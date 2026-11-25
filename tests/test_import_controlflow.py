"""torch.export control flow (torch.cond, while_loop, scan) imported into IR cond and loop, and
unbounded symbolic dims.

Every imported program is run by the float64 reference interpreter and compared with the PyTorch
module in float64, in both graph forms (export and core ATen), on inputs that take each side of
every branch. Gradients of the imported programs are compared with torch.autograd.
"""

import copy
import warnings

import pytest
import torch
from torch import nn
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.while_loop import while_loop
from torch.export import Dim

from truestar import ir
from truestar.ir.autodiff import grad

warnings.filterwarnings("ignore")

F64 = torch.float64
TOL = 1e-12


def rel(a, b):
    s = max(a.abs().max().item(), b.abs().max().item(), 1e-300)
    return (a - b).abs().max().item() / s


class Branch(nn.Module):
    """torch.cond with a parameter, two results and a Python scalar in one branch."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x):
        def yes(x):
            h = torch.relu(self.lin(x))
            return h, h.sum(1)

        def no(x):
            return -x * 0.5, x.mean(1)

        return torch.cond(x.sum() > 0, yes, no, (x,))


class NestedBranch(nn.Module):
    def forward(self, x, y):
        def outer_t(x, y):
            return torch.cond(y.mean() > 0, lambda a, b: a * b, lambda a, b: a - b, (x, y))

        def outer_f(x, y):
            return torch.tanh(x) + y

        return torch.cond(x.mean() > 0, outer_t, outer_f, (x, y))


class Grow(nn.Module):
    """while_loop with a data dependent trip count and a captured tensor."""

    def forward(self, x, limit):
        def c(i, h):
            return (h * h).sum() < limit.sum()

        def b(i, h):
            return i + 1, h * 1.5 + 0.1

        i, h = while_loop(c, b, (torch.zeros((), dtype=torch.int64), x))
        return h, i


class ScanRNN(nn.Module):
    """scan over time with a carried hidden state and stacked outputs."""

    def __init__(self, reverse=False):
        super().__init__()
        self.wx = nn.Parameter(torch.randn(4, 5) * 0.5)
        self.wh = nn.Parameter(torch.randn(5, 5) * 0.4)
        self.reverse = reverse

    def forward(self, xs, h0):
        def step(h, x):
            h2 = torch.tanh(x @ self.wx + h @ self.wh)
            return h2, h2 * 2.0

        h, ys = scan(step, h0, xs, reverse=self.reverse)
        return h, ys


class ScanWithBranch(nn.Module):
    """A cond inside a scan body."""

    def forward(self, xs, h0):
        def step(h, x):
            h2 = torch.cond(x.sum() > 0, lambda h, x: torch.tanh(h + x), lambda h, x: h * 0.9 - x, (h, x))
            return h2, h2.sum(-1)

        return scan(step, h0, xs)


def unrolled_rnn(m, xs, h0):
    h, ys = h0, []
    steps = range(xs.shape[0] - 1, -1, -1) if m.reverse else range(xs.shape[0])
    out = [None] * xs.shape[0]
    for t in steps:
        h = torch.tanh(xs[t] @ m.wx + h @ m.wh)
        out[t] = h * 2.0
    del ys
    return h, torch.stack(out)


def fresh_export_state():
    """Eager torch scan compiles its body with Dynamo; a cache left by an earlier eager call can
    make a later torch.export specialize a dim. Each test exports from a clean Dynamo state."""
    torch._dynamo.reset()


def check(model, args, dynamic, other_inputs=(), decompose=False):
    fresh_export_state()
    prog = ir.import_module(model, args, dynamic_shapes=dynamic, decompose=decompose)
    prog.verify()
    m64 = copy.deepcopy(model).double()
    worst = 0.0
    for inputs in [args] + list(other_inputs):
        x64 = [a.double() if a.is_floating_point() else a for a in inputs]
        got = ir.run(prog, *x64)
        want = m64(*x64)
        want = list(want) if isinstance(want, (tuple, list)) else [want]
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert tuple(g.shape) == tuple(w.shape)
            if w.is_floating_point():
                worst = max(worst, rel(g, w))
            else:
                assert torch.equal(g.to(w.dtype), w)
    return prog, worst


@pytest.mark.parametrize("decompose", [False, True], ids=["export", "core_aten"])
def test_cond_both_sides_and_symbolic_batch(decompose):
    torch.manual_seed(0)
    x = torch.rand(2, 3) + 0.1
    prog, worst = check(Branch(), (x,), ({0: Dim("B")},), [(-x,), (torch.rand(5, 3) + 0.1,), (-torch.rand(1, 3),)],
                        decompose)
    assert worst < TOL
    assert prog.op_counts()["cond"] == 1 and "B" in prog.symbols and prog.symbols["B"][1] is None


def test_nested_cond():
    torch.manual_seed(1)
    x, y = torch.rand(2, 3) + 0.1, torch.rand(2, 3) + 0.1
    cases = [(x, -y), (-x, y), (-x, -y)]
    _, worst = check(NestedBranch(), (x, y), ({0: Dim("B")}, {0: Dim("B")}), cases)
    assert worst < TOL


@pytest.mark.parametrize("decompose", [False, True], ids=["export", "core_aten"])
def test_while_loop_data_dependent_trip_count(decompose):
    torch.manual_seed(2)
    x = torch.rand(2, 3) * 0.1
    lim = torch.tensor([4.0])
    prog, worst = check(Grow(), (x, lim), None, [(x, torch.tensor([50.0])), (x * 30, lim)], decompose)
    assert worst < TOL and prog.op_counts()["loop"] == 1
    loop = [n for n in prog.nodes if n.op == "loop"][0]
    assert loop.attrs.get("cond") is not None


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
@pytest.mark.parametrize("decompose", [False, True], ids=["export", "core_aten"])
def test_scan_rnn_symbolic_time_and_batch(reverse, decompose):
    torch.manual_seed(3)
    m = ScanRNN(reverse)
    xs, h0 = torch.randn(6, 2, 4), torch.randn(2, 5)
    T, B = Dim("T"), Dim("B")
    others = [(torch.randn(1, 3, 4), torch.randn(3, 5)), (torch.randn(11, 2, 4), torch.randn(2, 5))]
    prog, worst = check(m, (xs, h0), ({0: T, 1: B}, {0: B}), others, decompose)
    assert worst < TOL
    assert {"T", "B"} <= set(prog.symbols)
    loop = [n for n in prog.nodes if n.op == "loop"][0]
    assert str(loop.attrs["trip_count"]) == "T"


def test_cond_inside_scan():
    torch.manual_seed(4)
    xs, h0 = torch.randn(7, 2, 3), torch.randn(2, 3)
    _, worst = check(ScanWithBranch(), (xs, h0), ({0: Dim("T")}, None), [(torch.randn(3, 2, 3), h0)])
    assert worst < TOL


@pytest.mark.parametrize("policy", ["store", "recompute", "sqrt"])
def test_imported_scan_gradients_match_torch_autograd(policy):
    """Train the scan RNN from torch.export: gradients of the imported program against torch on the
    unrolled loop, for every loop memory policy."""
    torch.manual_seed(5)
    m = ScanRNN()
    xs, h0 = torch.randn(5, 2, 4), torch.randn(2, 5)
    T, B = Dim("T"), Dim("B")
    fresh_export_state()
    prog = ir.import_module(m, (xs, h0), dynamic_shapes=({0: T, 1: B}, {0: B}))
    gp = grad(prog, ["xs", "h0", "wx", "wh"], seed="input", loop_memory=policy)
    m64 = copy.deepcopy(m).double()
    for T in (1, 5, 9):
        x64, h64 = torch.randn(T, 3, 4, dtype=F64), torch.randn(3, 5, dtype=F64)
        vh, vy = torch.randn(3, 5, dtype=F64), torch.randn(T, 3, 5, dtype=F64)
        res = ir.run(gp, x64, h64, vh, vy)
        xs_, h_ = x64.clone().requires_grad_(), h64.clone().requires_grad_()
        h, ys = unrolled_rnn(m64, xs_, h_)
        want = torch.autograd.grad((h * vh).sum() + (ys * vy).sum(), (xs_, h_, m64.wx, m64.wh))
        for g, w in zip(res[2:], want):
            assert rel(g, w) < TOL, (policy, T)


def test_imported_cond_gradients_match_torch_autograd():
    torch.manual_seed(6)
    m = Branch()
    prog = ir.import_module(m, (torch.rand(2, 3) + 0.1,), dynamic_shapes=({0: Dim("B")},))
    gp = grad(prog, ["x", "lin.weight", "lin.bias"], seed="input")
    m64 = copy.deepcopy(m).double()
    for x in (torch.rand(3, 3, dtype=F64) + 0.1, -torch.rand(4, 3, dtype=F64) - 0.1):
        v0, v1 = torch.randn(x.shape[0], 3, dtype=F64), torch.randn(x.shape[0], dtype=F64)
        res = ir.run(gp, x, v0, v1)
        xr = x.clone().requires_grad_()
        a, b = m64(xr)
        want = torch.autograd.grad((a * v0).sum() + (b * v1).sum(), (xr, m64.lin.weight, m64.lin.bias),
                                   allow_unused=True)
        want = [torch.zeros_like(p) if w is None else w for p, w in zip((x, m64.lin.weight, m64.lin.bias), want)]
        for g, w in zip(res[2:], want):
            assert rel(g, w) < TOL if w.abs().max() > 0 else g.abs().max() == 0


def test_unknown_op_inside_a_branch_is_named_with_the_rest():
    class Bad(nn.Module):
        def forward(self, x):
            return torch.cond(x.sum() > 0, lambda x: torch.special.bessel_j0(x), lambda x: torch.special.erfcx(x), (x,))

    with pytest.raises(ir.UnsupportedOpError) as err:
        ir.import_module(Bad(), (torch.randn(2, 3),))
    msg = str(err.value)
    assert "bessel_j0" in msg and "erfcx" in msg and "2 problem(s)" in msg


def test_non_tensor_operand_is_refused_by_name():
    class IntCarry(nn.Module):
        def forward(self, x):
            return while_loop(lambda i, h: i < 3, lambda i, h: (i + 1, h * 2), (0, x))[1]

    with pytest.raises(ir.UnsupportedOpError) as err:
        ir.import_module(IntCarry(), (torch.randn(2, 3),))
    assert "carry it as a tensor" in str(err.value)


def test_unbounded_dims_import_and_run_far_beyond_the_example():
    m = nn.Sequential(nn.Linear(3, 4), nn.Tanh())
    prog = ir.import_module(m, (torch.randn(2, 3),), dynamic_shapes=({0: Dim("B")},))
    assert prog.symbols["B"] == (1, None)
    x = torch.randn(4099, 3, dtype=F64)
    assert rel(ir.run(prog, x)[0], copy.deepcopy(m).double()(x)) < TOL
