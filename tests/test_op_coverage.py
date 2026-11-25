"""The ATen ops added with the op coverage work (ir/importer_extra.py, docs/MODEL_COVERAGE.md).

Every torch.nn layer and loss of ``truestar.testing_ops.nn_layers`` is imported and checked four
ways, all in float64:

1. the imported program against the PyTorch module (bar 1e-10 relative to the largest output);
2. its gradient (a random cotangent on output 0, with respect to every float param, or every float
   input for a module without params) against torch.autograd (bar 1e-10 relative);
3. vmap of the program over a new leading axis against the program run once per element (1e-12);
4. forward mode against a fourth order central difference along a random direction, and the dot
   product test against the reverse mode program (the bars of tests/test_forward.py).

Plus the composites written for these ops (expm1, log1p, softplus) at the points where a naive
formula fails, and checks that the tests can fail.
"""

import copy
import warnings

import pytest
import torch

import test_forward as tf
from truestar import ir
from truestar.ir.autodiff import grad
from truestar.ir.vmap import vmap
from truestar.testing_ops import current_ops, nn_layers

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

F64 = torch.float64
# the torch.nn layers and losses, and the ops and patterns of the 2025 to 2026 families
# (coverage refresh: diff, floor, bucketize, std, string conv padding, attention sinks, ...)
LAYERS = dict(nn_layers() + current_ops())
NAMES = sorted(LAYERS)
VALUE_TOL = 1e-10
GRAD_TOL = 1e-10
VMAP_TOL = 1e-12
# PyTorch's own float64 CPU backward of hardsigmoid multiplies by the float32 constant 1/6
# (0.1666666716337204, measured 2026-11-24, torch 2.11): Truestar's 1/6 is the exact one, so this
# layer's gradient is compared with the mathematically right value instead of torch.autograd.
TORCH_GRAD_IS_FLOAT32 = {"Hardsigmoid"}


def _rel(got, want):
    scale = want.abs().max().item() if want.numel() else 0.0
    if scale == 0.0:
        return (got - want).abs().max().item() if got.numel() else 0.0
    return (got.double() - want.double()).abs().max().item() / scale


def _to64(args):
    return tuple(a.double() if a.is_floating_point() else a for a in args)


_CACHE = {}


def _program(name):
    if name not in _CACHE:
        torch.manual_seed(0)
        module, args = LAYERS[name]()
        m64 = copy.deepcopy(module).double().eval()
        a64 = _to64(args)
        _CACHE[name] = (m64, a64, ir.import_module(m64, a64))
    return _CACHE[name]


def _outs(v):
    """The tensors of a result, nested tuples flattened (an LSTM returns (out, (h, c)))."""
    return [t for t in torch.utils._pytree.tree_leaves(v) if torch.is_tensor(t)]


@pytest.mark.parametrize("name", NAMES)
def test_import_matches_torch_float64(name):
    m64, a64, prog = _program(name)
    with torch.no_grad():
        want = _outs(m64(*a64))
    got = _outs(ir.run(prog, *a64))
    want = [w for w in want if torch.is_tensor(w)]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert tuple(g.shape) == tuple(w.shape)
        assert _rel(g, w) < VALUE_TOL, (name, _rel(g, w))


@pytest.mark.parametrize("name", NAMES)
def test_gradient_matches_torch_autograd(name):
    m64, a64, prog = _program(name)
    params = {k: v for k, v in m64.named_parameters() if v.requires_grad}
    names = [v.name for v in prog.params if v.type.dtype.is_floating_point and v.name in params]
    on_inputs = not names
    if on_inputs:
        names = [v.name for v, a in zip(prog.inputs, a64) if a.is_floating_point()]
    if not names:
        pytest.skip("integer inputs only")
    out0 = _outs(ir.run(prog, *a64))[0]
    cot = torch.randn(out0.shape, generator=torch.Generator().manual_seed(7), dtype=F64)
    got = _outs(ir.run(grad(prog, wrt=names, of=[0], seed="input"), *a64, cot))[len(prog.outputs):]
    leaves = [a.clone().requires_grad_(True) if a.is_floating_point() else a for a in a64]
    m64.zero_grad(set_to_none=True)
    _outs(m64(*leaves))[0].backward(cot)
    wants = [a.grad for a in leaves if a.is_floating_point()] if on_inputs else [params[n].grad for n in names]
    for n, g, w in zip(names, got, wants):
        w = torch.zeros_like(g) if w is None else w
        if name in TORCH_GRAD_IS_FLOAT32:
            # torch rounds 1/6 to float32: equal to float32 precision, and Truestar's is exactly 1/6
            assert _rel(g, w) < 1e-7
            inside = (a64[0] > -3) & (a64[0] < 3)
            assert torch.equal(g, torch.where(inside, cot * (1.0 / 6.0), torch.zeros_like(cot)))
            continue
        assert _rel(g, w) < GRAD_TOL, (name, n, _rel(g, w))


def _batched_args(a64, n=3):
    gen = torch.Generator().manual_seed(11)
    out = []
    for a in a64:
        if a.is_floating_point():
            out.append(torch.stack([a] + [torch.randn(a.shape, generator=gen, dtype=F64) for _ in range(n - 1)]))
        else:
            out.append(torch.stack([a] * n))
    return out


@pytest.mark.parametrize("name", NAMES)
def test_vmap_equals_the_per_element_loop(name):
    _, a64, prog = _program(name)
    flags = [a.is_floating_point() for a in a64]
    if not any(flags):   # integer inputs (embeddings): batch them too, the per element meaning is the same
        flags = [True] * len(a64)
    vp = vmap(prog, in_axes=[0 if f else None for f in flags])
    args = _batched_args(a64)
    args = [b if f else a for a, b, f in zip(a64, args, flags)]
    got = _outs(ir.run(vp, *args))
    per = [_outs(ir.run(prog, *[x[k] if f else x for x, f in zip(args, flags)])) for k in range(3)]
    for j, g in enumerate(got):
        w = torch.stack([p[j] for p in per])
        if not g.dtype.is_floating_point:
            assert torch.equal(g, w)
            continue
        assert _rel(g, w) < VMAP_TOL, (name, j, _rel(g, w))


_KINKS = {   # piecewise layers: keep finite differences off the corners
    "ReLU6": [0.0, 6.0], "Hardshrink": [-0.5, 0.5], "Softshrink": [-0.5, 0.5], "Threshold": [0.5],
    "LeakyReLU": [0.0], "PReLU": [0.0], "ELU": [0.0], "CELU": [0.0], "SELU": [0.0], "Hardswish": [-3.0, 3.0],
    "Hardsigmoid": [-3.0, 3.0], "Softsign": [0.0], "L1Loss": [0.0], "SmoothL1Loss": [0.0, 1.0, -1.0], "HuberLoss": [0.0, 0.5, -0.5],
    # coverage refresh: the steps of bucketize (its boundaries), floor(3 x) and the sign switch
    "bucketize": [-1.0, -0.25, 0.0, 0.5, 1.5], "bucketize_right": [-1.0, -0.25, 0.0, 0.5, 1.5],
    "floor": [k / 3.0 for k in range(-15, 16)], "bitwise_not": [0.0], "sum_of_bool": [0.0],
}
_NOT_SMOOTH = {"MaxPool1d", "MaxPool3d", "AdaptiveMaxPool1d", "EmbeddingBag_mean", "EmbeddingBag_sum", "Embedding",
               "L1Loss", "SmoothL1Loss", "HuberLoss", "PairwiseDistance"}


@pytest.mark.parametrize("name", NAMES)
def test_forward_mode_matches_finite_differences_and_reverse_mode(name):
    _, a64, prog = _program(name)
    if not any(a.is_floating_point() for a in a64):
        pytest.skip("integer inputs only")
    args = list(a64)
    if name in _KINKS:
        from test_autodiff import away_from
        args[0] = away_from(args[0], _KINKS[name], margin=0.05)
    if name in _NOT_SMOOTH:
        # max and abs based layers: distinct, well separated entries so no window or sign is near a tie
        x = args[0]
        perm = torch.randperm(x.numel(), generator=torch.Generator().manual_seed(3)).to(F64)
        args[0] = ((perm - x.numel() / 2 + 0.5) * 0.11).reshape(x.shape)
    fd, ts, jt, fouts = tf.fd_jvp_error(prog, args)
    assert fd < tf.FD_TOL, (name, fd)
    dot = tf.dot_error(prog, args, ts, jt, fouts)
    assert dot < tf.DOT_TOL, (name, dot)


# ----------------------------------------------------------------------------- the composites


def _unary_program(op_name):
    m = {"expm1": lambda x: torch.expm1(x), "log1p": lambda x: torch.log1p(x),
         "softplus": lambda x: torch.nn.functional.softplus(x), "log_sigmoid": torch.nn.functional.logsigmoid}[op_name]

    class M(torch.nn.Module):
        def forward(self, x):
            return m(x)
    return M(), m


@pytest.mark.parametrize("op_name", ["expm1", "log1p", "softplus", "log_sigmoid"])
def test_composites_are_accurate_where_the_naive_formula_is_not(op_name):
    """exp(x) - 1 and log(1 + x) lose every digit near 0 (1e-300 gives 0); softplus overflows for
    large x as log(1 + exp(x)). The composites keep 1e-15 relative per element everywhere."""
    module, fn = _unary_program(op_name)
    x = torch.tensor([-0.9, -0.5000001, -0.5, -1e-3, -1e-12, -1e-300, 0.0, 1e-300, 1e-12, 1e-3, 0.49, 0.5, 0.75, 3.0,
                      30.0], dtype=F64)
    if op_name in ("softplus", "log_sigmoid"):
        x = torch.cat([x, torch.tensor([-800.0, -40.0, 40.0, 800.0], dtype=F64)])
    prog = ir.import_module(module, (x,))
    got = ir.run(prog, x)[0]
    want = fn(x)
    rel = ((got - want).abs() / want.abs().clamp_min(1e-308)).max().item()
    assert rel < 4e-16 or (got - want).abs().max().item() == 0.0, (op_name, rel)
    assert torch.isfinite(got).all()


def test_naive_formulas_fail_the_same_points():
    """The check above can fail: exp(x) - 1 at 1e-12 is 1e-4 off relative, log(1 + x) at 1e-300 is 0."""
    x = torch.tensor([1e-12], dtype=F64)
    assert (((torch.exp(x) - 1) - torch.expm1(x)).abs() / torch.expm1(x)).item() > 1e-5
    assert torch.log(1 + torch.tensor([1e-300], dtype=F64)).item() == 0.0


def test_conv_transpose_is_the_adjoint_of_conv():
    """<conv(x, w), y> = <x, conv_transpose(y, w)>: the transposed convolution import is checked
    against the definition, not only against torch."""
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(2, 4, 9, 7, generator=gen, dtype=F64)   # sizes the transposed conv gives back exactly
    w = torch.randn(6, 2, 3, 3, generator=gen, dtype=F64)

    class T(torch.nn.Module):
        def forward(self, y):
            return torch.nn.functional.conv_transpose2d(y, w, stride=2, padding=1, groups=2, output_padding=0)
    y = torch.nn.functional.conv2d(x, w, stride=2, padding=1, groups=2)
    v = torch.randn(y.shape, generator=gen, dtype=F64)
    prog = ir.import_module(T(), (v,))
    xt = ir.run(prog, v)[0]
    assert xt.shape == x.shape
    lhs = (y * v).sum().item()
    rhs = (x * xt).sum().item()
    assert abs(lhs - rhs) / abs(lhs) < 1e-13


def test_unsupported_settings_are_refused_by_name():
    class Bad(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.avg_pool3d(x, 2, divisor_override=3)
    with pytest.raises(ir.UnsupportedOpError, match=r"\[TS860\].*divisor_override"):
        ir.import_module(Bad(), (torch.randn(1, 2, 4, 4, 4),))

    class LastOffset(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = torch.nn.EmbeddingBag(10, 4, mode="sum", include_last_offset=True)

        def forward(self, ids, offsets):
            return self.m(ids, offsets)
    with pytest.raises(ir.UnsupportedOpError, match=r"\[TS860\].*include_last_offset") as info:
        ir.import_module(LastOffset(), (torch.tensor([1, 2, 3, 4]), torch.tensor([0, 2, 4])))
    assert info.value.code == "TS860"

    class Shuffle(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.pixel_unshuffle(x, 2)
    H = torch.export.Dim("H", min=1, max=8)
    with pytest.raises(ir.UnsupportedOpError, match=r"\[TS861\].*must be fixed"):
        ir.import_module(Shuffle(), (torch.randn(1, 2, 4, 4),), dynamic_shapes=({2: 2 * H},))


@pytest.mark.parametrize("break_it", ["bicubic_a", "gru_gate"])
def test_a_broken_lowering_is_caught(monkeypatch, break_it):
    """The value test fails when a lowering is wrong: a cubic weight of a = -0.5 instead of -0.75, a GRU
    whose new gate reads h instead of r * h."""
    from truestar.ir import importer_extra as ie
    if break_it == "bicubic_a":
        orig = ie._cubic_matrix

        def wrong(in_size, out_size, factor, align):
            m = orig(in_size, out_size, factor, align)
            return m * 0.98 + 0.02 / in_size
        monkeypatch.setattr(ie, "_cubic_matrix", wrong)
        name = "Upsample_bicubic"
    else:
        orig = ie._cell

        def wrong(c, kind, x, h, cs, w_ih, w_hh, b_ih, b_hh, hidden):
            if kind == "gru":
                return orig(c, "rnn_tanh", x, h, cs, w_ih, w_hh, b_ih, b_hh, hidden * 3)[0], None
            return orig(c, kind, x, h, cs, w_ih, w_hh, b_ih, b_hh, hidden)
        monkeypatch.setattr(ie, "_cell", wrong)
        name = "GRU"
    torch.manual_seed(0)
    module, args = LAYERS[name]()
    m64 = copy.deepcopy(module).double().eval()
    a64 = _to64(args)
    try:
        prog = ir.import_module(m64, a64)
    except Exception:   # noqa: BLE001  a shape error is also a caught break
        return
    with torch.no_grad():
        want = _outs(m64(*a64))[0]
    assert _rel(_outs(ir.run(prog, *a64))[0], want) > VALUE_TOL




def test_frozen_carries_no_derivative():
    """frozen(x) is x with no derivative: d/dx (x * frozen(x)) = frozen(x), in both modes."""
    from truestar.ir.forward import jvp
    b = ir.Builder("f")
    x = b.input("x", F64, (3, 4))
    prog = b.build([b.op("mul", x, b.op("frozen", x))])
    xv = torch.randn(3, 4, dtype=F64)
    assert torch.equal(ir.run(prog, xv)[0], xv * xv)
    g = ir.run(grad(prog, ["x"], seed="ones"), xv)[1]
    assert torch.equal(g, xv)
    t = torch.randn(3, 4, dtype=F64)
    assert torch.equal(ir.run(jvp(prog, ["x"]), xv, t)[1], t * xv)


@pytest.mark.parametrize("fn", [torch.nn.functional.relu6, lambda x: torch.nn.functional.hardtanh(x, -1.0, 2.0),
                                lambda x: torch.clamp(x, -1.0, 2.0), lambda x: torch.clamp_min(x, -1.0),
                                lambda x: torch.clamp_max(x, 2.0)])
def test_clamp_family_gradient_at_the_bounds_follows_torch(fn):
    """hardtanh and relu6 send no gradient at x == bound; torch.clamp sends it before torch 2.14 and
    not from 2.14 on (pytorch #191142), so the import follows the running torch. mobilenet_v2 at its
    default init feeds 260 thousand exact zeros into ReLU6, so its imported gradient was 0.35 off
    before hardtanh and relu6 imported as clamp with strict bounds."""
    class M(torch.nn.Module):
        def forward(self, x):
            return fn(x)
    x = torch.tensor([-1.0, 0.0, 0.5, 2.0, 6.0, 7.0, -3.0], dtype=F64)
    prog = ir.import_module(M(), (x,))
    g = ir.run(grad(prog, [prog.inputs[0].name], seed="ones"), x)[1]
    xt = x.clone().requires_grad_()
    fn(xt).sum().backward()
    assert torch.equal(g, xt.grad)


# ----------------------------------------------------------------------------- coverage refresh (2025 to 2026)


def _imp(fn, *xs):
    class M(torch.nn.Module):
        def forward(self, *a):
            return fn(*a)
    return ir.import_module(M(), xs), M()


def test_floor_is_exact_at_the_edges():
    """Negative fractions, exact integers, -0.0, halves, the last fractional floats below 2^52, floats
    above it, the largest float, infinities and NaN: equal to torch.floor bit for bit."""
    big = 2.0 ** 52
    x = torch.tensor([-2.5, -1.0, -0.5, -1e-300, -0.0, 0.0, 1e-300, 0.5, 0.99999999999999989, 1.0, 2.5,
                      big - 0.5, -(big - 0.5), big, big + 2, 2.0 ** 60 + 2 ** 8, -(2.0 ** 60), 1.7976931348623157e308,
                      -1.7976931348623157e308, float("inf"), float("-inf"), float("nan")], dtype=F64)
    prog, m = _imp(torch.floor, x)
    got, want = ir.run(prog, x)[0], m(x)
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    ok = ~torch.isnan(want)
    assert torch.equal(got[ok], want[ok])
    assert torch.equal(torch.signbit(got[ok]), torch.signbit(want[ok]))
    g = ir.run(grad(prog, [prog.inputs[0].name], seed="ones"), x)[1]
    assert torch.equal(g, torch.zeros_like(x))     # torch.floor's gradient is 0 everywhere


def test_bucketize_at_the_boundaries_and_nan():
    b = torch.tensor([-1.0, 0.0, 0.0, 2.0], dtype=F64)
    x = torch.tensor([[-2.0, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0, float("inf"), float("-inf"), float("nan")]], dtype=F64)
    for right in (False, True):
        for out_int32 in (False, True):
            prog, m = _imp(lambda v, right=right, oi=out_int32: torch.bucketize(v, b, right=right, out_int32=oi), x)
            got, want = ir.run(prog, x)[0], m(x)
            assert got.dtype == want.dtype
            assert torch.equal(got[..., :-1], want[..., :-1]), (right, got, want)
            # NaN: torch places it after every boundary, the count of boundaries below it is 0 (documented)
            assert got[0, -1].item() == 0 and want[0, -1].item() == len(b)


def test_diff_of_booleans_is_inequality_and_integers_subtract():
    x = torch.tensor([[True, True, False, True, False]])
    prog, m = _imp(lambda v: torch.diff(v, dim=-1), x)
    assert torch.equal(ir.run(prog, x)[0], m(x))
    i = torch.tensor([[3, 1, 4, 1, 5, 9]])
    prog, m = _imp(lambda v: torch.diff(v, n=2, prepend=v[:, :1]), i)
    assert torch.equal(ir.run(prog, i)[0], m(i))


def test_new_settings_are_refused_by_name():
    x = torch.randn(3, 4, dtype=F64)
    with pytest.raises(ir.UnsupportedOpError, match=r"\[TS860\].*erfinv of a value that depends on an input"):
        _imp(lambda v: torch.special.erfinv(v.tanh()), x)


def test_erfinv_of_a_constant_is_folded_exactly():
    def f(v):
        p = torch.tensor([0.95, 0.5, 0.01, 0.0005], dtype=v.dtype)
        return v * torch.special.erfinv(2.0 * p - 1.0).sum()
    x = torch.randn(3, dtype=F64)
    prog, m = _imp(f, x)
    assert torch.equal(ir.run(prog, x)[0], m(x))
    assert "erfinv" not in " ".join(n.op for n in prog.nodes)   # folded: a constant, no IR op


class _Asserts(torch.nn.Module):
    """Asserts on a buffer (export traces buffers as fake tensors, so the assertion stays a graph
    node, as torch 2.15 leaves torch.distributions' argument validation) or on the input."""
    def __init__(self, scale, on_input=False):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=F64))
        self.on_input = on_input

    def forward(self, v):
        cond = v.sum() > -100 if self.on_input else self.scale > 0
        torch._assert_async(cond, "scale must be positive")
        return v * self.scale


def _assert_nodes(m, x):
    ep = torch.export.export(m, (x,))
    return [n for n in ep.graph.nodes if "_assert_async" in str(n.target)]


def test_assertion_on_constants_is_checked_at_import():
    """torch 2.15 exports torch.distributions' argument validation as aten._assert_async; on a
    condition computed from constants it is checked at import and leaves no IR op."""
    m, x = _Asserts(0.5), torch.randn(3, dtype=F64)
    assert _assert_nodes(m, x)      # the case under test reaches the importer
    prog = ir.import_module(m, (x,))
    assert torch.equal(ir.run(prog, x)[0], m(x))
    assert "assert" not in " ".join(n.op for n in prog.nodes)


def test_false_assertion_on_constants_names_its_message():
    m, x = _Asserts(-0.5), torch.randn(3, dtype=F64)
    assert _assert_nodes(m, x)
    with pytest.raises(ir.IRError, match=r"assertion fails on its constants: scale must be positive"):
        ir.import_module(m, (x,))


def test_assertion_on_an_input_is_refused_by_name():
    m, x = _Asserts(0.5, on_input=True), torch.randn(3, dtype=F64)
    assert _assert_nodes(m, x)
    with pytest.raises(ir.UnsupportedOpError, match=r"\[TS860\].*run time assertion on a value that depends on an input"):
        ir.import_module(m, (x,))


def _keep_last_column(orig):
    def handler(c, a):
        if a.get("end") == -1:      # [..., :-1] after the sink softmax: keep the sink column instead
            return orig(c, {**a, "start": 1, "end": None})
        return orig(c, a)
    return handler


@pytest.mark.parametrize("break_it", ["floor", "bucketize", "sinks"])
def test_a_broken_new_lowering_is_caught(monkeypatch, break_it):
    """The value test of the new ops fails when a lowering is wrong: floor as truncation, bucketize
    counting a boundary at the value as below it, a softmax that keeps the sink column."""
    from truestar.ir import importer_current as icur
    from truestar.ir.importer import HANDLERS
    if break_it == "floor":
        def trunc(c, a):
            x = c.tensor(a["self"])
            return icur._op(c, "cast", icur._op(c, "cast", x, dtype=torch.int64), dtype=x.type.dtype)
        monkeypatch.setitem(HANDLERS, "aten.floor.default", trunc)
        name = "floor"
    elif break_it == "bucketize":
        orig = HANDLERS["aten.bucketize.Tensor"]
        monkeypatch.setitem(HANDLERS, "aten.bucketize.Tensor", lambda c, a: orig(c, {**a, "right": not a.get("right")}))
        name = "bucketize"
    else:
        monkeypatch.setitem(HANDLERS, "aten.slice.Tensor", _keep_last_column(HANDLERS["aten.slice.Tensor"]))
        name = "attention_sinks"
    torch.manual_seed(0)
    module, args = LAYERS[name]()
    m64 = copy.deepcopy(module).double().eval()
    a64 = _to64(args)
    if name == "bucketize":     # values exactly on a boundary tell the two conventions apart
        a64 = (torch.tensor([[-1.0, -0.25, 0.0, 0.5, 1.5, 0.3]], dtype=F64),)
    prog = ir.import_module(m64, a64)
    with torch.no_grad():
        want = _outs(m64(*a64))[0]
    assert _rel(_outs(ir.run(prog, *a64))[0], want) > VALUE_TOL


class FakeSparseMoeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.gate(x)


class Mamba2Mixer(torch.nn.Module):
    def forward(self, x):
        return x.cumsum(-1)


class MambaMixer(Mamba2Mixer):       # the Mamba 1 sequential scan is supported: not refused
    pass


class SSD(torch.nn.Linear):          # a detector class name: not a state space layer
    pass


class GatedDeltaNetLayer(torch.nn.Linear):
    pass


@pytest.mark.parametrize("block, code", [(FakeSparseMoeBlock, "TS940"), (Mamba2Mixer, "TS942"),
                                         (lambda: GatedDeltaNetLayer(4, 4), "TS941"),
                                         (MambaMixer, None), (lambda: SSD(4, 4), None)])
def test_refusal_by_layer_name_without_transformers(block, code):
    """The rule reads class names (exact, then a name part), so a new family built from the same
    layers is refused too, on import_module and on import_exported; look alike names are not."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), block())
    x = torch.randn(2, 4)
    if code is None:
        ir.import_module(model, (x,))
        return
    with pytest.raises(ir.UnsupportedOpError) as info:
        ir.import_module(model, (x,))
    assert info.value.code == code and "at 1" in str(info.value)
    with pytest.raises(ir.UnsupportedOpError) as info:
        ir.import_exported(torch.export.export(model, (x,)))
    assert info.value.code == code
