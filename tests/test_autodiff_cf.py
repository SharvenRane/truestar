"""Gradients through control flow (truestar.ir.autodiff_cf): cond, loop, while, nested.

Three independent checks, as for the other rules (tests/test_autodiff.py):

1. every program against fourth order finite differences of the float64 reference interpreter;
2. against torch.autograd on the same computation unrolled in plain PyTorch (float64);
3. every loop memory policy ("store", "recompute", "sqrt") gives the same gradients, and its
   buffers have the size the policy promises.

Plus loud failures and tests that prove the checks can fail.
"""

import math
import warnings

import pytest
import sympy
import torch

import truestar
from truestar import Tensor, ir, ops
from truestar.ir import autodiff_cf
from truestar.ir.autodiff import grad

from test_autodiff import FD_TOL, fd_check, rel_err

warnings.filterwarnings("ignore", category=FutureWarning)

F64 = torch.float64
POLICIES = ["store", "recompute", "sqrt"]
TORCH_TOL = 1e-12  # same float64 math in a different order; measured worst 0 to 1e-15


# ----------------------------------------------------------------------------- programs


@truestar.function
def rnn(x: Tensor["B", "T", 4], wx: Tensor[4, 5], wh: Tensor[5, 5]) -> Tensor["B", 5]:
    h = ops.zeros((x.shape[0], 5))
    for t in range(x.shape[1]):
        h = ops.tanh(x[:, t] @ wx + h @ wh)
    return h


def rnn_torch(x, wx, wh):
    h = torch.zeros(x.shape[0], 5, dtype=x.dtype)
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ wx + h @ wh)
    return h


@truestar.function
def branchy(x: Tensor["B", 4], w: Tensor[4, 4]) -> Tensor["B", 4]:
    h = x @ w
    if h.mean() > 0:
        a = ops.tanh(h) * x
        c = h * 0.5
    else:
        a = h * h
        c = ops.sigmoid(x)
    return a + c * w.sum()


def branchy_torch(x, w):
    h = x @ w
    if h.mean() > 0:
        a, c = torch.tanh(h) * x, h * 0.5
    else:
        a, c = h * h, torch.sigmoid(x)
    return a + c * w.sum()


@truestar.function
def grow(x: Tensor["B", 3], w: Tensor[3, 3]) -> Tensor["B", 3]:
    while (x * x).sum() < 50.0:          # the iteration count depends on the data
        x = ops.tanh(x @ w) + x * 1.5
    return x


def grow_torch(x, w):
    while (x * x).sum() < 50.0:
        x = torch.tanh(x @ w) + x * 1.5
    return x


@truestar.function
def nested(x: Tensor["B", "T", 3], w: Tensor[3, 3]) -> Tensor[()]:
    h = x[:, 0]
    for t in range(x.shape[1]):
        v = x[:, t] @ w
        if v.sum() > 0:
            h = ops.tanh(h + v)
        else:
            s = h
            while (s * s).sum() < 4.0:
                s = s * 1.5 + 0.1
            h = s - v
    return (h * h).sum()


def nested_torch(x, w):
    h = x[:, 0]
    for t in range(x.shape[1]):
        v = x[:, t] @ w
        if v.sum() > 0:
            h = torch.tanh(h + v)
        else:
            s = h
            while (s * s).sum() < 4.0:
                s = s * 1.5 + 0.1
            h = s - v
    return (h * h).sum()


@truestar.function
def two_carry(x: Tensor["B", 3], w: Tensor[3, 3]) -> Tensor["B", 3]:
    h = x
    g = x * 0.5
    n = 0
    for t in range(x.shape[0] + 2):
        h, g = ops.tanh(h @ w + g), g * 0.9 + h * 0.1
        n = n + 1
    return h * g + n


def two_carry_torch(x, w):
    h, g, n = x, x * 0.5, 0
    for t in range(x.shape[0] + 2):
        h, g = torch.tanh(h @ w + g), g * 0.9 + h * 0.1
        n += 1
    return h * g + n


@truestar.function
def stack(x: Tensor["B", "T", 4], wh: Tensor[4, 4]) -> Tensor["T", "B", 4]:
    h = ops.zeros((x.shape[0], 4))
    ys = ops.zeros((x.shape[1], x.shape[0], 4))
    for t in range(x.shape[1]):
        h = ops.tanh(x[:, t] + h @ wh)
        ys = ops.index_copy(ys, 0, t.reshape(1), h.reshape(1, x.shape[0], 4))
    return ys


def stack_torch(x, wh):
    h = torch.zeros(x.shape[0], 4, dtype=x.dtype)
    ys = []
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] + h @ wh)
        ys.append(h)
    return torch.stack(ys)


@truestar.function
def short(x: Tensor["B", "T", 4], wx: Tensor[4, 5], wh: Tensor[5, 5]) -> Tensor["B", 5]:
    h = x[:, 0] @ wx
    for t in range(x.shape[1] - 1):       # zero iterations at T = 1
        h = ops.tanh(x[:, t + 1] @ wx + h @ wh)
    return h


def short_torch(x, wx, wh):
    h = x[:, 0] @ wx
    for t in range(x.shape[1] - 1):
        h = torch.tanh(x[:, t + 1] @ wx + h @ wh)
    return h


def _inputs(name, seed=0, **sizes):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=F64)  # noqa: E731
    B, T = sizes.get("B", 2), sizes.get("T", 5)
    if name in ("rnn", "short"):
        return [r(B, T, 4), r(4, 5) * 0.6, r(5, 5) * 0.4]
    if name == "branchy":
        x = r(B, 4).abs() + 0.5
        return [x if sizes.get("side", 1) > 0 else -x, r(4, 4).abs() * 0.3 + 0.1]
    if name == "grow":
        return [r(B, 3) * 0.5, r(3, 3) * 0.3]
    if name == "nested":
        return [r(B, T, 3) * 0.3, r(3, 3)]  # small enough that the inner while loop runs
    if name == "two_carry":
        return [r(B, 3), r(3, 3) * 0.5]
    if name == "stack":
        return [r(B, T, 4), r(4, 4) * 0.5]
    raise KeyError(name)


NATIVE = {"rnn": (rnn, rnn_torch), "branchy": (branchy, branchy_torch), "grow": (grow, grow_torch),
          "nested": (nested, nested_torch), "two_carry": (two_carry, two_carry_torch),
          "stack": (stack, stack_torch), "short": (short, short_torch)}


def _vjp(fn, inputs, policy, seed=1):
    """Gradients of <v, f(inputs)> from the grad program, and from torch.autograd on the unrolled
    PyTorch version of the same function."""
    native, torch_fn = NATIVE[fn]
    prog = native.program
    names = [v.name for v in prog.inputs]
    gp = grad(prog, names, seed="input", loop_memory=policy)
    outs = ir.run(prog, *inputs)
    gen = torch.Generator().manual_seed(seed)
    vs = [torch.randn(o.shape, generator=gen, dtype=F64) for o in outs]
    res = ir.run(gp, *inputs, *vs)
    xs = [t.clone().requires_grad_() for t in inputs]
    want_out = torch_fn(*xs)
    assert rel_err(res[0], want_out.detach()) < TORCH_TOL
    want = torch.autograd.grad((want_out * vs[0]).sum(), xs, allow_unused=True)
    want = [torch.zeros_like(x) if w is None else w for x, w in zip(inputs, want)]
    return res[len(outs):], want


# ----------------------------------------------------------------------------- 1. torch.autograd


TORCH_CASES = [("rnn", {"T": 1}), ("rnn", {"T": 3}), ("rnn", {"T": 8}), ("rnn", {"B": 1, "T": 6}),
               ("branchy", {"side": 1}), ("branchy", {"side": -1}),
               ("grow", {}), ("grow", {"B": 4}),
               ("nested", {"T": 6}), ("nested", {"T": 9, "B": 3}),
               ("two_carry", {"B": 3}), ("stack", {"T": 5}), ("short", {"T": 1}), ("short", {"T": 4})]


@pytest.mark.parametrize("policy", POLICIES + ["auto"])
@pytest.mark.parametrize("fn,sizes", TORCH_CASES, ids=[f"{f}-{'-'.join(f'{k}{v}' for k, v in s.items())}"
                                                        for f, s in TORCH_CASES])
def test_matches_torch_autograd_on_the_unrolled_program(fn, sizes, policy):
    got, want = _vjp(fn, _inputs(fn, **sizes), policy)
    for g, w in zip(got, want):
        assert tuple(g.shape) == tuple(w.shape)
        assert rel_err(g, w) < TORCH_TOL, (fn, sizes, policy)


def test_nested_case_takes_both_branches_and_runs_the_inner_while():
    """The nested case really exercises the cond inside the loop on both sides, and the while
    inside the else branch runs: otherwise its torch comparison would prove less than it claims."""
    x, w = _inputs("nested", T=9, B=3)
    sides, whiles = set(), 0
    h = x[:, 0]
    for t in range(9):
        v = x[:, t] @ w
        sides.add(bool(v.sum() > 0))
        if v.sum() > 0:
            h = torch.tanh(h + v)
        else:
            s = h
            while (s * s).sum() < 4.0:
                s, whiles = s * 1.5 + 0.1, whiles + 1
            h = s - v
    assert sides == {True, False} and whiles >= 2


# ----------------------------------------------------------------------------- 2. finite differences


FD_CASES = [("rnn", {"T": 4}), ("branchy", {"side": 1}), ("branchy", {"side": -1}), ("grow", {}),
            ("nested", {"T": 4}), ("two_carry", {"B": 2}), ("stack", {"T": 3}), ("short", {"T": 3})]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("fn,sizes", FD_CASES, ids=[f"{f}-{'-'.join(f'{k}{v}' for k, v in s.items())}"
                                                    for f, s in FD_CASES])
def test_matches_finite_differences(fn, sizes, policy):
    prog = NATIVE[fn][0].program
    inputs = _inputs(fn, **sizes)
    gp = grad(prog, [v.name for v in prog.inputs], seed="input", loop_memory=policy)
    assert fd_check(prog, inputs, gp) < FD_TOL


def _bounded_while():
    """A while loop with max_iterations (the IR allows it; the front end does not emit it), so the
    store policy can size its buffer."""
    p = grow.program
    (loop,) = [n for n in p.nodes if n.op == "loop"]
    loop.attrs["max_iterations"] = 64
    return p


def test_bounded_while_loop_stores_and_matches(monkeypatch):
    p = _bounded_while()
    try:
        inputs = _inputs("grow")
        for policy in ("store", "recompute", "sqrt"):
            gp = grad(p, ["x", "w"], seed="input", loop_memory=policy)
            assert fd_check(p, inputs, gp) < FD_TOL
        store = grad(p, ["x", "w"], seed="input", loop_memory="store")
        assert _buffer_rows(store, (2, 3), {sympy.Symbol("B", integer=True, positive=True): 2}) == [64]
    finally:
        (loop,) = [n for n in grow.program.nodes if n.op == "loop"]
        loop.attrs.pop("max_iterations", None)


# ----------------------------------------------------------------------------- 3. memory policies


def _walk(prog):
    for n in prog.nodes:
        yield n
        for r in [v for v in n.attrs.values() if isinstance(v, ir.Program)]:
            yield from _walk(r)


def _buffer_rows(prog, carry_shape, bindings):
    """Rows of every buffer of carried values (a ``full`` of shape [rows, *carry_shape]) anywhere in
    the program: what a policy keeps alive, counted in copies of the carried value."""
    rows = []
    for n in _walk(prog):
        if n.op != "full" or n.output.type.rank != len(carry_shape) + 1:
            continue
        shape = n.output.type.shape
        tail = [int(sympy.sympify(d).subs(bindings)) for d in shape[1:]]
        if tail == list(carry_shape):
            rows.append(int(sympy.sympify(shape[0]).subs(bindings)))
    return rows


@pytest.mark.parametrize("T", [8, 32, 48])
def test_policies_keep_what_they_promise(T):
    B = 2
    s = {sympy.Symbol("B", integer=True, positive=True): B, sympy.Symbol("T", integer=True, positive=True): T}
    progs = {p: grad(rnn.program, ["wx", "wh"], seed="ones", loop_memory=p) for p in POLICIES}
    store, recompute, sq = (_buffer_rows(progs[p], (B, 5), s) for p in POLICIES)
    assert store == [T]                       # every carried h, once
    assert recompute == []                    # nothing
    seg = math.isqrt(T - 1) + 1               # ceil(sqrt(T)) iterations per checkpoint
    assert sorted(sq) == sorted([-(-T // seg), seg])  # checkpoints, plus one segment replayed
    assert sum(sq) <= 2 * seg
    inputs = _inputs("rnn", T=T, B=B)
    ref = ir.run(progs["store"], *inputs)
    for p in ("recompute", "sqrt"):
        for a, b in zip(ir.run(progs[p], *inputs), ref):
            assert rel_err(a, b) < 1e-13


def test_store_keeps_only_what_the_backward_reads():
    """The stacked outputs of a scan are written by index_copy and never read by the backward, so
    the store policy keeps the hidden state (T rows) but no copy of the [T, B, 4] stack."""
    B, T = 2, 6
    s = {sympy.Symbol("B", integer=True, positive=True): B, sympy.Symbol("T", integer=True, positive=True): T}
    gp = grad(stack.program, ["x", "wh"], seed="ones", loop_memory="store")
    before = _buffer_rows(stack.program, (B, 4), s)   # the stack itself, [T, B, 4]
    assert before == [T]
    assert _buffer_rows(gp, (B, 4), s) == before + [T]  # plus the stored hidden states
    assert _buffer_rows(gp, (T, B, 4), s) == []


def test_one_grad_program_serves_every_length():
    gp = grad(rnn.program, ["x", "wx", "wh"], seed="ones")
    for T in (1, 2, 7, 13):
        inputs = _inputs("rnn", T=T)
        res = ir.run(gp, *inputs)
        xs = [t.clone().requires_grad_() for t in inputs]
        want = torch.autograd.grad(rnn_torch(*xs).sum(), xs)
        assert all(rel_err(g, w) < TORCH_TOL for g, w in zip(res[1:], want))


def test_capture_gradient_touches_one_row_per_step():
    """The gradient of x[:, t] inside the loop is added row by row (index_copy of acc[t] + g), not
    as a full zeros tensor plus one row every step."""
    gp = grad(rnn.program, ["x"], seed="ones", loop_memory="store")
    bwd = [n for n in gp.nodes if n.op == "loop"][-1]
    ops_in_body = bwd.attrs["body"].op_counts()
    assert ops_in_body.get("index_select") == 1 and ops_in_body.get("full_like", 0) == 0


# ----------------------------------------------------------------------------- loud failures


def _memory_attrs(prog):
    return [n.attrs.get("memory") for n in _walk(prog) if n.op == "loop" and "memory" in n.attrs]


def test_each_backward_loop_says_which_policy_it_got():
    """An unbounded while loop cannot size a buffer, so it recomputes whatever was asked, and its
    backward loop says so; counted loops follow the request."""
    for policy in POLICIES:
        assert _memory_attrs(grad(grow.program, ["x"], loop_memory=policy)) == ["recompute"]
        assert _memory_attrs(grad(rnn.program, ["x"], loop_memory=policy)) == [policy]
    assert _memory_attrs(grad(rnn.program, ["x"])) == ["store"]
    with pytest.raises(ir.IRError, match="loop_memory"):
        grad(rnn.program, ["x"], loop_memory="sometimes")


def test_rule_table_names_control_flow():
    from truestar.ir import autodiff
    table = autodiff.rule_table()
    assert table["cond"] == table["loop"] == "control flow"
    assert table["select_dynamic"] == "rule"


def test_grad_program_round_trips_through_the_artifact_format():
    from truestar.ir import serialize
    gp = grad(nested.program, ["x", "w"], seed="ones")
    back = serialize.loads(serialize.dumps(gp)).program
    inputs = _inputs("nested", T=5)
    for a, b in zip(ir.run(back, *inputs), ir.run(gp, *inputs)):
        assert torch.equal(a, b)


# ----------------------------------------------------------------------------- the checks can fail


def test_dropping_capture_gradients_is_caught(monkeypatch):
    monkeypatch.setattr(autodiff_cf, "_accumulate", lambda b, acc, g: acc)
    got, want = _vjp("rnn", _inputs("rnn", T=4), "store")
    assert rel_err(got[1], want[1]) > 1e-2 and rel_err(got[2], want[2]) > 1e-2


def test_corrupted_stored_carries_are_caught_and_recompute_is_independent(monkeypatch):
    def zero_row(b, x):
        z = b.op("full", shape=[], value=0.0, dtype=x.type.dtype)
        return b.op("reshape", b.op("mul", x, z), shape=[1] + list(x.type.shape))
    monkeypatch.setattr(autodiff_cf, "_row", zero_row)
    got, want = _vjp("rnn", _inputs("rnn", T=4), "store")
    assert rel_err(got[2], want[2]) > 1e-2
    got, want = _vjp("rnn", _inputs("rnn", T=4), "recompute")  # stores nothing: still right
    assert all(rel_err(g, w) < TORCH_TOL for g, w in zip(got, want))
