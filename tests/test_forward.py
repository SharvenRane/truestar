"""Tests for forward mode autodiff and the Jacobian family (truestar.ir.forward).

Two independent checks for every forward mode rule:

1. against finite differences of the float64 reference interpreter along a random direction
   (fourth order stencil, h = 1e-3, the same bar and reasoning as tests/test_autodiff.py);
2. for every op that also has a reverse mode rule, the dot product identity
   <J t, c> == <t, J^T c>, with J t from ``jvp`` and J^T c from ``grad``. The two rules are written
   independently, so a wrong rule in either breaks the identity.

Then jacfwd, jacrev and hessian against torch.func in float64, linearize against jvp, and broken
rules that must fail.
"""

import math
import warnings

import pytest
import torch

import truestar.train  # noqa: F401  registers log_softmax
import test_autodiff as ad
from truestar import ir
from truestar.ir import forward as fwd_mod
from truestar.ir.autodiff import GRAD_RULES, LOWERINGS, grad
from truestar.ir.forward import hessian, jacfwd, jacrev, jvp, jvp_rule_table, linearize

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

F64 = torch.float64
I64 = torch.int64
H = ad.FD_STEP
FD_TOL = ad.FD_TOL
# Bar for the dot product test. Both sides are exact linear algebra in float64 on the same inputs,
# so they agree to rounding of the two inner products: measured worst 2.0e-15 on 2026-11-24.
DOT_TOL = 1e-12

FD_RESULTS, DOT_RESULTS = {}, {}


def fd_jvp_error(prog, inputs, wrt=None, seed=1):
    """Worst relative error between jvp's tangents and a fourth order central difference along a
    random direction in every float input."""
    names = [v.name for v in prog.inputs]
    wrt = wrt or ad.float_inputs(prog)
    gen = torch.Generator().manual_seed(seed)
    ts = [torch.randn(inputs[names.index(w)].shape, generator=gen, dtype=F64) for w in wrt]
    jp = jvp(prog, wrt)
    res = ir.run(jp, *inputs, *ts)
    nout = len(prog.outputs)
    prim = ir.run(prog, *inputs)
    for a, b in zip(res[:nout], prim):
        assert torch.equal(a, b) or ad.rel_err(a.to(F64), b.to(F64)) < 1e-14
    fouts = [i for i, o in enumerate(prog.outputs) if o.type.dtype.is_floating_point]

    def at(s):
        args = list(inputs)
        for w, t in zip(wrt, ts):
            k = names.index(w)
            args[k] = inputs[k] + s * t
        return ir.run(prog, *args)
    vals = [at(m * H) for m in (-2, -1, 1, 2)]
    worst = 0.0
    for j, i in enumerate(fouts):
        fd = (vals[0][i] - 8 * vals[1][i] + 8 * vals[2][i] - vals[3][i]) / (12 * H)
        worst = max(worst, ad.rel_err(res[nout + j], fd))
    return worst, ts, res[nout:], fouts


def dot_error(prog, inputs, ts, jt, fouts, seed=2):
    """|<J t, c> - <t, J^T c>| relative to the larger side."""
    wrt = ad.float_inputs(prog)
    gen = torch.Generator().manual_seed(seed)
    cs = [torch.randn(tuple(x.shape), generator=gen, dtype=F64) for x in jt]
    gp = grad(prog, wrt, of=fouts, seed="input")
    gs = ir.run(gp, *inputs, *cs)[len(prog.outputs):]
    lhs = sum((a * c).sum() for a, c in zip(jt, cs)).item()
    rhs = sum((t * g).sum() for t, g in zip(ts, gs)).item()
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)


# ----------------------------------------------------------------------------- the rule corpus


@pytest.mark.parametrize("name", sorted(ad.CASES))
def test_jvp_matches_finite_differences_and_the_transposed_vjp(name):
    prog, inputs = ad.make(name, 0)
    fd, ts, jt, fouts = fd_jvp_error(prog, inputs)
    dot = dot_error(prog, inputs, ts, jt, fouts)
    FD_RESULTS[name], DOT_RESULTS[name] = fd, dot
    assert fd < FD_TOL, f"{name}: finite differences {fd:.2e}"
    assert dot < DOT_TOL, f"{name}: dot product test {dot:.2e}"


def test_every_op_with_a_reverse_rule_has_a_forward_rule_and_a_case():
    reverse = set(GRAD_RULES) | set(LOWERINGS)
    forward = set(jvp_rule_table())
    assert reverse - forward == set()
    covered = set()
    for name in ad.CASES:
        covered |= set(ad.make(name)[0].op_counts())
    assert reverse - covered == set(), "every op with both rules is in the dot product property test"


# ----------------------------------------------------------------------------- ops without a reverse rule


def _region(name, types, body):
    r = ir.Builder(name)
    vals = [r.input(f"a{i}", dt, s) for i, (s, dt) in enumerate(types)]
    out = body(r, *vals)
    return r.build(out if isinstance(out, list) else [out])


FWD_ONLY = {}


def fcase(name, shapes, build, prep=None, dtypes=None):
    FWD_ONLY[name] = (shapes, build, prep, dtypes)


def _pos(i):
    def prep(xs, g):
        xs[i] = xs[i].abs() + 0.5
        return xs
    return prep


fcase("log_sin_cos_reciprocal", [(3, 4)], lambda b, x: [b.op("log", x), b.op("sin", x), b.op("cos", x),
                                                       b.op("reciprocal", x)], _pos(0))
fcase("log_softmax", [(3, 5)], lambda b, x: [b.op("log_softmax", x, dim=-1), b.op("log_softmax", x, dim=0)])
fcase("abs", [(4, 5)], lambda b, x: b.op("abs", x), lambda xs, g: [ad.away_from(xs[0], [0.0])])
fcase("pow_both", [(3, 4), (4,)], lambda b, x, y: b.op("pow", x, y), _pos(0))
fcase("max_min", [(3, 4), (4,)], lambda b, x, y: [b.op("maximum", x, y), b.op("minimum", x, y)])
fcase("structure_ops", [(3, 4), (2, 4)],
      lambda b, x, s: [b.op("cumsum", x, dim=1), b.op("triu", x, diagonal=0), b.op("tril", x, diagonal=1),
                       b.op("pad", x, pads=[1, 1, 2, 0], value=3.0),
                       b.op("slice_scatter", x, s, dim=0, start=0, end=3, step=2),
                       b.op("select_scatter", x, b.op("select", s, dim=0, index=1), dim=0, index=1),
                       b.op("cast", x, dtype=torch.float32)])
fcase("avg_pool", [(2, 2, 6, 7)], lambda b, x: b.op("avg_pool2d", x, kernel_size=3, stride=2, padding=1,
                                                   ceil_mode=False, count_include_pad=True, divisor_override=None))


def _index_case(b, x):
    i = b.param("i", torch.tensor([[0, 2], [1, 1]]))
    k = b.param("k", torch.tensor(2))
    return [b.op("index", x, None, i), b.op("select_dynamic", x, k, dim=1)]


fcase("index_ops", [(3, 4)], _index_case)


def _cond_case(b, x, y):
    then = _region("then", [((3,), F64), ((3,), F64)], lambda r, a, c: [r.op("mul", r.op("tanh", a), c)])
    other = _region("else", [((3,), F64), ((3,), F64)], lambda r, a, c: [r.op("exp", c)])
    pred = b.op("gt", b.op("sum", x, dims=None, keepdim=False), b.op("full", shape=[], value=0.0, dtype=F64))
    return list(b.op_multi("cond", pred, x, y, then_branch=then, else_branch=other))


fcase("cond_then", [(3,), (3,)], _cond_case, lambda xs, g: [xs[0].abs() + 0.1, xs[1]])
fcase("cond_else", [(3,), (3,)], _cond_case, lambda xs, g: [-xs[0].abs() - 0.1, xs[1]])


def _rnn(b, x, w):
    """h = tanh(h w + x[i]) for every row of x: a loop over a symbolic length."""
    body = _region("body", [((), I64), ((4,), F64), (("T", 4), F64), ((4, 4), F64)],
                   lambda r, i, h, xs, ww: [r.op("tanh", r.op("add", r.op("reshape", r.op(
                       "matmul", r.op("reshape", h, shape=[1, 4]), ww), shape=[4]), r.op("select_dynamic", xs, i,
                                                                                       dim=0)))])
    h0 = b.op("full", shape=[4], value=0.1, dtype=F64)
    import sympy
    (h,) = b.op_multi("loop", h0, x, w, n_carried=1, trip_count=sympy.Symbol("T", integer=True, positive=True),
                      body=body)
    return h


fcase("loop_rnn", [("T", 4), (4, 4)], _rnn)


def _while(b, x):
    cond = _region("cond", [((3,), F64)], lambda r, v: r.op("gt", r.op("sum", r.op("mul", v, v), dims=None,
                                                                         keepdim=False),
                                                              r.op("full", shape=[], value=1.0, dtype=F64)))
    body = _region("body", [((), I64), ((3,), F64)],
                   lambda r, i, v: [r.op("mul", r.op("sin", v), r.op("full", shape=[], value=0.9, dtype=F64))])
    (y,) = b.op_multi("loop", x, n_carried=1, cond=cond, body=body)
    return y


fcase("while_loop", [(3,)], _while, lambda xs, g: [xs[0] * 0 + torch.tensor([1.3, -1.1, 0.9], dtype=F64)])
fcase("custom_vjp", [(3, 4), (4,)], ad.custom_vjp_case)


def make_fwd(name, seed=0):
    shapes, build, prep, dtypes = FWD_ONLY[name]
    b = ir.Builder(name)
    vals = [b.input(f"x{i}", F64, s) for i, s in enumerate(shapes)]
    outs = build(b, *vals)
    prog = b.build(outs if isinstance(outs, list) else [outs])
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(tuple(5 if isinstance(d, str) else d for d in s), generator=g, dtype=F64) for s in shapes]
    if prep:
        xs = prep(xs, g)
    return prog, xs


@pytest.mark.parametrize("name", sorted(FWD_ONLY))
def test_forward_only_rules_match_finite_differences(name):
    prog, inputs = make_fwd(name)
    fd, *_ = fd_jvp_error(prog, inputs)
    FD_RESULTS[f"fwd/{name}"] = fd
    assert fd < FD_TOL, f"{name}: {fd:.2e}"


def test_loop_jvp_carries_tangents_and_serves_every_length():
    prog, _ = make_fwd("loop_rnn")
    jp = jvp(prog, ["x0", "x1"])
    (loop,) = [n for n in jp.nodes if n.op == "loop"]
    assert loop.attrs["n_carried"] == 2          # the value and its tangent, nothing stored per step
    # Reference: torch.func.jvp of the same recurrence (exact). Finite differences lose accuracy
    # with the length here (the fifth derivative grows with T: 8.8e-14 at T = 1, 6.8e-9 at T = 8),
    # so the stencil checks the rule at the case length only (test_forward_only_rules...).
    for T in (1, 3, 8, 17):
        g = torch.Generator().manual_seed(T)
        x, w, tx, tw = (torch.randn(s, generator=g, dtype=F64) for s in ((T, 4), (4, 4), (T, 4), (4, 4)))

        def f(xx, ww):
            h = torch.full((4,), 0.1, dtype=F64)
            for i in range(T):
                h = torch.tanh(h @ ww + xx[i])
            return h
        _, want = torch.func.jvp(f, (x, w), (tx, tw))
        assert ad.rel_err(ir.run(jp, x, w, tx, tw)[1], want) < 1e-12


def test_custom_vjp_dot_product_with_its_own_rule():
    """The jvp of a custom_vjp node differentiates the primal; with a correct rule the dot product
    test against the user's backward holds."""
    prog, inputs = ad.make("custom_vjp", 0)
    _, ts, jt, fouts = fd_jvp_error(prog, inputs)
    assert dot_error(prog, inputs, ts, jt, fouts) < DOT_TOL


# ----------------------------------------------------------------------------- the Jacobian family


def _mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3), torch.nn.GELU()).double()


def test_jacfwd_and_jacrev_match_torch_func():
    from torch.func import jacfwd as tjacfwd, jacrev as tjacrev
    m = _mlp()
    x = torch.randn(4, 5, dtype=F64)
    prog = ir.import_module(m, (x,))
    names = [v.name for v in prog.inputs] + ["0.weight"]
    jf = jacfwd(prog, names)
    jr = jacrev(prog, names)
    y, jx, jw = ir.run(jf, x)
    y2, jx2, jw2 = ir.run(jr, x)
    params = {k: v.detach() for k, v in m.named_parameters()}

    def f_x(xx):
        return m(xx)

    def f_w(w):
        p = dict(params)
        p["0.weight"] = w
        return torch.func.functional_call(m, p, (x,))
    want_x = tjacfwd(f_x)(x)
    want_w = tjacrev(f_w)(params["0.weight"])
    assert tuple(jx.shape) == (4, 3, 4, 5) and tuple(jw.shape) == (4, 3, 7, 5)
    for got, want in ((jx, want_x), (jx2, want_x), (jw, want_w), (jw2, want_w)):
        assert ad.rel_err(got, want) < 1e-12


def test_jacfwd_with_a_symbolic_size():
    """The identity is built in the program from arange, so one jacfwd program serves every batch."""
    from torch.export import Dim
    m = _mlp()
    prog = ir.import_module(m, (torch.randn(2, 5, dtype=F64),), dynamic_shapes=({0: Dim("B", min=1, max=64)},))
    jf = jacfwd(prog, [prog.inputs[0].name])
    for B in (1, 3):
        x = torch.randn(B, 5, dtype=F64)
        got = ir.run(jf, x)[1]
        want = torch.func.jacfwd(m)(x)
        assert ad.rel_err(got, want) < 1e-12


class _TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.layer = torch.nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True,
                                                      activation="gelu").double().eval()

    def forward(self, x):
        y = self.layer(x)
        return (y * y).sum() * 0.5 + (y.tanh()).sum()


def test_hessian_of_a_transformer_layer_matches_torch_func():
    """torch.func.hessian (forward over reverse, like ours) raises on this stock layer on the CPU:
    forward AD is not implemented for the CPU flash attention kernel, and reverse over reverse
    (jacrev of jacrev) raises too (no double backward for it). With the math attention backend
    selected by hand torch.func.hessian works and is the reference."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        mod = _TinyLayer()
        x = torch.randn(1, 3, 8, dtype=F64)
        prog = ir.import_module(mod, (x,))
        hp = hessian(prog, [prog.inputs[0].name])
        loss, h = ir.run(hp, x)
        with pytest.raises(NotImplementedError, match="forward AD"):
            torch.func.hessian(mod)(x)
        with pytest.raises(RuntimeError, match="not implemented"):
            torch.func.jacrev(torch.func.jacrev(mod))(x)
        with sdpa_kernel(SDPBackend.MATH):
            want = torch.func.hessian(mod)(x)
        assert tuple(h.shape) == (1, 3, 8, 1, 3, 8)
        err = ad.rel_err(h, want)
        FD_RESULTS["hessian/transformer_layer"] = err
        assert err < 1e-10
        assert ad.rel_err(h.reshape(24, 24), h.reshape(24, 24).T) < 1e-12  # symmetric
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def test_hessian_layout_with_two_wrt_values():
    b = ir.Builder("f")
    x = b.input("x", F64, (3,))
    w = b.input("w", F64, (2, 3))
    y = b.op("sum", b.op("tanh", b.op("matmul", b.op("reshape", x, shape=[1, 3]), b.op("permute", w, dims=[1, 0]))),
             dims=None, keepdim=False)
    prog = b.build([y])
    hp = hessian(prog, ["x", "w"])
    xv, wv = torch.randn(3, dtype=F64), torch.randn(2, 3, dtype=F64)
    out = ir.run(hp, xv, wv)
    want = torch.func.hessian(lambda a, c: torch.tanh(a @ c.T).sum(), argnums=(0, 1))(xv, wv)
    shapes = [tuple(o.shape) for o in out[1:]]
    assert shapes == [(3, 3), (3, 2, 3), (2, 3, 3), (2, 3, 2, 3)]
    for got, wnt in zip(out[1:], [want[0][0], want[0][1], want[1][0], want[1][1]]):
        assert ad.rel_err(got, wnt) < 1e-12


def test_linearize_splits_primal_and_tangent_work():
    m = _mlp()
    x = torch.randn(4, 5, dtype=F64)
    prog = ir.import_module(m, (x,))
    primal, linear = linearize(prog, [prog.inputs[0].name])
    outs = ir.run(primal, x)
    y, res = outs[0], outs[1:]
    assert ad.rel_err(y, m(x).detach()) < 1e-14
    jp = jvp(prog, [prog.inputs[0].name])
    t1, t2 = torch.randn(4, 5, dtype=F64), torch.randn(4, 5, dtype=F64)
    l1 = ir.run(linear, *res, t1)[0]
    l2 = ir.run(linear, *res, t2)[0]
    assert ad.rel_err(l1, ir.run(jp, x, t1)[1]) < 1e-14
    assert ad.rel_err(ir.run(linear, *res, 2.0 * t1 - 3.0 * t2)[0], 2.0 * l1 - 3.0 * l2) < 1e-13  # linear
    primal_ops = {n.output.name for n in prog.nodes}
    assert not any(n.output.name in primal_ops for n in linear.nodes), "no primal work in the linear program"
    assert len(linear.nodes) < len(jp.nodes)


# ----------------------------------------------------------------------------- loud failures and broken rules


def test_op_without_a_forward_rule_is_refused_by_name(monkeypatch):
    # every float op has a forward rule since the op coverage work (2026-11-24), so one is removed
    from truestar.ir import forward
    monkeypatch.delitem(forward.JVP_RULES, "fmod")
    b = ir.Builder("f")
    x = b.input("x", F64, (3,))
    prog = b.build([b.op("fmod", x, b.op("full", shape=[], value=0.7, dtype=F64))])
    with pytest.raises(ir.UnsupportedOpError, match="no forward mode rule for IR op 'fmod'"):
        jvp(prog, ["x"])


def test_bad_requests_fail_clearly():
    prog, _ = ad.make("tanh", 0)
    with pytest.raises(ir.IRError, match="not an input or param"):
        jvp(prog, ["nope"])
    with pytest.raises(ir.IRError, match="not a scalar"):
        hessian(prog, ["x0"])


def _broken_tanh(e, n, ts):
    y = n.output
    return [e.op("mul", ts[0], e.op("sub", e.const(1.0, y), y))]  # 1 - y instead of 1 - y^2


def test_a_broken_rule_fails_both_checks(monkeypatch):
    monkeypatch.setitem(fwd_mod.JVP_RULES, "tanh", _broken_tanh)
    prog, inputs = ad.make("tanh", 0)
    fd, ts, jt, fouts = fd_jvp_error(prog, inputs)
    assert fd > 1e-2
    assert dot_error(prog, inputs, ts, jt, fouts) > 1e-2


def test_a_subtle_break_is_caught_by_the_dot_product_test(monkeypatch):
    """softmax without the sum(t y) term: wrong only off the diagonal of the Jacobian."""
    def broken(e, n, ts):
        return [e.op("mul", n.output, ts[0])]
    monkeypatch.setitem(fwd_mod.JVP_RULES, "softmax", broken)
    prog, inputs = ad.make("softmax_last", 0)
    fd, ts, jt, fouts = fd_jvp_error(prog, inputs)
    assert fd > 1e-2 and dot_error(prog, inputs, ts, jt, fouts) > 1e-2


def test_zz_report(capsys):
    if FD_RESULTS:
        with capsys.disabled():
            print(f"\n[forward] finite differences: {len(FD_RESULTS)} cases, worst {max(FD_RESULTS.values()):.1e}; "
                  f"dot product: {len(DOT_RESULTS)} cases, worst {max(DOT_RESULTS.values(), default=0):.1e}")
