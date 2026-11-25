"""Tests for vmap as a program transformation (truestar.ir.vmap).

The check for every rule is the definition of vmap: the batched program must equal the original
program run once per element (a Python loop of reference runs), output by output. Cases:

1. every program of the reverse mode rule corpus (tests/test_autodiff.py, every differentiable op)
   with every input batched, with only the first input batched, and with the batch on the last axis;
2. cases for the ops that corpus does not reach (comparisons, integer ops, indexing with a batched
   index, per element weights, random sampling, cond and loop with per element control flow);
3. per sample gradients and ensembles against explicit loops, a native function with data
   dependent control flow, symbolic batch sizes, loud failures, and broken rules that must fail.
"""

import math
import warnings

import pytest
import torch

import truestar
import truestar.train  # noqa: F401  registers log_softmax and all_reduce, which must be classified too
import test_autodiff as ad
from truestar import Tensor, ir, ops
from truestar.ir import vmap as vmap_mod
from truestar.ir.autodiff import grad
from truestar.ir.vmap import batching_table, vmap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

F64 = torch.float64
F32 = torch.float32
I64 = torch.int64
N = 3

# Bar. The batched program runs the same float64 kernels as the per element runs; only summation
# order inside a batched matmul or reduction can differ, a few ulp. Measured worst case on
# 2026-11-24: 4.4e-16 over every case. A wrong rule mixes up elements or dims, an error of order 1.
TOL = 1e-12


def rel_err(a, b):
    a, b = a.to(F64), b.to(F64)
    if a.shape != b.shape:
        return math.inf
    scale = max(a.abs().max().item(), b.abs().max().item()) if a.numel() else 0.0
    if scale == 0.0:
        return 0.0
    return (a - b).abs().max().item() / scale


def per_element(prog, args, flags, axes=None):
    """The reference: run ``prog`` once per element, stack the results on dim 0."""
    axes = axes or [0] * len(args)
    outs = []
    for k in range(N):
        one = [a.select(ax, k) if f else a for a, f, ax in zip(args, flags, axes)]
        outs.append(ir.run(prog, *one))
    return [torch.stack([o[j] for o in outs]) for j in range(len(outs[0]))]


def compare(vprog, prog, args, flags, axes=None):
    got = ir.run(vprog, *args)
    want = per_element(prog, args, flags, axes)
    assert len(got) == len(want)
    worst = 0.0
    for g, w in zip(got, want):
        assert tuple(g.shape) == tuple(w.shape)
        if not g.dtype.is_floating_point:
            assert torch.equal(g, w)
            continue
        worst = max(worst, rel_err(g, w))
    return worst


RESULTS = {}


# ----------------------------------------------------------------------------- 1. the rule corpus


@pytest.mark.parametrize("mode", ["all", "first", "last_axis"])
@pytest.mark.parametrize("name", sorted(ad.CASES))
def test_vmap_matches_per_element_loop_on_the_rule_corpus(name, mode):
    prog, base = ad.make(name, 0)
    per = [ad.make(name, k)[1] for k in range(N)]  # each element gets valid inputs from the case's prep
    n_in = len(base)
    flags = [True] * n_in if mode == "all" else [i == 0 for i in range(n_in)]
    axes = [(-1 if mode == "last_axis" and i == 0 else 0) for i in range(n_in)]
    in_axes = {prog.inputs[i].name: (axes[i] % (base[i].dim() + 1)) for i in range(n_in) if flags[i]}
    vp = vmap(prog, in_axes=in_axes)
    args = [torch.stack([per[k][i] for k in range(N)], dim=axes[i] % (base[i].dim() + 1)) if flags[i] else base[i]
            for i in range(n_in)]
    worst = compare(vp, prog, args, flags, [a % (base[i].dim() + 1) for i, a in enumerate(axes)])
    RESULTS[f"{name}/{mode}"] = worst
    assert worst < TOL, f"{name} ({mode}): {worst:.2e}"


# ----------------------------------------------------------------------------- 2. the other ops


EXTRA = {}


def extra(name, specs, build, flags=None):
    """specs: one generator per input, gen -> tensor (called once per element and once unbatched)."""
    EXTRA[name] = (specs, build, flags)


def _shape(shape):
    return shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape


def randn(*shape, dtype=F64):
    shape = _shape(shape)
    return lambda g: torch.randn(shape, generator=g, dtype=F64).to(dtype)


def randint(hi, *shape):
    shape = _shape(shape)
    return lambda g: torch.randint(0, hi, shape, generator=g)


def perm(n):
    return lambda g: torch.randperm(n, generator=g)[:2]


def positive(*shape):
    shape = _shape(shape)
    return lambda g: torch.rand(shape, generator=g, dtype=F64) + 0.5


def boolean(*shape):
    shape = _shape(shape)
    return lambda g: torch.rand(shape, generator=g) < 0.5


def region(name, types, body):
    r = ir.Builder(name)
    vals = [r.input(f"a{i}", dt, s) for i, (s, dt) in enumerate(types)]
    out = body(r, *vals)
    return r.build(out if isinstance(out, list) else [out])


extra("frozen", [randn(3, 4)], lambda b, x: b.op("mul", x, b.op("frozen", x)))
extra("compare_logic", [randn(3, 4), randn(4)],
      lambda b, x, y: [b.op("lt", x, y), b.op("logical_and", b.op("ge", x, y), b.op("logical_not", b.op("eq", x, y))),
                       b.op("logical_or", b.op("le", x, y), b.op("ne", x, y)),
                       b.op("any", b.op("gt", x, y), dims=[1], keepdim=False),
                       b.op("full_like", x, value=2.0, dtype=None)])
extra("where_batched_cond", [boolean(3, 4), randn(3, 4), randn(())],
      lambda b, c, x, y: b.op("where", c, x, y))
extra("scalar_promotion_f32", [randn(3, 4, dtype=F32), randn(())],
      lambda b, x, s: b.op("add", b.op("mul", x, s), s))
extra("int_ops", [randint(9, 3, 4), randint(4, 4)],
      lambda b, x, y: [b.op("fmod", x, b.op("add", y, b.op("full", shape=[], value=1, dtype=I64))),
                       b.op("remainder", x, b.op("full", shape=[], value=3, dtype=I64)),
                       b.op("maximum", x, y), b.op("minimum", x, y)])
extra("pow_log_trig", [positive(3, 4), randn(4)],
      lambda b, x, y: [b.op("pow", x, y), b.op("log", x), b.op("sin", y), b.op("cos", x), b.op("reciprocal", x),
                       b.op("abs", y), b.op("cast", b.op("mul", y, b.op("full", shape=[], value=3.0, dtype=F64)),
                                                dtype=I64)])
extra("arg_reductions", [randn(3, 5)],
      lambda b, x: [b.op("argmax", x, dim=None, keepdim=True), b.op("argmin", x, dim=1, keepdim=False),
                    b.op("argmax", x, dim=0, keepdim=True)])
extra("cumsum_tri_pad", [randn(3, 4)],
      lambda b, x: [b.op("cumsum", x, dim=0), b.op("triu", x, diagonal=1), b.op("tril", x, diagonal=-1),
                    b.op("pad", x, pads=[1, 2, 0, 1], value=0.5)])
extra("norms", [randn(2, 4, 3), randn(4), randn(4)],
      lambda b, x, w, c: [b.op("vector_norm", x, ord=3.0, dims=[1], keepdim=False),
                          b.op("group_norm", x, w, c, num_groups=2, eps=1e-5),
                          b.op("rms_norm", x, b.op("slice", w, dim=0, start=0, end=3, step=1),
                               normalized_shape=[3], eps=1e-6)])
extra("pools", [randn(2, 6, 7), randn(1, 2, 6, 6)],
      lambda b, x, y: [b.op("avg_pool2d", x, kernel_size=3, stride=2, padding=1, ceil_mode=False,
                            count_include_pad=False, divisor_override=None),
                       b.op("adaptive_max_pool2d", y, output_size=[4, 3]),
                       b.op("max_pool2d", y, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)])
extra("log_softmax", [randn(3, 5)], lambda b, x: [b.op("log_softmax", x, dim=1), b.op("log_softmax", x, dim=0)])
extra("rank0_reductions", [randn(())],
      lambda b, x: [b.op("sum", x, dims=None, keepdim=False), b.op("softmax", x, dim=0),
                    b.op_multi("sort", x, dim=-1, descending=False)[0], b.op("argmax", x, dim=None, keepdim=False),
                    b.op("flip", x, dims=[0])])
# indexing with a batched index
extra("index_adjacent_from_0", [randn(5, 4, 3), randint(5, 2, 3), randint(4, 3)],
      lambda b, x, i, j: b.op("index", x, i, j))
extra("index_adjacent_later", [randn(2, 5, 4, 3), randint(5, 6), randint(4, 6)],
      lambda b, x, i, j: b.op("index", x, None, i, j))
extra("index_split", [randn(5, 2, 4, 3), randint(5, 3), randint(4, 3)],
      lambda b, x, i, j: b.op("index", x, i, None, j))
extra("index_select_batched_index", [randn(4, 5, 3), randint(5, 6), randint(4, ())],
      lambda b, x, i, j: [b.op("index_select", x, i, dim=1), b.op("index_select", x, j, dim=0)])
extra("select_dynamic_batched_index", [randn(4, 5), randint(4, ()), randint(5, ())],
      lambda b, x, i, j: [b.op("select_dynamic", x, i, dim=0), b.op("select_dynamic", x, j, dim=1)])
extra("gather_batched_index", [randn(3, 5), randint(5, 3, 4)], lambda b, x, i: b.op("gather", x, i, dim=1))
extra("index_copy_batched_index", [randn(5, 3), perm(5), randn(2, 3)],
      lambda b, x, i, s: b.op("index_copy", x, i, s, dim=0))
extra("scatter_views", [randn(4, 5), randn(4, 2), randn(5)],
      lambda b, x, s, r: [b.op("slice_scatter", x, s, dim=1, start=1, end=5, step=2),
                          b.op("select_scatter", x, r, dim=0, index=2)])
# per element weights
extra("linear_batched_weight", [randn(2, 5), randn(3, 5), randn(3)],
      lambda b, x, w, c: b.op("linear", x, w, c), flags=[False, True, True])
extra("linear_rank1_batched_weight", [randn(5), randn(3, 5), randn(3)],
      lambda b, x, w, c: b.op("linear", x, w, c), flags=[True, True, False])
extra("linear_batched_bias_only", [randn(2, 5), randn(3, 5), randn(3)],
      lambda b, x, w, c: b.op("linear", x, w, c), flags=[False, False, True])
extra("conv2d_batched_weight", [randn(2, 4, 6, 6), randn(6, 2, 3, 3), randn(6)],
      lambda b, x, w, c: b.op("conv2d", x, w, c, stride=1, padding=1, dilation=1, groups=2))
extra("batch_norm_batched_stats", [randn(2, 3, 4), randn(3), randn(3), randn(3), positive(3)],
      lambda b, x, w, c, m, v: b.op("batch_norm", x, w, c, m, v, eps=1e-5), flags=[False, True, False, True, True])
extra("layer_norm_batched_weight", [randn(2, 5), randn(5), randn(5)],
      lambda b, x, w, c: b.op("layer_norm", x, w, c, normalized_shape=[5], eps=1e-5), flags=[True, True, False])
extra("norms_batched_weight", [randn(2, 4, 3), randn(4), randn(3)],
      lambda b, x, w, r: [b.op("group_norm", x, w, None, num_groups=2, eps=1e-5),
                          b.op("rms_norm", x, r, normalized_shape=[3], eps=1e-6)], flags=[False, True, True])
extra("attention_batched_mask_only", [randn(2, 4, 6), randn(2, 5, 6), randn(2, 5, 3), randn(4, 5)],
      lambda b, q, k, v, m: b.op("attention", q, k, v, m, scale=None, is_causal=False),
      flags=[False, False, False, True])
extra("attention_batched_k_only", [randn(2, 4, 6), randn(2, 5, 6), randn(2, 5, 3)],
      lambda b, q, k, v: b.op("attention", q, k, v, None, scale=0.5, is_causal=False), flags=[False, True, False])
extra("attention_grad_batched_do_only", [randn(2, 4, 6)] * 5,
      lambda b, q, k, v, o, do: b.op("attention_grad", q, k, v, o, do, scale=0.5, is_causal=False),
      flags=[False, False, False, False, True])
extra("creation_ops", [randn(3, 4)],
      lambda b, x: b.op("add", x, b.op("cast", b.op("reshape", b.op("arange", start=0, end=4, step=1, dtype=I64),
                                                           shape=[1, 4]), dtype=F64)))


def _cond_case(b, p, x, y):
    then = region("then", [((3,), F64), ((3,), F64)], lambda r, a, c: [r.op("mul", a, c), a])
    other = region("else", [((3,), F64), ((3,), F64)], lambda r, a, c: [r.op("sub", a, c), c])
    pred = b.op("gt", b.op("sum", p, dims=None, keepdim=False), b.op("full", shape=[], value=0.0, dtype=F64))
    return list(b.op_multi("cond", pred, x, y, then_branch=then, else_branch=other))


extra("cond_batched_predicate", [randn(2), randn(3), randn(3)], _cond_case)
extra("cond_unbatched_predicate", [randn(2), randn(3), randn(3)], _cond_case, flags=[False, True, False])


def _trip_loop(b, x, w):
    body = region("body", [((), I64), ((3,), F64), ((3,), F64), ((3,), F64)],
                  lambda r, i, h, acc, ww: [r.op("tanh", r.op("add", r.op("mul", h, ww), acc)),
                                            r.op("add", acc, h)])
    zero = b.op("full", shape=[3], value=0.0, dtype=F64)
    return list(b.op_multi("loop", zero, x, w, n_carried=2, trip_count=4, body=body))


extra("loop_carried_becomes_batched", [randn(3), randn(3)], _trip_loop, flags=[False, True])


def _while_loop(b, x, lim):
    """Halve until the norm is below lim: a different number of steps per element."""
    cond = region("cond", [((3,), F64), ((), I64), ((), F64)],
                  lambda r, v, k, l: r.op("gt", r.op("sum", r.op("mul", v, v), dims=None, keepdim=False), l))
    body = region("body", [((), I64), ((3,), F64), ((), I64), ((), F64)],
                  lambda r, i, v, k, l: [r.op("mul", v, r.op("full", shape=[], value=0.5, dtype=F64)),
                                         r.op("add", k, i)])
    k0 = b.op("full", shape=[], value=0, dtype=I64)
    return list(b.op_multi("loop", x, k0, lim, n_carried=2, cond=cond, body=body))


extra("while_per_element_trip_counts",
      [lambda g: torch.randn(3, generator=g, dtype=F64) * torch.exp(3 * torch.randn(1, generator=g, dtype=F64)),
       positive(())],
      _while_loop)
extra("custom_vjp_batched", [randn(3, 4), randn(4)], ad.custom_vjp_case, flags=[True, False])
# quantized KV decode attention (truestar.kvquant): int8 caches with per token scale groups of 2 and a
# live float tail; every operand batched, the tail length too (each element reads its own tail)
extra("kvq_attention_batched_cache",
      [randn(1, 2, 1, 4), lambda g: torch.randint(-127, 128, (1, 1, 6, 4), generator=g).to(torch.int8),
       positive(1, 1, 3), lambda g: torch.randint(-127, 128, (1, 1, 6, 4), generator=g).to(torch.int8),
       positive(1, 1, 3), randn(1, 1, 2, 4), randn(1, 1, 2, 4), lambda g: torch.randint(0, 3, (1,), generator=g)],
      lambda b, q, kd, ks, vd, vs, tk, tv, tl: b.op("kvq_attention", q, kd, ks, vd, vs, tk, tv, tl, None,
                                                     fmt="int8", group=2, vgroup=2, channel=False, scale=None))
# fused MLA latent attention (truestar.mla.latent): every operand batched, the position too (each
# element attends over its own cache prefix)
extra("mla_attention_batched",
      [randn(1, 2, 1, 3), randn(1, 2, 1, 2), randn(1, 5, 4), randn(1, 5, 2), randn(1, 1, 4), randn(1, 1, 2),
       lambda g: torch.randint(0, 6, (1,), generator=g), randn(2, 3, 4), randn(2, 3, 4)],
      lambda b, qn, qr, cc, rc, lat, kr, pos, wk, wv: b.op("mla_attention", qn, qr, cc, rc, lat, kr, pos, wk, wv,
                                                          scale=0.5))

# the derivative ops of linear_attention (P60): the cotangent of g and the tangent of o, every operand
# batched, the per element loop against the op alone
_LA = [randn(1, 4, 2, 3), randn(1, 4, 2, 3), randn(1, 4, 2, 2),
       lambda g: -torch.rand((1, 4, 2), generator=g, dtype=F64) * 0.3, lambda g: torch.rand((1, 4, 2), generator=g, dtype=F64)]
extra("linear_attention_vjp_batched", _LA + [randn(1, 4, 2, 2)],
      lambda b, q, k, v, g, beta, do: b.op("linear_attention_vjp", q, k, v, g, beta, do, rule="delta", scale=1.0, wrt=3))
extra("linear_attention_jvp_batched", _LA + [randn(1, 4, 2, 3), randn(1, 4, 2, 3), randn(1, 4, 2, 2), randn(1, 4, 2),
                                             randn(1, 4, 2)],
      lambda b, q, k, v, g, beta, tq, tk, tv, tg, tb: b.op("linear_attention_jvp", q, k, v, g, beta, tq, tk, tv, tg, tb,
                                                           rule="delta", scale=1.0))


def make_extra(name):
    specs, build, flags = EXTRA[name]
    b = ir.Builder(name)
    gens = [torch.Generator().manual_seed(k) for k in range(N + 1)]
    samples = [[s(g) for s in specs] for g in gens]
    vals = [b.input(f"x{i}", t.dtype, tuple(t.shape)) for i, t in enumerate(samples[0])]
    outs = build(b, *vals)
    prog = b.build(outs if isinstance(outs, list) else [outs])
    flags = flags or [True] * len(specs)
    args = [torch.stack([samples[k + 1][i] for k in range(N)]) if f else samples[0][i]
            for i, f in enumerate(flags)]
    return prog, args, flags


@pytest.mark.parametrize("name", sorted(EXTRA))
def test_vmap_matches_per_element_loop_on_other_ops(name):
    prog, args, flags = make_extra(name)
    vp = vmap(prog, in_axes={v.name: 0 for v, f in zip(prog.inputs, flags) if f})
    worst = compare(vp, prog, args, flags)
    RESULTS[f"extra/{name}"] = worst
    assert worst < TOL, f"{name}: {worst:.2e}"


def test_scalar_promotion_keeps_the_original_dtype():
    """f32[3, 4] * f64[] is f32 (a 0 dim operand does not widen). Batched, the scalar is f64[V],
    which would widen; the pass casts it so every output keeps its original dtype."""
    prog, args, flags = make_extra("scalar_promotion_f32")
    vp = vmap(prog)
    assert vp.outputs[0].type.dtype == F32
    assert "cast" in vp.op_counts()


def test_every_ir_op_has_a_batching_rule():
    table = batching_table()
    assert set(table) == set(ir.ir_ops())
    assert [op for op, how in table.items() if how == "unclassified"] == []
    assert not any(how.startswith("undefined") for how in table.values())
    covered = set()
    for name in ad.CASES:
        covered |= set(ad.make(name)[0].op_counts())
    for name in EXTRA:
        p = make_extra(name)[0]
        covered |= set(p.op_counts())
        for r in p.regions():
            covered |= set(r.op_counts())
    covered |= {"multinomial", "dim_scalar", "custom_vjp", "random", "random_batched"}  # tested below
    covered |= {"all_reduce"}  # needs torch.distributed to run; entrywise rule, not run here
    assert set(table) - covered == set(), f"ops without a vmap test: {sorted(set(table) - covered)}"


def test_while_loop_runs_each_element_to_its_own_end():
    prog, args, flags = make_extra("while_per_element_trip_counts")
    counts = [int(o[1]) for o in [ir.run(prog, args[0][k], args[1][k]) for k in range(N)]]
    assert len(set(counts)) > 1, "the case must give different trip counts per element"
    vp = vmap(prog)
    got = ir.run(vp, *args)
    assert [int(c) for c in got[1]] == counts


# ----------------------------------------------------------------------------- randomness


def _sampler(with_offset):
    b = ir.Builder("sampler")
    p = b.input("p", F64, (2, 5))
    ins = [p]
    if with_offset:
        ins.append(b.input("offset", I64, ()))
    y = b.op("multinomial", *ins, seed=7, num_samples=4, replacement=True)
    return b.build([y])


def test_random_ops_give_each_element_its_own_reproducible_stream():
    """Element k draws what the op draws alone with offset + k * 2**40: independent streams, each
    reproducible one element at a time; the same seed outside vmap is shared on purpose."""
    weights = torch.rand(2, 5, dtype=F64) + 0.1
    vp = vmap(_sampler(False), in_axes=None, size=N)  # nothing batched: draws still differ per element
    got = ir.run(vp, weights)[0]
    alone = _sampler(True)
    for k in range(N):
        want = ir.run(alone, weights, torch.tensor(k * vmap_mod.RANDOM_STRIDE))[0]
        assert torch.equal(got[k], want)
    assert not all(torch.equal(got[0], got[k]) for k in range(1, N))
    # with a run time offset: element k uses offset + k * 2**40
    vp2 = vmap(alone, in_axes={"p": 0})
    batch_w = torch.rand(N, 2, 5, dtype=F64) + 0.1
    got2 = ir.run(vp2, batch_w, torch.tensor(11))[0]
    for k in range(N):
        assert torch.equal(got2[k], ir.run(alone, batch_w[k], torch.tensor(11 + k * vmap_mod.RANDOM_STRIDE))[0])


# ----------------------------------------------------------------------------- symbolic sizes, axes


def test_one_program_serves_every_batch_size():
    prog, _ = ad.make("chain_two_outputs", 0)
    vp = vmap(prog)
    assert str(vp.inputs[0].type).startswith("f64[V,")
    for size in (1, 2, 5):
        xs = torch.randn(size, 3, 5, dtype=F64)
        w = torch.randn(4, 5, dtype=F64)
        got = ir.run(vmap(prog, in_axes={"x0": 0}), xs, w)
        want = [torch.stack([ir.run(prog, xs[k], w)[j] for k in range(size)]) for j in range(2)]
        assert max(rel_err(g, w_) for g, w_ in zip(got, want)) < TOL


def test_in_axes_out_axes_and_fixed_size():
    prog, _ = ad.make("matmul", 0)                # [2, 3, 4] @ [2, 4, 5]
    vp = vmap(prog, in_axes=[1, None], out_axes=2, size=6)
    assert [str(v.type) for v in vp.inputs] == ["f64[2, 6, 3, 4]", "f64[2, 4, 5]"]
    assert str(vp.outputs[0].type) == "f64[2, 3, 6, 5]"
    x, y = torch.randn(2, 6, 3, 4, dtype=F64), torch.randn(2, 4, 5, dtype=F64)
    got = ir.run(vp, x, y)[0]
    want = torch.stack([ir.run(prog, x[:, k], y)[0] for k in range(6)], dim=2)
    assert rel_err(got, want) < TOL


def test_ensemble_over_params():
    """A param named in in_axes becomes an input holding one copy per member."""
    b = ir.Builder("mlp")
    x = b.input("x", F64, ("B", 4))
    w = b.param("w", torch.randn(3, 4, dtype=F64))
    c = b.param("c", torch.randn(3, dtype=F64))
    prog = b.build([b.op("tanh", b.op("linear", x, w, c))])
    vp = vmap(prog, in_axes={"w": 0, "c": 0})
    assert [v.name for v in vp.inputs] == ["x", "w", "c"] and not vp.params
    xs, ws, cs = torch.randn(5, 4, dtype=F64), torch.randn(N, 3, 4, dtype=F64), torch.randn(N, 3, dtype=F64)
    got = ir.run(vp, xs, ws, cs)[0]
    want = torch.stack([torch.tanh(xs @ ws[k].T + cs[k]) for k in range(N)])
    assert rel_err(got, want) < TOL


# ----------------------------------------------------------------------------- per sample gradients


def _loss_program():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3)).double()

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, x, y):
            d = self.m(x) - y
            return (d * d).mean()
    mod = Loss()
    prog = ir.import_module(mod, (torch.randn(6, dtype=F64), torch.randn(3, dtype=F64)))
    return mod, prog


def test_per_sample_gradients_match_an_explicit_loop_and_torch_func():
    mod, prog = _loss_program()
    names = [n for n, _ in mod.named_parameters()]
    ps = vmap(grad(prog, wrt=names), in_axes={prog.inputs[0].name: 0, prog.inputs[1].name: 0})
    xs, ys = torch.randn(7, 6, dtype=F64), torch.randn(7, 3, dtype=F64)
    got = ir.run(ps, xs, ys)
    loop = []
    for k in range(7):
        mod.zero_grad()
        loss = mod(xs[k], ys[k])
        loop.append(torch.autograd.grad(loss, list(mod.parameters())))
    for j in range(len(names)):
        want = torch.stack([g[j] for g in loop])
        assert rel_err(got[1 + j], want) < 1e-12
    from torch.func import functional_call, grad as tgrad, vmap as tvmap
    params = {k: v.detach() for k, v in mod.named_parameters()}

    def loss_fn(p, x, y):
        return functional_call(mod, p, (x, y))
    tf = tvmap(tgrad(loss_fn), in_dims=(None, 0, 0))(params, xs, ys)
    for j, n in enumerate(names):
        assert rel_err(got[1 + j], tf[n]) < 1e-12


# ----------------------------------------------------------------------------- native functions


@truestar.function
def branchy(x: Tensor[4]) -> Tensor[4]:
    if x.sum() > 0:
        x = ops.relu(x) * 2.0
    else:
        x = ops.tanh(x)
    while (x * x).sum() > 1.0:
        x = x * 0.5
    return x


def test_vmap_of_a_native_function_with_data_dependent_control_flow():
    """The case torch.func.vmap refuses (a Python if and while on tensor data)."""
    xs = torch.randn(6, 4, dtype=F64) * 3
    xs[0] = -xs[0].abs()
    xs[1] = xs[1].abs()
    xs = xs.float().double()  # the function takes f32: give both sides the same values
    vp = vmap(branchy)
    got = ir.run(vp, xs)[0]
    want = torch.stack([branchy(xs[k].float()).to(F64) for k in range(6)])
    assert rel_err(got, want) < TOL
    from torch.func import vmap as tvmap

    def torch_branchy(x):
        if x.sum() > 0:
            x = torch.relu(x) * 2.0
        else:
            x = torch.tanh(x)
        while (x * x).sum() > 1.0:
            x = x * 0.5
        return x
    with pytest.raises(RuntimeError, match="data-dependent control flow"):
        tvmap(torch_branchy)(xs)


# ----------------------------------------------------------------------------- loud failures


def test_bad_requests_fail_clearly():
    prog, _ = ad.make("tanh", 0)
    with pytest.raises(ir.IRError, match="no input is batched"):
        vmap(prog, in_axes=None)
    with pytest.raises(ir.IRError, match="depends on a batched input"):
        vmap(prog, out_axes=None)
    with pytest.raises(ir.IRError, match="not inputs or params"):
        vmap(prog, in_axes={"nope": 0})
    with pytest.raises(ir.IRError, match="out of range"):
        vmap(prog, in_axes=5)


def test_unclassified_op_is_refused_by_name(monkeypatch):
    monkeypatch.delitem(vmap_mod.BATCH_RULES, "tanh")
    prog, _ = ad.make("tanh", 0)
    with pytest.raises(ir.UnsupportedOpError, match="no batching rule for IR op 'tanh'"):
        vmap(prog)


def test_undefined_op_is_refused_with_the_reason(monkeypatch):
    monkeypatch.setitem(vmap_mod.UNDEFINED, "tanh", "pretend tanh has no per element meaning")
    prog, _ = ad.make("tanh", 0)
    with pytest.raises(ir.UnsupportedOpError, match="pretend tanh"):
        vmap(prog)


# ----------------------------------------------------------------------------- the tests can fail


def test_a_rule_that_forgets_to_move_the_dim_is_caught(monkeypatch):
    def broken(bt, n, ins):
        return [(bt.e.op("softmax", ins[0][0], dim=n.attrs["dim"]), True)]
    monkeypatch.setitem(vmap_mod.BATCH_RULES, "softmax", broken)
    prog, _ = ad.make("softmax_mid", 0)
    per = [ad.make("softmax_mid", k)[1] for k in range(N)]
    args = [torch.stack([p[0] for p in per])]
    assert compare(vmap(prog), prog, args, [True]) > 1e-2


def test_a_rule_that_mixes_elements_is_caught(monkeypatch):
    """Reshaping [n, b, ...] as [b, n, ...] before a conv type checks when n == b but mixes elements."""
    def broken(bt, n, ins):
        x = ins[0][0]
        s = list(x.type.shape)
        xm = bt.e.op("reshape", bt.e.op("permute", x, dims=[1, 0, 2, 3, 4]), shape=[s[0] * s[1]] + s[2:])
        y = bt.e.op("conv2d", xm, *[None if p is None else p[0] for p in ins[1:]], **n.attrs)
        return [(bt.split(y, n.inputs[0].type.shape[0]), True)]
    monkeypatch.setitem(vmap_mod.BATCH_RULES, "conv2d", broken)
    b = ir.Builder("conv")
    x = b.input("x", F64, (N, 2, 5, 5))
    w = b.param("w", torch.randn(3, 2, 3, 3, dtype=F64))
    prog = b.build([b.op("conv2d", x, w, None, stride=1, padding=1, dilation=1, groups=1)])
    args = [torch.randn(N, N, 2, 5, 5, dtype=F64)]
    assert compare(vmap(prog, size=N), prog, args, [True]) > 1e-2


def test_a_rule_with_a_wrong_type_is_an_internal_error(monkeypatch):
    monkeypatch.setitem(vmap_mod.BATCH_RULES, "sum",
                        lambda bt, n, ins: [(bt.e.op("sum", ins[0][0], dims=None, keepdim=False), True)])
    prog, _ = ad.make("sum_dim", 0)
    with pytest.raises(ir.IRError, match="batching rule of sum"):
        vmap(prog)


def test_zz_report(capsys):
    if RESULTS:
        worst = max(RESULTS.values())
        with capsys.disabled():
            print(f"\n[vmap] {len(RESULTS)} cases, worst relative error against the per element loop {worst:.1e}")


def _branch_with_a_loop(b, x):
    """jax#2947: the then branch shrinks x toward 0 and only ends for x > 0 (for x <= 0 the loop
    never ends); the else branch ends for every x < 0."""
    def shrink(name, test):
        cond = region("c", [((), F64)], test)
        body = region("b", [((), I64), ((), F64)],
                      lambda r, i, v: [r.op("mul", v, r.op("full", shape=[], value=0.1, dtype=F64))])
        return region(name, [((), F64)], lambda r, v: list(r.op_multi("loop", v, n_carried=1, cond=cond, body=body,
                                                                       max_iterations=1000)))
    zero = lambda r: r.op("full", shape=[], value=0.0, dtype=F64)  # noqa: E731
    tiny = lambda r: r.op("full", shape=[], value=1e-6, dtype=F64)  # noqa: E731
    then = shrink("positive", lambda r, v: r.op("logical_or", r.op("le", v, zero(r)), r.op("gt", v, tiny(r))))
    other = shrink("negative", lambda r, v: r.op("lt", v, r.op("neg", tiny(r))))
    pred = b.op("ge", x, b.op("full", shape=[], value=0.0, dtype=F64))
    return list(b.op_multi("cond", pred, x, then_branch=then, else_branch=other))


def test_a_branch_no_element_takes_is_not_run():
    """JAX turns a batched cond into a select that runs both branches, so this never terminates
    (jax#2947). Here each branch is guarded by 'some element takes it': a batch that agrees runs
    one branch only and matches the loop. (A batch that disagrees still runs the positive branch on
    the negative elements, and that loop hits max_iterations: an error, never a hang.)"""
    b = ir.Builder("jax2947")
    x = b.input("x", F64, ())
    prog = b.build(_branch_with_a_loop(b, x))
    vp = vmap(prog)
    for xs in (torch.tensor([0.5, 3.0, 20.0], dtype=F64), torch.tensor([-0.5, -3.0, -20.0], dtype=F64)):
        got = ir.run(vp, xs)[0]
        want = torch.stack([ir.run(prog, v)[0] for v in xs])
        assert rel_err(got, want) < TOL
    with pytest.raises(ir.IRError, match="max_iterations"):
        ir.run(vp, torch.tensor([0.5, -0.5, 2.0], dtype=F64))


# ----------------------------------------------------------------------------- the keyed random op


from truestar import rng  # noqa: E402

KEYED = [("uniform", torch.float32, {}), ("normal", torch.float64, {}), ("normal", torch.float32, {}),
         ("bernoulli", torch.bool, {"p": 0.3}), ("randint", torch.int64, {"low": -5, "high": 17})]


def _keyed(dist, dtype, kw, shape=(4, 6), extra_stream=False):
    """y = random(key, s0[, s1]) plus a value that uses it, so the draw is not the only node."""
    b = ir.Builder("keyed")
    key = b.input("key", I64, ())
    s0 = b.input("s0", I64, ())
    streams = [s0] + ([b.input("s1", I64, ())] if extra_stream else [])
    y = b.op("random", key, *streams, dist=dist, shape=list(shape), dtype=dtype, site=rng.site_id("noise"),
             site_name="noise", **kw)
    return b.build([y])


@pytest.mark.parametrize("dist,dtype,kw", KEYED, ids=[f"{d}-{str(t)[6:]}" for d, t, _ in KEYED])
def test_keyed_random_is_one_vectorized_op_and_each_element_is_the_op_alone(dist, dtype, kw):
    """Element k draws exactly random(key, s0, k): the batch index is one more stream, as for a
    loop. One random_batched op, no loop, bitwise equal to each element run alone."""
    prog = _keyed(dist, dtype, kw)
    vp = vmap(prog, in_axes=None, size=5)
    assert vp.op_counts() == {"random_batched": 1}
    key, s0 = torch.tensor(rng.as_i64(0xDEADBEEF12345678)), torch.tensor(3)
    got = ir.run(vp, key, s0)[0]
    alone = _keyed(dist, dtype, kw, extra_stream=True)
    for k in range(5):
        assert torch.equal(got[k], ir.run(alone, key, s0, torch.tensor(k))[0])
    # and the same bits as the generator's own definition, element by element (python backend)
    want = rng.generate(dist, (4, 6), dtype, int(key), "noise", [3, 2], backend="python", **kw)
    assert torch.equal(got[2].to(want.dtype), want)


def test_keyed_random_elements_are_independent():
    prog = _keyed("normal", torch.float64, {}, shape=(50000,))
    got = ir.run(vmap(prog, in_axes=None, size=4), torch.tensor(7), torch.tensor(0))[0]
    assert len({tuple(r[:8].tolist()) for r in got}) == 4
    c = torch.corrcoef(got)
    off = (c - torch.eye(4, dtype=F64)).abs().max().item()
    assert off < 0.02, f"rows correlate: {off:.3f}"     # 1 / sqrt(50000) = 0.0045
    assert abs(got.mean().item()) < 0.01 and abs(got.std().item() - 1.0) < 0.01


def test_keyed_random_with_a_batched_key_and_vmap_of_vmap_keep_the_definition():
    """A per element key goes through the exact per element loop with the same rule (the index is
    one more stream); vmap of vmap appends both indices, outer first."""
    prog = _keyed("uniform", torch.float32, {})
    keys = torch.tensor([11, 12, 13])
    got = ir.run(vmap(prog, in_axes={"key": 0}), keys, torch.tensor(1))[0]
    alone = _keyed("uniform", torch.float32, {}, extra_stream=True)
    for k in range(3):
        assert torch.equal(got[k], ir.run(alone, keys[k], torch.tensor(1), torch.tensor(k))[0])
    vv = vmap(vmap(prog, in_axes=None, size=2), in_axes=None, size=3)
    got2 = ir.run(vv, torch.tensor(5), torch.tensor(9))[0]
    assert tuple(got2.shape) == (3, 2, 4, 6)
    for j in range(3):
        for k in range(2):
            want = rng.generate("uniform", (4, 6), torch.float32, 5, "noise", [9, j, k], backend="torch")
            assert torch.equal(got2[j, k].float(), want)


def test_generate_batch_matches_every_backend_row_by_row():
    rows = rng.generate_batch("normal", (3, 1000), torch.float32, 99, "s", [4])
    for backend in rng.available_backends("cpu"):
        for k in range(3):
            assert torch.equal(rows[k], rng.generate("normal", (1000,), torch.float32, 99, "s", [4, k], backend=backend))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dist,dtype,kw", KEYED, ids=[f"{d}-{str(t)[6:]}" for d, t, _ in KEYED])
def test_keyed_random_under_vmap_is_bitwise_equal_on_cpu_and_cuda(dist, dtype, kw):
    from truestar.compiler import CompiledExecutor
    prog = _keyed(dist, dtype, kw, shape=(64, 33))
    vp = vmap(prog, in_axes=None, size=7)
    key, s0 = torch.tensor(rng.as_i64(0x0123456789ABCDEF)), torch.tensor(2)
    cpu = ir.run(vp, key, s0)[0]
    cuda = ir.run(vp, key, s0, device="cuda")[0].cpu()
    assert torch.equal(cpu, cuda)
    ref_rows = rng.generate_batch(dist, (7, 64, 33), dtype, int(key), "noise", [2], device="cuda", **kw).cpu()
    assert torch.equal(cpu.to(ref_rows.dtype), ref_rows)
    for k in range(7):
        tri = rng.generate(dist, (64, 33), dtype, int(key), "noise", [2, k], device="cuda", **kw).cpu()
        assert torch.equal(ref_rows[k], tri)  # the Triton kernel's bits, row by row
    if dtype in (torch.float32, torch.int64, torch.bool):
        for dev in ("cpu", "cuda"):
            cp = CompiledExecutor(dev, "float32").prepare(vp)
            out = cp([key, s0])[0].cpu()
            assert torch.equal(out.to(ref_rows.dtype), ref_rows), dev
