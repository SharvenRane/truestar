"""The Metal backend (docs/METAL.md): generated Metal Shading Language, run on the MPS device.

Off macOS the tests check the generated source, the plans, the launch arithmetic and the refusals
(TS840 no Metal, TS841 float64, TS842 sentinels). On a Mac with MPS every kernel is compiled and run,
checked against float64 by the kernel layer on its first call (as on CUDA) and again here, run twice
for bitwise determinism, and the compiled executor runs an MLP and a TransformerEncoderLayer end to
end against float64 and against MPS eager.
"""

import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import ir as kir  # noqa: E402
from truestar.kernel import metal_backend as mb  # noqa: E402
from truestar.kernel import reference  # noqa: E402

HAS_METAL = mb.available()
WHY = mb.unavailable_reason() or ""
needs_metal = pytest.mark.skipif(not HAS_METAL, reason=f"needs Metal (macOS with MPS): {WHY}")
without_metal = pytest.mark.skipif(HAS_METAL, reason="checks the refusal on a machine without Metal")

g = torch.Generator().manual_seed(0)


def rnd(*shape, scale=1.0):
    return torch.randn(*shape, generator=g, dtype=torch.float64) * scale


# ----------------------------------------------------------------------------- kernels under test

def k_unary(op, dtype="float32"):
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), dtype)
    i, j = K.axis("i", M), K.axis("j", N)
    e = kir.Unary(op, x[i, j]) if op != "not" else kir.where(~(x[i, j] > 0.0), 1.0, 0.0)
    return K.kernel(f"u_{op}", [K.output("y", (i, j), e, dtype)])


def k_binary(dtype="float32"):
    M, N = K.sym("M"), K.sym("N")
    x, w = K.buffer("x", (M, N), dtype), K.buffer("w", (N,), dtype)
    i, j = K.axis("i", M), K.axis("j", N)
    e = K.maximum(x[i, j], w[j]) / (x[i, j] + 1.5) - K.minimum(x[i, j], 0.25) * K.index(j)
    e = K.where((K.index(j) <= K.index(i)) & (x[i, j] > -1.0), e, K.NEG_INF)
    return K.kernel("binary", [K.output("y", (i, j), e, dtype)])


def k_softmax(dtype="float32"):
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), dtype)
    i, j = K.axis("i", M), K.axis("j", N)
    return K.kernel("softmax", [K.output("y", (i, j), K.softmax_weight(x[i, j] * 0.5, over=j), dtype)])


def k_layernorm(n=96, dtype="float32"):
    M = K.sym("M")
    x, w, b = K.buffer("x", (M, n), dtype), K.buffer("w", (n,), dtype), K.buffer("b", (n,), dtype)
    i, j, r, s = K.axis("i", M), K.axis("j", n), K.axis("r", n), K.axis("s", n)
    mean = K.reduce_sum(r, x[i, r]) / n
    var = K.reduce_sum(s, (x[i, s] - mean) * (x[i, s] - mean)) / n
    y = (x[i, j] - mean) * K.rsqrt(var + 1e-5) * w[j] + b[j]
    return K.kernel("layernorm", [K.output("y", (i, j), y, dtype)])


def k_rowstats():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N))
    i, r1, r2, r3 = K.axis("i", M), K.axis("r1", N), K.axis("r2", N), K.axis("r3", N)
    return K.kernel("rowstats", [K.output("s", (i,), K.reduce_sum(r1, x[i, r1])),
                                 K.output("mx", (i,), K.reduce_max(r2, x[i, r2])),
                                 K.output("mn", (i,), K.reduce_min(r3, x[i, r3]))])


def k_total():
    N = K.sym("N")
    x = K.buffer("x", (N,))
    r = K.axis("r", N)
    return K.kernel("total", [K.output("t", (), K.reduce_sum(r, x[r] * x[r]))])


def k_matmul(M=37, N=45, Kd=29, dtype="float32", batch=False, epilogue=True):
    Ms, Ns = K.sym("M"), K.sym("N")
    if batch:
        Bs = K.sym("B")
        a = K.buffer("a", (Bs, Ms, Kd), dtype)
        b = K.buffer("b", (Bs, Kd, Ns), dtype)
        z = K.axis("z", Bs)
    else:
        a = K.buffer("a", (Ms, Kd), dtype)
        b = K.buffer("b", (Kd, Ns), dtype)
    bias = K.buffer("bias", (Ns,), dtype)
    i, j, r = K.axis("i", Ms), K.axis("j", Ns), K.axis("r", Kd)
    if batch:
        s = K.reduce_sum(r, a[z, i, r] * b[z, r, j])
        axes = (z, i, j)
    else:
        s = K.reduce_sum(r, K.tanh(a[i, r]) * b[r, j])     # a prologue on one factor
        axes = (i, j)
    y = K.relu(s + bias[j]) * 0.5 if epilogue else s
    return K.kernel("matmul", [K.output("y", axes, y, dtype)])


def k_linear3d(dtype="float32"):
    """y[z, i, j] = sum_r x[z, i, r] * w[r, j] + bias[j]: a linear layer on [B, T, C] activations,
    whose batch axis the weight does not depend on (the CUDA plan folds it into the rows)."""
    Bs, Ts, C, N = K.sym("B"), K.sym("T"), 24, 40
    x, w, bias = K.buffer("x", (Bs, Ts, C), dtype), K.buffer("w", (C, N), dtype), K.buffer("bias", (N,), dtype)
    z, i, j, r = K.axis("z", Bs), K.axis("i", Ts), K.axis("j", N), K.axis("r", C)
    y = K.reduce_sum(r, x[z, i, r] * w[r, j]) + bias[j]
    return K.kernel("linear3d", [K.output("y", (z, i, j), y, dtype)])


def k_conv1d():
    """y[n, o, h] = sum_c sum_t x[n, c, 2h + t - 1] * w[o, c, t], zero padding: the conv plan."""
    N, C, O, H, T = 2, 8, 6, 15, 3
    OH = (H + 2 - T) // 2 + 1
    x, w = K.buffer("x", (N, C, H)), K.buffer("w", (O, C, T))
    n, o, h = K.axis("n", N), K.axis("o", O), K.axis("h", OH)
    c, t = K.axis("c", C), K.axis("t", T)
    xs = kir.Load(x, (n, c, K.strided([(h, 2), (t, 1)], -1)), pad=0.0)
    y = K.reduce_sum(c, K.reduce_sum(t, xs * w[o, c, t]))
    return K.kernel("conv1d", [K.output("y", (n, o, h), y)])


def k_maxpool():
    N, H = 3, 17
    OH = (H + 2 - 3) // 2 + 1
    x = K.buffer("x", (N, H))
    n, h, t = K.axis("n", N), K.axis("h", OH), K.axis("t", 3)
    xs = kir.Load(x, (n, K.strided([(h, 2), (t, 1)], -1)), pad=-math.inf)
    return K.kernel("maxpool", [K.output("y", (n, h), K.reduce_max(t, xs))])


def attention_graph(dtype="float32", S=24, D=16, causal=True):
    gr = K.Graph()
    B, H = 2, 2
    q, k, v = (gr.input(n, (B, H, K.sym("S"), D), dtype) for n in "qkv")
    s = (q @ k.T) * D ** -0.5
    if causal:
        s = gr.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), s, with_index=True)
    return gr, {"o": gr.softmax(s) @ v}, (B, H, S, D)


def attention_ref(q, k, v, causal):
    s = (q.double() @ k.double().transpose(-1, -2)) * q.shape[-1] ** -0.5
    if causal:
        n = s.shape[-1]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v.double()


def _single(gr, outs):
    parts = gr.fuse(outs, "metal")
    assert len(parts) == 1, [k.name for _, k in parts]
    return parts[0][1]


CASES = {
    "binary": (k_binary, "pointwise", lambda: {"x": rnd(13, 21), "w": rnd(21)}),
    "softmax": (k_softmax, "rows", lambda: {"x": rnd(9, 300, scale=3)}),
    "layernorm": (k_layernorm, "rows", lambda: {"x": rnd(11, 96, scale=2) + 1, "w": rnd(96), "b": rnd(96)}),
    "rowstats": (k_rowstats, "rows", lambda: {"x": rnd(7, 1000)}),
    "total": (k_total, "rows", lambda: {"x": rnd(5000)}),
    "matmul": (k_matmul, "contraction", lambda: {"a": rnd(37, 29), "b": rnd(29, 45), "bias": rnd(45)}),
    "linear3d": (k_linear3d, "contraction",
                 lambda: {"x": rnd(3, 5, 24), "w": rnd(24, 40), "bias": rnd(40)}),
    "bmm": (lambda: k_matmul(batch=True, epilogue=False), "contraction",
            lambda: {"a": rnd(3, 33, 29), "b": rnd(3, 29, 70)}),
    "conv1d": (k_conv1d, "loops", lambda: {"x": rnd(2, 8, 15), "w": rnd(6, 8, 3)}),
    "maxpool": (k_maxpool, "loops", lambda: {"x": rnd(3, 17)}),
}


def _canon(k):
    return kir.canonicalize(k)[0]


# ----------------------------------------------------------------------------- everywhere

@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("simd", [True, False])
def test_source_is_generated_for_every_shape(name, simd):
    build, shape, _ = CASES[name]
    src, p = mb.generate(_canon(build()), {"simdgroup": simd})
    assert p.shape == shape
    assert src.count("{") == src.count("}")
    assert "#pragma METAL fp math_mode(safe)" in src and "kernel void kern(" in src
    assert "double" not in src                       # Metal has no float64
    if shape == "contraction":
        assert ("simdgroup_multiply_accumulate" in src) == simd
    if shape == "rows":
        assert "threadgroup_barrier" in src and "atomic" not in src


def test_attention_and_unary_ops_generate():
    gr, outs, _ = attention_graph()
    src, p = mb.generate(_canon(_single(gr, outs)), {})
    assert p.shape == "attention_simd" and p.tile == (32, 32) and p.group == 128 and "atomic" not in src
    assert src.count("{") == src.count("}") and "simdgroup_multiply_accumulate" in src
    assert "ts_jhi = min(ts_N, ts_i0 + 32 + (0))" in src      # causal: chunks past the tile skipped
    src, p = mb.generate(_canon(_single(gr, outs)), {"simdgroup": False})
    assert p.shape == "attention"                             # no SIMD group matrices: one row per group
    src, p = mb.generate(_canon(_single(gr, outs)), {"attention": "rows"})
    assert p.shape == "attention" and p.group == mb.ATT_GROUP and src.count("{") == src.count("}")
    src, p = mb.generate(_canon(_single(gr, outs)), {"attention": "loops"})
    assert p.shape == "loops" and p.via == "attention"
    gr, outs, _ = attention_graph(D=300)            # head dim above ATT_MAX_D: the loops shape
    assert mb.generate(_canon(_single(gr, outs)), {})[1].shape == "loops"
    gr, outs, _ = attention_graph(D=240)            # wide heads: smaller SIMD group tiles
    assert mb.generate(_canon(_single(gr, outs)), {})[1].tile == (8, 8)
    gr, outs, _ = attention_graph(D=20)             # not a multiple of 8: no SIMD group shape
    assert mb.generate(_canon(_single(gr, outs)), {})[1].shape == "attention"
    ka = _canon(_single(*attention_graph(S=40)[:2]))
    pa = mb.plan(ka)
    th, grp, work = mb.grid(pa, ka, {"s0": 40})
    assert grp == [pa.group, 1, 1] and th == [2 * 2 * 2 * pa.group, 1, 1]       # 2 query tiles of 32
    pr = mb.plan(ka, attention="rows")
    th, grp, work = mb.grid(pr, ka, {"s0": 40})
    assert grp == [mb.ATT_GROUP, 1, 1] and th == [2 * 2 * 40 * mb.ATT_GROUP, 1, 1]
    for op in ("neg", "exp", "log", "sqrt", "rsqrt", "sigmoid", "tanh", "abs", "erf", "cos", "sin", "not"):
        src, p = mb.generate(_canon(k_unary(op)), {})
        assert p.shape == "pointwise"
    assert "ts_erf(" in mb.generate(_canon(k_unary("erf")), {})[0]


def test_float64_is_refused_before_any_launch():
    with pytest.raises(mb.MetalUnsupported) as e:
        mb.generate(_canon(k_softmax("float64")), {})
    assert e.value.code == "TS841" and "float64" in str(e.value)


def test_literals_are_float32_and_saturate():
    assert mb._lit(0.5) == "0.5f" and mb._lit(-2.0) == "(-2.0f)"
    assert mb._lit(float("inf")) == "INFINITY" and mb._lit(float("nan")) == "NAN"
    assert mb._lit(-1e300) == "(-3.4028234663852886e+38f)"
    assert mb._lit(True) == "true"
    assert mb._lit(0.1) == "0.10000000149011612f"     # the float32 value, written exactly


def test_launch_grid_and_parameters():
    k = _canon(k_matmul(batch=True, epilogue=False))
    p = mb.plan(k)
    env = {"s0": 3, "s1": 33, "s2": 70}
    th, grp, work = mb.grid(p, k, env)
    assert grp == [128, 1, 1] and th == [3 * 128, 2, 3] and work == 3 * 33 * 70
    lay = mb.param_layout(k)
    assert [x[0] for x in lay][:3] == ["sym"] * 3
    rows = _canon(k_softmax())
    th, grp, work = mb.grid(mb.plan(rows), rows, {"s0": 9, "s1": 300})
    assert grp == [256, 1, 1] and th == [9 * 256, 1, 1] and work == 9
    with pytest.raises(mb.MetalUnsupported) as e:
        mb.params(rows, {"s0": 2 ** 31, "s1": 4}, {"b0": (2 ** 31, 4)}, {"o0": (2 ** 31, 4)})
    assert e.value.code == "TS845"


def test_codes_are_registered():
    from truestar import codes
    for c in ("TS840", "TS841", "TS842", "TS843", "TS844", "TS845"):
        assert c in codes.CODES and codes.slug(c).startswith("metal_")


def test_metal_check_backend_is_a_stack_not_eager_mps():
    """'metal' in truestar check must run Truestar's compiler, never eager MPS under its name."""
    from truestar.check import _STACK_KINDS
    from truestar.stacks import STACKS, StackBackend, TruestarRunner
    assert STACKS["metal"][0] in _STACK_KINDS
    b = StackBackend(name="metal", device="mps", description="", kind="truestar", option="metal")
    assert isinstance(b.make_runner(), TruestarRunner)


@without_metal
def test_no_metal_no_fallback():
    from truestar.backends import BackendUnavailable, resolve_backends
    from truestar.compiler import CompiledExecutor
    with pytest.raises(mb.MetalUnavailable) as e:
        K.compile(k_softmax(), "metal")
    assert "[TS840]" in str(e.value) and "never falls back" in str(e.value)
    with pytest.raises(mb.MetalUnavailable):
        CompiledExecutor("metal")
    with pytest.raises(BackendUnavailable) as e:
        resolve_backends(["metal"], allow_stacks=True)
    assert "metal" in str(e.value)


# ----------------------------------------------------------------------------- on a Mac

def _mps(inputs, k):
    dts = {b.name: getattr(torch, b.dtype) for b in k.inputs}
    return {n: v.to(dts[n]).to("mps") for n, v in inputs.items()}


def _reference(k, inputs, dtype):
    """float64 reference on the inputs as the kernel sees them (rounded to its dtype)."""
    import numpy as np
    np_in = {n: v.to(getattr(torch, dtype)).double().numpy() for n, v in inputs.items()}
    return reference.run_reference(k, np_in)


def _run_twice(ck, ins):
    a = ck(**ins)
    b = ck(**ins)
    a = a if isinstance(a, dict) else {"_": a}
    b = b if isinstance(b, dict) else {"_": b}
    for n in a:
        assert a[n].device.type == "mps"
        assert torch.equal(a[n].cpu().view(-1).view(torch.int16 if a[n].element_size() == 2 else torch.int32),
                           b[n].cpu().view(-1).view(torch.int16 if b[n].element_size() == 2 else torch.int32)), \
            f"{n} not bitwise reproducible"
    return a


@needs_metal
@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("simd", ["1", "0"])
def test_kernel_runs_and_matches_float64(name, simd, monkeypatch):
    monkeypatch.setenv("TRUESTAR_METAL_SIMDGROUP", simd)
    build, shape, make = CASES[name]
    k = build()
    ck = K.compile(k, "metal")
    assert ck.backend == "metal" and ck.plan_kind in ("tile", "contraction", "loops")
    inputs = make()
    outs = _run_twice(ck, _mps(inputs, k))      # the first call is checked against float64 inside
    assert ck.launch_configs()[0][1]["shape"] == shape


@needs_metal
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("name", ["softmax", "layernorm", "matmul"])
def test_half_precision_kernels(name, dtype):
    build = {"softmax": k_softmax, "layernorm": k_layernorm, "matmul": k_matmul}[name]
    k = build(dtype=dtype)
    ck = K.compile(k, "metal")
    _run_twice(ck, _mps(CASES[name][2](), k))


@needs_metal
@pytest.mark.parametrize("op", ["neg", "exp", "log", "sqrt", "rsqrt", "sigmoid", "tanh", "abs", "erf",
                                "cos", "sin", "not"])
def test_unary_ops_with_special_values(op):
    k = k_unary(op)
    ck = K.compile(k, "metal")
    x = torch.cat([torch.linspace(-7, 7, 401, dtype=torch.float64),
                   torch.tensor([0.0, -0.0, 1e-7, -1e-7, 30.0, -30.0, 80.0, float("inf"), float("-inf"),
                                 float("nan")], dtype=torch.float64)]).reshape(1, -1)
    y = ck(x=x.float().to("mps"))          # checked inside: NaN and inf at the same places
    ref = _reference(k, {"x": x}, "float32")["y"]
    import numpy as np
    got = y.cpu().double().numpy()
    assert np.array_equal(np.isnan(got), np.isnan(ref))


@needs_metal
def test_attention_kernel_runs():
    gr, outs, (B, H, S, D) = attention_graph()
    fn = gr.compile(outs, "metal")
    q, k, v = (rnd(B, H, S, D).float().to("mps") for _ in range(3))
    o = fn(q=q, k=k, v=v)["o"]
    assert o.device.type == "mps" and torch.isfinite(o.cpu()).all()


@needs_metal
@pytest.mark.parametrize("S,D,causal", [(24, 16, True), (100, 32, True), (200, 80, False), (1, 64, False),
                                        (129, 128, True)])
@pytest.mark.parametrize("group", ["auto", "simd16", "rows", "loops", "32", "256"])
def test_attention_shape_matches_float64(S, D, causal, group, monkeypatch):
    """Chunks of keys (S above and below the chunk size, a partial last chunk and query tile), head
    dims above the group size (several accumulators per thread), causal and full; bitwise equal run
    to run."""
    monkeypatch.delenv("TRUESTAR_METAL_ATTENTION", raising=False)
    monkeypatch.delenv("TRUESTAR_METAL_ATT_GROUP", raising=False)
    if group == "simd16":
        monkeypatch.setenv("TRUESTAR_METAL_ATTENTION", "simd")
        monkeypatch.setenv("TRUESTAR_METAL_ATT_TILE", "16")
    elif group in ("loops", "auto", "rows"):
        monkeypatch.setenv("TRUESTAR_METAL_ATTENTION", group)
    else:
        monkeypatch.setenv("TRUESTAR_METAL_ATT_GROUP", group)
    gr, outs, (B, H, _, _) = attention_graph(S=S, D=D, causal=causal)
    fn = gr.compile(outs, "metal")
    q, k, v = (rnd(B, H, S, D) for _ in range(3))
    o1 = fn(q=q.float().to("mps"), k=k.float().to("mps"), v=v.float().to("mps"))["o"].cpu()
    o2 = fn(q=q.float().to("mps"), k=k.float().to("mps"), v=v.float().to("mps"))["o"].cpu()
    ref = attention_ref(q, k, v, causal)
    err = float((o1.double() - ref).abs().max() / ref.abs().max())
    assert err < 2e-6, err
    assert torch.equal(o1, o2)
    shapes = {ck.plan_kind for _, ck in fn.kernels}
    assert shapes == {"loops" if group == "loops" else "attention"}, shapes
    metal_shape = {ck.art.plan.shape for _, ck in fn.kernels}
    simd = "attention_simd" if mb.simdgroup_enabled() and D % 8 == 0 else "attention"   # S is symbolic
    want = {"auto": simd, "simd16": simd, "loops": "loops"}.get(group, "attention")
    assert metal_shape == {want}, metal_shape


@needs_metal
@pytest.mark.parametrize("mode", ["auto", "rows"])
def test_attention_rows_have_the_same_bits_in_any_batch(mode, monkeypatch):
    """A query row's output bits do not depend on the other rows or batch entries computed with it."""
    monkeypatch.setenv("TRUESTAR_METAL_ATTENTION", mode)
    gr = K.Graph()
    q, k, v = (gr.input(n, (K.sym("B"), 2, K.sym("S"), 32), "float32") for n in "qkv")
    s = gr.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), (q @ k.T) * 32 ** -0.5, with_index=True)
    fn = gr.compile({"o": gr.softmax(s) @ v}, "metal")
    q, k, v = (rnd(3, 2, 75, 32).float() for _ in range(3))
    full = fn(q=q.to("mps"), k=k.to("mps"), v=v.to("mps"))["o"].cpu()
    one = fn(q=q[1:2].to("mps"), k=k[1:2].to("mps"), v=v[1:2].to("mps"))["o"].cpu()
    assert torch.equal(full[1:2], one)



@needs_metal
def test_broken_codegen_is_caught_on_first_call():
    k = k_unary("exp")
    ck = K.compile(k, "metal", _break={"exp": lambda x: f"(precise::exp({x}) * 1.01f)"})
    with pytest.raises(K.KernelMismatchError) as e:
        ck(x=rnd(4, 50).float().to("mps"))
    assert "metal" in str(e.value)


@needs_metal
def test_simdgroup_probe_reports():
    ok, why = mb.simdgroup_probe()
    print(f"simdgroup matrices usable: {ok} {why}")
    assert ok or why


@needs_metal
def test_sentinels_refused_on_metal():
    with pytest.raises(mb.MetalUnsupported) as e:
        K.compile(k_softmax(), "metal", checks="nan")
    assert e.value.code == "TS842"


# ----------------------------------------------------------------------------- compiled executor

def _import(model, example):
    from truestar import ir
    B = torch.export.Dim("B", min=1, max=64)
    return ir.import_module(model.eval(), (example,), dynamic_shapes=({0: B},))


def _rel(out, ref):
    out, ref = out.detach().cpu().double(), ref.detach().cpu().double()
    return (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


TOL = {"float32": 2e-4, "float16": 3e-2}


@needs_metal
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mlp_end_to_end(dtype):
    from truestar.compiler import compile_program
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 8),
                          nn.Softmax(-1)).eval()
    prog = _import(model, torch.randn(2, 16))
    cp = compile_program(prog, "metal", dtype)
    for batch in (5, 1, 9):
        x = torch.randn(batch, 16)
        y = cp([x])[0]
        assert y.device.type == "mps" and y.dtype == getattr(torch, dtype)
        ref = model.double()(x.double())
        model.float()
        eager = model.to("mps").to(getattr(torch, dtype))(x.to(getattr(torch, dtype)).to("mps"))
        model.float().cpu()
        e_metal, e_mps = _rel(y, ref), _rel(eager, ref)
        print(f"mlp {dtype} batch {batch}: metal {e_metal:.2e}, mps eager {e_mps:.2e}")
        assert e_metal < TOL[dtype]
    rep = cp.report
    assert rep.fallbacks == [] and rep.library == []
    assert [k["shape"] for k in rep.kernels][:3] == ["contraction"] * 3


@needs_metal
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("variant", ["relu_post", "gelu_pre"])
def test_transformer_layer_end_to_end(dtype, variant):
    from truestar.compiler import compile_program
    torch.manual_seed(1)
    layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True,
                                       activation="relu" if variant == "relu_post" else "gelu",
                                       norm_first=variant == "gelu_pre").eval()
    prog = _import(layer, torch.randn(2, 5, 32))
    cp = compile_program(prog, "metal", dtype)
    for batch in (3, 1):
        x = torch.randn(batch, 5, 32)
        y = cp([x])[0]
        ref = layer.double()(x.double())
        layer.float()
        with torch.no_grad():
            eager = layer.to("mps").to(getattr(torch, dtype))(x.to(getattr(torch, dtype)).to("mps"))
        layer.float().cpu()
        e_metal, e_mps = _rel(y, ref), _rel(eager, ref)
        print(f"transformer {variant} {dtype} batch {batch}: metal {e_metal:.2e}, mps eager {e_mps:.2e}")
        assert e_metal < TOL[dtype]
    rep = cp.report
    assert rep.fallbacks == [] and rep.library == []
    for op in ("linear", "attention", "layer_norm"):
        assert op in rep.kernel_ops()


@needs_metal
def test_check_backend_metal():
    from truestar.check import check
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).eval()
    rep = check(model, torch.randn(8, 16), backends=["cpu", "metal"])
    res = {r.backend.name: r for r in rep.results}
    assert res["metal"].status == "ok", res["metal"].as_dict() if hasattr(res["metal"], "as_dict") else res["metal"]
    assert res["metal"].stack is not None and res["metal"].stack["kind"] == "truestar"


def _cli(*args):
    import json
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    p = subprocess.run([sys.executable, "-m", "truestar", "check", *args, "--format", "agent", "--no-repro"],
                       capture_output=True, text=True, env=env, timeout=600)
    return p.returncode, json.loads(p.stdout)


@needs_metal
def test_cli_check_metal_clean_model(tmp_path):
    f = tmp_path / "mlp.py"
    f.write_text("import torch\nfrom torch import nn\n"
                 "def get_model():\n    torch.manual_seed(0)\n"
                 "    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8), nn.Softmax(-1)).eval()\n"
                 "def get_inputs():\n    return torch.randn(4, 16)\n")
    code, doc = _cli(str(f), "--backends", "cpu,metal")
    b = {x["name"]: x for x in doc["backends"]}
    print("cli metal clean:", code, b["metal"]["status"], b["metal"].get("codes"))
    assert code == 0 and b["metal"]["status"] == "ok"


@needs_metal
def test_cli_check_metal_refuses_loudly_where_float32_cannot_be_verified():
    """examples/check_model.py loses its digits in any float32 run. The compiled executor checks
    each kernel against float64 and refuses the cancelling variance kernel instead of returning its
    result, so the metal backend is reported as failed (TS002) with the reason, never as ok."""
    ex = os.path.join(os.path.dirname(__file__), "..", "examples", "check_model.py")
    code, doc = _cli(ex, "--backends", "cpu,metal")
    b = {x["name"]: x for x in doc["backends"]}
    m = b["metal"]
    msgs = [i["message"] for i in doc["issues"] if i.get("backend") == "metal"]
    print("cli metal example:", code, m["status"], m.get("codes"), [x[:300] for x in msgs])
    assert code == 1 and m["status"] == "divergent" and "TS002" in m["codes"]
    assert b["cpu"]["blamed_module"]["name"] == "norm"
    assert any("does not match" in x for x in msgs)


def test_no_combined_float64_copy_off_a_device():
    """MPS to("cpu", torch.float64) in one call returns zeros on torch 2.14.1 and raises on 2.15
    (docs/METAL.md); x.double().cpu() raises on MPS. Every copy off a device goes to the CPU first
    (kernel.verify.to_host_f64 is the one place that handles it)."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "truestar"
    bad = re.compile(r"""\.to\(\s*["']cpu["']\s*,\s*(dtype=)?torch\.(float64|double)\)|\.double\(\)\.cpu\(\)|"""
                     r"""\.to\(torch\.float64\)\.cpu\(\)""")
    hits = []
    for p in root.rglob("*.py"):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if bad.search(code) and not (p.name == "verify.py" and p.parent.name == "kernel"):
                hits.append(f"{p.relative_to(root)}:{n}: {line.strip()}")
    assert not hits, "\n".join(hits)
