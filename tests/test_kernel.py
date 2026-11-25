"""Kernel IR, fusion, backends, cache and verification.

Every result is compared with an independent float64 PyTorch computation, on top of the built in
verification each compiled kernel runs on first use.
"""

import math
import os
import tempfile

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import Graph  # noqa: E402
from truestar.kernel import compiler, triton_backend  # noqa: E402


def _has_triton():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_GPU = _has_triton()
BACKENDS = ["cpu"] + (["cuda"] if HAS_GPU else [])
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and Triton")


def dev(be):
    return "cuda" if be == "cuda" else "cpu"


def rnd(*shape, be="cpu", dtype=torch.float32, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32).to(dev(be), dtype)


def close(out, ref, tol):
    out = out.detach().double().cpu()
    ref = ref.detach().double().cpu()
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(torch.isnan(out), torch.isnan(ref)), "NaN positions differ"
    fin = ~torch.isnan(ref)
    err = (out[fin] - ref[fin]).abs().max().item() if fin.any() else 0.0
    scale = max(ref[fin].abs().max().item(), 1e-6) if fin.any() else 1.0
    assert err / scale <= tol, f"relative error {err / scale:.3g} > {tol}"


TOL = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 3e-2}

# ----------------------------------------------------------------------------- pointwise IR

UNARY = {
    "neg": (lambda e: -e, lambda t: -t),
    "exp": (K.exp, torch.exp),
    "log": (lambda e: K.log(K.abs_(e) + 0.1), lambda t: torch.log(t.abs() + 0.1)),
    "sqrt": (lambda e: K.sqrt(K.abs_(e)), lambda t: torch.sqrt(t.abs())),
    "rsqrt": (lambda e: K.rsqrt(K.abs_(e) + 0.5), lambda t: torch.rsqrt(t.abs() + 0.5)),
    "sigmoid": (K.sigmoid, torch.sigmoid),
    "tanh": (K.tanh, torch.tanh),
    "abs": (K.abs_, torch.abs),
    "relu": (K.relu, torch.relu),
}

BINARY = {
    "add": (lambda a, b: a + b, lambda a, b: a + b),
    "sub": (lambda a, b: a - b, lambda a, b: a - b),
    "mul": (lambda a, b: a * b, lambda a, b: a * b),
    "div": (lambda a, b: a / (K.abs_(b) + 0.5), lambda a, b: a / (b.abs() + 0.5)),
    "maximum": (K.maximum, torch.maximum),
    "minimum": (K.minimum, torch.minimum),
    "where_lt": (lambda a, b: K.where(a < b, a, b * 2.0), lambda a, b: torch.where(a < b, a, b * 2)),
    "where_ge_and": (lambda a, b: K.where((a >= 0.0) & (b <= 0.5), a, -b),
                     lambda a, b: torch.where((a >= 0) & (b <= 0.5), a, -b)),
}


def _pointwise_kernel(fn, n_in):
    M, N = K.sym("M"), K.sym("N")
    bufs = [K.buffer(n, (M, N)) for n in "ab"[:n_in]]
    i, j = K.axis("i", M), K.axis("j", N)
    e = fn(*[b[i, j] for b in bufs])
    return K.kernel("pw", [K.output("y", (i, j), e)])


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("op", list(UNARY))
def test_unary_ops(be, op):
    kfn, tfn = UNARY[op]
    fn = K.compile(_pointwise_kernel(kfn, 1), be)
    a = rnd(33, 130, be=be, seed=1)
    close(fn(a=a), tfn(a.double()), 1e-5)


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("op", list(BINARY))
def test_binary_ops(be, op):
    kfn, tfn = BINARY[op]
    fn = K.compile(_pointwise_kernel(kfn, 2), be)
    a, b = rnd(17, 300, be=be, seed=2), rnd(17, 300, be=be)
    close(fn(a=a, b=b), tfn(a.double(), b.double()), 1e-5)


@pytest.mark.parametrize("be", BACKENDS)
def test_broadcast_strided_and_index_values(be):
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N))
    row = K.buffer("row", (1, N))
    i, j = K.axis("i", M), K.axis("j", N)
    e = x[i, j] * row[0, j] + K.index(i) * 0.5 - K.index(j) + K.dim(N)
    fn = K.compile(K.kernel("bcast", [K.output("y", (i, j), e)]), be)
    base = rnd(70, 50, be=be, seed=3)
    xt = base.t()  # non contiguous view, shape (50, 70)
    r = rnd(1, 70, be=be)
    out = fn(x=xt, row=r)
    ii = torch.arange(50, dtype=torch.float64)[:, None]
    jj = torch.arange(70, dtype=torch.float64)[None, :]
    ref = xt.double().cpu() * r.double().cpu() + ii * 0.5 - jj + 70
    close(out, ref, 1e-6)


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_pointwise(be, dtype):
    name = str(dtype).replace("torch.", "")
    M = K.sym("M")
    x = K.buffer("x", (M,), name)
    i = K.axis("i", M)
    fn = K.compile(K.kernel("half", [K.output("y", (i,), K.sigmoid(x[i]) * x[i])]), be)
    a = rnd(1000, be=be, dtype=dtype, seed=4)
    out = fn(x=a)
    assert out.dtype == dtype
    close(out, torch.sigmoid(a.double()) * a.double(), TOL[dtype])


# ----------------------------------------------------------------------------- reductions


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("op", ["sum", "max", "min"])
@pytest.mark.parametrize("n", [7, 1000, 10000])
def test_reductions(be, op, n):
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N))
    i, r = K.axis("i", M), K.axis("r", N)
    red = {"sum": K.reduce_sum, "max": K.reduce_max, "min": K.reduce_min}[op]
    e = red(r, K.exp(x[i, r] * 0.1)) * 2.0
    fn = K.compile(K.kernel(f"red_{op}", [K.output("y", (i,), e)]), be)
    a = rnd(9, n, be=be, seed=5)
    t = torch.exp(a.double() * 0.1)
    ref = {"sum": t.sum(-1), "max": t.max(-1).values, "min": t.min(-1).values}[op] * 2
    close(fn(x=a), ref, 1e-5)


@pytest.mark.parametrize("be", BACKENDS)
def test_reduce_middle_axis(be):
    g = Graph()
    x = g.input("x", (K.sym("A"), K.sym("B"), K.sym("C")))
    y = g.sum(x, axis=1)
    fn = g.compile({"y": y}, be)
    a = rnd(5, 33, 20, be=be, seed=6)
    close(fn(x=a)["y"], a.double().sum(1), 1e-5)


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("n", [64, 1000, 9000])
def test_softmax_graph(be, n):
    g = Graph()
    x = g.input("x", (K.sym("M"), K.sym("N")))
    y = g.softmax(x * 0.5 + 1.0)
    fn = g.compile({"y": y}, be)
    assert fn.num_kernels == 1
    a = rnd(13, n, be=be, seed=7)
    close(fn(x=a)["y"], torch.softmax(a.double() * 0.5 + 1, -1), 1e-5)


@pytest.mark.parametrize("be", BACKENDS)
def test_softmax_over_first_axis(be):
    g = Graph()
    x = g.input("x", (K.sym("M"), K.sym("N")))
    fn = g.compile({"y": g.softmax(x, axis=0)}, be)
    a = rnd(300, 21, be=be, seed=8)
    close(fn(x=a)["y"], torch.softmax(a.double(), 0), 1e-5)


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("n", [96, 5000])
def test_layer_norm_with_residual(be, n):
    g = Graph()
    M, N = K.sym("M"), K.sym("N")
    x, res = g.input("x", (M, N)), g.input("res", (M, N))
    w, b = g.input("w", (N,)), g.input("b", (N,))
    y = g.layer_norm(x + res, w, b, eps=1e-5)
    fn = g.compile({"y": y}, be)
    assert fn.num_kernels == 1
    a, r_, ww, bb = rnd(19, n, be=be, seed=9), rnd(19, n, be=be), rnd(n, be=be), rnd(n, be=be)
    ref = torch.nn.functional.layer_norm((a + r_).double(), (n,), ww.double(), bb.double(), 1e-5)
    close(fn(x=a, res=r_, w=ww, b=bb)["y"], ref, 1e-4)


@pytest.mark.parametrize("be", BACKENDS)
def test_mean_and_max_graph(be):
    g = Graph()
    x = g.input("x", (K.sym("M"), K.sym("N")))
    y = g.mean(x, -1, keepdims=True)
    z = x - g.max(x, -1, keepdims=True) + y
    fn = g.compile({"z": z}, be)
    a = rnd(11, 77, be=be, seed=10)
    ad = a.double()
    close(fn(x=a)["z"], ad - ad.max(-1, keepdim=True).values + ad.mean(-1, keepdim=True), 1e-5)


# ----------------------------------------------------------------------------- matmul


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(70, 50, 90), (128, 64, 256), (5, 300, 17)])
def test_matmul_bias_relu(be, dtype, shape):
    if be == "cpu" and dtype == torch.float16 and shape[1] > 100:
        pytest.skip("slow on the scalar CPU path")
    m, kk, n = shape
    name = str(dtype).replace("torch.", "")
    g = Graph()
    a = g.input("a", (K.sym("M"), K.sym("K")), name)
    b = g.input("b", (K.sym("K"), K.sym("N")), name)
    bias = g.input("bias", (K.sym("N"),), name)
    y = g.relu(a @ b + bias)
    fn = g.compile({"y": y}, be)
    assert fn.num_kernels == 1
    if be == "cuda":
        assert fn.kernels[0][1].plan_kind == "contraction"
    x, w, c = rnd(m, kk, be=be, dtype=dtype, seed=11), rnd(kk, n, be=be, dtype=dtype), \
        rnd(n, be=be, dtype=dtype)
    ref = torch.relu(x.double() @ w.double() + c.double())
    close(fn(a=x, b=w, bias=c)["y"], ref, TOL[dtype])


@pytest.mark.parametrize("be", BACKENDS)
def test_batched_matmul_with_prologue(be):
    g = Graph()
    B, M, Kd, N = 3, K.sym("M"), K.sym("K"), K.sym("N")
    a = g.input("a", (B, M, Kd))
    b = g.input("b", (Kd, N))  # broadcast over the batch
    y = (g.exp(a * 0.1) @ g.tanh(b)) * 0.5
    fn = g.compile({"y": y}, be)
    assert fn.num_kernels == 1
    x, w = rnd(3, 40, 33, be=be, seed=12), rnd(33, 20, be=be)
    close(fn(a=x, b=w)["y"], (torch.exp(x.double() * 0.1) @ torch.tanh(w.double())) * 0.5, 1e-4)


@needs_gpu
def test_tf32_is_opt_in():
    g = Graph()
    a = g.input("a", (64, 64))
    b = g.input("b", (64, 64))
    f_ieee = g.compile({"y": a @ b}, "cuda")
    f_tf32 = g.compile({"y": a @ b}, "cuda", allow_tf32=True)
    assert 'input_precision="ieee"' in f_ieee.kernels[0][1].source
    assert 'input_precision="tf32"' in f_tf32.kernels[0][1].source
    x, w = rnd(64, 64, be="cuda", seed=13), rnd(64, 64, be="cuda")
    close(f_ieee(a=x, b=w)["y"], x.double() @ w.double(), 1e-5)


# ----------------------------------------------------------------------------- attention


def _attn_graph(dtype, mod=None, Nq=None, D=32, Dv=None):
    g = Graph()
    B, H, N, M = 2, 3, K.sym("N"), K.sym("M")
    q = g.input("q", (B, H, M if Nq else N, D), dtype)
    k = g.input("k", (B, H, N, D), dtype)
    v = g.input("v", (B, H, N, Dv or D), dtype)
    s = (q @ k.T) * (D ** -0.5)
    if mod is not None:
        s = g.pointwise(mod, s, with_index=True)
    return g, q, k, v, g.softmax(s) @ v


def _torch_attn(q, k, v, mask=None, mod=None):
    q, k, v = q.double().cpu(), k.double().cpu(), v.double().cpu()
    s = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if mod is not None:
        s = mod(s)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, -1) @ v


def _qkv(be, dtype, n, nq=None, D=32, Dv=None, seed=14):
    torch.manual_seed(seed)
    q = rnd(2, 3, nq or n, D, be=be, dtype=dtype)
    k = rnd(2, 3, n, D, be=be, dtype=dtype)
    v = rnd(2, 3, n, Dv or D, be=be, dtype=dtype)
    return q, k, v


ATTN_DTYPES = [("cpu", torch.float32)] + ([("cuda", torch.float32), ("cuda", torch.float16),
                                           ("cuda", torch.bfloat16)] if HAS_GPU else [])


@pytest.mark.parametrize("be,dtype", ATTN_DTYPES)
@pytest.mark.parametrize("n", [16, 100, 257])
def test_causal_attention(be, dtype, n):
    name = str(dtype).replace("torch.", "")
    g, *_, o = _attn_graph(name, lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF))
    fn = g.compile({"o": o}, be)
    assert fn.num_kernels == 1
    if be == "cuda":
        ck = fn.kernels[0][1]
        assert ck.plan_kind == "attention"
        assert (ck.art.plan.lo_off, ck.art.plan.hi_off) == (None, 0)  # causal: skip future blocks
    q, k, v = _qkv(be, dtype, n)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    close(fn(q=q, k=k, v=v)["o"], _torch_attn(q, k, v, mask), TOL[dtype])


@pytest.mark.parametrize("be,dtype", ATTN_DTYPES)
def test_noncausal_attention_rectangular(be, dtype):
    name = str(dtype).replace("torch.", "")
    g, *_, o = _attn_graph(name, None, Nq=True, D=40, Dv=24)
    fn = g.compile({"o": o}, be)
    q, k, v = _qkv(be, dtype, 150, nq=70, D=40, Dv=24)
    close(fn(q=q, k=k, v=v)["o"], _torch_attn(q, k, v), TOL[dtype])


@pytest.mark.parametrize("be,dtype", ATTN_DTYPES)
def test_sliding_window_softcap_alibi(be, dtype):
    """Window mask (both bounds), tanh softcap and a per head linear bias, written as plain Python."""
    name = str(dtype).replace("torch.", "")
    W = 20

    def mod(s, b, h, i, j):
        s = 10.0 * K.tanh(s / 10.0) - (h + 1.0) * 0.1 * (i - j)
        return K.where((j <= i) & (j > i - W), s, K.NEG_INF)

    g, *_, o = _attn_graph(name, mod)
    fn = g.compile({"o": o}, be)
    if be == "cuda":
        p = fn.kernels[0][1].art.plan
        assert (p.lo_off, p.hi_off) == (-W + 1, 0)
    n = 130
    q, k, v = _qkv(be, dtype, n)
    ii = torch.arange(n)[:, None]
    jj = torch.arange(n)[None, :]
    mask = (jj <= ii) & (jj > ii - W)
    hh = torch.arange(3, dtype=torch.float64)[None, :, None, None]

    def tmod(s):
        return 10 * torch.tanh(s / 10) - (hh + 1) * 0.1 * (ii - jj).double()

    close(fn(q=q, k=k, v=v)["o"], _torch_attn(q, k, v, mask, tmod), TOL[dtype])


def _diff_attn_graph(dtype, gated=False):
    """Differential attention (Ye et al. 2024): (softmax(Q1 K1) - lam * softmax(Q2 K2)) V."""
    g = Graph()
    B, H, N, D = 2, 3, K.sym("N"), 32
    q1, k1 = g.input("q1", (B, H, N, D), dtype), g.input("k1", (B, H, N, D), dtype)
    q2, k2 = g.input("q2", (B, H, N, D), dtype), g.input("k2", (B, H, N, D), dtype)
    v = g.input("v", (B, H, N, 2 * D), dtype)
    lam = g.input("lam", (1, H, 1, 1), "float32")
    causal = lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF)  # noqa: E731
    p1 = g.softmax(g.pointwise(causal, (q1 @ k1.T) * (D ** -0.5), with_index=True))
    p2 = g.softmax(g.pointwise(causal, (q2 @ k2.T) * (D ** -0.5), with_index=True))
    o = (p1 - lam * p2) @ v
    if gated:
        gate = g.input("gate", (B, H, N, 2 * D), dtype)
        o = o * g.sigmoid(gate)
    return g, o


@pytest.mark.parametrize("be,dtype", ATTN_DTYPES)
@pytest.mark.parametrize("gated", [False, True])
def test_differential_attention_fuses_into_one_kernel(be, dtype, gated):
    name = str(dtype).replace("torch.", "")
    g, o = _diff_attn_graph(name, gated)
    fn = g.compile({"o": o}, be)
    assert fn.num_kernels == 1
    if be == "cuda":
        ck = fn.kernels[0][1]
        assert ck.plan_kind == "attention" and len(ck.art.plan.terms) == 2
    n = 90
    torch.manual_seed(15)
    ins = {nm: rnd(2, 3, n, 32, be=be, dtype=dtype) for nm in ("q1", "k1", "q2", "k2")}
    ins["v"] = rnd(2, 3, n, 64, be=be, dtype=dtype)
    ins["lam"] = (torch.rand(1, 3, 1, 1) * 0.8).to(dev(be))
    if gated:
        ins["gate"] = rnd(2, 3, n, 64, be=be, dtype=dtype)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    d = {k_: x.double().cpu() for k_, x in ins.items()}
    a1 = torch.softmax(((d["q1"] @ d["k1"].transpose(-1, -2)) * 32 ** -0.5).masked_fill(~mask, -math.inf), -1)
    a2 = torch.softmax(((d["q2"] @ d["k2"].transpose(-1, -2)) * 32 ** -0.5).masked_fill(~mask, -math.inf), -1)
    ref = (a1 - d["lam"] * a2) @ d["v"]
    if gated:
        ref = ref * torch.sigmoid(d["gate"])
    close(fn(**ins)["o"], ref, TOL[dtype] * 2)


# ----------------------------------------------------------------------------- fusion decisions


@pytest.mark.parametrize("be", BACKENDS)
def test_fusion_splits_where_needed(be):
    g = Graph()
    M, N, Kd = K.sym("M"), K.sym("N"), K.sym("K")
    x = g.input("x", (M, Kd))
    w = g.input("w", (Kd, N))
    lw, lb = g.input("lw", (Kd,)), g.input("lb", (Kd,))
    y1 = g.softmax(x @ w)                       # row reduction over a contraction
    y2 = g.layer_norm(x, lw, lb) @ w            # contraction over a row reduction
    f1, f2 = g.compile({"y": y1}, be), g.compile({"y": y2}, be)
    expect = 2 if be == "cuda" else 1          # the CPU loop nest takes any kernel
    assert f1.num_kernels == expect and f2.num_kernels == expect
    a, b = rnd(20, 30, be=be, seed=16), rnd(30, 40, be=be)
    c, d = rnd(30, be=be), rnd(30, be=be)
    close(f1(x=a, w=b)["y"], torch.softmax(a.double() @ b.double(), -1), 1e-4)
    ln = torch.nn.functional.layer_norm(a.double(), (30,), c.double(), d.double())
    close(f2(x=a, w=b, lw=c, lb=d)["y"], ln @ b.double(), 1e-4)


@needs_gpu
def test_unschedulable_is_loud_not_a_fallback():
    M, N, P = K.sym("M"), K.sym("N"), K.sym("P")
    x, z = K.buffer("x", (M, N)), K.buffer("z", (M, P))
    i, r, s = K.axis("i", M), K.axis("r", N), K.axis("s", P)
    e = K.reduce_sum(r, x[i, r]) + K.reduce_max(s, z[i, s])   # two different reduced sizes
    k = K.kernel("two_sizes", [K.output("y", (i,), e)])
    with pytest.raises(K.Unschedulable, match="different sizes"):
        K.compile(k, "cuda")
    fn = K.compile(k, "cpu")
    a, b = rnd(4, 10, seed=17), rnd(4, 6)
    close(fn(x=a, z=b), a.double().sum(-1) + b.double().max(-1).values, 1e-6)


def test_shape_errors_name_both_shapes():
    g = Graph()
    a, b = g.input("a", (4, 5)), g.input("b", (6, 7))
    with pytest.raises(K.ShapeError, match=r"\[4, 5\] @ \[6, 7\]"):
        g.matmul(a, b)
    M = K.sym("M")
    x, y = K.buffer("x", (M,)), K.buffer("y", (M,))
    i = K.axis("i", M)
    fn = K.compile(K.kernel("sh", [K.output("o", (i,), x[i] + y[i])]), "cpu")
    with pytest.raises(K.ShapeError, match="M=3"):
        fn(x=torch.zeros(3), y=torch.zeros(4))


# ----------------------------------------------------------------------------- cache


def _chain(c):
    M = K.sym("M")
    x = K.buffer("x", (M,))
    i = K.axis("i", M)
    return K.kernel("chain", [K.output("y", (i,), K.sigmoid(x[i] * c) + 1.0)])


@pytest.mark.parametrize("be", BACKENDS)
def test_cache_hits_and_canonical_names(be):
    k1 = _chain(2.0)
    f1 = K.compile(k1, be)
    before = K.stats.memory_hits
    f2 = K.compile(_chain(2.0), be)
    assert K.stats.memory_hits == before + 1 and f2.key == f1.key
    # same structure, different names: same binary
    N = K.sym("Len")
    z = K.buffer("zz", (N,))
    a = K.axis("t", N)
    k3 = K.kernel("renamed", [K.output("w", (a,), K.sigmoid(z[a] * 2.0) + 1.0)])
    assert K.kernel_key(k3, be) == f1.key
    # an edit changes the key
    assert K.kernel_key(_chain(3.0), be) != f1.key
    x = rnd(100, be=be, seed=18)
    close(K.compile(k3, be)(zz=x), torch.sigmoid(x.double() * 2) + 1, 1e-6)


@pytest.mark.parametrize("be", BACKENDS)
def test_disk_cache_after_memory_cleared(be):
    k = _chain(4.5)
    K.compile(k, be)(x=rnd(10, be=be))
    K.clear_memory_cache()
    before = K.stats.disk_hits
    fn = K.compile(k, be)
    assert K.stats.disk_hits == before + 1
    x = rnd(10, be=be, seed=19)
    close(fn(x=x), torch.sigmoid(x.double() * 4.5) + 1, 1e-6)


@needs_gpu
def test_graph_edit_recompiles_only_changed_kernel():
    def build(scale):
        g = Graph()
        x = g.input("x", (K.sym("M"), K.sym("N")))
        y = g.softmax(x)                  # unchanged kernel
        z = g.layer_norm(y @ g.input("w", (K.sym("N"), K.sym("P"))) * scale)
        return g, {"z": z}

    g, outs = build(1.0)
    f = g.compile(outs, "cuda")
    keys1 = [ck.key for _, ck in f.kernels]
    g2, outs2 = build(2.0)
    f2 = g2.compile(outs2, "cuda")
    keys2 = [ck.key for _, ck in f2.kernels]
    assert len(keys1) == len(keys2) == 3
    assert sum(a == b for a, b in zip(keys1, keys2)) == 2  # only the scaled kernel changed


# ----------------------------------------------------------------------------- verification


@pytest.mark.parametrize("be", BACKENDS)
def test_verification_catches_broken_kernel(be):
    k = _chain(1.5)
    bad_sig = (lambda x: f"(tl.sigmoid({x}) * 1.01)") if be == "cuda" else \
        (lambda x: f"(_sigmoid({x}) * 1.01)")
    fn = K.compile(k, be, _break={"sigmoid": bad_sig})
    with pytest.raises(K.KernelMismatchError, match="kernel 'chain'"):
        fn(x=rnd(1000, be=be, seed=20))


@needs_gpu
def test_verification_catches_broken_attention_and_matmul():
    g, *_, o = _attn_graph("float16", lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF))
    k = g.fuse({"o": o}, "cuda")[0][1]
    fn = K.compile(k, "cuda", _break={"exp": lambda x: f"tl.exp(({x}) * 0.98)"})
    q, kk, v = _qkv("cuda", torch.float16, 64)
    with pytest.raises(K.KernelMismatchError, match="matmul"):
        fn(q=q, k=kk, v=v)
    g = Graph()
    a, b = g.input("a", (40, 30)), g.input("b", (30, 20))
    k = g.fuse({"y": a @ b + 1.0}, "cuda")[0][1]
    fn = K.compile(k, "cuda", _break={"add": lambda x, y: f"({x}) + ({y}) * 1.02"})
    with pytest.raises(K.KernelMismatchError):
        fn(a=rnd(40, 30, be="cuda"), b=rnd(30, 20, be="cuda"))


def test_verification_detects_nan_and_scale_errors():
    from truestar.kernel.verify import compare
    import numpy as np
    ref = np.array([1.0, 2.0, np.nan])
    assert compare("o", np.array([1.0, 2.0, np.nan]), ref, 1e-6).ok
    assert not compare("o", np.array([1.0, np.nan, np.nan]), ref, 1e-6).ok
    assert not compare("o", np.array([1.0, 2.0, 3.0]), ref, 1e-6).ok
    assert not compare("o", np.array([1.0, 2.1, np.nan]), ref, 1e-3).ok


def test_verify_off_skips_check(monkeypatch):
    k = _chain(7.25)
    fn = K.compile(k, "cpu", _break={"sigmoid": lambda x: f"(_sigmoid({x}) * 2.0)"})
    monkeypatch.setenv("TRUESTAR_VERIFY", "0")
    fn(x=torch.zeros(4))  # debugging escape hatch: no check
    monkeypatch.setenv("TRUESTAR_VERIFY", "all")
    with pytest.raises(K.KernelMismatchError):
        fn(x=torch.zeros(4))


def test_reference_matches_torch_independently():
    g, *_, o = _attn_graph("float32", lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF))
    q, k, v = _qkv("cpu", torch.float32, 33)
    ref = g.reference({"o": o}, {"q": q, "k": k, "v": v})["o"]
    mask = torch.ones(33, 33, dtype=torch.bool).tril()
    close(torch.from_numpy(ref), _torch_attn(q, k, v, mask), 1e-12)


def test_ir_rejects_bad_kernels():
    M = K.sym("M")
    x = K.buffer("x", (M,))
    i, j = K.axis("i", M), K.axis("j", 5)
    with pytest.raises(K.IRError, match="size"):
        x[j]
    with pytest.raises(K.IRError, match="neither output axes nor reduced"):
        K.output("y", (i,), x[i] + K.index(K.axis("q", 3)))
    with pytest.raises(K.IRError, match="truth value"):
        if x[i] > 0:
            pass


def test_chunked_reference_matches_unchunked(monkeypatch):
    from truestar.kernel import reference
    g, *_, o = _attn_graph("float32", lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF))
    q, k, v = _qkv("cpu", torch.float32, 40)
    whole = g.reference({"o": o}, {"q": q, "k": k, "v": v})["o"]
    monkeypatch.setattr(reference, "CHUNK_ELEMENTS", 100)
    chunked = g.reference({"o": o}, {"q": q, "k": k, "v": v})["o"]
    assert abs(whole - chunked).max() < 1e-12


@pytest.mark.parametrize("be", BACKENDS)
def test_new_shape_is_verified_and_launched_fresh(be):
    fn = K.compile(_chain(9.5), be, _break={"sigmoid": (lambda x: f"(tl.sigmoid({x}) * 1.01)")
                                            if be == "cuda" else (lambda x: f"(_sigmoid({x}) * 1.01)")})
    with pytest.raises(K.KernelMismatchError):
        fn(x=rnd(10, be=be))
    with pytest.raises(K.KernelMismatchError):  # a new shape signature is checked again
        fn(x=rnd(11, be=be))
    good = K.compile(_chain(9.5), be)
    x = rnd(12, be=be, seed=21)
    close(good(x=x), torch.sigmoid(x.double() * 9.5) + 1, 1e-6)
    close(good(x=x[::2]), torch.sigmoid(x[::2].double() * 9.5) + 1, 1e-6)  # strided view


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("n", [5, 3000, 9000])
def test_max_min_of_one_signed_rows(be, n):
    """All negative rows for max and all positive rows for min: a wrong initial value shows."""
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N))
    i, r, s = K.axis("i", M), K.axis("r", N), K.axis("s", N)
    neg = -K.exp(x[i, r]) - 1.0
    pos = K.exp(x[i, s]) + 1.0
    k = K.kernel("signed", [K.output("y", (i,), K.reduce_max(r, neg)),
                            K.output("z", (i,), K.reduce_min(s, pos))])
    fn = K.compile(k, be)
    a = rnd(6, n, be=be, seed=22)
    out = fn(x=a)
    ad = a.double()
    close(out["y"], (-torch.exp(ad) - 1).max(-1).values, 1e-6)
    close(out["z"], (torch.exp(ad) + 1).min(-1).values, 1e-6)
