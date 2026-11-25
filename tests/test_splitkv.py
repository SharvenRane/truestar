"""Split KV decode attention (launch config KV_CHUNK, kernels `kern` and `kern_splitkv`).

* A decode (up to DECODE_ROWS query rows) splits its keys into chunks of KV_CHUNK, one program
  each; every program stores its float32 running max, sum and accumulator and a second kernel
  merges them in the fixed chunk order 0, 1, 2, ... Every chunk size, dtype, a causal and a data
  mask, a two softmax (differential) decode and fully masked rows are compared with float64.
* Run to run: the same inputs and config give the same bits.
* Deterministic mode chunks by DET_KV_CHUNK whatever the SM count or batch, so a 26 SM and a 70
  SM GPU choose the same config and give the same bits; the fast mode may differ by SM count.
"""

import math
import os
import tempfile

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import Graph, ir  # noqa: E402
from truestar.kernel import triton_backend as tb  # noqa: E402


def _has_triton():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _has_triton(), reason="needs CUDA and Triton")


def decode_kernel(B, H, D, dtype, mask=None, Dv=None):
    """softmax(q k^T * scale [masked]) v with query length T and key length N, both symbolic."""
    g = Graph()
    T, N = K.sym("T"), K.sym("N")
    q = g.input("q", (B, H, T, D), dtype)
    k = g.input("k", (B, H, N, D), dtype)
    v = g.input("v", (B, H, N, Dv or D), dtype)
    s = (q @ k.T) * (D ** -0.5)
    if mask == "causal_tail":        # query row i sees keys up to N - T + i (a decode with history)
        s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i + 1000, s, K.NEG_INF), s, with_index=True)
    elif mask == "data":
        m = g.input("m", (B, 1, T, N), "float32")
        s = s + m
    (_, kern), = g.fuse({"o": g.softmax(s) @ v})
    return kern


def data(B, H, T, N, D, dtype, seed=0, Dv=None):
    dt = getattr(torch, dtype)
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, T, D, device="cuda", generator=g).to(dt)
    k = torch.randn(B, H, N, D, device="cuda", generator=g).to(dt)
    v = torch.randn(B, H, N, Dv or D, device="cuda", generator=g).to(dt)
    return q, k, v


def ref(q, k, v, add=None):
    s = (q.double() @ k.double().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    if add is not None:
        s = s + add.double()
    return torch.softmax(s, -1) @ v.double()


def rel(o, r):
    return ((o.double() - r).abs().max() / r.abs().max()).item()


TOL = {"float16": 2e-3, "bfloat16": 1.6e-2, "float32": 2e-5}


def plan_of(kern, det=False):
    opts = {"deterministic": True} if det else {}
    return tb.generate(ir.canonicalize(kern)[0], opts)[1]


@pytest.fixture
def sms(monkeypatch):
    def pin(n):
        monkeypatch.setattr(tb, "_sm_count", lambda: n)
    return pin


# ----------------------------------------------------------------------------- planning


def test_only_decodes_split_and_prefill_configs_are_unchanged(sms):
    sms(26)
    kern = decode_kernel(1, 8, 128, "float16")
    p = plan_of(kern)
    env = {p.i.size.name: 1, p.j.size.name: 4096}
    d = tb.default_config(p, env)
    assert d["BM"] == 16 and d["KV_CHUNK"] > 0
    env_pre = {p.i.size.name: 512, p.j.size.name: 4096}
    pre = tb.default_config(p, env_pre)
    assert pre["KV_CHUNK"] == 0 and pre["BM"] >= 64
    assert all(c["KV_CHUNK"] == 0 for c in tb.candidates(p, env_pre))


def test_fast_chunks_follow_the_sm_count_and_deterministic_ones_do_not(sms):
    kern = decode_kernel(4, 8, 128, "float16")
    p, pd = plan_of(kern), plan_of(kern, det=True)
    env = {p.i.size.name: 1, p.j.size.name: 8192}
    sms(26)
    few, few_d = tb.default_config(p, env), tb.default_config(pd, env)
    sms(70)
    many, many_d = tb.default_config(p, env), tb.default_config(pd, env)
    assert few["KV_CHUNK"] > many["KV_CHUNK"]          # fewer SMs: fewer, longer chunks
    assert few_d == many_d and few_d["KV_CHUNK"] == tb.DET_KV_CHUNK
    for b in (1, 32):                                  # and not on the batch either
        kb = decode_kernel(b, 8, 128, "float16")
        assert tb.default_config(plan_of(kb, det=True), env)["KV_CHUNK"] == tb.DET_KV_CHUNK


def test_check_config_rules_for_kv_chunk():
    kern = decode_kernel(1, 8, 128, "float16")
    p, pd = plan_of(kern), plan_of(kern, det=True)
    env = {p.i.size.name: 1, p.j.size.name: 4096}
    d = tb.default_config(p, env)
    assert tb.check_config(p, env, dict(d, KV_CHUNK=0)) is None
    assert "power of two" in tb.check_config(p, env, dict(d, KV_CHUNK=96))
    assert "power of two" in tb.check_config(p, env, dict(d, KV_CHUNK=32))     # below BN 64
    assert "deterministic" in tb.check_config(pd, env, dict(d, KV_CHUNK=512))
    assert tb.check_config(pd, env, dict(d, KV_CHUNK=tb.DET_KV_CHUNK)) is None


# ----------------------------------------------------------------------------- on the GPU


@gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("chunk", [0, 64, 128, 256, 1024])
def test_every_chunk_size_matches_float64(dtype, chunk):
    B, H, N, D = 2, 4, 1000, 128          # 1000 keys: a ragged last chunk for every size
    kern = decode_kernel(B, H, D, dtype)
    q, k, v = data(B, H, 1, N, D, dtype)
    cfg = dict(BM=16, BN=64, num_warps=4, num_stages=2, LPT=True, SPLIT=True, KV_CHUNK=chunk)
    out = K.compile(kern, "cuda", tune=False, _config=cfg)(q=q, k=k, v=v)
    assert rel(out, ref(q, k, v)) < TOL[dtype]


@gpu
@pytest.mark.parametrize("T", [1, 4, 16])
def test_several_query_rows_and_lpt_off(T):
    kern = decode_kernel(1, 8, 64, "float16")
    q, k, v = data(1, 8, T, 777, 64, "float16")
    for lpt in (True, False):
        cfg = dict(BM=16, BN=32, num_warps=4, num_stages=2, LPT=lpt, SPLIT=True, KV_CHUNK=128)
        out = K.compile(kern, "cuda", tune=False, _config=cfg)(q=q, k=k, v=v)
        assert rel(out, ref(q, k, v)) < TOL["float16"]


@gpu
def test_index_mask_and_data_mask_with_split():
    T, N = 4, 1200
    kern = decode_kernel(2, 4, 64, "float16", mask="causal_tail")
    q, k, v = data(2, 4, T, N, 64, "float16")
    i = torch.arange(T, device="cuda")[:, None]
    j = torch.arange(N, device="cuda")[None, :]
    add = torch.where(j <= i + 1000, 0.0, -math.inf)
    cfg = dict(BM=16, BN=64, num_warps=4, num_stages=2, LPT=True, SPLIT=True, KV_CHUNK=256)
    out = K.compile(kern, "cuda", tune=False, _config=cfg)(q=q, k=k, v=v)
    assert rel(out, ref(q, k, v, add)) < TOL["float16"]
    # a key padding mask given as data: the last 300 keys of batch 1 are padding
    kern = decode_kernel(2, 4, 64, "float16", mask="data")
    m = torch.zeros(2, 1, T, N, device="cuda")
    m[1, :, :, 900:] = -math.inf
    out = K.compile(kern, "cuda", tune=False, _config=cfg)(q=q, k=k, v=v, m=m)
    assert rel(out, ref(q, k, v, m)) < TOL["float16"]


@gpu
def test_a_chunk_with_every_key_masked_contributes_nothing():
    """Keys masked for whole chunks (here everything after key 100, chunks of 64): those chunks
    store max -inf and sum 0, and the merge must skip them instead of producing NaN."""
    T, N = 1, 1000
    kern = decode_kernel(1, 2, 64, "float32", mask="data")
    q, k, v = data(1, 2, T, N, 64, "float32")
    m = torch.zeros(1, 1, T, N, device="cuda")
    m[..., 100:] = -math.inf
    cfg = dict(BM=16, BN=64, num_warps=4, num_stages=2, LPT=True, SPLIT=True, KV_CHUNK=64)
    out = K.compile(kern, "cuda", tune=False, _config=cfg)(q=q, k=k, v=v, m=m)
    assert torch.isfinite(out).all()
    assert rel(out, ref(q, k, v, m)) < TOL["float32"]


@gpu
def test_differential_attention_decode_splits_both_softmaxes():
    g = Graph()
    T, N = K.sym("T"), K.sym("N")
    q1, k1, q2, k2 = (g.input(n, (1, 4, T if n[0] == "q" else N, 32), "float16")
                      for n in ("q1", "k1", "q2", "k2"))
    v = g.input("v", (1, 4, N, 64), "float16")
    lam = g.input("lam", (1, 4, 1, 1), "float32")
    p1 = g.softmax((q1 @ k1.T) * 32 ** -0.5)
    p2 = g.softmax((q2 @ k2.T) * 32 ** -0.5)
    (_, kern), = g.fuse({"o": (p1 - lam * p2) @ v})
    assert len(plan_of(kern).terms) == 2
    gen = torch.Generator(device="cuda").manual_seed(3)
    ins = {n: torch.randn(1, 4, 1 if n[0] == "q" else 900, 32, device="cuda", generator=gen).half()
           for n in ("q1", "k1", "q2", "k2")}
    ins["v"] = torch.randn(1, 4, 900, 64, device="cuda", generator=gen).half()
    ins["lam"] = torch.rand(1, 4, 1, 1, device="cuda", generator=gen) * 0.8
    d = {n: t.double() for n, t in ins.items()}
    a1 = torch.softmax((d["q1"] @ d["k1"].transpose(-1, -2)) * 32 ** -0.5, -1)
    a2 = torch.softmax((d["q2"] @ d["k2"].transpose(-1, -2)) * 32 ** -0.5, -1)
    r = (a1 - d["lam"] * a2) @ d["v"]
    cfg = dict(BM=16, BN=64, num_warps=4, num_stages=2, LPT=True, SPLIT=True, KV_CHUNK=128)
    out = K.compile(kern, "cuda", tune=False, _config=cfg)(**ins)
    assert rel(out, r) < 4e-3


@gpu
@pytest.mark.parametrize("det", [False, True])
def test_default_decode_is_run_to_run_deterministic(det):
    kern = decode_kernel(8, 8, 128, "bfloat16")
    q, k, v = data(8, 8, 1, 3000, 128, "bfloat16")
    ck = K.compile(kern, "cuda", tune=False, deterministic=det)
    first = ck(q=q, k=k, v=v)
    assert ck.launch_configs()[0][1]["KV_CHUNK"] > 0
    for _ in range(5):
        assert torch.equal(ck(q=q, k=k, v=v), first)
    assert rel(first, ref(q, k, v)) < TOL["bfloat16"]


@gpu
def test_deterministic_bits_are_the_same_for_26_and_70_sms(sms):
    kern = decode_kernel(2, 8, 128, "float16")
    q, k, v = data(2, 8, 1, 5000, 128, "float16")
    outs = []
    for n in (26, 70):
        sms(n)
        K.clear_memory_cache()
        ck = K.compile(kern, "cuda", tune=False, deterministic=True)
        outs.append((ck(q=q, k=k, v=v), ck.launch_configs()[0][1]))
    assert outs[0][1] == outs[1][1]
    assert torch.equal(outs[0][0], outs[1][0])


@gpu
def test_deterministic_decode_row_does_not_depend_on_the_batch():
    kern1, kern4 = decode_kernel(1, 8, 128, "float16"), decode_kernel(4, 8, 128, "float16")
    q, k, v = data(4, 8, 1, 2500, 128, "float16")
    full = K.compile(kern4, "cuda", tune=False, deterministic=True)(q=q, k=k, v=v)
    one = K.compile(kern1, "cuda", tune=False, deterministic=True)(
        q=q[2:3].contiguous(), k=k[2:3].contiguous(), v=v[2:3].contiguous())
    assert torch.equal(full[2:3], one)


@gpu
def test_verification_catches_a_broken_merge():
    """A wrong exponential (used by the chunk merge as well as the online softmax) is caught by
    the float64 check of the first call."""
    kern = decode_kernel(1, 4, 64, "float16")
    q, k, v = data(1, 4, 1, 1000, 64, "float16")
    cfg = dict(BM=16, BN=64, num_warps=4, num_stages=2, LPT=True, SPLIT=True, KV_CHUNK=128)
    fn = K.compile(kern, "cuda", tune=False, _config=cfg,
                   _break={"exp": lambda x: f"tl.exp(({x}) * 0.97)"})
    with pytest.raises(K.KernelMismatchError):
        fn(q=q, k=k, v=v)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("det", [False, True])
def test_unsplit_half_decode_takes_one_16_row_tile(dtype, det):
    """A half precision decode that does not split its keys runs one 16 row tile with 3 stages, at
    the prefill block's BN and warps (the bits do not change, the test below checks it); the
    prefill keeps its 64 row blocks."""
    p = plan_of(decode_kernel(1, 8, 128, dtype), det)
    c = tb.default_config(p, {p.i.size.name: 1, p.j.size.name: 1024})
    assert (c["BM"], c["BN"], c["num_warps"], c["num_stages"], c["KV_CHUNK"]) == (16, 32, 4, 3, 0)
    pre = tb.default_config(p, {p.i.size.name: 64, p.j.size.name: 1024})
    assert pre["BM"] == 64 and pre["KV_CHUNK"] == 0


@gpu
@pytest.mark.parametrize("B,T,N", [(1, 1, 1024), (8, 1, 1024), (1, 4, 2000), (2, 16, 1536)])
def test_unsplit_half_decode_tile_keeps_the_bits(B, T, N):
    """The 16 row decode tile gives the bits of the 64 row block it replaced, in both modes."""
    kern = decode_kernel(B, 8, 128, "float16")
    q, k, v = data(B, 8, T, N, 128, "float16", seed=N + T)
    for det in (False, True):
        new = K.compile(kern, "cuda", tune=False, deterministic=det)
        o = new(q=q, k=k, v=v)
        cfg = new.launch_configs()[0][1]
        if cfg["KV_CHUNK"]:   # default mode on a GPU where this decode splits: other blocks
            assert not det
            continue
        assert cfg["BM"] == 16
        old = K.compile(kern, "cuda", tune=False, deterministic=det,
                        _config=dict(cfg, BM=64, num_stages=2))
        assert torch.equal(o, old(q=q, k=k, v=v))
        assert rel(o, ref(q, k, v)) < TOL["float16"]
