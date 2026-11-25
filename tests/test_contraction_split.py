"""Split K contractions and the determinism guarantees around them.

* Split K (launch config SPLIT_K) cuts K into chunks, one program each, and a second kernel adds the
  partial sums in a fixed order. Every config is compared with float64, including ragged and empty
  chunks, batched contractions and epilogues that read other buffers.
* Run to run: the same inputs, shapes and config give the same bits (no atomics).
* Without split K a contraction's result does not depend on the launch config (tile sizes, BK,
  warps, stages, KTAIL), so not on the batch size either. With split K the chunk count follows M,
  so one row's rounding can depend on the batch it ran in; ``deterministic=True`` turns split K off
  and restores that guarantee.
"""

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


pytestmark = pytest.mark.skipif(not _has_triton(), reason="needs CUDA and Triton")


def linear_kernel(N, Kd, dtype="float32"):
    """relu(x @ w.T + b) with a symbolic row count, the shape of a linear layer."""
    g = Graph()
    x = g.input("x", (K.sym("M"), Kd), dtype)
    w = g.input("w", (N, Kd), dtype)
    b = g.input("b", (N,), dtype)
    (_, kern), = g.fuse({"y": g.relu(x @ g.transpose(w, 0, 1) + b)})
    return kern


def rel(out, ref):
    out, ref = out.double().cpu(), ref.double().cpu()
    return ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()


def data(M, N, Kd, seed=0, dtype=torch.float32):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, Kd, device="cuda", generator=gen).to(dtype)
    w = (torch.randn(N, Kd, device="cuda", generator=gen) / Kd ** 0.5).to(dtype)
    b = torch.randn(N, device="cuda", generator=gen).to(dtype)
    ref = torch.relu(x.double() @ w.double().T + b.double())
    return x, w, b, ref


@pytest.mark.parametrize("split", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("ktail", [True, False])
def test_split_k_matches_float64(split, ktail):
    # K 1000 is not a multiple of BK 32, and 16 chunks of 64 leave the last ones short or empty
    x, w, b, ref = data(37, 200, 1000)
    if split == 3:  # SPLIT_K must be a power of two: rejected by name, never run
        p = tb.plan(ir.canonicalize(linear_kernel(200, 1000))[0])
        assert "power of two" in tb.check_config(p, {"s0": 37}, tb._cfg(32, 64, 32, 4, 3,
                                                                          split=3))
        return
    kern = linear_kernel(200, 1000)
    cfg = tb._cfg(32, 64, 32, 4, 3, ktail=ktail, split=split)
    out = K.compile(kern, "cuda", _config=cfg)(x=x, w=w, b=b)   # verified against float64 too
    assert rel(out, ref) < 1e-5


@pytest.mark.parametrize("dtype,tol", [(torch.float16, 3e-3), (torch.bfloat16, 2e-2)])
def test_split_k_half_precision(dtype, tol):
    x, w, b, ref = data(20, 96, 777, dtype=dtype)
    kern = linear_kernel(96, 777, str(dtype).replace("torch.", ""))
    out = K.compile(kern, "cuda", _config=tb._cfg(16, 32, 64, 4, 3, split=4))(x=x, w=w, b=b)
    assert out.dtype == dtype and rel(out, ref) < tol


def test_split_k_batched_contraction_with_residual_epilogue():
    """Batch axes and an epilogue that loads another buffer, both in the second pass."""
    g = Graph()
    a = g.input("a", (3, K.sym("M"), 600), "float32")
    bb = g.input("b", (3, 600, 40), "float32")
    r = g.input("r", (3, K.sym("M"), 40), "float32")
    (_, kern), = g.fuse({"c": g.tanh(a @ bb) * 2.0 + r})
    torch.manual_seed(3)
    at, bt, rt = torch.randn(3, 50, 600, device="cuda"), torch.randn(3, 600, 40, device="cuda") / 25, \
        torch.randn(3, 50, 40, device="cuda")
    ref = torch.tanh(at.double() @ bt.double()) * 2 + rt.double()
    for split in (1, 4):
        out = K.compile(kern, "cuda", _config=tb._cfg(16, 32, 32, 4, 3, split=split))(a=at, b=bt, r=rt)
        assert rel(out, ref) < 1e-5


def test_default_uses_split_k_for_few_tiles_and_is_run_to_run_deterministic():
    kern = linear_kernel(1000, 2048)
    ck = K.compile(kern, "cuda", tune=False)
    x, w, b, ref = data(64, 1000, 2048)
    first = ck(x=x, w=w, b=b)
    (_, cfg, src), = ck.launch_configs()
    assert cfg["SPLIT_K"] > 1 and src == "default"
    assert rel(first, ref) < 1e-5
    for _ in range(5):
        assert torch.equal(ck(x=x, w=w, b=b), first)


def _rows_by_batch(ck, w, b, x):
    """Row block x[:7] computed inside batches of 7, 64 and 300 rows."""
    return [ck(x=x[:m].contiguous(), w=w, b=b)[:7] for m in (7, 64, 300)]


def test_without_split_k_results_do_not_depend_on_config_or_batch():
    kern = linear_kernel(256, 1536)
    x, w, b, _ = data(300, 256, 1536)
    outs = []
    for cfg in [tb._cfg(*g_) for g_ in tb.CONTRACTION_GRID["ieee"]] + [tb._cfg(64, 64, 64, 4, 3, ktail=False)]:
        outs.append(K.compile(kern, "cuda", _config=tb._fit_shared(
            tb.plan(ir.canonicalize(kern)[0]), cfg))(x=x, w=w, b=b))
    assert all(torch.equal(o, outs[0]) for o in outs)
    det = K.compile(kern, "cuda", deterministic=True, tune=False)
    rows = _rows_by_batch(det, w, b, x)
    assert all(torch.equal(r, rows[0]) for r in rows)   # K 1536: chunks fixed by K (det_k_chunk)


def test_split_k_makes_rows_depend_on_the_batch_and_deterministic_mode_does_not():
    """The documented price of split K: the chunk count follows M, so a row computed in a batch of
    7 and in a batch of 300 can differ in the last bits. deterministic=True gives equal bits."""
    kern = linear_kernel(1024, 4096)
    x, w, b, ref = data(300, 1024, 4096)
    fast = K.compile(kern, "cuda", tune=False)
    rows = _rows_by_batch(fast, w, b, x)
    splits = sorted({c["SPLIT_K"] for _, c, _ in fast.launch_configs()})
    assert len(splits) > 1                       # the chunk count changed with M
    assert not all(torch.equal(r, rows[0]) for r in rows)
    for r in rows:
        assert rel(r, ref[:7]) < 1e-5            # every one is still float32 accurate
    det = K.compile(kern, "cuda", tune=False, deterministic=True)
    drows = _rows_by_batch(det, w, b, x)
    assert all(torch.equal(r, drows[0]) for r in drows)


def test_deterministic_mode_rejects_a_cached_split_config(monkeypatch, tmp_path):
    """A split K config read back for a deterministic kernel would break its guarantee: rejected
    where K is one chain (K up to DET_K_MIN, or half precision), and beyond one program per chunk
    of a deterministic K split."""
    kern = linear_kernel(1024, 1024)
    p = tb.generate(ir.canonicalize(kern)[0], {"deterministic": True})[1]
    assert p.kchunk == 0
    why = tb.check_config(p, {"s0": 64}, tb._cfg(64, 128, 32, 4, 3, split=4))
    assert why is not None and "deterministic" in why
    with pytest.raises(ValueError, match="deterministic"):
        K.compile(kern, "cuda", deterministic=True, _config=tb._cfg(64, 128, 32, 4, 3, split=4))(
            *data(64, 1024, 1024)[:3])
    p = tb.generate(ir.canonicalize(linear_kernel(1024, 4096))[0], {"deterministic": True})[1]
    assert tb.check_config(p, {"s0": 64}, tb._cfg(64, 128, 32, 4, 3, split=8)) is None
    assert "one program per K chunk" in tb.check_config(p, {"s0": 64}, tb._cfg(64, 128, 32, 4, 3, split=16))


@pytest.mark.parametrize("Kd, dtype, opts, want", [
    (256, "float32", {"deterministic": True}, 0),
    (1024, "float32", {"deterministic": True}, 0),          # K up to DET_K_MIN: one chain, bits as before
    (1025, "float32", {"deterministic": True}, 512),
    (8192, "float32", {"deterministic": True}, 512),
    (8192, "float32", {}, 0),                               # default mode keeps its own split K
    (8192, "float32", {"deterministic": True, "allow_tf32": True}, 0),
    (8192, "float16", {"deterministic": True}, 0),
])
def test_deterministic_k_chunk_follows_k_alone(Kd, dtype, opts, want):
    p = tb.generate(ir.canonicalize(linear_kernel(64, Kd, dtype))[0], opts)[1]
    assert p.kchunk == want


def _chain_reference(x, w, b, kch, rows, cols):
    """The documented order in float64: each chunk a float32 fused multiply add chain from zero
    (a float32 product is exact in float64; the add rounds to float64 and then to float32, the
    same as one float32 rounding unless the float64 sum falls on a float32 tie, which these
    values never hit), chunk sums added left to right, then bias and relu."""
    import numpy as np
    x, w, b = x.cpu().numpy(), w.cpu().numpy(), b.cpu().numpy()
    K_ = x.shape[1]
    out = {}
    for r in rows:
        for c in cols:
            tot = np.float32(0)
            for lo in range(0, K_, kch):
                acc = np.float32(0)
                for k in range(lo, min(K_, lo + kch)):
                    acc = np.float32(float(x[r, k]) * float(w[c, k]) + float(acc))
                tot = np.float32(tot + acc)
            out[(r, c)] = max(np.float32(tot + b[c]), np.float32(0))
    return out


@pytest.mark.parametrize("M, N, Kd", [(64, 512, 8192), (33, 200, 1500), (1, 1024, 4096)])
def test_deterministic_k_chunks_same_bits_for_every_config_batch_and_layout(M, N, Kd):
    """Deterministic K chunks: one result for every tile, warp count, stage count and SPLIT_K
    (1 to the chunk count), for a row alone or in a batch, for the weight stored [N, K] or [K, N];
    it is the documented order exactly (float64 emulation) and float32 accurate."""
    kern = linear_kernel(N, Kd)
    x, w, b, ref = data(M, N, Kd, seed=3)
    det = K.compile(kern, "cuda", deterministic=True, tune=False)
    first = det(x=x, w=w, b=b).clone()
    assert torch.equal(det(x=x, w=w, b=b), first)
    nch = -(-Kd // tb.DET_K_CHUNK)
    for bm, bn, bk, warps, stages in ((16, 64, 64, 2, 3), (64, 128, 16, 4, 3), (32, 64, 32, 2, 4),
                                       (128, 128, 16, 4, 3), (64, 16, 32, 1, 3)):
        for split in sorted({1, 2, 1 << (nch.bit_length() - 1)}):
            c = tb._cfg(bm, bn, bk, warps, stages, split=split)
            out = K.compile(kern, "cuda", deterministic=True, _config=c)(x=x, w=w, b=b)
            assert torch.equal(out, first), (c, split)
    assert torch.equal(det(x=x, w=w.t().contiguous().t(), b=b), first)
    for r in sorted({0, M - 1}):
        assert torch.equal(det(x=x[r:r + 1].contiguous(), w=w, b=b)[0], first[r])
    em = _chain_reference(x, w, b, tb.DET_K_CHUNK, sorted({0, M - 1}), [0, N // 2, N - 1])
    assert all(float(first[r, c]) == float(v) for (r, c), v in em.items())
    assert rel(first, ref) < 1e-5
