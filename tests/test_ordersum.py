"""Order free sums (kernel/ordersum.py, docs/ORDER_FREE_SUM.md): the same float32 bits for every
order, split, block size and SM count, within a proven bound of the exact sum."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from truestar import kernel as K
from truestar.codes import CODES
from truestar.kernel import cpu_backend, ordersum as O
from truestar.kernel import triton_backend as tb

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def numba_ok():
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def bits(t):
    return t.contiguous().view(torch.int32)


def hard_rows(n=600, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = [
        torch.randn(n, generator=g),
        torch.randn(n, generator=g) * torch.exp2(torch.randint(-60, 60, (n,), generator=g).float()),
        torch.randn(n, generator=g) * 1e-40,                       # subnormal terms
        torch.randn(n, generator=g) * 1e-45,
        torch.randn(n, generator=g) * 3e38,                        # overflowing total
        torch.randint(-3, 4, (n,), generator=g).float() * 0.5,
        torch.full((n,), -0.0),
        torch.zeros(n),
    ]
    a = torch.randn(n // 2, generator=g) * 1e4
    rows.append(torch.cat([a, -a + torch.randn(n // 2, generator=g) * 1e-3]))
    # ties on the grid: 1.0 sets the grid to 2**-38, so 2**-39 and 3 * 2**-39 are halves
    tie = torch.zeros(n)
    tie[0], tie[1], tie[2], tie[3] = 1.0, 2.0 ** -39, 3 * 2.0 ** -39, -(5 * 2.0 ** -39)
    rows.append(tie)
    sp = torch.randn(4, n, generator=g)
    sp[0, 3] = float("nan")
    sp[1, 5] = float("inf")
    sp[2, 1], sp[2, 7] = float("inf"), float("-inf")
    sp[3, 0] = float("-inf")
    return torch.cat([torch.stack(rows), sp])


def test_codes_registered():
    assert CODES["TS1080"][1] == "order_free_sum_refused"
    assert CODES["TS1081"][1] == "order_free_sum_backend_unsupported"
    assert O.OrderFreeSumError.code == "TS1080"
    assert cpu_backend.HEADER.count(f"OS_BITS = {O.PRECISION_BITS}") == 1


def test_torch_and_numpy_definitions_agree_bitwise():
    x = hard_rows()
    a = O.order_free_sum(x, 1)
    b = torch.from_numpy(O.order_free_sum_numpy(x.numpy(), 1))
    assert torch.equal(bits(a), bits(b))


def test_any_order_and_split_gives_the_same_bits():
    x = hard_rows(seed=1)
    want = bits(O.order_free_sum(x, 1))
    for chunks in (1, 2, 3, 7, 64, 599):
        perm = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(chunks))
        assert torch.equal(bits(O.order_free_sum(x, 1, chunks=chunks, perm=perm)), want)


def test_special_values_and_signed_zero():
    x = hard_rows()
    r = O.order_free_sum(x, 1)
    assert torch.isnan(r[-4]) and r[-3] == float("inf") and torch.isnan(r[-2]) and r[-1] == float("-inf")
    assert bits(r[6:7]).item() == 0          # all -0.0 terms give +0.0 (documented)
    assert r[9].item() == 1.0 + 2.0 ** -38 * (0 + 2 - 2)   # ties: 0.5 -> 0, 1.5 -> 2, -2.5 -> -2


def test_within_the_proven_bound_of_the_exact_sum():
    x = hard_rows(n=300, seed=2)[:10]
    x = x[torch.isfinite(x).all(1)]
    r = O.order_free_sum(x, 1)
    bound = O.error_bound(x, 1)
    for i in range(x.shape[0]):
        exact = sum(Fraction(float(v)) for v in x[i].tolist())
        if not np.isfinite(r[i].item()):
            continue
        half_ulp = Fraction(float(np.spacing(np.abs(np.float32(r[i].item()))))) / 2
        assert abs(Fraction(r[i].item()) - exact) <= Fraction(float(bound[i])) + half_ulp


def test_more_accurate_than_float32_sums_on_cancellation():
    g = torch.Generator().manual_seed(4)
    a = torch.randn(64, 2000, generator=g) * 1e4
    x = torch.cat([a, -a + torch.randn(64, 2000, generator=g) * 1e-3], 1)
    ref = x.double().sum(1)
    assert (O.order_free_sum(x, 1).double() - ref).abs().max() < (x.sum(1).double() - ref).abs().max() / 100


def test_refusals():
    with pytest.raises(O.OrderFreeSumError) as e:
        O.order_free_sum(torch.zeros(3, dtype=torch.float16))
    assert e.value.code == "TS1080"
    with pytest.raises(O.OrderFreeSumError):
        O.order_free_sum(torch.zeros(O.MAX_TERMS + 1))


def _rowsum(body_mul=False):
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    w = K.buffer("w", (M, N), "float32")
    i, r = K.axis("i", M), K.axis("r", N)
    body = x[i, r] * w[i, r] if body_mul else x[i, r]
    return K.kernel("osum_rows", [K.output("y", (i,), K.order_free_sum(r, body))])


def test_ir_text_reference_and_metal_source():
    k = _rowsum()
    assert "osum[" in k.outputs[0].expr.__repr__()
    x = hard_rows(n=50)[:6].numpy().astype(np.float32)
    ref = K.run_reference(k, {"x": x})["y"]
    assert np.allclose(ref, x.astype(np.float64).sum(1), equal_nan=True)
    from truestar.kernel import ir as kir, metal_backend as mb
    src, p = mb.generate(kir.canonicalize(k)[0], {})
    assert p.shape == "rows" and "ts_os_finish(" in src and "ts_shl" in src and "atomic" not in src
    src, p = mb.generate(kir.canonicalize(_rowsum_static(40))[0], {})
    assert p.shape == "loops" and "ts_os_finish(" in src          # one thread per row, serial


def _rowsum_static(n, body_mul=False):
    M = K.sym("M")
    x = K.buffer("x", (M, n), "float32")
    w = K.buffer("w", (M, n), "float32")
    i, r = K.axis("i", M), K.axis("r", n)
    body = x[i, r] * w[i, r] if body_mul else x[i, r]
    return K.kernel("osum_rows_static", [K.output("y", (i,), K.order_free_sum(r, body))])


def _metal():
    from truestar.kernel import metal_backend as mb
    return mb.available()


metal_only = pytest.mark.skipif(not _metal(), reason="needs Metal (macOS with MPS)")


def _flush(t):
    """float32 subnormals to signed zero, as the Apple GPU's float arithmetic does."""
    return torch.where(t.abs() < 2.0 ** -126, torch.zeros_like(t) * torch.sign(t), t)


@metal_only
@pytest.mark.parametrize("n", [40, 700, 5000])
def test_metal_matches_the_definition_bitwise(n):
    """On the Mac's GPU: every hard row (subnormal, overflowing, cancelling, ties, NaN and inf) gives
    the definition's bits, which the CUDA and numba kernels also give (the tests above). With
    products in the body the terms are the GPU's own products, and the M2 flushes subnormal float32
    arithmetic to zero (loads and stores keep them), so the definition is applied to flushed ones."""
    x = hard_rows(n=n, seed=n)
    w = torch.randn(x.shape, generator=torch.Generator().manual_seed(n + 1))
    w[:, :3] = 1.0
    plain = K.compile(_rowsum(), "metal")(x=x.to("mps")).cpu()
    assert torch.equal(bits(plain), bits(O.order_free_sum(x, 1)))       # subnormal terms and totals too
    want = bits(O.order_free_sum(_flush(_flush(x) * _flush(w)), 1))
    k = _rowsum(body_mul=True)
    got = K.compile(k, "metal")(x=x.to("mps"), w=w.to("mps")).cpu()
    assert torch.equal(bits(got), want)
    again = K.compile(k, "metal")(x=x.to("mps"), w=w.to("mps")).cpu()
    assert torch.equal(bits(again), want)
    normal = torch.ones(x.shape[0], dtype=torch.bool)
    normal[2:4] = False                     # the two rows of subnormal terms
    assert torch.equal(bits(got)[normal], bits(O.order_free_sum(x * w, 1))[normal])
    if n <= 64:     # the same rows through one thread per row (serial) and a threadgroup per row (tree)
        ser = K.compile(_rowsum_static(n, body_mul=True), "metal")(x=x.to("mps"), w=w.to("mps")).cpu()
        assert torch.equal(bits(ser), want)


@metal_only
def test_metal_order_free_softmax_matches_rows_of_any_batch():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, j = K.axis("i", M), K.axis("j", N)
    sm = K.kernel("sm", [K.output("y", (i, j), K.softmax_weight(x[i, j], over=j))])
    X = torch.randn(48, 3000, generator=torch.Generator().manual_seed(3))
    fn = K.compile(sm, "metal", order_free=True)
    assert fn.order_free == {"applied": True, "sums": 1, "reason": ""}
    full = fn(x=X.to("mps")).cpu()
    part = fn(x=X[5:9].to("mps")).cpu()
    assert torch.equal(bits(full[5:9]), bits(part))
    assert torch.allclose(full, torch.softmax(X, 1), rtol=1e-5, atol=1e-7)


def test_order_free_tile_kernels_tune_in_deterministic_mode():
    p = tb.plan(_rowsum())
    assert p.kind == "tile" and not tb.bits_follow_config(p, {"M": 8, "N": 100})
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, j = K.axis("i", M), K.axis("j", N)
    sm = K.kernel("sm", [K.output("y", (i, j), K.softmax_weight(x[i, j], over=j))])
    assert tb.bits_follow_config(tb.plan(sm), {"M": 8, "N": 100})
    from truestar.kernel.compiler import _order_free_kernel
    k2, note = _order_free_kernel(sm)
    assert note["applied"] and not tb.bits_follow_config(tb.plan(k2), {"M": 8, "N": 100})
    a = K.buffer("a", (M, N), "float32")
    b = K.buffer("b", (N, M), "float32")
    r = K.axis("r", N)
    c = K.axis("c", M)
    mm = K.kernel("mm", [K.output("y", (i, c), K.reduce_sum(r, a[i, r] * b[r, c]))])
    k3, note = _order_free_kernel(mm)
    assert not note["applied"] and "contraction" in note["reason"] and k3 is mm


@pytest.mark.skipif(not numba_ok(), reason="needs numba")
def test_numba_backend_matches_the_definition_bitwise():
    k = _rowsum(body_mul=True)
    x = hard_rows(n=700, seed=5)
    w = torch.randn(x.shape, generator=torch.Generator().manual_seed(6))
    w[:, :3] = 1.0
    got = K.compile(k, "cpu")(x=x, w=w)
    assert torch.equal(bits(got), bits(O.order_free_sum(x * w, 1)))


@cuda_only
@pytest.mark.parametrize("n", [100, 4096, 9000])
def test_triton_matches_the_definition_for_every_launch_config(n):
    k = _rowsum(body_mul=True)
    x = hard_rows(n=n, seed=n).cuda()
    w = torch.randn(x.shape, device="cuda")
    want = bits(O.order_free_sum(x * w, 1))
    cfgs = ([dict(BR=1, num_warps=4), dict(BR=4, num_warps=8), dict(BR=2, num_warps=2)] if n <= 4096 else
            [dict(BC=128, num_warps=2), dict(BC=1024, num_warps=4), dict(BC=4096, num_warps=8)])
    for c in cfgs:
        got = K.compile(k, "cuda", _config=c)(x=x, w=w)
        assert torch.equal(bits(got), want), c
    assert torch.equal(bits(K.compile(k, "cuda", tune=True, deterministic=True)(x=x, w=w)), want)


@cuda_only
def test_order_free_option_on_softmax_is_config_free_and_verified():
    M, N = K.sym("M"), K.sym("N")
    x = K.buffer("x", (M, N), "float32")
    i, j = K.axis("i", M), K.axis("j", N)
    sm = K.kernel("sm", [K.output("y", (i, j), K.softmax_weight(x[i, j], over=j))])
    X = torch.randn(48, 3000, device="cuda")
    outs = [K.compile(sm, "cuda", order_free=True, _config=c)(x=X)
            for c in (dict(BR=1, num_warps=4), dict(BR=4, num_warps=8), dict(BR=2, num_warps=2))]
    assert all(torch.equal(bits(outs[0]), bits(o)) for o in outs[1:])
    fn = K.compile(sm, "cuda", order_free=True)
    assert fn.order_free == {"applied": True, "sums": 1, "reason": ""}
    assert torch.allclose(outs[0], torch.softmax(X, 1), rtol=1e-5, atol=1e-7)


@cuda_only
def test_standalone_cuda_sum_is_independent_of_blocks_and_programs():
    x = hard_rows(n=5000, seed=7)
    for row in (0, 1, 2, 4, 8, 9, 10, 11, 12, 13):
        xr = x[row].cuda()
        want = bits(O.order_free_sum(x[row], 0).reshape(1))
        for b, p in ((128, 1), (256, 3), (1024, None), (2048, 1000)):
            got = O.order_free_sum_cuda(xr, block=b, programs=p).reshape(1).cpu()
            assert torch.equal(bits(got), want), (row, b, p)
    perm = torch.randperm(5000)
    assert torch.equal(bits(O.order_free_sum_cuda(x[1].cuda()[perm.cuda()]).reshape(1).cpu()),
                       bits(O.order_free_sum(x[1], 0).reshape(1)))
