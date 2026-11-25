"""KV cache below 8 bits (truestar.kvquant, formats int4 and nvfp4, optionally Hadamard rotated).

* Storage: the codes follow the format's rule against the stored scales (round to nearest, ties
  to even); the GPU quantizer and the CPU fake quantizer store the same values; a rotation of a
  float16 cache and its inverse give back the cache bit for bit in float64.
* Kernel: the output matches float64 attention over the stored values turned back to the
  unrotated basis, for both formats, rotated or not, grouped query heads, several query rows.
* Against the unquantized cache: the measured error never exceeds ``error_bound`` (computed from
  the values in the unrotated basis) plus the kernel's own float16 rounding.
* Determinism: run to run equal; deterministic chunks ignore the SM count; a sequence alone equals
  it in a batch; paged and dense caches give the same bits.
* Memory: bytes per token as documented.
"""

import pytest
import torch

from truestar import codes
from truestar import kvquant as KQ
from truestar.kernel import KernelMismatchError
from truestar.kvquant import low
from truestar.quant import formats as QF
from truestar.rotate import hadamard as H
from truestar.rotate import walsh_kernel as W

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CASES = [("int4", False), ("int4", True), ("nvfp4", False), ("nvfp4", True)]


def data(B, Hq, Hk, T, N, D=128, dtype=torch.float16, seed=0, outlier=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, Hq, T, D, device="cuda", generator=g).to(dtype)
    k = torch.randn(B, Hk, N, D, device="cuda", generator=g)
    if outlier:
        k[..., 3] *= outlier
    v = torch.randn(B, Hk, N, D, device="cuda", generator=g).to(dtype)
    return q, k.to(dtype), v


def exact(q, k, v):
    G = q.shape[1] // k.shape[1]
    kr, vr = k.double().repeat_interleave(G, 1), v.double().repeat_interleave(G, 1)
    w = torch.softmax((q.double() @ kr.transpose(-1, -2)) * q.shape[-1] ** -0.5, -1)
    return w @ vr


# ----------------------------------------------------------------------------- storage


def test_rotation_round_trip_is_exact():
    x = torch.randn(3, 2, 77, 128, device="cuda").half()
    assert torch.equal(low.unrotate(H.walsh(x.double(), normalize=False)), x.double())


@pytest.mark.parametrize("fmt", ["int4", "nvfp4"])
def test_codes_follow_the_format_rule(fmt):
    _, k, _ = data(1, 1, 2, 1, 100, D=64)
    c = KQ.quantize(k, fmt, 32)
    deq = c.dequantize()
    bs = QF.E4M3.decode(c.bscale.view(torch.uint8)).repeat_interleave(16, 3)
    gs = c.scale.double().repeat_interleave(32, 2)[:, :, :100, None]
    sv = bs * gs
    y = k.double() / sv
    if fmt == "int4":
        want = torch.round(y).clamp(-7, 7)
    else:
        want = QF.E2M1.decode(QF.E2M1.encode(y))
    assert torch.equal(deq, want * sv)
    # the block scale holds the block amax within E4M3 rounding (3 mantissa bits)
    bam = k.double().abs().view(1, 2, 100, 4, 16).amax(-1)
    ratio = (QF.E4M3.decode(c.bscale.view(torch.uint8)) * c.scale.double().repeat_interleave(32, 2)[:, :, :100, None]
             * low.QM[fmt]) / bam
    assert float((ratio - 1).abs().max()) <= 2 ** -4 + 1e-12


@pytest.mark.parametrize("fmt,rot", CASES)
def test_gpu_quantizer_equals_the_cpu_fake_quantizer(fmt, rot):
    _, k, _ = data(2, 1, 3, 1, 333, D=64, outlier=10)
    assert torch.equal(KQ.quantize(k, fmt, 32, rot=rot).original().cpu(),
                       low.fake_quantize(k.cpu(), fmt, 32, rot))


def test_memory_per_token():
    # 8 KV heads of dim 128, groups of 32: K and V, 64 packed bytes + 8 E4M3 scales + 1/8 of a float32
    assert KQ.bytes_per_token("int4", 128, 32, heads=8) == 8 * 2 * (64 + 8 + 0.125) == 1154
    assert KQ.bytes_per_token("nvfp4", 128, 32, heads=8) == 1154
    assert KQ.bytes_per_token("int8", 128, 32, heads=8) == 2050
    assert KQ.bytes_per_token("float16", 128, 32, heads=8) == 4096
    _, k, _ = data(1, 1, 8, 1, 1024)
    c = KQ.quantize(k, "nvfp4", 32)
    stored = c.data.numel() + c.bscale.numel() + 4 * c.scale.numel()
    assert stored * 2 == 1024 * 1154


# ----------------------------------------------------------------------------- kernel


@pytest.mark.parametrize("fmt,rot", CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kernel_matches_float64_over_the_stored_values(fmt, rot, dtype):
    q, k, v = data(2, 8, 2, 1, 3000, dtype=dtype)
    kc, vc = KQ.quantize(k, fmt, 32, rot=rot), KQ.quantize(v, fmt, 32, rot=rot)
    out = KQ.decode_attention(q, kc, vc)
    ref = KQ.reference(q, kc, vc)
    tol = 2e-3 if dtype == torch.float16 else 1.6e-2
    assert float((out.double() - ref).abs().max() / ref.abs().max()) < tol


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("D", [64, 128, 256])
def test_query_rows_and_head_dims(T, D):
    q, k, v = data(1, 4, 4, T, 700, D=D)
    kc, vc = KQ.quantize(k, "nvfp4", 64, rot=True), KQ.quantize(v, "nvfp4", 64, rot=True)
    out = KQ.decode_attention(q, kc, vc)
    ref = KQ.reference(q, kc, vc)
    assert float((out.double() - ref).abs().max() / ref.abs().max()) < 2e-3


@pytest.mark.parametrize("fmt,rot", CASES)
@pytest.mark.parametrize("outlier", [0.0, 20.0])
def test_error_against_the_unquantized_cache_stays_within_the_bound(fmt, rot, outlier):
    q, k, v = data(2, 8, 2, 1, 2500, outlier=outlier)
    kc, vc = KQ.quantize(k, fmt, 32, rot=rot), KQ.quantize(v, fmt, 32, rot=rot)
    out = KQ.decode_attention(q, kc, vc, deterministic=True)
    err = (out.double() - exact(q, k, v)).abs()
    bound = KQ.error_bound(q, kc, vc, k, v)
    fp16 = 2 ** -11 * exact(q, k, v).abs() + 2e-3 * float(exact(q, k, v).abs().max())
    assert bool((err <= bound + fp16).all())


def test_rotation_shrinks_the_bound_on_channel_outliers_for_int4():
    q, k, v = data(2, 8, 2, 1, 2048, outlier=20)
    b = {}
    for rot in (False, True):
        kc, vc = KQ.quantize(k, "int4", 32, rot=rot), KQ.quantize(v, "int4", 32, rot=rot)
        b[rot] = float(KQ.error_bound(q, kc, vc, k, v).max())
    assert b[True] < b[False] / 2


# ----------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("fmt,rot", CASES)
def test_run_to_run_equal_and_sm_count_independent(fmt, rot, monkeypatch):
    q, k, v = data(3, 8, 2, 1, 4096)
    kc, vc = KQ.quantize(k, fmt, 32, rot=rot), KQ.quantize(v, fmt, 32, rot=rot)
    outs = []
    for n in (26, 70):
        monkeypatch.setattr(KQ, "_sms", lambda n=n: n)
        outs.append(KQ.decode_attention(q, kc, vc, deterministic=True))
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(KQ.decode_attention(q, kc, vc, deterministic=True), outs[0])
    one = KQ.decode_attention(
        q[1:2], KQ.QuantKV(kc.data[1:2], kc.scale[1:2], fmt, 32, False, kc.bscale[1:2], rot),
        KQ.QuantKV(vc.data[1:2], vc.scale[1:2], fmt, 32, False, vc.bscale[1:2], rot), deterministic=True)
    assert torch.equal(outs[0][1:2], one)


@pytest.mark.parametrize("fmt,rot", CASES)
def test_paged_and_dense_caches_give_the_same_bits(fmt, rot):
    B, Hk, page, N, D = 3, 2, 64, 1000, 128
    q, k, v = data(B, 8, Hk, 1, N)
    npg = -(-N // page)
    pool_k = torch.zeros(B * npg + 5, Hk, page, D, device="cuda", dtype=torch.float16)
    pool_v = torch.zeros_like(pool_k)
    perm = torch.randperm(B * npg + 5, device="cuda")[: B * npg].view(B, npg)
    kp = torch.cat([k, k.new_zeros(B, Hk, npg * page - N, D)], 2).view(B, Hk, npg, page, D)
    vp = torch.cat([v, v.new_zeros(B, Hk, npg * page - N, D)], 2).view(B, Hk, npg, page, D)
    pool_k[perm] = kp.permute(0, 2, 1, 3, 4)
    pool_v[perm] = vp.permute(0, 2, 1, 3, 4)
    pk, pv = KQ.quantize_pages(pool_k, fmt, 32, rot=rot), KQ.quantize_pages(pool_v, fmt, 32, rot=rot)
    paged = KQ.decode_attention(q, KQ.gather_pages(pk, perm, N), KQ.gather_pages(pv, perm, N))
    dense = KQ.decode_attention(q, KQ.quantize(k, fmt, 32, rot=rot), KQ.quantize(v, fmt, 32, rot=rot))
    assert torch.equal(paged, dense)


# ----------------------------------------------------------------------------- the Walsh kernel


@pytest.mark.parametrize("D", [32, 64, 128, 256])
def test_walsh_kernel_matches_float64_and_is_batch_invariant(D):
    x = torch.randn(45, D, device="cuda").half()
    y = W.fwht(x, out_dtype=torch.float32)
    ref = H.walsh(x.double(), normalize=False)
    assert float((y.double() - ref).abs().max() / ref.abs().max()) < 2 ** -22
    assert torch.equal(W.fwht(x[7:8], out_dtype=torch.float32), y[7:8])
    back = W.fwht(y, scale=1.0 / D, out_dtype=torch.float32)
    assert float((back.double() - x.double()).abs().max()) < 1e-5


def test_walsh_kernel_refusals():
    assert codes.slug("TS1023") == "walsh_kernel_mismatch"
    with pytest.raises(H.RotationError, match="TS1022"):
        W.fwht(torch.randn(4, 96, device="cuda"))
    with pytest.raises(H.RotationError, match="TS1022"):
        W.fwht(torch.randn(4, 64))


# ----------------------------------------------------------------------------- refusals


def test_unsupported_requests_raise_ts701():
    q, k, v = data(1, 4, 4, 1, 300)
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.quantize(k, "int8", rot=True)
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.quantize(k, "int4", channel=True)
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.quantize(k[..., :48], "nvfp4")
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.quantize(torch.randn(1, 1, 8, 96, device="cuda").half(), "int4", rot=True)
    kc, vc = KQ.quantize(k, "int4", rot=True), KQ.quantize(v, "int4")
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.decode_attention(q, kc, vc)
    tail = (k[:, :, :8].contiguous(), v[:, :, :8].contiguous(), torch.tensor([4], device="cuda", dtype=torch.int32))
    # the 4 bit decode reads a float tail since the compiled decode keeps one (no longer TS701);
    # tests/test_kvquant_decode.py checks its values against the growing cache
    out = KQ.decode_attention(q, kc, KQ.quantize(v, "int4", rot=True), tail=tail)
    assert out.shape == q.shape and bool(torch.isfinite(out).all())


def test_verification_catches_a_wrong_output():
    q, k, v = data(1, 4, 4, 1, 500)
    kc, vc = KQ.quantize(k, "nvfp4", rot=True), KQ.quantize(v, "nvfp4", rot=True)
    out = KQ.decode_attention(q, kc, vc)
    with pytest.raises(KernelMismatchError):
        KQ._check(q, kc, vc, 128 ** -0.5, out * 1.01 + 0.01)


@pytest.mark.parametrize("rot", [False, True])
def test_a_non_contiguous_cache_quantizes_like_its_contiguous_copy(rot):
    _, k, _ = data(2, 1, 3, 1, 200, D=64)
    kt = k.transpose(1, 2).contiguous().transpose(1, 2)      # [B, H, N, D] with token major strides
    assert not kt.is_contiguous()
    a, b = KQ.quantize(kt, "nvfp4", 32, rot=rot), KQ.quantize(k, "nvfp4", 32, rot=rot)
    assert torch.equal(a.data, b.data) and torch.equal(a.bscale.view(torch.uint8), b.bscale.view(torch.uint8))
