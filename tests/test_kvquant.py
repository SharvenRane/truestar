"""Quantized KV cache decode attention (truestar.kvquant).

* Storage: every int8 value is within amax / 254 of the original (half a step of its group's
  scale); every fp8 e4m3 value within 2^-4 of itself in the normal range, and within half of e4m3's
  smallest subnormal step (scaled) below it.
* Kernel: the output matches float64 attention over the dequantized cache (the values the kernel
  reads) for int8 and fp8, groups 32 and 64, grouped query heads, 1 to 16 query rows, ragged
  lengths and every chunk size.
* Against the unquantized cache: the measured error never exceeds ``error_bound`` (a proven upper
  bound from the cache's own rounding errors) plus the kernel's own float16 rounding.
* Determinism: run to run equal; deterministic chunks do not depend on the SM count or the batch;
  paged and dense caches give the same bits.
"""

import math

import pytest
import torch

from truestar import codes
from truestar import kvquant as KQ
from truestar.kernel import KernelMismatchError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def data(B, Hq, Hk, T, N, D=128, dtype=torch.float16, seed=0, spread=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, Hq, T, D, device="cuda", generator=g).to(dtype)
    k = (torch.randn(B, Hk, N, D, device="cuda", generator=g) * spread).to(dtype)
    v = torch.randn(B, Hk, N, D, device="cuda", generator=g).to(dtype)
    return q, k, v


def exact(q, k, v):
    G = q.shape[1] // k.shape[1]
    kr, vr = k.double().repeat_interleave(G, 1), v.double().repeat_interleave(G, 1)
    w = torch.softmax((q.double() @ kr.transpose(-1, -2)) * q.shape[-1] ** -0.5, -1)
    return w @ vr


def rel(a, b):
    return ((a.double() - b).abs().max() / b.abs().max()).item()


# ----------------------------------------------------------------------------- storage


@pytest.mark.parametrize("group", [32, 64])
def test_int8_values_are_within_half_a_step_of_their_group(group):
    x = torch.randn(2, 4, 1000, 64, device="cuda") * torch.linspace(0.1, 5, 1000, device="cuda")[:, None]
    c = KQ.quantize(x.half(), "int8", group)
    err = (c.dequantize() - x.half().double()).abs()
    amax = x.half().double().abs().view(2, 4, -1, 64)
    pad = c.scale.shape[2] * group - 1000
    amax = torch.cat([amax, amax.new_zeros(2, 4, pad, 64)], 2).view(2, 4, -1, group, 64).amax(dim=(3, 4))
    limit = (amax / 254).repeat_interleave(group, 2)[:, :, :1000, None]
    # plus float32 rounding of x / s near a half step: at most 2^-16 of a step (127 levels)
    assert (err <= limit * (1 + 2.0 ** -12) + 1e-12).all()


def test_fp8_values_are_within_two_to_the_minus_four():
    x = torch.randn(1, 2, 512, 64, device="cuda").half()
    c = KQ.quantize(x, "fp8", 32)
    xd, cd = x.double(), c.dequantize()
    s = c.scale.double().repeat_interleave(32, 2)[..., None]
    normal = (xd / s).abs() >= 2.0 ** -6                       # e4m3's smallest normal
    assert ((cd - xd).abs() <= 2.0 ** -4 * xd.abs() * (1 + 1e-9))[normal].all()
    assert ((cd - xd).abs() <= s * 2.0 ** -10 * (1 + 1e-9))[~normal].all()


def test_memory_per_token():
    assert KQ.bytes_per_token("float16", 128, 32, heads=8) == 4096
    assert KQ.bytes_per_token("int8", 128, 32, heads=8) == 2 * 8 * (128 + 4 / 32)   # 2050
    assert KQ.bytes_per_token("fp8", 128, 64, heads=8) == 2 * 8 * (128 + 4 / 64)


# ----------------------------------------------------------------------------- the kernel


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
@pytest.mark.parametrize("group", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kernel_matches_float64_over_the_dequantized_cache(fmt, group, dtype):
    q, k, v = data(2, 8, 2, 1, 1000, dtype=dtype)               # GQA 4:1, a ragged last group
    kc, vc = KQ.quantize(k, fmt, group), KQ.quantize(v, fmt, group)
    for chunk in (None, 64, 256, 1000):
        out = KQ.decode_attention(q, kc, vc, chunk=chunk)
        tol = 2e-3 if dtype == torch.float16 else 1.6e-2
        assert rel(out, KQ.reference(q, kc, vc)) < tol


@pytest.mark.parametrize("T", [1, 4, 16])
def test_several_query_rows(T):
    q, k, v = data(1, 4, 4, T, 3000)
    kc, vc = KQ.quantize(k, "int8"), KQ.quantize(v, "int8")
    assert rel(KQ.decode_attention(q, kc, vc), KQ.reference(q, kc, vc)) < 2e-3


def test_grouped_block_rule():
    assert KQ.grouped_block(4, 1, 32, False) == 32        # a block lies in one V group
    assert KQ.grouped_block(4, 1, 64, False) == 64
    assert KQ.grouped_block(1, 1, 32, False) == 0         # one row: the one row kernel
    assert KQ.grouped_block(4, 1, 32, True) == 0          # per channel keys: the one row kernel
    assert KQ.grouped_block(8, 16, 32, False) == 0        # 128 rows: above GRP_MAX_ROWS


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("Hq, Hk, T, group", [(32, 8, 1, 32), (8, 2, 3, 16), (16, 1, 4, 128), (4, 4, 2, 64)])
def test_grouped_query_kernel_matches_float64_with_tail_and_prefix(fmt, dtype, Hq, Hk, T, group):
    """The grouped kernel (all query heads and rows of a KV head in one tile): float64 over the
    dequantized cache, with a float tail and with a fixed capacity prefix, both modes, two runs
    equal."""
    B, N = 2, 1500
    q, k, v = data(B, Hq, Hk, T, N, dtype=dtype, seed=5)
    kc, vc = KQ.quantize(k, fmt, group), KQ.quantize(v, fmt, group)
    assert KQ.grouped_block(Hq // Hk, T, group, False)
    tol = 2e-3 if dtype == torch.float16 else 1.6e-2
    tk = torch.randn(B, Hk, 40, 128, device="cuda").to(dtype)
    tv = torch.randn(B, Hk, 40, 128, device="cuda").to(dtype)
    tlen = torch.tensor([29], dtype=torch.int32, device="cuda")
    pl = torch.tensor([1100], dtype=torch.int32, device="cuda")
    for det in (False, True):
        for kw in ({}, {"tail": (tk, tv, tlen)}, {"prefix_len": pl}):
            out = KQ.decode_attention(q, kc, vc, deterministic=det, **kw)
            assert torch.equal(out, KQ.decode_attention(q, kc, vc, deterministic=det, **kw))
            ref = KQ.reference(q, kc, vc, tail=kw.get("tail"), prefix_len=kw.get("prefix_len"))
            assert rel(out, ref) < tol, (det, list(kw))


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
@pytest.mark.parametrize("spread", [1.0, 4.0])
def test_error_against_the_unquantized_cache_stays_within_the_bound(fmt, spread):
    """spread 4 makes the scores peaked, where key rounding matters most."""
    q, k, v = data(2, 8, 8, 1, 4096, spread=spread, seed=1)
    kc, vc = KQ.quantize(k, fmt, 32), KQ.quantize(v, fmt, 32)
    out = KQ.decode_attention(q, kc, vc)
    ref = exact(q, k, v)
    kernel_part = 2e-3 * KQ.reference(q, kc, vc).abs().max().item()   # the kernel's own rounding
    bound = KQ.error_bound(q, kc, vc, k, v)
    assert ((out.double() - ref).abs() <= bound + kernel_part).all()


def test_quantized_error_against_the_fp16_cache_baseline():
    """The unquantized float16 cache (eager SDPA) is the baseline. Quantization rounding dominates
    it (measured: int8 about 36 times, fp8 about 98 times on this data), the order holds, and both
    stay within their proven bounds (test above). The ratios are recorded, not a guarantee."""
    q, k, v = data(1, 8, 8, 1, 4096, seed=2)
    ref = exact(q, k, v)
    base = (torch.nn.functional.scaled_dot_product_attention(q, k, v).double() - ref).abs().max().item()
    errs = {}
    for fmt in ("int8", "fp8"):
        kc, vc = KQ.quantize(k, fmt, 32), KQ.quantize(v, fmt, 32)
        errs[fmt] = (KQ.decode_attention(q, kc, vc).double() - ref).abs().max().item()
    assert base < errs["int8"] < errs["fp8"]
    assert errs["int8"] < 100 * base and errs["fp8"] < 1000 * base


# ----------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("det", [False, True])
def test_run_to_run_equal(det):
    q, k, v = data(8, 8, 8, 1, 5000)
    kc, vc = KQ.quantize(k, "int8"), KQ.quantize(v, "int8")
    first = KQ.decode_attention(q, kc, vc, deterministic=det)
    for _ in range(5):
        assert torch.equal(KQ.decode_attention(q, kc, vc, deterministic=det), first)


def test_deterministic_chunks_ignore_the_sm_count_and_the_batch(monkeypatch):
    for B in (1, 4, 32):
        assert KQ.chunk_for(B, 8, 8192, deterministic=True, sms=26) == KQ.DET_CHUNK
        assert KQ.chunk_for(B, 8, 8192, deterministic=True, sms=70) == KQ.DET_CHUNK
    # since 2026-11-25 the default mode chunk follows the shapes only, never the SM count
    assert KQ.chunk_for(1, 8, 8192, deterministic=False, sms=26) == \
        KQ.chunk_for(1, 8, 8192, deterministic=False, sms=70)
    assert KQ.chunk_for(1, 32, 1024, deterministic=False, Hkv=8) < 1024   # short contexts split
    assert KQ.chunk_for(1, 32, 1024, deterministic=True) == KQ.DET_CHUNK
    q, k, v = data(2, 8, 8, 1, 4096)
    kc, vc = KQ.quantize(k, "fp8"), KQ.quantize(v, "fp8")
    outs = []
    for n in (26, 70):
        monkeypatch.setattr(KQ, "_sms", lambda n=n: n)
        outs.append(KQ.decode_attention(q, kc, vc, deterministic=True))
    assert torch.equal(outs[0], outs[1])
    one = KQ.decode_attention(q[1:], KQ.QuantKV(kc.data[1:], kc.scale[1:], "fp8", 32),
                              KQ.QuantKV(vc.data[1:], vc.scale[1:], "fp8", 32), deterministic=True)
    assert torch.equal(outs[0][1:], one)


def test_paged_and_dense_caches_give_the_same_bits():
    B, H, page, N, D = 3, 4, 64, 1000, 128
    q, k, v = data(B, 8, H, 1, N)
    npg = -(-N // page)
    pool_k = torch.zeros(B * npg + 5, H, page, D, device="cuda", dtype=torch.float16)
    pool_v = torch.zeros_like(pool_k)
    perm = torch.randperm(B * npg + 5, device="cuda")[: B * npg].view(B, npg)    # scattered pages
    kp = torch.cat([k, k.new_zeros(B, H, npg * page - N, D)], 2).view(B, H, npg, page, D)
    vp = torch.cat([v, v.new_zeros(B, H, npg * page - N, D)], 2).view(B, H, npg, page, D)
    pool_k[perm] = kp.permute(0, 2, 1, 3, 4)
    pool_v[perm] = vp.permute(0, 2, 1, 3, 4)
    for fmt in ("int8", "fp8"):
        pk, pv = KQ.quantize_pages(pool_k, fmt, 32), KQ.quantize_pages(pool_v, fmt, 32)
        paged = KQ.decode_attention(q, KQ.gather_pages(pk, perm, N), KQ.gather_pages(pv, perm, N))
        dense = KQ.decode_attention(q, KQ.quantize(k, fmt, 32), KQ.quantize(v, fmt, 32))
        assert torch.equal(paged, dense)


# ----------------------------------------------------------------------------- refusals


def test_unsupported_requests_raise_ts701():
    assert codes.slug("TS701") == "kv_quant_unsupported"
    q, k, v = data(1, 4, 4, 1, 300)
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.quantize(k, "int3")
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.quantize(k, "int8", group=48)
    kc, vc = KQ.quantize(k, "int8"), KQ.quantize(v, "int8")
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.decode_attention(torch.randn(1, 4, 17, 128, device="cuda").half(), kc, vc)
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.decode_attention(q, kc, KQ.quantize(v, "fp8"))
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.decode_attention(q, kc, vc, chunk=100)
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.decode_attention(q.cpu(), kc, vc)


def test_verification_catches_a_wrong_output():
    q, k, v = data(1, 4, 4, 1, 500)
    kc, vc = KQ.quantize(k, "int8"), KQ.quantize(v, "int8")
    out = KQ.decode_attention(q, kc, vc)
    with pytest.raises(KernelMismatchError):
        KQ._check(q, kc, vc, 128 ** -0.5, out * 1.01 + 0.01)
    assert math.isfinite(out.double().abs().max().item())


# ----------------------------------------------------------------------------- per channel K scales


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_per_channel_keys_match_float64_and_stay_within_the_bound(fmt):
    q, k, v = data(2, 8, 2, 1, 3000, seed=4)
    k = k.clone()
    k[..., :4] *= 8                                           # a few outlier channels
    kc = KQ.quantize(k, fmt, 4096, channel=True)              # the memory of groups of 32
    vc = KQ.quantize(v, fmt, 32)
    assert kc.channel and kc.scale.shape == (2, 2, 1, 128)
    out = KQ.decode_attention(q, kc, vc)
    assert rel(out, KQ.reference(q, kc, vc)) < 2e-3
    kernel_part = 2e-3 * KQ.reference(q, kc, vc).abs().max().item()
    assert ((out.double() - exact(q, k, v)).abs() <= KQ.error_bound(q, kc, vc, k, v) + kernel_part).all()


def test_per_channel_int8_beats_per_group_on_outlier_channels_at_equal_memory():
    q, k, v = data(1, 8, 8, 1, 4096, seed=5)
    k = k.clone()
    k[..., :4] *= 20
    ref = exact(q, k, v)
    vc = KQ.quantize(v, "int8", 32)
    err = {lay: (KQ.decode_attention(q, kc, vc).double() - ref).abs().max().item()
           for lay, kc in (("group", KQ.quantize(k, "int8", 32)),
                           ("channel", KQ.quantize(k, "int8", 4096, channel=True)))}
    assert err["channel"] < err["group"] / 2
    assert KQ.bytes_per_token("int8", 128, 32, 8, k_channel_block=4096) == \
        KQ.bytes_per_token("int8", 128, 32, 8)


def test_per_channel_paged_equals_dense_and_v_must_be_per_group():
    B, H, page, N, D = 2, 4, 64, 700, 128
    q, k, v = data(B, 8, H, 1, N, seed=6)
    npg = -(-N // page)
    pool = torch.zeros(B * npg, H, page, D, device="cuda", dtype=torch.float16)
    table = torch.arange(B * npg, device="cuda").view(B, npg).flip(1)
    kp = torch.cat([k, k.new_zeros(B, H, npg * page - N, D)], 2).view(B, H, npg, page, D)
    pool[table] = kp.permute(0, 2, 1, 3, 4)
    pk = KQ.quantize_pages(pool, "int8", 64, channel=True)
    paged = KQ.gather_pages(pk, table, N)
    dense = KQ.quantize(k, "int8", 64, channel=True)
    assert torch.equal(paged.data, dense.data) and torch.equal(paged.scale, dense.scale)
    vc = KQ.quantize(v, "int8", 32)
    assert torch.equal(KQ.decode_attention(q, paged, vc), KQ.decode_attention(q, dense, vc))
    with pytest.raises(KQ.KVQuantError, match="TS701"):
        KQ.decode_attention(q, dense, KQ.quantize(v, "int8", 64, channel=True))


@pytest.mark.parametrize("fmt", ["int8", "fp8"])
def test_fixed_capacity_with_a_live_length_is_bitwise_the_exact_cache(fmt):
    """A cache allocated to its capacity with the live keys on the device (prefix_len) gives the bits
    of a cache of exactly the live keys at the same chunk size; in deterministic mode it chunks by
    DET_CHUNK always, which equals the exact cache from MIN_SERIAL_KEYS live keys on."""
    q, k, v = data(2, 8, 2, 1, 4096, seed=8)
    kc, vc = KQ.quantize(k, fmt, 32), KQ.quantize(v, fmt, 32)
    for n in (700, 2500):
        pl = torch.tensor([n], dtype=torch.int32, device="cuda")
        cut = lambda c: KQ.QuantKV(c.data[:, :, :n].contiguous(), c.scale[:, :, : -(-n // 32)].contiguous(), fmt, 32)  # noqa: E731
        big = KQ.decode_attention(q, kc, vc, prefix_len=pl, deterministic=True)
        exact = KQ.decode_attention(q, cut(kc), cut(vc), deterministic=True, chunk=KQ.DET_CHUNK)
        assert torch.equal(big, exact)
        assert rel(big, KQ.reference(q, kc, vc, prefix_len=pl)) < 2e-3
