"""Keyed random numbers (truestar.rng): bitwise equal on every backend, right distribution,
independent across keys, sites, streams and counters. See docs/RANDOM.md."""

from __future__ import annotations

import math

import pytest
import torch

from truestar import rng
from truestar.ir.reference import uniform01

CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    TRITON = CUDA
except ImportError:  # pragma: no cover
    TRITON = False
try:
    import numba  # noqa: F401
    NUMBA = True
except ImportError:  # pragma: no cover
    NUMBA = False

CASES = [
    ("uniform", torch.float32, {}), ("uniform", torch.float64, {}), ("uniform", torch.float16, {}),
    ("uniform", torch.bfloat16, {}), ("normal", torch.float32, {}), ("normal", torch.float64, {}),
    ("normal", torch.float16, {}), ("normal", torch.bfloat16, {}),
    ("bernoulli", torch.bool, {"p": 0.3}), ("bernoulli", torch.float32, {"p": 0.9}),
    ("randint", torch.int64, {"low": -5, "high": 17}), ("randint", torch.int32, {"low": 0, "high": 1000}),
]


def _backends():
    out = [("torch", "cpu")]
    if NUMBA:
        out.append(("numba", "cpu"))
    if CUDA:
        out.append(("torch", "cuda"))
    if TRITON:
        out.append(("triton", "cuda"))
    return out


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.cpu(), b.cpu()
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.view(iv), b.view(iv))
    return torch.equal(a, b)


def test_multinomial_generator_is_a_special_case():
    """uniform01 (what multinomial draws) is float64 uniform at stream_key(seed, offset)."""
    for seed, offset in [(0, 0), (7, 3), (2 ** 64 - 1, 12345)]:
        k = rng.stream_key(seed, offset)
        for i in range(200):
            assert rng.element("uniform", k, i, torch.float64) == uniform01(seed, offset, i)


@pytest.mark.parametrize("dist,dtype,kw", CASES, ids=[f"{d}-{str(t)[6:]}" for d, t, _ in CASES])
def test_every_backend_matches_the_definition_bit_for_bit(dist, dtype, kw):
    shape = (37, 61)   # not a multiple of any block or vector width
    ref = rng.generate(dist, shape, dtype, 2 ** 63 + 11, "site", (4, 9), backend="python", **kw)
    for backend, device in _backends():
        got = rng.generate(dist, shape, dtype, 2 ** 63 + 11, "site", (4, 9), device=device, backend=backend, **kw)
        assert got.device.type == device
        assert _same_bits(got, ref), f"{backend} on {device} differs from the definition"


@pytest.mark.parametrize("dist,dtype", [("uniform", torch.float32), ("normal", torch.float32),
                                        ("normal", torch.float64), ("bernoulli", torch.bool)])
def test_large_draws_are_bitwise_equal_across_backends(dist, dtype):
    n = 4_000_000
    kw = {"p": 0.9} if dist == "bernoulli" else {}
    outs = {b + "/" + d: rng.generate(dist, (n,), dtype, 99, "big", device=d, backend=b, **kw)
            for b, d in _backends()}
    first = next(iter(outs.values()))
    for name, t in outs.items():
        assert _same_bits(t, first), name


def test_counter_offset_composes():
    """A draw produced in two pieces equals the draw produced at once (counter based)."""
    for b, d in _backends():
        whole = rng.generate("normal", (1000,), torch.float32, 5, "x", device=d, backend=b)
        a = rng.generate("normal", (400,), torch.float32, 5, "x", device=d, backend=b)
        c = rng.generate("normal", (600,), torch.float32, 5, "x", device=d, backend=b, start=400)
        assert _same_bits(torch.cat([a, c]), whole)


def test_key_accepts_signed_and_unsigned_forms():
    a = rng.generate("uniform", (64,), torch.float32, -1, "s")
    b = rng.generate("uniform", (64,), torch.float32, 2 ** 64 - 1, "s")
    assert _same_bits(a, b)
    with pytest.raises(ValueError):
        rng.generate("uniform", (4,), torch.float32, 2 ** 64, "s")


# ----------------------------------------------------------------------------- statistics

N = 1_000_000


def _ks(sample: torch.Tensor, cdf) -> float:
    x, _ = torch.sort(sample.double())
    n = x.numel()
    f = cdf(x)
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(torch.maximum(i / n - f, f - (i - 1) / n).max())


# KS critical value at alpha = 1e-4 is about 2.1 / sqrt(n); the draws are fixed (keyed), so a pass
# is deterministic, and the bar is loose enough that a correct generator never sits near it.
KS_BAR = 2.1 / math.sqrt(N)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_statistics(dtype):
    u = rng.generate("uniform", (N,), dtype, 1, "stats").double()
    assert u.min() >= 0 and u.max() < 1
    assert abs(u.mean() - 0.5) < 5 * math.sqrt(1 / 12 / N)
    assert abs(u.var() - 1 / 12) < 5 * math.sqrt(1 / 180 / N)
    assert _ks(u, lambda x: x) < KS_BAR


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normal_statistics(dtype):
    z = rng.generate("normal", (N,), dtype, 2, "stats").double()
    assert abs(z.mean()) < 5 / math.sqrt(N)
    assert abs(z.var() - 1) < 5 * math.sqrt(2 / N)
    assert abs(((z - z.mean()) ** 3).mean()) < 5 * math.sqrt(15 / N)          # skewness
    assert abs((z ** 4).mean() - 3) < 5 * math.sqrt(96 / N)                    # kurtosis
    assert _ks(z, torch.special.ndtr) < KS_BAR
    tail = float((z.abs() > 4).double().mean())
    assert abs(tail - 6.334e-5) < 5 * math.sqrt(6.334e-5 / N)                   # P(|z| > 4)


def test_normal_is_accurate_against_the_exact_quantile():
    """float64: Acklam's bound (1.15e-9 relative); float32: within a few ulps of the exact quantile
    of the float32 qq it uses."""
    n = 200_000
    K = rng.stream_key(5, "acc")
    i = torch.arange(n, dtype=torch.int64)
    h = rng._mix_t(i ^ rng.as_i64(K))
    qq = ((rng._shr(h, 11) & ((1 << 52) - 1)) * 2 + 1).double() * 2.0 ** -54
    for dtype, q in [(torch.float64, qq), (torch.float32, qq.float().double())]:
        ex = torch.special.ndtri(q)
        ex = torch.where(h < 0, -ex, ex)
        z = rng.generate("normal", (n,), dtype, 5, "acc").double()
        if dtype == torch.float64:
            assert float(((z - ex).abs() / ex.abs()).max()) < 1.2e-9
        else:
            e32 = ex.float()
            ulp = (torch.nextafter(e32, torch.tensor(math.inf)).double() - e32.double()).abs()
            assert float(((z - ex).abs() / ulp).max()) < 8


def test_bernoulli_and_randint_statistics():
    for p in (0.0, 0.1, 0.5, 0.97, 1.0):
        b = rng.generate("bernoulli", (N,), torch.bool, 3, f"b{p}", p=p).double()
        assert abs(b.mean() - p) <= 5 * math.sqrt(max(p * (1 - p), 1e-12) / N)
        if p in (0.0, 1.0):
            assert b.mean() == p
    r = rng.generate("randint", (N,), torch.int64, 4, "r", low=-3, high=7)
    assert r.min() == -3 and r.max() == 6
    counts = torch.bincount(r + 3, minlength=10).double()
    chi2 = float(((counts - N / 10) ** 2 / (N / 10)).sum())
    assert chi2 < 33.7   # chi square, 9 dof, p = 1e-4


def test_independence_across_keys_sites_streams_and_counters():
    n = 200_000
    bar = 5 / math.sqrt(n)

    def u(key, site, streams=()):
        return rng.generate("uniform", (n,), torch.float64, key, site, streams) - 0.5

    def corr(a, b):
        return float((a * b).mean() / (a.std() * b.std()))

    base = u(10, "a")
    others = {"other site": u(10, "b"), "other key": u(11, "a"), "next stream": u(10, "a", (1,)),
              "stream 0": u(10, "a", (0,)), "swapped streams": u(10, "a", (2, 1)), "streams": u(10, "a", (1, 2))}
    for name, o in others.items():
        assert abs(corr(base, o)) < bar, name
        assert not torch.equal(base, o), name
    assert abs(corr(others["swapped streams"], others["streams"])) < bar
    assert abs(corr(base[1:], base[:-1])) < bar          # lag 1 along the counter
    # bits: each of the 64 output bits is set half the time
    h = rng._mix_t(torch.arange(n, dtype=torch.int64) ^ rng.as_i64(rng.stream_key(10, "a")))
    for bit in range(64):
        frac = float(((h >> bit) & 1).double().mean())
        assert abs(frac - 0.5) < 5 * 0.5 / math.sqrt(n), bit


def test_dropout_mask_and_scaling():
    x = torch.ones(1000, 100)
    y = rng.dropout(x, 0.25, key=1, site="d")
    kept = y != 0
    assert abs(float(kept.double().mean()) - 0.75) < 5 * math.sqrt(0.75 * 0.25 / x.numel())
    assert torch.all(y[kept] == 1 / 0.75)
    assert torch.equal(rng.dropout(x, 0.0, key=1), x)
    if CUDA:
        yc = rng.dropout(x.cuda(), 0.25, key=1, site="d")
        assert _same_bits(yc, y)


def test_bad_parameters_are_clear_errors():
    with pytest.raises(ValueError, match="unknown distribution"):
        rng.generate("gamma", (2,), torch.float32, 0)
    with pytest.raises(ValueError, match="float dtype"):
        rng.generate("normal", (2,), torch.int64, 0)
    with pytest.raises(ValueError, match="probability"):
        rng.generate("bernoulli", (2,), torch.bool, 0, p=1.5)
    with pytest.raises(ValueError, match="high - low"):
        rng.generate("randint", (2,), torch.int64, 0, low=3, high=3)
    with pytest.raises(ValueError, match="dropout probability"):
        rng.dropout(torch.ones(2), 1.0, key=0)


def test_signed_zero_normals_match_on_every_backend():
    """Found by the 1e8 draw comparison (scripts/random_vs_torch.py): when float32 qq rounds to exactly
    0.5 the draw is a zero whose sign comes from bit 63. Triton compiled -z as 0 - z, which gives +0.0
    where the definition gives -0.0; the kernel now multiplies by -1.0."""
    for idx in (51936059, 71749016):
        ref = rng.generate("normal", (1,), torch.float32, 2026, "evidence", backend="python", start=idx)
        assert ref.view(torch.int32).item() == -2 ** 31          # -0.0
        for backend, device in _backends():
            got = rng.generate("normal", (1,), torch.float32, 2026, "evidence", device=device, backend=backend,
                               start=idx)
            assert _same_bits(got, ref), f"{backend} on {device}"


def _unmix(h: int) -> int:
    """The inverse of rng.mix (SplitMix64 is a bijection), so a test can pick the 64 bits h itself."""
    def unxorshift(z, s):
        x = z
        for _ in range(64 // s + 1):
            x = z ^ (x >> s)
        return x
    z = unxorshift(h, 31)
    z = (z * pow(rng._M2, -1, 1 << 64)) & rng.MASK64
    z = unxorshift(z, 27)
    z = (z * pow(rng._M1, -1, 1 << 64)) & rng.MASK64
    z = unxorshift(z, 30)
    return (z - rng._GOLDEN) & rng.MASK64


def _chosen_normal_mantissas():
    """52 bit values m (qq = (2 m + 1) 2**-54) at every place a normal kernel can round or branch."""
    ms = {0, 1, 2, 3, (1 << 52) - 1, (1 << 52) - 2}
    # 2 m + 1 with exactly 25 bits sits halfway between two float32 values: round to nearest even
    for k in (0, 1, 2, 3, 1000, 1001, (1 << 23) - 2, (1 << 23) - 1):
        ms.add((1 << 23) + k)
    # longer odd values just above and just below a float32 halfway point, both mantissa parities
    for bits in (26, 30, 40, 53):
        s = bits - 24
        for mant in ((1 << 23) + 4, (1 << 23) + 5, (1 << 24) - 1):
            for low in ((1 << (s - 1)) | 1, (1 << (s - 1)) - 1):
                v = (mant << s) | low
                if v < 1 << 53:
                    ms.add((v - 1) >> 1)
    # PPND7 central split (qq = 0.075 in float32, ulp 2**-27, i.e. 2**26 in m), its r = 5 split
    # (qq = exp(-25)) and Acklam's tail split (qq = 0.02425 in float64)
    m0 = int(0.075 * 2 ** 53)
    ms.update(m0 + k * (1 << 25) + d for k in range(-6, 7) for d in (-1, 0, 1))
    m1 = int(math.exp(-25.0) * 2 ** 53)
    ms.update(range(m1 - 40, m1 + 41))
    m2 = int(0.02425 * 2 ** 53)
    ms.update(range(m2 - 3, m2 + 4))
    return sorted(m for m in ms if 0 <= m < 1 << 52)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_normal_rounding_ties_and_branch_splits_match_the_definition(dtype):
    """The normal kernels round qq to float32 from the 53 bit integer and select one rational per
    element before evaluating it; hit those places exactly by inverting the mixer: for a chosen h,
    the stream key mix^-1(h) draws h at element 0."""
    ms = _chosen_normal_mantissas()
    assert len(ms) > 150
    for m in ms:
        for sign in (0, 1):
            h = (sign << 63) | (m << 11) | 0x5A5
            K = _unmix(h)
            assert rng.mix(K) == h
            ref = rng.generate_python("normal", (1,), dtype, K)
            outs = [("torch", rng.generate_torch("normal", (1,), dtype, K))]
            if NUMBA:
                outs.append(("numba", rng.generate_numba("normal", (1,), dtype, K)))
            if TRITON:
                outs.append(("triton", rng.generate_triton("normal", (1,), dtype, K)))
            for name, got in outs:
                assert _same_bits(got, ref), f"{name}: m={m} sign={sign} {got.item()!r} vs {ref.item()!r}"


@pytest.mark.skipif(not TRITON, reason="needs CUDA and Triton")
def test_normal_kernel_bits_do_not_depend_on_launch_settings(monkeypatch):
    """The float64 kernel loops over blocks (a persistent grid) and over tiles of each block's tail
    list. One block per SM and tiles of 16 or 32 send every block through both loops several times
    (about 25 tail draws per 512); the float32 kernel takes its block and warp count from the same
    table. Every setting must give the reference path's bits, with a counter offset."""
    n, start = 300_001, 12345
    want = {dt: rng.generate("normal", (n,), dt, 7, "launch", device="cpu", backend="torch", start=start)
            for dt in (torch.float32, torch.float64)}
    for f32_launch, f64_launch in [((1024, 8), (2048, 256, 4)), ((256, 2), (512, 16, 1)), ((2048, 4), (1024, 32, 1))]:
        monkeypatch.setitem(rng.NORMAL_LAUNCH, False, f32_launch)
        monkeypatch.setitem(rng.NORMAL_LAUNCH, True, f64_launch)
        for dt, ref in want.items():
            got = rng.generate("normal", (n,), dt, 7, "launch", device="cuda", backend="triton", start=start)
            assert _same_bits(got, ref), f"{dt} with launch {f32_launch if dt == torch.float32 else f64_launch}"


def test_float32_kernel_literals_are_the_coefficient_table():
    """The float32 normal kernel writes PPND7's coefficients as literals instead of loading COEFS32
    per thread; Triton rounds a Python float literal to float32, so each literal must already be a
    COEFS32 value (or 0, 1/2, 1) for the bits to stay those of the definition."""
    import inspect
    import re

    import numpy as np

    src = inspect.getsource(rng._triton_kernel)
    body = src[src.index("def _tk_log32"):src.index("def _tk_qq")]
    table = {float(np.float32(v)) for v in rng.COEFS32} | {0.0, 0.5, 1.0}
    code = chr(10).join(line.split("#")[0] for line in body.splitlines())
    lits = re.findall(r"(?<![\w.])(\d+\.\d+)", code)
    assert len(lits) > 40
    for lit in lits:
        assert float(lit) in table, f"{lit} is not a float32 value of COEFS32"
