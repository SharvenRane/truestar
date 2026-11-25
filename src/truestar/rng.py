"""Keyed counter based random numbers that are bitwise identical on every backend.

Every random value is a pure function of three things: a 64 bit **key**, a **site** (a 64 bit id,
usually the hash of a name such as ``"block.dropout"``) and the element's position in the output
(its row major flat index). There is no generator object and no global state, so the same
(key, site, index) gives the same bits on the CPU, on CUDA, in the float64 reference interpreter,
in a compiled numba kernel and in a compiled Triton kernel. See docs/RANDOM.md.

The bits come from the SplitMix64 mixer that ``multinomial`` already uses
(``truestar.ir.reference.uniform01``):

    stream key  K = fold(fold(key, site), stream_1) ...      fold(a, b) = mix(mix(a) ^ b)
    element i   h = mix(K ^ i)                              (64 bits)

``uniform01(seed, offset, counter)`` is exactly ``uniform`` in float64 at stream key
``stream_key(seed, offset)`` and index ``counter``: the sampling op is a special case.

From the 64 bits ``h`` each distribution is defined with integer operations and with float
operations IEEE 754 rounds exactly (add, subtract, multiply, divide, square root), in one fixed
order, with no fused multiply add. That is what makes the bits portable:

* ``uniform`` in a float dtype with p significand bits: ``(h >> (64 - p)) * 2**-p``, exact in that
  dtype, in [0, 1). p is 24 (float32), 53 (float64), 11 (float16), 8 (bfloat16).
* ``bernoulli(p)``: ``(h >> 11) < ceil(p * 2**53)``, which is exactly ``(h >> 11) * 2**-53 < p``.
* ``randint(low, high)``: ``low + (((h >> 32) * (high - low)) >> 32)``; needs high - low <= 2**31
  (the bias of this multiply shift is under (high - low) / 2**32 per value).
* ``normal``: the sign is bit 63 of ``h``; ``qq = (2 m + 1) 2**-54`` in (0, 1/2) with m the low 52
  bits of ``h >> 11``; the result is ``-ndtri(qq)`` or ``ndtri(qq)``, with a logarithm written as a
  fixed atanh series (``_log``). float64 output is computed in float64 with Acklam's rational
  inverse normal CDF (relative error under 1.2e-9 in exact arithmetic). float32, float16 and
  bfloat16 output is computed in float32 arithmetic (qq rounded once to float32) with Wichura's
  AS241 PPND7 (relative error about 1.2e-7, all coefficients positive, so float32 evaluation does
  not cancel; Acklam's central rational was measured up to 3000 ulps off in float32), then rounded
  from float32 for the half types. float64 throughput on consumer GPUs is 1/64 of float32.
  Library ``log``, ``sqrt`` (torch's CPU kernel is 1 ulp off) and ``erfinv`` are never trusted.

Four implementations exist and tests hold them bitwise equal: ``python`` (the definition, one
element at a time), ``torch`` (vectorized torch ops on any device; the reference interpreter uses it
on the CPU), ``numba`` (compiled CPU kernel) and ``triton`` (compiled CUDA kernel).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB

DISTS = ("uniform", "normal", "bernoulli", "randint")
FLOAT_BITS = {torch.float32: 24, torch.float64: 53, torch.float16: 11, torch.bfloat16: 8}
INT_DTYPES = (torch.int64, torch.int32)
BACKENDS = ("python", "torch", "numba", "triton")

# Acklam's rational approximation of the inverse normal CDF (P. J. Acklam, 2003; the coefficients
# as published on his page and copied unchanged by QuantLib).
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
_P_LOW = 0.02425
_SQRT_HALF = 0.7071067811865476
_LN2 = 0.6931471805599453

# Wichura's AS241 PPND7 (Applied Statistics 37, 1988), the 7 digit version: central region for
# |q - 1/2| <= 0.425, then r = sqrt(-log q) with r <= 5 and r > 5.
_W_A = (3.3871327179, 50.434271938, 159.29113202, 59.109374720)
_W_B = (17.895169469, 78.757757664, 67.187563600)
_W_C = (1.4234372777, 2.7568153900, 1.3067284816, 0.17023821103)
_W_D = (0.73700164250, 0.12021132975)
_W_E = (6.6579051150, 3.0812263860, 0.42868294337, 0.017337203997)
_W_F = (0.24197894225, 0.012258202635)
_W_K = (0.425, 0.180625, 5.0, 1.6)

# Constant tables, one per arithmetic precision, with a fixed layout (the Triton kernel reads them by
# these offsets: a kernel cannot read Python globals, and Triton passes Python floats as float32).
NLOG64, NLOG32 = 12, 6     # atanh series terms 1, 1/3, 1/5, ...: enough for each precision
_LOG_TERMS = tuple(1.0 / (2 * k + 1) for k in range(NLOG64))
_OA, _OB, _OC, _OD, _OPLOW = 0, 6, 11, 17, 21                     # float64 table (Acklam)
COEFS64 = _A + _B + _C + _D + (_P_LOW, _SQRT_HALF, _LN2) + _LOG_TERMS
_LOG64 = (22, 23, 24, NLOG64)                                     # sqrt(1/2), ln 2, series, terms
# float32 table (PPND7): a 0-3, b 4-6, c 7-10, d 11-12, e 13-16, f 17-18, splits 19-22
COEFS32 = tuple(float(np.float32(c)) for c in _W_A + _W_B + _W_C + _W_D + _W_E + _W_F + _W_K
                + (_SQRT_HALF, _LN2) + _LOG_TERMS[:NLOG32])
_LOG32 = (23, 24, 25, NLOG32)


# ----------------------------------------------------------------------------- keys


def mix(z: int) -> int:
    """SplitMix64 (Steele, Lea and Flood 2014): a bijection on 64 bit unsigned integers."""
    z = (z + _GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * _M1) & MASK64
    z = ((z ^ (z >> 27)) * _M2) & MASK64
    return z ^ (z >> 31)


def as_u64(v: int) -> int:
    """An int (signed or unsigned, as an int64 tensor holds it) as its unsigned 64 bit pattern."""
    if not isinstance(v, int) or isinstance(v, bool):
        raise TypeError(f"a key is an int, got {type(v).__name__}")
    if not -(1 << 63) <= v <= MASK64:
        raise ValueError(f"a key must fit in 64 bits, got {v}")
    return v & MASK64


def as_i64(v: int) -> int:
    """The signed int64 with the same 64 bits (how a key is stored in an int64 tensor)."""
    v = as_u64(v)
    return v - (1 << 64) if v >> 63 else v


def site_id(name: Union[str, int]) -> int:
    """A 64 bit id for a site name (the first 8 bytes of BLAKE2b, little endian). Ints pass through."""
    if isinstance(name, int) and not isinstance(name, bool):
        return as_u64(name)
    if not isinstance(name, str):
        raise TypeError(f"a site is a name (str) or an int, got {type(name).__name__}")
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "little")


def fold(key: int, data: Union[str, int]) -> int:
    """A new key from a key and a name or an int: ``mix(mix(key) ^ site_id(data))``."""
    return mix(mix(as_u64(key)) ^ site_id(data))


def stream_key(key: int, site: Union[str, int], streams: Iterable[int] = ()) -> int:
    """The key one random op draws from: the key folded with its site, then with each stream value
    (for example the index of every enclosing loop)."""
    k = fold(key, site)
    for s in streams:
        k = fold(k, as_u64(int(s)))
    return k


def call_key(seed: int, call: int) -> int:
    """The key of call number ``call`` of a compiled function whose program seed is ``seed``."""
    return fold(seed, as_u64(call))


# ----------------------------------------------------------------------------- parameters


def check_params(dist: str, dtype: torch.dtype, p: Optional[float] = None, low: int = 0,
                 high: Optional[int] = None) -> None:
    """Raise ValueError when a distribution, dtype or parameter is not supported."""
    if dist not in DISTS:
        raise ValueError(f"unknown distribution {dist!r}; supported: {', '.join(DISTS)}")
    if dist in ("uniform", "normal") and dtype not in FLOAT_BITS:
        raise ValueError(f"{dist} needs a float dtype (float16, bfloat16, float32, float64), got {dtype}")
    if dist == "bernoulli":
        if dtype != torch.bool and dtype not in FLOAT_BITS:
            raise ValueError(f"bernoulli returns bool or a float dtype, got {dtype}")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"bernoulli needs a probability p in [0, 1], got {p!r}")
    if dist == "randint":
        if dtype not in INT_DTYPES:
            raise ValueError(f"randint returns int64 or int32, got {dtype}")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (low, high)):
            raise ValueError(f"randint needs int low and high, got {low!r}, {high!r}")
        if not 0 < high - low <= 1 << 31:
            raise ValueError(f"randint needs 0 < high - low <= 2**31, got low={low}, high={high}")
        if dtype == torch.int32 and not (-(1 << 31) <= low and high <= 1 << 31):
            raise ValueError(f"randint range [{low}, {high}) does not fit int32")


def bernoulli_threshold(p: float) -> int:
    """T = ceil(p * 2**53): ``(h >> 11) < T`` is exactly ``(h >> 11) * 2**-53 < p``."""
    return math.ceil(float(p) * 2.0 ** 53)


# ----------------------------------------------------------------------------- the definition


def _log(x, f32: bool):
    """Natural log from IEEE exact operations only: x = m 2^e with m in [sqrt(1/2), sqrt(2)),
    log m = 2 atanh(s), s = (m - 1) / (m + 1), as a fixed series in s^2. ``f32``: every operation in
    float32 (numpy float32 scalars), else in float64."""
    c, (osqh, oln2, olog, n) = (COEFS32, _LOG32) if f32 else (COEFS64, _LOG64)
    cv = np.float32 if f32 else float
    m, e = math.frexp(float(x))
    m = cv(m)
    if m < cv(c[osqh]):
        m = m + m
        e = e - 1
    one = cv(1.0)
    s = (m - one) / (m + one)
    s2 = s * s
    t = cv(c[olog + n - 1])
    for k in range(n - 2, -1, -1):
        t = t * s2 + cv(c[olog + k])
    return cv(e) * cv(c[oln2]) + (s + s) * t


def _ndtri_low(qq, f32: bool):
    """The inverse normal CDF at qq in (0, 1/2), a negative number: Acklam in float64 arithmetic,
    PPND7 in float32 arithmetic (numpy float32 scalars)."""
    if f32:
        c = [np.float32(v) for v in COEFS32]
        one = np.float32(1.0)
        q = qq - np.float32(0.5)
        if -q <= c[19]:
            r = c[20] - q * q
            num = ((c[3] * r + c[2]) * r + c[1]) * r + c[0]
            den = ((c[6] * r + c[5]) * r + c[4]) * r + one
            return (q * num) / den
        r = np.sqrt(-_log(qq, True))
        if r <= c[21]:
            r = r - c[22]
            num = ((c[10] * r + c[9]) * r + c[8]) * r + c[7]
            den = (c[12] * r + c[11]) * r + one
        else:
            r = r - c[21]
            num = ((c[16] * r + c[15]) * r + c[14]) * r + c[13]
            den = (c[18] * r + c[17]) * r + one
        return -(num / den)
    c = COEFS64
    if qq < c[_OPLOW]:
        t = math.sqrt(-2.0 * _log(qq, False))
        C, D = c[_OC:_OC + 6], c[_OD:_OD + 4]
        num = ((((C[0] * t + C[1]) * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5]
        den = (((D[0] * t + D[1]) * t + D[2]) * t + D[3]) * t + 1.0
        return num / den
    A, B = c[_OA:_OA + 6], c[_OB:_OB + 5]
    r = qq - 0.5
    s = r * r
    num = (((((A[0] * s + A[1]) * s + A[2]) * s + A[3]) * s + A[4]) * s + A[5]) * r
    den = ((((B[0] * s + B[1]) * s + B[2]) * s + B[3]) * s + B[4]) * s + 1.0
    return num / den


def element(dist: str, K: int, i: int, dtype: torch.dtype, p: Optional[float] = None, low: int = 0,
            high: Optional[int] = None):
    """The value at flat index ``i`` of a draw with stream key ``K``: the definition every
    implementation must match bit for bit."""
    h = mix((K ^ i) & MASK64)
    if dist == "uniform":
        bits = FLOAT_BITS[dtype]
        return (h >> (64 - bits)) * 2.0 ** -bits
    if dist == "bernoulli":
        hit = (h >> 11) < bernoulli_threshold(p)
        return hit if dtype == torch.bool else float(hit)
    if dist == "randint":
        return low + (((h >> 32) * (high - low)) >> 32)
    m = (h >> 11) & ((1 << 52) - 1)
    qq = (2 * m + 1) * 2.0 ** -54
    f32 = dtype != torch.float64
    if f32:
        qq = np.float32(qq)
    z = _ndtri_low(qq, f32)
    z = -z if h >> 63 else z
    if dtype in (torch.float16, torch.bfloat16):
        return float(torch.tensor(float(z), dtype=torch.float32).to(dtype))
    return float(z)


def generate_python(dist, shape, dtype, K, p=None, low=0, high=None, start: int = 0) -> torch.Tensor:
    n = math.prod(shape)
    vals = [element(dist, K, start + i, dtype, p, low, high) for i in range(n)]
    return torch.tensor(vals, dtype=dtype).reshape(shape)


# ----------------------------------------------------------------------------- torch


def _shr(z: torch.Tensor, k: int) -> torch.Tensor:
    """Logical right shift of an int64 tensor holding unsigned bits."""
    return (z >> k) & ((1 << (64 - k)) - 1)


def _mix_t(z: torch.Tensor) -> torch.Tensor:
    z = z + as_i64(_GOLDEN)
    z = (z ^ _shr(z, 30)) * as_i64(_M1)
    z = (z ^ _shr(z, 27)) * as_i64(_M2)
    return z ^ _shr(z, 31)


def _sqrt_t(x: torch.Tensor) -> torch.Tensor:
    """Correctly rounded square root. torch.sqrt on the CPU is not: its vectorized float64 kernel was
    measured 1 ulp off for 1 of 3000 inputs (torch 2.11), so CPU tensors go through numpy, whose
    sqrt is the hardware instruction. CUDA's sqrt is IEEE in torch's build (no fast math)."""
    if x.device.type == "cpu":
        return torch.from_numpy(np.sqrt(x.numpy()))
    return torch.sqrt(x)


def _log_t(x: torch.Tensor, f32: bool) -> torch.Tensor:
    c, (osqh, oln2, olog, n) = (COEFS32, _LOG32) if f32 else (COEFS64, _LOG64)
    m, e = torch.frexp(x)
    e = e.to(x.dtype)
    low = m < c[osqh]
    m = torch.where(low, m + m, m)
    e = torch.where(low, e - 1.0, e)
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    t = torch.full_like(s, c[olog + n - 1])
    for k in range(n - 2, -1, -1):
        t = t * s2 + c[olog + k]
    return e * c[oln2] + (s + s) * t


def _ppnd7_t(qq: torch.Tensor) -> torch.Tensor:
    c = COEFS32
    q = qq - 0.5
    r = c[20] - q * q
    num = ((c[3] * r + c[2]) * r + c[1]) * r + c[0]
    den = ((c[6] * r + c[5]) * r + c[4]) * r + 1.0
    central = (q * num) / den
    R = _sqrt_t(-_log_t(qq, True))
    r = R - c[22]
    num = ((c[10] * r + c[9]) * r + c[8]) * r + c[7]
    den = (c[12] * r + c[11]) * r + 1.0
    mid = -(num / den)
    r = R - c[21]
    num = ((c[16] * r + c[15]) * r + c[14]) * r + c[13]
    den = (c[18] * r + c[17]) * r + 1.0
    far = -(num / den)
    return torch.where(-q <= c[19], central, torch.where(R <= c[21], mid, far))


def _ndtri_low_t(qq: torch.Tensor, f32: bool) -> torch.Tensor:
    if f32:
        return _ppnd7_t(qq)
    c = COEFS64
    t = _sqrt_t(-2.0 * _log_t(qq, False))
    C, D = c[_OC:_OC + 6], c[_OD:_OD + 4]
    num = ((((C[0] * t + C[1]) * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5]
    den = (((D[0] * t + D[1]) * t + D[2]) * t + D[3]) * t + 1.0
    tail = num / den
    A, B = c[_OA:_OA + 6], c[_OB:_OB + 5]
    r = qq - 0.5
    s = r * r
    num = (((((A[0] * s + A[1]) * s + A[2]) * s + A[3]) * s + A[4]) * s + A[5]) * r
    den = ((((B[0] * s + B[1]) * s + B[2]) * s + B[3]) * s + B[4]) * s + 1.0
    return torch.where(qq < c[_OPLOW], tail, num / den)


def _from_bits_t(h: torch.Tensor, dist, dtype, p, low, high) -> torch.Tensor:
    if dist == "uniform":
        bits = FLOAT_BITS[dtype]
        u = _shr(h, 64 - bits).to(torch.float64) * 2.0 ** -bits
        return u.to(dtype)
    if dist == "bernoulli":
        hit = _shr(h, 11) < bernoulli_threshold(p)
        return hit if dtype == torch.bool else hit.to(dtype)
    if dist == "randint":
        return (low + _shr(_shr(h, 32) * (high - low), 32)).to(dtype)
    f32 = dtype != torch.float64
    m = _shr(h, 11) & ((1 << 52) - 1)
    qq = (m * 2 + 1).to(torch.float64) * 2.0 ** -54
    if f32:
        qq = qq.to(torch.float32)
    z = _ndtri_low_t(qq, f32)
    z = torch.where(h < 0, -z, z)          # bit 63 set
    return z if dtype in (torch.float32, torch.float64) else z.to(dtype)


_CHUNK = 1 << 22


def generate_torch(dist, shape, dtype, K, p=None, low=0, high=None, device="cpu",
                   start: int = 0) -> torch.Tensor:
    """Vectorized torch ops on ``device``, in chunks of 4M elements to bound temporary memory."""
    n = math.prod(shape)
    out = torch.empty(n, dtype=dtype, device=device)
    ks = as_i64(K)
    for lo in range(0, n, _CHUNK):
        hi = min(n, lo + _CHUNK)
        i = torch.arange(start + lo, start + hi, dtype=torch.int64, device=device)
        out[lo:hi] = _from_bits_t(_mix_t(i ^ ks), dist, dtype, p, low, high)
    return out.reshape(shape)


def generate_batch(dist, shape, dtype, key, site, streams=(), p=None, low=0, high=None,
                   device="cpu") -> torch.Tensor:
    """A batch of draws, one per row: row k of the result (``shape[0]`` rows) is bit for bit
    ``generate(dist, shape[1:], dtype, key, site, list(streams) + [k])``, the draw with the batch
    index as one more stream. That is what ``vmap`` gives each element of a ``random`` op
    (``random_batched``, truestar/ir/random_batched.py). Vectorized over rows and elements with the
    torch implementation: the per row keys ``fold(K, k) = mix(mix(K) ^ k)`` are one ``_mix_t`` over
    ``arange(rows)``, so no Python loop over the batch."""
    shape = tuple(int(d) for d in shape)
    check_params(dist, dtype, p, low, high)
    if not shape:
        raise ValueError("generate_batch needs a leading batch dim")
    n, inner = shape[0], shape[1:]
    m = math.prod(inner)
    dev = torch.device(device)
    K = stream_key(as_u64(int(key)), site, streams)
    rows = _mix_t(torch.arange(n, dtype=torch.int64, device=dev) ^ as_i64(mix(K)))   # fold(K, k)
    out = torch.empty(n * m, dtype=dtype, device=dev)
    if m == 0 or n == 0:
        return out.reshape(shape)
    for lo in range(0, n * m, _CHUNK):
        hi = min(n * m, lo + _CHUNK)
        j = torch.arange(lo, hi, dtype=torch.int64, device=dev)
        out[lo:hi] = _from_bits_t(_mix_t((j % m) ^ rows[j // m]), dist, dtype, p, low, high)
    return out.reshape(shape)


# ----------------------------------------------------------------------------- numba (CPU kernel)

_NUMBA: Dict[str, Any] = {}


def _numba_kernel():
    if "k" in _NUMBA:
        return _NUMBA["k"]
    from .deps import require

    require("numba", "compiled random numbers on the CPU")
    import numba

    u64, f32 = np.uint64, np.float32
    G, M1, M2 = u64(_GOLDEN), u64(_M1), u64(_M2)
    S11, S12, S27, S30, S31, S32, S63 = (u64(v) for v in (11, 12, 27, 30, 31, 32, 63))
    MASK52 = u64((1 << 52) - 1)
    ONE32, HALF32 = f32(1.0), f32(0.5)

    @numba.njit(cache=False, fastmath=False, nogil=True)
    def mixu(z):
        z = z + G
        z = (z ^ (z >> S30)) * M1
        z = (z ^ (z >> S27)) * M2
        return z ^ (z >> S31)

    @numba.njit(cache=False, fastmath=False, nogil=True)
    def log64(x, c):
        m, e = math.frexp(x)
        ef = float(e)
        if m < c[22]:
            m = m + m
            ef = ef - 1.0   # float64 table: sqrt(1/2) at 22, ln 2 at 23, series from 24
        s = (m - 1.0) / (m + 1.0)
        s2 = s * s
        t = c[24 + 11]
        for k in range(10, -1, -1):
            t = t * s2 + c[24 + k]
        return ef * c[23] + (s + s) * t

    @numba.njit(cache=False, fastmath=False, nogil=True)
    def ndtri64(qq, c):
        if qq < c[21]:
            t = math.sqrt(-2.0 * log64(qq, c))
            num = ((((c[11] * t + c[12]) * t + c[13]) * t + c[14]) * t + c[15]) * t + c[16]
            den = (((c[17] * t + c[18]) * t + c[19]) * t + c[20]) * t + 1.0
            return num / den
        r = qq - 0.5
        s = r * r
        num = (((((c[0] * s + c[1]) * s + c[2]) * s + c[3]) * s + c[4]) * s + c[5]) * r
        den = ((((c[6] * s + c[7]) * s + c[8]) * s + c[9]) * s + c[10]) * s + 1.0
        return num / den

    @numba.njit(cache=False, fastmath=False, nogil=True)
    def log32(x, c):
        m64, e = math.frexp(float(x))
        m = f32(m64)
        ef = f32(e)
        if m < c[23]:
            m = m + m
            ef = ef - ONE32
        s = (m - ONE32) / (m + ONE32)
        s2 = s * s
        t = c[25 + 5]
        for k in range(4, -1, -1):
            t = t * s2 + c[25 + k]
        return ef * c[24] + (s + s) * t

    @numba.njit(cache=False, fastmath=False, nogil=True)
    def ndtri32(qq, c):
        q = qq - HALF32
        if -q <= c[19]:
            r = c[20] - q * q
            num = ((c[3] * r + c[2]) * r + c[1]) * r + c[0]
            den = ((c[6] * r + c[5]) * r + c[4]) * r + ONE32
            return (q * num) / den
        r = np.sqrt(-log32(qq, c))
        if r <= c[21]:
            r = r - c[22]
            num = ((c[10] * r + c[9]) * r + c[8]) * r + c[7]
            den = (c[12] * r + c[11]) * r + ONE32
        else:
            r = r - c[21]
            num = ((c[16] * r + c[15]) * r + c[14]) * r + c[13]
            den = (c[18] * r + c[17]) * r + ONE32
        return -(num / den)

    @numba.njit(cache=False, fastmath=False, nogil=True)
    def run(out, start, K, mode, bits, scale, c64, c32, thresh, low, span):
        # mode 0 uniform (float64 buffer), 1 bernoulli (bool), 2 randint (int64),
        # 3 normal in float32 math (float32 buffer), 4 normal in float64 math (float64 buffer)
        n = out.shape[0]
        sh = u64(64 - bits)
        for j in range(n):
            h = mixu((u64(start) + u64(j)) ^ K)
            if mode == 0:
                out[j] = float(h >> sh) * scale
            elif mode == 1:
                out[j] = (h >> S11) < thresh
            elif mode == 2:
                out[j] = low + np.int64(((h >> S32) * u64(span)) >> S32)
            else:
                qq64 = float(((h >> S11) & MASK52) * u64(2) + u64(1)) * 5.551115123125783e-17
                neg = (h >> S63) == u64(1)
                if mode == 3:
                    z = ndtri32(f32(qq64), c32)
                    out[j] = -z if neg else z
                else:
                    z = ndtri64(qq64, c64)
                    out[j] = -z if neg else z

    _NUMBA["k"] = run
    return run


def generate_numba(dist, shape, dtype, K, p=None, low=0, high=None, start: int = 0) -> torch.Tensor:
    run = _numba_kernel()
    n = math.prod(shape)
    mode = {"uniform": 0, "bernoulli": 1, "randint": 2, "normal": 3}[dist]
    if mode == 3 and dtype == torch.float64:
        mode = 4
    buf = np.empty(n, dtype={0: np.float64, 1: np.bool_, 2: np.int64, 3: np.float32, 4: np.float64}[mode])
    bits = FLOAT_BITS.get(dtype, 53)
    run(buf, np.uint64(start), np.uint64(K), mode, bits, 2.0 ** -bits,
        np.asarray(COEFS64, dtype=np.float64), np.asarray(COEFS32, dtype=np.float32),
        np.uint64(bernoulli_threshold(p) if dist == "bernoulli" else 0), int(low), int((high or 1) - low))
    t = torch.from_numpy(buf)
    if t.dtype != dtype:
        t = t.to(dtype)
    return t.reshape(shape)


# ----------------------------------------------------------------------------- triton (CUDA kernel)

_TRITON: Dict[str, Any] = {}


def _triton_kernel():
    if "k" in _TRITON:
        return _TRITON["k"]
    import triton
    import triton.language as tl

    # Offsets into the constant tables are literals: a Triton kernel cannot read module globals.
    # The _tk prefix keeps these names apart from module functions, which Triton would resolve first.
    @triton.jit
    def _tk_mixu(z):
        z = z + ((tl.full([], 0x9E3779B9, tl.uint64) << 32) | 0x7F4A7C15)
        z = (z ^ (z >> 30)) * ((tl.full([], 0xBF58476D, tl.uint64) << 32) | 0x1CE4E5B9)
        z = (z ^ (z >> 27)) * ((tl.full([], 0x94D049BB, tl.uint64) << 32) | 0x133111EB)
        return z ^ (z >> 31)

    @triton.jit
    def _tk_div(a, b, F64: tl.constexpr):
        if F64:
            return a / b                       # float64 division is div.rn.f64 (IEEE)
        else:
            return tl.math.div_rn(a, b)        # float32 '/' is approximate in Triton; this is IEEE

    @triton.jit
    def _tk_log(x, c, F64: tl.constexpr, OSQH: tl.constexpr, NLOG: tl.constexpr):
        # frexp of a positive normal x by its bits (m in [0.5, 1)), then the atanh series
        if F64:
            ub = x.to(tl.uint64, bitcast=True)
            e = ((ub >> 52) & 0x7FF).to(tl.int32) - 1022
            m = ((ub & 0xFFFFFFFFFFFFF) | (0x3FE << 52)).to(tl.float64, bitcast=True)
            ef = e.to(tl.float64)
        else:
            ub = x.to(tl.uint32, bitcast=True)
            e = ((ub >> 23) & 0xFF).to(tl.int32) - 126
            m = ((ub & 0x7FFFFF) | (0x7E << 23)).to(tl.float32, bitcast=True)
            ef = e.to(tl.float32)
        low = m < tl.load(c + OSQH)
        m = tl.where(low, m + m, m)
        ef = tl.where(low, ef - 1.0, ef)
        s = _tk_div(m - 1.0, m + 1.0, F64)
        s2 = s * s
        t = s * 0.0 + tl.load(c + OSQH + 2 + NLOG - 1)
        for k in tl.static_range(NLOG - 2, -1, -1):
            t = t * s2 + tl.load(c + OSQH + 2 + k)
        return ef * tl.load(c + OSQH + 1) + (s + s) * t

    # float64 normal draws split the work by branch. Every element needs Acklam's central rational,
    # but only the tail (qq < 0.02425, 4.85% of draws) needs the float64 logarithm, square root and
    # tail rational, at 1/64 of the float32 rate on consumer GPUs. A warp of 32 almost always holds a
    # tail element, so selecting per element would still run the logarithm for everyone; instead each
    # block computes the central values, writes the local indices of its tail elements to a scratch
    # list (a prefix sum gives each one its slot) and then runs the tail code on that dense list.
    # Each formula is the definition's, with the same operations in the same order, and each value is
    # stored at its own index, so the bits do not depend on the block size or the grid. (The same
    # split for float32 measured slower than the per element select below on a T4, an L4 and an A10G.)
    @triton.jit
    def _tk_acklam_central(qq, c):
        r = qq - 0.5
        s = r * r
        num = (((((tl.load(c + 0) * s + tl.load(c + 1)) * s + tl.load(c + 2)) * s
                 + tl.load(c + 3)) * s + tl.load(c + 4)) * s + tl.load(c + 5)) * r
        den = ((((tl.load(c + 6) * s + tl.load(c + 7)) * s + tl.load(c + 8)) * s
                + tl.load(c + 9)) * s + tl.load(c + 10)) * s + 1.0
        return num / den

    @triton.jit
    def _tk_acklam_tail(qq, c):
        t = tl.sqrt(-2.0 * _tk_log(qq, c, True, 22, 12))   # sqrt.rn.f64, the only float64 sqrt in PTX
        num = ((((tl.load(c + 11) * t + tl.load(c + 12)) * t + tl.load(c + 13)) * t
                + tl.load(c + 14)) * t + tl.load(c + 15)) * t + tl.load(c + 16)
        den = (((tl.load(c + 17) * t + tl.load(c + 18)) * t + tl.load(c + 19)) * t
               + tl.load(c + 20)) * t + 1.0
        return num / den

    @triton.jit
    def _tk_log32(x):
        # _tk_log in float32 with the COEFS32 values written as literals (each is a float32 value,
        # so Triton's float32 literal is the same number): no table load per thread
        ub = x.to(tl.uint32, bitcast=True)
        e = ((ub >> 23) & 0xFF).to(tl.int32) - 126
        m = ((ub & 0x7FFFFF) | (0x7E << 23)).to(tl.float32, bitcast=True)
        ef = e.to(tl.float32)
        low = m < 0.7071067690849304
        m = tl.where(low, m + m, m)
        ef = tl.where(low, ef - 1.0, ef)
        s = tl.math.div_rn(m - 1.0, m + 1.0)
        s2 = s * s
        t = s * 0.0 + 0.09090909361839294
        t = t * s2 + 0.1111111119389534
        t = t * s2 + 0.1428571492433548
        t = t * s2 + 0.20000000298023224
        t = t * s2 + 0.3333333432674408
        t = t * s2 + 1.0
        return ef * 0.6931471824645996 + (s + s) * t

    @triton.jit
    def _tk_ppnd7_select(qq):
        # The other float32 layout: one rational and one division per element. Each element selects
        # the coefficients and the argument of its branch first; the two degree 2 tail denominators
        # get a leading zero coefficient, and ((0 r + b) r + ...) equals (b r + ...) exactly, since
        # 0 r is a signed zero for finite r and +-0 + b is b. Returns the unsigned value and the
        # central flag (the definition negates the tail values). The coefficients are the COEFS32
        # values as literals (PPND7: a 0-3, b 4-6, c 7-10, d 11-12, e 13-16, f 17-18). The r > 5
        # branch (qq < exp(-25), about 1 draw in 7e10) is selected only in a block that has one.
        q = qq - 0.5
        cen = -q <= 0.42500001192092896
        R = tl.math.sqrt_rn(-_tk_log32(qq))
        if tl.max((R > 5.0).to(tl.int32), axis=0) > 0:
            mid = R <= 5.0
            r = tl.where(cen, 0.18062500655651093 - q * q, R - tl.where(mid, 1.600000023841858, 5.0))
            a3 = tl.where(cen, 59.109375, tl.where(mid, 0.1702382117509842, 0.01733720488846302))
            a2 = tl.where(cen, 159.2911376953125, tl.where(mid, 1.306728482246399, 0.42868295311927795))
            a1 = tl.where(cen, 50.43427276611328, tl.where(mid, 2.7568154335021973, 3.081226348876953))
            a0 = tl.where(cen, 3.3871326446533203, tl.where(mid, 1.423437237739563, 6.657905101776123))
            b2 = tl.where(cen, 78.75775909423828, tl.where(mid, 0.12021133303642273, 0.012258202768862247))
            b1 = tl.where(cen, 17.895170211791992, tl.where(mid, 0.7370016574859619, 0.2419789433479309))
        else:
            r = tl.where(cen, 0.18062500655651093 - q * q, R - 1.600000023841858)
            a3 = tl.where(cen, 59.109375, 0.1702382117509842)
            a2 = tl.where(cen, 159.2911376953125, 1.306728482246399)
            a1 = tl.where(cen, 50.43427276611328, 2.7568154335021973)
            a0 = tl.where(cen, 3.3871326446533203, 1.423437237739563)
            b2 = tl.where(cen, 78.75775909423828, 0.12021133303642273)
            b1 = tl.where(cen, 17.895170211791992, 0.7370016574859619)
        b3 = tl.where(cen, 67.18756103515625, 0.0)
        num = ((a3 * r + a2) * r + a1) * r + a0
        den = ((b3 * r + b2) * r + b1) * r + 1.0
        return tl.math.div_rn(tl.where(cen, q * num, num), den), cen

    @triton.jit
    def _tk_qq(h, F64: tl.constexpr):
        odd = ((h >> 11) & 0xFFFFFFFFFFFFF) * 2 + 1          # 53 bits: qq = odd * 2**-54
        if F64:
            return odd.to(tl.float64) * 5.551115123125783e-17
        else:
            # qq rounded once to float32 with no float64 work (1/64 rate on consumer GPUs): the
            # integer rounds to nearest even (cvt.rn) and scaling by 2**-54 is exact in float32,
            # which equals rounding the exact float64 qq to float32.
            return odd.to(tl.float32) * 5.551115123125783e-17

    @triton.jit
    def _normal64_kernel(out, n, start, k_hi, k_lo, c, scratch, BLOCK: tl.constexpr, TILE: tl.constexpr):
        pid = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        tlane = tl.arange(0, TILE)
        mine = scratch + pid.to(tl.int64) * BLOCK
        K = (k_hi.to(tl.uint64) << 32) | k_lo.to(tl.uint64)
        for b in range(pid, tl.cdiv(n, BLOCK), tl.num_programs(0)):
            base = b.to(tl.int64) * BLOCK
            offs = base + lane.to(tl.int64)
            keep = offs < n
            h = _tk_mixu((offs + start).to(tl.uint64) ^ K)
            qq = _tk_qq(h, True)
            tail = qq < tl.load(c + 21)
            z = _tk_acklam_central(qq, c)
            z = tl.where((h >> 63) == 1, z * -1.0, z)   # not -z: Triton makes that 0 - z, which loses -0.0
            # Stored for the tail elements too, and overwritten by the tail loop after the barrier: a
            # store masked per element loses vector stores and measured twice the time on an A10G.
            tl.store(out + offs, z, mask=keep)
            tail = (tail & keep).to(tl.int32)
            count = tl.sum(tail, axis=0)
            tl.store(mine + tl.cumsum(tail, axis=0) - 1, lane, mask=tail == 1)
            tl.debug_barrier()
            for j in range(0, count, TILE):
                k = j + tlane
                ok = k < count
                at = base + tl.load(mine + k, mask=ok, other=0).to(tl.int64)
                ht = _tk_mixu((at + start).to(tl.uint64) ^ K)
                zt = _tk_acklam_tail(_tk_qq(ht, True), c)
                zt = tl.where((ht >> 63) == 1, zt * -1.0, zt)
                tl.store(out + at, zt, mask=ok)
            tl.debug_barrier()   # the list is rewritten by the next block

    @triton.jit
    def _kernel(out, n, start, k_hi, k_lo, c, t_hi, t_lo, low, span, MODE: tl.constexpr,
                BITS: tl.constexpr, F64: tl.constexpr, NLOG: tl.constexpr, OUT_DTYPE: tl.constexpr,
                BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        keep = offs < n
        K = (k_hi.to(tl.uint64) << 32) | k_lo.to(tl.uint64)
        h = _tk_mixu((offs + start).to(tl.uint64) ^ K)
        if MODE == 0:
            if BITS <= 24:   # exact in float32: no float64 work on the GPU
                v = (h >> (64 - BITS)).to(tl.float32) * (1.0 / (1 << BITS))
            else:
                v = (h >> (64 - BITS)).to(tl.float64) * tl.load(c + 40)
            tl.store(out + offs, v.to(OUT_DTYPE), mask=keep)
        elif MODE == 1:
            T = (t_hi.to(tl.uint64) << 32) | t_lo.to(tl.uint64)
            tl.store(out + offs, ((h >> 11) < T).to(OUT_DTYPE), mask=keep)
        elif MODE == 2:
            r = ((h >> 32) * span.to(tl.uint64)) >> 32
            tl.store(out + offs, (r.to(tl.int64) + low).to(OUT_DTYPE), mask=keep)
        else:   # normal, float32 arithmetic, one branch selected per element
            z, cen = _tk_ppnd7_select(_tk_qq(h, False))
            # tail values are negated by the definition and again for a set sign bit: the two cancel
            z = tl.where(cen != ((h >> 63) == 1), z, z * -1.0)
            tl.store(out + offs, z.to(OUT_DTYPE), mask=keep)

    _TRITON["k"] = _kernel
    _TRITON["normal64"] = _normal64_kernel
    _TRITON["tl"] = tl
    return _kernel


_COEF_DEV: Dict[Tuple[str, bool, int], torch.Tensor] = {}
_SM_COUNT: Dict[str, int] = {}
# Normal kernel launch settings, for speed only (every setting gives the same bits, tests/test_rng.py;
# timings in docs/RANDOM.md): float32 and narrower (BLOCK, warps) for the per element select kernel;
# float64 (BLOCK, TILE, blocks per SM) for the split kernel. Measured best of the settings tried on a
# T4, an L4 and an A10G.
NORMAL_LAUNCH = {False: (1024, 8), True: (2048, 256, 4)}


def generate_triton(dist, shape, dtype, K, p=None, low=0, high=None, device="cuda",
                    start: int = 0) -> torch.Tensor:
    kernel = _triton_kernel()
    tl = _TRITON["tl"]
    n = math.prod(shape)
    dev = torch.device(device)
    mode = {"uniform": 0, "bernoulli": 1, "randint": 2, "normal": 3}[dist]
    bits = FLOAT_BITS.get(dtype, 53)
    f64 = dist != "normal" or dtype == torch.float64
    ck = (str(dev), f64, bits)
    coef = _COEF_DEV.get(ck)
    if coef is None:
        table = list(COEFS64 if f64 else COEFS32)
        table += [0.0] * (40 - len(table)) + [2.0 ** -bits]
        coef = _COEF_DEV[ck] = torch.tensor(table, dtype=torch.float64 if f64 else torch.float32, device=dev)
    tl_of = {torch.float32: tl.float32, torch.float64: tl.float64, torch.float16: tl.float16,
             torch.bfloat16: tl.bfloat16, torch.bool: tl.int1, torch.int64: tl.int64, torch.int32: tl.int32}
    out = torch.empty(n, dtype=dtype, device=dev)
    if not n:
        return out.reshape(shape)
    # normal half draws are computed in float32 and rounded once by the store
    half_out = tl.float32 if dist == "normal" and dtype in (torch.float16, torch.bfloat16) else None
    if dist == "normal" and f64:
        # a persistent grid (a few blocks per SM), so the tail lists fit a small scratch buffer
        block, tile, waves = NORMAL_LAUNCH[True]
        sms = _SM_COUNT.get(str(dev))
        if sms is None:
            sms = _SM_COUNT[str(dev)] = torch.cuda.get_device_properties(dev).multi_processor_count
        grid = max(1, min((n + block - 1) // block, waves * sms))
        scratch = torch.empty(grid * block, dtype=torch.int32, device=dev)
        _TRITON["normal64"][(grid,)](out, n, start, K >> 32, K & 0xFFFFFFFF, coef, scratch, BLOCK=block,
                                     TILE=tile, num_warps=8, enable_fp_fusion=False)
        return out.reshape(shape)
    block, warps = NORMAL_LAUNCH[False] if dist == "normal" else (1024, 4)
    thresh = bernoulli_threshold(p) if dist == "bernoulli" else 0
    kernel[((n + block - 1) // block,)](
        out, n, start, K >> 32, K & 0xFFFFFFFF, coef, thresh >> 32, thresh & 0xFFFFFFFF, int(low),
        int((high or 1) - low), MODE=mode, BITS=bits, F64=f64, NLOG=NLOG64 if f64 else NLOG32,
        OUT_DTYPE=half_out or tl_of[dtype], BLOCK=block, num_warps=warps, enable_fp_fusion=False)
    return out.reshape(shape)


# ----------------------------------------------------------------------------- entry point


def available_backends(device: str = "cpu") -> Tuple[str, ...]:
    dev = torch.device(device).type
    out = ["python", "torch"]
    if dev == "cpu":
        try:
            import numba  # noqa: F401
            out.append("numba")
        except ImportError:
            pass
    if dev == "cuda":
        try:
            import triton  # noqa: F401
            out.append("triton")
        except ImportError:
            pass
    return tuple(out)


def generate(dist: str, shape: Sequence[int], dtype: torch.dtype, key: int, site: Union[str, int] = 0,
             streams: Iterable[int] = (), p: Optional[float] = None, low: int = 0,
             high: Optional[int] = None, device: Union[str, torch.device] = "cpu",
             backend: str = "auto", start: int = 0) -> torch.Tensor:
    """Draw a tensor of ``shape`` from ``dist`` for (key, site, streams) on ``device``.

    ``backend`` "auto" uses the compiled kernel of the device when it is installed (Triton on CUDA,
    numba on the CPU) and torch ops otherwise; every backend returns the same bits. ``start`` offsets
    the element counter, so a draw can be produced in pieces."""
    shape = tuple(int(d) for d in shape)
    check_params(dist, dtype, p, low, high)
    K = stream_key(as_u64(int(key)), site, streams)
    dev = torch.device(device)
    if backend == "auto":
        have = available_backends(str(dev))
        backend = "triton" if "triton" in have else ("numba" if "numba" in have else "torch")
    if backend == "python":
        return generate_python(dist, shape, dtype, K, p, low, high, start).to(dev)
    if backend == "torch":
        return generate_torch(dist, shape, dtype, K, p, low, high, dev, start)
    if backend == "numba":
        if dev.type != "cpu":
            raise ValueError("the numba backend runs on the CPU only")
        return generate_numba(dist, shape, dtype, K, p, low, high, start)
    if backend == "triton":
        if dev.type != "cuda":
            raise ValueError("the triton backend runs on CUDA only")
        return generate_triton(dist, shape, dtype, K, p, low, high, dev, start)
    raise ValueError(f"unknown backend {backend!r}; one of {', '.join(BACKENDS)} or 'auto'")


def dropout(x: torch.Tensor, p: float, key: int, site: Union[str, int] = "dropout",
            streams: Iterable[int] = (), backend: str = "auto") -> torch.Tensor:
    """Inverted dropout with a keyed mask: elements are kept with probability 1 - p and scaled by
    1 / (1 - p). The mask is the same on every device for the same key, site and shape."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")
    if p == 0.0:
        return x
    keep = generate("bernoulli", tuple(x.shape), torch.bool, key, site, streams, p=1.0 - p,
                    device=x.device, backend=backend)
    return torch.where(keep, x * (1.0 / (1.0 - p)), torch.zeros((), dtype=x.dtype, device=x.device))
