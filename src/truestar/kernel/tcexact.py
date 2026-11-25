"""Exact tensor core blocks: half precision and TF32 dot products with the same bits on every GPU.

Measured problem (docs/CROSS_CHIP.md, scripts/xchip_mma_probe.py): Triton's ``tl.dot`` on float16,
bfloat16 and TF32 operands compiles to ``mma.sync`` (``wgmma`` for big tiles on sm_90). Inside one
instruction the products are aligned to the largest one and the bits below a fixed window are
dropped. sm_86 and sm_89 keep a product 2**-24 below the largest and drop one 2**-25 below; sm_90
and sm_120 keep 2**-25. So the same program gives one set of bits on Ampere and Ada and another on
Hopper and Blackwell, while float32 (no tensor cores) is bitwise equal everywhere.

Fix: make every tensor core instruction's sum exact, then do all rounding in software in a fixed
order. For one K block of ``KB`` columns (a constant of the kernel, never a tuned value):

1. Operand values. float16 and bfloat16 values are taken exactly as float32. TF32 inputs are
   float32 values rounded to TF32 (10 stored fraction bits) to nearest even by integer arithmetic
   on their bits, so the TF32 value does not depend on what a GPU's converter does.
2. Block scale. For each row i of the A block, ``ea_i`` is the exponent with
   ``m * (1 + 2**-7) < 2**ea_i``, m = ``max_k |a_ik|`` (read from the float32 bits of the rounded
   product), clamped to [-119, 129] (a block whose maximum is below 2**-120, zero included, uses
   -119). Likewise ``eb_j`` for each column of B.
3. Digits. ``t = a * 2**(7 - ea_i)`` has ``|t| < 128 / (1 + 2**-7) < 127.01``. Three digits, each
   the nearest integer (ties to even) of what is left: ``d0 = rn(t)``, ``t1 = (t - d0) * 128``,
   ``d1 = rn(t1)``, ``t2 = (t1 - d1) * 128``, ``d2 = rn(t2)``: ``|d0| <= 127``, ``|d1|, |d2| <= 64``,
   and every step is exact float32 arithmetic (``rn`` is ``(t + 1.5 * 2**23) - 1.5 * 2**23``, exact
   for |t| < 2**22). What is left, at most ``2**(ea_i - 22)``, is dropped: float16 and TF32 values
   within 2**10 of their block maximum and bfloat16 values within 2**13 lose nothing.
4. Exact integer dot products on the tensor cores, one accumulator per digit weight w = p + q
   (w = 0, 1, 2; the pairs of weight 3 and 4 are not formed):
   ``I_w = sum_{p+q=w} sum_k d_p(a_ik) d_q(b_kj)``. For KB <= 128:
   ``|I_0| <= KB * 127**2``, ``|I_1| <= 2 KB * 127 * 64``, ``|I_2| <= KB (2 * 127 * 64 + 64**2)``,
   all at most 2,605,056, below 2**22.
5. Fixed order rounding in float32: ``s = (I_2 * 2**-7 + I_1) * 2**-7 + I_0`` (Horner, smallest
   weight first; each I_w converts to float32 exactly), then
   ``r = (s * 2**(ea_i - 7)) * 2**(eb_j - 7)``. The K loop adds the block results in increasing K
   order to a float32 accumulator that starts at zero: ``acc = acc + r``.

Every multiplication above is by a power of two, so it is exact (outside the subnormal range) and
contracting it with the following addition into a fused multiply add changes nothing. The result
is therefore a function of the input values and KB only: no tile size, warp count, pipeline depth,
instruction shape or GPU generation enters.

Two engines realize step 4 with the same digits, so they give the same bits; only speed differs:

* ``int8`` (KB from 32 to 128): digits as int8, ``tl.dot`` with int32 accumulation. Integer
  arithmetic has no alignment window, so the sum is exact on every GPU by definition.
* ``fp16`` (KB = 16; Triton's int8 dot needs K >= 32): digits as float16 (integers up to 2048 are
  exact in float16), chained ``tl.dot`` with a float32 accumulator that starts at
  C = 1.5 * 2**23 (not zero), read back as the integer ``bits(acc) - bits(C)``. Exactness rests on
  the alignment window, proven below.

Proof for the fp16 engine (the constant is TC_FRAC_BITS_MIN). The tensor core model of
docs/TENSOR_CORE_MODELS.md (Khattak and Mikaitis; fitted bit for bit by Truestar on sm_89 and
sm_120, and matching the xchip probe on sm_86, sm_89 and sm_90) takes one group of exact,
unnormalized products plus the accumulator c, finds the largest exponent E among them, truncates
every term to the grid 2**(E - F), adds exactly and truncates the sum to float32. F is 24 on sm_80,
sm_86 and sm_89 and 25 on sm_90 and sm_120 (23 on sm_70, which this mode refuses). Here every
product d_p d_q is an integer of magnitude below 2**14 (exponent at most 13), the accumulator
starts at C = 1.5 * 2**23 and every partial sum of the chain is C plus an integer of magnitude at
most 2,605,056 < 2**22, so it lies in [2**23, 2**24): every term is an integer, the largest
exponent is E = 23 and the grid 2**(E - F) <= 2**(23 - 24) < 1. No integer bit is ever dropped;
the exact sum is an integer in [2**23, 2**24), representable in float32, so any rounding returns
it unchanged. This holds for any grouping of the products (any instruction shape or group size,
c added early or late), because every partial sum satisfies the same bound. (sm_70's F = 23 gives
the grid 2**0 and would still do.) Starting at C also makes the result immune to Triton folding a
later addition into the accumulator (it folds ``x + dot(a, b, 0)`` into ``dot(a, b, x)``, which
broke an earlier draft): here the accumulator is never zero and every value read from it is
converted to an integer first. ``check_bounds`` asserts the numbers; tests/test_tcexact.py runs
the digits through every generation's model.

``emulate_block`` and ``emulate_matmul`` are the same algorithm in NumPy (float32 arithmetic,
int64 dot products): the CPU twin of the GPU kernels, bit for bit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Alignment window (fraction bits below the largest product's exponent) of the narrowest tensor
# core generation this mode accepts: sm_80, sm_86 and sm_89 keep 2**(E - 24) (measured by the
# xchip probe on A10G and L4: a product 2**-24 below 1.0 is kept, 2**-25 is dropped).
TC_FRAC_BITS_MIN = 24
DIGITS = 3                  # digits per operand value
DIGIT_BITS = 7              # digit weight 2**-7
TOP_MAX, LOW_MAX = 127, 64  # |d0| <= 127, |d1|, |d2| <= 64
TOP = 7                     # |t| < 2**TOP / (1 + 2**-7) before the first digit
HEADROOM = 1.0078125        # 1 + 2**-7: the block maximum is scaled by it before its exponent is read
SUM_BITS = 22               # every |I_w| is below 2**SUM_BITS
PORTABLE_BK = 64            # K block of contraction and convolution kernels in this mode
MIN_KB, INT8_MIN_KB, MAX_KB = 16, 32, 128
EXP_LO, EXP_HI = -119, 129  # block exponent clamp: 2**(TOP - e) and 2**(e - TOP) stay normal
MIN_CAPABILITY = (8, 0)
CODES = ("TS1330", "TS1331", "TS1332", "TS1333")
_RN = 12582912.0            # 1.5 * 2**23: (t + _RN) - _RN is t rounded to an integer, ties to even
_RN_BITS = 0x4B400000       # float32 bits of _RN


class PortableTensorCoreError(ValueError):
    """TS1330 to TS1333: a portable tensor core request Truestar refuses; nothing is downgraded."""

    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def engine_for(kb: int) -> str:
    """Engine of a K block of kb columns: 'int8' for 32 to 256, 'fp16' for 16; else TS1332."""
    if INT8_MIN_KB <= kb <= MAX_KB:
        return "int8"
    if MIN_KB <= kb < INT8_MIN_KB:
        return "fp16"
    raise PortableTensorCoreError(
        "TS1332", f"a tensor core K block of {kb} columns has no exact form ({MIN_KB} to {MAX_KB} "
                  f"only): its integer sums could exceed 2**{SUM_BITS}")


def weight_bounds(kb: int):
    """Largest |I_0|, |I_1|, |I_2| of a K block of kb columns."""
    t, lo = TOP_MAX, LOW_MAX
    return kb * t * t, 2 * kb * t * lo, kb * (2 * t * lo + lo * lo)


def check_bounds() -> None:
    """The inequalities the exactness proof uses, as numbers."""
    assert max(weight_bounds(MAX_KB)) < 2 ** SUM_BITS    # magic int to float conversion, float32 exact
    assert max(weight_bounds(2 * MAX_KB)) >= 2 ** SUM_BITS   # MAX_KB is the largest power of two
    assert TOP_MAX <= 127 and TOP_MAX <= 2 ** 11          # int8 and float16 hold every digit
    assert 2 ** TOP / HEADROOM * (1 + 2 ** -23) < TOP_MAX + 0.5   # d0 = rn(t) cannot reach 128
    assert 2 ** DIGIT_BITS * 0.5 == LOW_MAX               # |t - rn(t)| <= 1/2 gives |d1|, |d2| <= 64
    c = 1.5 * 2 ** 23                                      # fp16 engine: partial sums in [2**23, 2**24)
    w = max(weight_bounds(MAX_KB))
    assert 2 ** 23 <= c - w and c + w < 2 ** 24
    assert 23 - TC_FRAC_BITS_MIN < 0                       # grid below 1: integer bits all kept
    assert TOP - EXP_LO <= 126 and EXP_HI - TOP <= 127 and EXP_LO - TOP >= -126
    assert DIGITS == 3                                    # the Triton source and the emulation unroll 3


check_bounds()


def check_device(device_index: Optional[int] = None) -> None:
    """Refuse a GPU this mode was not proven for: ROCm or below sm_80 (TS1331)."""
    import torch
    if getattr(torch.version, "hip", None):
        raise PortableTensorCoreError(
            "TS1331", "portable tensor cores are proven for NVIDIA sm_80 and newer only; AMD matrix "
                      "cores were not measured")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(device_index)
        if tuple(cap) < MIN_CAPABILITY:
            raise PortableTensorCoreError(
                "TS1331", f"portable tensor cores need sm_80 or newer (int8 tensor cores and a "
                          f"{TC_FRAC_BITS_MIN} bit alignment window); this GPU is sm_{cap[0]}{cap[1]}")


# --------------------------------------------------------------------------------- Triton source

TRITON_SOURCE = '''
@triton.jit
def _tsx_exp(m):
    # m >= 0: e with m < 2**e (biased exponent - 126), clamped to [-119, 129]
    f = (m.to(tl.int32, bitcast=True) >> 23) & 255
    return tl.minimum(tl.maximum(f - 126, -119), 129)


@triton.jit
def _tsx_p2(e):
    # 2**e as float32 for int32 e in [-126, 127]
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _tsx_tf32(x):
    # round float32 to TF32 (10 fraction bits), nearest even, on the bits
    u = x.to(tl.int32, bitcast=True)
    u = (u + 4095 + ((u >> 13) & 1)) & -8192
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _tsx_digits(t):
    # three digits of t (|t| < 127.01): int32, the first in [-127, 127], the others in [-64, 64]
    r = t + 12582912.0
    d0 = r.to(tl.int32, bitcast=True) - 1262485504
    t = (t - (r - 12582912.0)) * 128.0
    r = t + 12582912.0
    d1 = r.to(tl.int32, bitcast=True) - 1262485504
    t = (t - (r - 12582912.0)) * 128.0
    r = t + 12582912.0
    d2 = r.to(tl.int32, bitcast=True) - 1262485504
    return d0, d1, d2


@triton.jit
def _tsx_f32(i):
    # int32 with |i| < 2**22 to float32, exactly
    return (i + 1262485504).to(tl.float32, bitcast=True) - 12582912.0


@triton.jit
def _tsx_xdot_e(a, b, TF32: tl.constexpr, INT8: tl.constexpr):
    a = a.to(tl.float32)
    b = b.to(tl.float32)
    if TF32:
        a = _tsx_tf32(a)
        b = _tsx_tf32(b)
    ea = _tsx_exp(tl.max(tl.abs(a), axis=1) * 1.0078125)
    eb = _tsx_exp(tl.max(tl.abs(b), axis=0) * 1.0078125)
    a0, a1, a2 = _tsx_digits(a * _tsx_p2(7 - ea)[:, None])
    b0, b1, b2 = _tsx_digits(b * _tsx_p2(7 - eb)[None, :])
    if INT8:
        a0 = a0.to(tl.int8)
        a1 = a1.to(tl.int8)
        a2 = a2.to(tl.int8)
        b0 = b0.to(tl.int8)
        b1 = b1.to(tl.int8)
        b2 = b2.to(tl.int8)
        i0 = tl.dot(a0, b0, out_dtype=tl.int32)
        i1 = tl.dot(a0, b1, out_dtype=tl.int32)
        i1 = tl.dot(a1, b0, i1, out_dtype=tl.int32)
        i2 = tl.dot(a0, b2, out_dtype=tl.int32)
        i2 = tl.dot(a1, b1, i2, out_dtype=tl.int32)
        i2 = tl.dot(a2, b0, i2, out_dtype=tl.int32)
    else:
        a0 = a0.to(tl.float16)
        a1 = a1.to(tl.float16)
        a2 = a2.to(tl.float16)
        b0 = b0.to(tl.float16)
        b1 = b1.to(tl.float16)
        b2 = b2.to(tl.float16)
        c = tl.full((a.shape[0], b.shape[1]), 12582912.0, tl.float32)
        i0 = tl.dot(a0, b0, c).to(tl.int32, bitcast=True) - 1262485504
        f = tl.dot(a0, b1, c)
        i1 = tl.dot(a1, b0, f).to(tl.int32, bitcast=True) - 1262485504
        f = tl.dot(a0, b2, c)
        f = tl.dot(a1, b1, f)
        i2 = tl.dot(a2, b0, f).to(tl.int32, bitcast=True) - 1262485504
    s = _tsx_f32(i2) * 0.0078125 + _tsx_f32(i1)
    s = s * 0.0078125 + _tsx_f32(i0)
    return s * _tsx_p2(ea - 7)[:, None] * _tsx_p2(eb - 7)[None, :]


@triton.jit
def _tsx_xdot(a, b, KB: tl.constexpr, TF32: tl.constexpr):
    """Exact block a (BM, KB) @ b (KB, BN), rounded in a fixed order to float32 (tcexact.py)."""
    tl.static_assert(KB >= 16 and KB <= 128)
    return _tsx_xdot_e(a, b, TF32, KB >= 32)
'''


def triton_source() -> str:
    """The Triton helpers a portable tensor core module defines (``_tsx_xdot`` and its parts)."""
    return TRITON_SOURCE


# ------------------------------------------------------------------------------ NumPy emulation

_F32 = np.float32


def _bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.int32)


def round_tf32(x: np.ndarray) -> np.ndarray:
    """float32 values rounded to TF32 to nearest even, as the kernels do it."""
    u = _bits(x).astype(np.int64)
    u = (u + 4095 + ((u >> 13) & 1)) & -8192
    return ((u + 2 ** 31) % 2 ** 32 - 2 ** 31).astype(np.int32).view(np.float32)


def _exp(m: np.ndarray) -> np.ndarray:
    f = (_bits(m) >> 23) & 255
    return np.clip(f.astype(np.int64) - 126, EXP_LO, EXP_HI)


def _p2(e: np.ndarray) -> np.ndarray:
    return ((np.asarray(e, dtype=np.int64) + 127) << 23).astype(np.int32).view(np.float32)


def digits(t: np.ndarray):
    """The three digits (int64) of float32 t, |t| < 127.01, as ``_tsx_digits`` cuts them."""
    rn = _F32(_RN)
    out = []
    t = np.asarray(t, dtype=np.float32)
    for p in range(DIGITS):
        r = (t + rn).astype(_F32)
        out.append(_bits(r).astype(np.int64) - _RN_BITS)
        if p < DIGITS - 1:
            t = ((t - (r - rn).astype(_F32)).astype(_F32) * _F32(128.0)).astype(_F32)
    return out


def block_digits(a: np.ndarray, axis: int):
    """(digits, exponents) of a block along its K axis (1 for A rows, 0 for B columns)."""
    with np.errstate(over="ignore"):
        e = _exp((np.abs(a).max(axis=axis) * _F32(HEADROOM)).astype(_F32))
    sc = _p2(TOP - e)
    with np.errstate(over="ignore", under="ignore"):
        t = (a * (sc[:, None] if axis == 1 else sc[None, :])).astype(_F32)
    return digits(t), e


def emulate_block(a: np.ndarray, b: np.ndarray, tf32: bool = False) -> np.ndarray:
    """One exact block (a [M, KB], b [KB, N], float16/bfloat16/float32 values) as float32, exactly
    as ``_tsx_xdot`` computes it (either engine)."""
    engine_for(a.shape[1])
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if tf32:
        a, b = round_tf32(a), round_tf32(b)
    da, ea = block_digits(a, 1)
    db, eb = block_digits(b, 0)
    f0 = (da[0] @ db[0]).astype(_F32)
    f1 = (da[0] @ db[1] + da[1] @ db[0]).astype(_F32)
    f2 = (da[0] @ db[2] + da[1] @ db[1] + da[2] @ db[0]).astype(_F32)
    w = _F32(2.0 ** -DIGIT_BITS)
    s = ((f2 * w).astype(_F32) + f1).astype(_F32)
    s = ((s * w).astype(_F32) + f0).astype(_F32)
    with np.errstate(over="ignore", under="ignore"):
        return ((s * _p2(ea - TOP)[:, None]).astype(_F32) * _p2(eb - TOP)[None, :]).astype(_F32)


def emulate_matmul(a: np.ndarray, b: np.ndarray, kb: int = PORTABLE_BK, tf32: bool = False) -> np.ndarray:
    """a [M, K] @ b [K, N] as a portable tensor core contraction: K blocks of kb columns (the last
    one zero padded) added in increasing K order to a float32 accumulator starting at zero."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    M, K = a.shape
    N = b.shape[1]
    acc = np.zeros((M, N), dtype=np.float32)
    for k0 in range(0, K, kb):
        ab = np.zeros((M, kb), dtype=np.float32)
        bb = np.zeros((kb, N), dtype=np.float32)
        w = min(kb, K - k0)
        ab[:, :w] = a[:, k0:k0 + w]
        bb[:w] = b[k0:k0 + w]
        acc = (acc + emulate_block(ab, bb, tf32)).astype(np.float32)
    return acc
