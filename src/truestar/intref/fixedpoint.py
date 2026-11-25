"""Exact integer requantization primitives, vectorized over numpy int64.

Each function is a transcription of the C it is named after, checked against the source on
2026-11-24 (links in docs/INTREF.md). Inputs are int64 arrays holding int32 values unless a function
says otherwise; every intermediate fits int64, so the result is exact, not a float approximation.

The point of having all of them side by side is that they disagree on a small set of inputs, and
which set depends on the rule. `truestar.intref.match` uses that to say which rule a device used.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _i64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def quantize_multiplier(real) -> Tuple[np.ndarray, np.ndarray]:
    """TFLite `QuantizeMultiplier`: real = M0 * 2^(shift - 31), M0 in [2^30, 2^31).

    frexp, then round half away from zero (`TfLiteRound` is std::round), renormalize when the
    mantissa rounds up to 2^31, flush shifts below -31 to zero. Works per element (per channel).
    """
    r = np.atleast_1d(np.asarray(real, dtype=np.float64))
    m0 = np.zeros(r.shape, dtype=np.int64)
    sh = np.zeros(r.shape, dtype=np.int64)
    for i, v in enumerate(r.ravel()):
        if v == 0.0:
            continue
        if v < 0 or not math.isfinite(v):
            raise ValueError(f"multiplier must be finite and positive, got {v}")
        q, e = math.frexp(float(v))
        qf = int(math.floor(q * (1 << 31) + 0.5))  # q > 0, so half up is half away
        if qf == (1 << 31):
            qf //= 2
            e += 1
        if e < -31:
            qf, e = 0, 0
        m0.ravel()[i] = qf
        sh.ravel()[i] = e
    if np.ndim(real) == 0:
        return m0.reshape(()), sh.reshape(())
    return m0.reshape(np.shape(real)), sh.reshape(np.shape(real))


def _shr(x: np.ndarray, n) -> np.ndarray:
    """Arithmetic shift right (C `>>` on a signed value: floor division by 2^n)."""
    return np.right_shift(_i64(x), _i64(n))


def _trunc_div_pow2(x: np.ndarray, n: int) -> np.ndarray:
    """C integer division by 2^n: truncation toward zero."""
    x = _i64(x)
    return np.where(x >= 0, x >> n, -((-x) >> n))


def srdhm(a, b) -> np.ndarray:
    """gemmlowp `SaturatingRoundingDoublingHighMul` (int32): round to nearest at bit 31.

    nudge is +2^30 for a non negative product and 1 - 2^30 for a negative one, then the sum is
    divided by 2^31 with C truncation. The 1 in 1 - 2^30 makes a negative tie go toward zero, so
    every tie goes toward plus infinity (srdhm(-1, 2^30) is 0, srdhm(-3, 2^30) is -1): the same
    result as CMSIS-NN's no saturation version on every input with a non negative multiplier
    (tests/test_intref.py). a == b == INT32_MIN saturates to INT32_MAX.
    """
    a, b = _i64(a), _i64(b)
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    out = _trunc_div_pow2(ab + nudge, 31)
    return np.where((a == b) & (a == INT32_MIN), INT32_MAX, out)


def cmsis_dhm_no_sat(a, b) -> np.ndarray:
    """CMSIS-NN `arm_nn_doubling_high_mult_no_sat` (C path): (a*b + 2^30) >> 31.

    The offset is +2^30 whatever the sign and the shift is arithmetic (floor), so ties round
    toward plus infinity, like gemmlowp written differently. The int32 cast wraps.
    """
    r = (_i64(a) * _i64(b) + (1 << 30)) >> 31
    return ((r + (1 << 31)) % (1 << 32)) - (1 << 31)


def rdbpot(x, exponent) -> np.ndarray:
    """gemmlowp `RoundingDivideByPOT` (same as CMSIS-NN `arm_nn_divide_by_power_of_two`):
    divide by 2^exponent, round half away from zero."""
    x, e = _i64(x), _i64(exponent)
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def _shifts(shift) -> Tuple[np.ndarray, np.ndarray]:
    s = _i64(shift)
    return np.maximum(s, 0), np.maximum(-s, 0)


def mbqm_double(x, m0, shift) -> np.ndarray:
    """TFLite `MultiplyByQuantizedMultiplier` default build (double rounding, gemmlowp)."""
    left, right = _shifts(shift)
    return rdbpot(srdhm(_i64(x) * (np.int64(1) << left), m0), right)


def mbqm_double_cmsis(x, m0, shift) -> np.ndarray:
    """CMSIS-NN `arm_nn_requantize` default build (double rounding, no saturation high mul)."""
    left, right = _shifts(shift)
    return rdbpot(cmsis_dhm_no_sat(_i64(x) * (np.int64(1) << left), m0), right)


def mbqm_single(x, m0, shift) -> np.ndarray:
    """TFLite `MultiplyByQuantizedMultiplier` with TFLITE_SINGLE_ROUNDING (and CMSIS-NN with
    CMSIS_NN_USE_SINGLE_ROUNDING, which is the same function written as two shifts):
    (x * M0 + 2^(t-1)) >> t with t = 31 - shift. One rounding, ties toward plus infinity."""
    t = 31 - _i64(shift)
    return (_i64(x) * _i64(m0) + (np.int64(1) << (t - 1))) >> t


def mbqm_single_cmsis_form(x, m0, shift) -> np.ndarray:
    """CMSIS-NN single rounding as written: ((x * M0) >> (t - 1) + 1) >> 1. Kept separate so a
    test can show it equals `mbqm_single` on every input instead of assuming it."""
    t = 31 - _i64(shift)
    return (((_i64(x) * _i64(m0)) >> (t - 1)) + 1) >> 1


def mbqm_exact(x, m0, shift, tie: str) -> np.ndarray:
    """Round x * M0 / 2^(31 - shift) exactly with the named tie rule: 'even', 'away', 'up'.

    'up' equals `mbqm_single`; 'even' and 'away' are the rules a float implementation of the
    same multiplier would follow if it had infinite precision.
    """
    t = 31 - _i64(shift)
    p = _i64(x) * _i64(m0)
    fl = p >> t
    rem = p - (fl << t)
    half = np.int64(1) << (t - 1)
    above = rem > half
    tie_ = rem == half
    if tie == "up":
        bump = above | tie_
    elif tie == "away":
        bump = above | (tie_ & (p >= 0))
    elif tie == "even":
        bump = above | (tie_ & ((fl & 1) == 1))
    else:
        raise ValueError(f"unknown tie rule {tie!r}")
    return fl + bump


def reduce_multiplier_q15(m0) -> np.ndarray:
    """TFLite / CMSIS-NN `REDUCE_MULTIPLIER`: the Q31 multiplier rounded to Q15."""
    m0 = _i64(m0)
    return np.where(m0 < 0x7FFF0000, (m0 + (1 << 15)) >> 16, 0x7FFF)


def mbqm_q15(x, m0, shift) -> np.ndarray:
    """TFLite `MultiplyByQuantizedMultiplier(int64_t x, ...)` and CMSIS-NN `arm_nn_requantize_s64`
    (C path): 16 bit multiplier, single rounding half up. Used for int16 activations, where x is an
    int64 accumulator in [-2^47, 2^47)."""
    rm = reduce_multiplier_q15(m0)
    t = 15 - _i64(shift)
    return (_i64(x) * rm + (np.int64(1) << (t - 1))) >> t


def round_half_away(v: np.ndarray) -> np.ndarray:
    """Round a float array half away from zero, exactly (std::round)."""
    v = np.asarray(v, dtype=np.float64)
    r = np.floor(np.abs(v) + 0.5)
    # Exact for values with at most a 24 bit mantissa below 2^28 (float32 products): |v| + 0.5 is
    # then representable in float64, so the floor sees the true sum.
    return np.copysign(r, v)


def round_half_even(v: np.ndarray) -> np.ndarray:
    """Round a float array half to even (np.rint, nearbyint in the default mode)."""
    return np.rint(np.asarray(v, dtype=np.float64))
