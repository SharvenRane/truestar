"""Correctly rounded reference for round once math (docs/ROUND_ONCE.md), built on mpmath.

Each function is evaluated at 200 bits and rounded to the target format by integer arithmetic
(round to nearest, ties to even, with the format's subnormals and overflow), so the reference never
goes through a float64 double rounding. A value within 2^-180 relative of a rounding midpoint is
recomputed at 2000 bits.
"""
from __future__ import annotations

import math

import mpmath

FORMATS = {"float32": (24, -126, 127), "float16": (11, -14, 15), "bfloat16": (8, -126, 127),
           "float64": (53, -1022, 1023)}


def _mp_value(name: str, x: float):
    """name(x) as an mpf (or a Python float for IEEE special cases)."""
    if math.isnan(x):
        return math.nan
    if name == "exp":
        return mpmath.exp(x)
    if name == "log":
        if x < 0:
            return math.nan
        if x == 0:
            return -math.inf
        return mpmath.log(x)
    if name == "tanh":
        if x == 0:
            return x
        return mpmath.tanh(x)
    if name == "erf":
        if x == 0:
            return x
        return mpmath.erf(x)
    if name == "rsqrt":
        if x == 0:
            return math.copysign(math.inf, x)
        if x < 0:
            return math.nan
        if math.isinf(x):
            return 0.0
        return 1 / mpmath.sqrt(x)
    if name == "sigmoid":
        if x == -math.inf:
            return 0.0
        return 1 / (1 + mpmath.exp(-x))
    raise ValueError(name)


def _round_exact(sign: int, man: int, exp: int, fmt: str):
    """(-1)^sign man 2^exp rounded to fmt; returns (float, distance to the nearest midpoint as a
    fraction of the value, or None when the value is exact)."""
    p, emin, emax = FORMATS[fmt]
    bc = man.bit_length()
    top = exp + bc - 1
    q = max(top, emin) - (p - 1)
    if exp >= q:
        r, near = man << (exp - q), None
    elif q - exp > bc + 1:
        r, near = 0, 1.0                 # below a quarter of the last place: rounds to zero
    else:
        sh = q - exp
        r = man >> sh
        rem = man - (r << sh)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (r & 1)):
            r += 1
        # relative distance to the midpoint (approximate)
        near = abs(rem - half) / float(man)
    if r == 0:
        v = 0.0
    elif q + r.bit_length() - 1 > emax:
        v = math.inf
    else:
        v = math.ldexp(float(r), q)
    return (-v if sign else v), near


def reference(name: str, x: float, fmt: str) -> float:
    """name(x) correctly rounded to fmt, as a Python float (inf and nan included)."""
    with mpmath.workprec(200):
        y = _mp_value(name, x)
        if isinstance(y, float):
            return _round_float(y, fmt)
        if y == 0:
            return 0.0
        if mpmath.isinf(y):
            return float(y)
        sign, man, exp, _ = y._mpf_
        v, near = _round_exact(sign, int(man), int(exp), fmt)
    if near is not None and near < 2.0 ** -180:
        with mpmath.workprec(2000):
            y = _mp_value(name, x)
            sign, man, exp, _ = y._mpf_
            v, _ = _round_exact(sign, int(man), int(exp), fmt)
    return v


def _round_float(y: float, fmt: str) -> float:
    if y == 0 or math.isinf(y) or math.isnan(y):
        return y
    m, e = math.frexp(abs(y))
    man = int(m * 2 ** 53)
    v, _ = _round_exact(1 if y < 0 else 0, man, e - 53, fmt)
    return v


def true_value(name: str, x: float):
    """name(x) at 200 bits (mpf), or a Python float for special cases."""
    with mpmath.workprec(200):
        return _mp_value(name, x)


def ulp_error(name: str, x: float, got: float, fmt: str) -> float:
    """|got - name(x)| in units of the last place of fmt at the true value (0 when both are the
    same special value; inf when got is inf or nan and the true value is not)."""
    with mpmath.workprec(200):
        y = _mp_value(name, x)
        if isinstance(y, float) or mpmath.isinf(y):
            yf = float(y)
            if (math.isnan(yf) and math.isnan(got)) or yf == got:
                return 0.0
            if math.isnan(got) or math.isinf(got) or math.isnan(yf) or math.isinf(yf):
                return math.inf
            y = mpmath.mpf(yf)
        if math.isnan(got) or math.isinf(got):
            return math.inf
        p, emin, emax = FORMATS[fmt]
        if y == 0:
            e = emin
        else:
            e = max(int(mpmath.floor(mpmath.log(abs(y), 2))), emin)
            if abs(y) < mpmath.ldexp(1, e):     # guard the log rounding
                e = max(e - 1, emin)
            elif abs(y) >= mpmath.ldexp(1, e + 1):
                e += 1
        ulp = mpmath.ldexp(1, e - p + 1)
        return float(abs(mpmath.mpf(got) - y) / ulp)


def relative_error(name: str, x: float, got64: float) -> float:
    """|got64 - name(x)| / |name(x)| at 200 bits (0 for matching specials)."""
    with mpmath.workprec(200):
        y = _mp_value(name, x)
        if isinstance(y, float) or mpmath.isinf(y):
            yf = float(y)
            same = (math.isnan(yf) and math.isnan(got64)) or yf == got64
            return 0.0 if same else math.inf
        if y == 0:
            return 0.0 if got64 == 0 else math.inf
        if math.isnan(got64) or math.isinf(got64):
            return math.inf
        return float(abs(mpmath.mpf(got64) - y) / abs(y))
