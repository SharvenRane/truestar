"""Round once elementwise math: exp, log, tanh, erf, rsqrt and sigmoid with portable bits.

Each function is evaluated in float64 by an algorithm written for Truestar that uses only
operations IEEE 754 defines as correctly rounded (+ - * /, sqrt, comparisons, selection) plus exact
operations (floor, negation, absolute value, scaling by a power of two built from its bits), in a
fixed order and with no fused multiply add. The float64 result is then rounded once to the output
format (float32, float16 or bfloat16) by an arithmetic rounding that is itself exact IEEE
arithmetic. Any device that runs the same sequence of correctly rounded float64 operations
therefore produces the same bits, whatever its native exp or tanh does.

One algorithm description drives three evaluators, so they cannot drift apart:

* torch tensors (``exp(x)``, ``value64("exp", x)``), on the CPU or any torch GPU;
* Triton source (``triton_source``), used by ``K.compile(..., math="round_once")``; those kernels
  launch with ``enable_fp_fusion=False`` so a*b+c is never contracted into one rounding;
* numba source (``numba_source``) for the 'cpu' kernel backend.

Two pitfalls measured while building this (docs/ROUND_ONCE.md): torch's float64 sqrt on the CPU
(torch 2.11, Windows, AVX512 build) is not correctly rounded for about 1.4 percent of inputs, so
the CPU evaluator takes its square root from NumPy; and torch on CUDA divides by a Python scalar as
a multiply by its reciprocal, so every division here is tensor by tensor.

Six float32 inputs (five for log, one for sigmoid) whose float64 value lies closer to a float32
rounding midpoint than the algorithm's error carry their exact answer (EXCEPTIONS); the exhaustive
float32 sweep found them. Every NaN result is one canonical quiet NaN, set on the integer bits.

Inputs must be float16, bfloat16 or float32 values (the algorithms rely on at most 24 significant
bits, for example to square exactly in erf). Correct rounding is proven exhaustively for float16
and bfloat16 against mpmath, and for float32 by an exhaustive sweep that is conditional on a
measured error bound; see docs/ROUND_ONCE.md for what is proven and what is sampled.
"""

from __future__ import annotations

import decimal
import math
from fractions import Fraction
from typing import Dict, Iterable, Sequence

FUNCTIONS = ("exp", "log", "tanh", "erf", "rsqrt", "sigmoid")
MODES = ("default", "round_once")
# ops that are already correctly rounded natively; round once mode only asks for the IEEE form
NATIVE_EXACT = ("neg", "abs", "not", "sqrt")
DTYPES = ("float32", "float16", "bfloat16")

# precision p (with the hidden bit) and the exponent range of each output format
FORMATS = {"float32": (24, -126, 127), "float16": (11, -14, 15), "bfloat16": (8, -126, 127)}

# positive quiet NaN with no payload, as the integer view of each format
CANONICAL_NAN = {"float32": ("int32", 0x7FC00000), "float16": ("int16", 0x7E00),
                 "bfloat16": ("int16", 0x7FC0)}

INF = math.inf
NAN = math.nan


class RoundOnceError(ValueError):
    """TS1089 to TS1093: a round once request Truestar refuses; nothing falls back silently."""

    def __init__(self, code: str, msg: str):
        super().__init__(f"[{code}] {msg}")
        self.code = code


# ============================================================================ constants
# Derived here from exact rational arithmetic (fractions, decimal); no table is copied.

def _pi_fraction() -> Fraction:
    """pi = 16 atan(1/5) - 4 atan(1/239), each arctangent summed far past 2^-200."""
    def atan_inv(n: int, terms: int) -> Fraction:
        s, p = Fraction(0), Fraction(1, n)
        for k in range(terms):
            s += (p if k % 2 == 0 else -p) / (2 * k + 1)
            p /= n * n
        return s
    return 16 * atan_inv(5, 60) - 4 * atan_inv(239, 30)


def _constants() -> Dict[str, object]:
    ctx = decimal.Context(prec=90)
    ln2 = Fraction(ctx.ln(decimal.Decimal(2)))
    pi = _pi_fraction()
    pi_dec = ctx.divide(decimal.Decimal(pi.numerator), decimal.Decimal(pi.denominator))
    two_over_sqrtpi = Fraction(ctx.divide(decimal.Decimal(2), ctx.sqrt(pi_dec)))
    # ln 2 split: the high part has 32 significant bits, so k * LN2_HI is exact for |k| < 2^21
    hi = Fraction(round(ln2 * 2 ** 32), 2 ** 32)
    c = {
        "LN2_HI": float(hi), "LN2_LO": float(ln2 - hi), "INV_LN2": float(1 / ln2),
        "SQRT2": float(Fraction(ctx.sqrt(decimal.Decimal(2)))),
        "TWO_OVER_SQRTPI": float(two_over_sqrtpi),
    }
    # exp on |r| <= ln2/2: Taylor degree 13 (remainder below 2^-57 relative)
    c["EXP_C"] = [float(Fraction(1, math.factorial(n))) for n in range(14)]
    # log(m) = 2 atanh(s), |s| <= 0.1716: 2 sum s^(2n+1)/(2n+1), n = 0..12
    c["LOG_C"] = [float(Fraction(2, 2 * n + 1)) for n in range(13)]
    # expm1(u) = u * sum u^n/(n+1)!, 0 <= u < 1, n = 0..19 (remainder below 1/21!)
    c["EXPM1_C"] = [float(Fraction(1, math.factorial(n + 1))) for n in range(20)]
    # erf(a) = 2/sqrt(pi) a exp(-a^2) sum (2a^2)^n / (2n+1)!!  (all terms positive), a <= 4:
    # nested as 1 + (2a^2/3)(1 + (2a^2/5)(1 + ...)), 62 levels (tail below 2^-61 at a = 4)
    c["ERF_N"] = 62
    c["ERF_C"] = [0.0] + [float(Fraction(2, 2 * n + 1)) for n in range(1, 63)]
    return c


C = _constants()
LN2_HI, LN2_LO, INV_LN2 = C["LN2_HI"], C["LN2_LO"], C["INV_LN2"]
SQRT2, TWO_OVER_SQRTPI = C["SQRT2"], C["TWO_OVER_SQRTPI"]
EXP_C, LOG_C, EXPM1_C, ERF_C, ERF_N = C["EXP_C"], C["LOG_C"], C["EXPM1_C"], C["ERF_C"], C["ERF_N"]

EXP_CLAMP = 800.0      # exp(800) overflows float64 and exp(-800) underflows it: both exact limits
SIGMOID_C3 = float(Fraction(-1, 48))
SIGMOID_SMALL_LO = 2.0 ** -26  # below: x/4 is under a quarter ulp of 1/2 in float32, rounds to 1/2
SIGMOID_SMALL_HI = 2.0 ** -14  # above: the ordinary path (its one float32 hard case is in EXCEPTIONS)
TANH_CLAMP = 20.0      # tanh(20) is 1 - 8.5e-18: rounds to 1 in float64 and every output format
TANH_SMALL = 0.5       # below: expm1 series; above: (1 - e^-2a) / (1 + e^-2a)
ERF_ONE = 4.0          # erfc(4) = 1.5e-8 < half an ulp of 1 in float32 (2.98e-8): erf rounds to 1


# ============================================================================ the algorithms
# F supplies the primitives (where, eq, ne, div, sqrt, floor, abs, signbit, pow2, frexp1); values
# support + - * and comparisons with values and Python floats. Each line is one IEEE operation,
# so the order of evaluation is the order written.

def _horner(coefs: Sequence[float], z):
    """((c_n z + c_{n-1}) z + ...) + c_0 with coefs = [c_0, ..., c_n]."""
    p = coefs[-1] * z + coefs[-2]
    for c in reversed(coefs[:-2]):
        p = p * z + c
    return p


def exp64(F, x):
    nan = F.ne(x, x)
    xs = F.where(nan, 0.0, x)
    xs = F.where(xs > EXP_CLAMP, EXP_CLAMP, xs)
    xs = F.where(xs < -EXP_CLAMP, -EXP_CLAMP, xs)
    k = F.floor(xs * INV_LN2 + 0.5)
    r = (xs - k * LN2_HI) - k * LN2_LO
    p = _horner(EXP_C, r)
    k1 = F.floor(k * 0.5)
    k2 = k - k1
    y = (p * F.pow2(k1)) * F.pow2(k2)
    return F.where(nan, x, y)


def log64(F, x):
    ok = (x > 0.0) & (x < INF)
    xs = F.where(ok, x, 1.0)
    m, e = F.frexp1(xs)                  # xs = m 2^e, 1 <= m < 2
    big = m > SQRT2
    m = F.where(big, m * 0.5, m)
    e = F.where(big, e + 1.0, e)
    f = m - 1.0                          # exact (Sterbenz)
    s = F.div(f, m + 1.0)                # m + 1 is exact for inputs of at most 24 bits
    z = s * s
    q = _horner(LOG_C, z)
    lm = s * q
    y = e * LN2_HI + (e * LN2_LO + lm)
    y = F.where(F.eq(x, 0.0), -INF, y)
    y = F.where(x < 0.0, NAN, y)
    y = F.where(F.eq(x, INF), INF, y)
    return F.where(F.ne(x, x), x, y)


def tanh64(F, x):
    nan = F.ne(x, x)
    xs = F.where(nan, 0.0, x)
    a = F.abs(xs)
    a = F.where(a > TANH_CLAMP, TANH_CLAMP, a)
    u = a * 2.0
    em1 = u * _horner(EXPM1_C, u)
    small = F.div(em1, em1 + 2.0)
    t = exp64(F, -u)
    large = F.div(1.0 - t, 1.0 + t)
    y = F.where(a < TANH_SMALL, small, large)
    y = F.where(F.signbit(xs), -y, y)    # tanh(-0) = -0
    return F.where(nan, x, y)


def sigmoid64(F, x):
    e = exp64(F, -x)
    y = F.div(1.0, 1.0 + e)
    # Near 0, sigmoid(x) = 1/2 + x/4 - x^3/48 + ...: for a float32 x the leading part A = 1/2 + x/4
    # is exact in float64 and is often exactly a float32 rounding midpoint, while the true value
    # sits x^3/48 below it, far under float64's resolution. So: A + c with c = -x^3/48 (its next
    # term is x^2/10 smaller), and if that sum rounds back to A, step one float64 ulp towards c.
    # The result then lies strictly on the true value's side of every midpoint near A.
    ax = F.abs(x)
    a = x * 0.25 + 0.5
    c = ((x * x) * x) * SIGMOID_C3
    t = a + c
    _, et = F.frexp1(F.where(t > 0.0, t, 1.0))
    ulp = F.pow2(et - 52.0)
    t = F.where(F.eq(t, a) & (x > 0.0), t - ulp, t)
    t = F.where(F.eq(t, a) & (x < 0.0), t + ulp, t)
    small = (ax >= SIGMOID_SMALL_LO) & (ax < SIGMOID_SMALL_HI)
    return F.where(small, t, y)


def erf64(F, x):
    nan = F.ne(x, x)
    xs = F.where(nan, 0.0, x)
    ax = F.abs(xs)
    a = F.where(ax > ERF_ONE, ERF_ONE, ax)
    y2 = a * a                           # exact for inputs of at most 26 bits
    acc = (y2 * ERF_C[ERF_N]) + 1.0
    for n in range(ERF_N - 1, 0, -1):
        acc = (y2 * acc) * ERF_C[n] + 1.0
    v = ((a * TWO_OVER_SQRTPI) * exp64(F, -y2)) * acc
    v = F.where(ax >= ERF_ONE, 1.0, v)
    v = F.where(F.signbit(xs), -v, v)    # erf(-0) = -0
    return F.where(nan, x, v)


def rsqrt64(F, x):
    return F.div(1.0, F.sqrt(x))


# float32 inputs whose float64 value lies closer to a float32 rounding midpoint than the algorithm's
# error, found by the exhaustive float32 sweep (scripts/roundonce_measure.py sweep) and settled by
# the mpmath reference: input -> the correctly rounded float32 result (exact in float64). None of
# them is a float16 or bfloat16 value.
EXCEPTIONS = {
    "log": [(float.fromhex("0x1.827a74p-7"), float.fromhex("-0x1.1c2b1ep+2")),
            (float.fromhex("0x1.2f1fd6p+3"), float.fromhex("0x1.1fcbcep+1")),
            (float.fromhex("0x1.bacb4ap+25"), float.fromhex("0x1.1e0696p+4")),
            (float.fromhex("0x1.b121a6p+76"), float.fromhex("0x1.a9a3f2p+5")),
            (float.fromhex("0x1.6351d8p+95"), float.fromhex("0x1.08b512p+6"))],
    "sigmoid": [(float.fromhex("-0x1.250c02p-10"), float.fromhex("0x1.ffb6bep-2"))],
}


def evaluate(F, name: str, x, patched: bool = True):
    """name(x) in float64, with the EXCEPTIONS inputs replaced by their exact float32 answer and
    every NaN result made the one canonical quiet NaN (so NaN bits do not depend on how a device
    propagates NaN payloads and signs)."""
    y = ALGORITHMS[name](F, x)
    for xin, yout in (EXCEPTIONS.get(name, ()) if patched else ()):
        y = F.where(F.eq(x, xin), yout, y)
    return F.where(F.ne(y, y), NAN, y)


ALGORITHMS = {"exp": exp64, "log": log64, "tanh": tanh64, "erf": erf64, "rsqrt": rsqrt64,
              "sigmoid": sigmoid64}


def round64(F, y, fmt: str):
    """y rounded to nearest even in format fmt, returned as a float64 holding that exact value.

    Adding C = 1.5 * 2^52 * ulp moves y's rounding point to the last bit of the sum, where
    float64's own round to nearest even does the rounding (C / ulp is even, so ties go to even
    multiples of ulp); subtracting C again is exact. ulp follows y's exponent, clamped at the
    format's subnormal exponent, and a result at 2^(emax+1) or beyond is an infinity."""
    p, emin, emax = FORMATS[fmt]
    a = F.abs(y)
    fin = (a < INF) & (a > 0.0)
    am = F.where(fin, a, 1.0)
    _, e = F.frexp1(am)
    e = F.where(e < float(emin), float(emin), e)
    e = F.where(e > float(emax), float(emax + 1), e)
    big = F.pow2(e + float(53 - p)) * 1.5
    ys = F.where(fin, y, 0.0)
    r = (ys + big) - big
    r = F.where(F.eq(r, 0.0) & F.signbit(ys), -r, r)   # a negative value rounded to zero is -0
    lim = 2.0 ** (emax + 1)
    r = F.where(r >= lim, INF, r)
    r = F.where(r <= -lim, -INF, r)
    return F.where(fin, r, y)


# ============================================================================ torch evaluator

class _TorchOps:
    def __init__(self, torch, cpu: bool):
        self.t = torch
        self.cpu = cpu

    def where(self, c, a, b):
        if not isinstance(a, self.t.Tensor) and not isinstance(b, self.t.Tensor):
            raise TypeError("internal: where of two constants")
        return self.t.where(c, a, b)

    def eq(self, a, b):
        return self.t.eq(a, b)

    def ne(self, a, b):
        return self.t.ne(a, b)

    def div(self, a, b):
        # tensor by tensor only: torch on CUDA turns a division by a Python scalar into a
        # multiply by the reciprocal, which is a different rounding
        if not isinstance(a, self.t.Tensor):
            a = self.t.full_like(b, a)
        if not isinstance(b, self.t.Tensor):
            b = self.t.full_like(a, b)
        return self.t.div(a, b)

    def sqrt(self, a):
        if self.cpu:
            # torch's CPU float64 sqrt is not correctly rounded on every build; NumPy's is
            import numpy as np
            with np.errstate(invalid="ignore", divide="ignore"):
                return self.t.from_numpy(np.sqrt(a.contiguous().numpy()))
        return self.t.sqrt(a)

    def floor(self, a):
        return self.t.floor(a)

    def abs(self, a):
        return self.t.abs(a)

    def signbit(self, a):
        return a.view(self.t.int64) < 0

    def pow2(self, k):
        t = self.t
        return ((k.to(t.int64) + 1023) << 52).view(t.float64)

    def frexp1(self, x):
        t = self.t
        b = x.view(t.int64)
        e = ((b >> 52) & 0x7FF) - 1023
        m = ((b & 0xFFFFFFFFFFFFF) | 0x3FF0000000000000).view(t.float64)
        return m, e.to(t.float64)


def _torch():
    import torch
    return torch


def _dtype_name(dt) -> str:
    return str(dt).replace("torch.", "")


def _check_name(name: str):
    if name not in ALGORITHMS:
        raise RoundOnceError("TS1092", f"round once math has no form of {name!r}; it covers "
                                       f"{', '.join(FUNCTIONS)}")


def value64(name: str, x, patched: bool = True):
    """The float64 value round once math computes for name(x), before the final rounding.

    x is a torch tensor of float16, bfloat16 or float32 values (a float64 tensor is accepted
    only if every value is a float32 value, which is not checked). patched=False skips the
    EXCEPTIONS table: the plain algorithm, accurate to a few float64 ulps everywhere, which the
    measurements use as a near exact value."""
    torch = _torch()
    _check_name(name)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RoundOnceError("TS1091", f"round once {name}: input dtype {_dtype_name(x.dtype)} is not "
                                       f"supported; use float16, bfloat16 or float32")
    x64 = x.to(torch.float64).contiguous()
    return evaluate(_TorchOps(torch, x64.device.type == "cpu"), name, x64, patched)


def round_to(y64, dtype):
    """Round a float64 tensor once, to nearest even, to float32, float16 or bfloat16."""
    torch = _torch()
    fmt = _dtype_name(dtype)
    if fmt not in FORMATS:
        raise RoundOnceError("TS1091", f"round once output dtype {fmt} is not supported; use "
                                       f"float32, float16 or bfloat16")
    r = round64(_TorchOps(torch, y64.device.type == "cpu"), y64, fmt)
    out = r.to(getattr(torch, fmt))
    # the cast of a NaN keeps or drops payload and sign bits differently on the CPU and on CUDA
    # (measured: every float16 NaN differs); every NaN becomes the format's canonical quiet NaN
    wide, bits = CANONICAL_NAN[fmt]
    w = getattr(torch, wide)
    canon = torch.full_like(out.view(w), bits)
    return torch.where(torch.isnan(out), canon, out.view(w)).view(getattr(torch, fmt))


def apply(name: str, x, dtype=None):
    """name(x) rounded once to dtype (default: x's dtype)."""
    out = x.dtype if dtype is None else dtype
    if _dtype_name(out) not in FORMATS:
        raise RoundOnceError("TS1091", f"round once {name}: output dtype {_dtype_name(out)} is not "
                                       f"supported; use float32, float16 or bfloat16")
    return round_to(value64(name, x), out)


def exp(x, dtype=None):
    return apply("exp", x, dtype)


def log(x, dtype=None):
    return apply("log", x, dtype)


def tanh(x, dtype=None):
    return apply("tanh", x, dtype)


def erf(x, dtype=None):
    return apply("erf", x, dtype)


def rsqrt(x, dtype=None):
    return apply("rsqrt", x, dtype)


def sigmoid(x, dtype=None):
    return apply("sigmoid", x, dtype)


# ============================================================================ source evaluators

class _Val:
    """A traced value: each operation appends one line of source."""

    __slots__ = ("tr", "v")

    def __init__(self, tr: "_Trace", v: str):
        self.tr, self.v = tr, v

    def _bin(self, o, op, rev=False):
        a, b = self.v, self.tr.lit(o)
        if rev:
            a, b = b, a
        return self.tr.binop(op, a, b)

    def __add__(self, o): return self._bin(o, "+")
    def __radd__(self, o): return self._bin(o, "+", True)
    def __sub__(self, o): return self._bin(o, "-")
    def __rsub__(self, o): return self._bin(o, "-", True)
    def __mul__(self, o): return self._bin(o, "*")
    def __rmul__(self, o): return self._bin(o, "*", True)
    def __lt__(self, o): return self._bin(o, "<")
    def __le__(self, o): return self._bin(o, "<=")
    def __gt__(self, o): return self._bin(o, ">")
    def __ge__(self, o): return self._bin(o, ">=")
    def __and__(self, o): return self._bin(o, "&")
    def __or__(self, o): return self._bin(o, "|")
    def __neg__(self): return self.tr.neg(self.v)

    def __truediv__(self, o):
        raise TypeError("internal: divide with F.div")

    def __bool__(self):
        raise TypeError("internal: a traced value has no truth value")


class _Trace:
    def __init__(self, dialect: str):
        self.dialect = dialect
        self.lines = []
        self.n = 0

    def emit(self, code: str) -> _Val:
        name = f"v{self.n}"
        self.n += 1
        if self.dialect == "metal":
            self.lines.append(f"    const auto {name} = {code};")
        else:
            self.lines.append(f"    {name} = {code}")
        return _Val(self, name)

    # float64 operations of the Metal dialect: binary64 on ulong bits (softf64_metal.py)
    _METAL_BIN = {"+": "ts_dadd({a}, {b})", "-": "ts_dsub({a}, {b})", "*": "ts_dmul({a}, {b})",
                  "<": "ts_dlt({a}, {b})", "<=": "ts_dle({a}, {b})", ">": "ts_dlt({b}, {a})",
                  ">=": "ts_dle({b}, {a})", "&": "(({a}) && ({b}))", "|": "(({a}) || ({b}))"}

    def binop(self, op, a, b):
        if self.dialect == "metal":
            return self.emit(self._METAL_BIN[op].format(a=a, b=b))
        return self.emit(f"({a}) {op} ({b})")

    def neg(self, a):
        if self.dialect == "metal":
            return self.emit(f"ts_dneg({a})")
        return self.emit(f"-({a})")

    def lit(self, o) -> str:
        if isinstance(o, _Val):
            return o.v
        v = float(o)
        if self.dialect == "metal":
            import struct
            return f"0x{struct.unpack('<Q', struct.pack('<d', v))[0]:016X}ul" if not math.isnan(v) else "TS_DQNAN"
        if math.isnan(v):
            return "math.nan" if self.dialect == "numba" else "float('nan')"
        if math.isinf(v):
            if self.dialect == "numba":
                return "math.inf" if v > 0 else "(-math.inf)"
            return "float('inf')" if v > 0 else "float('-inf')"
        return repr(v)

    # the primitives
    def where(self, c, a, b):
        c, a, b = self.lit(c), self.lit(a), self.lit(b)
        if self.dialect == "metal":
            return self.emit(f"(({c}) ? ({a}) : ({b}))")
        if self.dialect == "numba":
            return self.emit(f"{a} if {c} else {b}")
        return self.emit(f"tl.where({c}, {a}, {b})")

    def eq(self, a, b):
        if self.dialect == "metal":
            return self.emit(f"ts_deq({self.lit(a)}, {self.lit(b)})")
        return self.emit(f"({self.lit(a)}) == ({self.lit(b)})")

    def ne(self, a, b):
        if self.dialect == "metal":
            return self.emit(f"ts_dne({self.lit(a)}, {self.lit(b)})")
        return self.emit(f"({self.lit(a)}) != ({self.lit(b)})")

    def div(self, a, b):
        if self.dialect == "metal":
            return self.emit(f"ts_ddiv({self.lit(a)}, {self.lit(b)})")
        return self.emit(f"({self.lit(a)}) / ({self.lit(b)})")

    def sqrt(self, a):
        if self.dialect == "metal":
            return self.emit(f"ts_dsqrt({self.lit(a)})")
        if self.dialect == "numba":
            return self.emit(f"math.sqrt({self.lit(a)})")
        return self.emit(f"libdevice.sqrt_rn({self.lit(a)})")

    def floor(self, a):
        if self.dialect == "metal":
            return self.emit(f"ts_dfloor({self.lit(a)})")
        if self.dialect == "numba":
            return self.emit(f"np.float64(math.floor({self.lit(a)}))")
        return self.emit(f"libdevice.floor({self.lit(a)})")

    def abs(self, a):
        if self.dialect == "metal":
            return self.emit(f"ts_dabs({self.lit(a)})")
        if self.dialect == "numba":
            return self.emit(f"abs({self.lit(a)})")
        return self.emit(f"tl.abs({self.lit(a)})")

    def signbit(self, a):
        if self.dialect == "metal":
            return self.emit(f"ts_dsignbit({self.lit(a)})")
        if self.dialect == "numba":
            return self.emit(f"math.copysign(1.0, {self.lit(a)}) < 0.0")
        return self.emit(f"({self.lit(a)}).to(tl.int64, bitcast=True) < 0")

    def pow2(self, k):
        if self.dialect == "metal":
            return self.emit(f"ts_dpow2({self.lit(k)})")
        if self.dialect == "numba":
            return self.emit(f"math.ldexp(1.0, int({self.lit(k)}))")
        return self.emit(f"((({self.lit(k)}).to(tl.int64) + 1023) << 52).to(tl.float64, bitcast=True)")

    def frexp1(self, x):
        if self.dialect == "metal":
            xv = self.lit(x)
            return self.emit(f"ts_dfrexp_m({xv})"), self.emit(f"ts_dfrexp_e({xv})")
        if self.dialect == "numba":
            fr = self.emit(f"math.frexp({self.lit(x)})")
            m = self.emit(f"{fr.v}[0] * 2.0")
            e = self.emit(f"np.float64({fr.v}[1] - 1)")
            return m, e
        b = self.emit(f"({self.lit(x)}).to(tl.int64, bitcast=True)")
        e = self.emit(f"(((({b.v}) >> 52) & 0x7FF) - 1023).to(tl.float64)")
        m = self.emit(f"((({b.v}) & 0xFFFFFFFFFFFFF) | 0x3FF0000000000000).to(tl.float64, bitcast=True)")
        return m, e


def _helper(name: str, dialect: str) -> str:
    tr = _Trace(dialect)
    if dialect == "metal":
        x = tr.emit("ts_dfromf32(x)")
        if name == "sqrt":       # float64 sqrt rounded to float32 is the correctly rounded float32 sqrt
            y = tr.sqrt(x)
        else:
            y = evaluate(tr, name, x)
        return (f"static inline float _ro_{name}(float x) {{\n" + "\n".join(tr.lines)
                + f"\n    return ts_dtof32({y.v});\n}}\n")
    if dialect == "numba":
        # the argument is the float32 value the GPU backend would see
        x = tr.emit("np.float64(np.float32(x))")
    else:
        x = tr.emit("x.to(tl.float64)")
    y = evaluate(tr, name, x)
    if dialect == "numba":
        ret = f"np.float32({y.v})"
    else:
        # LLVM may fold a float select between two NaNs, so the canonical NaN is set on the bits
        f = tr.emit(f"{y.v}.to(tl.float32)")
        b = tr.emit(f"tl.where({f.v} != {f.v}, 0x7FC00000, {f.v}.to(tl.int32, bitcast=True))")
        ret = f"{b.v}.to(tl.float32, bitcast=True)"
    deco = "@numba.njit(inline=\"always\", fastmath=False)" if dialect == "numba" else "@triton.jit"
    return f"{deco}\ndef _ro_{name}(x):\n" + "\n".join(tr.lines) + f"\n    return {ret}\n"


def triton_source(names: Iterable[str]) -> str:
    """@triton.jit helpers _ro_<name>(x): float32 in, float32 out, evaluated round once.

    The module that holds them must be launched with enable_fp_fusion=False (the generated
    kernel module sets TS_FP_FUSION = False and triton_backend.prepare passes it)."""
    names = sorted(set(names))
    for n in names:
        _check_name(n)
    return "\n\n".join(_helper(n, "triton") for n in names)


def metal_source(names: Iterable[str]) -> str:
    """Metal helpers _ro_<name>(float x) -> float for the 'metal' kernel backend: the same IEEE
    float64 sequence, each operation done on the bits by softf64_metal (Apple GPUs have no float64).
    "sqrt" is added for the natively exact sqrt, whose Metal precise form is not documented as
    correctly rounded."""
    from .softf64_metal import SOURCE
    names = sorted(set(names))
    for n in names:
        if n != "sqrt":
            _check_name(n)
    return SOURCE + "\n" + "\n".join(_helper(n, "metal") for n in names)


def numba_source(names: Iterable[str]) -> str:
    """numba helpers _ro_<name>(x) for the 'cpu' kernel backend (needs math, numba, np)."""
    names = sorted(set(names))
    for n in names:
        _check_name(n)
    return "\n\n".join(_helper(n, "numba") for n in names)


# ============================================================================ kernel checks

def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise RoundOnceError("TS1090", f"unknown math mode {mode!r}; use one of {', '.join(MODES)}")
    return mode


def check_kernel(k, backend: str) -> None:
    """Refuse a kernel that round once mode cannot compile for backend (TS1089, TS1091, TS1092)."""
    from . import ir
    if backend not in ("triton", "numba", "metal"):
        raise RoundOnceError("TS1089", f"kernel {k.name}: round once math is generated for the "
                                       f"'cuda'/'rocm' (Triton), 'cpu' (numba) and 'metal' backends only, "
                                       f"not {backend!r}")
    for b in k.inputs:
        if b.dtype not in DTYPES:
            raise RoundOnceError("TS1091", f"kernel {k.name}: input {b.name} is {b.dtype}; round once "
                                           f"math takes float32, float16 or bfloat16 values")
    for o in k.outputs:
        if o.dtype is not None and o.dtype not in DTYPES:
            raise RoundOnceError("TS1091", f"kernel {k.name}: output {o.name} is {o.dtype}; round "
                                           f"once math writes float32, float16 or bfloat16")
        for n in ir.walk(o.expr):
            if isinstance(n, ir.Unary) and n.op not in FUNCTIONS and n.op not in NATIVE_EXACT:
                raise RoundOnceError("TS1092", f"kernel {k.name}: {n.op} has no round once form "
                                               f"(covered: {', '.join(FUNCTIONS)}); compile without "
                                               f"math='round_once' or remove it")


def used(k) -> list:
    """Round once functions kernel k uses."""
    from . import ir
    out = set()
    for o in k.outputs:
        for n in ir.walk(o.expr):
            if isinstance(n, ir.Unary) and n.op in FUNCTIONS:
                out.add(n.op)
    return sorted(out)
