"""Keyed stochastic rounding: the same rounded bits on every chip, checkable exactly.

Stochastic rounding (SR) rounds a value x that lies between two neighbours lo < |x| < hi of a
low precision format up to hi with probability (|x| - lo) / (hi - lo), so the rounded value is
unbiased in expectation. Hardware SR (the Blackwell FP4 converts, the CDNA4 `sr` converts) draws
its random bits from a per thread source: a step that uses it is not reproducible and two vendors
never agree.

Here the random bits are Truestar's keyed counter based bits (`truestar.rng`, docs/RANDOM.md):
element i of a tensor rounded at (key, site, step) uses

    r = top 32 bits of mix(K ^ i),       K = stream_key(key, site, (step,))

and rounds away from zero exactly when r < f * 2^32, f = (|x| - lo) / (hi - lo) in [0, 1).
Everything is integer arithmetic on the input's float32 bits (the fraction f is a ratio of
integers), so the torch path on the CPU, the torch path on CUDA, the Triton kernel and the
one element at a time definition (`round_python`, exact rationals) return identical codes; the
tests hold them bitwise equal. The reference knows every draw, so an SR kernel can be verified
exactly, not only statistically.

Bias: P(up) = ceil(f 2^32) / 2^32, so E[round(x)] - x lies in [0, 2^-32 (hi - lo)): bounded by
2^-32 of one quantum, measured in tests/test_lowp.py. Saturation (|x| at or above the largest
finite value) clamps to it, as round to nearest does with satfinite; that clamp is biased and is
counted separately.

Inputs: float32, float16 or bfloat16 (widened exactly to float32), optionally divided by a power
of two 2^e per element (an MX or NVFP4 block scale exponent) inside the integer arithmetic, so
no division rounds. Targets: the FP8 and FP4 element formats of `truestar.quant.formats`
and bfloat16 and float16 (`TARGETS`). Negative values that round to zero give code 0 (no negative
zero), as the round to nearest encoders here do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Optional, Union

import torch

from .. import rng
from .formats import E2M1, E4M3, E5M2, Codec, FormatError

BACKENDS = ("python", "torch", "triton")


@dataclass(frozen=True)
class Target:
    """A sign magnitude float format as SR needs it: mantissa bits m, the exponent of its
    smallest normal binade emin, its largest finite code and its width."""

    name: str
    bits: int
    m: int
    emin: int
    max_code: int
    codec: Optional[Codec] = None

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """float64 values of codes."""
        c = codes.long()
        if self.codec is not None:
            return self.codec.decode(c)
        if self.name == "bf16":
            return (c << 16).to(torch.int32).view(torch.float32).to(torch.float64)
        return c.to(torch.int16).view(torch.float16).to(torch.float64)


def _from_codec(c: Codec) -> Target:
    m = {"e4m3": 3, "e5m2": 2, "e2m1": 1}[c.name]
    emin = math.frexp(c.values[1 << m])[1] - 1          # code 1 << m is the smallest normal
    half = 1 << (c.bits - 1)
    finite = [i for i in range(half) if math.isfinite(c.values[i])]
    return Target(c.name, c.bits, m, emin, max(finite), c)


TARGETS: Dict[str, Target] = {c.name: _from_codec(c) for c in (E4M3, E5M2, E2M1)}
TARGETS["bf16"] = Target("bf16", 16, 7, -126, 0x7F7F)
TARGETS["fp16"] = Target("fp16", 16, 10, -14, 0x7BFF)


def target(t: Union[str, Codec, Target]) -> Target:
    if isinstance(t, Target):
        return t
    if isinstance(t, Codec):
        if t.name not in TARGETS:
            raise FormatError(f"no stochastic rounding for {t.name}")
        return TARGETS[t.name]
    key = {"bfloat16": "bf16", "float16": "fp16", "half": "fp16"}.get(str(t).lower(), str(t).lower())
    if key not in TARGETS:
        raise FormatError(f"no stochastic rounding for {t!r}; targets: {', '.join(TARGETS)}")
    return TARGETS[key]


def stream(key: int, site, step: int) -> int:
    """The 64 bit stream key of one rounding call."""
    return rng.stream_key(rng.as_u64(int(key)), site, (int(step),))


def _as_f32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return x.to(torch.float32)
    if x.dtype == torch.float64:
        x32 = x.to(torch.float32)
        if not bool((x32.to(torch.float64) == x).all()):
            raise FormatError("keyed stochastic rounding takes float32 (or narrower) values; this "
                              "float64 tensor is not exactly representable in float32")
        return x32
    raise FormatError(f"keyed stochastic rounding takes a float tensor, got {x.dtype}")


# ----------------------------------------------------------------------------- the definition


def _code_of(t: Target, k: int, qe: int) -> int:
    """Code of the magnitude k * 2^qe (k an integer, qe the quantum exponent of its binade)."""
    return ((qe + t.m - t.emin) << t.m) + k


def round_python(x: torch.Tensor, t, key: int, site="sround", step: int = 0,
                 scale_exp: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The definition, one element at a time, in exact rationals: codes int64 of x's shape."""
    tg = target(t)
    xs = _as_f32(x).reshape(-1).cpu().double().tolist()
    es = [0] * len(xs) if scale_exp is None else scale_exp.reshape(-1).cpu().long().tolist()
    K = stream(key, site, step)
    out = []
    for i, (v, e) in enumerate(zip(xs, es)):
        if math.isnan(v) or math.isinf(v):
            raise FormatError("keyed stochastic rounding: NaN or Inf in the input")
        a = Fraction(abs(v)) / Fraction(2) ** e
        if a == 0:
            out.append(0)
            continue
        E = max(_flog2(a), tg.emin)
        qe = E - tg.m
        quantum = Fraction(2) ** qe
        n = math.floor(a / quantum)
        f = a / quantum - n
        r = rng.mix(K ^ i) >> 32
        k = n + (1 if r < f * (1 << 32) else 0)
        code = min(_code_of(tg, k, qe), tg.max_code)
        if v < 0 and code:
            code |= 1 << (tg.bits - 1)
        out.append(code)
    return torch.tensor(out, dtype=torch.int64).reshape(x.shape)


def _flog2(a: Fraction) -> int:
    """floor(log2 a) for a positive rational, exactly."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if Fraction(2) ** (e + 1) <= a:
        e += 1
    return e


# ----------------------------------------------------------------------------- torch (any device)


def round_torch(x: torch.Tensor, t, key: int, site="sround", step: int = 0,
                scale_exp: Optional[torch.Tensor] = None, start: int = 0) -> torch.Tensor:
    """Vectorized integer implementation on x's device. Bitwise equal to `round_python`."""
    tg = target(t)
    x32 = _as_f32(x)
    if not bool(torch.isfinite(x32).all()):
        raise FormatError("keyed stochastic rounding: NaN or Inf in the input")
    dev = x32.device
    b = x32.reshape(-1).view(torch.int32).long() & 0xFFFFFFFF
    sign = b >> 31
    ex = (b >> 23) & 0xFF
    man = b & 0x7FFFFF
    M = torch.where(ex > 0, man | (1 << 23), man)              # |x| = M * 2^(eb - 23)
    eb = torch.where(ex > 0, ex - 127, torch.full_like(ex, -126))
    e = torch.zeros_like(eb) if scale_exp is None else scale_exp.reshape(-1).to(dev).long()
    # floor(log2 |x|) from the bits (subnormal inputs: from the leading one of M)
    nz = M > 0
    # floor(log2 M): M < 2^24 is exact in float32, so its exponent field is the answer
    lead = ((M.clamp(min=1).to(torch.float32).view(torch.int32).long() >> 23) & 0xFF) - 127
    fl = eb - 23 + lead - e                                     # floor(log2 (|x| / 2^e))
    E = torch.maximum(fl, torch.full_like(fl, tg.emin))
    qe = E - tg.m                                               # quantum of the scaled value
    sh = (qe + e) - (eb - 23)                                   # quantum in units of M's last bit
    n = torch.where(sh <= 0, M << (-sh).clamp(min=0, max=40), M >> sh.clamp(min=0, max=62))
    R = torch.where(sh > 0, M & ((torch.ones_like(M) << sh.clamp(min=0, max=62)) - 1), torch.zeros_like(M))
    # c = ceil(R * 2^(32 - sh)) exactly
    up_sh = (32 - sh).clamp(min=0, max=32)
    dn = (sh - 32).clamp(min=0, max=62)
    c_big = R << up_sh
    c_small = (R + (torch.ones_like(R) << dn) - 1) >> dn
    c = torch.where(sh <= 32, c_big, torch.where(dn >= 40, (R > 0).long(), c_small))
    Kk = stream(key, site, step)
    idx = torch.arange(start, start + b.numel(), device=dev, dtype=torch.int64)
    h = rng._mix_t(idx ^ rng.as_i64(Kk))
    r = rng._shr(h, 32)
    k = n + (r < c).long()
    code = ((qe + tg.m - tg.emin) << tg.m) + k
    big = sh <= -40                                             # far above the format: saturate
    code = torch.where(big, torch.full_like(code, tg.max_code), code).clamp(max=tg.max_code)
    code = torch.where(nz, code, torch.zeros_like(code))
    code = torch.where((sign == 1) & (code != 0), code | (1 << (tg.bits - 1)), code)
    return code.reshape(x.shape)


# ----------------------------------------------------------------------------- triton (CUDA)

_TRITON: Dict[str, object] = {}


def _triton_kernel():
    if "k" in _TRITON:
        return _TRITON["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def _sr_mixu(z):
        z = z + ((tl.full([], 0x9E3779B9, tl.uint64) << 32) | 0x7F4A7C15)
        z = (z ^ (z >> 30)) * ((tl.full([], 0xBF58476D, tl.uint64) << 32) | 0x1CE4E5B9)
        z = (z ^ (z >> 27)) * ((tl.full([], 0x94D049BB, tl.uint64) << 32) | 0x133111EB)
        return z ^ (z >> 31)

    @triton.jit
    def _sr_kernel(x_ptr, e_ptr, out_ptr, n, start, k_hi, k_lo, HAS_E: tl.constexpr,
                   MB: tl.constexpr, EMIN: tl.constexpr, MAXC: tl.constexpr, SIGNB: tl.constexpr,
                   BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        keep = offs < n
        xb = tl.load(x_ptr + offs, mask=keep, other=0.0).to(tl.int32, bitcast=True).to(tl.int64)
        xb = xb & 0xFFFFFFFF
        sign = xb >> 31
        ex = (xb >> 23) & 0xFF
        man = xb & 0x7FFFFF
        M = tl.where(ex > 0, man | (1 << 23), man)
        eb = tl.where(ex > 0, ex - 127, -126)
        if HAS_E:
            e = tl.load(e_ptr + offs, mask=keep, other=0).to(tl.int64)
        else:
            e = tl.zeros([BLOCK], dtype=tl.int64)
        nz = M > 0
        lead = ((tl.maximum(M, 1).to(tl.float32).to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
        fl = eb - 23 + lead - e
        E = tl.maximum(fl, EMIN)
        qe = E - MB
        sh = (qe + e) - (eb - 23)
        n_up = M << tl.minimum(tl.maximum(-sh, 0), 40)
        n_dn = M >> tl.minimum(tl.maximum(sh, 0), 62)
        nn = tl.where(sh <= 0, n_up, n_dn)
        one = tl.full([BLOCK], 1, tl.int64)
        R = tl.where(sh > 0, M & ((one << tl.minimum(tl.maximum(sh, 0), 62)) - 1), 0)
        up_sh = tl.minimum(tl.maximum(32 - sh, 0), 32)
        dn = tl.minimum(tl.maximum(sh - 32, 0), 62)
        c_big = R << up_sh
        c_small = (R + (one << dn) - 1) >> dn
        c = tl.where(sh <= 32, c_big, tl.where(dn >= 40, (R > 0).to(tl.int64), c_small))
        K = (k_hi.to(tl.uint64) << 32) | k_lo.to(tl.uint64)
        h = _sr_mixu((offs + start).to(tl.uint64) ^ K)
        r = (h >> 32).to(tl.int64)
        k = nn + (r < c).to(tl.int64)
        code = ((qe + MB - EMIN) << MB) + k
        code = tl.where(sh <= -40, MAXC, code)
        code = tl.minimum(code, MAXC)
        code = tl.where(nz, code, 0)
        code = tl.where((sign == 1) & (code != 0), code | SIGNB, code)
        tl.store(out_ptr + offs, code, mask=keep)

    _TRITON["k"] = _sr_kernel
    return _sr_kernel


def round_triton(x: torch.Tensor, t, key: int, site="sround", step: int = 0,
                 scale_exp: Optional[torch.Tensor] = None, start: int = 0) -> torch.Tensor:
    """The same rule as one fused Triton kernel on CUDA (int64 codes)."""
    tg = target(t)
    if x.device.type != "cuda":
        raise FormatError("the triton backend runs on CUDA only")
    x32 = _as_f32(x).contiguous().reshape(-1)
    if not bool(torch.isfinite(x32).all()):
        raise FormatError("keyed stochastic rounding: NaN or Inf in the input")
    n = x32.numel()
    out = torch.empty(n, dtype=torch.int64, device=x.device)
    e = None if scale_exp is None else scale_exp.reshape(-1).to(x.device, torch.int32).contiguous()
    K = stream(key, site, step)
    if n:
        blk = 1024
        _triton_kernel()[((n + blk - 1) // blk,)](
            x32, e if e is not None else x32, out, n, start, K >> 32, K & 0xFFFFFFFF,
            HAS_E=e is not None, MB=tg.m, EMIN=tg.emin, MAXC=tg.max_code,
            SIGNB=1 << (tg.bits - 1), BLOCK=blk)
    return out.reshape(x.shape)


def round_codes(x: torch.Tensor, t, key: int, site="sround", step: int = 0,
                scale_exp: Optional[torch.Tensor] = None, backend: str = "auto") -> torch.Tensor:
    """Keyed SR codes of x (optionally divided by 2^scale_exp). `backend` "auto" is Triton on
    CUDA when installed, torch otherwise; every backend gives the same codes."""
    if backend == "auto":
        backend = "torch"
        if x.device.type == "cuda":
            try:
                import triton  # noqa: F401
                backend = "triton"
            except ImportError:
                pass
    if backend == "python":
        return round_python(x, t, key, site, step, scale_exp).to(x.device)
    if backend == "torch":
        return round_torch(x, t, key, site, step, scale_exp)
    if backend == "triton":
        return round_triton(x, t, key, site, step, scale_exp)
    raise FormatError(f"unknown backend {backend!r}; one of {', '.join(BACKENDS)} or 'auto'")


def round_values(x: torch.Tensor, t, key: int, site="sround", step: int = 0,
                 scale_exp: Optional[torch.Tensor] = None, backend: str = "auto") -> torch.Tensor:
    """float64 values of the keyed SR of x (times 2^scale_exp back, so they compare with x)."""
    tg = target(t)
    v = tg.decode(round_codes(x, tg, key, site, step, scale_exp, backend))
    if scale_exp is not None:
        from .formats import pow2
        v = v * pow2(scale_exp.to(v.device).long())
    return v


def to_dtype(x: torch.Tensor, dtype: torch.dtype, key: int, site="sround", step: int = 0,
             backend: str = "auto") -> torch.Tensor:
    """Keyed SR cast of a float32 tensor to bfloat16 or float16 (the tensor itself)."""
    name = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype)
    if name is None:
        raise FormatError(f"to_dtype rounds to bfloat16 or float16, got {dtype}")
    c = round_codes(x, name, key, site, step, None, backend)
    if name == "bf16":
        return (c << 16).to(torch.int32).view(torch.float32).to(torch.bfloat16)
    return c.to(torch.int16).view(torch.float16)


# ----------------------------------------------------------------------------- verification


def verify_codes(codes: torch.Tensor, x: torch.Tensor, t, key: int, site="sround", step: int = 0,
                 scale_exp: Optional[torch.Tensor] = None):
    """Check an SR kernel's codes against the keyed reference, bit for bit. Returns a list of
    `Issue` (TS1006 naming the first differing element) or an empty list."""
    from ..codes import Issue

    ref = round_torch(x.detach().cpu(), t, key, site, step,
                      None if scale_exp is None else scale_exp.detach().cpu())
    got = codes.detach().cpu().long().reshape(ref.shape)
    bad = (ref != got).reshape(-1)
    if not bool(bad.any()):
        return []
    i = int(bad.nonzero()[0])
    return [Issue("TS1006", f"stochastic rounding to {target(t).name}: {int(bad.sum())} of "
                            f"{bad.numel()} codes differ from the keyed reference; first at flat "
                            f"index {i}: got 0x{int(got.reshape(-1)[i]):x}, want "
                            f"0x{int(ref.reshape(-1)[i]):x}")]


@dataclass
class BiasTest:
    """Mean of (rounded - exact) over many keyed draws, in units of each element's quantum."""

    draws: int
    elements: int
    mean_err: float          # mean of (round - x) / quantum over all draws and elements
    z: float                 # mean_err over its standard error
    var_ratio: float         # observed variance / f (1 - f) predicted (1 is ideal)
    max_abs_elem_z: float    # the worst single element's z score
    ok: bool


def bias_test(x: torch.Tensor, t, draws: int = 256, key: int = 1, site="sround.bias",
              z_limit: float = 5.0, backend: str = "torch") -> BiasTest:
    """Statistical unbiasedness of keyed SR: round x at `draws` steps and compare the mean with
    x in float64. Elements at or above the format's max are excluded (the clamp is biased by
    design). `ok` when the pooled z and the worst element's z (Bonferroni over the elements) are
    below `z_limit` and the variance matches f (1 - f) within 10 %."""
    tg = target(t)
    x32 = _as_f32(x).reshape(-1)
    x64 = x32.to(torch.float64)
    # quantum and fraction of each element, exactly
    vals = torch.stack([round_values(x32, tg, key, site, s, backend=backend).cpu() for s in range(draws)])
    a = x64.abs().cpu()
    fin = a < (tg.codec.max_finite if tg.codec is not None else float(tg.decode(torch.tensor([tg.max_code]))))
    m, e = torch.frexp(a.clamp(min=1e-300))
    E = torch.maximum(e.long() - 1, torch.full_like(e.long(), tg.emin))
    q = torch.ldexp(torch.ones_like(a), E - tg.m)
    f = a / q - torch.floor(a / q)
    err = (vals - x64.cpu()) / q                                    # [draws, n]
    err = err[:, fin]
    f = f[fin]
    n = int(fin.sum())
    mean = float(err.mean())
    var_pred = f * (1 - f)
    se = float((var_pred.sum() / draws).sqrt()) / max(n, 1)
    z = mean / se if se > 0 else (0.0 if mean == 0 else float("inf"))
    obs_var = float(err.var(dim=0, unbiased=True).sum())
    vr = obs_var / float(var_pred.sum()) if float(var_pred.sum()) > 0 else 1.0
    # per element z only where the normal approximation holds (draws f (1 - f) >= 5); an
    # element with f = 0 must never move
    el_se = (var_pred / draws).sqrt()
    normal = var_pred * draws >= 5
    el_z = err.mean(dim=0)[normal] / el_se[normal]
    exact_bad = bool((err[:, f == 0] != 0).any())
    bonf = max(z_limit, math.sqrt(2 * math.log(max(n, 2))) + 2.0)
    worst = float(el_z.abs().max()) if el_z.numel() else 0.0
    if exact_bad:
        worst = float("inf")
    ok = abs(z) < z_limit and worst < bonf and abs(vr - 1) < 0.1
    return BiasTest(draws, n, mean, z, vr, worst, ok)


def bias_issue(bt: BiasTest, t) -> list:
    from ..codes import Issue
    if bt.ok:
        return []
    return [Issue("TS1007", f"stochastic rounding to {target(t).name} is biased: mean error "
                            f"{bt.mean_err:.3g} quanta (z {bt.z:.2f}), variance ratio "
                            f"{bt.var_ratio:.3f}, worst element z {bt.max_abs_elem_z:.2f} over "
                            f"{bt.draws} draws of {bt.elements} elements")]
