"""Exact low precision formats, defined as data and evaluated in float64.

Every format here is a table of the values its codes stand for. Decoding is a table lookup;
encoding is round to nearest, ties to the even code, saturating at the largest finite value (the
rule of `cvt.rn.satfinite` on NVIDIA GPUs and of the OCP microscaling spec). Nothing here calls a
PyTorch or Triton cast, so the reference does not inherit a bug from the path it checks; the tests
compare the tables with torch's float8 casts and with the GPU's own conversions instead.

Formats: INT8 and INT4 (two's complement), FP8 E4M3 (`fn`: no infinity, 448 max) and E5M2 (IEEE
like, 57344 max), FP4 E2M1 (6 max), the E8M0 scale (a power of two) and the E4M3 block scale. The FP8 "fnuz" tables (E4M3FNUZ, max 240, and E5M2FNUZ: no
negative zero, 0x80 is NaN) and two more E4M3 variants (bias 11 fnuz, IEEE style with Inf) exist
for the variant check in `fp8variant.py`; MX block formats are in `mx.py`.

Weight formats built from them (`QuantWeight`, all for a weight W of shape [N, K], K the input
dim):

| spec | elements | scales |
|---|---|---|
| `int8` | INT8 in [-127, 127] | one float32 per output channel, amax / 127 |
| `int4:gG` | INT4 in [-8, 7] | one float16 per group of G inputs of a channel, amax / 7 |
| `fp8_e4m3:channel`, `fp8_e5m2:tensor`, ... | FP8 | float32 per channel or per tensor, amax / max |
| `nvfp4` | E2M1 | E4M3 per block of 16 inputs, times one float32 per tensor (amax / (448 * 6)) |
| `mxfp4` | E2M1 | E8M0 per block of 32 inputs: 2^(floor(log2 amax) - 2) |

The FP8 and FP4 specs also quantize the activations at run time (W8A8, W4A4): FP8 per token
(row), FP4 per block of the same size along K, NVFP4 with a per call tensor scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

F64 = torch.float64


class FormatError(ValueError):
    """A format spec Truestar does not define, or data that does not fit its format."""


# ----------------------------------------------------------------------------- element formats


@dataclass(frozen=True)
class Codec:
    """A float format given by its value table. `values[c]` is the value of code c (NaN for NaN
    codes). Codes with the top bit set are the negative values (sign magnitude)."""

    name: str
    bits: int
    values: Tuple[float, ...]
    max_finite: float

    def table(self, device=None) -> torch.Tensor:
        return torch.tensor(self.values, dtype=F64, device=device)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Exact float64 value of every code."""
        return self.table(codes.device)[codes.long()]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Codes (int64) for float64 values: nearest representable magnitude, ties to the even
        code, saturating at max_finite. NaN is refused (FormatError), never silently mapped."""
        x = x.to(F64)
        if torch.isnan(x).any():
            raise FormatError(f"{self.name}: cannot encode NaN")
        half = 1 << (self.bits - 1)
        mags = self.table(x.device)[:half]
        finite = torch.isfinite(mags)
        mags = mags[finite]                      # increasing, codes 0 .. len - 1
        a = x.abs().clamp(max=self.max_finite)
        hi = torch.searchsorted(mags, a).clamp(max=mags.numel() - 1)
        lo = (hi - 1).clamp(min=0)
        dlo = a - mags[lo]
        dhi = mags[hi] - a
        pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
        code = torch.where(pick_hi, hi, lo)
        code = torch.where(a == mags[hi], hi, code)
        neg = (x < 0) & (code != 0)
        return code + neg.long() * half

    def ambiguous(self, x: torch.Tensor, rel: float) -> torch.Tensor:
        """True where x lies within `rel` (relative) of a rounding midpoint, or of the
        saturation point: a kernel that rounded x once in float32 before encoding may legally
        pick the other code there (double rounding), so a mismatch is not counted as a bug."""
        x = x.to(F64).abs()
        half = 1 << (self.bits - 1)
        mags = self.table(x.device)[:half]
        mags = mags[torch.isfinite(mags)]
        mids = (mags[1:] + mags[:-1]) / 2
        i = torch.searchsorted(mids, x).clamp(max=mids.numel() - 1)
        near = (x - mids[i]).abs() <= rel * x
        j = (i - 1).clamp(min=0)
        near |= (x - mids[j]).abs() <= rel * x
        return near


def _minifloat(name: str, ebits: int, mbits: int, bias: int, ieee_special: bool,
               fn_nan_only: bool) -> Codec:
    vals = []
    for code in range(1 << (1 + ebits + mbits)):
        s = code >> (ebits + mbits)
        e = (code >> mbits) & ((1 << ebits) - 1)
        m = code & ((1 << mbits) - 1)
        emax = (1 << ebits) - 1
        if ieee_special and e == emax:
            v = math.inf if m == 0 else math.nan
        elif fn_nan_only and e == emax and m == (1 << mbits) - 1:
            v = math.nan
        elif e == 0:
            v = m * 2.0 ** (1 - bias - mbits)
        else:
            v = (1 + m / 2 ** mbits) * 2.0 ** (e - bias)
        vals.append(-v if s else v)
    finite = [abs(v) for v in vals if math.isfinite(v)]
    # sign magnitude codes: the negative half is the sign bit plus the positive code
    return Codec(name, 1 + ebits + mbits, tuple(vals), max(finite))


E4M3 = _minifloat("e4m3", 4, 3, 7, ieee_special=False, fn_nan_only=True)     # max 448
E5M2 = _minifloat("e5m2", 5, 2, 15, ieee_special=True, fn_nan_only=False)    # max 57344
E2M1 = _minifloat("e2m1", 2, 1, 1, ieee_special=False, fn_nan_only=False)    # max 6
FP8 = {"e4m3": E4M3, "e5m2": E5M2}


def _fnuz(name: str, ebits: int, mbits: int, bias: int) -> Codec:
    """An FP8 "fnuz" table (finite, no negative zero): every pattern is a finite value except
    the one that would be negative zero (sign bit only, 0x80), which is the single NaN. There is
    no infinity. The same bits mean other values than in the OCP formats (E4M3FNUZ has bias 8, so
    a normal code is worth half its OCP E4M3 value)."""
    vals = []
    for code in range(1 << (1 + ebits + mbits)):
        s = code >> (ebits + mbits)
        e = (code >> mbits) & ((1 << ebits) - 1)
        m = code & ((1 << mbits) - 1)
        if s and e == 0 and m == 0:
            vals.append(math.nan)
            continue
        v = m * 2.0 ** (1 - bias - mbits) if e == 0 else (1 + m / 2 ** mbits) * 2.0 ** (e - bias)
        vals.append(-v if s else v)
    return Codec(name, 1 + ebits + mbits, tuple(vals), max(abs(v) for v in vals if math.isfinite(v)))


E4M3FNUZ = _fnuz("e4m3fnuz", 4, 3, 8)        # max 240 (AMD MI300 class, gfx942)
E5M2FNUZ = _fnuz("e5m2fnuz", 5, 2, 16)       # max 57344
E4M3B11FNUZ = _fnuz("e4m3b11fnuz", 4, 3, 11)  # max 30 (ml_dtypes, XLA)
# IEEE style E4M3 (ml_dtypes float8_e4m3): exponent 15 holds Inf and NaN, max 240
E4M3IEEE = _minifloat("e4m3ieee", 4, 3, 7, ieee_special=True, fn_nan_only=False)


def e8m0_decode(codes: torch.Tensor) -> torch.Tensor:
    """E8M0 scale: 2^(code - 127); code 255 is NaN."""
    c = codes.long()
    v = pow2(c - 127)
    return torch.where(c == 255, torch.full_like(v, math.nan), v)


def pow2(e: torch.Tensor) -> torch.Tensor:
    """2^e exactly as float64, built from the exponent bits. torch.pow(2.0, e) and torch.ldexp
    on CUDA are not exact for every e (2^-19 came back as 2^-19 (1 - 2^-53) on torch 2.11,
    found when the quantizer check called a correct MXFP4 kernel wrong)."""
    e = e.long()
    if bool(((e < -1022) | (e > 1023)).any()):
        raise FormatError("pow2: exponent outside the float64 normal range")
    return ((e + 1023) << 52).view(F64)


def round_half_even_int(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Integer codes: round half to even (torch.round is half to even), then clamp."""
    return torch.round(x.to(F64)).clamp(lo, hi).long()


def div_f32_rn(a: torch.Tensor, d: float) -> torch.Tensor:
    """a / d correctly rounded to float32 (a any float tensor, d a nonzero float).

    PyTorch on CUDA divides a tensor by a Python scalar as a multiplication by the rounded
    reciprocal, which is one float32 ulp off for some inputs (37 of 64 rows of a float16 tensor
    divided by 448, measured on torch 2.11). Here the float64 quotient is rounded to float32 and
    then its two float32 neighbours are compared exactly (products of a float32 and a short
    constant are exact in float64)."""
    a64 = a.to(F64)
    q = (a64 / d).to(torch.float32)
    best = q
    berr = (q.to(F64) * d - a64).abs()
    for direction in (math.inf, -math.inf):
        c = torch.nextafter(q, torch.full_like(q, direction))
        e = (c.to(F64) * d - a64).abs()
        even = (c.view(torch.int32) & 1) == 0
        take = (e < berr) | ((e == berr) & even)
        best = torch.where(take, c, best)
        berr = torch.where(take, e, berr)
    return best


# ----------------------------------------------------------------------------- weight formats


@dataclass(frozen=True)
class Format:
    """A parsed format spec (see the module docstring for the table)."""

    kind: str                 # int8, int4, fp8, nvfp4, mxfp4
    group: int = 0            # INT4 group, FP4 block (inputs per scale); 0 = whole channel
    elem: str = ""            # e4m3 or e5m2 for fp8
    scale: str = "channel"    # fp8 weight scale: channel or tensor
    spec: str = ""

    @property
    def activations(self) -> bool:
        """True when activations are quantized too (W8A8 FP8, W4A4 FP4)."""
        return self.kind in ("fp8", "nvfp4", "mxfp4")

    @property
    def bits(self) -> int:
        return {"int8": 8, "int4": 4, "fp8": 8, "nvfp4": 4, "mxfp4": 4}[self.kind]


SPECS = ("int8", "int4:g32", "int4:g64", "int4:g128", "fp8_e4m3:channel", "fp8_e4m3:tensor",
         "fp8_e5m2:channel", "fp8_e5m2:tensor", "nvfp4", "mxfp4")


def parse(spec: str) -> Format:
    s = spec.strip().lower()
    if s == "int8":
        return Format("int8", 0, spec=s)
    if s.startswith("int4"):
        g = 128
        if ":" in s:
            part = s.split(":", 1)[1]
            if not part.startswith("g") or not part[1:].isdigit():
                raise FormatError(f"{spec!r}: write int4:g32, int4:g64 or int4:g128")
            g = int(part[1:])
        if g not in (32, 64, 128):
            raise FormatError(f"{spec!r}: INT4 group must be 32, 64 or 128, got {g}")
        return Format("int4", g, spec=f"int4:g{g}")
    if s.startswith("fp8"):
        head, _, sc = s.partition(":")
        elem = head[4:] if head.startswith("fp8_") else "e4m3"
        sc = sc or "channel"
        if elem not in FP8 or sc not in ("channel", "tensor"):
            raise FormatError(f"{spec!r}: write fp8_e4m3 or fp8_e5m2, then :channel or :tensor")
        return Format("fp8", 0, elem=elem, scale=sc, spec=f"fp8_{elem}:{sc}")
    if s == "nvfp4":
        return Format("nvfp4", 16, elem="e2m1", spec=s)
    if s == "mxfp4":
        return Format("mxfp4", 32, elem="e2m1", spec=s)
    raise FormatError(f"unknown format {spec!r}; known: {', '.join(SPECS)}")


@dataclass
class QuantWeight:
    """A quantized [N, K] weight in the layout its kernel reads.

    * int8: `data` int8 [N, K]; `scale` float32 [N].
    * int4: `data` uint8 [N, K / 2]. Within each group of G inputs, byte j holds input j in its
      low nibble and input j + G / 2 in its high nibble (two's complement). The kernel then
      needs no interleave: the low nibbles of a group are its first half, the high its second.
      `scale` float16 [N, K / G].
    * fp8: `data` float8 [N, K]; `scale` float32 [N] (channel) or [1] (tensor).
    * nvfp4: `data` uint8 [N, K / 2], two E2M1 codes per byte, input 2i in the low nibble (the
      layout of `tl.dot_scaled`); `scale` float8_e4m3fn [N, K / 16]; `tensor_scale` float32.
    * mxfp4: `data` as nvfp4; `scale` uint8 E8M0 codes [N, K / 32].
    """

    fmt: Format
    shape: Tuple[int, int]
    data: torch.Tensor
    scale: torch.Tensor
    tensor_scale: Optional[torch.Tensor] = None
    meta: Dict[str, float] = field(default_factory=dict)

    # -------------------------------------------------------------- exact decoding

    def codes(self) -> torch.Tensor:
        """Element codes as int64 [N, K] in natural input order (INT values for int8 / int4,
        format codes for fp8 / fp4)."""
        N, K = self.shape
        k = self.fmt.kind
        if k == "int8":
            return self.data.long()
        if k == "int4":
            G = self.fmt.group
            b = self.data.view(torch.uint8).long().view(N, K // G, G // 2)
            lo = b & 15
            hi = b >> 4
            q = torch.cat([lo, hi], dim=2).view(N, K)
            return torch.where(q >= 8, q - 16, q)
        if k == "fp8":
            return self.data.view(torch.uint8).long()
        b = self.data.long()
        return torch.stack([b & 15, b >> 4], dim=2).view(N, K)

    def scales_f64(self) -> torch.Tensor:
        """Per element scale as float64 [N, K] (exact)."""
        N, K = self.shape
        k = self.fmt.kind
        if k == "int8" or (k == "fp8"):
            s = self.scale.to(F64)
            s = s.expand(N) if s.numel() == 1 else s
            return s[:, None].expand(N, K)
        if k == "nvfp4":
            s = E4M3.decode(self.scale.view(torch.uint8)) * self.tensor_scale.to(F64)
        elif k == "mxfp4":
            s = e8m0_decode(self.scale)
        else:
            s = self.scale.to(F64)
        return s.repeat_interleave(self.fmt.group, dim=1)

    def elements_f64(self) -> torch.Tensor:
        """Decoded element values (before scaling), float64 [N, K]."""
        k = self.fmt.kind
        c = self.codes()
        if k in ("int8", "int4"):
            return c.to(F64)
        if k == "fp8":
            return FP8[self.fmt.elem].decode(c)
        return E2M1.decode(c)

    def dequantize(self) -> torch.Tensor:
        """The weight the format promises, exactly, as float64 [N, K]."""
        return self.elements_f64() * self.scales_f64()

    def nbytes(self) -> int:
        n = self.data.numel() * self.data.element_size() + self.scale.numel() * self.scale.element_size()
        return n + (4 if self.tensor_scale is not None else 0)


def _amax_groups(w: torch.Tensor, g: int) -> torch.Tensor:
    N, K = w.shape
    return w.view(N, K // g, g).abs().amax(dim=2)


def quantize_weight(w: torch.Tensor, spec) -> QuantWeight:
    """Quantize a float weight [N, K] (any float dtype; computed in float64) into `spec`.

    Scales are computed in float64 and stored in the format's scale dtype; element codes are then
    computed against the stored scale, so `dequantize()` is exactly what the kernel must compute
    with. A scale of zero (an all zero group) stores codes 0."""
    fmt = parse(spec) if isinstance(spec, str) else spec
    if w.dim() != 2:
        raise FormatError(f"weights must be 2D [N, K], got shape {tuple(w.shape)}")
    N, K = w.shape
    dev = w.device
    w64 = w.detach().to(F64)
    if fmt.group and K % fmt.group:
        raise FormatError(f"{fmt.spec}: the input dim {K} is not a multiple of the block {fmt.group}")
    if fmt.kind == "int4" and fmt.group % 2:
        raise FormatError("INT4 groups must be even")

    def safe_div(a, s):
        return torch.where(s > 0, a / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(a))

    if fmt.kind == "int8":
        s = (w64.abs().amax(dim=1) / 127).to(torch.float32)
        q = round_half_even_int(safe_div(w64, s.to(F64)[:, None]), -127, 127)
        return QuantWeight(fmt, (N, K), q.to(torch.int8).contiguous(), s.contiguous())
    if fmt.kind == "int4":
        G = fmt.group
        s = (_amax_groups(w64, G) / 7).to(torch.float16)
        if torch.isinf(s).any():
            raise FormatError("int4: a group scale overflows float16")
        sf = s.to(F64).repeat_interleave(G, dim=1)
        q = round_half_even_int(safe_div(w64, sf), -8, 7)
        u = (q & 15).view(N, K // G, G)
        packed = (u[:, :, : G // 2] | (u[:, :, G // 2:] << 4)).to(torch.uint8).view(N, K // 2)
        return QuantWeight(fmt, (N, K), packed.contiguous(), s.contiguous())
    if fmt.kind == "fp8":
        codec = FP8[fmt.elem]
        if fmt.scale == "channel":
            s = (w64.abs().amax(dim=1) / codec.max_finite).to(torch.float32)
            sf = s.to(F64)[:, None]
        else:
            s = (w64.abs().amax() / codec.max_finite).to(torch.float32).reshape(1)
            sf = s.to(F64).reshape(1, 1)
        c = codec.encode(safe_div(w64, sf.expand(N, K)))
        dt = torch.float8_e4m3fn if fmt.elem == "e4m3" else torch.float8_e5m2
        return QuantWeight(fmt, (N, K), c.to(torch.uint8).view(dt).contiguous(), s.contiguous())
    if fmt.kind in ("nvfp4", "mxfp4"):
        G = fmt.group
        am = _amax_groups(w64, G)
        ts = None
        if fmt.kind == "nvfp4":
            ts = (w64.abs().amax() / (E4M3.max_finite * E2M1.max_finite)).to(torch.float32)
            tsf = ts.to(F64)
            bcode = E4M3.encode(safe_div(am / E2M1.max_finite, tsf.expand_as(am)))
            sval = E4M3.decode(bcode) * tsf
            scale = bcode.to(torch.uint8).view(torch.float8_e4m3fn)
            ts = ts.reshape(1)
        else:
            bcode = mx_scale_code(am)
            sval = e8m0_decode(bcode)
            scale = bcode.to(torch.uint8)
        c = E2M1.encode(safe_div(w64, sval.repeat_interleave(G, dim=1)))
        packed = (c[:, 0::2] | (c[:, 1::2] << 4)).to(torch.uint8)
        return QuantWeight(fmt, (N, K), packed.contiguous(), scale.contiguous(), ts)
    raise FormatError(f"no quantizer for {fmt.spec}")


def mx_scale_code(amax: torch.Tensor) -> torch.Tensor:
    """OCP MX shared scale for E2M1: 2^(floor(log2 amax) - 2), as an E8M0 code (127 + exponent),
    clamped to the E8M0 range; amax 0 gives the smallest scale."""
    a = amax.to(F64)
    e = torch.where(a > 0, torch.floor(torch.log2(torch.where(a > 0, a, torch.ones_like(a)))),
                    torch.full_like(a, -127.0)) - 2
    # log2 of a float64 can be off by one ulp near powers of two: correct it exactly
    p = pow2(e + 2)
    e = torch.where((a > 0) & (p > a), e - 1, e)
    e = torch.where((a > 0) & (p * 2 <= a), e + 1, e)
    return (e + 127).clamp(0, 254).long()


# ----------------------------------------------------------------------------- activations


@dataclass
class QuantAct:
    """Activations [M, K] quantized at run time by a kernel: element codes, per row or per block
    scales (and the NVFP4 tensor scale). The reference reads these exact codes and scales."""

    fmt: Format
    data: torch.Tensor          # float8 [M, K] or uint8 packed [M, K / 2]
    scale: torch.Tensor         # float32 [M] (fp8), float8_e4m3fn [M, K/16], uint8 [M, K/32]
    tensor_scale: Optional[torch.Tensor] = None

    def dequantize(self) -> torch.Tensor:
        M = self.data.shape[0]
        k = self.fmt.kind
        if k == "fp8":
            c = self.data.view(torch.uint8).long()
            return FP8[self.fmt.elem].decode(c) * self.scale.to(F64)[:, None]
        b = self.data.long()
        c = torch.stack([b & 15, b >> 4], dim=2).view(M, -1)
        if k == "nvfp4":
            s = E4M3.decode(self.scale.view(torch.uint8)) * self.tensor_scale.to(F64)
        else:
            s = e8m0_decode(self.scale)
        return E2M1.decode(c) * s.repeat_interleave(self.fmt.group, dim=1)

    def codes(self) -> torch.Tensor:
        M = self.data.shape[0]
        if self.fmt.kind == "fp8":
            return self.data.view(torch.uint8).long()
        b = self.data.long()
        return torch.stack([b & 15, b >> 4], dim=2).view(M, -1)
