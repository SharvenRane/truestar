"""OCP microscaling (MX) block formats with exact tables: MXFP8 and MXFP4.

An MX block is 32 elements that share one UE8M0 scale X = 2^(code - 127). Elements are FP8
(E4M3 or E5M2) or FP4 (E2M1); every element value is its table value times
X, computed exactly in float64. Element rounding is round to nearest even with saturation at the
largest finite element value (OCP MX v1.1 section 6.3 step 2), or keyed stochastic rounding
(`truestar.quant.sround`), never a library cast.

Three rules for the shared scale are implemented, because they differ in which blocks saturate:

| rule | X | source |
|---|---|---|
| `floor` | 2^(floor(log2 amax) - emax_elem) | OCP MX v1.0 (and v1.1 with expAdj = 0) |
| `floor_adj` | the same, one power of two higher when the block max would round out of range | OCP MX v1.1 equation 4 (expAdj) |
| `ceil` | 2^ceil(log2(amax / max_elem)), at least 2^-127 | "Recipes for Pre-training LLMs with MXFP8" (arXiv 2506.08027) appendix A.1 |

`emax_elem` is the exponent of the element format's largest normal binade (8 for E4M3, 15 for
E5M2, 2 for E2M1). Every exponent here is read from the float64 bits
(`torch.frexp` is exact), never from a rounded `log2`. A block of zeros gets scale code 0.

Layout: `MXTensor.codes` holds one element code per int64 entry, `scale_codes` one UE8M0 code per
block of 32 along the last dimension.

MXFP6 (E2M3, E3M2) is not shipped: no GPU kernel runs it (Triton 3.7.1 `tl.dot_scaled` raises NYI
for e2m3), so it is not offered (docs/QUANT.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .formats import E2M1, E4M3, E5M2, F64, Codec, FormatError, e8m0_decode, pow2

BLOCK = 32

# name: (element codec, OCP format name)
MX_FORMATS: Dict[str, Codec] = {
    "mxfp8_e4m3": E4M3,
    "mxfp8_e5m2": E5M2,
    "mxfp4": E2M1,
}
SCALE_RULES = ("floor", "floor_adj", "ceil")


def emax(codec: Codec) -> int:
    """Exponent of the largest normal binade of an element format."""
    return math.frexp(codec.max_finite)[1] - 1


def mantissa_bits(codec: Codec) -> int:
    return {"e4m3": 3, "e5m2": 2, "e2m1": 1}[codec.name]


def element_codec(fmt: str) -> Codec:
    key = fmt.strip().lower()
    if key == "mxfp8":
        key = "mxfp8_e4m3"
    if key not in MX_FORMATS:
        raise FormatError(f"unknown MX format {fmt!r}; known: {', '.join(MX_FORMATS)}")
    return MX_FORMATS[key]


def _floor_log2(a: torch.Tensor) -> torch.Tensor:
    """floor(log2 a) for positive float64 a, exactly (from the exponent bits)."""
    _, e = torch.frexp(a)
    return e.long() - 1


def scale_exponent(amax: torch.Tensor, codec: Codec, rule: str = "floor") -> torch.Tensor:
    """Unbiased exponent of the shared scale (int64) for block maxima `amax` (float64, >= 0)."""
    if rule not in SCALE_RULES:
        raise FormatError(f"unknown MX scale rule {rule!r}; one of {', '.join(SCALE_RULES)}")
    a = amax.to(F64)
    pos = a > 0
    safe = torch.where(pos, a, torch.ones_like(a))
    fl = _floor_log2(safe)
    if rule in ("floor", "floor_adj"):
        e = fl - emax(codec)
        if rule == "floor_adj":
            m = mantissa_bits(codec)
            # the block max rounded to the element precision with an unbounded exponent
            frac = safe / pow2(fl)                                  # [1, 2), exact
            r = torch.round(frac * (1 << m))                        # half to even
            adj = r == (1 << (m + 1))                               # rounds into the next binade
            if codec.name == "e4m3":
                adj |= r == 15                                      # 1.875 * 2^8 = 480 is NaN in E4M3
            e = e + adj.long()
    else:
        # smallest k with 2^k * max_elem >= amax; 2^k * max_elem is exact in float64
        mx = codec.max_finite
        e = fl - (math.frexp(mx)[1] - 1)
        for _ in range(2):
            low = pow2(e.clamp(-1000, 1000)) * mx < safe
            e = torch.where(low, e + 1, e)
            high = pow2((e - 1).clamp(-1000, 1000)) * mx >= safe
            e = torch.where(high, e - 1, e)
    e = torch.where(pos, e, torch.full_like(e, -127))
    return e.clamp(-127, 127)


@dataclass
class MXTensor:
    """An MX quantized tensor [..., K]: element codes and one UE8M0 code per block of 32."""

    fmt: str
    codes: torch.Tensor          # int64 [..., K]
    scale_codes: torch.Tensor    # int64 [..., K / 32]
    rule: str = "floor"

    @property
    def codec(self) -> Codec:
        return element_codec(self.fmt)

    def scales_f64(self) -> torch.Tensor:
        return e8m0_decode(self.scale_codes).repeat_interleave(BLOCK, dim=-1)

    def dequantize(self) -> torch.Tensor:
        """Exact float64 values: element table value times the block scale."""
        return self.codec.decode(self.codes) * self.scales_f64()

    def clipped(self, x: torch.Tensor) -> int:
        """Elements of the source x that the clamp at the element max changed: their scaled
        magnitude lies above the midpoint between the max and the next value of the same
        format with an unbounded exponent, so round to nearest alone would have gone higher."""
        q = x.to(F64).abs() / self.scales_f64()
        c = self.codec
        mid = c.max_finite + 2.0 ** (emax(c) - mantissa_bits(c) - 1)
        return int((q > mid).sum())

    def nbytes(self) -> int:
        bits = self.codec.bits * self.codes.numel() + 8 * self.scale_codes.numel()
        return (bits + 7) // 8


def quantize(x: torch.Tensor, fmt: str, rule: str = "floor", rounding: str = "nearest",
             key: Optional[int] = None, site="mx", step: int = 0) -> MXTensor:
    """Quantize x [..., K] (K a multiple of 32) into an MX format, in float64.

    `rounding` "nearest" is round to nearest even with saturation; "keyed" is keyed stochastic
    rounding (`truestar.quant.sround`) of x / X with the given key, site and step, so the same
    codes come out on every device."""
    codec = element_codec(fmt)
    x64 = x.detach().to(F64)
    if x64.dim() == 0 or x64.shape[-1] % BLOCK:
        raise FormatError(f"{fmt}: the last dim must be a multiple of {BLOCK}, got {tuple(x64.shape)}")
    if not bool(torch.isfinite(x64).all()):
        raise FormatError(f"{fmt}: NaN or Inf in the input (the block scale would be NaN)")
    shp = x64.shape
    xb = x64.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK)
    e = scale_exponent(xb.abs().amax(dim=-1), codec, rule)
    sc = e + 127
    q = (xb / pow2(e)[..., None]).reshape(shp)          # exact: division by a power of two
    if rounding == "nearest":
        codes = codec.encode(q)
    elif rounding == "keyed":
        if key is None:
            raise FormatError("keyed stochastic rounding needs a key")
        from . import sround
        ee = e[..., None].expand(*e.shape, BLOCK).reshape(shp)
        codes = sround.round_codes(x.detach().reshape(shp), codec, key, site, step, scale_exp=ee)
    else:
        raise FormatError(f"unknown rounding {rounding!r}; 'nearest' or 'keyed'")
    return MXTensor(fmt.strip().lower(), codes, sc.long(), rule)


def table(fmt: str) -> Tuple[float, ...]:
    """The element value table of an MX format (code -> value), for docs and tests."""
    return element_codec(fmt).values


def matching_rules(x: torch.Tensor, scale_codes: torch.Tensor, fmt: str) -> Tuple[str, ...]:
    """The scale rules whose scales equal `scale_codes` on every block of x."""
    codec = element_codec(fmt)
    x64 = x.detach().to(F64)
    am = x64.reshape(*x64.shape[:-1], x64.shape[-1] // BLOCK, BLOCK).abs().amax(dim=-1)
    sc = scale_codes.detach().to(am.device).long()
    return tuple(r for r in SCALE_RULES if bool((scale_exponent(am, codec, r) + 127 == sc).all()))


def check_quantizer(x: torch.Tensor, codes: torch.Tensor, scale_codes: torch.Tensor, fmt: str,
                    rule: str = "floor") -> list:
    """Check an MX quantizer's output (from a kernel, a library or a file) against the exact
    rule: TS1008 when a block scale differs (naming the rules the scales do follow), TS1009 when
    an element code differs from round to nearest even of x / X with the quantizer's own scale."""
    from ..codes import Issue

    codec = element_codec(fmt)
    x64 = x.detach().to(F64)
    shp = x64.shape
    am = x64.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK).abs().amax(dim=-1)
    sc = scale_codes.detach().to(am.device).long().reshape(am.shape)
    want = scale_exponent(am, codec, rule) + 127
    out = []
    bad = (want != sc).reshape(-1)
    if bool(bad.any()):
        i = int(bad.nonzero()[0])
        follows = matching_rules(x64, sc, fmt)
        also = f"; the scales follow the {', '.join(follows)} rule" if follows else ""
        out.append(Issue("TS1008", f"{fmt}: {int(bad.sum())} of {bad.numel()} block scales differ "
                                   f"from the {rule} rule; first block {i}: code {int(sc.reshape(-1)[i])}, "
                                   f"want {int(want.reshape(-1)[i])}{also}"))
    X = e8m0_decode(sc).repeat_interleave(BLOCK, dim=-1)
    q = torch.where(X > 0, x64 / X, torch.zeros_like(x64))
    ref = codec.encode(q)
    got = codes.detach().to(ref.device).long().reshape(ref.shape)
    same = codec.decode(ref) == codec.decode(got)
    eb = (~same).reshape(-1)
    if bool(eb.any()):
        i = int(eb.nonzero()[0])
        out.append(Issue("TS1009", f"{fmt}: {int(eb.sum())} of {eb.numel()} element codes are not "
                                   f"the round to nearest even of x / X; first at flat index {i}: "
                                   f"got 0x{int(got.reshape(-1)[i]):x}, want 0x{int(ref.reshape(-1)[i]):x}"))
    return out
