"""Bit accurate model of a GPU matrix unit's inner product (docs/TENSOR_CORE_MODELS.md).

A matrix multiply accumulate instruction computes ``d = a1*b1 + ... + ak*bk + c``. The hardware
does not round after every term. Following the published description of NVIDIA tensor cores
(Khattak and Mikaitis, arXiv 2512.07004; Fasi, Higham, Mikaitis and Pranesh 2021), one *fused
group* of ``group`` products and the accumulator ``c`` is computed as:

1. every product is exact and is left unnormalized: its significand (up to two integer bits) keeps
   the exponent ``ea + eb``;
2. the largest exponent among the nonzero products and a nonzero ``c`` is found, raised to at least
   ``floor_exp`` when the unit has such a floor; call it ``E``;
3. every term is shifted onto the grid ``2**(E - frac_bits)``: bits below the grid are cut
   (``cut="rz"`` drops magnitude bits, ``cut="rd"`` cuts the two's complement value, which rounds
   toward minus infinity);
4. the cut terms are added exactly (the hardware keeps enough carry bits);
5. the sum is normalized and rounded once to the accumulator format, by truncation (``"rz"``) or
   round to nearest even (``"rne"``); a zero result is +0 whatever the signs.

A dot product longer than one group runs the groups in order, the rounded result of one group
being the ``c`` of the next, as one warp's chain of MMA instructions does. Two variants come from
the paper: ``pair_split`` (Hopper and B200 fp8 through ``mma.sync``: the 32 products go through
the fp16 unit as two interleaved halves of alternating pairs) and ``late_c`` (in that variant
``c`` is added after both halves, rounded to nearest even).

Everything is integer arithmetic in numpy int64, vectorised over the output elements, so results
are exact: 256 x 256 x 256 in fp16 takes about a second on a CPU.

Names here are Truestar's own; no code from any other implementation was used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------- number formats


@dataclass(frozen=True)
class Fmt:
    """A binary floating point format: ``man`` stored fraction bits, ``emin`` the exponent of the
    smallest normal number, ``maxval`` the largest finite value."""
    name: str
    man: int
    emin: int
    maxval: float


FORMATS: Dict[str, Fmt] = {f.name: f for f in (
    Fmt("fp32", 23, -126, float(np.finfo(np.float32).max)),
    Fmt("tf32", 10, -126, float(np.finfo(np.float32).max)),
    Fmt("fp16", 10, -14, 65504.0),
    Fmt("bf16", 7, -126, float.fromhex("0x1.fep127")),
    Fmt("e4m3", 3, -6, 448.0),
    Fmt("e5m2", 2, -14, 57344.0),
)}

_NEG = -(1 << 20)  # exponent of "no term"


class UnmodelledFormat(ValueError):
    """The unit has no data path for this input or accumulator format (TS1095)."""


def split(x: np.ndarray, fmt: Fmt) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite float64 values representable in ``fmt`` -> (negative, significand, exponent) with
    ``|x| = sig * 2**(exp - fmt.man)``. Subnormals get ``exp = emin`` and a significand below
    ``2**man``; zeros get significand 0. Raises ValueError for a value ``fmt`` cannot hold."""
    x = np.asarray(x, dtype=np.float64)
    neg = np.signbit(x)
    ax = np.abs(x)
    _, e = np.frexp(ax)
    ex = np.where(ax == 0, fmt.emin, np.maximum(e.astype(np.int64) - 1, fmt.emin))
    scaled = np.ldexp(ax, (fmt.man - ex).astype(np.int32))
    sig = scaled.astype(np.int64)
    bad = (scaled != sig) | (sig >= (1 << (fmt.man + 1))) | (ax > fmt.maxval)
    if bad.any():
        v = float(x[bad].flat[0])
        raise ValueError(f"{v!r} is not a finite {fmt.name} value")
    return neg, sig, ex


def round_to(x: np.ndarray, fmt: Fmt, mode: str) -> np.ndarray:
    """Round float64 ``x`` to ``fmt`` once ("rne": nearest even, "rz": toward zero); returns float64.

    Exact for every format here: the value is scaled by a power of two so that one unit in the
    last place of ``fmt`` is 1, rounded to an integer, and scaled back. (A torch cast from
    float64 to float16 or bfloat16 goes through float32 and can round twice; this does not.)
    Overflow gives infinity under "rne" and the largest finite value under "rz"."""
    x = np.asarray(x, dtype=np.float64)
    fin = np.isfinite(x) & (x != 0)
    _, e = np.frexp(np.where(fin, x, 1.0))
    ulp = np.maximum(e.astype(np.int64) - 1, fmt.emin) - fmt.man
    q = np.ldexp(np.where(fin, x, 0.0), (-ulp).astype(np.int32))
    if mode == "rne":
        r = np.rint(q)
    elif mode == "rz":
        r = np.trunc(q)
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")
    y = np.ldexp(r, ulp.astype(np.int32))
    over = np.abs(y) > fmt.maxval
    y = np.where(over, np.copysign(np.inf if mode == "rne" else fmt.maxval, x), y)
    return np.where(fin, y, x)


# ----------------------------------------------------------------------------- the unit


@dataclass(frozen=True)
class MatrixUnit:
    """Parameters of one matrix unit data path (one input format, one accumulator format)."""
    name: str
    inputs: Tuple[str, ...]  # input formats this data path accepts
    acc: str  # accumulator / output format: "fp32" or "fp16"
    group: int  # products summed in one fused step (the paper's N_FMA)
    frac_bits: int  # fraction bits of the alignment grid below the largest exponent
    cut: str = "rz"  # how bits below the grid are dropped: "rz" or "rd"
    final: str = "rz"  # rounding of the sum to the accumulator: "rz", "rne", or "via32" (rz to
    # fp32, then round to nearest even into a narrower accumulator)
    floor_exp: Optional[int] = None  # lowest exponent the alignment uses (None: no floor)
    pair_split: bool = False  # fp8 through the fp16 unit as two interleaved halves
    late_c: bool = False  # c added after the products, rounded to nearest even
    chunk: int = 0  # products per instruction for pair_split (0: same as group)

    def with_acc(self, acc: str, final: str) -> "MatrixUnit":
        return replace(self, acc=acc, final=final, name=f"{self.name}/{acc}")


def _shift(sig: np.ndarray, sh: np.ndarray, neg: np.ndarray, cut: str) -> np.ndarray:
    """Signed integer ``sig * 2**sh`` with bits below 2**0 cut per ``cut``."""
    sh = np.asarray(sh, dtype=np.int64)
    left = np.clip(sh, 0, 40)
    right = np.clip(-sh, 0, 62)
    if cut == "rz":
        mag = np.right_shift(np.left_shift(sig, left), right)
        mag = np.where(-sh > 62, 0, mag)
        return np.where(neg, -mag, mag)
    s = np.where(neg, -sig, sig)
    v = np.right_shift(np.left_shift(s, left), right)  # arithmetic shift: floor
    return np.where(-sh > 62, np.where(s < 0, -1, 0), v)


def _fused(pn, ps, pe, pfb, cn, cs, ce, cfb, unit: MatrixUnit, acc: Fmt,
           with_c: bool = True) -> np.ndarray:
    """One fused group. Products: arrays [..., g] (negative, significand, exponent) whose value is
    ``sig * 2**(exp - pfb)``; c: arrays [...] with value ``sig * 2**(exp - cfb)``. Returns float64
    values in the accumulator format."""
    nz = ps != 0
    pexp = np.where(nz, pe, _NEG)
    top = pexp.max(-1)
    if with_c:
        top = np.maximum(top, np.where(cs != 0, ce, _NEG))
    if unit.floor_exp is not None:
        top = np.where(top > _NEG, np.maximum(top, unit.floor_exp), top)
    base = top - unit.frac_bits  # exponent of the grid unit
    terms = _shift(ps, (pe - pfb) - base[..., None], pn, unit.cut).sum(-1)
    if with_c:
        terms = terms + _shift(cs, (ce - cfb) - base, cn, unit.cut)
    val = np.ldexp(terms.astype(np.float64), np.clip(base, -1074, 1023).astype(np.int32))
    val = np.where(top > _NEG, val, 0.0)
    if unit.final == "via32":  # truncated to fp32 first, then to nearest even (fp16 output)
        return _plus_zero(round_to(round_to(val, FORMATS["fp32"], "rz"), acc, "rne"))
    return _plus_zero(round_to(val, acc, unit.final))


def _plus_zero(x: np.ndarray) -> np.ndarray:
    """Every zero result is +0, also where IEEE addition would give -0 (all addends -0) and where
    a negative sum truncates to zero: measured on sm_120 for fp16, bf16 and e4m3, assumed for the
    other generations."""
    return np.where(x == 0, 0.0, x)


def _fmt_of(name: str) -> Fmt:
    if name not in FORMATS:
        raise UnmodelledFormat(f"unknown format {name!r}")
    return FORMATS[name]


def unit_matmul(a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray], unit: MatrixUnit,
                a_fmt: str, b_fmt: Optional[str] = None, max_temp: int = 1 << 21) -> np.ndarray:
    """``D = A @ B + C`` on the matrix unit, chaining groups along K in order.

    ``a`` [..., M, K] and ``b`` [..., K, N] (same leading batch shape, or none) hold values exactly
    representable in ``a_fmt`` / ``b_fmt``; ``c`` (broadcast to [..., M, N], None for zero) holds
    accumulator format values. Returns float64 [..., M, N], every value representable in the
    accumulator format. Outputs whose row, column or c holds inf or NaN follow IEEE float64
    arithmetic (the hardware's special value rules were not modelled)."""
    b_fmt = b_fmt or a_fmt
    for f in (a_fmt, b_fmt):
        if f not in unit.inputs:
            raise UnmodelledFormat(f"{unit.name} has no {f} input path (it takes "
                                   f"{', '.join(unit.inputs)})")
    fa, fb, acc = _fmt_of(a_fmt), _fmt_of(b_fmt), _fmt_of(unit.acc)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[:-2] != b.shape[:-2]:
        raise ValueError(f"batch shapes differ: {a.shape[:-2]} and {b.shape[:-2]}")
    lead = a.shape[:-2]
    m, k = a.shape[-2:]
    k2, n = b.shape[-2:]
    if k != k2:
        raise ValueError(f"inner sizes differ: {k} and {k2}")
    a = a.reshape(-1, m, k)
    b = b.reshape(-1, k, n)
    nb = a.shape[0]
    c = np.zeros((nb, m, n)) if c is None else         np.broadcast_to(np.asarray(c, dtype=np.float64), lead + (m, n)).reshape(nb, m, n)
    bad_a, bad_b, bad_c = ~np.isfinite(a), ~np.isfinite(b), ~np.isfinite(c)
    special = bad_a.any(2)[:, :, None] | bad_b.any(1)[:, None, :] | bad_c
    an, as_, ae = split(np.where(bad_a, 0.0, a), fa)  # [B, M, K]
    bn, bs, be = split(np.where(bad_b, 0.0, b).transpose(0, 2, 1), fb)  # [B, N, K]
    acc_val = np.where(bad_c, 0.0, c)
    pfb = fa.man + fb.man
    step = unit.chunk or unit.group
    per_mat = max(1, m * n * step)
    out = np.empty((nb, m, n))
    if per_mat <= max_temp:  # several whole matrices per pass
        bc = max_temp // per_mat
        pieces = [(slice(b0, min(nb, b0 + bc)), slice(0, m)) for b0 in range(0, nb, bc)]
    else:  # row blocks of one matrix per pass
        rows = max(1, max_temp // max(1, n * step))
        pieces = [(slice(bi, bi + 1), slice(r0, min(m, r0 + rows)))
                  for bi in range(nb) for r0 in range(0, m, rows)]
    for bsl, rsl in pieces:
        cur = acc_val[bsl, rsl]
        for k0 in range(0, k, step):
            ks = slice(k0, min(k, k0 + step))
            pn = an[bsl, rsl, None, ks] ^ bn[bsl, None, :, ks]
            ps = as_[bsl, rsl, None, ks] * bs[bsl, None, :, ks]
            pe = ae[bsl, rsl, None, ks] + be[bsl, None, :, ks]
            held = ~np.isfinite(cur)  # an accumulator that overflowed stays inf (or NaN)
            nxt = _apply(pn, ps, pe, pfb, np.where(held, 0.0, cur), unit, acc)
            cur = np.where(held, cur, nxt)
        out[bsl, rsl] = cur
    if special.any():
        with np.errstate(invalid="ignore", over="ignore"):
            out = np.where(special, a @ b + c, out)  # IEEE inf and NaN propagation
    return out.reshape(lead + (m, n))


def _apply(pn, ps, pe, pfb, cur, unit: MatrixUnit, acc: Fmt) -> np.ndarray:
    cn, cs, ce = split(cur, acc)
    if not unit.pair_split:
        return _fused(pn, ps, pe, pfb, cn, cs, ce, acc.man, unit, acc)
    # Two interleaved halves of alternating pairs through the fp16 unit (paper Fig. 5c): products
    # 1, 2, 5, 6, ... first with a zero accumulator, then 3, 4, 7, 8, ... on top of that result,
    # then c added and rounded to nearest even. Which half goes first is not published.
    f32 = FORMATS["fp32"]
    first = (np.arange(ps.shape[-1]) // 2) % 2 == 0
    zero = np.zeros(cur.shape, dtype=np.int64)
    part = _fused(pn[..., first], ps[..., first], pe[..., first], pfb, zero.astype(bool), zero,
                  zero, f32.man, unit, f32)
    qn, qs, qe = split(part, f32)
    part = _fused(pn[..., ~first], ps[..., ~first], pe[..., ~first], pfb, qn, qs, qe, f32.man,
                  unit, f32)
    # the float64 sum of two fp32 values rounds correctly
    return _plus_zero(round_to(part + cur, acc, "rne"))


def exact_matmul(a: np.ndarray, b: np.ndarray, c: Optional[np.ndarray], acc: str = "fp32") -> np.ndarray:
    """Correct arithmetic at the same formats: the float64 sum rounded once to nearest even (the
    ideal a unit model is compared against)."""
    out = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
    if c is not None:
        out = out + np.asarray(c, dtype=np.float64)
    return round_to(out, _fmt_of(acc), "rne")


# ----------------------------------------------------------------------------- generations
#
# Parameters per generation and input format, from Table 3 and section 4.1 of Khattak and
# Mikaitis, arXiv 2512.07004 (v4). "frac_bits" is the paper's product alignment fraction width
# (23 plus its extra alignment bits). The citations and the status of each value are the Rules of
# the matching sim: profiles in profiles.py; docs/TENSOR_CORE_MODELS.md has the table.


def _gen(name: str, **units: MatrixUnit) -> Dict[str, MatrixUnit]:
    return {fmt: replace(u, name=f"{name}:{fmt}") for fmt, u in units.items()}


_F16 = ("fp16",)
_B16 = ("bf16",)
_T32 = ("tf32",)
_FP8 = ("e4m3", "e5m2")

GENERATIONS: Dict[str, Dict[str, MatrixUnit]] = {
    # Volta: fp16 only, groups of 4, (2, 23) alignment, truncation.
    "v100": _gen("v100", fp16=MatrixUnit("", _F16, "fp32", 4, 23)),
    # Ampere data center (A100, A2, A30): groups of 8 for fp16 and bf16, 4 for tf32, one extra bit,
    # exponent floor -132.
    "a100": _gen("a100",
                 fp16=MatrixUnit("", _F16, "fp32", 8, 24, floor_exp=-132),
                 bf16=MatrixUnit("", _B16, "fp32", 8, 24, floor_exp=-132),
                 tf32=MatrixUnit("", _T32, "fp32", 4, 24, floor_exp=-132)),
    # Ada (L40S, RTX 1000, L4): A100's 16 bit paths (fitted bit for bit on an L4), fp8 with 13
    # fraction bits in groups of 16 (the paper's model; not reproduced on an L4 through Triton).
    "ada": _gen("ada",
                fp16=MatrixUnit("", _F16, "fp32", 8, 24, floor_exp=-132),
                bf16=MatrixUnit("", _B16, "fp32", 8, 24, floor_exp=-132),
                tf32=MatrixUnit("", _T32, "fp32", 4, 24, floor_exp=-132),
                e4m3=MatrixUnit("", _FP8, "fp32", 16, 13),
                e5m2=MatrixUnit("", _FP8, "fp32", 16, 13)),
    # Hopper (H100, H200): groups of 16, two extra bits, floor -133; tf32 groups of 8 (mma
    # m16n8k8); fp8 through wgmma: 13 fraction bits in groups of 32 (the narrow fp8 accumulator).
    "h100": _gen("h100",
                 fp16=MatrixUnit("", _F16, "fp32", 16, 25, floor_exp=-133),
                 bf16=MatrixUnit("", _B16, "fp32", 16, 25, floor_exp=-133),
                 tf32=MatrixUnit("", _T32, "fp32", 8, 25, floor_exp=-133),
                 e4m3=MatrixUnit("", _FP8, "fp32", 32, 13),
                 e5m2=MatrixUnit("", _FP8, "fp32", 32, 13)),
    # Hopper fp8 through mma.sync: converted to fp16, two interleaved halves on the fp16 unit, c
    # added late with round to nearest even.
    "h100-mma": _gen("h100-mma",
                     e4m3=MatrixUnit("", _FP8, "fp32", 16, 25, floor_exp=-133, pair_split=True,
                                     late_c=True, chunk=32),
                     e5m2=MatrixUnit("", _FP8, "fp32", 16, 25, floor_exp=-133, pair_split=True,
                                     late_c=True, chunk=32)),
    # Blackwell data center (B200): Hopper's 16 bit and tf32 paths; fp8 through tcgen05 with 25
    # fraction bits in groups of 32.
    "b200": _gen("b200",
                 fp16=MatrixUnit("", _F16, "fp32", 16, 25, floor_exp=-133),
                 bf16=MatrixUnit("", _B16, "fp32", 16, 25, floor_exp=-133),
                 tf32=MatrixUnit("", _T32, "fp32", 8, 25, floor_exp=-133),
                 e4m3=MatrixUnit("", _FP8, "fp32", 32, 25),
                 e5m2=MatrixUnit("", _FP8, "fp32", 32, 25)),
}
# Blackwell workstation and consumer (sm_120: RTX PRO 6000 in the paper, RTX 5070 Ti measured
# here): the paper reports B200's features with fp8 through mma.sync mapped to QMMA. Replaced by
# the measured parameters below when they differ.
GENERATIONS["sm120"] = _gen("sm120", **{f: u for f, u in GENERATIONS["b200"].items()})


def unit_for(gen: str, fmt: str, acc: str = "fp32") -> MatrixUnit:
    """The unit of generation ``gen`` for input format ``fmt``; ``acc="fp16"`` gives the fp16
    output path (the fp32 path's normalized sum rounded to nearest even fp16)."""
    units = GENERATIONS.get(gen)
    if units is None:
        raise KeyError(f"unknown tensor core generation {gen!r}; known: {', '.join(GENERATIONS)}")
    if fmt not in units:
        raise UnmodelledFormat(f"{gen} has no modelled {fmt} path (it has {', '.join(units)})")
    u = units[fmt]
    if acc == "fp32":
        return u
    if acc == "fp16":
        return u.with_acc("fp16", "rne")
    raise UnmodelledFormat(f"no {acc} accumulator path")


# ----------------------------------------------------------------------------- simulation entry

NOTE_FORMAT = "TS1095"  # prefixes of SimMode notes that truestar check turns into their own code
NOTE_OP = "TS1096"


def _quantize(x: np.ndarray, fmt: str) -> Tuple[np.ndarray, float]:
    """Round float64 ``x`` to the unit's input format; fp8 first scales per tensor by a power of
    two so the largest magnitude fits (the scale is returned and undone exactly after the unit)."""
    f = _fmt_of(fmt)
    scale = 1.0
    if fmt in ("e4m3", "e5m2"):
        fin = np.abs(x[np.isfinite(x)])
        amax = float(fin.max()) if fin.size else 0.0
        if amax > 0:
            scale = float(2.0 ** np.ceil(np.log2(amax / f.maxval)))
    return round_to(x / scale, f, "rne"), scale


def matmul_op(name: str, a: dict, gen: str, fmt: str, notes: set):
    """mm, addmm, bmm, baddbmm, mv or dot under a matrix unit model (float64 torch tensors in,
    float64 out, every value an fp32 value). ``gen="exact"``: the same input rounding with the
    exact sum rounded once (the profile's ideal)."""
    import torch

    if name in ("mm", "addmm"):
        x, w = a["mat1" if name == "addmm" else "self"], a["mat2"]
    elif name in ("bmm", "baddbmm"):
        x = a["batch1" if name == "baddbmm" else "self"]
        w = a["batch2" if name == "baddbmm" else "mat2"]
    elif name == "mv":
        x, w = a["self"], a["vec"].reshape(-1, 1)
    elif name == "dot":
        x, w = a["self"].reshape(1, -1), a["tensor"].reshape(-1, 1)
    else:
        raise KeyError(name)
    xn = x.detach().cpu().to(torch.float64).numpy()
    wn = w.detach().cpu().to(torch.float64).numpy()
    qx, sx = _quantize(xn, fmt)
    qw, sw = _quantize(wn, fmt)
    f32 = FORMATS["fp32"]
    if gen == "exact":
        acc = round_to(qx @ qw, f32, "rne")
    else:
        try:
            acc = unit_matmul(qx, qw, None, unit_for(gen, fmt), fmt)
        except UnmodelledFormat as exc:
            notes.add(f"{NOTE_FORMAT}: aten.{name}: {exc}; accumulated exactly instead")
            acc = round_to(qx @ qw, f32, "rne")
    if sx * sw != 1.0:
        acc = round_to(acc * (sx * sw), f32, "rne")
    out = torch.from_numpy(np.ascontiguousarray(acc))
    if name == "mv":
        out = out[:, 0]
    elif name == "dot":
        out = out.reshape(())
    if name in ("addmm", "baddbmm"):
        alpha, beta = float(a.get("alpha", 1)), float(a.get("beta", 1))
        bias = a["self"].to(torch.float64)
        val = out.numpy() * alpha + (bias.numpy() * beta if beta != 0 else 0.0)
        out = torch.from_numpy(np.ascontiguousarray(round_to(val, f32, "rne")))
    return out
