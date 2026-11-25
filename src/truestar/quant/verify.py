"""Two separate verdicts for a quantized matmul.

(a) Kernel verdict: is the kernel right? Compared with the exact float64 product of the operands
    the kernel was given, decoded exactly from their format (dequantized weights, and the
    activation codes and scales the run time quantizer produced). Only rounding the kernel is
    allowed to do may remain. Anything more is a kernel bug (TS460 for the matmul, TS461 for the
    activation quantizer), named by `diagnose` down to the scale index, group boundary or element
    order that is wrong.
(b) Format verdict: how much does the format itself change the answer? Computed only from float64
    references (the float model against the same model with every weight dequantized and every
    activation quantized exactly as the format says). No kernel takes part, so a kernel bug can
    never inflate or hide it. Reported as loss deltas, KL and decision flips (TS462, TS463).

Kernel bound, per output element (m, n), for a kernel that promises float32 accumulation of
exact products (every product of two codes here is exact in float32):

    |y - ref| <= u_out * |ref| + LAMBDA * sqrt(K) * 2^-24 * S + floor_out,

where S = sum_k |x_mk| |w_nk| (dequantized), u_out the unit roundoff of the output dtype and
floor_out half its smallest subnormal. The sqrt(K) term is the probabilistic accumulation bound
(Higham and Mary, 2019) with LAMBDA = 16; the worst case bound K * 2^-23 * S is reported too.
Correct kernels measured at most 0.036 of the bound over the formats, shapes and tile buckets in
the tests (docs/QUANT.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..codes import Issue
from . import formats as F
from . import kernels as Q

LAMBDA = 16.0
U32 = 2.0 ** -24
U_OUT = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8, torch.float32: 2.0 ** -24}
FLOOR_OUT = {torch.float16: 2.0 ** -25, torch.bfloat16: 2.0 ** -134, torch.float32: 2.0 ** -150}
TINY_OUT = {torch.float16: 2.0 ** -14, torch.bfloat16: 2.0 ** -126, torch.float32: 2.0 ** -126}
SCALE_AMBIGUITY = 2.0 ** -21     # NVFP4 block scales: two float32 divisions before E4M3 rounding


class QuantKernelMismatchError(AssertionError):
    """A quantized kernel disagrees with the exact dequantized float64 reference (TS460 / TS461).
    Never ignored silently."""

    def __init__(self, check: "KernelCheck"):
        self.check = check
        super().__init__(check.message())


# ----------------------------------------------------------------------------- verdict (a)


@dataclass
class KernelCheck:
    """Verdict (a) for one call of a quantized matmul."""

    fmt: str
    shape: Tuple[int, int, int]          # M, N, K
    ratio: float                         # max error / bound (fails above 1)
    worst_case_ratio: float              # max error / rigorous worst case bound
    rel_err: float                       # max |y - ref| / max |ref|
    worst_at: Tuple[int, int]
    nonfinite: int = 0                   # outputs not finite where the reference is
    quantizer_mismatches: int = 0        # activation codes or scales off the format (exact rule)
    quantizer_ambiguous: int = 0         # NVFP4 block scales at a double rounding tie (allowed)
    layer: str = ""
    diagnosis: List[str] = field(default_factory=list)

    @property
    def matmul_ok(self) -> bool:
        return self.ratio <= 1.0 and self.nonfinite == 0

    @property
    def ok(self) -> bool:
        return self.matmul_ok and self.quantizer_mismatches == 0

    @property
    def code(self) -> Optional[str]:
        if self.quantizer_mismatches:
            return "TS461"
        if not self.matmul_ok:
            return "TS460"
        return None

    def message(self) -> str:
        M, N, K = self.shape
        where = f"{self.layer}: " if self.layer else ""
        if self.quantizer_mismatches:
            return (f"{where}{self.fmt} activation quantizer wrote {self.quantizer_mismatches} codes "
                    f"or scales that differ from the format's rounding (M={M}, K={K})")
        if not self.matmul_ok:
            s = (f"{where}{self.fmt} kernel error {self.ratio:.3g}x its bound vs the exact "
                 f"dequantized float64 reference (max rel {self.rel_err:.2e}, worst at "
                 f"{self.worst_at}, M={M} N={N} K={K})")
            if self.nonfinite:
                s += f"; {self.nonfinite} outputs not finite where the reference is"
            if self.diagnosis:
                s += "; " + "; ".join(self.diagnosis)
            return s
        return f"{where}{self.fmt} kernel within {self.ratio:.3g} of its bound"

    def issue(self) -> Optional[Issue]:
        return Issue(self.code, self.message()) if self.code else None

    def as_dict(self) -> Dict:
        return {"layer": self.layer, "format": self.fmt, "shape": list(self.shape),
                "ratio": self.ratio, "worst_case_ratio": self.worst_case_ratio,
                "rel_err": self.rel_err, "worst_at": list(self.worst_at),
                "nonfinite": self.nonfinite, "quantizer_mismatches": self.quantizer_mismatches,
                "quantizer_ambiguous": self.quantizer_ambiguous, "ok": self.ok,
                "code": self.code, "diagnosis": self.diagnosis}


def _deq(qw: F.QuantWeight) -> torch.Tensor:
    cache = getattr(qw, "_deq_cache", None)
    if cache is None or cache.device != qw.data.device:
        cache = qw.dequantize()
        qw._deq_cache = cache
    return cache


def check_quantizer(x: torch.Tensor, act: F.QuantAct) -> Tuple[int, int]:
    """(mismatches, ambiguous) of a run time activation quantizer against the format's rule.

    Scales: FP8 per row amax / max as a correctly rounded float32; MXFP4 E8M0 exactly; NVFP4
    the tensor scale exactly and each E4M3 block scale except within 2^-21 of an E4M3 midpoint
    (two float32 divisions may round there either way). Element codes: exactly the format's
    round to nearest even of x / (the scale the kernel stored), no window."""
    fmt = act.fmt
    x64 = x.to(F.F64)
    M, K = x.shape
    bad = 0
    amb = 0
    if fmt.kind == "fp8":
        codec = F.FP8[fmt.elem]
        s_exp = F.div_f32_rn(x.abs().amax(dim=1), codec.max_finite)
        bad += int((s_exp != act.scale).sum())
        s = act.scale.to(F.F64)[:, None]
        sv = s.expand(M, K)
    else:
        G = fmt.group
        am = x64.abs().view(M, K // G, G).amax(dim=2)
        if fmt.kind == "nvfp4":
            ts_exp = F.div_f32_rn(x.abs().amax(), 448.0 * 6.0).reshape(1)
            bad += int((ts_exp != act.tensor_scale).sum())
            ts = act.tensor_scale.to(F.F64)
            want = am / 6.0 / ts if float(ts) > 0 else torch.zeros_like(am)
            got = act.scale.view(torch.uint8).long()
            exp_code = F.E4M3.encode(want)
            diff = F.E4M3.decode(exp_code) != F.E4M3.decode(got)
            near = F.E4M3.ambiguous(want, SCALE_AMBIGUITY)
            bad += int((diff & ~near).sum())
            amb += int((diff & near).sum())
            sv = F.E4M3.decode(got) * ts
        else:
            got = act.scale.long()
            bad += int((F.mx_scale_code(am) != got).sum())
            sv = F.e8m0_decode(got)
        sv = sv.repeat_interleave(G, dim=1)
        codec = F.E2M1
    ex = torch.where(sv > 0, x64 / torch.where(sv > 0, sv, torch.ones_like(sv)), torch.zeros_like(x64))
    want = codec.decode(codec.encode(ex))
    got = codec.decode(act.codes())
    bad += int((want != got).sum())
    return bad, amb


def check_matmul(x: torch.Tensor, qw: F.QuantWeight, y: torch.Tensor,
                 act: Optional[F.QuantAct] = None, layer: str = "") -> KernelCheck:
    """Verdict (a) for y = kernel(x, qw). Runs the float64 reference on y's device."""
    N, K = qw.shape
    M = x.shape[0]
    xd = act.dequantize() if act is not None else x.to(F.F64)
    Wd = _deq(qw)
    ref = xd @ Wd.T
    S = xd.abs() @ Wd.abs().T
    u = U_OUT[y.dtype]
    yd = y.to(F.F64)
    finite_ref = torch.isfinite(ref)
    nonfinite = int((~torch.isfinite(yd) & finite_ref).sum())
    err = (yd - ref).abs()
    err = torch.where(torch.isfinite(err), err, torch.full_like(err, math.inf))
    tol = u * ref.abs() + LAMBDA * math.sqrt(K) * U32 * S * (1 + u) + FLOOR_OUT[y.dtype]
    worst = u * ref.abs() + K * 2.0 * U32 * S * (1 + u) + FLOOR_OUT[y.dtype]
    r = err / tol
    flat = int(torch.argmax(torch.where(torch.isnan(r), torch.full_like(r, math.inf), r)))
    ratio = float(r.reshape(-1)[flat])
    scale = float(ref.abs().max().clamp(min=1e-30))
    qm, qa = check_quantizer(x, act) if act is not None else (0, 0)
    return KernelCheck(qw.fmt.spec, (M, N, K), ratio, float((err / worst).max()),
                       float(err.max()) / scale, (flat // N, flat % N), nonfinite, qm, qa, layer)


# ----------------------------------------------------------------------------- diagnosis


def effective_weight(qw: F.QuantWeight, run: Callable[[torch.Tensor], torch.Tensor],
                     dtype=torch.float16, chunk: int = 1024) -> torch.Tensor:
    """The weight a kernel actually multiplies with, read column by column: row k of the input
    is the unit vector e_k, so output row k is W[:, k] as the kernel sees it (one exact product,
    one scale, one rounding to `dtype`). Unit vectors quantize exactly in every format here."""
    N, K = qw.shape
    dev = qw.data.device
    cols = []
    for k0 in range(0, K, chunk):
        k1 = min(K, k0 + chunk)
        e = torch.zeros((k1 - k0, K), device=dev, dtype=dtype)
        e[torch.arange(k1 - k0), torch.arange(k0, k1)] = 1
        cols.append(run(e).to(F.F64))
    return torch.cat(cols, dim=0).T.contiguous()          # [N, K]


def diagnose(qw: F.QuantWeight, run: Callable[[torch.Tensor], torch.Tensor],
             dtype=torch.float16) -> List[str]:
    """Name what a failing kernel gets wrong, from its effective weight (no access to its code).

    Reports which channels and which groups or blocks hold wrong values, and for each pattern
    whether a whole group is off by one factor (a wrong scale: index, group size or boundary) or
    the right values sit at the wrong inputs (element order: nibbles, packing)."""
    N, K = qw.shape
    Wd = _deq(qw)
    We = effective_weight(qw, run, dtype)
    u = U_OUT[dtype]
    bad = (We - Wd).abs() > 2 * u * Wd.abs() + 2 * FLOOR_OUT[dtype] + 1e-30
    nb = int(bad.sum())
    if nb == 0:
        return ["the effective weight matches the dequantized weight: the error depends on the "
                "inputs (accumulation, activation scaling or a tile edge), not on a weight value"]
    out = []
    G = qw.fmt.group or K
    groups = bad.view(N, K // G, G).any(dim=2)                # [N, groups]
    gcols = groups.any(dim=0).nonzero().flatten().tolist()
    chans = bad.any(dim=1).nonzero().flatten()
    unit = "group" if qw.fmt.kind == "int4" else ("block" if G < K else "channel")
    out.append(f"{nb} of {N * K} weight values wrong, in {chans.numel()} of {N} channels"
               + (f" and {len(gcols)} of {K // G} {unit}s ({_ranges(gcols)}) of {G} inputs"
                  if G < K else ""))
    n0 = int(chans[0])
    out.extend(_explain_row(We[n0].cpu(), Wd[n0].cpu(), qw.scales_f64()[n0].cpu(), n0, G, K, unit, u,
                            TINY_OUT[dtype]))
    return out


def _explain_row(we: torch.Tensor, wd: torch.Tensor, sc: torch.Tensor, n0: int, G: int, K: int,
                 unit: str, u: float, tiny: float) -> List[str]:
    """Explain one wrong channel: element order (a permutation inside each group) or scales
    (runs of inputs off by one factor, and where the factor changes)."""
    tol = 4 * u
    if not torch.isfinite(we).all():
        bad = (~torch.isfinite(we)).nonzero().flatten().tolist()
        return [f"channel {n0}: {len(bad)} effective weights are not finite (overflow inside the "
                "kernel: accumulator or intermediate dtype too narrow)"]
    # element order: does each group hold the right values at other inputs?
    if G < K or unit == "channel":
        g = G if G < K else K
        e = we.view(-1, g)
        d = wd.view(-1, g)
        close = lambda x, y: bool(((x - y).abs() <= tol * y.abs() + 1e-30).all())
        if not close(e, d) and close(torch.sort(e, 1).values, torch.sort(d, 1).values):
            idx = torch.arange(g)
            if close(e, d[:, idx ^ 1]):
                how = "neighbouring inputs swapped in pairs (2i and 2i + 1): the two codes of each byte are read in the wrong order"
            elif close(e, d[:, (idx + g // 2) % g]):
                how = (f"the first and second half of each {unit} swapped: the low and high nibbles "
                       "of each byte are read in the wrong order")
            else:
                how = "inputs permuted inside each group: packing order"
            return [f"channel {n0}: the right values at the wrong inputs, {how}"]
    # entries whose effective value is subnormal in the output dtype carry no usable ratio
    nz = (wd.abs() > 0) & (we.abs() >= tiny)
    r = torch.where(nz, we / torch.where(nz, wd, torch.ones_like(wd)), torch.full_like(wd, math.nan))

    def block_factors(P: int):
        """Per block of P inputs: its one factor, or None when the block is not uniform."""
        f = []
        for b0 in range(0, K, P):
            v = r[b0:b0 + P]
            v = v[~torch.isnan(v)]
            if v.numel() == 0:
                f.append(1.0)
                continue
            m = float(v.median())
            if float((v - m).abs().max()) > tol * abs(m) + 1e-12:
                return None
            f.append(m)
        return f

    P = G if G < K else K
    facs = None
    while P >= 4:
        facs = block_factors(P)
        if facs is not None:
            break
        if P % 2:
            break
        P //= 2
    if facs is None:
        rel = float(((we - wd).abs() / wd.abs().clamp(min=1e-30)).max())
        return [f"channel {n0}: values neither scaled by one factor per block nor permuted "
                f"(relative error up to {rel:.3g})"]
    wrong = [i for i, f in enumerate(facs) if abs(f - 1.0) > tol]
    if not wrong:
        return [f"channel {n0}: only values near zero differ"]
    i0 = wrong[0]
    out = [f"channel {n0}: inputs {i0 * P}..{min(K, (i0 + 1) * P) - 1} off by the factor "
           f"{facs[i0]:.4g}" + (f" ({len(wrong)} of {len(facs)} blocks of {P} wrong)" if len(facs) > 1 else "")
           + ": a wrong scale"]
    if P < G and G < K:
        out.append(f"the factor is constant over {P} inputs but the format's {unit}s hold {G}: "
                   f"the kernel uses {unit}s of {P} (group size or group boundary error)")
    elif G < K:
        out.append(f"whole {unit}s off ({unit} {_ranges(wrong)}): a wrong scale index")
        want = sc[i0 * G] * facs[i0]
        per = sc[::G]
        hits = ((per - want).abs() <= tol * want.abs()).nonzero().flatten().tolist()
        hits = [h for h in hits if h != i0]
        if hits:
            out.append(f"{unit} {i0} got the scale of {unit} {hits[0]}")
    return out


def _ranges(xs: List[int]) -> str:
    if not xs:
        return ""
    out = []
    a = b = xs[0]
    for v in xs[1:]:
        if v == b + 1:
            b = v
            continue
        out.append(f"{a}" if a == b else f"{a}..{b}")
        a = b = v
    out.append(f"{a}" if a == b else f"{a}..{b}")
    s = ", ".join(out[:6])
    return s + (", ..." if len(out) > 6 else "")


def verify_kernel(x: torch.Tensor, qw: F.QuantWeight, inject: Optional[str] = None,
                  layer: str = "", diagnose_on_fail: bool = True) -> KernelCheck:
    """Run the kernel for qw's format on x and return verdict (a), with a diagnosis on failure."""
    y, act = Q.quantized_matmul(x, qw, inject=inject, return_act=True)
    chk = check_matmul(x, qw, y, act, layer)
    if not chk.matmul_ok and diagnose_on_fail:
        chk.diagnosis = diagnose(qw, lambda e: Q.quantized_matmul(e.to(x.dtype), qw, inject=inject),
                                 x.dtype)
    return chk


# ----------------------------------------------------------------------------- verdict (b) per layer


def quantize_act_ref(x: torch.Tensor, fmt: F.Format) -> torch.Tensor:
    """The format's activation quantization in float64 (what a correct quantizer computes, with
    float64 scales), dequantized: used by the float64 format reference model."""
    x64 = x.to(F.F64)
    shp = x64.shape
    x2 = x64.reshape(-1, shp[-1])
    M, K = x2.shape
    if fmt.kind == "fp8":
        codec = F.FP8[fmt.elem]
        s = x2.abs().amax(dim=1, keepdim=True) / codec.max_finite
        q = torch.where(s > 0, x2 / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(x2))
        return (codec.decode(codec.encode(q)) * s).reshape(shp)
    G = fmt.group
    am = x2.abs().view(M, K // G, G).amax(dim=2)
    if fmt.kind == "nvfp4":
        ts = x2.abs().amax() / (448.0 * 6.0)
        bs = F.E4M3.decode(F.E4M3.encode(am / 6.0 / ts)) * ts if float(ts) > 0 else torch.zeros_like(am)
    else:
        bs = F.e8m0_decode(F.mx_scale_code(am))
    sv = bs.repeat_interleave(G, dim=1)
    q = torch.where(sv > 0, x2 / torch.where(sv > 0, sv, torch.ones_like(sv)), torch.zeros_like(x2))
    return (F.E2M1.decode(F.E2M1.encode(q)) * sv).reshape(shp)


def layer_format_loss(x: torch.Tensor, w: torch.Tensor, qw: F.QuantWeight) -> float:
    """Relative L2 change of one layer's output caused by the format alone (float64)."""
    x64 = x.to(F.F64).reshape(-1, w.shape[1])
    xq = quantize_act_ref(x64, qw.fmt) if qw.fmt.activations else x64
    y = x64 @ w.to(F.F64).T
    yq = xq @ _deq(qw).T
    return float((yq - y).norm() / y.norm().clamp(min=1e-300))
