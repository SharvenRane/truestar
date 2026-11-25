"""Low precision GEMM training with every GEMM verified against float64 (docs/TRAIN_NUMERICS.md).

A linear layer y = x W^T + b runs three GEMMs per step: the forward (x, W), the input gradient
dX = dY W and the weight gradient dW = dY^T X. ``LPLinear`` runs each in a low precision recipe and
gives each call two separate verdicts, as truestar.quant does for inference:

* format loss: the float64 product of the dequantized operands (the exact codes and scales the
  kernel read) against the float64 product of the unquantized operands. What the format costs.
* kernel error: the kernel's output against that dequantized float64 product, within the
  float32 accumulation bound of truestar.quant.verify (u_out |ref| + 16 sqrt(K) 2^-24 sum|a||b|).
  Anything beyond is a kernel bug (TS1042), never counted as format loss.

and one invariant per layer: the weight the backward multiplies by must be the weight the forward
multiplied by (bitwise, after dequantization). A per tensor scale, or a 2D block scale, makes W and
W^T quantize to the same values; a per channel or 1x16 block scale does not, so the gradient is the
gradient of another weight (TS1043, measured and reported).

Recipes (``RECIPES``):

| name | forward | input gradient | weight gradient | kernel |
|---|---|---|---|---|
| ``fp8`` | E4M3 x rows, E4M3 W tensor | E4M3 dY rows, E4M3 W^T tensor | E4M3 dY^T rows, E4M3 X^T tensor | Truestar Triton (`quant.kernels.matmul_fp8`) |
| ``fp8_channel`` | as fp8, W per channel | W^T per channel (other values) | as fp8 | Truestar Triton |
| ``fp8_hybrid`` | E4M3 x E4M3, tensor scales | E5M2 dY x E4M3 W | E5M2 dY^T x E4M3 X^T | cuBLASLt (`torch._scaled_mm`) |
| ``nvfp4_fwd`` | NVFP4 x, NVFP4 W (Truestar Triton, native FP4 MMA on sm_120) | bf16 dY x the forward's dequantized W | bf16 | Truestar Triton, cuBLAS |
| ``bf16`` | bf16 | bf16 | bf16 | cuBLAS |

MXFP8 is not in truestar.quant on main (no E8M0 32 block FP8 format), so it is not offered here.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..codes import Issue

F64 = torch.float64
LAMBDA = 16.0
U32 = 2.0 ** -24
U_OUT = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8, torch.float32: 2.0 ** -24}
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DT = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


@dataclasses.dataclass(frozen=True)
class Gemm:
    a: str            # element format of the left operand: e4m3, e5m2, nvfp4, bf16, f32
    b: str
    a_scale: str = "row"      # row or tensor (fp8)
    b_scale: str = "tensor"   # tensor or channel (fp8)
    arm: str = "truestar"     # truestar (Triton), cublas (torch._scaled_mm or torch.matmul), torch


@dataclasses.dataclass(frozen=True)
class Recipe:
    name: str
    fwd: Gemm
    dgrad: Gemm
    wgrad: Gemm
    consistent_weight: bool = True    # the input gradient reads the forward's dequantized weight


RECIPES = {
    "fp8": Recipe("fp8", Gemm("e4m3", "e4m3"), Gemm("e4m3", "e4m3"), Gemm("e4m3", "e4m3")),
    "fp8_channel": Recipe("fp8_channel", Gemm("e4m3", "e4m3", b_scale="channel"),
                          Gemm("e4m3", "e4m3", b_scale="channel"), Gemm("e4m3", "e4m3")),
    "fp8_hybrid": Recipe("fp8_hybrid", Gemm("e4m3", "e4m3", "tensor", "tensor", "cublas"),
                         Gemm("e5m2", "e4m3", "tensor", "tensor", "cublas"),
                         Gemm("e5m2", "e4m3", "tensor", "tensor", "cublas")),
    "nvfp4_fwd": Recipe("nvfp4_fwd", Gemm("nvfp4", "nvfp4"), Gemm("bf16", "fwd", arm="cublas"),
                        Gemm("bf16", "bf16", arm="cublas")),
    "bf16": Recipe("bf16", Gemm("bf16", "bf16", arm="cublas"), Gemm("bf16", "bf16", arm="cublas"),
                   Gemm("bf16", "bf16", arm="cublas")),
}


class LowPrecisionKernelError(RuntimeError):
    """A low precision GEMM's output is off the float64 product of its own dequantized operands by
    more than float32 accumulation allows (TS1042)."""

    def __init__(self, rec: "GemmRecord"):
        self.record = rec
        self.code = "TS1042"
        super().__init__(f"truestar TS1042 low precision kernel error: {rec.layer} {rec.which} ({rec.fmt}): "
                         f"{rec.kernel_ratio:.3g} times the float32 accumulation bound "
                         f"(worst error {rec.kernel_error:.3g}); the result is wrong, not imprecise")


@dataclasses.dataclass
class GemmRecord:
    layer: str
    which: str                # fwd, dgrad, wgrad
    fmt: str
    shape: Tuple[int, int, int]   # M, N, K
    format_loss: float        # ||ref(dequantized) - ref(exact)||_F / ||ref(exact)||_F
    kernel_error: float       # max |y - ref(dequantized)|
    kernel_ratio: float       # max over entries of that error / its bound (<= 1 is a correct kernel)

    @property
    def ok(self) -> bool:
        return self.kernel_ratio <= 1.0


@dataclasses.dataclass
class LayerRecord:
    layer: str
    recipe: str
    gemms: List[GemmRecord] = dataclasses.field(default_factory=list)
    weight_mismatch: int = 0           # entries where the backward's weight differs from the forward's
    weight_mismatch_max: float = 0.0   # largest such difference, relative to max |W|
    dw_error: Optional[float] = None   # ||dW - dW_f64||_F / ||dW_f64||_F (set by grad_report)
    dx_error: Optional[float] = None

    @property
    def consistent(self) -> bool:
        return self.weight_mismatch == 0


# ----------------------------------------------------------------------------- one GEMM, quantized


def _bound(ad: torch.Tensor, bd: torch.Tensor, ref: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    K = ad.shape[1]
    S = ad.abs() @ bd.abs().T
    return U_OUT[out_dtype] * ref.abs() + LAMBDA * math.sqrt(K) * U32 * S + 2.0 ** -140


def _fp8_tensor(x: torch.Tensor, elem: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per tensor FP8 codes and the float32 scale amax / max (exact float64 scale computation,
    saturating round to nearest even through truestar.quant.formats)."""
    from ..quant.formats import FP8

    codec = FP8[elem]
    x64 = x.detach().to(F64)
    am = float(x64.abs().max())
    s = torch.tensor(am / codec.max_finite if am > 0 else 1.0, dtype=torch.float32, device=x.device)
    c = codec.encode(x64 / s.to(F64))
    return c.to(torch.uint8).view(FP8_DT[elem]), s


def _fp8_rows(x: torch.Tensor, elem: str) -> Tuple[torch.Tensor, torch.Tensor]:
    from ..quant.formats import FP8

    codec = FP8[elem]
    x64 = x.detach().to(F64)
    am = x64.abs().amax(dim=1)
    s = torch.where(am > 0, am / codec.max_finite, torch.ones_like(am)).to(torch.float32)
    c = codec.encode(x64 / s.to(F64)[:, None])
    return c.to(torch.uint8).view(FP8_DT[elem]), s


def _deq_fp8(codes: torch.Tensor, scale: torch.Tensor, elem: str) -> torch.Tensor:
    """Per tensor FP8 codes times their scale, exactly, in float64."""
    from ..quant.formats import FP8

    return FP8[elem].decode(codes.view(torch.uint8).long()) * float(scale.to(F64).reshape(-1)[0])


@dataclasses.dataclass
class WeightFactor:
    """The forward's dequantized weight as (block values, tensor scale): block values are element
    times block scale, exact in bfloat16 (at most 6 significant bits for NVFP4), and the tensor
    scale multiplies the GEMM's result once. So a backward can multiply by exactly the forward's
    weight without the rounding of forming it."""

    blocks: torch.Tensor     # float64 [N, K]
    scale: float

    def full(self) -> torch.Tensor:
        return self.blocks * self.scale


def gemm(a: torch.Tensor, b: torch.Tensor, g: Gemm, layer: str = "", which: str = "",
         b_deq: Optional[torch.Tensor] = None, record: bool = True, inject: Optional[str] = None
         ) -> Tuple[torch.Tensor, Optional[GemmRecord], torch.Tensor, Optional[WeightFactor]]:
    """y = a @ b^T (a [M, K], b [N, K]) in the format ``g``; returns (y float32, the verdict record,
    the dequantized b in float64, its WeightFactor for NVFP4). ``b_deq`` (format "fwd"): a
    WeightFactor whose exact values are used as b."""
    wf = None
    M, K = a.shape
    N = b.shape[0]
    dev = a.device
    out = torch.float32
    if g.a in ("e4m3", "e5m2"):
        if g.arm == "truestar":
            from ..quant import kernels as QK
            from ..quant.formats import QuantAct, QuantWeight, parse

            if g.a != g.b:
                raise ValueError("the Truestar FP8 kernel needs one element format for both operands")
            ac, as_ = _fp8_rows(a, g.a) if g.a_scale == "row" else _fp8_tensor(a, g.a)
            if g.a_scale != "row":
                as_ = as_.reshape(1).expand(M).contiguous()
            bc, bs = _fp8_rows(b, g.b) if g.b_scale == "channel" else _fp8_tensor(b, g.b)
            bs = bs.reshape(-1).contiguous()
            act = QuantAct(parse(f"fp8_{g.a}:channel"), ac.contiguous(), as_.contiguous())
            qw = QuantWeight(parse(f"fp8_{g.b}:{'channel' if g.b_scale == 'channel' else 'tensor'}"), (N, K),
                             bc.contiguous(), bs)
            y = QK.matmul_fp8(act, qw, out, inject=inject)
            ad = act.dequantize()
            bd = qw.dequantize()
        else:
            ac, sa = _fp8_tensor(a, g.a)
            bc, sb = _fp8_tensor(b, g.b)
            y = torch._scaled_mm(ac.contiguous(), bc.contiguous().t(), scale_a=sa, scale_b=sb, out_dtype=out)
            ad = _deq_fp8(ac, sa, g.a)
            bd = _deq_fp8(bc, sb, g.b)
        fmt = f"{g.a}x{g.b} {g.a_scale}/{g.b_scale} {g.arm}"
    elif g.a == "nvfp4":
        from ..quant import kernels as QK
        from ..quant.formats import quantize_weight

        a16 = a.to(torch.bfloat16).contiguous()
        act = QK.quant_act_fp4(a16, "nvfp4")
        qw = quantize_weight(b.detach(), "nvfp4")
        y = QK.matmul_fp4(act, qw, out, inject=inject)
        ad = act.dequantize()
        from ..quant.formats import E4M3

        blocks = qw.elements_f64() * E4M3.decode(qw.scale.view(torch.uint8)).repeat_interleave(qw.fmt.group, dim=1)
        wf = WeightFactor(blocks, float(qw.tensor_scale.to(F64).reshape(-1)[0]))
        bd = wf.full()
        fmt = "nvfp4 x nvfp4 truestar"
    elif g.a in ("bf16", "f32"):
        dt = torch.bfloat16 if g.a == "bf16" else torch.float32
        a_ = a.to(dt)
        if g.b == "fwd" and isinstance(b_deq, WeightFactor):
            b_ = b_deq.blocks.to(dt).to(dev)
            if not torch.equal(b_.to(F64).cpu(), b_deq.blocks.cpu()):
                raise AssertionError("internal: the forward's block values are not exact in " + str(dt))
            y = (a_ @ b_.T).to(out) * b_deq.scale
            bd = b_deq.full()
        else:
            b_ = b.to(dt)
            y = (a_ @ b_.T).to(out)
            bd = b_.to(F64)
        ad = a_.to(F64)
        fmt = f"{g.a} cublas" + (" (forward weight)" if g.b == "fwd" else "")
        out = dt
    else:
        raise ValueError(f"unknown element format {g.a!r}")
    rec = None
    if record:
        a64, b64 = a.detach().to(F64), b.detach().to(F64)
        ref_q = ad.to(F64).cpu() @ bd.to(F64).cpu().T
        exact = a64.cpu() @ b64.cpu().T
        den = float(exact.norm()) or 1.0
        err = (y.detach().to(F64).cpu() - ref_q).abs()
        bound = _bound(ad.to(F64).cpu(), bd.to(F64).cpu(), ref_q, out if out in U_OUT else torch.float32)
        rec = GemmRecord(layer, which, fmt, (M, N, K), float((ref_q - exact).norm()) / den, float(err.max()),
                         float((err / bound).max()))
    return y, rec, bd, wf


# ----------------------------------------------------------------------------- the W / W^T invariant


def quantize_dequantize(w: torch.Tensor, scheme: str) -> torch.Tensor:
    """W as a recipe's weight quantizer leaves it, in float64. Schemes: ``fp8_tensor``,
    ``fp8_channel`` (per output row), ``nvfp4`` (E4M3 scale per 1 x 16 block along the input dim,
    as inference uses) and ``nvfp4_2d`` (one E4M3 scale per 16 x 16 tile, the training layout
    of the NVFP4 pretraining recipe)."""
    from ..quant.formats import E2M1, E4M3, quantize_weight

    if scheme == "fp8_tensor":
        c, s = _fp8_tensor(w, "e4m3")
        return _deq_fp8(c, s, "e4m3")
    if scheme == "fp8_channel":
        c, s = _fp8_rows(w, "e4m3")
        from ..quant.formats import FP8
        return FP8["e4m3"].decode(c.view(torch.uint8).long()) * s.to(F64)[:, None]
    if scheme == "nvfp4":
        return quantize_weight(w.detach(), "nvfp4").dequantize()
    if scheme == "nvfp4_2d":
        w64 = w.detach().to(F64)
        N, K = w64.shape
        if N % 16 or K % 16:
            raise ValueError("nvfp4_2d needs both dims multiples of 16")
        ts = (w64.abs().amax() / (E4M3.max_finite * E2M1.max_finite)).to(torch.float32).to(F64)
        tiles = w64.reshape(N // 16, 16, K // 16, 16).abs().amax(dim=(1, 3))
        bcode = E4M3.encode(torch.where(ts > 0, tiles / E2M1.max_finite / ts, torch.zeros_like(tiles)))
        sv = E4M3.decode(bcode) * ts
        full = sv.repeat_interleave(16, 0).repeat_interleave(16, 1)
        c = E2M1.encode(torch.where(full > 0, w64 / torch.where(full > 0, full, torch.ones_like(full)),
                                    torch.zeros_like(w64)))
        return E2M1.decode(c) * full
    raise ValueError(f"unknown scheme {scheme!r}")


def transpose_consistency(w: torch.Tensor, scheme: str) -> Tuple[int, float]:
    """(entries where quantize(W)^T != quantize(W^T), the largest difference over max |W|): 0 means
    the forward and the input gradient multiply by the same weight."""
    a = quantize_dequantize(w, scheme).T
    b = quantize_dequantize(w.T.contiguous(), scheme)
    d = (a - b).abs()
    return int((d > 0).sum()), float(d.max()) / max(float(w.detach().abs().max()), 1e-300)


# ----------------------------------------------------------------------------- the layer


class _LPLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, mod):
        x2 = x.reshape(-1, x.shape[-1])
        r = mod.recipe
        y, rec, wdeq, wf = gemm(x2, w, r.fwd, mod.name, "fwd", record=mod.verify, inject=mod.inject.get("fwd"))
        if rec is not None:
            mod.record.gemms.append(rec)
            mod._strict(rec)
        ctx.save_for_backward(x2, w, wdeq)
        ctx.mod, ctx.shape, ctx.wf = mod, x.shape, wf
        y = y.to(x.dtype)
        if b is not None:
            y = y + b
        return y.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, w, wdeq = ctx.saved_tensors
        mod = ctx.mod
        r = mod.recipe
        g2 = gy.reshape(-1, gy.shape[-1]).float().contiguous()
        # input gradient: dX = dY W = dY @ (W^T)^T
        if r.dgrad.b == "fwd":
            fac = WeightFactor(ctx.wf.blocks.T.contiguous(), ctx.wf.scale) if ctx.wf is not None else                 WeightFactor(wdeq.T.contiguous(), 1.0)
            dx, rec, wb, _ = gemm(g2, w.T.contiguous(), r.dgrad, mod.name, "dgrad", b_deq=fac, record=mod.verify)
        else:
            dx, rec, wb, _ = gemm(g2, w.T.contiguous(), r.dgrad, mod.name, "dgrad", record=mod.verify,
                               inject=mod.inject.get("dgrad"))
        if rec is not None:
            mod.record.gemms.append(rec)
            mod._strict(rec)
        if mod.verify:
            diff = (wb.to(F64).cpu().T - wdeq.to(F64).cpu()).abs()
            mod.record.weight_mismatch = int((diff > 0).sum())
            mod.record.weight_mismatch_max = float(diff.max()) / max(float(w.detach().abs().max()), 1e-300)
        # weight gradient: dW = dY^T X = (dY^T) @ (X^T)^T
        dw, rec, _, _ = gemm(g2.T.contiguous(), x2.T.contiguous().float(), r.wgrad, mod.name, "wgrad",
                          record=mod.verify, inject=mod.inject.get("wgrad"))
        if rec is not None:
            mod.record.gemms.append(rec)
            mod._strict(rec)
        db = g2.sum(0) if mod.has_bias else None
        return dx.to(gy.dtype).reshape(ctx.shape), dw.to(w.dtype), db, None


class LPLinear(torch.nn.Module):
    """A Linear whose three GEMMs run in ``recipe`` with float64 verdicts on every call when
    ``verify`` (the records of the last call are in ``record``). ``strict`` raises
    LowPrecisionKernelError (TS1042) at the GEMM whose kernel verdict fails, before its result is
    used. ``inject`` maps fwd, dgrad or wgrad to a truestar.quant kernel bug name (tests only)."""

    def __init__(self, lin: torch.nn.Linear, recipe: str, name: str = "", verify: bool = True, strict: bool = False):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.has_bias = lin.bias is not None
        self.recipe = RECIPES[recipe] if isinstance(recipe, str) else recipe
        self.name = name
        self.verify = verify
        self.strict = strict
        self.inject: Dict[str, str] = {}
        self.record = LayerRecord(name, self.recipe.name)

    def _strict(self, rec: GemmRecord):
        if self.strict and not rec.ok:
            raise LowPrecisionKernelError(rec)

    def forward(self, x):
        self.record = LayerRecord(self.name, self.recipe.name)
        return _LPLinearFn.apply(x, self.weight, self.bias, self)


def convert(model: torch.nn.Module, recipe: str = "fp8", keep: Sequence[str] = (), verify: bool = True
            ) -> torch.nn.Module:
    """Replace every nn.Linear (except those named in ``keep``) by an LPLinear in place."""
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, torch.nn.Linear) and full not in keep:
                setattr(mod, cname, LPLinear(child, recipe, full, verify))
    return model


def layers(model: torch.nn.Module) -> List[LPLinear]:
    return [m for m in model.modules() if isinstance(m, LPLinear)]


# ----------------------------------------------------------------------------- gradients against float64


@dataclasses.dataclass
class GradReport:
    recipe: str
    layers: List[LayerRecord]
    loss: float
    loss_f64: float

    def issues(self) -> List[Issue]:
        out = []
        for l in self.layers:
            for g in l.gemms:
                if not g.ok:
                    out.append(Issue("TS1042", f"{l.layer} {g.which} ({g.fmt}): kernel error {g.kernel_ratio:.3g} "
                                               "times the float32 accumulation bound"))
            if not l.consistent:
                out.append(Issue("TS1043", f"{l.layer}: the input gradient multiplies by another weight than the "
                                           f"forward ({l.weight_mismatch} entries, up to {l.weight_mismatch_max:.3g} "
                                           "of max |W|)"))
        return out

    def table(self) -> str:
        rows = [f"recipe {self.recipe}: loss {self.loss:.6g} (float64 {self.loss_f64:.6g})",
                "layer | dW err vs f64 | dX err vs f64 | fwd loss | dgrad loss | wgrad loss | worst kernel / bound | W fwd == W bwd"]
        for l in self.layers:
            by = {g.which: g for g in l.gemms}
            k = max((g.kernel_ratio for g in l.gemms), default=0.0)
            f = lambda w: f"{by[w].format_loss:.3g}" if w in by else "-"
            rows.append(f"{l.layer} | {l.dw_error:.3g} | {(l.dx_error if l.dx_error is not None else float('nan')):.3g} | "
                        f"{f('fwd')} | {f('dgrad')} | {f('wgrad')} | {k:.3g} | "
                        f"{'yes' if l.consistent else f'no ({l.weight_mismatch} entries)'}")
        return "\n".join(rows)


def grad_report(model_f64: torch.nn.Module, model_lp: torch.nn.Module, loss_fn: Callable, x: torch.Tensor,
                y: torch.Tensor) -> GradReport:
    """One forward and backward of the low precision model and of its float64 twin (same weights,
    every GEMM in float64) on one batch: per layer weight and input gradient errors against
    float64, with the GEMM verdicts of the low precision step."""
    dev = next(model_lp.parameters()).device
    lps = layers(model_lp)
    names = [l.name for l in lps]
    acts64: Dict[str, torch.Tensor] = {}
    acts_lp: Dict[str, torch.Tensor] = {}
    hooks = []
    mods64 = dict(model_f64.named_modules())
    for n in names:
        def keep(store, key):
            def h(mod, inp):
                if inp[0].requires_grad:
                    inp[0].retain_grad()
                store[key] = inp[0]
            return h
        hooks.append(mods64[n].register_forward_pre_hook(keep(acts64, n)))
    for l in lps:
        def keep_lp(store, key):
            def h(mod, inp):
                if inp[0].requires_grad:
                    inp[0].retain_grad()
                store[key] = inp[0]
            return h
        hooks.append(l.register_forward_pre_hook(keep_lp(acts_lp, l.name)))
    try:
        model_f64.zero_grad(set_to_none=True)
        model_lp.zero_grad(set_to_none=True)
        l64 = loss_fn(model_f64(x.to(F64).cpu()), y.cpu())
        l64.backward()
        llp = loss_fn(model_lp(x.to(dev)), y.to(dev))
        llp.backward()
    finally:
        for h in hooks:
            h.remove()
    recs = []
    for l in lps:
        w64 = mods64[l.name].weight.grad
        rec = l.record
        rec.dw_error = float((l.weight.grad.detach().to(F64).cpu() - w64).norm() / max(float(w64.norm()), 1e-300))
        a64, alp = acts64[l.name], acts_lp[l.name]
        if a64.grad is not None and alp.grad is not None:
            rec.dx_error = float((alp.grad.detach().to(F64).cpu() - a64.grad).norm() / max(float(a64.grad.norm()), 1e-300))
        recs.append(rec)
    return GradReport(lps[0].recipe.name if lps else "", recs, float(llp), float(l64))


def plan(make_model: Callable[[], torch.nn.Module], recipe: str, loss_fn: Callable, x: torch.Tensor,
         y: torch.Tensor, budget: float, high: str = "bf16", device: str = "cuda", max_rounds: int = 32
         ) -> Tuple[List[str], List[GradReport]]:
    """Which layers stay in ``high`` precision so that every layer's weight gradient error against
    float64 is within ``budget``: measure with every layer in ``recipe``, move the layer with the
    worst error to ``high``, measure again (an upstream layer's error flows down through dX), until
    every layer is within budget. Returns (layers kept high, the report of each round)."""
    kept: List[str] = []
    hist: List[GradReport] = []
    for _ in range(max_rounds):
        m64 = make_model().to(F64)
        mlp = make_model().to(device)
        convert(mlp, recipe, keep=())
        for l in layers(mlp):
            if l.name in kept:
                l.recipe = RECIPES[high]
        rep = grad_report(m64, mlp, loss_fn, x, y)
        hist.append(rep)
        over = [l for l in rep.layers if l.dw_error is not None and l.dw_error > budget and l.layer not in kept]
        if not over:
            return kept, hist
        kept.append(max(over, key=lambda l: l.dw_error).layer)
    return kept, hist


__all__ = ["Gemm", "Recipe", "RECIPES", "GemmRecord", "LayerRecord", "GradReport", "LPLinear", "convert", "layers",
           "gemm", "grad_report", "plan", "quantize_dequantize", "transpose_consistency", "LowPrecisionKernelError"]
