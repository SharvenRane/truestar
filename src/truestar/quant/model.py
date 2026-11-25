"""Quantize the linear layers of a model and report the two verdicts per format.

* `quantize_model(model, spec)`: a copy whose `nn.Linear` layers run the quantized kernels
  (`QuantLinear`). Every layer checks its kernel against the exact dequantized float64 reference
  on the first call of each new input shape and dtype (verdict a), the same schedule as Truestar's
  other kernels; a failure raises `QuantKernelMismatchError` (TS460 / TS461) with a diagnosis.
* `format_reference(model, qmodel)`: the float64 model that computes exactly what the format
  promises: every quantized weight dequantized, every quantized activation rounded by the
  format's rule. No kernel runs in it.
* `compare(...)`: verdict (b) from float64 references only (loss delta, perplexity ratio, KL,
  decision flips beyond near ties) next to the usual baseline (the quantized model against the
  float16 model), and verdict (a) per layer.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as Fn
from torch import nn

from . import formats as F
from . import kernels as Q
from . import verify as V


class QuantLinear(nn.Module):
    """nn.Linear on a quantized kernel. Activations run in the model's float16 or bfloat16."""

    def __init__(self, lin: nn.Linear, spec, name: str = "", verify: bool = True):
        super().__init__()
        self.qw = F.quantize_weight(lin.weight.detach(), spec)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(),
                                                               requires_grad=False)
        self.name = name
        self.verify = verify
        self.raise_on_error = True
        self.inject: Optional[str] = None       # tests only: compile a deliberately wrong kernel
        self.checks: List[V.KernelCheck] = []
        self._seen = set()

    @property
    def in_features(self) -> int:
        return self.qw.shape[1]

    @property
    def out_features(self) -> int:
        return self.qw.shape[0]

    def _apply(self, fn, *args, **kwargs):  # keep the packed weight on the module's device
        super()._apply(fn, *args, **kwargs)
        probe = fn(torch.zeros(1))
        if probe.device != self.qw.data.device:
            for k in ("data", "scale", "tensor_scale"):
                t = getattr(self.qw, k)
                if t is not None:
                    setattr(self.qw, k, t.to(probe.device))
            self.qw._deq_cache = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, K = self.qw.shape
        shp = x.shape
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y, act = Q.quantized_matmul(x2, self.qw, inject=self.inject, return_act=True)
        key = (x2.shape[0], x2.dtype)
        if self.verify and key not in self._seen:
            self._seen.add(key)
            chk = V.check_matmul(x2, self.qw, y, act, self.name)
            if not chk.matmul_ok:
                inj = self.inject
                chk.diagnosis = V.diagnose(
                    self.qw, lambda e: Q.quantized_matmul(e.to(x2.dtype), self.qw, inject=inj),
                    x2.dtype)
            self.checks.append(chk)
            if not chk.ok and self.raise_on_error:
                raise V.QuantKernelMismatchError(chk)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shp[:-1], N)


class RefQuantLinear(nn.Module):
    """The format's promise in float64: dequantized weight, activations rounded exactly."""

    def __init__(self, qw: F.QuantWeight, bias: Optional[torch.Tensor]):
        super().__init__()
        self.fmt = qw.fmt
        self.register_buffer("w", V._deq(qw).clone())
        self.register_buffer("b", None if bias is None else bias.detach().to(F.F64).clone())

    def forward(self, x):
        x = x.to(F.F64)
        if self.fmt.activations:
            x = V.quantize_act_ref(x, self.fmt)
        y = x @ self.w.T
        return y if self.b is None else y + self.b


def _linears(model: nn.Module, skip: Iterable[str]) -> List[str]:
    skip = tuple(skip)
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not any(n == s or n.startswith(s + ".") for s in skip)]


def _set(model: nn.Module, name: str, mod: nn.Module):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, mod)


def quantize_model(model: nn.Module, spec, skip: Sequence[str] = ("lm_head",),
                   verify: bool = True) -> nn.Module:
    """A copy of model with every nn.Linear (except `skip`) on the quantized kernel for spec.
    The model must already be on CUDA in float16 or bfloat16."""
    q = copy.deepcopy(model)
    for n in _linears(model, skip):
        _set(q, n, QuantLinear(model.get_submodule(n), spec, n, verify))
    return q


def format_reference(model: nn.Module, qmodel: nn.Module) -> nn.Module:
    """float64 copy of model in which each layer quantized in qmodel computes the format's exact
    answer (same QuantWeight objects, so the same codes and scales the kernels read)."""
    r = copy.deepcopy(model).to(F.F64)
    for n, m in qmodel.named_modules():
        if isinstance(m, QuantLinear):
            _set(r, n, RefQuantLinear(m.qw, m.bias).to(next(r.parameters()).device))
    return r


def quant_layers(qmodel: nn.Module) -> List[QuantLinear]:
    return [m for m in qmodel.modules() if isinstance(m, QuantLinear)]


# ----------------------------------------------------------------------------- comparison


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(Fn.cross_entropy(logits.reshape(-1, logits.shape[-1]).to(F.F64),
                                  targets.reshape(-1)))


@dataclass
class Decisions:
    total: int
    flips: int
    near_ties: int

    @property
    def real(self) -> int:
        return self.flips - self.near_ties


def decisions(ref: torch.Tensor, other: torch.Tensor, tie: torch.Tensor) -> Decisions:
    """Top 1 changes of other against ref per position; a flip is a near tie when ref's top 1
    margin is within `tie` (per position)."""
    r = ref.reshape(-1, ref.shape[-1]).to(F.F64)
    o = other.reshape(-1, other.shape[-1]).to(F.F64)
    top2 = r.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    flip = r.argmax(-1) != o.argmax(-1)
    near = flip & (margin <= tie.reshape(-1))
    return Decisions(int(r.shape[0]), int(flip.sum()), int(near.sum()))


@dataclass
class FormatResult:
    spec: str
    weight_bytes: int
    # verdict (b): float64 format reference against the float64 float model, no kernel involved
    ce_float: float
    ce_format: float
    kl_format: float
    format_flips: Decisions
    # verdict (a): per layer kernel checks, and the kernel model against the format reference
    kernel_checks: List[V.KernelCheck]
    kernel_vs_format_flips: Decisions
    kernel_vs_format_rel: float
    # the usual baseline: quantized kernel model against the float16 model
    ce_kernel: float
    baseline_rel: float
    baseline_flips: Decisions
    layer_format_loss: Dict[str, float] = field(default_factory=dict)

    @property
    def kernel_ok(self) -> bool:
        return all(c.ok for c in self.kernel_checks)

    @property
    def codes(self) -> List[str]:
        out = sorted({c.code for c in self.kernel_checks if c.code})
        out.append("TS462" if self.format_flips.real else "TS463")
        return out

    def as_dict(self) -> Dict:
        d = lambda x: {"total": x.total, "flips": x.flips, "near_ties": x.near_ties}
        return {
            "format": self.spec, "weight_bytes": self.weight_bytes, "codes": self.codes,
            "format_verdict": {"ce_float": self.ce_float, "ce_format": self.ce_format,
                               "delta_ce": self.ce_format - self.ce_float,
                               "ppl_ratio": math.exp(self.ce_format - self.ce_float),
                               "kl": self.kl_format, "flips": d(self.format_flips),
                               "layer_rel_l2": self.layer_format_loss},
            "kernel_verdict": {"ok": self.kernel_ok,
                               "worst_ratio": max((c.ratio for c in self.kernel_checks), default=0.0),
                               "failing": [c.as_dict() for c in self.kernel_checks if not c.ok],
                               "layers_checked": len(self.kernel_checks),
                               "vs_format_reference_rel": self.kernel_vs_format_rel,
                               "vs_format_reference_flips": d(self.kernel_vs_format_flips)},
            "baseline_vs_fp16": {"ce_kernel": self.ce_kernel, "delta_ce_vs_fp16": None,
                                 "rel": self.baseline_rel, "flips": d(self.baseline_flips)},
        }


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(F.F64)
    b = b.to(F.F64)
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


@torch.no_grad()
def compare(model16: nn.Module, ids: torch.Tensor, specs: Sequence[str],
            skip: Sequence[str] = ("lm_head",), inject: Optional[Dict[str, str]] = None,
            layer_inputs: bool = True) -> Dict:
    """Both verdicts for each format on a causal LM `model16` (CUDA, float16 or bfloat16) and
    token ids [B, T]: logits at positions 0..T-2 score tokens 1..T-1.

    `inject` maps a layer name to a kernel bug (tests and the study only)."""
    targets = ids[:, 1:]
    f64 = copy.deepcopy(model16).to(F.F64)
    L64 = f64(ids)[:, :-1]
    L16 = model16(ids)[:, :-1]
    ce_float = _ce(L64, targets)
    noise = (L16.to(F.F64) - L64).abs().amax(dim=-1)            # float16 compute noise per position
    tie = 4 * noise
    p64 = torch.log_softmax(L64, -1)
    out = {"ce_float64": ce_float, "ce_fp16": _ce(L16, targets),
           "fp16_vs_float64_rel": _rel(L16, L64), "positions": int(targets.numel()), "formats": []}
    for spec in specs:
        qm = quantize_model(model16, spec, skip)
        for m in quant_layers(qm):
            m.raise_on_error = False
            if inject and m.name in inject:
                m.inject = inject[m.name]
        ref = format_reference(model16, qm)
        Lf = ref(ids)[:, :-1]
        Lk = qm(ids)[:, :-1]
        pf = torch.log_softmax(Lf, -1)
        kl = float((p64.exp() * (p64 - pf)).sum(-1).mean())
        lfl = {}
        if layer_inputs:
            lfl = _layer_format_loss(f64, ref, ids)
        res = FormatResult(
            spec=spec, weight_bytes=sum(m.qw.nbytes() for m in quant_layers(qm)),
            ce_float=ce_float, ce_format=_ce(Lf, targets), kl_format=kl,
            format_flips=decisions(L64, Lf, tie),
            kernel_checks=[c for m in quant_layers(qm) for c in m.checks],
            kernel_vs_format_flips=decisions(Lf, Lk, tie), kernel_vs_format_rel=_rel(Lk, Lf),
            ce_kernel=_ce(Lk, targets), baseline_rel=_rel(Lk, L16),
            baseline_flips=decisions(L16, Lk, tie), layer_format_loss=lfl)
        d = res.as_dict()
        d["baseline_vs_fp16"]["delta_ce_vs_fp16"] = res.ce_kernel - out["ce_fp16"]
        d["_result"] = res
        out["formats"].append(d)
        del qm, ref
        torch.cuda.empty_cache()
    return out


def _layer_format_loss(f64: nn.Module, ref: nn.Module, ids: torch.Tensor) -> Dict[str, float]:
    """Per layer: relative L2 change of the layer's output caused by the format alone, on the
    inputs the float64 float model gives that layer (so errors of earlier layers do not mix in)."""
    caps = {}
    hooks = []
    for n, m in f64.named_modules():
        if isinstance(m, nn.Linear) and isinstance(ref.get_submodule(n), RefQuantLinear):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, outp, n=n: caps.__setitem__(n, (inp[0], outp))))
    try:
        f64(ids)
    finally:
        for h in hooks:
            h.remove()
    res = {}
    for n, (x, y) in caps.items():
        yq = ref.get_submodule(n)(x)
        res[n] = float((yq - y).norm() / y.norm().clamp(min=1e-300))
    return res
