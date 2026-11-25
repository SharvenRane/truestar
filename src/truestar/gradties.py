"""Near tie mask flips: tell a parent replay that is off only because ReLU masks flipped apart from a
wrong backward.

A whole model (parent) replay in float16 compounds every child's rounding. Where a ReLU input is a
near tie (-5.3e-4 in float64, +1.2e-4 in float16), the backend, the correctly rounded run and float64
can each take a different branch, and the parent's gradients then differ by far more than any
threshold although every kernel is right (docs/GRAD_CHECK.md, Limits). The owner's rule
(2026-11-24): such a parent is not DIVERGENT when every leaf replay is clean and the deviation traces
to activation mask flips at near ties; it is SUSPECT with a note naming the layer and the number of
flipped elements. A real backward fault must stay DIVERGENT.

"Traces to" is checked, not assumed:

1. The parent's forward runs on the backend and in float64 with the reference inputs, recording every
   ReLU (aten.relu, aten.relu_) with the module that called it. Both runs must call the same ReLUs in
   the same order with the same shapes.
2. Every element whose mask differs between the backend and float64 must be a near tie: its float64
   input is within TIE_BAND (twice the precision floor) of zero, relative to the RMS of that ReLU's
   input. A flip far from zero is a forward bug, not rounding.
3. The correctly rounded run (float64 math, every op rounded to the dtype) is replayed again with
   the backend's masks forced into its ReLUs. If the backend's gradients land on that run within the
   usual thresholds, every bit of the deviation is explained by the flips; a wrong backward kernel
   does not land there.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from .capture import ROOT, map_tensors
from .compare import DTYPE_FLOOR, OK

RELU_OPS = ("aten.relu", "aten.relu_")
TIE_FACTOR = 2.0  # near tie: |x64| <= TIE_FACTOR * precision floor * RMS of that ReLU's input
NOTE = "near tie mask flip"


@dataclasses.dataclass
class _Relu:
    name: str  # module path and call count, e.g. layer4.1.relu#1
    x: torch.Tensor  # input (CPU copy)
    mask: torch.Tensor  # output > 0 (CPU bool)


class _Names:
    """Forward hooks that keep the path of the module running now (relative to the model)."""

    def __init__(self, module: nn.Module, prefix: str):
        self.stack: List[str] = []
        self.handles = []
        self.counts: Dict[str, int] = {}
        for sub_name, sub in module.named_modules():
            full = ".".join(p for p in (prefix, sub_name) if p and p != ROOT) or ROOT
            self.handles.append(sub.register_forward_pre_hook(self._pre(full)))
            self.handles.append(sub.register_forward_hook(self._post()))

    def _pre(self, name):
        def hook(mod, args):
            self.stack.append(name)
        return hook

    def _post(self):
        def hook(mod, args, out):
            if self.stack:
                self.stack.pop()
        return hook

    def current(self) -> str:
        name = self.stack[-1] if self.stack else "<model>"
        return name or "<model>"

    def close(self):
        for h in self.handles:
            h.remove()


class _Record(TorchDispatchMode):
    def __init__(self, names: _Names):
        super().__init__()
        self.names = names
        self.relus: List[_Relu] = []
        self.calls: Dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = str(func.overloadpacket)
        if packet not in RELU_OPS:
            return func(*args, **kwargs)
        x = args[0].detach().clone()
        out = func(*args, **kwargs)
        mod = self.names.current()
        k = self.calls.get(mod, 0)
        self.calls[mod] = k + 1
        self.relus.append(_Relu(f"{mod}#{k}", x.cpu(), (out > 0).cpu()))
        return out


class ForceMasks(TorchDispatchMode):
    """Give each ReLU, in call order, the mask the backend took. Where the forced mask is on but
    the input is not positive (a near tie the other way), the output is |x| plus a tiny amount, so
    the value stays a near tie and autograd's saved result keeps the forced mask for backward."""

    def __init__(self, masks: List[torch.Tensor]):
        super().__init__()
        self.masks = masks
        self.i = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = str(func.overloadpacket)
        if packet not in RELU_OPS or self.i >= len(self.masks):
            return func(*args, **kwargs)
        x = args[0]
        m = self.masks[self.i].to(x.device)
        self.i += 1
        tiny = torch.finfo(torch.float16).tiny
        pos = torch.where(x > 0, x, x.abs() + tiny)
        val = torch.where(m, pos, torch.zeros_like(x))
        if packet == "aten.relu_":
            x.copy_(val)
            return x
        return val


def _trace(module: nn.Module, prefix: str, args, kwargs) -> List[_Relu]:
    names = _Names(module, prefix)
    try:
        with torch.no_grad(), _Record(names) as rec:
            module(*args, **kwargs)
    finally:
        names.close()
    return rec.relus


@dataclasses.dataclass
class TieExplanation:
    flips: List[Tuple[str, int]]  # (ReLU call, flipped elements), in call order
    total: int
    max_tie: float  # largest |x64| / RMS among flipped elements
    band: float
    rounded_err: float  # worst distance of the backend gradient to the mask forced rounded run
    rounded_threshold: float

    def note(self, backend: str, dtype: torch.dtype) -> str:
        dt = str(dtype).replace("torch.", "")
        top = sorted(self.flips, key=lambda f: -f[1])  # most flips first
        where = ", ".join(f"{n} ({c})" for n, c in top[:4])
        more = f" and {len(self.flips) - 4} more" if len(self.flips) > 4 else ""
        return (f"{NOTE}: {self.total} ReLU element(s) take another branch on {backend} than in "
                f"float64 at near ties (|x| <= {self.max_tie:.1e} of the input RMS, band "
                f"{self.band:.0e}), in {where}{more}; a correctly rounded {dt} run with the "
                f"backend's masks matches the backend gradient (error {self.rounded_err:.2e} vs "
                f"threshold {self.rounded_threshold:.1e}), so the gap is {dt} rounding, not a "
                f"backward bug")

    def as_dict(self) -> Dict[str, Any]:
        return {"flips": [{"relu": n, "elements": c} for n, c in self.flips], "total": self.total,
                "max_tie": self.max_tie, "band": self.band, "rounded_err": self.rounded_err,
                "rounded_threshold": self.rounded_threshold}


def explain(dev_module: nn.Module, ref_module: nn.Module, emu_module: nn.Module, call, upstream,
            device: str, dtype: torch.dtype, tf32: Tuple[bool, bool], got, ref,
            tensors) -> Tuple[Optional[TieExplanation], str]:
    """(explanation, reason). The explanation is None, with the reason, unless all three checks of
    the module docstring hold for this parent call."""
    from .check import _cast_inputs
    from .gradcheck import _replay_emulated, _rounded_run_diff

    if dtype == torch.float64:
        return None, "float64 model"
    args_b = _cast_inputs(call.args, device, dtype)
    kwargs_b = _cast_inputs(call.kwargs, device, dtype)
    trace_b = _trace(dev_module, call.name, args_b, kwargs_b)
    args64 = _cast_inputs(call.args, "cpu", torch.float64)
    kwargs64 = _cast_inputs(call.kwargs, "cpu", torch.float64)
    trace_64 = _trace(ref_module, call.name, args64, kwargs64)
    if not trace_b:
        return None, "no ReLU in this call"
    if [(r.name, tuple(r.mask.shape)) for r in trace_b] != \
            [(r.name, tuple(r.mask.shape)) for r in trace_64]:
        return None, "the backend and float64 ran different ReLUs"
    band = TIE_FACTOR * DTYPE_FLOOR.get(dtype, 1e-4)
    flips: List[Tuple[str, int]] = []
    max_tie = 0.0
    for rb, r64 in zip(trace_b, trace_64):
        diff = rb.mask != r64.mask
        n = int(diff.sum())
        if not n:
            continue
        x = r64.x.double()
        rms = float(x.pow(2).mean().sqrt())
        tie = float(x[diff].abs().max()) / rms if rms > 0 else float("inf")
        max_tie = max(max_tie, tie)
        if tie > band:
            return None, (f"{rb.name}: a flipped element is {tie:.1e} of the input RMS from zero, "
                          f"not a near tie (band {band:.0e})")
        flips.append((rb.name, n))
    if not flips:
        return None, "no ReLU mask differs from float64"
    ups = upstream
    forced = _replay_emulated(emu_module, call, ups, dtype, tf32,
                              extra_mode=ForceMasks([r.mask for r in trace_b]))
    if forced is None or forced.labels != got.labels:
        return None, "the mask forced rounded run computed other gradients"
    worst_err, worst_thr = 0.0, 0.0
    for i, t in enumerate(tensors):
        if t.status == OK:
            continue
        d = _rounded_run_diff(got.grads[i], forced.grads[i], ref.grads[i], t.diff.rel_err, dtype)
        if d.status(t.threshold) != OK:
            return None, (f"{t.label}: {d.rel_err:.2e} from the mask forced rounded run "
                          f"(threshold {t.threshold:.1e}), not explained by the flips")
        if d.rel_err / t.threshold >= (worst_err / worst_thr if worst_thr else -1):
            worst_err, worst_thr = d.rel_err, t.threshold
    return TieExplanation(flips, sum(c for _, c in flips), max_tie, band, worst_err,
                          worst_thr), ""
