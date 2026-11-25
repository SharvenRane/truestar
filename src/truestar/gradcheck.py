"""`truestar check --backward`: find the module whose own backward pass computes wrong gradients.

Low precision training can diverge late with no NaN because a trusted kernel's backward is subtly
wrong (docs/PAIN_RESEARCH_2.md, N6). The forward check cannot see that. This is its backward
counterpart.

Method
------
1. Reference: the model in float64 on CPU, in eval mode, runs forward and backward through a fixed
   loss (default: mean squared error to a fixed random target, or the caller's `loss_fn`). Every
   module call is recorded with its inputs copied before forward, and a tensor hook on each output
   records the upstream gradient that reached it.
2. Isolated backward replay: for every module call that is replayed (leaves, and parents that run
   ops of their own once everything inside them matched, the same rule as the forward check), the
   module alone runs forward and backward with the reference inputs and the reference upstream
   gradients, giving the gradient of each floating input and of each trainable parameter of that
   call. This runs in float64 on CPU (the answer), in the model dtype on CPU (the noise floor), and
   in the model dtype on each backend. The error a module shows is therefore its own, not inherited
   from the modules after it.
3. Thresholds per gradient tensor, calibrated like the forward check: the precision's own floor,
   never below 10x the error the same tensor shows on CPU in the model dtype. Reported per module:
   relative L2 error (worst tensor against its own threshold), cosine (direction) and the norm
   ratio backend over reference. The module's forward output from the same replay is compared too,
   so a backward only bug (forward OK) is told apart from a forward bug that also spoils gradients.
4. Blame: the over threshold module that runs first in backward (the last in forward order), since
   corruption flows from there to everything before it.
5. Op drilldown: the blamed module's backward alone is traced at the ATen op level under a
   TorchDispatchMode on the reference and on the backend; ops are aligned and the first op whose own
   error crosses the threshold, confirmed by running that op alone with reference inputs, is named
   (for example aten.mm.default in AddmmBackward0).
6. Optional `steps`: a tiny SGD training loop from the same weights on the reference, on CPU in the
   model dtype (control) and on the backend, recording the total and per module gradient norms each
   step. A backend whose gradient norm ratio drifts away from 1 over the steps, faster than the CPU
   control does, is flagged (TS183): the reported failure mode is slow accumulation.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .backends import Backend, resolve_backends
from .capture import ROOT, Call, Recorder, flatten_tensors, map_tensors
from .check import (ReferenceFailed, _cast_inputs, _model_dtype, _module_by_name, _needs_replay, _sync,
                    _uses_tf32)
from .codes import Issue
from .compare import (
    DIVERGENT,
    DIVERGENT_FACTOR,
    DTYPE_FLOOR,
    OK,
    SUSPECT,
    Diff,
    diff_lists as _diff_lists,
    diff_tensors as _diff_tensors,
    threshold_for,
    worst,
)
from .opdrill import MAX_OPS, OpResult, _align, _comparable, _isolate, _leaves, _make_tracer
from .trainmode import active as active_trainmode, train_outputs


# Gradients keep the whole tensor scoring (docs/GRAD_CHECK.md, Limits): an input or weight
# gradient is sparse by construction (max pool routes, ReLU masks, near ties the rounded run
# explains), so its per element and per slice errors are not calibrated yet. The forward check
# (truestar check) uses both measures.
def diff_tensors(actual, reference):
    return _diff_tensors(actual, reference, local=False)


def diff_lists(actual, reference):
    return _diff_lists(actual, reference, local=False)


LossFn = Callable[[List[torch.Tensor], List[torch.Tensor]], torch.Tensor]


# ----------------------------------------------------------------------------- results


@dataclass
class GradTensor:
    label: str  # "input 0" or "param weight"
    diff: Diff
    threshold: float
    status: str
    norm_ratio: Optional[float]  # ||backend|| / ||reference||
    cosine: Optional[float]

    @property
    def ratio(self) -> float:
        return self.diff.rel_err / self.threshold if self.threshold else float("inf")

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "status": self.status, "rel_err": self.diff.rel_err,
                "threshold": self.threshold, "norm_ratio": self.norm_ratio, "cosine": self.cosine,
                "nonfinite_mismatch": self.diff.nonfinite_mismatch,
                "shape_mismatch": self.diff.shape_mismatch, "note": self.diff.note}


@dataclass
class GradLayer:
    key: str
    name: str
    type_name: str
    is_leaf: bool
    start: int
    status: str = OK
    replayed: bool = True
    tensors: List[GradTensor] = field(default_factory=list)
    forward: Optional[Diff] = None  # the module's own forward output in the same replay
    forward_threshold: Optional[float] = None
    forward_status: str = OK
    norm_ratio: Optional[float] = None  # over every gradient of this call together
    cosine: Optional[float] = None
    error: str = ""
    note: str = ""
    # Set when a parent's deviation is explained by near tie ReLU mask flips (gradties.py): the
    # explanation, and the tensor statuses before it was applied.
    tie: Optional[Any] = None
    tie_saved: List[str] = field(default_factory=list)

    @property
    def worst(self) -> Optional[GradTensor]:
        if not self.tensors:
            return None
        rank = {OK: 0, SUSPECT: 1, DIVERGENT: 2}
        # Prefer gradients that are not zero in exact math: they say more about what broke.
        return max(self.tensors, key=lambda t: (rank[t.status], not t.diff.note, t.ratio))

    @property
    def rel_err(self) -> Optional[float]:
        w = self.worst
        return None if w is None else w.diff.rel_err

    @property
    def threshold(self) -> Optional[float]:
        w = self.worst
        return None if w is None else w.threshold

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "name": self.name, "type": self.type_name, "leaf": self.is_leaf,
            "status": self.status, "replayed": self.replayed, "rel_err": self.rel_err,
            "threshold": self.threshold, "norm_ratio": self.norm_ratio, "cosine": self.cosine,
            "forward_status": self.forward_status,
            "forward_rel_err": None if self.forward is None else self.forward.rel_err,
            "forward_threshold": self.forward_threshold,
            "tensors": [t.as_dict() for t in self.tensors], "error": self.error, "note": self.note,
            "near_tie_mask_flip": None if self.tie is None else self.tie.as_dict(),
        }


@dataclass
class StepTrend:
    """Gradient norms over a tiny training loop: reference, CPU control and backend."""

    steps: int
    lr: float
    total_ratio: List[Optional[float]]  # backend / reference, per step
    # control name ("cpu", "emulated") -> its total norm / reference norm per step
    control_ratio: Dict[str, List[Optional[float]]]
    modules: Dict[str, List[Optional[float]]]  # module name -> backend / reference per step
    threshold: float
    flagged: List[Dict[str, Any]] = field(default_factory=list)  # {module, first, last, control}
    losses: List[float] = field(default_factory=list)  # backend loss per step

    def as_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "lr": self.lr, "threshold": self.threshold,
                "total_ratio": self.total_ratio, "control_ratio": self.control_ratio,
                "modules": self.modules, "flagged": self.flagged, "losses": self.losses}


@dataclass
class GradBackendResult:
    backend: Backend
    status: str = OK
    layers: List[GradLayer] = field(default_factory=list)  # forward start order
    first_divergent: Optional[GradLayer] = None
    first_divergent_op: Optional[OpResult] = None
    op_node: Optional[str] = None  # autograd node that ran the blamed op, e.g. AddmmBackward0
    trend: Optional[StepTrend] = None
    repro_path: Optional[str] = None
    op_repro_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    error: str = ""
    issues: List[Issue] = field(default_factory=list)
    blame_source: Optional[Dict[str, Any]] = None
    injected: List[str] = field(default_factory=list)  # truestar fault injectors active during it

    @property
    def codes(self) -> List[str]:
        return _unique(i.code for i in self.issues)

    def add(self, code: str, message: str) -> None:
        self.issues.append(Issue(code, message, self.backend.name))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.as_dict(), "status": self.status,
            "injected_faults": list(self.injected),
            "first_divergent": None if self.first_divergent is None else self.first_divergent.key,
            "first_divergent_op": None if self.first_divergent_op is None
            else dict(self.first_divergent_op.as_dict(), node=self.op_node),
            "trend": None if self.trend is None else self.trend.as_dict(),
            "repro_path": self.repro_path, "op_repro_path": self.op_repro_path,
            "notes": list(self.notes), "error": self.error, "codes": self.codes,
            "issues": [i.as_dict() for i in self.issues], "blame_source": self.blame_source,
            "layers": [l.as_dict() for l in self.layers],
        }


@dataclass
class GradReport:
    model_name: str
    dtype: torch.dtype
    reference: str
    loss: str
    results: List[GradBackendResult]
    skipped: List[Tuple[str, str]]
    notes: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)

    @property
    def all_issues(self) -> List[Issue]:
        return list(self.issues) + [i for r in self.results for i in r.issues]

    @property
    def codes(self) -> List[str]:
        return _unique(i.code for i in self.all_issues)

    @property
    def status(self) -> str:
        return worst(*[r.status for r in self.results]) if self.results else OK

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": "backward", "model": self.model_name,
            "dtype": str(self.dtype).replace("torch.", ""), "reference": self.reference,
            "loss": self.loss, "status": self.status,
            "skipped": [{"backend": n, "reason": r} for n, r in self.skipped],
            "notes": list(self.notes), "codes": self.codes,
            "issues": [i.as_dict() for i in self.issues],
            "results": [r.as_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)

    def to_text(self, top: int = 5) -> str:
        return render_text(self, top)

    def to_agent(self, model_file: Optional[str] = None) -> Dict[str, Any]:
        return report_to_agent(self, model_file)


# ----------------------------------------------------------------------------- helpers


def _unique(items) -> List[str]:
    out: List[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _cpu64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", dtype=torch.float64 if t.is_floating_point() else t.dtype,
                         copy=True)


def default_loss(outputs: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
    """Sum over floating outputs of the mean squared error to a fixed target (computed in at least
    float32, as mixed precision training computes its loss)."""
    total = None
    for o, t in zip(outputs, targets):
        if not o.is_floating_point():
            continue
        acc = torch.float64 if o.dtype == torch.float64 else torch.float32
        term = 0.5 * ((o.to(acc) - t.to(device=o.device, dtype=acc)) ** 2).mean()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("the model has no floating output to put a loss on")
    return total


def _targets(outputs: List[torch.Tensor], seed: int) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for o in outputs:
        o = o.detach().cpu().double()
        if not o.is_floating_point():
            out.append(o)
            continue
        scale = float(o.std()) if o.numel() > 1 else 1.0
        scale = scale if math.isfinite(scale) and scale > 0 else 1.0
        out.append(torch.randn(o.shape, generator=g, dtype=torch.float64) * scale)
    return out


class _GradRecorder(Recorder):
    """Recorder that also hooks every module output to capture the upstream gradient."""

    def __init__(self, model: nn.Module):
        super().__init__(model, keep_inputs=True, count_ops=True)
        self.upstream: Dict[str, List[Optional[torch.Tensor]]] = {}

    def _post(self, name: str):
        base = super()._post(name)

        def hook(module, args, kwargs, output):
            base(module, args, kwargs, output)
            key = self.calls[-1].key
            outs = flatten_tensors(output)
            slots: List[Optional[torch.Tensor]] = [None] * len(outs)
            self.upstream[key] = slots
            for i, t in enumerate(outs):
                if t.requires_grad:
                    t.register_hook(_store(slots, i))
            return None

        return hook


def _store(slots, i):
    def fn(g):
        if g is not None:
            slots[i] = g.detach().to("cpu", dtype=torch.float64, copy=True)
        return None

    return fn


def _norm_cos(a: List[torch.Tensor], b: List[torch.Tensor]):
    """(||a|| / ||b||, cosine(a, b)) over concatenated tensors, or Nones when undefined."""
    if not a or len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        return None, None
    x = torch.cat([t.detach().cpu().double().reshape(-1) for t in a])
    y = torch.cat([t.detach().cpu().double().reshape(-1) for t in b])
    if not (bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())):
        return None, None
    nx, ny = float(x.norm()), float(y.norm())
    if ny == 0.0:
        return (None if nx == 0.0 else float("inf")), None
    ratio = nx / ny
    cos = float(x @ y) / (nx * ny) if nx > 0 else None
    return ratio, cos


# ----------------------------------------------------------------------------- backward replay


@dataclass
class _Replay:
    labels: List[str]
    grads: List[torch.Tensor]  # CPU copies, in label order
    outputs: List[torch.Tensor]  # forward outputs, CPU copies


def _replay_backward(module: nn.Module, call: Call, upstream: List[Optional[torch.Tensor]],
                     device: str, dtype: torch.dtype) -> Optional[_Replay]:
    """Run one module's forward and backward alone with reference inputs and upstream gradients.

    Returns None when nothing in the call receives a gradient.
    """
    args = _cast_inputs(call.args, device, dtype)
    kwargs = _cast_inputs(call.kwargs, device, dtype)
    leaves: List[torch.Tensor] = []

    def prep(t: torch.Tensor) -> torch.Tensor:
        if not t.is_floating_point():
            return t
        leaf = t.detach().requires_grad_(True)
        leaves.append(leaf)
        return leaf.clone()  # modules with in place ops must not write into a leaf

    args, kwargs = map_tensors((args, kwargs), prep)
    params = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    with torch.enable_grad(), _replaying(call):
        out = module(*args, **kwargs)
        outs = flatten_tensors(out)
        if len(outs) != len(upstream):
            raise RuntimeError(f"{len(outs)} outputs on replay vs {len(upstream)} in the reference")
        pairs = [(o, g) for o, g in zip(outs, upstream)
                 if g is not None and o.requires_grad and o.is_floating_point()]
        wrt = leaves + [p for _, p in params]
        if not pairs or not wrt:
            return None
        grads = torch.autograd.grad(
            [o for o, _ in pairs], wrt,
            [g.to(device=o.device, dtype=o.dtype) for o, g in pairs], allow_unused=True)
    labels = [f"input {i}" for i in range(len(leaves))] + [f"param {n}" for n, _ in params]
    gl = [(torch.zeros_like(w) if g is None else g).detach().cpu() for g, w in zip(grads, wrt)]
    # training mode: the state after the call (running statistics) is compared with the forward
    return _Replay(labels, gl, [o.detach().cpu() for o in outs] + train_outputs(module))


def _replaying(call: Call):
    """Inside a training mode check: replay ``call`` from the reference run's state and keys."""
    tm = active_trainmode()
    return tm.replay(call.key, call.name) if tm is not None else contextlib.nullcontext()


def _replay_cpu64(module: nn.Module, call: Call, upstream) -> Optional[_Replay]:
    return _replay_backward(module, call, upstream, "cpu", torch.float64)


_CONV_OPS = {"aten.convolution", "aten.convolution_backward", "aten._convolution",
             "aten.conv2d", "aten.conv1d", "aten.conv3d"}
_MM_OPS = {"aten.mm", "aten.addmm", "aten.bmm", "aten.baddbmm", "aten.matmul", "aten.addbmm",
           "aten.linear"}


def _tf32(t: torch.Tensor) -> torch.Tensor:
    """Round a float64 tensor to TF32 (10 mantissa bits, round to nearest) and back."""
    x = t.to(torch.float32).contiguous()
    bits = x.view(torch.int32)
    finite = torch.isfinite(x)
    r = ((bits + 0x1000) & ~0x1FFF).view(torch.float32)
    return torch.where(finite, r, x).to(torch.float64).reshape(t.shape)


def _rounding_mode(dtype: torch.dtype, tf32_conv: bool = False, tf32_mm: bool = False):
    """Float64 math with every op's floating output rounded to `dtype`: what a correct kernel in
    `dtype` that rounds once per op computes. Views and other aliasing results are left alone
    (their base is already rounded); in place results are rounded in place. With TF32 enabled,
    the floating inputs of convolutions (and matmuls) are rounded to TF32 first, as the tensor
    cores do."""
    from torch.utils._python_dispatch import TorchDispatchMode

    def rnd(t):
        if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
            return t.to(dtype).to(torch.float64)
        return t

    def tf(t):
        if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
            return _tf32(t)
        return t

    class RoundingMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            packet = str(func.overloadpacket)
            if (tf32_conv and packet in _CONV_OPS) or (tf32_mm and packet in _MM_OPS):
                args, kwargs = map_tensors((args, kwargs), tf)
            out = func(*args, **kwargs)
            schema = func._schema
            if any(r.alias_info is not None for r in schema.returns):
                if schema.is_mutable:
                    for t in _leaves(out):
                        if t.dtype == torch.float64:
                            t.copy_(t.to(dtype).to(torch.float64))
                return out
            if isinstance(out, torch.Tensor):
                return rnd(out)
            if isinstance(out, (tuple, list)):
                return type(out)(rnd(t) for t in out)
            return out

    return RoundingMode()


def _replay_emulated(module64: nn.Module, call: Call, upstream, dtype: torch.dtype,
                     tf32: Tuple[bool, bool] = (False, False),
                     extra_mode=None) -> Optional[_Replay]:
    """The module's backward in float64 with inputs, upstream gradients and every op result
    rounded to `dtype` (module64 holds parameters already rounded to `dtype`). `extra_mode`, a
    dispatch mode, runs on top of the rounding (gradties.ForceMasks forces ReLU masks).

    The gap between this and the exact reference is error any correct `dtype` implementation
    shows: ReLU masks and max pool routes that flip because inputs round, cancellation in sums.
    It is the noise floor used where the CPU in the model dtype cannot be one (the CPU backend
    itself) and a second floor everywhere else.
    """
    def rnd(t):
        return t.to(dtype).to(torch.float64) if t.is_floating_point() else t

    c = dataclasses.replace(call, args=map_tensors(call.args, rnd),
                            kwargs=map_tensors(call.kwargs, rnd))
    ups = [None if g is None else rnd(g) for g in upstream]
    with _rounding_mode(dtype, *tf32), (extra_mode or contextlib.nullcontext()):
        return _replay_backward(module64, c, ups, "cpu", torch.float64)


def _tf32_flags(b: Backend, dtype: torch.dtype) -> Tuple[bool, bool]:
    """(TF32 convolutions, TF32 matmuls) as they apply to this backend and dtype."""
    if dtype != torch.float32 or b.name not in ("cuda", "rocm") or not b.tf32:
        return (False, False)
    return (bool(torch.backends.cudnn.allow_tf32), bool(torch.backends.cuda.matmul.allow_tf32))


def _floor(base_floor: float, cpu: Optional[float], emulated: Optional[float],
           zero_ref: bool = False, uncapped: bool = False) -> Optional[float]:
    """Noise floor for one tensor: the larger of the CPU run in the model dtype and the emulated
    rounding run. A value far above the precision floor is dropped, the same cap the forward check
    applies: a CPU value that large means the CPU kernel is itself wrong, and an emulated value
    that large means the module's math loses its digits in this precision (cancellation, as in a
    one pass variance). That precision loss is the divergence the check exists to report, so it
    must never raise the threshold past it. The one exception is a gradient that is zero in exact
    math (`zero_ref`): its value is only rounding, so the emulated rounding error is its honest
    floor at any size."""
    cap = math.inf if uncapped else base_floor * DIVERGENT_FACTOR
    ok = [v for v in (cpu,) if v is not None and v == v and v <= cap]
    if emulated is not None and emulated == emulated and (zero_ref or emulated <= cap):
        ok.append(emulated)
    return max(ok) if ok else None


def _zero_refs(ref: "_Replay") -> List[bool]:
    """Per gradient of one call: True where the reference is zero up to rounding (see _grad_diff)."""
    scale = _scale(ref)
    return [scale > 0 and r.is_floating_point() and float(r.double().norm()) < ZERO_GRAD * scale
            for r in ref.grads]


# ----------------------------------------------------------------------------- op drilldown


def _trace_backward(module: nn.Module, call: Call, upstream, device: str, dtype: torch.dtype,
                    limit: int):
    """ATen ops of one module's backward alone (forward runs outside the tracer)."""
    args = _cast_inputs(call.args, device, dtype)
    kwargs = _cast_inputs(call.kwargs, device, dtype)
    leaves: List[torch.Tensor] = []

    def prep(t):
        if not t.is_floating_point():
            return t
        leaf = t.detach().requires_grad_(True)
        leaves.append(leaf)
        return leaf.clone()

    args, kwargs = map_tensors((args, kwargs), prep)
    params = [p for p in module.parameters() if p.requires_grad]
    tracer = _make_tracer(limit)
    nodes: List[str] = []
    base_dispatch = type(tracer).__torch_dispatch__

    def dispatch(self, func, types, a=(), k=None):
        n = torch._C._current_autograd_node()
        before = len(self.ops)
        out = base_dispatch(self, func, types, a, k)
        if len(self.ops) > before:
            nodes.append(type(n).__name__ if n is not None else "")
        return out

    type(tracer).__torch_dispatch__ = dispatch
    with torch.enable_grad():
        with _replaying(call):
            outs = flatten_tensors(module(*args, **kwargs))
        pairs = [(o, g) for o, g in zip(outs, upstream)
                 if g is not None and o.requires_grad and o.is_floating_point()]
        grads = [g.to(device=o.device, dtype=o.dtype) for o, g in pairs]
        with tracer:
            torch.autograd.grad([o for o, _ in pairs], leaves + params, grads, allow_unused=True)
    _sync(device)
    return tracer.ops, nodes, tracer.truncated


class _RunInBackward(torch.autograd.Function):
    """Runs a callable while autograd executes a backward pass."""

    @staticmethod
    def forward(ctx, x, box):
        ctx.box = box
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        box = ctx.box
        try:
            box["out"] = box["fn"]()
        except Exception as exc:  # surfaced after the backward pass returns
            box["exc"] = exc
        return g, None


def _in_backward(fn):
    """fn() called inside a backward pass, as the op ran in the real graph. A kernel that is only
    wrong in backward (and fault injectors that only fire there) behaves the same when isolated."""
    box: Dict[str, Any] = {"fn": fn}
    x = torch.zeros((), dtype=torch.float64, requires_grad=True)
    with torch.enable_grad():
        _RunInBackward.apply(x, box).backward()
    if "exc" in box:
        raise box["exc"]
    return box.get("out")


def _op_emulated_noise(func, args, kwargs, reference, dtype: torch.dtype) -> Optional[float]:
    """Error of the op computed exactly from the same `dtype` inputs, its result rounded to `dtype`:
    what a correct kernel shows (cancellation in a sum of rounded values, for example)."""
    if dtype == torch.float64:
        return None
    try:
        a, k = map_tensors((args, kwargs),
                           lambda t: t.detach().cpu().to(torch.float64) if t.is_floating_point()
                           else t.detach().cpu())
        a, k = _retarget_cpu((a, k))
        with torch.no_grad():
            out = func(*a, **k)
        outs = [t.to(dtype).to(torch.float64) if t.is_floating_point() else t
                for t in _leaves(out)]
        return diff_lists(outs, reference).rel_err
    except Exception:
        return None


def _retarget_cpu(obj):
    if isinstance(obj, torch.device):
        return torch.device("cpu")
    if isinstance(obj, tuple):
        return tuple(_retarget_cpu(x) for x in obj)
    if isinstance(obj, list):
        return [_retarget_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _retarget_cpu(v) for k, v in obj.items()}
    return obj


def drill_backward(ref_module: nn.Module, dev_module: nn.Module, call: Call, upstream,
                   device: str, dtype: torch.dtype, threshold: float, is_cpu: bool,
                   limit: int = MAX_OPS):
    """First ATen op in one module's backward whose own result diverges on `device`.

    Returns (OpResult or None, autograd node name or None, notes).
    """
    notes: List[str] = []
    ref_ops, ref_nodes, t1 = _trace_backward(ref_module, call, upstream, "cpu", torch.float64, limit)
    dev_ops, dev_nodes, t2 = _trace_backward(dev_module, call, upstream, device, dtype, limit)
    if t1 or t2:
        notes.append(f"backward op drilldown skipped for {call.key}: more than {limit} ops.")
        return None, None, notes
    ref_keep = [i for i, r in enumerate(ref_ops) if _comparable(r)]
    dev_keep = [i for i, r in enumerate(dev_ops) if _comparable(r)]
    ref_c = [ref_ops[i] for i in ref_keep]
    dev_c = [dev_ops[i] for i in dev_keep]
    pairs = _align([r.name for r in ref_c], [r.name for r in dev_c])
    suspect = None
    for i, d in enumerate(dev_c):
        r = ref_c[pairs[i]] if i in pairs else None
        acc = diff_lists(d.outputs, r.outputs) if r is not None else None
        if acc is not None and acc.status(threshold) == OK:
            continue
        got = _in_backward(lambda: _isolate(d, r, device, is_cpu, threshold, dtype))
        if got is None:
            continue
        iso, thr, a, k, reference = got
        emu = _op_emulated_noise(d.func, a, k, reference, dtype)
        if emu is not None:
            thr = max(thr, threshold_for(dtype, False, emu))
        status = iso.status(thr)
        if status == OK:
            continue
        lv = _leaves((a, k))
        res = OpResult(
            op=d.name, module=call.key, index=i + 1, total=len(dev_c),
            input_shapes=[list(t.shape) for t in lv],
            input_dtypes=[str(t.dtype).replace("torch.", "") for t in lv],
            isolated=iso, accumulated=acc, threshold=thr, status=status, matched=r is not None,
            args=a, kwargs=k, reference=reference,
        )
        node = dev_nodes[dev_keep[i]] if dev_keep[i] < len(dev_nodes) else None
        if status == DIVERGENT:
            return res, node or None, notes
        suspect = suspect or (res, node or None)
    if suspect is not None:
        return suspect[0], suspect[1], notes
    only_here = sorted({d.name for i, d in enumerate(dev_c) if i not in pairs})
    msg = (f"backward op drilldown on {call.key}: no single ATen op was confirmed wrong for its "
           "own inputs")
    if only_here:
        msg += ("; ops that only run on this backend (cannot be checked alone against a float64 "
                "CPU run): " + ", ".join(only_here[:5]))
    notes.append(msg + ".")
    return None, None, notes


# ----------------------------------------------------------------------------- compare one call


ZERO_GRAD = 1e-6  # a reference gradient this much smaller than its call's largest is zero


def _scale(rep: "_Replay") -> float:
    return max([float(r.double().norm()) for r in rep.grads if r.is_floating_point()] or [0.0])


def _grad_diff(got: torch.Tensor, ref: torch.Tensor, scale: float) -> Diff:
    """diff_tensors, except that a gradient that is zero in exact math (the reference is only
    rounding, like the key bias of attention, which softmax cancels) is measured against
    ZERO_GRAD times the largest gradient of the same call instead of against itself."""
    d = diff_tensors(got, ref)
    if d.shape_mismatch or not ref.is_floating_point() or scale <= 0:
        return d
    rn = float(ref.cpu().double().norm())
    floor = ZERO_GRAD * scale
    if rn < floor and math.isfinite(d.rel_err):
        num = float((got.cpu().double() - ref.cpu().double()).norm())
        return Diff(num / floor, d.max_abs, d.nonfinite_mismatch, False,
                    note="the reference gradient is zero up to rounding")
    return d


_ROUNDED_NOTE = "measured against a correctly rounded"


def _rounded_run_diff(got: torch.Tensor, emu: torch.Tensor, ref: torch.Tensor,
                      emu_err: float, dtype: torch.dtype) -> Diff:
    """Distance from the backend gradient to the correctly rounded (emulated) run, over the norm
    of the exact reference, so it reads on the same scale as the error against the reference."""
    d = diff_tensors(got, emu)
    if d.shape_mismatch or d.nonfinite_mismatch:
        return d
    num = float((got.cpu().double() - emu.double()).nan_to_num().norm())
    den = float(ref.double().norm())
    dt = str(dtype).replace("torch.", "")
    return Diff(num / den if den > 0 else d.rel_err, d.max_abs, False, False,
                note=f"{_ROUNDED_NOTE} {dt} run: {dt} itself is "
                     f"{emu_err:.2e} off float64 here (rounding flips a ReLU mask, max pool route "
                     f"or similar)")


def _compare(got: _Replay, ref: _Replay, noise: Optional[List[float]], dtype: torch.dtype,
             tf32: bool, is_leaf: bool, emu_kept: Optional[List[Optional[torch.Tensor]]] = None,
             emu_err: Optional[List[float]] = None) -> Tuple[List[GradTensor], str]:
    """Per gradient tensor status. Where the emulated floor was too large to use (_floor drops it),
    a gradient that is off the exact reference is still OK when it matches the correctly rounded
    run within the usual threshold: a correct kernel in this precision computes that same value, so
    the error is the precision's, not a backward bug. A wrong kernel does not land on the rounded
    run, so it stays flagged."""
    out: List[GradTensor] = []
    status = OK
    scale = _scale(ref)
    for i, (label, g, r) in enumerate(zip(ref.labels, got.grads, ref.grads)):
        n = None if noise is None else noise[i]
        thr = threshold_for(dtype, tf32, n)
        d = _grad_diff(g, r, scale)
        st = d.status(thr)
        e = None if emu_kept is None else emu_kept[i]
        if st != OK and e is not None and emu_err is not None:
            de = _rounded_run_diff(g, e, r, emu_err[i], dtype)
            if de.status(thr) == OK:
                d, st = de, OK
        ratio, cos = _norm_cos([g], [r])
        out.append(GradTensor(label, d, thr, st, ratio, cos))
        status = worst(status, st)
    if not is_leaf and status == SUSPECT:
        status = OK  # parents compound their children's precision; flag only clear errors
    return out, status


# ----------------------------------------------------------------------------- training trend


def _owner_norms(model: nn.Module) -> Dict[str, float]:
    """Gradient norm per module that owns parameters directly, plus "<total>"."""
    sq: Dict[str, float] = {}
    total = 0.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        v = float(p.grad.detach().cpu().double().pow(2).sum())   # MPS has no float64
        owner = n.rsplit(".", 1)[0] if "." in n else ROOT
        sq[owner] = sq.get(owner, 0.0) + v
        total += v
    out = {k: math.sqrt(v) for k, v in sq.items()}
    out["<total>"] = math.sqrt(total)
    return out


def _train(base: nn.Module, args, kwargs, targets, loss_fn: LossFn, device: str,
           dtype: torch.dtype, steps: int, lr: float, emulate: Optional[torch.dtype] = None):
    """SGD steps from the same weights; per step gradient norms and losses.

    With `emulate`, the loop runs in float64 on CPU with weights, inputs and every op result
    (optimizer updates included) rounded to that dtype: the trajectory of a correct implementation.
    """
    if emulate is not None:
        device, dtype = "cpu", torch.float64
    model = copy.deepcopy(base).to(device=device, dtype=dtype)
    model = model.train() if base.training else model.eval()
    if active_trainmode() is not None:
        active_trainmode().attach(model)   # keyed dropout: the same masks in every training loop
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, foreach=False)
    a = _cast_inputs(args, device, emulate or dtype)
    k = _cast_inputs(kwargs, device, emulate or dtype)
    # targets are float64 on the CPU; the MPS device has no float64, so they go as float32 there
    tg = [t.to(torch.float32).to(device) if (str(device).startswith("mps") and t.dtype == torch.float64)
          else t.to(device) for t in targets]
    if emulate is not None:
        with torch.no_grad():
            for p in list(model.parameters()) + list(model.buffers()):
                if p.is_floating_point():
                    p.copy_(p.to(emulate).to(torch.float64))
        a, k = _cast_inputs((a, k), "cpu", torch.float64)
    mode = _rounding_mode(emulate) if emulate is not None else contextlib.nullcontext()
    with mode:
        return _train_loop(model, opt, a, k, tg, loss_fn, device, steps)


def _train_loop(model, opt, a, k, tg, loss_fn, device, steps):
    norms, losses = [], []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = loss_fn(flatten_tensors(model(*a, **k)), tg)
            loss.backward()
        _sync(device)
        norms.append(_owner_norms(model))
        losses.append(float(loss.detach().cpu().double()))
        opt.step()
    return norms, losses


def _ratios(got: List[Dict[str, float]], ref: List[Dict[str, float]], key: str):
    out: List[Optional[float]] = []
    for g, r in zip(got, ref):
        a, b = g.get(key), r.get(key)
        if a is None or b is None or b == 0.0 or not math.isfinite(a):
            out.append(None if a is None or b is None or b == 0.0 else float("inf"))
        else:
            out.append(a / b)
    return out


def _dev(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float("inf") if not math.isfinite(x) else abs(x - 1.0)


def _trend(res: GradBackendResult, base, args, kwargs, targets, loss_fn, dtype, steps, lr,
           ref_norms, controls: Dict[str, List[Dict[str, float]]]) -> StepTrend:
    b = res.backend
    got, losses = _train(base, args, kwargs, targets, loss_fn, b.device, dtype, steps, lr)
    thr = threshold_for(dtype, b.tf32, None)
    keys = ["<total>"] + sorted(k for k in ref_norms[0] if k != "<total>")
    mods = {k: _ratios(got, ref_norms, k) for k in keys}
    ctrl = {name: {k: _ratios(n, ref_norms, k) for k in keys} for name, n in controls.items()}
    # Low precision trajectories drift from float64 even when every kernel is right (small SGD
    # updates round away). So the backend is judged against the trajectory of a correct run in
    # its own precision: float64 math rounded to that precision ("emulated"), or float64 itself
    # when the model is float64. The CPU run in the same precision, when there is one, measures
    # how far two correct runs drift apart.
    basis = controls.get("emulated", ref_norms)
    flagged = []
    if steps >= 3:
        for k in keys:
            dv = [_dev(x) for x in _ratios(got, basis, k)]
            if any(x is None for x in dv):
                continue
            noise = 0.0
            for name, c in controls.items():
                if name != "emulated":
                    noise = max([noise] + [x for x in (_dev(y) for y in _ratios(c, basis, k))
                                           if x is not None])
            first, last = dv[0], dv[-1]
            # Growth: the gap at the end is over threshold, clearly larger than at the start, and
            # larger than two correct runs drift apart along the same loop.
            if last > max(thr, 10.0 * noise) and last > 2.0 * first + thr / 2:
                flagged.append({"module": k, "first_ratio": mods[k][0], "last_ratio": mods[k][-1],
                                "first_gap": first, "last_gap": last, "control_gap": noise})
    return StepTrend(steps, lr, mods["<total>"], {n: c["<total>"] for n, c in ctrl.items()},
                     {k: v for k, v in mods.items() if k != "<total>"}, thr, flagged, losses)


# ----------------------------------------------------------------------------- main entry


def grad_check(
    model: nn.Module,
    inputs: Union[torch.Tensor, Sequence[Any]],
    kwargs: Optional[Dict[str, Any]] = None,
    backends: Optional[List[str]] = None,
    loss_fn: Optional[LossFn] = None,
    steps: int = 0,
    lr: float = 1e-2,
    repro_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    drill_ops: bool = True,
    seed: int = 0,
    train: bool = False,
) -> GradReport:
    """Compare every module's own backward pass on every backend against a float64 CPU reference.

    Args:
        model: any nn.Module; copied, never modified, run in eval mode (dropout off), or with
            ``train=True`` in training mode (truestar.trainmode): batch statistics, running
            statistic updates compared with the forward, dropout masks keyed by ``seed`` and module
            so every backend draws the same ones (and the backward goes through the same mask).
        inputs: a tensor or a sequence of positional arguments.
        loss_fn: loss_fn(outputs, targets) -> scalar, where outputs is the flat list of output
            tensors and targets fixed random tensors of the same shapes. Default: mean squared error.
        steps: when 3 or more, also run that many SGD steps (learning rate `lr`) on each side and
            flag gradient norm ratios that drift over the steps.
        drill_ops: name the ATen op in the blamed module's backward.
    """
    from .trainmode import TrainMode

    tm = TrainMode(seed) if train else None
    with tm if tm is not None else contextlib.nullcontext():
        return _grad_check(model, inputs, kwargs, backends, loss_fn, steps, lr, repro_dir, model_name,
                           drill_ops, seed, train)


def _grad_check(model, inputs, kwargs, backends, loss_fn, steps, lr, repro_dir, model_name, drill_ops,
                seed, train) -> GradReport:
    from .trainmode import prepare

    args = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    kwargs = dict(kwargs or {})
    name = model_name or type(model).__name__
    dtype = _model_dtype(model)
    chosen, skipped = resolve_backends(backends)
    notes: List[str] = []
    issues: List[Issue] = []
    lf = loss_fn or default_loss
    loss_desc = "custom loss_fn" if loss_fn is not None else "mean squared error to a fixed random target"

    base = prepare(model, train)
    tm = active_trainmode()
    if tm is not None:
        notes.append("training mode: batch statistics, running statistic updates compared with the forward, "
                     "keyed dropout masks (the same on every backend).")

    # 1. Reference forward and backward in float64 on CPU, with upstream gradients per call.
    ref_model = copy.deepcopy(base).to(torch.float64)
    try:
        # Floating model inputs take gradients too, so a model with no parameters is checked.
        req = lambda t: t.requires_grad_(True).clone() if t.is_floating_point() else t  # noqa: E731
        a64 = map_tensors(_cast_inputs(args, "cpu", torch.float64), req)
        k64 = map_tensors(_cast_inputs(kwargs, "cpu", torch.float64), req)
        if tm is not None:
            tm.attach(ref_model)
            tm.recording = True
        try:
            with torch.enable_grad(), _GradRecorder(ref_model) as rec:
                out = ref_model(*a64, **k64)
        finally:
            if tm is not None:
                tm.recording = False
        outs = flatten_tensors(out)
        targets = _targets(outs, seed)
        loss = lf(outs, targets)
        if not loss.requires_grad:
            raise ReferenceFailed("the loss does not depend on any trainable parameter or input")
        loss.backward()
    except ReferenceFailed:
        raise
    except Exception as exc:
        raise ReferenceFailed(f"forward and backward failed on the float64 CPU reference: "
                              f"{type(exc).__name__}: {exc}") from exc
    calls, upstream, op_counting = rec.calls, rec.upstream, rec.op_counting
    if not op_counting:
        notes.append("op counting unavailable: every parent module is replayed (slower).")

    # 2. Reference and noise floor per call: isolated backward in float64 and in the model dtype.
    ref_rep: Dict[str, Optional[_Replay]] = {}
    noise: Dict[str, Tuple[Optional[List[float]], Optional[float]]] = {}
    ref_err: Dict[str, str] = {}
    cpu_model = copy.deepcopy(base)
    emu_model = None
    if dtype != torch.float64:
        emu_model = copy.deepcopy(ref_model)
        with torch.no_grad():
            for p in list(emu_model.parameters()) + list(emu_model.buffers()):
                if p.is_floating_point():
                    p.copy_(p.to(dtype).to(torch.float64))
    variants = _unique([_tf32_flags(b, dtype) for b in chosen])
    for c in calls:
        if not _needs_replay(c, op_counting):
            continue
        ups = upstream.get(c.key) or []
        if not any(g is not None for g in ups):
            ref_rep[c.key] = None
            continue
        try:
            ref_rep[c.key] = _replay_cpu64(_module_by_name(ref_model, c.name), c, ups)
        except Exception as exc:
            ref_rep[c.key] = None
            ref_err[c.key] = f"{type(exc).__name__}: {exc}"
            continue
        r = ref_rep[c.key]
        if r is None:
            continue
        cpu_g = cpu_f = None
        emu: Dict[Tuple[bool, bool], Tuple[Optional[List[float]], Optional[float]]] = {}
        try:
            n = _replay_backward(_module_by_name(cpu_model, c.name), c, ups, "cpu", dtype)
            cpu_g = [_grad_diff(g, x, _scale(r)).rel_err for g, x in zip(n.grads, r.grads)]
            cpu_f = diff_lists(n.outputs, c.outputs).rel_err
        except Exception:
            pass
        cap = DTYPE_FLOOR.get(dtype, 1e-4) * DIVERGENT_FACTOR
        for variant in variants if emu_model is not None else []:
            try:
                e = _replay_emulated(_module_by_name(emu_model, c.name), c, ups, dtype, variant)
                errs = [_grad_diff(g, x, _scale(r)).rel_err for g, x in zip(e.grads, r.grads)]
                # Keep the emulated gradient itself only where its floor will be dropped (see
                # _floor): there the backend is also compared with the correctly rounded run.
                kept = [g if v == v and v > cap else None for g, v in zip(e.grads, errs)]
                emu[variant] = (errs, diff_lists(e.outputs, c.outputs).rel_err, kept)
            except Exception:
                pass
        noise[c.key] = (cpu_g, cpu_f, emu)

    # Optional training loop on the reference and the CPU control.
    ref_norms = None
    controls: Dict[str, List[Dict[str, float]]] = {}
    if steps and steps >= 1:
        ref_norms, _ = _train(base, args, kwargs, targets, lf, "cpu", torch.float64, steps, lr)
        if dtype != torch.float64:
            try:
                controls["emulated"], _ = _train(base, args, kwargs, targets, lf, "cpu", dtype,
                                                 steps, lr, emulate=dtype)
            except Exception as exc:
                notes.append(f"emulated training control failed ({type(exc).__name__}: {exc}).")
        if steps < 3:
            notes.append("steps < 3: gradient norms are reported but no growth trend is judged.")

    results: List[GradBackendResult] = []
    for b in chosen:
        notes_b = [n for n in b.notes if dtype == torch.float32 or not n.startswith("TF32")]
        res = GradBackendResult(backend=b, notes=notes_b)
        try:
            _check_backend(res, base, ref_model, calls, upstream, ref_rep, ref_err, noise, dtype,
                           op_counting, drill_ops, repro_dir, name, emu_model)
            if ref_norms is not None:
                if "cpu" not in controls and b.name != "cpu":
                    controls["cpu"], _ = _train(base, args, kwargs, targets, lf, "cpu", dtype,
                                                steps, lr)
                ctl = {n: v for n, v in controls.items() if not (n == "cpu" and b.name == "cpu")}
                res.trend = _trend(res, base, args, kwargs, targets, lf, dtype, steps, lr,
                                   ref_norms, ctl)
                for f in res.trend.flagged:
                    res.status = worst(res.status, SUSPECT)
                    res.add("TS183",
                            f"gradient norm of {f['module']} drifts from a correct "
                            f"{str(dtype).replace('torch.', '')} run over {steps} steps: gap "
                            f"{_fmt(f['first_gap'])} at step 1, {_fmt(f['last_gap'])} at the end "
                            f"(ratio vs float64 {_fmt(f['first_ratio'])} to "
                            f"{_fmt(f['last_ratio'])}; two correct runs differ by "
                            f"{_fmt(f['control_gap'])})")
        except Exception as exc:
            res.status = DIVERGENT
            res.error = f"{type(exc).__name__}: {exc}"
            res.add("TS002", f"the model failed to run forward and backward on {b.name}: "
                             f"{res.error}")
            res.notes.append("The model failed on this backend:\n" + traceback.format_exc(limit=3))
        results.append(res)

    return GradReport(name, dtype, "CPU float64", loss_desc, results, skipped, notes, issues)


def _active_faults() -> List[str]:
    """Names of truestar fault injector modes active right now (tests and demos)."""
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        stack = _get_current_dispatch_mode_stack()
    except Exception:
        return []
    out = []
    for m in stack:
        if type(m).__module__ in ("truestar.gradfault", "truestar.testing"):
            out.append(f"{type(m).__name__}({getattr(m, 'op', '')})")
    return out


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf"
    return f"{x:.4g}"


def _explain_ties(lay: GradLayer, res: GradBackendResult, c: Call, dev_model, ref_model, emu_model,
                  ups, b: Backend, dtype, got: _Replay, ref: _Replay) -> None:
    """A parent replay over threshold: if near tie ReLU mask flips explain it (gradties.py), make it
    SUSPECT with a note; otherwise leave it DIVERGENT and say why it was not explained."""
    from . import gradties

    try:
        exp, why = gradties.explain(
            _module_by_name(dev_model, c.name), _module_by_name(ref_model, c.name),
            _module_by_name(emu_model, c.name), c, ups, b.device, dtype, _tf32_flags(b, dtype),
            got, ref, lay.tensors)
    except Exception as exc:  # the explanation is optional; never let it hide the finding
        exp, why = None, f"{type(exc).__name__}: {exc}"
    if exp is None:
        if why and why not in ("no ReLU in this call", "no ReLU mask differs from float64"):
            lay.note = f"not a near tie mask flip: {why}"
        return
    lay.tie = exp
    lay.tie_saved = [t.status for t in lay.tensors]
    for t in lay.tensors:
        if t.status != OK:
            t.status = SUSPECT
    lay.status = SUSPECT
    lay.note = exp.note(b.name, dtype)


def _check_backend(res: GradBackendResult, base, ref_model, calls, upstream, ref_rep, ref_err,
                   noise, dtype, op_counting, drill_ops, repro_dir, model_name,
                   emu_model=None) -> None:
    b = res.backend
    is_cpu = b.name == "cpu"
    res.injected = _active_faults()
    dev_model = copy.deepcopy(base).to(b.device)
    by_key = {c.key: c for c in calls}
    layers: Dict[str, GradLayer] = {}
    bad_inside: Dict[str, bool] = {}
    base_floor = DTYPE_FLOOR.get(dtype, 1e-4)
    for c in calls:  # completion order: children before parents
        lay = GradLayer(c.key, c.name, c.type_name, c.is_leaf, c.start)
        inner_bad = any(layers[k].status != OK or bad_inside.get(k, False) for k in c.children
                        if k in layers)
        bad_inside[c.key] = inner_bad
        layers[c.key] = lay
        if inner_bad or not _needs_replay(c, op_counting):
            lay.replayed = False
            continue
        if c.key in ref_err:
            lay.replayed = False
            lay.note = f"float64 backward replay failed: {ref_err[c.key]}"
            continue
        ref = ref_rep.get(c.key)
        if ref is None:
            lay.replayed = False
            lay.note = "no gradient reaches this call"
            continue
        cpu_g, cpu_f, emu = noise.get(c.key, (None, None, {}))
        emu_g, emu_f, emu_kept = emu.get(_tf32_flags(b, dtype), (None, None, None))
        zero = _zero_refs(ref)
        # Training mode parents: batch statistics couple every element, so a parent's gradient in a
        # correct low precision run is far off float64 (resnet18, batch 16, plain PyTorch float32 vs
        # float64: up to 1.1e-2, eval mode 6.8e-7). Their measured floors are kept uncapped; leaves,
        # where a wrong kernel shows, keep the cap.
        uncapped = active_trainmode() is not None and not c.is_leaf
        n_grads = [
            _floor(base_floor, None if is_cpu or cpu_g is None else cpu_g[i],
                   None if emu_g is None else emu_g[i], zero[i], uncapped)
            for i in range(len(ref.labels))]
        # The forward part uses the forward check's own threshold (the CPU noise floor, none on
        # the CPU backend itself), so the two checks never disagree on the same module.
        n_fwd = _floor(base_floor, None if is_cpu else cpu_f, None)
        try:
            got = _replay_backward(_module_by_name(dev_model, c.name), c, upstream[c.key],
                                   b.device, dtype)
            _sync(b.device)
            if got is None or got.labels != ref.labels:
                raise RuntimeError("the replay computed different gradients than the reference")
        except Exception as exc:
            lay.error = f"{type(exc).__name__}: {exc}"
            lay.status = DIVERGENT
            res.status = worst(res.status, DIVERGENT)
            continue
        tf32 = _uses_tf32(b, c, op_counting)  # TF32 rounds only matmuls and convolutions
        lay.tensors, lay.status = _compare(got, ref, n_grads, dtype, tf32, c.is_leaf,
                                           emu_kept, emu_g)
        lay.norm_ratio, lay.cosine = _norm_cos(got.grads, ref.grads)
        lay.forward_threshold = threshold_for(dtype, tf32, n_fwd)
        lay.forward = diff_lists(got.outputs, c.outputs)
        lay.forward_status = lay.forward.status(lay.forward_threshold)
        if not c.is_leaf and lay.forward_status == SUSPECT:
            lay.forward_status = OK  # same parent rule as the forward check
        if lay.forward_status == DIVERGENT:
            # Gradients of a module whose own forward is wrong cannot be called right.
            lay.status = DIVERGENT
        elif (lay.status == DIVERGENT and not c.is_leaf and emu_model is not None
              and lay.forward_status == OK):
            _explain_ties(lay, res, c, dev_model, ref_model, emu_model, upstream[c.key], b,
                          dtype, got, ref)
        res.status = worst(res.status, lay.status)

    # A parent excused by near tie mask flips stays excused only while every leaf replay is clean
    # (the owner's rule, gradties.py); otherwise it is judged as before.
    tie_parents = [l for l in layers.values() if l.tie is not None]
    if tie_parents and any(l.is_leaf and l.replayed and l.status != OK for l in layers.values()):
        for l in tie_parents:
            for t, st in zip(l.tensors, l.tie_saved):
                t.status = st
            l.status, l.note, l.tie, l.tie_saved = DIVERGENT, "", None, []
        res.status = worst(res.status, DIVERGENT)
    for l in layers.values():
        if l.tie is not None:
            res.notes.append(f"{l.key}: {l.note}.")
    res.layers = sorted(layers.values(), key=lambda l: l.start)
    rounded = [(l, t) for l in res.layers for t in l.tensors
               if t.diff.note.startswith(_ROUNDED_NOTE)]
    if rounded:
        l, t = rounded[0]
        res.notes.append(
            f"{len(rounded)} gradient tensor(s) are further off float64 than "
            f"{str(dtype).replace('torch.', '')} can resolve on these inputs, since a correctly "
            f"rounded run is equally off (first: {l.key} {t.label}); they were checked against that "
            f"rounded run instead (own error there {t.diff.rel_err:.1e}).")
    bad = [l for l in res.layers if l.status != OK]
    # Gradients flow from the last module in forward order to the first: the earliest bad module
    # in backward order is where the corruption starts.
    order = sorted(bad, key=lambda l: -l.start)
    divergent = [l for l in order if l.status == DIVERGENT]
    fd = divergent[0] if divergent else (order[0] if order else None)
    res.first_divergent = fd
    if fd is None:
        return
    others = len(bad) - 1
    tail = f"; {others} more module(s) over threshold" if others else ""
    if fd.error:
        res.add("TS184", f"module {fd.key} ({fd.type_name}) raised when its backward was replayed "
                         f"on {b.name}: {fd.error}")
    else:
        w = fd.worst
        what = (f"module {fd.key} ({fd.type_name}) gradient of {w.label}: own error "
                f"{w.diff.rel_err:.2e} vs threshold {w.threshold:.1e}, norm ratio "
                f"{_fmt(fd.norm_ratio)}, cosine {_fmt(fd.cosine)}")
        if fd.forward_status != OK:
            res.add("TS182", what + f"; its own forward also diverges (error "
                                    f"{fd.forward.rel_err:.2e}), so run the forward check first"
                    + tail)
        elif fd.tie is not None:
            res.add("TS188", what + f"; {fd.note}" + tail)
        else:
            res.add("TS180" if fd.status == DIVERGENT else "TS181",
                    what + f"; its forward matches (error {fd.forward.rel_err:.2e}), so the "
                           "backward itself is wrong" + tail)
    try:
        from .agent import module_source

        res.blame_source = module_source(base, fd.name)
    except Exception:
        res.blame_source = None

    c = by_key[fd.key]
    if drill_ops and not fd.error:
        try:
            ref_mod = _module_by_name(ref_model, c.name)
            op, node, op_notes = drill_backward(ref_mod, _module_by_name(dev_model, c.name), c,
                                                upstream[c.key], b.device, dtype,
                                                threshold_for(dtype, b.tf32, None), is_cpu)
            res.first_divergent_op, res.op_node = op, node
            res.notes.extend(op_notes)
        except Exception as exc:
            res.notes.append(f"backward op drilldown failed on {c.key}: {type(exc).__name__}: {exc}")

    if repro_dir and not fd.error and active_trainmode() is not None:
        res.notes.append("No repro written in training mode: a standalone repro would draw PyTorch's own "
                         "dropout masks and change the module's statistics; rerun with train=False for one.")
    elif repro_dir and not fd.error:
        try:
            res.repro_path = write_grad_repro(repro_dir, model_name, b, c,
                                              _module_by_name(base, c.name), dtype,
                                              upstream[c.key], ref_rep[c.key], fd)
        except Exception as exc:
            res.notes.append(f"No backward repro written ({type(exc).__name__}: {exc}).")
        if res.first_divergent_op is not None:
            try:
                from .repro import write_op_repro

                folder = dataclasses.replace(c, key=f"{c.key}_backward")
                res.op_repro_path = write_op_repro(repro_dir, model_name, b, folder,
                                                   res.first_divergent_op)
            except Exception as exc:
                res.notes.append(f"No op repro written ({type(exc).__name__}: {exc}).")


# ----------------------------------------------------------------------------- repro


GRAD_SCRIPT = '''"""Backward repro generated by truestar check --backward.

Module:   {layer} ({type_name}) from model {model}
Backend:  {backend} ({description})
Runs the module's forward and backward alone with the saved inputs and upstream gradients and
compares every input and parameter gradient with a float64 CPU reference.

Run:  python grad_repro.py         (uses the backend above)
      python grad_repro.py cpu     (control run, should pass)
Needs only PyTorch. Exit code 1 means the wrong gradient reproduced.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in {import_paths!r}:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
DTYPE = {dtype}


def tensors(o):
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, (list, tuple)):
        return [t for x in o for t in tensors(x)]
    if isinstance(o, dict):
        return [t for k in sorted(o, key=str) for t in tensors(o[k])]
    return []


def rel_err(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
        return float("inf")
    m = torch.isfinite(b)
    num = torch.linalg.vector_norm((a - b)[m]).item()
    den = torch.linalg.vector_norm(b[m]).item()
    return 0.0 if num == 0 else (num / den if den else float("inf"))


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else {device!r}
    p = torch.load(os.path.join(HERE, "grad_payload.pt"), weights_only=False)
    module = p["module"].to(device).eval()
    leaves = []

    def move(o):
        if isinstance(o, torch.Tensor):
            if not o.is_floating_point():
                return o.to(device)
            leaf = o.to(device=device, dtype=DTYPE).requires_grad_(True)
            leaves.append(leaf)
            return leaf.clone()
        if isinstance(o, tuple):
            return tuple(move(x) for x in o)
        if isinstance(o, list):
            return [move(x) for x in o]
        if isinstance(o, dict):
            return {{k: move(v) for k, v in o.items()}}
        return o

    args, kwargs = move(p["args"]), move(p["kwargs"])
    params = [q for _, q in module.named_parameters() if q.requires_grad]
    outs = tensors(module(*args, **kwargs))
    pairs = [(o, g) for o, g in zip(outs, p["upstream"]) if g is not None and o.requires_grad]
    grads = torch.autograd.grad([o for o, _ in pairs], leaves + params,
                                [g.to(device=o.device, dtype=o.dtype) for o, g in pairs],
                                allow_unused=True)
    failed = False
    for label, g, w, ref, thr in zip(p["labels"], grads, leaves + params, p["reference"],
                                     p["thresholds"]):
        g = torch.zeros_like(w) if g is None else g
        e = rel_err(g, ref)
        bad = e > thr
        failed |= bad
        print(f"{{label:<24}} relative error {{e:.3e}} (threshold {{thr:.1e}}){{'  WRONG' if bad else ''}}")
    print(f"torch {{torch.__version__}} on {{device}}")
    if failed:
        print("REPRODUCED: a gradient diverges from the float64 CPU reference")
        sys.exit(1)
    print("ok: every gradient matches the reference")


if __name__ == "__main__":
    main()
'''


def write_grad_repro(repro_dir: str, model_name: str, backend: Backend, call: Call,
                     module: nn.Module, dtype: torch.dtype, upstream, ref: _Replay,
                     layer: GradLayer) -> str:
    from .repro import _slug

    out_dir = os.path.join(repro_dir, _slug(f"{backend.name}_{call.key}_backward"))
    os.makedirs(out_dir, exist_ok=True)
    mod = copy.deepcopy(module).cpu().to(dtype).eval()
    torch.save({
        "module": mod, "args": call.args, "kwargs": call.kwargs, "upstream": list(upstream),
        "labels": ref.labels, "reference": ref.grads,
        "thresholds": [t.threshold for t in layer.tensors],
    }, os.path.join(out_dir, "grad_payload.pt"))
    import_paths: List[str] = []
    for m in mod.modules():
        f = getattr(sys.modules.get(type(m).__module__), "__file__", None)
        top = type(m).__module__.split(".")[0]
        if f and top not in ("torch", "truestar", "torchvision"):
            d = os.path.dirname(os.path.abspath(f))
            for _ in range(type(m).__module__.count(".")):
                d = os.path.dirname(d)
            if d not in import_paths:
                import_paths.append(d)
    script = GRAD_SCRIPT.format(layer=call.key, type_name=call.type_name, model=model_name,
                                backend=backend.name, description=backend.description,
                                dtype="torch." + str(dtype).replace("torch.", ""),
                                device=backend.device, import_paths=import_paths)
    path = os.path.join(out_dir, "grad_repro.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    return path


# ----------------------------------------------------------------------------- text report


_LABEL = {OK: "OK", SUSPECT: "SUSPECT", DIVERGENT: "DIVERGENT"}


def _e(x) -> str:
    if x is None:
        return "-"
    if not math.isfinite(x):
        return "inf"
    return f"{x:.1e}"


def render_text(report: GradReport, top: int = 5) -> str:
    dt = str(report.dtype).replace("torch.", "")
    lines = [f"truestar check --backward: {report.model_name} ({dt}), reference "
             f"{report.reference}, loss: {report.loss}"]
    for n in report.notes:
        lines.append(f"  note: {n}")
    for i in report.issues:
        lines.append(f"  {i.code}: {i.message}")
    for n, why in report.skipped:
        lines.append(f"  skipped {n} (TS001): {why}")
    lines.append("")
    for r in report.results:
        b = r.backend
        codes = f"  {' '.join(r.codes)}" if r.codes else ""
        lines.append(f"[{_LABEL[r.status]}] {b.name}: {b.description}{codes}")
        if r.error:
            lines.append(f"  TS002 failed: {r.error}")
        for n in r.notes:
            for k, part in enumerate(n.rstrip().splitlines()):
                lines.append(("  note: " if k == 0 else "        ") + part)
        checked = [l for l in r.layers if l.replayed]
        lines.append(f"  module backward passes checked: {len(checked)}")
        for i in r.issues:
            if i.code in ("TS180", "TS181", "TS182", "TS184"):
                kind = "suspect" if i.code == "TS181" else "divergent"
                lines.append(f"  {i.code} first {kind} backward: {i.message}")
        src = (r.blame_source or {}).get("user")
        if src is not None and r.first_divergent is not None:
            lines.append(f"  source: {src['file']}:{src['line']} ({src['qualname']})")
        op = r.first_divergent_op
        if op is not None:
            lines.append(f"  first divergent backward op: {op.op}"
                         + (f" in {r.op_node}" if r.op_node else "")
                         + f" (op {op.index} of {op.total} in the backward of {op.module}), own "
                           f"error {_e(op.isolated.rel_err)} vs threshold {_e(op.threshold)}")
            ins = ", ".join(f"{d}{s}" for d, s in zip(op.input_dtypes, op.input_shapes))
            lines.append(f"    inputs: {ins or 'none'}"
                         + ("" if op.matched else "; this op does not run in the reference"))
        if r.repro_path:
            lines.append(f"  repro: python {r.repro_path}   (control: add cpu)")
        if r.op_repro_path:
            lines.append(f"  op repro: python {r.op_repro_path}")
        ranked = sorted(checked, key=lambda l: -(l.worst.ratio if l.worst else 0.0))[:top]
        if ranked:
            lines.append("  worst gradients (own error / threshold, norm ratio, cosine):")
            for l in ranked:
                w = l.worst
                if w is None:
                    continue
                lines.append(f"    {_LABEL[l.status]:<9} {l.key:<28} {w.label:<20} "
                             f"{_e(w.diff.rel_err)} / {_e(w.threshold)}  norm {_fmt(l.norm_ratio)}"
                             f"  cos {_fmt(l.cosine)}  fwd {_LABEL[l.forward_status]}")
        t = r.trend
        if t is not None:
            series = " ".join(_fmt(x) for x in t.total_ratio)
            lines.append(f"  grad norm ratio vs float64 over {t.steps} SGD steps (lr {t.lr}): "
                         f"{series}")
            for cname, cr in t.control_ratio.items():
                label = f"CPU {dt}" if cname == "cpu" else f"emulated {dt}"
                lines.append(f"    {label} control: " + " ".join(_fmt(x) for x in cr))
            for f in t.flagged:
                lines.append(f"    TS183 drift in {f['module']}: gap to a correct run "
                             f"{_fmt(f['first_gap'])} -> {_fmt(f['last_gap'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------------------- agent JSON


def _num(x) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _agent_backend(r: GradBackendResult) -> Dict[str, Any]:
    from .agent import _repro_info

    fd = r.first_divergent
    blamed = None
    if fd is not None:
        w = fd.worst
        src = r.blame_source or {}
        blamed = {
            "key": fd.key, "name": fd.name, "type": fd.type_name, "leaf": fd.is_leaf,
            "own_code": not fd.is_leaf, "status": fd.status,
            "gradient": None if w is None else w.label,
            "rel_err": None if w is None else _num(w.diff.rel_err),
            "threshold": None if w is None else w.threshold,
            "ratio": None if w is None else _num(w.ratio),
            "norm_ratio": _num(fd.norm_ratio), "cosine": _num(fd.cosine),
            "nonfinite_mismatch": bool(w and w.diff.nonfinite_mismatch),
            "shape_mismatch": bool(w and w.diff.shape_mismatch),
            "forward_status": fd.forward_status,
            "forward_rel_err": None if fd.forward is None else _num(fd.forward.rel_err),
            "backward_only": fd.forward_status == OK and not fd.error,
            "error": fd.error or None,
            "source": src.get("module"), "user_source": src.get("user"),
        }
    op = r.first_divergent_op
    blamed_op = None
    if op is not None:
        rel = _num(op.isolated.rel_err)
        blamed_op = {
            "op": op.op, "node": r.op_node, "module": op.module, "index": op.index,
            "total": op.total,
            "inputs": [{"dtype": d, "shape": list(s)} for d, s in zip(op.input_dtypes,
                                                                      op.input_shapes)],
            "rel_err": rel, "threshold": op.threshold,
            "ratio": None if rel is None else rel / op.threshold,
            "runs_in_reference": bool(op.matched),
        }
    over = [l for l in r.layers if l.status != OK]
    checked = [l for l in r.layers if l.replayed]
    trend = None
    if r.trend is not None:
        t = r.trend
        trend = {"steps": t.steps, "lr": t.lr, "threshold": t.threshold,
                 "total_ratio": [_num(x) for x in t.total_ratio],
                 "control_ratio": {n: [_num(x) for x in v] for n, v in t.control_ratio.items()},
                 "flagged": [{k: (_num(v) if isinstance(v, float) else v) for k, v in f.items()}
                             for f in t.flagged],
                 "losses": [_num(x) for x in t.losses]}
    return {
        "name": r.backend.name, "device": r.backend.device,
        "description": r.backend.description, "status": r.status, "codes": r.codes,
        "injected_faults": list(r.injected),
        "blamed_module": blamed, "blamed_op": blamed_op,
        "modules_checked": len(checked),
        "modules_over_threshold_count": len(over),
        "modules_over_threshold": [
            {"key": l.key, "type": l.type_name, "status": l.status,
             "gradient": l.worst.label if l.worst else None,
             "rel_err": _num(l.rel_err), "threshold": l.threshold,
             "norm_ratio": _num(l.norm_ratio), "forward_status": l.forward_status}
            for l in over[:5]],
        "grad_norm_ratio": {l.key: _num(l.norm_ratio) for l in checked},
        "trend": trend,
        "repro": _repro_info(r.repro_path, r.backend.device),
        "op_repro": _repro_info(r.op_repro_path, r.backend.device),
        "error": r.error or None,
    }


def next_actions(report: GradReport) -> List[Dict[str, Any]]:
    from .agent import _repro_info, _torch_version

    acts: List[Dict[str, Any]] = []
    pin = f"torch=={_torch_version().split('+')[0]}"

    def add(kind, description, command=None, **extra):
        a = {"kind": kind, "description": description, "command": command}
        a.update(extra)
        acts.append(a)

    for r in report.results:
        if r.status == OK:
            continue
        b = r.backend.name
        if r.error:
            add("inspect_failure", f"The model raised on {b}: {r.error}.", backend=b)
            continue
        op_repro = _repro_info(r.op_repro_path, r.backend.device)
        repro = _repro_info(r.repro_path, r.backend.device)
        fd = r.first_divergent
        if op_repro:
            op = r.first_divergent_op
            add("confirm_op_repro",
                f"Run the single op repro for {op.op} (from {r.op_node or 'the backward'}) on {b}; "
                "exit code 1 shows this one op is wrong on its own.", op_repro["command"],
                backend=b, op=op.op)
        if repro:
            add("confirm_repro", f"Run the backward repro on {b}; exit code 1 confirms the wrong "
                                 "gradient on its own.", repro["command"], backend=b)
            if b != "cpu":
                add("control_repro", "Run the same backward repro on CPU; exit code 0 shows the "
                                     f"{b} backward kernel is at fault.",
                    repro["control_command"], backend=b)
        if r.injected:
            add("injected_fault",
                f"A truestar fault injector was active during this check ({', '.join(r.injected)}): "
                "this wrong gradient was put there on purpose, and the repro scripts will not "
                "reproduce it because the injector does not run in them.", backend=b)
            continue
        if fd is not None and fd.tie is not None:
            add("accept_near_tie",
                f"Module {fd.key} is off float64 only through {fd.tie.total} ReLU element(s) that "
                f"take another branch at near ties; a correctly rounded run with the same masks "
                f"matches it, so no backward kernel is wrong. Nothing to fix; to confirm, rerun with "
                "other inputs or in float32.", backend=b, module=fd.name)
            continue
        if "TS182" in r.codes:
            add("check_forward", f"Module {fd.key} is wrong in forward too: run the forward check "
                                 "(truestar check without --backward) and fix that first.",
                backend=b)
        if fd is not None and fd.forward_status == OK and not fd.error:
            add("mitigate_fp32",
                f"Until fixed, run module {fd.key} ({fd.type_name}) in float32 for training on "
                f"{b} (cast it and its inputs, or exclude it from autocast); recomputing the "
                "suspect backward in float32 is the mitigation the FlashAttention 3 report used.",
                backend=b, module=fd.name)
            src = (r.blame_source or {})
            own, user = src.get("module"), src.get("user")
            if own is not None and own["user_code"]:
                add("inspect_source", f"The blamed module {fd.key} is your class "
                                      f"{own['qualname']}: check its backward (custom "
                                      "autograd.Function) or the ops its forward uses.",
                    file=own["file"], line=own["line"], backend=b)
            else:
                if user is not None:
                    add("inspect_source", f"The blamed module {fd.key} ({fd.type_name}) is a "
                                          f"library class used in your class {user['qualname']}.",
                        file=user["file"], line=user["line"], backend=b)
                add("pin_torch", f"The wrong gradient is in a library backward on {b} with {pin}: "
                                 "test other torch releases and pin one that passes.",
                    current=pin, backend=b)
        if "TS183" in r.codes:
            mods = ", ".join(f["module"] for f in r.trend.flagged)
            add("watch_grad_norm",
                f"The gradient norm on {b} drifts from float64 over the training steps in: {mods}. "
                "Rerun with more steps to confirm the trend, and recompute those modules in "
                "float32.", backend=b)
    return acts


def report_to_agent(report: GradReport, model_file: Optional[str] = None) -> Dict[str, Any]:
    from .agent import INFO, _envelope, exit_code_for

    doc = _envelope(report.status, exit_code_for(report.status))
    doc.update({
        "mode": "backward",
        "model": {"name": report.model_name,
                  "file": os.path.abspath(model_file) if model_file else None,
                  "dtype": str(report.dtype).replace("torch.", "")},
        "reference": report.reference,
        "loss": report.loss,
        "codes": report.codes,
        "issues": [i.as_dict() for i in report.all_issues],
        "backends": [_agent_backend(r) for r in report.results],
        "skipped": [{"name": n, "reason": why, "code": "TS001", "severity": INFO}
                    for n, why in report.skipped],
        "next_actions": next_actions(report),
    })
    return doc
