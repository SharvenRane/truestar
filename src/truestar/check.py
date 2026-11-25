"""`truestar check`: run a model on every backend and find where results diverge.

Method
------
1. Reference: the model in float64 on CPU (the closest thing to exact math PyTorch offers).
   Every module call is recorded, with its inputs copied before forward runs (safe with in place
   ops), its nesting, and how many tensor ops ran in its own code outside its submodules.
2. Calibration: each module replayed alone on CPU in the model's dtype gives the error a correct
   implementation shows. Thresholds never go below 10x that noise floor.
3. For each backend:
   * full run: every module's output is compared with the reference (accumulated error);
   * isolated replay: each leaf module, and each parent module that runs ops of its own (functional
     code such as attention math or a residual add), is run alone on the backend with the reference
     inputs. Its error is therefore its own, not inherited from earlier layers. A parent is only
     replayed when every module inside it matched, so a parent is blamed only for its own code;
   * decision check: for classifier style outputs, count per input prediction flips, separating
     near ties from real flips;
   * lean test (truestar.bias): each replayed module's own errors must not lean one way beyond
     what unbiased round to nearest allows, even when their size is within threshold (TS961).
4. Op drilldown (opdrill.py): the blamed module alone is traced at the ATen op level on the
   reference and on the backend, and the first op whose own error crosses the threshold is named
   (for example aten.addmm.default), confirmed by running that op alone with reference inputs.
5. For the blamed module, write a standalone repro script that needs only PyTorch, plus a single op
   repro when an op was named.
"""

from __future__ import annotations

import contextlib
import copy
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .backends import Backend, resolve_backends
from .capture import ROOT, Call, Recorder, flatten_tensors, map_tensors
from .codes import Issue
from .opdrill import OpResult
from .compare import (
    DIVERGENT,
    DIVERGENT_FACTOR,
    DTYPE_FLOOR,
    OK,
    SUSPECT,
    Diff,
    diff_lists,
    threshold_for,
    worst,
)


_STACK_KINDS = ("inductor", "ort", "trt", "truestar")  # Backend.kind of compiler or runtime stacks (truestar.stacks)

@dataclass
class LayerResult:
    key: str
    name: str
    type_name: str
    is_leaf: bool
    isolated: Optional[Diff]
    propagated: Optional[Diff]
    threshold: float
    status: str
    replayed: bool = True
    error: str = ""
    # Lean test of the isolated replay (truestar.bias, TS961): the worst output's LeanResult as a
    # dict, None when the module was not replayed or diverged.
    bias: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        def d(x: Optional[Diff]):
            return None if x is None else {
                "rel_err": x.rel_err,
                "max_abs": x.max_abs,
                "nonfinite_mismatch": x.nonfinite_mismatch,
                "shape_mismatch": x.shape_mismatch,
                "note": x.note,
                "local_err": x.local_err,
                "local_at": x.local_at,
            }

        return {
            "key": self.key,
            "name": self.name,
            "type": self.type_name,
            "leaf": self.is_leaf,
            "status": self.status,
            "replayed": self.replayed,
            "threshold": self.threshold,
            "isolated": d(self.isolated),
            "propagated": d(self.propagated),
            "error": self.error,
            "bias": self.bias,
        }


@dataclass
class BackendResult:
    backend: Backend
    status: str = OK
    layers: List[LayerResult] = field(default_factory=list)  # in execution (start) order
    first_divergent: Optional[LayerResult] = None
    first_divergent_op: Optional[OpResult] = None  # the ATen op inside first_divergent to blame
    output: Optional[Diff] = None
    decisions: int = 0
    flips: int = 0
    tie_flips: int = 0
    repro_path: Optional[str] = None
    op_repro_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    error: str = ""
    issues: List[Issue] = field(default_factory=list)
    # Where the blamed module's class is defined, and the nearest enclosing module defined in user
    # code (see truestar.agent.module_source). None when nothing is blamed.
    blame_source: Optional[Dict[str, Any]] = None
    # Compiler or runtime backends only (truestar.stacks): how the engine was checked and, after a
    # mismatch, localized (compiles, seconds, kernels or layers of the blamed module).
    stack: Optional[Dict[str, Any]] = None
    # Backend specific facts, for example where each node ran on WebGPU (truestar.webgpu).
    extra: Dict[str, Any] = field(default_factory=dict)
    # What each op class really computes here in the model's dtype (truestar.precision_probe, R8):
    # input significand bits, accumulator, subnormal handling, plus the flags meant to control it.
    precision: Optional[Dict[str, Any]] = None

    @property
    def codes(self) -> List[str]:
        return _unique(i.code for i in self.issues)

    def add(self, code: str, message: str) -> None:
        self.issues.append(Issue(code, message, self.backend.name))

    @property
    def checked(self) -> List[LayerResult]:
        return [l for l in self.layers if l.replayed]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.as_dict(),
            "status": self.status,
            "first_divergent": None if self.first_divergent is None else self.first_divergent.key,
            "first_divergent_op": None if self.first_divergent_op is None
            else self.first_divergent_op.as_dict(),
            "output": None if self.output is None else {
                "rel_err": self.output.rel_err,
                "max_abs": self.output.max_abs,
                "nonfinite_mismatch": self.output.nonfinite_mismatch,
                "local_err": self.output.local_err,
                "local_at": self.output.local_at,
            },
            "decisions": self.decisions,
            "flips": self.flips,
            "tie_flips": self.tie_flips,
            "repro_path": self.repro_path,
            "op_repro_path": self.op_repro_path,
            "notes": list(self.notes),
            "error": self.error,
            "codes": self.codes,
            "issues": [i.as_dict() for i in self.issues],
            "blame_source": self.blame_source,
            "stack": self.stack,
            "precision": self.precision,
            **({"extra": self.extra} if self.extra else {}),
            "layers": [l.as_dict() for l in self.layers],
        }


@dataclass
class CheckReport:
    model_name: str
    dtype: torch.dtype
    reference: str
    results: List[BackendResult]
    skipped: List[Tuple[str, str]]
    notes: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)  # report level (not tied to one backend)

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
            "model": self.model_name,
            "dtype": str(self.dtype).replace("torch.", ""),
            "reference": self.reference,
            "status": self.status,
            "skipped": [{"backend": n, "reason": r} for n, r in self.skipped],
            "notes": list(self.notes),
            "codes": self.codes,
            "issues": [i.as_dict() for i in self.issues],
            "results": [r.as_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)

    def to_text(self, top: int = 5) -> str:
        from .report import render_text

        return render_text(self, top=top)


# ----------------------------------------------------------------------------- helpers


class ReferenceFailed(RuntimeError):
    """The model could not run on the CPU reference at all (code TS005)."""


def _unique(items) -> List[str]:
    out: List[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _model_dtype(model: nn.Module) -> torch.dtype:
    for p in list(model.parameters()) + list(model.buffers()):
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def _cast_inputs(obj: Any, device: str, dtype: torch.dtype) -> Any:
    """Move tensors to device; floating tensors take dtype, integer tensors keep theirs."""

    def fn(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point():
            return t.detach().to(device=device, dtype=dtype)
        return t.detach().to(device=device)

    return map_tensors(obj, fn)


def _run(model: nn.Module, args: Tuple, kwargs: Dict, keep_inputs: bool, count_ops: bool = False,
         ctx=None):
    # A simulated backend's mode is entered first, so the recorder's hooks see its results.
    with torch.no_grad(), (ctx or contextlib.nullcontext()),             Recorder(model, keep_inputs=keep_inputs, count_ops=count_ops) as rec:
        out = model(*args, **kwargs)
    return out, rec.calls, rec.op_counting


def _module_by_name(model: nn.Module, name: str) -> nn.Module:
    return model if name == ROOT else model.get_submodule(name)


def _replay(module: nn.Module, call: Call, device: str, dtype: torch.dtype,
            ctx=None) -> List[torch.Tensor]:
    from .trainmode import active, train_outputs

    args = _cast_inputs(call.args, device, dtype)
    kwargs = _cast_inputs(call.kwargs, device, dtype)
    tm = active()
    tm_ctx = tm.replay(call.key, call.name) if tm is not None else contextlib.nullcontext()
    with torch.no_grad(), tm_ctx, (ctx or contextlib.nullcontext()):
        out = module(*args, **kwargs)
    return [t.detach().cpu() for t in flatten_tensors(out)] + train_outputs(module)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _uses_tf32(b: Backend, c: Call, op_counting: bool) -> bool:
    """TF32 tolerance only for calls that run a matmul or convolution (TF32 rounds nothing else).
    Without op counting the op names are unknown, so every call keeps the backend's setting."""
    return b.tf32 and (not op_counting or c.tf32_ops > 0)


def _needs_replay(c: Call, op_counting: bool) -> bool:
    """Leaves always; parents only when they run tensor ops of their own (or counting failed)."""
    return c.is_leaf or not op_counting or c.own_ops > 0


class NoiseFloor:
    """Error a correct CPU run shows on each module replayed alone in a given dtype, against the
    float64 reference. Computed on first use and remembered. `get` returns None where the eager
    check never calibrates (modules it does not replay) and when there is no float64 reference."""

    def __init__(self, base: nn.Module, ref_calls, op_counting: bool, dtype: torch.dtype,
                 enabled: bool = True):
        self.base, self.op_counting, self.dtype, self.enabled = base, op_counting, dtype, enabled
        self.calls = {c.key: c for c in ref_calls}
        self._memo: Dict[Tuple[str, torch.dtype], float] = {}
        self._models: Dict[torch.dtype, nn.Module] = {}

    def get(self, key: str, dtype: Optional[torch.dtype] = None,
            always: bool = False) -> Optional[float]:
        c = self.calls.get(key)
        dtype = dtype or self.dtype
        if not self.enabled or c is None:
            return None
        if not always and dtype == self.dtype and not _needs_replay(c, self.op_counting):
            return None
        k = (key, dtype)
        if k not in self._memo:
            try:
                if dtype not in self._models:
                    self._models[dtype] = copy.deepcopy(self.base).to(dtype)
                outs = _replay(_module_by_name(self._models[dtype], c.name), c, "cpu", dtype)
                self._memo[k] = diff_lists(outs, c.outputs).rel_err
            except Exception:
                self._memo[k] = float("nan")
        return self._memo[k]


def _decisions(out: torch.Tensor, ref: torch.Tensor, threshold: float) -> Tuple[int, int, int]:
    """(decisions, flips, near tie flips) for [N, C] logits; (0, 0, 0) otherwise."""
    if out.ndim != 2 or out.shape[1] < 2 or out.shape != ref.shape:
        return 0, 0, 0
    if not (out.is_floating_point() and ref.is_floating_point()):
        return 0, 0, 0
    r = ref.double()
    a = out.cpu().double()
    flipped = r.argmax(dim=1) != a.argmax(dim=1)
    top2 = r.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    scale = r.abs().amax(dim=1).clamp_min(1e-12)
    # A tie is a margin smaller than what correct math in this precision can move.
    near_tie = margin <= 2.0 * threshold * scale
    return int(out.shape[0]), int(flipped.sum()), int((flipped & near_tie).sum())


# ----------------------------------------------------------------------------- main entry


def _tensor_core_codes(res, sim: str, sim_notes: set) -> None:
    """TS1094 to TS1097 for a tensor core profile (docs/TENSOR_CORE_MODELS.md). Tensor core notes
    are taken out of ``sim_notes`` so they are not repeated under TS525."""
    from .devsim import get_profile
    from .devsim.tensorcore import NOTE_FORMAT, NOTE_OP

    p = get_profile(sim)
    if not p.mma:
        return
    if not p.validated:
        res.add("TS1094", f"{p.name}: no device of this generation was measured by Truestar; the "
                          "unit model is exactly as good as the published description it follows")
    res.add("TS1097", f"{p.name}: matmuls chain the unit's groups in K order from zero; a library "
                      "may split K, pick another instruction shape or promote the accumulator, so "
                      "the device can differ even where the unit model is exact")
    for code, prefix in (("TS1095", NOTE_FORMAT), ("TS1096", NOTE_OP)):
        hits = sorted(n for n in sim_notes if n.startswith(prefix + ":"))
        if hits:
            res.add(code, "; ".join(h[len(prefix) + 2:] for h in hits))
            res.notes.extend(hits)
            sim_notes.difference_update(hits)


def _hexagon_codes(res, sim_notes: set) -> None:
    """TS1140 and TS1141 for sim:hexagon (docs/DEVICES.md section 7), taken out of ``sim_notes`` so
    they are not repeated under TS525."""
    from .devsim.mode import NOTE_FITTED, NOTE_UNMEASURED

    for code in (NOTE_FITTED, NOTE_UNMEASURED):
        hits = sorted(n for n in sim_notes if n.startswith(code + ":"))
        if hits:
            res.add(code, "; ".join(h[len(code) + 2:] for h in hits))
            res.notes.extend(hits)
            sim_notes.difference_update(hits)


def check(
    model: nn.Module,
    inputs: Union[torch.Tensor, Sequence[Any]],
    kwargs: Optional[Dict[str, Any]] = None,
    backends: Optional[List[str]] = None,
    repro_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    drill_ops: bool = True,
    stack_options: Optional[Dict[str, Dict[str, Any]]] = None,
    train: bool = False,
    seed: int = 0,
) -> CheckReport:
    """Compare a model's results on every backend against a float64 CPU reference.

    Args:
        model: any nn.Module. It is copied, never modified, and run in eval mode, or in training
            mode with ``train=True`` (truestar.trainmode): batch norm uses batch statistics and its
            running statistic update is compared like an output, dropout masks are keyed by
            ``seed`` and module so every backend draws the same ones.
        inputs: a tensor or a sequence of positional arguments.
        kwargs: optional keyword arguments for forward.
        backends: names like ["cpu", "cuda"]; default is every backend available here.
        repro_dir: where to write a standalone repro for each divergent backend.
        drill_ops: trace the blamed module at the ATen op level to name the op at fault.
        stack_options: per stack backend name, options for its runner (test and study hooks such
            as an injected fault; see truestar.stacks). Never needed for normal use.
    """
    args = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    kwargs = dict(kwargs or {})
    name = model_name or type(model).__name__
    dtype = _model_dtype(model)
    chosen, skipped = resolve_backends(backends, allow_sim=True, allow_stacks=True)
    if train and any(b.kind in _STACK_KINDS for b in chosen):
        from .backends import BackendUnavailable

        stacks = [b.name for b in chosen if b.kind in _STACK_KINDS]
        raise BackendUnavailable(
            "training mode is not supported on compiler and runtime backends ("
            + ", ".join(stacks) + "): they compile the model in eval mode", stacks)
    notes: List[str] = []
    issues: List[Issue] = []

    from .trainmode import TrainMode, prepare

    base = prepare(model, train)
    tm = TrainMode(seed) if train else None
    with tm if tm is not None else contextlib.nullcontext():
        return _check(base, args, kwargs, name, dtype, chosen, skipped, notes, issues, repro_dir,
                      drill_ops, tm, stack_options)


def _check(base, args, kwargs, name, dtype, chosen, skipped, notes, issues, repro_dir, drill_ops, tm,
           stack_options=None):
    if tm is not None:
        notes.append("training mode: batch statistics, running statistic updates compared as outputs, keyed "
                     "dropout masks (the same on every backend).")
        tm.recording = True
    # 1. Reference run in float64 on CPU.
    reference = "CPU float64"
    ref_dtype = torch.float64
    try:
        ref_model = copy.deepcopy(base).to(torch.float64)
        if tm is not None:
            tm.attach(ref_model)
        ref_out, ref_calls, op_counting = _run(
            ref_model, _cast_inputs(args, "cpu", torch.float64),
            _cast_inputs(kwargs, "cpu", torch.float64), keep_inputs=True, count_ops=True)
    except Exception as exc:  # some ops have no float64 kernel
        reference = f"CPU {str(dtype).replace('torch.', '')}"
        ref_dtype = dtype
        issues.append(Issue(
            "TS003",
            "float64 reference failed, so the reference is the CPU run in the model's dtype; "
            f"CPU bugs cannot be detected in this mode. Reason: {type(exc).__name__}: {exc}"
        ))
        ref_model = copy.deepcopy(base)
        if tm is not None:
            tm.attach(ref_model)
        try:
            ref_out, ref_calls, op_counting = _run(
                ref_model, _cast_inputs(args, "cpu", dtype), _cast_inputs(kwargs, "cpu", dtype),
                keep_inputs=True, count_ops=True)
        except Exception as exc2:
            raise ReferenceFailed(
                f"the model failed on CPU, so there is no reference: {type(exc2).__name__}: {exc2}"
            ) from exc2
    if tm is not None:
        tm.recording = False
    if not op_counting:
        notes.append("op counting unavailable: every parent module is replayed (slower).")
    ref_flat = [t.detach().cpu() for t in flatten_tensors(ref_out)]

    # 2. Noise floor: isolated replay on CPU in the model dtype, computed when first asked for
    # (a stack backend whose whole model engine matches never needs it).
    noise = NoiseFloor(base, ref_calls, op_counting, dtype, enabled=reference == "CPU float64")

    results: List[BackendResult] = []
    for b in chosen:
        notes_b = [n for n in b.notes if dtype == torch.float32 or not n.startswith("TF32")]
        res = BackendResult(backend=b, notes=notes_b)
        from .precision_probe import manifest

        res.precision = manifest(b.device, dtype) if b.kind == "torch" else None  # stacks and WebGPU are not eager PyTorch
        try:
            if b.kind == "webgpu":
                from .webgpu.onnx_check import check_backend as webgpu_check

                webgpu_check(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise,
                             op_counting, repro_dir, name, drill_ops)
            elif b.kind == "winml":  # a Windows ML execution provider (truestar.winml)
                from .winml import check_backend as winml_check

                winml_check(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise,
                            op_counting, repro_dir, name, drill_ops)
            elif b.kind == "webnn":  # WebNN in a headless browser (truestar.webnn)
                from .webnn import check_backend as webnn_check

                webnn_check(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise,
                            op_counting, repro_dir, name, drill_ops)
            elif b.kind in _STACK_KINDS:  # a compiler or runtime stack (truestar.stacks)
                from .stackcheck import check_stack

                check_stack(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise,
                            op_counting, repro_dir, name, ref_dtype if drill_ops else None,
                            stack_options.get(b.name) if stack_options else None)
            else:
                noise_b = noise
                if b.sim and reference == "CPU float64":
                    # A simulated backend is calibrated on correct arithmetic in its own precision
                    # (truestar.devsim.Profile.ideal), so only what the device does beyond correct
                    # low precision math is blamed.
                    noise_b = _calibrate(base, ref_calls, op_counting, dtype, b.calibration_context)
                _check_backend(res, base, args, kwargs, dtype, ref_calls, ref_flat, noise_b,
                               op_counting, repro_dir, name, ref_dtype if drill_ops else None)
        except Exception as exc:
            res.status = DIVERGENT
            res.error = f"{type(exc).__name__}: {exc}"
            res.add("TS002", f"the model failed to run on {b.name}: {res.error}")
            res.notes.append("The model failed to run on this backend:\n" + traceback.format_exc(limit=3))
        results.append(res)
    if any(b.kind == "webgpu" for b in chosen):
        from .webgpu import close_workers

        close_workers()  # each worker holds a Dawn device; never keep them past the check
    if any(b.kind == "winml" for b in chosen):
        from .winml import close_workers as close_winml

        close_winml()
    if any(b.kind == "webnn" for b in chosen):
        from .webnn import close_browsers

        close_browsers()

    return CheckReport(name, dtype, reference, results, skipped, notes, issues)


def _calibrate(base: nn.Module, ref_calls, op_counting: bool, dtype: torch.dtype,
               make_ctx) -> Dict[str, float]:
    """Each replayed module's own error on the CPU in the model dtype (``make_ctx`` None), or under
    a simulated backend's ideal profile (``make_ctx`` returns its context)."""
    noise: Dict[str, float] = {}
    cpu_model = copy.deepcopy(base)
    for c in ref_calls:
        if not _needs_replay(c, op_counting):
            continue
        try:
            outs = _replay(_module_by_name(cpu_model, c.name), c, "cpu", dtype,
                           make_ctx() if make_ctx else None)
            noise[c.key] = diff_lists(outs, c.outputs).rel_err
        except Exception:
            noise[c.key] = float("nan")
    return noise


def _lean(outs, refs, thr: float) -> Optional[Dict[str, Any]]:
    """The lean test of a module's replayed outputs against the reference outputs: the result of
    the output that came closest to firing, as a dict, or None when nothing could be tested. The
    effect floor is BIAS_FRACTION of the module's threshold; the denominator floor is the rms, as
    in the local measure (compare._local). Modules are black boxes: every set of dims with at least
    MIN_GROUPS_BLACKBOX groups is a grouping."""
    from . import bias as B

    best = None
    if len(outs) != len(refs):
        return None
    for i, (o, r) in enumerate(zip(outs, refs)):
        if not (o.is_floating_point() and r.is_floating_point()) or o.shape != r.shape or o.numel() < 2:
            continue
        lr = B.lean_test(f"output {i}", o, r, thr * B.BIAS_FRACTION, 1.0)
        if best is None or (lr.biased, lr.margin()) > (best.biased, best.margin()):
            best = lr
    return None if best is None else best.as_dict()


def _report_lean(res: BackendResult, ref_by_key) -> None:
    """TS961 for every replayed module whose own errors lean and none of whose children's do (a
    parent inherits its children's lean)."""
    by_key = {l.key: l for l in res.layers}

    def leans(l):
        return l is not None and l.bias is not None and l.bias.get("biased")

    own = [l for l in res.layers if leans(l)
           and not any(leans(by_key.get(k)) for k in ref_by_key[l.key].children)]
    if not own:
        return
    res.status = worst(res.status, SUSPECT)
    parts = []
    for l in own[:5]:
        fired = [x for x in l.bias["leans"] if x["p"] < l.bias["alpha"] and abs(x["mean"]) > l.bias["floor"]]
        x = max(fired, key=lambda z: (z["direction"] == "outward", abs(z["mean"])))
        way = {("up", True): "toward plus infinity", ("up", False): "toward minus infinity",
               ("outward", True): "away from zero", ("outward", False): "toward zero"}[
            (x["direction"], x["mean"] > 0)]
        parts.append(f"{l.key} ({l.type_name}): own errors lean {way}, mean {x['mean']:.2e} per element "
                     f"(effect floor {l.bias['floor']:.1e}, threshold {l.threshold:.1e}), t {x['t']:.3g} over "
                     f"{x['groups']} groups, p {x['p']:.1e}")
    more = f"; {len(own) - 5} more" if len(own) > 5 else ""
    res.add("TS961", "errors within the size threshold but leaning one way: " + "; ".join(parts) + more)


def _check_backend(res: BackendResult, base: nn.Module, args, kwargs, dtype, ref_calls, ref_flat,
                   noise, op_counting, repro_dir, model_name, ref_dtype=None) -> None:
    from .trainmode import active

    b = res.backend
    dev_model = copy.deepcopy(base).to(b.device)
    tm = active()
    if tm is not None:
        tm.attach(dev_model)
    is_cpu = b.name == "cpu"
    ref_by_key = {c.key: c for c in ref_calls}

    sim_notes: set = set()
    ctx = (lambda: b.context(sim_notes)) if b.sim else (lambda: None)
    tdt = b.floor_dtype(dtype)  # the dtype whose tolerance applies (a simulated float16 device)

    # Full run on the backend.
    out, calls, _ = _run(dev_model, _cast_inputs(args, b.device, dtype),
                         _cast_inputs(kwargs, b.device, dtype), keep_inputs=False, ctx=ctx())
    _sync(b.device)
    got_by_key = {c.key: c for c in calls}
    if [c.key for c in ref_calls] != [c.key for c in calls]:
        missing = [c.key for c in ref_calls if c.key not in got_by_key]
        extra = [c.key for c in calls if c.key not in ref_by_key]
        res.add(
            "TS004",
            "The modules that ran differ from the reference (data dependent control flow?). "
            f"Missing: {missing[:5]}; extra: {extra[:5]}."
        )
        res.status = worst(res.status, SUSPECT)

    # Isolated replay, children before parents (ref_calls is in completion order).
    base_floor = DTYPE_FLOOR.get(tdt, 1e-4)
    bad_inside: Dict[str, bool] = {}
    results: Dict[str, LayerResult] = {}
    for c in ref_calls:
        n = None if is_cpu else noise.get(c.key)
        if n is not None and n > base_floor * DIVERGENT_FACTOR:
            n = None  # a noise floor this large means the CPU itself is off; do not hide it
        thr = threshold_for(tdt, _uses_tf32(b, c, op_counting), n)
        prop = diff_lists(got_by_key[c.key].outputs, c.outputs) if c.key in got_by_key else None
        inner_bad = any(results[k].status != OK or bad_inside.get(k, False) for k in c.children
                        if k in results)
        bad_inside[c.key] = inner_bad
        # A parent whose output diverges while everything inside it matched must be replayed even
        # if the reference saw no own ops (for example a branch that only runs on this device).
        unexplained = prop is not None and prop.status(thr) == DIVERGENT
        if inner_bad or not (_needs_replay(c, op_counting) or unexplained):
            results[c.key] = LayerResult(c.key, c.name, c.type_name, c.is_leaf, None, prop, thr, OK,
                                         replayed=False)
            continue
        iso, err, lean = None, "", None
        try:
            outs = _replay(_module_by_name(dev_model, c.name), c, b.device, dtype, ctx())
            iso = diff_lists(outs, c.outputs)
            status = iso.status(thr)
            if not c.is_leaf and status == SUSPECT:
                status = OK  # parents compound their children's precision; flag only clear errors
            if status != DIVERGENT:
                lean = _lean(outs, c.outputs, thr)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            status = DIVERGENT
        results[c.key] = LayerResult(c.key, c.name, c.type_name, c.is_leaf, iso, prop, thr, status,
                                     True, err, lean)
        res.status = worst(res.status, status)

    res.layers = sorted(results.values(), key=lambda l: ref_by_key[l.key].start)
    _report_lean(res, ref_by_key)
    bad = [l for l in res.layers if l.status != OK]
    divergent = [l for l in bad if l.status == DIVERGENT]
    res.first_divergent = divergent[0] if divergent else (bad[0] if bad else None)
    fd = res.first_divergent
    if fd is not None:
        own = fd.isolated.rel_err if fd.isolated is not None else float("nan")
        why = (fd.isolated.explain(fd.threshold) if fd.isolated is not None
               else f"error {own:.2e} vs threshold {fd.threshold:.1e}")
        others = len(bad) - 1
        res.add(
            "TS010" if fd.status == DIVERGENT else "TS011",
            f"module {fd.key} ({fd.type_name}) own {why}"
            + (f"; {others} more module(s) over threshold" if others else ""),
        )
        if fd.error:
            res.add("TS014", f"module {fd.key} raised when replayed on {b.name}: {fd.error}")
        ideal = noise.get(fd.key) if b.sim else None
        if ideal is not None and ideal == ideal and ideal > fd.threshold:
            tag = str(tdt).replace("torch.", "")
            res.notes.append(
                f"{fd.key} is over its threshold under correct {tag} arithmetic too (own error "
                f"{ideal:.1e} with {b.name}/ideal): the model does not survive {tag} on any "
                f"device, so this is not specific to {b.name}. Values outside the {tag} range "
                f"(smallest normal {torch.finfo(tdt).tiny:.1e}, largest {torch.finfo(tdt).max:.1e}) "
                "are the usual cause.")
        try:
            from .agent import module_source

            res.blame_source = module_source(base, fd.name)
        except Exception:  # source lookup is best effort and must never break a check
            res.blame_source = None

    # Final output and decisions.
    out_flat = [t.detach().cpu() for t in flatten_tensors(out)]
    res.output = diff_lists(out_flat, ref_flat)
    model_tf32 = b.tf32 and (not op_counting or any(c.tf32_ops > 0 for c in ref_calls))
    out_status = res.output.status(threshold_for(tdt, model_tf32, None))
    if out_status == DIVERGENT:
        res.status = worst(res.status, DIVERGENT)
        if res.first_divergent is None:
            res.add(
                "TS012",
                "The final output diverges but no module is to blame on its own: the cause is code "
                "outside any module, or work that only runs on this backend. Wrap that code in a "
                "module to locate it."
            )
    if out_flat and ref_flat:
        n, flips, ties = _decisions(out_flat[0], ref_flat[0], threshold_for(tdt, model_tf32, None))
        res.decisions, res.flips, res.tie_flips = n, flips, ties
        if flips - ties > 0:
            res.status = worst(res.status, SUSPECT)
            res.add("TS013", f"{flips - ties} of {n} predictions changed beyond near ties "
                             f"({ties} near ties)")

    # Op drilldown: only the blamed module, never the whole model.
    fd = res.first_divergent
    if ref_dtype is not None and fd is not None and fd.replayed and not fd.error:
        from .opdrill import drill

        c = ref_by_key[fd.key]
        try:
            ref_mod = copy.deepcopy(_module_by_name(base, c.name)).to(ref_dtype)
            tm_ctx = tm.replay(c.key, c.name) if tm is not None else contextlib.nullcontext()
            with tm_ctx:
                op, op_notes = drill(ref_mod, _module_by_name(dev_model, c.name), c.key, c.args,
                                     c.kwargs, b.device, dtype, fd.threshold, is_cpu,
                                     ctx=ctx if b.sim else None,
                                     calibration=b.calibration_context if b.sim else None,
                                     tol_dtype=tdt)
            res.first_divergent_op = op
            res.notes.extend(op_notes)
        except Exception as exc:
            res.notes.append(f"op drilldown failed on {c.key}: {type(exc).__name__}: {exc}")

    if repro_dir and fd is not None and not fd.error and tm is not None:
        res.notes.append("No repro written in training mode: a standalone repro would draw PyTorch's own "
                         "dropout masks and change the module's statistics; rerun the check with train=False "
                         "to get one for an eval mode finding.")
    elif repro_dir and fd is not None and not fd.error:
        from .repro import write_op_repro, write_repro

        c = ref_by_key[fd.key]
        try:
            res.repro_path = write_repro(
                repro_dir, model_name, b, c, _module_by_name(base, c.name), dtype, fd.threshold,
            )
        except Exception as exc:  # an unpicklable module must not hide the finding itself
            res.notes.append(f"No repro written: the module could not be saved "
                             f"({type(exc).__name__}: {exc}).")
        if res.first_divergent_op is not None:
            try:
                res.op_repro_path = write_op_repro(repro_dir, model_name, b, c,
                                                   res.first_divergent_op)
            except Exception as exc:
                res.notes.append(f"No op repro written ({type(exc).__name__}: {exc}).")
    if b.sim:
        res.add("TS520", f"{b.name} is a CPU simulation of the device's numerics; confirm a "
                         "finding on the device itself")
        _tensor_core_codes(res, b.sim, sim_notes)
        _hexagon_codes(res, sim_notes)
        if sim_notes:
            res.add("TS525", "; ".join(sorted(sim_notes)))
            res.notes.extend(sorted(sim_notes))
