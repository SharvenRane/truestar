"""Op level drilldown: name the exact ATen op inside a blamed module.

Method
------
Module level blame says *which module* is wrong. A PyTorch maintainer needs *which op*. Once a
module is blamed, only that module is replayed twice under a TorchDispatchMode that records every
ATen op call (name, inputs, outputs):

1. on the reference (CPU float64), with the module's reference inputs;
2. on the backend, with the same inputs cast to the model dtype.

The two op sequences are aligned by op name (in order; ops that only one side runs, such as a
backend specific fused kernel, stay unmatched). Walking the backend's ops in order, an op whose
accumulated error crosses the threshold (or that has no reference counterpart) is re-run alone on
the backend with reference inputs, exactly as modules are isolated. Its error is then its own. The
first op whose own error crosses the threshold is reported. As with modules, the threshold never
goes below 10x the error the same op shows when run alone on CPU in the same dtype, so ill
conditioned ops (cancellation, argmax near ties) are not blamed for float noise.

Cost: only the blamed module is traced, and only ops that look wrong are re-run.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from .compare import DIVERGENT, OK, Diff, diff_lists, threshold_for

MAX_OPS = 5000  # a blamed module with more ops than this is not drilled (cost guard)

# Ops whose output is uninitialized memory: comparing them would raise false alarms.
_UNINITIALIZED = {
    "aten.empty", "aten.empty_like", "aten.empty_strided", "aten.new_empty",
    "aten.new_empty_strided", "aten.resize_", "aten.set_",
}


# ----------------------------------------------------------------------------- containers


def _leaves(obj: Any) -> List[torch.Tensor]:
    """Tensors in obj, in the same order `_replace` visits them."""
    out: List[torch.Tensor] = []

    def walk(o: Any) -> None:
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)

    walk(obj)
    return out


def _map(obj: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map(x, fn) for x in obj)
    if isinstance(obj, list):
        return [_map(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: _map(v, fn) for k, v in obj.items()}
    return obj


def _retarget(obj: Any, device: str) -> Any:
    """Move tensors to device and point device arguments (factory ops) at it."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, torch.device):
        return torch.device(device)
    if isinstance(obj, tuple):
        return tuple(_retarget(x, device) for x in obj)
    if isinstance(obj, list):
        return [_retarget(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _retarget(v, device) for k, v in obj.items()}
    return obj


def _cpu_copy(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


# ----------------------------------------------------------------------------- tracing


@dataclass
class OpRecord:
    name: str  # e.g. "aten.addmm.default"
    func: Any  # the OpOverload
    args: Any  # CPU copies taken before the op ran (safe with in place ops)
    kwargs: Dict[str, Any]
    outputs: List[torch.Tensor]  # CPU copies


def _make_tracer(limit: int):
    from torch.utils._python_dispatch import TorchDispatchMode

    class OpTracer(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops: List[OpRecord] = []
            self.truncated = False

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if len(self.ops) >= limit:
                self.truncated = True
                return func(*args, **kwargs)
            a, k = _map(args, _cpu_copy), _map(kwargs, _cpu_copy)
            out = func(*args, **kwargs)
            self.ops.append(OpRecord(str(func), func, a, k, [_cpu_copy(t) for t in _leaves(out)]))
            return out

    return OpTracer()


def trace_ops(module: nn.Module, args: Tuple, kwargs: Dict, limit: int = MAX_OPS, ctx=None):
    """Run module(*args, **kwargs) and return (op records, truncated). ``ctx`` (a simulated
    backend's mode) is entered before the tracer, so the tracer records the ops the module calls
    and the values the simulation gives them."""
    import contextlib

    tracer = _make_tracer(limit)
    with torch.no_grad(), (ctx or contextlib.nullcontext()), tracer:
        module(*args, **kwargs)
    return tracer.ops, tracer.truncated


def _comparable(r: OpRecord) -> bool:
    if not r.outputs:
        return False
    packet = str(r.func.overloadpacket)
    if packet in _UNINITIALIZED:
        return False
    return torch.Tag.nondeterministic_seeded not in getattr(r.func, "tags", ())


# ----------------------------------------------------------------------------- result


@dataclass
class OpResult:
    op: str  # e.g. "aten.addmm.default"
    module: str  # key of the blamed module, e.g. "fc1#0"
    index: int  # 1 based position among the backend's compared ops in that module
    total: int
    input_shapes: List[List[int]]
    input_dtypes: List[str]
    isolated: Diff  # op alone on the backend with reference inputs
    accumulated: Optional[Diff]  # op output inside the module run (None if no reference counterpart)
    threshold: float
    status: str
    matched: bool  # the reference ran the same op at the same place
    # Saved for the op repro: arguments as run on the backend (CPU copies) and the float64 answer.
    args: Any = field(default=None, repr=False, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    reference: List[torch.Tensor] = field(default_factory=list, repr=False, compare=False)
    # Compiler backends (truestar.stackcheck): the fused kernel computing this op, when known.
    kernel: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        def d(x: Optional[Diff]):
            return None if x is None else {
                "rel_err": x.rel_err, "max_abs": x.max_abs,
                "nonfinite_mismatch": x.nonfinite_mismatch, "shape_mismatch": x.shape_mismatch,
            }

        return {
            "op": self.op,
            "module": self.module,
            "index": self.index,
            "total": self.total,
            "input_shapes": self.input_shapes,
            "input_dtypes": self.input_dtypes,
            "status": self.status,
            "threshold": self.threshold,
            "isolated": d(self.isolated),
            "accumulated": d(self.accumulated),
            "matched": self.matched,
            "kernel": self.kernel,
        }


# ----------------------------------------------------------------------------- isolation


def _with_reference_tensors(dev: OpRecord, ref: OpRecord):
    """The backend call's arguments with every tensor replaced by the reference's, in the
    backend's dtype. Non tensor arguments (device, dtype, sizes) stay the backend's own."""
    dl, rl = _leaves((dev.args, dev.kwargs)), _leaves((ref.args, ref.kwargs))
    if len(dl) != len(rl) or any(a.shape != b.shape for a, b in zip(dl, rl)):
        return None
    it = iter(rl)
    return _map((dev.args, dev.kwargs), lambda t: next(it).to(dtype=t.dtype))


def _linear_backward_cpu(x, grad_output, weight, output_mask):
    """aten.linear_backward (MPS only: MPS runs a linear layer's backward as one op) in terms of
    ops every device has, so it can be checked against a float64 CPU run."""
    go = grad_output.reshape(-1, grad_output.shape[-1])
    gi = grad_output @ weight if output_mask[0] else None
    gw = go.t() @ x.reshape(-1, x.shape[-1]) if output_mask[1] else None
    gb = go.sum(0) if output_mask[2] else None
    return gi, gw, gb


# ops with no CPU kernel, as the same math in ops the CPU has
_CPU_FORMS = {"aten.linear_backward.default": _linear_backward_cpu}


def _call(func, args, kwargs, device: str, ctx=None) -> List[torch.Tensor]:
    import contextlib

    a, k = _retarget(_map((args, kwargs), torch.clone), device)
    fn = _CPU_FORMS.get(str(func), func) if str(device).startswith("cpu") else func
    with torch.no_grad(), (ctx or contextlib.nullcontext()):
        out = fn(*a, **k)
    return [_cpu_copy(t) for t in _leaves(out)]


def _upcast(obj: Any) -> Any:
    return _map(obj, lambda t: t.to(torch.float64) if t.is_floating_point() else t)


def _isolate(dev: OpRecord, ref: Optional[OpRecord], device: str, is_cpu: bool, base_thr: float,
             dtype: torch.dtype, ctx=None, calibration=None, tol_dtype=None):
    """Run one op alone on the backend with reference inputs. None when it cannot be checked."""
    inputs = _with_reference_tensors(dev, ref) if ref is not None else None
    if inputs is not None:
        args, kwargs = inputs
        reference = ref.outputs
    else:
        # No reference counterpart (or different arguments): use the backend's own inputs and
        # compute the answer for exactly those inputs on CPU in float64.
        args, kwargs = dev.args, dev.kwargs
        try:
            a64, k64 = _upcast((args, kwargs))
            reference = _call(dev.func, a64, k64, "cpu")
        except Exception:
            return None
    try:
        got = _call(dev.func, args, kwargs, device, ctx() if ctx else None)
    except Exception:
        return None  # e.g. a stride dependent op that rejects a copied input: not checkable
    iso = diff_lists(got, reference)
    thr = base_thr
    if not is_cpu:
        try:
            # Calibrate on the CPU in the model dtype, or for a simulated backend on correct
            # arithmetic in its own precision.
            noise = diff_lists(_call(dev.func, args, kwargs, "cpu",
                                     calibration() if calibration else None), reference).rel_err
            thr = max(base_thr, threshold_for(tol_dtype or dtype, False, noise))
        except Exception:
            pass  # no CPU kernel in this dtype: keep the module threshold
    return iso, thr, args, kwargs, reference


def _align(ref_names: List[str], dev_names: List[str]) -> Dict[int, int]:
    """dev index -> ref index for ops both sides ran at the same place in the sequence."""
    sm = difflib.SequenceMatcher(None, ref_names, dev_names, autojunk=False)
    pairs: Dict[int, int] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs[j1 + k] = i1 + k
    return pairs


# ----------------------------------------------------------------------------- dtype dependent code

_NON_CONSTANT = (torch.dtype, torch.device, torch.layout, torch.memory_format)


def _special_floats() -> Dict[float, str]:
    """Float constants that legitimately change with the dtype (finfo limits), mapped to a name."""
    out: Dict[float, str] = {}
    for dt in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        fi = torch.finfo(dt)
        for name in ("min", "max", "eps", "tiny"):
            out[float(getattr(fi, name))] = "finfo." + name
    return out


def _constants(r: OpRecord) -> List[Any]:
    """The non tensor arguments of one op call (scalars, sizes, flags), in order. Dtype, device
    and layout arguments are left out since they follow the dtype by design."""
    out: List[Any] = []

    def walk(o: Any) -> None:
        if isinstance(o, torch.Tensor) or o is None or isinstance(o, _NON_CONSTANT):
            return
        if isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for key in sorted(o):
                out.append(("kw", key))
                walk(o[key])
        else:
            out.append(o)

    walk((r.args, r.kwargs))
    return out


def _same_constant(a: Any, b: Any, tol: float, special: Dict[float, str]) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return type(a) == type(b) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    a, b = float(a), float(b)
    if a == b or (a != a and b != b):
        return True
    if a in special and b in special:
        return special[a] == special[b]
    return abs(a - b) <= tol * max(abs(a), abs(b))


def code_difference(ref_ops: List[OpRecord], low_ops: List[OpRecord],
                    dtype: torch.dtype) -> Optional[str]:
    """Where the module's own code differs between the float64 reference run and a CPU run in the
    model dtype: a different op sequence, or the same op called with different constants. Both
    runs are on CPU, so a difference can only come from the module's code branching on the dtype.
    None when the two runs execute the same ops with the same constants."""
    ref_names = [r.name for r in ref_ops]
    low_names = [r.name for r in low_ops]
    if ref_names != low_names:
        sm = difflib.SequenceMatcher(None, ref_names, low_names, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                return (f"op {j1 + 1}: the float64 run calls "
                        f"{', '.join(ref_names[i1:i2][:3]) or 'nothing'} where the "
                        f"{str(dtype).replace('torch.', '')} run calls "
                        f"{', '.join(low_names[j1:j2][:3]) or 'nothing'}")
    tol = min(1e-2, max(1e-6, 100 * torch.finfo(dtype).eps)) if dtype.is_floating_point else 1e-6
    special = _special_floats()
    for i, (r, l) in enumerate(zip(ref_ops, low_ops)):
        rc, lc = _constants(r), _constants(l)
        if len(rc) != len(lc) or not all(_same_constant(a, b, tol, special)
                                         for a, b in zip(rc, lc)):
            return (f"op {i + 1} ({r.name}) takes constants {rc[:4]} in the float64 run but "
                    f"{lc[:4]} in the {str(dtype).replace('torch.', '')} run")
    return None


def drill(ref_module: nn.Module, dev_module: nn.Module, module_key: str, args: Tuple,
          kwargs: Dict, device: str, dtype: torch.dtype, threshold: float, is_cpu: bool,
          limit: int = MAX_OPS, ctx=None, calibration=None,
          tol_dtype=None) -> Tuple[Optional[OpResult], List[str]]:
    """Find the first ATen op inside one module whose own result diverges on `device`.

    `args`/`kwargs` are the module's reference inputs (CPU). `ref_module` runs on CPU in the
    reference precision, `dev_module` on `device` in `dtype`. Returns (op or None, notes).
    """
    from .check import _cast_inputs  # local import: check imports this module

    notes: List[str] = []
    clone = lambda o: _map(o, torch.clone)  # modules with in place ops must not touch saved inputs
    ref_ops, t1 = trace_ops(ref_module, clone(args), clone(kwargs), limit)
    dev_ops, t2 = trace_ops(dev_module, _cast_inputs(clone(args), device, dtype),
                            _cast_inputs(clone(kwargs), device, dtype), limit,
                            ctx() if ctx else None)
    if t1 or t2:
        notes.append(f"op drilldown skipped for {module_key}: more than {limit} ops.")
        return None, notes
    # A module that branches on its input dtype (a "fast" float32 path) runs different code in the
    # float64 reference than in the model dtype. Comparing those op by op would blame a correct
    # stock op for the module's own math, so compare the code paths first, both on CPU.
    low_ops: Optional[List[OpRecord]] = dev_ops if is_cpu else None
    if low_ops is None:
        try:
            import copy

            low_mod = copy.deepcopy(ref_module).to(dtype)
            low_ops, t3 = trace_ops(low_mod, _cast_inputs(clone(args), "cpu", dtype),
                                    _cast_inputs(clone(kwargs), "cpu", dtype), limit)
            if t3:
                low_ops = None
        except Exception:
            low_ops = None  # no CPU kernel in this dtype: the code path check is skipped
    if low_ops is not None:
        where = code_difference(ref_ops, low_ops, dtype)
        if where is not None:
            notes.append(
                f"op drilldown on {module_key}: the module's own code differs by dtype, so no "
                f"stock op is blamed and no op repro is written. The float64 reference and the "
                f"{str(dtype).replace('torch.', '')} run take different code paths ({where}). "
                "Look for a branch on the dtype in this module's code.")
            return None, notes
    ref_ops = [r for r in ref_ops if _comparable(r)]
    dev_ops = [r for r in dev_ops if _comparable(r)]
    pairs = _align([r.name for r in ref_ops], [r.name for r in dev_ops])

    suspect: Optional[OpResult] = None
    for i, d in enumerate(dev_ops):
        r = ref_ops[pairs[i]] if i in pairs else None
        acc = diff_lists(d.outputs, r.outputs) if r is not None else None
        if acc is not None and acc.status(threshold) == OK:
            continue  # this op's output matches: nothing to blame here
        got = _isolate(d, r, device, is_cpu, threshold, dtype, ctx, calibration, tol_dtype)
        if got is None:
            continue
        iso, thr, a, k, reference = got
        status = iso.status(thr)
        if status == OK:
            continue  # error inherited from an earlier op, not its own
        leaves = _leaves((a, k))
        res = OpResult(
            op=d.name, module=module_key, index=i + 1, total=len(dev_ops),
            input_shapes=[list(t.shape) for t in leaves],
            input_dtypes=[str(t.dtype).replace("torch.", "") for t in leaves],
            isolated=iso, accumulated=acc, threshold=thr, status=status, matched=r is not None,
            args=a, kwargs=k, reference=reference,
        )
        if status == DIVERGENT:
            return res, notes
        suspect = suspect or res
    if suspect is not None:
        return suspect, notes
    matched_ref = set(pairs.values())
    only_here = sorted({d.name for i, d in enumerate(dev_ops) if i not in pairs})
    only_ref = sorted({r.name for j, r in enumerate(ref_ops) if j not in matched_ref})
    msg = (f"op drilldown on {module_key}: no single ATen op was confirmed wrong for its own "
           "inputs")
    if only_here or only_ref:
        msg += (", and the module runs different ops on this backend, so its own code (for "
                "example a device specific branch) is the likely cause. Ops only here: "
                + (", ".join(only_here[:5]) or "none") + "; only in the reference: "
                + (", ".join(only_ref[:5]) or "none"))
    notes.append(msg + ".")
    return None, notes
