"""Run a tensor IR program on real tensors through the fastest path that can run it.

Paths, tried in order for each (program, device type):

1. Executors registered with ``register_executor`` (the seam for the compiled executor being built
   elsewhere). A factory returns a runner, or None when it cannot take the program.
2. The kernel layer (``truestar.kernel``): Triton on CUDA, numba on CPU, when every op of the
   program lowers (see ``lower.py``). Each compiled kernel is checked by the kernel layer against
   its own float64 reference; on top of that the whole program is checked against the tensor IR
   reference interpreter on the first call per input signature, so a lowering mistake cannot pass.
3. The float64 reference interpreter, with the result cast back to the declared dtype and moved
   back to the input device. Correct and slow. Choosing it emits ``SlowPathWarning`` once per
   program and device, with the reason the faster path was refused (never introduce #1).

``plan(program, device_type)`` returns the chosen ``Plan``; ``Plan.path`` and ``Plan.reason`` say
what runs and why.
"""

from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..ir import reference
from ..ir.core import Program
from .lower import NotLowerable, lower

Runner = Callable[[Sequence[torch.Tensor]], List[torch.Tensor]]
Factory = Callable[[Program, str], Optional[Runner]]


class SlowPathWarning(UserWarning):
    """A custom op runs on the float64 reference interpreter instead of compiled kernels."""


class ProgramMismatchError(AssertionError):
    """A fast path disagreed with the float64 reference interpreter. Never silently ignored."""


# float tolerance of the whole program check, by the lowest precision among the outputs
_TOL = {torch.float64: 1e-9, torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 6e-2}

_EXECUTORS: List[Tuple[str, Factory]] = []
_PLANS: Dict[Tuple[int, str], "Plan"] = {}
_LOCK = threading.Lock()


def register_executor(name: str, factory: Factory, first: bool = True) -> None:
    """Plug in another executor. ``factory(program, device_type)`` returns a runner taking the input
    tensors and returning the output tensors (declared dtypes, on the input device), or None to
    decline. Registered executors are tried before the kernel layer. Clears cached plans."""
    unregister_executor(name)
    if first:
        _EXECUTORS.insert(0, (name, factory))
    else:
        _EXECUTORS.append((name, factory))
    clear_plans()


def unregister_executor(name: str) -> None:
    _EXECUTORS[:] = [(n, f) for n, f in _EXECUTORS if n != name]
    clear_plans()


def clear_plans() -> None:
    _PLANS.clear()


def _verify_mode() -> str:
    v = os.environ.get("TRUESTAR_VERIFY", "first").lower()
    return {"0": "off", "off": "off", "all": "all"}.get(v, "first")


def _rel_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    o = out.detach().cpu().to(torch.float64)
    r = ref.detach().cpu().to(torch.float64)
    if not out.dtype.is_floating_point:
        return 0.0 if torch.equal(o, r) else float("inf")
    fin = torch.isfinite(r)
    if not torch.equal(fin, torch.isfinite(o)) or not torch.equal(r[~fin], o[~fin]):
        return float("inf")
    if not fin.any():
        return 0.0
    scale = max(float(r[fin].abs().max()), 1e-6)
    return float((o[fin] - r[fin]).abs().max()) / scale


@dataclass
class Plan:
    program: Program
    device_type: str
    path: str                      # "kernel", "reference" or a registered executor name
    reason: str                    # why this path (for the reference path: why the faster one was refused)
    runner: Runner
    calls: int = 0
    verified: set = field(default_factory=set)
    kernels: int = 0

    def __call__(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        self.calls += 1
        outs = self.runner(inputs)
        if self.path != "reference":
            mode = _verify_mode()
            sig = tuple((tuple(x.shape), x.dtype) for x in inputs)
            if mode == "all" or (mode == "first" and sig not in self.verified):
                self.check(inputs, outs)
                self.verified.add(sig)
        return outs

    def check(self, inputs, outs) -> None:
        refs = reference.run(self.program, *inputs)
        for i, (o, r, v) in enumerate(zip(outs, refs, self.program.outputs)):
            if tuple(o.shape) != tuple(r.shape) or o.dtype != v.type.dtype:
                raise ProgramMismatchError(
                    f"program {self.program.name} on the {self.path} path: output {i} is {o.dtype} "
                    f"{list(o.shape)}, the IR declares {v.type} = {list(r.shape)}")
            tol = _TOL.get(v.type.dtype, 0.0)
            err = _rel_error(o, r)
            if not err <= tol:
                raise ProgramMismatchError(
                    f"program {self.program.name} on the {self.path} path ({self.device_type}): output {i} "
                    f"differs from the float64 reference by {err:.3g} (tolerance {tol:g})")


def _reference_runner(program: Program) -> Runner:
    def run(inputs):
        dev = inputs[0].device if inputs else torch.device("cpu")
        outs = reference.run(program, *inputs)
        res = []
        ptrs = {x.data_ptr() for x in inputs if x.numel()}
        for o, v in zip(outs, program.outputs):
            y = o.to(device=dev, dtype=v.type.dtype)
            if y.numel() and y.data_ptr() in ptrs:
                y = y.clone()
            res.append(y.contiguous())
        return res
    return run


def _kernel_runner(program: Program, device_type: str) -> Tuple[Runner, int]:
    from ..ir.custom_vjp import inline_custom_vjp
    low = lower(inline_custom_vjp(program))  # custom_vjp nodes run their forward region
    backend = "cuda" if device_type == "cuda" else "cpu"
    try:
        compiled = low.graph.compile(low.outputs, backend)
    except NotLowerable:
        raise
    except Exception as e:  # Unschedulable and friends: the kernel layer refused the graph
        raise NotLowerable(f"the kernel layer refused the fused graph: {type(e).__name__}: {e}") from None

    def run(inputs):
        feed = {name: x for name, x in zip(low.input_names, inputs)}
        if low.param_inputs:
            dev = inputs[0].device
            for k, t in low.param_inputs.items():
                feed[k] = t.to(dev)
        res = compiled(**feed)
        outs = []
        for i in range(low.n_outputs):
            if i in low.passthrough:
                outs.append(inputs[low.passthrough[i]].clone())
            else:
                outs.append(res[f"out{i}"])
        return outs
    return run, compiled.num_kernels


def plan(program: Program, device_type: str, allow_kernel: bool = True) -> Plan:
    """The execution plan for ``program`` on a device type ("cpu" or "cuda"), cached per program."""
    key = (id(program), device_type, allow_kernel)
    with _LOCK:
        hit = _PLANS.get(key)
        if hit is not None and hit.program is program:
            return hit
        reasons = []
        for name, factory in _EXECUTORS:
            try:
                r = factory(program, device_type)
            except Exception as e:  # a plugged executor that crashes declines loudly in the reason
                r = None
                reasons.append(f"executor {name} failed: {type(e).__name__}: {e}")
            if r is not None:
                p = Plan(program, device_type, name, f"registered executor {name}", r)
                _PLANS[key] = p
                return p
            reasons.append(f"executor {name} declined")
        if allow_kernel and device_type in ("cpu", "cuda"):
            try:
                r, nk = _kernel_runner(program, device_type)
                p = Plan(program, device_type, "kernel", f"kernel layer, {nk} fused kernel(s)", r, kernels=nk)
                _PLANS[key] = p
                return p
            except NotLowerable as e:
                reasons.append(f"kernel layer: {e}")
        elif not allow_kernel:
            reasons.append("kernel layer disabled for this op")
        else:
            reasons.append(f"kernel layer: no backend for device type {device_type}")
        reason = "; ".join(reasons)
        warnings.warn(f"truestar custom op program {program.name} on {device_type} runs on the float64 reference "
                      f"interpreter (slow): {reason}", SlowPathWarning, stacklevel=3)
        p = Plan(program, device_type, "reference", reason, _reference_runner(program))
        _PLANS[key] = p
        return p


def execute(program: Program, inputs: Sequence[torch.Tensor], allow_kernel: bool = True) -> List[torch.Tensor]:
    """Run ``program`` on ``inputs`` (all on one device) through the fastest available path."""
    dev = inputs[0].device.type if inputs else "cpu"
    return plan(program, dev, allow_kernel)(inputs)
