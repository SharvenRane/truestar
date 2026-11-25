"""Record every module call during a forward pass, with nesting and own op counts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

ROOT = "<model>"


def flatten_tensors(obj: Any) -> List[torch.Tensor]:
    """All tensors inside nested tuples, lists and dicts, in a stable order (depth first, dict keys
    sorted by their text).

    An explicit stack, not a recursive inner function: a nested function that calls itself refers
    to itself through its closure, a reference cycle that also holds the result list, so every
    tensor flattened here stayed alive until the garbage collector ran."""
    out: List[torch.Tensor] = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            stack.extend(reversed(o))
        elif isinstance(o, dict):
            stack.extend(o[k] for k in reversed(sorted(o.keys(), key=str)))
        elif hasattr(o, "to_tuple") and callable(o.to_tuple):  # Hugging Face ModelOutput
            stack.append(o.to_tuple())
    return out


def map_tensors(obj: Any, fn) -> Any:
    """Apply fn to every tensor inside nested containers, keeping the structure."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(map_tensors(x, fn) for x in obj)
    if isinstance(obj, list):
        return [map_tensors(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: map_tensors(v, fn) for k, v in obj.items()}
    return obj


def _detach_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", copy=True)


@dataclass
class Call:
    key: str  # "name#index"
    name: str
    type_name: str
    index: int
    start: int  # order in which calls started
    depth: int
    parent: Optional[str]  # key of the enclosing call
    is_leaf: bool
    args: Any = None  # CPU copies taken BEFORE forward (safe with in place ops); kept on request
    kwargs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[torch.Tensor] = field(default_factory=list)
    own_ops: int = 0  # tensor ops run in this module's own code, outside its submodules
    # Matmul and convolution ops run anywhere inside this call, submodules included: the ops TF32
    # applies to. A call with none gets no TF32 tolerance.
    tf32_ops: int = 0
    children: List[str] = field(default_factory=list)


class Recorder:
    """Context manager that records inputs (before forward) and outputs of every module call.

    `calls` is in completion order, so children always come before their parent.
    """

    def __init__(self, model: nn.Module, keep_inputs: bool = False, count_ops: bool = False):
        self.model = model
        self.keep_inputs = keep_inputs
        self.count_ops = count_ops
        self.calls: List[Call] = []
        self._counts: Dict[str, int] = {}
        self._stack: List[Call] = []
        self._handles = []
        self._starts = 0
        self.in_hook = False
        self._mode = None
        self.op_counting = False

    def __enter__(self) -> "Recorder":
        for name, mod in self.model.named_modules():
            label = name or ROOT
            leaf = next(iter(mod.children()), None) is None
            self._handles.append(mod.register_forward_pre_hook(self._pre(label, leaf), with_kwargs=True))
            self._handles.append(mod.register_forward_hook(self._post(label), with_kwargs=True))
        if self.count_ops:
            self._mode = _make_counter(self)
            if self._mode is not None:
                self._mode.__enter__()
                self.op_counting = True
        return self

    def __exit__(self, *exc) -> None:
        if self._mode is not None:
            self._mode.__exit__(*exc)
            self._mode = None
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _pre(self, name: str, leaf: bool):
        def hook(module, args, kwargs):
            self.in_hook = True
            try:
                idx = self._counts.get(name, 0)
                self._counts[name] = idx + 1
                parent = self._stack[-1] if self._stack else None
                call = Call(
                    key=f"{name}#{idx}", name=name, type_name=type(module).__name__, index=idx,
                    start=self._starts, depth=len(self._stack),
                    parent=parent.key if parent else None, is_leaf=leaf,
                )
                self._starts += 1
                if self.keep_inputs:
                    call.args = map_tensors(args, _detach_cpu)
                    call.kwargs = map_tensors(kwargs, _detach_cpu)
                if parent is not None:
                    parent.children.append(call.key)
                self._stack.append(call)
            finally:
                self.in_hook = False
            return None

        return hook

    def _post(self, name: str):
        def hook(module, args, kwargs, output):
            self.in_hook = True
            try:
                call = self._stack.pop()
                call.outputs = [_detach_cpu(t) for t in flatten_tensors(output)]
                from .trainmode import train_outputs   # training mode: the state after the call too

                call.outputs += train_outputs(module)
                self.calls.append(call)
            finally:
                self.in_hook = False
            return None

        return hook


# ATen ops that run as GEMMs or convolutions, the math TF32 rounds on CUDA and ROCm.
_TF32_OPS = {
    "mm", "addmm", "bmm", "baddbmm", "addbmm", "matmul", "linear", "_addmm_activation",
    "convolution", "_convolution", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
    "conv_transpose2d", "conv_transpose3d", "cudnn_convolution", "cudnn_convolution_transpose",
    "mkldnn_convolution", "miopen_convolution", "_scaled_dot_product_attention_math",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu",
    "scaled_dot_product_attention", "einsum", "tensordot", "_int_mm",
}


def is_tf32_op(func) -> bool:
    """True for a matmul or convolution ATen op (an OpOverload such as aten.addmm.default)."""
    try:
        name = str(func.overloadpacket).split(".")[-1]
    except Exception:
        name = str(func).split(".")[1] if "." in str(func) else str(func)
    return name in _TF32_OPS


def _make_counter(rec: Recorder):
    """A dispatch mode that credits each tensor op to the innermost running module."""
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:  # pragma: no cover - very old torch
        return None

    class OwnOpCounter(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if not rec.in_hook and rec._stack:
                rec._stack[-1].own_ops += 1
                if is_tf32_op(func):
                    for c in rec._stack:
                        c.tf32_ops += 1
            return func(*args, **(kwargs or {}))

    return OwnOpCounter()
