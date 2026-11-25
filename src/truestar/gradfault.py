"""Fault injection for backward passes: an ATen op that is wrong only while autograd runs backward.

Used to prove that `truestar check --backward` finds a backward only bug (correct forward, wrong
gradient, no NaN) and names the module and the op, while the forward check reports OK:

    with BackwardOpFault("aten.mm.default", "cuda", amount=0.5):
        report = truestar.grad_check(model, x)

The fault fires only inside a backward pass (autograd's graph task id is set), only when the op
runs on `device_type`, and never on float64 tensors, so the float64 CPU reference stays exact even
when the backend under test is the CPU. `node` narrows it to one autograd node (for example
"SoftmaxBackward0"). `ramp` adds that much more corruption per backward pass, to simulate an error
that grows slowly over a training run.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _floats(obj, out):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _floats(x, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _floats(v, out)
    return out


def in_backward() -> bool:
    """True while the autograd engine is running a backward pass on this thread."""
    return torch._C._current_graph_task_id() != -1


class BackwardOpFault(TorchDispatchMode):
    """Corrupt one ATen op, only in backward, only on `device_type`, never in float64.

    kinds:
        "scale":    output * (1 + amount)                       a wrong constant
        "row_leak": for aten._softmax_backward_data, output + amount * y * rowsum(y * g): a
                    softmax backward that leaks its row sum term (arXiv 2609.34272); for other
                    ops, output + amount * mean |row|
    `hits` counts corrupted calls and `passes` the backward passes seen, so a test can prove the
    fault fired.
    """

    def __init__(self, op, device_type: str = "cuda", amount: float = 0.5, kind: str = "scale",
                 node: Optional[str] = None, ramp: float = 0.0):
        super().__init__()
        if kind not in ("scale", "row_leak"):
            raise ValueError(f"unknown backward fault kind {kind}")
        self.op = str(op)
        self.device_type = device_type
        self.amount = amount
        self.kind = kind
        self.node = node
        self.ramp = ramp
        self.hits = 0
        self.passes = 0
        self._last_task = None

    def _corrupt(self, t: torch.Tensor, amount: float, func=None, args=()) -> torch.Tensor:
        if self.kind == "scale":
            return t * (1.0 + amount)
        if func is not None and str(func.overloadpacket) == "aten._softmax_backward_data":
            # grad_in = y * (g - rowsum(y * g)); a leaked row sum adds amount * y * rowsum(y * g).
            g, y, dim = args[0], args[1], args[2]
            return t + amount * y * (g * y).sum(dim=dim, keepdim=True)
        if t.ndim == 0:
            return t + amount * t
        return t + amount * t.abs().mean(dim=-1, keepdim=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.op not in (str(func), str(func.overloadpacket)) or not in_backward():
            return out
        fl = _floats((args, kwargs), [])
        if not fl or any(t.dtype == torch.float64 for t in fl):
            return out
        if not any(t.device.type == self.device_type for t in fl):
            return out
        if self.node is not None:
            n = torch._C._current_autograd_node()
            if n is None or type(n).__name__ != self.node:
                return out
        task = torch._C._current_graph_task_id()
        if task != self._last_task:
            self._last_task = task
            self.passes += 1
        amount = self.amount + self.ramp * (self.passes - 1)
        self.hits += 1
        if isinstance(out, torch.Tensor):
            return self._corrupt(out, amount, func, args) if out.is_floating_point() else out
        if isinstance(out, (tuple, list)):
            return type(out)(self._corrupt(t, amount)
                             if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                             for t in out)
        return out
