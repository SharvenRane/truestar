"""Fault injection: layers (DeviceFault) and single ATen ops (OpFault) that misbehave on one backend.

Used to prove that truestar check finds a known divergence, names the right layer, and stays
quiet on clean models. Users can drop these into their own models to validate a setup.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode


class DeviceFault(nn.Module):
    """Identity everywhere except on `device_type`, where it corrupts its input.

    kinds:
        "scale":    multiply by (1 + amount)       a wrong kernel constant
        "zero":     zero the second half of dim 1  dropped channels
        "nan":      put a NaN in the first element a numerically unsafe kernel
        "rows":     scale only rows >= `from_row`  input dependent bug (like allocator state)
        "noise":    add amount * |x| relative noise below the threshold (must NOT be flagged)
    localized kinds, each multiplying one part by (1 + amount) (-1 zeroes it, -2 negates it):
        "row":      index `from_row` of dim 0             one wrong row (a last tile, one sequence)
        "element":  the element of largest magnitude      one wrong element
        "last_col": the last index of the last dim        a wrong last column (tail handling)
        "region":   the first `from_row` of the last dim  a model's small value channels

    `device_type="lowp"` corrupts every run that is not float64, on any device, so a CPU only
    check has a faulty backend while its float64 reference stays exact.
    """

    def __init__(self, device_type: str = "cuda", kind: str = "scale", amount: float = 0.05,
                 from_row: int = 1):
        super().__init__()
        self.device_type = device_type
        self.kind = kind
        self.amount = amount
        self.from_row = from_row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.device_type == "lowp":
            if x.dtype == torch.float64:
                return x
        elif x.device.type != self.device_type:
            return x
        if self.kind in ("row", "element", "last_col", "region"):
            y = x.clone()
            f = 1.0 + self.amount
            if self.kind == "row":
                y[self.from_row] = y[self.from_row] * f
            elif self.kind == "element":
                i = int(y.detach().abs().reshape(-1).float().argmax())
                y.view(-1)[i] = y.view(-1)[i] * f
            elif self.kind == "last_col":
                y[..., -1] = y[..., -1] * f
            else:
                y[..., :self.from_row] = y[..., :self.from_row] * f
            return y
        if self.kind == "scale":
            return x * (1.0 + self.amount)
        if self.kind == "zero":
            y = x.clone()
            half = y.shape[1] // 2
            y[:, half:] = 0
            return y
        if self.kind == "nan":
            y = x.clone()
            y.view(-1)[0] = float("nan")
            return y
        if self.kind == "rows":
            y = x.clone()
            y[self.from_row:] = y[self.from_row:] * (1.0 + self.amount)
            return y
        if self.kind == "noise":
            g = torch.Generator(device=x.device).manual_seed(0)
            return x + self.amount * x.abs() * torch.randn(x.shape, generator=g, device=x.device,
                                                           dtype=x.dtype)
        raise ValueError(f"unknown fault kind {self.kind}")


def _on_device(obj, device_type: str) -> bool:
    if isinstance(obj, torch.Tensor):
        return obj.device.type == device_type
    if isinstance(obj, (list, tuple)):
        return any(_on_device(x, device_type) for x in obj)
    if isinstance(obj, dict):
        return any(_on_device(v, device_type) for v in obj.values())
    return False


class OpFault(TorchDispatchMode):
    """Context manager that corrupts one ATen op, only when it runs on `device_type`.

    Simulates a wrong backend kernel such as MPS linear with bias, below the module level:

        with OpFault("aten.addmm.default", "cuda", amount=0.5):
            report = truestar.check(model, x)

    `op` is a full overload name ("aten.addmm.default") or a packet name ("aten.addmm", every
    overload). Floating outputs are multiplied by (1 + amount); in place ops are corrupted in
    place. `hits` counts how many calls were corrupted, so a test can prove the fault fired.

    `when(args, kwargs)` limits the fault to some calls, for example batch norm in training mode
    only: ``when=lambda a, k: a[5]`` on ``aten.native_batch_norm`` (the same op runs in eval mode
    with training=False). ``target="state"`` corrupts the tensors the op writes in place (batch
    norm's running statistics) instead of its outputs. ``skip_float64`` leaves float64 calls alone,
    so a CPU fault does not also corrupt the CPU float64 reference.
    """

    def __init__(self, op, device_type: str = "cuda", amount: float = 0.5, when=None, target: str = "output",
                 skip_float64: bool = False):
        super().__init__()
        if target not in ("output", "state"):
            raise ValueError(f"target must be 'output' or 'state', got {target!r}")
        self.op = str(op)
        self.device_type = device_type
        self.amount = amount
        self.when = when
        self.target = target
        self.skip_float64 = skip_float64
        self.hits = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.op not in (str(func), str(func.overloadpacket)):
            return out
        if not _on_device((args, kwargs), self.device_type):
            return out
        if self.skip_float64 and args and isinstance(args[0], torch.Tensor) and args[0].dtype == torch.float64:
            return out
        if self.when is not None and not self.when(args, kwargs):
            return out
        self.hits += 1
        factor = 1.0 + self.amount
        if self.target == "state":
            # batch norm's schemas do not mark running_mean and running_var as written, though the
            # kernels update them in place, so those are matched by name too
            for a, t in zip(func._schema.arguments, args):
                written = (a.alias_info is not None and a.alias_info.is_write) or a.name in ("running_mean",
                                                                                                 "running_var")
                if written and isinstance(t, torch.Tensor) and t.is_floating_point():
                    t.mul_(factor)
            return out
        if func._schema.is_mutable:
            for t in ([out] if isinstance(out, torch.Tensor) else list(out or ())):
                if isinstance(t, torch.Tensor) and t.is_floating_point():
                    t.mul_(factor)
            return out
        if isinstance(out, torch.Tensor):
            return out * factor if out.is_floating_point() else out
        if isinstance(out, (tuple, list)):
            return type(out)(t * factor if isinstance(t, torch.Tensor) and t.is_floating_point()
                             else t for t in out)
        return out
