"""How a remedy is applied without changing the user's code.

Two mechanisms, both scoped to one device type:

* "mode" (default): a TorchDispatchMode that sees every ATen call and, when the blamed op runs on
  the blamed device, computes it with the decomposition instead. It sits above the backend
  kernels, so it also works when the wrong result comes from Python level code such as
  truestar.testing.OpFault, as long as it is entered inside that code. Cost: Python dispatch on
  every op while active.
* "library": a torch.library kernel registered for the op on the device's dispatch key
  (CUDA, MPS, XPU, CPU). It replaces the backend kernel itself for the whole process, which is the
  right fix for a real kernel bug (for example aten.linear on MPS). It cannot fix a Python level
  fault, because a dispatch mode runs before any kernel. Cost: Python only for the replaced op.

Both run below autograd, so gradients still use the original op's derivative formula.
"""

from __future__ import annotations

import contextlib
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from .table import Decomposition, lookup

MECHANISMS = ("mode", "library")
DISPATCH_KEYS = {"cuda": "CUDA", "mps": "MPS", "xpu": "XPU", "cpu": "CPU"}


@dataclass(frozen=True)
class Remedy:
    """Replace `op` by the decomposition `decomposition` when it runs on `device_type`."""

    op: str
    device_type: str
    decomposition: str
    # A decomposition from a custom table (fix(..., table=...)); None means the built in table.
    custom: Optional[Decomposition] = field(default=None, compare=False, repr=False)

    @property
    def entry(self) -> Decomposition:
        return self.custom if self.custom is not None else lookup(self.op, self.decomposition)

    def as_dict(self) -> Dict[str, str]:
        e = self.entry
        return {"op": self.op, "device_type": self.device_type, "decomposition": self.decomposition,
                "description": e.description}

    def __str__(self) -> str:
        return f"{self.op} on {self.device_type} -> {self.decomposition} ({self.entry.description})"


def _on_device(obj: Any, device_type: str) -> bool:
    if isinstance(obj, torch.Tensor):
        return obj.device.type == device_type
    if isinstance(obj, (list, tuple)):
        return any(_on_device(x, device_type) for x in obj)
    if isinstance(obj, dict):
        return any(_on_device(v, device_type) for v in obj.values())
    return False


class RemedyMode(TorchDispatchMode):
    """Dispatch mode applying remedies. `hits` counts replaced calls per op, so a caller can prove
    the remedy fired (a remedy that never fires fixes nothing)."""

    def __init__(self, remedies: Sequence[Remedy]):
        super().__init__()
        self.remedies = list(remedies)
        self._by_op: Dict[str, List[Remedy]] = {}
        for r in self.remedies:
            self._by_op.setdefault(r.op, []).append(r)
        self._fns = {r: r.entry for r in self.remedies}
        self.hits: Counter = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        rs = self._by_op.get(str(func))
        if rs:
            for r in rs:
                if _on_device((args, kwargs), r.device_type):
                    self.hits[r.op] += 1
                    return self._fns[r](*args, **kwargs)
        return func(*args, **kwargs)


class LibraryOverride:
    """torch.library kernels for the remedies on their device dispatch keys. `remove()` restores
    the previous kernels (the dispatcher keeps them underneath)."""

    def __init__(self, remedies: Sequence[Remedy]):
        self.remedies = list(remedies)
        self.hits: Counter = Counter()
        self._lib = None

    def install(self) -> "LibraryOverride":
        if self._lib is not None:
            return self
        lib = torch.library.Library("aten", "IMPL")
        try:
            for r in self.remedies:
                e = r.entry
                if e.calls_self:
                    raise ValueError(f"{r.op} -> {r.decomposition} calls the op it replaces and "
                                     "cannot be a library kernel; use mechanism='mode'")
                key = DISPATCH_KEYS.get(r.device_type)
                if key is None:
                    raise ValueError(f"no dispatch key known for device type {r.device_type!r}")
                _, packet, overload = r.op.split(".")
                name = packet if overload == "default" else f"{packet}.{overload}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # "overriding a previously registered kernel"
                    try:  # stack on top of a kernel that was itself registered from Python
                        lib.impl(name, self._counted(r, e), key, allow_override=True)
                    except TypeError:  # torch without allow_override
                        lib.impl(name, self._counted(r, e), key)
        except Exception:
            lib._destroy()
            raise
        self._lib = lib
        return self

    def _counted(self, r: Remedy, e: Decomposition):
        def kernel(*args, **kwargs):
            self.hits[r.op] += 1
            return e(*args, **kwargs)

        return kernel

    def remove(self) -> None:
        if self._lib is not None:
            self._lib._destroy()
            self._lib = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False


# ----------------------------------------------------------------------------- public entry points


def _remedies_of(target: Any) -> List[Remedy]:
    """Remedies from a FixReport, a CheckReport (unverified first choices), a model given to
    apply(), a Remedy or a list of them."""
    if target is None:
        return []
    if isinstance(target, Remedy):
        return [target]
    if isinstance(target, nn.Module):
        return list(getattr(target, "_truestar_remedies", []))
    if hasattr(target, "remedies") and callable(getattr(target, "remedies")):
        return list(target.remedies())  # FixReport: only verified remedies
    if hasattr(target, "results") and hasattr(target, "dtype"):  # CheckReport
        from .table import candidates

        out = []
        for r in target.results:
            op = getattr(r, "first_divergent_op", None)
            if op is None:
                continue
            dt = torch.device(r.backend.device).type
            cands = candidates(op.op, dt)
            if cands:
                out.append(Remedy(op.op, dt, cands[0].name))
        if out:
            warnings.warn("truestar.fix.patched(CheckReport) applies the first table entry for "
                          "each blamed op without verifying it; run truestar.fix.fix() for a "
                          "verified remedy.", stacklevel=3)
        return out
    if isinstance(target, Iterable):
        out = []
        for t in target:
            out.extend(_remedies_of(t))
        return out
    raise TypeError(f"cannot take remedies from {type(target).__name__}")


@contextlib.contextmanager
def patched(target: Any, mechanism: str = "mode") -> Iterator[Union[RemedyMode, LibraryOverride]]:
    """Run code with remedies applied, only on their device:

        report = truestar.fix.fix(model, x)
        with truestar.fix.patched(report):
            y = model(x)

    `target` is a FixReport (its verified remedies), a CheckReport (first table entry per blamed
    op, unverified, with a warning), a model passed to apply(), a Remedy or a list of Remedies.
    Yields the mode or library override; its `hits` counter shows how often the remedy fired.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    remedies = _remedies_of(target)
    if mechanism == "mode":
        mode = RemedyMode(remedies)
        with mode:
            yield mode
    else:
        lib = LibraryOverride(remedies)
        with lib:
            yield lib


class Applied:
    """Handle returned by apply(); `remove()` undoes it."""

    def __init__(self, remedies: List[Remedy], mechanism: str, model: Optional[nn.Module],
                 handles: list, lib: Optional[LibraryOverride], state: Optional[dict]):
        self.remedies = remedies
        self.mechanism = mechanism
        self.model = model
        self._handles = handles
        self._lib = lib
        self._state = state

    @property
    def hits(self) -> Counter:
        if self._lib is not None:
            return self._lib.hits
        return self._state["hits"] if self._state else Counter()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._lib is not None:
            self._lib.remove()
        if self.model is not None and hasattr(self.model, "_truestar_remedies"):
            try:
                del self.model._truestar_remedies
            except AttributeError:
                pass


def apply(target: Any, model: Optional[nn.Module] = None, mechanism: str = "mode") -> Applied:
    """Permanent opt in, until `remove()` is called on the returned handle.

    * mechanism "mode" needs `model`: every forward of that model runs with the remedies, and
      nothing else in the process is touched. The model's code and weights are not changed.
    * mechanism "library" applies the remedies process wide as kernels on the device's dispatch
      key; `model` is optional and only recorded.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    remedies = _remedies_of(target)
    if not remedies:
        raise ValueError("nothing to apply: no verified remedy in the target")
    if mechanism == "library":
        lib = LibraryOverride(remedies).install()
        if model is not None:
            model._truestar_remedies = list(remedies)
        return Applied(remedies, mechanism, model, [], lib, None)
    if model is None:
        raise ValueError("mechanism 'mode' applies to one model: pass model=...")
    state = {"stack": [], "hits": Counter()}

    def pre(mod, args, kwargs):
        m = RemedyMode(remedies)
        m.__enter__()
        state["stack"].append(m)
        return None

    def post(mod, args, kwargs, out):
        m = state["stack"].pop()
        m.__exit__(None, None, None)
        state["hits"].update(m.hits)
        return None

    h1 = model.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = model.register_forward_hook(post, with_kwargs=True, always_call=True)
    model._truestar_remedies = list(remedies)
    return Applied(remedies, mechanism, model, [h1, h2], None, state)
