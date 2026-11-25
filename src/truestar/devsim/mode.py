"""Run eager PyTorch code with a device numerics profile (the `sim:*` check backends).

``SimMode`` is a ``TorchDispatchMode``. For every ATen op that reads a float tensor it:

1. rounds each float input to the dtype the device reads it in (the profile's storage dtype, its
   low precision dtype for ops such as Neuron's matmuls, or the tensor's own dtype) under the
   profile's rules, and widens it to float64;
2. computes the op in float64, with the profile's accumulator and formulation where it has one
   (:mod:`.kernels`), otherwise exactly;
3. rounds every float output to the storage dtype (or the dtype the op declares) and returns it in
   the dtype PyTorch would have returned, so the rest of the model and every comparison in
   ``truestar check`` see an ordinary tensor that holds the device's values.

View ops pass through untouched (they compute nothing and must keep aliasing); on a profile that
fuses GELU into the linear layer before it (``sim:hexagon``) a view of a linear layer's result also
carries its unrounded accumulator, and any in place write drops it. In place ops are
computed out of place on float64 copies and written back. Random ops pass through, so a model
draws the same numbers with and without the simulation. Forward only: autograd is not simulated.

This is the eager twin of :func:`truestar.train.amp.emulate_precision`, which applies the same
profile's storage rules to Truestar IR programs.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any, Dict, List, Optional, Set

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from . import hexagon as HX
from . import kernels as K
from . import tensorcore as TC
from .profiles import Profile, get_profile

F64 = torch.float64
_MATMUL = {"mm", "addmm", "bmm", "baddbmm", "mv", "dot"}
_VAR = {"var", "std", "var_mean"}
# A linear layer, as PyTorch dispatches it (2d input: addmm or mm; nd input: view, addmm, view).
# On a profile with ``fuse_linear_gelu`` their unrounded result is kept for a GELU that reads it.
_LINEAR = {"addmm", "mm"}
# Prefixes of SimMode notes that truestar check turns into their own code (docs/DEVICES.md 7).
NOTE_FITTED = "TS1140"
NOTE_UNMEASURED = "TS1141"
_HEXAGON_UNMEASURED = {
    "_softmax": "on the device softmax is 1.7x to 2.0x further from float64 than correct float16; "
                "computed here as correct float16",
    "_scaled_dot_product_flash_attention_for_cpu": "on the device attention is 1.7x to 2.0x "
                "further from float64 than correct float16; computed here as correct float16",
}
# Ops a GPU runs on its matrix units that the tensor core profiles do not route through the unit
# model (reported as TS1096 when a model uses them).
_TC_UNMODELLED = {"convolution", "_scaled_dot_product_flash_attention_for_cpu",
                  "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
                  "_scaled_mm"}
# Ops that reduce many elements but are computed exactly and rounded once even on a profile with
# a narrow or tiled accumulator. Each hit is named in the backend's notes.
_UNMODELLED = {"cumsum", "cumprod", "prod", "avg_pool1d", "avg_pool2d", "avg_pool3d",
               "_adaptive_avg_pool2d", "_adaptive_avg_pool3d", "native_group_norm",
               "native_batch_norm", "_native_batch_norm_legit", "norm", "nll_loss_forward",
               "nll_loss2d_forward", "embedding_bag", "_embedding_bag", "trace", "einsum",
               "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
               "_transformer_encoder_layer_fwd", "_native_multi_head_attention"}


def _leaves(obj: Any) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []

    def walk(o):
        if isinstance(o, torch.Tensor):
            out.append(o)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)

    walk(obj)
    return out


def _map(obj: Any, fn) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map(x, fn) for x in obj)
    if isinstance(obj, list):
        return [_map(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: _map(v, fn) for k, v in obj.items()}
    return obj


def _bind(func, args, kwargs) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    schema = func._schema.arguments
    for i, arg in enumerate(schema):
        if i < len(args) and not arg.kwarg_only:
            out[arg.name] = args[i]
        elif arg.name in kwargs:
            out[arg.name] = kwargs[arg.name]
        elif arg.has_default_value():
            out[arg.name] = arg.default_value
        else:
            out[arg.name] = None
    return out


def _is_float(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.is_floating_point()


class SimMode(TorchDispatchMode):
    """Evaluate every ATen op under ``profile`` (see the module docstring)."""

    def __init__(self, profile: Profile, notes: Optional[Set[str]] = None):
        super().__init__()
        self.p = profile
        self.notes: Set[str] = notes if notes is not None else set()
        self.count = 0
        # id(result of a linear layer) -> (weak reference to it, its unrounded float64 value)
        self._wide: Dict[int, Any] = {}

    def _keep_wide(self, t: Any, wide: Any) -> None:
        if not (isinstance(t, torch.Tensor) and isinstance(wide, torch.Tensor)):
            return
        key = id(t)
        self._wide[key] = (weakref.ref(t, lambda _r, k=key, d=self._wide: d.pop(k, None)), wide)

    def _forget_wide(self, t: Any) -> None:
        """An in place write to ``t`` makes every kept value on its storage stale."""
        if not self._wide or not isinstance(t, torch.Tensor):
            return
        ptr = t.untyped_storage().data_ptr()
        for key, (ref, _) in list(self._wide.items()):
            kept = ref()
            if kept is None or kept.untyped_storage().data_ptr() == ptr:
                self._wide.pop(key, None)

    def _wide_of(self, t: Any) -> Optional[torch.Tensor]:
        hit = self._wide.get(id(t)) if isinstance(t, torch.Tensor) else None
        return hit[1] if hit is not None and hit[0]() is t else None

    # -- helpers

    def _declared(self, func, args, kwargs) -> Optional[List[torch.dtype]]:
        """Output dtypes PyTorch would produce, from a meta run (no compute)."""
        try:
            ma, mk = _map((args, kwargs), lambda t: torch.empty_like(t, device="meta"))
            out = func(*ma, **mk)
            return [t.dtype for t in _leaves(out)]
        except Exception:
            return None

    def _widen(self, t: torch.Tensor, name: str) -> torch.Tensor:
        if not t.is_floating_point():
            return t
        p = self.p
        if p.mma and name in _MATMUL:
            # the tensor core model rounds (and for fp8 scales) the operands itself, exactly
            return t.detach().to(F64)
        dt = p.low_dtype if name in p.low_ops else (p.storage or t.dtype)
        if torch.finfo(dt).bits > torch.finfo(t.dtype).bits:
            dt = t.dtype  # a device never reads more bits than the tensor holds
        return K.rnd(t.detach().to(F64), dt, p)

    def _compute(self, name: str, func, args, kwargs, sdt: torch.dtype):
        p = self.p
        plain_acc = p.acc == "wide" and not p.tile
        stepwise = K.stepwise(p)
        has_out = any(a.name == "out" for a in func._schema.arguments)
        if not has_out:
            a = _bind(func, args, kwargs)
            got = None
            if p.mma and name in _MATMUL and "out_dtype" not in a:
                got = TC.matmul_op(name, a, p.mma, p.mma_fmt, self.notes)
            elif p.mma and name in _TC_UNMODELLED:
                self.notes.add(f"{TC.NOTE_OP}: aten.{name}: runs on tensor cores on the device "
                               f"but was computed exactly here (no unit model for it)")
            if got is not None:
                return got
            if name in _MATMUL and not plain_acc and "out_dtype" not in a:
                got = K.matmul_like(name, a, p, sdt, self.notes)
            elif name == "convolution" and not plain_acc:
                got = K.convolution(a, p, sdt, self.notes)
            elif name in ("sum", "mean") and not plain_acc and "self" in a and _is_float(a["self"]):
                got = K.reduce_sum(a, p, sdt, mean=name == "mean")
            elif name in ("_softmax", "_log_softmax") and stepwise:
                got = K.softmax(a, p, sdt, log=name == "_log_softmax")
            elif name == "logsumexp" and (stepwise or "logsumexp" in p.overrides):
                got = K.logsumexp(a, p, sdt)
            elif name == "softplus" and "softplus" in p.overrides:
                got = K.softplus(a, p, sdt)
            elif name == "mish" and "mish" in p.overrides:
                got = K.mish(a, p, sdt)
            elif name == "gelu" and "gelu_hexagon" in p.overrides:
                if a.get("approximate", "none") == "none":
                    got = HX.gelu(a["self"])
                else:
                    self.notes.add(f"{NOTE_UNMEASURED}: aten.gelu(approximate="
                                   f"'{a.get('approximate')}'): not measured on the device; "
                                   "computed as correct float16")
            elif name == "native_layer_norm" and stepwise:
                got = K.layer_norm(a, p, sdt)
            elif name in _VAR and stepwise and "self" in a:
                got = K.var_like(name, a, p, sdt)
            elif name == "linalg_vector_norm" and stepwise:
                got = K.vector_norm(a, p, sdt)
            elif name == "_scaled_dot_product_flash_attention_for_cpu" and stepwise:
                got = K.attention(a, p, sdt)
            if got is not None:
                return got
        if name in _UNMODELLED and not plain_acc:
            self.notes.add(f"aten.{name}: computed exactly and rounded once; this profile's "
                           f"accumulator is not modelled for it")
        return func(*args, **kwargs)

    # -- dispatch

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "is_view", False):
            out = func(*args, **kwargs)
            if self._wide and args:
                wide = self._wide_of(args[0])
                if wide is not None and isinstance(out, torch.Tensor):
                    try:
                        self._keep_wide(out, func(wide, *args[1:], **kwargs))
                    except Exception:  # a view the float64 copy cannot take: no fusion
                        pass
            return out
        if (torch.Tag.nondeterministic_seeded in getattr(func, "tags", ())
                or not any(_is_float(t) for t in _leaves((args, kwargs)))):
            return func(*args, **kwargs)
        name = func.overloadpacket.__name__
        declared = self._declared(func, args, kwargs)
        if declared is None:  # no meta kernel: take the dtypes from a real run
            declared = [t.dtype for t in _leaves(func(*[_map(x, torch.clone) for x in args],
                                                      **_map(kwargs, torch.clone)))]
        floats = [d for d in declared if d.is_floating_point]
        sdt = self.p.storage or (floats[0] if floats else torch.float32)
        self.count += 1
        a64 = _map(list(args), lambda t: self._widen(t, name))
        k64 = _map(dict(kwargs), lambda t: self._widen(t, name))
        ov = self.p.overrides
        if name == "gelu" and "gelu_hexagon" in ov and args:
            wide = self._wide_of(args[0]) if "fuse_linear_gelu" in ov else None
            if wide is not None:
                a64[0] = wide
                self.notes.add(f"{NOTE_FITTED}: aten.gelu: the curve fitted to Snapdragon NPU "
                               "outputs, fused with the linear layer before it")
            else:
                self.notes.add(f"{NOTE_FITTED}: aten.gelu: the curve fitted to Snapdragon NPU "
                               "outputs; its input is not a linear layer's output, so it reads "
                               "the float16 input (fusion there was not measured)")
        elif "gelu_hexagon" in ov and name in _HEXAGON_UNMEASURED:
            self.notes.add(f"{NOTE_UNMEASURED}: aten.{name}: {_HEXAGON_UNMEASURED[name]}")
        out = self._compute(name, func, a64, k64, sdt)

        it = iter(declared)

        def finish(t: torch.Tensor) -> torch.Tensor:
            d = next(it, t.dtype)
            if not (t.is_floating_point() and d.is_floating_point):
                return t.to(d) if t.dtype != d else t
            store = self.p.storage or d
            if torch.finfo(store).bits > torch.finfo(d).bits:
                store = d
            return K.rnd(t.to(F64), store, self.p).to(d)

        mutated = [i for i, arg in enumerate(func._schema.arguments)
                   if arg.alias_info is not None and arg.alias_info.is_write]
        if mutated:
            originals = []
            for i in mutated:
                arg = func._schema.arguments[i]
                orig = args[i] if i < len(args) else kwargs.get(arg.name)
                new = a64[i] if i < len(args) else k64.get(arg.name)
                if isinstance(orig, torch.Tensor) and isinstance(new, torch.Tensor):
                    self._forget_wide(orig)
                    store = self.p.storage or orig.dtype
                    if orig.is_floating_point() and torch.finfo(store).bits > torch.finfo(orig.dtype).bits:
                        store = orig.dtype
                    val = K.rnd(new.to(F64), store, self.p) if orig.is_floating_point() else new
                    orig.copy_(val.to(orig.dtype))
                    originals.append(orig)
            if isinstance(out, torch.Tensor) and originals:
                return originals[0]
            if isinstance(out, (tuple, list)) and originals:
                return type(out)(originals[: len(out)])
            return out
        result = _map(out, finish)
        if name in _LINEAR and "fuse_linear_gelu" in self.p.overrides:
            self._keep_wide(result, out)  # unrounded, for a GELU fused after this linear layer
        return result


@contextlib.contextmanager
def simulate(profile, notes: Optional[Set[str]] = None):
    """``with simulate("sim:ane"): model(x)`` runs the model with the profile's numerics."""
    p = get_profile(profile) if isinstance(profile, str) else profile
    mode = SimMode(p, notes)
    with mode:
        yield mode
