"""The three IR ops training needs and the IR does not have yet.

* ``cast(x) dtype=...``: change the dtype. Mixed precision is impossible without it. The reference
  interpreter computes every float in float64, so its kernel rounds through the target dtype and
  back: ``cast(x, f16)`` returns float64 numbers that are exactly representable in float16, with
  overflow to infinity where float16 overflows. A real backend just converts.
* ``log_softmax(x) dim=...``: ``x - max - log(sum(exp(x - max)))``, the numerically stable form.
  The IR has no ``log`` and no max reduction, and composing them would differentiate through the
  max shift for nothing, so the stable form is one op with its own gradient rule.
* ``all_reduce(x) reduce="mean"|"sum"``: the gradient all reduce of data parallel training. Its kernel
  calls ``torch.distributed.all_reduce`` (looked up at call time, so a CollectiveRecorder that
  patched it records every call). It has no gradient rule; it is only ever applied to gradients.

Shape rules and reference kernels are registered in the IR's own registries (``shapes.RULES``,
``reference.IMPLS``) the way ``grad_rules.py`` registers ``erf``, and only when the IR does not
already define the op. The gradient rules of ``cast`` and ``log_softmax`` are kept here, in
``TRAIN_GRAD_RULES``, and installed only for the duration of a ``grad`` call made by this package
(see :func:`grad`), so the global gradient rule table that other tests enumerate is unchanged.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, List

import torch

from ..ir import grad_rules as _grad_rules  # noqa: F401  registers erf, which gradient programs use
from ..ir import reference, shapes
from ..ir.core import IRError, ShapeError, TensorType

NEW_OPS = ("cast", "log_softmax", "all_reduce")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------- shape rules


def _cast_rule(types, attrs):
    (t,) = types
    dtype = attrs.get("dtype")
    if not isinstance(dtype, torch.dtype):
        raise ShapeError("cast", f"attribute dtype must be a torch.dtype, got {dtype!r}", [("x", t)])
    return TensorType(dtype, t.shape)


def _log_softmax_rule(types, attrs):
    (t,) = types
    if not t.dtype.is_floating_point:
        raise ShapeError("log_softmax", "needs a floating point input", [("x", t)])
    return shapes.RULES["softmax"](types, attrs)


def _all_reduce_rule(types, attrs):
    (t,) = types
    if attrs.get("reduce") not in ("mean", "sum"):
        raise ShapeError("all_reduce", f"reduce must be 'mean' or 'sum', got {attrs.get('reduce')!r}", [("x", t)])
    if not t.dtype.is_floating_point:
        raise ShapeError("all_reduce", "needs a floating point input", [("x", t)])
    return t


# ----------------------------------------------------------------------------- reference kernels


def _cast_impl(attrs, args, shape):
    x = args[0]
    dtype = attrs["dtype"]
    if dtype.is_floating_point:
        # round through the target dtype, then back to the interpreter's float64
        return x.to(dtype).to(reference.COMPUTE_DTYPE)
    return x.to(dtype)


def _log_softmax_impl(attrs, args, shape):
    x = args[0]
    d = attrs["dim"]
    shifted = x - x.amax(dim=d, keepdim=True)
    return shifted - torch.log(torch.exp(shifted).sum(dim=d, keepdim=True))


def _all_reduce_impl(attrs, args, shape):
    import torch.distributed as dist

    from ..diag import collectives

    for sub in ("train", "ir", "frontend"):  # record the user's line that ran the step, not ours
        collectives.skip_frames_from(os.path.join(_PKG, sub))

    if not (dist.is_available() and dist.is_initialized()):
        raise IRError("all_reduce needs torch.distributed: call init_process_group on every rank before "
                      "running a data parallel training step")
    t = args[0].clone()
    dist.all_reduce(t)  # attribute lookup at call time: a CollectiveRecorder sees this call
    if attrs["reduce"] == "mean":
        t = t / dist.get_world_size()
    return t


_SHAPES = {"cast": _cast_rule, "log_softmax": _log_softmax_rule, "all_reduce": _all_reduce_rule}
_IMPLS = {"cast": _cast_impl, "log_softmax": _log_softmax_impl, "all_reduce": _all_reduce_impl}

for _name in NEW_OPS:
    if _name not in shapes.RULES:
        shapes.RULES[_name] = _SHAPES[_name]
    if _name not in reference.IMPLS:
        reference.IMPLS[_name] = _IMPLS[_name]


# ----------------------------------------------------------------------------- gradient rules


def _grad_cast(e, n, g):
    """y = cast(x): the cotangent goes back in the dtype of x (torch's autocast does the same)."""
    (x,) = n.inputs
    if not x.type.dtype.is_floating_point:
        return [None]
    return [lambda: e.op("cast", g, dtype=x.type.dtype)]


def _grad_log_softmax(e, n, g):
    """y = log_softmax(x): x_bar = g - exp(y) sum(g) along the dim."""
    (x,) = n.inputs
    y = n.output

    def fn():
        d = n.attrs["dim"] % max(x.type.rank, 1)
        total = e.op("sum", g, dims=[d], keepdim=True)
        return e.op("sub", g, e.op("mul", e.op("exp", y), total))
    return [fn]


TRAIN_GRAD_RULES: Dict[str, Any] = {"cast": _grad_cast, "log_softmax": _grad_log_softmax}


@contextlib.contextmanager
def training_rules():
    """Install the gradient rules of ``cast`` and ``log_softmax`` for the duration of the block."""
    grad_rules = _grad_rules
    added: List[str] = []
    for name, fn in TRAIN_GRAD_RULES.items():
        if name not in grad_rules.GRAD_RULES:
            grad_rules.GRAD_RULES[name] = fn
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            grad_rules.GRAD_RULES.pop(name, None)


def grad(program, wrt, **kw):
    """``truestar.ir.grad`` with the training ops' gradient rules available."""
    from ..ir.autodiff import grad as _grad

    with training_rules():
        return _grad(program, wrt, **kw)
