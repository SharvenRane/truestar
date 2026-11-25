"""Training mode ops written as primitive IR ops, shared by the importer and the native front end.

Training batch norm and dropout are not new IR ops. They are short compositions of ops the IR
already has (mean, sub, mul, rsqrt, where, random), so the reference interpreter, autodiff, the
compiler and the serializer cover them with no new rule. What makes them training ops is that their
hidden parts are explicit values:

* batch norm's running statistics are state in and state out: ``(y, new_mean, new_var)`` from
  ``(x, weight, bias, mean, var)``. Nothing is mutated, so recompute, vmap or a replayed module cannot
  apply the update twice;
* dropout's mask comes from a ``random`` op with an explicit key operand and site, so it is the
  same on every backend (docs/RANDOM.md).

Both functions take ``emit(op, inputs, **attrs) -> Value`` and ``const(number, like) -> Value``
from the caller (importer context or front end translator), so errors keep the caller's source line.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import sympy
import torch

from .core import UnsupportedOpError, Value

Emit = Callable[..., Value]
Const = Callable[[float, Value], Value]


def batch_norm_train(emit: Emit, const: Const, x: Value, weight: Optional[Value], bias: Optional[Value],
                     running_mean: Optional[Value], running_var: Optional[Value], momentum: Optional[float],
                     eps: float) -> Tuple[Value, Optional[Value], Optional[Value]]:
    """PyTorch's training batch norm. Normalizes with the batch statistics over every dim but the
    channel dim 1 (biased variance); when running statistics are given, returns their updates
    ``(1 - momentum) * running + momentum * batch``, with the unbiased variance ``var * n / (n - 1)``
    as torch does. Returns ``(y, new_running_mean, new_running_var)``."""
    rank = x.type.rank
    if rank < 2:
        raise UnsupportedOpError(f"training batch norm needs an input [N, C, ...], got {x.type}")
    if (running_mean is None) != (running_var is None):
        raise UnsupportedOpError("training batch norm needs both running_mean and running_var, or neither")
    if running_mean is not None and momentum is None:
        raise UnsupportedOpError("training batch norm with momentum=None (a cumulative average driven by "
                                 "num_batches_tracked) is not supported; give a float momentum")
    dims = [0] + list(range(2, rank))
    n = sympy.Integer(1)
    for d in dims:
        n = n * sympy.sympify(x.type.shape[d])
    n = sympy.simplify(n)
    if n.is_Integer and int(n) <= 1 and running_mean is not None:
        raise UnsupportedOpError(f"training batch norm over {int(n)} value per channel has no unbiased "
                                 f"variance (input {x.type}); PyTorch refuses it too")
    mean_k = emit("mean", [x], dims=dims, keepdim=True)
    d = emit("sub", [x, mean_k])
    var_k = emit("mean", [emit("mul", [d, d])], dims=dims, keepdim=True)
    y = emit("mul", [d, emit("rsqrt", [emit("add", [var_k, const(float(eps), var_k)])])])
    view = [x.type.shape[1]] + [1] * (rank - 2)
    if weight is not None:
        y = emit("mul", [y, _view(emit, weight, view)])
    if bias is not None:
        y = emit("add", [y, _view(emit, bias, view)])
    if running_mean is None:
        return y, None, None
    c = [x.type.shape[1]]
    mean = emit("reshape", [mean_k], shape=c)
    var = emit("reshape", [var_k], shape=c)
    if n.is_Integer:
        unbiased = emit("mul", [var, const(int(n) / (int(n) - 1), var)])
    else:
        num = emit("dim_scalar", [], expr=n, dtype=var.type.dtype)
        den = emit("dim_scalar", [], expr=n - 1, dtype=var.type.dtype)
        unbiased = emit("div", [emit("mul", [var, num]), den])
    m = float(momentum)
    new_mean = emit("add", [emit("mul", [running_mean, const(1.0 - m, running_mean)]),
                            emit("mul", [mean, const(m, mean)])])
    new_var = emit("add", [emit("mul", [running_var, const(1.0 - m, running_var)]),
                           emit("mul", [unbiased, const(m, unbiased)])])
    return y, new_mean, new_var


def _view(emit: Emit, v: Value, shape: Sequence) -> Value:
    if v.type.rank == len(shape) and all(str(a) == str(b) for a, b in zip(v.type.shape, shape)):
        return v
    return emit("reshape", [v], shape=list(shape))


def dropout(emit: Emit, const: Const, x: Value, p: float, key: Value, streams: Sequence[Value],
            site_name: str) -> Tuple[Value, Value]:
    """Inverted dropout: keep each element with probability 1 - p (a keyed ``random`` bernoulli draw
    at ``site_name``), scale kept elements by 1 / (1 - p). Returns ``(y, keep_mask)``."""
    from .. import rng

    p = float(p)
    if not 0.0 <= p < 1.0:
        raise UnsupportedOpError(f"dropout probability must be in [0, 1), got {p}")
    keep = emit("random", [key] + list(streams), dist="bernoulli", shape=list(x.type.shape), dtype=torch.bool,
                site=rng.site_id(site_name), site_name=site_name, p=1.0 - p)
    if p == 0.0:
        return x, keep
    y = emit("where", [keep, emit("mul", [x, const(1.0 / (1.0 - p), x)]), const(0.0, x)])
    return y, keep
