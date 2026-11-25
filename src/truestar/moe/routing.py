"""Deterministic top k routing for mixture of experts layers.

The tie rule (the same on every device, in eager and in the IR):

* a larger score ranks first;
* equal scores (bitwise equal in the working precision) rank by expert index, lower first;
* NaN ranks above every number (as ``torch.sort``), so a NaN score is chosen, never silently skipped.

``torch.topk`` promises no order among equal values, and its CUDA and CPU kernels do not agree on
one. ``route_topk`` is the first k entries of a stable descending sort, which is the rule above on
CPU and CUDA alike; the IR's ``topk`` op computes the same rule (ir/reference.py, the compiled
runtime's ``_index_topk``). ``stable_topk`` makes every ``torch.topk`` call inside a block use it.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.overrides import TorchFunctionMode

TIE_RULE = ("larger score first; bitwise equal scores go to the lower expert index; NaN ranks above "
            "every number")


def route_topk(scores: torch.Tensor, k: int, dim: int = -1, largest: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """(values, indices) of the k best scores along ``dim`` under ``TIE_RULE``, sorted best first."""
    v, i = torch.sort(scores, dim=dim, descending=largest, stable=True)
    return v.narrow(dim, 0, k), i.narrow(dim, 0, k)


class stable_topk(TorchFunctionMode):
    """Inside this mode every ``torch.topk`` (function or method) follows ``TIE_RULE``. Results are
    always sorted best first, which ``sorted=False`` callers accept (any order is allowed there)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if func is torch.topk or func is torch.Tensor.topk:
            names = ("input", "k", "dim", "largest", "sorted")
            bound = dict(zip(names, args))
            bound.update(kwargs)
            out = route_topk(bound["input"], int(bound["k"]), bound.get("dim", -1), bound.get("largest", True))
            return torch.return_types.topk(out)
        return func(*args, **kwargs)
