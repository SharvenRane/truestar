"""Model summary and parameter counting for truestar.nn layers and parameter trees.

    print(nn.summary(net))           # one row per parameter, subtotals per child, totals
    nn.count(p)                      # (elements, bytes) of any tree of tensors
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, List, Tuple

import torch

from .. import pytree
from .core import Layer, _dtype


@dataclasses.dataclass(frozen=True)
class Row:
    path: str
    shape: Tuple[int, ...]
    dtype: str
    count: int
    bytes: int
    trained: bool          # a parameter (True) or carried state such as running statistics (False)


@dataclasses.dataclass(frozen=True)
class Summary:
    name: str
    rows: Tuple[Row, ...]

    @property
    def params(self) -> int:
        return sum(r.count for r in self.rows if r.trained)

    @property
    def state(self) -> int:
        return sum(r.count for r in self.rows if not r.trained)

    @property
    def bytes(self) -> int:
        return sum(r.bytes for r in self.rows)

    def by_child(self) -> List[Tuple[str, int]]:
        """Trained parameter count per top level child, in order."""
        out: dict = {}
        for r in self.rows:
            if r.trained:
                top = r.path.split(".", 1)[0] if "." in r.path else "(own)"
                out[top] = out.get(top, 0) + r.count
        return list(out.items())

    def __str__(self) -> str:
        w = max([len(r.path) for r in self.rows] + [9])
        lines = [f"{self.name}", f"{'parameter':<{w}}  {'shape':<18} {'dtype':<5} {'count':>12}"]
        for r in self.rows:
            tag = "" if r.trained else "  (state)"
            lines.append(f"{r.path:<{w}}  {str(list(r.shape)):<18} {r.dtype:<5} {r.count:>12,}{tag}")
        lines.append(f"trained parameters {self.params:,} ({_human(self.params)}), state {self.state:,}, "
                     f"{_mib(self.bytes)}")
        return "\n".join(lines)


def _human(n: int) -> str:
    for unit, k in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= k:
            return f"{n / k:.2f}{unit}"
    return str(n)


def _mib(b: int) -> str:
    return f"{b / 2 ** 20:.2f} MiB"


_SHORT = {torch.float32: "f32", torch.float64: "f64", torch.bfloat16: "bf16", torch.float16: "f16"}


def _rows(layer: Layer, which: str, prefix: str = "") -> List[Row]:
    own = layer.params() if which == "params" else layer.state()
    dt = _dtype(layer.dtype)
    size = torch.empty((), dtype=dt).element_size()
    out = []
    for name, prm in own.items():
        n = math.prod(prm.shape)
        out.append(Row(prefix + name, tuple(prm.shape), _SHORT.get(dt, str(dt)), n, n * size, which == "params"))
    for name, child in layer.parts.items():
        out += _rows(child, which, f"{prefix}{name}.")
    return out


def summary(layer: Layer, name: str = None) -> Summary:
    """Every parameter and state entry of ``layer`` with its shape, dtype and size, from the config
    alone (nothing is drawn)."""
    if not isinstance(layer, Layer):
        raise TypeError(f"summary takes a truestar.nn layer, got {type(layer).__name__}")
    rows = _rows(layer, "params") + _rows(layer, "state")
    return Summary(name or type(layer).__name__, tuple(rows))


def count(tree: Any) -> Tuple[int, int]:
    """(elements, bytes) over every tensor leaf of a tree (parameters, optimizer state, a checkpoint)."""
    leaves, _ = pytree.flatten(tree, lambda x: isinstance(x, torch.Tensor))
    return sum(t.numel() for t in leaves), sum(t.numel() * t.element_size() for t in leaves)


__all__ = ["Row", "Summary", "summary", "count"]
