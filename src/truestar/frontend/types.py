"""Shape typed tensor annotations for the native front end.

``Tensor["B", 128, "f32"]`` is a tensor with a symbolic first dim ``B``, a fixed second dim of 128
and dtype float32. Dims are ints, symbol names or small expressions over symbol names (``"2*B"``).
The dtype is the last item when it names a dtype; it defaults to f32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import sympy
import torch

from ..ir.core import Dim, TensorType, norm_dim, symbol

DTYPES: Dict[str, torch.dtype] = {
    "f64": torch.float64, "f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16,
    "i64": torch.int64, "i32": torch.int32, "i16": torch.int16, "i8": torch.int8, "u8": torch.uint8,
    "bool": torch.bool,
    "float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "int64": torch.int64, "int32": torch.int32,
}


def parse_dim(d: Any) -> Dim:
    if isinstance(d, bool):
        raise TypeError(f"a dim cannot be a bool: {d!r}")
    if isinstance(d, int):
        if d < 0:
            raise TypeError(f"a dim must be non negative, got {d}")
        return d
    if isinstance(d, sympy.Basic):
        return norm_dim(d)
    if isinstance(d, str):
        names = {n: symbol(n) for n in _identifiers(d)}
        try:
            return norm_dim(sympy.sympify(d, locals=names))
        except (sympy.SympifyError, TypeError) as e:
            raise TypeError(f"cannot parse dim {d!r}: {e}") from None
    raise TypeError(f"a dim must be an int, a symbol name or an expression string, got {d!r}")


def _identifiers(s: str):
    import re
    return set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s))


@dataclass(frozen=True)
class TensorSpec:
    """The parsed form of a ``Tensor[...]`` annotation."""

    shape: Tuple[Dim, ...]
    dtype: torch.dtype

    def to_type(self) -> TensorType:
        return TensorType(self.dtype, self.shape)

    def symbols(self):
        out = []
        for d in self.shape:
            if not isinstance(d, int):
                for s in sorted(sympy.sympify(d).free_symbols, key=str):
                    if str(s) not in out:
                        out.append(str(s))
        return out

    def __str__(self) -> str:
        return str(self.to_type())

    def __call__(self, *a, **k):
        # Callable only so typing.NamedTuple accepts Tensor[...] as a field annotation (it checks
        # callable()); a spec is never meant to be called.
        raise TypeError(f"{self} is a shape annotation, not a constructor")


class Tensor:
    """Annotation only: ``x: Tensor["B", 128, "f32"]``. Never instantiated."""

    def __init__(self, *a, **k):
        raise TypeError("truestar.Tensor is an annotation, write Tensor[\"B\", 128, \"f32\"]")

    def __class_getitem__(cls, items) -> TensorSpec:
        if not isinstance(items, tuple):
            items = (items,)
        dtype = torch.float32
        dims = list(items)
        if dims and isinstance(dims[-1], torch.dtype):
            dtype = dims.pop()
        elif dims and isinstance(dims[-1], str) and dims[-1] in DTYPES:
            dtype = DTYPES[dims.pop()]
        if any(isinstance(d, str) and d in DTYPES for d in dims):
            raise TypeError(f"the dtype must be the last item of Tensor[...], got {items!r}")
        return TensorSpec(tuple(parse_dim(d) for d in dims), dtype)
