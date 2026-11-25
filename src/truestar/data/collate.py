"""Collate: a list of samples into one batch, and the batch into torch tensors.

``collate`` works on numpy so it runs the same in a thread or a worker process; ``to_torch`` runs in
the consuming process and wraps the arrays without a copy. Python ints become int64, floats float64
and bools bool on every OS (numpy's default int is 32 bits on Windows with numpy 1.x).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .errors import DataError, raise_


def _leaf(x: Any, where: str) -> Any:
    if isinstance(x, bool):
        return np.bool_(x)
    if isinstance(x, int):
        return np.int64(x)
    if isinstance(x, float):
        return np.float64(x)
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return x


def collate(samples: List[Any], where: str = "batch") -> Any:
    """Stack a list of samples: tuples and lists field by field, dicts key by key, arrays and numbers
    with ``np.stack``; strings, bytes and other objects stay a list."""
    if not samples:
        raise_("TS1385", f"{where}: no samples to collate")
    first = samples[0]
    if isinstance(first, (tuple, list)):
        n = len(first)
        for s in samples:
            if not isinstance(s, (tuple, list)) or len(s) != n:
                raise_("TS1385", f"{where}: samples have different structures ({_desc(first)} and {_desc(s)})")
        return tuple(collate([s[k] for s in samples], f"{where}[{k}]") for k in range(n))
    if isinstance(first, dict):
        keys = list(first)
        for s in samples:
            if not isinstance(s, dict) or list(s) != keys:
                raise_("TS1385", f"{where}: samples have different keys ({keys} and "
                                 f"{list(s) if isinstance(s, dict) else _desc(s)})")
        return {k: collate([s[k] for s in samples], f"{where}[{k!r}]") for k in keys}
    leaves = [_leaf(s, where) for s in samples]
    if isinstance(leaves[0], (np.ndarray, np.generic)):
        shapes = {np.shape(x) for x in leaves}
        dtypes = {np.asarray(x).dtype for x in leaves}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise_("TS1385", f"{where}: samples have shapes {sorted(shapes)} and dtypes "
                             f"{sorted(str(d) for d in dtypes)}; a batch needs one shape and one dtype "
                             "(pad or pass collate=)")
        return np.stack([np.asarray(x) for x in leaves])
    return list(samples)


def _desc(x: Any) -> str:
    if isinstance(x, (tuple, list)):
        return f"{type(x).__name__} of {len(x)}"
    return type(x).__name__


def to_torch(batch: Any, pin: bool = False, device: Any = None) -> Any:
    """numpy arrays in ``batch`` as torch tensors (zero copy when the array is writable), optionally
    in pinned memory and moved to ``device`` without blocking."""
    import torch

    if isinstance(batch, tuple):
        return tuple(to_torch(b, pin, device) for b in batch)
    if isinstance(batch, dict):
        return {k: to_torch(v, pin, device) for k, v in batch.items()}
    if isinstance(batch, (np.ndarray, np.generic)):
        a = np.asarray(batch)
        if a.dtype.kind in "biufc":
            if not a.flags.writeable or not a.flags.c_contiguous:
                a = np.ascontiguousarray(a).copy()
            t = torch.from_numpy(a)
            if pin:
                t = t.pin_memory()
            if device is not None:
                t = t.to(device, non_blocking=pin)
            return t
        return a
    return batch


__all__ = ["collate", "to_torch", "DataError"]
