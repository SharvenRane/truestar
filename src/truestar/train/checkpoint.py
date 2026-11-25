"""Checkpoint files: named tensors plus JSON state in the ``.tsx`` artifact format (docs/EXPORT.md).

A checkpoint is a program with no nodes whose params are the saved tensors in their own dtypes, with
the JSON state under ``metadata["truestar_checkpoint"]``. The format gives it a sha256 per tensor
and over the whole file, and the same state always gives the same bytes. Writes are atomic: the
bytes go to a temporary file in the target folder, are flushed to disk, then renamed over the
target, so a crash leaves either the old checkpoint or the new one, never half of one.

Checkpoint versions: 1 (params, optimizer state, step count, history), 2 (adds ``rng``: the seed the
trainer derives every step's key from, and ``loader``: the data loader position). Version 1 files
still load; a resume that needs the seed or the loader position refuses them (TS1389).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch

from ..data.errors import DataError
from ..ir import serialize
from ..ir.core import Program, TensorType, Value

CHECKPOINT_KEY = "truestar_checkpoint"
CHECKPOINT_VERSION = 2
READS = (1, 2)


def dumps(tensors: Dict[str, torch.Tensor], state: Dict[str, Any]) -> bytes:
    """Checkpoint bytes for ``tensors`` (name -> tensor) and plain JSON ``state``."""
    params, data = [], {}
    for n, t in tensors.items():
        t = t.detach().cpu().contiguous()
        params.append(Value(n, TensorType(t.dtype, tuple(t.shape)), "param"))
        data[n] = t
    prog = Program("checkpoint", [], params, data, [], list(params))
    return serialize.dumps(prog, {CHECKPOINT_KEY: {"version": CHECKPOINT_VERSION, **state}})


def save(path: str, tensors: Dict[str, torch.Tensor], state: Dict[str, Any]) -> str:
    """Write a checkpoint atomically. Returns the content hash."""
    return serialize.write_bytes(dumps(tensors, state), path)


def loads(path_or_bytes: Union[str, bytes]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """``(tensors, state)`` of a checkpoint; every checksum is verified first."""
    try:
        art = (serialize.loads(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray))
               else serialize.load(path_or_bytes))
    except serialize.ArtifactError as e:
        raise DataError("TS1386", str(e)) from None
    except OSError as e:
        raise DataError("TS1386", f"cannot read checkpoint: {e}") from None
    meta = art.metadata.get(CHECKPOINT_KEY)
    if not meta:
        raise DataError("TS1386", "not a truestar training checkpoint: the metadata has no checkpoint section")
    if meta.get("version") not in READS:
        raise DataError("TS1387", f"checkpoint version {meta.get('version')} is not supported (this truestar "
                                  f"reads versions {', '.join(map(str, READS))})")
    return dict(art.program.param_data), dict(meta)


load = loads

__all__ = ["CHECKPOINT_KEY", "CHECKPOINT_VERSION", "dumps", "save", "loads", "load"]
