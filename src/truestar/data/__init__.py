"""Deterministic data loading (docs/DATA.md).

``Loader`` serves the same batches on every machine, OS, worker count and worker mode, shards across
data parallel ranks with no overlap and no drop unless ``drop_last``, and resumes mid epoch from a
saved position. Sources: ``ArraySource`` (in memory), ``MemmapSource`` (numpy memory maps),
``FolderSource`` (a folder of files), ``LinesSource`` and ``JsonLinesSource`` (text and JSONL).
"""

from .collate import collate, to_torch
from .errors import DATA_CODES, DataError
from .loader import Loader
from .order import epoch_order, mix_array, sample_key, sample_keys, shard
from .sources import ArraySource, FolderSource, JsonLinesSource, LinesSource, MemmapSource
from .transforms import RandomCropFlip, key_ints

__all__ = ["Loader", "ArraySource", "MemmapSource", "FolderSource", "LinesSource", "JsonLinesSource",
           "RandomCropFlip", "key_ints", "collate", "to_torch", "epoch_order", "sample_key", "sample_keys", "shard", "mix_array", "DataError", "DATA_CODES"]
