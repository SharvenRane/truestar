"""A safetensors reader that needs only NumPy: memory mapped, sharded checkpoints, every dtype.

``truestar.interop.safetensors`` reads and writes files for torch tensors. This module reads the
same format without importing torch, for tools that only inspect or convert a checkpoint, and is
the reader behind ``truestar.models.from_pretrained``.

    from truestar.interop import safetensors_np as st
    ck = st.open_checkpoint("path/to/model")        # a folder, a .safetensors file or an index json
    ck.names()                                      # every tensor name across every shard
    t = ck["model.layers.0.self_attn.q_proj.weight"]
    t.dtype, t.shape                                # "BF16", (4096, 4096): from the header only
    t.raw()                                         # the stored bits, a read only view of the map
    t.numpy()                                       # values; BF16 and FP8 widened to float32 exactly

Header checks are those of ``safetensors.read_header``: duplicate names, unknown dtypes, byte
lengths that do not match the shape, overlaps, holes, a buffer of the wrong length. A sharded
checkpoint (``model.safetensors.index.json``) is checked as a whole before any tensor is read:
every shard the index names exists, every name in the index is in the shard it is mapped to, no
name is in two shards, and no shard holds a tensor the index does not list. Every problem is
reported at once with the stable code TS1395 (``SafetensorsError.code`` keeps the module's slug).

Values: BF16 is the top half of a float32, so widening is a 16 bit shift (exact). FP8 E4M3FN and
E5M2 are widened through a 256 entry table built from their bit layouts (exact; E4M3FN has no
infinity and one NaN pattern per sign). ``RawTensor.torch()`` gives a zero copy torch view of the
stored bits in the stored dtype when torch is present.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MAX_HEADER = 100 * 1024 * 1024
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"

# safetensors dtype name: (bytes per element, NumPy dtype of the stored bits, NumPy dtype of values)
DTYPES: Dict[str, Tuple[int, str, str]] = {
    "F64": (8, "<f8", "<f8"), "F32": (4, "<f4", "<f4"), "F16": (2, "<f2", "<f2"), "BF16": (2, "<u2", "<f4"),
    "I64": (8, "<i8", "<i8"), "I32": (4, "<i4", "<i4"), "I16": (2, "<i2", "<i2"), "I8": (1, "i1", "i1"),
    "U64": (8, "<u8", "<u8"), "U32": (4, "<u4", "<u4"), "U16": (2, "<u2", "<u2"), "U8": (1, "u1", "u1"),
    "BOOL": (1, "u1", "?"), "F8_E4M3": (1, "u1", "<f4"), "F8_E5M2": (1, "u1", "<f4"), "C64": (8, "<c8", "<c8"),
}

TORCH_NAMES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16", "I64": "int64",
               "I32": "int32", "I16": "int16", "I8": "int8", "U64": "uint64", "U32": "uint32", "U16": "uint16",
               "U8": "uint8", "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
               "C64": "complex64"}

SHARD_CODE = "TS1395"


class SafetensorsError(ValueError):
    """A file, an index or a read that is wrong. ``problems`` lists every one found."""

    def __init__(self, code: str, summary: str, problems: Sequence[str] = (), suggestion: str = "",
                 ts_code: str = ""):
        self.code = code
        self.ts_code = ts_code
        self.summary = summary
        self.problems = list(problems)
        self.suggestion = suggestion
        lines = [summary] + [f"  {p}" for p in self.problems[:50]]
        if len(self.problems) > 50:
            lines.append(f"  ... and {len(self.problems) - 50} more")
        if suggestion:
            lines.append(f"  fix: {suggestion}")
        super().__init__("\n".join(lines))

    def to_dict(self):
        return {"code": self.code, "ts_code": self.ts_code, "summary": self.summary, "problems": self.problems,
                "suggestion": self.suggestion}


# --------------------------------------------------------------------------- FP8 tables


def _fp8_table(exp_bits: int, man_bits: int, bias: int, finite_only: bool) -> np.ndarray:
    """The float32 value of each of the 256 bit patterns."""
    out = np.empty(256, dtype=np.float32)
    top = (1 << exp_bits) - 1
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> man_bits) & top
        m = b & ((1 << man_bits) - 1)
        if finite_only and e == top and m == (1 << man_bits) - 1:
            v = math.nan                      # E4M3FN: S.1111.111 is NaN, there is no infinity
        elif not finite_only and e == top:
            v = s * math.inf if m == 0 else math.nan
        elif e == 0:
            v = s * (m / (1 << man_bits)) * 2.0 ** (1 - bias)
        else:
            v = s * (1.0 + m / (1 << man_bits)) * 2.0 ** (e - bias)
        out[b] = v
    return out


FP8_E4M3FN = _fp8_table(4, 3, 7, True)
FP8_E5M2 = _fp8_table(5, 2, 15, False)


def widen(bits: np.ndarray, dtype: str) -> np.ndarray:
    """Values of stored bits as a NumPy array (BF16 and FP8 widened to float32, exactly)."""
    if dtype == "BF16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F8_E4M3":
        return FP8_E4M3FN[bits]
    if dtype == "F8_E5M2":
        return FP8_E5M2[bits]
    if dtype == "BOOL":
        return bits != 0
    return bits


# --------------------------------------------------------------------------- one file


@dataclass(frozen=True)
class Entry:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    begin: int
    end: int


def _no_duplicates(pairs):
    out, dup = {}, []
    for k, v in pairs:
        if k in out:
            dup.append(k)
        out[k] = v
    if dup:
        out["\0duplicates"] = dup
    return out


def read_header(path: str) -> Tuple[Dict[str, Entry], Dict[str, str], int, int]:
    """Validate a file's header. Returns (entries, metadata, data start offset, data length)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise SafetensorsError("safetensors_truncated", f"{path} is {size} bytes, shorter than the 8 byte "
                                   "header length")
        (n,) = struct.unpack("<Q", head)
        if n > MAX_HEADER:
            raise SafetensorsError("safetensors_bad_header", f"{path}: header length {n} is over the 100 MB limit")
        if 8 + n > size:
            raise SafetensorsError("safetensors_truncated", f"{path}: header length {n} runs past the end of the "
                                   f"{size} byte file")
        raw = f.read(n)
    if not raw.startswith(b"{"):
        raise SafetensorsError("safetensors_bad_header", f"{path}: the header does not start with '{{'")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SafetensorsError("safetensors_bad_header", f"{path}: the header is not UTF 8 JSON: {e}") from None
    if not isinstance(obj, dict):
        raise SafetensorsError("safetensors_bad_header", f"{path}: the header is not a JSON object")
    problems: List[str] = [f"{d}: the name appears more than once in the header" for d in obj.pop("\0duplicates", [])]
    meta = obj.pop("__metadata__", None) or {}
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        problems.append("__metadata__ must map strings to strings")
        meta = {}
    data_len = size - 8 - n
    entries: Dict[str, Entry] = {}
    for name, e in obj.items():
        if not isinstance(e, dict) or set(e) != {"dtype", "shape", "data_offsets"}:
            problems.append(f"{name}: entry must have exactly dtype, shape and data_offsets, got {e!r}"[:200])
            continue
        info = DTYPES.get(e["dtype"])
        if info is None:
            problems.append(f"{name}: unknown dtype {e['dtype']!r} (known: {sorted(DTYPES)})")
            continue
        shape, off = e["shape"], e["data_offsets"]
        if not (isinstance(shape, list) and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                                                for d in shape)):
            problems.append(f"{name}: shape must be a list of non negative ints, got {shape!r}")
            continue
        if not (isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                and x >= 0 for x in off) and off[0] <= off[1]):
            problems.append(f"{name}: data_offsets must be [begin, end] with 0 <= begin <= end, got {off!r}")
            continue
        want = math.prod(shape) * info[0]
        if off[1] - off[0] != want:
            problems.append(f"{name}: {e['dtype']}{shape} needs {want} bytes but data_offsets {off} hold "
                            f"{off[1] - off[0]}")
            continue
        entries[name] = Entry(name, e["dtype"], tuple(shape), off[0], off[1])
    pos = 0
    for en in sorted(entries.values(), key=lambda x: (x.begin, x.end)):
        if en.begin < pos:
            problems.append(f"{en.name}: bytes [{en.begin}, {en.end}) overlap the tensor before it (ends at {pos})")
        elif en.begin > pos:
            problems.append(f"{en.name}: a hole of {en.begin - pos} bytes before it (at {pos})")
        pos = max(pos, en.end)
    if pos != data_len:
        problems.append(f"the tensors cover {pos} bytes but the data buffer is {data_len} bytes"
                        + (" (truncated file)" if pos > data_len else " (trailing bytes)"))
    if problems:
        raise SafetensorsError("safetensors_invalid", f"{path} is not a valid safetensors file "
                               f"({len(problems)} problem(s), nothing was loaded):", problems)
    return entries, meta, 8 + n, data_len


class RawTensor:
    """One stored tensor: dtype and shape from the header; data read from the map on first touch."""

    __slots__ = ("name", "dtype", "shape", "_file", "_entry")

    def __init__(self, file: "SafetensorsFile", entry: Entry):
        self.name = entry.name
        self.dtype = entry.dtype
        self.shape = entry.shape
        self._file = file
        self._entry = entry

    @property
    def nbytes(self) -> int:
        return self._entry.end - self._entry.begin

    @property
    def path(self) -> str:
        return self._file.path

    def raw(self) -> np.ndarray:
        """The stored bits as a read only NumPy view of the memory map (a copy when unaligned)."""
        e = self._entry
        size, bits, _ = DTYPES[e.dtype]
        if e.end == e.begin:
            return np.empty(e.shape, dtype=bits)
        buf = self._file.buffer()[e.begin:e.end]
        if e.begin % size:
            buf = buf.copy()
        return buf.view(bits).reshape(e.shape)

    def numpy(self) -> np.ndarray:
        """Values: BF16, FP8 widened to float32 exactly; BOOL as bool; others as stored."""
        return widen(self.raw(), self.dtype)

    def torch(self):
        """A torch tensor of the stored dtype, sharing the map's memory (read only: never write it)."""
        import torch
        import warnings

        dt = getattr(torch, TORCH_NAMES[self.dtype])
        r = self.raw()
        with warnings.catch_warnings():   # torch warns that the map is read only; nothing writes it
            warnings.simplefilter("ignore", UserWarning)
            if self.dtype == "BOOL":
                return torch.from_numpy(r).view(torch.bool)
            if self.dtype in ("BF16", "F8_E4M3", "F8_E5M2"):
                return torch.from_numpy(r).view(dt)
            return torch.from_numpy(r)

    def __repr__(self):
        return f"<{self.name} {self.dtype}{list(self.shape)} in {os.path.basename(self.path)}>"


class SafetensorsFile(Mapping[str, RawTensor]):
    """One validated file. The data buffer is memory mapped read only, on first touch."""

    def __init__(self, path: str):
        self.path = str(path)
        self.entries, self.metadata, self._start, self._len = read_header(self.path)
        self._buf: Optional[np.ndarray] = None

    def buffer(self) -> np.ndarray:
        if self._buf is None:
            self._buf = (np.empty(0, np.uint8) if self._len == 0 else
                         np.memmap(self.path, dtype=np.uint8, mode="r", offset=self._start, shape=(self._len,)))
        return self._buf

    def __getitem__(self, k: str) -> RawTensor:
        return RawTensor(self, self.entries[k])

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


# --------------------------------------------------------------------------- a checkpoint (one file or shards)


class Checkpoint(Mapping[str, RawTensor]):
    """Every tensor of a checkpoint, one file or the shards of an index, by name."""

    def __init__(self, files: Sequence[SafetensorsFile], where: Dict[str, SafetensorsFile], source: str,
                 index_metadata: Optional[Dict] = None):
        self.files = list(files)
        self._where = where
        self.source = source
        self.index_metadata = dict(index_metadata or {})

    def names(self) -> List[str]:
        return list(self._where)

    def __getitem__(self, k: str) -> RawTensor:
        return self._where[k][k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)

    @property
    def nbytes(self) -> int:
        return sum(e.end - e.begin for f in self.files for e in f.entries.values())

    def dtypes(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k in self._where:
            d = self._where[k].entries[k].dtype
            out[d] = out.get(d, 0) + 1
        return out


def _open_index(index_path: str) -> Checkpoint:
    base = os.path.dirname(os.path.abspath(index_path))
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SafetensorsError("safetensors_bad_index", f"{index_path}: cannot read the index: {e}",
                               ts_code=SHARD_CODE) from None
    wm = idx.get("weight_map") if isinstance(idx, dict) else None
    if not isinstance(wm, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in wm.items()):
        raise SafetensorsError("safetensors_bad_index", f"{index_path}: no weight_map of names to shard files",
                               ts_code=SHARD_CODE)
    problems: List[str] = []
    files: Dict[str, SafetensorsFile] = {}
    for shard in sorted(set(wm.values())):
        if os.path.isabs(shard) or ".." in shard.replace("\\", "/").split("/"):
            problems.append(f"{shard}: a shard must be a file next to the index")
            continue
        p = os.path.join(base, shard)
        if not os.path.isfile(p):
            problems.append(f"{shard}: named by the index but not found next to it")
            continue
        try:
            files[shard] = SafetensorsFile(p)
        except SafetensorsError as e:
            problems.extend(f"{shard}: {q}" for q in ([e.summary] + e.problems))
    where: Dict[str, SafetensorsFile] = {}
    for name, shard in wm.items():
        f = files.get(shard)
        if f is None:
            continue
        if name not in f.entries:
            problems.append(f"{name}: the index maps it to {shard}, which does not hold it")
            continue
        where[name] = f
    listed = set(wm)
    for shard, f in files.items():
        for name in f.entries:
            if name not in listed:
                owner = [s for s, g in files.items() if name in g.entries and s != shard]
                problems.append(f"{name}: in {shard} but not in the index" +
                                (f" (also in {', '.join(owner)})" if owner else ""))
            elif wm[name] != shard:
                problems.append(f"{name}: in {shard} and mapped to {wm[name]}: one name in two shards")
    if problems:
        raise SafetensorsError("safetensors_bad_index", f"{index_path}: the sharded checkpoint is not consistent "
                               f"({len(problems)} problem(s), nothing was loaded):", problems,
                               "re save the checkpoint, or fix the weight_map to match the shards", ts_code=SHARD_CODE)
    return Checkpoint(list(files.values()), where, index_path, idx.get("metadata"))


def open_checkpoint(path: str) -> Checkpoint:
    """A folder (``model.safetensors.index.json`` or ``model.safetensors``, else its only
    ``*.safetensors`` file), an index json, or one ``.safetensors`` file."""
    path = str(path)
    if os.path.isdir(path):
        idx = os.path.join(path, INDEX_NAME)
        if os.path.isfile(idx):
            return _open_index(idx)
        one = os.path.join(path, SINGLE_NAME)
        if not os.path.isfile(one):
            found = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
            if len(found) != 1:
                raise SafetensorsError("safetensors_not_found", f"{path}: no {INDEX_NAME} or {SINGLE_NAME}, and "
                                       f"{len(found)} other .safetensors files", found,
                                       "point at the index json or at one file", ts_code=SHARD_CODE)
            one = os.path.join(path, found[0])
        f = SafetensorsFile(one)
        return Checkpoint([f], {k: f for k in f.entries}, one)
    if path.endswith(".json"):
        return _open_index(path)
    f = SafetensorsFile(path)
    return Checkpoint([f], {k: f for k in f.entries}, path)
