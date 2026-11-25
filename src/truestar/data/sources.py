"""Data sources: anything with ``len(source)`` and ``source[i]``.

A source may also have ``get_batch(indices)`` (a whole batch in one vectorized read, used when no
per sample transform is set) and ``fingerprint()`` (a short string that changes when the data
changes; a saved loader position records it and a resume onto other data is refused). Every source
here is picklable, so it can be sent to worker processes; file handles are reopened per process.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .errors import raise_


def _as_numpy(a: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(a)


def _check_index(i: int, n: int) -> int:
    i = int(i)
    if not -n <= i < n:
        raise_("TS1381", f"index {i} is out of range for a source of {n} samples")
    return i % n if n else i


class ArraySource:
    """In memory arrays (numpy or torch) of the same length; sample ``i`` is ``(a[i], b[i], ...)``,
    or ``a[i]`` for one array."""

    def __init__(self, *arrays: Any):
        if not arrays:
            raise_("TS1381", "ArraySource needs at least one array")
        self.arrays = [_as_numpy(a) for a in arrays]
        lens = [len(a) for a in self.arrays]
        if len(set(lens)) != 1:
            raise_("TS1381", f"the arrays have different lengths {lens}")

    def __len__(self) -> int:
        return len(self.arrays[0])

    def _pack(self, parts):
        return parts[0] if len(parts) == 1 else tuple(parts)

    def __getitem__(self, i: int):
        i = _check_index(i, len(self))
        return self._pack([a[i] for a in self.arrays])

    def get_batch(self, indices: np.ndarray):
        idx = np.asarray(indices, dtype=np.int64)
        return self._pack([np.ascontiguousarray(a[idx]) for a in self.arrays])

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for a in self.arrays:
            h.update(f"{a.dtype.str}{a.shape}".encode())
        return h.hexdigest()[:16]


class MemmapSource(ArraySource):
    """Arrays on disk read through numpy memory maps: ``.npy`` files (``MemmapSource("x.npy",
    "y.npy")``) or raw files given as ``(path, dtype, shape)``. Only the rows a batch needs are read."""

    def __init__(self, *files: Any, offset: int = 0):
        self.files = []
        for f in files:
            if isinstance(f, (str, os.PathLike)):
                f = (os.fspath(f), None, None)
            path, dtype, shape = f
            if not os.path.exists(path):
                raise_("TS1381", f"memmap file {path} does not exist")
            self.files.append((path, None if dtype is None else np.dtype(dtype).str,
                               None if shape is None else tuple(int(s) for s in shape)))
        self.offset = int(offset)
        self._open()

    def _open(self) -> None:
        arrays = []
        for path, dtype, shape in self.files:
            if dtype is None:
                arrays.append(np.load(path, mmap_mode="r"))
            else:
                arrays.append(np.memmap(path, dtype=np.dtype(dtype), mode="r", offset=self.offset, shape=shape))
        ArraySource.__init__(self, *arrays)

    def __getstate__(self):
        return {"files": self.files, "offset": self.offset}

    def __setstate__(self, st):
        self.files, self.offset = st["files"], st["offset"]
        self._open()

    def fingerprint(self) -> str:
        h = hashlib.sha256(ArraySource.fingerprint(self).encode())
        for path, _, _ in self.files:
            h.update(str(os.path.getsize(path)).encode())
        return h.hexdigest()[:16]


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class FolderSource:
    """The files under ``root`` matching ``suffixes`` (all files when empty), in the byte order of
    their UTF-8 relative paths with ``/`` separators, which is the same on every OS. Sample ``i`` is
    ``reader(path)`` (the file's bytes by default); with ``labels=True`` it is ``(reader(path),
    label)`` where the label is the index of the file's top level folder among the sorted folders."""

    def __init__(self, root: str, suffixes: Sequence[str] = (), reader: Callable[[str], Any] = _read_bytes,
                 labels: bool = False):
        root = os.path.abspath(os.fspath(root))
        if not os.path.isdir(root):
            raise_("TS1381", f"folder {root} does not exist")
        suf = tuple(s.lower() for s in suffixes)
        rel: List[str] = []
        for d, _, names in os.walk(root):
            for name in names:
                if suf and not name.lower().endswith(suf):
                    continue
                rel.append(os.path.relpath(os.path.join(d, name), root).replace(os.sep, "/"))
        if not rel:
            raise_("TS1381", f"folder {root} has no files" + (f" ending in {', '.join(suffixes)}" if suf else ""))
        rel.sort(key=lambda s: s.encode("utf-8"))
        self.root, self.files, self.reader, self.labels = root, rel, reader, labels
        self.classes: List[str] = []
        if labels:
            tops = sorted({r.split("/", 1)[0] for r in rel if "/" in r}, key=lambda s: s.encode("utf-8"))
            if not tops:
                raise_("TS1381", f"labels=True needs one sub folder per class under {root}")
            self.classes = tops
            self._label = {c: k for k, c in enumerate(tops)}

    def __len__(self) -> int:
        return len(self.files)

    def path(self, i: int) -> str:
        return os.path.join(self.root, *self.files[_check_index(i, len(self))].split("/"))

    def __getitem__(self, i: int):
        i = _check_index(i, len(self))
        x = self.reader(self.path(i))
        if self.labels:
            return x, np.int64(self._label[self.files[i].split("/", 1)[0]])
        return x

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for r in self.files:
            h.update(r.encode("utf-8") + b"\0")
        return h.hexdigest()[:16]


class LinesSource:
    """One sample per line of a text file (UTF-8). ``parse`` maps the line (without its line
    ending) to a sample; ``JsonLinesSource`` parses each line as JSON. Line offsets are indexed once;
    a read seeks to the line, so the file is never held in memory."""

    CHUNK = 1 << 22

    def __init__(self, path: str, parse: Optional[Callable[[str], Any]] = None, skip_empty: bool = True):
        path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(path):
            raise_("TS1381", f"text file {path} does not exist")
        self.path, self.parse, self.skip_empty = path, parse, skip_empty
        newlines, pos = [], 0
        with open(path, "rb") as f:
            while True:
                buf = f.read(self.CHUNK)
                if not buf:
                    break
                newlines.append(np.flatnonzero(np.frombuffer(buf, dtype=np.uint8) == 10).astype(np.int64) + pos)
                pos += len(buf)
        nl = np.concatenate(newlines) if newlines else np.zeros(0, dtype=np.int64)
        s = np.concatenate([np.zeros(1, dtype=np.int64), nl + 1])
        e = np.concatenate([nl, np.asarray([pos], dtype=np.int64)])
        if s[-1] >= pos:            # no text after the last newline
            s, e = s[:-1], e[:-1]
        if skip_empty:              # lines of zero bytes (a line holding only "\r" is kept as "")
            keep = (e - s) > 0
            s, e = s[keep], e[keep]
        self._starts, self._ends = s, e
        self._size = pos
        self._fh = None

    def __getstate__(self):
        st = dict(self.__dict__)
        st["_fh"] = None
        return st

    def __len__(self) -> int:
        return len(self._starts)

    def _line(self, i: int) -> str:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        s, e = int(self._starts[i]), int(self._ends[i])
        self._fh.seek(s)
        raw = self._fh.read(e - s)
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        return raw.decode("utf-8")

    def __getitem__(self, i: int):
        line = self._line(_check_index(i, len(self)))
        return self.parse(line) if self.parse is not None else line

    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self._size}:{len(self)}".encode()).hexdigest()[:16]

    def __del__(self):
        fh = getattr(self, "_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


class JsonLinesSource(LinesSource):
    """A JSONL file: sample ``i`` is ``json.loads`` of line ``i``."""

    def __init__(self, path: str, skip_empty: bool = True):
        super().__init__(path, json.loads, skip_empty)


def fingerprint_of(source: Any) -> str:
    fp = getattr(source, "fingerprint", None)
    return fp() if callable(fp) else ""


__all__ = ["ArraySource", "MemmapSource", "FolderSource", "LinesSource", "JsonLinesSource", "fingerprint_of"]
