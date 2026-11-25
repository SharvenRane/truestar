"""The .tsrec container: a JSON header plus deduplicated raw tensor bytes, with a checksum.

Layout (all integers little endian):

    8 bytes   magic b"TSREC\\x00v1"
    8 bytes   header length H
    H bytes   header, UTF-8 JSON with sorted keys and no whitespace
    ...       tensor blobs, concatenated in the order of header["blobs"]
    32 bytes  sha256 of everything before it

Every tensor is stored once per distinct content (dtype, shape, bytes), so a module output that is
the next module's input costs nothing extra. Nothing in the file depends on the time, the host or
the process, so two identical runs give identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import torch

MAGIC = b"TSREC\x00v1"
FORMAT = "truestar.record"
VERSION = 1


class RecordError(Exception):
    """The file is not a valid recording (bad magic, version, truncation or checksum)."""


class Unreplayable(Exception):
    """A recorded argument could not be stored, so the call cannot be fed back in."""


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def dtype_from(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise RecordError(f"unknown dtype {name!r} in recording")
    return dt


def tensor_bytes(t: torch.Tensor) -> bytes:
    if t.layout != torch.strided or t.is_quantized:
        raise Unreplayable(f"cannot store a {t.layout} / quantized tensor")
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        return b""
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_hash(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(f"{dtype_name(t.dtype)}|{tuple(t.shape)}|".encode())
    h.update(tensor_bytes(t))
    return h.hexdigest()[:32]


def hashes(ts: List[torch.Tensor]) -> List[str]:
    return [tensor_hash(t) for t in ts]


class BlobStore:
    """Tensors written so far, deduplicated by content hash."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.data: List[bytes] = []
        self._index: Dict[str, int] = {}
        self.nbytes = 0

    def add(self, t: torch.Tensor) -> int:
        raw = tensor_bytes(t)
        h = tensor_hash(t)
        if h in self._index:
            return self._index[h]
        i = len(self.entries)
        self.entries.append({"hash": h, "dtype": dtype_name(t.dtype), "shape": list(t.shape),
                             "offset": self.nbytes, "nbytes": len(raw)})
        self.data.append(raw)
        self._index[h] = i
        self.nbytes += len(raw)
        return i


# ----------------------------------------------------------------------------- argument trees


def alias_map(obj: Any, fn, _memo: Optional[Dict[int, Any]] = None) -> Any:
    """Apply fn to every tensor in nested containers; a tensor object seen twice maps to one
    result object, so `attn(x, x, x)` stays `attn(y, y, y)`."""
    memo = {} if _memo is None else _memo
    if isinstance(obj, torch.Tensor):
        if id(obj) not in memo:
            memo[id(obj)] = (obj, fn(obj))  # keep obj alive so its id is not reused
        return memo[id(obj)][1]
    if isinstance(obj, tuple):
        return tuple(alias_map(x, fn, memo) for x in obj)
    if isinstance(obj, list):
        return [alias_map(x, fn, memo) for x in obj]
    if isinstance(obj, dict):
        return {k: alias_map(v, fn, memo) for k, v in obj.items()}
    return obj


def encode(obj: Any, store: Optional[BlobStore], _ids: Optional[Dict[int, int]] = None) -> Any:
    """JSON tree for a forward or op argument. Tensors become {"T": blob, "id": n} (or only
    their hash {"H": ...} when store is None). `id` numbers distinct tensor objects in order of
    first use, so a tensor passed twice (attention called as attn(x, x, x)) is the same object
    again on replay: modules such as nn.MultiheadAttention pick a different kernel otherwise.
    Values that cannot be stored become {"opaque": type}."""
    ids = {} if _ids is None else _ids
    if isinstance(obj, torch.Tensor):
        n = ids.setdefault(id(obj), len(ids))
        if store is None:
            return {"H": tensor_hash(obj), "dtype": dtype_name(obj.dtype),
                    "shape": list(obj.shape), "id": n}
        return {"T": store.add(obj), "id": n}
    if obj is None or isinstance(obj, (bool, str)):
        return {"v": obj}
    if isinstance(obj, int):
        return {"v": int(obj)}
    if isinstance(obj, float):
        if math.isfinite(obj):
            return {"v": obj}
        return {"f": "nan" if obj != obj else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, tuple):
        return {"tuple": [encode(x, store, ids) for x in obj]}
    if isinstance(obj, list):
        return {"list": [encode(x, store, ids) for x in obj]}
    if isinstance(obj, dict):
        items = []
        for k in obj:
            if not isinstance(k, (str, int)):
                return {"opaque": "dict with " + type(k).__name__ + " keys"}
            items.append([k, encode(obj[k], store, ids)])
        return {"dict": items}
    if isinstance(obj, torch.dtype):
        return {"dtype": dtype_name(obj)}
    if isinstance(obj, torch.device):
        return {"device": str(obj)}
    if isinstance(obj, torch.memory_format):
        return {"memfmt": str(obj).replace("torch.", "")}
    if isinstance(obj, torch.layout):
        return {"layout": str(obj).replace("torch.", "")}
    return {"opaque": type(obj).__name__}


def encode_call(args: Any, kwargs: Any, store: Optional[BlobStore]) -> Tuple[Any, Any]:
    """Encode forward args and kwargs with one shared tensor numbering."""
    ids: Dict[int, int] = {}
    return encode(args, store, ids), encode(kwargs, store, ids)


def decode_call(args_tree: Any, kwargs_tree: Any, tensor) -> Tuple[Any, Any]:
    objs: Dict[int, torch.Tensor] = {}
    return decode(args_tree, tensor, objs), decode(kwargs_tree, tensor, objs)


def tree_hash(tree: Any, entries: List[Dict[str, Any]]) -> str:
    """Hash of an encoded tree where blob references are replaced by content hashes."""

    def norm(t: Any) -> Any:
        if isinstance(t, dict):
            if "T" in t:
                e = entries[t["T"]]
                return {"H": e["hash"], "dtype": e["dtype"], "shape": e["shape"]}
            return {k: norm(v) for k, v in t.items()}
        if isinstance(t, list):
            return [norm(x) for x in t]
        return t

    s = json.dumps(norm(tree), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:32]


def opaque_parts(tree: Any) -> List[str]:
    out: List[str] = []

    def walk(t: Any) -> None:
        if isinstance(t, dict):
            if "opaque" in t:
                out.append(t["opaque"])
            elif "H" in t:
                out.append("tensor stored as hash only")
            else:
                for v in t.values():
                    walk(v)
        elif isinstance(t, list):
            for x in t:
                walk(x)

    walk(tree)
    return out


# ----------------------------------------------------------------------------- file


def dumps(header: Dict[str, Any], store: BlobStore) -> bytes:
    header = dict(header, blobs=store.entries)
    hj = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body = MAGIC + struct.pack("<Q", len(hj)) + hj + b"".join(store.data)
    return body + hashlib.sha256(body).digest()


def loads(raw: bytes, path: str = "<bytes>") -> Tuple[Dict[str, Any], memoryview]:
    """(header, blob area). Raises RecordError naming what is wrong."""
    if len(raw) < len(MAGIC) + 8 + 32:
        raise RecordError(f"{path}: file is truncated ({len(raw)} bytes)")
    if raw[:5] != MAGIC[:5]:
        raise RecordError(f"{path}: not a truestar recording (bad magic)")
    if raw[:8] != MAGIC:
        raise RecordError(f"{path}: unsupported recording format version {raw[5:8]!r}; this "
                          f"truestar reads {MAGIC[5:8]!r}")
    body, digest = raw[:-32], raw[-32:]
    if hashlib.sha256(body).digest() != digest:
        raise RecordError(f"{path}: checksum mismatch: the file is truncated or was modified")
    (hlen,) = struct.unpack("<Q", body[8:16])
    if 16 + hlen > len(body):
        raise RecordError(f"{path}: header length {hlen} runs past the end of the file")
    try:
        header = json.loads(body[16:16 + hlen].decode())
    except Exception as exc:
        raise RecordError(f"{path}: header is not valid JSON ({exc})") from exc
    if header.get("format") != FORMAT or header.get("version") != VERSION:
        raise RecordError(f"{path}: unsupported recording {header.get('format')!r} version "
                          f"{header.get('version')!r}")
    blobs = memoryview(body)[16 + hlen:]
    total = sum(e["nbytes"] for e in header.get("blobs", []))
    if total != len(blobs):
        raise RecordError(f"{path}: blob area is {len(blobs)} bytes, header says {total}")
    return header, blobs


def blob_tensor(entry: Dict[str, Any], blobs: memoryview) -> torch.Tensor:
    dt = dtype_from(entry["dtype"])
    shape = tuple(entry["shape"])
    if entry["nbytes"] == 0:
        return torch.empty(shape, dtype=dt)
    raw = bytearray(blobs[entry["offset"]:entry["offset"] + entry["nbytes"]])
    return torch.frombuffer(raw, dtype=torch.uint8).view(dt).reshape(shape)


def decode(tree: Any, tensor, _objs: Optional[Dict[int, torch.Tensor]] = None) -> Any:
    """Inverse of encode; `tensor(i)` returns blob i. Tensors that were one object when recorded
    are one (fresh) object again. Raises Unreplayable on opaque parts."""
    objs = {} if _objs is None else _objs
    if not isinstance(tree, dict):
        raise RecordError(f"bad argument tree node {tree!r}")
    if "T" in tree:
        n = tree.get("id")
        if n is None:
            return tensor(tree["T"]).clone()
        if n not in objs:
            objs[n] = tensor(tree["T"]).clone()
        return objs[n]
    if "v" in tree:
        return tree["v"]
    if "f" in tree:
        return float(tree["f"])
    if "tuple" in tree:
        return tuple(decode(x, tensor, objs) for x in tree["tuple"])
    if "list" in tree:
        return [decode(x, tensor, objs) for x in tree["list"]]
    if "dict" in tree:
        return {k: decode(v, tensor, objs) for k, v in tree["dict"]}
    if "dtype" in tree and len(tree) == 1:
        return dtype_from(tree["dtype"])
    if "device" in tree:
        return torch.device(tree["device"])
    if "memfmt" in tree:
        return getattr(torch, tree["memfmt"])
    if "layout" in tree:
        return getattr(torch, tree["layout"])
    if "H" in tree:
        raise Unreplayable("the tensor was stored as a hash only (outputs only recording)")
    if "opaque" in tree:
        raise Unreplayable(f"an argument of type {tree['opaque']} could not be recorded")
    raise RecordError(f"bad argument tree node {tree!r}")
