"""GGUF files: write, read and decode, implemented from the published format description.

A GGUF file is a little endian header (magic, version 3, tensor count, metadata count), the
metadata as typed key value pairs, one info record per tensor (name, dims in ggml order, type,
offset into the data section), padding to `general.alignment` (default 32) and the tensor data.
Dims are stored innermost first: a torch weight `[out, in]` is `ne = [in, out]`.

Truestar uses this to build tiny test models and to decode the weights an engine actually loads,
so it can compare them with the reference by content. Decoding is exact: every block format below
is decoded to float64 from its integer codes and float16 scales with no rounding of its own.
Nothing here quantizes weights for an engine to run; that is the engine's own tool.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"GGUF"
VERSION = 3
DEFAULT_ALIGNMENT = 32

# metadata value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?",
           U64: "<Q", I64: "<q", F64: "<d"}

# tensor types: id -> (name, elements per block, bytes per block)
TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}
TYPE_ID = {v[0]: k for k, v in TYPES.items()}
FLOAT_TYPES = ("F32", "F16", "BF16")


class GGUFError(ValueError):
    """A GGUF file is malformed or holds a tensor type Truestar cannot decode."""


# ----------------------------------------------------------------------------- writing


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _infer_type(v: Any) -> int:
    if isinstance(v, bool):
        return BOOL
    if isinstance(v, int):
        return U32 if 0 <= v < 2 ** 32 else I64
    if isinstance(v, float):
        return F32
    if isinstance(v, str):
        return STRING
    if isinstance(v, (list, tuple)):
        return ARRAY
    raise GGUFError(f"cannot store metadata value of type {type(v).__name__}")


def _pack_value(vtype: int, v: Any, elem_type: Optional[int] = None) -> bytes:
    if vtype == STRING:
        return _pack_string(v)
    if vtype == ARRAY:
        items = list(v)
        et = elem_type if elem_type is not None else (_infer_type(items[0]) if items else U32)
        if et == ARRAY:
            raise GGUFError("nested arrays are not written")
        out = [struct.pack("<IQ", et, len(items))]
        if et == STRING:
            out.extend(_pack_string(x) for x in items)
        else:
            out.append(np.asarray(items, dtype=np.dtype(_SCALAR[et])).tobytes())
        return b"".join(out)
    return struct.pack(_SCALAR[vtype], v)


@dataclass
class Meta:
    """A metadata value with an explicit GGUF type (plain Python values get an inferred type)."""
    value: Any
    vtype: int
    elem_type: Optional[int] = None


@dataclass
class TensorData:
    name: str
    type_name: str
    ne: Tuple[int, ...]  # ggml order, innermost first
    data: bytes


def float_tensor(name: str, t: Any, type_name: str = "F32") -> TensorData:
    """A float tensor in torch layout (`[out, in]` for a weight) stored as F32, F16 or BF16."""
    a = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
    ne = tuple(reversed(a.shape)) if a.ndim else (1,)
    if type_name == "F32":
        raw = a.astype("<f4").tobytes()
    elif type_name == "F16":
        raw = a.astype("<f2").tobytes()
    elif type_name == "BF16":
        f = a.astype("<f4").view("<u4").astype(np.uint64)
        # round to nearest even on the upper 16 bits
        f = (f + 0x7FFF + ((f >> 16) & 1)) >> 16
        raw = f.astype("<u2").tobytes()
    else:
        raise GGUFError(f"float_tensor writes F32, F16 or BF16, not {type_name}")
    return TensorData(name, type_name, ne, raw)


def write(path: str, metadata: Dict[str, Any], tensors: Sequence[TensorData],
          alignment: int = DEFAULT_ALIGNMENT) -> None:
    """Write a GGUF v3 file atomically. Metadata values are Python values or `Meta`."""
    names = [t.name for t in tensors]
    if len(set(names)) != len(names):
        raise GGUFError("duplicate tensor names")
    md = dict(metadata)
    if alignment != DEFAULT_ALIGNMENT:
        md["general.alignment"] = Meta(alignment, U32)
    head = [MAGIC, struct.pack("<IQQ", VERSION, len(tensors), len(md))]
    for k, v in md.items():
        m = v if isinstance(v, Meta) else Meta(v, _infer_type(v))
        head.append(_pack_string(k) + struct.pack("<I", m.vtype) + _pack_value(m.vtype, m.value, m.elem_type))
    offset = 0
    offsets = []
    for t in tensors:
        _, bs, ts = TYPES[TYPE_ID[t.type_name]]
        n = int(np.prod(t.ne))
        if n % bs or len(t.data) != n // bs * ts:
            raise GGUFError(f"{t.name}: {len(t.data)} bytes do not fit {t.type_name} with ne {t.ne}")
        offsets.append(offset)
        offset += -(-len(t.data) // alignment) * alignment
    for t, off in zip(tensors, offsets):
        head.append(_pack_string(t.name) + struct.pack("<I", len(t.ne))
                    + struct.pack(f"<{len(t.ne)}Q", *t.ne) + struct.pack("<IQ", TYPE_ID[t.type_name], off))
    blob = b"".join(head)
    blob += b"\0" * (-len(blob) % alignment)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
        for t in tensors:
            f.write(t.data)
            f.write(b"\0" * (-len(t.data) % alignment))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- reading


@dataclass
class TensorInfo:
    name: str
    type_id: int
    ne: Tuple[int, ...]
    offset: int  # absolute file offset of the data

    @property
    def type_name(self) -> str:
        return TYPES.get(self.type_id, (f"type{self.type_id}", 0, 0))[0]

    @property
    def shape(self) -> Tuple[int, ...]:
        """torch layout (outermost first)."""
        return tuple(reversed(self.ne))

    @property
    def nbytes(self) -> int:
        if self.type_id not in TYPES:
            raise GGUFError(f"{self.name}: unknown tensor type id {self.type_id}")
        _, bs, ts = TYPES[self.type_id]
        return int(np.prod(self.ne)) // bs * ts


@dataclass
class GGUFFile:
    path: str
    version: int
    metadata: Dict[str, Any]
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)

    def raw(self, name: str) -> bytes:
        ti = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(ti.offset)
            b = f.read(ti.nbytes)
        if len(b) != ti.nbytes:
            raise GGUFError(f"{name}: file ends inside the tensor data")
        return b

    def tensor(self, name: str) -> np.ndarray:
        """The tensor decoded exactly to float64, in torch layout."""
        ti = self.tensors[name]
        return dequantize(self.raw(name), ti.type_name, int(np.prod(ti.ne))).reshape(ti.shape)

    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", ""))


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt: str):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError("file ends inside the header")
        return struct.unpack(fmt, b)

    def string(self) -> str:
        (n,) = self.take("<Q")
        if n > 1 << 30:
            raise GGUFError(f"string of {n} bytes in the header")
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError("file ends inside a string")
        return b.decode("utf-8", errors="replace")

    def value(self, vtype: int) -> Any:
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            et, n = self.take("<IQ")
            if et == STRING:
                return [self.string() for _ in range(n)]
            if et == ARRAY:
                return [self.value(ARRAY) for _ in range(n)]
            dt = np.dtype(_SCALAR[et])
            b = self.f.read(dt.itemsize * n)
            return np.frombuffer(b, dtype=dt).tolist()
        if vtype not in _SCALAR:
            raise GGUFError(f"unknown metadata type {vtype}")
        return self.take(_SCALAR[vtype])[0]


def read(path: str) -> GGUFFile:
    """Read the header and tensor infos (no tensor data)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file (bad magic)")
        r = _Reader(f)
        (version,) = r.take("<I")
        if version not in (2, 3):
            raise GGUFError(f"GGUF version {version} is not supported (2 or 3)")
        n_tensors, n_kv = r.take("<QQ")
        md: Dict[str, Any] = {}
        for _ in range(n_kv):
            k = r.string()
            (vt,) = r.take("<I")
            if k in md:
                raise GGUFError(f"metadata key {k!r} appears twice")
            md[k] = r.value(vt)
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            (nd,) = r.take("<I")
            ne = r.take(f"<{nd}Q")
            tid, off = r.take("<IQ")
            infos.append((name, tid, tuple(int(x) for x in ne), off))
        align = int(md.get("general.alignment", DEFAULT_ALIGNMENT))
        start = f.tell()
        start += -start % align
    g = GGUFFile(path, version, md)
    for name, tid, ne, off in infos:
        if name in g.tensors:
            raise GGUFError(f"tensor {name!r} appears twice")
        ti = TensorInfo(name, tid, ne, start + off)
        if tid in TYPES and ti.offset + ti.nbytes > size:
            raise GGUFError(f"{name}: data runs past the end of the file")
        g.tensors[name] = ti
    return g


# ----------------------------------------------------------------------------- decoding


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view("<f2").astype(np.float64).reshape(-1)


def _blocks(raw: bytes, n: int, type_name: str) -> np.ndarray:
    _, bs, ts = TYPES[TYPE_ID[type_name]]
    if n % bs:
        raise GGUFError(f"{n} elements is not a whole number of {type_name} blocks")
    a = np.frombuffer(raw, dtype=np.uint8)
    if a.size != n // bs * ts:
        raise GGUFError(f"{a.size} bytes do not hold {n} {type_name} elements")
    return a.reshape(n // bs, ts)


def _scale_min_k4(sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The 8 six bit scales and 8 six bit mins packed in 12 bytes (K quant super blocks)."""
    s = sc.astype(np.int64)
    scale = np.empty(s.shape[:-1] + (8,), np.int64)
    mins = np.empty_like(scale)
    scale[..., :4] = s[..., 0:4] & 63
    mins[..., :4] = s[..., 4:8] & 63
    scale[..., 4:] = (s[..., 8:12] & 0xF) | ((s[..., 0:4] >> 6) << 4)
    mins[..., 4:] = (s[..., 8:12] >> 4) | ((s[..., 4:8] >> 6) << 4)
    return scale, mins


def dequantize(raw: bytes, type_name: str, n: int) -> np.ndarray:
    """Decode n elements of a GGUF tensor to float64, exactly."""
    if type_name == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n).astype(np.float64)
    if type_name == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n).astype(np.float64)
    if type_name == "BF16":
        u = np.frombuffer(raw, dtype="<u2", count=n).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    if type_name == "Q8_0":
        b = _blocks(raw, n, type_name)
        d = _f16(b[:, 0:2].copy())
        q = b[:, 2:34].view(np.int8).astype(np.float64)
        return (d[:, None] * q).reshape(-1)
    if type_name == "Q4_0":
        b = _blocks(raw, n, type_name)
        d = _f16(b[:, 0:2].copy())
        qs = b[:, 2:18].astype(np.int64)
        q = np.concatenate([qs & 0xF, qs >> 4], axis=1) - 8
        return (d[:, None] * q).reshape(-1)
    if type_name in ("Q4_K", "Q5_K"):
        b = _blocks(raw, n, type_name)
        d = _f16(b[:, 0:2].copy())
        dmin = _f16(b[:, 2:4].copy())
        scale, mins = _scale_min_k4(b[:, 4:16])
        if type_name == "Q4_K":
            qs = b[:, 16:144].astype(np.int64).reshape(-1, 4, 32)
            hi = None
        else:
            qh = b[:, 16:48].astype(np.int64)  # 32 bytes, bit 2j and 2j+1 for chunk j
            qs = b[:, 48:176].astype(np.int64).reshape(-1, 4, 32)
            hi = qh
        out = np.empty((b.shape[0], 4, 64), np.float64)
        for j in range(4):
            lo_q = qs[:, j, :] & 0xF
            up_q = qs[:, j, :] >> 4
            if hi is not None:
                lo_q = lo_q + ((hi >> (2 * j)) & 1) * 16
                up_q = up_q + ((hi >> (2 * j + 1)) & 1) * 16
            s1, m1 = scale[:, 2 * j], mins[:, 2 * j]
            s2, m2 = scale[:, 2 * j + 1], mins[:, 2 * j + 1]
            out[:, j, :32] = (d * s1)[:, None] * lo_q - (dmin * m1)[:, None]
            out[:, j, 32:] = (d * s2)[:, None] * up_q - (dmin * m2)[:, None]
        return out.reshape(-1)
    if type_name == "Q6_K":
        b = _blocks(raw, n, type_name)
        ql = b[:, 0:128].astype(np.int64)
        qh = b[:, 128:192].astype(np.int64)
        sc = b[:, 192:208].view(np.int8).astype(np.float64)
        d = _f16(b[:, 208:210].copy())
        out = np.empty((b.shape[0], 256), np.float64)
        for half in range(2):
            l_ = ql[:, 64 * half: 64 * half + 64]
            h_ = qh[:, 32 * half: 32 * half + 32]
            s_ = sc[:, 8 * half: 8 * half + 8]
            q1 = ((l_[:, :32] & 0xF) | (((h_ >> 0) & 3) << 4)) - 32
            q2 = ((l_[:, 32:] & 0xF) | (((h_ >> 2) & 3) << 4)) - 32
            q3 = ((l_[:, :32] >> 4) | (((h_ >> 4) & 3) << 4)) - 32
            q4 = ((l_[:, 32:] >> 4) | (((h_ >> 6) & 3) << 4)) - 32
            base = 128 * half
            for k, q in enumerate((q1, q2, q3, q4)):
                # elements l (0..31) of quarter k use scale index l // 16 + 2k
                sidx = np.where(np.arange(32) < 16, 2 * k, 2 * k + 1)
                out[:, base + 32 * k: base + 32 * k + 32] = d[:, None] * s_[:, sidx] * q
        return out.reshape(-1)
    raise GGUFError(f"tensor type {type_name} is not decoded by Truestar (F32, F16, BF16, Q4_0, Q8_0, "
                    "Q4_K, Q5_K, Q6_K are)")


def decodable(type_name: str) -> bool:
    return type_name in ("F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K")


# ----------------------------------------------------------------------------- activation rounding


def round_like_dot_type(x: np.ndarray, weight_type: str) -> np.ndarray:
    """What the ggml CPU backend does to the f32 activations before a dot product with weights of
    this type: F16 weights take F16 activations, Q8_0 and Q4_0 weights take Q8_0 blocks of 32, K
    quant weights take Q8_K blocks of 256. Rows are along the last axis. Used to model the rounding
    the engine is allowed to do, not to run anything."""
    x = np.asarray(x, dtype=np.float32)
    if weight_type in ("F32",):
        return x.astype(np.float64)
    if weight_type == "F16":
        return x.astype(np.float16).astype(np.float64)
    if weight_type == "BF16":
        f = x.view(np.uint32).astype(np.uint64)
        f = ((f + 0x7FFF + ((f >> 16) & 1)) >> 16) << 16
        return f.astype(np.uint32).view(np.float32).astype(np.float64)
    if weight_type in ("Q8_0", "Q4_0"):
        bs = 32
    elif weight_type in ("Q4_K", "Q5_K", "Q6_K"):
        bs = 256
    else:
        raise GGUFError(f"no activation rounding model for weights of type {weight_type}")
    shp = x.shape
    if shp[-1] % bs:
        raise GGUFError(f"row length {shp[-1]} is not a multiple of {bs}")
    blk = x.reshape(-1, bs)
    amax = np.abs(blk).max(axis=1)
    if bs == 32:
        d = (amax / np.float32(127)).astype(np.float32)
        inv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, 1), 0).astype(np.float32)
        q = np.rint(blk * inv[:, None])  # nearest; ties are measure zero for real data
        d = d.astype(np.float16).astype(np.float64)  # the Q8_0 scale is stored as f16
    else:
        # Q8_K: scale from the signed extreme, iscale = -127 / max, stored as f32
        idx = np.abs(blk).argmax(axis=1)
        mx = blk[np.arange(blk.shape[0]), idx]
        inv = np.where(amax > 0, np.float32(-127) / np.where(amax > 0, mx, 1), 0).astype(np.float32)
        q = np.minimum(np.rint(blk * inv[:, None]), 127)
        d = np.where(inv != 0, np.float32(1) / np.where(inv != 0, inv, 1), 0).astype(np.float64)
    return (q.astype(np.float64) * d[:, None]).reshape(shp)


def iter_types(g: GGUFFile) -> Iterable[Tuple[str, str]]:
    for name, ti in g.tensors.items():
        yield name, ti.type_name
