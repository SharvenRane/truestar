"""On disk format for Truestar IR programs (``.tsx`` files).

One file holds the whole program: inputs with their symbolic dims, params with their weights, every
node with its op name, typed attributes and source location, the outputs, and free form metadata
(the export path stores its verification evidence there). Format version 1, all integers little
endian::

    offset  size  field
    0       8     magic b"TRUESTAR"
    8       4     u32 format version (1)
    12      4     u32 reserved, must be 0
    16      8     u64 header length H
    24      H     header: UTF 8 JSON, sorted keys, no whitespace
    ...           zero padding up to a multiple of 64
    ...     D     data section: raw little endian tensor bytes, each tensor 64 byte aligned
    end-32  32    sha256 of every byte before it

The header is ``{"format", "format_version", "program_hash", "data_nbytes", "program",
"metadata"}``. ``program_hash`` is the sha256 of the canonical program JSON plus the data section,
so it identifies the program and its weights and does not change when only the metadata does.
Writing the same program twice gives the same bytes.

Ops are encoded generically: op name, input value names, attributes, the output value and, for
nodes that define several values (``cond``, ``loop``), the extra outputs. Attribute
values carry a type tag (``int``, ``float`` as an exact hex string, ``bool``, ``str``, ``none``,
``list``, ``tuple``, ``dict``, ``dtype``, ``dim`` for symbolic expressions, ``value`` for a
reference to an SSA value, ``program`` for a nested program such as a control flow region, and
``tensor`` for tensor data). A new op whose attributes use these types needs no change here. Any
other attribute type is refused with an error that names the op, the node and the attribute.

Reading checks, in this order, and raises ``ArtifactError`` on the first failure: the magic, the
format version (an unknown version is never guessed at), the file checksum (and, when the header
still parses, which tensors were damaged), the program hash, every op name, and finally
``Program.verify()`` (SSA order plus every shape rule).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sympy
import torch

from .core import IRError, Node, Program, TensorType, Value, norm_dim, symbol

FORMAT = "truestar.program"
FORMAT_VERSION = 1
MAGIC = b"TRUESTAR"
ALIGN = 64
_PREFIX = struct.Struct("<8sIIQ")  # magic, version, reserved, header length
_DIGEST = 32

_DTYPES = {
    torch.float64: "f64", torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.int64: "i64", torch.int32: "i32", torch.int16: "i16", torch.int8: "i8",
    torch.uint8: "u8", torch.bool: "bool",
}
_DTYPE_BY_NAME = {v: k for k, v in _DTYPES.items()}

# sympy functions a dim expression may use. Anything else is refused on write and on read.
_EXPR_FNS = {
    "Add": sympy.Add, "Mul": sympy.Mul, "Pow": sympy.Pow, "Mod": sympy.Mod,
    "floor": sympy.floor, "ceiling": sympy.ceiling, "Max": sympy.Max, "Min": sympy.Min,
}

_KNOWN_PROGRAM_FIELDS = {"name", "inputs", "params", "param_data", "nodes", "outputs", "symbols",
                         "state", "state_init", "key"}   # the last three: optional keys, absent when empty
_KNOWN_NODE_FIELDS = {"op", "inputs", "attrs", "output", "source", "extra_outputs"}
_KNOWN_VALUE_FIELDS = {"name", "type", "kind"}

SUPPORTED_ATTR_TYPES = ("None", "bool", "int", "float", "str", "list", "tuple", "dict with str keys",
                        "torch.dtype", "sympy expression (dim)", "Value (reference)", "Program (nested)",
                        "torch.Tensor")


class SerializeError(IRError):
    """The program holds something the format cannot store. Raised, never skipped."""


class ArtifactError(IRError):
    """A file is not a valid artifact: wrong magic, unknown version, corrupted, or inconsistent."""


@dataclass
class Artifact:
    """A loaded artifact: the program, its metadata, and its identity."""

    program: Program
    metadata: Dict[str, Any] = field(default_factory=dict)
    program_hash: str = ""
    format_version: int = FORMAT_VERSION


# ----------------------------------------------------------------------------- helpers


def _canonical(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise SerializeError(f"metadata is not plain JSON (str, int, finite float, bool, None, list, "
                             f"dict): {e}") from None


def _check_fields(obj: Any, known: set, what: str) -> None:
    names = {f.name for f in dataclasses.fields(obj)}
    extra = sorted(names - known)
    if extra:
        raise SerializeError(f"{what} has field(s) {extra} that format version {FORMAT_VERSION} cannot "
                             "store; extend truestar.ir.serialize and bump FORMAT_VERSION instead of "
                             "dropping them")


def _dtype_name(dtype: torch.dtype, what: str) -> str:
    name = _DTYPES.get(dtype)
    if name is None:
        raise SerializeError(f"{what}: dtype {dtype} has no encoding (supported: {sorted(_DTYPE_BY_NAME)})")
    return name


def _dtype_of(name: Any, what: str) -> torch.dtype:
    dtype = _DTYPE_BY_NAME.get(name)
    if dtype is None:
        raise ArtifactError(f"{what}: unknown dtype {name!r}")
    return dtype


def _expr_to_json(e: sympy.Basic, what: str) -> Any:
    if isinstance(e, sympy.Integer):
        return int(e)
    if isinstance(e, sympy.Rational):
        return {"rat": [int(e.p), int(e.q)]}
    if isinstance(e, sympy.Symbol):
        if e != symbol(e.name):
            raise SerializeError(f"{what}: symbol {e} does not have the IR's assumptions (positive "
                                 "integer); build dims with truestar.ir.symbol")
        return {"sym": e.name}
    name = type(e).__name__
    if name not in _EXPR_FNS:
        raise SerializeError(f"{what}: dim expression {e} uses {name}, which has no encoding "
                             f"(supported: {sorted(_EXPR_FNS)})")
    return {"fn": name, "args": [_expr_to_json(a, what) for a in e.args]}


def _expr_from_json(j: Any, what: str) -> sympy.Expr:
    if isinstance(j, bool):
        raise ArtifactError(f"{what}: bad dim expression {j!r}")
    if isinstance(j, int):
        return sympy.Integer(j)
    if isinstance(j, dict) and set(j) == {"rat"}:
        p, q = j["rat"]
        return sympy.Rational(p, q)
    if isinstance(j, dict) and set(j) == {"sym"}:
        return symbol(j["sym"])
    if isinstance(j, dict) and set(j) == {"fn", "args"} and j["fn"] in _EXPR_FNS:
        return _EXPR_FNS[j["fn"]](*[_expr_from_json(a, what) for a in j["args"]])
    raise ArtifactError(f"{what}: bad dim expression {j!r}")


def _dim_to_json(d: Any, what: str) -> Any:
    d = norm_dim(d)
    return d if isinstance(d, int) else {"expr": _expr_to_json(d, what)}


def _dim_from_json(j: Any, what: str) -> Any:
    if isinstance(j, int) and not isinstance(j, bool):
        return j
    if isinstance(j, dict) and set(j) == {"expr"}:
        return norm_dim(_expr_from_json(j["expr"], what))
    raise ArtifactError(f"{what}: bad dim {j!r}")


def _type_to_json(t: TensorType, what: str) -> Dict[str, Any]:
    return {"dtype": _dtype_name(t.dtype, what), "shape": [_dim_to_json(d, what) for d in t.shape]}


def _type_from_json(j: Dict[str, Any], what: str) -> TensorType:
    return TensorType(_dtype_of(j["dtype"], what), tuple(_dim_from_json(d, what) for d in j["shape"]))


# ----------------------------------------------------------------------------- encoder


class _Encoder:
    def __init__(self):
        self.chunks: List[bytes] = []
        self.size = 0

    def tensor(self, t: torch.Tensor, what: str) -> Dict[str, Any]:
        if not isinstance(t, torch.Tensor):
            raise SerializeError(f"{what}: expected a tensor, got {type(t).__name__}")
        if t.layout != torch.strided or t.is_quantized:
            raise SerializeError(f"{what}: only dense tensors can be stored, got layout {t.layout}")
        if sys.byteorder != "little":
            raise SerializeError("writing artifacts on a big endian host is not supported")
        dtype = _dtype_name(t.dtype, what)
        t = t.detach().cpu().contiguous()
        raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        pad = (-self.size) % ALIGN
        if pad:
            self.chunks.append(b"\0" * pad)
            self.size += pad
        ref = {"dtype": dtype, "shape": [int(s) for s in t.shape], "offset": self.size,
               "nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.chunks.append(raw)
        self.size += len(raw)
        return ref

    def program(self, prog: Program, outer: List[Dict[str, Value]]) -> Dict[str, Any]:
        _check_fields(prog, _KNOWN_PROGRAM_FIELDS, "Program")
        scope: Dict[str, Value] = {}
        scopes = outer + [scope]

        def define(v: Value, kind: str, what: str) -> Dict[str, Any]:
            _check_fields(v, _KNOWN_VALUE_FIELDS, "Value")
            if v.kind != kind:
                raise SerializeError(f"{what}: value {v.ref} has kind {v.kind!r}, expected {kind!r}")
            if v.name in scope:
                raise SerializeError(f"{what}: value name {v.name!r} is defined twice in program {prog.name}")
            scope[v.name] = v
            return {"name": v.name, "type": _type_to_json(v.type, what)}

        def ref(v: Optional[Value], what: str) -> Optional[str]:
            if v is None:
                return None
            if not isinstance(v, Value):
                raise SerializeError(f"{what}: expected a Value, got {type(v).__name__}")
            for s in reversed(scopes):
                if v.name in s:
                    if s[v.name] is not v:
                        raise SerializeError(f"{what}: {v.ref} is shadowed by another value with the same "
                                             "name, so a name reference would be ambiguous")
                    return v.name
            raise SerializeError(f"{what}: {v.ref} is not defined before it is used")

        out: Dict[str, Any] = {"name": prog.name}
        out["symbols"] = [[str(k), int(lo), None if hi is None else int(hi)] for k, (lo, hi) in prog.symbols.items()]
        out["inputs"] = [define(v, "input", f"input %{v.name}") for v in prog.inputs]
        params = []
        for v in prog.params:
            what = f"param @{v.name}"
            j = define(v, "param", what)
            if v.name not in prog.param_data:
                raise SerializeError(f"{what} has no data")
            j["data"] = self.tensor(prog.param_data[v.name], what)
            params.append(j)
        out["params"] = params
        nodes = []
        for node in prog.nodes:
            _check_fields(node, _KNOWN_NODE_FIELDS, "Node")
            what = f"{node.op} node %{node.output.name}"
            if not isinstance(node.op, str):
                raise SerializeError(f"{what}: op name must be a str")
            if not isinstance(node.attrs, dict):
                raise SerializeError(f"{what}: attrs must be a dict")
            j = {"op": node.op, "inputs": [ref(x, what) for x in node.inputs]}
            attrs = []
            for k, v in node.attrs.items():
                if not isinstance(k, str):
                    raise SerializeError(f"{what}: attribute name {k!r} is not a str")
                attrs.append([k, self.attr(v, f"attribute {k!r} of {what}", scopes)])
            j["attrs"] = attrs
            j["output"] = define(node.output, "op", what)
            extra = tuple(getattr(node, "extra_outputs", ()) or ())
            if type(getattr(node, "extra_outputs", ())) is not tuple:
                raise SerializeError(f"{what}: extra_outputs must be a tuple of Values")
            if extra:  # omitted when empty, so single output programs keep their bytes
                j["extra_outputs"] = [define(v, "op", what) for v in extra]
            j["source"] = str(node.source or "")
            nodes.append(j)
        out["nodes"] = nodes
        out["outputs"] = [ref(v, f"output {getattr(v, 'ref', v)}") for v in prog.outputs]
        if prog.state:   # training state (docs/RANDOM.md); written only when present, so older hashes hold
            out["state"] = [[str(t), int(i), int(o)] for t, i, o in prog.state]
            out["state_init"] = [[str(t), self.tensor(prog.state_init[t], f"state {t}")]
                                 for t, _, _ in prog.state if t in prog.state_init]
        if prog.key is not None:
            out["key"] = str(prog.key)
        return out

    def attr(self, v: Any, what: str, scopes: List[Dict[str, Value]]) -> Dict[str, Any]:
        t = type(v)
        if v is None:
            return {"none": None}
        if t is bool:
            return {"bool": v}
        if t is int:
            return {"int": v}
        if t is float:
            return {"float": v.hex()}
        if t is str:
            return {"str": v}
        if t is list:
            return {"list": [self.attr(x, f"{what}[{i}]", scopes) for i, x in enumerate(v)]}
        if t is tuple:
            return {"tuple": [self.attr(x, f"{what}[{i}]", scopes) for i, x in enumerate(v)]}
        if t is dict:
            items = []
            for k, x in v.items():
                if type(k) is not str:
                    raise SerializeError(f"{what}: dict key {k!r} is not a str")
                items.append([k, self.attr(x, f"{what}[{k!r}]", scopes)])
            return {"dict": items}
        if isinstance(v, torch.dtype):
            return {"dtype": _dtype_name(v, what)}
        if isinstance(v, sympy.Basic):
            return {"dim": _expr_to_json(v, what)}
        if isinstance(v, Value):
            for s in reversed(scopes):
                if v.name in s:
                    if s[v.name] is not v:
                        raise SerializeError(f"{what}: {v.ref} is shadowed by another value with the same name")
                    return {"value": v.name}
            raise SerializeError(f"{what}: {v.ref} is not defined before it is used")
        if isinstance(v, Program):
            return {"program": self.program(v, scopes)}
        if isinstance(v, torch.Tensor):
            return {"tensor": self.tensor(v, what)}
        raise SerializeError(f"cannot serialize {what}: a {t.__module__}.{t.__qualname__} ({_short(v)}) has "
                             f"no encoding. Supported attribute types: {', '.join(SUPPORTED_ATTR_TYPES)}")


def _short(v: Any) -> str:
    r = repr(v)
    return r if len(r) <= 80 else r[:77] + "..."


# ----------------------------------------------------------------------------- decoder


class _Decoder:
    def __init__(self, data: memoryview):
        self.data = data

    def tensor(self, j: Dict[str, Any], what: str) -> torch.Tensor:
        dtype = _dtype_of(j["dtype"], what)
        shape = [int(s) for s in j["shape"]]
        off, n = int(j["offset"]), int(j["nbytes"])
        itemsize = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for s in shape:
            numel *= s
        if n != numel * itemsize or off < 0 or off + n > len(self.data):
            raise ArtifactError(f"{what}: tensor bytes [{off}, {off + n}) do not fit shape {shape} {dtype} "
                                f"inside a data section of {len(self.data)} bytes")
        if numel == 0:
            return torch.empty(shape, dtype=dtype)
        raw = bytearray(self.data[off:off + n])
        return torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape)

    def program(self, j: Dict[str, Any], outer: List[Dict[str, Value]]) -> Program:
        scope: Dict[str, Value] = {}
        scopes = outer + [scope]
        name = j["name"]

        def define(vj: Dict[str, Any], kind: str, what: str) -> Value:
            v = Value(vj["name"], _type_from_json(vj["type"], what), kind)
            if v.name in scope:
                raise ArtifactError(f"{what}: value name {v.name!r} is defined twice in program {name}")
            scope[v.name] = v
            return v

        def ref(n: Optional[str], what: str) -> Optional[Value]:
            if n is None:
                return None
            for s in reversed(scopes):
                if n in s:
                    return s[n]
            raise ArtifactError(f"{what}: reference to undefined value {n!r}")

        symbols = {s: (lo, hi) for s, lo, hi in j["symbols"]}
        inputs = [define(v, "input", f"input %{v['name']}") for v in j["inputs"]]
        params, param_data = [], {}
        for pj in j["params"]:
            what = f"param @{pj['name']}"
            v = define(pj, "param", what)
            params.append(v)
            param_data[v.name] = self.tensor(pj["data"], what)
        nodes = []
        for nj in j["nodes"]:
            what = f"{nj['op']} node %{nj['output']['name']}"
            ins = tuple(ref(x, what) for x in nj["inputs"])
            attrs = {k: self.attr(a, f"attribute {k!r} of {what}", scopes) for k, a in nj["attrs"]}
            out = define(nj["output"], "op", what)
            extra = tuple(define(v, "op", what) for v in nj.get("extra_outputs", []))
            nodes.append(Node(nj["op"], ins, attrs, out, nj.get("source", ""), extra))
        outputs = [ref(n, "program outputs") for n in j["outputs"]]
        prog = Program(name, inputs, params, param_data, nodes, outputs, symbols)
        prog.state = [(t, int(i), int(o)) for t, i, o in j.get("state", [])]
        prog.state_init = {t: self.tensor(tj, f"state {t}") for t, tj in j.get("state_init", [])}
        prog.key = j.get("key")
        return prog

    def attr(self, j: Dict[str, Any], what: str, scopes: List[Dict[str, Value]]) -> Any:
        if not isinstance(j, dict) or len(j) != 1:
            raise ArtifactError(f"{what}: bad attribute encoding {_short(j)}")
        (tag, v), = j.items()
        if tag == "none":
            return None
        if tag in ("bool", "int", "str"):
            return v
        if tag == "float":
            return float.fromhex(v)
        if tag == "list":
            return [self.attr(x, what, scopes) for x in v]
        if tag == "tuple":
            return tuple(self.attr(x, what, scopes) for x in v)
        if tag == "dict":
            return {k: self.attr(x, what, scopes) for k, x in v}
        if tag == "dtype":
            return _dtype_of(v, what)
        if tag == "dim":
            return norm_dim(_expr_from_json(v, what))
        if tag == "value":
            for s in reversed(scopes):
                if v in s:
                    return s[v]
            raise ArtifactError(f"{what}: reference to undefined value {v!r}")
        if tag == "program":
            return self.program(v, scopes)
        if tag == "tensor":
            return self.tensor(v, what)
        raise ArtifactError(f"{what}: unknown attribute type tag {tag!r}; the artifact was probably written by "
                            "a newer truestar")


def _ops_in(j: Dict[str, Any], acc: set) -> set:
    """Every op name in a program JSON, including nested programs in attributes."""
    for nj in j["nodes"]:
        acc.add(nj["op"])
        for _, a in nj["attrs"]:
            _ops_in_attr(a, acc)
    return acc


def _ops_in_attr(a: Any, acc: set) -> None:
    if isinstance(a, dict):
        if "program" in a and len(a) == 1:
            _ops_in(a["program"], acc)
            return
        for v in a.values():
            _ops_in_attr(v, acc)
    elif isinstance(a, list):
        for v in a:
            _ops_in_attr(v, acc)


# ----------------------------------------------------------------------------- public API


def _encode(program: Program) -> Tuple[Dict[str, Any], bytes, str]:
    enc = _Encoder()
    pj = enc.program(program, [])
    data = b"".join(enc.chunks)
    phash = hashlib.sha256(_canonical(pj) + b"\0" + data).hexdigest()
    return pj, data, phash


def program_hash(program: Program) -> str:
    """sha256 identifying the program and its weights (metadata excluded)."""
    return _encode(program)[2]


def dumps(program: Program, metadata: Optional[Dict[str, Any]] = None) -> bytes:
    """Serialize ``program`` (and plain JSON ``metadata``) to bytes. Same input, same bytes."""
    pj, data, phash = _encode(program)
    header = _canonical({"format": FORMAT, "format_version": FORMAT_VERSION, "program_hash": phash,
                         "data_nbytes": len(data), "program": pj, "metadata": metadata or {}})
    head = _PREFIX.pack(MAGIC, FORMAT_VERSION, 0, len(header)) + header
    head += b"\0" * ((-len(head)) % ALIGN)
    body = head + data
    return body + hashlib.sha256(body).digest()


def loads(blob: bytes, verify: bool = True) -> Artifact:
    """Parse an artifact. Raises ArtifactError on any problem (see the module docstring for the order)."""
    blob = bytes(blob)
    if len(blob) < _PREFIX.size + _DIGEST:
        raise ArtifactError(f"not a truestar artifact: only {len(blob)} bytes")
    magic, version, reserved, hlen = _PREFIX.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ArtifactError(f"not a truestar artifact: magic is {magic!r}, expected {MAGIC!r}")
    if version != FORMAT_VERSION:
        newer = version > FORMAT_VERSION
        raise ArtifactError(f"unsupported artifact format version {version}; this truestar reads version "
                            f"{FORMAT_VERSION} only" + (". The file was written by a newer truestar: upgrade "
                                                         "truestar to load it" if newer else ""))
    if reserved != 0:
        raise ArtifactError(f"artifact header reserved field is {reserved}, expected 0 (corrupted file?)")
    body, digest = blob[:-_DIGEST], blob[-_DIGEST:]
    header_end = _PREFIX.size + hlen
    data_start = header_end + ((-header_end) % ALIGN)
    if hashlib.sha256(body).digest() != digest:
        raise ArtifactError("artifact is corrupted: the file checksum does not match"
                            + _locate_damage(blob, header_end, data_start))
    try:
        header = json.loads(blob[_PREFIX.size:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ArtifactError(f"artifact header is not valid JSON: {e}") from None
    if header.get("format") != FORMAT or header.get("format_version") != version:
        raise ArtifactError(f"artifact header says format {header.get('format')!r} version "
                            f"{header.get('format_version')!r}, expected {FORMAT!r} version {version}")
    data = memoryview(body)[data_start:]
    if len(data) != header["data_nbytes"]:
        raise ArtifactError(f"artifact data section is {len(data)} bytes, header says {header['data_nbytes']}")
    pj = header["program"]
    if verify:
        from .shapes import RULES
        unknown = sorted(_ops_in(pj, set()) - set(RULES))
        if unknown:
            raise ArtifactError(f"artifact uses IR op(s) {unknown} that this truestar does not define; it was "
                                "written by a newer truestar or by hand")
    try:
        prog = _Decoder(data).program(pj, [])
    except (KeyError, TypeError, ValueError) as e:
        raise ArtifactError(f"artifact program is malformed: {type(e).__name__}: {e}") from None
    phash = hashlib.sha256(_canonical(pj) + b"\0" + bytes(data)).hexdigest()
    if phash != header["program_hash"]:
        raise ArtifactError(f"artifact program hash {phash} does not match the recorded {header['program_hash']}")
    if verify:
        for v in prog.params:
            data_dtype = prog.param_data[v.name].dtype
            if data_dtype != v.type.dtype and not (data_dtype == torch.float64 and v.type.dtype.is_floating_point
                                                   and v.type.rank == 0):
                raise ArtifactError(f"param @{v.name}: stored {data_dtype} data for a {v.type} value")
        try:
            prog.verify()
        except IRError as e:
            raise ArtifactError(f"artifact program failed verification: {e}") from None
        from .. import compile_cache    # the hash was just checked: the compile cache need not recompute it
        compile_cache.remember_hash(prog, phash)
    return Artifact(prog, header["metadata"], phash, version)


def _locate_damage(blob: bytes, header_end: int, data_start: int) -> str:
    """Best effort: name the damaged tensors when the header itself still parses."""
    try:
        header = json.loads(blob[_PREFIX.size:header_end].decode("utf-8"))
        data = blob[data_start:-_DIGEST]
        bad = []

        def walk(pj, prefix):
            for p in pj["params"]:
                d = p["data"]
                if hashlib.sha256(data[d["offset"]:d["offset"] + d["nbytes"]]).hexdigest() != d["sha256"]:
                    bad.append(prefix + "@" + p["name"])
        walk(header["program"], "")
    except Exception:
        return " (the header is damaged too)"
    if bad:
        return f"; damaged tensor(s): {', '.join(bad[:10])}" + (" ..." if len(bad) > 10 else "")
    return "; the tensor data checks out, so the damage is in the header, padding or checksum"


def save(program: Program, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Write ``program`` to ``path`` atomically. Returns the program hash."""
    return write_bytes(dumps(program, metadata), path)


def write_bytes(blob: bytes, path: str) -> str:
    """Write already serialized artifact bytes atomically (temp file plus rename). Returns the program hash."""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tsx-", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())    # the bytes are on disk before the rename makes them the file
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    hlen = _PREFIX.unpack_from(blob, 0)[3]
    return json.loads(blob[_PREFIX.size:_PREFIX.size + hlen].decode("utf-8"))["program_hash"]


def load(path: str, verify: bool = True) -> Artifact:
    """Read an artifact file. See ``loads``."""
    with open(path, "rb") as f:
        return loads(f.read(), verify=verify)


__all__ = ["Artifact", "ArtifactError", "SerializeError", "FORMAT_VERSION", "dumps", "loads", "save", "load",
           "program_hash", "write_bytes"]
