"""Truestar IR to Core ML (an ML Program in a .mlpackage), with no coremltools.

coremltools has no Windows wheel, so the writer encodes the Core ML protobuf itself: the Model
message (specification version 6, iOS 15 / macOS 12) holding an ML Program in opset CoreML5, with
every weight as an immediate value (no weight blob file), inside the .mlpackage directory layout
(Manifest.json plus Data/com.apple.CoreML/model.mlmodel). Field numbers are those of Apple's
published schema (Model.proto, FeatureTypes.proto, MIL.proto).

Only Apple's Core ML framework runs the file, on macOS and iOS. Here the file is checked by
decoding its own bytes back (``decode``) and evaluating the decoded MIL program in float64
(``evaluate64``): that is the conversion verdict (the file's math against Truestar's float64
reference). The runtime verdict stays pending until the file runs on a Mac (TS1413).
"""

from __future__ import annotations

import hashlib
import json
import re
import math
import struct
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...ir.core import Value
from .lower import Lowered, LoweringError, pair, static_shape

T = "coreml"
SPEC_VERSION = 6
OPSET = "CoreML5"
MODEL_PATH = "Data/com.apple.CoreML/model.mlmodel"

# MIL DataType and ArrayFeatureType codes
_MIL = {"bool": 1, "string": 2, "fp32": 11, "int32": 23}
_ARRAY = {"fp32": 65568, "int32": 131104}

# ----------------------------------------------------------------------------- protobuf wire


def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wt: int) -> bytes:
    return _varint(field << 3 | wt)


def f_int(field: int, n: int) -> bytes:
    return _key(field, 0) + _varint(int(n))


def f_len(field: int, data: bytes) -> bytes:
    return _key(field, 2) + _varint(len(data)) + data


def f_str(field: int, s: str) -> bytes:
    return f_len(field, s.encode("utf-8"))


def f_map(field: int, items: List[Tuple[str, bytes]]) -> bytes:
    return b"".join(f_len(field, f_str(1, k) + f_len(2, v)) for k, v in items)


def parse(buf: bytes) -> Dict[int, List[Any]]:
    """Generic protobuf decode: field number to the list of its raw values (ints for varints,
    bytes for length delimited, fixed fields as bytes)."""
    out: Dict[int, List[Any]] = {}
    i, n = 0, len(buf)
    while i < n:
        k, i = _read_varint(buf, i)
        field, wt = k >> 3, k & 7
        if wt == 0:
            v, i = _read_varint(buf, i)
        elif wt == 2:
            ln, i = _read_varint(buf, i)
            v, i = buf[i:i + ln], i + ln
            if i > n:
                raise ValueError("truncated length delimited field")
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        else:
            raise ValueError(f"unsupported wire type {wt}")
        out.setdefault(field, []).append(v)
    return out


def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _signed(n: int) -> int:
    return n - (1 << 64) if n >= 1 << 63 else n

# ----------------------------------------------------------------------------- MIL values


def _tensor_type(dt: str, shape) -> bytes:
    body = f_int(1, _MIL[dt]) + f_int(2, len(shape))
    body += b"".join(f_len(3, f_len(1, f_int(1, int(d)))) for d in shape)
    return f_len(1, body)  # ValueType.tensorType


def _value(dt: str, arr: np.ndarray) -> bytes:
    arr = np.asarray(arr)
    if dt == "fp32":
        tv = f_len(1, f_len(1, np.ascontiguousarray(arr, "<f4").tobytes()))
    elif dt == "int32":
        tv = f_len(2, f_len(1, b"".join(_varint(int(v)) for v in arr.reshape(-1))))
    elif dt == "bool":
        tv = f_len(3, f_len(1, b"".join(_varint(int(bool(v))) for v in arr.reshape(-1))))
    elif dt == "string":
        tv = f_len(4, b"".join(f_str(1, str(v)) for v in arr.reshape(-1)))
    else:
        raise ValueError(dt)
    return f_len(2, _tensor_type(dt, arr.shape)) + f_len(3, f_len(1, tv))


def _dt(dtype: torch.dtype) -> str:
    if dtype.is_floating_point:
        return "fp32"
    if dtype == torch.bool:
        return "bool"
    return "int32"


class _Prog:
    def __init__(self):
        self.ops: List[bytes] = []
        self.n = 0
        self.types: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
        self._mil: Dict[str, str] = {}

    def mil(self, name: str) -> str:
        """The MIL identifier of a value name: Core ML's MIL parser takes [A-Za-z_][A-Za-z0-9_]* only
        (a parameter name like fc1.weight failed to load on macOS 26 with "got '.' when expecting
        token of type 'EQUAL'"), kept unique."""
        if name in self._mil:
            return self._mil[name]
        m = re.sub(r"\W", "_", name)
        if not m or m[0].isdigit():
            m = "v_" + m
        used = set(self._mil.values())
        while m in used:
            m += "_"
        self._mil[name] = m
        return m

    def fresh(self, base: str) -> str:
        self.n += 1
        return f"{base}_{self.n}"

    def _named(self, name: str, dt: str, shape) -> bytes:
        self.types[name] = (dt, tuple(int(s) for s in shape))
        return f_str(1, self.mil(name)) + f_len(2, _tensor_type(dt, shape))

    def const(self, base: str, arr, dt: Optional[str] = None) -> str:
        arr = np.asarray(arr)
        dt = dt or ("fp32" if arr.dtype.kind == "f" else "bool" if arr.dtype.kind == "b" else
                    "string" if arr.dtype.kind in "US" else "int32")
        name = self.fresh(base)
        attrs = [("name", _value("string", np.asarray(self.mil(name)))), ("val", _value(dt, arr))]
        self.ops.append(f_str(1, "const") + f_len(3, self._named(name, dt, arr.shape)) + f_map(5, attrs))
        return name

    def op(self, typ: str, inputs: Dict[str, Any], out: str, dt: str, shape) -> str:
        binds = []
        for k, v in inputs.items():
            if v is None:
                continue
            names = v if isinstance(v, list) else [v]
            binds.append((k, b"".join(f_len(1, f_str(1, self.mil(x))) for x in names)))
        attrs = [("name", _value("string", np.asarray(self.mil(out))))]
        self.ops.append(f_str(1, typ) + f_map(2, binds) + f_len(3, self._named(out, dt, shape)) + f_map(5, attrs))
        return out


def write(low: Lowered) -> Dict[str, bytes]:
    """The .mlpackage as {relative path: bytes}."""
    if not low.static():
        raise LoweringError(T, None, "Core ML files are written with static shapes (export with --static-batch)")
    p = _Prog()
    env: Dict[str, str] = {}
    shapes: Dict[str, Tuple[int, ...]] = {}
    fn_inputs = []
    for v in low.inputs:
        shp = static_shape(T, v)
        if v.type.dtype not in (torch.float32, torch.int64, torch.int32):
            raise LoweringError(T, None, f"input %{v.name} has dtype {v.type.dtype}")
        fn_inputs.append(p._named(v.name, _dt(v.type.dtype), shp))
        env[v.name], shapes[v.name] = v.name, shp

    def get(v: Value) -> str:
        if v.name in env:
            return env[v.name]
        t = low.consts[v.name]
        arr = t.float().numpy() if t.dtype.is_floating_point else t.numpy()
        if arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        env[v.name] = p.const(v.name, arr)
        shapes[v.name] = tuple(arr.shape)
        return env[v.name]

    def i32(base, vals):
        return p.const(base, np.asarray(vals, np.int32))

    def f32(base, val):
        return p.const(base, np.asarray(val, np.float32))

    def boolc(base, val):
        return p.const(base, np.asarray(val, np.bool_))

    def emit(n, typ, inputs, name=None, shape=None, dt=None):
        name = name or n.output.name
        shape = static_shape(T, n.output, n) if shape is None else shape
        out = p.op(typ, inputs, name, dt or _dt(n.output.type.dtype), shape)
        if name == n.output.name:
            env[name], shapes[name] = name, tuple(shape)
        return out

    unary = {"relu": "relu", "sigmoid": "sigmoid", "tanh": "tanh", "exp": "exp", "sqrt": "sqrt", "abs": "abs",
             "sin": "sin", "cos": "cos", "erf": "erf", "logical_not": "logical_not"}
    binary = {"add": "add", "sub": "sub", "mul": "mul", "div": "real_div", "maximum": "maximum",
              "minimum": "minimum", "eq": "equal", "lt": "less", "gt": "greater", "le": "less_equal",
              "ge": "greater_equal"}
    for n in low.nodes:
        a, ins, o = n.attrs, n.inputs, n.output.name
        if n.op in unary:
            emit(n, unary[n.op], {"x": get(ins[0])})
        elif n.op == "neg":
            emit(n, "mul", {"x": get(ins[0]), "y": f32(o + "_m1", -1.0)})
        elif n.op == "rsqrt":
            s = emit(n, "sqrt", {"x": get(ins[0])}, name=o + "_sqrt")
            emit(n, "real_div", {"x": f32(o + "_one", 1.0), "y": s})
        elif n.op in binary:
            emit(n, binary[n.op], {"x": get(ins[0]), "y": get(ins[1])})
        elif n.op == "gelu":
            mode = "TANH_APPROXIMATION" if a.get("approximate", "none") == "tanh" else "EXACT"
            emit(n, "gelu", {"x": get(ins[0]), "mode": p.const(o + "_mode", np.asarray(mode), "string")})
        elif n.op == "linear":
            x, w, b = ins
            emit(n, "linear", {"x": get(x), "weight": get(w), "bias": get(b) if b is not None else None})
        elif n.op == "matmul":
            emit(n, "matmul", {"x": get(ins[0]), "y": get(ins[1]), "transpose_x": boolc(o + "_tx", False),
                               "transpose_y": boolc(o + "_ty", False)})
        elif n.op == "conv2d":
            x, w, b = ins
            (sh, sw), (ph, pw), (dh, dw) = pair(a.get("stride", 1)), pair(a.get("padding", 0)), pair(a.get("dilation", 1))
            emit(n, "conv", {"x": get(x), "weight": get(w), "bias": get(b) if b is not None else None,
                             "strides": i32(o + "_strides", [sh, sw]),
                             "pad_type": p.const(o + "_pad_type", np.asarray("custom"), "string"),
                             "pad": i32(o + "_pad", [ph, ph, pw, pw]), "dilations": i32(o + "_dil", [dh, dw]),
                             "groups": i32(o + "_groups", int(a.get("groups", 1)))})
        elif n.op == "batch_norm":
            x, g, b, mean, var = ins
            emit(n, "batch_norm", {"x": get(x), "mean": get(mean), "variance": get(var),
                                   "gamma": get(g) if g is not None else None, "beta": get(b) if b is not None else None,
                                   "epsilon": f32(o + "_eps", float(a["eps"]))})
        elif n.op == "layer_norm":
            x, g, b = ins
            k = len(a["normalized_shape"])
            emit(n, "layer_norm", {"x": get(x), "axes": i32(o + "_axes", list(range(-k, 0))),
                                   "gamma": get(g) if g is not None else None, "beta": get(b) if b is not None else None,
                                   "epsilon": f32(o + "_eps", float(a["eps"]))})
        elif n.op == "max_pool2d":
            (kh, kw), (ph, pw) = pair(a["kernel_size"]), pair(a.get("padding", 0))
            sh, sw = pair(a.get("stride") or a["kernel_size"])
            if pair(a.get("dilation", 1)) != (1, 1):
                raise LoweringError(T, n, "dilated max pooling")
            emit(n, "max_pool", {"x": get(ins[0]), "kernel_sizes": i32(o + "_k", [kh, kw]),
                                 "strides": i32(o + "_s", [sh, sw]),
                                 "pad_type": p.const(o + "_pad_type", np.asarray("custom"), "string"),
                                 "pad": i32(o + "_pad", [ph, ph, pw, pw]),
                                 "ceil_mode": boolc(o + "_ceil", bool(a.get("ceil_mode", False)))})
        elif n.op == "adaptive_avg_pool2d":
            if tuple(int(d) for d in n.output.type.shape[-2:]) != (1, 1):
                raise LoweringError(T, n, "adaptive average pooling to more than one cell")
            emit(n, "reduce_mean", {"x": get(ins[0]), "axes": i32(o + "_axes", [-2, -1]),
                                    "keep_dims": boolc(o + "_keep", True)})
        elif n.op == "mean":
            rank = len(static_shape(T, ins[0], n))
            dims = a.get("dim")
            dims = list(range(rank)) if dims is None else ([dims] if isinstance(dims, int) else list(dims))
            emit(n, "reduce_mean", {"x": get(ins[0]), "axes": i32(o + "_axes", dims),
                                    "keep_dims": boolc(o + "_keep", bool(a.get("keepdim", False)))})
        elif n.op == "softmax":
            emit(n, "softmax", {"x": get(ins[0]), "axis": i32(o + "_axis", int(a["dim"]))})
        elif n.op == "reshape":
            emit(n, "reshape", {"x": get(ins[0]), "shape": i32(o + "_shape", list(static_shape(T, n.output, n)))})
        elif n.op == "permute":
            emit(n, "transpose", {"x": get(ins[0]), "perm": i32(o + "_perm", [int(d) for d in a["dims"]])})
        elif n.op in ("select", "slice"):
            xs = static_shape(T, ins[0], n)
            d = int(a["dim"]) % len(xs)
            begin, end, stride = [0] * len(xs), list(xs), [1] * len(xs)
            squeeze = [False] * len(xs)
            if n.op == "select":
                i = int(a["index"]) % xs[d]
                begin[d], end[d], squeeze[d] = i, i + 1, True
            else:
                st = a.get("step", 1) or 1
                if st < 1:
                    raise LoweringError(T, n, "a slice with a negative step")
                begin[d], end[d], stride[d] = slice(a.get("start"), a.get("end"), st).indices(xs[d])
            emit(n, "slice_by_index", {"x": get(ins[0]), "begin": i32(o + "_b", begin), "end": i32(o + "_e", end),
                                       "stride": i32(o + "_st", stride), "squeeze_mask": boolc(o + "_sq", squeeze)})
        elif n.op == "where":
            emit(n, "select", {"cond": get(ins[0]), "a": get(ins[1]), "b": get(ins[2])})
        elif n.op == "index":
            if len(ins) != 2 or ins[1] is None:
                raise LoweringError(T, n, "advanced indexing other than one index tensor on dim 0")
            emit(n, "gather", {"x": get(ins[0]), "indices": get(ins[1]), "axis": i32(o + "_axis", 0)})
        elif n.op == "attention":
            q, k, v, mask = ins
            qs, ks = static_shape(T, q, n), static_shape(T, k, n)
            scale = a.get("scale")
            scale = 1.0 / math.sqrt(qs[-1]) if scale is None else float(scale)
            sshape = qs[:-1] + (ks[-2],)
            s = emit(n, "matmul", {"x": get(q), "y": get(k), "transpose_x": boolc(o + "_tx", False),
                                   "transpose_y": boolc(o + "_ty", True)}, name=o + "_scores", shape=sshape)
            s = emit(n, "mul", {"x": s, "y": f32(o + "_scale", scale)}, name=o + "_scaled", shape=sshape)
            keep = np.tril(np.ones((qs[-2], ks[-2]), np.bool_)) if a.get("is_causal") else None
            if mask is not None:
                if mask.type.dtype != torch.bool or not low.is_const(mask):
                    raise LoweringError(T, n, "attention with a computed or additive mask")
                mk = low.consts[mask.name].numpy()
                keep = mk if keep is None else keep & mk
            if keep is not None:
                s = emit(n, "select", {"cond": boolc(o + "_keep", keep), "a": s,
                                       "b": f32(o + "_ninf", -np.inf)}, name=o + "_masked", shape=sshape)
            pr = emit(n, "softmax", {"x": s, "axis": i32(o + "_axis", -1)}, name=o + "_probs", shape=sshape)
            emit(n, "matmul", {"x": pr, "y": get(v), "transpose_x": boolc(o + "_tx2", False),
                               "transpose_y": boolc(o + "_ty2", False)})
        else:
            raise LoweringError(T, n, f"op {n.op} has no Core ML lowering")

    outs = []
    for v in low.outputs:
        if v.name not in env or v.name in low.consts:
            raise LoweringError(T, None, f"output %{v.name} does not depend on the inputs")
        outs.append(env[v.name])
    block = b"".join(f_str(2, p.mil(x)) for x in outs) + b"".join(f_len(3, op) for op in p.ops)
    function = b"".join(f_len(1, i) for i in fn_inputs) + f_str(2, OPSET) + f_map(3, [(OPSET, block)])
    program = f_int(1, 1) + f_map(2, [("main", function)])

    def feature(v: Value) -> bytes:
        dt = "fp32" if v.type.dtype.is_floating_point else "int32"
        arr = b"".join(f_int(1, int(d)) for d in static_shape(T, v)) + f_int(2, _ARRAY[dt])
        return f_str(1, p.mil(v.name)) + f_len(3, f_len(5, arr))

    desc = b"".join(f_len(1, feature(v)) for v in low.inputs) + b"".join(f_len(10, feature(v)) for v in low.outputs)
    model = f_int(1, SPEC_VERSION) + f_len(2, desc) + f_len(502, program)
    ident = str(uuid.UUID(hashlib.sha256(model).hexdigest()[:32])).upper()
    manifest = {"fileFormatVersion": "1.0.0", "rootModelIdentifier": ident,
                "itemInfoEntries": {ident: {"author": "com.apple.CoreML", "description": "CoreML Model Specification",
                                            "name": "model.mlmodel", "path": "com.apple.CoreML/model.mlmodel"}}}
    return {"Manifest.json": json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"), MODEL_PATH: model}

# ----------------------------------------------------------------------------- decode and evaluate


def _decode_value(buf: bytes) -> np.ndarray:
    v = parse(buf)
    tt = parse(parse(v[2][0])[1][0])
    dt = tt[1][0]
    shape = tuple(parse(parse(d)[1][0])[1][0] for d in tt.get(3, []))
    tv = parse(parse(v[3][0])[1][0]) if 3 in v else {}
    if 1 in tv:
        arr = np.frombuffer(parse(tv[1][0]).get(1, [b""])[0], "<f4").astype(np.float64)
    elif 2 in tv:
        raw, vals, i = parse(tv[2][0]).get(1, [b""])[0], [], 0
        while i < len(raw):
            x, i = _read_varint(raw, i)
            vals.append(_signed(x))
        arr = np.asarray(vals, np.int64)
    elif 3 in tv:
        raw = parse(tv[3][0]).get(1, [b""])[0]
        arr = np.frombuffer(raw, np.uint8).astype(np.bool_)
    elif 4 in tv:
        return np.asarray(parse(tv[4][0])[1][0].decode("utf-8") if dt == _MIL["string"] else "")
    else:
        arr = np.zeros(0)
    return arr.reshape(shape)


def decode(model_bytes: bytes) -> Dict[str, Any]:
    """The written model read back from its own bytes: spec version, the feature descriptions and
    the main function's ops as (type, {param: [names]}, output name, attribute values)."""
    m = parse(model_bytes)
    desc = parse(m[2][0])

    def feat(b):
        f = parse(b)
        arr = parse(parse(f[3][0])[5][0])
        return {"name": f[1][0].decode(), "shape": [int(s) for s in arr.get(1, [])], "dtype": int(arr[2][0])}

    prog = parse(m[502][0])
    fn = None
    for e in prog[2]:
        kv = parse(e)
        if kv[1][0].decode() == "main":
            fn = parse(kv[2][0])
    opset = fn[2][0].decode()
    block = None
    for e in fn[3]:
        kv = parse(e)
        if kv[1][0].decode() == opset:
            block = parse(kv[2][0])
    ops = []
    for ob in block.get(3, []):
        o = parse(ob)
        inputs = {}
        for e in o.get(2, []):
            kv = parse(e)
            inputs[kv[1][0].decode()] = [parse(b)[1][0].decode() for b in parse(kv[2][0]).get(1, [])]
        attrs = {}
        for e in o.get(5, []):
            kv = parse(e)
            attrs[kv[1][0].decode()] = _decode_value(kv[2][0])
        out = parse(o[3][0])[1][0].decode()
        ops.append((o[1][0].decode(), inputs, out, attrs))
    return {"spec_version": int(m[1][0]), "inputs": [feat(b) for b in desc.get(1, [])],
            "outputs": [feat(b) for b in desc.get(10, [])], "opset": opset,
            "block_outputs": [b.decode() for b in block.get(2, [])], "ops": ops}


def evaluate64(decoded: Dict[str, Any], feeds: List[np.ndarray]) -> List[np.ndarray]:
    """The decoded ML Program's math in float64 (MIL op semantics, CoreML5 opset)."""
    import torch.nn.functional as F

    env: Dict[str, Any] = {}
    for f, x in zip(decoded["inputs"], feeds):
        x = np.asarray(x)
        env[f["name"]] = torch.from_numpy(x.astype(np.float64) if x.dtype.kind == "f" else x.astype(np.int64))

    def t(name):
        return env[name]

    def lst(name):
        return [int(v) for v in np.asarray(env[name]).reshape(-1)]

    for typ, ins, out, attrs in decoded["ops"]:
        g = {k: (v[0] if len(v) == 1 else v) for k, v in ins.items()}
        if typ == "const":
            val = attrs["val"]
            env[out] = val if val.dtype.kind in "US" else torch.from_numpy(np.array(val))
            continue
        x = env.get(g.get("x")) if isinstance(g.get("x"), str) else None
        if typ in ("relu", "sigmoid", "tanh", "exp", "sqrt", "abs", "sin", "cos", "erf", "logical_not"):
            r = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh, "exp": torch.exp,
                 "sqrt": torch.sqrt, "abs": torch.abs, "sin": torch.sin, "cos": torch.cos, "erf": torch.erf,
                 "logical_not": torch.logical_not}[typ](x)
        elif typ in ("add", "sub", "mul", "real_div", "maximum", "minimum", "equal", "less", "greater",
                     "less_equal", "greater_equal"):
            y = t(g["y"])
            r = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "real_div": torch.div,
                 "maximum": torch.maximum, "minimum": torch.minimum, "equal": torch.eq, "less": torch.lt,
                 "greater": torch.gt, "less_equal": torch.le, "greater_equal": torch.ge}[typ](x, y)
        elif typ == "gelu":
            mode = str(env[g["mode"]]) if "mode" in g else "EXACT"
            r = F.gelu(x, approximate="tanh" if mode == "TANH_APPROXIMATION" else "none")
        elif typ == "linear":
            r = F.linear(x, t(g["weight"]), t(g["bias"]) if "bias" in g else None)
        elif typ == "matmul":
            a_, b_ = x, t(g["y"])
            if bool(t(g["transpose_x"])):
                a_ = a_.transpose(-2, -1)
            if bool(t(g["transpose_y"])):
                b_ = b_.transpose(-2, -1)
            r = torch.matmul(a_, b_)
        elif typ == "conv":
            pad = lst(g["pad"])
            xp = F.pad(x, (pad[2], pad[3], pad[0], pad[1]))
            r = F.conv2d(xp, t(g["weight"]), t(g["bias"]) if "bias" in g else None, stride=lst(g["strides"]),
                         dilation=lst(g["dilations"]), groups=lst(g["groups"])[0])
        elif typ == "batch_norm":
            c = (1, -1) + (1,) * (x.dim() - 2)
            r = (x - t(g["mean"]).reshape(c)) / torch.sqrt(t(g["variance"]).reshape(c) + float(t(g["epsilon"])))
            if "gamma" in g:
                r = r * t(g["gamma"]).reshape(c)
            if "beta" in g:
                r = r + t(g["beta"]).reshape(c)
        elif typ == "layer_norm":
            axes = [a_ % x.dim() for a_ in lst(g["axes"])]
            mu = x.mean(dim=axes, keepdim=True)
            var = ((x - mu) ** 2).mean(dim=axes, keepdim=True)
            r = (x - mu) / torch.sqrt(var + float(t(g["epsilon"])))
            if "gamma" in g:
                r = r * t(g["gamma"])
            if "beta" in g:
                r = r + t(g["beta"])
        elif typ == "max_pool":
            pad = lst(g["pad"])
            xp = F.pad(x, (pad[2], pad[3], pad[0], pad[1]), value=-math.inf)
            r = F.max_pool2d(xp, lst(g["kernel_sizes"]), lst(g["strides"]), ceil_mode=bool(t(g["ceil_mode"])))
        elif typ == "reduce_mean":
            r = x.mean(dim=lst(g["axes"]), keepdim=bool(t(g["keep_dims"])))
        elif typ == "softmax":
            r = torch.softmax(x, dim=lst(g["axis"])[0])
        elif typ == "reshape":
            r = x.reshape(lst(g["shape"]))
        elif typ == "transpose":
            r = x.permute(*lst(g["perm"]))
        elif typ == "slice_by_index":
            b, e, s = lst(g["begin"]), lst(g["end"]), lst(g["stride"])
            sq = [bool(v) for v in np.asarray(env[g["squeeze_mask"]]).reshape(-1)]
            r = x[tuple(slice(b[i], e[i], s[i]) for i in range(x.dim()))]
            r = r.reshape([d for i, d in enumerate(r.shape) if not sq[i]])
        elif typ == "select":
            r = torch.where(t(g["cond"]).bool(), t(g["a"]), t(g["b"]))
        elif typ == "gather":
            r = torch.index_select(x, lst(g["axis"])[0], t(g["indices"]).reshape(-1).long()).reshape(
                tuple(t(g["indices"]).shape) + tuple(x.shape[1:]))
        else:
            raise ValueError(f"no float64 rule for MIL op {typ}")
        env[out] = r.double() if r.is_floating_point() else r
    return [env[nm].numpy() for nm in decoded["block_outputs"]]
