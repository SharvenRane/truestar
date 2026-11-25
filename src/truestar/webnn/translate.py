"""ONNX graph to a WebNN graph description (Truestar's own translation, research item 6.3).

The description is plain JSON the page script (`page.js`) turns into `MLGraphBuilder` calls one
node at a time, so every ONNX node is one WebNN operation and nothing is silently left to another
runtime: a node without a rule here raises `Unsupported` (TS1127) before anything runs.

Shapes come from ONNX shape inference; Reshape, Flatten, Squeeze and Unsqueeze become a WebNN
reshape to the inferred static shape. Only float32 graphs are translated (WebNN has no float64,
and float16 needs Float16Array, which the page does not assume).
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Sequence

import numpy as np


class Unsupported(ValueError):
    """An ONNX node or dtype with no WebNN translation (TS1127)."""


_DTYPES = {np.dtype(np.float32): "float32", np.dtype(np.int32): "int32", np.dtype(np.int64): "int64",
           np.dtype(np.uint8): "uint8", np.dtype(np.int8): "int8"}

UNARY = {"Relu": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Erf": "erf", "Sqrt": "sqrt",
         "Exp": "exp", "Log": "log", "Abs": "abs", "Neg": "neg", "Reciprocal": "reciprocal",
         "Identity": "identity", "HardSwish": "hardSwish", "Floor": "floor", "Ceil": "ceil",
         "Sin": "sin", "Cos": "cos", "Softsign": "softsign", "Softplus": "softplus"}
BINARY = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Pow": "pow", "Max": "max",
          "Min": "min", "MatMul": "matmul"}
RESHAPES = ("Reshape", "Flatten", "Squeeze", "Unsqueeze")
SUPPORTED = sorted(set(UNARY) | set(BINARY) | set(RESHAPES) | {
    "Gemm", "Softmax", "LayerNormalization", "ReduceMean", "Transpose", "Conv", "MaxPool",
    "AveragePool", "GlobalAveragePool", "BatchNormalization", "Concat", "Clip", "HardSigmoid",
    "Constant", "Cast", "LeakyRelu", "Elu"})


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _attrs(node) -> Dict[str, Any]:
    from onnx import helper

    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _dt(a: np.ndarray, what: str) -> str:
    if a.dtype == np.float64:
        a = a.astype(np.float32)
    if a.dtype not in _DTYPES:
        raise Unsupported(f"{what}: dtype {a.dtype} has no WebNN type here")
    return _DTYPES[a.dtype]


def translate(model: bytes, feeds: Dict[str, np.ndarray], outputs: Sequence[str] = ()) -> Dict[str, Any]:
    """{"inputs", "consts", "nodes", "outputs"} for page.js. Raises Unsupported."""
    import onnx
    from onnx import numpy_helper

    proto = onnx.load_from_string(model)
    try:
        inferred = onnx.shape_inference.infer_shapes(proto, strict_mode=False)
    except Exception:
        inferred = proto
    shapes: Dict[str, List[int]] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            shapes[vi.name] = [int(d.dim_value) for d in dims]
    consts: Dict[str, np.ndarray] = {i.name: numpy_helper.to_array(i) for i in proto.graph.initializer}
    desc_consts: Dict[str, Dict[str, Any]] = {}
    inputs = []
    for name, a in feeds.items():
        a = np.asarray(a)
        if a.dtype != np.float32:
            raise Unsupported(f"graph input {name}: dtype {a.dtype}; the WebNN backend translates float32 graphs only")
        inputs.append({"name": name, "dtype": "float32", "shape": list(a.shape)})
        shapes.setdefault(name, list(a.shape))
    nodes = []

    def const(name: str) -> np.ndarray:
        if name not in consts:
            raise Unsupported(f"input {name} must be a constant for this op")
        return consts[name]

    def need(name: str):
        if name in consts and name not in desc_consts:
            a = consts[name]
            if a.dtype == np.float64:
                a = a.astype(np.float32)
            desc_consts[name] = {"name": name, "dtype": _dt(a, name), "shape": list(a.shape), "data": b64(a)}

    for n in proto.graph.node:
        op, at = n.op_type, _attrs(n)
        ins = [i for i in n.input]
        out = n.output[0]
        e: Dict[str, Any] = {"name": n.name, "type": op, "outputs": [out]}
        if op == "Constant":
            v = at.get("value")
            if v is None:
                raise Unsupported(f"Constant {n.name} without a tensor value")
            consts[out] = numpy_helper.to_array(v)
            continue
        if op in UNARY:
            e.update(op=UNARY[op], inputs=ins[:1])
        elif op in BINARY:
            e.update(op=BINARY[op], inputs=ins[:2])
        elif op in RESHAPES:
            if out not in shapes:
                raise Unsupported(f"{op} {n.name}: output shape not static")
            e.update(op="reshape", inputs=ins[:1], shape=shapes[out])
        elif op == "Gemm":
            e.update(op="gemm", inputs=ins[:2], c=ins[2] if len(ins) > 2 and ins[2] else None,
                     alpha=float(at.get("alpha", 1.0)), beta=float(at.get("beta", 1.0)),
                     aTranspose=bool(at.get("transA", 0)), bTranspose=bool(at.get("transB", 0)))
        elif op == "Softmax":
            rank = len(shapes.get(ins[0], [])) or 2
            e.update(op="softmax", inputs=ins[:1], axis=int(at.get("axis", -1)) % rank)
        elif op == "LayerNormalization":
            rank = len(shapes.get(ins[0], []))
            if not rank:
                raise Unsupported(f"LayerNormalization {n.name}: input rank unknown")
            ax = int(at.get("axis", -1)) % rank
            e.update(op="layerNormalization", inputs=ins[:1], scale=ins[1] if len(ins) > 1 else None,
                     bias=ins[2] if len(ins) > 2 and ins[2] else None, axes=list(range(ax, rank)),
                     epsilon=float(at.get("epsilon", 1e-5)))
        elif op == "ReduceMean":
            rank = len(shapes.get(ins[0], []))
            axes = at.get("axes")
            if axes is None and len(ins) > 1 and ins[1]:
                axes = [int(v) for v in const(ins[1]).reshape(-1)]
            axes = list(range(rank)) if axes is None else [int(a) % rank for a in axes]
            e.update(op="reduceMean", inputs=ins[:1], axes=axes, keepDimensions=bool(at.get("keepdims", 1)))
        elif op == "Transpose":
            e.update(op="transpose", inputs=ins[:1], permutation=[int(p) for p in at["perm"]])
        elif op in ("Conv", "MaxPool", "AveragePool"):
            k = len(shapes.get(ins[0], [])) - 2
            if k != 2:
                raise Unsupported(f"{op} {n.name}: only 2D (NCHW) is translated, input rank {k + 2}")
            if at.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
                raise Unsupported(f"{op} {n.name}: auto_pad")
            p = [int(v) for v in at.get("pads", [0, 0, 0, 0])]
            common = {"padding": [p[0], p[2], p[1], p[3]], "strides": [int(v) for v in at.get("strides", [1, 1])],
                      "dilations": [int(v) for v in at.get("dilations", [1, 1])]}
            if op == "Conv":
                e.update(op="conv2d", inputs=ins[:2], bias=ins[2] if len(ins) > 2 and ins[2] else None,
                         groups=int(at.get("group", 1)), **common)
            else:
                if op == "AveragePool" and not at.get("count_include_pad", 0) and any(p):
                    raise Unsupported(f"AveragePool {n.name}: count_include_pad=0 with padding")
                if at.get("ceil_mode", 0):
                    raise Unsupported(f"{op} {n.name}: ceil_mode")
                e.update(op="maxPool2d" if op == "MaxPool" else "averagePool2d", inputs=ins[:1],
                         windowDimensions=[int(v) for v in at["kernel_shape"]], **common)
        elif op == "GlobalAveragePool":
            e.update(op="averagePool2d", inputs=ins[:1])
        elif op == "BatchNormalization":
            e.update(op="batchNormalization", inputs=[ins[0], ins[3], ins[4]], scale=ins[1], bias=ins[2],
                     epsilon=float(at.get("epsilon", 1e-5)), axis=1)
        elif op == "Concat":
            rank = len(shapes.get(out, []))
            e.update(op="concat", inputs=ins, axis=int(at["axis"]) % rank)
        elif op == "Clip":
            lo = float(const(ins[1]).reshape(-1)[0]) if len(ins) > 1 and ins[1] else None
            hi = float(const(ins[2]).reshape(-1)[0]) if len(ins) > 2 and ins[2] else None
            e.update(op="clamp", inputs=ins[:1], minValue=lo, maxValue=hi)
        elif op == "HardSigmoid":
            e.update(op="hardSigmoid", inputs=ins[:1], alpha=float(at.get("alpha", 0.2)), beta=float(at.get("beta", 0.5)))
        elif op == "LeakyRelu":
            e.update(op="leakyRelu", inputs=ins[:1], alpha=float(at.get("alpha", 0.01)))
        elif op == "Elu":
            e.update(op="elu", inputs=ins[:1], alpha=float(at.get("alpha", 1.0)))
        elif op == "Cast":
            to = {1: "float32", 6: "int32", 7: "int64"}.get(int(at["to"]))
            if to is None:
                raise Unsupported(f"Cast {n.name} to ONNX type {at['to']}")
            e.update(op="cast", inputs=ins[:1], dataType=to)
        else:
            raise Unsupported(f"ONNX op {op} (node {n.name}) has no WebNN translation in Truestar "
                              f"(translated: {', '.join(SUPPORTED)})")
        for i in e.get("inputs", []):
            if i:
                need(i)
        for k in ("c", "scale", "bias"):
            if e.get(k):
                need(e[k])
        nodes.append(e)
    outs = list(outputs) if outputs else [o.name for o in proto.graph.output]
    for o in outs:
        need(o)
    return {"inputs": inputs, "consts": list(desc_consts.values()), "nodes": nodes, "outputs": outs,
            "shapes": {o: shapes.get(o) for o in outs}}
