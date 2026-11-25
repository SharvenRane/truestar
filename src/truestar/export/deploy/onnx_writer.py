"""Truestar IR to ONNX (opset 17), written node by node from the verified program.

The batch dim stays symbolic when the program's is (a reshape target with one symbolic entry is
written with -1). Ops without an exact ONNX form are refused (TS1412), never approximated.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from ...ir.core import Node, Value, is_const
from .lower import Lowered, LoweringError, pair, symbolic_entries

T = "onnx"
OPSET = 17

_UNARY = {"relu": "Relu", "sigmoid": "Sigmoid", "tanh": "Tanh", "exp": "Exp", "neg": "Neg", "abs": "Abs",
          "sqrt": "Sqrt", "sin": "Sin", "cos": "Cos", "log": "Log", "erf": "Erf", "reciprocal": "Reciprocal",
          "logical_not": "Not"}
_BINARY = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div", "maximum": "Max", "minimum": "Min",
           "eq": "Equal", "lt": "Less", "gt": "Greater", "le": "LessOrEqual", "ge": "GreaterOrEqual",
           "logical_and": "And", "logical_or": "Or"}


def _onnx():
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        from ...deps import MissingDependencyError

        raise MissingDependencyError("onnx", "Writing a .onnx file (truestar export --to onnx)",
                                     extra="onnx") from None
    return onnx, TensorProto, helper, numpy_helper


def _elem(dtype: torch.dtype) -> int:
    _, TP, _, _ = _onnx()
    return {torch.float32: TP.FLOAT, torch.float64: TP.DOUBLE, torch.float16: TP.FLOAT16,
            torch.int64: TP.INT64, torch.int32: TP.INT32, torch.bool: TP.BOOL, torch.int8: TP.INT8,
            torch.uint8: TP.UINT8}[dtype]


def _dims(shape) -> List:
    return [int(d) if is_const(d) else str(d).replace(" ", "") for d in shape]


class _G:
    def __init__(self):
        self.onnx, self.TP, self.H, self.NH = _onnx()
        self.nodes, self.inits = [], []
        self.n = 0

    def fresh(self, base: str) -> str:
        self.n += 1
        return f"{base}__{self.n}"

    def const(self, base: str, a) -> str:
        name = self.fresh(base)
        self.inits.append(self.NH.from_array(np.ascontiguousarray(a), name))
        return name

    def node(self, op: str, ins: List[str], out: Optional[str] = None, **attrs) -> str:
        out = out or self.fresh(op.lower())
        self.nodes.append(self.H.make_node(op, ins, [out], name=out + "_" + op, **attrs))
        return out


def write(low: Lowered) -> bytes:
    g = _G()
    H, TP = g.H, g.TP
    names: Dict[str, str] = {v.name: v.name for v in low.inputs}

    def nm(v: Optional[Value]) -> str:
        if v is None:
            return ""
        if v.name not in names:  # constants become initializers when first read
            g.inits.append(g.NH.from_array(np.ascontiguousarray(low.consts[v.name].numpy()), v.name))
            names[v.name] = v.name
        return names[v.name]

    def cast_like(name: str, v: Value, want: torch.dtype) -> str:
        if v.type.dtype == want:
            return name
        if v.name in low.consts:
            return g.const(v.name + "_as", low.consts[v.name].to(want).numpy())
        return g.node("Cast", [name], to=_elem(want))

    for n in low.nodes:
        out = n.output.name
        a = n.attrs
        ins = n.inputs
        od = n.output.type.dtype
        if n.op in _UNARY:
            g.node(_UNARY[n.op], [nm(ins[0])], out)
        elif n.op == "rsqrt":
            g.node("Reciprocal", [g.node("Sqrt", [nm(ins[0])])], out)
        elif n.op in _BINARY:
            if n.op in ("eq", "lt", "gt", "le", "ge"):
                want = ins[0].type.dtype if ins[0].type.dtype.is_floating_point else ins[1].type.dtype
            else:
                want = od
            g.node(_BINARY[n.op], [cast_like(nm(ins[0]), ins[0], want), cast_like(nm(ins[1]), ins[1], want)], out)
        elif n.op == "gelu":
            x = nm(ins[0])
            half = g.const("half", np.float32(0.5))
            one = g.const("one", np.float32(1.0))
            if a.get("approximate", "none") == "tanh":
                c = g.const("c", np.float32(math.sqrt(2.0 / math.pi)))
                k = g.const("k", np.float32(0.044715))
                x3 = g.node("Mul", [g.node("Mul", [x, x]), x])
                inner = g.node("Mul", [c, g.node("Add", [x, g.node("Mul", [k, x3])])])
                g.node("Mul", [g.node("Mul", [half, x]), g.node("Add", [one, g.node("Tanh", [inner])])], out)
            else:
                r2 = g.const("sqrt2", np.float32(math.sqrt(2.0)))
                e = g.node("Erf", [g.node("Div", [x, r2])])
                g.node("Mul", [g.node("Mul", [half, x]), g.node("Add", [one, e])], out)
        elif n.op == "silu":
            x = nm(ins[0])
            g.node("Mul", [x, g.node("Sigmoid", [x])], out)
        elif n.op == "clamp":
            lo, hi = a.get("min"), a.get("max")
            np_t = np.float32 if od.is_floating_point else np.int64
            g.node("Clip", [nm(ins[0]), g.const("lo", np_t(lo)) if lo is not None else "",
                            g.const("hi", np_t(hi)) if hi is not None else ""], out)
        elif n.op == "linear":
            x, w, b = ins
            if not low.is_const(w):
                wt = g.node("Transpose", [nm(w)], perm=[1, 0])
            else:
                wt = g.const(w.name + "_t", low.consts[w.name].t().contiguous().numpy())
            if x.type.rank == 2:
                g.node("Gemm", [nm(x), wt] + ([nm(b)] if b is not None else []), out)
            elif b is None:
                g.node("MatMul", [nm(x), wt], out)
            else:
                g.node("Add", [g.node("MatMul", [nm(x), wt]), nm(b)], out)
        elif n.op == "matmul":
            g.node("MatMul", [nm(ins[0]), nm(ins[1])], out)
        elif n.op == "conv2d":
            x, w, b = ins
            (sh, sw), (ph, pw), (dh, dw) = pair(a.get("stride", 1)), pair(a.get("padding", 0)), pair(a.get("dilation", 1))
            if isinstance(a.get("padding"), str):
                raise LoweringError(T, n, f"string padding {a['padding']!r}")
            g.node("Conv", [nm(x), nm(w)] + ([nm(b)] if b is not None else []), out, strides=[sh, sw],
                   pads=[ph, pw, ph, pw], dilations=[dh, dw], group=int(a.get("groups", 1)))
        elif n.op == "batch_norm":
            x, w, b, mean, var = ins
            c = int(x.type.shape[1])
            sc = nm(w) if w is not None else g.const("ones", np.ones(c, np.float32))
            bi = nm(b) if b is not None else g.const("zeros", np.zeros(c, np.float32))
            g.node("BatchNormalization", [nm(x), sc, bi, nm(mean), nm(var)], out, epsilon=float(a["eps"]))
        elif n.op == "layer_norm":
            x, w, b = ins
            k = len(a["normalized_shape"])
            norm = [int(d) for d in a["normalized_shape"]]
            sc = nm(w) if w is not None else g.const("ones", np.ones(norm, np.float32))
            g.node("LayerNormalization", [nm(x), sc] + ([nm(b)] if b is not None else []), out, axis=-k,
                   epsilon=float(a["eps"]))
        elif n.op == "max_pool2d":
            (kh, kw), (ph, pw), (dh, dw) = pair(a["kernel_size"]), pair(a.get("padding", 0)), pair(a.get("dilation", 1))
            sh, sw = pair(a.get("stride") or a["kernel_size"])
            g.node("MaxPool", [nm(ins[0])], out, kernel_shape=[kh, kw], strides=[sh, sw], pads=[ph, pw, ph, pw],
                   dilations=[dh, dw], ceil_mode=int(bool(a.get("ceil_mode", False))))
        elif n.op == "adaptive_avg_pool2d":
            oh, ow = (int(d) for d in n.output.type.shape[-2:])
            h, w = ins[0].type.shape[-2:]
            if (oh, ow) == (1, 1):
                g.node("GlobalAveragePool", [nm(ins[0])], out)
            elif is_const(h) and is_const(w) and int(h) % oh == 0 and int(w) % ow == 0:
                kh, kw = int(h) // oh, int(w) // ow
                g.node("AveragePool", [nm(ins[0])], out, kernel_shape=[kh, kw], strides=[kh, kw])
            else:
                raise LoweringError(T, n, "adaptive average pooling with uneven cells")
        elif n.op == "mean" or n.op == "sum":
            dims = a.get("dim")
            dims = list(range(ins[0].type.rank)) if dims is None else ([dims] if isinstance(dims, int) else list(dims))
            keep = int(bool(a.get("keepdim", False)))
            if n.op == "mean":
                g.node("ReduceMean", [nm(ins[0])], out, axes=dims, keepdims=keep)
            else:
                g.node("ReduceSum", [nm(ins[0]), g.const("axes", np.asarray(dims, np.int64))], out, keepdims=keep)
        elif n.op == "softmax":
            g.node("Softmax", [nm(ins[0])], out, axis=int(a["dim"]))
        elif n.op == "log_softmax":
            g.node("LogSoftmax", [nm(ins[0])], out, axis=int(a["dim"]))
        elif n.op == "reshape":
            shape = list(n.output.type.shape)
            sym = symbolic_entries(shape)
            if len(sym) > 1:
                raise LoweringError(T, n, f"reshape to {shape} has {len(sym)} symbolic entries (ONNX Reshape "
                                          "can infer one); export with --static-batch")
            tgt = [-1 if i in sym else int(d) for i, d in enumerate(shape)]
            g.node("Reshape", [nm(ins[0]), g.const("shape", np.asarray(tgt, np.int64))], out)
        elif n.op == "permute":
            g.node("Transpose", [nm(ins[0])], out, perm=[int(d) % ins[0].type.rank for d in a["dims"]])
        elif n.op == "select":
            g.node("Gather", [nm(ins[0]), g.const("index", np.asarray(int(a["index"]), np.int64))], out,
                   axis=int(a["dim"]))
        elif n.op == "slice":
            start, end, step = a.get("start"), a.get("end"), a.get("step", 1)
            if any(not (v is None or isinstance(v, int)) for v in (start, end, step)):
                raise LoweringError(T, n, "a slice with symbolic bounds")
            big = np.iinfo(np.int64).max
            g.node("Slice", [nm(ins[0]), g.const("starts", np.asarray([start or 0], np.int64)),
                             g.const("ends", np.asarray([big if end is None else min(end, big)], np.int64)),
                             g.const("axes", np.asarray([int(a["dim"])], np.int64)),
                             g.const("steps", np.asarray([step], np.int64))], out)
        elif n.op == "cat":
            g.node("Concat", [nm(v) for v in ins], out, axis=int(a["dim"]))
        elif n.op == "expand":
            shape = list(n.output.type.shape)
            if symbolic_entries(shape):
                raise LoweringError(T, n, "expand to a symbolic shape")
            g.node("Expand", [nm(ins[0]), g.const("shape", np.asarray([int(d) for d in shape], np.int64))], out)
        elif n.op == "where":
            c, x, y = ins
            g.node("Where", [nm(c), cast_like(nm(x), x, od), cast_like(nm(y), y, od)], out)
        elif n.op == "cast":
            g.node("Cast", [nm(ins[0])], out, to=_elem(od))
        elif n.op == "index":
            x, idx = ins[0], list(ins[1:])
            if len(idx) != 1 or idx[0] is None:
                raise LoweringError(T, n, "advanced indexing other than one index tensor on dim 0")
            g.node("Gather", [nm(x), nm(idx[0])], out, axis=0)
        elif n.op == "index_select":
            g.node("Gather", [nm(ins[0]), nm(ins[1])], out, axis=int(a["dim"]))
        elif n.op == "attention":
            q, k, v, mask = ins
            if a.get("is_causal") and mask is not None:
                raise LoweringError(T, n, "attention with both is_causal and a mask")
            e = q.type.shape[-1]
            scale = a.get("scale")
            scale = 1.0 / math.sqrt(int(e)) if scale is None else float(scale)
            rank = k.type.rank
            kt = g.node("Transpose", [nm(k)], perm=list(range(rank - 2)) + [rank - 1, rank - 2])
            s = g.node("Mul", [g.node("MatMul", [nm(q), kt]), g.const("scale", np.float32(scale))])
            ninf = g.const("ninf", np.float32(-np.inf)) if a.get("is_causal") or (
                mask is not None and mask.type.dtype == torch.bool) else None
            if a.get("is_causal"):
                L, S = q.type.shape[-2], k.type.shape[-2]
                if not (is_const(L) and is_const(S)):
                    raise LoweringError(T, n, "causal attention with a symbolic length")
                keep = np.tril(np.ones((int(L), int(S)), np.bool_))
                s = g.node("Where", [g.const("causal", keep), s, ninf])
            if mask is not None:
                if mask.type.dtype == torch.bool:
                    s = g.node("Where", [nm(mask), s, ninf])
                else:
                    s = g.node("Add", [s, nm(mask)])
            g.node("MatMul", [g.node("Softmax", [s], axis=-1), nm(v)], out)
        else:
            raise LoweringError(T, n, f"op {n.op} has no ONNX lowering")
        names[out] = out

    def vi(v: Value):
        return H.make_tensor_value_info(v.name, _elem(v.type.dtype), _dims(v.type.shape))

    graph = H.make_graph(g.nodes, low.name or "truestar", [vi(v) for v in low.inputs],
                         [vi(v) for v in low.outputs], g.inits)
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", OPSET)], producer_name="truestar",
                         ir_version=8)
    g.onnx.checker.check_model(model)
    return model.SerializeToString()
