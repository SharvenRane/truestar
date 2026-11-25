"""Truestar IR to LiteRT (.tflite), written from the verified program with static shapes.

Convolutions in LiteRT are NHWC, the IR is NCHW. The writer keeps a layout per value: a conv,
pool or batch norm output is stored NHWC, elementwise ops keep their operand's layout, and a value
is transposed back to NCHW only where an op needs the logical layout (a reshape, a program output).
Padding that LiteRT's SAME rule does not reproduce exactly is written as an explicit PADV2 (with
-inf for max pooling) followed by a VALID window. Batch norm is written as a per channel multiply
and add whose constants are computed in float64 and rounded once.

Ops without an exact LiteRT form are refused (TS1412).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ...conform.tflite_build import TfliteBuildError, _Writer
from ...ir.core import Node, Value
from .lower import Lowered, LoweringError, pair, static_shape

T = "litert"
NHWC, NCHW = "nhwc", "nchw"
TO_NHWC, TO_NCHW = (0, 2, 3, 1), (0, 3, 1, 2)

_UNARY = {"relu": "RELU", "sigmoid": "LOGISTIC", "tanh": "TANH", "exp": "EXP", "neg": "NEG", "abs": "ABS",
          "sqrt": "SQRT", "rsqrt": "RSQRT", "sin": "SIN", "cos": "COS", "log": "LOG",
          "logical_not": "LOGICAL_NOT"}
_BINARY = {"add": ("ADD", "AddOptions"), "sub": ("SUB", "SubOptions"), "mul": ("MUL", "MulOptions"),
           "div": ("DIV", "DivOptions"), "maximum": ("MAXIMUM", None), "minimum": ("MINIMUM", None)}


class _W(_Writer):
    def tensor(self, name, shape, dtype=np.float32, data=None):
        S = self.S
        kinds = {np.dtype(np.float32): S.TensorType.FLOAT32, np.dtype(np.int32): S.TensorType.INT32,
                 np.dtype(np.int64): S.TensorType.INT64, np.dtype(np.bool_): S.TensorType.BOOL}
        t = S.TensorT()
        t.name = name
        t.shape = [int(s) for s in shape]
        t.type = kinds[np.dtype(dtype)]
        t.buffer = 0 if data is None else self.buffer(np.asarray(data, dtype=dtype))
        self.tensors.append(t)
        return len(self.tensors) - 1


_NP = {torch.float32: np.float32, torch.int64: np.int64, torch.int32: np.int32, torch.bool: np.bool_}


def write(low: Lowered) -> bytes:
    try:
        w = _W()
    except TfliteBuildError as e:
        from ...deps import MissingDependencyError

        raise MissingDependencyError("ai_edge_litert", "Writing a .tflite file (truestar export --to litert)",
                                     extra="litert", package="ai-edge-litert") from e
    S, B, O = w.S, w.S.BuiltinOperator, w.S.BuiltinOptions
    env: Dict[str, Tuple[int, str]] = {}  # value name -> (tensor index, layout)
    shapes: Dict[str, Tuple[int, ...]] = {}

    def np_dtype(v: Value):
        if v.type.dtype not in _NP:
            raise LoweringError(T, None, f"%{v.name} has dtype {v.type.dtype}, which this writer does not emit")
        return _NP[v.type.dtype]

    for v in low.inputs:
        shapes[v.name] = static_shape(T, v)
        env[v.name] = (w.tensor(v.name, shapes[v.name], np_dtype(v)), NCHW)

    def const_t(v: Value, layout=NCHW) -> int:
        a = low.consts[v.name].numpy()
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        if layout == NHWC:
            a = a.transpose(TO_NHWC)
        return w.tensor(v.name, a.shape, a.dtype.type, a)

    def i32(name, vals) -> int:
        a = np.asarray(vals, np.int32)
        return w.tensor(name, a.shape, np.int32, a)

    def f32(name, val) -> int:
        a = np.asarray(val, np.float32)
        return w.tensor(name, a.shape, np.float32, a)

    def get(v: Value, want: Optional[str] = None) -> int:
        """Tensor index of v in layout `want` (None: as stored)."""
        if v.name in low.consts:
            return const_t(v, want or NCHW)
        idx, lay = env[v.name]
        if want is None or want == lay or len(shapes[v.name]) != 4:
            return idx
        if v.name + "@" + want in env:
            return env[v.name + "@" + want][0]
        perm = TO_NHWC if want == NHWC else TO_NCHW
        shp = shapes[v.name]
        tshape = tuple(shp[p] for p in TO_NHWC) if want == NHWC else shp
        out = w.tensor(f"{v.name}/{want}", tshape, np_dtype(v))
        w.op(B.TRANSPOSE, [idx, i32(f"{v.name}/{want}.perm", perm)], [out])
        env[v.name + "@" + want] = (out, want)
        return out

    def layout(v: Value) -> str:
        return NCHW if v.name in low.consts else env[v.name][1]

    def new(n: Node, lay: str) -> int:
        shp = static_shape(T, n.output, n)
        shapes[n.output.name] = shp
        stored = tuple(shp[p] for p in TO_NHWC) if lay == NHWC else shp
        idx = w.tensor(n.output.name, stored, np_dtype(n.output))
        env[n.output.name] = (idx, lay)
        return idx

    def pad_for(n, x_idx, hw, k, s, d, p, value) -> Tuple[int, int]:
        """(input tensor, padding enum): SAME when LiteRT's split equals the IR's padding, else an
        explicit PADV2 and VALID."""
        (H, W), (kh, kw), (sh, sw), (dh, dw), (ph, pw) = hw, k, s, d, p
        if ph == 0 and pw == 0:
            return x_idx, S.Padding.VALID
        same = True
        for size, kk, ss, dd, pp in ((H, kh, sh, dh, ph), (W, kw, sw, dw, pw)):
            out = -(-size // ss)
            total = max((out - 1) * ss + (kk - 1) * dd + 1 - size, 0)
            torch_out = (size + 2 * pp - dd * (kk - 1) - 1) // ss + 1
            if out != torch_out or total // 2 != pp or total - total // 2 != pp:
                same = False
        if same:
            return x_idx, S.Padding.SAME
        n_, c = shapes[n.inputs[0].name][0], shapes[n.inputs[0].name][1]
        padded = w.tensor(f"{n.output.name}/pad", (n_, H + 2 * ph, W + 2 * pw, c))
        pads = i32(f"{n.output.name}/pads", [[0, 0], [ph, ph], [pw, pw], [0, 0]])
        if value == 0.0:  # PAD: XNNPACK takes it, PADV2 it leaves on the builtin kernel
            w.op(B.PAD, [x_idx, pads], [padded], O.PadOptions, S.PadOptionsT())
        else:
            w.op(B.PADV2, [x_idx, pads, f32(f"{n.output.name}/padval", value)], [padded], O.PadV2Options,
                 S.PadV2OptionsT())
        return padded, S.Padding.VALID

    for n in low.nodes:
        a, ins, od = n.attrs, n.inputs, n.output.type.dtype
        if n.op in _UNARY:
            lay = layout(ins[0])
            w.op(getattr(B, _UNARY[n.op]), [get(ins[0])], [new(n, lay)])
        elif n.op == "gelu":
            lay = layout(ins[0])
            opt = S.GeluOptionsT()
            opt.approximate = a.get("approximate", "none") == "tanh"
            w.op(B.GELU, [get(ins[0])], [new(n, lay)], O.GeluOptions, opt)
        elif n.op in _BINARY:
            code, oname = _BINARY[n.op]
            x, y = ins
            r4 = [v for v in (x, y) if len(static_shape(T, v, n)) == 4 and v.name not in low.consts]
            lay = NHWC if r4 and all(layout(v) == NHWC for v in r4) else NCHW
            ops = []
            for v in (x, y):
                vs = static_shape(T, v, n)
                if v.name in low.consts:
                    c = low.consts[v.name].to(od).numpy()
                    if lay == NHWC and c.ndim == 4:
                        c = c.transpose(TO_NHWC)
                    elif lay == NHWC and c.ndim > 0 and c.size > 1:
                        raise LoweringError(T, n, "a broadcast constant against an NHWC value")
                    ops.append(w.tensor(f"{n.output.name}/{v.name}", c.shape, c.dtype.type, c))
                else:
                    ops.append(get(v, lay if len(vs) == 4 else None))
            if oname:
                w.op(getattr(B, code), ops, [new(n, lay)], getattr(O, oname), getattr(S, oname + "T")())
            else:
                w.op(getattr(B, code), ops, [new(n, lay)])
        elif n.op == "linear":
            x, wt, b = ins
            if not low.is_const(wt):
                raise LoweringError(T, n, "linear with a computed weight")
            opt = S.FullyConnectedOptionsT()
            opt.keepNumDims = len(static_shape(T, x, n)) > 2
            w.op(B.FULLY_CONNECTED, [get(x, NCHW), const_t(wt), const_t(b) if b is not None else -1],
                 [new(n, NCHW)], O.FullyConnectedOptions, opt)
        elif n.op == "conv2d":
            x, wt, b = ins
            if int(a.get("groups", 1)) != 1:
                raise LoweringError(T, n, "grouped convolution (depthwise is not written yet)")
            if not low.is_const(wt):
                raise LoweringError(T, n, "convolution with a computed weight")
            xs = static_shape(T, x, n)
            wnp = low.consts[wt.name].numpy().astype(np.float32).transpose(0, 2, 3, 1)  # OHWI
            bias = (low.consts[b.name].numpy().astype(np.float32) if b is not None
                    else np.zeros(wnp.shape[0], np.float32))
            (sh, sw), (dh, dw), (ph, pw) = pair(a.get("stride", 1)), pair(a.get("dilation", 1)), pair(a.get("padding", 0))
            src, padding = pad_for(n, get(x, NHWC), xs[2:], wnp.shape[1:3], (sh, sw), (dh, dw), (ph, pw), 0.0)
            opt = S.Conv2DOptionsT()
            opt.padding, opt.strideH, opt.strideW = padding, sh, sw
            opt.dilationHFactor, opt.dilationWFactor = dh, dw
            w.op(B.CONV_2D, [src, w.tensor(wt.name, wnp.shape, np.float32, wnp),
                             w.tensor(f"{n.output.name}/bias", bias.shape, np.float32, bias)],
                 [new(n, NHWC)], O.Conv2DOptions, opt)
        elif n.op == "batch_norm":
            x, g_, b_, mean, var = ins
            if not all(v is None or low.is_const(v) for v in (g_, b_, mean, var)):
                raise LoweringError(T, n, "batch norm with computed statistics")
            m = low.consts[mean.name].double()
            inv = 1.0 / torch.sqrt(low.consts[var.name].double() + float(a["eps"]))
            scale = inv * (low.consts[g_.name].double() if g_ is not None else 1.0)
            shift = (low.consts[b_.name].double() if b_ is not None else 0.0) - m * scale
            xs = static_shape(T, x, n)
            lay = NHWC if len(xs) == 4 else NCHW
            if len(xs) != 4 and len(xs) != 2:
                raise LoweringError(T, n, f"batch norm on a rank {len(xs)} tensor")
            sc, sf = scale.float().numpy(), shift.float().numpy()
            mid = w.tensor(f"{n.output.name}/scaled", tuple(xs[p] for p in TO_NHWC) if lay == NHWC else xs)
            w.op(B.MUL, [get(x, lay if len(xs) == 4 else None), f32(f"{n.output.name}/scale", sc)], [mid],
                 O.MulOptions, S.MulOptionsT())
            w.op(B.ADD, [mid, f32(f"{n.output.name}/shift", sf)], [new(n, lay)], O.AddOptions, S.AddOptionsT())
        elif n.op == "max_pool2d":
            x = ins[0]
            xs = static_shape(T, x, n)
            k, (sh, sw) = pair(a["kernel_size"]), pair(a.get("stride") or a["kernel_size"])
            if pair(a.get("dilation", 1)) != (1, 1):
                raise LoweringError(T, n, "dilated max pooling")
            ph, pw = pair(a.get("padding", 0))
            os_ = static_shape(T, n.output, n)
            if a.get("ceil_mode") and any(o != (s + 2 * p - kk) // st + 1 for o, s, p, kk, st in
                                          zip(os_[2:], xs[2:], (ph, pw), k, (sh, sw))):
                raise LoweringError(T, n, "ceil mode pooling that adds a partial window")
            src, padding = pad_for(n, get(x, NHWC), xs[2:], k, (sh, sw), (1, 1), (ph, pw), -np.inf)
            opt = S.Pool2DOptionsT()
            opt.padding, opt.strideH, opt.strideW, opt.filterHeight, opt.filterWidth = padding, sh, sw, k[0], k[1]
            w.op(B.MAX_POOL_2D, [src], [new(n, NHWC)], O.Pool2DOptions, opt)
        elif n.op == "adaptive_avg_pool2d":
            if tuple(int(d) for d in n.output.type.shape[-2:]) != (1, 1):
                raise LoweringError(T, n, "adaptive average pooling to more than one cell")
            opt = S.ReducerOptionsT()
            opt.keepDims = True
            w.op(B.MEAN, [get(ins[0], NHWC), i32(f"{n.output.name}/axes", [1, 2])], [new(n, NHWC)],
                 O.ReducerOptions, opt)
        elif n.op == "mean":
            dims = a.get("dim")
            rank = len(static_shape(T, ins[0], n))
            dims = list(range(rank)) if dims is None else ([dims] if isinstance(dims, int) else list(dims))
            opt = S.ReducerOptionsT()
            opt.keepDims = bool(a.get("keepdim", False))
            w.op(B.MEAN, [get(ins[0], NCHW), i32(f"{n.output.name}/axes", [d % rank for d in dims])],
                 [new(n, NCHW)], O.ReducerOptions, opt)
        elif n.op == "softmax":
            rank = len(static_shape(T, ins[0], n))
            if int(a["dim"]) % rank != rank - 1:
                raise LoweringError(T, n, "softmax over a dim other than the last")
            opt = S.SoftmaxOptionsT()
            opt.beta = 1.0
            w.op(B.SOFTMAX, [get(ins[0], NCHW)], [new(n, NCHW)], O.SoftmaxOptions, opt)
        elif n.op == "layer_norm":
            x, g_, b_ = ins
            if len(a["normalized_shape"]) != 1:
                raise LoweringError(T, n, "layer norm over more than one dim")
            xs = static_shape(T, x, n)
            red = xs[:-1] + (1,)
            base = n.output.name
            xi = get(x, NCHW)
            axis = i32(base + "/axis", [len(xs) - 1])
            ro = S.ReducerOptionsT()
            ro.keepDims = True
            mu = w.tensor(base + "/mean", red)
            w.op(B.MEAN, [xi, axis], [mu], O.ReducerOptions, ro)
            sq = w.tensor(base + "/sqdiff", xs)
            w.op(B.SQUARED_DIFFERENCE, [xi, mu], [sq])
            var = w.tensor(base + "/var", red)
            w.op(B.MEAN, [sq, axis], [var], O.ReducerOptions, ro)
            ve = w.tensor(base + "/var_eps", red)
            w.op(B.ADD, [var, f32(base + "/eps", float(a["eps"]))], [ve], O.AddOptions, S.AddOptionsT())
            inv = w.tensor(base + "/inv", red)
            w.op(B.RSQRT, [ve], [inv])
            cen = w.tensor(base + "/centered", xs)
            w.op(B.SUB, [xi, mu], [cen], O.SubOptions, S.SubOptionsT())
            last = new(n, NCHW) if g_ is None and b_ is None else w.tensor(base + "/normed", xs)
            w.op(B.MUL, [cen, inv], [last], O.MulOptions, S.MulOptionsT())
            if g_ is not None:
                gout = new(n, NCHW) if b_ is None else w.tensor(base + "/scaled", xs)
                w.op(B.MUL, [last, const_t(g_)], [gout], O.MulOptions, S.MulOptionsT())
                last = gout
            if b_ is not None:
                w.op(B.ADD, [last, const_t(b_)], [new(n, NCHW)], O.AddOptions, S.AddOptionsT())
        elif n.op == "reshape":
            shp = static_shape(T, n.output, n)
            w.op(B.RESHAPE, [get(ins[0], NCHW), i32(n.output.name + "/shape", shp)], [new(n, NCHW)])
        elif n.op == "permute":
            rank = len(static_shape(T, ins[0], n))
            w.op(B.TRANSPOSE, [get(ins[0], NCHW), i32(n.output.name + "/perm", [d % rank for d in a["dims"]])],
                 [new(n, NCHW)])
        elif n.op in ("select", "slice"):
            xs = static_shape(T, ins[0], n)
            d = int(a["dim"]) % len(xs)
            begin, end, strides = [0] * len(xs), list(xs), [1] * len(xs)
            opt = S.StridedSliceOptionsT()
            if n.op == "select":
                i = int(a["index"]) % xs[d]
                begin[d], end[d] = i, i + 1
                opt.shrinkAxisMask = 1 << d
            else:
                st = a.get("step", 1) or 1
                if st < 1:
                    raise LoweringError(T, n, "a slice with a negative step")
                s0, e0, _ = slice(a.get("start"), a.get("end"), st).indices(xs[d])
                begin[d], end[d], strides[d] = s0, e0, st
            base = n.output.name
            w.op(B.STRIDED_SLICE, [get(ins[0], NCHW), i32(base + "/begin", begin), i32(base + "/end", end),
                                   i32(base + "/strides", strides)], [new(n, NCHW)], O.StridedSliceOptions, opt)
        elif n.op == "matmul":
            opt = S.BatchMatMulOptionsT()
            w.op(B.BATCH_MATMUL, [get(ins[0], NCHW), get(ins[1], NCHW)], [new(n, NCHW)], O.BatchMatMulOptions, opt)
        elif n.op == "where":
            c, x, y = ins
            ops = []
            for v in (c, x, y):
                if v.name in low.consts:
                    arr = low.consts[v.name]
                    arr = (arr.to(od) if v is not c else arr).numpy()
                    ops.append(w.tensor(f"{n.output.name}/{v.name}", arr.shape, arr.dtype.type, arr))
                else:
                    ops.append(get(v, NCHW))
            w.op(B.SELECT_V2, ops, [new(n, NCHW)], O.SelectV2Options, S.SelectV2OptionsT())
        elif n.op == "index":
            x, idx = ins[0], list(ins[1:])
            if len(idx) != 1 or idx[0] is None or not low.is_const(x):
                raise LoweringError(T, n, "advanced indexing other than a constant table gathered on dim 0")
            opt = S.GatherOptionsT()
            opt.axis = 0
            w.op(B.GATHER, [const_t(x), get(idx[0])], [new(n, NCHW)], O.GatherOptions, opt)
        elif n.op == "attention":
            q, k, v, mask = ins
            qs = static_shape(T, q, n)
            ks = static_shape(T, k, n)
            scale = a.get("scale")
            scale = 1.0 / math.sqrt(qs[-1]) if scale is None else float(scale)
            base = n.output.name
            sshape = qs[:-1] + (ks[-2],)
            s = w.tensor(base + "/scores", sshape)
            opt = S.BatchMatMulOptionsT()
            opt.adjY = True
            w.op(B.BATCH_MATMUL, [get(q, NCHW), get(k, NCHW)], [s], O.BatchMatMulOptions, opt)
            s2 = w.tensor(base + "/scaled", sshape)
            w.op(B.MUL, [s, f32(base + "/scale", scale)], [s2], O.MulOptions, S.MulOptionsT())
            s = s2
            keep = None
            if a.get("is_causal"):
                keep = np.tril(np.ones((qs[-2], ks[-2]), np.bool_))
            if mask is not None:
                if mask.type.dtype != torch.bool or not low.is_const(mask):
                    raise LoweringError(T, n, "attention with a computed or additive mask")
                mk = low.consts[mask.name].numpy()
                keep = mk if keep is None else (keep & mk)
            if keep is not None:
                s3 = w.tensor(base + "/masked", sshape)
                w.op(B.SELECT_V2, [w.tensor(base + "/keep", keep.shape, np.bool_, keep), s,
                                   f32(base + "/ninf", -np.inf)], [s3], O.SelectV2Options, S.SelectV2OptionsT())
                s = s3
            p = w.tensor(base + "/probs", sshape)
            so = S.SoftmaxOptionsT()
            so.beta = 1.0
            w.op(B.SOFTMAX, [s], [p], O.SoftmaxOptions, so)
            w.op(B.BATCH_MATMUL, [p, get(v, NCHW)], [new(n, NCHW)], O.BatchMatMulOptions, S.BatchMatMulOptionsT())
        else:
            raise LoweringError(T, n, f"op {n.op} has no LiteRT lowering")

    outs = []
    for v in low.outputs:
        if v.name in low.consts:
            raise LoweringError(T, None, f"output %{v.name} does not depend on the inputs")
        outs.append(get(v, NCHW))
    in_idx = [env[v.name][0] for v in low.inputs]
    return w.finish(in_idx, outs, [v.name for v in low.inputs], [v.name for v in low.outputs])
