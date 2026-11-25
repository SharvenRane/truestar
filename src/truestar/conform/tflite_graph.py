"""Read, cut and evaluate .tflite graphs (from the published TFLite flatbuffer schema).

* `TfliteGraph(data)`: operators with their builtin names, tensor names, shapes and constants.
* `cut(op_indices, inputs, outputs)`: a new .tflite holding only those operators, whose inputs are
  the given tensors. Conformance compiles a module's cut alone and feeds it float64 reference
  inputs, so the error it measures is the module's own and the run that is judged is never
  instrumented.
* `evaluate64(feeds, wanted)`: the graph's own math in float64 (numpy), for the ops listed in
  `_EVAL`. It separates a conversion fault (the file computes something else than the model) from a
  runtime fault (the runtime computes the file wrongly). An op without a rule makes this verdict
  unavailable for the cut, never approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


def _schema():
    from ai_edge_litert import schema_py_generated as S

    return S


_NUMPY = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 6: np.bool_,
          7: np.int16, 9: np.int8, 10: np.float64}


@dataclass
class Op:
    index: int
    name: str  # builtin operator name, e.g. FULLY_CONNECTED
    inputs: List[int]
    outputs: List[int]
    options: object = None


@dataclass
class TfliteGraph:
    data: bytes
    model: object = None
    ops: List[Op] = field(default_factory=list)

    def __post_init__(self):
        S = _schema()
        self.S = S
        self.model = S.ModelT.InitFromPackedBuf(bytearray(self.data), 0)
        names = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
        sg = self.model.subgraphs[0]
        self.sg = sg
        for i, o in enumerate(sg.operators or []):
            oc = self.model.operatorCodes[o.opcodeIndex]
            code = max(int(oc.builtinCode), int(oc.deprecatedBuiltinCode))
            nm = names.get(code, f"BUILTIN_{code}")
            if code == S.BuiltinOperator.CUSTOM:
                nm = f"CUSTOM:{oc.customCode.decode() if isinstance(oc.customCode, bytes) else oc.customCode}"
            self.ops.append(Op(i, nm, [int(x) for x in (o.inputs if o.inputs is not None else [])],
                               [int(x) for x in (o.outputs if o.outputs is not None else [])],
                               o.builtinOptions))

    # ------------------------------------------------------------------ tensors
    def tensor_name(self, i: int) -> str:
        n = self.sg.tensors[i].name
        return n.decode() if isinstance(n, bytes) else (n or f"tensor_{i}")

    def shape(self, i: int):
        s = self.sg.tensors[i].shape
        return tuple(int(x) for x in (s if s is not None else []))

    def dtype(self, i: int):
        return _NUMPY.get(int(self.sg.tensors[i].type), np.float32)

    def constant(self, i: int) -> Optional[np.ndarray]:
        t = self.sg.tensors[i]
        b = self.model.buffers[t.buffer] if t.buffer else None
        if b is None or b.data is None or len(b.data) == 0:
            return None
        raw = np.asarray(b.data, dtype=np.uint8).tobytes()
        return np.frombuffer(raw, dtype=self.dtype(i)).reshape(self.shape(i))

    def quant(self, i: int):
        """(scales float64, zero points int64, axis) of a quantized integer tensor, else None."""
        t = self.sg.tensors[i]
        q = t.quantization
        if q is None or q.scale is None or len(q.scale) == 0 or np.dtype(self.dtype(i)).kind not in "iu":
            return None
        zp = np.asarray(q.zeroPoint if q.zeroPoint is not None and len(q.zeroPoint) else [0], np.int64)
        return np.asarray(q.scale, np.float64), zp, int(q.quantizedDimension or 0)

    def _per_axis(self, i: int, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        """`v` (one value per channel, or one) shaped to broadcast against `a` along the tensor's axis."""
        v = np.asarray(v)
        if v.size == 1 or a.ndim == 0:
            return v.reshape(()) if v.size == 1 else v
        shape = [1] * a.ndim
        shape[self.quant(i)[2]] = v.size
        return v.reshape(shape)

    def dequantize(self, i: int, q: np.ndarray) -> np.ndarray:
        """Integer codes of tensor `i` as real numbers, float64: (q - zero point) * scale."""
        s, zp, _ = self.quant(i)
        q = np.asarray(q, np.int64)
        return (q - self._per_axis(i, q, zp)).astype(np.float64) * self._per_axis(i, q, s)

    def quantize(self, i: int, x: np.ndarray, float32: bool = False) -> np.ndarray:
        """Real numbers as tensor `i`'s integer codes: round half away from zero of x / scale, plus the
        zero point, clamped to the type (TFLite's reference QUANTIZE, which divides in float32 for a
        float32 input: `float32=True`)."""
        s, zp, _ = self.quant(i)
        x = np.asarray(x, np.float64)
        if float32:
            r = (x.astype(np.float32) / self._per_axis(i, x, s).astype(np.float32)).astype(np.float64)
        else:
            r = x / self._per_axis(i, x, s)
        q = np.sign(r) * np.floor(np.abs(r) + 0.5) + self._per_axis(i, x, zp)
        info = np.iinfo(self.dtype(i))
        return np.clip(q, info.min, info.max).astype(self.dtype(i))

    def real_constant(self, i: int) -> Optional[np.ndarray]:
        """A constant as the number it stands for: dequantized when quantized, float64 when float."""
        c = self.constant(i)
        if c is None:
            return None
        if self.quant(i) is not None:
            return self.dequantize(i, c)
        return c.astype(np.float64) if c.dtype.kind == "f" else c

    @property
    def inputs(self) -> List[int]:
        return [int(x) for x in self.sg.inputs]

    @property
    def outputs(self) -> List[int]:
        return [int(x) for x in self.sg.outputs]

    def producers(self) -> Dict[int, int]:
        return {t: op.index for op in self.ops for t in op.outputs}

    # ------------------------------------------------------------------ cut
    def cut(self, op_indices: Sequence[int], inputs: Sequence[int], outputs: Sequence[int]) -> bytes:
        """A model of only these operators, with `inputs` as graph inputs and `outputs` as graph
        outputs. Constants are copied; tensor indices are renumbered."""
        import copy

        import flatbuffers

        S = self.S
        keep = sorted(set(op_indices))
        used: List[int] = []
        for i in list(inputs) + [t for k in keep for t in self.ops[k].inputs + self.ops[k].outputs]:
            if i >= 0 and i not in used:
                used.append(i)
        remap = {old: new for new, old in enumerate(used)}
        m = S.ModelT()
        m.version = self.model.version
        m.description = "cut by truestar conform"
        m.buffers = [S.BufferT()]
        tensors = []
        for old in used:
            t = copy.copy(self.sg.tensors[old])
            if t.buffer and old not in inputs:
                b = S.BufferT()
                b.data = self.model.buffers[t.buffer].data
                m.buffers.append(b)
                t.buffer = len(m.buffers) - 1
            else:
                t.buffer = 0
            tensors.append(t)
        codes: Dict[int, int] = {}
        m.operatorCodes = []
        ops = []
        for k in keep:
            o = copy.copy(self.sg.operators[k])
            if o.opcodeIndex not in codes:
                codes[o.opcodeIndex] = len(m.operatorCodes)
                m.operatorCodes.append(self.model.operatorCodes[o.opcodeIndex])
            o.opcodeIndex = codes[o.opcodeIndex]
            o.inputs = [remap[i] if i >= 0 else -1 for i in (o.inputs if o.inputs is not None else [])]
            o.outputs = [remap[i] for i in o.outputs]
            ops.append(o)
        sg = S.SubGraphT()
        sg.tensors, sg.operators, sg.name = tensors, ops, "cut"
        sg.inputs = [remap[i] for i in inputs]
        sg.outputs = [remap[i] for i in outputs]
        m.subgraphs = [sg]
        sig = S.SignatureDefT()
        sig.signatureKey = "serving_default"
        sig.subgraphIndex = 0

        def tmap(i):
            t = S.TensorMapT()
            t.name, t.tensorIndex = f"t{i}", remap[i]
            return t

        sig.inputs = [tmap(i) for i in inputs]
        sig.outputs = [tmap(i) for i in outputs]
        m.signatureDefs = [sig]
        b = flatbuffers.Builder(1 << 16)
        b.Finish(m.Pack(b), file_identifier=b"TFL3")
        return bytes(b.Output())

    # ------------------------------------------------------------------ float64 meaning
    def evaluate64(self, feeds: Dict[int, np.ndarray], wanted: Sequence[int],
                   op_indices: Optional[Sequence[int]] = None) -> Optional[List[np.ndarray]]:
        """The graph's math in float64 for `wanted` tensors; None when an op has no rule.

        Every value is a real number: a quantized tensor holds its dequantized codes, so feeds and
        results are in the model's units whatever the file's integer types. An op whose output is
        quantized has its exact real result snapped to the output's grid (round half away from zero,
        clamped), which is the integer arithmetic the file asks for, without the fixed point
        multiplier a runtime uses for it (that can move a value one step at a near tie)."""
        vals: Dict[int, np.ndarray] = {}
        for k, v in feeds.items():
            v = np.asarray(v)
            if self.quant(k) is None:
                vals[k] = v.astype(np.float64)
            else:  # codes, or real numbers put on the tensor's grid (what a runtime is fed)
                vals[k] = self.dequantize(k, v if v.dtype.kind in "iu" else self.quantize(k, v))
        order = sorted(op_indices) if op_indices is not None else range(len(self.ops))
        for k in order:
            op = self.ops[k]
            if all(o in vals for o in op.outputs):
                continue
            fn = _EVAL.get(op.name)
            if fn is None:
                return None
            args = []
            for i in op.inputs:
                if i < 0:
                    args.append(None)
                elif i in vals:
                    args.append(vals[i])
                else:
                    c = self.real_constant(i)
                    if c is None:
                        return None
                    args.append(c)
            try:
                res = fn(args, op.options)
            except KeyError:
                return None
            out = op.outputs[0]
            res = np.asarray(res, dtype=np.float64).reshape(self.shape(out))
            if self.quant(out) is not None:
                f32 = op.name == "QUANTIZE" and op.inputs and np.dtype(self.dtype(op.inputs[0])) == np.float32
                res = self.dequantize(out, self.quantize(out, res, float32=bool(f32)))
            vals[out] = res
        try:
            return [vals[i] for i in wanted]
        except KeyError:
            return None


# ----------------------------------------------------------------------------- float64 rules

def _act(x, opts):
    f = getattr(opts, "fusedActivationFunction", 0) if opts is not None else 0
    if f == 0:
        return x
    if f == 1:
        return np.maximum(x, 0.0)
    if f == 3:
        return np.clip(x, 0.0, 6.0)
    if f == 2:
        return np.clip(x, -1.0, 1.0)
    raise KeyError(f"fused activation {f}")


def _fc(a, o):
    x, w = a[0], a[1]
    b = a[2] if len(a) > 2 else None
    keep = bool(getattr(o, "keepNumDims", False)) if o is not None else False
    lead = x.shape[:-1]
    y = x.reshape(-1, w.shape[1]) @ w.T
    if b is not None:
        y = y + b
    y = _act(y, o)
    return y.reshape(lead + (w.shape[0],)) if keep else y


def _erf(x):
    from math import erf

    return np.vectorize(erf)(x)


def _gelu(a, o):
    x = a[0]
    if o is not None and getattr(o, "approximate", False):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _mean(a, o):
    axes = tuple(int(v) for v in np.asarray(a[1]).reshape(-1))
    keep = bool(getattr(o, "keepDims", False)) if o is not None else False
    return a[0].mean(axis=axes, keepdims=keep)


def _softmax(a, o):
    beta = float(getattr(o, "beta", 1.0)) if o is not None else 1.0
    z = a[0] * beta
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _cumsum(a, o):
    if o is not None and (getattr(o, "exclusive", False) or getattr(o, "reverse", False)):
        raise KeyError("exclusive or reverse cumsum")
    return np.cumsum(a[0], axis=int(np.asarray(a[1]).reshape(-1)[0]))


_EVAL = {
    "FULLY_CONNECTED": _fc,
    "ADD": lambda a, o: _act(a[0] + a[1], o),
    "SUB": lambda a, o: _act(a[0] - a[1], o),
    "MUL": lambda a, o: _act(a[0] * a[1], o),
    "DIV": lambda a, o: _act(a[0] / a[1], o),
    "RELU": lambda a, o: np.maximum(a[0], 0.0),
    "RELU6": lambda a, o: np.clip(a[0], 0.0, 6.0),
    "LOGISTIC": lambda a, o: 1.0 / (1.0 + np.exp(-a[0])),
    "TANH": lambda a, o: np.tanh(a[0]),
    "GELU": _gelu,
    "MEAN": _mean,
    "SQUARED_DIFFERENCE": lambda a, o: (a[0] - a[1]) ** 2,
    "RSQRT": lambda a, o: 1.0 / np.sqrt(a[0]),
    "SQRT": lambda a, o: np.sqrt(a[0]),
    "SOFTMAX": _softmax,
    "CUMSUM": _cumsum,
    "RESHAPE": lambda a, o: a[0],  # the output shape comes from the output tensor
    "EXP": lambda a, o: np.exp(a[0]),
    "SIN": lambda a, o: np.sin(a[0]),
    "COS": lambda a, o: np.cos(a[0]),
    "ABS": lambda a, o: np.abs(a[0]),
    "NEG": lambda a, o: -a[0],
    # values are real numbers (see evaluate64): quantizing is the snap to the output's grid, and
    # dequantizing changes nothing
    "QUANTIZE": lambda a, o: a[0],
    "DEQUANTIZE": lambda a, o: a[0],
}


# float64 rules for the operators truestar export --to litert writes (docs/EXPORT.md)

def _same_pads(size, k, s, d):
    out = -(-size // s)
    total = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return total // 2, total - total // 2


def _window(x, kh, kw, sh, sw, dh, dw, padding, value):
    """NHWC windows: (N, OH, OW, kh, kw, C) after LiteRT's SAME or VALID padding."""
    n, h, w, c = x.shape
    if padding == 0:  # SAME
        (t, b), (l, r) = _same_pads(h, kh, sh, dh), _same_pads(w, kw, sw, dw)
        x = np.pad(x, ((0, 0), (t, b), (l, r), (0, 0)), constant_values=value)
    n, h, w, c = x.shape
    oh, ow = (h - (kh - 1) * dh - 1) // sh + 1, (w - (kw - 1) * dw - 1) // sw + 1
    out = np.empty((n, oh, ow, kh, kw, c), np.float64)
    for u in range(kh):
        for v in range(kw):
            out[:, :, :, u, v, :] = x[:, u * dh:u * dh + sh * (oh - 1) + 1:sh, v * dw:v * dw + sw * (ow - 1) + 1:sw, :]
    return out


def _conv(a, o):
    x, w = a[0], a[1]
    win = _window(x, w.shape[1], w.shape[2], o.strideH, o.strideW, o.dilationHFactor or 1, o.dilationWFactor or 1,
                  o.padding, 0.0)
    y = np.einsum("nhwuvc,ouvc->nhwo", win, w)
    if len(a) > 2 and a[2] is not None:
        y = y + a[2]
    return _act(y, o)


def _maxpool(a, o):
    win = _window(a[0], o.filterHeight, o.filterWidth, o.strideH, o.strideW, 1, 1, o.padding, -np.inf)
    return _act(win.max(axis=(3, 4)), o)


def _padv2(a, o):
    pads = np.asarray(a[1]).astype(np.int64)
    value = float(np.asarray(a[2]).reshape(-1)[0]) if len(a) > 2 and a[2] is not None else 0.0
    return np.pad(a[0], [tuple(p) for p in pads], constant_values=value)


def _strided_slice(a, o):
    x = a[0]
    b, e, s = (np.asarray(v).astype(np.int64).reshape(-1) for v in a[1:4])
    if getattr(o, "beginMask", 0) or getattr(o, "endMask", 0) or getattr(o, "ellipsisMask", 0) or \
            getattr(o, "newAxisMask", 0) or getattr(o, "offset", False):
        raise KeyError("strided slice masks")
    y = x[tuple(slice(int(b[i]), int(e[i]), int(s[i])) for i in range(x.ndim))]
    shrink = getattr(o, "shrinkAxisMask", 0) or 0
    return y.reshape([d for i, d in enumerate(y.shape) if not shrink >> i & 1])


def _bmm(a, o):
    x, y = a[0], a[1]
    if getattr(o, "adjX", False):
        x = np.swapaxes(x, -1, -2)
    if getattr(o, "adjY", False):
        y = np.swapaxes(y, -1, -2)
    return np.matmul(x, y)


def _gather(a, o):
    axis = int(getattr(o, "axis", 0) or 0)
    if getattr(o, "batchDims", 0):
        raise KeyError("gather batch dims")
    return np.take(a[0], np.asarray(a[1]).astype(np.int64), axis=axis)


_EVAL.update({
    "CONV_2D": _conv,
    "MAX_POOL_2D": _maxpool,
    "PADV2": _padv2,
    "PAD": lambda a, o: np.pad(a[0], [tuple(p) for p in np.asarray(a[1]).astype(np.int64)]),
    "TRANSPOSE": lambda a, o: np.transpose(a[0], np.asarray(a[1]).astype(np.int64).reshape(-1)),
    "STRIDED_SLICE": _strided_slice,
    "BATCH_MATMUL": _bmm,
    "GATHER": _gather,
    "SELECT_V2": lambda a, o: np.where(np.asarray(a[0]).astype(bool), a[1], a[2]),
    "LOGICAL_NOT": lambda a, o: ~np.asarray(a[0]).astype(bool),
    "LOG": lambda a, o: np.log(a[0]),
    "MAXIMUM": lambda a, o: np.maximum(a[0], a[1]),
    "MINIMUM": lambda a, o: np.minimum(a[0], a[1]),
})
