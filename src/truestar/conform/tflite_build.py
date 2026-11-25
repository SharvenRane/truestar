"""A small PyTorch to .tflite writer, so conformance has real .tflite files to check.

No converter from PyTorch to TFLite runs on this machine without pinning another torch, so this
module writes the flatbuffer itself from the published TFLite schema (through the schema object API
that ships with the LiteRT wheel, `ai_edge_litert.schema_py_generated`). It covers the ops small
test models use: linear, relu, sigmoid, tanh, sin, cos, exp, abs, sqrt, rsqrt, neg, gelu, layer_norm (decomposed into MEAN,
SQUARED_DIFFERENCE, RSQRT and so on, as converters do), softmax, cumsum, add, sub, mul, reshape.

Every tensor is named `<module path>/<node name>` from torch.export's module stack, the way
converters that keep source locations name them, so conformance can map tensors back to modules.

`faults` changes the decomposition on purpose, to show what conformance finds:

* "gelu_tanh": gelu written as the tanh approximation (an exporter decomposition change);
* "layernorm_eps_outside": layer_norm as (x - mean) / (sqrt(var) + eps) instead of
  (x - mean) / sqrt(var + eps).

This is a writer for tests and demonstrations, not a general converter: an op outside the list is
refused with its name and module, never written approximately.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


class TfliteBuildError(RuntimeError):
    pass


def _schema():
    try:
        from ai_edge_litert import schema_py_generated as S
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise TfliteBuildError("writing .tflite needs the ai-edge-litert package "
                               f"({type(exc).__name__}: {exc})") from exc
    return S


_UNARY = {"aten.relu.default": "RELU", "aten.sigmoid.default": "LOGISTIC",
          "aten.tanh.default": "TANH", "aten.sin.default": "SIN", "aten.cos.default": "COS",
          "aten.exp.default": "EXP", "aten.abs.default": "ABS", "aten.sqrt.default": "SQRT",
          "aten.rsqrt.default": "RSQRT", "aten.neg.default": "NEG"}


def _module_path(node) -> str:
    stack = node.meta.get("nn_module_stack") or {}
    if not stack:
        return ""
    return list(stack.values())[-1][0] or ""


class _Writer:
    def __init__(self):
        self.S = _schema()
        self.tensors = []
        self.buffers = [self.S.BufferT()]
        self.codes: Dict[int, int] = {}
        self.code_list = []
        self.ops = []

    def buffer(self, a: np.ndarray) -> int:
        b = self.S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(a).tobytes(), np.uint8)
        self.buffers.append(b)
        return len(self.buffers) - 1

    def tensor(self, name: str, shape, dtype=np.float32, data: Optional[np.ndarray] = None,
               scale=None, zero_point=None, axis: int = 0) -> int:
        """A tensor; `scale` (one value, or one per channel along `axis`) makes it quantized."""
        S = self.S
        t = S.TensorT()
        t.name = name
        t.shape = [int(s) for s in shape]
        t.type = {np.float32: S.TensorType.FLOAT32, np.int32: S.TensorType.INT32,
                  np.int8: S.TensorType.INT8}[dtype]
        t.buffer = 0 if data is None else self.buffer(np.asarray(data, dtype=dtype))
        if scale is not None:
            q = S.QuantizationParametersT()
            q.scale = [float(v) for v in np.atleast_1d(np.asarray(scale, np.float32))]
            zp = np.zeros(len(q.scale), np.int64) if zero_point is None else np.atleast_1d(zero_point)
            q.zeroPoint = [int(v) for v in zp]
            q.quantizedDimension = int(axis)
            t.quantization = q
        self.tensors.append(t)
        return len(self.tensors) - 1

    def const(self, name: str, value) -> int:
        a = np.asarray(value, dtype=np.float32)
        return self.tensor(name, a.shape, np.float32, a)

    def op(self, code: int, ins: Sequence[int], outs: Sequence[int], opt_type: int = 0,
           opt=None, version: int = 1) -> None:
        S = self.S
        if code not in self.codes:
            oc = S.OperatorCodeT()
            oc.builtinCode = code
            oc.deprecatedBuiltinCode = min(code, 127)
            oc.version = version  # the TFLite op version the operand types need (op_version.cc)
            self.codes[code] = len(self.code_list)
            self.code_list.append(oc)
        o = S.OperatorT()
        o.opcodeIndex = self.codes[code]
        o.inputs = list(ins)
        o.outputs = list(outs)
        o.builtinOptionsType = opt_type
        o.builtinOptions = opt
        self.ops.append(o)

    def finish(self, inputs: List[int], outputs: List[int], in_names: List[str],
               out_names: List[str]) -> bytes:
        import flatbuffers

        S = self.S
        sg = S.SubGraphT()
        sg.tensors, sg.inputs, sg.outputs, sg.operators, sg.name = (
            self.tensors, inputs, outputs, self.ops, "main")
        m = S.ModelT()
        m.version = 3
        m.operatorCodes = self.code_list
        m.subgraphs = [sg]
        m.buffers = self.buffers
        m.description = "written by truestar.conform.tflite_build"
        sig = S.SignatureDefT()
        sig.signatureKey = "serving_default"
        sig.subgraphIndex = 0

        def tmap(name, idx):
            t = S.TensorMapT()
            t.name, t.tensorIndex = name, idx
            return t

        sig.inputs = [tmap(n, i) for n, i in zip(in_names, inputs)]
        sig.outputs = [tmap(n, i) for n, i in zip(out_names, outputs)]
        m.signatureDefs = [sig]
        b = flatbuffers.Builder(1 << 16)
        b.Finish(m.Pack(b), file_identifier=b"TFL3")
        return bytes(b.Output())


def build_tflite(model: nn.Module, args: Tuple, faults: Sequence[str] = ()) -> bytes:
    """Write `model` called on `args` (static shapes) as .tflite bytes."""
    unknown = set(faults) - {"gelu_tanh", "layernorm_eps_outside"}
    if unknown:
        raise TfliteBuildError(f"unknown fault(s): {sorted(unknown)}")
    w = _Writer()
    S = w.S
    B = S.BuiltinOperator
    O = S.BuiltinOptions
    ep = torch.export.export(model.eval(), tuple(args))
    sig = ep.graph_signature
    state = dict(ep.state_dict)
    consts = dict(getattr(ep, "constants", {}) or {})
    param_of = {**sig.inputs_to_parameters, **sig.inputs_to_buffers,
                **getattr(sig, "inputs_to_lifted_tensor_constants", {})}
    env: Dict[str, object] = {}  # fx node name -> tensor index or numpy constant
    inputs, in_names = [], []

    def shape_of(n):
        return tuple(n.meta["val"].shape)

    def val(a):
        if isinstance(a, torch.fx.Node):
            return env[a.name]
        return a

    def tid(a, name):  # a tensor index, materializing constants
        v = val(a)
        if isinstance(v, int) and not isinstance(v, bool) and isinstance(a, torch.fx.Node):
            return v
        if isinstance(v, np.ndarray):
            return w.const(name, v)
        return w.const(name, np.float32(v))

    for n in ep.graph.nodes:
        if n.op == "placeholder":
            if n.name in param_of:
                key = param_of[n.name]
                t = state.get(key, consts.get(key))
                env[n.name] = t.detach().cpu().numpy().astype(np.float32)
            else:
                idx = w.tensor(n.name, shape_of(n))
                env[n.name] = idx
                inputs.append(idx)
                in_names.append(n.name)
            continue
        if n.op == "output":
            outs = [val(a) for a in n.args[0]]
            names = [a.name for a in n.args[0]]
            return w.finish(inputs, [int(o) for o in outs], in_names, names)
        if n.op != "call_function":
            raise TfliteBuildError(f"unsupported graph node {n.op} {n.name}")
        path = _module_path(n)
        name = f"{path}/{n.name}"
        t = str(n.target)
        out = w.tensor(name, shape_of(n))
        a = n.args
        if t == "aten.linear.default":
            wt = val(a[1])
            ins = [val(a[0]), w.const(name + ".weight", wt)]
            ins.append(w.const(name + ".bias", val(a[2])) if len(a) > 2 and a[2] is not None else -1)
            opt = S.FullyConnectedOptionsT()
            opt.keepNumDims = len(shape_of(n)) > 2
            w.op(B.FULLY_CONNECTED, ins, [out], O.FullyConnectedOptions, opt)
        elif t in _UNARY:
            code = getattr(B, _UNARY[t])
            w.op(code, [val(a[0])], [out])
        elif t == "aten.gelu.default":
            approx = n.kwargs.get("approximate", "none") == "tanh"
            if "gelu_tanh" in faults:
                approx = True
            opt = S.GeluOptionsT()
            opt.approximate = approx
            w.op(B.GELU, [val(a[0])], [out], O.GeluOptions, opt)
        elif t in ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor"):
            code, ot, oc = {"aten.add.Tensor": (B.ADD, O.AddOptions, S.AddOptionsT),
                            "aten.sub.Tensor": (B.SUB, O.SubOptions, S.SubOptionsT),
                            "aten.mul.Tensor": (B.MUL, O.MulOptions, S.MulOptionsT)}[t]
            if n.kwargs.get("alpha", 1) != 1:
                raise TfliteBuildError(f"{t} with alpha in {path or 'the top module'}")
            w.op(code, [tid(a[0], name + ".a"), tid(a[1], name + ".b")], [out], ot, oc())
        elif t == "aten.softmax.int" or t == "aten._softmax.default":
            dim = a[1] if a[1] >= 0 else a[1] + len(shape_of(n))
            if dim != len(shape_of(n)) - 1:
                raise TfliteBuildError(f"softmax over a non last dim in {path}")
            opt = S.SoftmaxOptionsT()
            opt.beta = 1.0
            w.op(B.SOFTMAX, [val(a[0])], [out], O.SoftmaxOptions, opt)
        elif t == "aten.cumsum.default":
            dim = w.tensor(name + ".axis", (), np.int32, np.int32(a[1]))
            w.op(B.CUMSUM, [val(a[0]), dim], [out], O.CumsumOptions, S.CumsumOptionsT())
        elif t in ("aten.view.default", "aten.reshape.default"):
            shp = w.tensor(name + ".shape", (len(shape_of(n)),), np.int32,
                           np.asarray(shape_of(n), np.int32))
            w.op(B.RESHAPE, [val(a[0]), shp], [out])
        elif t == "aten.layer_norm.default":
            x = val(a[0])
            norm = a[1]
            gamma = val(a[2]) if len(a) > 2 and a[2] is not None else None
            beta = val(a[3]) if len(a) > 3 and a[3] is not None else None
            eps = float(a[4]) if len(a) > 4 else 1e-5
            if len(norm) != 1:
                raise TfliteBuildError(f"layer_norm over {len(norm)} dims in {path}")
            shp = shape_of(n)
            red = list(shp[:-1]) + [1]
            axis = w.tensor(name + ".axis", (1,), np.int32, np.asarray([len(shp) - 1], np.int32))

            def mean(src, nm):
                o = w.tensor(f"{path}/{nm}", red)
                opt = S.ReducerOptionsT()
                opt.keepDims = True
                w.op(B.MEAN, [src, axis], [o], O.ReducerOptions, opt)
                return o

            mu = mean(x, n.name + ".mean")
            sq = w.tensor(f"{path}/{n.name}.sqdiff", shp)
            w.op(B.SQUARED_DIFFERENCE, [x, mu], [sq])
            var = mean(sq, n.name + ".var")
            cen = w.tensor(f"{path}/{n.name}.centered", shp)
            w.op(B.SUB, [x, mu], [cen], O.SubOptions, S.SubOptionsT())
            inv = w.tensor(f"{path}/{n.name}.inv", red)
            if "layernorm_eps_outside" in faults:
                sd = w.tensor(f"{path}/{n.name}.sd", red)
                w.op(B.SQRT, [var], [sd])
                sde = w.tensor(f"{path}/{n.name}.sd_eps", red)
                w.op(B.ADD, [sd, w.const(name + ".eps", np.float32(eps))], [sde], O.AddOptions,
                     S.AddOptionsT())
                one = w.const(name + ".one", np.float32(1.0))
                w.op(B.DIV, [one, sde], [inv], O.DivOptions, S.DivOptionsT())
            else:
                ve = w.tensor(f"{path}/{n.name}.var_eps", red)
                w.op(B.ADD, [var, w.const(name + ".eps", np.float32(eps))], [ve], O.AddOptions,
                     S.AddOptionsT())
                w.op(B.RSQRT, [ve], [inv])
            last = out if gamma is None and beta is None else w.tensor(
                f"{path}/{n.name}.normed", shp)
            w.op(B.MUL, [cen, inv], [last], O.MulOptions, S.MulOptionsT())
            if gamma is not None:
                g_out = out if beta is None else w.tensor(f"{path}/{n.name}.scaled", shp)
                w.op(B.MUL, [last, w.const(name + ".gamma", gamma)], [g_out], O.MulOptions,
                     S.MulOptionsT())
                last = g_out
            if beta is not None:
                w.op(B.ADD, [last, w.const(name + ".beta", beta)], [out], O.AddOptions,
                     S.AddOptionsT())
        else:
            raise TfliteBuildError(
                f"op {t} (node {n.name} in {path or 'the top module'}) is not in this writer's "
                "list; nothing was written")
        env[n.name] = out
    raise TfliteBuildError("graph has no output node")


def build_int8_mlp(layers: Sequence[Dict], x_shape: Tuple[int, int], x_scale: float, x_zero_point: int,
                   taps: bool = True, prefix: str = "", bare_outputs: bool = False,
                   input_name: str = "input") -> bytes:
    """A fully connected INT8 network as .tflite, the way a full integer converter writes one.

    float32 input -> QUANTIZE (int8, per tensor) -> FULLY_CONNECTED per layer (int8 activations,
    int8 weights [N, K] quantized per output channel, `quantizedDimension` 0, int32 bias with scale
    x_scale * w_scale[n] and zero point 0, ReLU fused) -> DEQUANTIZE. The weights are stored in the
    layout FULLY_CONNECTED reads, so no TRANSPOSE of a per channel tensor is needed (the op AI Hub's
    ONNX to TFLite converter refuses, docs/QAIHUB.md).

    Each layer is a dict: `name`, `w` (int8 [N, K]), `w_scale` (float32 [N]), `bias` (int32 [N]),
    `y_scale`, `y_zero_point` (int8 output), `relu`. With `taps` every layer output is dequantized
    and is a graph output named `name` (tensor `<prefix><name>/out`, or `name` itself with
    `bare_outputs`, for runtimes that report outputs by tensor name); otherwise only the last."""
    w = _Writer()
    S = w.S
    B = S.BuiltinOperator
    rows, k_in = (int(v) for v in x_shape)
    x = w.tensor(f"{prefix}{input_name}", (rows, k_in))
    cur = w.tensor(f"{prefix}{input_name}/quantized", (rows, k_in), np.int8, scale=x_scale,
                   zero_point=x_zero_point)
    w.op(B.QUANTIZE, [x], [cur])
    s_in = np.float64(np.float32(x_scale))
    outs, out_names = [], []
    for j, L in enumerate(layers):
        name = L["name"]
        wq = np.asarray(L["w"], np.int8)
        n_out, k = wq.shape
        if k != w.tensors[cur].shape[1]:
            raise TfliteBuildError(f"layer {name}: weight [N, K] = {list(wq.shape)} does not take "
                                   f"{w.tensors[cur].shape[1]} inputs")
        sw = np.asarray(L["w_scale"], np.float32).reshape(-1)
        wt = w.tensor(f"{prefix}{name}/weight", (n_out, k), np.int8, wq, scale=sw, axis=0)
        bt = w.tensor(f"{prefix}{name}/bias", (n_out,), np.int32, np.asarray(L["bias"], np.int32),
                      scale=(s_in * sw.astype(np.float64)).astype(np.float32), axis=0)
        yq = w.tensor(f"{prefix}{name}/quantized", (rows, n_out), np.int8, scale=L["y_scale"],
                      zero_point=L["y_zero_point"])
        opt = S.FullyConnectedOptionsT()
        opt.fusedActivationFunction = (S.ActivationFunctionType.RELU if L.get("relu")
                                       else S.ActivationFunctionType.NONE)
        w.op(B.FULLY_CONNECTED, [cur, wt, bt], [yq], S.BuiltinOptions.FullyConnectedOptions, opt,
             version=4)  # int8 input, weight and output
        if taps or j == len(layers) - 1:
            y = w.tensor(name if bare_outputs else f"{prefix}{name}/out", (rows, n_out))
            w.op(B.DEQUANTIZE, [yq], [y], version=2)  # int8 input
            outs.append(y)
            out_names.append(name)
        cur, s_in = yq, np.float64(np.float32(L["y_scale"]))
    return w.finish([x], outs, [input_name], out_names)
