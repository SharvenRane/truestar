"""Fixed battery for hosted phone, laptop and IoT accelerators (Qualcomm AI Hub), docs/QAIHUB.md.

Two models, each one ONNX file, so one compile and one inference job cover every case:

* ``float``: three float32 networks side by side, every intermediate tensor worth judging is an
  extra output (a tap): an MLP (512, 1024, 1024, 10, GELU then ReLU), a small residual conv net
  (64x64 input, two residual blocks, a stride 2 conv, global pool, linear head) and one transformer
  encoder layer written out op by op (4 heads, model width 128, 64 tokens, post norm, GELU FFN).
* ``int8``: a three layer MLP as a QDQ graph: uint8 activations with per tensor scales from a
  separate calibration draw, int8 weights per output channel from ``truestar.quant`` (``int8``
  spec), int32 biases, ReLU before each requantize. Its intended arithmetic is computed with
  ``truestar.intref`` (exact integer accumulation, one requantization per layer).

Weights and inputs come from numpy's PCG64 ``random()`` (uniform doubles, exact on every
platform) scaled to unit variance, so any host regenerates the same bytes; the hashes are pinned
in tests/test_qaihub.py. Torch is imported, onnx only when a file is written.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

BATTERY_VERSION = 1
SEED = 20261124
INT8_ROWS = 16

# input name -> shape, per model
FLOAT_INPUTS = {"x_mlp": (8, 512), "x_conv": (1, 3, 64, 64), "x_enc": (1, 64, 128)}
INT8_INPUTS = {"x_q": (INT8_ROWS, 256)}
INT8_DIMS = (256, 256, 256, 10)


def _uniform(seed: int, shape, scale: float = 1.0) -> np.ndarray:
    """Unit variance (times scale) uniform values, float64, the same bytes on every platform."""
    u = np.random.Generator(np.random.PCG64(seed)).random(int(np.prod(shape)))
    return ((u * 2.0 - 1.0) * math.sqrt(3.0) * scale).reshape(shape)


def _f32(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).astype(np.float32)


class _Param:
    """Hands out seeded tensors in a fixed order."""

    def __init__(self, base: int):
        self.base, self.n = base, 0

    def __call__(self, shape, scale: float) -> torch.Tensor:
        self.n += 1
        return torch.from_numpy(_f32(_uniform(self.base * 1000 + self.n, shape, scale)))


def _linear(p: _Param, i: int, o: int, gain: float = 1.0) -> nn.Linear:
    m = nn.Linear(i, o)
    with torch.no_grad():
        m.weight.copy_(p((o, i), gain / math.sqrt(i)))
        m.bias.copy_(p((o,), 0.1))
    return m


def _conv(p: _Param, i: int, o: int, stride: int = 1, gain: float = 1.0) -> nn.Conv2d:
    m = nn.Conv2d(i, o, 3, stride=stride, padding=1)
    with torch.no_grad():
        m.weight.copy_(p((o, i, 3, 3), gain / math.sqrt(9 * i)))
        m.bias.copy_(p((o,), 0.1))
    return m


class BatteryMLP(nn.Module):
    def __init__(self, p: _Param):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = _linear(p, 512, 1024), _linear(p, 1024, 1024, 1.4), _linear(p, 1024, 10)

    def forward(self, x):
        h1 = torch.nn.functional.gelu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return h1, h2, self.fc3(h2)


class BatteryConv(nn.Module):
    def __init__(self, p: _Param):
        super().__init__()
        self.stem = _conv(p, 3, 32, gain=1.4)
        self.b1a, self.b1b = _conv(p, 32, 32, gain=1.4), _conv(p, 32, 32, gain=0.7)
        self.down = _conv(p, 32, 64, stride=2, gain=1.0)
        self.b2a, self.b2b = _conv(p, 64, 64, gain=1.4), _conv(p, 64, 64, gain=0.7)
        self.head = _linear(p, 64, 10, gain=3.0)

    def forward(self, x):
        s = torch.relu(self.stem(x))
        b1 = torch.relu(s + self.b1b(torch.relu(self.b1a(s))))
        d = torch.relu(self.down(b1))
        b2 = torch.relu(d + self.b2b(torch.relu(self.b2a(d))))
        pooled = b2.mean(dim=(2, 3))
        return s, b1, d, b2, pooled, self.head(pooled)


class BatteryEncoder(nn.Module):
    """One post norm transformer encoder layer, every op explicit (no fused attention)."""

    def __init__(self, p: _Param, d: int = 128, heads: int = 4, ff: int = 512):
        super().__init__()
        self.d, self.h = d, heads
        self.q, self.k, self.v, self.o = (_linear(p, d, d) for _ in range(4))
        self.ff1, self.ff2 = _linear(p, d, ff, 1.4), _linear(p, ff, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        with torch.no_grad():
            for n in (self.n1, self.n2):
                n.weight.copy_(1.0 + p((d,), 0.1))
                n.bias.copy_(p((d,), 0.1))

    def forward(self, x):
        B, T, D = x.shape
        hd = D // self.h

        def split(t):
            return t.reshape(B, T, self.h, hd).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(hd)), dim=-1)
        ctx = torch.matmul(att, v).transpose(1, 2).reshape(B, T, D)
        a = self.o(ctx)
        h = self.n1(x + a)
        f = self.ff2(torch.nn.functional.gelu(self.ff1(h)))
        return att, a, h, self.n2(h + f)


class FloatBattery(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp, self.conv, self.enc = BatteryMLP(_Param(1)), BatteryConv(_Param(2)), BatteryEncoder(_Param(3))

    def forward(self, x_mlp, x_conv, x_enc):
        return (*self.mlp(x_mlp), *self.conv(x_conv), *self.enc(x_enc))


FLOAT_TAPS = ("mlp.fc1_gelu", "mlp.fc2_relu", "mlp.out", "conv.stem", "conv.block1", "conv.down",
              "conv.block2", "conv.pool", "conv.out", "enc.softmax", "enc.attn_out", "enc.norm1",
              "enc.out")
INT8_TAPS = ("int8.l1", "int8.l2", "int8.out")
# float tap -> case, so a report can say which network a tap belongs to
CASES = ("mlp", "conv", "enc", "int8")


def case_of(tap: str) -> str:
    return tap.split(".", 1)[0]


def float_model() -> FloatBattery:
    return FloatBattery().eval()


def inputs(model: str) -> Dict[str, np.ndarray]:
    """The battery's float32 inputs (one sample per input name)."""
    specs = FLOAT_INPUTS if model == "float" else INT8_INPUTS
    base = 50 if model == "float" else 60
    return {k: _f32(_uniform(SEED + base + i, s)) for i, (k, s) in enumerate(specs.items())}


def calibration_inputs() -> np.ndarray:
    """A separate draw for the INT8 activation ranges (never judged)."""
    return _f32(_uniform(SEED + 70, (256, INT8_DIMS[0])))


def input_hash(arrs: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256(f"v{BATTERY_VERSION}".encode())
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(f"|{k}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def float_reference(arrs: Optional[Dict[str, np.ndarray]] = None, dtype=torch.float64) -> Dict[str, np.ndarray]:
    """Every float tap in `dtype` (float64: the reference; float32: what a correct float32 device gives)."""
    arrs = arrs or inputs("float")
    m = float_model().to(dtype)
    with torch.no_grad():
        outs = m(*[torch.from_numpy(arrs[k]).to(dtype) for k in FLOAT_INPUTS])
    return {t: o.detach().to(torch.float64).numpy() for t, o in zip(FLOAT_TAPS, outs)}


def simulated(profile: str, arrs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Every float tap as the ``sim:*`` profile predicts it (truestar.devsim)."""
    from ..devsim import simulate

    arrs = arrs or inputs("float")
    m = float_model()
    with torch.no_grad(), simulate(profile):
        outs = m(*[torch.from_numpy(arrs[k]) for k in FLOAT_INPUTS])
    return {t: o.detach().to(torch.float64).numpy() for t, o in zip(FLOAT_TAPS, outs)}


# ----------------------------------------------------------------------------- INT8 model

def _int8_float_layers():
    p = _Param(4)
    return [_linear(p, i, o, 1.4 if j < 2 else 1.0) for j, (i, o) in enumerate(zip(INT8_DIMS[:-1], INT8_DIMS[1:]))]


def _u8_params(lo: float, hi: float) -> Tuple[float, int]:
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = float(np.float32((hi - lo) / 255.0))
    zp = int(np.clip(round(-lo / scale), 0, 255))
    return scale, zp


def int8_params() -> Dict:
    """Quantization of the INT8 MLP: weight codes, scales, zero points, int32 biases.

    Activation ranges are the min and max over the calibration draw of the float64 model; the
    weight codes and per channel float32 scales are ``truestar.quant.quantize_weight(w, "int8")``.
    """
    from ..quant.formats import quantize_weight

    layers = _int8_float_layers()
    x = torch.from_numpy(calibration_inputs()).double()
    sx, zx = _u8_params(float(x.min()), float(x.max()))
    out = {"input": {"scale": sx, "zp": zx}, "layers": []}
    s_in = sx
    with torch.no_grad():
        for j, lin in enumerate(layers):
            qw = quantize_weight(lin.weight.detach(), "int8")
            y = x @ lin.weight.double().T + lin.bias.double()
            last = j == len(layers) - 1
            if not last:
                y = torch.relu(y)
            sy, zy = _u8_params(float(y.min()), float(y.max()))
            sw = qw.scale.double().numpy()
            bias = np.round(lin.bias.double().numpy() / (s_in * sw)).astype(np.int64)
            out["layers"].append({"w": qw.data.numpy().astype(np.int8), "sw": qw.scale.numpy().astype(np.float32),
                                  "bias": bias.astype(np.int32), "sy": sy, "zy": zy, "relu": not last,
                                  "w_float": lin.weight.detach().double().numpy(),
                                  "b_float": lin.bias.detach().double().numpy()})
            x, s_in = y, sy
    return out


def int8_intended(arrs: Optional[Dict[str, np.ndarray]] = None, convention: str = "onnx_qlinear",
                  params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """INT8 taps (dequantized, float64) as `convention` computes the QDQ graph (truestar.intref)."""
    from ..intref import ops as Q

    arrs = arrs or inputs("int8")
    P = params or int8_params()
    sx, zx = P["input"]["scale"], P["input"]["zp"]
    q = Q.quantize(arrs["x_q"], {"scale": sx, "zp": zx, "out_signed": False}, convention)
    out, s_in, z_in = {}, sx, zx
    for tap, L in zip(INT8_TAPS, P["layers"]):
        p = {"x_zp": z_in, "sx": s_in, "sw": L["sw"].astype(np.float64), "sy": L["sy"], "y_zp": L["zy"],
             "out_signed": False}
        if L["relu"]:
            p["act_min"] = L["zy"]
        q = Q.fully_connected(q, L["w"], L["bias"], p, convention)
        out[tap] = (q.astype(np.float64) - L["zy"]) * L["sy"]
        s_in, z_in = L["sy"], L["zy"]
    return out


def int8_float_reference(arrs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """The unquantized MLP in float64 (to see the quantization error itself)."""
    arrs = arrs or inputs("int8")
    x = arrs["x_q"].astype(np.float64)
    out = {}
    for tap, L in zip(INT8_TAPS, int8_params()["layers"]):
        x = x @ L["w_float"].T + L["b_float"]
        if L["relu"]:
            x = np.maximum(x, 0.0)
        out[tap] = x
    return out


def int8_steps(params: Optional[Dict] = None) -> Dict[str, float]:
    P = params or int8_params()
    return {t: L["sy"] for t, L in zip(INT8_TAPS, P["layers"])}


# ----------------------------------------------------------------------------- ONNX files

def float_onnx_bytes() -> bytes:
    """The float battery as ONNX (opset 18, torch.onnx dynamo exporter), outputs named by tap."""
    import io
    import contextlib
    import warnings

    m = float_model()
    args = tuple(torch.from_numpy(a) for a in inputs("float").values())
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        prog = torch.onnx.export(m, args, dynamo=True, verbose=False, opset_version=18,
                                 input_names=list(FLOAT_INPUTS), output_names=[_onnx_name(t) for t in FLOAT_TAPS])
    return prog.model_proto.SerializeToString()


def _onnx_name(tap: str) -> str:
    return tap.replace(".", "__")


def tap_of_output(name: str) -> str:
    return name.replace("__", ".")


def int8_onnx_bytes(gemm: bool = False) -> bytes:
    """The INT8 MLP as a QDQ ONNX graph (opset 17): Q and DQ around every MatMul, per channel weights.

    ``gemm=True`` writes the same arithmetic as Gemm(transB=1) on [N, K] weights quantized along
    axis 0 (the layout PyTorch exports). AI Hub's ONNX to TFLite converter rejects both forms
    ("'tfl.transpose' op has mismatched quantized axes", docs/QAIHUB.md); the TFLite runs take
    ``int8_tflite_bytes`` instead."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    P = int8_params()
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    sx, zx = P["input"]["scale"], P["input"]["zp"]
    const("x_s", np.float32(sx))
    const("x_z", np.uint8(zx))
    nodes += [helper.make_node("QuantizeLinear", ["x_q", "x_s", "x_z"], ["x_u8"]),
              helper.make_node("DequantizeLinear", ["x_u8", "x_s", "x_z"], ["x_dq"])]
    cur, s_in = "x_dq", sx
    outs = []
    for j, (tap, L) in enumerate(zip(INT8_TAPS, P["layers"])):
        n = f"l{j + 1}"
        const(f"{n}_w", L["w"].copy() if gemm else L["w"].T.copy())  # [N, K] for Gemm, [K, N] for MatMul
        const(f"{n}_ws", L["sw"])
        const(f"{n}_wz", np.zeros(L["sw"].shape, np.int8))
        const(f"{n}_b", L["bias"])
        const(f"{n}_bs", (np.float64(s_in) * L["sw"].astype(np.float64)).astype(np.float32))
        const(f"{n}_bz", np.zeros(L["sw"].shape, np.int32))
        const(f"{n}_ys", np.float32(L["sy"]))
        const(f"{n}_yz", np.uint8(L["zy"]))
        nodes += [helper.make_node("DequantizeLinear", [f"{n}_w", f"{n}_ws", f"{n}_wz"], [f"{n}_wf"],
                                   axis=0 if gemm else 1),
                  helper.make_node("DequantizeLinear", [f"{n}_b", f"{n}_bs", f"{n}_bz"], [f"{n}_bf"], axis=0)]
        if gemm:
            nodes.append(helper.make_node("Gemm", [cur, f"{n}_wf", f"{n}_bf"], [f"{n}_acc"], transB=1))
        else:
            nodes += [helper.make_node("MatMul", [cur, f"{n}_wf"], [f"{n}_mm"]),
                      helper.make_node("Add", [f"{n}_mm", f"{n}_bf"], [f"{n}_acc"])]
        pre = f"{n}_acc"
        if L["relu"]:
            nodes.append(helper.make_node("Relu", [pre], [f"{n}_relu"]))
            pre = f"{n}_relu"
        out_name = _onnx_name(tap)
        nodes += [helper.make_node("QuantizeLinear", [pre, f"{n}_ys", f"{n}_yz"], [f"{n}_u8"]),
                  helper.make_node("DequantizeLinear", [f"{n}_u8", f"{n}_ys", f"{n}_yz"], [out_name])]
        outs.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [INT8_ROWS, INT8_DIMS[j + 1]]))
        cur, s_in = out_name, L["sy"]
    g = helper.make_graph(nodes, "truestar_int8_battery",
                          [helper.make_tensor_value_info("x_q", TensorProto.FLOAT, list(INT8_INPUTS["x_q"]))],
                          outs, inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="truestar")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def int8_tflite_bytes() -> bytes:
    """The INT8 MLP written directly as a full integer .tflite (``tflite_build.build_int8_mlp``).

    AI Hub's ONNX to TFLite converter refuses both QDQ forms above (docs/QAIHUB.md), so the TFLite
    runs take this file, which needs no conversion: AI Hub runs a .tflite as it is. TFLite's
    integer FULLY_CONNECTED with per channel weights takes int8 activations, so each uint8
    activation (scale s, zero point z) is written as int8 with the same scale and zero point
    z - 128: the same grid of real values, every code shifted by 128, the same arithmetic. Outputs
    are named by tap, like the ONNX outputs."""
    from .tflite_build import build_int8_mlp

    P = int8_params()
    layers = [{"name": _onnx_name(tap), "w": L["w"], "w_scale": L["sw"], "bias": L["bias"],
               "y_scale": L["sy"], "y_zero_point": L["zy"] - 128, "relu": L["relu"]}
              for tap, L in zip(INT8_TAPS, P["layers"])]
    return build_int8_mlp(layers, INT8_INPUTS["x_q"], P["input"]["scale"], P["input"]["zp"] - 128,
                          bare_outputs=True, input_name="x_q")


def tap_names(model: str) -> List[str]:
    return list(FLOAT_TAPS if model == "float" else INT8_TAPS)
