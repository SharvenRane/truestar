"""Models and injected faults for the compiler and runtime backends (tests/test_stacks.py,
scripts/stack_study.py). Each fault is one real change to one stack, at one known module."""

from __future__ import annotations

import contextlib

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width=32, act="sigmoid"):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc = nn.Linear(width, width)
        self.act = act

    def forward(self, x):
        h = self.fc(self.norm(x))
        g = torch.sigmoid(h) if self.act == "sigmoid" else nn.functional.gelu(h)
        return x + g * x


class Net(nn.Module):
    """Input projection, four residual blocks, head. Blocks are identical in structure."""

    def __init__(self, act="sigmoid", blocks=4):
        super().__init__()
        self.inp = nn.Linear(16, 32)
        self.blocks = nn.Sequential(*[Block(32, act) for _ in range(blocks)])
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        return self.head(torch.relu(self.blocks(self.inp(x))))


class DetHead(nn.Module):
    def forward(self, x):
        return torch.linalg.det(x[:, :4, :4])


class DetNet(nn.Module):
    """A model with a determinant: ONNX Det has no CUDA kernel in ONNX Runtime."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.det = DetHead()

    def forward(self, x):
        y = self.fc(x)
        return y.sum(-1) + self.det(y).unsqueeze(-1)


class RMSNorm(nn.Module):
    """Hand written RMSNorm, as in many model repos: x.pow(2) overflows float16 once |x| > 256."""

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class RmsNet(nn.Module):
    """Activations around 400 into a hand written RMSNorm: exact in float32, Inf in float16."""

    def __init__(self, width=256):
        super().__init__()
        self.fc = nn.Linear(64, width)
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, 10)

    def forward(self, x):
        return self.head(self.norm(self.fc(x) * 400.0))


class GeluMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(64, 64)
        self.act2 = nn.GELU()
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.act2(self.fc2(self.act1(self.fc1(x)))))


class ShiftNorm(nn.Module):
    """LayerNorm on activations with a large mean (300): exact in float32, not in float16."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 8)

    def forward(self, x):
        return self.head(self.norm(self.fc(x) + 300.0))


# ----------------------------------------------------------------------------- injected faults


@contextlib.contextmanager
def inductor_wrong_sigmoid(scale: float = 1.05):
    """Inductor miscompile: its sigmoid lowering returns scale * sigmoid(x). The FX graph cache is
    off inside so neither a correct cached kernel hides the fault nor the faulty one leaks out."""
    import torch._inductor.lowering as L
    from torch._inductor import config as icfg

    aten = torch.ops.aten
    orig = L.lowerings[aten.sigmoid.default]
    L.lowerings[aten.sigmoid.default] = lambda t: L.lowerings[aten.mul.Tensor](orig(t), scale)
    try:
        with icfg.patch(fx_graph_cache=False):
            torch._dynamo.reset()
            yield
    finally:
        L.lowerings[aten.sigmoid.default] = orig
        torch._dynamo.reset()


def onnx_gelu_ignores_approximate():
    """Exporter fault: aten.gelu translated to the tanh approximation whatever `approximate`
    says (custom translation table of torch.onnx.export, the stack's compiler step)."""
    from onnxscript import opset18 as op

    def gelu(x, approximate: str = "none"):
        c = op.CastLike(0.7978845608, x)
        k = op.CastLike(0.044715, x)
        half = op.CastLike(0.5, x)
        one = op.CastLike(1.0, x)
        return half * x * (one + op.Tanh(c * (x + k * x * x * x)))

    return {"translations": {torch.ops.aten.gelu.default: gelu}}


def trt_norm_in_fp16():
    """TensorRT precision fault: normalization layers forced to FP16 in an FP32 engine, every
    other floating layer kept FP32 (what a per layer precision choice of the builder can do)."""
    import tensorrt as trt

    def hook(net, cfg):
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(net.num_layers):
            layer = net.get_layer(i)
            if layer.num_outputs == 0 or layer.get_output(0).dtype not in (trt.float32,
                                                                           trt.float16):
                continue
            if layer.type in (trt.LayerType.CONSTANT, trt.LayerType.SHUFFLE):
                continue
            layer.precision = (trt.float16 if layer.type == trt.LayerType.NORMALIZATION
                               else trt.float32)

    return {"layer_hook": hook}
