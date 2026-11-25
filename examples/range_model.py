"""`truestar range examples/range_model.py --target fp16`: a residual stream that outgrows float16.

A small decoder style stack (pre and post RMS norms around each MLP, as in Gemma) where a large post
norm weight on one channel adds about 9,000 to the residual every block, the pattern behind
onnxruntime#26732 (Gemma 3 unusable in float16 on WebGPU). The float64 peak is about 73,500, past
float16's 65,504.

Expected: status REMEDIED, exit 1. The first site is the residual add of block 9 (margin about -0.17
bits; block 8 is a near overflow); the chosen remedy is an exact power of two rescale of the residual
stream, verified by emulating float16 (error back to the float16 precision floor, no flips).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, ff, step):
        super().__init__()
        self.pre = nn.Parameter(torch.ones(d))
        self.post = nn.Parameter(torch.ones(d))
        self.gate = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)
        with torch.no_grad():
            self.down.weight[0] = self.down.weight[0].abs() * 4  # a positive outlier channel
            self.post[0] = step

    def forward(self, x):
        h = F.rms_norm(x, (x.shape[-1],), weight=self.pre)
        g = self.gate(h)
        y = self.down(F.gelu(g, approximate="tanh") * g)
        return x + F.rms_norm(y, (y.shape[-1],), weight=self.post)


class Stack(nn.Module):
    def __init__(self, d=64, ff=128, n=12, classes=32):
        super().__init__()
        self.inp = nn.Linear(16, d)
        self.blocks = nn.ModuleList(Block(d, ff, 1200.0 if 2 <= i < 10 else 1.0) for i in range(n))
        self.norm = nn.Parameter(torch.ones(d))
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.head(F.rms_norm(x, (x.shape[-1],), weight=self.norm))


def get_model():
    torch.manual_seed(0)
    return Stack().eval()


def get_inputs():
    torch.manual_seed(1)
    return (torch.randn(8, 16),)


def get_calibration():
    torch.manual_seed(2)
    return [(torch.randn(8, 16),)]
