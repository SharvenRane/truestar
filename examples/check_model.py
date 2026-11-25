"""Example for `truestar check`: a hand written layer norm that loses its precision in float32.

    truestar check examples/check_model.py --backends cpu
    truestar check examples/check_model.py --backends cpu --format agent

`OnePassNorm` computes the variance as mean(x * x) - mean(x) ** 2. The features sit near 3000
with a spread of about 1, so both terms are near 9e6, where float32 resolves steps of 1: the
difference, the variance, is mostly rounding error. float64 has digits to spare, so the float64
CPU reference is right and every float32 run is wrong. Truestar blames `norm` with this file and
line, and names the op where the digits are lost. Random weights, CPU by default (add
`--backends cpu,cuda` on a GPU machine), a few seconds. Exit code 1.
"""
import torch
from torch import nn


class OnePassNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x * x).mean(-1, keepdim=True) - mean * mean  # the bug: cancellation
        return (x - mean) * torch.rsqrt(var.clamp_min(0) + 1e-5)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = OnePassNorm()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.norm(x))))


def get_model():
    torch.manual_seed(0)
    return Net().eval()


def get_inputs():
    torch.manual_seed(1)
    return 3000 + torch.randn(8, 32)  # for example raw sensor readings with a large offset
