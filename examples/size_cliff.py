"""A dot product of n ones, the reproducer of openvinotoolkit/openvino#22613.

    truestar check examples/size_cliff.py --backends cpu,sim:ieee-f16,sim:openvino-gpu-f16 --sweep n:512..8192

OpenVINO's GPU plugin computes the gemm in f16 by default; an f16 accumulator stops at 2048, since
2048 + 1 rounds back to 2048. The issue saw 2048.0 at n = 2049. The sweep reports the CPU and
textbook float16 (sim:ieee-f16, a float32 class accumulator) clean at every size, and
sim:openvino-gpu-f16 clean up to n = 2058, where 2048 is still within float16 tolerance of n,
and SUSPECT from n = 2059, naming the module (dot) and the op (aten.bmm.default). Exit code 1.
"""

import torch
from torch import nn


class Dot(nn.Module):
    def forward(self, a, b):
        return a @ b


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dot = Dot()

    def forward(self, a, b):
        return self.dot(a, b)


def get_model():
    return Net()


def get_inputs(n=2049):
    return torch.ones(1, 1, n), torch.ones(1, n, 1)
