"""Example for `truestar record` and `truestar diff`: find what an "upgrade" changed.

    truestar record examples/record_diff_model.py -o before.tsrec --backend cpu --ops all
    truestar diff before.tsrec --replay                                   # same environment: identical
    TRUESTAR_EXAMPLE_UPGRADE=1 truestar diff before.tsrec --replay        # simulated upgrade: divergent

A real upgrade changes torch, CUDA, cuDNN, Triton or the driver. Here the environment variable
TRUESTAR_EXAMPLE_UPGRADE stands in for it: when set, `Norm` computes its variance with a biased
divisor, as a changed kernel might. The diff names `norm#0`, the ATen op inside it and writes an
op repro. Random weights, CPU only. diff exits 0 when the replay is identical, 1 when divergent.
"""
import os

import torch
from torch import nn


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        upgraded = os.environ.get("TRUESTAR_EXAMPLE_UPGRADE") == "1"
        var = x.var(-1, keepdim=True, unbiased=not upgraded)
        return (x - x.mean(-1, keepdim=True)) / torch.sqrt(var + 1e-5) * self.weight


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.norm = Norm(32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.norm(torch.relu(self.fc1(x))))


def get_model():
    torch.manual_seed(0)
    return Net().eval()


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(8, 16)
