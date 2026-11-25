"""Example for `truestar invariance`: a row whose result depends on the rest of the batch.

    truestar invariance examples/invariance_model.py --backends cpu
    truestar invariance examples/invariance_model.py --backends cpu --format agent

`Standardize` centers features with the mean over the batch dim, a common inference bug (batch
statistics where fixed statistics belong). A row then gets a different answer depending on which
other rows share its batch. The check names `norm` and the `aten.mean.dim` op inside it (TS052),
and reports whatever float level batch variance the stock matmul kernels show on this machine.
Random weights, CPU only. Exit code 1.
"""
import torch
from torch import nn


class Standardize(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=0, keepdim=True)  # the bug: statistics over the batch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Standardize()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.norm(x))))


def get_model():
    torch.manual_seed(0)
    return Net()


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(8, 16)
