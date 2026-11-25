"""Candidate for the `truestar port` example: a hand port of port_reference.py with a bug.

    truestar port examples/port_reference.py examples/port_candidate.py

Other module names (blocks.N.norm, .w1, .w2) and the same weights, copied by an explicit mapping
the way a converter script does. The bug: `Norm` hardcodes eps 1e-6 where the reference uses
1e-5. The port check maps modules by weight content, names `blocks.0.norm` and the
`aten.add.Tensor` whose constant differs. Exit code 1.
"""
import torch
from torch import nn


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight  # the bug


class PortBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = Norm(dim)
        self.w1 = nn.Linear(dim, 4 * dim)
        self.w2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.w2(torch.relu(self.w1(self.norm(x))))


class PortModel(nn.Module):
    def __init__(self, dim=16, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList(PortBlock(dim) for _ in range(depth))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def get_model(reference):
    port = PortModel()
    names = {"input_layernorm": "norm", "up_proj": "w1", "down_proj": "w2"}
    state = {}
    for key, value in reference.state_dict().items():
        _, i, mod, leaf = key.split(".")  # layers.0.up_proj.weight
        state[f"blocks.{i}.{names[mod]}.{leaf}"] = value
    port.load_state_dict(state)
    return port.eval()
