"""Reference model for the `truestar port` example (the candidate is port_candidate.py).

    truestar port examples/port_reference.py examples/port_candidate.py
    truestar port examples/port_reference.py examples/port_candidate.py --format agent

A two block MLP with RMSNorm, written with Hugging Face style names. Random weights, CPU only.
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.input_layernorm = RMSNorm(dim)
        self.up_proj = nn.Linear(dim, 4 * dim)
        self.down_proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.down_proj(torch.relu(self.up_proj(self.input_layernorm(x))))


class RefModel(nn.Module):
    def __init__(self, dim=16, depth=2):
        super().__init__()
        self.layers = nn.ModuleList(Block(dim) for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_model():
    torch.manual_seed(0)
    m = RefModel()
    with torch.no_grad():  # norm weights that are not all ones, so they match by value
        for p in m.parameters():
            if p.dim() == 1 and p.shape[0] == 16:
                p.add_(0.1 * torch.randn_like(p))
    return m.eval()


def get_inputs():
    torch.manual_seed(1)
    return 0.01 * torch.randn(4, 16)  # small activations, where eps matters
