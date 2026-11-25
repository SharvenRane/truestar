"""`truestar parallel examples/parallel_model.py --layout tp=2,dp=2,ga=2,sp=1`: a small pre norm
transformer under tensor, data and sequence parallel with gradient accumulation, on 4 simulated ranks.

Expected: overall OK, exit 0 (every module output, gradient and updated weight within 1e-9 of the
single device step, both in float64).

With `--inject missing_tp_sync` (the gradient all reduce over tp of the sequence parallel norm
weights left out): TS684 at blocks.0.norm1 ("their sum over tp equals the single device
gradient"), exit 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(1.0 + 0.1 * torch.randn(d))

    def forward(self, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.weight


class Attention(nn.Module):
    def __init__(self, d, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        B, T, w = q.shape  # from q: under sequence parallel x is this rank's piece of the sequence
        h = w // self.head_dim  # heads on this rank
        q, k, v = (t.view(B, T, h, self.head_dim).transpose(1, 2) for t in (q, k, v))
        a = torch.softmax(q @ k.transpose(-1, -2) / self.head_dim ** 0.5, dim=-1) @ v
        return self.o(a.transpose(1, 2).reshape(B, T, w))


class Block(nn.Module):
    def __init__(self, d, ff, head_dim):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, head_dim)
        self.gate_up = nn.Linear(d, 2 * ff, bias=False)  # fused gate and up: two parts
        self.down = nn.Linear(ff, d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        g, u = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(g.contiguous()) * u)


class Model(nn.Module):
    def __init__(self, din=12, d=16, ff=24, head_dim=4, n=2, dout=6):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.blocks = nn.ModuleList(Block(d, ff, head_dim) for _ in range(n))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, dout)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def get_model():
    torch.manual_seed(0)
    return Model()


def get_inputs():
    return torch.randn(8, 4, 12, generator=torch.Generator().manual_seed(1))


def get_parallel_plan(layout):
    plan = {}
    for i in range(2):
        p = f"blocks.{i}."
        plan.update({p + "attn.q": "column", p + "attn.k": "column", p + "attn.v": "column", p + "attn.o": "row",
                     p + "gate_up": "column:2", p + "down": "row"})
    if layout.sp:
        plan.update({"inp": "scatter", "norm": "gather"})  # the residual stream is sequence parallel between
    return plan
