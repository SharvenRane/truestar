"""Small models for tests/test_parallel.py and examples/parallel_model.py: a pre norm transformer with
attention (head count derived from the local width, so it runs on a tensor parallel shard), a fused
gate and up projection, and plans for tensor and sequence parallel."""

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
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        B, T, _ = q.shape  # from q, not x: under sequence parallel x is this rank's sequence piece
        h = q.shape[-1] // self.head_dim  # heads on this rank
        q, k, v = (t.view(B, T, h, self.head_dim).transpose(1, 2) for t in (q, k, v))
        a = torch.softmax(q @ k.transpose(-1, -2) / self.head_dim ** 0.5, dim=-1) @ v
        return self.o(a.transpose(1, 2).reshape(B, T, h * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, d, ff, act=F.silu):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * ff, bias=False)  # gate and up fused: two parts
        self.down = nn.Linear(ff, d)
        self.act = act

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.act(g.contiguous()) * u)


class Block(nn.Module):
    def __init__(self, d, ff, head_dim, act=F.silu):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, head_dim)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, ff, act)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Tiny(nn.Module):
    def __init__(self, din=12, d=16, ff=24, head_dim=4, n=2, dout=6, act=F.silu):
        super().__init__()
        self.inp = nn.Linear(din, d)
        self.blocks = nn.ModuleList(Block(d, ff, head_dim, act) for _ in range(n))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, dout)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def tiny_plan(n=2, sp=False):
    plan = {}
    for i in range(n):
        p = f"blocks.{i}."
        plan.update({p + "attn.q": "column", p + "attn.k": "column", p + "attn.v": "column", p + "attn.o": "row",
                     p + "mlp.gate_up": "column:2", p + "mlp.down": "row"})
    if sp:
        plan.update({"inp": "scatter", "norm": "gather"})
    return plan


class MLP(nn.Module):
    def __init__(self, din=10, h=32, dout=5):
        super().__init__()
        self.up = nn.Linear(din, h)
        self.act = nn.GELU()
        self.down = nn.Linear(h, dout)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class Classifier(nn.Module):
    """Token classifier for the padded cross entropy case (ignore_index rows)."""

    def __init__(self, din=8, h=16, classes=7):
        super().__init__()
        self.up = nn.Linear(din, h)
        self.down = nn.Linear(h, classes)

    def forward(self, x):
        return self.down(torch.tanh(self.up(x)))


def tiny(seed=0):
    torch.manual_seed(seed)
    return Tiny()


def tiny_inputs(seed=1, B=8, T=4):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 12, generator=g)


def padded_targets(B=8, T=6, classes=7, seed=2):
    """Token labels with padding (-100) concentrated in the first rows, so micro batches and replicas
    hold different token counts."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, classes, (B, T), generator=g)
    y[0, 1:] = -100
    y[1, 3:] = -100
    y[2, 5:] = -100
    return y
