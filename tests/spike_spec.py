"""Spike spec file for tests/test_spikes.py and examples: a scale free regression net (a hand written
RMS norm first, so a batch scaled by any factor has the same float64 loss), SGD, mean squared error."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandRMSNorm(nn.Module):
    """x / sqrt(mean(x * x)): in float16, x * x overflows once |x| passes 256."""

    def forward(self, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)


class Net(nn.Module):
    def __init__(self, d=16, h=32, out=4):
        super().__init__()
        self.norm = HandRMSNorm()
        self.up = nn.Linear(d, h)
        self.act = nn.GELU()
        self.down = nn.Linear(h, out)

    def forward(self, x):
        return self.down(self.act(self.up(self.norm(x))))


def get_model():
    torch.manual_seed(0)
    return Net()


def get_loss():
    return F.mse_loss


def get_optimizer(params):
    return torch.optim.SGD(params, lr=0.05, momentum=0.9)
