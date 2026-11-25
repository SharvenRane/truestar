"""Example for `truestar check --backward`: a forward that is right with a backward that is wrong.

    truestar check examples/backward_model.py --backward --backends cpu
    truestar check examples/backward_model.py --backends cpu            # the forward check: OK

`ScaledTanh` is a custom autograd.Function whose hand written backward forgets the square in
1 - tanh(x)^2, but only for float32 (a "fast" backward kernel). Its forward is exact, so the
forward check reports OK and only the backward check blames `act`. Random weights, CPU only.
Exit code 1 with --backward, 0 without.
"""
import torch
from torch import nn


class _ScaledTanhFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        if y.dtype == torch.float32:
            return g * (1 - y)  # the bug: should be 1 - y * y
        return g * (1 - y * y)


class ScaledTanh(nn.Module):
    def forward(self, x):
        return _ScaledTanhFn.apply(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = ScaledTanh()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def get_model():
    torch.manual_seed(0)
    return Net()


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(8, 16)
