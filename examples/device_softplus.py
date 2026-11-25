"""A small classifier whose softplus sees inputs above 10.4 (coremltools#2687).

    truestar check examples/device_softplus.py --backends cpu,sim:ieee-f16,sim:ane

The CPU and textbook float16 (sim:ieee-f16) are clean. The simulated Apple Neural Engine
(sim:ane) is DIVERGENT: the first divergent module is act (Act), the op aten.softplus.default,
because the ANE returns 0 for softplus once x passes ln(2^15) = 10.397. Exit code 1.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Act(nn.Module):
    def forward(self, x):
        return F.softplus(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = Act()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def get_model():
    torch.manual_seed(0)
    m = Net()
    with torch.no_grad():
        m.fc1.weight.mul_(8)  # pre activations up to about 20
    return m


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(64, 16)
