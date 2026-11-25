"""Demo: a ResNet18 with a layer that is wrong only on CUDA.

    truestar check examples/fault_demo.py

Truestar should report CPU as OK, CUDA as DIVERGENT, blame layer3.1.bn2.1, and write a repro.
"""
import torch
import torchvision
from torch import nn

from truestar.testing import DeviceFault


def get_model():
    torch.manual_seed(0)
    m = torchvision.models.resnet18(num_classes=10)
    m.layer3[1].bn2 = nn.Sequential(m.layer3[1].bn2, DeviceFault("cuda", "rows", 0.5, from_row=2))
    return m


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(4, 3, 64, 64)
