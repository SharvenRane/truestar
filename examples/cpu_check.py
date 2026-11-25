"""CPU only example: a small conv net with a transformer block, no torchvision needed.

    truestar check examples/cpu_check.py --format agent

On a machine with only a CPU this compares CPU float32 against the float64 CPU reference and
should report OK with exit code 0. CI uses it to prove a fresh `pip install truestar` works.
"""
import torch
from torch import nn


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.block = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32,
                                                dropout=0.0, batch_first=True)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem(x)                      # (B, 16, H, W)
        x = x.flatten(2).transpose(1, 2)      # (B, H*W, 16)
        x = self.block(x)
        return self.head(x.mean(dim=1))


def get_model():
    torch.manual_seed(0)
    return SmallNet().eval()


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(4, 3, 16, 16)
