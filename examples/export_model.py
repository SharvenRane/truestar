"""Example for `truestar export` and `truestar run`: a verified artifact that runs without this file.

    truestar export examples/export_model.py -o small_cnn.tsx
    python examples/export_model.py inputs.pt          # writes a batch of 3 example inputs
    truestar run small_cnn.tsx inputs.pt -o outputs.pt
    truestar run small_cnn.tsx --info

The export checks the artifact against PyTorch in float64 at the example batch (2) and at a
second batch (7) and refuses to write anything if they disagree. The batch dim stays symbolic, so
`run` accepts batch 3 although the export saw 2 and 7. Random weights, CPU only.
"""
import sys

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def get_model():
    torch.manual_seed(0)
    return SmallCNN().eval()


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(2, 3, 16, 16)


if __name__ == "__main__":
    torch.manual_seed(2)
    torch.save(torch.randn(3, 3, 16, 16), sys.argv[1] if len(sys.argv) > 1 else "inputs.pt")
