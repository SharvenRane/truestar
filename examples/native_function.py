"""Example for the native front end: `@truestar.function`, typed shapes and real control flow.

    python examples/native_function.py

The RNN below loops over a symbolic length T and branches on tensor data. It is translated to
Truestar IR once, when the decorator runs; calls with other batch sizes and lengths reuse that
program. The script compares each call with the same math in eager PyTorch float64 and exits 1
if any call differs by more than 1e-10 relative error. A shape bug would be raised at decoration,
before any data exists (see `shape_bug_demo`). CPU only, random weights.
"""
import sys

import torch

import truestar
from truestar import Tensor, ops


@truestar.function
def rnn(x: Tensor["B", "T", 16], wx: Tensor[16, 32], wh: Tensor[32, 32]) -> Tensor["B", 32]:
    h = ops.zeros((x.shape[0], 32))
    for t in range(x.shape[1]):          # a loop over a symbolic length
        h = ops.tanh(x[:, t] @ wx + h @ wh)
    if h.sum() > 0:                      # a branch on tensor data
        h = ops.relu(h)
    return h


def eager(x, wx, wh):
    x, wx, wh = x.double(), wx.double(), wh.double()
    h = torch.zeros(x.shape[0], 32, dtype=torch.float64)
    for t in range(x.shape[1]):
        h = torch.tanh(x[:, t] @ wx + h @ wh)
    return torch.relu(h) if h.sum() > 0 else h


def shape_bug_demo():
    try:
        @truestar.function
        def buggy(h: Tensor["B", 32], w: Tensor[64, 32]) -> Tensor["B", 32]:
            return h @ w                 # inner dims differ: 32 vs 64
    except Exception as exc:             # raised at decoration, no data needed
        return str(exc).splitlines()[0]
    return None


def main():
    torch.manual_seed(0)
    wx, wh = 0.3 * torch.randn(16, 32), 0.3 * torch.randn(32, 32)
    print(rnn.signature)
    worst = 0.0
    for b, t in ((2, 3), (5, 9), (1, 1)):
        x = torch.randn(b, t, 16)
        got, want = rnn(x, wx, wh), eager(x, wx, wh)
        err = ((got - want).norm() / want.norm().clamp_min(1e-30)).item()
        worst = max(worst, err)
        print(f"B={b} T={t}: output {tuple(got.shape)}, relative error vs eager float64 {err:.1e}")
    print(f"shape bug caught at decoration: {shape_bug_demo()}")
    return 0 if worst <= 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
