"""A float16 training run with one loss spike, then `truestar replay RUN --step 45`.

    python examples/spike_demo.py run_dir          # trains 50 steps under truestar.spikes.SpikeWatch
    truestar replay run_dir --step 45              # explains the spike at step 45

This file is also the spike spec (get_model, get_loss, get_optimizer). The net normalizes its input
first, so batch 45, a valid batch at 7.5 times the usual scale, is as easy as any other in exact
arithmetic, but in float16 the norm's x * x overflows.

Expected: the run logs a spike at step 45 with a TS691 shadow flag; the replay prints
`verdict: NUMERICS [TS692] the spike vanishes in float64` with `first split norm`, exit 1.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandRMSNorm(nn.Module):
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


def main(run_dir):
    from truestar.spikes import SpikeWatch

    W = torch.randn(16, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float64) * 0.8
    with SpikeWatch(os.path.abspath(__file__), run_dir, dtype="float16", shadow_every=10, ckpt_every=20) as w:
        for n in range(50):
            g = torch.Generator().manual_seed(100 + n)
            x = torch.randn(16, 16, generator=g, dtype=torch.float64) * (150.0 if n == 45 else 20.0)
            y = HandRMSNorm()(x) @ W
            rec = w.step(x.float(), y.float())
            if rec["spike"]:
                print(f"step {n}: loss {rec['loss']:.4g} spike, shadow {rec.get('code', 'ok')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spike_run")
