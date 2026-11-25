"""Constructed training runs for tests/test_spikes.py and scripts/spike_study.py: a clean run, and
three runs with one spike each whose cause is known by construction.

* numerics: batch S is a valid batch at 7.5 times the usual input scale. The net normalizes its
  input first, so in exact arithmetic the batch is as easy as any other, but in float16 the norm's
  x * x overflows and the output collapses.
* data: batch S has the targets of rows 3 and 9 replaced by large values (a corrupted batch).
* optimizer: the learning rate is 40 times its value at step S - 1 (a schedule bug).
"""

import os

import torch

import spike_spec
from truestar.spikes import SpikeWatch

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_spec.py")
TEACHER = torch.Generator().manual_seed(7)
W = torch.randn(16, 4, generator=TEACHER, dtype=torch.float64) * 0.8


def batch(n, scale=20.0, rows=16, seed=100):
    g = torch.Generator().manual_seed(seed + n)
    x = torch.randn(rows, 16, generator=g, dtype=torch.float64) * scale
    y = spike_spec.HandRMSNorm()(x) @ W
    return x.float(), y.float()


def run(run_dir, case="clean", steps=60, spike=45, dtype="float16", shadow_every=10, ckpt_every=20, data_seed=100,
        **kw):
    w = SpikeWatch(SPEC, run_dir, dtype=dtype, shadow_every=shadow_every, ckpt_every=ckpt_every, **kw)
    recs = []
    with w:
        for n in range(steps):
            x, y = batch(n, scale=150.0 if (case == "numerics" and n == spike) else 20.0, seed=data_seed)
            if case == "data" and n == spike:
                y = y.clone()
                y[3] = 40.0
                y[9] = -40.0
            lr = 0.05 * 40 if (case == "optimizer" and n == spike - 1) else None
            recs.append(w.step(x, y, lr=lr))
    return w, recs
