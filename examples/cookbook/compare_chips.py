"""Compare two chips you do not own: run one model under simulated device numerics.

    python examples/cookbook/compare_chips.py        # CPU, about 10 s

A simulated device (`sim:*`) is a profile of how a chip does float arithmetic: storage dtype,
accumulator, special values, op formulations, each rule labelled with its source
(docs/DEVICES.md). It runs on the CPU. Here one native classifier runs in float64 (the reference),
on textbook float16 (`sim:ieee-f16`, what correct float16 arithmetic gives), and on two device
profiles: the Apple Neural Engine (`sim:ane`) and an Arm Mali GPU through ncnn (`sim:mali-fp16`).
Nothing here runs on the devices; every number is a simulation of their documented arithmetic.
Recipe: docs/cookbook/compare_chips.md.
"""

import sys

import torch

from truestar import nn
from truestar.devsim import get_profile, simulate

D, H, C, N = 64, 256, 10, 512

net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.LayerNorm(H), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, C))
CHIPS = ["sim:ieee-f16", "sim:ane", "sim:mali-fp16"]


def cast(tree, dtype):
    return {k: cast(v, dtype) for k, v in tree.items()} if isinstance(tree, dict) else tree.to(dtype)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def main():
    p = net.init(seed=0)
    x = torch.randn(N, D, generator=torch.Generator().manual_seed(0)) * 4
    ref = net(cast(p, torch.float64), x.double())            # float64 reference
    outs = {}
    for chip in CHIPS:
        with simulate(chip):
            outs[chip] = net(cast(p, torch.float32), x)          # every op under the chip's arithmetic
    print(f"{'device':16} {'rel error vs float64':>22} {'top 1 changed':>14}  profile")
    for chip in CHIPS:
        flips = int((outs[chip].argmax(-1) != ref.argmax(-1)).sum())
        print(f"{chip:16} {rel(outs[chip], ref):22.1e} {flips:>9} of {N}  {get_profile(chip).description.split(':')[0]}")
    a, b = CHIPS[1], CHIPS[2]
    differ = int((outs[a].argmax(-1) != outs[b].argmax(-1)).sum())
    print(f"\n{a} against {b}: relative difference {rel(outs[a], outs[b]):.1e}, "
          f"{differ} of {N} predictions differ between the two chips")
    print("A chip's error well above sim:ieee-f16 is the device's arithmetic, not float16 itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
