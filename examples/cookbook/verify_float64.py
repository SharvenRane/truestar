"""Verify a model against float64: every compiled kernel, every region and the whole program.

    python examples/cookbook/verify_float64.py        # CPU, about 10 s

A native model compiled for the CPU in float32 is checked on its first call against Truestar's
float64 reference interpreter, at three levels (each kernel, each fused region, the whole program),
before its result is returned. This script prints what was compiled and verified, then repeats the
comparison by hand: the same layers run eagerly in float64 on the same weights.
Recipe: docs/cookbook/verify_float64.md.
"""

import sys

import torch

import truestar
from truestar import Tensor, nn
from truestar.compiler import CompiledExecutor

net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 64), nn.SiLU(),
                    nn.Linear(64, 10))
ex = CompiledExecutor("cpu", dtype="float32")


@truestar.native(executor=ex)
def forward(p: net.spec, x: Tensor["B", 32]) -> Tensor["B", 10]:
    return net(p, x)


def to64(tree):
    return {k: to64(v) for k, v in tree.items()} if isinstance(tree, dict) else tree.double()


def main():
    p = net.init(seed=0)
    x = torch.randn(16, 32, generator=torch.Generator().manual_seed(0))
    y = forward(p, x)                      # compiled, then checked against float64, then returned
    cp = ex.last
    print(cp.report)
    print(f"\nverified on the first call: {cp.verified}; end to end error vs float64 {cp.max_error:.1e}")

    ref = net(to64(p), x.double())         # the same layers, eagerly, in float64
    rel = float((y.double() - ref).norm() / ref.norm())
    print(f"by hand: relative L2 error of the float32 result against eager float64: {rel:.1e}")

    y2 = forward(p, torch.randn(5, 32))    # a new batch size: the same program, no recompile
    print(f"batch 5 ran on the same compiled program (compiles: {forward.compile_count}), output {tuple(y2.shape)}")
    ok = cp.verified and rel < 1e-5
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
