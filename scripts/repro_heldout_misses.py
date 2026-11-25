"""CPU stand ins for the held out derived bound misses (docs/KERNEL_CHECK.md, held out section).

The held out kernels behind the misses sum in float32 one term after another (one thread per row
of a cumsum or a sum) or compute an instance norm in two passes. On the constant tensor family
every step of a sequential sum adds the same value (or zero, under a mask), so its rounding
errors repeat instead of averaging out; instance norm (aten.native_batch_norm on the instance view) had no reduction model.

    python -I scripts/repro_heldout_misses.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from truestar.kernelcheck import check_kernel, family_variants  # noqa: E402
from truestar.tolerance import clear_cache, derive  # noqa: E402


def seq_cumsum(x):
    """One thread per row, float32 running sum (KernelBench level 1 task 89 samples)."""
    out = torch.empty_like(x)
    s = torch.zeros(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[1]):
        s = s + x[:, i]
        out[:, i] = s
    return out


def seq_sum(x):
    return seq_cumsum(x)[:, -1:]


def seq_mean(x):
    return seq_sum(x) / x.shape[1]


def inorm_two_pass(x):
    mu = x.mean((2, 3), keepdim=True)
    var = ((x - mu) ** 2).mean((2, 3), keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5)


def const(x):
    return next(v.inputs for v in family_variants((x,)) if v.name.startswith("constant"))


def cases():
    g = torch.Generator().manual_seed(0)
    r = torch.randn(64, 4000, generator=g)
    return {
        "sum, constant tensor": (lambda x: x.sum(1, keepdim=True), seq_sum, const(r)),
        "mean, constant tensor": (lambda x: x.mean(1, keepdim=True), seq_mean, const(r + 0.3)),
        "cumsum, constant tensor": (lambda x: torch.cumsum(x, 1), seq_cumsum, const(r - 0.2)),
        "masked cumsum, constant tensor": (lambda x, m: torch.cumsum(x * m, 1),
                                           lambda x, m: seq_cumsum(x * m),
                                           const(r - 0.2) + (torch.rand(64, 4000, generator=g)
                                                             > 0.5,)),
        "instance norm": (lambda x: torch.nn.functional.instance_norm(x), inorm_two_pass,
                          (torch.randn(8, 16, 32, 32, generator=g) * 3 + 1,)),
    }


def main():
    for name, (ref, cand, x) in cases().items():
        clear_cache()
        out = cand(*x)
        tol = derive(ref, x, like=out)
        rep = tol.check(out)
        print(f"{name:26s} ok={rep.ok} worst={rep.worst.worst_ratio:.2f}x of the bound")
        rk = check_kernel(ref, cand, x, drill_ops=False, shapes=False, strides=False)
        print(f"{'':26s} check_kernel over the families: {rk.status} {rk.codes}")


if __name__ == "__main__":
    main()
