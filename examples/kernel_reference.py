"""Reference for `truestar kernel`: RMSNorm, written plainly in PyTorch.

    truestar kernel examples/kernel_reference.py examples/kernel_candidate.py

The candidate is a blocked RMSNorm that is right on the example input (1024 columns, a multiple of
its block) and wrong when the row length is not: the check moves every size by one and names the
failing shape (TS200 on d1=1025), where the bad values sit, and why no single ATen op is to blame.
"""

import torch

DTYPES = ["float16"]


def reference(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def get_inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(64, 1024, generator=g), torch.rand(1024, generator=g) + 0.5
