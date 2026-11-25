"""Candidate for `truestar kernel`: a blocked RMSNorm, as a kernel author would tile it.

The sum of squares runs over whole blocks of 256 columns only, so a row length that is not a
multiple of 256 leaves its last partial block out of the mean. A single allclose on the example
input (1024 columns) passes it.
"""

import torch

BLOCK = 256


def kernel(x, w):
    n = x.shape[-1]
    xf = x.float()
    ss = torch.zeros(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device)
    for j in range(0, (n // BLOCK) * BLOCK, BLOCK):
        ss += xf[..., j:j + BLOCK].pow(2).sum(-1, keepdim=True)
    return (xf * torch.rsqrt(ss / n + 1e-6) * w.float()).to(x.dtype)
