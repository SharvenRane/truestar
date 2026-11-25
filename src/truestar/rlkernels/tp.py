"""Tensor parallel SIMULATION on one device, with a fixed all reduce order.

This is not distributed code. All ranks run one after another in the calling process on one
GPU, and the "all reduce" is a function that adds a Python list of tensors. What it shows is the
arithmetic a real TP 2 run has to reproduce for its logprobs to equal TP 1 bit for bit:

* A column parallel layer (each rank owns some output features) changes no reduction: every
  output element is computed from the same K values in the same order as at TP 1.
* A row parallel layer (each rank owns a contiguous 1/tp of the input features K) splits the
  reduction. The RL matmul already reduces K as `SEGMENTS` equal segments added as a balanced
  binary tree, so rank r's partial is exactly one subtree when tp divides `SEGMENTS`.
  `all_reduce` adds the ranks' float32 partials in the same binary tree, in rank order, which
  rebuilds the TP 1 tree. A ring or a reduction in another order or another dtype would not.

A real multi GPU run would need a collective with this fixed order and float32 payloads (for
example an all gather of the partials followed by this tree on every rank); that was not run
here (one GPU).
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from . import kernels as K


def shard_rows(w: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Rank `rank`'s contiguous block of output features of a [out, in] weight."""
    n = w.shape[0] // tp
    return w[rank * n:(rank + 1) * n]


def shard_cols(w: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Rank `rank`'s contiguous block of input features of a [out, in] weight."""
    n = w.shape[1] // tp
    return w[:, rank * n:(rank + 1) * n]


def all_reduce(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum of the ranks' partials as a balanced binary tree in rank order: ((r0 + r1) + (r2 + r3))."""
    if len(parts) & (len(parts) - 1):
        raise ValueError(f"all_reduce needs a power of two rank count, got {len(parts)}")
    return K.tree_sum(torch.stack(list(parts)), 0)


def all_gather(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(parts), dim=-1)


def row_parallel_linear(xs: Sequence[torch.Tensor], weight: torch.Tensor, tp: int) -> torch.Tensor:
    """y = x @ weight.T where rank r holds x's r-th block of features `xs[r]` and the matching
    columns of `weight`. Each rank reduces its K block in SEGMENTS / tp segments; the float32
    partials are all reduced in a fixed tree, then cast once."""
    if K.SEGMENTS % tp:
        raise ValueError(f"tp {tp} must divide the K segment count {K.SEGMENTS}")
    local: List[torch.Tensor] = []
    for r, x in enumerate(xs):
        w = shard_cols(weight, tp, r)
        local.append(K.matmul_partials(x, w.t(), K.SEGMENTS // tp))
    return all_reduce(local).to(weight.dtype)
