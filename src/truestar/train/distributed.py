"""Data parallel gradient averaging inside the training step program.

Every rank runs the same step program on its own shard of the batch. The program flattens the loss
and every gradient into one bucket, averages the bucket across ranks with one ``all_reduce`` node,
and splits it back before clipping and the optimizer update. One collective per step, in the same
place on every rank, so ranks cannot disagree on collective order by construction. Because the
bucket also carries the loss, every rank reports the global mean loss.

The ``all_reduce`` kernel calls ``torch.distributed.all_reduce`` at run time. Run training inside
``truestar.diag.collectives.record_collectives`` and a hang (a rank that runs out of data, crashes,
or takes a different number of steps) is diagnosed with the rank and collective sequence number.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ..ir.core import IRError, Value


def _numel(v: Value) -> int:
    n = 1
    for d in v.type.shape:
        if not isinstance(d, int):
            raise IRError(f"data parallel bucket: {v.ref} has symbolic shape {v.type}; gradients of "
                          "trainable params always have fixed shapes, so this is an internal error")
        n *= d
    return n


def all_reduce_mean(e, values: Sequence[Value]) -> List[Value]:
    """Average ``values`` across ranks with one ``all_reduce`` of a flat bucket."""
    e.source = "truestar.train data parallel gradient bucket"
    flats, sizes = [], []
    for v in values:
        n = _numel(v)
        sizes.append(n)
        flats.append(e.op("reshape", v, shape=[n], name=f"{v.name}_flat"))
    bucket = flats[0] if len(flats) == 1 else e.op("cat", *flats, dim=0, name="grad_bucket")
    e.source = "truestar.train data parallel all_reduce (mean over ranks)"
    red = e.op("all_reduce", bucket, reduce="mean", name="grad_bucket_mean")
    e.source = "truestar.train data parallel gradient bucket"
    outs, off = [], 0
    for v, n in zip(values, sizes):
        part = e.op("slice", red, dim=0, start=off, end=off + n, step=1, name=f"{v.name}_mean_flat")
        part = e.op("reshape", part, shape=list(v.type.shape), name=f"{v.name}_mean")
        if part.type.dtype != v.type.dtype:
            part = e.op("cast", part, dtype=v.type.dtype)
        outs.append(part)
        off += n
    return outs
