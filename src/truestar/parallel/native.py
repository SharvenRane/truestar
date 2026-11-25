"""Data parallel and accumulated steps for the native training step (`truestar.train_step(...,
data_parallel=comm, accumulate=k)`, docs/TRAINING.md, Multi GPU).

The native step is one IR program per step on one device. With data parallel it becomes two
programs (fitting.Step): forward and backward, run once per micro batch, and the update (clipping
and the optimizer), run once. Between them, on the host:

1. the micro batch losses and gradients are summed in micro batch order;
2. the sums are summed over replicas, gradients in fixed buckets (in parameter order, filled up to
   `bucket_bytes`), each started as soon as it is flattened, so the transfer of one bucket overlaps
   the flattening of the next;
3. everything is divided once by the number of micro batches over all replicas.

With equal micro batches and a loss that is a mean over its rows this is the whole batch mean, and
every replica then runs the same update program on the same inputs, so replicas stay bitwise equal.
The backward pass itself is one executor call, so its compute cannot overlap the transfers here;
the layout runtime (gradsync.py) overlaps with torch autograd instead.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def _buckets(ts: Sequence[torch.Tensor], cap: int) -> List[List[int]]:
    out, cur, size = [], [], 0
    for i, t in enumerate(ts):
        cur.append(i)
        size += t.numel() * t.element_size()
        if size >= cap:
            out.append(cur)
            cur, size = [], 0
    if cur:
        out.append(cur)
    return out


def reduce_mean(comm, loss_sum: torch.Tensor, grad_sums: List[torch.Tensor], count: float,
                bucket_bytes: int = 25 << 20) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """(sum over replicas of loss_sum) / count and the same for every gradient, in fixed buckets."""
    if comm is None or comm.world == 1:
        return loss_sum / count, [g / count for g in grad_sums]
    group = tuple(range(comm.world))
    start = getattr(comm, "all_reduce_start", None)
    pend = []
    for b, idx in enumerate(_buckets(grad_sums, bucket_bytes)):
        flat = torch.cat([grad_sums[i].reshape(-1) for i in idx]) if len(idx) > 1 else grad_sums[idx[0]].reshape(-1)
        what = f"native step gradient bucket {b} ({flat.numel()} entries) over replicas"
        if start is None:
            pend.append((idx, comm.all_reduce(flat, group, what)))
        else:   # a fresh concatenation is donated: the communicator may reduce it in place
            pend.append((idx, start(flat, group, what, donate=len(idx) > 1)))
    loss = comm.all_reduce(loss_sum.reshape(1), group, "native step loss over replicas").reshape(loss_sum.shape)
    out: List[Optional[torch.Tensor]] = [None] * len(grad_sums)
    for idx, p in pend:
        red = p.wait() if hasattr(p, "wait") else p
        off = 0
        for i in idx:
            k = grad_sums[i].numel()
            out[i] = red[off:off + k].view_as(grad_sums[i]) / count
            off += k
    return loss / count, out


def fingerprint(params: Sequence[torch.Tensor]) -> torch.Tensor:
    """Two float64 numbers per tensor (sum and sum of magnitudes): cheap to all gather, and equal on
    two replicas only if they hold the same values in practice."""
    return torch.stack([torch.stack([p.double().sum(), p.double().abs().sum()]) for p in params]).reshape(-1)


def replicas_agree(comm, names: Sequence[str], params: Sequence[torch.Tensor], step: int) -> None:
    """Raise FitError TS824 when the replicas hold different parameters after `step`."""
    from ..fitting import FitError

    fp = fingerprint(params)
    got = comm.all_gather(fp.reshape(1, -1), tuple(range(comm.world)), 0, "replica fingerprint check")
    for r in range(1, got.shape[0]):
        bad = (got[r] != got[0]).nonzero()
        if bad.numel():
            i = int(bad[0]) // 2
            raise FitError(f"after step {step} replica {r} holds other values than replica 0 for {names[i]} (sum "
                           f"{float(got[r, 2 * i]):.17g} against {float(got[0, 2 * i]):.17g}); {bad.numel()} of "
                           f"{fp.numel()} fingerprint entries differ. The replicas no longer train one model: the "
                           "gradient sum or the update differs between ranks", "TS824")
