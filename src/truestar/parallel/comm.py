"""Collectives for simulated ranks, and the autograd functions tensor and sequence parallel layers use.

Two communicators share one interface:

* `ThreadComm`: every rank is a thread of this process; a collective is a barrier over the group, a
  read of every member's tensor, and a sum in rank order, so every rank computes bitwise the same
  result. No process start up, no ports.
* `GlooComm`: every rank is a separate CPU process with torch.distributed and the gloo backend,
  the same library calls a real job makes (reduce scatter is an all reduce and a slice, since gloo
  has no reduce scatter).

Both are simulations of a multi GPU job on one machine: the arithmetic of the layout is real, the
interconnect is not. Every call is logged with its purpose, so a finding can name the collective.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Dict, List, Sequence, Tuple

import torch


def file_init_method(path: str) -> str:
    """The `init_method` URL of a torch.distributed file store at `path`, right on every OS.

    torch adds the rank and world size to the URL and joins it again with urlunparse. On POSIX a
    URL with four slashes (`"file:///" + "/tmp/x"`) comes back as `file://tmp/x`, so `tmp` becomes
    the host and every rank waits on a file in a directory that does not exist until the store
    times out. A Windows path needs the third slash before its drive letter instead."""
    path = os.path.abspath(path)
    if sys.platform == "win32":
        return "file:///" + path.replace("\\", "/")
    return "file://" + path


class CollectiveAborted(RuntimeError):
    """Another rank failed, so this rank's collective can never complete."""


class UnevenSplit(ValueError):
    """TS1211: a reduce scatter over n ranks of a dim whose length n does not divide."""


def check_even_split(t: torch.Tensor, group: Sequence[int], dim: int, purpose: str) -> None:
    """Refuse a reduce scatter whose dim does not split into equal pieces: chunk would hand out unequal
    pieces and NCCL's reduce scatter needs equal ones, so every backend refuses it the same way."""
    n = len(group)
    if n > 1 and t.shape[dim] % n:
        raise UnevenSplit(f"[TS1211] reduce_scatter ({purpose}): dim {dim} has {t.shape[dim]} entries, which do not "
                          f"split into {n} equal pieces over ranks {tuple(group)}; pad it to a multiple of {n} "
                          "(FlatShard pads the flat state that way)")


class Comm:
    rank: int
    world: int

    def __init__(self) -> None:
        self.log: List[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]] = []

    def _note(self, op: str, purpose: str, group: Sequence[int], t: torch.Tensor) -> None:
        self.log.append((op, purpose, tuple(group), tuple(t.shape)))

    def all_reduce(self, t: torch.Tensor, group: Sequence[int], purpose: str) -> torch.Tensor:
        raise NotImplementedError

    def all_gather(self, t: torch.Tensor, group: Sequence[int], dim: int, purpose: str) -> torch.Tensor:
        raise NotImplementedError

    def reduce_scatter(self, t: torch.Tensor, group: Sequence[int], dim: int, purpose: str) -> torch.Tensor:
        check_even_split(t, group, dim, purpose)
        full = self.all_reduce(t, group, purpose)
        idx = list(group).index(self.rank)
        return full.chunk(len(group), dim=dim)[idx].contiguous()


class _Hub:
    """Shared state of the thread ranks: one barrier and one slot list per group."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.lock = threading.Lock()
        self.groups: Dict[Tuple[int, ...], Tuple[threading.Barrier, list]] = {}
        self.aborted = False

    def group(self, members: Tuple[int, ...]):
        with self.lock:
            if members not in self.groups:
                self.groups[members] = (threading.Barrier(len(members), timeout=self.timeout),
                                        [None] * len(members))
            return self.groups[members]

    def abort(self) -> None:
        with self.lock:
            self.aborted = True
            for b, _ in self.groups.values():
                b.abort()


class ThreadComm(Comm):
    def __init__(self, hub: _Hub, rank: int, world: int) -> None:
        super().__init__()
        self.hub, self.rank, self.world = hub, rank, world

    def _exchange(self, t: torch.Tensor, group: Sequence[int]) -> List[torch.Tensor]:
        members = tuple(group)
        if len(members) == 1:
            return [t]
        barrier, slots = self.hub.group(members)
        try:
            slots[members.index(self.rank)] = t.clone()  # the caller may overwrite t once we return
            barrier.wait()
            got = list(slots)
            barrier.wait()  # nobody overwrites a slot before every member has read it
        except threading.BrokenBarrierError:
            raise CollectiveAborted("a collective could not complete: another rank failed or never "
                                    "called it") from None
        return got

    def all_reduce(self, t, group, purpose):
        self._note("all_reduce", purpose, group, t)
        got = self._exchange(t.detach(), group)
        acc = got[0].clone()
        for g in got[1:]:
            acc = acc + g
        return acc

    def all_gather(self, t, group, dim, purpose):
        self._note("all_gather", purpose, group, t)
        return torch.cat(self._exchange(t.detach().contiguous(), group), dim=dim)


class GlooComm(Comm):
    def __init__(self, rank: int, world: int, groups: Dict[Tuple[int, ...], object]) -> None:
        super().__init__()
        self.rank, self.world, self.groups = rank, world, groups

    def all_reduce(self, t, group, purpose):
        import torch.distributed as dist

        self._note("all_reduce", purpose, group, t)
        out = t.detach().clone().contiguous()
        if len(group) > 1:
            dist.all_reduce(out, group=self.groups[tuple(group)])
        return out

    def all_gather(self, t, group, dim, purpose):
        import torch.distributed as dist

        self._note("all_gather", purpose, group, t)
        t = t.detach().contiguous()
        if len(group) == 1:
            return t.clone()
        bufs = [torch.empty_like(t) for _ in group]
        dist.all_gather(bufs, t, group=self.groups[tuple(group)])
        return torch.cat(bufs, dim=dim)


# ----------------------------------------------------------------------------- autograd functions
# Each pairs a forward collective with the backward collective its gradient needs.


class ToColumn(torch.autograd.Function):
    """Input of a column layer, no sequence parallel: forward identity, backward sum over tp (each
    rank's weight shard contributes part of the input gradient)."""

    @staticmethod
    def forward(ctx, x, comm, group, purpose, reduce_grad):
        ctx.comm, ctx.group, ctx.purpose, ctx.reduce_grad = comm, group, purpose, reduce_grad
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if ctx.reduce_grad:
            g = ctx.comm.all_reduce(g, ctx.group, ctx.purpose + " (backward)")
        return g, None, None, None, None


class FromRow(torch.autograd.Function):
    """Output of a row layer, no sequence parallel: forward sum over tp, backward identity."""

    @staticmethod
    def forward(ctx, x, comm, group, purpose):
        return comm.all_reduce(x, group, purpose)

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class GatherSeq(torch.autograd.Function):
    """Input of a column layer under sequence parallel: forward all gather along the sequence,
    backward reduce scatter (sum the partial input gradients, keep this rank's sequence piece)."""

    @staticmethod
    def forward(ctx, x, comm, group, dim, purpose, reduce_grad=True):
        ctx.comm, ctx.group, ctx.dim, ctx.purpose, ctx.reduce_grad = comm, group, dim, purpose, reduce_grad
        return comm.all_gather(x, group, dim, purpose)

    @staticmethod
    def backward(ctx, g):
        if ctx.reduce_grad:
            g = ctx.comm.reduce_scatter(g, ctx.group, ctx.dim, ctx.purpose + " (backward)")
        else:
            g = g.chunk(len(ctx.group), dim=ctx.dim)[list(ctx.group).index(ctx.comm.rank)].contiguous()
        return g, None, None, None, None, None


class ScatterSeqReduce(torch.autograd.Function):
    """Output of a row layer under sequence parallel: forward reduce scatter, backward all gather."""

    @staticmethod
    def forward(ctx, x, comm, group, dim, purpose):
        ctx.comm, ctx.group, ctx.dim, ctx.purpose = comm, group, dim, purpose
        return comm.reduce_scatter(x, group, dim, purpose)

    @staticmethod
    def backward(ctx, g):
        return ctx.comm.all_gather(g, ctx.group, ctx.dim, ctx.purpose + " (backward)"), None, None, None, None


class SplitSeq(torch.autograd.Function):
    """Entering the sequence parallel region: forward keep this rank's piece, backward all gather."""

    @staticmethod
    def forward(ctx, x, comm, group, dim, purpose):
        ctx.comm, ctx.group, ctx.dim, ctx.purpose = comm, group, dim, purpose
        idx = list(group).index(comm.rank)
        return x.chunk(len(group), dim=dim)[idx].contiguous()

    @staticmethod
    def backward(ctx, g):
        return ctx.comm.all_gather(g, ctx.group, ctx.dim, ctx.purpose + " (backward)"), None, None, None, None


class GatherOut(torch.autograd.Function):
    """Leaving the sequence parallel region: forward all gather, backward keep this rank's piece of a
    gradient every rank computed in full."""

    @staticmethod
    def forward(ctx, x, comm, group, dim, purpose):
        ctx.comm, ctx.group, ctx.dim = comm, group, dim
        return comm.all_gather(x, group, dim, purpose)

    @staticmethod
    def backward(ctx, g):
        idx = list(ctx.group).index(ctx.comm.rank)
        return g.chunk(len(ctx.group), dim=ctx.dim)[idx].contiguous(), None, None, None, None
