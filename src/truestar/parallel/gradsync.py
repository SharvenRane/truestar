"""Gradient sum over data parallel replicas in fixed buckets, started while the backward pass runs.

Buckets are fixed when the sync is built: parameters in reverse registration order (roughly the
order their gradients finish in backward), filled up to `bucket_bytes` (the first one up to
`first_bucket_bytes` when given: a small first bucket starts the first reduction early in backward). Every rank builds the same
buckets from the same model, and bucket i is started only after buckets 0 to i-1 have started, so
the collectives run in the same order on every rank whatever order the gradients finish in (a
rank that started bucket 3 before bucket 2 while its peer did the opposite would sum the wrong
tensors or hang).

With `overlap=True` a bucket starts from a hook that fires when the last of its gradients is
accumulated (`register_post_accumulate_grad_hook`), so the reduction of late layers runs while the
backward pass is still computing early ones. With gradient accumulation the hooks are disarmed for
the first micro batches (`no_sync()`); only the backward of the last one starts buckets. Without a
communicator that can start a reduction asynchronously (the simulated ranks), or with
`overlap=False`, every bucket is reduced after the backward pass, in bucket order.

`finish()` waits for every bucket in order and writes the sums back into `.grad`. A parameter with
no gradient contributes zeros to its bucket and keeps `grad=None`.

Each bucket owns one flat buffer, padded to a multiple of the group size and kept from step to step:
the gradients are copied into it by one concatenation, the communicator reduces it without a copy
of its own (`donate=True`), and the sums go back by one multi tensor copy. The padding (zeros) lets
the fixed order sum cut the buffer into equal pieces without another copy.

Written from the definitions above; no code from any data parallel wrapper is used.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import List, Optional, Sequence, Tuple

import torch


class GradientSync:
    def __init__(self, named_params: Sequence[Tuple[str, torch.nn.Parameter]], comm, group: Sequence[int], *,
                 bucket_bytes: int = 25 << 20, overlap: bool = True, purpose: str = "gradient bucket all reduce over dp",
                 first_bucket_bytes: Optional[int] = None):
        self.comm, self.group, self.purpose = comm, tuple(group), purpose
        self.params = [(n, p) for n, p in named_params if p.requires_grad]
        self.can_async = hasattr(comm, "all_reduce_start") and len(self.group) > 1
        self.overlap = bool(overlap) and self.can_async
        self.buckets: List[List[int]] = []
        cur, size = [], 0
        for i in reversed(range(len(self.params))):
            p = self.params[i][1]
            if cur and (p.dtype != self.params[cur[0]][1].dtype or p.device != self.params[cur[0]][1].device):
                self.buckets.append(cur)        # one dtype and device per flat buffer
                cur, size = [], 0
            cur.append(i)
            size += p.numel() * p.element_size()
            if size >= (bucket_bytes if self.buckets or first_bucket_bytes is None else first_bucket_bytes):
                self.buckets.append(cur)
                cur, size = [], 0
        if cur:
            self.buckets.append(cur)
        self.bucket_of = {}
        for b, idx in enumerate(self.buckets):
            for i in idx:
                self.bucket_of[i] = b
        self._bufs: List[Optional[torch.Tensor]] = [None] * len(self.buckets)
        self._sizes = [sum(self.params[i][1].numel() for i in b) for b in self.buckets]
        self._lock = threading.Lock()
        self._armed = True
        self._reset()
        self._handles = []
        if self.overlap:
            for i, (_, p) in enumerate(self.params):
                self._handles.append(p.register_post_accumulate_grad_hook(lambda t, i=i: self._ready(i)))
        self.launch_times: List[float] = []   # perf_counter when each bucket started, for overlap reports

    def _reset(self) -> None:
        self._left = [len(b) for b in self.buckets]
        self._started: List[Optional[object]] = [None] * len(self.buckets)
        self._next = 0

    @contextlib.contextmanager
    def no_sync(self):
        """Backward passes inside the block only accumulate (the first micro batches)."""
        prev, self._armed = self._armed, False
        try:
            yield
        finally:
            self._armed = prev

    def _flat(self, b: int) -> torch.Tensor:
        """The bucket's gradients, concatenated into its buffer (zeros for a missing gradient and for
        the padding)."""
        size, n = self._sizes[b], max(1, len(self.group))
        first = self.params[self.buckets[b][0]][1]
        buf = self._bufs[b]
        if buf is None:
            buf = torch.zeros(-(-size // n) * n, dtype=first.dtype, device=first.device)
            self._bufs[b] = buf
        parts = []
        for i in self.buckets[b]:
            p = self.params[i][1]
            parts.append(p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=p.dtype,
                                                                                   device=p.device))
        torch.cat(parts, out=buf[:size])
        return buf

    def _start(self, b: int) -> None:
        flat = self._flat(b)
        what = f"{self.purpose}, bucket {b} of {len(self.buckets)} ({self._sizes[b]} entries)"
        if self.can_async:
            self._started[b] = self.comm.all_reduce_start(flat, self.group, what, donate=True)
        else:
            self._started[b] = self.comm.all_reduce(flat, self.group, what)
        self.launch_times.append(time.perf_counter())

    def _ready(self, i: int) -> None:
        if not self._armed:
            return
        with self._lock:
            b = self.bucket_of[i]
            self._left[b] -= 1
            while self._next < len(self.buckets) and self._left[self._next] <= 0:
                self._start(self._next)
                self._next += 1

    def finish(self, scale: Optional[float] = None) -> None:
        """Start what has not started (in bucket order), wait for every bucket, write the sums back.
        `scale` multiplies every sum (1 / dp for a mean)."""
        with self._lock:
            while self._next < len(self.buckets):
                self._start(self._next)
                self._next += 1
        for b, pend in enumerate(self._started):
            red = pend.wait() if hasattr(pend, "wait") else pend
            if scale is not None:
                red = red * scale
            off, dst, src = 0, [], []
            for i in self.buckets[b]:
                p = self.params[i][1]
                k = p.numel()
                if p.grad is not None:
                    dst.append(p.grad)
                    src.append(red[off:off + k].view_as(p.grad))
                off += k
            if dst:
                torch._foreach_copy_(dst, src)
        self._reset()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
