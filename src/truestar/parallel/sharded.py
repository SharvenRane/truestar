"""State sharded over data parallel replicas: each replica keeps a slice of the optimizer state (and
optionally of the parameters) instead of a full copy.

All trained parameters of a rank are laid out in one flat vector, in registration order, padded with
zeros to a multiple of the group size n. Replica i owns entries [i * per, (i + 1) * per). One step:

1. the flat gradient is reduce scattered over the group: replica i receives the sum over replicas
   of its own slice only;
2. the squared norm of the slice is summed over the group (and over tp, by the caller) for clipping;
3. the optimizer steps one flat parameter, the slice, with its own state; SGD and AdamW act per
   element, so the slice steps exactly as the same entries of the whole tensors would;
4. the updated slices are all gathered and written back into the parameters (level 1), or kept as
   the only copy until the next step gathers them for compute (level 2).

Optimizer state per replica is 1/n of the unsharded state; with level 2 the parameters between
steps are too. Written from the description above, no code from any sharding library.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


class FlatShard:
    def __init__(self, params: Sequence[Tuple[str, torch.nn.Parameter]], group_rank: int, group_size: int):
        self.params = [(n, p) for n, p in params if p.requires_grad]
        self.sizes = [p.numel() for _, p in self.params]
        self.shapes = [tuple(p.shape) for _, p in self.params]
        self.total = sum(self.sizes)
        self.n = group_size
        self.per = -(-self.total // group_size)
        self.padded = self.per * group_size
        self.lo, self.hi = group_rank * self.per, (group_rank + 1) * self.per
        ref = self.params[0][1]
        self.dtype, self.device = ref.dtype, ref.device

    def _cat(self, ts: List[Optional[torch.Tensor]]) -> torch.Tensor:
        parts = [(t.reshape(-1) if t is not None else torch.zeros(k, dtype=self.dtype, device=self.device))
                 for t, k in zip(ts, self.sizes)]
        if self.padded > self.total:
            parts.append(torch.zeros(self.padded - self.total, dtype=self.dtype, device=self.device))
        return torch.cat(parts)

    def flat_params(self) -> torch.Tensor:
        return self._cat([p.detach() for _, p in self.params])

    def flat_grads(self) -> torch.Tensor:
        return self._cat([p.grad for _, p in self.params])

    def flat_mask(self, keep) -> torch.Tensor:
        """1 for entries of parameters where keep(name) is true, else 0 (for norms counted once)."""
        return self._cat([torch.full((k,), 1.0 if keep(n) else 0.0, dtype=self.dtype, device=self.device)
                          for (n, _), k in zip(self.params, self.sizes)])

    def mine(self, flat: torch.Tensor) -> torch.Tensor:
        return flat[self.lo:self.hi]

    def pieces(self, flat: torch.Tensor) -> List[torch.Tensor]:
        out, off = [], 0
        for k, sh in zip(self.sizes, self.shapes):
            out.append(flat[off:off + k].view(sh))
            off += k
        return out

    def write_params(self, flat: torch.Tensor) -> None:
        """Make every parameter a view of `flat` (one buffer, no copy per parameter)."""
        for (_, p), piece in zip(self.params, self.pieces(flat)):
            p.data = piece

    def release_params(self) -> None:
        """Drop the full parameters (level 2 between steps): each becomes an empty tensor."""
        for _, p in self.params:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    def state_bytes(self, slots: int) -> int:
        return slots * self.per * torch.empty((), dtype=self.dtype).element_size()
