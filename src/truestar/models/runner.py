"""The loaded model: the native layer, its parameter tree and two compiled native functions (the
prompt and a decode step through the KV cache). No ``from __future__ import annotations`` here:
the native functions' annotations are the layer's own specs."""

from typing import Any, Dict, Optional

import torch

from ..frontend.function import NativeFunction
from ..frontend.types import Tensor
from .decoder import DecoderLM


def native_functions(m: DecoderLM, executor=None):
    """(prefill, decode): ``prefill(p, ids, pos) -> logits``; ``decode(p, ids, pos, cache) ->
    (logits, cache)``. One compile each serves every batch, length, position and cache capacity."""
    spec = m.spec
    cspec = m.cache_spec()

    def prefill(p: spec, ids: Tensor["B", "T", "i64"], pos: Tensor["T", "i64"]):
        return m(p, ids, pos)

    def decode(p: spec, ids: Tensor["B", "T", "i64"], pos: Tensor["T", "i64"], cache: cspec):
        logits, c2 = m(p, ids, pos, cache)
        return logits, c2

    pre = NativeFunction(prefill, executor=executor, eager=False)
    dec = NativeFunction(decode, executor=executor, eager=False)
    pre.name = f"{m.__class__.__name__}.prefill"
    dec.name = f"{m.__class__.__name__}.decode"
    return pre, dec


class PretrainedLM:
    """``lm(ids)`` gives the logits [B, T, vocab] of int64 ids [B, T]; ``lm.init_cache(B, S)`` and
    ``lm.step(ids, pos, cache)`` run the decoder through a KV cache; ``lm.generate(ids, n)`` decodes
    greedily. ``lm.model`` is the ``truestar.nn`` layer, ``lm.params`` its tree (call
    ``lm.model(lm.params, ids, pos)`` to run it eagerly through PyTorch)."""

    def __init__(self, model: DecoderLM, params: Dict[str, Any], config, info, executor=None):
        self.model = model
        self.params = params
        self.config = config
        self.info = info
        self.executor = executor
        self._fns = None

    @property
    def fns(self):
        if self._fns is None:
            self._fns = native_functions(self.model, self.executor)
        return self._fns

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.forward(ids)

    def forward(self, ids: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos = torch.arange(ids.shape[1]) if pos is None else pos
        return self.fns[0](self.params, ids, pos)

    def init_cache(self, batch: int, capacity: int):
        return self.model.init_cache(batch, capacity)

    def step(self, ids: torch.Tensor, pos: torch.Tensor, cache):
        return self.fns[1](self.params, ids, pos, cache)

    def generate(self, ids: torch.Tensor, steps: int, capacity: Optional[int] = None) -> torch.Tensor:
        """Greedy decoding: the prompt through the cache, then ``steps`` one token steps."""
        B, T = ids.shape
        cache = self.init_cache(B, capacity or T + steps)
        logits, cache = self.step(ids, torch.arange(T), cache)
        out = [ids]
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        for t in range(steps):
            out.append(nxt)
            if t == steps - 1:
                break
            logits, cache = self.step(nxt, torch.tensor([T + t]), cache)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
        return torch.cat(out, dim=1)

    def __repr__(self):
        c = self.config
        return (f"<truestar.models {c.family} {c.depth} layers, dim {c.dim}, {c.heads}/{c.kv_heads} heads, "
                f"{self.model.dtype}, {self.model.num_params():,} params>")
