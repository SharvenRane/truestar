"""Speculative decoding whose tokens are bitwise those of plain greedy decoding.

    from truestar import specdec as SD
    inv = SD.check_rows(target, prompt, cache, rows=k + 1)       # TS706, or TS705 when it fails
    toks, stats = SD.speculative(target, draft, prompt, steps, k, target_cache, draft_cache)

A small draft model proposes ``k`` tokens; the target model checks them in one call over ``k + 1``
query rows and keeps the longest prefix it agrees with, plus its own next token. In exact arithmetic
the kept tokens are the target's greedy tokens. In floating point they are only if the target's
logits for a row do not depend on how many query rows the call has: a kernel that splits or tiles
its reduction by the row count gives the verify call other bits than the one row decode, an argmax
near a tie flips, and the output changes (arXiv 2510.22876, 2607.17283, 2609.15504 measure this).

``check_rows`` runs the target both ways on the same prefix (``rows`` tokens as one call, then the
same tokens one at a time) and compares logits and caches bitwise: TS706 when they are equal, TS705
when they are not (the first row and the largest difference are named). With a row invariant target
(Truestar deterministic mode: no split K, and per row reductions over keys, docs/PREFIX_CHECK.md)
``speculative`` returns bitwise the tokens of ``greedy``; the same ties break the same way because
the same logits reach the same argmax.

Steps are dense cache callables ``step(ids [B, T], pos [T], kc, vc) -> (logits [B, T, V], kc, vc)``
(examples/llama_paged_decode.py), whose cache rows beyond the current position are masked by
position: a rejected draft token's K and V stay in the cache, masked, until a later token at that
position overwrites them. Batch size 1 (each sequence keeps its own count of accepted tokens).

Limits: greedy only. Speculative *sampling* (Leviathan et al. 2023, Chen et al. 2023) keeps the
target's distribution, not its samples: the accept test draws extra random numbers, so even with a
fixed seed the sampled tokens are not those of plain sampling. A keyed sampler (one random number
per position, drawn from the position and a key) would make the accepted tokens equal to plain
keyed sampling; not done here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

Step = Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass
class RowCheck:
    invariant: bool
    rows: int
    first_row: Optional[int]
    max_abs: float
    codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"invariant": self.invariant, "rows": self.rows, "first_row": self.first_row,
                "max_abs": self.max_abs, "codes": self.codes}


def _pos(lo: int, hi: int, device) -> torch.Tensor:
    return torch.arange(lo, hi, device=device)


def check_rows(step: Step, prompt: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, rows: int,
               tokens: Optional[torch.Tensor] = None) -> RowCheck:
    """Prefill ``prompt``, then feed ``rows`` tokens as one call and one at a time from the same cache;
    compare the logits and the caches bitwise. The caches passed in are not changed."""
    dev = kc.device
    t0 = prompt.shape[1]
    _, k0, v0 = step(prompt.to(dev), _pos(0, t0, dev), kc.clone(), vc.clone())
    if tokens is None:              # any valid tokens: the check is about the rows, not the text
        tokens = prompt.flip(1).repeat(1, -(-rows // t0))[:, :rows]
    tokens = tokens.to(dev)
    many, km, vm = step(tokens, _pos(t0, t0 + rows, dev), k0.clone(), v0.clone())
    k1, v1, one = k0.clone(), v0.clone(), []
    for r in range(rows):
        lg, k1, v1 = step(tokens[:, r:r + 1], _pos(t0 + r, t0 + r + 1, dev), k1, v1)
        one.append(lg)
    one = torch.cat(one, 1)
    diff = (many.double() - one.double()).abs()
    same = torch.equal(many, one) and torch.equal(km, k1) and torch.equal(vm, v1)
    first = None
    if not same:
        bad = torch.nonzero(diff.amax(dim=(0, 2)) > 0)
        first = int(bad[0]) if len(bad) else 0
    return RowCheck(same, rows, first, diff.max().item(), ["TS706" if same else "TS705"])


def greedy(step: Step, prompt: torch.Tensor, steps: int, kc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
    """Plain greedy decoding: ``steps`` new tokens [B, steps]."""
    dev = kc.device
    t0 = prompt.shape[1]
    lg, kc, vc = step(prompt.to(dev), _pos(0, t0, dev), kc, vc)
    nxt = lg[:, -1].argmax(-1)
    out = [nxt]
    for s in range(1, steps):
        lg, kc, vc = step(nxt[:, None], _pos(t0 + s - 1, t0 + s, dev), kc, vc)
        nxt = lg[:, -1].argmax(-1)
        out.append(nxt)
    return torch.stack(out, 1)


def speculative(target: Step, draft: Step, prompt: torch.Tensor, steps: int, k: int,
                tk: torch.Tensor, tv: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
                verify: Optional[Step] = None) -> Tuple[torch.Tensor, Dict]:
    """Greedy speculative decoding, batch 1: ``steps`` new tokens and statistics.

    ``verify`` is the target called with ``k + 1`` rows (default ``target``; a separately graphed
    callable for that shape is the fast path). Per round: the draft proposes ``d1..dk`` from the
    last token ``x``; the target scores ``[x, d1..dk]``; ``n`` proposals agree with its argmax and the
    round emits ``n + 1`` target tokens."""
    if prompt.shape[0] != 1:
        raise ValueError("speculative(): batch 1 only")
    verify = verify or target
    dev = tk.device
    t0 = prompt.shape[1]
    p = prompt.to(dev)
    lg, tk, tv = target(p, _pos(0, t0, dev), tk, tv)
    _, dk, dv = draft(p, _pos(0, t0, dev), dk, dv)
    x = lg[:, -1].argmax(-1)                     # the first new token, at position t0
    out: List[torch.Tensor] = [x]
    pos = t0                                     # position of x (not yet in either cache)
    rounds = accepted = 0
    dpending: Optional[torch.Tensor] = None      # a draft token the draft cache still lacks
    while len(out) < steps:
        props = []
        if dpending is not None:                 # all k accepted last round: feed d_k first
            _, dk, dv = draft(dpending[:, None], _pos(pos - 1, pos, dev), dk, dv)
            dpending = None
        y = x
        for i in range(k):
            dl, dk, dv = draft(y[:, None], _pos(pos + i, pos + i + 1, dev), dk, dv)
            y = dl[:, -1].argmax(-1)
            props.append(y)
        cand = torch.stack([x] + props, 1)       # [1, k + 1]
        vl, tk, tv = verify(cand, _pos(pos, pos + k + 1, dev), tk, tv)
        g = vl[0].argmax(-1)                     # target tokens after each row
        d = torch.stack(props, 1)[0]
        agree = (d == g[:k]).int()
        n = int(agree.cumprod(0).sum())
        rounds += 1
        accepted += n
        new = g[: n + 1]
        out.extend(new[i:i + 1] for i in range(len(new)))
        if n == k:
            dpending = d[k - 1:k]
        x = new[-1:]
        pos += n + 1
    toks = torch.cat(out)[:steps][None]
    return toks, {"rounds": rounds, "accepted": accepted, "proposed": rounds * k,
                  "acceptance": accepted / max(1, rounds * k), "tokens_per_round": (steps - 1) / max(1, rounds)}
