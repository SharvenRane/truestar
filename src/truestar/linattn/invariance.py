"""Chunk size invariance and error growth along the chunk chain, measured against float64.

``chunk_invariance`` runs one call at several chunk sizes and reports, per chunk size, the per
element error against the float64 recurrence, and per pair of chunk sizes how far apart the two
results are and whether they are bitwise equal. In exact arithmetic every chunk size gives the
same result (docs/LINEAR_ATTENTION.md, the proof), so in float64 the pairs agree to rounding and
in float32 the pair difference is the rounding the reduction order changes, which is what this
reports.

``chain_error`` splits the output error against float64 by chunk index (the error of the outputs
of chunk n, which see the state after n chunk steps) and fits its growth: the slope of the error
over the chunk index by least squares, and the ratio of the last quarter's error to the first's
(TS1062 when it is above GROWTH_WARN).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from .op import linear_attention
from .reference import recurrent

GROWTH_WARN = 4.0   # last quarter over first quarter above this: TS1062


def _per_element(out: torch.Tensor, ref: torch.Tensor, floor: float = 2.0) -> torch.Tensor:
    """|out - ref| / max(|ref|, floor * rms(ref)), the kernel layer's measure (reduction floor)."""
    out, ref = out.to(torch.float64), ref.to(torch.float64)
    rms = ref.pow(2).mean().sqrt().clamp_min(1e-300)
    lo = torch.minimum(floor * rms, ref.abs().max()).clamp_min(1e-6)
    return (out - ref).abs() / torch.maximum(ref.abs(), lo)


def chunk_invariance(q, k, v, g=None, beta=None, *, rule="add", scale=1.0, initial_state=None,
                     chunks: Iterable[int] = (16, 32, 64), impl: Optional[str] = None) -> Dict[str, object]:
    with torch.no_grad():
        ref, sref = recurrent(q, k, v, g, beta, rule=rule, scale=scale, initial_state=initial_state)
        outs = {}
        for c in chunks:
            o, s = linear_attention(q, k, v, g, beta, rule=rule, scale=scale, initial_state=initial_state,
                                    chunk=c, deterministic=False, impl=impl, output_final_state=True)
            outs[c] = (o, s)
    per_chunk = {c: {"o": float(_per_element(o, ref).max()), "final_state": float(_per_element(s, sref).max())}
                 for c, (o, s) in outs.items()}
    pairs = {}
    cs = list(outs)
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            oa, ob = outs[a][0], outs[b][0]
            pairs[f"{a}-{b}"] = {"bitwise_equal": bool(torch.equal(oa, ob)),
                                 "max_abs": float((oa.double() - ob.double()).abs().max()),
                                 "per_element": float(_per_element(oa, ob.to(torch.float64)).max()),
                                 "elements_differing": float((oa != ob).double().mean())}
    return {"vs_float64": per_chunk, "pairs": pairs, "dtype": str(v.dtype), "T": int(q.shape[1])}


def chain_error(q, k, v, g=None, beta=None, *, rule="add", scale=1.0, chunk=64, impl=None,
                deterministic=False) -> Dict[str, object]:
    with torch.no_grad():
        ref, _ = recurrent(q, k, v, g, beta, rule=rule, scale=scale)
        o = linear_attention(q, k, v, g, beta, rule=rule, scale=scale, chunk=chunk, impl=impl,
                             deterministic=deterministic)
    e = _per_element(o, ref)                                   # [B, T, H, dv]
    T = e.shape[1]
    n = -(-T // chunk)
    per = [float(e[:, i * chunk:(i + 1) * chunk].max()) for i in range(n)]
    x = torch.arange(n, dtype=torch.float64)
    y = torch.tensor(per, dtype=torch.float64)
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum().clamp_min(1e-300)) if n > 1 else 0.0
    qn = max(1, n // 4)
    first, last = float(y[:qn].mean()), float(y[-qn:].mean())
    ratio = last / first if first > 0 else (1.0 if last == 0 else float("inf"))
    return {"per_chunk": per, "slope_per_chunk": slope, "last_over_first": ratio,
            "worst": float(y.max()), "chunks": n, "chunk": chunk,
            "code": "TS1062" if ratio > GROWTH_WARN else None}
