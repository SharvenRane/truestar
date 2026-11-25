"""The absorbed decode proven equal to the expanded decode in float64, before it is compiled.

``prove_absorption`` decodes greedily in float64 with the expanded attention (Hugging Face's
computation) and, at every step, runs the absorbed attention on the same inputs and the same cache.
It compares, layer by layer, the latent cache the step returns and the attention output (the o_proj
input), then the logits; a point holds when its largest difference is within ``TOL`` of its largest
magnitude. Layer 0's cache is written before any attention and is equal bit for bit; a later
layer's cache reads the attention output of the layers before it. A wrong
absorption (a head mixed up, W_uk transposed, RoPE folded into the latent query) fails at the first
layer it touches, and the error names it (``MLAAbsorptionError``, TS1201).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .model import MLAModel

F64 = torch.float64
TOL = 1e-9     # relative to a point's largest magnitude; a correct absorption measures about 1e-15


@dataclass
class AbsorptionReport:
    tol: float
    steps: int
    points: List[tuple] = field(default_factory=list)    # (step, name, relative difference)
    first_bad: Optional[tuple] = None

    @property
    def ok(self) -> bool:
        return self.first_bad is None

    @property
    def worst(self) -> float:
        return max(p[2] for p in self.points)

    def message(self) -> str:
        if self.ok:
            return (f"absorbed MLA decode equals the expanded decode in float64 at {len(self.points)} points over "
                    f"{self.steps} steps (worst relative difference {self.worst:.2e}, bound {self.tol:.0e})")
        s, name, rel = self.first_bad
        return (f"[TS1201] the absorbed MLA attention differs from the expanded one at step {s}, {name} "
                f"(relative {rel:.3e}, bound {self.tol:.0e}); the absorbed weights of that layer (W_uk into "
                f"the query, W_uv after the attention) do not compute its keys and values. Nothing was compiled.")


class MLAAbsorptionError(AssertionError):
    """TS1201: the absorbed decode does not compute the expanded decode in float64."""
    code = "TS1201"

    def __init__(self, report: AbsorptionReport):
        super().__init__(report.message())
        self.report = report


def _rel(a, b) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


@torch.no_grad()
def prove_absorption(model: MLAModel, prompt: torch.Tensor, steps: int = 4, capacity: Optional[int] = None,
                     tol: float = TOL, raise_on_error: bool = True, mode: str = "absorb") -> AbsorptionReport:
    """Prompt [B, T] plus ``steps - 1`` greedy tokens, in float64 on the model's device. ``mode`` is
    the absorbed mode checked against the expanded one: ``"absorb"`` or ``"fused"`` (mla/latent.py)."""
    m = model if next(model.parameters()).dtype == F64 else copy.deepcopy(model).double()
    dev = next(m.parameters()).device
    b, t0 = prompt.shape
    capacity = capacity or t0 + steps
    cc, rc = m.empty_cache(b, capacity, F64, dev)
    ids, pos = prompt.to(dev), torch.arange(t0, device=dev)
    rep = AbsorptionReport(tol, steps)
    for s in range(steps):
        ra, rb = [], []
        le, cce, rce = m.step(ids, pos, cc, rc, "expand", ra)
        la, cca, rca = m.step(ids, pos, cc, rc, mode, rb)
        pts = []
        for i, (x, y) in enumerate(zip(rb, ra)):
            pts.append((f"model.layers.{i} latent cache", torch.cat((cca[i], rca[i]), -1),
                        torch.cat((cce[i], rce[i]), -1)))
            pts.append((f"model.layers.{i} attention output", x, y))
        pts.append(("logits", la, le))
        for name, got, ref in pts:
            rel = _rel(got, ref)
            rep.points.append((s, name, rel))
            if rel > tol and rep.first_bad is None:
                rep.first_bad = (s, name, rel)
        if rep.first_bad is not None:
            break
        cc, rc = cce, rce
        ids, pos = le[:, -1].argmax(-1)[:, None], torch.tensor([t0 + s], device=dev)
    if not rep.ok and raise_on_error:
        raise MLAAbsorptionError(rep)
    return rep


__all__ = ["prove_absorption", "AbsorptionReport", "MLAAbsorptionError", "TOL"]
