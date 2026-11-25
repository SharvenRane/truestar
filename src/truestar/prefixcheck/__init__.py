"""Prefix cache check: does decoding with a cached prefix give the bits of one full prefill?

    from truestar import prefixcheck as PC
    rep = PC.compare(full_logits, cached_logits, full_layers={...}, cached_layers={...},
                     reference=ref_logits)
    print(rep.to_text())

A serving engine reuses the KV cache of a shared prefix (prefix caching, chunked prefill): it runs the
prefix once, keeps its K and V, and runs only the new tokens against them. In exact arithmetic that is
the full prefill of the whole sequence. In floating point it is the same only when every reduction
walks the same values in the same order in both runs; a split chosen from the token count, a
different attention schedule for a short query chunk, or a quantized prefix makes them differ.

``compare`` takes the two runs' logits at the new tokens (``[..., tokens, vocab]``, the token axis
second to last) and, optionally, per layer tensors of both runs (the K and V caches, keyed by layer
name, in model order) and a float64 reference of the logits. It reports whether the logits are
bitwise equal, the first token whose logits differ and by how much, the first layer whose tensors
differ, and each run's error against the reference, with these codes:

* TS704 (info) ``prefix_cache_bitwise``: the cached run gives the full prefill's bits.
* TS702 (warning) ``prefix_cache_divergent``: the runs differ; the first token and the first layer
  that differ are named, with the largest difference. Within rounding of each other when TS703 is
  not raised.
* TS703 (error) ``prefix_cache_wrong``: with a reference, the cached run's error exceeds the full
  prefill's by more than ``tolerance`` (relative to the reference's largest value): the cached path
  is less accurate, not only rounded differently.

``check_files`` runs the same comparison on files written by any engine (``.pt`` or ``.npz`` holding
``logits`` and optionally ``layers``), so an external engine's cached and uncached outputs can be
checked without Truestar running the model. The command is ``truestar prefixcheck``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

DEFAULT_TOL = 1e-3   # TS703: allowed extra relative error of the cached run against the reference


@dataclass
class PrefixReport:
    bitwise: bool
    max_abs: float
    first_token: Optional[int]          # first token (along the second to last axis) that differs
    tokens: int
    first_layer: Optional[str] = None   # first layer whose tensors differ, in the order given
    layer_diffs: Dict[str, float] = field(default_factory=dict)
    error_full: Optional[float] = None  # relative error against the reference, or None
    error_cached: Optional[float] = None
    tolerance: float = DEFAULT_TOL
    issues: List[Dict[str, str]] = field(default_factory=list)

    @property
    def codes(self) -> List[str]:
        return [i["code"] for i in self.issues]

    def to_dict(self) -> Dict[str, Any]:
        return {"bitwise": self.bitwise, "max_abs_difference": self.max_abs, "first_token": self.first_token,
                "tokens": self.tokens, "first_layer": self.first_layer, "layer_max_abs": self.layer_diffs,
                "error_full_vs_reference": self.error_full, "error_cached_vs_reference": self.error_cached,
                "tolerance": self.tolerance, "issues": self.issues}

    def to_text(self) -> str:
        lines = [f"prefix cache check: {'BITWISE' if self.bitwise else 'DIVERGENT'} over {self.tokens} tokens"]
        if not self.bitwise:
            lines.append(f"  first token that differs: {self.first_token}, largest difference {self.max_abs:.3e}")
            if self.first_layer is not None:
                lines.append(f"  first layer that differs: {self.first_layer} "
                             f"(largest difference {self.layer_diffs[self.first_layer]:.3e})")
        if self.error_full is not None:
            lines.append(f"  against the reference: full prefill {self.error_full:.3e}, cached {self.error_cached:.3e}")
        for i in self.issues:
            lines.append(f"  {i['code']} {i['message']}")
        return "\n".join(lines)


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.double() - ref.double()).abs().max() / ref.double().abs().max().clamp_min(1e-300)).item()


def compare(full: torch.Tensor, cached: torch.Tensor, full_layers: Optional[Dict[str, torch.Tensor]] = None,
            cached_layers: Optional[Dict[str, torch.Tensor]] = None, reference: Optional[torch.Tensor] = None,
            tolerance: float = DEFAULT_TOL) -> PrefixReport:
    """Compare a full prefill's logits with a cached prefix run's, at the same new tokens."""
    if tuple(full.shape) != tuple(cached.shape):
        raise ValueError(f"the two runs' logits have different shapes {tuple(full.shape)} and {tuple(cached.shape)}")
    f, c = full.detach().cpu(), cached.detach().cpu()
    bitwise = f.dtype == c.dtype and torch.equal(f, c)
    diff = (f.double() - c.double()).abs()
    first = None
    if not bitwise:
        per_token = diff.reshape(-1, diff.shape[-2], diff.shape[-1]).amax(dim=(0, 2)) if diff.dim() >= 2 else diff
        nz = torch.nonzero(per_token > 0)
        first = int(nz[0]) if len(nz) else 0
    rep = PrefixReport(bitwise, diff.max().item() if diff.numel() else 0.0, first,
                       f.shape[-2] if f.dim() >= 2 else 1, tolerance=tolerance)
    if full_layers is not None and cached_layers is not None:
        for name, a in full_layers.items():
            b = cached_layers[name]
            d = (a.detach().cpu().double() - b.detach().cpu().double()).abs().max().item()
            rep.layer_diffs[name] = d
            if rep.first_layer is None and not torch.equal(a.detach().cpu(), b.detach().cpu()):
                rep.first_layer = name
    if reference is not None:
        rep.error_full, rep.error_cached = _rel(f, reference.cpu()), _rel(c, reference.cpu())
    if bitwise:
        rep.issues.append({"code": "TS704", "message": "the cached prefix run gives the full prefill's bits"})
    else:
        where = f"token {first}" + (f", layer {rep.first_layer}" if rep.first_layer is not None else "")
        rep.issues.append({"code": "TS702", "message": f"the cached prefix run differs from the full prefill "
                                                      f"first at {where}, largest difference {rep.max_abs:.3e}"})
        if rep.error_full is not None and rep.error_cached > rep.error_full + tolerance:
            rep.issues.append({"code": "TS703", "message": f"the cached run is less accurate: error "
                                                          f"{rep.error_cached:.3e} against the reference, the "
                                                          f"full prefill {rep.error_full:.3e} (tolerance {tolerance:g})"})
    return rep


def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npz"):
        import numpy as np
        z = np.load(path)
        out: Dict[str, Any] = {"logits": torch.from_numpy(z["logits"])}
        layers = {k[len("layers/"):]: torch.from_numpy(z[k]) for k in z.files if k.startswith("layers/")}
        if layers:
            out["layers"] = layers
        return out
    d = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(d, torch.Tensor):
        return {"logits": d}
    if "logits" not in d:
        raise ValueError(f"{path}: no 'logits' entry")
    return d


def check_files(full_path: str, cached_path: str, reference_path: Optional[str] = None,
                tolerance: float = DEFAULT_TOL) -> PrefixReport:
    """``compare`` on files: ``.pt`` (a tensor, or a dict with ``logits`` and optionally a dict
    ``layers``) or ``.npz`` (``logits`` and ``layers/<name>`` arrays), from any engine."""
    a, b = _load(full_path), _load(cached_path)
    ref = _load(reference_path)["logits"] if reference_path else None
    return compare(a["logits"], b["logits"], a.get("layers"), b.get("layers"), ref, tolerance)
