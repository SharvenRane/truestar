"""Precision remedies: keep the fewest layers in float so INT8 prediction flips fall under a target.

Offline and local. The INT8 model is simulated on CPU with fake quantization (w8a8: weights per
output channel symmetric INT8, each quantized layer's input per tensor asymmetric UINT8 with
min/max ranges from a calibration batch), and every prediction is compared with the float64 model.
This is the intended INT8 arithmetic, the same thing the NPU study's host simulation measures
(docs/NPU.md), not a vendor runtime.

Method
------
1. Units: the quantizable layers (Conv1d, Conv2d, Linear) by default, or any module names (for
   example the block taps of the NPU study); a unit kept in float keeps every quantizable layer
   under it.
2. Ranking: from per layer error data when given (a tap report: the error each tap adds over the
   largest error before it, so inherited error is not counted twice), otherwise measured: the
   output error with only that unit kept in float, one simulation per unit.
3. Greedy with verification: units are kept in ranking order, and after each one the whole model
   is simulated again and its flips counted, until flips are at most the target.
4. Prune: each kept unit is dropped again in turn, least useful first, if the simulation still
   meets the target without it. The result is verified, and no single unit can be removed.

Reported: flips and output error before and after, the kept units and layers, and the fraction of
multiply accumulates (MACs) that stay in higher precision.

Qualcomm execution is not done here. The hook is `PrecisionPlan.qai_hub_hint()`: the layer names to
exclude from quantization, for a later quantize step in truestar.remote.qai_hub_quant
(HubSession.quantize) to pass to the vendor tool. That module is not changed by this one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
from torch import nn

from ..codes import Issue

QUANT_TYPES = (nn.Conv1d, nn.Conv2d, nn.Linear)
BATCH = 64


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    n = float((a - b).norm())
    d = float(b.norm())
    return 0.0 if n == 0 else (n / d if d > 0 else float("inf"))


class Int8Simulator:
    """Fake quantized copy of a model on CPU. `run(keep)` returns logits with the layers in `keep`
    left in float32 and every other quantizable layer in simulated INT8."""

    def __init__(self, model: nn.Module, eval_x: torch.Tensor, calib_x: torch.Tensor,
                 w_bits: int = 8, a_bits: int = 8):
        self.model = copy.deepcopy(model).cpu().float().eval()
        self.eval_x = eval_x.detach().cpu().float()
        self.w_max = 2 ** (w_bits - 1) - 1
        self.a_max = 2 ** a_bits - 1
        self.runs = 0
        ref = copy.deepcopy(model).cpu().double().eval()
        with torch.no_grad():
            self.reference = torch.cat([ref(x.double()) for x in self.eval_x.split(BATCH)])
        del ref
        self.leaves: List[str] = []
        self.ranges: Dict[str, Tuple[float, float]] = {}
        self.macs: Dict[str, int] = {}
        mods = {n: m for n, m in self.model.named_modules() if isinstance(m, QUANT_TYPES)}
        hooks = []

        def pre(name):
            def h(mod, inp):
                x = inp[0].detach()
                lo, hi = float(x.min()), float(x.max())
                if name in self.ranges:
                    a, b = self.ranges[name]
                    lo, hi = min(lo, a), max(hi, b)
                elif name not in self.leaves:
                    self.leaves.append(name)
                self.ranges[name] = (lo, hi)
            return h

        def post(name):
            def h(mod, inp, out):
                per = out.numel() // max(out.shape[0], 1)
                if isinstance(mod, nn.Linear):
                    k = mod.in_features
                else:
                    k = (mod.in_channels // mod.groups) * int(torch.tensor(mod.kernel_size).prod())
                self.macs[name] = per * k
            return h

        for n, m in mods.items():
            hooks.append(m.register_forward_pre_hook(pre(n)))
            hooks.append(m.register_forward_hook(post(n)))
        with torch.no_grad():
            for x in calib_x.detach().cpu().float().split(BATCH):
                self.model(x)
        for h in hooks:
            h.remove()
        self._mods = {n: mods[n] for n in self.leaves}
        self._float_w = {n: m.weight.detach().clone() for n, m in self._mods.items()}
        self._q_w = {n: self._quant_weight(w) for n, w in self._float_w.items()}

    @property
    def total_macs(self) -> int:
        return sum(self.macs.values())

    def _quant_weight(self, w: torch.Tensor) -> torch.Tensor:
        amax = w.abs().reshape(w.shape[0], -1).amax(1).clamp_min(1e-12)
        scale = (amax / self.w_max).double()
        return torch.fake_quantize_per_channel_affine(
            w, scale.float(), torch.zeros(w.shape[0], dtype=torch.int32), 0, -self.w_max,
            self.w_max)

    def _act_hook(self, name: str):
        lo, hi = self.ranges[name]
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        scale = max((hi - lo) / self.a_max, 1e-12)
        zp = int(round(-lo / scale))
        zp = min(max(zp, 0), self.a_max)

        def h(mod, inp):
            return (torch.fake_quantize_per_tensor_affine(inp[0], scale, zp, 0, self.a_max),
                    ) + tuple(inp[1:])
        return h

    def run(self, keep: Iterable[str] = ()) -> torch.Tensor:
        keep = set(keep)
        hooks = []
        try:
            with torch.no_grad():
                for n, m in self._mods.items():
                    if n in keep:
                        m.weight.copy_(self._float_w[n])
                    else:
                        m.weight.copy_(self._q_w[n])
                        hooks.append(m.register_forward_pre_hook(self._act_hook(n)))
                out = torch.cat([self.model(x) for x in self.eval_x.split(BATCH)])
        finally:
            for h in hooks:
                h.remove()
            with torch.no_grad():
                for n, m in self._mods.items():
                    m.weight.copy_(self._float_w[n])
        self.runs += 1
        return out

    def flips(self, logits: torch.Tensor) -> int:
        return int((logits.argmax(1) != self.reference.argmax(1)).sum())

    def error(self, logits: torch.Tensor) -> float:
        return _rel(logits, self.reference)


@dataclass
class PrecisionPlan:
    model_name: str
    scheme: str
    n_inputs: int
    target_flips: int
    keep: List[str]  # units kept in float
    keep_leaves: List[str]  # quantizable layers those units cover
    n_leaves: int
    flips_before: int
    flips_after: int
    err_before: float
    err_after: float
    compute_fraction: float  # MACs kept in float / all quantizable MACs
    ranking_source: str  # "measured" or "layer error data"
    ranking: List[Tuple[str, float]]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    simulations: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def met(self) -> bool:
        return self.flips_after <= self.target_flips

    @property
    def issue(self) -> Issue:
        if self.met:
            return Issue("TS094", f"verified in simulation: keeping {len(self.keep_leaves)} of "
                                  f"{self.n_leaves} layers in float takes flips "
                                  f"{self.flips_before} -> {self.flips_after} (target "
                                  f"{self.target_flips}) at {100 * self.compute_fraction:.2f}% "
                                  "of compute in float")
        return Issue("TS095", f"target of {self.target_flips} flips not reached: best verified "
                              f"{self.flips_after} (from {self.flips_before}) with "
                              f"{len(self.keep_leaves)} layers in float")

    def qai_hub_hint(self) -> Dict[str, Any]:
        """Hook for vendor execution (not run here): which layers to leave out of INT8."""
        return {"keep_float_layers": list(self.keep_leaves), "scheme": self.scheme,
                "verified_flips": self.flips_after, "target_flips": self.target_flips,
                "where": "truestar.remote.qai_hub_quant.HubSession.quantize (not wired yet)"}

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["ranking"] = [{"unit": u, "score": s} for u, s in self.ranking]
        d["met"] = self.met
        d["issue"] = self.issue.as_dict()
        d["qai_hub_hint"] = self.qai_hub_hint()
        return d

    def to_text(self) -> str:
        from .output import plan_text

        return plan_text(self)

    def to_agent(self) -> Dict[str, Any]:
        from .output import plan_agent

        return plan_agent(self)


def _errors_in_order(layer_errors: Any) -> List[Tuple[str, float]]:
    """(name, error) in forward order from a mapping, a LayerBlame, or rows with name and error."""
    rows = getattr(layer_errors, "rows", None)
    if rows is not None:
        return [(r.name, float(r.rel_vs_control)) for r in rows]
    if isinstance(layer_errors, Mapping):
        return [(str(k), float(v)) for k, v in layer_errors.items()]
    out = []
    for r in layer_errors:
        if isinstance(r, Mapping):
            err = r.get("error", r.get("rel_vs_control", r.get("rel_err")))
            out.append((str(r["name"]), float(err)))
        else:
            out.append((str(r[0]), float(r[1])))
    return out


def own_increase(errors: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Error each tap adds over the largest error seen before it (inherited error removed)."""
    out, high = [], 0.0
    for n, e in errors:
        out.append((n, max(0.0, e - high)))
        high = max(high, e)
    return out


def select_precision(model: nn.Module, eval_x: torch.Tensor, calib_x: Optional[torch.Tensor] = None,
                     target_flips: int = 0, units: Optional[Sequence[str]] = None,
                     layer_errors: Any = None, model_name: Optional[str] = None,
                     simulator: Optional[Int8Simulator] = None) -> PrecisionPlan:
    """Choose the fewest units to keep in float so simulated INT8 flips are at most target_flips.

    `layer_errors`: optional per layer error data (mapping name -> error in forward order, a
    truestar.remote.agreement.LayerBlame, or rows with name and error). When given, its names are
    the units and the ranking comes from it; otherwise each unit is measured.
    """
    sim = simulator or Int8Simulator(model, eval_x, eval_x if calib_x is None else calib_x)
    notes: List[str] = []
    if layer_errors is not None:
        ordered = _errors_in_order(layer_errors)
        names = [n for n, _ in ordered] if units is None else list(units)
        ranking_src = "layer error data"
        scores = dict(own_increase(ordered))
    else:
        names = list(units) if units is not None else list(sim.leaves)
        ranking_src = "measured"
        scores = {}

    def leaves_of(u: str) -> List[str]:
        return [l for l in sim.leaves if l == u or l.startswith(u + ".")]

    unit_leaves = {u: leaves_of(u) for u in names}
    empty = [u for u, ls in unit_leaves.items() if not ls]
    if empty:
        notes.append("units with no quantizable layer, ignored: " + ", ".join(empty))
    names = [u for u in names if unit_leaves[u]]
    macs = {u: sum(sim.macs[l] for l in unit_leaves[u]) for u in names}

    base = sim.run(())
    f0, e0 = sim.flips(base), sim.error(base)
    if ranking_src == "measured":
        for u in names:
            scores[u] = e0 - sim.error(sim.run(unit_leaves[u]))
    order = sorted(names, key=lambda u: (-scores.get(u, 0.0), macs[u]))
    ranking = [(u, float(scores.get(u, 0.0))) for u in order]

    steps: List[Dict[str, Any]] = [{"action": "start", "unit": "-", "flips": f0, "err": e0}]
    keep: List[str] = []
    f, e = f0, e0

    def leaves_for(us: Iterable[str]) -> Set[str]:
        s: Set[str] = set()
        for u in us:
            s.update(unit_leaves[u])
        return s

    for u in order:
        if f <= target_flips:
            break
        keep.append(u)
        out = sim.run(leaves_for(keep))
        f, e = sim.flips(out), sim.error(out)
        steps.append({"action": "keep", "unit": u, "flips": f, "err": e})
    if f <= target_flips:
        for u in list(reversed(keep)):
            trial = [k for k in keep if k != u]
            out = sim.run(leaves_for(trial))
            ft = sim.flips(out)
            if ft <= target_flips:
                keep = trial
                f, e = ft, sim.error(out)
                steps.append({"action": "drop", "unit": u, "flips": f, "err": e})
    # Final verification of exactly the reported set.
    out = sim.run(leaves_for(keep))
    f, e = sim.flips(out), sim.error(out)
    kl = [l for l in sim.leaves if l in leaves_for(keep)]
    frac = sum(sim.macs[l] for l in kl) / sim.total_macs if sim.total_macs else 0.0
    return PrecisionPlan(
        model_name=model_name or type(model).__name__, scheme="w8a8 fake quant",
        n_inputs=int(sim.reference.shape[0]), target_flips=int(target_flips),
        keep=[u for u in order if u in keep], keep_leaves=kl, n_leaves=len(sim.leaves),
        flips_before=f0, flips_after=f, err_before=e0, err_after=e, compute_fraction=frac,
        ranking_source=ranking_src, ranking=ranking, steps=steps, simulations=sim.runs,
        notes=notes)
