"""Per input agreement and per layer blame for runs of one model on several devices.

Pure numpy and torch: nothing here talks to a remote service, so all of it is unit tested offline.

Agreement follows arXiv 2609.16085: for every pair of runs, count on how many inputs the top 1
prediction is the same. Aggregate accuracy is reported next to it because the point of the paper is
that accuracy can stay flat while individual predictions move.

Near ties. When two runs pick different classes c_a and c_b, the swing is how far the logit
difference between exactly those two classes moved: (a[c_a] - a[c_b]) + (b[c_b] - b[c_a]). The
disagreement is a near tie when the swing is within what correct arithmetic in both runs can move,
the sum of each run's tie tolerance:

* float64 reference: 0 (it is the reference).
* float16 run: 2 * float16 floor * the input's largest logit magnitude (same rule as truestar.check).
* float32 run: the same with the float32 floor.
* INT8 run: 2 output quantization steps. Dequantized INT8 outputs live on a grid; the step is
  estimated from the data (`estimate_step`) when it is not given.

A large swing is a clear flip, even if one run alone looks nearly tied: that run moved its logits a
long way to get there.

An exact tie is a disagreement where one run scores the two contested classes exactly equal. That happens often in INT8
because logits are rounded to the output grid, and argmax then picks whichever index comes first.
Exact ties are a subset of near ties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..compare import DTYPE_FLOOR

FP16_FLOOR = DTYPE_FLOOR[torch.float16]
# A tensor counts as INT8 dequantized if this fraction of its values sit on one grid.
GRID_FRACTION = 0.99


def estimate_step(x: np.ndarray) -> Optional[float]:
    """Quantization step of a dequantized tensor, or None if the values are not on a grid.

    The step is the smallest gap between distinct values; the tensor is accepted as quantized only
    if almost every value is an integer number of steps away from the smallest value.
    """
    v = np.unique(np.asarray(x, dtype=np.float64).ravel())
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    span = float(v[-1] - v[0])
    if span <= 0:
        return None
    gaps = np.diff(v)
    gaps = gaps[gaps > span * 1e-6]
    if gaps.size == 0:
        return None
    step = float(gaps.min())
    if span / step > 70000:  # far more levels than 16 bits: a float tensor
        return None
    k = (v - v[0]) / step
    on_grid = np.abs(k - np.round(k)) < 0.02
    if on_grid.mean() < GRID_FRACTION:
        return None
    return step


@dataclass
class RunLogits:
    """Logits of one run: [N, C], with how close a top two margin must be to count as a tie."""

    name: str
    logits: np.ndarray
    precision: str = "float64"  # float64, float32, float16 or int8
    step: Optional[float] = None  # INT8 output step; estimated when None
    jobs: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.logits = np.asarray(self.logits, dtype=np.float64)
        if self.logits.ndim != 2 or self.logits.shape[1] < 2:
            raise ValueError(f"{self.name}: logits must be [N, C>=2], got {self.logits.shape}")
        if self.precision == "int8" and self.step is None:
            self.step = estimate_step(self.logits)

    def tie_tolerance(self) -> np.ndarray:
        """Per input margin below which this run's top 1 is a coin flip."""
        n = self.logits.shape[0]
        if self.precision == "int8":
            step = self.step if self.step is not None else 0.0
            return np.full(n, 2.0 * step)
        if self.precision == "float16":
            return 2.0 * FP16_FLOOR * np.abs(self.logits).max(axis=1)
        if self.precision == "float32":
            return 2.0 * DTYPE_FLOOR[torch.float32] * np.abs(self.logits).max(axis=1)
        return np.zeros(n)

    def top1(self) -> np.ndarray:
        return self.logits.argmax(axis=1)

    def margin(self) -> np.ndarray:
        s = np.sort(self.logits, axis=1)
        return s[:, -1] - s[:, -2]

    def accuracy(self, labels: np.ndarray) -> float:
        return float((self.top1() == np.asarray(labels)).mean())


@dataclass
class PairAgreement:
    a: str
    b: str
    n: int
    agree: int
    near_tie: int  # disagreements explained by a near tie in either run
    exact_tie: int  # disagreements where either run's top two are exactly equal
    flipped: List[int] = field(default_factory=list)  # indices of the disagreeing inputs

    @property
    def disagree(self) -> int:
        return self.n - self.agree

    @property
    def clear(self) -> int:
        """Disagreements that are not near ties: the prediction really moved."""
        return self.disagree - self.near_tie

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "n": self.n, "agree": self.agree,
                "disagree": self.disagree, "near_tie": self.near_tie, "exact_tie": self.exact_tie,
                "clear": self.clear, "flipped": self.flipped}


def pair_agreement(ra: RunLogits, rb: RunLogits) -> PairAgreement:
    if ra.logits.shape != rb.logits.shape:
        raise ValueError(f"shape {ra.logits.shape} ({ra.name}) vs {rb.logits.shape} ({rb.name})")
    ta, tb = ra.top1(), rb.top1()
    differ = ta != tb
    rows = np.arange(ta.size)
    # The two contested classes: how far each run prefers its own pick over the other's.
    gap_a = ra.logits[rows, ta] - ra.logits[rows, tb]
    gap_b = rb.logits[rows, tb] - rb.logits[rows, ta]
    swing = gap_a + gap_b  # how much the logit difference between the two classes moved
    tol = ra.tie_tolerance() + rb.tie_tolerance()
    near = differ & (swing <= tol * (1 + 1e-9) + 1e-12)  # slack for float rounding of the grid
    exact = differ & ((gap_a == 0) | (gap_b == 0))
    return PairAgreement(a=ra.name, b=rb.name, n=int(ta.size), agree=int((~differ).sum()),
                         near_tie=int(near.sum()), exact_tie=int(exact.sum()),
                         flipped=[int(i) for i in np.nonzero(differ)[0]])


def agreement_matrix(runs: Sequence[RunLogits]) -> List[PairAgreement]:
    """Every unordered pair, in the order the runs are given."""
    out = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            out.append(pair_agreement(runs[i], runs[j]))
    return out


def agreement_text(runs: Sequence[RunLogits], pairs: Sequence[PairAgreement],
                   labels: Optional[np.ndarray] = None) -> str:
    lines = []
    if labels is not None:
        lines.append("accuracy on the batch (top 1 vs label):")
        for r in runs:
            lines.append(f"  {r.name:<28} {r.accuracy(labels) * 100:6.2f}%  "
                         f"({int((r.top1() == labels).sum())}/{len(labels)})"
                         + (f"  output step {r.step:.4g}" if r.step else ""))
    lines.append("per input agreement (same top 1):")
    for p in pairs:
        lines.append(f"  {p.a} vs {p.b}: {p.agree}/{p.n} agree, {p.disagree} differ "
                     f"({p.near_tie} near ties, {p.exact_tie} exact ties, {p.clear} clear)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- per layer blame


@dataclass
class LayerRow:
    name: str
    shape: Tuple[int, ...]
    rel_vs_control: float  # device vs control run of the same compiled model
    rel_vs_reference: float  # device vs float64 reference
    control_vs_reference: float
    threshold: float
    step: Optional[float]  # INT8 grid step of the control tensor, if quantized
    frac_over_1lsb: Optional[float]  # INT8: share of elements that differ by more than one step
    crossed: bool

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["shape"] = list(self.shape)
        return d


@dataclass
class LayerBlame:
    rows: List[LayerRow]
    quantized: bool

    @property
    def first(self) -> Optional[LayerRow]:
        for r in self.rows:
            if r.crossed:
                return r
        return None

    def to_text(self) -> str:
        kind = "INT8: 1 LSB rms of the control" if self.quantized else "float16 floor"
        lines = [f"per layer, device vs control (threshold: {kind}); errors include what was inherited"]
        for r in self.rows:
            extra = f"  >1 LSB {r.frac_over_1lsb * 100:5.2f}%" if r.frac_over_1lsb is not None else ""
            lines.append(f"  {'X' if r.crossed else ' '} {r.name:<12} {str(r.shape):<18} "
                         f"vs control {r.rel_vs_control:.2e} (thr {r.threshold:.1e})  "
                         f"vs ref {r.rel_vs_reference:.2e}  control vs ref {r.control_vs_reference:.2e}{extra}")
        f = self.first
        lines.append(f"first layer over threshold: {f.name}" if f else "no layer over threshold")
        return "\n".join(lines)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    num = float(np.linalg.norm((a - b).ravel()))
    den = float(np.linalg.norm(b.ravel()))
    if num == 0.0:
        return 0.0
    return num / den if den > 0 else float("inf")


def layer_threshold(control: np.ndarray, quantized: bool) -> Tuple[float, Optional[float]]:
    """(relative L2 threshold, INT8 step).

    float: the float16 floor. INT8: the relative error of being off by one output step on every
    element (step / rms of the control tensor); anything below is rounding of the last bit.
    """
    if not quantized:
        return FP16_FLOOR, None
    step = estimate_step(control)
    c = np.asarray(control, dtype=np.float64)
    rms = float(np.sqrt(np.mean(c * c))) if c.size else 0.0
    if step is None or rms == 0.0:
        # Not on a grid (the runtime kept this tensor in float): fall back to the float16 floor.
        return FP16_FLOOR, None
    return step / rms, step


def layer_blame(names: Sequence[str], device: Sequence[np.ndarray], control: Sequence[np.ndarray],
                reference: Sequence[np.ndarray], quantized: bool) -> LayerBlame:
    """Compare tapped tensors layer by layer and mark the first that crosses its threshold.

    `device`, `control` and `reference` hold one array per tap, all inputs stacked on axis 0, in
    the order of `names` (forward order). Because taps come from one full run, a layer's error
    includes what it inherited; the first layer over threshold is where the divergence starts.
    """
    if not (len(names) == len(device) == len(control) == len(reference)):
        raise ValueError("names, device, control and reference must have the same length")
    rows = []
    for n, d, c, r in zip(names, device, control, reference):
        d = np.asarray(d, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        thr, step = layer_threshold(c, quantized)
        rel = _rel(d, c)
        frac = None
        if step is not None and d.shape == c.shape:
            frac = float((np.abs(d - c) > 1.5 * step).mean())
        rows.append(LayerRow(name=n, shape=tuple(d.shape), rel_vs_control=rel,
                             rel_vs_reference=_rel(d, r), control_vs_reference=_rel(c, r),
                             threshold=thr, step=step, frac_over_1lsb=frac,
                             crossed=bool(rel > thr or math.isinf(rel))))
    return LayerBlame(rows=rows, quantized=quantized)
