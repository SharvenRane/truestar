"""Which convention did a device use? Three tools:

* `match_op`: given an op's integer inputs, its quantization parameters and the device's integer
  output, run the op under every convention and report which ones reproduce the device bit for bit.
  Conventions that give identical outputs on these inputs are grouped, so the answer never claims
  more than the inputs can tell.
* `distinguishing_case`: build inputs on which the conventions disagree (the CMSIS-NN#214
  situation), so a second device run can settle what the first could not.
* `pair_signature` and `rounding_only_test`: when the layer's parameters are unknown (a hosted NPU
  whose quantized model is not on this machine) compare the device with a control run of the
  same model. Two documented requantization rules applied to the same accumulator differ almost
  only in one direction per channel and sign of the value (each is a monotone rounding of nearly
  the same product); measured over every pair and 64 multipliers, the minority direction never
  exceeds 0.041% of elements. Differences in both directions at a higher rate inside one channel
  mean the accumulators themselves differ, so the cause is before the requantizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .conventions import CONVENTIONS, RULES, documented_conventions
from .ops import qrange, run_op


@dataclass
class ConventionResult:
    convention: str
    equal: int
    total: int
    max_lsb: int
    error: Optional[str] = None

    @property
    def rate(self) -> float:
        return self.equal / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {"convention": self.convention, "equal": self.equal, "total": self.total,
                "rate": self.rate, "max_lsb": self.max_lsb, "error": self.error}


@dataclass
class OpVerdict:
    op: str
    results: List[ConventionResult]
    groups: List[List[str]]  # conventions with identical outputs on these inputs
    matched: List[str]
    code: str
    text: str

    def as_dict(self) -> dict:
        return {"op": self.op, "matched": self.matched, "groups": self.groups, "code": self.code,
                "verdict": self.text, "results": [r.as_dict() for r in self.results]}


def _groups(outs: Dict[str, np.ndarray]) -> List[List[str]]:
    groups: List[List[str]] = []
    keys: List[np.ndarray] = []
    for name, o in outs.items():
        for g, k in zip(groups, keys):
            if k.shape == o.shape and np.array_equal(k, o):
                g.append(name)
                break
        else:
            groups.append([name])
            keys.append(o)
    return groups


def match_op(op: str, inputs: Dict[str, np.ndarray], params: dict, device: np.ndarray,
             conventions: Optional[Sequence[str]] = None) -> OpVerdict:
    """Compare a device output with every convention on the same integer inputs."""
    names = list(conventions or CONVENTIONS)
    device = np.asarray(device).astype(np.int64)
    outs: Dict[str, np.ndarray] = {}
    results: List[ConventionResult] = []
    for n in names:
        try:
            o = run_op(op, inputs, params, n)
        except Exception as exc:  # noqa: BLE001 (a convention that cannot run this op is reported)
            results.append(ConventionResult(n, 0, int(device.size), -1, f"{type(exc).__name__}: {exc}"))
            continue
        if o.shape != device.shape:
            results.append(ConventionResult(n, 0, int(device.size), -1,
                                            f"shape {o.shape} vs device {device.shape}"))
            continue
        outs[n] = o
        d = np.abs(o - device)
        results.append(ConventionResult(n, int((d == 0).sum()), int(d.size), int(d.max(initial=0))))
    groups = _groups(outs)
    matched = [r.convention for r in results if r.error is None and r.equal == r.total]
    results.sort(key=lambda r: (-r.rate, r.max_lsb))
    if matched and len(matched) == len(outs) and len(outs) > 1:
        code = "TS542"
        text = (f"{op}: every convention gives the same output on these inputs, so they cannot tell "
                f"which one the device uses; run `truestar intref cases` for inputs that separate them")
    elif matched:
        code = "TS540"
        others = [g for g in groups if not set(g) & set(matched)]
        text = (f"{op}: device matches {', '.join(matched)} bit for bit"
                + (f"; differs from {', '.join(n for g in others for n in g)}" if others else ""))
    else:
        best = results[0] if results else None
        code = "TS541"
        text = (f"{op}: device matches no known convention" +
                (f"; closest {best.convention} on {best.rate:.2%} of elements, worst {best.max_lsb} LSB"
                 if best else ""))
    return OpVerdict(op, results, groups, matched, code, text)


# ----------------------------------------------------------------------------- distinguishing inputs


def separating_accumulators(real: float, rules: Optional[Sequence[str]] = None,
                            lo: int = -(1 << 20), hi: int = 1 << 20) -> Dict[tuple, np.ndarray]:
    """For one multiplier, the accumulators in [lo, hi) on which each pair of rules disagrees."""
    rules = list(rules or RULES)
    acc = np.arange(lo, hi, dtype=np.int64)
    out = {r: RULES[r].requant(acc, real) for r in rules}
    res = {}
    for i, a in enumerate(rules):
        for b in rules[i + 1:]:
            res[(a, b)] = acc[out[a] != out[b]]
    return res


def distinguishing_case(op: str, params: dict, conventions: Optional[Sequence[str]] = None,
                        bits: int = 8, seed: int = 0) -> Dict[str, np.ndarray]:
    """Integer inputs for `op` on which as many convention pairs as possible disagree.

    MUL and ADD: every pair of int8 values (65,536 elements), which is exhaustive. REQUANTIZE: every
    input value. FULLY_CONNECTED and CONV: for each convention pair, accumulators where their
    rules disagree, reached with two taps (x0 * 1 + x1 * 127) plus the bias.
    """
    lo, hi = qrange(bits)
    rng = np.random.default_rng(seed)
    if op in ("mul", "add"):
        a, b = np.meshgrid(np.arange(lo, hi + 1), np.arange(lo, hi + 1), indexing="ij")
        dt = np.int16 if bits == 16 else np.int8
        return {"a": a.ravel().astype(dt), "b": b.ravel().astype(dt)}
    if op == "requantize":
        return {"x": np.arange(lo, hi + 1).astype(np.int16 if bits == 16 else np.int8)}
    if op in ("fully_connected", "conv"):
        names = list(conventions or documented_conventions())
        rules = sorted({CONVENTIONS[n].rule_for("fully_connected" if op == "fully_connected" else "conv").name
                        for n in names})
        sw = np.atleast_1d(np.asarray(params["sw"], dtype=np.float64))
        real = float(params["sx"]) * float(sw[0]) / float(params["sy"])
        accs: List[int] = []
        for _pair, vals in separating_accumulators(real, rules, -(1 << 14), 1 << 14).items():
            if vals.size:
                accs.extend(rng.choice(vals, size=min(64, vals.size), replace=False).tolist())
        accs.extend(rng.integers(-(1 << 13), 1 << 13, 256).tolist())
        accs = np.asarray(sorted(set(accs)), dtype=np.int64)
        xz = int(params.get("x_zp", 0))
        x1 = np.clip(np.round(accs / 127.0), lo - xz, hi - xz).astype(np.int64)
        x0 = accs - 127 * x1
        ok = (x0 >= lo - xz) & (x0 <= hi - xz)
        x = np.stack([x0[ok], x1[ok]], 1) + xz
        if op == "fully_connected":
            return {"x": x.astype(np.int8), "w": np.array([[1, 127]], dtype=np.int8),
                    "b": np.zeros(1, dtype=np.int32)}
        n = x.shape[0]
        return {"x": x.T.reshape(1, 2, 1, n).astype(np.int8),
                "w": np.array([[1, 127]], dtype=np.int8).reshape(1, 2, 1, 1),
                "b": np.zeros(1, dtype=np.int32)}
    raise ValueError(f"no distinguishing construction for {op!r}")


def pairs_distinguished(op: str, inputs: Dict[str, np.ndarray], params: dict,
                        conventions: Sequence[str]) -> Dict[str, object]:
    """How many convention pairs give different outputs on these inputs."""
    outs = {n: run_op(op, inputs, params, n) for n in conventions}
    names = list(conventions)
    same, diff = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (same if np.array_equal(outs[a], outs[b]) else diff).append((a, b))
    return {"pairs": len(same) + len(diff), "distinguished": len(diff), "identical": same,
            "groups": _groups(outs)}


# ----------------------------------------------------------------------------- unknown parameters


@dataclass
class PairSignature:
    """How a device's integer output differs from a control run of the same layer."""

    elements: int
    differ: int
    up: int
    down: int
    max_lsb: int
    by_magnitude: List[dict] = field(default_factory=list)
    channel_minority: float = 0.0  # median over channels of min(up, down) / elements, value > 0
    channels_two_sided: int = 0
    channels: int = 0

    @property
    def rate(self) -> float:
        return self.differ / self.elements if self.elements else 0.0

    def as_dict(self) -> dict:
        return {"elements": self.elements, "differ": self.differ, "rate": self.rate, "up": self.up,
                "down": self.down, "max_lsb": self.max_lsb, "by_magnitude": self.by_magnitude,
                "channel_minority_median": self.channel_minority,
                "channels_two_sided": self.channels_two_sided, "channels": self.channels}


# Largest share of elements, per channel and value sign, on which a pair of documented rules
# differs in the minority direction: measured 0.041% (q15_single_up vs q31_double); the bound
# keeps a 12x margin. tests/test_intref.py re-measures it.
ROUNDING_MINORITY_BOUND = 0.005

BINS = ((1, 15), (16, 31), (32, 63), (64, 127), (128, 254))


def pair_signature(device: np.ndarray, control: np.ndarray, channel_axis: int = 1,
                   zero_point: int = 0, min_count: int = 20) -> PairSignature:
    """Difference statistics between two integer outputs of the same layer (levels above the zero
    point, so a relu output's 0 is level 0)."""
    d = np.asarray(device, np.int64) - np.asarray(control, np.int64)
    lv = np.asarray(control, np.int64) - zero_point
    bins = []
    for lo, hi in BINS:
        sel = (lv >= lo) & (lv <= hi)
        n = int(sel.sum())
        bins.append({"levels": [lo, hi], "elements": n,
                     "rate": float((d[sel] != 0).mean()) if n else 0.0,
                     "up": float((d[sel] > 0).mean()) if n else 0.0,
                     "down": float((d[sel] < 0).mean()) if n else 0.0})
    pos = lv > 0
    dd = np.moveaxis(np.where(pos, d, 0), channel_axis, 0).reshape(d.shape[channel_axis], -1)
    up = (dd > 0).sum(1)
    dn = (dd < 0).sum(1)
    tot = up + dn
    npos = np.moveaxis(pos, channel_axis, 0).reshape(d.shape[channel_axis], -1).sum(1)
    keep = (tot >= min_count) & (npos > 0)
    minority = np.minimum(up, dn)[keep] / npos[keep] if keep.any() else np.zeros(0)
    return PairSignature(int(d.size), int((d != 0).sum()), int((d > 0).sum()), int((d < 0).sum()),
                         int(np.abs(d).max(initial=0)), bins,
                         float(np.median(minority)) if minority.size else 0.0,
                         int((minority > ROUNDING_MINORITY_BOUND).sum()), int(keep.sum()))


def rounding_only_test(sig: PairSignature) -> Dict[str, object]:
    """Can a requantization rule difference alone explain the signature? Only if the minority
    direction stays within ROUNDING_MINORITY_BOUND of elements in most channels."""
    explained = sig.channel_minority <= ROUNDING_MINORITY_BOUND
    return {"rounding_only_possible": explained,
            "channel_minority_median": sig.channel_minority,
            "bound": ROUNDING_MINORITY_BOUND,
            "channels_two_sided": sig.channels_two_sided, "channels": sig.channels,
            "reason": ("differences go one way per channel, as a rounding rule difference would"
                       if explained else
                       f"{sig.channels_two_sided} of {sig.channels} channels differ in both "
                       f"directions (median minority share {sig.channel_minority:.2%} of elements, "
                       f"a documented rule pair stays under {ROUNDING_MINORITY_BOUND:.1%}); "
                       "the accumulators differ, so the cause is before the requantizer")}
