"""Qualcomm QAIRT (QNN) quantization range modes as rounding conventions.

QAIRT quantizes a float tensor in three steps (quantization page of the QAIRT docs, read
2026-11-24, docs/INTREF.md has the link and the quotes):

1. pick a float range [min, max] from the calibration data: the *range mode* (`tf`,
   `symmetric`, `enhanced`, `adjusted`);
2. enforce a minimum range of 0.0001 and make float zero exactly representable, which moves the
   range by up to half a step;
3. code = round((x - min) / scale), scale = (max - min) / (2^bw - 1), clamped to [0, 2^bw - 1].

Two devices with the same kernels but different range modes give codes that differ by a step or
more on part of the tensor; two runs with the same range but another tie rule differ by exactly
one step on the inputs that sit on a half step. `attribute` uses that to say which mode (and which
tie rule) a device result came from, so a 1 LSB difference on Snapdragon is named as a range mode
choice and not reported as a kernel bug.

What is documented and what is not (each mode carries the flag):

* `tf` and `symmetric`: the doc gives the formulas and two worked examples (input [-1.8, 0.5]
  gives codes [0, 89, 200, 255]; input [-5.1, 5.1] gives min -5.12, max 5.08); both are tests.
* `enhanced`: the doc says only that it searches for a range that improves accuracy and may clip
  long tails. Truestar's version is its own search (minimum expected squared error over a grid of
  clipped ranges) and is labelled a hypothesis.
* `adjusted`: the doc says the max is expanded or the min decreased "if necessary" while zero stays
  exact. Truestar reads that as: widen (never shift inward) until the whole true range is covered.
  Also a hypothesis.
* the tie rule of round() is not documented; `half_away` (C++ std::round) and `half_even` are both
  offered and both are scored.

Signed codes are the unsigned codes minus 2^(bw-1) (the doc's signed asymmetric schema). Per
channel weights are symmetric with offset -2^(bw-1), as the doc requires.

The hook for device comparisons is `attribute(reference, device, ...)`; `simulate` runs a torch
model with every module output and weight fake quantized under chosen modes, in float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import fixedpoint as fp

MODES = ("tf", "symmetric", "enhanced", "adjusted")
TIES = ("half_away", "half_even")
MIN_RANGE = 0.0001
DOCUMENTED = {"tf": True, "symmetric": True, "enhanced": False, "adjusted": False}
SOURCE = "https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/quantization.html"


def _round(x, ties: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64)
    if ties == "half_away":
        # exact for every float64: v - trunc(v) has no rounding error (unlike floor(|v| + 0.5),
        # which turns 0.49999999999999994 into 1)
        t = np.trunc(v)
        return t + np.sign(v) * (np.abs(v - t) >= 0.5)
    if ties == "half_even":
        return fp.round_half_even(v)
    raise ValueError(f"unknown tie rule {ties!r}; use one of {', '.join(TIES)}")


@dataclass(frozen=True)
class Encoding:
    """One QNN encoding: the representable range and its integer grid."""

    bw: int
    min: float  # float value of unsigned code 0
    max: float  # float value of unsigned code 2^bw - 1
    scale: float
    offset: int  # QNN offset: round(min / scale), <= 0; float = (code + offset) * scale
    mode: str = "tf"
    ties: str = "half_away"
    signed: bool = False

    @property
    def levels(self) -> int:
        return (1 << self.bw) - 1

    @property
    def zero_point(self) -> int:
        """The code that represents float zero, in the signed or unsigned code space."""
        return -self.offset - (1 << (self.bw - 1) if self.signed else 0)

    def quantize(self, x) -> np.ndarray:
        """Float to codes: round((x - min) / scale), clamped (int64)."""
        x = np.asarray(x, dtype=np.float64)
        q = np.clip(_round((x - self.min) / self.scale, self.ties), 0, self.levels).astype(np.int64)
        return q - (1 << (self.bw - 1)) if self.signed else q

    def dequantize(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.int64)
        if self.signed:
            q = q + (1 << (self.bw - 1))
        return (q + self.offset).astype(np.float64) * self.scale

    def fake(self, x) -> np.ndarray:
        """x quantized and dequantized in float64."""
        return self.dequantize(self.quantize(x))

    def as_dict(self) -> Dict[str, Any]:
        return {"bitwidth": self.bw, "min": self.min, "max": self.max, "scale": self.scale,
                "offset": self.offset, "mode": self.mode, "ties": self.ties, "signed": self.signed,
                "zero_point": self.zero_point}


def from_range(lo: float, hi: float, bw: int = 8, *, ties: str = "half_away",
               signed: bool = False, mode: str = "tf") -> Encoding:
    """Steps 2 and 3 of the doc for a chosen float range: include zero, minimum range, then move
    the range so float zero is a code."""
    if bw < 2 or bw > 32:
        raise ValueError(f"bitwidth {bw} out of range")
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("the range must be finite (the calibration data has NaN or inf)")
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    hi = max(hi, lo + MIN_RANGE)
    n = (1 << bw) - 1
    scale = (hi - lo) / n
    off = int(_round(lo / scale, ties))
    off = max(-n, min(0, off))
    lo = off * scale
    return Encoding(bw, lo, lo + n * scale, scale, off, mode, ties, signed)


def _adjusted(lo: float, hi: float, bw: int, ties: str, signed: bool) -> Encoding:
    """The smallest scale whose grid, with an integer offset, covers [lo, hi] without moving
    either end inward (Truestar's reading of the doc's 'adjusted' mode)."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    hi = max(hi, lo + MIN_RANGE)
    n = (1 << bw) - 1
    o = np.arange(-n, 1, dtype=np.float64)  # candidate offsets
    with np.errstate(divide="ignore", invalid="ignore"):
        need_lo = np.where(o < 0, lo / o, np.where(lo < 0, np.inf, 0.0))
        need_hi = np.where(o + n > 0, hi / (o + n), np.inf)
    s = np.maximum(need_lo, need_hi)
    s = np.where(np.isfinite(s) & (s > 0), s, np.inf)
    k = int(np.argmin(s))
    scale, off = float(s[k]), int(o[k])
    return Encoding(bw, off * scale, (off + n) * scale, scale, off, "adjusted", ties, signed)


def _enhanced(x: np.ndarray, bw: int, ties: str, signed: bool, grid: int = 48) -> Encoding:
    """Range that minimizes the mean squared quantization error of x, searched over clipped
    ranges [lo * a, hi * b] with a, b in (0, 1] (Truestar's own search; the doc does not give the
    algorithm)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    lo, hi = float(x.min()), float(x.max())
    best, best_err = from_range(lo, hi, bw, ties=ties, signed=signed, mode="enhanced"), np.inf
    if x.size > 65536:  # deterministic subsample: every k-th element of the sorted data
        x = np.sort(x)[:: x.size // 65536 + 1]
    fr = np.linspace(1.0, 0.05, grid)
    for a in fr if lo < 0 else (1.0,):
        for b in fr if hi > 0 else (1.0,):
            e = from_range(lo * a, hi * b, bw, ties=ties, signed=signed, mode="enhanced")
            err = float(np.mean((e.fake(x) - x) ** 2))
            if err < best_err:
                best, best_err = e, err
    return best


def encoding(x, bw: int = 8, mode: str = "tf", *, ties: str = "half_away", signed: bool = False,
             axis: Optional[int] = None):
    """The encoding QAIRT's `mode` picks for calibration data x. With `axis`, one symmetric
    encoding per channel along that axis (offset -2^(bw-1), the doc's per channel rule); `mode`
    must then be `symmetric`."""
    if mode not in MODES:
        raise ValueError(f"unknown QAIRT range mode {mode!r}; use one of {', '.join(MODES)}")
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no calibration data")
    if not np.all(np.isfinite(x)):
        raise ValueError("calibration data has NaN or inf")
    if axis is not None:
        if mode != "symmetric":
            raise ValueError("per channel encodings are symmetric in QNN (offset -2^(bw-1))")
        moved = np.moveaxis(x, axis, 0).reshape(x.shape[axis], -1)
        return [encoding(row, bw, "symmetric", ties=ties, signed=signed) for row in moved]
    lo, hi = float(x.min()), float(x.max())
    if mode == "tf":
        return from_range(lo, hi, bw, ties=ties, signed=signed, mode="tf")
    if mode == "symmetric":
        # The doc's procedure on (-a, a): scale 2a / (2^bw - 1), and min / scale is exactly
        # -(2^bw - 1) / 2, a tie that both tie rules send to -2^(bw-1). Written out directly:
        # computed in floating point the quotient can land a hair above the tie (-127.4999...)
        # and round to -127, which QNN's per channel rule (offset -2^(bw-1)) forbids.
        a = max(abs(lo), abs(hi), MIN_RANGE / 2)
        n = (1 << bw) - 1
        scale = 2.0 * a / n
        off = -(1 << (bw - 1))
        return Encoding(bw, off * scale, (off + n) * scale, scale, off, "symmetric", ties, signed)
    if mode == "adjusted":
        return _adjusted(lo, hi, bw, ties, signed)
    return _enhanced(x, bw, ties, signed)


def candidates(calibration, bw: int = 8, signed: bool = False,
               modes: Sequence[str] = MODES, ties: Sequence[str] = TIES) -> Dict[str, Encoding]:
    """Every (mode, tie rule) encoding for this calibration data, keyed 'mode/ties'."""
    return {f"{m}/{t}": encoding(calibration, bw, m, ties=t, signed=signed) for m in modes for t in ties}


# ----------------------------------------------------------------------------- device hook

@dataclass
class ModeScore:
    name: str  # "tf/half_away"
    documented: bool
    encoding: Dict[str, Any]
    grid_match: bool  # the device's values lie on this encoding's grid
    equal: int
    total: int
    max_lsb: int

    @property
    def rate(self) -> float:
        return self.equal / self.total if self.total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "documented": self.documented, "encoding": self.encoding,
                "grid_match": self.grid_match, "equal": self.equal, "total": self.total,
                "rate": self.rate, "max_lsb": self.max_lsb}


@dataclass
class Attribution:
    code: str  # TS1130, TS1131, TS1132 or TS1133
    verdict: str
    best: Optional[str]
    scores: List[ModeScore] = field(default_factory=list)
    device_step: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "verdict": self.verdict, "best": self.best,
                "device_step": self.device_step, "scores": [s.as_dict() for s in self.scores]}


def device_step(values, tol: float = 0.02) -> Optional[float]:
    """The grid step of dequantized device values (None when they are not on one grid)."""
    v = np.unique(np.asarray(values, dtype=np.float64).ravel())
    if v.size < 2:
        return None
    d = np.diff(v)
    s = float(d[d > 1e-12].min()) if np.any(d > 1e-12) else None
    if not s:
        return None
    k = (v - v[0]) / s
    return s if np.abs(k - np.rint(k)).max() <= tol else None


def attribute(reference, device, *, calibration=None, bw: int = 8, signed: bool = False,
              device_codes: bool = False, encoding_hint: Optional[Dict[str, Any]] = None,
              modes: Sequence[str] = MODES) -> Attribution:
    """Which QAIRT range mode and tie rule explains a device's quantized tensor.

    reference: the float values that were quantized (float64 reference of that tensor).
    device: the device output for the same tensor, dequantized floats (default) or integer codes
      (`device_codes=True`, then `encoding_hint` must give the device's scale and offset, or the
      codes are compared with each mode's codes directly).
    calibration: the calibration data the encoding was computed from (default: reference).

    Verdicts: TS1130 one mode/tie rule reproduces every element; TS1131 the device's grid is one
    mode's grid but some elements differ by exactly one step (a tie rule or input rounding
    difference, not a range mode or kernel bug); TS1132 no mode's grid matches the device's (the
    encoding came from somewhere else: other calibration data, an override file, or a kernel);
    TS1133 the device values are not on any grid, so there is nothing to attribute."""
    ref = np.asarray(reference, dtype=np.float64)
    dev = np.asarray(device)
    cal = ref if calibration is None else np.asarray(calibration, dtype=np.float64)
    if dev.shape != ref.shape:
        raise ValueError(f"device shape {dev.shape} differs from reference shape {ref.shape}")
    step = None
    if not device_codes:
        step = device_step(dev)
        if step is None:
            return Attribution("TS1133", "the device values are not on one quantization grid; no "
                                         "range mode can be attributed", None, [], None)
    scores: List[ModeScore] = []
    for name, enc in candidates(cal, bw, signed, modes).items():
        pred = enc.quantize(ref)
        if device_codes:
            got = dev.astype(np.int64)
            grid = True
            if encoding_hint is not None:
                grid = (abs(float(encoding_hint["scale"]) - enc.scale) <= 1e-6 * enc.scale
                        and int(encoding_hint["offset"]) == enc.offset)
        else:
            grid = abs(step - enc.scale) <= 1e-4 * enc.scale
            raw = dev.astype(np.float64) / enc.scale - enc.offset
            grid = grid and bool(np.abs(raw - np.rint(raw)).max() <= 0.02)
            got = np.rint(raw).astype(np.int64) - ((1 << (bw - 1)) if signed else 0)
        diff = np.abs(got - pred)
        scores.append(ModeScore(name, DOCUMENTED[name.split("/")[0]], enc.as_dict(), bool(grid),
                                int((diff == 0).sum()), int(diff.size), int(diff.max()) if diff.size else 0))
    scores.sort(key=lambda s: (not s.grid_match, -s.equal, not s.documented, s.name))
    best = scores[0]
    same = [s.name for s in scores if s.grid_match and s.equal == best.equal]
    if best.grid_match and best.equal == best.total:
        names = ", ".join(same)
        return Attribution("TS1130", f"reproduced bit for bit by QAIRT range mode {names}"
                                     + ("" if best.documented else " (an undocumented mode, "
                                        "Truestar's reading of it)"), best.name, scores, step)
    if best.grid_match and best.max_lsb <= 1:
        return Attribution(
            "TS1131", f"the device uses the grid of {best.name}; {best.total - best.equal} of "
                      f"{best.total} elements differ by exactly one step (a tie or input rounding "
                      "difference inside the same encoding, not a kernel bug)", best.name, scores, step)
    if not any(s.grid_match for s in scores):
        return Attribution(
            "TS1132", "no QAIRT range mode computed from this calibration data gives the device's "
                      f"grid (device step {step!r}); the encoding came from other data, an override "
                      "file or a different quantizer", None, scores, step)
    return Attribution(
        "TS1132", f"the device uses the grid of {best.name} but {best.total - best.equal} of "
                  f"{best.total} elements differ by up to {best.max_lsb} steps: more than a range "
                  "mode or tie rule explains (look at the kernel before the quantizer)",
        best.name, scores, step)


# ----------------------------------------------------------------------------- model simulation

@dataclass
class SimConfig:
    act_mode: str = "tf"
    param_mode: str = "tf"
    act_bw: int = 8
    param_bw: int = 8
    per_channel: bool = False  # symmetric per output channel weights for conv and linear
    ties: str = "half_away"
    float16_modules: Tuple[str, ...] = ()  # QAIRT float fallback: these modules compute in float16

    def as_dict(self) -> Dict[str, Any]:
        return dict(vars(self), float16_modules=list(self.float16_modules))


def _leaf_modules(model) -> List[Tuple[str, Any]]:
    return [(n, m) for n, m in model.named_modules() if n and not list(m.children())]


def calibrate(model, calibration_inputs: Iterable[Sequence[Any]]) -> Dict[str, Tuple[float, float]]:
    """Float64 min and max of every leaf module's output (and of the model input, key '')."""
    import torch

    model = model.double().eval()
    ranges: Dict[str, List[float]] = {}

    def put(key, t):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            r = ranges.setdefault(key, [np.inf, -np.inf])
            r[0] = min(r[0], float(t.min()))
            r[1] = max(r[1], float(t.max()))

    hooks = [m.register_forward_hook(lambda mod, a, o, n=n: put(n, o)) for n, m in _leaf_modules(model)]
    try:
        with torch.no_grad():
            for args in calibration_inputs:
                args = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args]
                for a in args:
                    put("", a)
                model(*args)
    finally:
        for h in hooks:
            h.remove()
    return {k: (v[0], v[1]) for k, v in ranges.items()}


def simulate(model, calibration_inputs, inputs: Sequence[Any], config: Optional[SimConfig] = None
             ) -> Dict[str, Any]:
    """Predicted outputs of the quantized model, in float64: the input, every leaf module's output
    and every conv or linear weight fake quantized under QAIRT range modes. Returns
    {"output": tensor, "taps": {module: tensor}, "encodings": {...}, "config": {...}}.

    Bias is left in float64 (QNN stores it in 32 bits, far below the activation step). Modules in
    float16_modules run with float16 rounded inputs, weights and outputs instead (the float fallback
    and FP16 override modes). The prediction is the simulation's, never checked against a device
    here; `attribute` is how a device run is compared with it."""
    import copy

    import torch

    cfg = config or SimConfig()
    model = copy.deepcopy(model).double().eval()
    ranges = calibrate(model, calibration_inputs)
    enc_act = {k: from_range(lo, hi, cfg.act_bw, ties=cfg.ties, mode=cfg.act_mode)
               if cfg.act_mode in ("tf",) else None for k, (lo, hi) in ranges.items()}
    if cfg.act_mode != "tf":
        # modes other than tf need the data, not only the range: collect it once more
        data: Dict[str, List[np.ndarray]] = {}

        def grab(key, t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                data.setdefault(key, []).append(t.detach().numpy().ravel())

        hs = [m.register_forward_hook(lambda mod, a, o, n=n: grab(n, o)) for n, m in _leaf_modules(model)]
        try:
            with torch.no_grad():
                for args in calibration_inputs:
                    args = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args]
                    for a in args:
                        grab("", a)
                    model(*args)
        finally:
            for h in hs:
                h.remove()
        enc_act = {k: encoding(np.concatenate(v), cfg.act_bw, cfg.act_mode, ties=cfg.ties)
                   for k, v in data.items()}
    encs: Dict[str, Any] = {"activations": {k: e.as_dict() for k, e in enc_act.items() if e is not None},
                            "params": {}}
    f16 = set(cfg.float16_modules)
    with torch.no_grad():
        for name, m in _leaf_modules(model):
            w = getattr(m, "weight", None)
            if not isinstance(w, torch.Tensor) or w.dim() < 2:
                continue
            if name in f16:
                w.copy_(w.half().double())
                continue
            wn = w.detach().numpy()
            if cfg.per_channel:
                per = encoding(wn, cfg.param_bw, "symmetric", ties=cfg.ties, signed=True, axis=0)
                fq = np.stack([e.fake(row) for e, row in zip(per, wn.reshape(wn.shape[0], -1))])
                encs["params"][name + ".weight"] = [e.as_dict() for e in per]
            else:
                e = encoding(wn, cfg.param_bw, cfg.param_mode, ties=cfg.ties, signed=True)
                fq = e.fake(wn)
                encs["params"][name + ".weight"] = e.as_dict()
            w.copy_(torch.from_numpy(fq.reshape(wn.shape)))
    taps: Dict[str, Any] = {}

    def fq_t(key, t):
        e = enc_act.get(key)
        if e is None or not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return t
        return torch.from_numpy(e.fake(t.detach().numpy()))

    def hook(mod, args, out, name):
        if name in f16:
            out = out.half().double()
        else:
            out = fq_t(name, out)
        taps[name] = out
        return out

    def pre(mod, args, name):
        if name in f16:
            return tuple(a.half().double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                         for a in args)
        return None

    hooks = []
    for n, m in _leaf_modules(model):
        hooks.append(m.register_forward_pre_hook(lambda mod, a, n=n: pre(mod, a, n)))
        hooks.append(m.register_forward_hook(lambda mod, a, o, n=n: hook(mod, a, o, n)))
    try:
        with torch.no_grad():
            args = [fq_t("", a.double()) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                    for a in inputs]
            out = model(*args)
    finally:
        for h in hooks:
            h.remove()
    return {"output": out, "taps": taps, "encodings": encs, "config": cfg.as_dict()}


def compare_to_device(reference_taps: Dict[str, Any], device_taps: Dict[str, Any], *,
                      calibration_taps: Optional[Dict[str, Any]] = None, bw: int = 8,
                      signed: bool = False) -> Dict[str, Dict[str, Any]]:
    """Hook for real device runs (for example a QNN job's per layer outputs): `attribute` per tap.

    reference_taps: float64 values of each tapped tensor (module name -> array), the values the
    device quantized. device_taps: the device's dequantized outputs for the same names.
    calibration_taps: the calibration data per name (default: the reference itself). Returns
    {name: Attribution.as_dict()} for every name present in both."""
    out: Dict[str, Dict[str, Any]] = {}
    for k in reference_taps:
        if k not in device_taps:
            continue
        cal = None if calibration_taps is None else calibration_taps.get(k)
        out[k] = attribute(np.asarray(reference_taps[k], dtype=np.float64), np.asarray(device_taps[k]),
                           calibration=cal, bw=bw, signed=signed).as_dict()
    return out
