"""`truestar conform qaihub DIR`: judge saved Qualcomm AI Hub outputs of the fixed battery (docs/QAIHUB.md).

No network. DIR holds what a run on hosted devices downloaded: ``summary.json`` (battery version,
input hashes, one entry per run with its device, runtime, compute unit and per layer placement)
and one ``<run>.npz`` per run with every output tensor. This side regenerates the inputs, checks
the hashes, recomputes every reference itself and gives one verdict per run and tap.

Float runs. Each tap's relative L2 error against float64 is set against two predictions computed
here from the same model: float32 eager PyTorch (what a correct float32 device gives) and
``sim:ieee-f16`` (correct float16 arithmetic: float16 storage, wide accumulation). A tap within
``F32_FACTOR`` times the float32 error (or 1e-5) ran in float32; one within ``F16_FACTOR`` times
the ``sim:ieee-f16`` error ran in float16 as simulated (TS923 on a float32 model); anything above
is beyond correct float16 (TS924) and the report gives the ratio to ``sim:mali-fp16`` (a float16
accumulator), the next simulated explanation.

INT8 runs. Every tap is a requantized layer output, so its values are integers times a known
step. Each layer is replayed with ``truestar.intref`` from the device's own previous tap (the
battery input for the first layer), so a layer is judged on its own and not on inherited drift:
the share of elements one step away from the intended arithmetic (ONNX QLinear: exact integer
accumulation, float32 requantization, ties to even) is TS926, any element further away TS927, and
the intref convention that matches the layer best is named (TS928).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import qaihub_battery as B

F32_FACTOR = 8.0
F32_FLOOR = 1e-5
F16_FACTOR = 4.0
CONVENTIONS = ("onnx_qlinear", "tflite_ref", "exact_half_away", "npu_fp16_input")
ACCELERATOR_UNITS = ("npu", "gpu")


class QaiHubBundleError(Exception):
    pass


def rel_l2(a: np.ndarray, ref: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(ref.shape)
    n = float(np.linalg.norm(ref))
    return float(np.linalg.norm(a - ref)) / n if n > 0 else float(np.linalg.norm(a - ref))


@dataclass
class TapVerdict:
    tap: str
    verdict: str  # float32, float16, beyond_float16, exact, one_step, divergent, missing
    metrics: Dict = field(default_factory=dict)

    def as_dict(self):
        return {"tap": self.tap, "verdict": self.verdict, "metrics": self.metrics}


@dataclass
class RunVerdict:
    run: str
    model: str
    device: str
    runtime: str
    unit: str
    verdict: str  # float32, float16, beyond_float16, int8_exact, int8_one_step, divergent, failed
    taps: List[TapVerdict] = field(default_factory=list)
    issues: List[tuple] = field(default_factory=list)
    layer_units: Dict = field(default_factory=dict)

    def as_dict(self):
        return {"run": self.run, "model": self.model, "device": self.device, "runtime": self.runtime,
                "unit": self.unit, "verdict": self.verdict, "layer_units": self.layer_units,
                "taps": [t.as_dict() for t in self.taps],
                "issues": [{"code": c, "message": m} for c, m in self.issues]}


def load_summary(path: str) -> Dict:
    f = os.path.join(path, "summary.json") if os.path.isdir(path) else path
    try:
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
    except (OSError, ValueError) as exc:
        raise QaiHubBundleError(f"{f}: {type(exc).__name__}: {exc}") from None
    if not isinstance(s, dict) or "runs" not in s or "input_hashes" not in s:
        raise QaiHubBundleError(f"{f}: not an AI Hub battery summary (no runs or input_hashes)")
    if s.get("battery_version") != B.BATTERY_VERSION or s.get("seed") != B.SEED:
        raise QaiHubBundleError(f"{f}: battery version {s.get('battery_version')} seed {s.get('seed')}, "
                                f"this truestar has {B.BATTERY_VERSION} seed {B.SEED}")
    return s


def write(path: str, runs: Dict[str, tuple]) -> str:
    """Write a results directory: runs maps a run name to (row, outputs or None). A row has model
    ("float" or "int8"), device, runtime, unit, optional layer_units and, for a failed job, error."""
    os.makedirs(path, exist_ok=True)
    s = {"battery": "qaihub", "battery_version": B.BATTERY_VERSION, "seed": B.SEED,
         "input_hashes": {m: B.input_hash(B.inputs(m)) for m in ("float", "int8")}, "runs": {}}
    for name, (row, outs) in runs.items():
        s["runs"][name] = dict(row)
        if outs is not None:
            np.savez(os.path.join(path, name + ".npz"), **{k: np.asarray(v) for k, v in outs.items()})
    with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)
    return path


def match_outputs(got: Dict[str, np.ndarray], model: str, shapes: Dict[str, tuple]) -> Dict[str, np.ndarray]:
    """Device outputs keyed by tap. Runtimes keep the ONNX output names (``mlp__out``) or rename
    them; a renamed output is matched by element count when that count is unique among the taps."""
    taps = B.tap_names(model)
    out = {}
    rest = {}
    for k, v in got.items():
        t = B.tap_of_output(k)
        if t in taps:
            out[t] = np.asarray(v)
        else:
            rest[k] = np.asarray(v)
    # AI Hub's TFLite and QNN targets name outputs output_0, output_1, ... in the ONNX order: map by
    # position when every remaining output has the size of the tap at its position
    todo = [t for t in taps if t not in out]
    names = sorted(rest, key=lambda k: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", k)])
    if todo and len(names) == len(todo) and all(rest[k].size == int(np.prod(shapes[t])) for k, t in zip(names, todo)):
        for k, t in zip(names, todo):
            out[t] = rest.pop(k)
    for t in taps:
        if t in out:
            continue
        n = int(np.prod(shapes[t]))
        same = [t2 for t2 in taps if int(np.prod(shapes[t2])) == n]
        cand = [k for k, v in rest.items() if v.size == n]
        if len(same) == 1 and len(cand) == 1:
            out[t] = rest.pop(cand[0])
    return out


def _to_shape(a: np.ndarray, shape) -> Optional[np.ndarray]:
    """Reshape a device tensor to the reference layout. TFLite may hand back NHWC for NCHW taps."""
    if a.size != int(np.prod(shape)):
        return None
    if a.shape == tuple(shape):
        return a.astype(np.float64)
    if len(shape) == 4 and a.ndim == 4 and a.shape == (shape[0], shape[2], shape[3], shape[1]):
        return a.transpose(0, 3, 1, 2).astype(np.float64)
    return a.reshape(shape).astype(np.float64)


class _Refs:
    """References computed once per check."""

    def __init__(self):
        self._c: Dict[str, object] = {}

    def get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]


def judge_float(name: str, row: Dict, got: Dict[str, np.ndarray], refs: _Refs) -> RunVerdict:
    arrs = refs.get("float_inputs", lambda: B.inputs("float"))
    f64 = refs.get("f64", lambda: B.float_reference(arrs))
    import torch

    f32 = refs.get("f32", lambda: B.float_reference(arrs, dtype=torch.float32))
    s16 = refs.get("sim16", lambda: B.simulated("sim:ieee-f16", arrs))
    acc16 = refs.get("simacc", lambda: B.simulated("sim:mali-fp16", arrs))
    rv = RunVerdict(name, "float", row.get("device", ""), row.get("runtime", ""), row.get("unit", ""), "float32",
                    layer_units=row.get("layer_units") or {})
    m = match_outputs(got, "float", {t: f64[t].shape for t in B.FLOAT_TAPS})
    order = {"float32": 0, "float16": 1, "beyond_float16": 2, "missing": 3}
    worst = "float32"
    for t in B.FLOAT_TAPS:
        a = _to_shape(m[t], f64[t].shape) if t in m else None
        if a is None:
            rv.taps.append(TapVerdict(t, "missing"))
            rv.issues.append(("TS929", f"{name}: tap {t} missing or of the wrong size"))
            worst = "missing"
            continue
        e = rel_l2(a, f64[t])
        e32, e16, eacc = rel_l2(f32[t], f64[t]), rel_l2(s16[t], f64[t]), rel_l2(acc16[t], f64[t])
        met = {"rel_l2": e, "float32_pred": e32, "sim_ieee_f16_pred": e16, "sim_mali_fp16_pred": eacc,
               "ratio_to_sim_f16": e / e16 if e16 else float("inf"),
               "max_abs_over_rms": float(np.max(np.abs(a - f64[t])) / max(np.sqrt(np.mean(f64[t] ** 2)), 1e-300)),
               "nonfinite": int(np.sum(~np.isfinite(a)))}
        if met["nonfinite"] or not np.isfinite(e):
            v = "beyond_float16"
        elif e <= max(F32_FACTOR * e32, F32_FLOOR):
            v = "float32"
        elif e <= F16_FACTOR * e16:
            v = "float16"
        else:
            v = "beyond_float16"
        rv.taps.append(TapVerdict(t, v, met))
        if order[v] > order[worst]:
            worst = v
    rv.verdict = worst if worst != "missing" else "divergent"
    beyond = [tv for tv in rv.taps if tv.verdict == "beyond_float16"]
    if beyond:
        b = beyond[0]
        rv.issues.append(("TS924", f"{name}: {len(beyond)} taps beyond correct float16, first {b.tap} at rel_l2 "
                                   f"{b.metrics['rel_l2']:.2e} = {b.metrics['ratio_to_sim_f16']:.1f}x sim:ieee-f16 "
                                   f"({b.metrics['rel_l2'] / max(b.metrics['sim_mali_fp16_pred'], 1e-300):.2f}x "
                                   "sim:mali-fp16, a float16 accumulator)"))
    elif worst == "float16":
        hi = max(rv.taps, key=lambda tv: tv.metrics.get("ratio_to_sim_f16", 0.0))
        rv.issues.append(("TS923", f"{name}: the float32 model ran at float16 level, within {F16_FACTOR:g}x of "
                                   f"sim:ieee-f16 on every tap (largest {hi.metrics['ratio_to_sim_f16']:.2f}x at {hi.tap})"))
    if worst == "float32" and rv.unit in ACCELERATOR_UNITS:
        units = ", ".join(f"{k} {v}" for k, v in sorted(rv.layer_units.items()) if v) or "no layer placement saved"
        rv.issues.append(("TS925", f"{name}: asked for {rv.unit} but every tap is float32 accurate (layers: {units})"))
    return rv


def _levels(x: np.ndarray, step: float, zp: int) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) / step).astype(np.int64) + zp


def judge_int8(name: str, row: Dict, got: Dict[str, np.ndarray], refs: _Refs) -> RunVerdict:
    from ..intref import ops as Q

    arrs = refs.get("int8_inputs", lambda: B.inputs("int8"))
    P = refs.get("int8_params", B.int8_params)
    intended = refs.get("int8_intended", lambda: B.int8_intended(arrs, params=P))
    rv = RunVerdict(name, "int8", row.get("device", ""), row.get("runtime", ""), row.get("unit", ""), "int8_exact",
                    layer_units=row.get("layer_units") or {})
    m = match_outputs(got, "int8", {t: intended[t].shape for t in B.INT8_TAPS})
    sx, zx = P["input"]["scale"], P["input"]["zp"]
    first_div = first_one = None
    prev_q = {c: Q.quantize(arrs["x_q"], {"scale": sx, "zp": zx, "out_signed": False}, c) for c in CONVENTIONS}
    s_in, z_in = sx, zx
    for t, L in zip(B.INT8_TAPS, P["layers"]):
        a = _to_shape(m[t], intended[t].shape) if t in m else None
        if a is None:
            rv.taps.append(TapVerdict(t, "missing"))
            rv.issues.append(("TS929", f"{name}: tap {t} missing or of the wrong size"))
            rv.verdict = "divergent"
            break
        step, zy = L["sy"], L["zy"]
        dev_q = _levels(a, step, zy)
        off_grid = float(np.max(np.abs(np.asarray(a, np.float64) / step - np.rint(np.asarray(a, np.float64) / step))))
        p = {"x_zp": z_in, "sx": s_in, "sw": L["sw"].astype(np.float64), "sy": step, "y_zp": zy, "out_signed": False}
        if L["relu"]:
            p["act_min"] = zy
        per_conv = {}
        for c in CONVENTIONS:
            want = Q.fully_connected(prev_q[c], L["w"], L["bias"], p, c)
            d = np.abs(dev_q - want)
            per_conv[c] = {"frac_off": float(np.mean(d > 0)), "max_steps": int(d.max())}
        main = per_conv["onnx_qlinear"]
        best = min(CONVENTIONS, key=lambda c: (per_conv[c]["max_steps"] > 1, per_conv[c]["frac_off"]))
        v = "exact" if main["max_steps"] == 0 else ("one_step" if main["max_steps"] == 1 else "divergent")
        chained = np.abs(dev_q - _levels(intended[t], step, zy))
        met = {"frac_off": main["frac_off"], "max_steps": main["max_steps"], "off_grid": off_grid,
               "chained_frac_off": float(np.mean(chained > 0)), "chained_max_steps": int(chained.max()),
               "conventions": per_conv, "best_convention": best}
        rv.taps.append(TapVerdict(t, v, met))
        if v == "divergent" and first_div is None:
            first_div = t
        if v == "one_step" and first_one is None:
            first_one = t
        # the next layer reads what the device produced
        prev_q = {c: dev_q for c in CONVENTIONS}
        s_in, z_in = step, zy
    if first_div:
        tv = next(x for x in rv.taps if x.tap == first_div)
        rv.issues.append(("TS927", f"{name}: layer {first_div} replayed from the device's own input has an element "
                                   f"{tv.metrics['max_steps']} steps from the intended arithmetic "
                                   f"({tv.metrics['frac_off']:.2%} of elements off)"))
        rv.verdict = "divergent"
    elif first_one:
        offs = ", ".join(f"{x.tap} {x.metrics['frac_off']:.2%}" for x in rv.taps if x.metrics)
        rv.issues.append(("TS926", f"{name}: elements one step from the intended arithmetic, per layer replayed from "
                                   f"the device's own input: {offs}"))
        if rv.verdict != "divergent":
            rv.verdict = "int8_one_step"
    for tv in rv.taps:
        if tv.metrics and tv.metrics["best_convention"] != "onnx_qlinear" and tv.metrics["frac_off"] > 0:
            b = tv.metrics["best_convention"]
            rv.issues.append(("TS928", f"{name}: {tv.tap} matches intref convention {b} best "
                                       f"({tv.metrics['conventions'][b]['frac_off']:.2%} off, onnx_qlinear "
                                       f"{tv.metrics['frac_off']:.2%})"))
    return rv


def check(path: str, runs: Optional[List[str]] = None):
    """Returns (run verdicts, top level issues). Raises QaiHubBundleError for an unreadable bundle."""
    s = load_summary(path)
    root = path if os.path.isdir(path) else os.path.dirname(path)
    names = runs or sorted(s["runs"])
    for n in names:
        if n not in s["runs"]:
            raise QaiHubBundleError(f"run {n!r} not in the bundle (has {', '.join(sorted(s['runs']))})")
    refs = _Refs()
    top, out = [], []
    hashes_ok = {}
    for model in ("float", "int8"):
        want = s["input_hashes"].get(model)
        if want is None:
            continue
        have = B.input_hash(B.inputs(model))
        hashes_ok[model] = want == have
        if want != have:
            top.append(("TS921", f"{model}: the run's input hash {str(want)[:12]} is not this battery's {have[:12]}; "
                                 "its runs are not judged"))
    for n in names:
        row = s["runs"][n]
        model = row.get("model", "")
        if model not in ("float", "int8"):
            raise QaiHubBundleError(f"run {n!r}: unknown model {model!r}")
        if not hashes_ok.get(model, False):
            continue
        base = RunVerdict(n, model, row.get("device", ""), row.get("runtime", ""), row.get("unit", ""), "failed")
        if row.get("error"):
            base.issues.append(("TS922", f"{n} on {base.device}: {str(row['error'])[:240]}"))
            out.append(base)
            continue
        f = os.path.join(root, n + ".npz")
        try:
            with np.load(f, allow_pickle=False) as z:
                got = {k: z[k] for k in z.files}
        except (OSError, ValueError) as exc:
            raise QaiHubBundleError(f"{f}: {type(exc).__name__}: {exc}") from None
        out.append(judge_float(n, row, got, refs) if model == "float" else judge_int8(n, row, got, refs))
    return out, top


ERROR_CODES = {"TS921", "TS922", "TS924", "TS927", "TS929"}


def is_finding(verdicts, top) -> bool:
    return bool(top) or any(c in ERROR_CODES for v in verdicts for c, _ in v.issues)


def to_text(verdicts, top) -> str:
    lines = ["truestar conform qaihub: saved AI Hub outputs against float64, float32 and sim:ieee-f16 "
             "(float) or the intended INT8 arithmetic replayed per layer"]
    for c, msg in top:
        lines.append(f"  {c} {msg}")
    for v in verdicts:
        where = f"{v.device} {v.runtime}/{v.unit}"
        units = " ".join(f"{k}:{n}" for k, n in sorted(v.layer_units.items()) if n)
        lines.append(f"  {v.run:34s} {v.verdict:15s} {where}" + (f"  [{units}]" if units else ""))
        for t in v.taps:
            mt = t.metrics
            if not mt:
                lines.append(f"      {t.tap:16s} {t.verdict}")
            elif v.model == "float":
                lines.append(f"      {t.tap:16s} {t.verdict:15s} rel_l2 {mt['rel_l2']:.2e}  "
                             f"{mt['ratio_to_sim_f16']:.2f}x sim:ieee-f16")
            else:
                lines.append(f"      {t.tap:16s} {t.verdict:15s} {mt['frac_off']:.2%} off (max {mt['max_steps']}), "
                             f"chained {mt['chained_frac_off']:.2%}, best {mt['best_convention']}")
        for c, msg in v.issues:
            lines.append(f"    {c} {msg}")
    return "\n".join(lines)
