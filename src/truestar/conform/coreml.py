"""`truestar conform coreml DIR`: judge saved Core ML runs of a fixed battery (docs/COREML.md).

Two halves. ``collect(DIR)`` runs on a Mac with coremltools: it converts the battery (an MLP, a
small conv net, a transformer encoder layer and the MLP with int8 weights) to ML programs at
FLOAT32 and FLOAT16 compute precision, runs each with the compute units CPU_ONLY, CPU_AND_GPU,
CPU_AND_NE and ALL, and saves every output (``<run>.npz``), the run to run bit check, the latency
and Core ML's own compute plan (``MLComputePlan``: the device each operation is planned on and the
devices that support it) into ``summary.json``. ``check(DIR)`` runs anywhere with torch: it
regenerates the inputs and weights (portable draws, hashed), recomputes float64, float32,
``sim:ieee-f16`` and ``sim:mali-fp16`` references itself and gives one verdict per run.

Accuracy uses the rule of ``truestar conform qaihub``: within ``F32_FACTOR`` times the float32
error (or ``F32_FLOOR``) is float32, within ``F16_FACTOR`` times ``sim:ieee-f16`` is float16, above
is beyond correct float16 (TS1163), with the ratio to ``sim:mali-fp16`` (a float16 accumulator).
Placement: an accelerator asked for while Core ML plans operations on the CPU is a silent
fallback (TS1160, and TS1161 when the requested unit does not support the operations at the run's
precision, as the Neural Engine with a FLOAT32 program). Int8 runs are judged against the float64
model with the dequantized weights, so the weight rounding itself is not counted as an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

BATTERY_VERSION = 1
SEED = 1160
MODELS = ("mlp", "convnet", "encoder", "mlp_int8")
PRECISIONS = ("FLOAT32", "FLOAT16")
UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")
WANTED = {"CPU_ONLY": {"CPU"}, "CPU_AND_GPU": {"GPU"}, "CPU_AND_NE": {"NE"}, "ALL": {"GPU", "NE"}}
F32_FACTOR = 8.0
F32_FLOOR = 1e-5
F16_FACTOR = 4.0
STRUCTURAL_OPS = {"const", "reshape", "transpose", "expand_dims", "squeeze", "slice_by_index", "concat",
                  "identity", "cast"}


class CoreMLBundleError(Exception):
    pass


# ----------------------------------------------------------------------------- the battery

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build(model: str):
    """(module, input) of a battery model, both from portable draws (bitwise the same on every OS)."""
    torch, nn, F = _torch()
    from ..fingerprint.models import portable_init, portable_normal

    if model in ("mlp", "mlp_int8"):
        m = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 16))
        x = portable_normal(SEED, 1, (128, 1024))
    elif model == "convnet":
        m = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                          nn.Linear(64, 16))
        x = portable_normal(SEED, 2, (8, 3, 128, 128))
    elif model == "encoder":
        m = _StaticEncoder(nn.TransformerEncoderLayer(256, 8, 1024, dropout=0.0, batch_first=True), 4, 128)
        x = portable_normal(SEED, 3, (4, 128, 256))
    else:
        raise CoreMLBundleError(f"unknown battery model {model!r} (models: {', '.join(MODELS)})")
    portable_init(m, SEED + MODELS.index(model) if model != "mlp_int8" else SEED)
    m.eval()
    if model == "mlp_int8":
        with torch.no_grad():
            for layer in m:
                if isinstance(layer, nn.Linear):
                    layer.weight.copy_(dequantized_int8(layer.weight))
    return m, x


def dequantized_int8(w):
    """Per output channel symmetric int8 (Core ML's linear_symmetric): scale max|w| / 127 per row,
    codes round half to even clipped to [-127, 127], back to float32."""
    torch, _, _ = _torch()
    w64 = w.detach().double()
    scale = (w64.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) / 127.0).float().double()
    q = torch.clamp(torch.round(w64 / scale), -127, 127)
    return (q * scale).float()


def _static_encoder_cls():
    torch, nn, F = _torch()

    class StaticEncoder(nn.Module):
        """nn.TransformerEncoderLayer (post norm, relu) with constant shapes: the trace has no shape
        arithmetic, which coremltools 9.0 cannot convert under numpy 2.5."""

        def __init__(self, layer, B, T):
            super().__init__()
            self.l, self.B, self.T = layer, B, T
            self.d, self.h = layer.self_attn.embed_dim, layer.self_attn.num_heads

        def forward(self, x):
            l, B, T, d, h = self.l, self.B, self.T, self.d, self.h
            hd = d // h
            qkv = F.linear(x, l.self_attn.in_proj_weight, l.self_attn.in_proj_bias)
            q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
            q = q.reshape(B, T, h, hd).permute(0, 2, 1, 3)
            k = k.reshape(B, T, h, hd).permute(0, 2, 1, 3)
            v = v.reshape(B, T, h, hd).permute(0, 2, 1, 3)
            a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5), dim=-1)
            a = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(B, T, d)
            x = l.norm1(x + l.self_attn.out_proj(a))
            return l.norm2(x + l.linear2(torch.relu(l.linear1(x))))

    return StaticEncoder


def _StaticEncoder(layer, B, T):
    return _static_encoder_cls()(layer, B, T)


def input_hash(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(x.numpy()).tobytes()).hexdigest()


def references(model: str) -> Dict[str, np.ndarray]:
    """float64, float32, sim:ieee-f16 and sim:mali-fp16 outputs of a battery model."""
    torch, _, _ = _torch()
    from ..devsim import simulate

    m, x = build(model)
    out = {}
    with torch.no_grad():
        out["f64"] = m.double()(x.double()).numpy()
        m.float()
        out["f32"] = m(x).double().numpy()
        for key, prof in (("sim16", "sim:ieee-f16"), ("simacc", "sim:mali-fp16")):
            with simulate(prof):
                out[key] = m(x).double().numpy()
    return out


# ----------------------------------------------------------------------------- collect (macOS)

def _device_name(d) -> str:
    from coremltools.models import compute_device as cd
    if isinstance(d, cd.MLNeuralEngineComputeDevice):
        return "NE"
    if isinstance(d, cd.MLGPUComputeDevice):
        return "GPU"
    if isinstance(d, cd.MLCPUComputeDevice):
        return "CPU"
    return "none" if d is None else type(d).__name__


def _plan(compiled: str, units) -> List[Dict]:
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(compiled, units)
    ops: List[Dict] = []

    def walk(block):
        for op in block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is not None:
                ops.append({"op": op.operator_name.split(".")[-1], "preferred": _device_name(u.preferred_compute_device),
                            "supported": sorted({_device_name(d) for d in u.supported_compute_devices})})
            for b in getattr(op, "blocks", None) or []:
                walk(b)
    walk(plan.model_structure.program.functions["main"].block)
    return ops


def collect(path: str, models=MODELS, precisions=PRECISIONS, units=UNITS, reps: int = 30, log=print) -> str:
    """Convert, run and save the battery on this Mac. Needs macOS and coremltools."""
    import coremltools as ct
    torch, _, _ = _torch()

    os.makedirs(path, exist_ok=True)
    s = {"battery": "coreml", "battery_version": BATTERY_VERSION, "seed": SEED, "coremltools": ct.__version__,
         "input_hashes": {}, "runs": {}}
    try:
        import platform
        s["macos"] = platform.mac_ver()[0]
        s["chip"] = os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
    except Exception:  # informational only
        pass
    for model in models:
        m, x = build(model)
        s["input_hashes"][model] = input_hash(x)
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            traced = torch.jit.trace(m, x)
        for prec in precisions:
            base = f"{model}_{prec.lower()}"
            try:
                ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=tuple(x.shape))], convert_to="mlprogram",
                                compute_precision=getattr(ct.precision, prec),
                                minimum_deployment_target=ct.target.macOS15)
                if model == "mlp_int8":
                    from coremltools.optimize.coreml import (OpLinearQuantizerConfig, OptimizationConfig,
                                                             linear_quantize_weights)
                    ml = linear_quantize_weights(ml, OptimizationConfig(global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric", dtype="int8")))
                pkg = os.path.join(path, base + ".mlpackage")
                ml.save(pkg)
                compiled = str(ct.utils.compile_model(pkg, os.path.join(path, base + ".mlmodelc")))
            except Exception as exc:  # recorded per run, never hidden
                for u in units:
                    s["runs"][f"{base}_{u.lower()}"] = {"model": model, "precision": prec, "units": u,
                                                        "error": f"convert: {type(exc).__name__}: {exc}"[:600]}
                continue
            for u in units:
                name = f"{base}_{u.lower()}"
                row = {"model": model, "precision": prec, "units": u}
                cu = getattr(ct.ComputeUnit, u)
                try:
                    mm = ct.models.CompiledMLModel(compiled, compute_units=cu)
                    feed = {"x": x.numpy()}
                    first = mm.predict(feed)
                    key = sorted(first)[0]
                    y = np.asarray(first[key])
                    again = np.asarray(mm.predict(feed)[key])
                    row["run_to_run_equal"] = bool(np.array_equal(y, again))
                    for _ in range(5):
                        mm.predict(feed)
                    ts = []
                    for _ in range(reps):
                        t0 = time.perf_counter()
                        mm.predict(feed)
                        ts.append((time.perf_counter() - t0) * 1e3)
                    row["median_ms"] = round(statistics.median(ts), 4)
                    np.savez(os.path.join(path, name + ".npz"), y=y)
                except Exception as exc:
                    row["error"] = f"run: {type(exc).__name__}: {exc}"[:600]
                try:
                    row["ops"] = _plan(compiled, cu)
                except Exception as exc:
                    row["plan_error"] = f"{type(exc).__name__}: {exc}"[:300]
                s["runs"][name] = row
                if log:
                    log(f"{name}: " + (row.get("error") or f"{row.get('median_ms')} ms"))
    with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)
    return path


# ----------------------------------------------------------------------------- check (anywhere)

@dataclass
class RunVerdict:
    run: str
    model: str
    precision: str
    units: str
    verdict: str          # float32, float16, beyond_float16, failed
    metrics: Dict = field(default_factory=dict)
    placement: Dict = field(default_factory=dict)
    issues: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self):
        return {"run": self.run, "model": self.model, "precision": self.precision, "units": self.units,
                "verdict": self.verdict, "metrics": self.metrics, "placement": self.placement,
                "issues": [{"code": c, "message": m} for c, m in self.issues]}


def load_summary(path: str) -> Dict:
    f = os.path.join(path, "summary.json") if os.path.isdir(path) else path
    try:
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CoreMLBundleError(f"{f}: {type(exc).__name__}: {exc}") from None
    if not isinstance(s, dict) or s.get("battery") != "coreml" or "runs" not in s:
        raise CoreMLBundleError(f"{f}: not a Core ML battery summary (battery 'coreml' with runs)")
    if s.get("battery_version") != BATTERY_VERSION or s.get("seed") != SEED:
        raise CoreMLBundleError(f"{f}: battery version {s.get('battery_version')} seed {s.get('seed')}, this "
                                f"truestar has {BATTERY_VERSION} seed {SEED}")
    return s


def _rel(a: np.ndarray, ref: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(ref.shape)
    return float(np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-300))


def placement(row: Dict) -> Dict:
    ops = [o for o in row.get("ops") or [] if o.get("op") not in STRUCTURAL_OPS]
    counts: Dict[str, int] = {}
    for o in ops:
        counts[o["preferred"]] = counts.get(o["preferred"], 0) + 1
    want = WANTED.get(row.get("units", ""), set())
    off = [o for o in ops if o["preferred"] not in want] if row.get("units") != "CPU_ONLY" else []
    return {"planned": counts, "compute_ops": len(ops), "off_requested": len(off),
            "off_but_supported_there": sum(1 for o in off if want & set(o.get("supported", []))),
            "off_ops": sorted({o["op"] for o in off})}


def judge(name: str, row: Dict, y: np.ndarray, refs: Dict[str, np.ndarray]) -> RunVerdict:
    rv = RunVerdict(name, row["model"], row["precision"], row["units"], "float32")
    f64 = refs["f64"]
    if y.size != f64.size:
        rv.verdict = "failed"
        rv.issues.append(("TS1164", f"{name}: output of {y.size} elements, the battery's has {f64.size}"))
        return rv
    e = _rel(y, f64)
    e32, e16, eacc = (_rel(refs[k], f64) for k in ("f32", "sim16", "simacc"))
    rv.metrics = {"rel_err": e, "float32_pred": e32, "sim_ieee_f16_pred": e16, "sim_mali_fp16_pred": eacc,
                  "ratio_to_sim_f16": e / e16 if e16 else float("inf"), "nonfinite": int(np.sum(~np.isfinite(y))),
                  "median_ms": row.get("median_ms"), "run_to_run_equal": row.get("run_to_run_equal")}
    if rv.metrics["nonfinite"] or not np.isfinite(e) or e > F16_FACTOR * e16:
        rv.verdict = "beyond_float16"
    elif e > max(F32_FACTOR * e32, F32_FLOOR):
        rv.verdict = "float16"
    if rv.verdict == "beyond_float16":
        rv.issues.append(("TS1163", f"{name}: error {e:.2e} is {e / e16:.1f}x correct float16 (sim:ieee-f16 {e16:.2e}); "
                                    f"{e / eacc:.2f}x sim:mali-fp16 (a float16 accumulator)"))
    elif rv.verdict == "float16" and rv.precision == "FLOAT32":
        rv.issues.append(("TS1162", f"{name}: a FLOAT32 program ran at float16 level ({e:.2e}, float32 predicts {e32:.2e})"))
    if row.get("run_to_run_equal") is False:
        rv.issues.append(("TS1167", f"{name}: two identical predictions gave different bits"))
    p = placement(row)
    rv.placement = p
    if "ops" not in row:
        rv.issues.append(("TS1165", f"{name}: no compute plan saved ({row.get('plan_error', 'not collected')})"))
    elif p["off_requested"]:
        where = ", ".join(f"{k} {v}" for k, v in sorted(p["planned"].items()))
        if p["off_but_supported_there"] == 0:
            rv.issues.append(("TS1161", f"{name}: asked for {row['units']} but none of the {p['off_requested']} operations "
                                        f"planned elsewhere is supported there at {row['precision']} (planned: {where})"))
        else:
            rv.issues.append(("TS1160", f"{name}: asked for {row['units']} but Core ML planned {p['off_requested']} of "
                                        f"{p['compute_ops']} operations elsewhere, {p['off_but_supported_there']} of them "
                                        f"supported on the requested unit (planned: {where}; ops: {', '.join(p['off_ops'])})"))
    return rv


def check(path: str, runs: Optional[List[str]] = None):
    """Returns (run verdicts, top level issues). Raises CoreMLBundleError for an unreadable bundle."""
    s = load_summary(path)
    root = path if os.path.isdir(path) else os.path.dirname(path)
    names = runs or sorted(s["runs"])
    for n in names:
        if n not in s["runs"]:
            raise CoreMLBundleError(f"run {n!r} not in the bundle (has {', '.join(sorted(s['runs']))})")
    top, out, refs, ok = [], [], {}, {}
    for model, want in (s.get("input_hashes") or {}).items():
        if model not in MODELS:
            raise CoreMLBundleError(f"unknown battery model {model!r} in the summary")
        have = input_hash(build(model)[1])
        ok[model] = want == have
        if want != have:
            top.append(("TS1166", f"{model}: the run's input hash {str(want)[:12]} is not this battery's {have[:12]}; "
                                  "its runs are not judged"))
    for n in names:
        row = s["runs"][n]
        model = row.get("model")
        if model not in MODELS:
            raise CoreMLBundleError(f"run {n!r}: unknown model {model!r}")
        if not ok.get(model, False):
            continue
        if row.get("error"):
            rv = RunVerdict(n, model, row.get("precision", ""), row.get("units", ""), "failed")
            rv.issues.append(("TS1164", f"{n}: {str(row['error'])[:300]}"))
            out.append(rv)
            continue
        f = os.path.join(root, n + ".npz")
        try:
            with np.load(f, allow_pickle=False) as z:
                y = np.asarray(z["y"])
        except (OSError, ValueError, KeyError) as exc:
            raise CoreMLBundleError(f"{f}: {type(exc).__name__}: {exc}") from None
        if model not in refs:
            refs[model] = references(model)
        out.append(judge(n, row, y, refs[model]))
    return out, top


ERROR_CODES = {"TS1163", "TS1164", "TS1166"}


def is_finding(verdicts, top) -> bool:
    return bool(top) or any(c in ERROR_CODES for v in verdicts for c, _ in v.issues)


def to_text(verdicts, top) -> str:
    lines = ["truestar conform coreml: saved Core ML runs against float64, float32, sim:ieee-f16 and the "
             "compute plan's placement"]
    for c, msg in top:
        lines.append(f"  {c} {msg}")
    for v in verdicts:
        p = v.placement or {}
        where = " ".join(f"{k}:{n}" for k, n in sorted((p.get("planned") or {}).items()))
        if v.metrics:
            lines.append(f"  {v.run:34s} {v.verdict:15s} err {v.metrics['rel_err']:.2e} "
                         f"({v.metrics['ratio_to_sim_f16']:.2f}x sim:ieee-f16)  [{where}]")
        else:
            lines.append(f"  {v.run:34s} {v.verdict}")
        for c, msg in v.issues:
            lines.append(f"    {c} {msg}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m truestar.conform.coreml",
                                 description="collect the Core ML battery on a Mac (then: truestar conform coreml DIR)")
    ap.add_argument("action", choices=("collect",))
    ap.add_argument("dir")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args(argv)
    collect(a.dir, models=tuple(a.models.split(",")), reps=a.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
