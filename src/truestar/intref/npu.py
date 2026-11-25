"""Explain the 1 LSB differences of docs/NPU.md from the saved S24 results, with no device jobs.

Inputs are the INT8 tap runs already in the AI Hub job cache (`~/.cache/truestar/qai_hub`, or
TRUESTAR_CACHE): S24 NPU job jp0810ee5 and S24 CPU job j5w0d6nmp, 16 inputs, taps relu (conv1 +
bn1 + relu) and maxpool. The quantized model itself (scales, integer weights) was never saved on
this machine, so the conv1 parameters are unknown. Three steps:

1. Signature of NPU vs CPU at relu (`match.pair_signature`) and the rounding only test: can any
   pair of documented requantization rules produce it?
2. Max pool control: is the NPU's max pool exact on its own relu output? (max pool has no
   rounding, so it should be.)
3. Hypotheses, simulated with the integer reference on the same conv1 (BN folded, per channel
   symmetric int8 weights, the measured output step, a sweep of plausible input scales): each
   documented convention against the CPU's convention, and the float16 input quantization
   hypothesis. Each is scored by how close its signature is to the observed one.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .conventions import CONVENTIONS, documented_conventions
from .match import pair_signature, rounding_only_test
from .ops import conv2d, max_pool, quantize

NPU_JOB = "jp0810ee5"
CPU_JOB = "j5w0d6nmp"
INPUT_SCALES = (0.030, 0.0333, 0.03675, 0.040, 0.0421, 0.045)
INPUT_ZP = 126
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "data", "npu_s24_conv1.json")


def cache_dir(directory: Optional[str] = None) -> str:
    return directory or os.environ.get("TRUESTAR_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "truestar", "qai_hub")


def load_taps(job_id: str, directory: Optional[str] = None) -> List[np.ndarray]:
    """Every output of a cached inference job, inputs concatenated, in output order."""
    from truestar.remote.qai_hub_quant import outputs_in_order  # numpy only at import

    d = cache_dir(directory)
    with open(os.path.join(d, "hub_cache.json"), encoding="utf-8") as f:
        index = json.load(f)
    keys = [k for k, v in index.items() if v.get("job_id") == job_id]
    path = next((os.path.join(d, "outputs", k + ".npz") for k in keys
                 if os.path.exists(os.path.join(d, "outputs", k + ".npz"))), None)
    if path is None:
        raise FileNotFoundError(f"no saved outputs for job {job_id} under {d}")
    z = np.load(path, allow_pickle=False)
    out: Dict[str, List[np.ndarray]] = {}
    for k in sorted(z.files):
        name = k.split("::")[0]
        out.setdefault(name, []).append(z[k])
    return outputs_in_order(out)


def to_levels(x: np.ndarray, step: float) -> np.ndarray:
    q = np.rint(np.asarray(x, np.float64) / step)
    if np.abs(q * step - x).max() > step * 0.02:
        raise ValueError("tensor is not on the quantization grid")
    return q.astype(np.int64)


@dataclass
class Hypothesis:
    name: str
    documented: bool
    rate: List[float]  # per input scale
    by_magnitude: List[List[float]]
    minority: List[float]
    distance: float  # mean absolute gap to the observed per bin rates

    def as_dict(self) -> dict:
        return {"name": self.name, "documented": self.documented, "rate": self.rate,
                "by_magnitude": self.by_magnitude, "channel_minority": self.minority,
                "distance": self.distance}


@dataclass
class NpuExplanation:
    step: float
    signature: dict
    rounding_test: dict
    maxpool: dict
    hypotheses: List[Hypothesis] = field(default_factory=list)
    best: Optional[str] = None
    verdict: str = ""
    code: str = ""
    seconds: float = 0.0
    jobs_submitted: int = 0

    def as_dict(self) -> dict:
        return {"step": self.step, "signature": self.signature, "rounding_test": self.rounding_test,
                "maxpool": self.maxpool, "hypotheses": [h.as_dict() for h in self.hypotheses],
                "best": self.best, "verdict": self.verdict, "code": self.code,
                "seconds": self.seconds, "jobs_submitted": self.jobs_submitted}

    def to_text(self) -> str:
        s = self.signature
        lines = [
            "truestar intref npu: S24 NPU vs S24 CPU, INT8 relu tap (conv1 + bn1 + relu), 16 inputs",
            f"  output step {self.step:.6g}; {s['differ']} of {s['elements']} elements differ "
            f"({s['rate']:.2%}), {s['up']} up, {s['down']} down, worst {s['max_lsb']} LSB",
            "  rate by output level: " + ", ".join(
                f"{b['levels'][0]}..{b['levels'][1]}: {b['rate']:.1%}" for b in s["by_magnitude"]),
            f"  rounding only test: {self.rounding_test['reason']}",
            f"  max pool control: {self.maxpool['text']}",
            "  hypotheses simulated on conv1 (mean gap to the observed rates per level bin):",
        ]
        for h in self.hypotheses:
            lines.append(f"    {h.name:28s} rate {min(h.rate):.2%} to {max(h.rate):.2%}, gap "
                         f"{h.distance:.3f}{'' if h.documented else ' (hypothesis, undocumented)'}")
        lines.append(f"verdict [{self.code}]: {self.verdict}")
        lines.append(f"{self.jobs_submitted} device jobs submitted, {self.seconds:.1f} s")
        return "\n".join(lines)


def _maxpool_check(relu: np.ndarray, pool: np.ndarray, step: float, pool_step: float) -> dict:
    if abs(pool_step - step) > 1e-6 * step:
        return {"exact": None, "text": f"max pool step {pool_step:.6g} differs from relu step; not compared"}
    pr = max_pool(relu, {"kernel": 3, "stride": 2, "padding": 1})
    eq = float((pr == pool).mean())
    return {"exact": eq == 1.0, "equal_fraction": eq,
            "text": (f"NPU max pool equals the exact max pool of its own relu output on {eq:.2%} of "
                     "elements" + (" (no rounding here, as expected)" if eq == 1.0 else ""))}


def _conv1_case(n: int):
    import torch
    from truestar.remote.synthetic import build_task

    task = build_task()
    m = task.model.double()
    bn = m.bn1
    g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    W = (m.conv1.weight * g[:, None, None, None]).detach().numpy()
    B = (bn.bias - bn.running_mean * g).detach().numpy()
    x = task.eval_x[:n].float().numpy()
    sw = np.abs(W).reshape(W.shape[0], -1).max(1) / 127.0
    wq = np.rint(W / sw[:, None, None, None]).astype(np.int8)
    return x, wq, B, sw


def simulate(observed: dict, step: float, n: int = 4,
             scales: Sequence[float] = INPUT_SCALES) -> List[Hypothesis]:
    """Each convention's signature against the CPU's (onnx_qlinear: XNNPACK and ONNX Runtime use
    float32 requantization with round half to even, and ONNX Runtime's integer kernels matched the
    S24 CPU exactly in docs/NPU.md) on a simulated conv1."""
    x, wq, B, sw = _conv1_case(n)
    names = [c for c in documented_conventions() if c != "onnx_qlinear"] + ["npu_fp16_input"]
    res: Dict[str, Dict[str, list]] = {k: {"rate": [], "bins": [], "min": []} for k in names}
    for sx in scales:
        bq = np.rint(B / (sx * sw)).astype(np.int64)
        p = {"x_zp": INPUT_ZP, "sx": sx, "sw": sw, "sy": step, "y_zp": 0, "stride": 2,
             "padding": 3, "out_signed": False}
        qp = {"scale": sx, "zp": INPUT_ZP, "out_signed": False}
        xq = quantize(x, qp, "onnx_qlinear")
        ctrl = conv2d(xq, wq, bq, p, "onnx_qlinear")
        for k in names:
            xin = quantize(x, qp, k) if k == "npu_fp16_input" else xq
            sig = pair_signature(conv2d(xin, wq, bq, p, k), ctrl)
            res[k]["rate"].append(sig.rate)
            res[k]["bins"].append([b["rate"] for b in sig.by_magnitude])
            res[k]["min"].append(sig.channel_minority)
    obs = np.array([b["rate"] for b in observed["by_magnitude"]])
    out = []
    for k in names:
        bins = np.array(res[k]["bins"])
        out.append(Hypothesis(k, CONVENTIONS[k].documented, res[k]["rate"], bins.tolist(),
                              res[k]["min"], float(np.abs(bins - obs).mean())))
    out.sort(key=lambda h: h.distance)
    return out


def explain(directory: Optional[str] = None, simulate_inputs: int = 4) -> NpuExplanation:
    from truestar.remote.agreement import estimate_step

    t0 = time.time()
    npu, cpu = load_taps(NPU_JOB, directory), load_taps(CPU_JOB, directory)
    step = estimate_step(cpu[1])
    if step is None:
        raise ValueError("the CPU relu tap is not on a quantization grid")
    rn, rc = to_levels(npu[1], step), to_levels(cpu[1], step)
    sig = pair_signature(rn, rc)
    rt = rounding_only_test(sig)
    pool_step = estimate_step(cpu[2]) or step
    mp = _maxpool_check(rn, to_levels(npu[2], pool_step), step, pool_step)
    hyps = simulate(sig.as_dict(), step, simulate_inputs)
    return _verdict(step, sig.as_dict(), rt, mp, hyps, time.time() - t0)


def _verdict(step, sig, rt, mp, hyps, seconds) -> NpuExplanation:
    best = hyps[0] if hyps else None
    documented_best = next((h for h in hyps if h.documented), None)
    if rt["rounding_only_possible"]:
        code = "TS540"
        verdict = "the differences are consistent with a requantization rounding rule difference"
    else:
        code = "TS543"
        verdict = ("not a rounding convention difference: " + rt["reason"])
        if best is not None and not best.documented:
            verdict += (f". Best match: {best.name}, simulated rate {min(best.rate):.2%} to "
                        f"{max(best.rate):.2%} against {sig['rate']:.2%} observed, gap {best.distance:.3f}"
                        + (f"; the closest documented convention ({documented_best.name}) is off by "
                           f"{documented_best.distance:.3f}" if documented_best else "")
                        + ". The NPU most likely quantizes the float input in float16; this is an "
                        "inference from the saved data, not a documented Qualcomm behaviour")
    e = NpuExplanation(step, sig, rt, mp, hyps, best.name if best else None, verdict, code,
                       round(seconds, 2), 0)
    return e


def save_summary(e: NpuExplanation, path: str = SUMMARY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(e.as_dict(), f, indent=1)


def from_summary(path: str = SUMMARY_PATH) -> NpuExplanation:
    """The explanation recorded from the saved jobs (used when the job cache is not on this
    machine); the verdict is recomputed from the stored numbers, not copied."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    hyps = [Hypothesis(h["name"], h["documented"], h["rate"], h["by_magnitude"],
                       h["channel_minority"], h["distance"]) for h in d["hypotheses"]]
    from .match import PairSignature

    s = d["signature"]
    sig = PairSignature(s["elements"], s["differ"], s["up"], s["down"], s["max_lsb"],
                        s["by_magnitude"], s["channel_minority_median"], s["channels_two_sided"],
                        s["channels"])
    return _verdict(d["step"], s, rounding_only_test(sig), d["maxpool"], hyps, d["seconds"])
