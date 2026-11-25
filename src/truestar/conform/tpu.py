"""`truestar conform tpu DIR`: judge saved accelerator outputs of the fixed battery (docs/TPU.md).

The accelerator does not need to be here. DIR is what `tpu_battery.py` wrote on the machine that
has it (summary.json and one .npy per backend and case). This side regenerates every input from
its seed, checks the input hashes the run recorded, computes the float64 reference itself (the
numbers in the run's own summary are not trusted) and gives one verdict per backend and case.

Budgets, by the dtype the case asks for:

* float32 matmul and sum: effective bits (-log2 of the worst error over |A||B|, or over sum |v|)
  at least 24 - log2(n)/2 - 3. That is the probabilistic rounding bound lambda sqrt(n) u with
  lambda = 8 (Higham and Mary, SIAM J. Sci. Comput. 2019). Any float32 summation order meets it;
  one bfloat16 or TF32 rounding of the inputs does not.
* other float32 cases: relative L2 error at most 1e-5; elementwise functions also within 4 ulp.
* bfloat16 cases: relative L2 error at most 3e-2.

A float32 case outside its budget but within the bfloat16 budget is reduced precision (TS902,
the TPU's default matmul precision is the usual cause); beyond that it is divergent (TS901).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import tpu_battery as B

F32_REL_L2 = 1e-5
BF16_REL_L2 = 3e-2
F32_ULP = 4.0
LAMBDA_BITS = 3.0  # log2(lambda), lambda = 8
ELEMENT_WRONG = 1.0 / 16  # one element this far off (in output rms) is wrong at any precision


class TpuBundleError(Exception):
    pass


def _inner_n(case: str) -> int:
    specs = B.CASES[case][3]
    if B.CASES[case][0] == "sum":
        return int(np.prod(specs["v"][0]))
    return int(specs["a"][0][-1])


def bits_budget(case: str) -> float:
    return 24.0 - 0.5 * math.log2(_inner_n(case)) - LAMBDA_BITS


@dataclass
class CaseVerdict:
    backend: str
    case: str
    verdict: str  # ok, reduced_precision, divergent, unstable_ok, failed, missing
    metrics: Dict = field(default_factory=dict)
    budget: str = ""
    issues: List[tuple] = field(default_factory=list)

    def as_dict(self):
        return {"backend": self.backend, "case": self.case, "verdict": self.verdict,
                "budget": self.budget, "metrics": self.metrics,
                "issues": [{"code": c, "message": m} for c, m in self.issues]}


def judge(case: str, m: Dict, backend: str = "") -> tuple:
    """(verdict, budget text, [(code, message)]) for one case's metrics."""
    op, dtype, prec, _ = B.CASES[case]
    if m.get("nonfinite"):
        return "divergent", "finite", [("TS901", f"{case}: {m['nonfinite']} non finite values")]
    if m.get("max_abs_over_rms", 0.0) > ELEMENT_WRONG:
        return "divergent", f"max_abs <= {ELEMENT_WRONG:g} rms(ref)", [
            ("TS901", f"{case}: an element is off by {m['max_abs_over_rms']:.3g} times the output's rms: "
                      "a wrong value, not rounding")]
    if dtype == "bfloat16":
        ok = m["rel_l2"] <= BF16_REL_L2
        return ("ok" if ok else "divergent", f"rel_l2 <= {BF16_REL_L2:g}",
                [] if ok else [("TS901", f"{case}: rel_l2 {m['rel_l2']:.3g} above the bfloat16 budget")])
    if op in ("matmul", "matmul_acc32", "sum"):
        need = bits_budget(case)
        text = f"eff_bits >= {need:.1f}"
        if m["eff_bits"] >= need:
            return "ok", text, []
        if m["rel_l2"] <= BF16_REL_L2:
            hint = ""
            if op != "sum" and "tpu" in backend and prec != "highest":
                hint = (" (JAX default precision on TPU: pass precision=HIGHEST or set "
                        "jax_default_matmul_precision='highest')")
            elif op != "sum" and "tf32" in backend:
                hint = " (TF32 is on: torch.backends.cuda.matmul.allow_tf32 = False)"
            return "reduced_precision", text, [("TS902", f"{case}: float32 result behaves as "
                                                f"{m['eff_bits']:.1f} bits, budget {need:.1f}{hint}")]
        return "divergent", text, [("TS901", f"{case}: {m['eff_bits']:.1f} effective bits, rel_l2 "
                                    f"{m['rel_l2']:.3g}")]
    text = f"rel_l2 <= {F32_REL_L2:g}" + (f", max_ulp <= {F32_ULP:g}" if op in ("exp", "tanh", "log", "rsqrt")
                                           else "")
    if m["rel_l2"] <= F32_REL_L2:
        if op in ("exp", "tanh", "log", "rsqrt") and m.get("max_ulp", 0.0) > F32_ULP:
            return "ulp_high", text, [("TS905", f"{case}: {m['max_ulp']:.1f} ulp worst element "
                                       f"(budget {F32_ULP:g}); relative error is float32 level")]
        return "ok", text, []
    if m["rel_l2"] <= BF16_REL_L2:
        return "reduced_precision", text, [("TS902", f"{case}: float32 case at rel_l2 {m['rel_l2']:.3g}, "
                                            "bfloat16 or TF32 level; check the matmul precision of its "
                                            "dots")]
    return "divergent", text, [("TS901", f"{case}: rel_l2 {m['rel_l2']:.3g}")]


def load_summary(path: str) -> Dict:
    f = os.path.join(path, "summary.json") if os.path.isdir(path) else path
    try:
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
    except (OSError, ValueError) as exc:
        raise TpuBundleError(f"{f}: {type(exc).__name__}: {exc}") from None
    if not isinstance(s, dict) or "backends" not in s or "input_hashes" not in s:
        raise TpuBundleError(f"{f}: not a battery summary (no backends or input_hashes)")
    if s.get("battery_version") != B.BATTERY_VERSION or s.get("master_seed") != B.MASTER_SEED:
        raise TpuBundleError(f"{f}: battery version {s.get('battery_version')} seed {s.get('master_seed')}, "
                             f"this truestar has {B.BATTERY_VERSION} seed {B.MASTER_SEED}")
    return s


def check(path: str, backends: Optional[List[str]] = None, cases: Optional[List[str]] = None):
    """Returns (verdicts, top level issues). Raises TpuBundleError for an unreadable bundle."""
    s = load_summary(path)
    root = path if os.path.isdir(path) else os.path.dirname(path)
    top, out = [], []
    names = backends or sorted(s["backends"])
    for b in names:
        if b not in s["backends"]:
            raise TpuBundleError(f"backend {b!r} not in the bundle (has {', '.join(sorted(s['backends']))})")
    cache = {}
    for case in cases or []:
        if case not in B.CASES:
            raise TpuBundleError(f"unknown case {case!r}")
    present = {B._case_of(k) for b in names for k in s["backends"][b].get("results", {})}
    for case in [c for c in (cases or list(B.CASES)) if c in present]:
        arrs = B.inputs(case)
        if s["input_hashes"].get(case) != B.input_hash(arrs):
            top.append(("TS904", f"{case}: the run's inputs hash {str(s['input_hashes'].get(case))[:12]} is "
                                 f"not this battery's {B.input_hash(arrs)[:12]}; its outputs are not judged"))
            continue
        cache[case] = (arrs, B.reference(case, arrs))
    for b in names:
        rows = s["backends"][b].get("results", {})
        for key, row in sorted(rows.items()):
            case = B._case_of(key)
            if case not in cache:
                continue
            if "error" in row:
                out.append(CaseVerdict(b, key, "failed", issues=[("TS906", f"{key} on {b} raised: "
                                                                            f"{row['error'][:200]}")]))
                continue
            f = os.path.join(root, b, key.replace("@", "__") + ".npy")
            try:
                y = np.load(f, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise TpuBundleError(f"{f}: {type(exc).__name__}: {exc}") from None
            arrs, ref = cache[case]
            if y.size != ref.size:
                raise TpuBundleError(f"{f}: {y.size} values, the reference has {ref.size}")
            m = B.metrics(case, y, ref, arrs)
            verdict, budget, issues = judge(case, m, b)
            hashes = row.get("hashes") or []
            m["bitwise_stable"] = len(set(hashes)) <= 1
            m["runs"] = len(hashes)
            if len(set(hashes)) > 1:
                issues.append(("TS903", f"{key} on {b}: {len(set(hashes))} different results over "
                                        f"{len(hashes)} identical calls"))
                if verdict == "ok":
                    verdict = "unstable_ok"
            out.append(CaseVerdict(b, key, verdict, m, budget, issues))
    return out, top


def to_text(verdicts, top) -> str:
    lines = ["truestar conform tpu: saved outputs against the float64 reference"]
    for c, msg in top:
        lines.append(f"  {c} {msg}")
    for v in verdicts:
        m = v.metrics
        if not m:
            lines.append(f"  {v.backend:16s} {v.case:26s} {v.verdict}")
            continue
        bits = f" bits {m['eff_bits']:.1f}" if "eff_bits" in m else ""
        ulp = f" ulp {m['max_ulp']:.1f}" if "max_ulp" in m and "eff_bits" not in m else ""
        lines.append(f"  {v.backend:16s} {v.case:26s} {v.verdict:17s} rel_l2 {m['rel_l2']:.2e}{bits}{ulp}"
                     f" {'stable' if m.get('bitwise_stable') else 'UNSTABLE'}  [{v.budget}]")
        for c, msg in v.issues:
            lines.append(f"      {c} {msg}")
    return "\n".join(lines)
