"""Text and agent JSON for a precision plan."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _us(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.0f} us"


def _row(m) -> str:
    hold = f"{m.hold.flips}/{m.hold.decisions}" if m.hold else "-"
    err = m.cal.rel_err
    e = "inf" if not math.isfinite(err) else f"{err:.2e}"
    lo, hi = m.latency_range
    return (f"  {m.name:<28} {_us(m.latency_us):>10} ({lo:.0f} to {hi:.0f})  flips cal "
            f"{m.cal.flips}/{m.cal.decisions}  held out {hold}  rel err {e}")


def plan_text(plan, top: int = 60) -> str:
    L: List[str] = []
    r = plan.result
    counts = ", ".join(f"{v} {k}" for k, v in sorted(plan.counts().items(), key=lambda kv: -kv[1]))
    L.append(f"truestar compile --precision auto: {plan.model_name} on {plan.gpu} "
             f"(torch {plan.torch_version})")
    L.append(f"budget: {plan.budget.text()} flips -> at most {plan.allowed_cal} calibration and "
             f"{plan.allowed_hold} held out decisions may change against float64")
    status = "met" if plan.met else "NOT met (float32 everywhere)"
    L.append(f"plan ({status}): {counts}; {len(plan.units)} units, "
             f"{sum(1 for g in plan.groups if len(g) > 1)} tie groups, {plan.verifications} "
             f"verification runs, planned in {plan.seconds.get('planning', 0):.1f} s (baselines {plan.seconds.get('baselines', 0):.1f} s)")
    L.append("")
    L.append("measured end to end (median GPU time per forward, round range):")
    L.append(_row(r))
    for b in plan.baselines:
        L.append(_row(b))
    L.append("")
    L.append("per unit (execution order):")
    for u in sorted(plan.units, key=lambda u: u.order)[:top]:
        L.append(f"  {u.name or '<model>'} ({u.module_type}): {u.why}")
    if len(plan.units) > top:
        L.append(f"  ... {len(plan.units) - top} more in the JSON")
    if plan.issues:
        L.append("")
        for i in plan.issues:
            L.append(f"[{i.code}] {i.message}")
    for n in plan.notes:
        L.append(f"note: {n}")
    return "\n".join(L)


def plan_agent(plan, model_file: Optional[str] = None) -> Dict[str, Any]:
    from ..agent import _envelope

    bug = "TS664" in plan.codes
    status = "ok" if plan.met and not bug else ("budget_not_met" if not plan.met else "kernel_rejected")
    doc = _envelope(status, 0 if status == "ok" else 1)
    doc["command"] = "compile"
    actions = []
    if plan.met:
        actions.append({"kind": "apply_plan",
                        "description": "Run the model with this plan: save it with --plan plan.json, "
                                       "then truestar.precplan.apply(model, 'plan.json').",
                        "command": None})
    else:
        actions.append({"kind": "raise_budget",
                        "description": "Float32 everywhere already flips more decisions than the "
                                       "budget allows; raise --max-flips or find the op with "
                                       "truestar check.",
                        "command": f"truestar check {model_file} --format agent" if model_file else None})
    if bug:
        actions.append({"kind": "report_kernel_bug",
                        "description": "A Truestar quantized kernel failed its exact check on a "
                                       "layer (TS664); the plan does not use it. Report it.",
                        "command": None})
    doc.update({"model_file": model_file, "codes": plan.codes,
                "issues": [i.as_dict() for i in plan.issues], "next_actions": actions,
                "plan": plan.as_dict()})
    return doc
