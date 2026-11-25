"""Text and agent JSON for `truestar fix` and for precision plans.

The agent document follows docs/AGENT_INTERFACE.md (same envelope, schema_version, strict JSON,
stable codes, `next_actions`), with `command` set to "fix".
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..agent import SCHEMA_VERSION, _torch_version

if TYPE_CHECKING:
    from .engine import FixReport
    from .precision import PrecisionPlan

EXIT_OK, EXIT_NOT_FIXED, EXIT_USAGE = 0, 1, 2


def _e(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.2e}"


def _clean(obj: Any) -> Any:
    """Strict JSON: non finite floats become None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def exit_code(status: str) -> int:
    return EXIT_OK if status in ("clean", "resolved") else EXIT_NOT_FIXED


def _snap_line(s) -> str:
    flips = f", flips {s.real_flips} real of {s.decisions}" if s.decisions else ""
    op = f", op {s.op} {_e(s.op_rel_err)}" if s.op else ""
    mod = f", module {s.module} {_e(s.module_rel_err)}" if s.module else ""
    return f"{s.status}, output error {_e(s.output_rel_err)}{flips}{mod}{op}"


def fix_text(rep: "FixReport") -> str:
    lines = [f"truestar fix: {rep.model_name} (mechanism {rep.mechanism}), status {rep.status}"]
    if not rep.backends:
        lines.append("  every backend already matches the reference: nothing to fix")
    for b in rep.backends:
        lines.append("")
        lines.append(f"[{b.outcome.upper()}] {b.backend} ({b.device})  "
                     + " ".join(i.code for i in b.issues))
        lines.append(f"  before: {_snap_line(b.before)}")
        lines.append(f"  after:  {_snap_line(b.after)}")
        for r in b.applied:
            lines.append(f"  remedy: {r}")
        for a in b.attempts:
            tag = "WON " if a.won else ("skip" if a.skipped else ("fail" if not a.op_ok else "try "))
            extra = f" ({a.skipped})" if a.skipped else ""
            if a.module_status is not None:
                extra += (f"; rerun check: module {a.module_status} {_e(a.module_rel_err)}, "
                          f"model {a.model_status}, {a.hits} call(s) replaced")
            lines.append(f"    {tag} {a.remedy.op} -> {a.remedy.decomposition}: op alone "
                         f"{_e(a.op_rel_err)} vs {_e(a.op_threshold)}{extra}")
        if b.cost is not None:
            c = b.cost
            lines.append(f"  cost: forward {c.baseline_ms:.3f} ms -> {c.remedied_ms:.3f} ms "
                         f"({100 * c.overhead:+.1f}%, median of {c.reps})")
        if b.reason and b.outcome != "resolved":
            lines.append(f"  why: {b.reason}")
        up = [f"{k} {v}" for k, v in b.upstream.items() if v]
        if up and b.outcome != "clean":
            lines.append("  upstream report: " + "; ".join(up))
        lines.append(f"  search and verify: {b.seconds:.2f} s")
    if rep.remedies():
        lines.append("")
        lines.append("use it: with truestar.fix.patched(report): ...   "
                     "or truestar.fix.apply(report, model=model)")
    return "\n".join(lines)


def _actions(rep: "FixReport") -> List[Dict[str, Any]]:
    acts: List[Dict[str, Any]] = []
    for b in rep.backends:
        rems = [r.as_dict() for r in b.applied]
        if b.outcome in ("resolved", "improved"):
            acts.append({
                "kind": "apply_remedy" if b.outcome == "resolved" else "apply_partial_remedy",
                "backend": b.backend,
                "description": (f"Apply the verified remedy on {b.backend} without editing the "
                                "model: wrap calls in truestar.fix.patched(report), or call "
                                "truestar.fix.apply(report, model=model) once."
                                + ("" if b.outcome == "resolved" else
                                   " It lowers the error but the check is not clean.")),
                "command": None, "remedies": rems,
                "python": "with truestar.fix.patched(report):\n    out = model(*inputs)",
                "cost": None if b.cost is None else b.cost.as_dict(),
            })
        path = b.upstream.get("op_repro") or b.upstream.get("repro")
        if b.outcome != "clean" and (path or b.upstream.get("draft")):
            acts.append({
                "kind": "report_upstream", "backend": b.backend,
                "description": ("The remedy is a workaround; the kernel is still wrong. "
                                if b.outcome == "resolved" else
                                "No remedy cleared this backend. ")
                + "Report it with the repro"
                + (f" ({path})" if path else "")
                + (f"; a drafted report is at {b.upstream['draft']}" if b.upstream.get("draft")
                   else "") + ".",
                "command": None if not path else f'python "{os.path.abspath(path)}"',
                "path": path, "draft": b.upstream.get("draft"),
            })
        if b.outcome == "not_applicable":
            acts.append({"kind": "inspect_source", "backend": b.backend,
                         "description": b.reason + ". Run truestar check for the blamed module "
                                        "and its source location.", "command": None})
    return acts


def _issues(issues) -> List[Dict[str, Any]]:
    return [i.as_dict() for i in issues]


def fix_agent(rep: "FixReport", model_file: Optional[str] = None) -> Dict[str, Any]:
    from .. import __version__

    doc = {
        "schema_version": SCHEMA_VERSION, "tool": "truestar", "command": "fix",
        "truestar_version": __version__, "torch_version": _torch_version(),
        "status": rep.status, "exit_code": exit_code(rep.status),
        "model": {"name": rep.model_name,
                  "file": os.path.abspath(model_file) if model_file else None,
                  "dtype": str(rep.before.dtype).replace("torch.", "")},
        "mechanism": rep.mechanism,
        "codes": rep.codes,
        "issues": _issues(rep.issues),
        "backends": [b.as_dict() for b in rep.backends],
        "next_actions": _actions(rep),
    }
    return _clean(doc)


# ----------------------------------------------------------------------------- precision plans


def plan_text(plan: "PrecisionPlan") -> str:
    lines = [f"truestar fix (precision): {plan.model_name}, {plan.scheme} simulated on CPU vs "
             f"float64, {plan.n_inputs} inputs, target at most {plan.target_flips} flips"]
    lines.append(f"  ranking from {plan.ranking_source}; {plan.simulations} simulations")
    lines.append(f"  before: all quantized, {plan.flips_before} flips, output error "
                 f"{_e(plan.err_before)}")
    lines.append(f"  after:  keep {len(plan.keep)} unit(s) in float ({', '.join(plan.keep) or 'none'}),"
                 f" {plan.flips_after} flips, output error {_e(plan.err_after)}")
    lines.append(f"  compute kept in higher precision: {100 * plan.compute_fraction:.2f}% of "
                 f"MACs ({len(plan.keep_leaves)} of {plan.n_leaves} layers)")
    lines.append(f"  {'MET' if plan.met else 'NOT MET'}: " + plan.issue.message)
    for s in plan.steps:
        lines.append(f"    {s['action']:<6} {s['unit']:<22} flips {s['flips']:>4}  "
                     f"error {_e(s['err'])}")
    return "\n".join(lines)


def plan_agent(plan: "PrecisionPlan") -> Dict[str, Any]:
    from .. import __version__

    i = plan.issue
    doc = {
        "schema_version": SCHEMA_VERSION, "tool": "truestar", "command": "fix",
        "truestar_version": __version__, "torch_version": _torch_version(),
        "status": "resolved" if plan.met else "unresolved",
        "exit_code": EXIT_OK if plan.met else EXIT_NOT_FIXED,
        "model": {"name": plan.model_name, "file": None, "dtype": "float32"},
        "mechanism": "precision",
        "codes": [i.code], "issues": [i.as_dict()],
        "precision_plan": plan.as_dict(),
        "next_actions": [{
            "kind": "keep_layers_in_float" if plan.met else "raise_target_or_retrain",
            "description": (f"Exclude {', '.join(plan.keep_leaves) or 'no layers'} from INT8 "
                            f"quantization; verified in simulation: {plan.flips_before} -> "
                            f"{plan.flips_after} flips at {100 * plan.compute_fraction:.2f}% of "
                            "compute in float.") if plan.met else
                           ("Keeping every candidate unit in float did not reach the target; "
                            "the flips come from somewhere the plan cannot keep in float."),
            "command": None, "keep": list(plan.keep_leaves),
        }],
    }
    return _clean(doc)
