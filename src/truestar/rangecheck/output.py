"""Text and agent JSON for `truestar range` (envelope as in docs/AGENT_INTERFACE.md, command "range")."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..agent import SCHEMA_VERSION, _torch_version
from ..codes import Issue

if TYPE_CHECKING:
    from .analysis import RangeReport, Site

EXIT_OK, EXIT_FINDING, EXIT_USAGE = 0, 1, 2


def exit_code(status: str) -> int:
    return EXIT_OK if status == "ok" else EXIT_FINDING


def _e(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.3g}"


def _bits(x: float) -> str:
    if math.isinf(x):
        return "unbounded" if x < 0 else "inf"
    return f"{x:+.2f} bits"


def _clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


_KIND_TEXT = {
    "overflow": "overflows", "hidden_overflow": "hidden intermediate overflows",
    "near_overflow": "near overflow", "underflow": "underflows", "precision_loss": "loses precision",
    "constant_saturated": "constant saturates", "hidden_in_fused": "would overflow if decomposed",
}


def _site_line(s: "Site", fmt) -> str:
    from .analysis import _KIND_CODE

    code = _KIND_CODE.get(s.kind, "TS309")
    what = f"{s.hidden} " if s.hidden else ""
    if s.kind == "underflow":
        detail = f"{100 * s.share_below_normal:.0f}% of nonzero values below {fmt.min_normal:.3g}, peak {_e(s.peak)}"
    elif s.kind == "precision_loss":
        detail = f"keeps {s.precision_bits:.1f} correct bits alone"
    else:
        detail = f"{what}peaks {_e(s.peak)} vs max {_e(fmt.max)}, margin {_bits(s.headroom_bits)}"
        detail += (f"; static bound {_e(s.static_peak)} (proven safe)" if s.proven
                   else f"; static bound {_e(s.static_peak)}")
    w = s.where
    loc = f"{w.file}:{w.line}" if w.file else ""
    mod = f"{w.module} ({w.module_type})" if w.module else "<model>"
    line = f"  {code} {s.op} %{s.value.split('.')[0]} in {mod} {_KIND_TEXT.get(s.kind, s.kind)}: {detail}"
    if loc:
        line += f"\n         at {loc}: {w.code}" if w.code else f"\n         at {loc}"
    if s.effect != "untested":
        line += f"  [{s.effect}]"
    return line


def _cmp(c) -> str:
    return (f"cosine min {c.cosine_min:.6f} mean {c.cosine_mean:.6f}, error {_e(c.rel_err)}, "
            f"flips {c.real_flips} real of {c.flips} in {c.rows} rows, non finite {c.nonfinite}")


def report_text(rep: "RangeReport", top: int = 12) -> str:
    t = rep.target
    fmt = t.fmt
    lines = [f"truestar range: {rep.model_name}, target {t.name} ({t.description}), "
             f"headroom {rep.headroom_bits:g} bit(s), {rep.calibration} calibration input(s)",
             f"status: {rep.status.upper()}  " + " ".join(rep.codes)]
    c = rep.counts
    lines.append(f"  {c['float_values']} float values: {c['proven_safe_statically']} proven safe by static "
                 f"bounds for any input in the calibration box, {c['safe_on_calibration_only']} safe on the "
                 f"calibration inputs only")
    lines.append(f"  twin (format precision, unlimited range): {_cmp(rep.twin)}")
    lines.append(f"  target {t.name}:                            {_cmp(rep.target_run)}")
    if rep.sites:
        causes = [s for s in rep.sites if s.origin]
        carried = [s for s in rep.sites if not s.origin]
        lines.append("")
        lines.append(f"sites ({len(causes)} causes, {len(carried)} downstream "
                     f"values still out of range):")
        for s in causes[:top]:
            lines.append(_site_line(s, fmt))
        if len(causes) > top:
            lines.append(f"  ... {len(causes) - top} more (all in --json / --format agent)")
        if carried:
            mods = sorted({s.where.module or "<model>" for s in carried})
            lines.append(f"  carried: {len(carried)} values in {len(mods)} module(s) from {mods[0]} to "
                         f"{mods[-1]} stay out of range (listed in --json / --format agent)")
    if rep.attempts:
        lines.append("")
        lines.append("remedies (verified by emulating the target):")
        for a in rep.attempts:
            tag = "PASS" if a.passed else "fail"
            res = f": {_cmp(a.result)}" if a.result is not None else ""
            ex = f", float64 rewrite error {_e(a.exact_rel_err)}" if a.exact_rel_err is not None else ""
            lines.append(f"  {tag} {a.remedy}: {a.detail}{ex}{res}")
        if rep.chosen is not None:
            lines.append(f"  chosen: {rep.chosen.remedy}")
    if rep.info:
        kinds: Dict[str, int] = {}
        for s in rep.info:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        lines.append("")
        lines.append("information (TS309, no effect on the result): "
                     + ", ".join(f"{n} {_KIND_TEXT.get(k, k)}" + (" without effect" if k in ("overflow", "hidden_overflow", "underflow", "precision_loss") else "")
                                 for k, n in kinds.items()))
    s = rep.seconds
    lines.append("")
    lines.append(f"time: {s['total']:.2f} s (import {s['import']:.2f}, float64 observe {s['observe']:.2f}, "
                 f"static {s['static']:.2f}, emulate {s['emulate']:.2f}, remedies {s['remedies']:.2f})")
    return "\n".join(lines)


def _issues(rep: "RangeReport") -> List[Dict[str, Any]]:
    from .analysis import _KIND_CODE

    out = []
    for s in rep.sites:
        out.append(Issue(_KIND_CODE[s.kind], _site_line(s, rep.target.fmt).strip()).as_dict())
    if rep.info:
        out.append(Issue("TS309", f"{len(rep.info)} range events without effect on the result").as_dict())
    for code in rep._run_codes():
        msg = {"TS300": "every value fits and the target matches its twin",
               "TS306": f"target run: cosine min {rep.target_run.cosine_min:.6f}, "
                        f"{rep.target_run.real_flips} real flips, {rep.target_run.nonfinite} non finite",
               "TS307": f"remedy {rep.chosen.remedy if rep.chosen else ''} verified in emulation",
               "TS308": "no remedy passed, or no site explains the failure"}[code]
        out.append(Issue(code, msg).as_dict())
    return out


def _next_actions(rep: "RangeReport", model_file: Optional[str]) -> List[Dict[str, Any]]:
    acts: List[Dict[str, Any]] = []
    for s in rep.sites[:5]:
        if s.where.file:
            acts.append({"kind": "inspect_source", "description": f"{s.kind} at {s.op} %{s.value}",
                         "file": s.where.file, "line": s.where.line, "command": None})
    if rep.chosen is not None:
        a = rep.chosen
        desc = {"rescale": "Fold the power of two scale into the listed weights and norm eps (exact in "
                           "float64, no run time cost), then export for the target.",
                "fp32": "Keep the listed ops in float32 on the target (for example a precision block "
                        "list in the converter).",
                "clamp": "Insert Clip nodes at the listed ops' outputs before export."}[a.remedy]
        acts.append({"kind": f"apply_{a.remedy}", "description": desc, "command": None})
    elif rep.status == "overflow":
        acts.append({"kind": "keep_float32", "description": "No remedy passed in emulation: keep the "
                     "model in float32 on this target or retrain with range control.", "command": None})
    if rep.status == "at_risk":
        target = f'"{os.path.abspath(model_file)}"' if model_file else "<model.py>"
        acts.append({"kind": "add_calibration", "description": "A value sits near the format's maximum "
                     "without a static proof: add representative inputs (get_calibration()) and rerun.",
                     "command": f"truestar range {target} --target {rep.target.name} --format agent"})
    return acts


def report_agent(rep: "RangeReport", model_file: Optional[str] = None) -> Dict[str, Any]:
    from .. import __version__

    fmt = rep.target.fmt
    doc: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "tool": "truestar", "command": "range",
        "truestar_version": __version__, "torch_version": _torch_version(),
        "status": rep.status, "exit_code": exit_code(rep.status),
        "model": {"name": rep.model_name, "file": os.path.abspath(model_file) if model_file else None},
        "target": {"name": rep.target.name, "format": fmt.name, "max": fmt.max,
                   "min_normal": fmt.min_normal, "significand_bits": fmt.bits,
                   "decomposed": rep.target.decomposed, "nan_to_inf": rep.target.nan_to_inf,
                   "description": rep.target.description},
        "headroom_bits": rep.headroom_bits,
        "calibration_inputs": rep.calibration,
        "codes": rep.codes,
        "issues": _issues(rep),
        "counts": rep.counts,
        "twin": rep.twin.as_dict(),
        "target_run": rep.target_run.as_dict(),
        "sites": [s.as_dict() for s in rep.sites],
        "info": [s.as_dict() for s in rep.info],
        "remedies": [a.as_dict() for a in rep.attempts],
        "chosen_remedy": rep.chosen.remedy if rep.chosen else None,
        "seconds": rep.seconds,
        "next_actions": _next_actions(rep, model_file),
    }
    return _clean(doc)
