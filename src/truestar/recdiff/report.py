"""Text and agent JSON output for `truestar diff`."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from ..capture import ROOT
from .diff import (DIVERGED, IDENTICAL, NOISE, PRECISION_CHANGE, RANK, UNCHECKED, WEIGHTS,
                   DiffReport, _num)
from .fingerprint import PRECISION, VERSION

SCHEMA_VERSION = 1
MAX_MODULES = 5

_VERDICT = {
    IDENTICAL: "IDENTICAL: every module gives bitwise the same result",
    NOISE: "NOISE: results moved, but only within the dtype tolerance",
    PRECISION_CHANGE: "PRECISION CHANGE: explained by a precision setting that differs",
    DIVERGED: "DIVERGENT: a module's own result changed beyond what precision explains",
    "not_comparable": "NOT COMPARABLE: the runs used different weights or inputs",
}


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None or x != x else f"{x:.2e}"


def render_text(rep: DiffReport, top: int = 8) -> str:
    lines: List[str] = []
    how = {"recordings": "two recordings", "replay": "replay, every module alone",
           "replay_outputs_only": "replay of an outputs only recording"}[rep.mode]
    lines.append(f"truestar diff: {rep.model} ({rep.dtype_name()}, {rep.device}), {how}")
    lines.append(f"  recorded: {rep.recorded}")
    lines.append(f"  current:  {rep.current}")
    if rep.env_changes:
        lines.append("Environment changes:")
        for k, a, b, kd in rep.env_changes[:12]:
            lines.append(f"  {k}: {a} -> {b}  [{kd}]")
        if len(rep.env_changes) > 12:
            lines.append(f"  ... {len(rep.env_changes) - 12} more")
    else:
        lines.append("Environment: no differences in the fingerprint")
    lines.append("")
    lines.append("Verdict: " + _VERDICT[rep.status])
    b = rep.blamed
    if b is not None and RANK[b.cls] >= RANK[NOISE]:
        rel = b.diff.rel_err if b.diff is not None else None
        band = f"noise band {b.noise_band:.1e}"
        if b.precision_band is not None:
            band += f", precision band {b.precision_band:.1e}"
        lines.append(f"First module whose own result changed: {b.key} ({b.type_name})")
        lines.append(f"  class {b.cls}, own change {_fmt(rel)} ({band}), measured by "
                     f"{'isolated replay' if b.method == 'replay' else 'equal input hashes'}")
        if b.exact_current is not None:
            closer = "current" if b.exact_current < (b.exact_recorded or 0) else "recorded"
            lines.append(f"  vs float64 math: recorded {_fmt(b.exact_recorded)}, current "
                         f"{_fmt(b.exact_current)} (the {closer} result is closer to exact)")
        src = (rep.blame_source or {}).get("user")
        if src:
            lines.append(f"  inside your code: {src['file']}:{src['line']} ({src['qualname']})")
        op = rep.blamed_op
        if op is not None:
            how_op = ("fed its recorded inputs" if op.source == "recorded_trace"
                      else "compared with float64 math")
            approx = "" if op.isolated else " (approximate: its inputs also differ)"
            lines.append(f"  op: {op.op} (op {op.index} of {op.total}), class {op.cls}, change "
                         f"{_fmt(op.diff.rel_err)} vs band {op.band:.1e}, {how_op}{approx}")
        if rep.op_repro_path:
            lines.append(f"  op repro: {rep.op_repro_path}")
    c = rep.counts()
    lines.append(f"Modules: {c[IDENTICAL]} identical, {c[NOISE]} noise, {c[PRECISION_CHANGE]} "
                 f"precision, {c[DIVERGED]} divergent, {c['inherited']} inherited"
                 + (f", {c[UNCHECKED]} not checked" if c[UNCHECKED] else "")
                 + (f", {c[WEIGHTS]} with different weights" if c[WEIGHTS] else ""))
    if rep.final is not None:
        lines.append(f"Final output change: {_fmt(rep.final.rel_err)} (max abs "
                     f"{_fmt(rep.final.max_abs)})")
    if rep.baseline:
        lines.append(f"Baseline, {rep.baseline['method']}: "
                     f"{'pass' if rep.baseline['allclose'] else 'FAIL'}; it names no module "
                     "and no op")
    changed = [m for m in rep.modules if m.own and RANK[m.cls] >= RANK[NOISE]]
    if changed:
        lines.append("")
        lines.append(f"Modules that own a change (first {min(top, len(changed))} of "
                     f"{len(changed)}):")
        for m in changed[:top]:
            rel = m.diff.rel_err if m.diff is not None else None
            lines.append(f"  {m.key:<32} {m.type_name:<24} {m.cls:<10} {_fmt(rel)}")
    if rep.issues:
        lines.append("")
        for i in rep.issues:
            lines.append(f"[{i.code}] {i.message}")
    for n in rep.notes:
        lines.append(f"note: {n}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- agent JSON


def _torch_pin(version: Any) -> Optional[str]:
    if not isinstance(version, str):
        return None
    return f"torch=={version.split('+')[0]}"


def next_actions(rep: DiffReport, recorded_path: Optional[str] = None,
                 model_file: Optional[str] = None) -> List[Dict[str, Any]]:
    acts: List[Dict[str, Any]] = []

    def add(kind: str, description: str, command: Optional[str] = None, **extra):
        a = {"kind": kind, "description": description, "command": command}
        a.update(extra)
        acts.append(a)

    codes = rep.codes
    if "TS069" in codes:
        add("check_inputs", "The two recordings fed the model different inputs. Record both "
                            "runs from the same get_inputs() before comparing.")
    if "TS064" in codes:
        add("check_weights", "The model's weights differ from the recorded ones, so this is not "
                             "an environment change. Load the same checkpoint (or seed) and "
                             "compare again.")
    status = rep.status
    b = rep.blamed
    if status == PRECISION_CHANGE:
        for k, old, new, kd in rep.env_changes:
            if kd == PRECISION:
                add("restore_precision_flag",
                    f"Set {k} back to {old} to get the recorded numerics, or accept the change: "
                    "it is within that precision mode's tolerance.", key=k, recorded=old,
                    current=new)
    if status == DIVERGED and b is not None:
        op = rep.blamed_op
        if rep.op_repro_path:
            add("confirm_op_repro",
                f"Run the single op repro for {op.op}; exit code 1 shows this one op's result "
                "moved for the exact inputs it had in the recorded environment.",
                f'"{sys.executable}" "{os.path.abspath(rep.op_repro_path)}"', op=op.op)
        src = rep.blame_source or {}
        own, user = src.get("module"), src.get("user")
        if own is not None and own.get("qualname", "").startswith("truestar.testing."):
            add("injected_fault", f"The blamed module {b.key} is a truestar.testing fault "
                                  "injector: this change was put there on purpose.")
        elif own is not None and own.get("user_code"):
            add("inspect_source", f"The blamed module {b.key} is your class {own['qualname']}; "
                                  "check its forward for code that depends on library "
                                  "behaviour.", file=own["file"], line=own["line"])
        elif user is not None:
            add("inspect_source", f"The blamed module {b.key} ({b.type_name}) is a library "
                                  f"class used inside your class {user['qualname']}.",
                file=user["file"], line=user["line"])
        versions = [(k, old, new) for k, old, new, kd in rep.env_changes if kd == VERSION]
        for k, old, new in versions:
            pin = _torch_pin(old) if k == "torch" else None
            add("pin_version",
                f"{k} changed from {old} to {new}. Pin {k} to the recorded {old} until the "
                "change is understood" + (" (install it from the index matching your CUDA "
                                         "build)" if pin else "") + ".",
                f'pip install "{pin}"' if pin else None, key=k, recorded=old, current=new)
        if op is not None or versions:
            add("report_upstream",
                "Report upstream with "
                + (f"the op repro folder {os.path.dirname(os.path.abspath(rep.op_repro_path))}"
                   if rep.op_repro_path else f"the blamed module {b.key} ({b.type_name})"
                   + (f" and op {op.op}" if op is not None else ""))
                + ", both version sets from environment_changes, and the GPU name.",
                repro_dir=os.path.dirname(os.path.abspath(rep.op_repro_path))
                if rep.op_repro_path else None)
        if op is None or op.source != "recorded_trace":
            add("record_op_trace",
                f"Record in the old environment with --ops {b.name} so diff can feed every op "
                "the inputs it had there and name the op exactly.",
                f"truestar record <model.py> -o run.tsrec --ops {b.name}")
    if "TS068" in codes:
        target = f'"{recorded_path}"' if recorded_path else "a.tsrec"
        add("replay_for_isolation",
            "Some changed modules saw different inputs, so only the first change is isolated. "
            "Replay the old recording in the new environment to measure every module alone.",
            f"truestar diff {target} --replay"
            + (f' --model "{model_file}"' if model_file else "") + " --format agent")
    if "TS067" in codes:
        add("inspect_replay_failure", "Some module calls could not be replayed (see TS067); "
                                      "their own behaviour was not measured.")
    return acts


def to_agent(rep: DiffReport, recorded_path: Optional[str] = None,
             model_file: Optional[str] = None, strict: bool = False) -> Dict[str, Any]:
    from ..agent import _envelope

    doc = _envelope(rep.status, rep.exit_code(strict))
    doc["command"] = "diff"
    b = rep.blamed
    blamed = None
    if b is not None:
        src = rep.blame_source or {}
        blamed = dict(b.as_dict(), source=src.get("module"), user_source=src.get("user"))
    changed = [m for m in rep.modules if m.own and RANK[m.cls] >= RANK[NOISE]]
    doc.update({
        "mode": rep.mode,
        "recorded": rep.recorded,
        "current": rep.current,
        "model": {"name": rep.model, "dtype": rep.dtype_name(), "device": rep.device},
        "codes": rep.codes,
        "issues": [i.as_dict() for i in rep.issues],
        "environment_changes": [{"key": k, "recorded": a, "current": v, "kind": kd}
                                for k, a, v, kd in rep.env_changes],
        "precision_explained": rep.precision_explained,
        "blamed_module": blamed,
        "blamed_op": None if rep.blamed_op is None else rep.blamed_op.as_dict(),
        "counts": rep.counts(),
        "modules_changed_count": len(changed),
        "modules_changed": [{"key": m.key, "type": m.type_name, "class": m.cls,
                             "rel_err": _num(m.diff.rel_err) if m.diff else None}
                            for m in changed[:MAX_MODULES]],
        "final_output": None if rep.final is None else {
            "rel_err": _num(rep.final.rel_err), "max_abs": _num(rep.final.max_abs)},
        "baseline": rep.baseline,
        "op_repro": None if not rep.op_repro_path else {
            "path": os.path.abspath(rep.op_repro_path),
            "command": f'"{sys.executable}" "{os.path.abspath(rep.op_repro_path)}"'},
        "notes": list(rep.notes),
        "next_actions": next_actions(rep, recorded_path, model_file),
    })
    return doc
