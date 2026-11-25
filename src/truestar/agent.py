"""Agent output: compact, stable, versioned JSON for AI coding agents (`--format agent`).

The schema is documented in docs/AGENT_INTERFACE.md. Rules that keep it safe to act on:

* `schema_version` changes only when a field is removed or changes meaning. Adding fields does not
  bump it, so agents must ignore unknown fields.
* Strict JSON: no NaN or Infinity. A number that is not finite or was not measured is null, and the
  `nonfinite_mismatch` / `shape_mismatch` flags say why.
* Every finding carries a stable code from truestar.codes.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .codes import CODES, INFO, Issue, severity, slug

if TYPE_CHECKING:
    from torch import nn

    from .check import BackendResult, CheckReport

# This module does not import torch, so error documents (and the MCP server) stay light.
OK = "ok"  # truestar.compare.OK


def _torch_version() -> str:
    t = sys.modules.get("torch")
    if t is not None:
        return t.__version__
    try:
        from importlib.metadata import version

        return version("torch")
    except Exception:
        return "unknown"

SCHEMA_VERSION = 1
EXIT_OK, EXIT_DIVERGENT, EXIT_USAGE = 0, 1, 2
MAX_MODULES = 5  # modules_over_threshold is capped so the output stays small


# ----------------------------------------------------------------------------- source lookup


def _library_roots() -> List[str]:
    roots = []
    for mod in ("torch", "torchvision", "torchaudio", "transformers", "timm"):
        m = sys.modules.get(mod)
        f = getattr(m, "__file__", None) if m is not None else None
        if f:
            roots.append(os.path.dirname(os.path.abspath(f)))
    roots.append(os.path.dirname(os.path.abspath(__file__)))  # truestar itself
    return [os.path.normcase(r) for r in roots]


def _is_library(path: str) -> bool:
    p = os.path.normcase(os.path.abspath(path))
    if any(p.startswith(r + os.sep) for r in _library_roots()):
        return True
    parts = p.replace("\\", "/").split("/")
    return "site-packages" in parts or "dist-packages" in parts


def class_source(cls: type) -> Optional[Dict[str, Any]]:
    """File and first line of a class definition, or None when Python cannot tell."""
    try:
        path = inspect.getsourcefile(cls)
        if path is None:
            return None
        _, line = inspect.getsourcelines(cls)
    except (TypeError, OSError):
        return None
    return {
        "file": os.path.abspath(path),
        "line": int(line),
        "qualname": f"{cls.__module__}.{cls.__qualname__}",
        "user_code": not _is_library(path),
    }


def module_source(model: "nn.Module", name: str) -> Dict[str, Any]:
    """Source location for the module at `name` inside `model`.

    `module` is where the module's own class is defined. `user` is the nearest module, itself or an
    ancestor, whose class is defined outside torch and other installed libraries: that is the code
    the user can edit (for a blamed nn.Linear inside a user's MLP it points at the MLP class).
    """
    from .capture import ROOT

    chain = [ROOT] if name == ROOT else [ROOT] + [
        ".".join(name.split(".")[: i + 1]) for i in range(len(name.split(".")))
    ]
    mods = [(n, model if n == ROOT else model.get_submodule(n)) for n in chain]
    own = class_source(type(mods[-1][1]))
    user = None
    for n, m in reversed(mods):
        src = class_source(type(m))
        if src is not None and src["user_code"]:
            user = dict(src, module=n)
            break
    return {"module": own, "user": user}


# ----------------------------------------------------------------------------- helpers


def _num(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def exit_code_for(status: str) -> int:
    return EXIT_OK if status == OK else EXIT_DIVERGENT


def _repro_info(path: Optional[str], device: str) -> Optional[Dict[str, str]]:
    if not path:
        return None
    return {
        "path": os.path.abspath(path),
        "command": f'"{sys.executable}" "{os.path.abspath(path)}"',
        "control_command": f'"{sys.executable}" "{os.path.abspath(path)}" cpu',
        "device": device,
    }


def _backend(r: "BackendResult") -> Dict[str, Any]:
    from .compare import LOCAL_FACTOR

    fd = r.first_divergent
    blamed = None
    if fd is not None:
        iso = fd.isolated
        rel = _num(iso.rel_err) if iso is not None else None
        src = r.blame_source or {}
        blamed = {
            "key": fd.key,
            "name": fd.name,
            "type": fd.type_name,
            "leaf": fd.is_leaf,
            "own_code": not fd.is_leaf,  # True: the error is in this module's own forward code
            "status": fd.status,
            "rel_err": rel,
            "threshold": fd.threshold,
            "ratio": None if rel is None else rel / fd.threshold,
            "local_err": _num(iso.local_err) if iso is not None else None,
            "local_limit": None if fd.threshold is None else fd.threshold * LOCAL_FACTOR,
            "local_at": (iso.local_at or None) if iso is not None else None,
            "localized": (iso is not None and fd.threshold is not None
                          and bool(iso.localized(fd.threshold))),
            "nonfinite_mismatch": bool(iso.nonfinite_mismatch) if iso is not None else False,
            "shape_mismatch": bool(iso.shape_mismatch) if iso is not None else False,
            "error": fd.error or None,
            "source": src.get("module"),
            "user_source": src.get("user"),
        }
    over = [l for l in r.layers if l.status != OK]
    output = None
    if r.output is not None:
        output = {
            "rel_err": _num(r.output.rel_err),
            "max_abs": _num(r.output.max_abs),
            "nonfinite_mismatch": bool(r.output.nonfinite_mismatch),
            "shape_mismatch": bool(r.output.shape_mismatch),
            "local_err": _num(r.output.local_err),
            "local_at": r.output.local_at or None,
        }
    return {
        "name": r.backend.name,
        "device": r.backend.device,
        "description": r.backend.description,
        "status": r.status,
        "codes": r.codes,
        "output": output,
        "decisions": {"total": r.decisions, "flips": r.flips, "near_tie_flips": r.tie_flips},
        "blamed_module": blamed,
        "modules_checked": len([l for l in r.layers if l.replayed]),
        "modules_over_threshold_count": len(over),
        "modules_over_threshold": [
            {
                "key": l.key,
                "type": l.type_name,
                "status": l.status,
                "rel_err": _num(l.isolated.rel_err) if l.isolated is not None else None,
                "local_err": _num(l.isolated.local_err) if l.isolated is not None else None,
                "accumulated_rel_err": _num(l.propagated.rel_err) if l.propagated else None,
                "threshold": l.threshold,
            }
            for l in over[:MAX_MODULES]
        ],
        "blamed_op": _op(r),
        "repro": _repro_info(r.repro_path, r.backend.device),
        "op_repro": _repro_info(getattr(r, "op_repro_path", None), r.backend.device),
        "error": r.error or None,
        "stack": _stack(r),
        "precision": r.precision,
    }


def _stack(r: "BackendResult") -> Optional[Dict[str, Any]]:
    """Compiler or runtime backends: how the engine was checked and localized (JSON safe)."""
    s = getattr(r, "stack", None)
    if not s:
        return None
    return {k: v for k, v in s.items() if not callable(v)}


def _op(r: "BackendResult") -> Optional[Dict[str, Any]]:
    """The exact ATen op inside the blamed module, when op drilldown found one."""
    op = getattr(r, "first_divergent_op", None)
    if op is None:
        return None
    rel = _num(op.isolated.rel_err) if op.isolated is not None else None
    return {
        "op": op.op,
        "module": op.module,
        "index": op.index,
        "total": op.total,
        "inputs": [{"dtype": d, "shape": list(sh)} for d, sh in zip(op.input_dtypes, op.input_shapes)],
        "rel_err": rel,
        "threshold": op.threshold,
        "ratio": None if rel is None else rel / op.threshold,
        "runs_in_reference": bool(op.matched),
        "kernel": getattr(op, "kernel", None),
    }


def next_actions(report: "CheckReport") -> List[Dict[str, Any]]:
    """Short, ordered list of what an agent should do next. Empty when everything matches."""
    acts: List[Dict[str, Any]] = []
    torch_pin = f"torch=={_torch_version().split('+')[0]}"

    def add(kind: str, description: str, command: Optional[str] = None, **extra):
        a = {"kind": kind, "description": description, "command": command}
        a.update(extra)
        acts.append(a)

    if any(i.code == "TS003" for i in report.issues):
        add("note_reference_degraded",
            "The float64 reference could not run, so CPU results are not checked. Find the op with "
            "no float64 kernel if CPU correctness matters.")
    for r in report.results:
        if r.status == OK:
            continue
        b = r.backend.name
        if r.error:
            add("inspect_failure",
                f"The model raised on {b}: {r.error}. Fix or report that error first; nothing "
                "else on this backend was checked.", backend=b)
            continue
        fd = r.first_divergent
        repro = _repro_info(r.repro_path, r.backend.device)
        op_repro = _repro_info(getattr(r, "op_repro_path", None), r.backend.device)
        if op_repro:
            op = r.first_divergent_op
            what = "ONNX node" if r.backend.kind == "webgpu" else "PyTorch op"
            add("confirm_op_repro",
                f"Run the single op repro for {op.op} on {b}; exit code 1 shows this one {what} "
                "is wrong on its own. This is the smallest possible bug report.",
                op_repro["command"], backend=b, op=op.op)
        if repro:
            add("confirm_repro",
                f"Run the repro on {b}; exit code 1 confirms the divergence on its own.",
                repro["command"], backend=b)
            add("control_repro",
                "Run the same repro on CPU; exit code 0 shows the module itself is fine and the "
                f"{b} kernel is at fault.", repro["control_command"], backend=b)
        if fd is not None:
            src = r.blame_source or {}
            own, user = src.get("module"), src.get("user")
            if own is not None and own["user_code"]:
                add("inspect_source",
                    f"The blamed module {fd.key} is your class {own['qualname']}. Check its forward "
                    f"for code that behaves differently on {b}.",
                    file=own["file"], line=own["line"], backend=b)
            elif r.backend.kind == "webgpu":
                if user is not None:
                    add("inspect_source",
                        f"The blamed module {fd.key} ({fd.type_name}) is used inside your class "
                        f"{user['qualname']}.", file=user["file"], line=user["line"], backend=b)
                _webgpu_actions(add, r, b, repro, op_repro)
            elif own is not None and own["qualname"].startswith("truestar.testing."):
                add("injected_fault",
                    f"The blamed module {fd.key} is a truestar.testing fault injector "
                    f"({fd.type_name}): this divergence was put there on purpose.", backend=b)
            else:
                owner = own["qualname"].split(".")[0] if own is not None else "unknown"
                if user is not None:
                    add("inspect_source",
                        f"The blamed module {fd.key} ({fd.type_name}) is a library class; it is "
                        f"used inside your class {user['qualname']}.",
                        file=user["file"], line=user["line"], backend=b)
                add("pin_torch",
                    f"The divergence is inside a library module ({fd.type_name}) on {b} with "
                    f"{torch_pin}. Run the repro under another torch release and pin one that "
                    "passes, or move this module to a backend that passes.",
                    current=torch_pin, backend=b)
                if repro:
                    add("report_upstream",
                        f"Report to PyTorch"
                        + ("" if owner == "torch" else f" (or {owner}, which defines "
                           f"{fd.type_name}, if its code branches on device)")
                        + f" with the repro folder {os.path.dirname(repro['path'])} "
                        + ("(op_repro.py is the smallest report, plus repro.py and payloads), "
                           if op_repro else "(repro.py plus payload.pt), ")
                        + f"{torch_pin} and the {b} device description.",
                        backend=b, repro_dir=os.path.dirname(repro["path"]))
        codes = r.codes
        if "TS012" in codes:
            add("wrap_in_module",
                "The output diverges but no module is to blame: move the code between modules "
                "into an nn.Module and rerun to locate it.", backend=b)
        if "TS004" in codes:
            add("check_control_flow",
                f"Different modules ran on {b} than on the reference: look for branches on device, "
                "shape or data.", backend=b)
        if "TS013" in codes:
            add("check_flips",
                f"{r.flips - r.tie_flips} predictions changed on {b}. Treat the backend as "
                "not safe for this model until the blamed module is fixed.", backend=b)
    return acts


def _webgpu_actions(add, r, b, repro, op_repro) -> None:
    """Remedies for a WebGPU finding: the runtime is ONNX Runtime's WebGPU EP, not torch."""
    op = r.first_divergent_op
    opts = r.backend.options or {}
    low = op is not None and any("float16" in d for d in op.input_dtypes)
    if (low and any(k in op.op for k in ("MatMul", "Gemm"))
            and opts.get("enableMatmulFp32Accumulation") != "1"):
        add("try_ep_option",
            f"Rerun with --backends \"{b}+enableMatmulFp32Accumulation=1\": ONNX Runtime's WebGPU "
            "EP accumulates some float16 MatMul shapes in float16 by default (docs/WEBGPU.md). "
            "If the rerun is clean, set that EP option where the model is deployed.",
            backend=b, ep_option="enableMatmulFp32Accumulation=1")
    if repro:
        info = r.backend.info or {}
        add("report_upstream",
            "Report to ONNX Runtime (WebGPU EP) with the repro folder "
            f"{os.path.dirname(repro['path'])} ("
            + ("op_repro.py is the smallest report, " if op_repro else "")
            + f"repro.py, the .onnx files and inputs), onnxruntime {info.get('onnxruntime', '?')}, "
            f"onnxruntime-ep-webgpu {info.get('ep_version', '?')} and the adapter: "
            f"{r.backend.description}.", backend=b, repro_dir=os.path.dirname(repro["path"]))


# ----------------------------------------------------------------------------- documents


def _envelope(status: str, exit_code: int) -> Dict[str, Any]:
    from . import __version__

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "truestar",
        "command": "check",
        "truestar_version": __version__,
        "torch_version": _torch_version(),
        "status": status,
        "exit_code": exit_code,
    }


def report_to_agent(report: "CheckReport", model_file: Optional[str] = None) -> Dict[str, Any]:
    doc = _envelope(report.status, exit_code_for(report.status))
    doc.update({
        "model": {
            "name": report.model_name,
            "file": os.path.abspath(model_file) if model_file else None,
            "dtype": str(report.dtype).replace("torch.", ""),
        },
        "reference": report.reference,
        "codes": report.codes,
        "issues": [i.as_dict() for i in report.all_issues],
        "backends": [_backend(r) for r in report.results],
        "skipped": [
            {"name": n, "reason": why, "code": "TS001", "severity": INFO} for n, why in report.skipped
        ],
        "next_actions": next_actions(report),
    })
    return doc


def error_to_agent(code: str, message: str, model_file: Optional[str] = None,
                   **extra: Any) -> Dict[str, Any]:
    """Agent document for a run that could not produce a report (exit code 2)."""
    doc = _envelope("error", EXIT_USAGE)
    issue = Issue(code, message).as_dict()
    actions = []
    if code == "TS001":
        avail = ",".join(extra.get("available") or ["cpu"])
        target = f'"{os.path.abspath(model_file)}"' if model_file else "<model.py>"
        actions.append({"kind": "use_available_backends",
                        "description": "Rerun with backends that exist here (listed in "
                                       "`available`).",
                        "command": f"truestar check {target} --backends {avail} --format agent"})
    elif code == "TS020":
        actions.append({"kind": "fix_model_file",
                        "description": "The model file must define get_model() returning an "
                                       "nn.Module and get_inputs() returning a tensor or tuple.",
                        "command": None})
    doc.update({
        "model": {"name": None, "file": os.path.abspath(model_file) if model_file else None,
                  "dtype": None},
        "reference": None,
        "codes": [code],
        "issues": [issue],
        "backends": [],
        "skipped": [],
        "next_actions": actions,
    })
    doc.update(extra)
    return doc


def dumps(doc: Dict[str, Any]) -> str:
    """Strict JSON (raises on NaN or Infinity so a broken document is never emitted)."""
    return json.dumps(doc, indent=2, allow_nan=False, sort_keys=False)


def _code_number(code: str) -> int:
    """TS1060 sorts after TS886: codes compare by number, not as text (four digit codes, 2026-11-24)."""
    return int(code[2:])


def code_table() -> List[Dict[str, str]]:
    return [{"code": c, "severity": severity(c), "slug": slug(c), "meaning": CODES[c][2]}
            for c in sorted(CODES, key=_code_number)]
