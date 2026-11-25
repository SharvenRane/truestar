"""`truestar compile model.py --precision auto --max-flips N`: the command line side of
truestar.precplan.

The model file is the one `truestar check` takes (get_model, get_inputs), plus optional
get_calibration() (a list of more forward argument tuples, used with get_inputs() for planning)
and get_holdout() (forward argument tuples that only verify the finished plan). Without
get_holdout() the calibration inputs are split in two halves.

Exit codes: 0 a plan within budget on the calibration and held out inputs, 1 the budget cannot be
met (the plan is float32 everywhere) or a kernel failed its exact check, 2 usage error.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

EXIT_OK, EXIT_FINDING, EXIT_USAGE = 0, 1, 2
# kept here so building the parser does not import torch (tests check it equals units.PRECISIONS)
PRECISION_NAMES = ("fp32", "tf32", "bf16", "fp16", "int8w", "fp8", "int4w")


def add_parser(sub) -> None:
    c = sub.add_parser("compile", help="choose a precision per layer: fastest plan on this GPU whose "
                                       "decision flips against float64 stay within a budget")
    c.add_argument("model", help="python file defining get_model() and get_inputs()")
    c.add_argument("--precision", choices=("auto",), default="auto",
                   help="auto: plan per layer (the only mode; uniform precisions are reported as "
                        "baselines)")
    c.add_argument("--max-flips", default="0",
                   help="decisions allowed to change against float64: a count on the calibration "
                        "inputs (scaled to the held out size) or a share such as 0.1%% (default 0)")
    c.add_argument("--precisions", default=",".join(PRECISION_NAMES),
                   help="candidate precisions, comma separated (fp32 is always included)")
    c.add_argument("--plan", dest="plan_path", help="write the plan JSON here (apply it with "
                                                     "truestar.precplan.apply)")
    c.add_argument("--no-range", action="store_true", help="skip the range check")
    c.add_argument("--no-baselines", action="store_true", help="skip the uniform precision baselines")
    c.add_argument("--refine", type=int, default=6, help="greedy refinement steps after the "
                                                         "bisection (default 6)")
    c.add_argument("--top", type=int, default=60, help="units to list in the text report")
    c.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    c.add_argument("--format", choices=("text", "agent"), default="text")


def _module_of(path: str):
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f) == os.path.abspath(path):
            return m
    return None


def run(a) -> int:
    from ..agent import dumps, error_to_agent
    from ..cli import ModelFileError, _load

    agent = a.format == "agent"

    def fail(code: str, msg: str) -> int:
        if agent:
            doc = error_to_agent(code, msg, a.model)
            doc["command"] = "compile"
            print(dumps(doc))
        else:
            print(f"error [{code}]: {msg}", file=sys.stderr)
        return EXIT_USAGE

    precisions = [p.strip() for p in a.precisions.split(",") if p.strip()]
    if "fp32" not in precisions:
        precisions.insert(0, "fp32")
    bad = [p for p in precisions if p not in PRECISION_NAMES]
    if bad:
        return fail("TS021", f"unknown precision(s) {', '.join(bad)}; known: {', '.join(PRECISION_NAMES)}")
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    with quiet:
        try:
            model, inputs, kwargs = _load(a.model)
        except ModelFileError as exc:
            return fail("TS020", str(exc))
        if kwargs:
            return fail("TS021", "truestar compile takes positional tensor inputs only; get_kwargs() "
                                 "is not supported")
        mod = _module_of(a.model)
        calib = [inputs]
        hold = None
        try:
            if mod is not None and hasattr(mod, "get_calibration"):
                calib += list(mod.get_calibration())
            if mod is not None and hasattr(mod, "get_holdout"):
                hold = mod.get_holdout()
        except Exception as exc:
            return fail("TS020", f"get_calibration() / get_holdout() failed: {type(exc).__name__}: {exc}")
        from .planner import Budget
        from .units import PlanError
        from . import planner

        try:
            Budget.parse(a.max_flips)
        except ValueError as exc:
            return fail("TS021", str(exc))
        import torch

        if not torch.cuda.is_available():
            return fail("TS669", "truestar compile --precision auto times real kernels and needs a "
                                 "CUDA GPU; none is visible (nothing is planned on the CPU)")
        try:
            plan = planner.plan(model, calib, hold, max_flips=a.max_flips, precisions=precisions,
                                use_range=not a.no_range, baselines=not a.no_baselines,
                                refine=a.refine,
                                model_name=os.path.splitext(os.path.basename(a.model))[0])
        except PlanError as exc:
            msg = str(exc)
            code = msg[:5] if msg.startswith("TS6") else "TS021"
            return fail(code, msg)
    doc = plan.to_agent(a.model)
    if agent:
        print(dumps(doc))
    else:
        print(plan.to_text(top=a.top))
    if a.plan_path:
        plan.save(a.plan_path)
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2, allow_nan=False))
    if not plan.met or "TS664" in plan.codes:
        return EXIT_FINDING
    return EXIT_OK
