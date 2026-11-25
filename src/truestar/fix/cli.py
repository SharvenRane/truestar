"""`truestar fix model.py`: the command line side of truestar.fix.

The model file is the same as for `truestar check` (get_model, get_inputs, optional get_kwargs),
plus two optional hooks:

* get_context(): a context manager every run happens inside, for example a
  truestar.testing.OpFault that simulates a wrong kernel (examples/fix_demo.py).
* get_calibration(): calibration inputs for `--int8` (default: the inputs themselves).

Exit codes: 0 clean or every backend resolved (or the INT8 target met), 1 improved, unresolved or
not applicable, 2 usage error.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

EXIT_USAGE = 2


def add_parser(sub) -> None:
    f = sub.add_parser("fix", help="find, apply and verify remedies for what check finds")
    f.add_argument("model", nargs="?", help="python file defining get_model() and get_inputs()")
    f.add_argument("--backends", help="comma separated, e.g. cpu,cuda (default: all available)")
    f.add_argument("--mechanism", choices=("mode", "library"), default="mode",
                   help="mode: dispatch mode (default); library: kernel override on the device key")
    f.add_argument("--format", choices=("text", "json", "agent"), default="text")
    f.add_argument("--json", dest="json_path", help="also write the full JSON report here")
    f.add_argument("--repro-dir", default="truestar_repro", help="where repros are written")
    f.add_argument("--reps", type=int, default=20, help="timed forward runs for the cost (0: skip)")
    f.add_argument("--int8", action="store_true",
                   help="precision remedy: fewest layers kept in float for simulated INT8")
    f.add_argument("--target-flips", type=int, default=0, help="with --int8: flips allowed")
    f.add_argument("--list", action="store_true", help="print the remedy table as JSON")


def _load_extra(path: str):
    """get_context and get_calibration from the module cli._load registered."""
    mod = None
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.abspath(f) == os.path.abspath(path):
            mod = m
    ctx = mod.get_context() if mod is not None and hasattr(mod, "get_context") else None
    calib = mod.get_calibration() if mod is not None and hasattr(mod, "get_calibration") else None
    return ctx, calib


def run(a) -> int:
    from ..cli import ModelFileError, _load

    if a.list:
        from .table import table_rows

        print(json.dumps({"remedies": table_rows()}, indent=2))
        return 0
    if not a.model:
        print("error: truestar fix needs a model file", file=sys.stderr)
        return EXIT_USAGE
    quiet = contextlib.redirect_stdout(sys.stderr) if a.format != "text" else contextlib.nullcontext()
    try:
        with quiet:
            model, inputs, kwargs = _load(a.model)
            ctx, calib = _load_extra(a.model)
    except ModelFileError as exc:
        if a.format == "agent":
            from ..agent import dumps, error_to_agent

            doc = error_to_agent("TS020", str(exc), a.model)
            doc["command"] = "fix"
            print(dumps(doc))
        else:
            print(f"error [TS020]: {exc}", file=sys.stderr)
        return EXIT_USAGE
    sims = [s for s in (a.backends or "").split(",") if s.strip().lower().startswith("sim:")]
    if sims and not a.int8:
        msg = (f"truestar fix does not support simulated backends ({', '.join(sims)}); check them "
               "with truestar check --backends")
        if a.format == "agent":
            from ..agent import dumps as _dumps, error_to_agent

            doc = error_to_agent("TS001", msg, a.model)
            doc["command"] = "fix"
            print(_dumps(doc))
        else:
            print(f"error [TS001]: {msg}", file=sys.stderr)
        return EXIT_USAGE
    name = os.path.splitext(os.path.basename(a.model))[0]
    from ..agent import dumps
    from .output import exit_code

    with quiet, (ctx if ctx is not None else contextlib.nullcontext()):
        if a.int8:
            import torch

            from .precision import select_precision

            x = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
            plan = select_precision(model, x, calib, target_flips=a.target_flips, model_name=name)
            doc_full, doc_agent, text = plan.as_dict(), plan.to_agent(), plan.to_text()
            code = 0 if plan.met else 1
        else:
            from .engine import fix

            backends = [s for s in a.backends.split(",") if s] if a.backends else None
            rep = fix(model, inputs, kwargs, backends=backends, mechanism=a.mechanism,
                      repro_dir=a.repro_dir, time_reps=a.reps, model_name=name)
            doc_full, doc_agent, text = rep.as_dict(), rep.to_agent(a.model), rep.to_text()
            code = exit_code(rep.status)
    if a.format == "agent":
        print(dumps(doc_agent))
    elif a.format == "json":
        print(json.dumps(doc_full, indent=2, default=str))
    else:
        print(text)
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(doc_full, f, indent=2, default=str)
    return code
