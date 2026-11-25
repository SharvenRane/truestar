"""`truestar range model.py --target fp16`: the command line side of truestar.rangecheck.

The model file is the one `truestar check` takes (get_model, get_inputs, optional get_kwargs is not
supported here), plus an optional get_calibration() returning a list of more forward argument tuples
with the same shapes. Exit codes: 0 every value fits and the target run matches its twin, 1 a finding
(overflow, near overflow, underflow, precision loss), also when a remedy was verified, 2 usage error.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

EXIT_USAGE = 2


# kept here (not imported from .formats) so building the parser does not import torch;
# tests/test_rangecheck.py checks it equals sorted(formats.TARGETS)
TARGET_NAMES = ("ane", "bf16", "fp16", "fp8-e4m3", "fp8-e5m2", "webgpu-f16")


def add_parser(sub) -> None:
    r = sub.add_parser("range", help="predict low precision overflow and underflow before deployment, "
                                     "with a verified remedy")
    r.add_argument("model", help="python file defining get_model() and get_inputs()")
    r.add_argument("--target", default="fp16", choices=TARGET_NAMES,
                   help="fp16, bf16, fp8-e4m3, fp8-e5m2, ane (fp16, NaN to infinity), webgpu-f16 "
                        "(fp16, decomposed norms)")
    r.add_argument("--headroom", type=float, default=1.0,
                   help="bits of margin a value must keep below the format's maximum (default 1)")
    r.add_argument("--no-remedy", action="store_true", help="report sites only, do not try remedies")
    r.add_argument("--top", type=int, default=12, help="sites to list in the text report")
    r.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    r.add_argument("--format", choices=("text", "agent"), default="text")


def run(a) -> int:
    from ..agent import dumps, error_to_agent
    from ..cli import ModelFileError, _load
    from .output import exit_code

    agent = a.format == "agent"

    def fail(code: str, msg: str) -> int:
        if agent:
            doc = error_to_agent(code, msg, a.model)
            doc["command"] = "range"
            print(dumps(doc))
        else:
            print(f"error [{code}]: {msg}", file=sys.stderr)
        return EXIT_USAGE

    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    with quiet:
        try:
            model, inputs, kwargs = _load(a.model)
        except ModelFileError as exc:
            failure = ("TS020", str(exc))
        else:
            failure = None
        if failure is None and kwargs:
            failure = ("TS021", "truestar range takes positional tensor inputs only; get_kwargs() is "
                                "not supported")
        if failure is None:
            calib = None
            for m in list(sys.modules.values()):
                f = getattr(m, "__file__", None)
                if f and os.path.abspath(f) == os.path.abspath(a.model) and hasattr(m, "get_calibration"):
                    calib = m.get_calibration()
            from .analysis import range_check

            try:
                rep = range_check(model, inputs, target=a.target, headroom_bits=a.headroom,
                                  calibration=calib, remedies=not a.no_remedy,
                                  model_name=os.path.splitext(os.path.basename(a.model))[0])
            except Exception as exc:  # import or shape errors name the op and line themselves
                failure = ("TS021", f"{type(exc).__name__}: {exc}")
    if failure is not None:
        return fail(*failure)
    doc = rep.to_agent(a.model)
    if agent:
        print(dumps(doc))
    else:
        print(rep.to_text(top=a.top))
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2, allow_nan=False))
    return exit_code(rep.status)
