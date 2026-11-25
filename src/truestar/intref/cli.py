"""`truestar intref`: integer reference conventions (docs/INTREF.md).

    truestar intref conventions                 every rule and convention, JSON
    truestar intref match case.npz              which convention a device output follows, per op
    truestar intref cases                       cited cases: CMSIS-NN#214, ONNX Runtime, pair table
    truestar intref npu                         explain docs/NPU.md from the saved S24 jobs (no jobs)
    truestar intref golden -o golden.h          C golden vectors for a microcontroller harness

A case file (.npz) holds `op` (string), the integer inputs by name (x, w, b, or a, b), `device`
(the device's integer output), and `params` (a JSON string of zero points, scales and shapes,
keys as in truestar.intref.ops).
"""

from __future__ import annotations

import json
import os
import sys

EXIT_OK, EXIT_FINDING, EXIT_USAGE = 0, 1, 2


def add_parser(sub) -> None:
    p = sub.add_parser("intref", help="integer INT8 reference: which rounding convention a device uses")
    s = p.add_subparsers(dest="intref_cmd", required=True)
    c = s.add_parser("conventions", help="list rules and conventions as JSON")
    c.set_defaults(format="agent")
    m = s.add_parser("match", help="which convention a device output follows")
    m.add_argument("case", help=".npz with op, inputs, params (JSON string) and device")
    m.add_argument("--conventions", default=None, help="comma list (default: all)")
    m.add_argument("--format", choices=("text", "agent"), default="text")
    k = s.add_parser("cases", help="cited distinguishing cases and the ONNX Runtime oracle")
    k.add_argument("--format", choices=("text", "agent"), default="text")
    k.add_argument("--no-onnxruntime", action="store_true")
    n = s.add_parser("npu", help="explain the NPU 1 LSB differences from the saved jobs")
    n.add_argument("--cache-dir", default=None)
    n.add_argument("--summary", action="store_true",
                   help="use the recorded summary instead of the job cache")
    n.add_argument("--format", choices=("text", "agent"), default="text")
    g = s.add_parser("golden", help="write a C header of golden vectors")
    g.add_argument("-o", "--output", required=True)
    g.add_argument("--convention", default="tflite_ref")
    g.add_argument("--tolerance", choices=("exact", "portable"), default="exact")
    g.add_argument("--format", choices=("text", "agent"), default="text")


def _envelope(status: str, exit_code: int, codes, body: dict) -> dict:
    from truestar import __version__
    from truestar.codes import Issue

    issues = [Issue(c, m).as_dict() for c, m in codes]
    return {"schema_version": 1, "tool": "truestar", "command": "intref",
            "truestar_version": __version__, "status": status, "exit_code": exit_code,
            "codes": sorted({c for c, _ in codes}), "issues": issues, **body}


def _emit(a, doc: dict, text: str) -> None:
    if getattr(a, "format", "text") == "agent":
        print(json.dumps(doc, allow_nan=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        print(text)


def _load_case(path: str):
    import numpy as np

    z = np.load(path, allow_pickle=False)
    for key in ("op", "device", "params"):
        if key not in z.files:
            raise ValueError(f"case file has no '{key}'")
    op = str(z["op"])
    params = json.loads(str(z["params"]))
    inputs = {k: z[k] for k in z.files if k not in ("op", "device", "params")}
    return op, inputs, params, z["device"]


def run(a) -> int:
    from .conventions import describe

    cmd = a.intref_cmd
    try:
        if cmd == "conventions":
            print(json.dumps({"conventions": describe()}, indent=1))
            return EXIT_OK
        if cmd == "match":
            from .match import match_op

            try:
                op, inputs, params, device = _load_case(a.case)
                convs = a.conventions.split(",") if a.conventions else None
                v = match_op(op, inputs, params, device, convs)
            except (OSError, ValueError, KeyError) as exc:
                doc = _envelope("error", EXIT_USAGE, [("TS544", str(exc))], {})
                _emit(a, doc, f"truestar intref match: {exc}")
                return EXIT_USAGE
            code = EXIT_FINDING if v.code == "TS541" else EXIT_OK
            doc = _envelope("no_convention" if v.code == "TS541" else "ok", code,
                            [(v.code, v.text)], {"op": v.as_dict()})
            _emit(a, doc, v.text + "\n" + "\n".join(
                f"  {r.convention:18s} {r.rate:8.4%}  worst {r.max_lsb} LSB" for r in v.results))
            return code
        if cmd == "cases":
            from .cases import cases_text, run_cases

            res = run_cases(with_onnxruntime=not a.no_onnxruntime)
            _emit(a, _envelope("ok", EXIT_OK, [], {"cases": res}), cases_text(res))
            return EXIT_OK
        if cmd == "npu":
            from . import npu

            try:
                e = npu.from_summary() if a.summary else npu.explain(a.cache_dir)
            except FileNotFoundError as exc:
                msg = f"{exc}; rerun with --summary to use the recorded numbers"
                _emit(a, _envelope("error", EXIT_USAGE, [("TS544", msg)], {}), msg)
                return EXIT_USAGE
            code = EXIT_FINDING if e.code == "TS543" else EXIT_OK
            _emit(a, _envelope("not_rounding" if e.code == "TS543" else "ok", code,
                               [(e.code, e.verdict)], {"npu": e.as_dict()}), e.to_text())
            return code
        if cmd == "golden":
            from .golden import build_layers, demo_network, emit_header, verify_roundtrip

            x, specs = demo_network()
            layers = build_layers(x, specs, a.convention, a.tolerance)
            text = emit_header(layers, a.convention, a.tolerance)
            bad = verify_roundtrip(layers, text)
            if bad:
                _emit(a, _envelope("error", EXIT_FINDING, [("TS545", "; ".join(bad))], {}),
                      "golden header failed its parse back check: " + "; ".join(bad))
                return EXIT_FINDING
            with open(a.output, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            info = [{"name": L.name, "op": L.op, "elements": int(L.expected.size), "tol_lsb": L.tol_lsb,
                     "spread": L.spread} for L in layers]
            _emit(a, _envelope("ok", EXIT_OK, [], {"output": os.path.abspath(a.output),
                                                   "convention": a.convention, "layers": info}),
                  f"wrote {a.output}: {len(layers)} layers under {a.convention}, tolerance {a.tolerance}; "
                  "parsed back and checked")
            return EXIT_OK
    except ValueError as exc:
        print(f"truestar intref {cmd}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_USAGE
