"""`truestar conform`: check a deployed artifact against its PyTorch model (docs/CONFORM.md).

    truestar conform model.py model.onnx                      default runtimes for the format
    truestar conform model.py model.tflite --runtimes tflite:interp,tflite:gpu --golden model.tsgold
    truestar conform model.py model.pte --inputs calib.pt --format agent
    truestar conform replay model.tsgold model.onnx           after an upgrade: what changed
    truestar conform runtimes                                 every runtime and why it cannot run
    truestar conform qaihub RESULTS_DIR                       saved AI Hub battery outputs (docs/QAIHUB.md)
    truestar conform tpu RESULTS_DIR                          saved TPU battery outputs vs float64 (docs/TPU.md)
    truestar conform llamacpp MODEL.gguf|HF_DIR --llama-cpp DIR   llama.cpp vs float64 (docs/CONFORM_LLAMACPP.md)
    truestar conform vllm [RESULTS_DIR]                       a GPU serving engine vs float64 and vs itself
    truestar conform sglang [RESULTS_DIR]                     (docs/CONFORM_SERVING.md)
    truestar conform coreml RESULTS_DIR                       saved Core ML battery runs (docs/COREML.md)

Exit codes: 0 every runtime conforms (replay: identical or changed within tolerance), 1 a finding,
2 usage error (bad arguments, unreadable model file or artifact, unknown runtime).
"""

from __future__ import annotations

import json
import os
import sys

EXIT_OK, EXIT_FINDING, EXIT_USAGE = 0, 1, 2


def add_parser(sub) -> None:
    p = sub.add_parser("conform", help="check a deployed .onnx, .tflite, .pte or .tsx file against "
                                       "the PyTorch model, in its own runtime")
    p.add_argument("model", help="python file with get_model() and get_inputs(), or 'replay', 'runtimes', "
                                   "'vllm' or 'sglang'")
    p.add_argument("artifact", nargs="?", help="the deployed file (for replay: the bundle, then the file)")
    p.add_argument("extra", nargs="?", help=argparse_suppress())
    p.add_argument("--runtimes", default=None, help="comma list, e.g. ort:cpu,ort:webgpu (default: the "
                                                    "format's runtimes that can run here)")
    p.add_argument("--samples", type=int, default=8, help="inputs drawn around get_inputs() (default 8)")
    p.add_argument("--inputs", default=None, help=".pt file with a list of samples (each a tensor or tuple)")
    p.add_argument("--golden", default=None, help="write the golden bundle here (default: <artifact>.tsgold)")
    p.add_argument("--no-golden", action="store_true", help="do not write a bundle")
    p.add_argument("--no-localize", action="store_true", help="skip module localization")
    p.add_argument("--out", default=None, help="vllm, sglang: where a run here writes its results directory "
                                               "(default truestar_serving_<engine>)")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"),
                   help="vllm, sglang: serving dtype of a run here (default bfloat16)")
    p.add_argument("--format", choices=("text", "agent"), default="text")
    g = p.add_argument_group("llamacpp (truestar conform llamacpp MODEL.gguf|HF_DIR)")
    g.add_argument("--llama-cpp", default=None, help="llama.cpp source tree built with cmake, or install prefix "
                                                     "(default: $TRUESTAR_LLAMA_CPP)")
    g.add_argument("--formats", default=None, help="for a model directory: comma list of f32,f16,q8_0,q4_k_m (default all)")
    g.add_argument("--source", default=None, help="for a GGUF: the Hugging Face directory it came from (conversion and "
                                                  "quantization verdicts)")
    g.add_argument("--threads", default="1,4", help="engine threads to try (default 1,4)")
    g.add_argument("--batch", default="512,7", help="engine batch and ubatch sizes to try (default 512,7)")
    g.add_argument("--flash", default="off,on", help="flash attention settings to try (default off,on)")
    g.add_argument("--kv", default="f16", choices=("f16", "f32"), help="KV cache type (default f16, llama.cpp's)")
    g.add_argument("--new-tokens", type=int, default=16, help="greedy steps per prompt (default 16)")
    g.add_argument("--workdir", default=None, help="where converted GGUF files go (default a temporary directory)")


def argparse_suppress():
    import argparse

    return argparse.SUPPRESS


def _envelope(status, exit_code, codes, body):
    from truestar import __version__
    from truestar.codes import Issue

    return {"schema_version": 1, "tool": "truestar", "command": "conform", "truestar_version": __version__,
            "status": status, "exit_code": exit_code, "codes": sorted({c for c, _ in codes}),
            "issues": [Issue(c, m).as_dict() for c, m in codes], **body}


def _emit(a, doc, text):
    if a.format == "agent":
        print(json.dumps(doc, allow_nan=False, default=str))
    else:
        print(text)


def _usage(a, code, msg, actions=None) -> int:
    if a.format == "agent":
        print(json.dumps(_envelope("error", EXIT_USAGE, [(code, msg)], {"next_actions": actions or []})))
    else:
        print(f"truestar conform: {msg}", file=sys.stderr)
    return EXIT_USAGE


def _next_actions(report, a):
    acts = []
    for r in report.results:
        if r.blamed:
            b = r.blamed
            src = b.user_source or {}
            acts.append({"kind": "fix_conversion" if b.cause == "conversion" else "report_runtime",
                         "runtime": r.runtime, "module": b.module, "file": src.get("file"),
                         "line": src.get("line"),
                         "description": (f"module {b.module}: the file computes something else than the model "
                                         "(conversion); re-export or fix the exporter's decomposition for it"
                                         if b.cause == "conversion" else
                                         f"module {b.module}: {r.runtime} computes the file wrongly; report it "
                                         "with the golden bundle")})
        res = r.residency or {}
        if res.get("off_target"):
            acts.append({"kind": "inspect_fallback", "runtime": r.runtime,
                         "description": f"{len(res['off_target'])} node(s) ran off {res['requested']}: "
                                        + ", ".join(p["op"] for p in res["off_target"][:5])})
        if r.precision.get("observed") == "float16":
            acts.append({"kind": "force_precision", "runtime": r.runtime,
                         "description": "the runtime computes in float16; force float32 (tflite:gpu-f32) or "
                                        "accept float16 and check decisions"})
    if report.golden:
        acts.append({"kind": "replay_after_upgrade", "description": "after a runtime, SDK or driver change",
                     "command": f"truestar conform replay {report.golden} {report.artifact['path']}"})
    return acts


def _llamacpp(a) -> int:
    from .llamacpp import LlamaCppError, Setting, conform_llamacpp

    if not a.artifact:
        return _usage(a, "TS021", "usage: truestar conform llamacpp MODEL.gguf|HF_DIR [--llama-cpp DIR]")
    try:
        threads = [int(x) for x in a.threads.split(",")]
        batch = [int(x) for x in a.batch.split(",")]
        flash = [{"on": True, "off": False}[x.strip()] for x in a.flash.split(",")]
    except (ValueError, KeyError):
        return _usage(a, "TS021", "--threads and --batch take integers, --flash takes on and off")
    if min(threads + batch) < 1 or a.new_tokens < 1:
        return _usage(a, "TS021", "--threads, --batch and --new-tokens must be at least 1")
    settings = [Setting(t, b, fl, a.kv) for t in threads for b in batch for fl in flash]
    formats = a.formats.split(",") if a.formats else None
    try:
        rep = conform_llamacpp(a.artifact, llama_cpp=a.llama_cpp, formats=formats, settings=settings,
                               n_new=a.new_tokens, source=a.source, workdir=a.workdir)
    except LlamaCppError as exc:
        return _usage(a, exc.code, exc.message)
    code = EXIT_FINDING if rep.finding else EXIT_OK
    acts = []
    for c, m in rep.issues:
        if c == "TS1311" or c == "TS1312":
            acts.append({"kind": "report_engine", "description": "llama.cpp computes the file differently from its "
                                                                 "float64 replay: " + m})
        elif c == "TS1316":
            acts.append({"kind": "fix_conversion", "description": m})
    doc = _envelope("divergent" if code else "ok", code, rep.issues,
                    {"mode": "llamacpp", **rep.as_dict(), "next_actions": acts})
    _emit(a, doc, rep.to_text())
    return code


SERVING_NAMES = ("vllm", "sglang", "tensorrt-llm", "tensorrt_llm", "trtllm")


def _serving_unavailable(engine: str) -> "str | None":
    """Why `engine` cannot run here, or None. The package is looked up before CUDA is touched."""
    import importlib.util

    if importlib.util.find_spec(engine) is None:
        return f"{engine} is not installed here"
    import torch

    if not torch.cuda.is_available():
        return f"{engine} is installed but there is no CUDA GPU here"
    return None


def _serving(a) -> int:
    """`truestar conform vllm|sglang [RESULTS_DIR]` (docs/CONFORM_SERVING.md)."""
    from .serving_battery import REFUSED

    engine = a.model
    if engine in REFUSED:
        return _usage(a, "TS1321", f"{REFUSED[engine]} is not supported in this release (it is listed for after "
                                   "launch in docs/CONFORM_SERVING.md); supported serving engines: vllm, sglang",
                      [{"kind": "use_supported_engine", "description": "check the same model with vLLM or SGLang",
                        "command": "truestar conform vllm"}])
    from .serving import ServingBundleError, check, to_text

    results = a.artifact
    if results is None:
        why = _serving_unavailable(engine)
        out = a.out or f"truestar_serving_{engine}"
        if why:
            return _usage(a, "TS1320", f"{why}; run the battery where {engine} and a CUDA GPU are, then judge its "
                                       "results directory here (no GPU needed)",
                          [{"kind": "run_battery_on_gpu_host",
                            "description": f"on a Linux machine with {engine} and a CUDA GPU",
                            "command": f"python -m truestar.conform.serving_battery run --engine {engine} "
                                       f"--out {out} --dtype {a.dtype}"},
                           {"kind": "judge_results", "description": "then, anywhere",
                            "command": f"truestar conform {engine} {out}"}])
        from .serving_battery import run_all
        from .serving_model import ServeSpec

        results = run_all(engine, out, ServeSpec(dtype=a.dtype))
    try:
        rep = check(results, engine)
    except ServingBundleError as exc:
        return _usage(a, "TS1322", str(exc))
    code = EXIT_FINDING if rep.finding else EXIT_OK
    found = {c for c, _ in rep.issues}
    status = ("divergent" if found & {"TS1322", "TS1323", "TS1324"} else
              "nondeterministic" if rep.finding else "ok")
    doc = _envelope(status, code, rep.issues,
                    {"mode": "serving", **rep.as_dict(), "next_actions": []})
    _emit(a, doc, to_text(rep))
    return code


def run(a) -> int:
    from ..cli import ModelFileError, _load
    from ..deps import MissingDependencyError
    from .artifact import ArtifactError
    from .golden import GoldenError

    if a.model == "runtimes":
        from . import runtimes as R

        rows = []
        for n, (fmt, tgt) in R.RUNTIMES.items():
            why = R.make(n).unavailable()
            rows.append({"runtime": n, "format": fmt, "available": why is None, "reason": why})
        print(json.dumps({"runtimes": rows}, indent=None if a.format == "agent" else 1))
        return EXIT_OK
    if a.model == "qaihub":  # saved Qualcomm AI Hub outputs of the fixed battery (docs/QAIHUB.md)
        if not a.artifact:
            return _usage(a, "TS021", "usage: truestar conform qaihub RESULTS_DIR [--runtimes RUN,...]")
        from .qaihub import QaiHubBundleError, check as qcheck, is_finding, to_text as qtext

        try:
            verdicts, top = qcheck(a.artifact, a.runtimes.split(",") if a.runtimes else None)
        except QaiHubBundleError as exc:
            return _usage(a, "TS920", str(exc))
        bad = is_finding(verdicts, top)
        code = EXIT_FINDING if bad else EXIT_OK
        issues = list(top) + [i for v in verdicts for i in v.issues]
        doc = _envelope("divergent" if bad else "ok", code, issues,
                        {"mode": "qaihub", "results_dir": a.artifact, "runs": [v.as_dict() for v in verdicts],
                         "next_actions": []})
        _emit(a, doc, qtext(verdicts, top))
        return code
    if a.model == "coreml":  # saved Core ML battery runs: accuracy and planned placement (docs/COREML.md)
        if not a.artifact:
            return _usage(a, "TS021", "usage: truestar conform coreml RESULTS_DIR [--runtimes RUN,...] "
                                      "(collect on a Mac: python -m truestar.conform.coreml collect DIR)")
        from .coreml import CoreMLBundleError, check as ccheck, is_finding as cfinding, to_text as ctext

        try:
            verdicts, top = ccheck(a.artifact, a.runtimes.split(",") if a.runtimes else None)
        except CoreMLBundleError as exc:
            return _usage(a, "TS1168", str(exc))
        bad = cfinding(verdicts, top)
        code = EXIT_FINDING if bad else EXIT_OK
        issues = list(top) + [i for v in verdicts for i in v.issues]
        doc = _envelope("divergent" if bad else "ok", code, issues,
                        {"mode": "coreml", "results_dir": a.artifact, "runs": [v.as_dict() for v in verdicts],
                         "next_actions": []})
        _emit(a, doc, ctext(verdicts, top))
        return code
    if a.model == "tpu":  # saved accelerator outputs of the fixed battery (docs/TPU.md)
        if not a.artifact:
            return _usage(a, "TS021", "usage: truestar conform tpu RESULTS_DIR [--runtimes BACKEND,...]")
        from .tpu import TpuBundleError, check, to_text

        try:
            verdicts, top = check(a.artifact, a.runtimes.split(",") if a.runtimes else None)
        except TpuBundleError as exc:
            return _usage(a, "TS900", str(exc))
        bad = bool(top) or any(v.verdict in ("divergent", "reduced_precision", "failed") for v in verdicts)
        code = EXIT_FINDING if bad else EXIT_OK
        issues = list(top) + [i for v in verdicts for i in v.issues]
        doc = _envelope("divergent" if bad else "ok", code, issues,
                        {"mode": "tpu", "results_dir": a.artifact, "cases": [v.as_dict() for v in verdicts],
                         "next_actions": []})
        _emit(a, doc, to_text(verdicts, top))
        return code
    if a.model == "llamacpp":  # an official llama.cpp build against float64 (docs/CONFORM_LLAMACPP.md)
        return _llamacpp(a)
    if a.model in SERVING_NAMES:
        return _serving(a)
    if a.model == "replay":
        if not a.artifact or not a.extra:
            return _usage(a, "TS021", "usage: truestar conform replay BUNDLE.tsgold ARTIFACT")
        from .bundle import replay, replay_text

        try:
            results, top = replay(a.artifact, a.extra, a.runtimes.split(",") if a.runtimes else None)
        except GoldenError as exc:
            return _usage(a, "TS632", str(exc))
        except MissingDependencyError as exc:
            return _usage(a, exc.code, str(exc), exc.to_dict()["next_actions"])
        except ArtifactError as exc:
            return _usage(a, "TS620", str(exc))
        bad = bool(top) or any(r.verdict in ("regressed", "failed") or not r.conformant for r in results)
        code = EXIT_FINDING if bad else EXIT_OK
        issues = list(top) + [i for r in results for i in r.issues]
        doc = _envelope("divergent" if bad else "ok", code, issues,
                        {"mode": "replay", "bundle": a.artifact, "artifact": a.extra,
                         "runtimes": [r.as_dict() for r in results], "next_actions": []})
        _emit(a, doc, replay_text(results, top))
        return code
    if not a.artifact:
        return _usage(a, "TS021", "usage: truestar conform MODEL.py ARTIFACT")
    try:
        model, example, kwargs = _load(a.model)
    except ModelFileError as exc:
        return _usage(a, "TS020", str(exc))
    if kwargs:
        return _usage(a, "TS021", "conform passes positional inputs only, so get_kwargs() is not supported: "
                                  "return every forward argument from get_inputs() as a tuple and drop "
                                  "get_kwargs()")
    inputs = None
    if a.inputs:
        import torch

        try:
            inputs = torch.load(a.inputs, weights_only=True)
        except Exception as exc:
            return _usage(a, "TS021", f"--inputs {a.inputs}: {type(exc).__name__}: {exc}")
        if not isinstance(inputs, (list, tuple)) or not inputs:
            return _usage(a, "TS021", "--inputs must hold a non empty list of samples")
    golden = None if a.no_golden else (a.golden or a.artifact.rstrip("/\\") + ".tsgold")
    from .core import conform

    try:
        rep = conform(model, example, a.artifact, runtimes=a.runtimes.split(",") if a.runtimes else None,
                      samples=a.samples, inputs=inputs, golden=golden, localize=not a.no_localize,
                      model_file=a.model)
    except MissingDependencyError as exc:  # the artifact's format needs a module not installed here
        return _usage(a, exc.code, str(exc), exc.to_dict()["next_actions"])
    except ArtifactError as exc:
        return _usage(a, "TS620", str(exc))
    except ValueError as exc:
        return _usage(a, "TS021", str(exc))
    st = rep.status
    ran = [r for r in rep.results if r.status != "unavailable"]
    missing = [r.runtime for r in rep.results if r.status == "unavailable"]
    if (a.runtimes and missing) or not ran or any(c == "TS637" for c, _ in rep.issues):
        # a runtime asked for by name that cannot run, or nothing ran: usage error, never a pass
        code = EXIT_USAGE
        st = "error" if not ran or any(c == "TS637" for c, _ in rep.issues) else st
    else:
        code = EXIT_OK if st == "ok" else EXIT_FINDING
    issues = list(rep.issues) + [i for r in rep.results for i in r.issues]
    doc = _envelope(st, code, issues, {"artifact": rep.artifact, "samples": rep.samples,
                                       "modules_mapped": rep.units, "golden": rep.golden,
                                       "runtimes": [r.as_dict() for r in rep.results],
                                       "next_actions": _next_actions(rep, a)})
    _emit(a, doc, rep.to_text())
    return code
