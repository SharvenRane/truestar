"""Command line: truestar check model.py, truestar export model.py -o model.tsx, truestar run model.tsx,
truestar record model.py -o run.tsrec, truestar diff run.tsrec --replay, truestar range model.py --target fp16,
truestar compile model.py --precision auto --max-flips 0

model.py defines get_model() returning an nn.Module and get_inputs() returning a tensor or a tuple
of forward arguments (optionally a (args, kwargs) pair via get_kwargs()).

Exit codes: 0 every backend matches, 1 divergence or suspect result, 2 usage error.
For export: 0 artifact written, 1 export refused (nothing written), 2 usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys

EXIT_OK, EXIT_DIVERGENT, EXIT_USAGE = 0, 1, 2


class ModelFileError(Exception):
    """The model file is missing, fails to import, or lacks the required functions (TS020)."""


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of exiting, so `--format agent` can still emit JSON."""

    def error(self, message):
        raise _UsageError(message)


class _UsageError(Exception):
    pass


def _load(path: str):
    if not os.path.isfile(path):
        raise ModelFileError(f"{path} not found")
    # Register the file under its own name, so classes defined in it have findable source
    # (inspect) and can be pickled into repros that import them again.
    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem if stem.isidentifier() and stem not in sys.modules else "truestar_user_model"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise ModelFileError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    for fn in ("get_model", "get_inputs"):
        if not hasattr(mod, fn):
            raise ModelFileError(f"{path} must define {fn}(): get_model() returns an nn.Module, get_inputs() a "
                                  "tensor or a tuple of forward arguments (see examples/)")
    try:
        kwargs = mod.get_kwargs() if hasattr(mod, "get_kwargs") else None
        return mod.get_model(), mod.get_inputs(), kwargs
    except Exception as exc:
        raise ModelFileError(
            f"calling get_model() / get_inputs() in {path} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _load_sized(path: str, name: str):
    """For --sweep: a build(size) function over a model file whose get_model() and/or
    get_inputs() take the size as the keyword ``name``."""
    import inspect

    if not os.path.isfile(path):
        raise ModelFileError(f"{path} not found")
    stem = os.path.splitext(os.path.basename(path))[0]
    modname = stem if stem.isidentifier() and stem not in sys.modules else "truestar_user_model"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        raise ModelFileError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    for fn in ("get_model", "get_inputs"):
        if not hasattr(mod, fn):
            raise ModelFileError(f"{path} must define {fn}(): get_model() returns an nn.Module, get_inputs() a "
                                  "tensor or a tuple of forward arguments (see examples/)")

    def takes(fn) -> bool:
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())

    sized = [f for f in ("get_model", "get_inputs") if takes(getattr(mod, f))]
    if not sized:
        raise ModelFileError(f"--sweep {name}: neither get_model() nor get_inputs() in {path} takes "
                             f"a keyword argument {name!r}; add it to the one that builds the size")

    def build(size: int):
        try:
            model = mod.get_model(**({name: size} if "get_model" in sized else {}))
            inputs = mod.get_inputs(**({name: size} if "get_inputs" in sized else {}))
            kwargs = mod.get_kwargs() if hasattr(mod, "get_kwargs") else None
        except Exception as exc:
            raise ModelFileError(f"building {path} at {name}={size} failed: "
                                 f"{type(exc).__name__}: {exc}") from exc
        return model, inputs, kwargs

    return build


def _build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="truestar", description="The same answer on every chip.")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    c = sub.add_parser("check", help="compare a model across every backend on this machine")
    c.add_argument("model", help="python file defining get_model() and get_inputs()")
    c.add_argument("--backends", help="comma separated, e.g. cpu,cuda (default: all available)")
    c.add_argument("--json", dest="json_path", help="also write the full report as JSON here")
    c.add_argument("--repro-dir", default="truestar_repro", help="where to write repro scripts")
    c.add_argument("--no-repro", action="store_true", help="do not write repro scripts")
    c.add_argument("--top", type=int, default=5, help="layers to list per backend")
    c.add_argument("--format", choices=("text", "agent"), default="text",
                   help="text (default, for people) or agent (versioned JSON on stdout, see "
                        "docs/AGENT_INTERFACE.md)")
    c.add_argument("--backward", action="store_true",
                   help="check every module's backward pass (gradients) instead of the forward "
                        "(docs/GRAD_CHECK.md)")
    c.add_argument("--train", action="store_true",
                   help="check the training mode forward (and backward): batch statistics with the running "
                        "statistic update compared, keyed dropout masks identical on every backend")
    c.add_argument("--steps", type=int, default=0,
                   help="with --backward: also run this many SGD steps and flag gradient norm drift")
    c.add_argument("--lr", type=float, default=1e-2, help="with --steps: SGD learning rate")
    c.add_argument("--sweep", default=None, metavar="NAME:LO..HI",
                   help="find the first size where a backend diverges: get_model() and/or "
                        "get_inputs() take NAME as a keyword (docs/DEVICES.md)")
    pr = sub.add_parser("probe", help="special value probe: NaN, inf, subnormals, minus zero and "
                                      "casts on every backend against IEEE (docs/DEVICES.md)")
    pr.add_argument("--specials", action="store_true", required=True,
                    help="run the special value battery (the only probe today)")
    pr.add_argument("--backends", help="comma separated (default: every real backend here and "
                                       "every sim:* profile)")
    pr.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    pr.add_argument("--json", dest="json_path", help="also write the JSON here")
    pr.add_argument("--format", choices=("text", "agent"), default="text")
    from .fix.cli import add_parser as _add_fix

    _add_fix(sub)
    from .rangecheck.cli import add_parser as _add_range

    _add_range(sub)
    from .precplan.cli import add_parser as _add_compile  # docs/PRECISION.md

    _add_compile(sub)

    from .intref.cli import add_parser as _add_intref  # docs/INTREF.md

    _add_intref(sub)
    from .parallel.cli import add_parser as _add_parallel  # docs/PARALLEL.md

    _add_parallel(sub)
    from .spikes.cli import add_parser as _add_replay  # docs/SPIKES.md

    _add_replay(sub)

    from .conform.cli import add_parser as _add_conform  # docs/CONFORM.md

    _add_conform(sub)
    from .fingerprint.cli import add_parser as _add_fingerprint  # docs/CROSS_CHIP.md

    _add_fingerprint(sub)
    from .prefixcheck.cli import add_parser as _add_prefixcheck  # docs/PREFIX_CHECK.md

    _add_prefixcheck(sub)
    from .serve.cli import add_parser as _add_serve  # docs/SERVE.md

    _add_serve(sub)

    from .orderprobe import add_parser as _add_orderprobe  # docs/ORDERPROBE.md

    _add_orderprobe(sub)
    b = sub.add_parser("backends", help="list the backends on this machine as JSON")
    b.add_argument("--sims", action="store_true",
                   help="also list every simulated backend (sim:*) with its rules and sources")
    b.set_defaults(format="agent")
    cc = sub.add_parser("cache", help="the persistent compile cache: info (JSON) or clear")
    cc.add_argument("action", choices=("info", "clear", "trim"))
    cc.set_defaults(format="agent")
    k = sub.add_parser("codes", help="list every truestar code as JSON")
    k.set_defaults(format="agent")
    m = sub.add_parser("mcp", help="run the MCP server on stdin/stdout")
    m.add_argument("--impl", choices=("auto", "sdk", "builtin"), default="auto",
                   help="auto uses the official mcp package when installed, else the builtin one")
    m.set_defaults(format="text")
    e = sub.add_parser("export", help="export a model to a verified .tsx artifact, or with --to to a deployable "
                                      "file verified in its own runtime")
    e.add_argument("model", help="python file defining get_model() and get_inputs()")
    e.add_argument("-o", "--output", required=True, help="artifact path, e.g. model.tsx or model.onnx")
    e.add_argument("--to", choices=("tsx", "onnx", "litert", "tflite", "coreml", "executorch", "pte"), default="tsx",
                   help="file format (default tsx); onnx, litert, coreml and executorch run the written file in "
                        "its runtime and refuse it when it fails (docs/EXPORT.md, Deploy targets)")
    e.add_argument("--runtime", default=None, help="runtime that verifies the file (default: ort:cpu, "
                                                   "tflite:interp, pte:cpu)")
    e.add_argument("--force", action="store_true", help="write the file even when its runtime check fails (TS1415)")
    e.add_argument("--samples", type=int, default=4, help="inputs the file is checked on, the example first (default 4)")
    e.add_argument("--json", dest="json_path", default=None, help="--to: also write the report as JSON here")
    e.add_argument("--static-batch", action="store_true", help="freeze the batch size (default: symbolic B)")
    e.add_argument("--max-batch", type=int, default=None, help="upper bound of the batch symbol (default: none)")
    e.add_argument("--dynamic", action="append", metavar="ARG:AXIS[=NAME]",
                   help="make more dims symbolic, for example --dynamic x:1=T (repeatable)")
    e.add_argument("--second-batch", type=int, default=None, help="batch size of the second check (default 7)")
    e.add_argument("--tolerance", type=float, default=None, help="max relative error vs PyTorch float64")
    e.add_argument("--decompose", action="store_true", help="import the core ATen graph")
    pc = sub.add_parser("port", help="check that a second implementation computes the same model")
    pc.add_argument("reference", help="python file defining get_model() and get_inputs()")
    pc.add_argument("candidate", help="python file defining get_model() or get_model(reference); "
                                      "gguf:model.gguf (llama.cpp), jax:file.py or keras:file.py "
                                      "(docs/PORT_ENGINES.md)")
    pc.add_argument("--kv-type", choices=("f32", "f16"), default="f32", help="gguf: KV cache type (default f32)")
    pc.add_argument("--threads", type=int, default=4, help="gguf: engine threads (default 4)")
    pc.add_argument("--keras-backends", default="jax,torch", help="keras: backends to run (default jax,torch)")
    pc.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    pc.add_argument("--top", type=int, default=5, help="modules to list")
    pc.add_argument("--format", choices=("text", "agent"), default="text")
    kc = sub.add_parser("kernel", help="check a kernel against a reference over many inputs "
                                       "with derived tolerances")
    kc.add_argument("reference", help="python file defining reference(*inputs) (or a KernelBench "
                                      "Model) and get_inputs()")
    kc.add_argument("candidate", help="python file defining kernel(*inputs), candidate(*inputs) "
                                      "or a KernelBench ModelNew")
    kc.add_argument("--device", default=None, help="cpu or cuda (default: cuda when present)")
    kc.add_argument("--dtypes", default=None, help="also check these dtypes, e.g. float16,bfloat16")
    kc.add_argument("--no-shapes", action="store_true", help="skip the shape variants")
    kc.add_argument("--repeats", type=int, default=2, help="runs per input for the determinism check")
    kc.add_argument("--no-lean", action="store_true", help="skip the test of whether errors "
                    "within the tolerance lean one way (TS1300)")
    kc.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    kc.add_argument("--format", choices=("text", "agent"), default="text")
    v = sub.add_parser("invariance", help="check that a row's result does not depend on the batch")
    v.add_argument("model", help="python file defining get_model() and get_inputs()")
    v.add_argument("--backends", help="comma separated, e.g. cpu,cuda (default: all available)")
    v.add_argument("--sizes", default="2,7,64", help="batch sizes to place the row in")
    v.add_argument("--rows", type=int, default=2, help="example rows to check")
    v.add_argument("--decode", type=int, default=None, help="greedy tokens for token models (default 4)")
    v.add_argument("--seq-pad", type=int, default=None, help="sequence right padding for token models (default 5)")
    v.add_argument("--no-flags", action="store_true", help="do not rerun with torch flag sets")
    v.add_argument("--train", action="store_true",
                   help="training mode: keyed dropout masks drawn per row; training batch norm is batch dependent")
    v.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    v.add_argument("--format", choices=("text", "agent"), default="text")
    r = sub.add_parser("run", help="run a .tsx artifact (no model code needed)")
    r.add_argument("artifact")
    r.add_argument("inputs", nargs="?", help=".pt file holding a tensor or a list of tensors")
    r.add_argument("-o", "--output", help="save the outputs here")
    r.add_argument("--info", action="store_true", help="print the signature and verification evidence")
    r.add_argument("--executor", default=None,
                   help="reference (default), cpu, cuda, cpu:float16 or cuda:float16 (see docs/COMPILER.md)")
    from .recdiff.cli import add_parsers as _recdiff_parsers  # record, diff (docs/RECORD_DIFF.md)
    _add_rlpair_parser(sub)  # rlpair (docs/RL_PAIR.md)

    _recdiff_parsers(sub)
    _add_profile_parser(sub)  # profile (docs/PROFILE.md)
    return p


def _add_profile_parser(sub) -> None:
    """`truestar profile`; kept here so building the parser does not import torch."""
    r = sub.add_parser("profile", help="GPU time per compiled step, mapped to IR ops and source lines")
    r.add_argument("model", help="python file defining get_model() and get_inputs()")
    r.add_argument("--dtype", default="float32", help="float32 (default), float16 or bfloat16")
    r.add_argument("--iters", type=int, default=30, help="timed calls (default 30)")
    r.add_argument("--warmup", type=int, default=3, help="untimed calls after the first (default 3)")
    r.add_argument("--tf32", action="store_true", help="allow TF32 for float32 dot products and convolutions")
    r.add_argument("--sentinels", default=None, help="profile with sentinels (compiled in checks), e.g. nan,div or all")
    r.add_argument("--crosscheck", action="store_true",
                   help="also run the calls under torch.profiler and compare counts and totals")
    r.add_argument("--top", type=int, default=15, help="steps and lines to list in the text table")
    r.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    r.add_argument("--format", choices=("text", "agent"), default="text")


def _add_rlpair_parser(sub) -> None:
    """`truestar rlpair` (P15); kept here so building the parser does not import torch."""
    r = sub.add_parser("rlpair", help="compare RL trainer and sampler logprobs and localize the gap")
    r.add_argument("pair", help="python file defining get_model(), get_tokens(), get_prompt_len() "
                                "and sample_logits(model, tokens, prompt_len)")
    r.add_argument("--device", default=None, help="move the models here first, e.g. cpu or cuda")
    r.add_argument("--threshold", type=float, default=1e-3,
                   help="logprob difference counted as over threshold (default 1e-3)")
    r.add_argument("--clip", type=float, default=0.2, help="importance ratio clip range (default 0.2)")
    r.add_argument("--batch-sizes", default="2,7",
                   help="sampler batch composition check sizes, empty to skip (default 2,7)")
    r.add_argument("--step-module", default=None,
                   help="module whose calls are the decode steps (default: the model)")
    r.add_argument("--no-ops", action="store_true", help="skip op drilldown")
    r.add_argument("--strict", action="store_true",
                   help="exit 0 only for bitwise identical logprobs and a batch invariant sampler")
    r.add_argument("--kernels", choices=("default", "rl"), default="default",
                   help="rl: run trainer and sampler on the batch and position invariant RL kernel "
                        "set (truestar.rlkernels, CUDA) and name any op whose kernel differs")
    r.add_argument("--tp", type=int, default=None,
                   help="also compare the trainer at TP 1 with the pair file's tp_logits(model, "
                        "tokens, tp) bitwise (a one device simulation)")
    r.add_argument("--top", type=int, default=5, help="worst positions and modules to list")
    r.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    r.add_argument("--format", choices=("text", "agent"), default="text")


def _wants_agent(argv) -> bool:
    argv = list(argv)
    for i, a in enumerate(argv):
        if a == "--format=agent" or (a == "--format" and i + 1 < len(argv) and argv[i + 1] == "agent"):
            return True
    return False


def _model_arg(argv):
    argv = list(argv)
    if argv and argv[0] == "check":
        skip = False
        for a in argv[1:]:
            if skip:
                skip = False
            elif a in _VALUE_FLAGS:
                skip = True
            elif not a.startswith("-"):
                return a
    return None


_VALUE_FLAGS = ("--backends", "--json", "--repro-dir", "--top", "--format", "--steps", "--lr",
                "--sweep")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        a = _build_parser().parse_args(argv)
    except _UsageError as exc:
        if _wants_agent(argv):
            from .agent import dumps, error_to_agent

            print(dumps(error_to_agent("TS021", str(exc), _model_arg(argv))))
        else:
            _build_parser().print_usage(sys.stderr)
            print(f"truestar: error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if a.cmd == "backends":
        from .backends import available_backends

        import torch

        doc = {"backends": [x.as_dict() for x in available_backends()],
               "torch_version": torch.__version__}
        if a.sims:
            from .devsim import PROFILES

            doc["simulated"] = [p.as_dict() for p in PROFILES.values()]
        print(json.dumps(doc, indent=2))
        return EXIT_OK
    if a.cmd == "probe":
        return _probe(a)
    if a.cmd == "codes":
        from .agent import code_table

        print(json.dumps({"codes": code_table()}, indent=2))
        return EXIT_OK
    if a.cmd == "export":
        return _export(a)
    if a.cmd == "port":
        from .portcheck import cli as port_cli

        return port_cli(a)
    if a.cmd == "kernel":
        from .kernelcheck import cli as kernel_cli

        return kernel_cli(a)
    if a.cmd == "run":
        from .export.runner import main as run_main
        rargs = [a.artifact] + ([a.inputs] if a.inputs else []) + (["-o", a.output] if a.output else [])
        rargs += ["--info"] if a.info else []
        return run_main(rargs + (["--executor", a.executor] if a.executor else []))
    if a.cmd == "invariance":
        from .invariance import run_cli

        return run_cli(a)
    if a.cmd == "rlpair":
        from .rlpair import run_cli as rlpair_cli

        return rlpair_cli(a)
    if a.cmd in ("record", "diff"):
        from .recdiff.cli import run_diff, run_record

        return run_record(a) if a.cmd == "record" else run_diff(a)
    if a.cmd == "range":
        from .rangecheck.cli import run as range_run

        return range_run(a)
    if a.cmd == "compile":
        from .precplan.cli import run as compile_run

        return compile_run(a)

    if a.cmd == "replay":
        from .spikes.cli import run as replay_run

        return replay_run(a)
    if a.cmd == "parallel":
        from .parallel.cli import run as parallel_run

        return parallel_run(a)


    if a.cmd == "conform":
        from .conform.cli import run as conform_run

        return conform_run(a)
    if a.cmd == "orderprobe":
        from .orderprobe import run as orderprobe_run

        return orderprobe_run(a)
    if a.cmd == "fingerprint":
        from .fingerprint.cli import run as fingerprint_run

        return fingerprint_run(a)
    if a.cmd == "serve":
        from .serve.cli import run as serve_run

        return serve_run(a)
    if a.cmd == "prefixcheck":
        from .prefixcheck.cli import run as prefixcheck_run

        return prefixcheck_run(a)
    if a.cmd == "intref":
        from .intref.cli import run as intref_run

        return intref_run(a)
    if a.cmd == "fix":
        from .fix.cli import run as fix_run

        return fix_run(a)
    if a.cmd == "profile":
        from .compiler.profile import run_cli as profile_cli

        return profile_cli(a)
    if a.cmd == "cache":
        from . import compile_cache as CC

        doc = CC.info()
        if a.action == "clear":
            doc = {"dir": doc["dir"], "deleted": CC.clear(), "after": CC.info()}
        elif a.action == "trim":
            doc = {"dir": doc["dir"], "deleted": CC.trim(), "after": CC.info()}
        print(json.dumps(doc, indent=2))
        return EXIT_OK
    if a.cmd == "mcp":
        from .mcp_server import serve

        return serve(a.impl)
    return _check(a)


def _check(a) -> int:
    agent = a.format == "agent"
    # In agent mode stdout carries exactly one JSON document: anything the user's model prints
    # goes to stderr instead.
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()

    def fail(code: str, message: str, **extra) -> int:
        if agent:
            from .agent import dumps, error_to_agent

            print(dumps(error_to_agent(code, message, a.model, **extra)))
        else:
            print(f"error [{code}]: {message}", file=sys.stderr)
        return EXIT_USAGE

    from .backends import BackendUnavailable, available_backends
    from .check import ReferenceFailed, check

    backends = [s for s in a.backends.split(",") if s] if a.backends else None
    if a.sweep:
        return _sweep(a, backends, fail, quiet)
    failure = None
    with quiet:
        try:
            model, inputs, kwargs = _load(a.model)
            run, extra = check, {}
            if a.backward:
                from .gradcheck import grad_check as run

                extra = {"steps": a.steps, "lr": a.lr}
            report = run(model, inputs, kwargs=kwargs, backends=backends,
                         repro_dir=None if a.no_repro else a.repro_dir,
                         model_name=os.path.splitext(os.path.basename(a.model))[0], train=a.train, **extra)
        except ModelFileError as exc:
            failure = ("TS020", str(exc), {})
        except BackendUnavailable as exc:
            failure = ("TS001", str(exc), {"available": [x.name for x in available_backends()]})
        except ReferenceFailed as exc:
            failure = ("TS005", str(exc), {})
        except RuntimeError as exc:
            failure = ("TS021", str(exc), {})
    if failure is not None:  # printed after the redirect ends, so it lands on stdout
        return fail(failure[0], failure[1], **failure[2])
    if agent:
        from .agent import dumps, report_to_agent

        doc = report.to_agent(a.model) if a.backward else report_to_agent(report, a.model)
        print(dumps(doc))
    else:
        print(report.to_text(top=a.top))
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
    return EXIT_OK if report.status == "ok" else EXIT_DIVERGENT


def _parse_dynamic(items) -> dict:
    """--dynamic x:1=T,2 -> {"x": {1: "T", 2: "x_2"}}; repeatable."""
    out: dict = {}
    for item in items:
        name, sep, axes = item.partition(":")
        if not sep or not name or not axes:
            raise ValueError(f"--dynamic {item!r}: write ARG:AXIS[=NAME][,AXIS[=NAME]...], for example x:1=T")
        for part in axes.split(","):
            ax, _, sym = part.partition("=")
            try:
                ax = int(ax)
            except ValueError:
                raise ValueError(f"--dynamic {item!r}: axis {ax!r} is not an integer") from None
            out.setdefault(name, {})[ax] = sym or f"{name}_{ax}"
    return out

def _sweep(a, backends, fail, quiet) -> int:
    from .backends import BackendUnavailable, available_backends
    from .check import ReferenceFailed
    from .devsim.sweep import parse_sweep, sweep

    if a.backward:
        return fail("TS021", "--sweep checks the forward pass; it cannot be combined with --backward")
    failure = None
    with quiet:
        try:
            name, lo, hi = parse_sweep(a.sweep)
            build = _load_sized(a.model, name)
            if backends is None:
                backends = [b.name for b in available_backends()]
            rep = sweep(build, name, lo, hi, backends,
                        model_name=os.path.splitext(os.path.basename(a.model))[0],
                        repro_dir=None if a.no_repro else a.repro_dir)
        except ValueError as exc:
            failure = ("TS021", str(exc), {})
        except ModelFileError as exc:
            failure = ("TS020", str(exc), {})
        except BackendUnavailable as exc:
            failure = ("TS001", str(exc), {"available": [x.name for x in available_backends()]})
        except ReferenceFailed as exc:
            failure = ("TS005", str(exc), {})
    if failure is not None:
        return fail(failure[0], failure[1], **failure[2])
    doc = rep.as_dict()
    if a.format == "agent":
        from .agent import _envelope, dumps
        from .codes import Issue

        env = _envelope(rep.status, EXIT_OK if rep.status == "ok" else EXIT_DIVERGENT)
        env["command"] = "check --sweep"
        issues = []
        for r in rep.results:
            if r.verdict in ("cliff", "fails_at_lo"):
                where = f"{name}={r.first_bad}" + (f" (OK at {name}={r.last_ok})" if r.last_ok else "")
                blame = f", module {r.module}" if r.module else ""
                blame += f", op {r.op}" if r.op else ""
                issues.append(Issue("TS521" if r.verdict == "cliff" else "TS522",
                                    f"{r.backend} diverges from {where}{blame}", r.backend).as_dict())
        env.update({"model": {"name": rep.model, "file": os.path.abspath(a.model)},
                    "codes": sorted({i["code"] for i in issues}), "issues": issues, "sweep": doc,
                    "next_actions": [{"kind": "run_repro", "backend": r.backend,
                                      "description": f"Reproduce at {name}={r.first_bad}.",
                                      "command": f"python {r.repro_path}"}
                                     for r in rep.results if r.repro_path]})
        print(dumps(env))
    else:
        print(rep.to_text())
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
    return EXIT_OK if rep.status == "ok" else EXIT_DIVERGENT


def _probe(a) -> int:
    import torch

    from .backends import BackendUnavailable, available_backends
    from .devsim.specials import probe_specials

    backends = [s for s in a.backends.split(",") if s] if a.backends else None
    try:
        rep = probe_specials(backends, getattr(torch, a.dtype))
    except BackendUnavailable as exc:
        if a.format == "agent":
            from .agent import dumps, error_to_agent

            print(dumps(error_to_agent("TS001", str(exc), None,
                                       available=[x.name for x in available_backends()])))
        else:
            print(f"error [TS001]: {exc}", file=sys.stderr)
        return EXIT_USAGE
    doc = rep.as_dict()
    if a.format == "agent":
        from .agent import _envelope, dumps

        env = _envelope(rep.status, EXIT_OK if rep.status == "ok" else EXIT_DIVERGENT)
        env["command"] = "probe --specials"
        env.update({"codes": rep.codes(), "probe": doc, "next_actions": []})
        print(dumps(env))
    else:
        print(rep.to_text())
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
    return EXIT_OK if rep.status == "ok" else EXIT_DIVERGENT


def _export(a) -> int:
    from .export import DEFAULT_TOLERANCE, ExportError, export

    try:
        model, inputs, kwargs = _load(a.model)
    except ModelFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        dyn = _parse_dynamic(a.dynamic or [])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if a.to != "tsx":
        return _export_to(a, model, inputs, kwargs)
    try:
        res = export(model, inputs, a.output, kwargs=kwargs, dynamic_batch=not a.static_batch,
                     max_batch=a.max_batch, dynamic_dims=dyn, second_batch=a.second_batch,
                     tolerance=DEFAULT_TOLERANCE if a.tolerance is None else a.tolerance,
                     decompose=a.decompose, name=os.path.splitext(os.path.basename(a.model))[0])
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(res.to_text())
    return 0


def _export_to(a, model, inputs, kwargs) -> int:
    from .deps import MissingDependencyError
    from .export import DEFAULT_TOLERANCE
    from .export.deploy import DeployError, export_to

    if kwargs:
        print("error: TS1419: deploy targets take positional tensor inputs; get_kwargs() exports to .tsx only",
              file=sys.stderr)
        return EXIT_USAGE
    if a.dynamic:
        print("error: TS1419: --dynamic is for .tsx; deploy targets keep only the batch symbolic (onnx)",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        res = export_to(model, inputs, a.output, to=a.to, runtime=a.runtime, force=a.force,
                        static_batch=True if a.static_batch else None, samples=a.samples,
                        second_batch=a.second_batch or 7,
                        tolerance=DEFAULT_TOLERANCE if a.tolerance is None else a.tolerance,
                        decompose=a.decompose, name=os.path.splitext(os.path.basename(a.model))[0])
    except MissingDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except DeployError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if a.json_path and exc.report is not None:
            with open(a.json_path, "w", encoding="utf-8") as f:
                json.dump(exc.report.as_dict(), f, indent=2, default=str)
        return EXIT_USAGE if exc.code == "TS1419" else 1
    print(res.to_text())
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(res.as_dict(), f, indent=2, default=str)
    return 1 if res.status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
