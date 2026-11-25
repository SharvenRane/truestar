"""Command line for `truestar record` and `truestar diff` (wired into truestar.cli).

Exit codes. record: 0 written, 2 usage error (model file, backend, size limit).
diff: 0 identical, noise or precision change (precision is 1 with --strict), 1 divergent or not
comparable, 2 usage error (bad recording, missing model, missing backend).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys

EXIT_OK, EXIT_DIVERGENT, EXIT_USAGE = 0, 1, 2


def add_parsers(sub) -> None:
    r = sub.add_parser("record", help="run a model and record every module call to a .tsrec file")
    r.add_argument("model", help="python file defining get_model() and get_inputs()")
    r.add_argument("-o", "--output", required=True, help="recording path, e.g. run.tsrec")
    r.add_argument("--backend", default="auto",
                   help="cpu, cuda, mps, xpu (default auto: the first accelerator, else cpu)")
    r.add_argument("--outputs-only", action="store_true",
                   help="store only module outputs and input hashes (much smaller)")
    r.add_argument("--ops", default="none",
                   help="trace ATen ops: none (default), all, or comma separated module names")
    r.add_argument("--max-mb", type=float, default=1024.0, help="size limit in MiB (default 1024)")
    r.add_argument("--format", choices=("text", "agent"), default="text")

    d = sub.add_parser("diff", help="compare a recording with another one or with this environment")
    d.add_argument("recorded", help="recording from the old environment")
    d.add_argument("current", nargs="?", help="recording from the new environment")
    d.add_argument("--replay", action="store_true",
                   help="replay the recording here, every module alone with its recorded inputs")
    d.add_argument("--model", help="model file for --replay (default: the file named in the recording)")
    d.add_argument("--backend", help="replay on this backend (default: the recorded device)")
    d.add_argument("--no-exact", action="store_true",
                   help="skip the float64 calibration in replay (faster, dtype bands only)")
    d.add_argument("--repro-dir", default="truestar_repro", help="where to write the op repro")
    d.add_argument("--no-repro", action="store_true", help="do not write an op repro")
    d.add_argument("--strict", action="store_true", help="exit 1 on precision changes too")
    d.add_argument("--json", dest="json_path", help="also write the full report as JSON here")
    d.add_argument("--top", type=int, default=8, help="changed modules to list")
    d.add_argument("--format", choices=("text", "json", "agent"), default="text")


def _device_for(name: str) -> str:
    from ..backends import available_backends, resolve_backends

    if name == "auto":
        accel = [b for b in available_backends() if b.name != "cpu"]
        return accel[0].device if accel else "cpu"
    chosen, _ = resolve_backends([name])
    return chosen[0].device


def _emit_error(a, command: str, code: str, message: str, **extra) -> int:
    if a.format == "agent":
        from ..agent import dumps, error_to_agent

        doc = error_to_agent(code, message, getattr(a, "model", None), **extra)
        doc["command"] = command
        print(dumps(doc))
    else:
        print(f"error [{code}]: {message}", file=sys.stderr)
    return EXIT_USAGE


def run_record(a) -> int:
    from ..backends import BackendUnavailable, available_backends
    from ..cli import ModelFileError, _load
    from .record import RecordTooLarge, record

    agent = a.format == "agent"
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure = None
    ops = a.ops if a.ops in ("none", "all") else [s.strip() for s in a.ops.split(",") if s.strip()]
    with quiet:
        try:
            device = _device_for(a.backend)
            model, inputs, kwargs = _load(a.model)
            rec = record(model, inputs, kwargs, device=device, outputs_only=a.outputs_only,
                         ops=ops, max_bytes=int(a.max_mb * 2**20),
                         model_name=os.path.splitext(os.path.basename(a.model))[0],
                         model_file=a.model)
            rec.save(a.output)
        except ModelFileError as exc:
            failure = ("TS020", str(exc), {})
        except BackendUnavailable as exc:
            failure = ("TS001", str(exc), {"available": [x.name for x in available_backends()]})
        except RecordTooLarge as exc:
            failure = ("TS070", str(exc), {})
        except ValueError as exc:
            failure = ("TS021", str(exc), {})
    if failure is not None:
        return _emit_error(a, "record", failure[0], failure[1], **failure[2])
    digest = hashlib.sha256(rec.raw).hexdigest()
    info = {"path": os.path.abspath(a.output), "bytes": len(rec.raw), "sha256": digest,
            "device": rec.device, "mode": rec.header["mode"], "calls": len(rec.calls),
            "ops_traced": len(rec.ops), "ops_truncated": rec.header["ops_truncated"],
            "torch": rec.env.get("torch"), "cudnn": rec.env.get("cudnn"),
            "gpu": rec.env.get("gpu.name")}
    if agent:
        from ..agent import _envelope, dumps

        doc = _envelope("ok", EXIT_OK)
        doc["command"] = "record"
        doc.update({"recording": info, "env": rec.env, "codes": [], "issues": [],
                    "next_actions": [{
                        "kind": "diff_after_upgrade",
                        "description": "After changing torch, CUDA, cuDNN, Triton or the driver, "
                                       "replay this recording to find what changed.",
                        "command": f'truestar diff "{info["path"]}" --replay --format agent'}]})
        print(dumps(doc))
    else:
        print(f"recorded {info['calls']} module calls"
              + (f" and {info['ops_traced']} ops" if info["ops_traced"] else "")
              + f" on {info['device']} ({info['mode']}) to {info['path']}, "
              f"{info['bytes'] / 1024:.1f} KiB, sha256 {digest[:16]}")
        print(f"environment: torch {info['torch']}, cuDNN {info['cudnn']}, GPU {info['gpu']}")
    return EXIT_OK


def run_diff(a) -> int:
    from ..backends import BackendUnavailable, available_backends
    from ..cli import ModelFileError, _load
    from .diff import ReplayError, diff_recordings, diff_replay
    from .record import load
    from .store import RecordError

    agent = a.format == "agent"
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    if a.current and a.replay:
        return _emit_error(a, "diff", "TS021", "give either a second recording or --replay, "
                                               "not both")
    if not a.current and not a.replay:
        return _emit_error(a, "diff", "TS021", "give a second recording, or --replay to compare "
                                               "with this environment")
    failure = None
    model_file = None
    with quiet:
        try:
            rec_a = load(a.recorded)
            if a.current:
                rep = diff_recordings(rec_a, load(a.current))
            else:
                model_file = a.model or rec_a.header["model"].get("file")
                if not model_file:
                    raise ModelFileError("the recording names no model file; pass --model")
                model, _, _ = _load(model_file)
                device = _device_for(a.backend) if a.backend else None
                rep = diff_replay(rec_a, model, device=device, exact=not a.no_exact,
                                  repro_dir=None if a.no_repro else a.repro_dir)
        except RecordError as exc:
            failure = ("TS065", str(exc), {})
        except ModelFileError as exc:
            failure = ("TS020", str(exc), {})
        except BackendUnavailable as exc:
            failure = ("TS001", str(exc), {"available": [x.name for x in available_backends()]})
        except ReplayError as exc:
            failure = ("TS021", str(exc), {})
    if failure is not None:
        return _emit_error(a, "diff", failure[0], failure[1], **failure[2])
    code = rep.exit_code(a.strict)
    if agent:
        from ..agent import dumps

        doc = rep.to_agent(os.path.abspath(a.recorded),
                           os.path.abspath(model_file) if model_file else None, a.strict)
        print(dumps(doc))
    elif a.format == "json":
        print(rep.to_json())
    else:
        print(rep.to_text(top=a.top))
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(rep.to_json())
    return code
