"""`truestar port reference.py gguf:model.gguf | jax:cand.py | keras:cand.py` (docs/PORT_ENGINES.md)."""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Dict, Optional


def _reference(path: str):
    from ..portcheck import PortFileError, _import

    mod = _import(path, "truestar_port_reference")
    if not hasattr(mod, "get_inputs"):
        raise PortFileError(f"{path} must define get_inputs()")
    try:
        model = mod.get_model()
        inputs = mod.get_inputs()
        extra = {"prompt": mod.get_prompt() if hasattr(mod, "get_prompt") else None,
                 "config": mod.get_config() if hasattr(mod, "get_config") else None,
                 "module": mod}
    except Exception as exc:
        raise PortFileError(f"calling get_model() / get_inputs() in {path} failed: "
                            f"{type(exc).__name__}: {exc}") from exc
    return model, inputs, extra


def _config(c: Any):
    from .llama_ref import LlamaConfig

    if c is None or isinstance(c, LlamaConfig):
        return c
    if isinstance(c, dict):
        return LlamaConfig(**c)
    return LlamaConfig.from_hf(c)


def cli(a) -> int:
    from ..agent import dumps, error_to_agent
    from ..portcheck import PortFileError

    agent = a.format == "agent"
    kind, _, target = a.candidate.partition(":")
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure = None
    doc: Optional[Dict[str, Any]] = None
    text = ""
    status_ok = False
    with quiet:
        try:
            model, inputs, extra = _reference(a.reference)
            if kind == "gguf":
                from .engine_port import EnginePortError, engine_port_check, report_to_agent
                from .llamacpp import EngineError

                if not os.path.isfile(target):
                    raise PortFileError(f"{target} not found")
                try:
                    rep = engine_port_check(model, target, inputs, config=_config(extra["config"]),
                                            prompt=extra["prompt"], kv_type=a.kv_type, threads=a.threads,
                                            reference_name=os.path.basename(a.reference),
                                            candidate_name=os.path.basename(target))
                except (EnginePortError, EngineError) as exc:
                    failure = (exc.code, str(exc))
                else:
                    doc = report_to_agent(rep, a.reference, target)
                    text = rep.to_text()
                    status_ok = rep.status == "ok"
            else:
                from .xport import FrameworkError, framework_port_check, report_to_agent as x_agent

                if not os.path.isfile(target):
                    raise PortFileError(f"{target} not found")
                try:
                    backends = [b.strip() for b in a.keras_backends.split(",") if b.strip()] if kind == "keras" else None
                    rep = framework_port_check(model, inputs, kind, target, backends=backends,
                                               reference_name=os.path.basename(a.reference))
                except FrameworkError as exc:
                    failure = (exc.code, str(exc))
                else:
                    doc = x_agent(rep, a.reference, target)
                    text = rep.to_text()
                    status_ok = rep.status == "ok"
        except PortFileError as exc:
            failure = ("TS020", str(exc))
        except Exception as exc:
            failure = ("TS021", f"{type(exc).__name__}: {exc}")
    if failure is not None:
        if agent:
            d = error_to_agent(failure[0], failure[1], a.reference)
            d["command"] = "port"
            print(dumps(d))
        else:
            print(f"error [{failure[0]}]: {failure[1]}", file=sys.stderr)
        return 2
    if agent:
        print(dumps(doc))
    else:
        print(text)
    if getattr(a, "json_path", None):
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(dumps(doc))
    return 0 if status_ok else 1
