"""Run a ``.tsx`` artifact with nothing but the file.

No torch.export, no model class, no model source: the artifact holds the whole program and its
weights. Execution goes through an executor chosen by name. Today the only executor is
``reference`` (the float64 IR reference interpreter, outputs cast to the declared dtypes); the
compiled executor plugs in through ``register_executor`` without changing callers.

Command line::

    python -m truestar.export.runner model.tsx inputs.pt [-o outputs.pt] [--info]

``inputs.pt`` holds a tensor or a list or tuple of tensors (loaded with ``weights_only=True``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from ..ir import Program, run as _reference_run
from ..ir.core import IRError
from ..ir.serialize import Artifact, load as _load

Executor = Callable[[Program, Sequence[torch.Tensor]], List[torch.Tensor]]
EXECUTORS: Dict[str, Executor] = {}


def register_executor(name: str, fn: Executor) -> None:
    """Add an executor: ``fn(program, inputs) -> outputs`` (outputs in any float precision)."""
    EXECUTORS[name] = fn


def _reference(program: Program, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return _reference_run(program, *inputs)


register_executor("reference", _reference)


COMPILED_EXECUTORS = ("cuda", "cpu", "cuda:float16", "cpu:float16")


def _ensure_executor(name: str) -> None:
    """The compiled executors (see docs/COMPILER.md) register themselves when first asked for, so
    running an artifact through the reference interpreter never imports the compiler."""
    if name not in EXECUTORS and name in COMPILED_EXECUTORS:
        from .. import compiler  # noqa: F401  (registers "cuda", "cpu" and their float16 forms)


class Runner:
    """A loaded artifact ready to run. ``runner(*inputs)`` returns a list of output tensors."""

    def __init__(self, artifact: Artifact, executor: str = "reference", cast_outputs: bool = True):
        _ensure_executor(executor)
        if executor not in EXECUTORS:
            raise ValueError(f"unknown executor {executor!r}; available: {sorted(EXECUTORS)}")
        self.artifact = artifact
        self.program = artifact.program
        self.executor = executor
        self.cast_outputs = cast_outputs

    @classmethod
    def load(cls, path: str, **kw: Any) -> "Runner":
        return cls(_load(path), **kw)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.artifact.metadata

    @property
    def evidence(self) -> Optional[Dict[str, Any]]:
        return self.artifact.metadata.get("truestar_export")

    def signature(self) -> str:
        p = self.program
        ins = ", ".join(f"%{v.name}: {v.type}" for v in p.inputs)
        outs = ", ".join(str(v.type) for v in p.outputs)
        syms = "".join(f", {s} in [{lo}, {'inf' if hi is None else hi}]" for s, (lo, hi) in p.symbols.items())
        return f"{p.name}({ins}) -> ({outs}){syms}"

    def __call__(self, *inputs: Any, **kwargs: Any) -> Any:
        """Run the artifact. Called with plain tensors in the program's input order (the form every
        artifact accepts), it returns the flat list of output tensors. Called the way the model was
        exported, with keyword arguments or nested structures, it is ``call`` and returns the
        model's own output structure."""
        if kwargs or any(not isinstance(t, torch.Tensor) for t in inputs):
            return self.call(*inputs, **kwargs)
        return self._run_flat(inputs)

    def _run_flat(self, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        p = self.program
        if len(inputs) != len(p.inputs):
            raise IRError(f"{p.name} takes {len(p.inputs)} inputs ({self.signature()}), got {len(inputs)}")
        for v, t in zip(p.inputs, inputs):
            if not isinstance(t, torch.Tensor):
                raise IRError(f"input %{v.name} must be a tensor, got {type(t).__name__}")
            if t.dtype != v.type.dtype:
                raise IRError(f"input %{v.name} is {t.dtype} but the artifact declares {v.type}; cast it with "
                              f".to({v.type.dtype})")
        outs = EXECUTORS[self.executor](p, inputs)
        if self.cast_outputs:
            outs = [o.to(v.type.dtype) for o, v in zip(outs, p.outputs)]
        return outs

    @property
    def io(self) -> Optional[Dict[str, Any]]:
        """The stored input and output structures (artifacts exported since 2026-11-24)."""
        return (self.evidence or {}).get("io")

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Run with the arguments structured as at export (nested dicts, lists, tuples, NamedTuples,
        dataclasses, keyword arguments); returns the model's output structure. A structure that
        differs names the exact leaf, for example ``params['w']: missing key 'w'``."""
        from .. import pytree

        io = self.io
        if io is None:
            if kwargs:
                raise IRError(f"{self.program.name} was exported without a stored input structure; pass its "
                              f"{len(self.program.inputs)} input tensors positionally")
            return self._run_flat(args)
        args_spec, kw_spec = io["inputs"]["children"]
        names = io.get("arg_names") or []
        want_pos = len(args_spec["children"])
        if len(args) != want_pos:
            raise pytree.StructureError("", f"{self.program.name} was exported with {want_pos} positional "
                                            f"argument(s) ({', '.join(names) or '...'}), got {len(args)}")
        inputs = self.program.inputs

        def check(path, t, i):
            v = inputs[i]
            if t.dtype != v.type.dtype:
                raise pytree.StructureError(path, f"is {t.dtype} but the artifact declares {v.type}; cast it "
                                                  f"with .to({v.type.dtype})")
            if t.dim() != v.type.rank or any(isinstance(d, int) and d != s for d, s in zip(v.type.shape, t.shape)):
                raise pytree.StructureError(path, f"has shape {list(t.shape)} but the artifact declares {v.type}")

        is_t = lambda x: isinstance(x, torch.Tensor)  # noqa: E731
        count = [0]
        leaves: List[torch.Tensor] = []
        for i, (spec, a) in enumerate(zip(args_spec["children"], args)):
            leaves += pytree.match(spec, a, is_t, names[i] if i < len(names) else f"args[{i}]", check, count)
        want_kw, got_kw = list(kw_spec.get("keys", [])), list(kwargs)
        missing = [k for k in want_kw if k not in kwargs]
        extra = [k for k in got_kw if k not in want_kw]
        if missing or extra:
            parts = ([f"missing keyword argument{'s' if len(missing) > 1 else ''} {', '.join(missing)}"] if missing else [])                 + ([f"unexpected keyword argument{'s' if len(extra) > 1 else ''} {', '.join(extra)}"] if extra else [])
            raise pytree.StructureError(missing[0] if missing else extra[0],
                                        "; ".join(parts) + f" (exported with keywords: {', '.join(want_kw) or 'none'})")
        for k, spec in zip(want_kw, kw_spec.get("children", [])):
            leaves += pytree.match(spec, kwargs[k], is_t, k, check, count)
        return pytree.unflatten(io["outputs"], self._run_flat(leaves))


def load(path: str, **kw: Any) -> Runner:
    """Load an artifact for running. Raises ArtifactError if the file is damaged or of an unknown version."""
    return Runner.load(path, **kw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="truestar run", description="run a .tsx artifact")
    p.add_argument("artifact")
    p.add_argument("inputs", nargs="?", help=".pt file with a tensor or a list of tensors")
    p.add_argument("-o", "--output", help="write the outputs (a list of tensors) here with torch.save")
    p.add_argument("--info", action="store_true", help="print the signature and the verification evidence")
    p.add_argument("--executor", default="reference", choices=sorted(set(EXECUTORS) | set(COMPILED_EXECUTORS)))
    a = p.parse_args(argv)
    try:
        r = load(a.artifact, executor=a.executor)
    except (IRError, OSError, ImportError) as e:  # ImportError: MissingDependencyError (TS022)
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.info or not a.inputs:
        print(r.signature())
        print(f"program hash {r.artifact.program_hash}, format version {r.artifact.format_version}")
        if r.evidence:
            print(json.dumps(r.evidence, indent=2, sort_keys=True))
        if not a.inputs:
            return 0
    x = torch.load(a.inputs, weights_only=True)
    from .. import pytree
    try:
        if isinstance(x, dict):            # keyword arguments, as exported
            outs = r.call(**x)
        else:
            x = [x] if isinstance(x, torch.Tensor) else list(x)
            outs = r(*x)
    except (IRError, ImportError, pytree.StructureError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    leaves, spec = pytree.flatten(outs, lambda t: isinstance(t, torch.Tensor))
    paths = pytree.leaf_paths(spec) if not isinstance(outs, list) else [f"output {i}" for i in range(len(leaves))]
    for pth, o in zip(paths, leaves):
        print(f"{pth or 'output'}: {str(o.dtype).replace('torch.', '')}{list(o.shape)}")
    if a.output:
        torch.save(outs, a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
