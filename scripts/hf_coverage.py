"""Hugging Face model coverage from configs (no downloads): check, IR import, compile, export.

    PYTHONPATH=src python scripts/hf_coverage.py [--models llama,gpt2] [--no-cuda]

Prints one row per model and stage with PASS or the first error line. See docs/MODEL_COVERAGE.md.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import traceback

import torch

from truestar import ir
from truestar.testing_hf import MODELS, build


def _first_line(e: BaseException) -> str:
    text = f"{type(e).__name__}: {e}"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # the first named op problem is the useful line in an import error
    for ln in lines:
        if ln.startswith("unsupported op") or ln.startswith("no gradient rule"):
            return ln[:200]
    return lines[0][:200] if lines else type(e).__name__


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().cpu()
    b = b.double().cpu()
    scale = b.abs().max().item() or 1.0
    return (a - b).abs().max().item() / scale


def run_model(name: str, cuda: bool):
    res = {}
    model, args = build(name, batch=2)

    # 1. truestar check
    t0 = time.perf_counter()
    try:
        from truestar.check import check
        rep = check(model, args, backends=["cpu", "cuda"] if cuda else ["cpu"], drill_ops=False)
        res["check"] = "PASS" if rep.status.upper() == "OK" else f"{rep.status} {rep.codes}"
    except Exception as e:  # noqa: BLE001
        res["check"] = "ERROR " + _first_line(e)
    res["check_s"] = time.perf_counter() - t0

    # 2. IR import (float64, symbolic batch)
    prog = None
    m64 = copy.deepcopy(model).double()
    a64 = tuple(a.double() if a.is_floating_point() else a for a in args)
    try:
        B = torch.export.Dim("B", min=1, max=64)
        prog = ir.import_module(m64, a64, dynamic_shapes=({0: B},))
        errs = []
        for batch in (2, 5):
            _, xs = build(name, batch=batch, seed=1)
            xs = tuple(x.double() if x.is_floating_point() else x for x in xs)
            with torch.no_grad(), ir.reference.exact_float32_casts():
                want = m64(*xs)
            got = ir.run(prog, *xs)
            got = got if isinstance(got, (list, tuple)) else [got]
            errs.append(max(_rel(g, w) for g, w in zip(got, want)))
        worst = max(errs)
        res["import"] = f"PASS {worst:.1e}" if worst < 1e-10 else f"FAIL err {worst:.1e}"
    except Exception as e:  # noqa: BLE001
        res["import"] = "ERROR " + _first_line(e)
        prog = None

    # 3. compile (float32) on cuda, else cpu
    if prog is None:
        res["compile"] = "blocked by import"
    else:
        try:
            from truestar.compiler import compile_program
            dev = "cuda" if cuda else "cpu"
            cp = compile_program(prog, dev, "float32")
            _, xs = build(name, batch=3, seed=2)
            outs = cp(list(xs))
            fb = cp.report.fallbacks
            ops = sorted({o for f in fb for o in f["ops"]})
            res["compile"] = f"PASS ({dev}, {len(cp.report.kernels)} kernels, {len(fb)} fallbacks {ops})"
        except Exception as e:  # noqa: BLE001
            res["compile"] = "ERROR " + _first_line(e)

    # 4. export
    try:
        from truestar import export
        export.export(copy.deepcopy(model), args, output=None)
        res["export"] = "PASS"
    except Exception as e:  # noqa: BLE001
        res["export"] = "ERROR " + _first_line(e)
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--no-cuda", action="store_true")
    a = p.parse_args(argv)
    cuda = torch.cuda.is_available() and not a.no_cuda
    for name in a.models.split(","):
        try:
            r = run_model(name, cuda)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            continue
        print(f"== {name}")
        for k in ("check", "import", "compile", "export"):
            print(f"   {k:8s} {r.get(k)}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
