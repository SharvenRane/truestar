"""Sweep of the text, retrieval, speech and vision language families (truestar.testing_hf_text).

Per family: build and run (float32 and float64), import (``ir.import_module`` of the float64 model,
symbolic batch where the wrapper allows it) checked against PyTorch float64 at two batch sizes,
compile (``compile_program(prog, "cpu", "float32")``, the first call verified against the float64
reference, no float64 CPU fallback allowed), and one training step's gradient with respect to every
trainable weight against torch.autograd in float64. docs/MODEL_COVERAGE.md has the results.

    PYTHONPATH=src python scripts/text_coverage.py --json text.json
    PYTHONPATH=src python scripts/text_coverage.py --models t5,bart
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
import warnings

import torch

from truestar import ir
from truestar.ir.core import UnsupportedOpError
from truestar import testing_hf_text as T


def _to64(xs):
    return tuple(x.double() if x.is_floating_point() else x for x in xs)


def _rel(got, want):
    scale = want.double().abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


def _first_line(e: BaseException) -> str:
    s = str(e).strip().splitlines()
    return f"{type(e).__name__}: {s[0][:300] if s else ''}"


def import_program(name):
    model, args = T.build(name, batch=2)
    m64 = copy.deepcopy(model).double()
    a64 = _to64(args)
    ds = T.dynamic_shapes(model.kind, args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prog = ir.import_module(m64, a64, dynamic_shapes=ds)
    return model, m64, prog


def import_error(name, m64, prog):
    """Worst relative error of every output at two batch sizes (one when the batch is fixed)."""
    worst = 0.0
    batches = ((2, 1), (5, 2)) if T.kind_of(name) in T.DYNAMIC_BATCH else ((2, 1),)
    for batch, seed in batches:
        _, xs = T.build(name, batch=batch, seed=seed)
        xs = _to64(xs)
        with torch.no_grad(), ir.reference.exact_float32_casts():
            want = m64(*xs)
        got = ir.run(prog, *xs)
        got = got if isinstance(got, (list, tuple)) else [got]
        assert len(got) == len(want), (len(got), len(want))
        for g, w in zip(got, want):
            assert g.shape == w.shape, (g.shape, w.shape)
            worst = max(worst, _rel(g, w))
    return worst


def train_error(name, m64, prog, seed=3):
    from truestar.ir.autodiff import grad
    batch = 2
    _, xs = T.build(name, batch=batch, seed=seed)
    xs = _to64(xs)
    sd = m64.state_dict(keep_vars=True)
    params = {v.name: sd[v.name] for v in prog.params
              if v.name in sd and isinstance(sd[v.name], torch.nn.Parameter) and sd[v.name].requires_grad}
    names = list(params)
    trainable = {id(p) for p in m64.parameters() if p.requires_grad}
    missing = trainable - {id(p) for p in params.values()}
    if missing:
        byid = {id(p): n for n, p in m64.named_parameters()}
        raise AssertionError("trainable weights not in the program: " + ", ".join(sorted(byid[i] for i in missing)))
    out0 = ir.run(prog, *xs)
    out0 = (out0 if isinstance(out0, (list, tuple)) else [out0])[0]
    cot = torch.randn(out0.shape, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    got = ir.run(grad(prog, wrt=names, of=[0], seed="input"), *xs, cot)[len(prog.outputs):]
    m64.zero_grad(set_to_none=True)
    with ir.reference.exact_float32_casts():
        m64(*xs)[0].backward(cot)
    scale = max(params[n].grad.abs().max().item() for n in names if params[n].grad is not None) or 1.0
    worst = 0.0
    for n, g in zip(names, got):
        want = params[n].grad
        want = torch.zeros_like(g) if want is None else want
        worst = max(worst, (g - want).abs().max().item() / scale)
    return worst


def compile_check(name, prog):
    from truestar.compiler import compile_program
    cp = compile_program(prog, "cpu", "float32")
    batches = ((3, 5), (2, 6)) if T.kind_of(name) in T.DYNAMIC_BATCH else ((2, 5),)
    worst = 0.0
    for batch, seed in batches:
        _, xs = T.build(name, batch=batch, seed=seed)
        outs = cp(list(xs))           # verified against the float64 reference on the first call
        want = ir.run(prog, *_to64(xs))
        want = want if isinstance(want, (list, tuple)) else [want]
        worst = max(worst, _rel(outs[0].cpu(), want[0]))
    fb = sorted({o for f in cp.report.fallbacks for o in f["ops"]})
    return worst, fb


def measure(name):
    res = {"name": name, "model_type": T.MODELS_TEXT[name].model_type}
    t0 = time.perf_counter()
    try:
        model, args = T.build(name, batch=2)
        with torch.no_grad():
            model(*args)
        res["params"] = sum(p.numel() for p in model.parameters())
    except Exception as e:  # noqa: BLE001
        res["build"] = _first_line(e)
        res["trace"] = traceback.format_exc()[-3000:]
        return res
    try:
        model, m64, prog = import_program(name)
        res["dynamic_batch"] = "B" in prog.symbols
        res["import"] = f"PASS {import_error(name, m64, prog):.1e}"
    except UnsupportedOpError as e:
        res["import"] = f"refused [{getattr(e, 'code', '')}] {str(e)[:1500]}"
        res["trace"] = traceback.format_exc()[-3000:]
        return res
    except Exception as e:  # noqa: BLE001
        res["import"] = "error " + _first_line(e)
        res["trace"] = traceback.format_exc()[-3000:]
        return res
    try:
        err, fb = compile_check(name, prog)
        res["compile"] = f"PASS {err:.1e}" + (f" fallbacks {' '.join(fb)}" if fb else "")
    except Exception as e:  # noqa: BLE001
        res["compile"] = "error " + _first_line(e)
        res["compile_trace"] = traceback.format_exc()[-3000:]
    try:
        err = train_error(name, m64, prog)
        res["train"] = ("PASS" if err < 1e-8 else "FAIL") + f" {err:.1e}"
    except Exception as e:  # noqa: BLE001
        res["train"] = "error " + _first_line(e)
        res["train_trace"] = traceback.format_exc()[-3000:]
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--trace", action="store_true", help="print the traceback of a failing stage")
    a = ap.parse_args(argv)
    names = [n for n in T.available() if not a.models or n in a.models.split(",")]
    out = []
    for n in names:
        r = measure(n)
        out.append(r)
        short = {k: v for k, v in r.items() if not k.endswith("trace")}
        print(json.dumps(short), flush=True)
        if a.trace:
            for k in ("trace", "compile_trace", "train_trace"):
                if k in r:
                    print(r[k], flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
