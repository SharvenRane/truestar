"""CUDA rows of the 2026 coverage refresh: the current Hugging Face families compiled for CUDA.

    PYTHONPATH=src python scripts/cov26_cuda.py [--models a,b] [--json out.jsonl]

For each family of ``testing_hf.MODELS_2026`` this transformers knows, minus ``KNOWN_GAPS_2026``
(26 on transformers 5.20 to 5.24), with the bar of
``tests/test_hf_models.py::test_compiled_on_cuda_matches_the_reference_on_first_call``:

* import: ``model.double()`` with a symbolic batch (as the test does);
* compile: ``compile_program(prog, "cuda", "float32")``; the first call at each signature is
  verified against the float64 reference (it raises otherwise);
* at batch 3 (seed 5) and batch 2 (seed 6), the first output's error relative to ``ir.run`` in
  float64 must be under 1e-3, and nothing may run on a fallback.

Also per family: eager PyTorch float32 on CUDA, its own error against the same float64 reference
(context for Truestar's error), and latency at batch 3: Truestar's compiled call and eager PyTorch
under ``torch.no_grad()``, and Truestar replayed from a CUDA graph (``cp.graphed()``, checked bitwise
against the ordinary call), each the median of 30 calls after 5 warm up calls, CUDA synchronized.
One JSON line per family.
"""

import argparse
import copy
import json
import statistics
import sys
import time
import traceback

import torch


def rel(got, want):
    scale = want.abs().max().item() or 1.0
    return (got.double().cpu() - want.double().cpu()).abs().max().item() / scale


def first(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    if hasattr(x, "logits"):
        return x.logits
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if isinstance(x, dict):
        return next(iter(x.values()))
    return x


def latency(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def run(name):
    from truestar import ir
    from truestar.compiler import compile_program
    from truestar.testing_hf import build

    row = {"family": name}
    model, args = build(name, batch=2)
    m64 = copy.deepcopy(model).double()
    a64 = tuple(a.double() if a.is_floating_point() else a for a in args)
    t = time.perf_counter()
    prog = ir.import_module(m64, a64, dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
    row["import_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    cp = compile_program(prog, "cuda", "float32")
    errs, eager_errs = [], []
    eager = copy.deepcopy(model).float().cuda().eval()
    for batch, seed in ((3, 5), (2, 6)):
        _, xs = build(name, batch=batch, seed=seed)
        outs = cp(list(xs))                       # verified on the first call at this signature
        want = first(ir.run(prog, *[x.double() if x.is_floating_point() else x for x in xs]))
        errs.append(rel(outs[0], want))
        with torch.no_grad():
            e = first(eager(*[x.cuda() for x in xs]))
        eager_errs.append(rel(e, want))
    row["compile_and_first_calls_s"] = round(time.perf_counter() - t, 1)
    row["kernels"] = len(cp.report.kernels)
    row["fallbacks"] = sorted({o for f in cp.report.fallbacks for o in f["ops"]})
    row["max_error_vs_float64"] = max(errs)
    row["eager_max_error_vs_float64"] = max(eager_errs)
    row["pass"] = max(errs) < 1e-3 and not row["fallbacks"]
    _, xs = build(name, batch=3, seed=5)
    xc = [x.cuda() for x in xs]
    row["truestar_ms"] = latency(lambda: cp(xc))
    try:                                              # CUDA graph replay of the same compiled program
        g = cp.graphed()
        gout = g(xc)                                  # records after the verified call; checked bitwise
        row["graphed_equal"] = bool(torch.equal(gout[0], cp(xc)[0]))
        row["truestar_graphed_ms"] = latency(lambda: g(xc))
    except Exception as e:  # noqa: BLE001  a refused graph is recorded, not fatal
        row["truestar_graphed_ms"] = None
        row["graph_refused"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    with torch.no_grad():
        row["eager_ms"] = latency(lambda: eager(*xc))
    row["speedup_vs_eager"] = row["eager_ms"] / row["truestar_ms"]
    if row["truestar_graphed_ms"]:
        row["graphed_speedup_vs_eager"] = row["eager_ms"] / row["truestar_graphed_ms"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models")
    ap.add_argument("--json")
    a = ap.parse_args()
    import transformers
    from truestar import testing_hf as T
    names = a.models.split(",") if a.models else sorted(
        n for n in T.available(T.MODELS_2026) if n not in T.KNOWN_GAPS_2026)
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}, "
          f"{len(names)} families", flush=True)
    out = open(a.json, "w", encoding="utf-8") if a.json else None
    for n in names:
        try:
            r = run(n)
        except Exception as e:  # noqa: BLE001  a failure is a row, with its first line
            lines = [ln for ln in f"{type(e).__name__}: {e}".splitlines() if ln.strip()]
            r = {"family": n, "pass": False, "error": lines[0][:300], "trace": traceback.format_exc()[-1500:]}
        if out:
            out.write(json.dumps(r) + "\n")
            out.flush()
        if "error" in r:
            print(f"{n:24} FAIL  {r['error'][:150]}", flush=True)
        else:
            print(f"{n:24} {'PASS' if r['pass'] else 'FAIL'}  err {r['max_error_vs_float64']:.1e} "
                  f"(eager {r['eager_max_error_vs_float64']:.1e})  {r['kernels']} kernels, fallbacks "
                  f"{r['fallbacks'] or 'none'}  truestar {r['truestar_ms']:.2f} ms, eager {r['eager_ms']:.2f} ms "
                  f"({r['speedup_vs_eager']:.2f}x); graphed "
                  + (f"{r['truestar_graphed_ms']:.2f} ms ({r['graphed_speedup_vs_eager']:.2f}x)"
                     if r.get("truestar_graphed_ms") else f"refused: {r.get('graph_refused')}"), flush=True)
    if out:
        out.close()


if __name__ == "__main__":
    sys.exit(main())
