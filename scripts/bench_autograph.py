"""Ordinary compiled calls before and after automatic CUDA graph replay, against eager PyTorch.

    PYTHONPATH=src python scripts/bench_autograph.py [--models a,b] [--json out.jsonl]

The 26 families of the coverage refresh (``testing_hf.MODELS_2026`` minus ``KNOWN_GAPS_2026``, as
scripts/cov26_cuda.py). One ``compile_program(prog, "cuda", "float32")`` per family, called the
ordinary way (``cp(inputs)``) with automatic graphs off (``cp.auto_graphs = False``: every call
launches every kernel from Python, as before this change) and on (the default: from the second call
at a signature the call is replayed from a CUDA graph). Per family:

* bits: at batch 3 (seed 5) and batch 2 (seed 6) the outputs with graphs on are compared with
  torch.equal against the outputs with graphs off, on the first replay and on a later one;
* latency at batch 3: median of 30 calls after 5 warm up calls, CUDA synchronized, for graphs off,
  graphs on and eager PyTorch float32 under ``torch.no_grad()``.
"""

import argparse
import copy
import json
import statistics
import time

import torch


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

    model, args = build(name, batch=2)
    m64 = copy.deepcopy(model).double()
    a64 = tuple(a.double() if a.is_floating_point() else a for a in args)
    prog = ir.import_module(m64, a64, dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
    cp = compile_program(prog, "cuda", "float32")
    eager = copy.deepcopy(model).float().cuda().eval()
    row = {"family": name, "bitwise_equal": True}
    for batch, seed in ((3, 5), (2, 6)):
        _, xs = build(name, batch=batch, seed=seed)
        xc = [x.cuda() for x in xs]
        cp.auto_graphs = False
        ref = [o.clone() for o in cp(xc)]                 # the first (verified) call, ordinary
        ref2 = [o.clone() for o in cp(xc)]                # ordinary again
        cp.auto_graphs = True
        got = [cp(xc) for _ in range(3)]                  # first auto call records, the rest replay
        same = all(torch.equal(a, b) for a, b in zip(ref, ref2)) and all(
            torch.equal(a, b) for g in got for a, b in zip(ref, g))
        row["bitwise_equal"] = row["bitwise_equal"] and same
    row["replays"] = cp.auto_graph_replays
    row["refusals"] = list(cp.auto_graph_refusals.values())
    _, xs = build(name, batch=3, seed=5)
    xc = [x.cuda() for x in xs]
    cp.auto_graphs = False
    row["before_ms"] = latency(lambda: cp(xc))
    cp.auto_graphs = True
    row["after_ms"] = latency(lambda: cp(xc))
    with torch.no_grad():
        row["eager_ms"] = latency(lambda: eager(*xc))
    row["before_vs_eager"] = row["eager_ms"] / row["before_ms"]
    row["after_vs_eager"] = row["eager_ms"] / row["after_ms"]
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
        except Exception as e:  # noqa: BLE001  a family that fails is a row with its first line
            r = {"family": n, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
        if out:
            out.write(json.dumps(r) + "\n")
            out.flush()
        if "error" in r:
            print(f"{n:24} ERROR {r['error']}", flush=True)
        else:
            print(f"{n:24} before {r['before_ms']:6.2f} ms ({r['before_vs_eager']:.2f}x)  after {r['after_ms']:6.2f} ms "
                  f"({r['after_vs_eager']:.2f}x)  eager {r['eager_ms']:6.2f} ms  bitwise {r['bitwise_equal']}  "
                  f"replays {r['replays']}{'  refused ' + str(r['refusals']) if r['refusals'] else ''}", flush=True)


if __name__ == "__main__":
    main()
