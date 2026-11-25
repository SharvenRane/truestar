"""Linear layer sweep: Truestar contraction kernels against cuBLAS and torch.compile.

    PYTHONPATH=src python scripts/bench_matmul.py                  # strict float32 (TF32 off)
    PYTHONPATH=src python scripts/bench_matmul.py --tf32           # float32 with TF32 allowed
    PYTHONPATH=src python scripts/bench_matmul.py --dtype float16 --rounds 9 --json out.json

Every case is ``relu(x @ W.T + b)`` with weights ``[N, K]`` and a symbolic row count M. Tools:

* ``truestar``: the default launch config (tuning off), what a first call at a size that is not
  worth tuning gets.
* ``tuned``: the same kernel tuned for this shape (tuning on, its own cold cache).
* ``F.linear``: ``torch.relu(F.linear(x, W, b))``, cuBLAS plus one elementwise kernel.
* ``matmul``: ``x @ W.T``, the bare cuBLAS product without bias or relu (a floor, not the same work).
* ``compile``: ``torch.compile`` of the F.linear version (Inductor, default mode).

Each tool runs as a CUDA graph of 20 calls; times are per call. The GPU may be shared, so rounds
are interleaved across tools and the report gives the median and the minimum over rounds. Every
output is compared with float64 and the largest relative error is printed. Nothing is tuned per
case except the ``tuned`` row.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile

SHAPES = [(1, 4096, 4096), (16, 4096, 1024), (64, 2048, 1024), (64, 2048, 2048), (64, 1000, 2048),
          (512, 1024, 256), (512, 256, 1024), (1024, 1024, 1024), (2048, 4096, 1024),
          (4096, 4096, 4096),
          # tall K: few output tiles, a long reduction
          (64, 512, 8192), (16, 1024, 16384), (128, 256, 8192),
          # the transformer layer of scripts/bench_compiler.py at batch 64 (M = 64 x 128 tokens)
          (8192, 768, 256), (8192, 1024, 256), (8192, 256, 1024)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--tf32", action="store_true", help="float32 with TF32 allowed (both sides)")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--shapes", help="comma separated MxNxK list (default: the built in sweep)")
    ap.add_argument("--tools", default="truestar,tuned,F.linear,matmul,compile")
    ap.add_argument("--json")
    args = ap.parse_args()
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_bench_mm_"))
    os.environ.setdefault("TRUESTAR_TUNE_BUDGET", "120")

    import torch
    import torch.nn.functional as F
    import triton.testing
    from truestar import kernel as K
    from truestar.kernel import Graph

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    tools = args.tools.split(",")
    shapes = SHAPES if not args.shapes else [tuple(int(v) for v in s.split("x"))
                                              for s in args.shapes.split(",")]
    dt = getattr(torch, args.dtype)

    def kernel(N, Kd):
        g = Graph()
        x = g.input("x", (K.sym("M"), Kd), args.dtype)
        w = g.input("w", (N, Kd), args.dtype)
        b = g.input("b", (N,), args.dtype)
        (_, kern), = g.fuse({"y": g.relu(x @ g.transpose(w, 0, 1) + b)})
        return kern

    def graph_of(fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        return g

    rows = []
    for M, N, Kd in shapes:
        gen = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(M, Kd, device="cuda", generator=gen).to(dt)
        w = (torch.randn(N, Kd, device="cuda", generator=gen) / Kd ** 0.5).to(dt)
        b = torch.randn(N, device="cuda", generator=gen).to(dt)
        ref = torch.relu(x.double() @ w.double().T + b.double())
        fns, cfgs = {}, {}
        for tool in tools:
            if tool in ("truestar", "tuned"):
                ck = K.compile(kernel(N, Kd), "cuda", allow_tf32=args.tf32, tune=(tool == "tuned"))
                fns[tool] = (lambda ck=ck: ck(x=x, w=w, b=b))
                fns[tool]()
                cfgs[tool] = ck.launch_configs()[0][1]
            elif tool == "F.linear":
                fns[tool] = lambda: torch.relu(F.linear(x, w, b))
            elif tool == "matmul":
                fns[tool] = lambda: x @ w.T
            elif tool == "compile":
                cf = torch.compile(lambda x, w, b: torch.relu(F.linear(x, w, b)))
                fns[tool] = (lambda cf=cf: cf(x, w, b))
        errs = {}
        for tool, fn in fns.items():
            if tool != "matmul":
                out = fn().double()
                errs[tool] = ((out - ref).abs().max() / ref.abs().max()).item()
        graphs = {t: graph_of(f) for t, f in fns.items()}
        times = {t: [] for t in graphs}
        for _ in range(args.rounds):
            for t, g in graphs.items():
                times[t].append(triton.testing.do_bench(g.replay, warmup=3, rep=25,
                                                        return_mode="median") / 20 * 1e3)
        row = {"shape": [M, N, Kd], "median_us": {t: statistics.median(v) for t, v in times.items()},
               "min_us": {t: min(v) for t, v in times.items()}, "error": errs,
               "configs": {t: c for t, c in cfgs.items()}}
        rows.append(row)
        med = row["median_us"]
        line = f"{M}x{N}x{Kd}: " + "  ".join(f"{t} {med[t]:.1f}/{row['min_us'][t]:.1f}" for t in med)
        if "truestar" in med and "F.linear" in med:
            line += f"  truestar/F.linear {med['truestar'] / med['F.linear']:.2f}"
        print(line, flush=True)
        del graphs
    print()
    print("| M x N x K | " + " | ".join(f"{t} (us, median / min)" for t in rows[0]["median_us"])
          + " | Truestar / F.linear | max error Truestar | max error F.linear |")
    print("|---|" + "---|" * (len(rows[0]["median_us"]) + 3))
    for r in rows:
        med, mn = r["median_us"], r["min_us"]
        ratio = f"{med['truestar'] / med['F.linear']:.2f}" if "truestar" in med and "F.linear" in med else ""
        print(f"| {' x '.join(map(str, r['shape']))} | "
              + " | ".join(f"{med[t]:.1f} / {mn[t]:.1f}" for t in med)
              + f" | {ratio} | {r['error'].get('truestar', float('nan')):.1e} | "
              f"{r['error'].get('F.linear', float('nan')):.1e} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"dtype": args.dtype, "tf32": args.tf32, "gpu": torch.cuda.get_device_name(0),
                       "torch": torch.__version__, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
