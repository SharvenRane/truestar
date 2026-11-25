"""Compiled linear layers (relu(x @ W.T + b), the shapes of scripts/bench_matmul.py) through
K.compile: float32 without tensor cores, plain tensor cores and portable tensor cores, per call
time over 20 back to back calls (launches overlap the GPU work, so small shapes are launch bound for
every arm alike; a CUDA graph capture of the compiled kernels failed with an illegal address on
replay, also for plain float32, so it is not used here), error against float64 and the output hash.

    PYTHONPATH=src python scripts/tcexact_compiled_matmul.py --json out/compiled_matmul.json
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truestar import kernel as K  # noqa: E402
from truestar.kernel import Graph  # noqa: E402
from bench_matmul import SHAPES  # noqa: E402

ARMS = [("float32_fma", "float32", dict(deterministic=True)),
        ("tf32_plain", "float32", dict(deterministic=True, allow_tf32=True)),
        ("tf32_portable", "float32", dict(allow_tf32=True, portable_tensor_cores=True)),
        ("float16_plain", "float16", dict(deterministic=True)),
        ("float16_portable", "float16", dict(portable_tensor_cores=True)),
        ("bfloat16_plain", "bfloat16", dict(deterministic=True)),
        ("bfloat16_portable", "bfloat16", dict(portable_tensor_cores=True))]


def kernel(N, Kd, dtype):
    g = Graph()
    x = g.input("x", (K.sym("M"), Kd), dtype)
    w = g.input("w", (N, Kd), dtype)
    b = g.input("b", (N,), dtype)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--first", type=int, default=0, help="only the first n shapes")
    ap.add_argument("--no-graph", action="store_true", help="time the calls directly (debugging)")
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    doc = {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "rows": []}
    for M, N, Kd in (SHAPES[:args.first] if args.first else SHAPES):
        gen = torch.Generator().manual_seed(M * 7 + N * 3 + Kd)
        x32 = torch.randn(M, Kd, generator=gen)
        w32 = torch.randn(N, Kd, generator=gen) / Kd ** 0.5
        b32 = torch.randn(N, generator=gen)
        row = {"shape": [M, N, Kd], "us": {}, "err": {}, "sha": {}}
        graphs = {}
        for name, dt, kw in ARMS:
            td = getattr(torch, dt)
            x, w, b = x32.to(td).cuda(), w32.to(td).cuda(), b32.to(td).cuda()
            ref = torch.relu(x.double() @ w.double().T + b.double())
            try:
                ck = K.compile(kernel(N, Kd, dt), "cuda", **kw)
                fn = (lambda ck=ck, x=x, w=w, b=b: ck(x=x, w=w, b=b))
                out = fn()
                row["err"][name] = float((out.double() - ref).abs().max() / ref.abs().max())
                row["sha"][name] = hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest()[:12]
                torch.cuda.synchronize()
                graphs[name] = fn if args.no_graph else (lambda fn=fn: [fn() for _ in range(20)])
            except Exception as e:  # recorded, never hidden
                row["err"][name] = f"{type(e).__name__}: {str(e)[:300]}"
        times = {t: [] for t in graphs}
        for _ in range(args.rounds):
            for t, g in graphs.items():
                if args.no_graph:
                    times[t].append(triton.testing.do_bench(g, warmup=3, rep=25, return_mode="median") * 1e3)
                else:
                    times[t].append(triton.testing.do_bench(g, warmup=3, rep=25, return_mode="median") / 20 * 1e3)
        row["us"] = {t: statistics.median(v) for t, v in times.items()}
        u = row["us"]
        if "float32_fma" in u:
            row["fma_over"] = {t: u["float32_fma"] / v for t, v in u.items()}
        doc["rows"].append(row)
        print(json.dumps({"shape": row["shape"], "us": {k: round(v, 1) for k, v in u.items()}}), flush=True)
        del graphs
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(doc, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
