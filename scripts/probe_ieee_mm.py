"""Strict float32 contraction probe: Truestar against cuBLAS, a wider tile sweep, bit checks.

    PYTHONPATH=src python scripts/probe_ieee_mm.py baseline [--shapes 1024x1024x1024,...]
    PYTHONPATH=src python scripts/probe_ieee_mm.py sweep --shapes 1024x1024x1024 [--grid wide]
    PYTHONPATH=src python scripts/probe_ieee_mm.py ttgir --shapes 1024x1024x1024

Every case is ``relu(x @ W.T + b)`` in float32 with TF32 off on both sides, as in
scripts/bench_matmul.py. ``baseline`` times the bare cuBLAS product, ``F.linear`` plus relu, and
Truestar's default launch config in deterministic mode (never split K) and in the default mode,
each as a CUDA graph of 20 calls (median over interleaved rounds), with the max relative error
against float64. ``sweep`` forces every config of a grid in deterministic mode (SPLIT_K 1) and
checks that each output has the same bits as the default config's. ``ttgir`` prints the layouts
Triton chose for the default config.
"""

import argparse
import itertools
import json
import os
import statistics
import sys
import tempfile

SHAPES = [(1, 4096, 4096), (16, 4096, 1024), (64, 2048, 1024), (64, 2048, 2048), (64, 1000, 2048),
          (512, 1024, 256), (512, 256, 1024), (1024, 1024, 1024), (2048, 4096, 1024),
          (4096, 4096, 4096), (64, 512, 8192), (16, 1024, 16384), (128, 256, 8192),
          (8192, 768, 256), (8192, 1024, 256), (8192, 256, 1024)]


def parse_shapes(s):
    return SHAPES if not s else [tuple(int(v) for v in t.split("x")) for t in s.split(",")]


def setup():
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="ts_probe_mm_"))
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def kernel_of(N, Kd):
    from truestar import kernel as K
    from truestar.kernel import Graph
    g = Graph()
    x = g.input("x", (K.sym("M"), Kd), "float32")
    w = g.input("w", (N, Kd), "float32")
    b = g.input("b", (N,), "float32")
    (_, kern), = g.fuse({"y": g.relu(x @ g.transpose(w, 0, 1) + b)})
    return kern


def data(torch, M, N, Kd):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M, Kd, device="cuda", generator=gen)
    w = torch.randn(N, Kd, device="cuda", generator=gen) / Kd ** 0.5
    b = torch.randn(N, device="cuda", generator=gen)
    return x, w, b


def graph_of(torch, fn, calls=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    return g


def timed(torch, fns, rounds):
    import triton.testing
    graphs = {t: graph_of(torch, f) for t, f in fns.items()}
    times = {t: [] for t in graphs}
    for _ in range(rounds):
        for t, g in graphs.items():
            times[t].append(triton.testing.do_bench(g.replay, warmup=3, rep=25,
                                                    return_mode="median") / 20 * 1e3)
    del graphs
    torch.cuda.empty_cache()
    return {t: statistics.median(v) for t, v in times.items()}, {t: min(v) for t, v in times.items()}


def rel_err(out, ref):
    return ((out.double() - ref).abs().max() / ref.abs().max()).item()


def baseline(args):
    torch = setup()
    import torch.nn.functional as F
    from truestar import kernel as K
    rows = []
    print(f"gpu {torch.cuda.get_device_name(0)}, SMs {torch.cuda.get_device_properties(0).multi_processor_count}")
    for M, N, Kd in parse_shapes(args.shapes):
        x, w, b = data(torch, M, N, Kd)
        ref = torch.relu(x.double() @ w.double().T + b.double())
        kern = kernel_of(N, Kd)
        fns, cfgs, errs = {}, {}, {}
        for name, det in (("ts_det", True), ("ts", False)):
            ck = K.compile(kern, "cuda", tune=False, deterministic=det)
            fns[name] = (lambda ck=ck: ck(x=x, w=w, b=b))
            errs[name] = rel_err(fns[name](), ref)
            cfgs[name] = ck.launch_configs()[0][1]
        fns["F.linear"] = lambda: torch.relu(F.linear(x, w, b))
        errs["F.linear"] = rel_err(fns["F.linear"](), ref)
        fns["matmul"] = lambda: x @ w.T
        med, mn = timed(torch, fns, args.rounds)
        best = min(med["F.linear"], med["matmul"])
        row = dict(shape=[M, N, Kd], median_us=med, min_us=mn, error=errs, configs=cfgs)
        rows.append(row)
        print(f"{M}x{N}x{Kd}: " + "  ".join(f"{t} {med[t]:.1f}" for t in med)
              + f"  det/F.linear {med['ts_det'] / med['F.linear']:.2f}"
              + f"  det/matmul {med['ts_det'] / med['matmul']:.2f}"
              + f"  ts/best {med['ts'] / best:.2f}"
              + f"  err det {errs['ts_det']:.1e} ts {errs['ts']:.1e} cublas {errs['F.linear']:.1e}"
              + f"  cfg det {cfgs['ts_det']} ts {cfgs['ts']}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}, f)


GRIDS = {
    # (BM, BN, BK, warps, stages, GROUP)
    "wide": [c for c in itertools.product((64, 128, 256), (64, 128, 256), (8, 16, 32), (4, 8), (2, 3, 4), (8,))
             if 4096 <= c[0] * c[1] <= 32768 and c[0] * c[1] // (32 * c[3]) <= 128],
    "small": [c for c in itertools.product((16, 32, 64), (16, 32, 64, 128), (32, 64, 128), (1, 2, 4), (3,), (8,))
              if 4 <= c[0] * c[1] // (32 * c[3]) <= 32 and c[1] >= 8 * c[3]],
    "split": [c + (s,) for c in ((64, 16, 32, 1, 3, 8), (64, 16, 32, 2, 3, 8), (16, 32, 64, 2, 3, 8),
                                  (16, 128, 64, 2, 3, 8), (32, 64, 32, 2, 4, 8), (64, 128, 16, 4, 3, 8),
                                  (16, 64, 64, 2, 3, 8)) for s in (1, 2, 4, 8, 16, 32)],
    "group": [(128, 128, 16, 4, 3, g) for g in (1, 4, 8, 16, 32)] + [(128, 256, 16, 8, 3, g) for g in (1, 4, 16)],
}


def sweep(args):
    torch = setup()
    from truestar import kernel as K
    from truestar.kernel import triton_backend as tb
    for M, N, Kd in parse_shapes(args.shapes):
        x, w, b = data(torch, M, N, Kd)
        kern = kernel_of(N, Kd)
        base = K.compile(kern, "cuda", tune=False, deterministic=True)
        y0 = base(x=x, w=w, b=b).clone()
        default = base.launch_configs()[0][1]
        p = base.art.plan
        env = {} if isinstance(p.m.size, int) else {p.m.size.name: M}
        fns, bad, skipped = {"default": (lambda: base(x=x, w=w, b=b))}, [], 0
        for g in GRIDS[args.grid]:
            bm, bn, bk, warps, stages, group = g[:6]
            split = g[6] if len(g) > 6 else 1
            c = dict(BM=bm, BN=bn, BK=bk, GROUP=group, num_warps=warps, num_stages=stages, KTAIL=True, SPLIT_K=split)
            if tb.check_config(p, env, c) is not None:
                skipped += 1
                continue
            try:
                ck = K.compile(kern, "cuda", tune=False, deterministic=split == 1, _config=c)
                y = ck(x=x, w=w, b=b)
                y2 = ck(x=x, w=w, b=b)
                torch.cuda.synchronize()
                if not torch.equal(y, y2):
                    print(f"  RUN TO RUN DIFFERENCE {c}")
            except Exception as e:  # recorded, never hidden
                print(f"  fail {c}: {type(e).__name__}: {str(e)[:120]}")
                continue
            if split == 1 and not torch.equal(y, y0):
                bad.append(c)
            fns[tuple(g)] = (lambda ck=ck: ck(x=x, w=w, b=b))
        import torch.nn.functional as F
        fns["F.linear"] = lambda: torch.relu(F.linear(x, w, b))
        fns["matmul"] = lambda: x @ w.T
        # time in chunks so few graphs are alive at once
        keys = list(fns)
        med = {}
        for i in range(0, len(keys), 12):
            part = {k: fns[k] for k in keys[i:i + 12]}
            if "F.linear" not in part:
                part["F.linear"] = fns["F.linear"]
            m, _ = timed(torch, part, args.rounds)
            med.update({k: v for k, v in m.items() if k != "F.linear" or i + 12 >= len(keys)})
        ranked = sorted((v, k) for k, v in med.items() if k not in ("F.linear", "matmul", "default"))
        print(f"{M}x{N}x{Kd}: default {default} {med['default']:.1f} us, F.linear {med['F.linear']:.1f},"
              f" matmul {med['matmul']:.1f}, configs {len(ranked)}, skipped {skipped}, bit mismatches {len(bad)}")
        for v, k in ranked[:8]:
            print(f"   {k}: {v:.1f} us  ({v / med['F.linear']:.2f} of F.linear, {v / med['matmul']:.2f} of matmul)")
        if bad:
            print("   MISMATCH", bad[:5])
        sys.stdout.flush()


def ttgir(args):
    torch = setup()
    from truestar import kernel as K
    for M, N, Kd in parse_shapes(args.shapes):
        x, w, b = data(torch, M, N, Kd)
        ck = K.compile(kernel_of(N, Kd), "cuda", tune=False, deterministic=True)
        ck(x=x, w=w, b=b)
        print(M, N, Kd, ck.launch_configs()[0][1])
        fn = ck.art.module.kern
        caches = getattr(fn, "device_caches", None)
        if not caches:
            print("   no device_caches on", type(fn).__name__, [a for a in dir(fn) if "cache" in a])
        for dev_cache in (caches or {}).values():
            for compiled in dev_cache[0].values():
                src = compiled.asm.get("ttgir", "")
                for line in src.splitlines():
                    if line.startswith("#") or "tt.dot" in line or "local_load" in line:
                        print("  ", line.strip()[:240])
                ptx = compiled.asm.get("ptx", "")
                print("   ptx: fma", ptx.count("fma.rn.f32"), "ld.shared.v4", ptx.count("ld.shared.v4"),
                      "ld.shared.v2", ptx.count("ld.shared.v2"), "ld.shared", ptx.count("ld.shared"))
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="+", choices=["baseline", "sweep", "ttgir"])
    ap.add_argument("--shapes")
    ap.add_argument("--grid", default="wide", choices=sorted(GRIDS))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--json")
    args = ap.parse_args()
    for part in args.part:
        {"baseline": baseline, "sweep": sweep, "ttgir": ttgir}[part](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
