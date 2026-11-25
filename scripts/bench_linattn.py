"""Speed, chunk size invariance and chain error of truestar.linattn (docs/LINEAR_ATTENTION.md).

    PYTHONPATH=src python scripts/bench_linattn.py              # speed table
    PYTHONPATH=src python scripts/bench_linattn.py --invariance # chunk size and chain error numbers
    PYTHONPATH=src python scripts/bench_linattn.py --sweep      # launch config sweep behind config()

Speed rows, all float32 inputs on one CUDA GPU, forward only, median of triton.testing.do_bench:
* sequential: the recurrence one step at a time in float32 (reference.recurrent with dtype
  float32), the shape of Mamba 1's and the models' own slow path;
* eager chunked: chunked.py in float32 with chunk 64 (the eager PyTorch form written from the math);
* truestar: the Triton kernels, deterministic config (strict float32 dots), and with TF32 dots.
Every Truestar row is checked against float64 once before it is timed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar import linattn as LA  # noqa: E402
from truestar.linattn import triton_kernels as TK  # noqa: E402
from truestar.linattn.chunked import chunked  # noqa: E402
from truestar.linattn.reference import recurrent  # noqa: E402

# name, B, T, H, dk, dv, rule
SHAPES = [
    ("mamba2-2.7b layer (SSD)", 1, 4096, 80, 128, 64, "add"),
    ("qwen3-next layer (GDN)", 1, 4096, 32, 128, 128, "delta"),
    ("small batch 4 (GDN)", 4, 2048, 16, 64, 64, "delta"),
    ("small batch 4 (SSD)", 4, 2048, 16, 64, 64, "add"),
]


def inputs(B, T, H, dk, dv, rule, seed=0, gscale=0.3, dev="cuda"):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, dk, generator=gen) * dk ** -0.5
    k = torch.nn.functional.normalize(torch.randn(B, T, H, dk, generator=gen), dim=-1)
    v = torch.randn(B, T, H, dv, generator=gen)
    g = -torch.rand(B, T, H, generator=gen) * gscale
    beta = torch.sigmoid(torch.randn(B, T, H, generator=gen)) if rule == "delta" else None
    return tuple(None if x is None else x.to(dev) for x in (q, k, v, g, beta))


def bench(fn, rep=50):
    import triton
    return triton.testing.do_bench(fn, warmup=5, rep=rep)


def speed():
    rows = []
    for name, B, T, H, dk, dv, rule in SHAPES:
        q, k, v, g, beta = inputs(B, T, H, dk, dv, rule)
        LA.linear_attention(q, k, v, g, beta, rule=rule, deterministic=True)       # checked once
        LA.linear_attention(q, k, v, g, beta, rule=rule, allow_tf32=True)
        r = {"shape": name, "B": B, "T": T, "H": H, "dk": dk, "dv": dv}
        r["sequential_ms"] = bench(lambda: recurrent(q, k, v, g, beta, rule=rule, dtype=torch.float32), rep=200)
        r["eager_chunked_ms"] = bench(lambda: chunked(q, k, v, g, beta, rule=rule, chunk=64))
        r["truestar_ms"] = bench(lambda: LA.linear_attention(q, k, v, g, beta, rule=rule, deterministic=True))
        r["truestar_tf32_ms"] = bench(lambda: LA.linear_attention(q, k, v, g, beta, rule=rule, allow_tf32=True))
        rows.append(r)
        print(f"{name:28s} seq {r['sequential_ms']:8.2f} ms  eager {r['eager_chunked_ms']:7.2f} ms  "
              f"truestar {r['truestar_ms']:6.2f} ms ({r['sequential_ms'] / r['truestar_ms']:5.0f}x seq, "
              f"{r['eager_chunked_ms'] / r['truestar_ms']:4.1f}x eager)  tf32 {r['truestar_tf32_ms']:6.2f} ms", flush=True)
    return rows


def invariance():
    out = {}
    for rule, dtype in itertools.product(("add", "delta"), (torch.float32, torch.bfloat16)):
        q, k, v, g, beta = inputs(2, 2048, 4, 128, 128, rule)
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        rep = LA.chunk_invariance(q, k, v, g, beta, rule=rule, chunks=(16, 32, 64))
        key = f"{rule} {str(dtype)[6:]}"
        out[key] = rep
        print(key, "vs float64:", {c: "%.2e" % e["o"] for c, e in rep["vs_float64"].items()},
              "pairs:", {p: ("%.2e" % d["per_element"], "%.0f%% differ" % (100 * d["elements_differing"]))
                         for p, d in rep["pairs"].items()}, flush=True)
    q64 = [None if x is None else x.double().cpu() for x in inputs(1, 512, 2, 64, 64, "delta")]
    o_ref, _ = recurrent(*q64, rule="delta")
    f64 = {c: float(((chunked(*q64, rule="delta", chunk=c)[0] - o_ref).abs().max() / o_ref.abs().max()))
           for c in (1, 16, 32, 64, 128)}
    out["delta float64 chunked"] = f64
    print("delta float64 chunked vs sequential (max abs / max |ref|):", {c: "%.1e" % e for c, e in f64.items()})
    for rule, gscale in itertools.product(("add", "delta"), (0.0, 0.01, 0.3)):
        q, k, v, g, beta = inputs(1, 16384, 2, 64, 64, rule, gscale=gscale)
        if gscale == 0.0:
            g = None
        rep = LA.chain_error(q, k, v, g, beta, rule=rule, chunk=TK.DET_CHUNK)
        key = f"chain {rule} decay {gscale}"
        out[key] = {k_: rep[k_] for k_ in ("worst", "slope_per_chunk", "last_over_first", "chunks", "code")}
        print(key, {k_: rep[k_] for k_ in ("worst", "slope_per_chunk", "last_over_first", "chunks", "code")}, flush=True)
    return out


def sweep():
    import triton
    for rule, dk, BH in itertools.product(("add", "delta"), (64, 128), (8, 32)):
        q, k, v, g, beta = inputs(BH // 4, 4096, 4, dk, dk, rule)
        res = []
        for c, bv, w in itertools.product(TK.CHUNKS, (16, 32, 64), (4, 8)):
            cfg = {"CHUNK": c, "BK": dk, "BV": bv, "num_warps": w}
            try:
                ms = bench(lambda: TK.forward(q, k, v, g, beta, rule=rule, scale=1.0, initial_state=None, cfg=cfg), rep=30)
            except Exception:  # out of shared memory
                ms = float("nan")
            res.append((ms, c, bv, w))
        pick = TK.config(BH // 4, 4, dk, dk, deterministic=True)
        mine = [r[0] for r in res if r[1:] == (pick["CHUNK"], pick["BV"], pick["num_warps"])][0]
        best = min(r for r in res if r[0] == r[0])
        print(f"{rule} dk {dk} B*H {BH}: config() {mine:.2f} ms, best {best[0]:.2f} ms (C{best[1]} BV{best[2]} "
              f"W{best[3]}), ratio {mine / best[0]:.2f}", flush=True)
    del triton


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--invariance", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    print(torch.cuda.get_device_name(0), torch.__version__)
    res = sweep() if a.sweep else invariance() if a.invariance else speed()
    if a.json and res is not None:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
