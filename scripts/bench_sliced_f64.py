"""Sliced float64 matmul (docs/SLICED_F64.md): speed against native float64 on the GPU and the CPU,
accuracy against exact rational dot products, and the time of a kernel verification.

Run: PYTHONPATH=src python scripts/bench_sliced_f64.py [--sizes 1024,2048,4096] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import f64emu as F  # noqa: E402
from truestar.kernel import reference as R  # noqa: E402


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def exact_err(a, b, C, n=30):
    g = torch.Generator().manual_seed(9)
    pts = list(zip(torch.randint(0, a.shape[0], (n,), generator=g).tolist(),
                   torch.randint(0, b.shape[1], (n,), generator=g).tolist()))
    A, Bt = a.cpu().tolist(), b.cpu().t().tolist()
    worst = 0.0
    for i, j in pts:
        ex = sum(Fraction(x) * Fraction(y) for x, y in zip(A[i], Bt[j]))
        sc = float(sum(abs(x * y) for x, y in zip(A[i], Bt[j])))
        worst = max(worst, float(abs(Fraction(float(C[i, j])) - ex)) / sc)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096,8192")
    ap.add_argument("--out", default="")
    ap.add_argument("--cpu-max", type=int, default=4096)
    args = ap.parse_args()
    doc = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": []}
    torch.manual_seed(0)
    for n in [int(s) for s in args.sizes.split(",")]:
        for kind in ("gaussian", "wide"):
            a = torch.randn(n, n, dtype=torch.float64, device="cuda")
            b = torch.randn(n, n, dtype=torch.float64, device="cuda")
            if kind == "wide":
                a *= torch.exp2(torch.randint(-20, 20, (n, n), device="cuda").double())
                b *= torch.exp2(torch.randint(-20, 20, (n, n), device="cuda").double())
            C, info = F.sliced_matmul(a, b, check_cpu=n <= args.cpu_max)
            row = {"n": n, "kind": kind, "info": info.as_dict(),
                   "ms_sliced": round(timed(lambda: F.sliced_matmul(a, b, check_cpu=False)), 2),
                   "ms_native_gpu_float64": round(timed(lambda: a @ b), 2)}
            if n <= args.cpu_max:
                ac, bc = a.cpu(), b.cpu()
                t = time.perf_counter()
                ac @ bc
                row["ms_cpu_float64"] = round((time.perf_counter() - t) * 1e3, 2)
            if n <= 2048:
                row["err_over_absprod_sliced"] = exact_err(a, b, C)
                row["err_over_absprod_native_gpu"] = exact_err(a, b, a @ b)
                row["gamma_K"] = F.gamma(n)
            row["speedup_vs_native_gpu"] = round(row["ms_native_gpu_float64"] / row["ms_sliced"], 2)
            print(json.dumps(row))
            doc["rows"].append(row)
            del a, b, C
            torch.cuda.empty_cache()
    # one kernel verification (first call float64 check) of a float32 matmul, native vs sliced
    n = 4096
    A = K.buffer("a", (n, n), "float32")
    B = K.buffer("b", (n, n), "float32")
    i, j, r = K.axis("i", n), K.axis("j", n), K.axis("r", n)
    k = K.kernel("mm", [K.output("y", (i, j), K.reduce_sum(r, A[i, r] * B[r, j]))])
    x, y = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    ver = {}
    for mode in ("native", "sliced"):
        os.environ["TRUESTAR_VERIFY_F64"] = mode
        os.environ["TRUESTAR_VERIFY"] = "all"
        fn = K.compile(k, "cuda", tune=False)
        fn(a=x, b=y)
        torch.cuda.synchronize()
        t_ver = timed(lambda: fn(a=x, b=y))
        os.environ["TRUESTAR_VERIFY"] = "0"
        t_run = timed(lambda: fn(a=x, b=y))
        ver[mode] = {"ms_call_with_check": round(t_ver, 2), "ms_call_no_check": round(t_run, 2)}
    os.environ.pop("TRUESTAR_VERIFY", None)
    ver["sliced_used"] = R.sliced_stats["used"]
    doc["verification_4096_matmul"] = ver
    print(json.dumps(ver))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)


if __name__ == "__main__":
    main()
