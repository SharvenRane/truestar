"""Speed of the quantized matmul kernels against float16 (Truestar's contraction kernel and
cuBLAS through eager PyTorch) and against torch._scaled_mm for FP8.

Shapes are the linear layers of a Llama 7B class block (dim 4096, hidden 11008). Times are the
median of `--rounds` rounds, each the median of triton.testing.do_bench. Quantized rows include
the run time activation quantizer (FP8, FP4) unless marked "matmul only".

    PYTHONPATH=src python scripts/bench_quant.py --out bench_quant.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
import triton

from truestar.quant import formats as F
from truestar.quant import kernels as Q

SHAPES = [("qkvo 4096x4096", 4096, 4096), ("gate/up 11008x4096", 11008, 4096),
          ("down 4096x11008", 4096, 11008)]
MS = [1, 16, 256, 2048]
FORMATS = ["int8", "int4:g128", "fp8_e4m3:channel", "nvfp4", "mxfp4"]


REPS = 20


def timeit(fn, rounds: int) -> float:
    """GPU time per call in ms: REPS calls captured in one CUDA graph (so host launch overhead,
    which differs between Triton and cuBLAS launches, is not measured), replayed by do_bench;
    median over rounds."""
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            fn()
    torch.cuda.synchronize()
    t = statistics.median(triton.testing.do_bench(g.replay, warmup=10, rep=100, return_mode="median")
                          for _ in range(rounds))
    del g
    return t / REPS


def truestar_fp16(M: int, N: int, K: int):
    from truestar import kernel as KK
    g = KK.Graph()
    Ms = KK.sym("M")
    x = g.input("x", (Ms, K), "float16")
    w = g.input("w", (N, K), "float16")
    fn = g.compile({"y": x @ w.T}, "cuda")
    return lambda xx, ww: fn(x=xx, w=ww)["y"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    args = ap.parse_args()
    ms = [int(v) for v in args.ms.split(",")]
    torch.manual_seed(0)
    rows = []
    for label, N, K in SHAPES:
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        qws = {s: F.quantize_weight(w, s) for s in FORMATS}
        w8 = qws["fp8_e4m3:channel"]
        qwt = F.quantize_weight(w, "fp8_e4m3:tensor")
        tt = truestar_fp16(1, N, K)
        for M in ms:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            r = {"shape": label, "M": M, "N": N, "K": K}
            # cuBLAS picked a slower M = 1 kernel inside a CUDA graph in some runs: the baseline
            # is the faster of graph and plain launches, so no speedup comes from that
            r["eager fp16 (cuBLAS) ms"] = min(timeit(lambda: x @ w.T, args.rounds), statistics.median(
                triton.testing.do_bench(lambda: x @ w.T, warmup=25, rep=100, return_mode="median")
                for _ in range(args.rounds)))
            r["truestar fp16 ms"] = timeit(lambda: tt(x, w), args.rounds)
            for s, qw in qws.items():
                r[f"{s} ms"] = timeit(lambda: Q.quantized_matmul(x, qw), args.rounds)
            act = Q.quant_act_fp8(x, "e4m3")
            r["fp8_e4m3:channel matmul only ms"] = timeit(lambda: Q.matmul_fp8(act, w8), args.rounds)
            a4 = Q.quant_act_fp4(x, "nvfp4")
            r["nvfp4 matmul only ms"] = timeit(lambda: Q.matmul_fp4(a4, qws["nvfp4"]), args.rounds)
            # torch._scaled_mm: row wise scales, weight as a column major [K, N]
            wt = w8.data.t()                                   # [K, N], column major view
            sw = w8.scale.reshape(1, N)
            try:
                def torch_fp8():
                    s = (x.abs().amax(dim=1, keepdim=True).float() / 448.0)
                    xq = (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn)
                    return torch._scaled_mm(xq, wt, scale_a=s, scale_b=sw, out_dtype=torch.float16)
                sa = act.scale.reshape(M, 1)
                xq0 = act.data
                r["torch._scaled_mm fp8 ms"] = timeit(torch_fp8, args.rounds)
                r["torch._scaled_mm fp8 matmul only ms"] = timeit(
                    lambda: torch._scaled_mm(xq0, wt, scale_a=sa, scale_b=sw, out_dtype=torch.float16),
                    args.rounds)
            except Exception as e:  # report, never hide
                r["torch._scaled_mm fp8 rowwise"] = f"failed: {type(e).__name__}: {str(e)[:120]}"
                # torch 2.11 on sm_120 has tensorwise scaling only: one scale per operand
                wt_t = qwt.data.t()
                st = torch.ones((), device="cuda")

                def torch_fp8_t():
                    s = (x.abs().amax().float() / 448.0)
                    xq = (x.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn)
                    return torch._scaled_mm(xq, wt_t, scale_a=s, scale_b=qwt.scale.reshape(()),
                                            out_dtype=torch.float16)
                xq1 = act.data
                r["torch._scaled_mm fp8 tensorwise ms"] = timeit(torch_fp8_t, args.rounds)
                r["torch._scaled_mm fp8 tensorwise matmul only ms"] = timeit(
                    lambda: torch._scaled_mm(xq1, wt_t, scale_a=st, scale_b=st,
                                             out_dtype=torch.float16), args.rounds)
            rows.append(r)
            base = r["eager fp16 (cuBLAS) ms"]
            print(f"{label:20s} M={M:5d} " + "  ".join(
                f"{k.replace(' ms', '')}={v * 1000:.0f}us({base / v:.2f}x)" for k, v in r.items()
                if k.endswith(" ms")), flush=True)
        del qws, w, qwt
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                       "triton": triton.__version__, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
