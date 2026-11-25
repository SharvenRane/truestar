"""Is the fair benchmark's causal attention number real? (FAIR_BENCH_2026_11, attn / bf16_causal:
Truestar 0.379 ms, 90.6 TFLOPS counting the causal half, 103% of the rated 87.9.)

    PYTHONPATH=<truestar src> python scripts/profile_causal_attn.py [--json out.json]

Same workload and inputs as the fair benchmark (fb_models.build("attn", "bf16_causal"): B 4, H 16,
S 2048, D 64, bfloat16, the inputs in data/attn__bf16_causal.npz). Measures:

* torch.profiler (CUPTI) device time of every kernel of 50 back to back calls: which kernels a
  call runs and their summed GPU time per call, next to cuDNN SDPA profiled the same way;
* the SM clock during those calls (nvidia-smi, every 100 ms), for the clock adjusted peak;
* an independent FLOP count: (a) the useful causal FLOPs, 4 B H D S (S + 1) / 2 (QK^T and PV,
  one multiply add = 2 FLOPs; softmax not counted), checked against torch's FlopCounterMode for
  the dense SDPA (which must be 4 B H S^2 D); (b) the FLOPs the kernel executes, from its launch
  config (BM x BN blocks: every key block that starts at or before the last query of a query block
  is computed whole) and the mma instruction count of its compiled PTX;
* the output against float64 attention, so the timed kernel does the full work.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("TRUESTAR_FAIR_BENCH", "fair_bench"), "scripts"))  # unpublished fair benchmark

DATA = os.path.join(os.environ.get("TRUESTAR_FAIR_BENCH", "fair_bench"), "data", "attn__bf16_causal.npz")


class Clocks(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], False

    def run(self):
        while not self.stop:
            out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            try:
                self.samples.append([float(x) for x in out.strip().split(",")])
            except ValueError:
                pass
            time.sleep(0.1)


def profile(torch, fn, calls=50):
    from torch.profiler import ProfilerActivity, profile as tprof
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    clk = Clocks()
    clk.start()
    t_end = time.time() + 1.5
    while time.time() < t_end:            # let the clock settle under load, as in the benchmark
        fn()
    torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    t_end = time.time() + 1.0
    while time.time() < t_end:
        fn()
    torch.cuda.synchronize()
    clk.stop = True
    clk.join()
    kernels = {}
    for e in prof.events():
        if e.device_type.name == "CUDA" and e.device_time > 0:
            k = kernels.setdefault(e.name, [0, 0.0])
            k[0] += 1
            k[1] += e.device_time       # microseconds
    per_call = {n: dict(launches=c / calls, us_per_call=t / calls) for n, (c, t) in kernels.items()}
    sm = sorted(s[0] for s in clk.samples if s[1] > 50)
    return per_call, (sm[len(sm) // 2] if sm else None), clk.samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    import perf_gate
    perf_gate.wait_idle(quiet=False)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import fb_models as M
    from truestar import ir
    from truestar.compiler import compile_program

    B, H, S, D = 4, 16, 2048, 64
    m64, ins64, _ = M.build("attn", "bf16_causal")
    z = np.load(DATA)
    print("npz keys", list(z.keys()))
    m32 = m64.to(torch.float32).eval()
    ex = tuple(t.to(torch.float32) for t in ins64)
    xs = [t.to("cuda", torch.bfloat16) for t in ins64]
    prog = ir.import_module(m32, ex, name="attn_bf16_causal")
    cp = compile_program(prog, "cuda", "bfloat16")
    ts = lambda: cp(xs)[0]  # noqa: E731
    sd = lambda: F.scaled_dot_product_attention(*xs, is_causal=True)  # noqa: E731
    out = ts()
    q64, k64, v64 = (t.to("cuda", torch.float64) for t in ins64)
    ref = torch.cat([F.scaled_dot_product_attention(q64[b:b + 1], k64[b:b + 1], v64[b:b + 1], is_causal=True)
                     for b in range(B)])
    err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
    err_sd = ((sd().double() - ref).abs().max() / ref.abs().max()).item()
    print(f"error against float64: truestar {err:.3e}, sdpa {err_sd:.3e}")

    # (a) useful FLOPs, and torch's own count for the dense op
    useful = 4 * B * H * D * S * (S + 1) // 2
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        F.scaled_dot_product_attention(*xs)
    dense_torch = fc.get_total_flops()
    print(f"FLOPs: useful causal {useful:.4e}; torch FlopCounterMode dense SDPA {dense_torch:.4e} "
          f"(formula 4BHS^2D {4 * B * H * S * S * D:.4e})")

    # (b) the attention kernel's launch config, blocks computed, and its PTX mma count
    steps = [s for s in cp.plan.steps if getattr(s, "kind", "") == "kernel"]
    info = []
    for s in steps:
        for sig, cfg, src in s.ck.launch_configs():
            info.append(dict(kernel=s.name, config=cfg, source=src))
    print("kernel steps:", json.dumps(info))
    att = [i for i in info if "BN" in i["config"] and "SPLIT" in i["config"]]
    executed = None
    if att:
        c = att[0]["config"]
        bm, bn = c["BM"], c["BN"]
        blocks = sum(min(-(-S // bn), (i * bm + bm - 1) // bn + 1) for i in range(-(-S // bm)))
        executed = 4 * B * H * D * bm * bn * blocks
        print(f"BM {bm} BN {bn}: {blocks} key blocks per head of {(-(-S // bm)) * (-(-S // bn))} "
              f"-> executed {executed:.4e} FLOPs ({executed / useful:.3f}x the useful count)")
    ptx_mma = {}
    cache = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    newest = []
    for root, _, files in os.walk(cache):
        for f in files:
            if f.endswith(".ptx"):
                p = os.path.join(root, f)
                newest.append((os.path.getmtime(p), p))
    for _, p in sorted(newest)[-40:]:
        txt = open(p, encoding="utf-8", errors="replace").read()
        n = len(re.findall(r"\bmma\.sync\.aligned\.m16n8k16[^;]*bf16", txt))
        if n:
            ptx_mma[os.path.basename(p)] = dict(mma_m16n8k16_bf16=n,
                                                entry=(re.findall(r"\.entry (\w+)", txt) or ["?"])[0])
    print("PTX mma.sync m16n8k16 bf16 counts (static):", json.dumps(ptx_mma))

    res = {}
    for name, fn in (("truestar", ts), ("sdpa_default", sd)):
        perf_gate.wait_idle()
        per_call, sm_mhz, samples = profile(torch, fn)
        tot = sum(v["us_per_call"] for v in per_call.values())
        tflops = useful / (tot * 1e-6) / 1e12
        res[name] = dict(kernels=per_call, gpu_us_per_call=tot, sm_mhz_median=sm_mhz,
                         useful_tflops=tflops,
                         executed_tflops=None if executed is None or name != "truestar"
                         else executed / (tot * 1e-6) / 1e12)
        print(f"{name}: GPU time per call {tot:.1f} us over {len(per_call)} kernel(s) "
              f"{json.dumps(per_call)} | SM clock median {sm_mhz} MHz | useful {tflops:.1f} TFLOPS"
              + (f" | executed {res[name]['executed_tflops']:.1f} TFLOPS" if res[name]["executed_tflops"] else ""),
              flush=True)
    peak_rated = 87.9
    for name, r in res.items():
        adj = peak_rated * (r["sm_mhz_median"] or 2452) / 2452
        r["clock_adjusted_peak"] = adj
        print(f"{name}: useful {r['useful_tflops']:.1f} TFLOPS = {100 * r['useful_tflops'] / peak_rated:.0f}% "
              f"of rated 87.9, {100 * r['useful_tflops'] / adj:.0f}% of the clock adjusted {adj:.1f}"
              + (f"; executed {r['executed_tflops']:.1f} = {100 * r['executed_tflops'] / adj:.0f}% of it"
                 if r.get("executed_tflops") else ""))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(error=err, error_sdpa=err_sd, useful_flops=useful, dense_torch_flops=dense_torch,
                           executed_flops=executed, kernels=info, ptx_mma=ptx_mma, results=res), f, indent=1)


if __name__ == "__main__":
    main()
