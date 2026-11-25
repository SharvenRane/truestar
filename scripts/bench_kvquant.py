"""Decode attention over a quantized KV cache (truestar.kvquant) against the float16 cache.

    PYTHONPATH=src python scripts/bench_kvquant.py --json out.json

q [B, 8, 1, 128] float16 against a cache of N tokens with 8 KV heads. Tools:

* ``kvq-int8``, ``kvq-fp8``: ``kvquant.decode_attention`` over an int8 or fp8 e4m3 cache with one
  float32 scale per head and 32 tokens (default chunks).
* ``splitkv``: Truestar's generated split KV attention over the float16 cache (kernel layer default).
* ``sdpa``: ``F.scaled_dot_product_attention`` eager over the float16 cache.

Timing as scripts/bench_decode_attention.py: CUDA events around each call, 10 warm up calls, 5
interleaved rounds of 25 timed calls, median and minimum in microseconds. Reported with each row:
KV bytes per token (all heads, K and V, scales included) and every output's largest absolute
error against float64 attention over the unquantized cache. Provisional numbers on a laptop GPU.
"""

import argparse
import json
import os
import statistics
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--contexts", default="2048,8192")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--json")
    a = ap.parse_args()
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_bench_kvq_"))
    os.environ.setdefault("TRUESTAR_VERIFY_DEVICE", "cpu")

    import torch
    import torch.nn.functional as F
    from truestar import kernel as K
    from truestar import kvquant as KQ
    from truestar.kernel import Graph

    H, D = a.heads, a.dim

    def kernel(B):
        g = Graph()
        T, N = K.sym("T"), K.sym("N")
        q = g.input("q", (B, H, T, D), "float16")
        k = g.input("k", (B, H, N, D), "float16")
        v = g.input("v", (B, H, N, D), "float16")
        (_, kern), = g.fuse({"o": g.softmax((q @ k.T) * (D ** -0.5)) @ v})
        return kern

    def time_calls(fn, reps):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(reps)]
        for s, e in ev:
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 for s, e in ev]

    rows = []
    for B in (int(x) for x in a.batches.split(",")):
        ck = K.compile(kernel(B), "cuda", tune=False)
        for N in (int(x) for x in a.contexts.split(",")):
            gen = torch.Generator(device="cuda").manual_seed(0)
            q = torch.randn(B, H, 1, D, device="cuda", generator=gen).half()
            k = torch.randn(B, H, N, D, device="cuda", generator=gen).half()
            v = torch.randn(B, H, N, D, device="cuda", generator=gen).half()
            ref = torch.softmax((q.double() @ k.double().transpose(-1, -2)) * D ** -0.5, -1) @ v.double()
            caches = {f: (KQ.quantize(k, f, a.group), KQ.quantize(v, f, a.group)) for f in KQ.FORMATS}
            fns = {"kvq-int8": lambda: KQ.decode_attention(q, *caches["int8"]),
                   "kvq-fp8": lambda: KQ.decode_attention(q, *caches["fp8"]),
                   "splitkv": lambda: ck(q=q, k=k, v=v),
                   "sdpa": lambda: F.scaled_dot_product_attention(q, k, v)}
            errs = {}
            for t, fn in fns.items():
                out = fn()
                torch.cuda.synchronize()
                errs[t] = (out.double() - ref).abs().max().item()
                for _ in range(10):
                    fn()
            times = {t: [] for t in fns}
            for _ in range(a.rounds):
                for t, fn in fns.items():
                    times[t] += time_calls(fn, a.reps)
            mem = {"kvq-int8": KQ.bytes_per_token("int8", D, a.group, H),
                   "kvq-fp8": KQ.bytes_per_token("fp8", D, a.group, H),
                   "splitkv": KQ.bytes_per_token("float16", D, a.group, H),
                   "sdpa": KQ.bytes_per_token("float16", D, a.group, H)}
            row = {"B": B, "N": N, "H": H, "D": D, "group": a.group,
                   "median_us": {t: statistics.median(x) for t, x in times.items()},
                   "min_us": {t: min(x) for t, x in times.items()},
                   "max_abs_error_vs_float64": errs, "kv_bytes_per_token": mem,
                   "kvq_chunk": KQ.chunk_for(B, H, N), "splitkv_config": ck.launch_configs()[-1][1]}
            rows.append(row)
            med = row["median_us"]
            print(f"B{B} N{N}: " + "  ".join(f"{t} {med[t]:.1f}/{row['min_us'][t]:.1f}us err {errs[t]:.1e}"
                                             for t in fns), flush=True)
            del q, k, v, ref, caches
            torch.cuda.empty_cache()
    if a.json:
        props = torch.cuda.get_device_properties(0)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"gpu": props.name, "sms": props.multi_processor_count, "torch": torch.__version__,
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
