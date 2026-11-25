"""Decode attention (one query token against a KV cache): Truestar against eager SDPA and
torch.compile max-autotune, at batch 1, 8 and 32 and context 512 to 8192.

    PYTHONPATH=src python scripts/bench_decode_attention.py --json out.json
    PYTHONPATH=src python scripts/bench_decode_attention.py --dtypes float16 --tools truestar

The case is softmax(q k^T / sqrt(D)) v with q [B, H, 1, D] and k, v [B, H, N, D] (H 8, D 128 by
default; a grouped query model's KV heads are expanded before attention in Truestar, so H is the
query head count). Tools:

* ``truestar``: the kernel layer's default launch config (tuning off), what a first decode step gets;
  its config (split KV chunk) is recorded. Verified against float64 on its first call.
* ``sdpa``: ``F.scaled_dot_product_attention`` eager.
* ``compile``: ``torch.compile(mode="max-autotune")`` of the same SDPA call, compiled per shape.

Timing: CUDA events around one call, 10 warm up calls, then ``--reps`` timed calls per round and
``--rounds`` rounds interleaved across tools; the median and the minimum over all timed calls are
reported in microseconds. Every output is compared with a float64 computation. Numbers from a shared
or a laptop GPU are provisional: clocks move with temperature and power.
"""

import argparse
import json
import os
import statistics
import tempfile

BATCHES = (1, 8, 32)
CONTEXTS = (512, 1024, 2048, 4096, 8192)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="float16,bfloat16")
    ap.add_argument("--tools", default="truestar,sdpa,compile")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    ap.add_argument("--json")
    a = ap.parse_args()
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_bench_dec_"))
    os.environ.setdefault("TRUESTAR_VERIFY_DEVICE", "cpu")   # the float64 KV cache would not fit

    import torch
    import torch.nn.functional as F
    from truestar import kernel as K
    from truestar.kernel import Graph

    tools = a.tools.split(",")
    H, D = a.heads, a.dim
    props = torch.cuda.get_device_properties(0)
    rows = []

    def kernel(B, dtype):
        g = Graph()
        T, N = K.sym("T"), K.sym("N")
        q = g.input("q", (B, H, T, D), dtype)
        k = g.input("k", (B, H, N, D), dtype)
        v = g.input("v", (B, H, N, D), dtype)
        (_, kern), = g.fuse({"o": g.softmax((q @ k.T) * (D ** -0.5)) @ v})
        return kern

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    def time_calls(fn, reps):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(reps)]
        for s, e in ev:
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 for s, e in ev]

    for dtype in a.dtypes.split(","):
        dt = getattr(torch, dtype)
        for B in (int(x) for x in a.batches.split(",")):
            ck = K.compile(kernel(B, dtype), "cuda", tune=False) if "truestar" in tools else None
            for N in (int(x) for x in a.contexts.split(",")):
                gen = torch.Generator(device="cuda").manual_seed(0)
                q = torch.randn(B, H, 1, D, device="cuda", generator=gen).to(dt)
                k = torch.randn(B, H, N, D, device="cuda", generator=gen).to(dt)
                v = torch.randn(B, H, N, D, device="cuda", generator=gen).to(dt)
                s = (q.double() @ k.double().transpose(-1, -2)) * D ** -0.5
                ref = torch.softmax(s, -1) @ v.double()
                fns, cfg, errs = {}, None, {}
                if "truestar" in tools:
                    fns["truestar"] = lambda: ck(q=q, k=k, v=v)
                if "sdpa" in tools:
                    fns["sdpa"] = lambda: sdpa(q, k, v)
                if "compile" in tools:
                    torch._dynamo.reset()
                    cf = torch.compile(sdpa, mode="max-autotune", dynamic=False)
                    fns["compile"] = lambda cf=cf: cf(q, k, v)
                for t, fn in fns.items():
                    out = fn()                                    # first call: compile, verify
                    torch.cuda.synchronize()
                    errs[t] = ((out.double() - ref).abs().max() / ref.abs().max()).item()
                    for _ in range(10):
                        fn()
                if "truestar" in tools:
                    cfg = ck.launch_configs()[-1][1]                # the signature of this N
                times = {t: [] for t in fns}
                for _ in range(a.rounds):
                    for t, fn in fns.items():
                        times[t] += time_calls(fn, a.reps)
                row = {"dtype": dtype, "B": B, "N": N, "H": H, "D": D,
                       "median_us": {t: statistics.median(x) for t, x in times.items()},
                       "min_us": {t: min(x) for t, x in times.items()},
                       "error": errs, "truestar_config": cfg}
                rows.append(row)
                med = row["median_us"]
                line = f"{dtype} B{B} N{N}: " + "  ".join(f"{t} {med[t]:.1f}/{row['min_us'][t]:.1f}"
                                                          for t in med)
                if "truestar" in med and "sdpa" in med:
                    line += f"  truestar/sdpa {med['truestar'] / med['sdpa']:.2f}"
                if "truestar" in med and "compile" in med:
                    line += f"  truestar/compile {med['truestar'] / med['compile']:.2f}"
                if cfg:
                    line += f"  KV_CHUNK {cfg.get('KV_CHUNK', 'n/a')}"
                print(line, flush=True)
                del q, k, v, ref, s
                torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"gpu": props.name, "sms": props.multi_processor_count, "torch": torch.__version__,
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
