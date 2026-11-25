"""Deterministic mode before and after order keeping tuning: speed, and the bits unchanged.

    PYTHONPATH=src python scripts/bench_detautotune.py --json out.json

For each kernel of scripts/order_classes.py (attention causal, full and decode, softmax and layer
norm, float16 and float32), two deterministic compiles: the default config (what a
deterministic kernel ran before, source ``pinned``) and ``tune=True`` (now tuned over the parameters
that keep the summation order, triton_backend.ORDER_FREE). Reports each one's time (CUDA events,
best of 7 interleaved rounds, each the median of 5 runs of 50 launches), the speedup, the config chosen and whether the outputs are bitwise equal.
"""

import argparse
import json
import os
import tempfile


def timed(fn, reps=50, warm=10):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return sorted(ts)[2]


def kernel_cases():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import order_classes
    return order_classes.cases()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="ts_detat_"))
    import torch
    from truestar import kernel as K
    from truestar.kernel import triton_backend as tb

    rows = []
    for name, kern, shapes, dtype in kernel_cases():
        gen = torch.Generator(device="cuda").manual_seed(0)
        dt = getattr(torch, dtype)
        ins = {n: torch.randn(*s, device="cuda", generator=gen).to(dt) for n, s in shapes.items()}
        tuned = K.compile(kern, "cuda", tune=True, deterministic=True)
        out_t = tuned(**ins)
        L = next(iter(tuned._fast.values()))
        p, env = tuned.art.plan, L.env
        d = tb.default_config(p, env)
        pinned = K.compile(kern, "cuda", tune=False, deterministic=True, _config=d)
        out_p = pinned(**ins)
        # interleaved rounds, best of each: same config pairs read equal (laptop clocks drift)
        ps, ts = [], []
        for _ in range(7):
            ps.append(timed(lambda: pinned(**ins)))
            ts.append(timed(lambda: tuned(**ins)))
        us_p, us_t = min(ps), min(ts)
        (_, cfg, src), = tuned.launch_configs()
        changed = {k: (d[k], cfg[k]) for k in cfg if cfg[k] != d.get(k)}
        row = {"case": name, "before_us": us_p, "after_us": us_t, "speedup": us_p / us_t, "source": src,
               "changed": {k: list(v) for k, v in changed.items()}, "bitwise_equal": torch.equal(out_p, out_t)}
        rows.append(row)
        print(f"{name:42} before {us_p:8.1f} us  after {us_t:8.1f} us  x{us_p / us_t:5.2f}  {src:12} "
              f"{changed or 'default'}  bitwise {row['bitwise_equal']}", flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
