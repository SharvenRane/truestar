"""attention_grad (compiler/attention_bwd.py) against torch's SDPA backward, one layer's shapes.

    PYTHONPATH=src python scripts/bench_attention_grad.py --B 8 --H 16 --T 512 --D 64 --sweep

Times each of the three kernels for every block config in the sweep (each config is a fixed
function of the shapes once chosen; the sweep is how the table in ``configs`` was picked), the
whole op with the default table, and torch's ``scaled_dot_product_attention`` backward
(memory efficient kernel in float32). Also prints the error of each against float64.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar.compiler import attention_bwd as AB  # noqa: E402
from truestar.ir.attention_grad import reference_grad  # noqa: E402


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--kernels", default=None)
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    g = torch.Generator().manual_seed(0)
    shp = (a.B, a.T, a.H, a.D)
    q, k, v, do = (torch.randn(shp, generator=g).cuda().to(dt).transpose(1, 2) for _ in range(4))
    do = do.contiguous()
    scale = a.D ** -0.5
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = o.contiguous()
    res = {"gpu": torch.cuda.get_device_name(), "shape": [a.B, a.H, a.T, a.D], "dtype": a.dtype}
    # torch
    ql, kl, vl = (t.detach().clone().requires_grad_() for t in (q, k, v))
    ot = F.scaled_dot_product_attention(ql, kl, vl, is_causal=True)

    def tb():
        torch.autograd.grad(ot, (ql, kl, vl), do, retain_graph=True)
    res["torch_sdpa_bwd_ms"] = round(timed(tb), 4)
    res["truestar_default_ms"] = round(timed(lambda: AB._triton(q, k, v, o, do, scale, True, False)), 4)
    got = AB._triton(q, k, v, o, do, scale, True, False)
    tg = torch.autograd.grad(ot, (ql, kl, vl), do, retain_graph=True)
    ref = reference_grad(*(t.double() for t in (q, k, v, do)), scale, True)
    res["rel_err_truestar"] = [float((x.double() - r).norm() / r.norm()) for x, r in zip(got, ref)]
    res["rel_err_torch"] = [float((x.double() - r).norm() / r.norm()) for x, r in zip(tg, ref)]
    print(json.dumps(res), flush=True)
    cands = {"rowstats": [(64, 64, 4, 1), (64, 128, 8, 1), (128, 64, 8, 1), (32, 64, 4, 1)],
             "dkdv": [(32, 32, 4, 1), (32, 64, 4, 1), (64, 32, 4, 1), (64, 64, 8, 1), (32, 32, 2, 1), (64, 32, 8, 1)],
             "dq": [(32, 32, 4, 1), (64, 32, 4, 1), (32, 64, 4, 1), (64, 64, 8, 1), (32, 32, 2, 1), (64, 32, 8, 1)]}
    base = AB.configs(a.D, dt)
    order = ["rowstats", "dkdv", "dq"]
    for kern in order:
        for c in cands[kern]:
            cfg = dict(base)
            cfg[kern] = c
            try:
                per = []
                for _ in range(8):
                    ev = []
                    AB._triton(q, k, v, o, do, scale, True, False, cfg=cfg, events=ev)
                    torch.cuda.synchronize()
                    per.append([ev[i].elapsed_time(ev[i + 1]) for i in range(3)])
                per.sort(key=lambda r: sum(r))
                row = dict(zip(order, (round(x, 4) for x in per[len(per) // 2])))
                print(json.dumps({"kernel": kern, "cfg": c, **row}), flush=True)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"kernel": kern, "cfg": c, "error": type(e).__name__}), flush=True)
    if not a.sweep:
        return
    base = AB.configs(a.D, dt)
    for kern in (a.kernels or "dq,rowstats,dkdv").split(","):
        rows = []
        for bm, bn, w, ns in itertools.product((32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3)):
            cfg = dict(base)
            cfg[kern] = (bm, bn, w, ns)
            try:
                ms = timed(lambda: AB._triton(q, k, v, o, do, scale, True, False, cfg=cfg), reps=8)
            except Exception as e:  # noqa: BLE001 (out of shared memory and the like)
                rows.append({"cfg": [bm, bn, w, ns], "error": type(e).__name__})
                continue
            rows.append({"cfg": [bm, bn, w, ns], "total_ms": round(ms, 4)})
        ok = sorted([r for r in rows if "total_ms" in r], key=lambda r: r["total_ms"])
        print(json.dumps({"kernel": kern, "best": ok[:6], "failed": len(rows) - len(ok)}), flush=True)


if __name__ == "__main__":
    main()
