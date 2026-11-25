"""Min cut recompute and host offload against the greedy recompute pass (docs/MEMORY.md, P55).

Two training programs (forward, loss, backward in one IR program, truestar.train.loss_and_grad):
the 6 layer transformer language model of bench_memory.py (d256, batch 16, sequence 128) and
resnet18 at 64 px (batch 32, 10 classes, batch norm in eval mode, Truestar convolution kernels).
For budgets of 75%, 50% and 35% of the peak with reuse: planned and measured peak, step time, number
of recomputed or offloaded values, compile time, and whether the gradients are bitwise equal to the
program without recompute. Programs run with verify="off": on origin/main e96cd2c the LM's training
program fails its end to end float64 check before any recompute (d_matmul_59, relative 0.00754 against
0.00153), so the per value recompute and reload checks are exercised in tests/test_remat.py instead.

    PYTHONPATH=src python scripts/bench_remat.py [--models lm,resnet] [--out results/remat/remat.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
from torch.export import Dim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from bench_memory import LM, MiB, _time_all, measure  # noqa: E402
from truestar import ir  # noqa: E402
from truestar import train as T  # noqa: E402
from truestar.compiler import compile_program  # noqa: E402

MODES = ("greedy", "mincut", "offload", "mincut+offload")


def build_lm(batch=16, seq=128):
    torch.manual_seed(0)
    model = LM().eval()
    names = [n for n, _ in model.named_parameters()]
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, seq, 256),), dynamic_shapes=({0: Dim("B", max=64)},))
    gp, _, _ = T.step.loss_and_grad(prog, T.CausalLM(), names)
    x = torch.randn(batch, seq, 256, device="cuda")
    y = torch.randint(0, 512, (batch, seq), device="cuda")
    params = [dict(model.named_parameters())[n].detach().cuda() for n in names]
    return "transformer LM 6x d256, batch 16, seq 128", gp, [x, y] + params, {"B": batch}


def build_resnet(batch=32, px=64):
    import torchvision
    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=10).eval()
    names = [n for n, _ in model.named_parameters()]
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 3, px, px),), dynamic_shapes=({0: Dim("B", max=64)},))
    gp, _, _ = T.step.loss_and_grad(prog, T.CrossEntropy(), names)
    x = torch.randn(batch, 3, px, px, device="cuda")
    y = torch.randint(0, 10, (batch,), device="cuda")
    params = [dict(model.named_parameters())[n].detach().cuda() for n in names]
    return f"resnet18 {px} px, batch {batch}", gp, [x] + [y] + params, {"B": batch}


def run(builder, fracs=(0.75, 0.5, 0.35)):
    label, gp, inputs, dims = builder()
    base = compile_program(gp, "cuda", "float32", verify="off")
    mp0 = base.memory_plan(dims)
    base(inputs)
    ref = base(inputs)
    peak0, _ = measure(lambda: base(inputs))
    rows = [{"model": label, "variant": "reuse, no recompute", "planned": mp0.peak_bytes, "measured": peak0,
             "timer": "base"}]
    timers = {"base": lambda: base(inputs)}
    keep = [base]
    for frac in fracs:
        budget = mp0.resident_bytes + int(frac * mp0.peak_bytes)
        for mode in MODES:
            name = f"{mode}, budget {frac:.0%}"
            t = time.perf_counter()
            try:
                cp = compile_program(gp, "cuda", "float32", memory_budget=budget, memory_dims=dims, recompute=mode,
                                     verify="off")
            except Exception as e:  # noqa: BLE001 - an unreachable budget is a result
                rows.append({"model": label, "variant": name, "error": type(e).__name__,
                             "message": str(e).splitlines()[0][:200]})
                print(json.dumps(rows[-1]), flush=True)
                continue
            cs = time.perf_counter() - t
            mp = cp.memory_plan(dims)
            out = cp(inputs)                        # outputs compared bitwise with the program without recompute
            same = all(torch.equal(a, b) for a, b in zip(out, ref))
            del out
            cp(inputs)
            peak, _ = measure(lambda: cp(inputs))
            key = (mode, frac)
            timers[key] = (lambda c: (lambda: c(inputs)))(cp)
            keep.append(cp)
            rep = cp.recompute_report
            rows.append({"model": label, "variant": name, "planned": mp.peak_bytes, "measured": peak,
                         "budget_peak": budget - mp0.resident_bytes, "timer": key,
                         "recomputed": sum(1 for r in rep if r.get("method") != "offload"),
                         "offloaded": sum(1 for r in rep if r.get("method") == "offload"),
                         "recompute_work": sum(int(r.get("work", 0)) for r in rep),
                         "bitwise_equal": same, "compile_s": round(cs, 2)})
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "timer"}), flush=True)
    best = _time_all(timers, n=3, rounds=9)
    for r in rows:
        if "timer" in r:
            r["ms"] = round(1e3 * best[r.pop("timer")], 2)
        for k in ("planned", "measured", "budget_peak"):
            if k in r:
                r[k + "_mib"] = round(r.pop(k) / MiB, 2)
    del keep
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lm,resnet")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for m in a.models.split(","):
        rows += run({"lm": build_lm, "resnet": build_resnet}[m])
    for r in rows:
        print(json.dumps(r), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
