"""Memory planner and recompute measurements (docs/MEMORY.md).

    PYTHONPATH=src python scripts/bench_memory.py            # both parts
    PYTHONPATH=src python scripts/bench_memory.py --part plan
    PYTHONPATH=src python scripts/bench_memory.py --part recompute

Part "plan": for MLP, TransformerEncoderLayer, resnet18 and an attention block, at several batch
sizes, the planned peak of one call (before any launch) against the measured peak of a real call
(torch.cuda.max_memory_allocated and torch's requested_bytes, above what was allocated before the
call), with reuse (the planner's schedule), without reuse (the executor before the planner: every
value held to the end of the call) and PyTorch eager under no_grad.

Part "recompute": a transformer language model's training program (forward, loss, backward), its peak
with and without recomputation at budgets, step time, and torch eager with and without
torch.utils.checkpoint per layer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar import ir  # noqa: E402
from truestar.compiler import compile_program  # noqa: E402

MiB = 2 ** 20


def _stats():
    s = torch.cuda.memory_stats()
    return s.get("requested_bytes.all.current", 0), s.get("requested_bytes.all.peak", 0)


def measure(fn):
    """(peak allocated, peak requested) of one call of fn, above what existed before it."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    req0, _ = _stats()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    _, reqp = _stats()
    del out
    return peak, reqp - req0


class Attention(nn.Module):
    def __init__(self, d=256, h=8):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


def cases():
    import torchvision
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.ReLU(),
                        nn.Linear(2048, 1000), nn.Softmax(-1)).eval()
    tel = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0, batch_first=True).eval()
    rn = torchvision.models.resnet18().eval()
    att = Attention().eval()
    return [("MLP 1024-2048x2-1000", mlp, (1024,), [1, 64, 256]),
            ("TransformerEncoderLayer d256 seq128", tel, (128, 256), [1, 8, 64]),
            ("resnet18 224px", rn, (3, 224, 224), [1, 16, 64]),
            ("causal attention d256 h8 seq512", att, (512, 256), [1, 8, 32])]


def part_plan():
    rows = []
    for name, model, shape, batches in cases():
        B = Dim("B", max=max(batches))
        prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, *shape),), dynamic_shapes=({0: B},))
        cps = {"reuse": compile_program(prog, "cuda", "float32"),
               "hold": compile_program(prog, "cuda", "float32", memory_reuse=False)}
        gm = model.cuda()
        for b in batches:
            x = torch.randn(b, *shape, device="cuda")
            row = {"case": name, "batch": b}
            for k, cp in cps.items():
                t0 = time.perf_counter()
                mp = cp.memory_plan({"B": b})
                plan_s = time.perf_counter() - t0
                req = cp.memory_plan({"B": b}, rounding=False)
                cp([x])
                cp([x])                       # past the first call checks
                peak, reqp = measure(lambda: cp([x]))
                row[k] = {"planned": mp.peak_bytes, "planned_requested": req.peak_bytes,
                          "measured_allocated": peak, "measured_requested": reqp, "plan_seconds": plan_s,
                          "inplace_steps": mp.inplace_steps, "notes": mp.notes,
                          "peak_at": f"{mp.peak_step.get('name')} ({mp.peak_step.get('source')})"}
            with torch.no_grad():
                gm(x)
                peak, reqp = measure(lambda: gm(x))
            row["eager"] = {"measured_allocated": peak, "measured_requested": reqp}
            rows.append(row)
            r, h, e = row["reuse"], row["hold"], row["eager"]
            print(f"{name:38s} B={b:<4d} reuse planned {r['planned'] / MiB:9.2f} measured {r['measured_allocated'] / MiB:9.2f} "
                  f"(req {r['planned_requested'] / MiB:.2f}/{r['measured_requested'] / MiB:.2f}) | hold planned "
                  f"{h['planned'] / MiB:9.2f} measured {h['measured_allocated'] / MiB:9.2f} | eager {e['measured_allocated'] / MiB:9.2f} MiB"
                  f" | in place {r['inplace_steps']}", flush=True)
        del cps, gm
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------- recompute


class LM(nn.Module):
    def __init__(self, d=256, layers=6, vocab=512, ckpt=False):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True)
                                     for _ in range(layers)])
        self.head = nn.Linear(d, vocab)
        self.ckpt = ckpt

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False) if self.ckpt else layer(x)
        return self.head(x)


def _time_all(fns, n=5, rounds=9):
    """Fastest of ``rounds`` rounds per function, the functions interleaved within every round, so a
    burst of load from other jobs on a shared GPU hits all of them alike."""
    best = {k: float("inf") for k in fns}
    for f in fns.values():
        f()
    for _ in range(rounds):
        for k, f in fns.items():
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                f()
            torch.cuda.synchronize()
            best[k] = min(best[k], (time.perf_counter() - t) / n)
    return best


def part_recompute(batch=16, seq=128):
    from truestar import train as T
    torch.manual_seed(0)
    model = LM().eval()
    names = [n for n, _ in model.named_parameters()]
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, seq, 256),),
                            dynamic_shapes=({0: Dim("B", max=64)},))
    gp, _, _ = T.step.loss_and_grad(prog, T.CausalLM(), names)
    x = torch.randn(batch, seq, 256, device="cuda")
    y = torch.randint(0, 512, (batch, seq), device="cuda")
    params = [dict(model.named_parameters())[n].detach().cuda() for n in names]
    inputs = [x, y] + params
    dims = {"B": batch}
    rows = []
    base = compile_program(gp, "cuda", "float32")
    mp0 = base.memory_plan(dims)
    base(inputs)
    ref = base(inputs)
    peak0, _ = measure(lambda: base(inputs))
    timers = {"reuse": lambda: base(inputs)}
    hold = compile_program(gp, "cuda", "float32", memory_reuse=False)
    hold(inputs)
    hold(inputs)
    peakh, _ = measure(lambda: hold(inputs))
    rows.append({"variant": "Truestar, every value held (before the planner)", "planned": hold.memory_plan(dims).peak_bytes,
                 "measured": peakh, "timer": "hold"})
    timers["hold"] = lambda: hold(inputs)
    rows.append({"variant": "Truestar, liveness and reuse", "planned": mp0.peak_bytes, "measured": peak0, "timer": "reuse"})
    resident = mp0.resident_bytes
    for frac in (0.75, 0.5, 0.35):
        budget = resident + int(frac * mp0.peak_bytes)
        t = time.perf_counter()
        try:
            cp = compile_program(gp, "cuda", "float32", memory_budget=budget, memory_dims=dims, recompute=True)
        except Exception as e:  # noqa: BLE001 (an unreachable budget is reported, not hidden)
            rows.append({"variant": f"Truestar recompute, budget {frac:.0%} of the peak", "error": type(e).__name__,
                         "message": str(e).splitlines()[0]})
            print(rows[-1], flush=True)
            continue
        cs = time.perf_counter() - t
        mp = cp.memory_plan(dims)
        outs = cp(inputs)                      # check run: every recomputed value compared bitwise
        same = all(torch.equal(a, b) for a, b in zip(outs, ref))
        cp(inputs)
        peak, _ = measure(lambda: cp(inputs))
        timers[frac] = (lambda c: (lambda: c(inputs)))(cp)
        rows.append({"variant": f"Truestar recompute, budget {frac:.0%} of the peak", "planned": mp.peak_bytes,
                     "measured": peak, "timer": frac, "recomputed": len(cp.recompute_report),
                     "gradients_bitwise_equal_to_no_recompute": same, "compile_seconds": cs,
                     "budget_peak": budget - resident})
        print(rows[-1], flush=True)
    for ckpt in (False, True):
        m = LM(ckpt=ckpt).cuda().train()
        m.load_state_dict(model.state_dict())

        def step(m=m):
            out = m(x)
            loss = F.cross_entropy(out[:, :-1].reshape(-1, 512), y[:, 1:].reshape(-1))
            return torch.autograd.grad(loss, list(m.parameters()))
        step()
        peak, _ = measure(step)
        rows.append({"variant": "PyTorch eager" + (", torch.utils.checkpoint per layer" if ckpt else ""),
                     "measured": peak, "timer": ("eager", ckpt)})
        timers[("eager", ckpt)] = step
    best = _time_all(timers)
    for r in rows:
        if "timer" in r:
            r["ms"] = 1e3 * best[r.pop("timer")]
    for r in rows:
        print(json.dumps({k: (round(v / MiB, 2) if k in ("planned", "measured", "budget_peak") else v)
                          for k, v in r.items()}), flush=True)
    return rows


def part_donate(batch=64):
    """A training step (forward, MSE, backward, SGD with momentum) whose params and momentum are
    donated: the update writes over the old buffers instead of a second copy."""
    from truestar import train as T
    from truestar.ir.core import eval_dim, symbol
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1024, 4096), nn.Tanh(), nn.Linear(4096, 1024)).eval()
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 1024),), dynamic_shapes=({0: Dim("B", max=256)},))
    step = T.build_train_step(prog, T.MSE(), T.SGD(1e-3, momentum=0.9), params=model)
    cp = compile_program(step.program, "cuda", "float32")
    g = torch.Generator().manual_seed(1)
    base = []
    for v in step.program.inputs:
        if v.name in step.initial:
            t = step.initial[v.name].clone()
        elif v.name == "lr":
            t = torch.tensor(1e-3, dtype=torch.float64)
        else:
            t = torch.randn(tuple(eval_dim(d, {symbol("B"): batch}) for d in v.type.shape), generator=g)
        base.append(t.to("cuda", torch.float32) if t.dtype.is_floating_point else t.cuda())
    ref = cp([t.clone() for t in base])
    cp([t.clone() for t in base])
    rows = []
    for donate in (False, True):
        mp = cp.memory_plan({"B": batch}, donate=donate)
        ins = [t.clone() for t in base]
        peak, req = measure(lambda: cp(ins, donate=donate))
        ins = [t.clone() for t in base]
        same = all(torch.equal(a, b) for a, b in zip(cp(ins, donate=donate), ref))
        rows.append({"donate": donate, "planned": mp.peak_bytes, "measured": peak, "resident": mp.resident_bytes,
                     "inplace_steps": mp.inplace_steps, "bitwise_equal": same,
                     "params_and_state_bytes": sum(t.numel() * 4 for n, t in zip([v.name for v in step.program.inputs], base)
                                                    if n in step.initial)})
        print(json.dumps({k: (round(v / MiB, 2) if isinstance(v, int) and not isinstance(v, bool) and v > 4096 else v)
                          for k, v in rows[-1].items()}), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["plan", "recompute", "donate", "all"], default="all")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    if a.part in ("plan", "all"):
        res["plan"] = part_plan()
    if a.part in ("donate", "all"):
        res["donate"] = part_donate()
    if a.part in ("recompute", "all"):
        res["recompute"] = part_recompute()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
