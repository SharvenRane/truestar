"""Measurements behind docs/TRAIN_NUMERICS.md: Muon against AdamW (compiled CPU step), Muon's
orthogonality per executor, low precision GEMM recipes in training (CUDA), logit bound tightness.

    PYTHONPATH=src python scripts/bench_trainnum.py [--steps 300] [--out results/trainnum.json]
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))

import truestar  # noqa: E402
from truestar import lpgemm as G, optim  # noqa: E402
from truestar.optim import muon as M  # noqa: E402


def muon_vs_adamw(steps, executor):
    import nn_mlp as E

    net = E.model("f32")
    loss = E.make_loss(net, "f32")
    p0 = {k: {kk: t.float() for kk, t in v.items()} for k, v in net.init(seed=0).items()}
    xs, ys = E.data(steps + 8, torch.float32)
    held_x, held_y = xs[-8:].reshape(-1, xs.shape[-1]), ys[-8:].reshape(-1)
    out = {}
    sched = lambda peak: optim.warmup_cosine(peak, warmup=steps // 10, total=steps)
    cases = {"AdamW lr 3e-3": optim.AdamW(lr=sched(3e-3), weight_decay=0.01),
             "Muon lr 0.02 (bf16 NS), AdamW 3e-3 for biases": optim.Muon(lr=sched(0.02), weight_decay=0.01, adam_lr=sched(3e-3)),
             "Muon lr 0.02 (f32 NS)": optim.Muon(lr=sched(0.02), weight_decay=0.01, adam_lr=sched(3e-3), ns_dtype="f32")}
    for name, opt in cases.items():
        step = truestar.train_step(loss, opt, clip=1.0, executor=executor)
        ts = step.init(p0)
        losses, times = [], []
        for i in range(steps):
            t0 = time.perf_counter()
            ts, r = step(ts, xs[i], ys[i])
            times.append(time.perf_counter() - t0)
            losses.append(r.loss)
        with torch.no_grad():
            logits = net(ts.params, held_x)
            hl = float(torch.nn.functional.cross_entropy(logits.float(), held_y))
            acc = float((logits.argmax(-1) == held_y).float().mean())
        out[name] = {"first_loss": losses[0], "mean_last_20": statistics.fmean(losses[-20:]), "held_out_loss": hl,
                     "held_out_acc": acc, "median_step_ms": 1e3 * statistics.median(times[3:])}
        print(name, out[name], flush=True)
    return out


def muon_rounding():
    from torch.optim._muon import _zeropower_via_newtonschulz

    torch.manual_seed(3)
    rows = []
    for shape in [(128, 512), (512, 128), (256, 256)]:
        u = torch.randn(*shape)
        r = {"shape": list(shape)}
        r["torch.optim.Muon bf16 cpu"] = M.orthogonality_report(
            u, _zeropower_via_newtonschulz(u, M.DEFAULT_COEFFICIENTS, 5, 1e-7), dtype=torch.bfloat16).as_dict()
        if torch.cuda.is_available():
            r["torch.optim.Muon bf16 cuda"] = M.orthogonality_report(
                u, _zeropower_via_newtonschulz(u.cuda(), M.DEFAULT_COEFFICIENTS, 5, 1e-7), dtype=torch.bfloat16).as_dict()
        for dt in ("bf16", "f16", "f32"):
            o = optim.Muon(ns_dtype=dt)
            r[f"truestar eager {dt}"] = M.orthogonality_report(u, o.orthogonalize(u), dtype=o.ns_torch_dtype).as_dict()
        rows.append(r)
        print(shape, {k: (round(v["rounding"], 5), round(v["estimate"], 5), round(v["intended"], 4))
                      for k, v in r.items() if k != "shape"}, flush=True)
    return rows


def lowp_training(steps):
    def make():
        torch.manual_seed(0)
        return torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(),
                                   torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 64))
    g = torch.Generator().manual_seed(0)
    teacher = torch.randn(256, 64, generator=g)
    xs = torch.randn(steps + 4, 256, 256, generator=g)
    ys = (torch.tanh(xs) @ teacher).argmax(-1)
    out = {}
    lf = torch.nn.functional.cross_entropy
    x0, y0 = xs[0], ys[0]
    for recipe in ["f32", "bf16", "fp8", "fp8_channel", "fp8_hybrid", "nvfp4_fwd"]:
        m = make().cuda()
        rec = {}
        if recipe != "f32":
            rep = G.grad_report(make().double(), G.convert(make().cuda(), recipe), lf, x0, y0)
            rec["step0"] = [{"layer": l.layer, "dw_error": l.dw_error, "dx_error": l.dx_error,
                             "fwd_loss": next(g.format_loss for g in l.gemms if g.which == "fwd"),
                             "dgrad_loss": next(g.format_loss for g in l.gemms if g.which == "dgrad"),
                             "wgrad_loss": next(g.format_loss for g in l.gemms if g.which == "wgrad"),
                             "kernel_ratio": max(g.kernel_ratio for g in l.gemms), "consistent": l.consistent}
                            for l in rep.layers]
            G.convert(m, recipe, verify=False)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
        losses, times = [], []
        for i in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad()
            loss = lf(m(xs[i].cuda()), ys[i].cuda())
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            losses.append(float(loss))
        with torch.no_grad():
            hl = float(lf(m(xs[-1].cuda()), ys[-1].cuda()))
        rec.update(first=losses[0], mean_last_20=statistics.fmean(losses[-20:]), held_out=hl,
                   median_step_ms=1e3 * statistics.median(times[3:]))
        out[recipe] = rec
        print(recipe, {k: v for k, v in rec.items() if k != "step0"}, flush=True)
        if "step0" in rec:
            for r in rec["step0"]:
                print("   ", r, flush=True)
    return out


def logit_tightness():
    from truestar import logitbound as L

    torch.manual_seed(0)
    enc = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(256, 8, 512, norm_first=True, batch_first=True),
                                      4, enable_nested_tensor=False)
    t0 = time.perf_counter()
    rep = L.check_module(enc, "fp16")
    dt = time.perf_counter() - t0
    gaps = [h.gap for h in rep.heads]
    res = {"heads": len(gaps), "gap_min": min(gaps), "gap_max": max(gaps), "seconds": dt,
           "worst_upper": max(h.upper for h in rep.heads)}
    print("logit bound", res, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lp-steps", type=int, default=200)
    ap.add_argument("--out", default="results/trainnum_2026_11_24.json")
    a = ap.parse_args()
    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    res["muon_rounding"] = muon_rounding()
    res["muon_vs_adamw_cpu"] = muon_vs_adamw(a.steps, "cpu")
    res["logit_bound"] = logit_tightness()
    if torch.cuda.is_available():
        res["lowp_training"] = lowp_training(a.lp_steps)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=str))
