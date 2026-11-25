"""Mixed precision training steps: Truestar compiled (mixed=True) against PyTorch AMP and torch.compile AMP.

    PYTHONPATH=src python scripts/bench_train_amp.py --model mlp --dtype bfloat16
    PYTHONPATH=src python scripts/bench_train_amp.py --model tel --dtype float16 --parity 200 --bitwise 20
    PYTHONPATH=src python scripts/bench_train_amp.py --model llama --dtype bfloat16 --tools truestar,graph,eager

One step is forward, loss, backward and SGD with momentum 0.9 (default; --opt adamw: AdamW, weight decay 0.01), lr 1e-3, on the GPU, with
float32 master weights and optimizer state on every side. Low precision matmuls, convolutions and
attention accumulate in float32 on every side.

Sides:
  truestar   CompiledTrainStep on build_train_step(amp=AutocastPolicy.cuda(dtype)), compiled with
             mixed=True (default verification: every kernel and the step end to end on the first
             call, end to end bars derived for the declared mixed precision, TS1480)
  graph      the same step replayed from one CUDA graph after its checked calls
  eager      torch.autocast(dtype) forward and loss, torch.amp.GradScaler for float16 (disabled for
             bfloat16), torch.optim.AdamW(fused=True)
  compile    the same step with the forward plus loss under torch.compile (default) and the
             optimizer step compiled, as the torch.compile docs train
  autotune   the same under torch.compile(mode="max-autotune") (CUDA graphs on)

Models (the fair benchmark sizes, docs FAIR_BENCH_2026_11): mlp 1024-4096-4096-1024 batch 512,
cross entropy; tel TransformerEncoderLayer d512 h8 ff2048 batch 32 x 256, MSE; llama two Llama
style decoder blocks (d512, 8 query and 2 KV heads, SwiGLU 1408, RMSNorm, rotary) batch 8 x 256,
causal LM loss over a 4096 vocabulary; resnet18 at 224 batch 32, cross entropy. Eval form models
(batch norm with running statistics, dropout 0) on every side, as bench_train.py.

Timing as bench_train.py: warmup, then rounds in which each side runs reps steps, each timed with
its own CUDA events, sides interleaved; median and min. ``--parity N``: N steps from the same
weights on a fixed cycle of 8 batches, Truestar against eager AMP, losses printed every 10 steps and
the largest relative difference. ``--bitwise N``: two Truestar runs of N steps, every carried value
compared bit for bit. Results are JSON lines.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

sys.path.insert(0, os.environ.get("TRUESTAR_SRC") or os.path.join(os.path.dirname(__file__), "..", "src"))

LR = 1e-3


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Block(nn.Module):
    """A Llama style decoder block: RMSNorm, grouped query attention with rotary positions, SwiGLU."""

    def __init__(self, d=512, heads=8, kv=2, ff=1408, seq=256):
        super().__init__()
        self.h, self.kv, self.hd = heads, kv, d // heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, kv * self.hd, bias=False)
        self.v = nn.Linear(d, kv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, ff, bias=False)
        self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)
        inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2).float() / self.hd))
        ang = torch.arange(seq).float()[:, None] * inv[None, :]
        self.register_buffer("cos", torch.cat([ang.cos(), ang.cos()], -1), persistent=False)
        self.register_buffer("sin", torch.cat([ang.sin(), ang.sin()], -1), persistent=False)

    def rope(self, x):
        a, b = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return x * self.cos + torch.cat([-b, a], -1) * self.sin

    def forward(self, x):
        B, S, D = x.shape
        h = self.n1(x)
        q = self.rope(self.q(h).view(B, S, self.h, self.hd).transpose(1, 2))
        k = self.rope(self.k(h).view(B, S, self.kv, self.hd).transpose(1, 2))
        v = self.v(h).view(B, S, self.kv, self.hd).transpose(1, 2)
        rep = self.h // self.kv
        k = k[:, :, None].expand(B, self.kv, rep, S, self.hd).reshape(B, self.h, S, self.hd)
        v = v[:, :, None].expand(B, self.kv, rep, S, self.hd).reshape(B, self.h, S, self.hd)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, D))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Llama(nn.Module):
    def __init__(self, d=512, layers=2, vocab=4096, seq=256):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, seq=seq) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + (x if self.down is None else self.down(x)))


class ResNet18(nn.Module):
    """The ResNet-18 layout (the Modal image has no torchvision): 7x7 stem, max pool, four stages of
    two basic blocks (64, 128, 256, 512 channels), global average pool, a 1000 way head."""

    def __init__(self, classes=1000):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        blocks, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            blocks += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.fc(self.blocks(self.stem(x)).mean((2, 3)))


def make_case(name, batch=None, small=False):
    """(model, example input, batch generator g -> (x, y), truestar loss, torch loss fn, dynamic shapes)."""
    from truestar import train as T
    if name == "mlp":
        dims = (64, 128, 128, 64) if small else (1024, 4096, 4096, 1024)
        batch = batch or (16 if small else 512)
        model = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2]), nn.GELU(),
                              nn.Linear(dims[2], dims[3]))

        def data(g):
            return torch.randn(batch, dims[0], generator=g), torch.randint(0, dims[3], (batch,), generator=g)
        return model, torch.randn(2, dims[0]), data, T.CrossEntropy(), F.cross_entropy
    if name == "tel":
        d, h, ff, s = (64, 4, 128, 16) if small else (512, 8, 2048, 256)
        batch = batch or (4 if small else 32)
        model = nn.TransformerEncoderLayer(d, h, ff, dropout=0.0, batch_first=True)

        def data(g):
            return torch.randn(batch, s, d, generator=g), torch.randn(batch, s, d, generator=g)
        return model, torch.randn(2, s, d), data, T.MSE(), F.mse_loss
    if name == "llama":
        d, s, vocab = (64, 16, 128) if small else (512, 256, 4096)
        batch = batch or (2 if small else 8)
        model = Llama(d, 2, vocab, s)

        def data(g):
            return torch.randn(batch, s, d, generator=g), torch.randint(0, vocab, (batch, s), generator=g)

        def lm_loss(out, y):
            return F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]), y[:, 1:].reshape(-1))
        return model, torch.randn(2, s, d), data, T.CausalLM(), lm_loss
    if name == "resnet18":
        model = ResNet18()
        px = 32 if small else 224
        batch = batch or (2 if small else 32)

        def data(g):
            return torch.randn(batch, 3, px, px, generator=g), torch.randint(0, 1000, (batch,), generator=g)
        return model, torch.randn(2, 3, px, px), data, T.CrossEntropy(), F.cross_entropy
    raise SystemExit(f"unknown model {name!r}")


class TruestarStep:
    def __init__(self, model, ex, loss, dtype, graph=False, device="cuda", opt="sgd"):
        from truestar import ir, train as T
        from truestar.train.compiled import CompiledTrainStep
        t0 = time.perf_counter()
        prog = ir.import_module(copy.deepcopy(model).eval(), (ex,), dynamic_shapes=({0: Dim("B", max=1024)},))
        o = T.SGD(LR, momentum=0.9) if opt == "sgd" else T.AdamW(LR, weight_decay=0.01)
        self.step = T.build_train_step(prog, loss, o, params=model,
                                       amp=T.AutocastPolicy.cuda(dtype))
        self.build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.cs = CompiledTrainStep(self.step, device, graph=graph)
        self.compile_s = time.perf_counter() - t0

    def __call__(self, x, y):
        return self.cs(x, y, lr=LR)[0]


def torch_step(model, loss_fn, dtype, mode=None, device="cuda", opt_kind="sgd"):
    m = copy.deepcopy(model).to(device).eval()
    opt = (torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, fused=device == "cuda") if opt_kind == "sgd"
           else torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01, fused=device == "cuda"))
    scaler = torch.amp.GradScaler(device, enabled=dtype == torch.float16)

    def fwd(x, y):
        with torch.autocast(device, dtype=dtype):
            out = m(x)
        return loss_fn(out.float(), y)
    f = fwd if mode is None else torch.compile(fwd, mode=mode)
    ostep = opt.step if mode is None else torch.compile(opt.step, mode=mode)

    def step(x, y):
        loss = f(x, y)
        scaler.scale(loss).backward()
        if dtype == torch.float16:
            scaler.step(opt)        # unscales, skips on inf; the compiled opt.step is used for bfloat16
            scaler.update()
        else:
            ostep()
        opt.zero_grad(set_to_none=True)
        return loss
    return step, m


def time_sides(sides, x, y, rounds, reps):
    times = {k: [] for k in sides}
    for _ in range(rounds):
        for k, f in sides.items():
            for _ in range(reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                f(x, y)
                b.record()
                b.synchronize()
                times[k].append(a.elapsed_time(b))
    return times


def autocast_lists(dtype):
    """What torch.autocast("cuda") does to the ops the policy names (checked live, printed)."""
    x = torch.randn(4, 8, device="cuda")
    out = {}
    with torch.autocast("cuda", dtype=dtype):
        out["linear"] = F.linear(x, torch.randn(8, 8, device="cuda")).dtype
        h = x.to(dtype)
        out["softmax"] = torch.softmax(h, -1).dtype
        out["log_softmax"] = torch.log_softmax(h, -1).dtype
        out["layer_norm"] = F.layer_norm(h, (8,)).dtype
        out["gelu"] = F.gelu(h).dtype
        out["silu"] = F.silu(h).dtype
        out["rsqrt"] = torch.rsqrt(h).dtype
        out["mean"] = h.mean(-1).dtype
        out["pow"] = h.pow(2).dtype
    return {k: str(v).replace("torch.", "") for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="one or more of mlp, tel, llama, resnet18 (comma separated)")
    ap.add_argument("--dtype", default="bfloat16", help="bfloat16, float16 or both (comma separated)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--tools", default="truestar,graph,eager,compile")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--parity", type=int, default=0)
    ap.add_argument("--bitwise", type=int, default=0)
    ap.add_argument("--no-extra", default="", help="models that skip --parity and --bitwise (comma separated)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    t0 = time.perf_counter()
    for name in a.model.split(","):
        for dt in a.dtype.split(","):
            try:
                run_case(a, name, dt, name not in a.no_extra.split(","))
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"model": name, "dtype": dt, "case_failed": f"{type(e).__name__}: {str(e)[:800]}"}),
                      flush=True)
            torch.cuda.empty_cache()
            print(json.dumps({"elapsed_s": round(time.perf_counter() - t0, 1)}), flush=True)


def run_case(a, model_name, dtype_name, extra):
    dtype = getattr(torch, dtype_name)
    a.model, a.dtype = model_name, dtype_name
    torch.manual_seed(0)
    model, ex, data, tloss, loss_fn = make_case(a.model, a.batch)
    g = torch.Generator().manual_seed(1)
    batches = [tuple(t.cuda() for t in data(g)) for _ in range(8)]
    x, y = batches[0]
    print(json.dumps({"autocast_cuda": autocast_lists(dtype), "gpu": torch.cuda.get_device_name(),
                      "torch": torch.__version__, "model": a.model, "dtype": a.dtype}), flush=True)
    tools = a.tools.split(",")
    sides, info = {}, {}
    for side in [t for t in ("truestar", "graph") if t in tools]:
        try:
            ts = TruestarStep(model, ex, tloss, dtype, graph=side == "graph", opt=a.opt)
            t0 = time.perf_counter()
            ts(x, y)
            torch.cuda.synchronize()
            info[side] = {"build_s": round(ts.build_s, 2), "compile_s": round(ts.compile_s, 2),
                          "first_call_s": round(time.perf_counter() - t0, 2),
                          "verify_s": round(ts.cs.cp.verify_seconds, 2),
                          "fallbacks": sum(1 for _, s in ts.cs.cp.plan.all_steps() if s.kind == "fallback"),
                          "kernels": ts.cs.cp.num_kernels,
                          "near_ties": sum(1 for b in ts.cs.cp.mixed_bounds if b and b.get("near_tie")),
                          "worst_bar_ratio": max((b["error"] / b["bar"] for b in ts.cs.cp.mixed_bounds
                                                  if b and "error" in b and not b.get("near_tie")), default=0.0)}
            if side == "graph":
                for _ in range(3):
                    ts(x, y)
                torch.cuda.synchronize()
            sides[side] = ts
        except Exception as e:  # noqa: BLE001 (a refused or failed side is a row, not a crash)
            print(json.dumps({"model": a.model, "dtype": a.dtype, "side": side,
                              "failed": f"{type(e).__name__}: {str(e)[:600]}"}), flush=True)
    for name, mode in (("eager", None), ("compile", "default"), ("autotune", "max-autotune")):
        if name not in tools:
            continue
        t0 = time.perf_counter()
        f, m = torch_step(model, loss_fn, dtype, mode, opt_kind=a.opt)
        f(x, y)
        torch.cuda.synchronize()
        info[name] = {"first_call_s": round(time.perf_counter() - t0, 2)}
        sides[name] = f
    for f in sides.values():
        for _ in range(a.warmup):
            f(x, y)
    times = time_sides(sides, x, y, a.rounds, a.reps)
    for k, t in times.items():
        print(json.dumps({"model": a.model, "dtype": a.dtype, "side": k, "median_ms": round(statistics.median(t), 3),
                          "min_ms": round(min(t), 3), "n": len(t), **info.get(k, {})}), flush=True)
    del sides
    torch.cuda.empty_cache()
    if a.parity and extra:
        ts = TruestarStep(model, ex, tloss, dtype, graph=True, opt=a.opt)
        f, m = torch_step(model, loss_fn, dtype, None, opt_kind=a.opt)
        lt, le = [], []
        for k in range(a.parity):
            bx, by = batches[k % len(batches)]
            lt.append(ts(bx, by).item())
            le.append(f(bx, by).item())
        rel = [abs(u - v) / max(abs(v), 1e-12) for u, v in zip(lt, le)]
        got = ts.cs.params()
        prel = max(((got[n].float() - p.detach().float()).norm() / p.detach().float().norm()).item()
                   for n, p in m.named_parameters())
        print(json.dumps({"model": a.model, "dtype": a.dtype, "parity_steps": a.parity,
                          "loss_truestar_every10": [round(v, 5) for v in lt[::10]],
                          "loss_eager_every10": [round(v, 5) for v in le[::10]],
                          "max_rel_loss_diff": max(rel), "mean_rel_loss_diff_last50": statistics.mean(rel[-50:]),
                          "final_param_rel_l2_diff": prel,
                          "loss_drop_truestar": lt[0] - statistics.mean(lt[-8:]),
                          "loss_drop_eager": le[0] - statistics.mean(le[-8:])}), flush=True)
        del ts
    if a.bitwise and extra:
        runs = []
        for _ in range(2):
            ts = TruestarStep(model, ex, tloss, dtype, graph=True, opt=a.opt)
            losses = [ts(*batches[k % len(batches)]).item() for k in range(a.bitwise)]
            st = {n: t.clone() for n, t in ts.cs.carried_state().items()}
            runs.append((losses, st))
            del ts
        (la, sa), (lb, sb) = runs
        same = all(torch.equal(sa[n].reshape(-1).view(torch.uint8), sb[n].reshape(-1).view(torch.uint8)) for n in sa)
        print(json.dumps({"model": a.model, "dtype": a.dtype, "bitwise_steps": a.bitwise,
                          "losses_equal": la == lb, "state_bitwise_equal": same, "values": len(sa)}), flush=True)


if __name__ == "__main__":
    main()
