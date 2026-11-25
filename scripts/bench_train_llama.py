"""Training step of a small Llama style decoder: Truestar compiled against eager PyTorch and torch.compile,
with a per kernel time breakdown by category on every side.

    PYTHONPATH=src python scripts/bench_train_llama.py --preset tiny --device cpu --tools truestar
    PYTHONPATH=src python scripts/bench_train_llama.py --preset 94m --batch 8 --tools graph,eager,compile
    PYTHONPATH=src python scripts/bench_train_llama.py --preset 94m --batch 8 --breakdown

The model (written here with torch.nn, imported to the IR by Truestar): token embedding, ``layers``
pre norm blocks of RMSNorm, causal multi head attention with rotary position embedding
(``F.scaled_dot_product_attention``), RMSNorm and a SwiGLU feed forward, a final RMSNorm and an
untied output head; causal LM loss. One training step is forward, loss, backward and AdamW
(weight decay 0.01) on every side, float32 with TF32 off unless ``--tf32``.

Sides (as scripts/bench_train.py): ``truestar`` (CompiledTrainStep, params and state donated),
``graph`` (the same replayed from one CUDA graph), ``eager`` (torch.optim.AdamW(fused=True), SDPA),
``compile`` (torch.compile of forward plus loss and of the optimizer step).

``--breakdown`` prints the GPU time of one step by category for every requested side:
Truestar from ``truestar.compiler.profile.profile_program`` (each step's ops and values), torch from
``torch.profiler`` kernel records (kernel names). Categories: forward and backward matmuls,
attention forward and backward, norms, softmax and cross entropy, optimizer, elementwise and
layout, other. Host gaps are the call span minus GPU busy time.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

sys.path.insert(0, os.environ.get("TRUESTAR_SRC") or os.path.join(os.path.dirname(__file__), "..", "src"))

MiB = 2 ** 20
PRESETS = {   # d, layers, heads, ffn, vocab, seq
    "tiny": (64, 2, 4, 128, 128, 32),
    "small": (256, 2, 4, 704, 1024, 256),
    "94m": (1024, 6, 16, 2816, 8192, 512),
}


# ----------------------------------------------------------------------------- model


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.RMSNorm(d, eps=1e-5)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.mlp_norm = nn.RMSNorm(d, eps=1e-5)
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    @staticmethod
    def rope(x, cos, sin):
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * cos + rot * sin

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        hd = d // self.heads
        h = self.attn_norm(x)
        q = self.wq(h).view(b, t, self.heads, hd).transpose(1, 2)
        k = self.wk(h).view(b, t, self.heads, hd).transpose(1, 2)
        v = self.wv(h).view(b, t, self.heads, hd).transpose(1, 2)
        q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).reshape(b, t, d))
        h = self.mlp_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Llama(nn.Module):
    def __init__(self, d, layers, heads, ffn, vocab, seq):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn) for _ in range(layers)])
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.head = nn.Linear(d, vocab, bias=False)
        hd = d // heads
        inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        ang = torch.outer(torch.arange(seq, dtype=torch.float64), inv)
        ang = torch.cat([ang, ang], dim=-1)
        self.register_buffer("cos", ang.cos().float(), persistent=False)
        self.register_buffer("sin", ang.sin().float(), persistent=False)
        g = torch.Generator().manual_seed(0)
        for p in self.parameters():
            if p.dim() == 2:
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        return self.head(self.norm(x))


def lm_loss(out, y):
    return F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]), y[:, 1:].reshape(-1))


LR = 1e-3


# ----------------------------------------------------------------------------- sides


class TruestarSide:
    def __init__(self, model, tokens, device, dtype, graph, group, compile_kw):
        from truestar import ir, train as T
        from truestar.train.compiled import CompiledTrainStep
        t0 = time.perf_counter()
        ex = tokens[:2].cpu()
        prog = ir.import_module(copy.deepcopy(model).eval(), (ex,), dynamic_shapes=({0: Dim("B", max=256)},))
        self.step = T.build_train_step(prog, T.CausalLM(), T.AdamW(LR, weight_decay=0.01), params=model,
                                       param_group=group)
        self.build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.cs = CompiledTrainStep(self.step, device, dtype, graph=graph, accept_ill_conditioned=True,
                                    **compile_kw)
        self.cp = self.cs.cp
        self.compile_s = time.perf_counter() - t0
        self.x = self.y = tokens
        self.n_ops = len(self.step.program.nodes) if hasattr(self.step.program, "nodes") else None

    def __call__(self):
        return self.cs(self.x, self.y, lr=LR)[0]

    def inputs_for_profile(self):
        st = self.cs.carried_state()
        b = {self.step.batch_inputs[0]: self.x, self.step.batch_inputs[1]: self.y}
        lr = torch.tensor(LR, dtype=torch.float32, device=self.x.device)
        return [b[n] if n in b else lr if n == "lr" else st[n].clone() for n in self.cs.names]


def torch_side(model, tokens, mode=None, dtype=torch.float32):
    m = copy.deepcopy(model).to(tokens.device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01, fused=tokens.is_cuda)
    amp = dtype != torch.float32

    def fwd(x, y):
        with torch.autocast("cuda", dtype=dtype, enabled=amp):
            return lm_loss(m(x), y)

    f = fwd if mode is None else torch.compile(fwd, mode=mode)
    o = opt.step if mode is None else torch.compile(opt.step, mode=mode)

    def step():
        loss = f(tokens, tokens)
        loss.backward()
        o()
        opt.zero_grad(set_to_none=True)
        return loss
    return step, (m, opt)


# ----------------------------------------------------------------------------- breakdown


CATS = ["matmul fwd", "matmul bwd", "attention fwd", "attention bwd", "norm", "softmax / xent",
        "optimizer", "elementwise / layout", "embedding / index", "other"]


def phases_of(prog):
    """value name -> "fwd", "bwd" or "opt" from the step program's node order: forward and loss up to
    the loss value, the backward up to the first non finite check, the norm and update after it."""
    loss = prog.outputs[0].name
    names = [[o.name for o in n.outputs] for n in prog.nodes]
    li = next(i for i, ns in enumerate(names) if loss in ns)
    oi = next((i for i in range(li + 1, len(prog.nodes)) if prog.nodes[i].op in ("logical_or", "isfinite", "isnan")),
              len(prog.nodes))
    return {nm: ("fwd" if i <= li else "bwd" if i < oi else "opt") for i, ns in enumerate(names) for nm in ns}


def truestar_category(r, phase):
    ops = set(r["ops"])
    ph = [phase.get(v) for v in r["values"]]
    ph = "opt" if "opt" in ph else "bwd" if "bwd" in ph else "fwd"
    if ops & {"index", "index_select", "gather", "scatter_add", "embedding"}:
        return "embedding / index"
    if ph == "opt":
        return "optimizer"
    if r.get("shape") == "attention":
        return "attention " + ph
    if ops & {"matmul", "linear"} or r.get("shape") == "contraction":
        return "matmul " + ph
    if "softmax" in ops:
        return "attention " + ph
    if "log_softmax" in ops or "logsumexp" in ops:
        return "softmax / xent"
    if ops & {"rms_norm", "rsqrt", "mean", "sqrt"}:
        return "norm"
    if ops & {"sum", "amax"}:
        return "reduction " + ph
    return "elementwise / layout"


def truestar_breakdown(side, iters):
    from truestar.compiler.profile import profile_program
    p = profile_program(side.cp, side.inputs_for_profile(), iters=iters, warmup=1)
    phase = phases_of(side.step.program)
    by = defaultdict(float)
    count = defaultdict(int)
    for r in p["steps"]:
        c = truestar_category(r, phase)
        by[c] += r["gpu_ms"]
        count[c] += r["calls"]
    t = p["totals"]
    sig = defaultdict(lambda: [0.0, 0])      # (category, ops) -> [ms, launches]: the small kernels summed
    for r in p["steps"]:
        k = (truestar_category(r, phase), r["kind"], " + ".join(r["ops"])[:100])
        sig[k][0] += r["gpu_ms"]
        sig[k][1] += r["calls"]
    groups = [{"cat": c, "kind": kd, "ops": o, "ms": round(v[0], 3), "launches": v[1]}
              for (c, kd, o), v in sorted(sig.items(), key=lambda kv: -kv[1][0])[:45]]
    return {"by_category_ms": {k: round(v, 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
            "groups": groups,
            "launches": dict(count),
            "gpu_busy_ms": round(t["gpu_busy_ms"], 3), "call_span_ms": round(t["call_span_ms"], 3),
            "host_issue_ms": round(t["host_issue_ms"], 3), "steps_sum_ms": round(t["step_gpu_ms"], 3),
            "first_call": p["first_call"],
            "top": [{"ms": round(r["gpu_ms"], 4), "calls": r["calls"], "kind": r["kind"], "shape": r["shape"],
                     "cat": truestar_category(r, phase), "ops": " + ".join(r["ops"])[:120],
                     "tflops": r["tflop_per_s"] and round(r["tflop_per_s"], 2)}
                    for r in p["steps"][:40]]}


def torch_category(name, phase):
    n = name.lower()
    if "multi_tensor" in n or "adam" in n or "fused_adam" in n:
        return "optimizer"
    if phase == "opt":
        return "optimizer"
    if "fmha" in n or "flash" in n or "attention" in n or "efficient" in n:
        return "attention bwd" if ("bwd" in n or "backward" in n or phase == "bwd") else "attention fwd"
    if "gemm" in n or "cutlass" in n or "cublas" in n or "xmma" in n or re.search(r"\bsm\d+_", n) or "matmul" in n:
        return "matmul bwd" if phase == "bwd" else "matmul fwd"
    if "rms" in n or "norm" in n:
        return "norm"
    if "softmax" in n or "nll" in n or "cross_entropy" in n or "log_softmax" in n:
        return "softmax / xent"
    if "embedding" in n or "index" in n or "scatter" in n or "gather" in n:
        return "embedding / index"
    return "elementwise / layout"


def torch_breakdown(step, iters):
    """Kernel time per category with torch.profiler; the phase of a kernel (forward, backward,
    optimizer) is taken from record_function ranges around the three parts of the step."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
    by = defaultdict(float)
    names = defaultdict(float)
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = e.device_time if hasattr(e, "device_time") else e.cuda_time
        phase = "bwd" if _is_bwd(e) else ("opt" if _is_opt(e) else "fwd")
        c = torch_category(e.name, phase)
        by[c] += t / 1000.0 / iters
        names[(c, e.name[:90])] += t / 1000.0 / iters
    top = sorted(names.items(), key=lambda kv: -kv[1])[:30]
    return {"by_category_ms": {k: round(v, 3) for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
            "kernels_ms": round(sum(by.values()), 3),
            "top": [{"cat": c, "kernel": n, "ms": round(v, 4)} for (c, n), v in top]}


def _is_bwd(e):
    # kernels launched from autograd's engine thread (backward) or under a *Backward op
    try:
        p = e.cpu_parent
        while p is not None:
            if "Backward" in p.name or "autograd::engine" in p.name or "CompiledFunctionBackward" in p.name:
                return True
            p = p.cpu_parent
    except AttributeError:
        pass
    return False


def _is_opt(e):
    try:
        p = e.cpu_parent
        while p is not None:
            if "Optimizer" in p.name or "adam" in p.name.lower():
                return True
            p = p.cpu_parent
    except AttributeError:
        pass
    return False


# ----------------------------------------------------------------------------- timing


def time_sides(sides, rounds, reps):
    times = {k: [] for k in sides}
    for _ in range(rounds):
        for k, f in sides.items():
            for _ in range(reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                f()
                b.record()
                b.synchronize()
                times[k].append(a.elapsed_time(b))
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="94m", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--tools", default="graph,eager,compile")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--group", action="store_true")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--breakdown-iters", type=int, default=1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--unfused-attention", action="store_true",
                    help="TRUESTAR_FUSED_ATTENTION_GRAD=0: the attention backward lowered to its parts (before)")
    a = ap.parse_args()
    if a.unfused_attention:
        os.environ["TRUESTAR_FUSED_ATTENTION_GRAD"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    d, layers, heads, ffn, vocab, seq = PRESETS[a.preset]
    layers = a.layers or layers
    torch.manual_seed(0)
    model = Llama(d, layers, heads, ffn, vocab, seq)
    nparam = sum(p.numel() for p in model.parameters())
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, vocab, (a.batch, seq), generator=g).to(a.device)
    meta = {"preset": a.preset, "d": d, "layers": layers, "heads": heads, "ffn": ffn, "vocab": vocab, "seq": seq,
            "batch": a.batch, "params_m": round(nparam / 1e6, 2), "dtype": a.dtype, "tf32": a.tf32,
            "device": a.device, "unfused_attention": a.unfused_attention, "gpu": torch.cuda.get_device_name() if a.device == "cuda" else None,
            "torch": torch.__version__}
    print(json.dumps({"meta": meta}), flush=True)
    tools = a.tools.split(",")
    sides, info, out = {}, {}, []
    for side in ("truestar", "graph", "truestar_unfused", "graph_unfused"):
        if side not in tools:
            continue
        # *_unfused: the attention backward lowered to its parts (the tree before attention_grad)
        os.environ["TRUESTAR_FUSED_ATTENTION_GRAD"] = "0" if (a.unfused_attention or "unfused" in side) else "1"
        ts = TruestarSide(model, tokens, a.device, a.dtype, side.startswith("graph"), a.group, {})
        t0 = time.perf_counter()
        loss = ts()
        if a.device == "cuda":
            torch.cuda.synchronize()
        info[side] = {"build_s": round(ts.build_s, 1), "compile_s": round(ts.compile_s, 1),
                      "first_call_s": round(time.perf_counter() - t0, 1), "loss0": float(loss),
                      "conditioning": len(getattr(ts.cp, "conditioning", []) or [])}
        print(json.dumps({"side": side, **info[side]}), flush=True)
        if a.device != "cuda":
            for _ in range(3):
                t0 = time.perf_counter()
                loss = ts()
                print(json.dumps({"side": side, "cpu_step_s": round(time.perf_counter() - t0, 3),
                                  "loss": float(loss)}), flush=True)
            continue
        if side.startswith("graph"):
            for _ in range(3):
                ts()
            torch.cuda.synchronize()
        if a.breakdown and side == "truestar":
            bd = truestar_breakdown(ts, a.breakdown_iters)
            print(json.dumps({"side": side, "breakdown": bd}), flush=True)
            out.append({"side": side, "breakdown": bd})
        sides[side] = ts
    if a.device != "cuda":
        return
    dt = getattr(torch, a.dtype)
    for name, mode in (("eager", None), ("compile", "default"), ("autotune", "max-autotune")):
        if name not in tools:
            continue
        t0 = time.perf_counter()
        f, keep = torch_side(model, tokens, mode, dt)
        loss = f()
        torch.cuda.synchronize()
        info[name] = {"first_call_s": round(time.perf_counter() - t0, 1), "loss0": float(loss)}
        print(json.dumps({"side": name, **info[name]}), flush=True)
        if a.breakdown and name in ("eager", "compile"):
            bd = torch_breakdown(f, max(1, a.breakdown_iters))
            print(json.dumps({"side": name, "breakdown": bd}), flush=True)
            out.append({"side": name, "breakdown": bd})
        sides[name] = f
        info[name]["_keep"] = keep
    for f in sides.values():
        for _ in range(a.warmup):
            f()
    torch.cuda.synchronize()
    times = time_sides(sides, a.rounds, a.reps)
    rows = []
    for k, ts_ in times.items():
        row = {"side": k, "median_ms": round(statistics.median(ts_), 3), "min_ms": round(min(ts_), 3),
               "n": len(ts_), "tokens_per_s": round(a.batch * seq / (statistics.median(ts_) / 1000)),
               **{kk: vv for kk, vv in info.get(k, {}).items() if not kk.startswith("_")}}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"peak_mem_mib": round(torch.cuda.max_memory_allocated() / MiB)}), flush=True)
    if a.json:
        with open(a.json, "a") as fh:
            fh.write(json.dumps({"meta": meta, "rows": rows, "breakdowns": out}) + "\n")


if __name__ == "__main__":
    main()
