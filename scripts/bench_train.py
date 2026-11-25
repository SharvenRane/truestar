"""Training step speed and peak memory: Truestar compiled against eager PyTorch and torch.compile.

    PYTHONPATH=src python scripts/bench_train.py --model mlp --batch 64 --opt sgd
    PYTHONPATH=src python scripts/bench_train.py --model lm --batch 8 --opt adamw --tools truestar,eager
    PYTHONPATH=src python scripts/bench_train.py --model resnet18 --batch 16 --profile

One training step is forward, loss, backward and the optimizer update (SGD with momentum 0.9 or
AdamW), on the GPU, float32. Every side runs the same model, the same batch and the same
optimizer settings, with the precision flags pinned and printed (IEEE float32 unless --tf32).

Sides:
  truestar      truestar.train.build_train_step compiled by truestar.compiler, params and state
                donated each step (one executor call per step), default verification (every
                kernel checked against float64 on its first call)
  eager         model(x), loss.backward(), torch.optim with fused=True (the fastest eager optimizer)
  compile       the same step function under torch.compile (default mode)
  autotune      the same under torch.compile(mode="max-autotune") (CUDA graphs on)
  graph         the Truestar step replayed from one CUDA graph (compare with autotune: graphs on
                both sides); no peak is reported for graph replays (their pool is reserved at capture)

Models run in eval form on every side (batch norm uses running statistics), because the IR imports
inference graphs (docs/TRAINING.md, Limits); dropout is 0.

Timing (docs/QUALITY_BAR.md never introduce #4, ADVERSARIAL_REVIEW 1.9): every side warms up, then
``--rounds`` rounds in which each side runs ``--reps`` steps, each step timed with its own pair of
CUDA events; sides interleave inside a round so load from other jobs hits all of them alike. Median
and min of all timed steps. Peak memory is torch.cuda.max_memory_allocated during one step above what
was allocated before it (params, optimizer state and the batch are resident on every side). GPU
clocks are sampled with nvidia-smi during the timed rounds.

``--profile`` prints `truestar profile` style per step times of the Truestar step program instead.
Results are printed as one JSON line per side.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

# TRUESTAR_SRC: another source tree to measure (the before rows); default this tree
sys.path.insert(0, os.environ.get("TRUESTAR_SRC") or os.path.join(os.path.dirname(__file__), "..", "src"))

MiB = 2 ** 20


# ----------------------------------------------------------------------------- models


class LM(nn.Module):
    def __init__(self, d=256, layers=6, vocab=512):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True)
                                     for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def make_case(name: str, batch: int):
    """(model, example input for import, batch x, batch y, truestar loss, torch loss fn)."""
    from truestar import train as T
    g = torch.Generator().manual_seed(1)
    if name == "mlp":
        model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.GELU(),
                              nn.Linear(2048, 1000))
        x = torch.randn(batch, 1024, generator=g)
        y = torch.randint(0, 1000, (batch,), generator=g)
        ex = torch.randn(2, 1024)
        return model, ex, x, y, T.CrossEntropy(), lambda out, y: F.cross_entropy(out, y)
    if name == "lm":
        model = LM()
        x = torch.randn(batch, 128, 256, generator=g)
        y = torch.randint(0, 512, (batch, 128), generator=g)
        ex = torch.randn(2, 128, 256)

        def lm_loss(out, y):
            return F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]), y[:, 1:].reshape(-1))
        return model, ex, x, y, T.CausalLM(), lm_loss
    if name == "resnet18":
        import torchvision
        model = torchvision.models.resnet18(num_classes=1000)
        x = torch.randn(batch, 3, 224, 224, generator=g)
        y = torch.randint(0, 1000, (batch,), generator=g)
        ex = torch.randn(2, 3, 224, 224)
        return model, ex, x, y, T.CrossEntropy(), lambda out, y: F.cross_entropy(out, y)
    raise SystemExit(f"unknown model {name!r}")


LR = 1e-3


def truestar_opt(kind):
    from truestar import train as T
    if kind == "sgd":
        return T.SGD(LR, momentum=0.9)
    return T.AdamW(LR, weight_decay=0.01)


def torch_opt(kind, params, fused=True):
    if kind == "sgd":
        return torch.optim.SGD(params, lr=LR, momentum=0.9, fused=fused)
    return torch.optim.AdamW(params, lr=LR, weight_decay=0.01, fused=fused)


# ----------------------------------------------------------------------------- sides


class TruestarStep:
    """Build the one program step and run it through truestar.train.CompiledTrainStep (params and
    state donated every step; with graph=True replayed from one CUDA graph after its checked calls)."""

    def __init__(self, model, ex, x, y, loss, opt_kind, graph=False, group=False):
        from truestar import ir, train as T
        dyn = ({0: Dim("B", max=512)},)
        t0 = time.perf_counter()
        prog = ir.import_module(copy.deepcopy(model).eval(), (ex,), dynamic_shapes=dyn)
        kw = {"param_group": True} if group else {}
        self.step = T.build_train_step(prog, loss, truestar_opt(opt_kind), params=model, **kw)
        self.build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        try:
            from truestar.train.compiled import CompiledTrainStep
        except ImportError:      # a tree from before CompiledTrainStep (the before rows): same calls, by hand
            CompiledTrainStep = _OldStep
        # AdamW's first step is ill conditioned in float32 (each update divides by |g|): opt in to the
        # end to end allowance against an independent float32 evaluation (TS782, docs/TRAINING.md)
        extra = {"accept_ill_conditioned": True} if opt_kind == "adamw" and CompiledTrainStep is not _OldStep else {}
        self.cs = CompiledTrainStep(self.step, "cuda", "float32", graph=graph, **extra)
        self.cp = self.cs.cp
        self.compile_s = time.perf_counter() - t0
        self.x, self.y = x, y

    def __call__(self):
        return self.cs(self.x, self.y, lr=LR)[0]

    def inputs_for_profile(self):
        """A fresh, non donated input list (the profiler calls the program many times)."""
        st = self.cs.carried_state()
        b = {self.step.batch_inputs[0]: self.x, self.step.batch_inputs[1]: self.y}
        lr = torch.tensor(LR, dtype=torch.float32, device="cuda")
        return [b[n] if n in b else lr if n == "lr" else st[n].clone() for n in self.cs.names]


class _OldStep:
    """What CompiledTrainStep does without a graph, for source trees that predate it."""

    def __init__(self, step, device, dtype, graph=False, **_):
        from truestar.compiler import compile_program
        assert not graph, "no CUDA graph training step in this tree"
        self.step, self.cp = step, compile_program(step.program, device, dtype)
        self.names = [v.name for v in step.program.inputs]
        self.carried = list(step.carried)
        self.state = {n: step.initial[n].detach().to(device, torch.float32 if step.initial[n].dtype.is_floating_point
                                                      else step.initial[n].dtype).clone() for n in self.carried}

    def carried_state(self):
        return self.state

    def __call__(self, x, y, lr):
        b = {self.step.batch_inputs[0]: x, self.step.batch_inputs[1]: y}
        lr_t = torch.tensor(lr, dtype=torch.float32, device="cuda")
        ins = [b[n] if n in b else lr_t if n == "lr" else self.state.pop(n) for n in self.names]
        outs = self.cp(ins, donate=True)
        for n, t in zip(self.carried, outs[3:]):
            self.state[n] = t
        return outs[:3]


def torch_step(model, x, y, loss_fn, opt_kind, mode=None):
    m = copy.deepcopy(model).cuda().eval()
    opt = torch_opt(opt_kind, m.parameters())

    def step():
        out = m(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    if mode is None:
        return step, m, opt
    # compile the forward plus loss (backward through AOT autograd) and the optimizer step: the
    # documented way to compile a training step (a graph break at backward is expected)
    fwd = torch.compile(lambda x, y: loss_fn(m(x), y), mode=mode)
    opt_step = torch.compile(opt.step, mode=mode)

    def cstep():
        loss = fwd(x, y)
        loss.backward()
        opt_step()
        opt.zero_grad(set_to_none=True)
        return loss
    return cstep, m, opt


# ----------------------------------------------------------------------------- measurement


class ClockSampler:
    def __init__(self, period=0.5):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                self.samples.append([float(v) for v in out.stdout.strip().split(",")])
            except Exception:  # noqa: BLE001 (a missing sample is not a result)
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def summary(self):
        if not self.samples:
            return None
        cols = list(zip(*self.samples))
        return {k: [min(c), statistics.median(c), max(c)] for k, c in
                zip(("sm_mhz", "mem_mhz", "power_w", "util_pct"), cols)}


def peak_of(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


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


def flags():
    return {"matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "cudnn.deterministic": torch.backends.cudnn.deterministic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp", "lm", "resnet18"])
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--tools", default="truestar,eager,compile,autotune")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--group", action="store_true", help="Truestar sides with param_group=True")
    ap.add_argument("--profile-iters", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    model, ex, x, y, tloss, loss_fn = make_case(a.model, a.batch)
    x, y = x.cuda(), y.cuda()
    tools = a.tools.split(",")
    sides, info = {}, {}
    keep = []
    for side in [t for t in ("truestar", "graph") if t in tools] or (["truestar"] if a.profile else []):
        ts = TruestarStep(model, ex, x, y, tloss, a.opt, graph=side == "graph", group=a.group)
        t0 = time.perf_counter()
        ts()
        torch.cuda.synchronize()
        info[side] = {"build_s": round(ts.build_s, 2), "compile_s": round(ts.compile_s, 2),
                      "first_call_s": round(time.perf_counter() - t0, 2)}
        if a.profile:
            from truestar.compiler.profile import profile_program, to_text
            p = profile_program(ts.cp, ts.inputs_for_profile(), iters=a.profile_iters, warmup=2)
            print(to_text(p, top=40))
            return
        if side == "graph":
            for _ in range(3):          # checked call, warmup, capture and its verified replay
                ts()
            torch.cuda.synchronize()
            info[side]["copied_back"] = len(ts.cs.copied_back)
        sides[side] = ts
        keep.append(ts)
    for name, mode in (("eager", None), ("compile", "default"), ("autotune", "max-autotune")):
        if name not in tools:
            continue
        t0 = time.perf_counter()
        f, m, opt = torch_step(model, x, y, loss_fn, a.opt, mode)
        f()
        torch.cuda.synchronize()
        info[name] = {"first_call_s": round(time.perf_counter() - t0, 2)}
        sides[name] = f
        keep.append((m, opt))
    for f in sides.values():
        for _ in range(a.warmup):
            f()
    peaks = {k: peak_of(f) for k, f in sides.items()}
    # CUDA graph replays (graph, and torch.compile max-autotune with its CUDA graph trees) allocate
    # nothing: their memory is a private pool reserved at capture, which max_memory_allocated does not
    # see. No number is reported for them; the graph replays the truestar side's exact schedule.
    for k in ("graph", "autotune"):
        if k in peaks:
            peaks[k] = None
    with ClockSampler() as cs:
        times = time_sides(sides, a.rounds, a.reps)
    rows = []
    for k, ts_ in times.items():
        row = {"model": a.model, "batch": a.batch, "opt": a.opt, "tf32": a.tf32, "group": a.group, "side": k,
               "median_ms": round(statistics.median(ts_), 3), "min_ms": round(min(ts_), 3),
               "max_ms": round(max(ts_), 3), "n": len(ts_), "peak_mib": None if peaks[k] is None else round(peaks[k] / MiB, 2), **info.get(k, {})}
        rows.append(row)
        print(json.dumps(row), flush=True)
    meta = {"flags": flags(), "clocks[min,median,max]": cs.summary(), "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__}
    print(json.dumps(meta), flush=True)
    if a.json:
        with open(a.json, "a") as fh:
            for r in rows:
                fh.write(json.dumps({**r, **meta}) + "\n")


if __name__ == "__main__":
    main()
