"""Transformer inference: Truestar compiled against eager PyTorch and torch.compile, measured fairly.

    PYTHONPATH=src python scripts/bench_xfmr.py --case tel --prec fp32 [--rounds 5 --reps 20]
    PYTHONPATH=src python scripts/bench_xfmr.py --all --json out.json

Cases (models are defined here, random weights, nothing downloaded):

* ``tel``: ``nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True)`` at batch 64, sequence 128
  (eval, no dropout: eager runs PyTorch's fused fast path).
* ``prefill``: one Llama style block (RMSNorm, grouped query attention with rotary embeddings,
  SwiGLU MLP; width 1024, 16 query and 4 KV heads, MLP 2816) on a 512 token prompt at batch 1.
* ``decode1``, ``decode8``, ``decode32``: the same block, one new token per sequence against a
  KV cache of capacity 1024 holding 511 positions, at batch 1, 8 and 32.

Method (docs/QUALITY_BAR.md, the P38 entry; <research>/ADVERSARIAL_REVIEW.md 1.9):

* Every tool runs in the same process, interleaved round by round; each round times ``--reps``
  single calls with a CUDA event pair around each call, after a warm up. The L2 cache (48 MB) is
  flushed before every timed call by writing a 128 MB buffer outside the events, so weights and
  caches come from DRAM for every tool alike. Reported: median and minimum over all timed calls.
* Precision is pinned on both sides before anything is built: ``fp32`` sets
  ``torch.backends.cuda.matmul.allow_tf32`` and ``torch.backends.cudnn.allow_tf32`` False, ``tf32``
  sets both True; ``bf16`` and ``fp16`` cast the model and inputs. Truestar follows PyTorch's flags
  (``allow_tf32=None``), so both sides see the same setting. ``allow_fp16_reduced_precision_reduction``
  is left at PyTorch's default and printed.
* Eager runs with every SDPA backend that works for the case forced in turn (flash, efficient,
  cudnn, math); the fastest is the eager row and its backend is named. The decode cases also time
  eager with the cache sliced to the live positions (no masked keys), and keep the faster.
* torch.compile runs in ``mode="default"`` and ``mode="max-autotune"`` (which also captures CUDA
  graphs). Truestar runs with its defaults, the float64 check of the first call included in the warm
  up and not in the times; Truestar also runs replayed from CUDA graphs (``cp.graphed()``, the KV
  cache updated in place), unless ``--no-graphs``.
* GPU clocks, power and utilization are sampled by ``nvidia-smi`` during the run and reported.
* Every output is compared with a float64 run of the same module; the error is in the result.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PRECS = ("fp32", "tf32", "bf16", "fp16")
CASES = ("tel", "prefill", "decode1", "decode8", "decode32")

# ----------------------------------------------------------------------------- the models


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) + 0.1 * torch.randn(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), dim=-1)
    return x * cos + rot * sin


class Block(nn.Module):
    """One pre norm decoder block. ``mode``: "prefill" (causal over the prompt, returns its K and V),
    "decode" (writes K and V at ``pos`` with ``index_copy``, attends over the cache with a mask; value
    semantics, returns the new cache) or "decode_inplace" (the same with ``index_copy_``: eager's best
    form, returns only the output) or "decode_sliced" (in place, attends only the live positions)."""

    def __init__(self, dim=1024, heads=16, kv=4, hidden=2816, mode="prefill"):
        super().__init__()
        self.dim, self.heads, self.kv, self.hd = dim, heads, kv, dim // heads
        self.mode = mode
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, kv * self.hd, bias=False)
        self.wv = nn.Linear(dim, kv * self.hd, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def _qkv(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.norm1(x)
        q = self.wq(h).view(B, T, self.heads, self.hd).transpose(1, 2)
        k = self.wk(h).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.wv(h).view(B, T, self.kv, self.hd).transpose(1, 2)
        return rotate(q, cos, sin), rotate(k, cos, sin), v

    def _out(self, x, a):
        B, T, _ = x.shape
        x = x + self.wo(a.transpose(1, 2).reshape(B, T, self.dim))
        h = self.norm2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

    def forward(self, x, cos, sin, pos=None, kc=None, vc=None):
        q, k, v = self._qkv(x, cos, sin)
        if self.mode == "prefill":
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            return self._out(x, a), k, v
        if self.mode == "decode":
            kc = kc.index_copy(2, pos, k)
            vc = vc.index_copy(2, pos, v)
            keep = torch.arange(kc.shape[2], device=x.device) <= pos[:, None]
            a = F.scaled_dot_product_attention(q, kc, vc, attn_mask=keep, enable_gqa=True)
            return self._out(x, a), kc, vc
        kc.index_copy_(2, pos, k)
        vc.index_copy_(2, pos, v)
        if self.mode == "decode_sliced":
            n = int(pos[-1]) + 1 if not isinstance(pos, int) else pos + 1
            a = F.scaled_dot_product_attention(q, kc[:, :, :n], vc[:, :, :n], enable_gqa=True)
        else:
            keep = torch.arange(kc.shape[2], device=x.device) <= pos[:, None]
            a = F.scaled_dot_product_attention(q, kc, vc, attn_mask=keep, enable_gqa=True)
        return self._out(x, a)


def rope(positions, hd, theta=10000.0):
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    f = positions.to(torch.float64)[:, None] * inv[None]
    f = torch.cat((f, f), dim=-1)
    return f.cos(), f.sin()


CAP, LIVE = 1024, 511


def build(case: str, seed=0):
    """(module factory by mode, example inputs on the CPU in float64)."""
    torch.manual_seed(seed)
    if case == "tel":
        m = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0, batch_first=True).eval()
        return {"main": m}, (torch.randn(64, 128, 256, dtype=torch.float64),)
    base = Block(mode="prefill").eval()
    with torch.no_grad():
        for p in base.parameters():
            if p.dim() == 2:
                p.mul_(1.5)
    sd = base.state_dict()

    def make(mode):
        b = Block(mode=mode).eval()
        b.load_state_dict(sd)
        return b

    if case == "prefill":
        T = 512
        cos, sin = rope(torch.arange(T), 64)
        return {"main": make("prefill")}, (torch.randn(1, T, 1024, dtype=torch.float64), cos, sin)
    B = int(case.replace("decode", ""))
    cos, sin = rope(torch.tensor([LIVE]), 64)
    kc = torch.randn(B, 4, CAP, 64, dtype=torch.float64)
    vc = torch.randn(B, 4, CAP, 64, dtype=torch.float64)
    kc[:, :, LIVE:] = 0
    vc[:, :, LIVE:] = 0
    pos = torch.tensor([LIVE])
    return ({"main": make("decode"), "inplace": make("decode_inplace"), "sliced": make("decode_sliced")},
            (torch.randn(B, 1, 1024, dtype=torch.float64), cos, sin, pos, kc, vc))


# ----------------------------------------------------------------------------- timing


class Clocks:
    """nvidia-smi samples (SM clock, memory clock, power, utilization, memory used) during the run."""

    def __init__(self):
        self.rows = []
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "-lms", "250"], stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except OSError:
            self.proc = None
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.rows.append([float(v) for v in line.strip().split(",")])
            except ValueError:
                pass

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
        return False

    def summary(self):
        if not self.rows:
            return {}
        cols = list(zip(*self.rows))
        names = ("sm_mhz", "mem_mhz", "power_w", "util_pct", "mem_used_mib")
        return {n: {"min": min(c), "median": statistics.median(c), "max": max(c)} for n, c in zip(names, cols)}


_FLUSH = None


def flush_l2():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device="cuda")
    _FLUSH.zero_()


def time_calls(fn, reps):
    out = []
    for _ in range(reps):
        flush_l2()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e))
    return out


# ----------------------------------------------------------------------------- tools

DT = {"fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
TS_DT = {"fp32": "float32", "tf32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def pin(prec):
    tf = prec == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf
    torch.backends.cudnn.allow_tf32 = tf
    torch.backends.cudnn.benchmark = True
    return {"matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}


def dev_args(args, dt):
    return tuple(a.to("cuda", dt) if a.is_floating_point() else a.to("cuda") for a in args)


def sdpa_backends():
    from torch.nn.attention import SDPBackend
    return {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}


def first_out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def rel_err(out, ref):
    out = out.detach().to("cpu", torch.float64)
    return float((out - ref).norm() / ref.norm().clamp_min(1e-300))


def tools_for(case, prec, mods, args, graphs: bool, want):
    """name -> (callable, description). Each callable runs one call; built and warmed lazily."""
    from torch.nn.attention import sdpa_kernel
    dt = DT[prec]
    decode = case.startswith("decode")
    tools = {}
    dargs = dev_args(args, dt)

    def fresh_cache():
        return tuple(a.clone() for a in dargs)

    if "eager" in want:
        for mname in (("inplace", "sliced") if decode else ("main",)):
            m = copy.deepcopy(mods[mname]).to("cuda", dt)
            for bname, be in sdpa_backends().items():
                a = fresh_cache() if decode else dargs

                def call(m=m, a=a, be=be):
                    with torch.no_grad(), sdpa_kernel([be]):
                        return m(*a)
                try:
                    call()
                    torch.cuda.synchronize()
                except Exception:   # this backend cannot run the case
                    continue
                tools[f"eager:{bname}" + (":sliced" if mname == "sliced" else "")] = call
    for mode in ("default", "max-autotune"):
        if f"compile:{mode}" not in want and "compile" not in want:
            continue
        m = copy.deepcopy(mods["inplace" if decode else "main"]).to("cuda", dt)
        torch._dynamo.reset()
        cm = torch.compile(m, mode=mode)
        a = fresh_cache() if decode else dargs

        def call(cm=cm, a=a):
            with torch.no_grad():
                return cm(*a)
        tools[f"compile:{mode}"] = call
    if "truestar" in want:
        from truestar import ir
        from truestar.compiler import compile_program
        m = copy.deepcopy(mods["main"]).to("cpu", torch.float32)
        ex = tuple(a.to(torch.float32) if a.is_floating_point() else a for a in args)
        dyn = None
        if decode:   # exported at batch 2 (an example at batch 1 would specialize B)
            ex = tuple(a.to(torch.float32) if a.is_floating_point() else a for a in build("decode2")[1])
            B = torch.export.Dim("B", min=1, max=64)
            dyn = ({0: B}, None, None, None, {0: B}, {0: B})
        prog = ir.import_module(m, ex, dynamic_shapes=dyn, name=case)
        cp = compile_program(prog, "cuda", TS_DT[prec])
        lst = list(fresh_cache()) if decode else list(dargs)
        upd = {4: 1, 5: 2} if decode else None      # the KV cache is written in place, as eager does

        def call(cp=cp, lst=lst):
            return cp(lst, update=upd)
        tools["truestar"] = call
        tools["_truestar_cp"] = cp
        if graphs:
            g = cp.graphed(update=upd)
            glst = list(fresh_cache()) if decode else list(dargs)
            tools["truestar:graphs"] = lambda g=g, glst=glst: g(glst)
    return tools


def run_case(case, prec, rounds, reps, want, graphs=False):
    flags = pin(prec)
    mods, args = build(case)
    ref_mod = mods["main"].to("cpu", torch.float64)
    with torch.no_grad():
        ref = first_out(ref_mod(*args)).to(torch.float64)
    res = {"case": case, "prec": prec, "flags": flags, "torch": torch.__version__, "tools": {}}
    import triton
    res["triton"] = triton.__version__
    res["gpu"] = torch.cuda.get_device_name()
    t0 = time.perf_counter()
    tools = tools_for(case, prec, mods, args, graphs, want)
    cp = tools.pop("_truestar_cp", None)
    # warm up: first calls (compile, Truestar's float64 checks), then a few more
    warm = {}
    for name, fn in tools.items():
        s = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        warm[name] = time.perf_counter() - s
        res["tools"][name] = {"first_call_s": warm[name], "rel_err_vs_f64": rel_err(first_out(out), ref)}
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    res["build_s"] = time.perf_counter() - t0
    if cp is not None:
        rep = cp.report.counts()
        res["truestar_report"] = {k: rep[k] for k in ("kernel", "library", "fallback", "layout") if k in rep}
        res["truestar_steps"] = [s for s in (e.get("shape") for e in cp.report.kernels)]
    samples = {n: [] for n in tools}
    with Clocks() as clk:
        for _ in range(rounds):
            for n, fn in tools.items():
                samples[n] += time_calls(fn, reps)
    res["clocks"] = clk.summary()
    for n, xs in samples.items():
        res["tools"][n].update({"median_ms": statistics.median(xs), "min_ms": min(xs), "n": len(xs)})
    best_eager = min((n for n in tools if n.startswith("eager")), key=lambda n: res["tools"][n]["median_ms"],
                     default=None)
    res["best_eager"] = best_eager
    return res


def fmt(res):
    lines = [f"{res['case']} {res['prec']}  ({res['gpu']}, torch {res['torch']}, triton {res['triton']})",
             f"  flags {res['flags']}"]
    c = res.get("clocks") or {}
    if c:
        lines.append(f"  SM clock {c['sm_mhz']['min']:.0f} to {c['sm_mhz']['max']:.0f} MHz, "
                     f"util {c['util_pct']['min']:.0f} to {c['util_pct']['max']:.0f}%, "
                     f"mem used {c['mem_used_mib']['max']:.0f} MiB")
    base = res["tools"].get(res["best_eager"], {}).get("median_ms")
    for n, t in sorted(res["tools"].items(), key=lambda kv: kv[1].get("median_ms", 1e9)):
        sp = f"  {base / t['median_ms']:.2f}x vs best eager" if base else ""
        lines.append(f"  {n:24s} median {t['median_ms']:8.4f} ms  min {t['min_ms']:8.4f} ms  "
                     f"err {t['rel_err_vs_f64']:.1e}  first {t['first_call_s']:.2f} s{sp}")
    if "truestar_report" in res:
        lines.append(f"  truestar {res['truestar_report']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="tel", choices=CASES)
    ap.add_argument("--prec", default="fp32", choices=PRECS)
    ap.add_argument("--all", action="store_true", help="every case and precision, one process each")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tools", default="eager,compile,truestar")
    ap.add_argument("--no-graphs", dest="graphs", action="store_false",
                    help="leave out Truestar with CUDA graph replay (cp.graphed())")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    want = set(a.tools.split(","))
    if a.all:
        results = []
        for case in CASES:
            for prec in PRECS:
                cmd = [sys.executable, os.path.abspath(__file__), "--case", case, "--prec", prec,
                       "--rounds", str(a.rounds), "--reps", str(a.reps), "--tools", a.tools,
                       "--json", "-"] + ([] if a.graphs else ["--no-graphs"])
                p = subprocess.run(cmd, capture_output=True, text=True)
                try:
                    r = json.loads(p.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    r = {"case": case, "prec": prec, "error": p.stderr[-2000:]}
                results.append(r)
                print(fmt(r) if "tools" in r else f"{case} {prec}: FAILED\n{r['error']}", flush=True)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(results, f, indent=1)
        return 0
    r = run_case(a.case, a.prec, a.rounds, a.reps, want, a.graphs)
    if a.json == "-":
        print(json.dumps(r))
    else:
        print(fmt(r))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(r, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
