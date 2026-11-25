"""Benchmark the compiled executor against PyTorch eager and torch.compile on the GPU.

    PYTHONPATH=src python scripts/bench_compiler.py                # every case, one process at a time
    PYTHONPATH=src python scripts/bench_compiler.py --cases mlp    # one case
    PYTHONPATH=src python scripts/bench_compiler.py --repeat 3 --json out.json

Cases: an MLP, a TransformerEncoderLayer and resnet18 imported through torch.export, and a native
``@truestar.function`` with a data dependent branch and a loop over a symbolic length.

Every (case, tool) pair runs in its own fresh process with cold caches for Truestar, Triton and
Inductor, never two at once (the GPU is shared). Measured per pair:

* first: time from the model in memory to the first result at batch 1 (import or decoration,
  compile, backend JIT). For Truestar the built in verification is timed and listed apart.
* batch 7 and 64: time of the first call at each new batch size, and whether anything recompiled.
* run: time of one call at batch 64 and at batch 1 once warm (CUDA synchronized): the fastest of
  seven batches of calls, because the GPU is shared and other jobs only ever add time.
* edit: one constant of the model is changed and the model is compiled again in the same process.

The native case runs both sides of its branch at batch 1 before anything else is timed, for
every tool, so a branch compiled on first use does not look like a batch size recompile.

Nothing is tuned per case. Losses are printed like wins. Each result records the GPU utilization
by other jobs right before its process started. ``--repeat N`` runs the whole matrix N times and
keeps, for every time, the fastest observation (other jobs on a shared GPU only add time) and, for
every compile count, the largest.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

CASES = ("mlp", "transformer", "resnet18", "native")
BATCHES = (1, 7, 64)     # the first call is at batch 1, then each new size in this order
SEQ = 32          # sequence length of the native RNN at the timed batch sizes


# ----------------------------------------------------------------------------- shared helpers


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    return time.perf_counter() - t, out


def run_time(fn, budget: float = 0.7) -> float:
    """Seconds per call: the fastest of seven batches of calls, after a warm up."""
    for _ in range(5):
        fn()
    sync()
    one, _ = timed(fn)
    n = max(3, min(2000, int(budget / 7 / max(one, 1e-6))))
    samples = []
    for _ in range(7):
        sync()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        samples.append((time.perf_counter() - t) / n)
    return min(samples)


def models(case: str, edited: bool = False):
    """The PyTorch module of an imported case. ``edited`` changes one constant of one layer."""
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    if case == "mlp":
        m = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(approximate="tanh"), nn.Linear(2048, 2048),
                          nn.GELU(approximate="tanh"), nn.Linear(2048, 2048), nn.ReLU(),
                          nn.Linear(2048, 1000))
        if edited:
            m[5] = nn.Hardtanh(-1.0, 1.0)          # the last activation changes; the layers do not
        return m.eval(), (1024,)
    if case == "transformer":
        m = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0, batch_first=True,
                                       layer_norm_eps=1e-4 if edited else 1e-5)
        return m.eval(), (128, 256)
    if case == "resnet18":
        import torchvision
        m = torchvision.models.resnet18()
        if edited:
            m.layer4[1].bn2.eps = 1e-3             # one batch norm constant
        return m.eval(), (3, 224, 224)
    raise ValueError(case)


def rnn_inputs(batch: int, steps: int, device: str):
    import torch
    gen = torch.Generator().manual_seed(batch * 1000 + steps)
    x = torch.randn(batch, steps, 256, generator=gen)
    wx = torch.randn(256, 512, generator=gen) * 0.05
    wh = torch.randn(512, 512, generator=gen) * 0.04
    return [t.to(device) for t in (x, wx, wh)]


# ----------------------------------------------------------------------------- Truestar worker


def truestar_worker(case: str, dtype: str, allow_tf32: bool) -> dict:
    import torch
    import truestar
    from truestar import Tensor, ir, ops
    from truestar import kernel as K
    from truestar.compiler import STATS, CompiledExecutor, compile_program

    out = {"case": case, "tool": "truestar", "dtype": dtype, "allow_tf32": allow_tf32}

    def kernel_verify(cp):
        return sum(s.ck.verify_seconds for _, s in cp.plan.all_steps() if s.kind == "kernel")

    from truestar.kernel import tune as ktune

    def first_call(cp, call):
        """Seconds of a first call, split into verification and the rest (JIT and the run).
        The launch config search time inside it is added to out["tune"]."""
        v0 = cp.verify_seconds + kernel_verify(cp)
        s0, n0 = ktune.stats.seconds, ktune.stats.searches
        t, _ = timed(call)
        v = cp.verify_seconds + kernel_verify(cp) - v0
        out.setdefault("tune", []).append({"seconds": ktune.stats.seconds - s0,
                                           "searches": ktune.stats.searches - n0})
        return t, v

    if case == "native":
        ex = CompiledExecutor("cuda", dtype, allow_tf32=allow_tf32)
        t0 = time.perf_counter()

        @truestar.function(executor=ex)
        def rnn(x: Tensor["B", "T", 256], wx: Tensor[256, 512], wh: Tensor[512, 512],
                decay: float = 0.5) -> Tensor["B", 512]:
            h = ops.zeros((x.shape[0], 512))
            for t in range(x.shape[1]):                   # a loop over a symbolic length
                h = ops.tanh(x[:, t] @ wx + h @ wh)
            if h.mean() > 0:                              # a branch on tensor data
                h = ops.relu(h)
            else:
                h = h * decay
            return h

        fn = rnn
        out["compile_s"] = time.perf_counter() - t0
        cp = ex.last
        args = {b: rnn_inputs(b, SEQ, "cuda") for b in BATCHES}
        call = lambda b=1: fn(*args[b])                    # noqa: E731

        def make_edit():                                   # the constant of the else branch changes
            fn.specialize(decay=0.25)
            return fn, ex

        edit_call = lambda f: f(*args[1], decay=0.25)      # noqa: E731
    else:
        model, shape = models(case)
        t0 = time.perf_counter()
        prog = ir.import_module(model, (torch.randn(2, *shape),),
                                dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
        out["import_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cp = compile_program(prog, "cuda", dtype, allow_tf32=allow_tf32)
        out["compile_s"] = time.perf_counter() - t0
        args = {b: [torch.randn(b, *shape, device="cuda")] for b in BATCHES}
        call = lambda b=1: cp(args[b])                     # noqa: E731

        def make_edit():
            m2, _ = models(case, edited=True)
            p2 = ir.import_module(m2, (torch.randn(2, *shape),),
                                  dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
            return compile_program(p2, "cuda", dtype, allow_tf32=allow_tf32)

        edit_call = lambda c: c(args[1])                   # noqa: E731

    stats0 = dict(STATS)
    t, v = first_call(cp, call)
    if case == "native":                                   # the other side of the branch (tanh is odd)
        other = [-args[1][0]] + args[1][1:]
        t2, v2 = first_call(cp, lambda: fn(*other))
        t, v = t + t2, v + v2
    out["first_call_s"], out["first_verify_s"] = t, v
    out["first_s"] = out.get("import_s", 0.0) + out["compile_s"] + t - v      # without verification
    out["max_error_vs_float64"] = cp.max_error
    out["run_b1_ms"] = run_time(call, 0.3) * 1e3
    out["new_batch"] = {}
    for b in BATCHES[1:]:
        before = (dict(STATS), K.stats.misses)
        t, v = first_call(cp, lambda: call(b))
        out["new_batch"][str(b)] = {"first_call_ms": t * 1e3, "verify_ms": v * 1e3,
                                    "tune_ms": out["tune"][-1]["seconds"] * 1e3,
                                    "searches": out["tune"][-1]["searches"],
                                    "truestar_compiles": sum(STATS.values()) - sum(before[0].values()),
                                    "kernels_generated": K.stats.misses - before[1]}
    out["run_ms"] = run_time(lambda: call(64)) * 1e3
    out["compiles_after_first"] = sum(STATS.values()) - sum(stats0.values())
    if case == "native":                                   # other loop lengths through the same program
        lens = {}
        for steps in (8, 48):
            a = rnn_inputs(64, steps, "cuda")
            before = sum(STATS.values())
            t, v = first_call(cp, lambda: fn(*a))
            lens[str(steps)] = {"first_call_ms": t * 1e3, "verify_ms": v * 1e3,
                                "truestar_compiles": sum(STATS.values()) - before}
        out["new_length"] = lens
    misses = K.stats.misses
    t0 = time.perf_counter()
    edited = make_edit()
    cp2 = edited[1].last if case == "native" else edited
    target = edited[0] if case == "native" else edited
    t_build = time.perf_counter() - t0
    t, v = first_call(cp2, lambda: edit_call(target))
    out["edit_s"] = t_build + t - v
    out["edit_verify_s"] = v
    out["edit_kernels_generated"] = K.stats.misses - misses
    rep = cp.report
    c = rep.counts()
    out["report"] = {"kernels": c["kernel"], "library": c["library"], "fallbacks": c["fallback"],
                     "host": c["host"], "layout": c["layout"],
                     "kernel_shapes": sorted(k["shape"] for k in rep.kernels)}
    return out


# ----------------------------------------------------------------------------- PyTorch worker


def torch_worker(case: str, tool: str) -> dict:
    import torch
    out = {"case": case, "tool": tool}
    counters = torch._dynamo.utils.counters

    def graphs():
        return int(counters["stats"]["unique_graphs"])

    if case == "native":
        def make(decay):
            def rnn(x, wx, wh):
                h = torch.zeros((x.shape[0], 512), device=x.device)
                for t in range(x.shape[1]):
                    h = torch.tanh(x[:, t] @ wx + h @ wh)
                if h.mean() > 0:
                    h = torch.relu(h)
                else:
                    h = h * decay
                return h
            return rnn

        args = {b: rnn_inputs(b, SEQ, "cuda") for b in BATCHES}
        raw, raw_edit = make(0.5), make(0.25)
        wrap = (lambda f: torch.compile(f)) if tool == "compile" else (lambda f: f)
    else:
        model, shape = models(case)
        model = model.cuda()
        args = {b: [torch.randn(b, *shape, device="cuda")] for b in BATCHES}
        raw = model
        raw_edit = None
        wrap = (lambda f: torch.compile(f)) if tool == "compile" else (lambda f: f)

    with torch.no_grad():
        t0 = time.perf_counter()
        fn = wrap(raw)
        t, _ = timed(lambda: fn(*args[1]))
        if case == "native":                               # the other side of the branch
            timed(lambda: fn(-args[1][0], *args[1][1:]))
        out["first_s"] = time.perf_counter() - t0
        out["run_b1_ms"] = run_time(lambda: fn(*args[1]), 0.3) * 1e3
        out["new_batch"] = {}
        for b in BATCHES[1:]:
            g0 = graphs()
            t, _ = timed(lambda: fn(*args[b]))
            out["new_batch"][str(b)] = {"first_call_ms": t * 1e3, "graphs_compiled": graphs() - g0}
        out["run_ms"] = run_time(lambda: fn(*args[64])) * 1e3
        if case == "native":
            lens = {}
            for steps in (8, 48):
                a = rnn_inputs(64, steps, "cuda")
                g0 = graphs()
                t, _ = timed(lambda: fn(*a))
                lens[str(steps)] = {"first_call_ms": t * 1e3, "graphs_compiled": graphs() - g0}
            out["new_length"] = lens
        out["graphs_total"] = graphs()
        if tool == "compile":
            if case != "native":
                raw_edit, _ = models(case, edited=True)
                raw_edit = raw_edit.cuda()
            t0 = time.perf_counter()
            fn2 = wrap(raw_edit)
            timed(lambda: fn2(*args[1]))
            out["edit_s"] = time.perf_counter() - t0
        try:
            out["graph_breaks"] = len(torch._dynamo.explain(raw)(*args[64]).break_reasons) if tool == "compile" else 0
        except Exception as e:  # noqa: BLE001
            out["graph_breaks"] = f"explain failed: {type(e).__name__}"
    return out


# ----------------------------------------------------------------------------- first call at a size

FIRST_CASES = {            # case: (kind, first call shape) for --first-call
    "mlp_b64": ("mlp", 64),
    "transformer_b64": ("transformer", 64),
    "resnet18_b64": ("resnet18", 64),
    "attention_s1024": ("attention", (1, 8, 1024, 64)),
    "attention_s8192": ("attention", (1, 2, 8192, 128)),
}


def attention_program(heads: int, dim: int):
    """Causal attention over [B, H, S, D] with symbolic B and S (the IR attention op)."""
    import torch
    from truestar.ir.core import Builder
    b = Builder("attention")
    q = b.input("q", torch.float32, ("B", heads, "S", dim))
    k = b.input("k", torch.float32, ("B", heads, "S", dim))
    v = b.input("v", torch.float32, ("B", heads, "S", dim))
    return b.build([b.op("attention", q, k, v, None, scale=None, is_causal=True)])


def first_call_worker(case: str, dtype: str) -> dict:
    """Compile in a fresh process and time the first call at one size, split into compile, the
    kernel layer's checks, the region checks, the end to end check and the rest (JIT and the run)."""
    import torch
    from truestar import ir
    from truestar.compiler import compile_program
    kind, size = FIRST_CASES[case]
    out = {"case": case, "dtype": dtype, "verify_device": os.environ.get("TRUESTAR_VERIFY_DEVICE", "auto")}
    gen = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    if kind == "attention":
        prog = attention_program(size[1], size[3])
        args = [torch.randn(*size, generator=gen).cuda() for _ in range(3)]
    else:
        model, shape = models(kind)
        prog = ir.import_module(model, (torch.randn(2, *shape),),
                                dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
        args = [torch.randn(size, *shape, generator=gen).cuda()]
    out["import_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cp = compile_program(prog, "cuda", dtype)
    out["compile_s"] = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    t, _ = timed(lambda: cp(args))
    out["first_call_s"] = t
    out["kernel_verify_s"] = cp.kernel_verify_seconds
    out["region_verify_s"] = cp.region_verify_seconds
    out["end_to_end_verify_s"] = cp.end_to_end_verify_seconds
    out["rest_s"] = t - out["kernel_verify_s"] - out["region_verify_s"] - out["end_to_end_verify_s"]
    out["max_error_vs_float64"] = cp.max_error
    out["max_region_error"] = cp.max_region_error
    out["end_to_end_device"] = getattr(cp, "end_to_end_device", None)
    out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return out


def first_call_table(results) -> str:
    rows = ["| case | first call (s) | kernel checks (s) | region checks (s) | end to end check (s) | rest (s) | "
            "max error vs float64 | peak GPU memory (MB) |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            rows.append(f"| {r['case']} | failed: {r['error'].strip().splitlines()[-1][:120]} | | | | | | |")
            continue
        rows.append(f"| {r['case']} | {fmt(r['first_call_s'])} | {fmt(r['kernel_verify_s'])} | "
                    f"{fmt(r['region_verify_s'])} | {fmt(r['end_to_end_verify_s'])} | {fmt(r['rest_s'])} | "
                    f"{fmt(r['max_error_vs_float64'])} | {fmt(r['peak_gpu_mb'])} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- driver


def worker(case: str, tool: str, dtype: str, allow_tf32: bool) -> dict:
    import torch
    out = truestar_worker(case, dtype, allow_tf32) if tool == "truestar" else torch_worker(case, tool)
    out["gpu"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__
    return out


def gpu_busy() -> str:
    """GPU utilization and memory in use right now (other jobs: this script has no process yet)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return q.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def spawn(case: str, tool: str, dtype: str = "float32", allow_tf32: bool = False,
          first_call: bool = False, cache: str = None) -> dict:
    """One worker process. ``cache`` reuses the cache directories of an earlier process (a warm
    start); by default every process starts with empty caches."""
    env = dict(os.environ)
    busy = gpu_busy()
    cold = cache or tempfile.mkdtemp(prefix="truestar_bench_")
    env["TRUESTAR_CACHE_DIR"] = os.path.join(cold, "truestar")
    env["TRITON_CACHE_DIR"] = os.path.join(cold, "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cold, "inductor")
    env["NUMBA_CACHE_DIR"] = os.path.join(cold, "numba")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", case, tool, "--dtype", dtype]
    if allow_tf32:
        cmd.append("--allow-tf32")
    if first_call:
        cmd.append("--first-call")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or not lines:
        return {"case": case, "tool": tool, "error": (p.stderr or p.stdout)[-1500:]}
    out = json.loads(lines[-1][7:])
    out["process_s"] = time.perf_counter() - t0
    out["gpu_before"] = busy
    out["cache_dir"] = cold
    return out


COUNT_FIELDS = ("truestar_compiles", "kernels_generated", "graphs_compiled", "graphs_total", "compiles_after_first",
                "edit_kernels_generated", "graph_breaks", "searches")


def merge(a, b):
    """Combine two observations of the same measurement: fastest time, largest count."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k not in a:
                out[k] = v
            elif k in COUNT_FIELDS and isinstance(v, (int, float)) and isinstance(a[k], (int, float)):
                out[k] = max(a[k], v)
            elif k == "max_error_vs_float64":
                out[k] = max(a[k], v)
            elif k == "gpu_before":
                out[k] = f"{a[k]}; {v}"
            else:
                out[k] = merge(a[k], v)
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return min(a, b)
    return a


def fmt(v, unit=""):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{v:.3g}{unit}"


def table(results) -> str:
    rows = ["| case | tool | first result, batch 1 (s) | run, batch 64 (ms) | run, batch 1 (ms) | "
            "first call at batch 7 / 64 (ms) | recompiles for batch 7, 64 | after an edit (s) | notes |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            rows.append(f"| {r['case']} | {r['tool']} | failed | | | | | | {r['error'].strip().splitlines()[-1][:120]} |")
            continue
        nb = r.get("new_batch", {})
        calls = " / ".join(fmt(nb[b]["first_call_ms"]) for b in ("7", "64") if b in nb)
        if r["tool"].startswith("truestar"):
            name = "Truestar" + (" (TF32 allowed)" if r.get("allow_tf32") else "") + (
                ", warm cache" if "warm" in r["tool"] else "")
            rec = str(sum(nb[b]["truestar_compiles"] for b in nb))
            ver = sum(nb[b]["verify_ms"] for b in nb)
            rep = r["report"]
            tune_ms = " / ".join(fmt(nb[b].get("tune_ms")) for b in ("7", "64") if b in nb)
            notes = (f"launch config search at batch 7 / 64: {tune_ms} ms; "
                     f"verification on first call {fmt(r['first_verify_s'], ' s')}, at batch 7 and 64 {fmt(ver, ' ms')} "
                     f"(inside the times to the left); {rep['kernels']} kernels, {rep['library']} library, "
                     f"{rep['fallbacks']} fallbacks; max error vs float64 {fmt(r['max_error_vs_float64'])}")
            if "new_length" in r:
                notes += "; new loop lengths 8 and 48: " + ", ".join(
                    f"{fmt(v['first_call_ms'], ' ms')} ({v['truestar_compiles']} compiles)"
                    for v in r["new_length"].values())
            edit = fmt(r.get("edit_s")) + f" ({r.get('edit_kernels_generated')} kernels generated)"
        else:
            name = "torch.compile" if r["tool"] == "compile" else "PyTorch eager"
            rec = str(sum(nb[b]["graphs_compiled"] for b in nb)) if r["tool"] == "compile" else "none"
            notes = f"graph breaks {r.get('graph_breaks')}" if r["tool"] == "compile" else ""
            if "new_length" in r and r["tool"] == "compile":
                notes += "; new loop lengths 8 and 48: " + ", ".join(
                    f"{fmt(v['first_call_ms'], ' ms')} ({v['graphs_compiled']} graphs)" for v in r["new_length"].values())
            edit = fmt(r.get("edit_s")) if r["tool"] == "compile" else "none"
        rows.append(f"| {r['case']} | {name} | {fmt(r['first_s'])} | {fmt(r['run_ms'])} | {fmt(r['run_b1_ms'])} | "
                    f"{calls} | {rec} | {edit} | {notes} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("CASE", "TOOL"))
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--allow-tf32", action="store_true")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--tools", nargs="*", default=["truestar", "eager", "compile"])
    ap.add_argument("--json")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--first-call", action="store_true",
                    help="time compile plus the first call at one size per case, split by check "
                         f"(cases: {', '.join(FIRST_CASES)}); each repeat is a fresh process, median reported")
    ap.add_argument("--warm", action="store_true",
                    help="run each Truestar case a second time on the caches of the first (warm start)")
    a = ap.parse_args()
    if a.worker and a.first_call:
        print("RESULT " + json.dumps(first_call_worker(a.worker[0], a.dtype)))
        return
    if a.first_call:
        cases = [c for c in a.cases if c in FIRST_CASES] or list(FIRST_CASES)
        results = []
        for case in cases:
            runs = []
            for _ in range(a.repeat):
                r = spawn(case, "truestar", a.dtype, first_call=True)
                print(json.dumps(r), flush=True)
                runs.append(r)
            ok = [r for r in runs if "error" not in r]
            if not ok:
                results.append(runs[-1])
                continue
            med = dict(ok[0])
            for k, v in ok[0].items():
                if isinstance(v, float):
                    med[k] = statistics.median(r[k] for r in ok)
            med["runs"] = len(ok)
            results.append(med)
        print()
        print(f"median of {a.repeat} fresh processes per case")
        print(first_call_table(results))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        return
    if a.worker:
        print("RESULT " + json.dumps(worker(a.worker[0], a.worker[1], a.dtype, a.allow_tf32)))
        return
    best = {}
    for _ in range(a.repeat):
        for case in a.cases:
            for tool in a.tools:
                jobs = [(case, tool, False)]
                if tool == "truestar" and case == "resnet18":   # PyTorch's own default for convolutions is TF32
                    jobs.append((case, tool, True))
                if tool == "truestar" and a.warm:
                    jobs += [(case, "truestar (warm cache)", j[2]) for j in list(jobs)]
                cache = {}
                for job in jobs:
                    if job[1] == "truestar (warm cache)":
                        r = spawn(job[0], "truestar", allow_tf32=job[2], cache=cache.get(job[2]))
                        r["tool"] = job[1]
                    else:
                        r = spawn(job[0], job[1], allow_tf32=job[2])
                        cache[job[2]] = r.get("cache_dir")
                    print(json.dumps(r), flush=True)
                    if "error" in r:
                        best.setdefault(job, r)
                    elif job not in best or "error" in best[job]:
                        best[job] = r
                    else:
                        best[job] = merge(best[job], r)
    results = list(best.values())
    print()
    if a.repeat > 1:
        print(f"fastest of {a.repeat} runs per time, largest compile count")
    print(table(results))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
