"""Overhead of sentinels (compiled in runtime checks) against torch.autograd.detect_anomaly and against no checks.

    PYTHONPATH=src python scripts/bench_sentinels.py                 # every model, one process each
    PYTHONPATH=src python scripts/bench_sentinels.py --model mlp     # one model in this process
    PYTHONPATH=src python scripts/bench_sentinels.py --faults        # each check on an injected fault

Times are per call, from N calls followed by one synchronize (so a host read per call shows), the
fastest of several rounds because the GPU is shared. Truestar runs with verification after the
first call off (the first call is excluded), like the PyTorch rows run without any checking.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("TRUESTAR_VERIFY_GPU_MB", "1024")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from truestar import Tensor, ops  # noqa: E402  (annotations of the native functions below)


def sync():
    torch.cuda.synchronize()


def per_call(fn, budget=0.25, rounds=5):
    for _ in range(3):
        fn()
    sync()
    t = time.perf_counter()
    fn()
    sync()
    one = max(time.perf_counter() - t, 1e-5)
    n = max(3, min(200, int(budget / one)))
    best = []
    for _ in range(rounds):
        sync()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        best.append((time.perf_counter() - t) / n * 1000.0)
    return min(best)


def gpu_busy(fn, n=5):
    """GPU time of one call with the GPU kept busy while the host issues it (a sleep kernel queued
    first, as truestar profile does): the call's own GPU cost, without host time."""
    from truestar.compiler.profile import _sleep_cycles
    fn()
    sync()
    t = time.perf_counter()
    fn()
    host = (time.perf_counter() - t) * 1000.0
    sync()
    cyc = _sleep_cycles(3 * host + 0.5)
    best = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(cyc)
        a.record()
        fn()
        b.record()
        sync()
        best.append(a.elapsed_time(b))
    return min(best)


def build(name):
    torch.manual_seed(0)
    if name == "mlp":
        m = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.ReLU(),
                          nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 1000))
        x = torch.randn(64, 1024)
    elif name == "transformer":
        m = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0, batch_first=True)
        x = torch.randn(64, 128, 256)
    else:
        import torchvision
        m = torchvision.models.resnet18()
        x = torch.randn(16, 3, 224, 224)
    return m.eval(), x


def run_model(name, rounds=7):
    """Every variant is built first, then timed in turn in each round (the GPU is shared with other
    jobs, so interleaving keeps them under the same load); a variant's time is its fastest round."""
    import warnings

    from truestar import ir
    from truestar.compiler import compile_program
    from truestar.ir.autodiff import grad
    warnings.filterwarnings("ignore")

    m, x = build(name)
    xd = x.cuda()
    prog = ir.import_module(m, (x,))
    variants = []      # (tool, mode, variant, fn, baseline variant)

    # PyTorch eager: forward with autograd recording (what detect_anomaly instruments) and a step
    md = m.cuda()
    params = list(md.parameters())

    def fwd():
        return md(xd)

    def step():
        for p in params:
            p.grad = None
        md(xd).float().sum().backward()

    def anomaly(f):
        def g():
            with torch.autograd.detect_anomaly(check_nan=True):
                f()
        return g

    hooks = []

    def hooked():
        if not hooks:
            for sub in md.modules():
                hooks.append(sub.register_forward_hook(_finite_hook))
        try:
            return md(xd)
        finally:
            for h in hooks:
                h.remove()
            hooks.clear()

    variants += [("torch eager", "forward", "plain", fwd, None),
                 ("torch eager", "forward", "detect_anomaly", anomaly(fwd), "plain"),
                 ("torch eager", "forward", "isfinite hook per module", hooked, "plain"),
                 ("torch eager", "train step", "plain", step, None),
                 ("torch eager", "train step", "detect_anomaly", anomaly(step), "plain")]
    keep = []
    for dtype, names in (("float32", ["off", "off + one host read", "nan", "inf", "div", "nan,div", "all",
                                      "nan,div deferred", "all deferred"]),
                         ("float16", ["off", "overflow", "all", "all deferred"])):
        for v in names:
            chk = None if v.startswith("off") else v.split()[0]
            cp = compile_program(prog, "cuda", dtype, sentinels=chk,
                                 sentinel_read="deferred" if v.endswith("deferred") else "call")
            cp([xd])
            cp.verify = "off"
            keep.append(cp)
            if v == "off + one host read":
                fn = (lambda c: (lambda: bool(c([xd])[0].reshape(-1)[0] != 0)))(cp)
            else:
                fn = (lambda c: (lambda: c([xd])))(cp)
            variants.append(("truestar", f"forward {dtype}", v, fn, None if v == "off" else "off"))
    # Truestar compiled training step: the forward and the gradient of every parameter. Built with
    # verification off (see the first_call_verified field: the float32 check of some backward sums is
    # too strict for sums that cancel, a separate issue).
    gp = grad(prog, wrt=[p.name for p in prog.params], seed="ones")
    first_ok = None
    os.environ["TRUESTAR_VERIFY"] = "0"
    for v in ("off", "nan,div", "all", "nan,div deferred"):
        cp = compile_program(gp, "cuda", "float32", sentinels=None if v == "off" else v.split()[0], verify="off",
                             sentinel_read="deferred" if v.endswith("deferred") else "call")
        cp([xd])
        keep.append(cp)
        variants.append(("truestar", "train step float32", v, (lambda c: (lambda: c([xd])))(cp),
                         None if v == "off" else "off"))
    best, gbest = {}, {}
    for r in range(rounds):
        for tool, mode, v, fn, _ in variants:
            ms = per_call(fn, budget=0.15, rounds=1)
            k = (tool, mode, v)
            best[k] = min(best.get(k, ms), ms)
            g = gpu_busy(fn, 3)
            gbest[k] = min(gbest.get(k, g), g)
            print(json.dumps({"round": r, "tool": tool, "mode": mode, "variant": v, "ms": ms}), file=sys.stderr, flush=True)
    os.environ.pop("TRUESTAR_VERIFY")
    try:
        compile_program(gp, "cuda", "float32")([xd])
        first_ok = True
    except Exception as e:  # noqa: BLE001
        first_ok = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    rows = []
    for tool, mode, v, _, base in variants:
        r = {"tool": tool, "mode": mode, "variant": v, "ms": best[(tool, mode, v)], "gpu_ms": gbest[(tool, mode, v)]}
        if base:
            r["overhead_pct"] = 100.0 * (r["ms"] / best[(tool, mode, base)] - 1.0)
            r["gpu_overhead_pct"] = 100.0 * (r["gpu_ms"] / gbest[(tool, mode, base)] - 1.0)
        rows.append(r)
    return {"model": name, "batch": list(x.shape), "rows": rows, "rounds": rounds,
            "grad_first_call_verified": first_ok}


def _finite_hook(mod, inp, o):
    if isinstance(o, torch.Tensor) and not bool(torch.isfinite(o).all()):
        raise RuntimeError(f"non finite output in {type(mod).__name__}")


def faults():
    """Each class caught on an injected fault, against what detect_anomaly says on the same fault."""
    import truestar
    from truestar.compiler import SentinelError, CompiledExecutor
    res = []

    def show(name, f, *args):
        try:
            f(*args)
            res.append({"fault": name, "caught": False})
        except SentinelError as e:
            d = e.to_dict()
            res.append({"fault": name, "caught": True, "code": d["code"], "op": d["op"], "line": d["line"],
                        "index": d["index"], "message": str(e).splitlines()[0]})
        print(json.dumps(res[-1]), file=sys.stderr)

    ex = CompiledExecutor("cuda", sentinels="all")
    ex16 = CompiledExecutor("cuda", "float16", sentinels="all")

    @truestar.function(executor=ex)
    def nan_f(x: Tensor["B", 64]) -> Tensor["B", 64]:
        return ops.log(x) * 2.0 + 1.0

    @truestar.function(executor=ex)
    def inf_f(x: Tensor["B", 64]) -> Tensor["B", 64]:
        return ops.exp(x * 50.0) - 1.0

    @truestar.function(executor=ex16)
    def ovf_f(x: Tensor["B", 64], w: Tensor[64, 64]) -> Tensor["B", 64]:
        return (x @ w) * 100.0

    @truestar.function(executor=ex)
    def div_f(x: Tensor["B", 64]) -> Tensor["B", 64]:
        return x / x.sum(-1, keepdim=True)

    x = torch.rand(32, 64, device="cuda") + 0.5
    for f, args in ((nan_f, (x,)), (inf_f, (x * 0.01,)), (ovf_f, (x, torch.rand(64, 64, device="cuda") * 0.01)),
                    (div_f, (x,))):
        f(*args)
    bad = x.clone()
    bad[7, 11] = -1.0
    show("nan: log of a negative entry", nan_f, bad)
    big = x * 0.01
    big[3, 5] = 3.0
    show("inf: exp overflows float32", inf_f, big)
    xo = x.clone()
    xo[9] = 40.0
    show("overflow: float16 product too large", ovf_f, xo, torch.rand(64, 64, device="cuda"))
    xz = x.clone()
    xz[12] = 0.0
    show("div: a row sums to zero", div_f, xz)
    # detect_anomaly on the same NaN, in PyTorch eager
    xt = bad.clone().requires_grad_(True)
    try:
        with torch.autograd.detect_anomaly():
            (torch.log(xt) * 2.0 + 1.0).sum().backward()
        msg = "no error"
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
    res.append({"fault": "nan, torch detect_anomaly", "message": msg})
    print(json.dumps(res[-1]), file=sys.stderr)
    # the same NaN with detect_anomaly in a forward only (inference) call
    try:
        with torch.autograd.detect_anomaly(), torch.no_grad():
            torch.log(bad) * 2.0 + 1.0
        msg = "no error (forward only, nothing checked)"
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
    res.append({"fault": "nan, torch detect_anomaly forward under no_grad", "message": msg})
    print(json.dumps(res[-1]), file=sys.stderr)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("mlp", "transformer", "resnet18"))
    ap.add_argument("--faults", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.faults:
        print(json.dumps(faults(), indent=1))
        return
    if a.model:
        print(json.dumps(run_model(a.model)))
        return
    results = []
    for m in ("mlp", "transformer", "resnet18"):
        p = subprocess.run([sys.executable, __file__, "--model", m], capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stderr[-3000:], file=sys.stderr)
            continue
        results.append(json.loads(p.stdout.strip().splitlines()[-1]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    for r in results:
        print(f"\n{r['model']} {r['batch']}")
        for row in r["rows"]:
            ov = f"{row['overhead_pct']:+.1f}%" if "overhead_pct" in row else "baseline"
            gov = f"{row['gpu_overhead_pct']:+.1f}%" if "gpu_overhead_pct" in row else "baseline"
            print(f"  {row['tool']:<12} {row['mode']:<20} {row['variant']:<26} {row['ms']:8.3f} ms {ov:>9}   "
                  f"GPU {row['gpu_ms']:8.3f} ms {gov:>9}")


if __name__ == "__main__":
    main()
