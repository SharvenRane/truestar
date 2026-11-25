"""Where the time of a compiled decode step and of an imported linear attention model goes.

For each case: eager PyTorch, the compiled ordinary call (automatic graphs off), the compiled call
as it runs by default (automatic CUDA graph from the second call), GPU busy time from
``truestar profile`` and the heaviest steps. CPU cases report the ordinary call and cProfile.

    PYTHONPATH=src python scripts/profile_decodeperf.py --cases mla,linattn --device cuda
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import io
import json
import pstats
import statistics
import time

import torch

PROFILE_ITERS = 5
DETAIL = False
MLA_MODES, MLA_BATCHES = ["absorb", "expand", "fused"], [1, 8]


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def _t(fn, dev, reps):
    fn()
    fn()
    ts = []
    for _ in range(reps):
        _sync(dev)
        t = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t)
    return round(1e3 * statistics.median(ts), 4)


def _loop(fn, dev, n):
    """n dependent calls back to back, one sync at the end: pipelined time per call."""
    fn()
    _sync(dev)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(dev)
    return round(1e3 * (time.perf_counter() - t) / n, 4)


def _report(cp, inputs, eager, dev, reps, top):
    r = {}
    with torch.no_grad():
        r["eager_ms"] = _t(eager, dev, reps)
        r["eager_loop_ms"] = _loop(eager, dev, reps)
        cp(list(inputs))                                        # first call: the float64 checks
        r["fallbacks"] = len(cp.report.fallbacks)
        auto = getattr(cp, "auto_graphs", False)
        cp.auto_graphs = False
        r["ordinary_ms"] = _t(lambda: cp(list(inputs)), dev, reps)
        if dev == "cpu":
            pr = cProfile.Profile()
            pr.enable()
            for _ in range(reps):
                cp(list(inputs))
            pr.disable()
            buf = io.StringIO()
            pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(top)
            r["cprofile"] = [ln for ln in buf.getvalue().splitlines() if ln.strip()][-top - 1:]
        cp.auto_graphs = auto
        if dev == "cuda":
            r["auto_ms"] = _t(lambda: cp(list(inputs)), dev, reps)
            r["auto_loop_ms"] = _loop(lambda: cp(list(inputs)), dev, reps)
            r["auto_replays"] = cp.auto_graph_replays
            r["auto_refusals"] = {str(k)[:60]: v for k, v in cp.auto_graph_refusals.items()}
            from truestar.compiler.profile import profile_program
            p = profile_program(cp, list(inputs), iters=PROFILE_ITERS)
            r["totals"] = {k: p["totals"][k] for k in ("step_gpu_ms", "gpu_busy_ms", "host_issue_ms", "steps")}
            r["top"] = [(s["step"][:50], s["kind"], s.get("impl"), round(s["gpu_ms"], 4), s["ops"][:6])
                        for s in p["steps"][:top]]
            kinds = {}
            for s in p["steps"]:
                kinds[s["kind"]] = kinds.get(s["kind"], 0.0) + s["gpu_ms"]
            r["gpu_ms_by_kind"] = {k: round(v, 4) for k, v in kinds.items()}
            if DETAIL:
                r["non_kernel_steps"] = [(s["step"][:40], s["kind"], round(s["gpu_ms"], 4), s["ops"][:4],
                                          [str(x)[:40] for x in (s.get("shape"),)])
                                         for s in p["steps"] if s["kind"] != "kernel"]
            from torch.profiler import ProfilerActivity, profile as tprof
            with tprof(activities=[ProfilerActivity.CUDA]) as pt:
                for _ in range(5):
                    eager()
                _sync(dev)
            ev = [e for e in pt.key_averages() if e.device_type.name == "CUDA"]
            r["eager_gpu_ms"] = round(sum(e.self_device_time_total for e in ev) / 5 / 1e3, 4)
            r["eager_kernels"] = sum(e.count for e in ev) / 5
            r["eager_top"] = [(e.key[:60], round(e.self_device_time_total / 5 / 1e3, 4))
                              for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:top]]
    return r


def mla_cases(dev, reps, top):
    import transformers  # noqa: F401
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("bench_mla_decode", __file__.replace("profile_decodeperf", "bench_mla_decode"))
    bm = importlib.util.module_from_spec(spec)
    sys.modules["bench_mla_decode"] = bm
    spec.loader.exec_module(bm)
    from truestar.mla import compile_steps, from_hf
    hf = bm.build_hf()
    model = from_hf(hf)
    c = model.c
    e32 = from_hf(hf).to(dev)
    out = {}
    for mode in MLA_MODES:
        st = compile_steps(model, dev, "float32", decode=mode)
        for batch in MLA_BATCHES:
            prompt = torch.randint(0, c.vocab, (batch, 64), generator=torch.Generator().manual_seed(3)).to(dev)
            _, cc, rc = st.prefill([prompt, torch.arange(64, device=dev),
                                    *e32.empty_cache(batch, 192, torch.float32, dev)])
            ids, pos = prompt[:, -1:].contiguous(), torch.tensor([64], device=dev)
            inputs = [ids, pos, cc, rc]
            eager = lambda: e32.step(ids, pos, cc, rc, mode)   # noqa: E731
            out[f"mla {mode} b{batch}"] = _report(st.decode, inputs, eager, dev, reps, top)
            print(f"mla {mode} b{batch}", json.dumps(out[f"mla {mode} b{batch}"]), flush=True)
    return out


def linattn_cases(dev, reps, top, families, seq, batch):
    from truestar import ir
    from truestar import testing_hf as th
    from truestar.compiler import compile_program
    out = {}
    for name in families:
        torch.manual_seed(0)
        model, _ = th.build(name, batch=batch)
        ids = torch.randint(3, th.VOCAB, (batch, seq), generator=torch.Generator().manual_seed(1))
        prog = ir.import_module(copy.deepcopy(model).double(), (ids,))
        eager_m = model.to(dev)
        x = ids.to(dev)
        cp = compile_program(prog, dev, "float32")
        out[name] = _report(cp, [x], lambda: eager_m(x), dev, reps, top)
        out[name]["ops"] = dict(sorted(prog.op_counts().items(), key=lambda kv: -kv[1])[:30])
        print(name, json.dumps(out[name]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="mla,linattn")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--families", default="mamba2,falcon_h1,qwen3_5,bamba,zamba2")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mla-modes", default="absorb,expand,fused")
    ap.add_argument("--mla-batches", default="1,8")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    global MLA_MODES, MLA_BATCHES, DETAIL
    MLA_MODES, MLA_BATCHES = a.mla_modes.split(","), [int(x) for x in a.mla_batches.split(",")]
    DETAIL = a.detail
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name() if a.device == "cuda" else None}
    if "mla" in a.cases and a.device == "cuda":
        res["mla"] = mla_cases(a.device, a.reps, a.top)
    if "linattn" in a.cases:
        res["linattn"] = linattn_cases(a.device, a.reps, a.top, a.families.split(","), a.seq, a.batch)
    print("RESULT_JSON", json.dumps(res))


if __name__ == "__main__":
    main()
