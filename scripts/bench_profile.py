"""`truestar profile` totals against whole call CUDA events and torch.profiler.

    PYTHONPATH=src python scripts/bench_profile.py

For each model: the sum of the per step event times, the whole call span with the GPU kept busy
(CUDA events around the call, no step events), the call as it runs, torch.profiler's kernel time
and its count of Truestar kernel launches, and how many steps map to a source line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ.setdefault("TRUESTAR_VERIFY_GPU_MB", "1024")


def one(name: str, repeat: int) -> dict:
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from bench_sentinels import build
    from truestar import ir
    from truestar.compiler import compile_program
    from truestar.compiler.profile import profile_program

    m, x = build(name)
    prog = ir.import_module(m, (x,))
    cp = compile_program(prog, "cuda")
    xd = x.cuda()
    runs = []
    for _ in range(repeat):
        p = profile_program(cp, [xd], iters=30, warmup=3, crosscheck=True)
        t, c = p["totals"], p["crosscheck"]
        runs.append({"step_sum_ms": t["step_gpu_ms"], "gpu_busy_ms": t["gpu_busy_ms"],
                     "as_run_ms": t["call_span_ms"], "host_issue_ms": t["host_issue_ms"],
                     "instrumented_span_ms": t["instrumented_span_ms"],
                     "profiler_gpu_ms": c["gpu_ms_per_call"], "profiler_truestar_kernel_ms": c["truestar_kernel_ms_per_call"],
                     "launched": c["truestar_kernels_launched"], "seen": c["truestar_kernels_seen"],
                     "steps_run": p["accounting"]["steps_run"], "steps_timed": p["accounting"]["steps_timed"],
                     "steps": len(p["steps"]), "steps_without_line": sum(1 for r in p["steps"] if not r["lines"]),
                     "kernel_steps_without_line": sum(1 for r in p["steps"] if not r["lines"] and r["kind"] == "kernel"),
                     "top": [(r["step"], r["kind"], round(r["gpu_ms"], 4)) for r in p["steps"][:3]]})
        torch.cuda.empty_cache()
    return {"model": name, "batch": list(x.shape), "runs": runs}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--model":
        print(json.dumps(one(sys.argv[2], 3)))
        return
    out = []
    for m in ("mlp", "transformer", "resnet18"):
        p = subprocess.run([sys.executable, __file__, "--model", m], capture_output=True, text=True)
        if p.returncode:
            print(p.stderr[-2000:], file=sys.stderr)
            continue
        r = json.loads(p.stdout.strip().splitlines()[-1])
        out.append(r)
        for run in r["runs"]:
            print(m, json.dumps(run))
    if len(sys.argv) > 2 and sys.argv[1] == "--out":
        with open(sys.argv[2], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
