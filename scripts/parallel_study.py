"""Measurements for docs/PARALLEL.md: detection of every injected fault, false alarms on correct
layouts, and the cost of one check on threads and on gloo processes.

    PYTHONPATH=src;tests python scripts/parallel_study.py

Every rank is simulated on this CPU (threads, or gloo processes); no GPU interconnect is involved.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

import torch  # noqa: E402

import parallel_models as M  # noqa: E402
from truestar.parallel import FAULTS, CrossEntropy, parallel_check  # noqa: E402


def detection():
    print("## faults, layout tp=2,dp=4,ga=3,sp=1, batch 24 x 4 tokens (factors dp 4, ga 3, tp 2, world 8 all differ);"
          " ga_mean_of_means: classifier with padding, tp=2,dp=2,ga=2")
    print("| fault | first code | module | collective | loss at step 1 differs | seconds |")
    print("|---|---|---|---|---|---|")
    for fault in FAULTS:
        t0 = time.perf_counter()
        if fault == "ga_mean_of_means":
            x = torch.randn(8, 6, 8, generator=torch.Generator().manual_seed(3))
            r = parallel_check(M.Classifier(), x, "tp=2,dp=2,ga=2", targets=M.padded_targets(), loss=CrossEntropy(),
                               faults=[fault])
        else:
            r = parallel_check(M.tiny(), M.tiny_inputs(B=24), "tp=2,dp=4,ga=3,sp=1", plan=M.tiny_plan(sp=True),
                               faults=[fault], clip=0.5)
        f = r.first()
        lossdiff = abs(r.loss_parallel - r.loss_reference) > 1e-9 * abs(r.loss_reference) \
            if r.loss_parallel == r.loss_parallel else "no loss (crash)"
        print(f"| {fault} | {f.code if f else 'none'} | {f.module if f else ''} | {f.collective or ''} | "
              f"{lossdiff} | {time.perf_counter() - t0:.2f} |")


def false_alarms():
    layouts = [("tp=2", False), ("tp=4", False), ("tp=2,dp=2", False), ("tp=2,dp=2,ga=2", False),
               ("tp=2,ga=2,sp=1", True), ("tp=2,dp=2,ga=2,sp=1", True), ("tp=4,dp=2,sp=1", True),
               ("dp=4,ga=2", False), ("dp=8", False)]
    runs = bad = 0
    worst = 0.0
    for seed in range(4):
        for lay, sp in layouts:
            for opt, clip in (("sgd", 0.0), ("adamw", 0.5)):
                r = parallel_check(M.tiny(seed), M.tiny_inputs(seed + 10, B=16, T=8), lay,
                                   plan=M.tiny_plan(sp=sp) if "tp" in lay else None, optimizer=opt, clip=clip)
                runs += 1
                bad += r.status != "ok"
                worst = max([worst] + list(r.worst.values()))
    for seed in range(4):
        for lay in ("tp=2,dp=2,ga=2", "dp=4", "dp=2,ga=4"):
            x = torch.randn(8, 6, 8, generator=torch.Generator().manual_seed(seed))
            r = parallel_check(M.Classifier(), x, lay, targets=M.padded_targets(seed=seed), loss=CrossEntropy())
            runs += 1
            bad += r.status != "ok"
            worst = max([worst] + list(r.worst.values()))
        r = parallel_check(M.MLP(), torch.randn(16, 10), "tp=2,dp=2,ga=2", optimizer="adamw", clip=1.0)
        runs += 1
        bad += r.status != "ok"
    print(f"## false alarms: {bad} of {runs} correct runs flagged; worst error over all stages {worst:.1e} "
          "(threshold 1e-9)")


def cost():
    print("## cost of one check (threads vs gloo processes)")
    f = os.path.join(HERE, "..", "tests", "parallel_models_file.py")
    from truestar.cli import _load

    model, x, _ = _load(f)
    for ranks in ("threads", "gloo"):
        for lay in ("tp=2,dp=2,ga=2,sp=1", "tp=2,dp=1"):
            t0 = time.perf_counter()
            r = parallel_check(model, x, lay, plan=M.tiny_plan(sp="sp=1" in lay), ranks=ranks, model_file=f)
            print(f"  {ranks:7s} {lay:20s} {r.status:9s} total {time.perf_counter() - t0:5.2f} s "
                  f"(reference {r.timings['reference']:.2f}, layout {r.timings['parallel']:.2f}, "
                  f"diagnosis {r.timings['diagnosis']:.2f}), {r.collectives} collectives on rank 0")
    big = M.Tiny(din=64, d=256, ff=512, head_dim=32, n=4, dout=64)
    xb = torch.randn(16, 32, 64)
    t0 = time.perf_counter()
    r = parallel_check(big, xb, "tp=2,dp=2,ga=2,sp=1", plan=M.tiny_plan(n=4, sp=True))
    print(f"  threads 4 blocks d=256, batch 16 x 32 tokens, tp=2,dp=2,ga=2,sp=1: {r.status}, "
          f"{time.perf_counter() - t0:.2f} s, worst {max(r.worst.values()):.1e}")


if __name__ == "__main__":
    detection()
    false_alarms()
    cost()
