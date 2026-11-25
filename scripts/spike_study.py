"""Measurements for docs/SPIKES.md: watch overhead, verdicts on the constructed spikes over several data
seeds, and false alarms on clean runs.

    PYTHONPATH=src;tests python scripts/spike_study.py
"""

import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

import torch  # noqa: E402

import spike_models as SM  # noqa: E402
import spike_spec  # noqa: E402
from truestar.spikes import replay, train_step  # noqa: E402


WIDE = '''
import torch, torch.nn as nn, torch.nn.functional as F
import spike_spec

def get_model():
    torch.manual_seed(0)
    return spike_spec.Net(d=256, h=1024, out=256)

def get_loss():
    return F.mse_loss

def get_optimizer(params):
    return torch.optim.SGD(params, lr=0.05, momentum=0.9)
'''


def overhead(steps=200, wide=False):
    spec_path = SM.SPEC
    if wide:
        tmp = tempfile.mkdtemp()
        spec_path = os.path.join(tmp, "wide_spec.py")
        with open(spec_path, "w", encoding="utf-8") as f:
            f.write(WIDE)
        g = torch.Generator().manual_seed(0)
        xs = [(torch.randn(256, 256, generator=g) * 20, torch.randn(256, 256, generator=g)) for _ in range(steps)]
        import importlib.util
        sp = importlib.util.spec_from_file_location("wide_spec", spec_path)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
    else:
        xs = [SM.batch(n) for n in range(steps)]
        mod = spike_spec
    print(f"## overhead, {steps} steps of the {'wide (256, 1024, 256), batch 256' if wide else 'spec'} net in "
          "float16 on the CPU (median of 3 runs)")

    def plain():
        torch.manual_seed(0)
        m = mod.get_model().half()
        opt = mod.get_optimizer(m.parameters())
        t0 = time.perf_counter()
        for x, y in xs:
            train_step(m, opt, mod.get_loss(), x, y, torch.float16, 0.05, 0.0)
        return time.perf_counter() - t0

    def watched(every, ckpt=20):
        d = tempfile.mkdtemp()
        try:
            from truestar.spikes import SpikeWatch

            t0 = time.perf_counter()
            with SpikeWatch(spec_path, d, dtype="float16", shadow_every=every, ckpt_every=ckpt) as w:
                for x, y in xs:
                    w.step(x, y)
            return time.perf_counter() - t0
        finally:
            shutil.rmtree(d, ignore_errors=True)

    base = sorted(plain() for _ in range(3))[1]
    print(f"  plain loop: {1e3 * base / steps:.2f} ms per step")
    for every in (0, 50, 10, 1):
        t = sorted(watched(every) for _ in range(3))[1]
        what = "no shadow (checkpoints and batches only)" if every == 0 else f"shadow every {every}"
        print(f"  watch, {what}: {1e3 * t / steps:.2f} ms per step, {100 * (t / base - 1):+.0f}%")


def verdicts(seeds=5):
    print(f"## verdicts at the constructed spike (step 45), {seeds} data seeds each")
    for case, want in (("numerics", "numerics"), ("data", "data"), ("optimizer", "optimizer"), ("clean", "none")):
        got, secs, flagged = [], [], 0
        for s in range(seeds):
            d = tempfile.mkdtemp()
            try:
                _, recs = SM.run(d, case, data_seed=100 + 17 * s)
                flagged += recs[45].get("code") == "TS691"
                t0 = time.perf_counter()
                rep = replay(d, 45)
                secs.append(time.perf_counter() - t0)
                got.append(rep.verdict)
            finally:
                shutil.rmtree(d, ignore_errors=True)
        ok = sum(g == want for g in got)
        print(f"  {case:9s}: {ok}/{seeds} {want}, watch flagged the step {flagged}/{seeds}, replay "
              f"{1e3 * sum(secs) / len(secs):.0f} ms, verdicts {got}")


def false_alarms(seeds=5):
    print(f"## false alarms on clean runs ({seeds} seeds, 60 steps, shadow every 5, replay of every 5th step)")
    shadows = flags = replays = wrong = 0
    worst = 0.0
    for s in range(seeds):
        d = tempfile.mkdtemp()
        try:
            _, recs = SM.run(d, "clean", data_seed=300 + s, shadow_every=5)
            for r in recs:
                if "shadow" in r:
                    shadows += 1
                    flags += r.get("code") == "TS691"
                    worst = max(worst, r["shadow"]["loss_error"])
            for n in range(5, 60, 5):
                replays += 1
                wrong += replay(d, n).verdict != "none"
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"  shadow flags {flags} of {shadows} shadowed steps (worst loss error {worst:.1e}); replay verdicts "
          f"other than none: {wrong} of {replays}")
    for dt in ("bfloat16",):
        d = tempfile.mkdtemp()
        try:
            _, recs = SM.run(d, "numerics", dtype=dt)
            print(f"  the float16 overflow batch under {dt}: spike {recs[45]['spike']}, replay {replay(d, 45).verdict}")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    overhead()
    overhead(steps=60, wide=True)
    verdicts()
    false_alarms()
