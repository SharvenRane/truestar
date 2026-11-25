"""Idle gate for the perfmm benchmarks: the fair benchmark's wait_idle (FAIR_BENCH_2026_11/
scripts/fb_common.py) with desktop browsers that hold a GPU context (Brave, Edge, Firefox) counted
as desktop, as Chrome already is there."""

import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("TRUESTAR_FAIR_BENCH", "fair_bench"), "scripts"))  # unpublished fair benchmark
import fb_common  # noqa: E402

DESKTOP_EXTRA = ("brave.exe", "msedge.exe", "firefox.exe")


def _own():
    return [os.getpid()] + [b["pid"] for b in fb_common.foreign_compute()
                            if any(n in b["name"].lower() for n in DESKTOP_EXTRA)]


def wait_idle(quiet=True):
    g = fb_common.wait_idle(own_pids=_own(), log=lambda s: print(s, flush=True))
    if not quiet or g["waited_s"] > 5:
        print(f"idle gate: waited {g['waited_s']} s, util {g['util_samples']}", flush=True)
    return g


def busy() -> bool:
    """Another process is using the GPU now (sampled while this process is not launching)."""
    import time
    time.sleep(1.0)   # utilization is averaged over the last sample period: let ours leave it
    for _ in range(3):
        if fb_common.gpu_state().get("util_pct", 100.0) > 10.0:
            return True
        time.sleep(0.3)
    return False


def timed(torch, fns, rounds):
    """probe_ieee_mm.timed with the fair benchmark's per round gate: the GPU is checked idle before
    every round and again after it; a round during which another process worked is thrown away
    and rerun. Returns (median, min, rounds redone)."""
    import statistics
    import triton.testing
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import probe_ieee_mm as P
    graphs = {t: P.graph_of(torch, f) for t, f in fns.items()}
    times = {t: [] for t in graphs}
    redone = 0
    done = 0
    while done < rounds:
        wait_idle()
        # the clocks drop while the gate waits: run the GPU for about half a second first, or
        # the first tool of a round is timed at idle clocks (seen 2026-11-25: up to 4x slower)
        import time
        first = next(iter(graphs.values()))
        t_end = time.time() + 0.5
        while time.time() < t_end:
            first.replay()
            torch.cuda.synchronize()
        got = {t: triton.testing.do_bench(g.replay, warmup=3, rep=25, return_mode="median") / 20 * 1e3
               for t, g in graphs.items()}
        torch.cuda.synchronize()
        if busy():
            redone += 1
            if redone > 50:
                raise RuntimeError("the GPU never stayed idle for a round")
            continue
        for t, v in got.items():
            times[t].append(v)
        done += 1
    del graphs
    torch.cuda.empty_cache()
    return ({t: statistics.median(v) for t, v in times.items()}, {t: min(v) for t, v in times.items()},
            redone)
