"""Host and device timing helpers used by scripts/bench_overhead.py and the runtime tests.

Three numbers per callable, all in microseconds per call, each the median of several repeats:

* host: wall time to enqueue n calls back to back with no synchronization, divided by n. This is
  the Python, dispatch and launch cost; the GPU runs behind it as long as the launch queue does
  not fill (n stays well below its depth of about 1000 launches).
* wall: the same loop followed by one synchronize, divided by n. Steady state throughput: the
  larger of host and device time when the two overlap.
* device: the calls recorded k times into one CUDA graph and timed with CUDA events around the
  replay, divided by k. No host work is in the measured window, so this is GPU time plus the
  small gaps between kernels.

The GPU on the development machine is shared with other work, so callers repeat and report
medians, never single runs.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List

import torch


def _spin(fn: Callable, seconds: float):
    # GPUs drop clocks when idle; run the workload briefly so the timed loop starts at speed.
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        fn()
    torch.cuda.synchronize()


def host_and_wall(fn: Callable, n: int = 200, repeats: int = 7, spin: float = 0.1
                  ) -> Dict[str, float]:
    """Median host enqueue time and median wall time per call, in microseconds."""
    _spin(fn, spin)
    hosts: List[float] = []
    walls: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        hosts.append((t1 - t0) / n * 1e6)
        walls.append((t2 - t0) / n * 1e6)
    return {"host_us": statistics.median(hosts), "wall_us": statistics.median(walls)}


def host_only(fn: Callable, n: int = 2000, repeats: int = 7) -> float:
    """Median time per call in microseconds for host only work (no GPU launches expected)."""
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        vals.append((time.perf_counter() - t0) / n * 1e6)
    return statistics.median(vals)


def device_time(fn: Callable, k: int = 20, repeats: int = 7, spin: float = 0.1) -> float:
    """Median GPU time per call in microseconds, from k calls recorded into one CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k):
            fn()
    _spin(g.replay, spin)
    vals = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end) * 1e3 / k)
    del g
    return statistics.median(vals)


def median(xs) -> float:
    return statistics.median(list(xs))
