"""Autotuning of GPU launch configs, persisted in the on disk kernel cache.

A launch config (block sizes, num_warps, num_stages, and for attention the LPT and SPLIT
switches) is picked once per (kernel hash, shape bucket, input dtypes, device name, Triton
version). The shape bucket rounds every symbolic size up to a power of two, so nearby sizes share
one search. The winner is written to `<cache>/triton/tune/<key>.json` with a digest; a later
process reads it back and pays nothing. A file that does not parse, has the wrong version, key or
digest, or holds a config the scheduler rejects is reported with a TuneCacheWarning and the search
runs again, overwriting it.

Modes (`compile(..., tune=...)` overrides the environment variable `TRUESTAR_TUNE`):

* auto (default): tune launches big enough to matter (triton_backend.worth_tuning); small ones
  use the default config, since a search would cost more than it could ever save.
* on: tune every launch.  * off: always use the default config.

The search compiles and times the default config first, compiles the other candidates in a
thread pool (`TRUESTAR_TUNE_WORKERS`, default up to 8), and skips whatever is not compiled when the
budget (`TRUESTAR_TUNE_BUDGET`, seconds, default 30) runs out; most of the cost is the Triton
compile of each candidate. Candidates that fail to compile or launch (for example too much shared
memory) are skipped. Timing uses triton.testing.do_bench (L2 flushed between runs, median), in two
rounds: every candidate briefly, then the best three again for longer.

Correctness does not depend on tuning: every config runs the same generated source, and the
config that is used is the one the first call verifies against the float64 reference.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

TUNE_VERSION = 1
# Bumped when triton_backend's default configs or candidates change, so configs tuned over an old
# search space are looked up under a new key (a plain miss, no warning) and searched again.
SEARCH_SPACE = 12
DEFAULT_BUDGET_S = 30.0


class TuneCacheWarning(UserWarning):
    """A cached launch config was unreadable or invalid and was tuned again."""


@dataclass
class TuneStats:
    searches: int = 0          # searches run (cache misses)
    cache_hits: int = 0        # configs read back from disk
    rejected: int = 0          # cached files found corrupt or invalid
    candidates: int = 0        # configs compiled and timed
    failed: int = 0            # candidates that failed to compile or launch
    seconds: float = 0.0       # wall time spent searching
    log: List[Dict] = field(default_factory=list)  # one record per search

    def reset(self):
        self.searches = self.cache_hits = self.rejected = self.candidates = self.failed = 0
        self.seconds = 0.0
        self.log = []


stats = TuneStats()


def mode(explicit: Optional[bool] = None) -> str:
    if explicit is True:
        return "on"
    if explicit is False:
        return "off"
    v = os.environ.get("TRUESTAR_TUNE", "auto").strip().lower()
    if v in ("0", "off", "false", "no"):
        return "off"
    if v in ("1", "on", "true", "yes", "all"):
        return "on"
    return "auto"


def budget_seconds() -> float:
    try:
        return float(os.environ.get("TRUESTAR_TUNE_BUDGET", DEFAULT_BUDGET_S))
    except ValueError:
        return DEFAULT_BUDGET_S


def _pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def _pow4(n: int) -> int:
    b = 1
    while b < n:
        b *= 4
    return b


def bucket(env: Dict[str, int], ignore=(), coarse=()) -> List[Tuple[str, int]]:
    """Every symbolic size rounded up to a power of two, those named in ``coarse`` (the row count
    of a contraction) to a power of four. Sizes named in ``ignore`` (batch like symbols, see
    triton_backend.batch_symbols) are left out, so a config tuned at one batch size is the config
    at every batch size and no new Triton variant is compiled when the batch changes."""
    return sorted((name, _pow4(v) if name in coarse else _pow2(v)) for name, v in env.items()
                  if name not in ignore)


def tune_key(kernel_key: str, env: Dict[str, int], dtypes, device: str, ignore=(), coarse=()) -> str:
    try:  # CPU only installs have no triton: tuning there must not need it
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    text = json.dumps({"v": TUNE_VERSION, "space": SEARCH_SPACE, "kernel": kernel_key, "bucket": bucket(env, ignore, coarse),
                       "dtypes": list(dtypes), "device": device, "triton": triton_version},
                      sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def tune_dir() -> Path:
    from .compiler import cache_dir
    return cache_dir() / "triton" / "tune"


def path_for(key: str) -> Path:
    return tune_dir() / f"{key}.json"


def _digest(key: str, cfg: Dict) -> str:
    return hashlib.sha256(json.dumps({"key": key, "config": cfg}, sort_keys=True)
                          .encode()).hexdigest()


def load(key: str, check: Callable[[Dict], Optional[str]]) -> Optional[Dict]:
    """The cached config for key, or None (missing, or corrupt: warned and counted)."""
    from .. import compile_cache
    if not compile_cache.enabled():      # TRUESTAR_COMPILE_CACHE=0: every process measures again
        return None
    path = path_for(key)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rec, dict):
            raise ValueError("not a JSON object")
        if rec.get("version") != TUNE_VERSION:
            raise ValueError(f"version {rec.get('version')!r}, expected {TUNE_VERSION}")
        if rec.get("key") != key:
            raise ValueError("key does not match the file name")
        cfg = rec.get("config")
        if rec.get("digest") != _digest(key, cfg):
            raise ValueError("digest mismatch (the file was edited or truncated)")
        why = check(cfg)
        if why is not None:
            raise ValueError(why)
    except (OSError, ValueError, TypeError) as e:
        stats.rejected += 1
        warnings.warn(f"truestar: cached launch config {path} is invalid ({e}); tuning again",
                      TuneCacheWarning, stacklevel=3)
        return None
    stats.cache_hits += 1
    try:
        os.utime(path)                   # least recently used order (compile_cache.trim)
    except OSError:
        pass
    return cfg


def store(key: str, cfg: Dict, record: Dict):
    d = tune_dir()
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(record, version=TUNE_VERSION, key=key, config=cfg, digest=_digest(key, cfg))
    path = path_for(key)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _bench(fn, rep_ms: float, stat: str = "median") -> float:
    import triton.testing
    return float(triton.testing.do_bench(fn, warmup=2, rep=rep_ms, return_mode=stat))


def _workers() -> int:
    try:
        n = int(os.environ.get("TRUESTAR_TUNE_WORKERS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else max(1, min(8, (os.cpu_count() or 2) - 1))


def search(make_launch: Callable[[Dict], Tuple[Callable[[], None], Callable[[], None]]],
           cands: List[Dict], budget: Optional[float] = None) -> Tuple[Dict, Dict]:
    """Time candidate configs; return (best config, record).

    make_launch(cfg) returns (run, build): run launches the kernel once, build compiles it
    without launching. cands[0] is the default: its failure propagates (it would fail without
    tuning too). Every candidate, the default included, is built in one thread pool (Triton's
    compiler releases the GIL for most of the work), so a search costs about one compile of wall
    time when there are enough workers; then they are timed one at a time. Whatever is not built
    when the budget runs out is skipped (the default is always waited for)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor, wait
    budget = budget_seconds() if budget is None else budget
    t0 = time.perf_counter()
    runs: Dict[int, Callable[[], None]] = {}
    failed: List[Tuple[Dict, str]] = []

    def build(n):
        run, b = make_launch(cands[n])
        b()
        return n, run

    ex = ThreadPoolExecutor(min(_workers(), len(cands)))
    futs = [ex.submit(build, n) for n in range(len(cands))]
    futs[0].result()  # the default: raises here when it cannot be built
    rest = list(range(1, len(cands)))
    if time.perf_counter() - t0 >= budget:  # the default alone used up the budget
        done, pending = set(), set(futs[1:])
    else:
        done, pending = wait(futs[1:], timeout=max(0.0, budget - (time.perf_counter() - t0)))
    for f in pending:
        f.cancel()
    ex.shutdown(wait=True)
    skipped = 0
    runs[0] = futs[0].result()[1]
    for f, n in zip(futs[1:], rest):
        if f.cancelled() or f not in done:
            skipped += 1
            continue
        err = f.exception()
        if err is not None:
            failed.append((cands[n], f"{type(err).__name__}: {str(err)[:200]}"))
            stats.failed += 1
        else:
            runs[n] = f.result()[1]
    run0 = runs[0]
    run0()
    torch.cuda.synchronize()
    t_built = time.perf_counter()
    # the GPU idles at low clocks after compiling; spin it up before the first timing
    t = time.perf_counter()
    while time.perf_counter() - t < 0.05:
        run0()
    torch.cuda.synchronize()
    timings: List[Tuple[int, float]] = []
    for n in sorted(runs):
        try:
            ms = _bench(runs[n], 10)
        except Exception as e:  # noqa: BLE001 (a launch failure skips the config)
            if n == 0:
                raise
            torch.cuda.synchronize()
            failed.append((cands[n], f"{type(e).__name__}: {str(e)[:200]}"))
            stats.failed += 1
            continue
        stats.candidates += 1
        timings.append((n, ms))
    # second round: the best three again, longer, so noise does not pick the winner
    top = sorted(timings, key=lambda x: x[1])[:3]
    final = {n: min(ms, _bench(runs[n], 40)) for n, ms in top}
    best_n = min(final, key=final.get)
    secs = time.perf_counter() - t0
    first = dict(timings)
    default_ms = final.get(0, first[0])
    record = {"best_ms": final[best_n], "default_ms": default_ms, "tune_seconds": secs,
              "compile_seconds": t_built - t0, "timing_seconds": time.perf_counter() - t_built,
              "timed": [[cands[n], final.get(n, ms)] for n, ms in timings],
              "skipped_for_budget": skipped,
              "failed": [[c, why] for c, why in failed]}
    stats.searches += 1
    stats.seconds += secs
    return cands[best_n], record
