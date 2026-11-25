"""R7 benchmark: `@truestar.verified_autotune` against `@triton.autotune` (CUDA and Triton only).

Kernels (truestar.kernelbench_triton style, written here):

* mm_nokmask: a tiled matmul with no K mask, so it is right only for configs whose BK divides K.
  At K=1040 only BK=16 divides; the BK=32 and 64 configs read past each row and are faster.
* mm_masked: the same matmul with a K mask (every config right): the overhead with nothing removed.
* softmax: one row per program, BLOCK the padded row, a num_warps sweep (every config right).

For each it reports, for triton.autotune and for verified_autotune: the config picked, whether the
output is right (truestar.close_report), the tuning time of the first call (compile caches warm,
median of 3), and for verified_autotune the configs removed and why.

    PYTHONPATH=src python scripts/bench_autotune.py [--json out.json]
"""

import argparse
import json
import statistics
import sys
import time
import warnings

import torch
import triton

from truestar.autotune import VerifiedAutotuneWarning, verified_autotune
from truestar.kernelbench_triton import (AUTOTUNE_MM_CONFIGS as MM_CONFIGS,
                                         AUTOTUNE_SM_CONFIGS as SM_CONFIGS,
                                         autotune_mm_kernel as mm_kernel,
                                         autotune_softmax_kernel as softmax_kernel)
from truestar.tolerance import close_report


def mm_ref(a, b):
    return a @ b


def softmax_ref(x):
    return torch.softmax(x, -1)


def _mm_runner(decorate, kmask: bool):
    tuned = decorate(mm_kernel)

    def run(a, b):
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty(M, N, device=a.device, dtype=a.dtype)
        grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa
        tuned[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1), KMASK=kmask)
        return c

    return tuned, run


def _sm_runner(decorate):
    tuned = decorate(softmax_kernel)

    def run(x):
        M, N = x.shape
        y = torch.empty_like(x)
        tuned[(M,)](x, y, N, x.stride(0), y.stride(0), BLOCK=triton.next_power_of_2(N))
        return y

    return tuned, run


def _best(tuned):
    if hasattr(tuned, "best_config") and tuned.best_config is not None:
        cfg = tuned.best_config
        kw = dict(cfg.all_kwargs())
        return ", ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
    return "?"


def bench_case(name, make_runner_t, make_runner_v, inputs, reference, trials=3):
    rows = {"triton.autotune": [], "verified_autotune": []}
    picks = {}
    removed = []
    make_runner_t()[1](*inputs)  # compile every config once so both sides start warm
    torch.cuda.synchronize()
    for _ in range(trials):
        for label, mk in (("triton.autotune", make_runner_t), ("verified_autotune", make_runner_v)):
            tuned, run = mk()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", VerifiedAutotuneWarning)
                out = run(*inputs)
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rows[label].append(dt)
            ok = close_report(reference, inputs, out).ok
            picks[label] = (_best(tuned), ok)
            if label == "verified_autotune":
                rep = next(iter(tuned.reports.values()))
                removed = [(r.config, r.reason[:160]) for r in rep.rejected]
                split = (rep.derive_seconds, rep.check_seconds, rep.bench_seconds)
    t_t = statistics.median(rows["triton.autotune"])
    t_v = statistics.median(rows["verified_autotune"])
    print(f"\n{name}")
    for label in rows:
        cfg, ok = picks[label]
        print(f"  {label:18s} tune {statistics.median(rows[label]):6.3f} s  picked [{cfg}]  "
              f"output {'right' if ok else 'WRONG'}")
    print(f"  verified overhead {100 * (t_v - t_t) / t_t:+.1f}%  (derive {split[0]:.3f} s, "
          f"checks {split[1]:.3f} s, timing {split[2]:.3f} s)")
    for cfg, why in removed:
        print(f"  removed [{cfg}]: {why}")
    return {"case": name, "triton_s": t_t, "verified_s": t_v,
            "overhead_pct": 100 * (t_v - t_t) / t_t, "picks": {k: list(v) for k, v in picks.items()},
            "removed": removed}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    g = torch.Generator().manual_seed(0)
    results = []
    for K, kmask, label in ((1040, False, "mm_nokmask K=1040 (only BK=16 is right)"),
                            (1024, False, "mm_nokmask K=1024 (every BK divides)"),
                            (1040, True, "mm_masked K=1040 (every config right)")):
        a_ = torch.randn(512, K, generator=g).cuda().half()
        b_ = torch.randn(K, 384, generator=g).cuda().half()
        mk_t = lambda km=kmask: _mm_runner(lambda f: triton.autotune(MM_CONFIGS, key=["M", "N", "K"])(f), km)  # noqa
        mk_v = lambda km=kmask: _mm_runner(lambda f: verified_autotune(  # noqa
            MM_CONFIGS, key=["M", "N", "K"], reference=mm_ref, inputs=["a_ptr", "b_ptr"],
            outputs=["c_ptr"])(f), km)
        results.append(bench_case(label, mk_t, mk_v, (a_, b_), mm_ref))
    x = torch.randn(1024, 3000, generator=g).cuda().half()
    mk_t = lambda: _sm_runner(lambda f: triton.autotune(SM_CONFIGS, key=["N"])(f))  # noqa
    mk_v = lambda: _sm_runner(lambda f: verified_autotune(  # noqa
        SM_CONFIGS, key=["N"], reference=softmax_ref, inputs=["x_ptr"], outputs=["y_ptr"])(f))
    results.append(bench_case("softmax N=3000 (every config right)", mk_t, mk_v, (x,),
                              softmax_ref))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
