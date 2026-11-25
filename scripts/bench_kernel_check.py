"""R1 benchmark: `truestar kernel` against a single input allclose on correct and subtly wrong kernels.

Every case of truestar.kernelbench (and, with --cuda, the Triton cases of truestar.kernelbench_triton)
is checked on several draws of its example input by:

* truestar kernel: `truestar.kernelcheck.check_kernel(reference, candidate, inputs)`; a kernel is
  rejected when the status is not ok (divergent, or suspect for nondeterminism)
* truestar --lean: the same with the lean verdict (TS1300) on
* allclose 1e-2 (KernelBench's check), allclose 1e-4 (the YAKK baseline) and torch's default
  `assert_close`, each on the example input only, against the reference in the same dtype

and prints false fail (correct kernels rejected) and false pass (wrong kernels accepted) rates,
the codes truestar gives per wrong kernel, and seconds per kernel.

    PYTHONPATH=src python scripts/bench_kernel_check.py [--draws 3] [--cuda] [--json out.json]
"""

import argparse
import json
import sys
import time

import torch

from truestar import kernelbench
from truestar.kernelcheck import check_kernel


def _baselines(case, inputs):
    out = case.candidate(*inputs)
    ref = case.reference(*inputs)
    o, r = out.double(), ref.double()
    try:
        torch.testing.assert_close(out, ref)
        td = True
    except AssertionError:
        td = False
    return {"allclose 1e-2": bool(torch.allclose(o, r, rtol=1e-2, atol=1e-2)),
            "allclose 1e-4": bool(torch.allclose(o, r, rtol=1e-4, atol=1e-4)),
            "torch default": td}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--only", default=None, help="substring of case names")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    runs = [(c, "cpu") for c in kernelbench.cases()]
    if a.cuda and torch.cuda.is_available():
        from truestar import kernelbench_triton

        runs += [(c, "cuda") for c in kernelbench_triton.cases()]
    if a.only:
        runs = [r for r in runs if a.only in r[0].name]
    names = ["truestar kernel", "allclose 1e-2", "allclose 1e-4", "torch default",
             "truestar --lean"]
    ff = {n: 0 for n in names}
    fp = {n: 0 for n in names}
    n_ok = n_bad = 0
    rows, times = [], []
    for case, device in runs:
        passed = {n: 0 for n in names}
        codes = set()
        for d in range(a.draws):
            g = torch.Generator().manual_seed(2000 + d)
            inputs = tuple(t.to(device) for t in case.make_inputs(case.dims, case.dtype, g))
            verdict = _baselines(case, inputs)
            t0 = time.perf_counter()
            rep = check_kernel(case.reference, case.candidate, inputs, device=device, lean=True)
            times.append(time.perf_counter() - t0)
            verdict["truestar --lean"] = rep.status == "ok"
            verdict["truestar kernel"] = not any(v.status != "ok" and v.status != "skipped"
                                                 and v.code != "TS1300" for v in rep.variants)
            codes.update(rep.codes)
            for n in names:
                passed[n] += verdict[n]
                if case.correct and not verdict[n]:
                    ff[n] += 1
                if not case.correct and verdict[n]:
                    fp[n] += 1
            n_ok += case.correct
            n_bad += not case.correct
            if d == 0:
                first = rep.first_failure.as_dict() if rep.first_failure else None
                blamed = rep.blamed_op.op if rep.blamed_op else None
        rows.append({"case": case.name, "device": device, "correct": case.correct, "bug": case.bug,
                     "needs": list(case.needs), "passed": passed, "codes": sorted(codes),
                     "first_failure": first, "blamed_op": blamed})
        cells = "  ".join(f"{passed[n]}/{a.draws}" for n in names)
        print(f"{case.name:30s} {device:4s} {'correct' if case.correct else 'WRONG  '} "
              f"passed: {cells}  {','.join(sorted(codes)) or '-'}"
              f"{'  [' + first['name'] + ']' if first else ''}", flush=True)
    print()
    print(f"{len(runs)} kernels x {a.draws} draws: {n_ok} correct runs, {n_bad} wrong runs")
    print(f"{'checker':18s} {'false fail':>14s} {'false pass':>14s}")
    summary = {}
    for n in names:
        summary[n] = {"false_fail": ff[n] / max(1, n_ok), "false_pass": fp[n] / max(1, n_bad),
                      "false_fail_count": ff[n], "false_pass_count": fp[n]}
        print(f"{n:18s} {ff[n] / max(1, n_ok):9.1%} ({ff[n]:3d}) {fp[n] / max(1, n_bad):8.1%} "
              f"({fp[n]:3d})")
    times.sort()
    print(f"truestar kernel seconds per kernel: median {times[len(times) // 2]:.2f}, "
          f"max {times[-1]:.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "rows": rows, "median_seconds": times[len(times) // 2],
                       "max_seconds": times[-1]}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
