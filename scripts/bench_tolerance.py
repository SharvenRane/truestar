"""R6 benchmark: derived tolerances (`truestar.assert_close`) against hand picked ones.

For every case of truestar.kernelbench whose bug shows on the example input (cases that need
strided inputs, large values or another shape are for `truestar kernel`, scripts/bench_kernel_check.py),
and for several input draws, it asks four checkers whether the candidate's output is right:

* truestar: `truestar.tolerance.close_report(reference, inputs, out)` (derived, no arguments)
* torch default: `torch.testing.assert_close(out, reference(*inputs))`, the reference in the same
  dtype on the same device, default rtol and atol (the common unit test)
* torch default vs float64: the same, the reference computed in float64 and cast to the dtype
* allclose 1e-2: `torch.allclose(out, reference(*inputs), rtol=1e-2, atol=1e-2)` (KernelBench)

and prints the false fail rate (correct candidates rejected) and false pass rate (wrong candidates
accepted) of each, plus per case results and the time truestar takes.

    PYTHONPATH=src python scripts/bench_tolerance.py [--draws 5] [--cuda] [--json out.json]
"""

import argparse
import json
import sys
import time

import torch

from truestar import kernelbench
from truestar.tolerance import clear_cache, close_report


def _torch_default(out, ref):
    try:
        torch.testing.assert_close(out, ref)
        return True
    except AssertionError:
        return False


def _run_case(case, draw, device):
    g = torch.Generator().manual_seed(1000 + draw)
    inputs = case.make_inputs(case.dims, case.dtype, g)
    inputs = tuple(t.to(device) for t in inputs)
    out = case.candidate(*inputs)
    same = case.reference(*inputs)
    ref64 = case.reference(*(t.double() for t in inputs)).to(case.dtype)
    t0 = time.perf_counter()
    rep = close_report(case.reference, inputs, out)
    dt = time.perf_counter() - t0
    return {
        "truestar": rep.ok,
        "torch default": _torch_default(out, same),
        "torch default vs float64": _torch_default(out, ref64),
        "allclose 1e-2": bool(torch.allclose(out.double(), same.double(), rtol=1e-2, atol=1e-2,
                                             equal_nan=True)),
    }, dt, rep


def cuda_cases():
    """Correct float32 kernels under TF32 flags and a cuBLAS float16 GEMM, on the GPU."""
    C = kernelbench.Case
    out = []

    def mm_cuda_tf32(a, b):
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return a @ b
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    out.append(C("cuda_mm_cublas_fp16", kernelbench.mm_ref, lambda a, b: a @ b, True,
                 kernelbench._mm_inputs, dict(M=256, K=4096, N=200), torch.float16))
    out.append(C("cuda_mm_tf32_policy_fp32", kernelbench.mm_ref, mm_cuda_tf32, True,
                 kernelbench._mm_inputs, dict(M=256, K=4096, N=200), torch.float32,
                 note="TF32 enabled while checking, so the policy is TF32"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--cuda", action="store_true", help="also run the CUDA cases")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    cases = [c for c in kernelbench.cases() if not c.needs]
    runs = [(c, "cpu") for c in cases]
    if a.cuda and torch.cuda.is_available():
        runs += [(c, "cuda") for c in cuda_cases()]
    names = ["truestar", "torch default", "torch default vs float64", "allclose 1e-2"]
    tally = {n: {"ff": 0, "fp": 0} for n in names}
    n_correct = n_wrong = 0
    rows = []
    times = []
    margins = []  # (worst element ratio, RMS ratio) of correct candidates: how close to the bound
    for case, device in runs:
        res_case = {n: 0 for n in names}
        for d in range(a.draws):
            clear_cache()
            old = torch.backends.cuda.matmul.allow_tf32
            if case.name == "cuda_mm_tf32_policy_fp32":
                torch.backends.cuda.matmul.allow_tf32 = True
            try:
                verdicts, dt, rep = _run_case(case, d, device)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old
            times.append(dt)
            if case.correct:
                w = max(rep.outputs, key=lambda o: o.worst_ratio)
                margins.append((w.worst_ratio, max(o.rms_ratio for o in rep.outputs), case.name))
            for n in names:
                passed = verdicts[n]
                res_case[n] += int(passed)
                if case.correct and not passed:
                    tally[n]["ff"] += 1
                if not case.correct and passed:
                    tally[n]["fp"] += 1
            if case.correct:
                n_correct += 1
            else:
                n_wrong += 1
            if d == 0:
                msg = rep.message()
        row = {"case": case.name, "device": device, "correct": case.correct, "bug": case.bug,
               "passed": res_case, "draws": a.draws, "truestar_message": msg}
        rows.append(row)
        cells = "  ".join(f"{res_case[n]}/{a.draws}" for n in names)
        print(f"{case.name:30s} {device:4s} {'correct' if case.correct else 'WRONG  '}  "
              f"passed: {cells}", flush=True)
    print()
    print(f"{len(runs)} cases x {a.draws} draws: {n_correct} correct runs, {n_wrong} wrong runs")
    print(f"{'checker':28s} {'false fail':>12s} {'false pass':>12s}")
    summary = {}
    for n in names:
        ff = tally[n]["ff"] / max(1, n_correct)
        fp = tally[n]["fp"] / max(1, n_wrong)
        summary[n] = {"false_fail": ff, "false_pass": fp, "false_fail_count": tally[n]["ff"],
                      "false_pass_count": tally[n]["fp"]}
        print(f"{n:28s} {ff:11.1%} ({tally[n]['ff']:3d}) {fp:6.1%} ({tally[n]['fp']:3d})")
    if margins:
        we = max(margins, key=lambda m: m[0])
        wr = max(margins, key=lambda m: m[1])
        print(f"correct candidates, closest to the bound: element {we[0]:.3f} of its bound "
              f"({we[2]}), RMS ratio {wr[1]:.3f} of limit 2 ({wr[2]})")
        summary_margin = {"max_element_ratio": we[0], "max_element_case": we[2],
                          "max_rms_ratio": wr[1], "max_rms_case": wr[2]}
    else:
        summary_margin = {}
    times.sort()
    print(f"truestar seconds per check: median {times[len(times) // 2]:.3f}, max {times[-1]:.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "margin": summary_margin, "rows": rows, "n_correct": n_correct,
                       "n_wrong": n_wrong}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
