"""Validate the AMD (ROCm) path on a machine with an AMD GPU. Written before any such machine was
available: every step below is untested on hardware until this script has run once.

    PYTHONPATH=src python scripts/rocm_validate.py --out rocm-results
    PYTHONPATH=src python scripts/rocm_validate.py --out rocm-results --quick    # skip the long tests

Steps, each recorded in <out>/summary.json with its command, exit code, seconds and last lines:

1. environment: torch, HIP, Triton, the attached GPU as truestar.kernel.rocm.current_target()
   (arch, wave size, CUs, LDS); refuses (exit 2) without a ROCm build of torch and an AMD GPU;
2. compile matrix: scripts/rocm_compile_matrix.py for the attached arch, if the table knows it;
3. kernel tests: tests/test_rocm.py plus the kernel, tuning, SM aware, split K and conv tests, on
   the GPU (torch calls it "cuda"); every kernel is checked against float64 on its first call;
4. fingerprint: scripts/laptop_fingerprint.py (float32, float16, bfloat16), bit level hashes of the
   compiled executor and eager PyTorch to diff against an NVIDIA machine's file;
   `truestar fingerprint` as well when this version's CLI has that command;
5. short benchmark: scripts/bench_matmul.py on six shapes in float32 and float16, Truestar's
   default config against F.linear (rocBLAS or hipBLASLt).

Exit 0 when every step passed, 1 when one failed, 2 when ROCm is not here. docs/ROCM.md says how
to run it on AMD Developer Cloud.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_TESTS = ["tests/test_rocm.py", "tests/test_kernel.py", "tests/test_kernel_gpu_verify.py",
                "tests/test_kernel_verify_elements.py", "tests/test_kernel_magnitude.py",
                "tests/test_kernel_tune.py", "tests/test_sm_aware.py", "tests/test_contraction_split.py",
                "tests/test_conv_kernels.py", "tests/test_autotune.py", "tests/test_codes_unique.py"]
QUICK_TESTS = ["tests/test_rocm.py", "tests/test_kernel.py", "tests/test_codes_unique.py"]
BENCH_SHAPES = "16x4096x1024,64x2048x2048,512x1024x256,1024x1024x1024,4096x4096x4096,64x512x8192"


def run(name, cmd, out_dir, summary, timeout):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    log = os.path.join(out_dir, f"{name}.log")
    t0 = time.perf_counter()
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    try:
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        code = p.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    with open(log, encoding="utf-8", errors="replace") as fh:
        tail = fh.read().splitlines()[-15:]
    rec = dict(name=name, command=cmd, exit=code, seconds=round(time.perf_counter() - t0, 1), log=log, tail=tail)
    summary["steps"].append(rec)
    print(f"   exit {code} in {rec['seconds']} s", flush=True)
    for line in tail[-3:]:
        print("   " + line, flush=True)
    return code == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rocm-results")
    ap.add_argument("--quick", action="store_true", help="only tests/test_rocm.py, test_kernel.py and codes")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per step")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch
    from truestar.kernel import rocm

    summary = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), platform=platform.platform(),
                   python=sys.version.split()[0], torch=torch.__version__, hip=rocm.hip_version(), steps=[])
    try:
        import triton
        summary["triton"] = triton.__version__
    except Exception as e:
        summary["triton"] = f"not importable: {e}"
    ok, why = rocm.hip_available()
    if not ok:
        summary["refused"] = why
        print(f"ROCm not available: {why}")
        with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        return 2
    t = rocm.current_target()
    summary["target"] = t.as_dict()
    summary["device_name"] = torch.cuda.get_device_name(0)
    print(f"AMD GPU: {summary['device_name']}, {t.arch}, wave {t.wave}, {t.cus} CUs, LDS {t.lds} bytes, "
          f"{t.memory / 2 ** 30:.0f} GiB, HIP {summary['hip']}, Triton {summary['triton']}", flush=True)

    py = sys.executable
    good = True
    if t.arch in rocm.TARGETS:
        good &= run("compile_matrix", [py, "scripts/rocm_compile_matrix.py", "--targets", t.arch,
                                       "--json", os.path.join(a.out, "compile_matrix.json")], a.out, summary, a.timeout)
    tests = QUICK_TESTS if a.quick else KERNEL_TESTS
    good &= run("kernel_tests", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
                                 "--junitxml", os.path.join(a.out, "kernel_tests.xml"), *tests],
                a.out, summary, a.timeout)
    good &= run("fingerprint", [py, "scripts/laptop_fingerprint.py", "--json", os.path.join(a.out, "fingerprint.json"),
                                "--dtypes", "float32,float16,bfloat16"], a.out, summary, a.timeout)
    help_text = subprocess.run([py, "-m", "truestar", "--help"], cwd=ROOT, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))).stdout
    if "fingerprint" in help_text:
        good &= run("truestar_fingerprint", [py, "-m", "truestar", "fingerprint"], a.out, summary, a.timeout)
    else:
        summary["truestar_fingerprint"] = "this version's CLI has no fingerprint command; laptop_fingerprint.py ran"
    for dt in ("float32", "float16"):
        good &= run(f"bench_matmul_{dt}", [py, "scripts/bench_matmul.py", "--dtype", dt, "--rounds", "5",
                                           "--shapes", BENCH_SHAPES, "--tools", "truestar,F.linear",
                                           "--json", os.path.join(a.out, f"bench_matmul_{dt}.json")],
                    a.out, summary, a.timeout)
    summary["passed"] = bool(good)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"{'all steps passed' if good else 'a step failed'}; summary in {os.path.join(a.out, 'summary.json')}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
