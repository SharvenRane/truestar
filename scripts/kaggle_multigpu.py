"""Build the private Kaggle kernel that runs the real multi GPU checks on GPU T4 x2 (two T4s, NCCL).

    python scripts/kaggle_multigpu.py OUT_DIR [--slug truestar-multigpu-t4x2] [--owner sharvenrane]
    kaggle kernels push -p OUT_DIR
    kaggle kernels pull OWNER/SLUG -p CHECK_DIR -m      # check "is_private": true before trusting it

The kernel embeds this checkout (src, the script, the example and the tests it runs) as an xz tarball,
runs scripts/multigpu_verify.py on both GPUs, the `truestar parallel --ranks devices` command line,
and the real rank tests, and leaves multigpu_t4.json in /kaggle/working. `machine_shape`
NvidiaTeslaT4 is Kaggle's GPU T4 x2 accelerator. The kernel is always private.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILES = ["scripts/multigpu_verify.py", "examples/parallel_model.py", "tests/parallel_models.py",
         "tests/parallel_models_file.py", "tests/test_multigpu.py", "tests/test_parallel.py",
         "tests/test_codes_unique.py", "pyproject.toml"]

RUNNER = r'''
os.chdir("/kaggle/working/ts")
env = dict(os.environ, PYTHONPATH="src:tests", TRUESTAR_GIT_COMMIT=COMMIT, PYTHONIOENCODING="utf-8")
def sh(*a, timeout=None):
    print("$", " ".join(a), flush=True)
    try:
        r = subprocess.run(list(a), env=env, timeout=timeout)
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print("EXIT", code, flush=True)
    return code
sh("nvidia-smi")
sh(sys.executable, "-c", "import torch, torch.distributed as d; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], d.is_nccl_available(), torch.cuda.nccl.version())")
sh(sys.executable, "scripts/multigpu_verify.py", "--out", "/kaggle/working/multigpu_t4.json", timeout=5400)
for lay in ("tp=2", "tp=2,ga=2,sp=1", "dp=2,ga=2", "dp=2,shard=2"):
    sh(sys.executable, "-m", "truestar", "parallel", "examples/parallel_model.py", "--layout", lay, "--ranks", "devices",
       "--optimizer", "adamw", "--clip", "0.5", timeout=900)
sh(sys.executable, "-m", "truestar", "parallel", "examples/parallel_model.py", "--layout", "tp=2,sp=1", "--ranks",
   "devices", "--inject", "missing_tp_sync", "--format", "agent", timeout=900)
sh(sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfEs", "tests/test_multigpu.py",
   "tests/test_codes_unique.py", timeout=2400)
import shutil
shutil.rmtree("/kaggle/working/ts", ignore_errors=True)
'''


def build(out: str, owner: str, slug: str) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "src", "truestar")):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(dirpath, fn)
                tar.add(full, arcname=os.path.relpath(full, ROOT).replace("\\", "/"))
        for rel in FILES:
            tar.add(os.path.join(ROOT, rel), arcname=rel)
    blob = base64.b64encode(buf.getvalue()).decode("ascii")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    os.makedirs(out, exist_ok=True)
    code = (f"import base64, io, os, subprocess, sys, tarfile\nCOMMIT = {commit!r}\n"
            "os.makedirs('/kaggle/working/ts', exist_ok=True)\n"
            f"tarfile.open(fileobj=io.BytesIO(base64.b64decode({blob!r})), mode='r:xz')"
            ".extractall('/kaggle/working/ts', filter='data')\n" + RUNNER)
    with open(os.path.join(out, f"{slug}.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    meta = {"id": f"{owner}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--owner", default="sharvenrane")
    ap.add_argument("--slug", default="truestar-multigpu-t4x2")
    a = ap.parse_args()
    print(build(a.out, a.owner, a.slug))


if __name__ == "__main__":
    main()
