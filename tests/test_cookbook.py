"""Every cookbook recipe and switching guide example runs on the CPU in under a minute and prints its
verdict (docs/cookbook/, docs/SWITCHING.md). Each script runs in its own process with the GPU hidden."""

import os
import re
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DOCS = os.path.join(ROOT, "docs")

# script -> a line its output must contain
SCRIPTS = {
    "cookbook/verify_float64.py": "verified on the first call: True",
    "cookbook/compare_chips.py": "predictions differ between the two chips",
    "cookbook/deterministic_training.py": "equal to the uninterrupted run True",
    "cookbook/cross_gpu_bitwise.py": "Truestar compiled bitwise equal on 12, eager PyTorch on 0",
    "cookbook/quantize_safely.py": "the bfloat16 scale is caught",
    "cookbook/catch_lying_runtime.py": "computes this float32 file in float16",
    "cookbook/speculative_decoding.py": "returned the greedy tokens bit for bit",
    "cookbook/linear_attention.py": "chunk size as a reduction order",
    "cookbook/near_tie.py": "the device flips only near ties",
    "cookbook/memory_recompute.py": "bitwise equal to the plain step: losses True, parameters True",
    "switching/train_native.py": "validation accuracy",
    "switching/infer_native.py": "verified against float64 on the first call: True",
    "switching/import_once.py": "fine tuned 30 steps",
}
LIMIT_S = 60.0


def _run(script):
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "examples", script)], cwd=ROOT, env=env,
                       capture_output=True, encoding="utf-8", errors="replace", timeout=2 * LIMIT_S)
    return r, time.perf_counter() - t0


def test_every_script_is_listed():
    found = {f"{d}/{f}" for d in ("cookbook", "switching")
             for f in os.listdir(os.path.join(ROOT, "examples", d)) if f.endswith(".py")}
    assert found == set(SCRIPTS)


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_script_runs_on_cpu_in_under_a_minute(script):
    r, seconds = _run(script)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert SCRIPTS[script] in r.stdout, r.stdout[-3000:]
    assert "Warning" not in r.stderr, r.stderr[-3000:]
    assert seconds < LIMIT_S, f"{script} took {seconds:.1f} s"


def test_every_recipe_page_points_at_a_script_and_is_in_the_nav():
    pages = sorted(f for f in os.listdir(os.path.join(DOCS, "cookbook")) if f.endswith(".md"))
    with open(os.path.join(ROOT, "mkdocs.yml"), encoding="utf-8") as f:
        nav = f.read()
    for page in pages:
        if page == "index.md":
            continue
        with open(os.path.join(DOCS, "cookbook", page), encoding="utf-8") as f:
            text = f.read()
        script = f"cookbook/{page[:-3]}.py"
        assert script in SCRIPTS and f"examples/{script}" in text, page
        assert f"cookbook/{page}" in nav, page
    assert len(pages) - 1 == sum(s.startswith("cookbook/") for s in SCRIPTS)


def test_every_nav_entry_exists():
    with open(os.path.join(ROOT, "mkdocs.yml"), encoding="utf-8") as f:
        entries = re.findall(r":\s+([\w./-]+\.md)\s*$", f.read(), flags=re.M)
    assert entries
    missing = [e for e in entries if not os.path.exists(os.path.join(DOCS, e))]
    assert not missing, missing
