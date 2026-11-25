"""Download the public held out kernel sets (sources and labels only) used by
scripts/bench_heldout.py. Nothing here runs the downloaded code.

* KernelBench problems, levels 1 and 2 (MIT, github.com/ScalingIntelligence/KernelBench at a pinned
  commit) -> tests/data/heldout/kernelbench/problems/level{1,2}/
* KernelBench samples, baseline_eval (Apache 2.0, huggingface.co/datasets/ScalingIntelligence/
  kernelbench-samples): generated kernels with the published eval result, for the given models and
  levels -> tests/data/heldout/kernelbench/samples/<level>/<model>/problem_<id>.json
* METR KernelBenchFiltered (github.com/METR/KernelBenchFiltered, pinned commit): tasks and best agent
  solutions, levels 1 and 2. The repository has no license file, so these files go to
  results/heldout_metr/, which ignores itself and is never committed; --metr skips nothing else.

    python scripts/fetch_heldout.py [--models deepseek-r1,openai-o1] [--levels 1,2] [--metr]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_COMMIT = "b866d6ab57"  # 2025-01-10, the last change to level1 before the 2025-06 removals
KB_SHA = None  # resolved at run time from KB_COMMIT
METR_COMMIT = "9507be4154168be123b6bfe764aea2263625178e"  # the commit the YAKK post links
HF = "https://huggingface.co"
HF_DS = "ScalingIntelligence/kernelbench-samples"


def _get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "truestar-heldout-fetch"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


def _json(url: str):
    return json.loads(_get(url).decode("utf-8"))


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _gh_tree(repo: str, ref: str):
    return _json(f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1")["tree"]


def short_name(name, limit=48, keep=40):
    """Shorten a long upstream file name so deep Windows clones stay under the path limit."""
    if len(name) <= limit:
        return name
    stem = re.sub(r"_+", "_", name[:-3]).strip("_")
    return stem[:keep].rstrip("_") + ".py"


def fetch_kernelbench_problems(levels):
    repo = "ScalingIntelligence/KernelBench"
    sha = _json(f"https://api.github.com/repos/{repo}/commits/{KB_COMMIT}")["sha"]
    tree = _gh_tree(repo, sha)
    dst = os.path.join(ROOT, "tests", "data", "heldout", "kernelbench")
    n = 0
    names = {}
    for e in tree:
        p = e["path"]
        for lv in levels:
            pre = f"KernelBench/level{lv}/"
            if e["type"] == "blob" and p.startswith(pre) and p.endswith(".py"):
                raw = _get(f"https://raw.githubusercontent.com/{repo}/{sha}/{p}")
                name = p[len(pre):]
                short = short_name(name)
                if short != name:
                    names[f"level{lv}/{name}"] = short
                _write(os.path.join(dst, "problems", f"level{lv}", short), raw)
                n += 1
    _write(os.path.join(dst, "problems", "NAMES.json"),
           (json.dumps(names, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    _write(os.path.join(dst, "LICENSE"),
           _get(f"https://raw.githubusercontent.com/{repo}/{sha}/LICENSE"))
    print(f"KernelBench problems: {n} files at {sha}")
    return sha


def fetch_samples(levels, models):
    dst = os.path.join(ROOT, "tests", "data", "heldout", "kernelbench", "samples")
    n = 0
    for lv in levels:
        for m in models:
            base = f"baseline_eval/level{lv}/{m}"
            listing = _json(f"{HF}/api/datasets/{HF_DS}/tree/main/{base}")
            for d in listing:
                if d["type"] != "directory":
                    continue
                pid = d["path"].rsplit("_", 1)[-1]
                out = os.path.join(dst, f"level{lv}", m, f"problem_{pid}.json")
                if os.path.exists(out):
                    continue
                try:
                    raw = _get(f"{HF}/datasets/{HF_DS}/resolve/main/{d['path']}/sample_0/kernel.json")
                except Exception as exc:
                    print("skip", d["path"], exc)
                    continue
                _write(out, raw)
                n += 1
    print(f"KernelBench samples: {n} new files")


def fetch_metr(levels):
    repo = "METR/KernelBenchFiltered"
    tree = _gh_tree(repo, METR_COMMIT)
    dst = os.path.join(ROOT, "results", "heldout_metr")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, ".gitignore"), "w") as f:
        f.write("# METR/KernelBenchFiltered has no license file: kept local, never committed\n*\n")
    n = 0
    names = {}
    for e in tree:
        p = e["path"]
        if e["type"] != "blob" or not p.endswith(".py"):
            continue
        for lv in levels:
            if p.startswith(f"tasks/level_{lv}/") or p.startswith(f"best_agent_solutions/level_{lv}/"):
                _write(os.path.join(dst, p),
                       _get(f"https://raw.githubusercontent.com/{repo}/{METR_COMMIT}/{p}"))
                n += 1
    print(f"METR: {n} files at {METR_COMMIT}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="deepseek-r1,openai-o1")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--metr", action="store_true")
    ap.add_argument("--no-kernelbench", action="store_true")
    a = ap.parse_args(argv)
    levels = [int(x) for x in a.levels.split(",")]
    if not a.no_kernelbench:
        fetch_kernelbench_problems(levels)
        fetch_samples(levels, [m for m in a.models.split(",") if m])
    if a.metr:
        fetch_metr(levels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
