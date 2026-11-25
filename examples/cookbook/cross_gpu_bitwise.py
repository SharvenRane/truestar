"""Bitwise reproducible results across GPUs: read two fingerprint files and say which results match.

    python examples/cookbook/cross_gpu_bitwise.py        # CPU, about 2 s
    python examples/cookbook/cross_gpu_bitwise.py A.json B.json

`truestar fingerprint --json fp.json` on a GPU hashes the raw output bytes of a few fixed models
(same weight and input hashes everywhere) for Truestar's compiled executor in deterministic mode,
and for eager PyTorch as the baseline. This script compares two such files. Without arguments it
reads the two committed in docs/cross_chip/: an RTX 5070 Ti (Blackwell, Windows) and a Tesla T4
(Turing, Linux), both with Triton 3.7.1. No GPU is needed to compare. Recipe:
docs/cookbook/cross_gpu_bitwise.md.
"""

import os
import sys

from truestar.fingerprint import compare_files, load, match_table

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [os.path.join(HERE, "..", "..", "docs", "cross_chip", n)
           for n in ("fp2_rtx5070ti.json", "fp2_t4_triton371.json")]
CASES = ("mlp", "transformer", "llama_step")


def gpu(doc):
    e = doc["env"]
    return f"{e.get('gpu')} (cc {e.get('capability')}, {e.get('sm_count')} SMs, Triton {e.get('triton')})"


def main(argv):
    paths = argv[1:3] if len(argv) >= 3 else DEFAULT
    docs = [("A", load(paths[0])), ("B", load(paths[1]))]
    print(f"A: {gpu(docs[0][1])}\nB: {gpu(docs[1][1])}")
    cmp = compare_files(*paths)
    print("environment differences: " + ", ".join(d["key"] for d in cmp.env_diff))

    rows = {arm: [r for r in match_table(docs, arm=arm) if r["case"] in CASES and r["dtype"] == "float32"]
            for arm in ("truestar_det", "eager")}
    print(f"\n{'case':12} {'batch':>5}  {'Truestar compiled, A / B':28} {'equal':6} eager PyTorch equal")
    eager = {(r["case"], r["batch"]): r["all_equal"] for r in rows["eager"]}
    for r in rows["truestar_det"]:
        cells = " / ".join(r["cells"].get(n, "-") for n in ("A", "B"))
        print(f"{r['case']:12} {r['batch']:>5}  {cells:28} {str(r['all_equal']):6} {eager.get((r['case'], r['batch']))}")
    same = sum(r["all_equal"] for r in rows["truestar_det"])
    same_eager = sum(r["all_equal"] for r in rows["eager"])
    print(f"\nfloat32, {len(rows['truestar_det'])} rows: Truestar compiled bitwise equal on {same}, "
          f"eager PyTorch on {same_eager}")
    return 0 if rows["truestar_det"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
