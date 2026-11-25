"""Rewrite the rows of tests/data/contraction_defaults_70sm.json that the deterministic K chunks
(triton_backend.det_k_chunk, 2026-11-25) changed: strict float32 deterministic rows with K above
DET_K_MIN. Every other row must still match the current rules; the script stops if one does not.

    PYTHONPATH=src python scripts/regen_contraction_table.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from test_sm_aware import TABLE, plan  # noqa: E402
from truestar.kernel import triton_backend as tb  # noqa: E402


def main():
    tb.device_limits = lambda: (tb.REF_SMS, tb.REF_SMEM, tb.REF_MEMORY)
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    changed = 0
    for key, (default, cands) in table.items():
        variant, shape = key.split(" ")
        M, N, Kd = (int(v) for v in shape.split("x"))
        p = plan(N, Kd, variant.split("-")[0], tf32="tf32" in variant, det=variant.endswith("-det"))
        env = {p.m.size.name: M}
        new = [tb.default_config(p, env), tb.candidates(p, env)]
        if new != [default, cands]:
            if not p.kchunk:
                sys.exit(f"{key} changed but has no K chunks: not a change of this branch")
            table[key] = new
            changed += 1
            print(f"{key}: {default} -> {new[0]}")
    TABLE.write_text(json.dumps(table, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{changed} rows rewritten")


if __name__ == "__main__":
    main()
