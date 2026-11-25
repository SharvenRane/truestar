"""Compile every GPU loop shape for AMD targets without an AMD GPU (Triton's AMD backend).

    PYTHONPATH=src python scripts/rocm_compile_matrix.py --targets gfx942,gfx1100,gfx1103 --json out.json
    PYTHONPATH=src python scripts/rocm_compile_matrix.py --candidates     # every tuning candidate too

For each case of truestar.kernel.rocm_cases and each target: the loop shape, the default launch
config planned with the target's CUs and LDS, and per compiled function the wave size, lanes per
program, LDS bytes, VGPRs, spills and scratch, or the code that refused it. Nothing runs: every
result is compiled only (TS885). Exit 1 when a default config does not compile.
"""

import argparse
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_rocm_"))

from truestar.kernel import rocm, rocm_cases  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(rocm.TARGETS))
    ap.add_argument("--cases", help="comma separated case names (default: all)")
    ap.add_argument("--candidates", action="store_true", help="also build every tuning candidate")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ok, why = rocm.compile_available()
    if not ok:
        print(f"AMD compile path unavailable: {why}")
        return 2
    want = set(a.cases.split(",")) if a.cases else None
    rows, failed = [], 0
    for arch in a.targets.split(","):
        for name, k, shapes, kw in rocm_cases.cases():
            if want and name not in want:
                continue
            t0 = time.perf_counter()
            row = dict(target=arch, case=name)
            try:
                b = rocm.build(k, arch, shapes, **kw)
                row.update(b.as_dict())
                row.pop("kernel", None)
                f = b.functions[0]
                print(f"{arch:8} {name:34} {b.plan_kind:11} {str(b.config):92} wave {b.target.wave} "
                      f"lanes {f.lanes:4} lds {f.lds:6} vgpr {f.vgprs:3} spill {f.vgpr_spills} scratch {f.scratch}"
                      f"{' +splitk' if len(b.functions) > 1 else ''}")
            except Exception as e:
                failed += 1
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"{arch:8} {name:34} FAILED {row['error'][:300]}")
            row["seconds"] = round(time.perf_counter() - t0, 2)
            if a.candidates:
                cands = []
                for cfg, res in rocm.build_candidates(k, arch, shapes, **kw):
                    if isinstance(res, Exception):
                        cands.append(dict(config=cfg, refused=f"{type(res).__name__}: {str(res)[:300]}"))
                    else:
                        f = res.functions[0]
                        cands.append(dict(config=cfg, lds=f.lds, vgprs=f.vgprs, scratch=f.scratch))
                row["candidates"] = cands
                n_bad = sum("refused" in c for c in cands)
                print(f"         candidates: {len(cands)} built, {n_bad} refused"
                      + "".join(f"\n           refused {c['config']}: {c['refused'][:200]}" for c in cands if "refused" in c))
            rows.append(row)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows), fh, indent=1, default=str)
    print(f"{len(rows) - failed} of {len(rows)} default launches compiled")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
