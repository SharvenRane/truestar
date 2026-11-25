"""Before / after table from two scripts/bench_perfmm.py JSON files.

    python scripts/perfmm_table.py results/perfmm/mm_before.json results/perfmm/mm_after.json
"""

import json
import sys


def main():
    before, after = (json.load(open(p)) for p in sys.argv[1:3])
    print("| M x N x K | cuBLAS F.linear | det before | det after | det x | default before | default after | default x | det err before / after | bits |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b, a in zip(before["rows"], after["rows"]):
        assert b["shape"] == a["shape"]
        cub = a["median_us"]["F.linear"]
        db, da = b["median_us"]["lin_det"], a["median_us"]["lin_det"]
        fb, fa = b["median_us"]["lin_def"], a["median_us"]["lin_def"]
        same = all(a["run_to_run_equal"].values())
        bits = a["bits_equal_to_nk"]
        print(f"| {'x'.join(map(str, a['shape']))} | {cub:.1f} | {db:.1f} ({cub / db:.2f}) | {da:.1f} ({cub / da:.2f}) "
              f"| {db / da:.2f}x | {fb:.1f} ({cub / fb:.2f}) | {fa:.1f} ({cub / fa:.2f}) | {fb / fa:.2f}x "
              f"| {b['error']['lin_det']:.1e} / {a['error']['lin_det']:.1e} "
              f"| {'runs equal' if same else 'RUNS DIFFER'}, lin_det=det_nk {bits.get('lin_det')} |")


if __name__ == "__main__":
    main()
