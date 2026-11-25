"""Summarize a TRUESTAR_BIAS_LOG file: how close every tested kernel output came to the lean test's
two conditions (docs/BIAS.md, false positive survey).

    TRUESTAR_BIAS_LOG=lean.jsonl python -m pytest tests/test_kernel.py ...
    python scripts/bias_survey.py lean.jsonl [more.jsonl ...]
    python scripts/bias_survey.py --zoo cpu,cuda     # truestar check over scripts/zoo_check.py models

Prints, over every output tested: how many fired, the largest effect ratio |mean| / floor, the
smallest p, and the largest margin (the smaller of the two ratios, 1 means firing), with the
kernels closest to firing.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter


def main(paths) -> int:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        print("no rows")
        return 1
    fired = [r for r in rows if r["biased"]]
    tested = [r for r in rows if r["leans"]]
    eff = [max(abs(l["mean"]) / r["floor"] for l in r["leans"]) for r in tested]
    sig = [min(l["p"] for l in r["leans"]) for r in tested]
    print(f"outputs tested: {len(tested)} ({len(rows)} rows), kernels: {len({r['kernel'] for r in rows})}, "
          f"backends: {dict(Counter(r['backend'] for r in rows))}")
    print(f"fired: {len(fired)}")
    for r in fired:
        print(f"  FIRED {r['kernel']} on {r['backend']}: " + ", ".join(
            f"{l['direction']} mean {l['mean']:.3g} p {l['p']:.2g}" for l in r["leans"]))
    print(f"largest |mean| / floor: {max(eff):.3g}; p below 1e-6 on {sum(p < 1e-6 for p in sig)} outputs; "
          f"smallest p: {min(sig):.3g}")
    clean = sorted((r for r in tested if not r["biased"]), key=lambda r: -r["margin"])
    print(f"largest margin among outputs that did not fire: {clean[0]['margin']:.3g}" if clean else "")
    for r in clean[:8]:
        l = max(r["leans"], key=lambda z: abs(z["mean"]))
        print(f"  {r['margin']:.3f}  {r['kernel']} ({r['backend']}, n {r['n']}): {l['direction']} mean "
              f"{l['mean']:.3g} vs floor {r['floor']:.3g}, p {l['p']:.2g} over {l['groups']} groups "
              f"({l['grouping']}){' [' + r['note'] + ']' if r['note'] else ''}")
    big = [e for e in eff if e > 1]
    print(f"outputs over the effect floor (significance not reached): {len(big)}")
    return 0


def zoo(backends: str) -> int:
    """The lean test of truestar check (TS961) over the zoo: every replayed module's margin."""
    import os
    import time
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    import truestar
    from zoo_check import zoo as models

    rows, fired = [], 0
    for name, (build, inputs) in models().items():
        torch.manual_seed(0)
        t = time.time()
        rep = truestar.check(build().eval(), inputs(), backends=backends.split(","), model_name=name, drill_ops=False)
        for r in rep.results:
            fired += "TS961" in r.codes
            for l in r.layers:
                if l.bias is not None:
                    rows.append((l.bias["margin"], name, r.backend.name, l.key, l.type_name, l.bias))
        print(f"{name}: {rep.status}, codes {rep.codes}, {time.time() - t:.1f}s", flush=True)
    rows.sort(key=lambda x: -x[0])
    print(f"module outputs tested: {len(rows)}; backends with TS961: {fired}")
    for m, name, be, key, typ, b in rows[:8]:
        l = max(b["leans"], key=lambda z: abs(z["mean"]))
        print(f"  {m:.3f}  {name} {be} {key} ({typ}): {l['direction']} mean {l['mean']:.3g} vs floor "
              f"{b['floor']:.3g}, p {l['p']:.2g} over {l['groups']} groups ({l['grouping']})")
    return 1 if fired else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--zoo"]:
        sys.exit(zoo(sys.argv[2] if len(sys.argv) > 2 else "cpu"))
    sys.exit(main(sys.argv[1:]))
