"""Rotary angles at long positions in a float32 program: the error before and after float64 islands.

    PYTHONPATH=src python scripts/rope_long_positions.py [--positions 100,1000,2100,4000] [--json out.json]

A Llama style decoder (examples/llama_paged_decode.py) decodes one token at each position into a
cache that is empty but for that token, compiled for CUDA in deterministic float32, three ways:

* ``float64 trace``: the default import (``model.double()``), compiled in float32: every angle is
  computed in float32 (position times inverse frequency, then cos and sin).
* ``float32 trace``: the model imported in float32, angles in float32: the same arithmetic.
* ``float32 trace, rope_f64``: the model's angles, cos and sin declared float64 (``.double()``):
  a float64 island inside the float32 program (compiler/lower.py is_island); cos and sin are rounded
  to float32 once, where the model casts.

Each row gives the error relative to the eager model in float64 (largest absolute difference over
the largest value) of the logits and of the new token's rotated keys in the cache (the one cache row
the step writes, where the angle error shows undiluted), the angle error a float32 product makes at that position (bound), and
whether the program passed Truestar's own end to end check (the default verify).
"""

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="100,2100,4000,8000,16000,32000")
    ap.add_argument("--json")
    a = ap.parse_args()
    import torch
    import llama_paged_decode as P
    from truestar.compiler import compile_program
    from truestar.compiler.runtime import ProgramMismatchError

    positions = [int(x) for x in a.positions.split(",")]
    cap = max(positions) + 16
    rows = []
    for mode in ("float64 trace", "float32 trace", "float32 trace, rope_f64"):
        torch.manual_seed(0)
        c = P.Cfg(vocab=256, dim=128, heads=4, kv_heads=2, head_dim=32, layers=3, ffn=256, block=16,
                  rope_f64=mode.endswith("rope_f64"))
        model = P.Llama(c).eval()
        tdt = torch.float64 if mode == "float64 trace" else torch.float32
        ir, _ = P.import_steps(model, capacity=cap, max_capacity=max(4096, 2 * cap), trace_dtype=tdt)
        ref_model = P.DenseStep(copy.deepcopy(model)).double().cuda()
        loose = compile_program(ir, "cuda", "float32", deterministic=True, verify="off")
        checked = compile_program(ir, "cuda", "float32", deterministic=True)
        isl = [x for x in getattr(loose.plan, "islands", [])]
        for p in positions:
            ids = torch.tensor([[0], [11]], device="cuda")
            pos = torch.tensor([p], device="cuda")
            z = torch.zeros(c.layers, 2, c.kv_heads, cap, c.head_dim, device="cuda")
            out, kc, _ = loose([ids, pos, z, z.clone()])
            with torch.no_grad():
                ref, kref, _ = ref_model(ids, pos, z.double(), z.double())
            err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
            krow, kr = kc[:, :, :, p].double(), kref[:, :, :, p]        # the new token's rotated keys
            kerr = ((krow - kr).abs().max() / kr.abs().max()).item()
            try:
                checked([ids, pos, z, z.clone()])
                ok, why = True, ""
            except ProgramMismatchError as e:
                ok, why = False, str(e).splitlines()[1].strip()[:110]
            angle = p * float(model.inv_freq.max()) * 2 ** -24      # half an ulp of the float32 product
            row = {"mode": mode, "position": p, "relative_error": err, "key_row_relative_error": kerr, "end_to_end_check": ok, "why": why,
                   "angle_error_bound_rad": angle, "islands": len(isl)}
            rows.append(row)
            print(f"{mode:26} position {p:6d}: logits {err:.2e}, new key row {kerr:.2e} vs float64, "
                  f"check {'pass' if ok else 'FAIL'}"
                  f"{'' if ok else ' (' + why + ')'}; islands {len(isl)}", flush=True)
    # the case that failed in the prefix cache check (docs/PREFIX_CHECK.md): capacity 2200, zero ids
    for f64 in (False, True):
        torch.manual_seed(0)
        c = P.Cfg(vocab=256, dim=128, heads=4, kv_heads=2, head_dim=32, layers=3, ffn=256, block=16, rope_f64=f64)
        model = P.Llama(c).eval()
        ir, _ = P.import_steps(model, capacity=2200, max_capacity=8192,
                               trace_dtype=torch.float32 if f64 else torch.float64)
        cp = compile_program(ir, "cuda", "float32", deterministic=True)
        z = torch.zeros(c.layers, 2, c.kv_heads, 2200, c.head_dim, device="cuda")
        try:
            cp([torch.zeros(2, 1, dtype=torch.long, device="cuda"), torch.tensor([2100], device="cuda"), z, z.clone()])
            ok, why = True, ""
        except ProgramMismatchError as e:
            ok, why = False, str(e).splitlines()[1].strip()
        mode = "float32 trace, rope_f64" if f64 else "float64 trace"
        rows.append({"mode": mode, "case": "prefix check failure, capacity 2200, position 2100",
                     "end_to_end_check": ok, "why": why})
        print(f"{mode:26} capacity 2200 position 2100 (the prefix check case): check "
              f"{'pass' if ok else 'FAIL (' + why + ')'}", flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
