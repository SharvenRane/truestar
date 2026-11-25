"""Native from_pretrained against transformers and against the torch.export import path.

    PYTHONPATH=src python scripts/bench_models_native.py [--compiled cpu|cuda] [--json out.json]

For each family transformers knows: a tiny random model (built from a config in code, nothing
downloaded) saved with save_pretrained, then

* the worst relative error of the native logits (reference interpreter, float64) and of four
  decode steps through the KV cache against transformers in float64 (its float32 casts kept exact,
  as in tests/test_hf_models.py);
* the time to build the native model from the folder and to compile it, against the time
  ``truestar.ir.import_module`` (torch.export) takes on the transformers model;
* with ``--compiled``: the native prefill on Truestar kernels in float32 (CompiledExecutor, which
  checks its first run against the float64 reference) and its worst error against transformers.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import test_models_native as T  # noqa: E402

from truestar import ir, models  # noqa: E402


def one(name, compiled):
    m = T._hf(name)
    d = tempfile.mkdtemp(prefix=f"ts_{name}_")
    m.save_pretrained(d, safe_serialization=True)
    ids = T._ids(2, 8, seed=5)
    want = T._want(m, ids)
    t0 = time.perf_counter()
    lm = models.from_pretrained(d)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = lm(ids)
    t_first = time.perf_counter() - t0
    row = {"family": name, "params": lm.model.num_params(), "load_s": round(t_load, 3),
           "compile_and_first_call_s": round(t_first, 3), "prefill_rel": T._rel(got, want)}
    cache = lm.init_cache(2, 8)
    lg, cache = lm.step(ids[:, :4], torch.arange(4), cache)
    worst = T._rel(lg, want[:, :4])
    for t in range(4, 8):
        lg, cache = lm.step(ids[:, t:t + 1], torch.tensor([t]), cache)
        worst = max(worst, T._rel(lg[:, 0], want[:, t]))
    row["decode_rel"] = worst

    class W(torch.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.m = mm

        def forward(self, x):
            return (self.m(input_ids=x, use_cache=False).logits,)
    t0 = time.perf_counter()
    try:
        ir.import_module(W(copy.deepcopy(m)), (ids,))
        row["torch_export_import_s"] = round(time.perf_counter() - t0, 3)
    except Exception as e:   # noqa: BLE001  (reported, not hidden)
        row["torch_export_import_s"] = f"failed: {type(e).__name__}: {str(e)[:120]}"
    if compiled:
        from truestar.compiler import CompiledExecutor
        ex = CompiledExecutor(compiled, "float32")
        lm32 = models.from_pretrained(d, dtype="f32", executor=ex)
        t0 = time.perf_counter()
        out = lm32(ids.to(compiled) if compiled == "cuda" else ids)
        row["compiled_first_call_s"] = round(time.perf_counter() - t0, 3)
        row["compiled_f32_rel"] = T._rel(out.cpu(), want)
        t0 = time.perf_counter()
        for _ in range(5):
            out = lm32(ids.to(compiled) if compiled == "cuda" else ids)
        if compiled == "cuda":
            torch.cuda.synchronize()
        row["compiled_call_ms"] = round((time.perf_counter() - t0) / 5 * 1e3, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    import transformers
    rows = []
    for name in a.only or T.FAMILIES:
        r = one(name, a.compiled)
        rows.append(r)
        print(json.dumps(r))
    print(f"transformers {transformers.__version__}, torch {torch.__version__}; skipped (not in this transformers): "
          f"{T.SKIPPED}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"transformers": transformers.__version__, "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
