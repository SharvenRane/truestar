"""One transformer encoder layer (d_model 1024, 16 heads, ffn 4096, 8 x 512 tokens) compiled by
Truestar three ways: float32 with no tensor cores (the bitwise portable baseline), half precision
on plain tensor cores (deterministic=True), and half precision with portable_tensor_cores=True.
Median milliseconds per call, the output hash and the error against float64.

    PYTHONPATH=src python scripts/tcexact_layer_bench.py --json out/layer.json
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn as nn
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from truestar import ir  # noqa: E402
from truestar.compiler import compile_program  # noqa: E402
from truestar.fingerprint.models import portable_init, portable_normal  # noqa: E402


def sha(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    model = portable_init(nn.TransformerEncoderLayer(1024, 16, 4096, dropout=0.0, batch_first=True), 5).eval()
    x = portable_normal(6, 0, (args.batch, args.tokens, 1024))
    with torch.no_grad():
        ref = copy.deepcopy(model).double()(x.double())
    p = torch.cuda.get_device_properties(0)
    doc = {"gpu": p.name, "capability": f"{p.major}.{p.minor}", "shape": [args.batch, args.tokens, 1024],
           "runs": []}
    prog = ir.import_module(model, (x[:2].clone(),), dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},),
                            name="layer")
    arms = [("float32_fma", "float32", dict(deterministic=True, allow_tf32=False)),
            ("tf32_plain", "float32", dict(deterministic=True, allow_tf32=True)),
            ("tf32_portable", "float32", dict(allow_tf32=True, portable_tensor_cores=True))]
    for dt in ("float16", "bfloat16"):
        arms += [(f"{dt}_plain", dt, dict(deterministic=True)),
                 (f"{dt}_portable", dt, dict(portable_tensor_cores=True))]
    for name, dt, kw in arms:
        t0 = time.time()
        try:
            cp = compile_program(prog, "cuda", dt, **kw)
            xs = [x.to("cuda", getattr(torch, dt))]
            out = cp(xs)[0]
            torch.cuda.synchronize()
            ms = triton.testing.do_bench(lambda: cp(xs), warmup=5, rep=50)
            err = float(((out.double().cpu() - ref).abs().max() / ref.abs().max()))
            row = {"arm": name, "ms": ms, "sha": sha(out), "max_err_over_max": err,
                   "compile_s": time.time() - t0, "again_equal": sha(cp(xs)[0]) == sha(out)}
        except Exception as e:   # recorded, never hidden
            row = {"arm": name, "error": f"{type(e).__name__}: {str(e)[:600]}"}
        doc["runs"].append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(doc, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
