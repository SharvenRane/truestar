"""Compile every torchvision classifier with the compiled executor and count what is not a
Truestar kernel (library kernels, fallbacks), with the reasons.

    PYTHONPATH=src python scripts/conv_zoo.py                 # every classifier, compile and one checked run
    PYTHONPATH=src python scripts/conv_zoo.py --no-run resnet18 mobilenet_v2

Random weights, no downloads, eval mode, a symbolic batch. Each model is imported at batch 2 and
the input size export_zoo.py uses (64 pixels, 224 for vision transformers and swin, 299 for
inception_v3), compiled for the GPU in float32, and (unless --no-run) run once at batch 2: that run
is the built in float64 check of every kernel and of the whole program, and its error against a
float64 run of the PyTorch model is printed. Models the importer cannot take are listed with the
reason and counted apart.
"""

import argparse
import json
import sys
import time

import torch
import torchvision

SIZE_224 = ("vit_", "swin", "maxvit")
SIZE_299 = ("inception_v3",)
MAX_PARAMS = 100_000_000


def input_size(name):
    if name.startswith(SIZE_224):
        return 224
    if name in SIZE_299:
        return 299
    return 64


def one(name, run=True):
    from truestar import ir
    from truestar.compiler import compile_program
    torch.manual_seed(0)
    kw = {"aux_logits": False, "init_weights": False} if name in ("inception_v3", "googlenet") else {}
    model = torchvision.models.get_model(name, **kw).eval()
    if sum(p.numel() for p in model.parameters()) > MAX_PARAMS:
        return {"model": name, "status": "skipped", "reason": "over 100M parameters"}
    hw = input_size(name)
    x = torch.randn(2, 3, hw, hw)
    try:
        B = torch.export.Dim("B", min=1, max=64)
        prog = ir.import_module(model, (x,), dynamic_shapes=({0: B},))
    except Exception as e:  # the importer's reason is the row
        return {"model": name, "status": "import", "reason": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}
    t = time.perf_counter()
    cp = compile_program(prog, "cuda")
    rep = cp.report
    row = {"model": name, "status": "compiled", "compile_s": round(time.perf_counter() - t, 2),
           "kernels": len(rep.kernels), "library": [(e["ops"][0], e.get("reason")) for e in rep.library],
           "fallbacks": [(e["ops"][0], e["reason"]) for e in rep.fallbacks]}
    shapes = {}
    for e in rep.kernels:
        shapes[e["shape"]] = shapes.get(e["shape"], 0) + 1
    row["shapes"] = shapes
    if run:
        try:
            out = cp([x])[0]
            ref = model.double()(x.double())
            row["err"] = ((out.double().cpu() - ref).abs().max() / ref.abs().max()).item()
            row["verified"] = bool(cp.verified)
        except Exception as e:
            row["status"] = "run"
            row["reason"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    names = a.names or torchvision.models.list_models(module=torchvision.models)
    rows = []
    for n in names:
        r = one(n, run=not a.no_run)
        rows.append(r)
        print(json.dumps(r), flush=True)
        torch.cuda.empty_cache()
    done = [r for r in rows if r["status"] == "compiled"]
    clean = [r for r in done if not r["library"] and not r["fallbacks"]]
    print(f"imported and compiled: {len(done)} of {len(rows)}; zero library kernels and zero fallbacks: "
          f"{len(clean)}; verified run: {sum(1 for r in done if r.get('verified'))}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
