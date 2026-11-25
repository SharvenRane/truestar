"""CNNs fully in Truestar kernels against cuDNN (PyTorch eager) and torch.compile.

    PYTHONPATH=src python scripts/bench_conv.py                          # every model and precision
    PYTHONPATH=src python scripts/bench_conv.py --models resnet18 --modes fp16 --batches 64
    PYTHONPATH=src python scripts/bench_conv.py --json out.json

Models (torchvision, random weights, batch norm statistics randomized, eval mode, 224 x 224):
resnet18, mobilenet_v3_small and mobilenet_v2 (the depthwise heavy one: 17 depthwise 3 x 3
convolutions). Precisions: fp32 (IEEE float32 everywhere: cuDNN's TF32 is turned off for eager and
torch.compile), tf32 (TF32 allowed on every side), fp16 (model and input in float16).

Each (model, precision) runs in one process. For each batch size every tool is warmed up (Truestar
compiles and verifies once, tunes launch configs that are worth it; torch.compile compiles a graph
per batch size), then the tools are timed interleaved: 7 rounds of 20 synchronized calls each,
median of the rounds. The graph row is the same Truestar program captured once into a CUDA graph
(truestar.runtime.capture, docs/RUNTIME.md), which removes the host launch cost. Each Truestar program is checked against the float64 reference on its first
call, as always; the error against a float64 run of the PyTorch model is printed too. The compile
report gives the number of library kernels left (torch kernels Truestar did not generate).

The GPU is shared with other jobs, so numbers move by up to about 10% between runs.
"""

import argparse
import json
import statistics
import sys
import time

import torch

MODELS = ("resnet18", "mobilenet_v3_small", "mobilenet_v2")
MODES = ("fp32", "tf32", "fp16")
BATCHES = (1, 8, 64)


def make_model(name):
    import torchvision
    torch.manual_seed(0)
    m = getattr(torchvision.models, name)().eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.running_var.uniform_(0.5, 1.5)
    return m


def set_tf32(on: bool):
    torch.backends.cudnn.allow_tf32 = on
    torch.backends.cuda.matmul.allow_tf32 = on


def time_calls(fn, calls=20):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / calls * 1e3


def run(model_name, mode, batches, rounds=7, tools=("eager", "compile", "truestar")):
    from truestar import ir
    from truestar.compiler import compile_program
    set_tf32(mode == "tf32")
    dtype = torch.float16 if mode == "fp16" else torch.float32
    model = make_model(model_name)
    ref_model = make_model(model_name).double()
    model = model.to("cuda", dtype)
    res = {"model": model_name, "mode": mode, "batches": {}}
    t = time.perf_counter()
    B = torch.export.Dim("B", min=1, max=64)
    prog = ir.import_module(make_model(model_name), (torch.randn(2, 3, 224, 224),), dynamic_shapes=({0: B},))
    cp = compile_program(prog, "cuda", "float16" if mode == "fp16" else "float32", allow_tf32=(mode == "tf32"))
    rep = cp.report
    res["truestar_compile_s"] = time.perf_counter() - t
    res["truestar_kernels"] = len(rep.kernels)
    res["library_kernels"] = len(rep.library)
    res["fallbacks"] = len(rep.fallbacks)
    res["shapes"] = {}
    for e in rep.kernels:
        res["shapes"][e["shape"]] = res["shapes"].get(e["shape"], 0) + 1
    compiled = torch.compile(model) if "compile" in tools else None
    for bs in batches:
        x = torch.randn(bs, 3, 224, 224, generator=torch.Generator().manual_seed(bs))
        xd = x.to("cuda", dtype)
        fns = {}
        with torch.no_grad():
            if "eager" in tools:
                fns["eager"] = lambda: model(xd)
            if "compile" in tools:
                t = time.perf_counter()
                compiled(xd)
                torch.cuda.synchronize()
                res.setdefault("compile_first_s", {})[bs] = time.perf_counter() - t
                fns["compile"] = lambda: compiled(xd)
            if "truestar" in tools:
                t = time.perf_counter()
                out = cp([xd])[0]
                torch.cuda.synchronize()
                first = time.perf_counter() - t
                ref = ref_model(x.double())
                err = ((out.double().cpu() - ref).abs().max() / ref.abs().max()).item()
                res.setdefault("truestar_first_s", {})[bs] = first
                res.setdefault("truestar_err", {})[bs] = err
                fns["truestar"] = lambda: cp([xd])
            if "graph" in tools:     # Truestar's kernels replayed from one CUDA graph (truestar.runtime)
                from truestar.runtime import capture
                step = capture(lambda t: cp([t])[0], (xd,))
                fns["graph"] = lambda: step(xd)
            for f in fns.values():
                for _ in range(3):
                    f()
            times = {k: [] for k in fns}
            for _ in range(rounds):
                for k, f in fns.items():
                    times[k].append(time_calls(f))
        res["batches"][bs] = {k: statistics.median(v) for k, v in times.items()}
        print(model_name, mode, bs, {k: round(v, 3) for k, v in res["batches"][bs].items()},
              "err", res.get("truestar_err", {}).get(bs), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--tools", default="eager,compile,truestar,graph")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = []
    for m in a.models.split(","):
        for mode in a.modes.split(","):
            r = run(m, mode, [int(b) for b in a.batches.split(",")], a.rounds, tuple(a.tools.split(",")))
            print(json.dumps({k: v for k, v in r.items() if k != "batches"}), flush=True)
            out.append(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
