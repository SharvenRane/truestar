"""Bit level fingerprint of the compiled executor and of eager PyTorch CUDA on this machine.

    PYTHONPATH=src python scripts/laptop_fingerprint.py --json fingerprint.json

Run the same command on two machines and diff the JSON files: every sha256 is taken over the raw
bytes of a tensor, so equal hashes mean bit for bit equal results.

Models: a small MLP, a TransformerEncoderLayer and resnet18. Their weights are built on the CPU
after ``torch.manual_seed(0)``, saved to a file and loaded back from it, and the hash of every
parameter's bytes is recorded so the two machines can confirm they start from the same weights.
Inputs come from a CPU generator with a fixed seed and are hashed too.

``--dtypes float32,float16,bfloat16`` adds half precision cases (eager CUDA and the compiled
executor in that precision). Settings: float32 with TF32 off (matmul and cuDNN), ``torch.use_deterministic_algorithms(True)``,
cuDNN deterministic and not benchmarking, and the compiled executor built with
``deterministic=True, allow_tf32=False``. The compiled program is called twice; both calls must hash
the same. Eager CUDA and eager CPU outputs are hashed as well, with the largest absolute difference
between the compiled output and eager CUDA.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

BATCH = 4


def sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu().reshape(-1)          # 0 dim tensors (batch norm counters) too
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest()


def build(case: str):
    torch.manual_seed(0)
    if case == "mlp":
        m = nn.Sequential(nn.Linear(256, 512), nn.GELU(approximate="tanh"), nn.Linear(512, 512),
                          nn.ReLU(), nn.Linear(512, 10))
        return m, (256,)
    if case == "transformer":
        m = nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0, batch_first=True)
        return m, (32, 128)
    if case == "resnet18":
        import torchvision
        return torchvision.models.resnet18(), (3, 64, 64)
    raise ValueError(case)


def load_model(case: str, weights_dir: str):
    model, shape = build(case)
    path = os.path.join(weights_dir, f"{case}.pt")
    torch.save(model.state_dict(), path)
    fresh, _ = build(case)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    fresh.eval()
    weights = hashlib.sha256()
    for name, t in fresh.state_dict().items():
        weights.update(name.encode())
        weights.update(sha(t).encode())
    return fresh, shape, weights.hexdigest()


def outputs(x):
    if isinstance(x, torch.Tensor):
        return [x]
    return [t for t in x if isinstance(t, torch.Tensor)]


def run_case(case: str, weights_dir: str, dtype: str = "float32") -> dict:
    from truestar import ir
    from truestar.compiler import compile_program

    model, shape, weights_sha = load_model(case, weights_dir)
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(BATCH, *shape, generator=gen)
    name = case if dtype == "float32" else f"{case}_{dtype}"
    rec = {"case": name, "input_shape": list(x.shape), "weights_sha256": weights_sha, "input_sha256": sha(x)}
    dt = getattr(torch, dtype)

    with torch.no_grad():
        if dtype == "float32":       # CPU half kernels are not what this compares
            rec["eager_cpu"] = [sha(o) for o in outputs(model(x))]
        cuda_model = model.to("cuda", dt)
        xc = x.to("cuda", dt)
        eager = outputs(cuda_model(xc))
        torch.cuda.synchronize()
        rec["eager_cuda"] = [sha(o) for o in eager]
        rec["eager_cuda_again"] = [sha(o) for o in outputs(cuda_model(xc))]
        model = cuda_model.to("cpu", torch.float32)

    try:
        t0 = time.perf_counter()
        prog = ir.import_module(model, (torch.randn(2, *shape),),
                                dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
        cp = compile_program(prog, "cuda", dtype, allow_tf32=False, deterministic=True)
        first = outputs(cp([xc]))
        torch.cuda.synchronize()
        rec["compile_and_first_call_s"] = round(time.perf_counter() - t0, 3)
        second = outputs(cp([xc]))
        torch.cuda.synchronize()
        rec["compiled_cuda"] = [sha(o) for o in first]
        rec["compiled_cuda_again"] = [sha(o) for o in second]
        rec["compiled_run_to_run_equal"] = rec["compiled_cuda"] == rec["compiled_cuda_again"]
        rec["compiled_max_error_vs_float64"] = getattr(cp, "max_error", None)
        rec["compiled_vs_eager_cuda_max_abs"] = [float((a.float() - b.float()).abs().max())
                                                 for a, b in zip(first, eager)]
        rec["compiled_equals_eager_cuda_bitwise"] = rec["compiled_cuda"] == rec["eager_cuda"]
    except Exception as e:                                      # recorded, never hidden
        rec["compiled_error"] = f"{type(e).__name__}: {e}"[:2000]
    rec["eager_run_to_run_equal"] = rec["eager_cuda"] == rec["eager_cuda_again"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="fingerprint.json")
    ap.add_argument("--cases", nargs="*", default=["mlp", "transformer", "resnet18"])
    ap.add_argument("--dtypes", default="float32",
                    help="comma separated: float32 (TF32 off), float16, bfloat16; cases of other "
                         "dtypes are named <case>_<dtype>")
    ap.add_argument("--weights-dir", default=os.path.join(tempfile.gettempdir(), "truestar_fingerprint"))
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    os.makedirs(a.weights_dir, exist_ok=True)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import truestar
    props = torch.cuda.get_device_properties(0)
    env = {"gpu": props.name, "sm_count": props.multi_processor_count,
           "capability": f"{props.major}.{props.minor}", "driver_cuda": torch.version.cuda,
           "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
           "truestar": getattr(truestar, "__version__", None), "python": platform.python_version(),
           "cpu": platform.processor(), "platform": platform.platform()}
    try:
        import triton
        env["triton"] = triton.__version__
    except ImportError:
        env["triton"] = None
    results = []
    for dtype in a.dtypes.split(","):
        for case in a.cases:
            r = run_case(case, a.weights_dir, dtype)
            print(json.dumps(r), flush=True)
            results.append(r)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump({"env": env, "results": results}, f, indent=2)
    print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
