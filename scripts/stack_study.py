"""Compilers and runtimes as check backends: false alarms, injected faults, overhead, baselines.

Run in an environment with the stacks installed (here the cv venv: torch 2.11, TensorRT 10.16,
onnxruntime-gpu 1.29):

    PYTHONPATH=src python scripts/stack_study.py clean      # resnet18, TransformerEncoderLayer
    PYTHONPATH=src python scripts/stack_study.py faults     # one injected fault per backend
    PYTHONPATH=src python scripts/stack_study.py polygraphy # Polygraphy on the TensorRT fault

Overhead is `truestar check` wall time minus the stack alone (export or torch.compile, build,
one run) on the same model, both warm (each measured twice, the second kept).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
warnings.filterwarnings("ignore")

from truestar.check import _cast_inputs, check  # noqa: E402
from truestar.stacks import probe  # noqa: E402

import stack_models as S  # noqa: E402

BACKENDS = ["cuda+inductor", "ort:cpu", "ort:cuda", "trt:fp32", "trt:fp16"]


def _models():
    import torchvision

    torch.manual_seed(0)
    r18 = torchvision.models.resnet18(weights=None).eval()
    enc = torch.nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True).eval()
    return [("resnet18", r18, torch.randn(4, 3, 224, 224)),
            ("TransformerEncoderLayer", enc, torch.randn(4, 64, 256))]


def stack_alone(name, model, x) -> float:
    b, why = probe(name)
    t = time.perf_counter()
    r = b.make_runner()
    a = _cast_inputs((x,), "cpu", torch.float32)
    e = r.compile(model, a, {}, whole_model=True)
    e(a, {})
    e.close()
    r.close()
    return time.perf_counter() - t


def clean() -> None:
    print("| model | backend | status | output rel err | check s | stack alone s | overhead s |")
    print("|---|---|---|---|---|---|---|")
    for mname, model, x in _models():
        for be in BACKENDS:
            b, why = probe(be)
            if b is None:
                print(f"| {mname} | {be} | unavailable: {why} | | | | |")
                continue
            for _ in range(2):
                ta = stack_alone(be, model, x)
                t = time.perf_counter()
                rep = check(model, x, backends=[be])
                tc = time.perf_counter() - t
            r = rep.results[0]
            err = r.output.rel_err if r.output is not None else float("nan")
            print(f"| {mname} | {be} | {r.status} {' '.join(r.codes)} | {err:.1e} | {tc:.2f} | "
                  f"{ta:.2f} | {tc - ta:+.2f} |", flush=True)


FAULTS = [
    ("cuda+inductor", "Inductor sigmoid lowering x1.05", lambda: (S.Net(), torch.randn(8, 16)),
     None, "inductor"),
    ("ort:cpu", "exporter: gelu as tanh approximation", lambda: (S.GeluMLP(), torch.randn(32, 16) * 2),
     S.onnx_gelu_ignores_approximate, None),
    ("ort:cuda", "Det has no CUDA kernel (silent CPU fallback)",
     lambda: (S.DetNet(), torch.randn(4, 6, 8)), None, None),
    ("trt:fp32", "LayerNorm forced to FP16 in an FP32 engine",
     lambda: (S.ShiftNorm(), torch.randn(8, 16)), S.trt_norm_in_fp16, None),
    ("trt:fp16", "hand written RMSNorm overflows float16",
     lambda: (S.RmsNet(), torch.randn(64, 64)), None, None),
]


def faults() -> None:
    print("| backend | fault | status | codes | blamed module | blamed op | check s | compiles |")
    print("|---|---|---|---|---|---|---|---|")
    for be, label, make, opts, special in FAULTS:
        torch.manual_seed(0)
        model, x = make()
        model.eval()
        ctx = S.inductor_wrong_sigmoid() if special == "inductor" else _null()
        with ctx:
            t = time.perf_counter()
            rep = check(model, x, backends=[be],
                        stack_options={be: opts()} if opts else None)
            tc = time.perf_counter() - t
        r = rep.results[0]
        fd = r.first_divergent.key if r.first_divergent is not None and r.layers and len(
            r.layers) > 1 else "-"
        op = r.first_divergent_op.op if r.first_divergent_op is not None else "-"
        k = getattr(r.first_divergent_op, "kernel", None) if r.first_divergent_op else None
        if k:
            op += f" in {k['kernel']}"
        print(f"| {be} | {label} | {r.status} | {' '.join(r.codes)} | {fd} | {op} | {tc:.2f} | "
              f"{r.stack.get('compiles')} |", flush=True)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def polygraphy() -> None:
    """Polygraphy's documented accuracy workflow on the ShiftNorm model: compare TensorRT with ORT
    marking every tensor as an output. Its layer hook cannot be injected through the CLI, so this
    measures time and what the per layer comparison costs on the clean engine, and whether
    marking outputs changes the engine (layer count with and without)."""
    torch.manual_seed(0)
    model, x = S.ShiftNorm().eval(), torch.randn(8, 16)
    d = tempfile.mkdtemp(prefix="truestar_poly_")
    path = os.path.join(d, "m.onnx")
    torch.onnx.export(model, (x,), path, dynamo=True, verbose=False)
    exe = os.path.join(os.path.dirname(sys.executable), "polygraphy")
    for mark in ([], ["--trt-outputs", "mark", "all", "--onnx-outputs", "mark", "all"]):
        t = time.perf_counter()
        p = subprocess.run([exe, "run", path, "--trt", "--onnxrt"] + mark, capture_output=True,
                           text=True)
        print(f"polygraphy {'mark all' if mark else 'outputs only'}: exit {p.returncode}, "
              f"{time.perf_counter() - t:.2f} s")
        tail = [l for l in (p.stdout + p.stderr).splitlines() if "PASSED" in l or "FAILED" in l]
        print("  " + " | ".join(tail[-2:]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "clean"
    {"clean": clean, "faults": faults, "polygraphy": polygraphy}[what]()
