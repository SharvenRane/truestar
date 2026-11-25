"""End to end: models compiled by Truestar for Metal against float64 and against MPS eager.

For each model and precision: the compiled executor on "metal" (every kernel and region checked
against float64 on its first run), the same model in eager PyTorch on MPS, and the float64 CPU
reference. Prints and writes (--json) the relative error of both against float64, the compile
time, and the median time per call of both after warmup (speed is provisional: hosted runners are
virtual machines with a paravirtual GPU).

    PYTHONPATH=src python scripts/metal_e2e.py --json metal-e2e.json
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn as nn


def _rel(out, ref):
    out, ref = out.detach().cpu().double(), ref.detach().cpu().double()
    return (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


class TinyTransformer(nn.Module):
    """Two pre norm encoder layers and a head: a small transformer."""

    def __init__(self, d=64, heads=4, ff=128, layers=2, classes=10):
        super().__init__()
        self.inp = nn.Linear(16, d)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, activation="gelu",
                                       norm_first=True) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def models():
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10),
                        nn.Softmax(-1)).eval()
    torch.manual_seed(1)
    tr = TinyTransformer().eval()
    torch.manual_seed(2)
    tr128 = TinyTransformer(d=128, heads=4, ff=256).eval()
    return [("mlp", mlp, (32, 64)), ("transformer", tr, (4, 32, 16)), ("transformer_s128", tr128, (4, 128, 16))]


def timeit(fn, reps):
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def run(reps):
    from truestar import ir
    from truestar.compiler import compile_program

    rows = []
    for name, model, shape in models():
        for dtype in ("float32", "float16"):
            tdt = getattr(torch, dtype)
            B = torch.export.Dim("B", min=1, max=256)
            prog = ir.import_module(model.float().eval(), (torch.randn(*shape),), dynamic_shapes=({0: B},))
            t0 = time.perf_counter()
            cp = compile_program(prog, "metal", dtype)
            x = torch.randn(*shape)
            y = cp([x])[0]                                  # compiles kernels, checks against float64
            torch.mps.synchronize()
            first = time.perf_counter() - t0
            with torch.no_grad():
                ref = model.double()(x.double())
                model.float()
                eager_model = model.to("mps").to(tdt)
                xm = x.to(tdt).to("mps")
                eager = eager_model(xm)
            row = {
                "model": name, "dtype": dtype, "shape": list(shape),
                "metal_rel_err": _rel(y, ref), "mps_eager_rel_err": _rel(eager, ref),
                "metal_vs_eager": _rel(y, eager),
                "first_call_s": round(first, 3),
                "verify_s": round(cp.verify_seconds, 3),
                "kernels": [k["shape"] for k in cp.report.kernels],
                "fallbacks": len(cp.report.fallbacks), "library": len(cp.report.library),
            }
            xm32 = x.to("mps")
            with torch.no_grad():
                row["metal_ms"] = round(1e3 * timeit(lambda: cp([xm32]), reps), 3)
                row["mps_eager_ms"] = round(1e3 * timeit(lambda: eager_model(xm), reps), 3)
            model.float().cpu()
            print(json.dumps(row))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    rows = run(a.reps)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
