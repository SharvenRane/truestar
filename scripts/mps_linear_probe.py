"""Characterize a matmul / linear divergence on a backend against a float64 CPU reference.

    python scripts/mps_linear_probe.py --device mps
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform

import torch
import torch.nn.functional as F


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    n = torch.linalg.vector_norm(b).item()
    return torch.linalg.vector_norm(a - b).item() / (n if n else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--json", dest="json_path")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)
    print(platform.platform(), "torch", torch.__version__, "device", dev)
    rows = []
    batches = [(4,), (1,), (64,), (4, 16), (2, 17)]
    dims = [(512, 10), (128, 256), (256, 128), (768, 3072), (576, 1024), (1024, 10), (64, 64), (16, 32)]
    for lead, (k, n) in itertools.product(batches, dims):
        x = torch.randn(*lead, k)
        w = torch.randn(n, k) / k ** 0.5
        b = torch.randn(n)
        ref_lin = F.linear(x.double(), w.double(), b.double())
        ref_mm = x.double() @ w.double().t()
        cases = {
            "linear_bias": lambda: F.linear(x.to(dev), w.to(dev), b.to(dev)),
            "linear_nobias": lambda: F.linear(x.to(dev), w.to(dev)),
            "matmul_wT": lambda: x.to(dev) @ w.to(dev).t(),
            "matmul_wT_contig": lambda: x.to(dev) @ w.t().contiguous().to(dev),
        }
        refs = {"linear_bias": ref_lin, "linear_nobias": ref_mm, "matmul_wT": ref_mm,
                "matmul_wT_contig": ref_mm}
        if x.ndim == 2:
            cases["addmm"] = lambda: torch.addmm(b.to(dev), x.to(dev), w.to(dev).t())
            refs["addmm"] = ref_lin
        for name, fn in cases.items():
            try:
                out = fn()
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                err = rel(out, refs[name])
            except Exception as exc:  # noqa: BLE001
                err = float("nan")
                name = f"{name} ERROR {type(exc).__name__}: {str(exc)[:60]}"
            rows.append({"lead": list(lead), "k": k, "n": n, "op": name, "rel_err": err})
            flag = "  <-- WRONG" if err > 1e-3 else ""
            print(f"x{tuple(lead) + (k,)} w({n},{k}) {name:<18} rel_err {err:.2e}{flag}")
    bad = [r for r in rows if r["rel_err"] > 1e-3]
    print(f"\n{len(bad)} of {len(rows)} cases exceed 1e-3")
    if a.json_path:
        with open(a.json_path, "w") as f:
            json.dump({"platform": platform.platform(), "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
