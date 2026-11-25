"""Run truestar check over a small zoo of real architectures and summarize.

    python scripts/zoo_check.py --backends cpu,mps --json zoo.json

Models use random weights (no downloads). Exit code is 0 even when divergence is found, so CI
keeps the report; read the summary or the JSON for results.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import truestar  # noqa: E402


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 128)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, batch_first=True), num_layers=2,
            enable_nested_tensor=False)
        self.head = nn.Linear(128, 10)

    def forward(self, ids):
        return self.head(self.enc(self.emb(ids)).mean(dim=1))


def zoo():
    import torchvision as tv

    def vit():
        m = tv.models.vit_b_16(num_classes=10, image_size=64)
        nn.init.normal_(m.heads.head.weight, std=0.02)  # torchvision zero inits the head
        return m

    img = lambda s=64, n=4: torch.randn(n, 3, s, s)  # noqa: E731
    return {
        "resnet18": (lambda: tv.models.resnet18(num_classes=10), img),
        "mobilenet_v3_small": (lambda: tv.models.mobilenet_v3_small(num_classes=10), img),
        "efficientnet_b0": (lambda: tv.models.efficientnet_b0(num_classes=10), img),
        "convnext_tiny": (lambda: tv.models.convnext_tiny(num_classes=10), img),
        "vit_b_16": (vit, lambda: img(64, 2)),
        "tiny_transformer": (TinyTransformer, lambda: torch.randint(0, 1000, (4, 16))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=None)
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--only", default=None, help="comma separated model names")
    ap.add_argument("--repro-dir", default=None, help="write repro scripts for divergent backends")
    a = ap.parse_args()
    backends = a.backends.split(",") if a.backends else None
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backends": [b.as_dict() for b in truestar.available_backends()],
    }
    print(json.dumps(env, indent=2))
    rows, reports = [], {}
    models = zoo()
    if a.only:
        models = {k: v for k, v in models.items() if k in a.only.split(",")}
    for name, (mk, inp) in models.items():
        torch.manual_seed(0)
        model, x = mk().eval(), inp()
        t = time.time()
        rep = truestar.check(model, x, backends=backends, model_name=name,
                             repro_dir=os.path.join(a.repro_dir, name) if a.repro_dir else None)
        dt = time.time() - t
        print(rep.to_text(top=5))
        print(f"-- {name}: {rep.status} in {dt:.1f}s\n")
        reports[name] = rep.as_dict()
        for r in rep.results:
            rows.append({
                "model": name, "backend": r.backend.name, "status": r.status,
                "first_divergent": r.first_divergent.key if r.first_divergent else None,
                "out_rel_err": r.output.rel_err if r.output else None,
                "flips": r.flips, "decisions": r.decisions, "seconds": round(dt, 1),
                "error": r.error,
            })
    print("SUMMARY")
    for row in rows:
        print(f"  {row['model']:<20} {row['backend']:<6} {row['status']:<9} "
              f"out_err={row['out_rel_err'] if row['out_rel_err'] is None else format(row['out_rel_err'], '.1e')} "
              f"first={row['first_divergent']} {row['error'][:80]}")
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump({"env": env, "summary": rows, "reports": reports}, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
