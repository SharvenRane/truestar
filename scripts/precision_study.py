"""Per layer precision plans against uniform baselines on four models trained in code.

Models (nothing downloaded, every one trained here for seconds on a synthetic task so its
decisions have real margins):

* MLP: OnePassNorm on raw readings near 30 (the planted sensitive layer: float32 is fine, 16 bit
  floats lose the variance), Linear 1024 -> 4096 -> 4096 -> 100 with GELU.
* Llama style decoder (truestar.rlkernels.llama: RMSNorm, rotary, grouped query attention,
  SwiGLU), dim 512, 4 layers, hidden 1536, vocab 512, trained on the Markov language of
  scripts/quant_study.py.
* torchvision resnet18 (10 classes) and a torchvision VisionTransformer (64 x 64 images, patch 8,
  6 layers, width 384) trained on synthetic class templates with noise.

Per model and budget: the plan (`truestar.precplan.plan`) and the baselines all fp32, all tf32,
all fp16, all bf16, fp16 except norms and torch.autocast fp16, each with flips against float64 on
the calibration and held out inputs and GPU time per forward (CUDA graph replay).

    PYTHONPATH=src python scripts/precision_study.py --out precision_study.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truestar.precplan import plan  # noqa: E402


class OnePassNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x * x).mean(-1, keepdim=True) - mean * mean
        return (x - mean) * torch.rsqrt(var.clamp_min(0) + 1e-5)


class MLP(nn.Module):
    def __init__(self, d=1024, h=4096, classes=100):
        super().__init__()
        self.norm = OnePassNorm()
        self.fc1, self.act1 = nn.Linear(d, h), nn.GELU()
        self.fc2, self.act2 = nn.Linear(h, h), nn.GELU()
        self.head = nn.Linear(h, classes)

    def forward(self, x):
        return self.head(self.act2(self.fc2(self.act1(self.fc1(self.norm(x))))))


def mlp_data(n, seed, d=1024, classes=100):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, d, generator=torch.Generator().manual_seed(7))
    y = torch.randint(0, classes, (n,), generator=g)
    return 30 + 0.3 * (centers[y] * 0.1 + torch.randn(n, d, generator=g)), y


def image_data(n, seed, classes=10, size=64):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(classes, 3, 8, 8, generator=torch.Generator().manual_seed(11))
    templates = nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    y = torch.randint(0, classes, (n,), generator=g)
    x = templates[y] * 0.12 + torch.randn(n, 3, size, size, generator=g)
    return x, y


def fit(m, data_fn, steps, lr, bs=128, seed=0):
    torch.manual_seed(seed)
    m = m.cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time()
    for s in range(steps):
        x, y = data_fn(bs, 1000 + s)
        loss = nn.functional.cross_entropy(m(x.cuda()), y.cuda())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    x, y = data_fn(1024, 99)
    with torch.no_grad():
        acc = float((m.eval()(x.cuda()).argmax(1).cpu() == y).float().mean())
    print(f"  trained {type(m).__name__} {steps} steps in {time.time() - t0:.0f} s, accuracy {acc:.3f}",
          file=sys.stderr)
    return m.cpu().eval(), acc


def build(name):
    if name == "mlp":
        m, acc = fit(MLP(), mlp_data, 300, 1e-3, bs=256)
        cal = [(mlp_data(1024, 1)[0],), (mlp_data(1024, 2)[0],)]
        hold = [(mlp_data(1024, 3)[0],), (mlp_data(1024, 4)[0],)]
        return m, cal, hold, acc
    if name == "llama":
        from quant_study import markov_corpus, train
        from truestar.rlkernels import llama as L

        cfg = L.Config(vocab=512, dim=512, layers=4, heads=8, kv_heads=4, hidden=1536, max_len=256)
        m = train(cfg, 800).cpu().eval()
        data = markov_corpus(cfg.vocab, 32, 257, seed=5)
        cal = [(data[0:8, :256],), (data[8:16, :256],)]
        hold = [(data[16:24, :256],), (data[24:32, :256],)]
        with torch.no_grad():
            lg = m.cuda()(data[16:32, :256].cuda())
            acc = float((lg.argmax(-1).cpu() == data[16:32, 1:257]).float().mean())
        return m.cpu(), cal, hold, acc
    if name == "resnet18":
        import torchvision

        m, acc = fit(torchvision.models.resnet18(num_classes=10), image_data, 300, 1e-3)
        cal = [(image_data(256, 1)[0],), (image_data(256, 2)[0],)]
        hold = [(image_data(256, 3)[0],), (image_data(256, 4)[0],)]
        return m, cal, hold, acc
    if name == "vit":
        from torchvision.models.vision_transformer import VisionTransformer

        vit = VisionTransformer(image_size=64, patch_size=8, num_layers=6, num_heads=6,
                                hidden_dim=384, mlp_dim=1536, num_classes=10)
        m, acc = fit(vit, image_data, 400, 3e-4)
        cal = [(image_data(256, 1)[0],), (image_data(256, 2)[0],)]
        hold = [(image_data(256, 3)[0],), (image_data(256, 4)[0],)]
        return m, cal, hold, acc
    raise ValueError(name)


def row(m):
    h = m["held_out"]
    lo, hi = m["latency_range_us"]
    return (f"| {m['name']} | {m['latency_us']:.0f} ({lo:.0f} to {hi:.0f}) | "
            f"{m['eager_us']:.0f} | {m['calibration']['flips']} | {h['flips'] if h else '-'} | "
            f"{m['calibration']['rel_err'] if m['calibration']['rel_err'] is not None else 'inf'} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,llama,resnet18,vit")
    ap.add_argument("--budgets", default="0,0.5%")
    ap.add_argument("--out", default="precision_study.json")
    a = ap.parse_args()
    torch.set_num_threads(4)
    results = []
    for name in a.models.split(","):
        print(f"== {name}", file=sys.stderr)
        m, cal, hold, acc = build(name)
        for b in a.budgets.split(","):
            t = time.time()
            p = plan(m, cal, hold, max_flips=b, model_name=name)
            d = p.as_dict()
            d["accuracy"] = acc
            d["wall_s"] = time.time() - t
            results.append(d)
            print(p.to_text(top=12), file=sys.stderr)
            print(f"\n### {name}, max flips {b} (accuracy {acc:.3f}, planning {d['seconds']['total']:.1f} s, "
                  f"{d['verification_runs']} verification runs, plan {d['counts']})\n")
            print("| config | GPU us per forward (graph, range) | eager us | flips cal | flips held out | rel err |")
            print("|---|---|---|---|---|---|")
            print(row(d["result"]))
            for bl in d["baselines"]:
                print(row(bl))
            sys.stdout.flush()
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1, default=str)
        del m
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
