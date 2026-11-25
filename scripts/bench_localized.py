"""False fail and false pass of `truestar check` scoring, before and after localized scoring.

    python scripts/bench_localized.py --json after.json
    python scripts/bench_localized.py --old --json before.json   # aggregate only (the old scoring)

Clean set: the zoo of scripts/zoo_check.py plus the clean models of tests/test_check.py, each in
float32, float16 and bfloat16, on cpu and on the first accelerator. A clean run that is not OK is a
false fail. Fault set: the existing DeviceFault kinds (tests/test_check.py) and the localized ones
(one row in 1024, one element, the last column, a small value region) on a 1024 x 1024 layer, in
every dtype, injected on the accelerator (DeviceFault by device) and on the CPU (`lowp`). A fault
run is a false pass when the backend is not flagged at the expected level or another module is
blamed. `--old` scores with the whole tensor relative error alone, exactly as before 2026-11-24.

Also prints the clean margin: the largest local error over its local limit, and the largest
aggregate error over its threshold, over every replayed module and the final output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import truestar  # noqa: E402
from truestar import compare  # noqa: E402
from truestar.testing import DeviceFault  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class MLP(nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fault = fault if fault is not None else nn.Identity()
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.fc2(self.fault(self.act(self.fc1(x))))


class Wide(nn.Module):
    """A 1024 x 1024 layer whose first 32 channels are 32 times smaller than the rest."""

    def __init__(self, fault=None):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        g = torch.ones(1024)
        g[:32] = 1.0 / 32.0
        self.register_buffer("gain", g)
        self.fault = fault if fault is not None else nn.Identity()
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        return self.fc2(self.fault(self.fc1(x) * self.gain))


class TwoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)
        self.a = nn.Linear(8, 3)
        self.b = nn.Linear(8, 2)

    def forward(self, x):
        h = torch.tanh(self.shared(x))
        h = torch.tanh(self.shared(h))
        return self.a(h), {"aux": self.b(h)}


class SmallTransformer(nn.Module):
    def __init__(self, vocab=1000, d=128, heads=4, ff=256, out=10):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True), num_layers=2,
            enable_nested_tensor=False)
        self.head = nn.Linear(d, out)

    def forward(self, ids):
        return self.head(self.enc(self.emb(ids)).mean(dim=1))


class InplaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3)
        self.a1 = nn.SiLU(inplace=True)
        self.c2 = nn.Conv2d(8, 8, 3)
        self.a2 = nn.ReLU(inplace=True)
        self.a3 = nn.Hardswish(inplace=True)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        x = self.a3(self.a2(self.c2(self.a1(self.c1(x)))))
        return self.head(x.mean(dim=(2, 3)))


def clean_models():
    import torchvision as tv

    def vit():
        m = tv.models.vit_b_16(num_classes=10, image_size=64)
        nn.init.normal_(m.heads.head.weight, std=0.02)
        return m

    img = lambda s=64, n=4: torch.randn(n, 3, s, s)  # noqa: E731
    return {
        "resnet18": (lambda: tv.models.resnet18(num_classes=10), img),
        "mobilenet_v3_small": (lambda: tv.models.mobilenet_v3_small(num_classes=10), img),
        "efficientnet_b0": (lambda: tv.models.efficientnet_b0(num_classes=10), img),
        "convnext_tiny": (lambda: tv.models.convnext_tiny(num_classes=10), img),
        "vit_b_16": (vit, lambda: img(64, 2)),
        "tiny_transformer": (SmallTransformer, lambda: torch.randint(0, 1000, (4, 16))),
        "mlp": (MLP, lambda: torch.randn(32, 16)),
        "two_heads": (TwoHeads, lambda: torch.randn(4, 8)),
        "transformer_small": (lambda: SmallTransformer(50, 32, 4, 64, 7),
                              lambda: torch.randint(0, 50, (4, 12))),
        "inplace_net": (InplaceNet, lambda: torch.randn(4, 3, 16, 16)),
        "wide": (Wide, lambda: torch.randn(1024, 256)),
    }


def fault_cases(dev: str):
    """(name, model factory, input factory, dtype key, expected status) per fault."""
    cases = []
    for dk in DTYPES:
        hard = dk != "fp32"  # half precision: completely wrong parts; float32: 5 percent wrong
        loc = {"row": -1.0 if hard else 0.05, "element": -1.0 if hard else 0.05,
               "last_col": -1.0 if hard else 0.05, "region": -2.0 if hard else 0.05}
        for kind, amount in loc.items():
            fr = 517 if kind == "row" else 32
            cases.append((f"wide_{kind}", lambda k=kind, a=amount, f=fr: Wide(DeviceFault(dev, k, a, f)),
                          lambda: torch.randn(1024, 256), dk, compare.DIVERGENT))
        for kind, amount, exp in [("scale", 0.5, compare.DIVERGENT), ("zero", 0.0, compare.DIVERGENT),
                                  ("nan", 0.0, compare.DIVERGENT), ("rows", 0.5, compare.DIVERGENT)]:
            cases.append((f"mlp_{kind}", lambda k=kind, a=amount: MLP(DeviceFault(dev, k, a, 20)),
                          lambda: torch.randn(32, 16), dk, exp))
        if dk == "fp32":
            cases.append(("mlp_scale_0.02", lambda: MLP(DeviceFault(dev, "scale", 0.02, 20)),
                          lambda: torch.randn(32, 16), dk, compare.SUSPECT))
    return cases


def _ratios(r, dtype):
    """Largest local error over its limit and aggregate over its threshold, with where."""
    best = {"local": (0.0, ""), "aggregate": (0.0, "")}

    def see(d, thr, name):
        if d is None or thr <= 0 or d.shape_mismatch or d.nonfinite_mismatch:
            return
        lr = d.local_err / compare.local_limit(thr)
        ar = d.rel_err / thr
        if lr > best["local"][0]:
            best["local"] = (lr, f"{name} {d.local_at}")
        if ar > best["aggregate"][0]:
            best["aggregate"] = (ar, name)

    for l in r.layers:
        if l.replayed and l.isolated is not None and (l.is_leaf or True):
            see(l.isolated, l.threshold, l.key)
    if r.output is not None:
        see(r.output, compare.threshold_for(dtype, r.backend.tf32, None), "output")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", action="store_true", help="aggregate scoring only (before)")
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--only", default=None, help="clean,faults")
    ap.add_argument("--models", default=None, help="comma separated clean model names")
    ap.add_argument("--dtypes", default="fp32,fp16,bf16")
    a = ap.parse_args()
    if a.old:
        compare.Diff.local_status = lambda self, threshold: compare.OK
    accel = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
    backends = ["cpu"] + ([accel.name] if accel else [])
    accel_type = torch.device(accel.device).type if accel else None
    parts = a.only.split(",") if a.only else ["clean", "faults"]
    dks = a.dtypes.split(",")
    out = {"scoring": "old" if a.old else "new", "backends": backends, "torch": torch.__version__,
           "clean": [], "faults": []}

    if "clean" in parts:
        models = clean_models()
        if a.models:
            models = {k: v for k, v in models.items() if k in a.models.split(",")}
        for name, (mk, inp) in models.items():
            for dk in dks:
                torch.manual_seed(0)
                m, x = mk().eval().to(DTYPES[dk]), inp()
                t = time.time()
                rep = truestar.check(m, x, backends=backends, drill_ops=False)
                for r in rep.results:
                    rat = _ratios(r, DTYPES[dk])
                    row = {"model": name, "dtype": dk, "backend": r.backend.name, "status": r.status,
                           "first": r.first_divergent.key if r.first_divergent else None,
                           "local_margin": rat["local"][0], "local_where": rat["local"][1],
                           "agg_margin": rat["aggregate"][0], "agg_where": rat["aggregate"][1],
                           "flips": r.flips - r.tie_flips, "seconds": round(time.time() - t, 1)}
                    out["clean"].append(row)
                    print(f"clean {name:<20} {dk} {r.backend.name:<5} {r.status:<9} "
                          f"local {row['local_margin']:.2f} ({row['local_where'][:60]}) "
                          f"agg {row['agg_margin']:.2f} first={row['first']}", flush=True)

    if "faults" in parts:
        targets = [("cpu", "lowp")] + ([(accel.name, accel_type)] if accel else [])
        for bname, dev in targets:
            for name, mk, inp, dk, exp in fault_cases(dev):
                if dk not in dks:
                    continue
                torch.manual_seed(0)
                m, x = mk().eval().to(DTYPES[dk]), inp()
                rep = truestar.check(m, x, backends=[bname], drill_ops=False)
                r = rep.results[0]
                fd = r.first_divergent
                blamed = fd.name if fd is not None else None
                if exp == compare.SUSPECT:
                    caught = r.status in (compare.SUSPECT, compare.DIVERGENT)
                else:
                    caught = r.status == exp
                ok = caught and blamed == "fault"
                row = {"case": name, "dtype": dk, "backend": bname, "expected": exp,
                       "status": r.status, "blamed": blamed, "caught": ok,
                       "message": next((i.message for i in r.issues if i.code in ("TS010", "TS011")), "")}
                out["faults"].append(row)
                print(f"fault {name:<16} {dk} {bname:<5} expect {exp:<9} got {r.status:<9} "
                      f"blamed={blamed} {'CAUGHT' if ok else 'MISSED'}", flush=True)

    clean, faults = out["clean"], out["faults"]
    if clean:
        ff = [c for c in clean if c["status"] != compare.OK]
        out["false_fail"] = f"{len(ff)} of {len(clean)}"
        top = max(clean, key=lambda c: c["local_margin"])
        print(f"\nfalse fail: {len(ff)} of {len(clean)} clean runs")
        print(f"largest clean local margin {top['local_margin']:.3f}: {top['model']} {top['dtype']} "
              f"{top['backend']} {top['local_where']}")
        topa = max(clean, key=lambda c: c["agg_margin"])
        print(f"largest clean aggregate margin {topa['agg_margin']:.3f}: {topa['model']} {topa['dtype']} "
              f"{topa['backend']} {topa['agg_where']}")
        for c in ff:
            print("  FALSE FAIL", c)
    if faults:
        fp = [f for f in faults if not f["caught"]]
        out["false_pass"] = f"{len(fp)} of {len(faults)}"
        print(f"false pass: {len(fp)} of {len(faults)} fault runs")
        for f in fp:
            print(f"  MISSED {f['case']} {f['dtype']} {f['backend']}: {f['status']} blamed={f['blamed']}")
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
