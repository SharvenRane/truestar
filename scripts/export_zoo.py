"""Export every torchvision classifier with truestar export and report which ones round trip.

Run: PYTHONPATH=src python scripts/export_zoo.py [name ...]   (CPU only, random weights, no downloads)

For each model: export at batch 2 with a symbolic batch (this includes the float64 checks against
PyTorch at batch 2 and 7), write the artifact, load it back, run it at batch 3 through the runner
and compare with eager PyTorch float32. Models the importer cannot take are listed with the reason.
"""

import io
import os
import sys
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch
import torchvision

from truestar.export import ExportError, export, load

SIZE_224 = ("vit_", "swin", "maxvit")
SIZE_299 = ("inception_v3",)
MAX_PARAMS = 100_000_000  # float64 reference runs of bigger models take minutes on a CPU


def input_size(name):
    if name.startswith(SIZE_224):
        return 224
    if name in SIZE_299:
        return 299
    return 64


def rows_identical(model, x) -> bool:
    with torch.no_grad():
        y = model(x)
    return bool(((y - y[:1]).abs().max() <= 1e-9 * y.abs().max().clamp_min(1e-30)).item())


def measure_batch_norm(model, size, batches=4):
    """Batch norm statistics measured on random inputs (cumulative average), then eval mode.

    torchvision initializes googlenet's convolutions with std 0.01; with the default running
    statistics (mean 0, var 1) the activations shrink to zero through its 57 convolutions and every
    input gives the same output, so no export check can see an input dependent error. Measured
    statistics put each layer back near unit variance; only the buffers change, not the weights."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(batches):
            model(torch.randn(8, 3, size, size))
    model.eval()
    return len(bns)


def main(names):
    torch.manual_seed(0)
    names = names or torchvision.models.list_models(module=torchvision.models)
    out_dir = tempfile.mkdtemp()
    rows = []
    for name in names:
        t0 = time.time()
        try:
            model = torchvision.models.get_model(name, weights=None).eval()
        except Exception as e:
            rows.append((name, "SKIP", f"cannot build: {e}", 0.0))
            continue
        n_params = sum(p.numel() for p in model.parameters())
        if n_params > MAX_PARAMS:
            rows.append((name, "SKIP", f"{n_params / 1e6:.0f}M params, above the sweep limit", 0.0))
            continue
        with torch.no_grad():  # zero initialized heads (ViT) make every output 0 and every check vacuous
            for prm in model.parameters():
                if prm.dim() >= 2 and prm.abs().max() == 0:
                    prm.normal_(0, 0.02)
        s = input_size(name)
        x = torch.randn(2, 3, s, s)
        note = ""
        if rows_identical(model, x) and measure_batch_norm(model, s):
            note = "; batch norm statistics measured on random inputs (input independent outputs otherwise)"
        path = os.path.join(out_dir, f"{name}.tsx")
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            res = export(model, x, path)
            run = load(path)
            x3 = torch.randn(3, 3, s, s)
            with torch.no_grad():
                want = model(x3)
            got = run(x3)[0]
            err = ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
            checks = res.evidence["checks"]
            detail = (f"rel err f64 {max(c['max_rel_error'] for c in checks):.1e}, runner f32 vs eager {err:.1e}, "
                      f"{os.path.getsize(path) / 1e6:.1f} MB{note}")
            status = "PASS" if err < 1e-4 else "FAIL"
            if res.evidence["warnings"]:
                status, detail = "WEAK", detail + "; " + res.evidence["warnings"][0]
        except ExportError as e:
            msg = str(e)
            ops = sorted({ln.split("unsupported op ")[1].split(" ")[0] for ln in msg.splitlines()
                          if "unsupported op " in ln})
            detail = ("unsupported ops: " + ", ".join(ops)) if ops else msg.splitlines()[0][:150]
            status = "REFUSED"
        except Exception as e:
            status, detail = "ERROR", f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"
        finally:
            sys.stdout = stdout
            if os.path.exists(path):
                os.remove(path)
        rows.append((name, status, detail, time.time() - t0))
        print(f"{name:28s} {status:8s} {time.time() - t0:6.1f}s  {detail}", flush=True)
    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    print("\nsummary:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for r in rows:
        if r[1] == "SKIP":
            print(f"{r[0]:28s} SKIP     {r[2]}")


if __name__ == "__main__":
    main(sys.argv[1:])
