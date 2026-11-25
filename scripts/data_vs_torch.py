"""truestar.data.Loader against torch.utils.data.DataLoader on the same synthetic data (CPU).

Measures samples per second for a plain epoch (no per sample work) and an augmented epoch (pad 4,
random crop, random flip), and checks determinism: the sha256 of every batch of an epoch across
worker counts and modes, and across two separate processes.

    python scripts/data_vs_torch.py [--n 50000] [--out results/data_loader/data_vs_torch.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from truestar.data import ArraySource, Loader, RandomCropFlip  # noqa: E402

B = 256


def make(n):
    g = np.random.default_rng(0)
    return g.standard_normal((n, 3, 32, 32)).astype(np.float32), (np.arange(n) % 10).astype(np.int64)


class TorchAug(Dataset):
    """The same augmentation for torch's loader, drawing from torch's per worker generator."""

    def __init__(self, x, y, pad=4):
        self.x, self.y, self.pad = x, y, pad

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img, p = self.x[i], self.pad
        c, h, w = img.shape
        dy, dx = (int(v) for v in torch.randint(0, 2 * p + 1, (2,)))
        flip = int(torch.randint(0, 2, ()))
        padded = np.zeros((c, h + 2 * p, w + 2 * p), dtype=img.dtype)
        padded[:, p:p + h, p:p + w] = img
        out = padded[:, dy:dy + h, dx:dx + w]
        if flip:
            out = out[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(out)), int(self.y[i])


class PerSample:
    """RandomCropFlip without its batch form: the per sample path, one call per sample."""

    def __init__(self):
        self.t = RandomCropFlip(4)

    def __call__(self, sample, key):
        return self.t(sample, key)


def digest(batches):
    h = hashlib.sha256()
    for b in batches:
        for t in b:
            h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rate(loader, n, epochs=2):
    """Samples per second of the last epoch (the first one starts the workers)."""
    best = None
    for _ in range(epochs):
        t = time.perf_counter()
        for _b in loader:
            pass
        best = time.perf_counter() - t
    return n / best


def ts_loader(x, y, aug, workers=0, mode="thread"):
    t = {False: None, True: RandomCropFlip(4), "per_sample": PerSample()}[aug]
    return Loader(ArraySource(x, y), B, seed=0, transform=t,
                  workers=workers, mode=mode)


def torch_loader(x, y, aug, workers=0):
    ds = TorchAug(x, y) if aug else TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    g = torch.Generator().manual_seed(0)
    return DataLoader(ds, B, shuffle=True, num_workers=workers, generator=g,
                      persistent_workers=workers > 0)


def digest_only(n):
    x, y = make(n)
    print(digest(ts_loader(x, y, True)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--out", default=os.path.join("results", "data_loader", "data_vs_torch.json"))
    ap.add_argument("--digest-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.digest_only:
        return digest_only(a.n)
    x, y = make(a.n)
    rows = []

    def add(name, kind, aug, workers, mode, ld):
        r = rate(ld, a.n)
        case = {False: "plain", True: "augment", "per_sample": "augment_per_sample"}[aug]
        rows.append({"loader": name, "case": case, "workers": workers, "mode": mode, "samples_per_s": round(r)})
        print(f"{name:9s} {case:18s} workers {workers} {mode:8s} {r:12,.0f} samples/s")
        if hasattr(ld, "close"):
            ld.close()

    for aug in (False, True, "per_sample"):
        for w, mode in ((0, "-"), (4, "thread"), (4, "process")):
            if not aug and w:
                continue
            add("truestar", "ts", aug, w, mode, ts_loader(x, y, aug, w, "thread" if mode == "-" else mode))
        if aug == "per_sample":
            continue
        for w in (0, 4):
            if not aug and w:
                continue
            add("torch", "torch", aug, w, "process" if w else "-", torch_loader(x, y, aug, w))

    # determinism
    n_det = min(a.n, 8192)
    xd, yd = make(n_det)
    ts_dig = {}
    for w, mode in ((0, "thread"), (1, "thread"), (2, "thread"), (4, "thread"), (2, "process"), (4, "process")):
        with ts_loader(xd, yd, True, w, mode) as ld:
            ts_dig[f"{mode}x{w}"] = digest(ld)
    other = subprocess.run([sys.executable, os.path.abspath(__file__), "--digest-only", "--n", str(n_det)],
                           capture_output=True, text=True, check=True).stdout.strip()
    torch_dig = {}
    for w in (0, 2, 4):
        torch.manual_seed(0)
        torch_dig[f"workers{w}"] = digest(torch_loader(xd, yd, True, w))
    det = {"samples": n_det, "truestar": ts_dig, "truestar_second_process": other,
           "truestar_all_equal": len(set(ts_dig.values()) | {other}) == 1,
           "torch": torch_dig, "torch_distinct_digests": len(set(torch_dig.values()))}
    print(json.dumps(det, indent=1))
    out = {"machine": {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                       "numpy": np.__version__, "cpu_count": os.cpu_count(), "torch_threads": 4},
           "batch_size": B, "samples": a.n, "throughput": rows, "determinism": det}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
