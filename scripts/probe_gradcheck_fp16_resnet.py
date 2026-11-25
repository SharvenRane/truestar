"""Probe: truestar's backward check on a float16 resnet18 on CPU, across seeds and thread counts.

Prints, per (threads, seed), the status, the worst gradient's own error, its threshold and where it is,
so a platform difference in tests/test_gradcheck.py::test_clean_float16_resnet_reports_clean can be
told apart from seed luck.
"""

import platform
import sys

import torch
import torchvision

from truestar import grad_check

print(f"torch {torch.__version__}, python {sys.version.split()[0]}, {platform.system()} {platform.machine()}")
for threads in (1, 4):
    torch.set_num_threads(threads)
    for seed in range(8):
        torch.manual_seed(seed)
        model = torchvision.models.resnet18(num_classes=10).half()
        x = torch.randn(2, 3, 32, 32)
        rep = grad_check(model, x, backends=["cpu"])
        worst = None
        for r in rep.results:
            for l in r.layers:
                for g in l.tensors:
                    ratio = g.diff.rel_err / g.threshold if g.threshold else float("inf")
                    if worst is None or ratio > worst[0]:
                        worst = (ratio, l.name, g.label, g.diff.rel_err, g.threshold)
        w = "n/a" if worst is None else f"{worst[1]} {worst[2]} {worst[3]:.2e}/{worst[4]:.2e} ratio {worst[0]:.2f}"
        print(f"threads {threads} seed {seed}: {rep.status}  worst {w}", flush=True)
