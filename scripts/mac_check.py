"""Five minute check on a real Apple silicon Mac. Needs only PyTorch.

Setup and run (Terminal):

    python3 -m venv ts && source ts/bin/activate
    pip install torch==2.14.1
    python mac_check.py
    pip install --upgrade torch
    python mac_check.py

Each run appends to truestar_mac_report.txt next to this file. Attach it to an issue if you want to share it.
Nothing is uploaded anywhere; the script only does a few matrix multiplies.
"""
import os
import platform
import subprocess
import sys

import torch
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truestar_mac_report.txt")
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    n = torch.linalg.vector_norm(b).item()
    return torch.linalg.vector_norm(a - b).item() / (n if n else 1.0)


log("=" * 70)
log(f"macOS {platform.mac_ver()[0]}  ({sh(['sw_vers', '-buildVersion'])})")
log(f"chip  {sh(['sysctl', '-n', 'machdep.cpu.brand_string'])}")
log(f"gpu   {sh(['system_profiler', 'SPDisplaysDataType']).splitlines()[2:4] if platform.system() == 'Darwin' else 'n/a'}")
log(f"python {platform.python_version()}  torch {torch.__version__}")
log(f"mps built {torch.backends.mps.is_built()}  available {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    log("MPS not available: nothing to test on this machine.")
else:
    torch.manual_seed(0)
    shapes = [((4, 512), 10), ((4, 16, 128), 256), ((2, 17, 768), 3072), ((64, 64), 64), ((1, 1024), 10)]
    wrong = 0
    total = 0
    for xs, n in shapes:
        x = torch.randn(*xs)
        k = xs[-1]
        w = torch.randn(n, k) / k ** 0.5
        b = torch.randn(n)
        ref_b = F.linear(x.double(), w.double(), b.double())
        ref_nb = F.linear(x.double(), w.double())
        xm, wm, bm = x.to("mps"), w.to("mps"), b.to("mps")
        got_b = F.linear(xm, wm, bm)
        got_nb = F.linear(xm, wm)
        got_mm = xm @ wm.t() + bm
        e_b, e_nb, e_mm = rel(got_b, ref_b), rel(got_nb, ref_nb), rel(got_mm, ref_b)
        bias_dropped = rel(got_b, ref_nb) < 1e-4
        total += 1
        bad = e_b > 1e-3
        wrong += bad
        log(f"x{tuple(xs)} w({n},{k})  linear+bias {e_b:.1e}  linear {e_nb:.1e}  "
            f"matmul+bias {e_mm:.1e}{'  <-- WRONG' if bad else ''}"
            f"{'  (bias was ignored)' if bad and bias_dropped else ''}")
    log(f"RESULT: F.linear with bias wrong in {wrong} of {total} cases on this Mac")
log("")
with open(OUT, "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"saved to {OUT}")
