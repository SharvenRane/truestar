"""conftest.py for the CPU smoke tests in .github/workflows/install-check.yml.

The workflow copies tests/ out of the checkout (so the tests import the installed wheel, not src/)
and drops this file in as tests/conftest.py. It hides MPS and XPU so tests that need an
accelerator skip, as they do on a machine with only a CPU; CUDA is hidden by
CUDA_VISIBLE_DEVICES="". GitHub's macOS runners expose a paravirtual GPU as MPS, which is not the
CPU check this job is for. CLI child processes the tests start do not load this file; the
workflow prints `truestar backends` first so a run that saw MPS there is visible in the log.
"""
import functools

import torch

if getattr(torch.backends, "mps", None) is not None:
    _orig = torch.backends.mps.is_available

    @functools.wraps(_orig)   # torch 2.15 dynamo reads __wrapped__ on it
    def _no_mps():
        return False

    torch.backends.mps.is_available = _no_mps
if getattr(torch, "xpu", None) is not None:
    torch.xpu.is_available = lambda: False
