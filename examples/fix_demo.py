"""Demo for `truestar fix`: an MLP whose CUDA matmul with bias (aten.addmm) is wrong.

    truestar fix examples/fix_demo.py --backends cuda

The bad kernel is simulated with truestar.testing.OpFault, entered around every run through
get_context(). truestar fix names the op, swaps in mm plus add only on CUDA, reruns the check to
verify it is clean and reports what the remedy costs.
"""

import torch
from torch import nn

from truestar.testing import OpFault


def get_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                         nn.Linear(128, 10))


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(32, 64)


def get_context():
    return OpFault("aten.addmm.default", "cuda", amount=0.05)
