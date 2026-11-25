"""Catch a runtime that lies: check the shipped file in the runtime that runs it, against float64.

    python examples/cookbook/catch_lying_runtime.py        # CPU, about 15 s

1. Export a model once to a Truestar artifact (`.tsx`); the export itself is refused unless the
   file matches the model.
2. `conform` runs the file in its runtimes (the float64 reference interpreter and the compiled CPU
   executor) on 8 samples and judges each against the model in float64: conversion (the file's
   meaning) and runtime (what the runtime computes) separately.
3. A runtime that lies: the same compiled runtime, but every output it returns is rounded through
   float16, as a runtime that quietly switches to half precision would. Conform reports it, with
   the decisions it changed and the module where the error starts.
Recipe: docs/cookbook/catch_lying_runtime.md.
"""

import os
import sys
import tempfile

import numpy as np
import torch

from truestar.conform import conform
from truestar.export import export


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.nn.Linear(16, 64)
        self.norm = torch.nn.LayerNorm(64)
        self.hidden = torch.nn.Linear(64, 64)
        self.head = torch.nn.Linear(64, 10)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.inp(x))
        h = torch.nn.functional.gelu(self.hidden(self.norm(h)))
        return self.head(h)


def half_precision(outs):
    """What the lying runtime does to every result before handing it back."""
    return [o.astype(np.float16).astype(o.dtype) for o in outs]


def main():
    torch.manual_seed(0)
    model, x = Classifier().eval(), torch.randn(64, 16)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "classifier.tsx")
        export(model, x, path)

        honest = conform(model, x, path, runtimes=["tsx:reference", "tsx:cpu"])
        print(honest.to_text())
        lying = conform(model, x, path, runtimes=["tsx:cpu"], options={"tsx:cpu": {"fault_outputs": half_precision}})
        print("\nthe same runtime, returning float16 rounded results:")
        print(lying.to_text())
    ok = all(r.status == "ok" for r in honest.results) and lying.results[0].status != "ok"
    print("\nOK: the honest runtimes conform, the lying one is caught" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
