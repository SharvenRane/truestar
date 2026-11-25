"""The Truestar native version of a typical PyTorch inference script (docs/SWITCHING.md, section 2).

    python examples/switching/infer_native.py        # CPU, about 10 s

Load weights from a safetensors file, compile the forward pass for the CPU in float32 (checked
against float64 on its first call), run it on batches of any size, take the predictions.
"""

import os
import sys
import tempfile

import torch

import truestar
from truestar import Tensor, nn
from truestar.compiler import CompiledExecutor
from truestar.interop.safetensors import load_file, save_file

D, H, C = 20, 64, 5
net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, C))
ex = CompiledExecutor("cpu", "float32")


@truestar.native(executor=ex)
def predict(p: net.spec, x: Tensor["B", D]) -> Tensor["B", "i64"]:
    return net(p, x).argmax(-1)


def flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def unflatten(flat):
    tree = {}
    for name, t in flat.items():
        *path, leaf = name.split(".")
        node = tree
        for k in path:
            node = node.setdefault(k, {})
        node[leaf] = t
    return tree


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "weights.safetensors")
        save_file(flatten(net.init(seed=0)), path)       # stands in for weights you trained earlier
        p = unflatten(load_file(path, mmap=False))      # mmap=False: the file can be deleted at once

    for n in (1, 7, 256):                                 # any batch size, one compiled program
        x = torch.randn(n, D)
        labels = predict(p, x)
        same = torch.equal(labels, net(p, x).argmax(-1))  # the same layers run eagerly
        print(f"batch {n:3}: predictions {labels[:8].tolist()}{' ...' if n > 8 else ''}; equal to eager {same}")
    print(f"compiles: {predict.compile_count}; verified against float64 on the first call: {ex.last.verified}")
    return 0 if ex.last.verified else 1


if __name__ == "__main__":
    sys.exit(main())
