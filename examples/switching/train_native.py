"""The Truestar native version of a typical PyTorch training loop (docs/SWITCHING.md, section 1).

    python examples/switching/train_native.py        # CPU, about 15 s

A classifier with dropout, AdamW with warmup and cosine decay, gradient clipping, a checkpoint, an
evaluation pass. One compiled program per step: forward, backward, clipping and the optimizer.
"""

import os
import sys
import tempfile

import torch

import truestar
from truestar import Tensor, nn, optim
from truestar.compiler import CompiledExecutor

D, H, C, B, STEPS = 20, 64, 5, 64, 60

# --- model: an immutable config; parameters are a tree you hold
net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(), nn.Linear(H, C))


# --- loss: a native function; `training` is a compile time argument, so eval has no dropout op
@truestar.native
def loss(p: net.spec, x: Tensor["B", D], y: Tensor["B", "i64"], training: bool = False) -> Tensor[()]:
    return nn.cross_entropy(net(p, x, training=training), y)


@truestar.native
def accuracy(p: net.spec, x: Tensor["B", D], y: Tensor["B", "i64"]) -> Tensor[()]:
    return nn.accuracy(net(p, x), y)


def data(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g)
    return x, (x[:, :C] + 0.5 * x[:, C:2 * C]).argmax(-1)


def batches(steps, seed=0):
    for i in range(steps):
        yield data(B, seed * 100000 + i)


def main():
    opt = optim.AdamW(lr=optim.warmup_cosine(3e-3, warmup=6, total=STEPS), weight_decay=0.01)
    step = truestar.train_step(loss, opt, clip=1.0, static={"training": True},
                               executor=CompiledExecutor("cpu", "float32"))
    ts = step.init(net.init(seed=0))          # TrainState: params, optimizer slots, step count

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "run.safetensors")
        ts, history = truestar.fit(step, ts, batches(STEPS), log_every=20, checkpoint=ckpt)
        ts = step.load(ckpt)                  # resume later from here, bit for bit

    xv, yv = data(2048, seed=7)
    acc = float(accuracy(ts.params, xv, yv))  # evaluation: a separate program, no dropout in it
    print(f"loss {history[0].loss:.3f} -> {history[-1].loss:.3f}; validation accuracy {acc:.3f}")
    return 0 if history[-1].loss < history[0].loss and acc > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
