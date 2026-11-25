"""Deterministic training: the same run twice gives the same bits, and a resumed run continues the
uninterrupted one bit for bit, dropout included.

    python examples/cookbook/deterministic_training.py        # CPU, about 15 s

Everything that changes during training is an explicit value in the TrainState: parameters,
optimizer moments, the step count. Dropout masks are keyed by the function's seed, the step and
the call site, never drawn from a global generator, so a replayed step draws the same mask
(docs/RANDOM.md). The step is compiled for the CPU in float32. Recipe:
docs/cookbook/deterministic_training.md.
"""

import hashlib
import os
import sys
import tempfile

import torch

import truestar
from truestar import Tensor, nn, optim
from truestar.compiler import CompiledExecutor

D, H, C, B, STEPS = 16, 64, 4, 32, 20

net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, C))


@truestar.native
def loss(p: net.spec, x: Tensor["B", D], y: Tensor["B", "i64"], training: bool = False) -> Tensor[()]:
    return nn.cross_entropy(net(p, x, training=training), y)


def batches():
    g = torch.Generator().manual_seed(0)
    for _ in range(STEPS):
        x = torch.randn(B, D, generator=g)
        yield x, (x[:, :C] - x[:, C:2 * C]).argmax(-1)


def make_step():
    opt = optim.AdamW(lr=optim.warmup_cosine(1e-2, warmup=4, total=STEPS), weight_decay=0.01)
    return truestar.train_step(loss, opt, clip=1.0, static={"training": True},
                               executor=CompiledExecutor("cpu", "float32"))


def digest(tree):
    h = hashlib.sha256()
    for k in sorted(tree):
        v = tree[k]
        if isinstance(v, dict):
            h.update(digest(v).encode())
        else:
            h.update(k.encode() + v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    # 1. two independent runs from the same seed
    step = make_step()
    a, hist_a = truestar.fit(step, step.init(net.init(seed=0)), batches())
    step2 = make_step()
    b, hist_b = truestar.fit(step2, step2.init(net.init(seed=0)), batches())
    same_runs = digest(a.params) == digest(b.params) and [r.loss for r in hist_a] == [r.loss for r in hist_b]
    print(f"run 1 params {digest(a.params)}, run 2 params {digest(b.params)}: identical {same_runs}")
    print(f"loss {hist_a[0].loss:.4f} -> {hist_a[-1].loss:.4f} over {STEPS} steps")

    # 2. stop after 10 steps, save, load into a fresh step, finish: equal to the uninterrupted run
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "half.safetensors")
        step3 = make_step()
        half, _ = truestar.fit(step3, step3.init(net.init(seed=0)), batches(), steps=STEPS // 2, checkpoint=ckpt)
        step4 = make_step()
        resumed = step4.load(ckpt)
        rest = list(batches())[STEPS // 2:]
        c, _ = truestar.fit(step4, resumed, rest)
    same_resume = digest(c.params) == digest(a.params)
    print(f"stopped at step {half.step}, resumed from the checkpoint: params {digest(c.params)}, "
          f"equal to the uninterrupted run {same_resume}")
    ok = same_runs and same_resume
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
