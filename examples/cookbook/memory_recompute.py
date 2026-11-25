"""Plan a training step's memory before running it, then fit it under a budget by recomputing.

    python examples/cookbook/memory_recompute.py        # CPU, about 20 s

1. Build a native training step (forward, backward, AdamW in one program) on the compiled CPU
   executor and ask for its memory plan at batch 2048. The plan comes from shapes alone.
2. Compile the same step with a budget below that peak and recompute=True: values the backward
   needs are recomputed instead of stored until the plan fits; each recompute is checked.
3. Train both for a few steps on the same batches and compare every loss and parameter bit.
Recipe: docs/cookbook/memory_recompute.md.
"""

import sys

import torch

import truestar
from truestar import Tensor, nn, optim
from truestar.compiler import CompiledExecutor

D, H, C, B, STEPS = 32, 128, 10, 2048, 3

net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, H), nn.GELU(),
                    nn.Linear(H, C))


@truestar.native
def loss(p: net.spec, x: Tensor["B", D], y: Tensor["B", "i64"]) -> Tensor[()]:
    return nn.cross_entropy(net(p, x), y)


def batches():
    g = torch.Generator().manual_seed(0)
    for _ in range(STEPS):
        x = torch.randn(B, D, generator=g)
        yield x, (x[:, :C] * 3).argmax(-1)


def train(executor):
    step = truestar.train_step(loss, optim.AdamW(lr=1e-3), executor=executor)
    ts = step.init(net.init(seed=0))
    losses = []
    for x, y in batches():
        ts, r = step(ts, x, y)
        losses.append(r.loss)
    return ts, losses


def leaves(tree):
    return [t for v in tree.values() for t in leaves(v)] if isinstance(tree, dict) else [tree]


def main():
    plain_ex = CompiledExecutor("cpu", "float32")
    plain, plain_losses = train(plain_ex)
    plan = plain_ex.last.memory_plan({"B": B})
    print("memory plan of the plain step:")
    print(plan)

    budget = int(plan.total_bytes * 0.8)
    tight_ex = CompiledExecutor("cpu", "float32", memory_budget=budget, memory_dims={"B": B}, recompute=True)
    tight, tight_losses = train(tight_ex)
    cp = tight_ex.last
    tight_plan = cp.memory_plan({"B": B})
    print(f"\nbudget {budget / 2**20:.2f} MiB (80% of the plain step, {plan.total_bytes / 2**20:.2f} MiB)")
    print(f"planned with recompute: {tight_plan.total_bytes / 2**20:.2f} MiB")
    for e in cp.recompute_report:
        print(f"  recomputed: {e['value']} ({e['bytes'] / 2**20:.2f} MiB, {'+'.join(e['ops'])}) from "
              f"{e['source']}, rerun before {e['recomputed_before']}")

    same_loss = plain_losses == tight_losses
    same_params = all(torch.equal(a, b) for a, b in zip(leaves(plain.params), leaves(tight.params)))
    print(f"\nlosses {plain_losses[0]:.4f} -> {plain_losses[-1]:.4f}; bitwise equal to the plain step: "
          f"losses {same_loss}, parameters {same_params}")
    ok = same_loss and same_params and tight_plan.total_bytes <= budget
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
