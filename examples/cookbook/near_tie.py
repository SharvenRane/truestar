"""Near tie diagnosis: tell a prediction that flips because two classes were tied from one that flips
because something is wrong.

    python examples/cookbook/near_tie.py        # CPU, about 5 s

Any run in lower precision changes some predictions. Where the float64 top two logits of a row are
closer than the rounding a correct run in that precision makes on that row, either answer is
correct and the flip is a near tie. Truestar counts those apart (TS013 in `truestar check` says
"3/8 predictions changed (0 near ties, 3 real)"). This script does the same by hand on a native
classifier. The rounding bound per row comes from a correct run in bfloat16 (textbook arithmetic,
sim:ieee-bf16); it then judges a simulated device that runs its matmuls in bfloat16
(sim:neuron-bf16), whose flips are near ties, and a run with a planted bug (one class's weights
negated), whose extra flips are real. Recipe: docs/cookbook/near_tie.md.
"""

import sys

import torch

from truestar import nn
from truestar.devsim import simulate
from truestar.quant.model import decisions

D, H, C, N = 32, 128, 20, 4000
net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, C))


def cast(tree, dtype):
    return {k: cast(v, dtype) for k, v in tree.items()} if isinstance(tree, dict) else tree.to(dtype)


def explain(ref, out, tie, title, show=3):
    d = decisions(ref, out, tie)
    print(f"{title}: {d.flips} of {d.total} predictions changed ({d.near_ties} near ties, {d.real} real)")
    top2 = ref.topk(2, dim=-1)
    flipped = (ref.argmax(-1) != out.argmax(-1)).nonzero().flatten()[:show]
    for r in flipped.tolist():
        (a, b), (ia, ib) = top2.values[r].tolist(), top2.indices[r].tolist()
        margin = a - b
        kind = "near tie" if margin <= float(tie[r]) else "REAL"
        print(f"  row {r:4}: float64 class {ia} by {margin:.2e} over class {ib}; this run picked "
              f"{int(out[r].argmax())}; rounding bound on the row {float(tie[r]):.2e} -> {kind}")
    return d


def main():
    p = net.init(seed=0)
    x = torch.randn(N, D, generator=torch.Generator().manual_seed(0))
    ref = net(cast(p, torch.float64), x.double())
    with simulate("sim:ieee-bf16"):
        bf16 = net(cast(p, torch.float32), x).double()
    # The rounding a correct bfloat16 run makes on each row, times 4: a flip with a float64 margin
    # inside it is a near tie (the rule of truestar.quant.model.compare).
    tie = 4 * (bf16 - ref).abs().amax(-1)

    with simulate("sim:neuron-bf16"):                       # a device: matmuls in bfloat16
        dev = net(cast(p, torch.float32), x).double()
    a = explain(ref, dev, tie, "sim:neuron-bf16")
    bad = cast(p, torch.float32)
    bad["4"]["w"] = bad["4"]["w"].clone()
    bad["4"]["w"][3] = -bad["4"]["w"][3]                    # the planted bug: class 3's weights negated
    with simulate("sim:ieee-bf16"):
        broken = net(bad, x).double()
    b = explain(ref, broken, tie, "\nbfloat16 with the planted bug")
    ok = a.real == 0 and a.flips > 0 and b.real > 0
    print("\nOK: the device flips only near ties; the bug flips real predictions" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
