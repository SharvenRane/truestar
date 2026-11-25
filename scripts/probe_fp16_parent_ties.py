"""Probe: float16 resnet18 backward check on 80 portable draws, CPU only.

Every weight, buffer and input comes from float64 `torch.rand` (the same bits on every CPU
capability, unlike float32 `normal_`), so the set is the same on every machine. Per draw it prints
the status, the codes, the blamed module and any near tie mask flip note. With `--fault` it also
injects a backward fault (default: a 30% `aten.threshold_backward` fault, a wrong ReLU backward; see
--fault-op and --amount) on every draw and counts how many are still reported DIVERGENT.

    PYTHONPATH=src python scripts/probe_fp16_parent_ties.py            # false alarms
    PYTHONPATH=src python scripts/probe_fp16_parent_ties.py --fault    # injected faults
    PYTHONPATH=src python scripts/probe_fp16_parent_ties.py --seeds 0:20
"""

import argparse
import json
import math
import sys
import time

import torch
import torchvision

from truestar import grad_check


class _WrongBackward(torch.autograd.Function):
    """Identity forward; backward scales the gradient by (1 + amount) except in float64, so the
    reference stays exact. It runs in the model's own code, so only the whole model replay sees it."""

    @staticmethod
    def forward(ctx, x, amount):
        ctx.amount = amount
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return (g if g.dtype == torch.float64 else g * (1 + ctx.amount)), None


class RootFaultResNet(torchvision.models.ResNet):
    root_fault = 0.0

    def forward(self, x):
        return _WrongBackward.apply(self._forward_impl(x), self.root_fault)


def portable_resnet18(seed: int, root_fault: float = 0.0):
    """resnet18 (10 classes) with every floating parameter and buffer drawn from float64 rand.
    With root_fault, the model's own code has a backward that is off by that much (relative)."""
    g = torch.Generator().manual_seed(seed)
    model = RootFaultResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10)
    model.root_fault = root_fault

    def u(shape, lo, hi):
        return (torch.rand(shape, generator=g, dtype=torch.float64) * (hi - lo) + lo).float()

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:  # conv and linear weights: uniform with variance 2 / fan_in
                fan_in = p[0].numel()
                bound = math.sqrt(6.0 / fan_in)
                p.copy_(u(p.shape, -bound, bound))
            elif name.endswith("weight"):  # batch norm scale
                p.copy_(u(p.shape, 0.5, 1.5))
            else:  # biases
                p.copy_(u(p.shape, -0.1, 0.1))
        for name, b in model.named_buffers():
            if not b.is_floating_point():
                continue
            if name.endswith("running_mean"):
                b.copy_(u(b.shape, -0.1, 0.1))
            elif name.endswith("running_var"):
                b.copy_(u(b.shape, 0.5, 1.5))
    x = (torch.rand((2, 3, 32, 32), generator=g, dtype=torch.float64) * 2 - 1).float()
    return model, x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0:80", help="start:stop")
    ap.add_argument("--fault", action="store_true", help="inject a backward fault on every draw")
    ap.add_argument("--fault-op", default="aten.threshold_backward",
                    help="ATen op to corrupt in backward (default: ReLU backward)")
    ap.add_argument("--amount", type=float, default=0.3, help="relative corruption (default 0.3)")
    ap.add_argument("--root-fault", type=float, default=0.0,
                    help="a wrong backward of this size in the model's own code (seen only by "
                         "the whole model replay)")
    ap.add_argument("--json", default=None, help="write per draw results here")
    a = ap.parse_args(argv)
    lo, hi = (int(v) for v in a.seeds.split(":"))
    torch.set_num_threads(2)
    rows = []
    t0 = time.time()
    for seed in range(lo, hi):
        model, x = portable_resnet18(seed, a.root_fault)
        model = model.half()
        if a.fault:
            from truestar.gradfault import BackwardOpFault

            with BackwardOpFault(a.fault_op, "cpu", amount=a.amount):
                rep = grad_check(model, x, backends=["cpu"], drill_ops=False)
        else:
            rep = grad_check(model, x, backends=["cpu"], drill_ops=False)
        r = rep.results[0]
        fd = r.first_divergent
        tie = [n for n in r.notes if "near tie mask flip" in n]
        row = {"seed": seed, "status": r.status, "codes": r.codes,
               "blamed": None if fd is None else fd.key,
               "leaf": None if fd is None else fd.is_leaf, "tie_note": tie[0] if tie else None}
        rows.append(row)
        print(f"seed {seed:3d}: {r.status:9s} {','.join(r.codes) or '-':12s} "
              f"{row['blamed'] or '-'}{'  [' + tie[0][:150] + ']' if tie else ''}", flush=True)
    n = len(rows)
    count = {s: sum(1 for r in rows if r["status"] == s) for s in ("ok", "suspect", "divergent")}
    print(f"\n{n} draws in {time.time() - t0:.0f} s: ok {count['ok']}, suspect {count['suspect']}, "
          f"divergent {count['divergent']}  (fault: "
          f"{f'{a.fault_op} x{a.amount}' if a.fault else 'none'}"
          f"{f', root backward x{a.root_fault}' if a.root_fault else ''})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
