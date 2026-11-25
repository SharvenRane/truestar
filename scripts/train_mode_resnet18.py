"""Evidence for training mode in the IR (docs/RANDOM.md, pain ledger P20): torchvision resnet18 in
training mode, imported with explicit batch norm state, trained for N SGD steps on the float64
reference interpreter with IR autodiff, side by side with PyTorch float64 autograd on the same data.

Prints per step the worst relative error of the loss, every weight gradient, the updated weights and
every running statistic, next to a control: PyTorch against PyTorch with weights moved by 1e-15.
Training at this learning rate is chaotic, so free running errors grow about 100x per step in both;
``resync`` starts every step from PyTorch's weights and statistics, which measures each step alone.

Usage: PYTHONPATH=src python scripts/train_mode_resnet18.py [steps] [export|decompose] [lr] [resync]
"""

from __future__ import annotations

import copy
import sys
import time

import torch
import torchvision

from truestar import ir
from truestar.ir.autodiff import grad


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def main(steps: int = 20, decompose: bool = False, lr: float = 0.05, resync: bool = False):
    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=10).double().train()
    for m in model.modules():   # non trivial statistics and affine parameters
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.3, 0.3)
    x0 = torch.randn(4, 3, 32, 32, dtype=torch.float64)

    class WithLoss(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x, target):
            d = self.net(x) - target
            return (d * d).mean()

    wrapped = WithLoss(model).train()
    t0 = time.perf_counter()
    prog = ir.import_module(copy.deepcopy(wrapped), (x0, torch.zeros(4, 10, dtype=torch.float64)),
                            training=True, decompose=decompose)
    weights = [p.name for p in prog.params if p.name.startswith("net.") and p.type.rank > 0
               and p.name.removeprefix("net.") in dict(model.named_parameters())]
    gprog = grad(prog, wrt=weights, of=[0])
    print(f"imported in {time.perf_counter() - t0:.1f} s: {len(prog.nodes)} nodes, {len(prog.state)} state "
          f"tensors, {len(weights)} weight tensors")
    n_state = len(prog.state)
    state = ir.initial_state(prog)
    ref = copy.deepcopy(model)
    opt = torch.optim.SGD(ref.parameters(), lr=lr)
    # Control: PyTorch against PyTorch with every weight moved by a relative 1e-15. Training is
    # chaotic, so any two float64 runs drift apart; the IR run is judged against this drift.
    ctl = copy.deepcopy(model)
    with torch.no_grad():
        for prm in ctl.parameters():
            prm.mul_(1 + 1e-15)
    ctl_opt = torch.optim.SGD(ctl.parameters(), lr=lr)
    worst = worst_ctl = 0.0
    for step in range(steps):
        torch.manual_seed(100 + step)
        x = torch.randn(4, 3, 32, 32, dtype=torch.float64)
        target = torch.randn(4, 10, dtype=torch.float64)
        t1 = time.perf_counter()
        if resync:   # start the step from torch's weights and statistics: measures this step alone
            named0, bufs0 = dict(ref.named_parameters()), ref.state_dict()
            for w in weights:
                gprog.param_data[w] = named0[w.removeprefix("net.")].detach().clone()
            state = [bufs0[t.removeprefix("net.")].clone() for t, _, _ in prog.state]
        outs = ir.run(gprog, x, target, *state)
        loss, new_state, grads = outs[0], outs[1:1 + n_state], outs[1 + n_state:]
        state = [s.to(t.dtype) for s, t in zip(new_state, ir.initial_state(prog))]
        # torch float64 autograd on the same step
        opt.zero_grad()
        tl = ((ref(x) - target) ** 2).mean()
        tl.backward()
        ctl_opt.zero_grad()
        cl = ((ctl(x) - target) ** 2).mean()
        cl.backward()
        named = dict(ref.named_parameters())
        cnamed = dict(ctl.named_parameters())
        errs = {"loss": rel(loss, tl.detach())}
        errs["grads"] = max(rel(g, named[w.removeprefix("net.")].grad) for w, g in zip(weights, grads))
        ctl_errs = {"loss": rel(cl.detach(), tl.detach()),
                    "grads": max(rel(cnamed[k].grad, named[k].grad) for k in named)}
        opt.step()
        ctl_opt.step()
        for w, g in zip(weights, grads):   # the same SGD step on the IR weights
            prog_w = gprog.param_data[w]
            gprog.param_data[w] = prog_w - lr * g
        errs["weights"] = max(rel(gprog.param_data[w], named[w.removeprefix("net.")].detach()) for w in weights)
        bufs = ref.state_dict()
        errs["running stats"] = max(rel(s, bufs[t.removeprefix("net.")]) for s, (t, _, _) in zip(state, prog.state))
        ctl_errs["weights"] = max(rel(cnamed[k].detach(), named[k].detach()) for k in named)
        cb = ctl.state_dict()
        ctl_errs["running stats"] = max(rel(cb[k], bufs[k]) for k in bufs if bufs[k].is_floating_point())
        worst = max(worst, max(errs.values()))
        worst_ctl = max(worst_ctl, max(ctl_errs.values()))
        print(f"step {step + 1:2d} ({time.perf_counter() - t1:.1f} s): IR vs torch " +
              ", ".join(f"{k} {v:.1e}" for k, v in errs.items()) + " | control torch vs torch(w*(1+1e-15)) " +
              ", ".join(f"{k} {v:.1e}" for k, v in ctl_errs.items()))
    print(f"worst relative error over {steps} steps: IR {worst:.2e}, control {worst_ctl:.2e}")
    return worst, worst_ctl


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, len(sys.argv) > 2 and sys.argv[2] == "decompose",
         float(sys.argv[3]) if len(sys.argv) > 3 else 0.05, len(sys.argv) > 4 and sys.argv[4] == "resync")
