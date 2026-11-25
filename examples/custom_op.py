"""Example for `truestar.torch_op`: a typed Python function registered as a PyTorch custom op.

    python examples/custom_op.py

No C++ or CUDA compiler is involved. The op gets a schema, a fake implementation for
torch.compile, and a backward derived by Truestar's autodiff. The script checks the forward and
the gradients against plain PyTorch in float64 and runs torch.library.opcheck; it exits 1 on
any mismatch. CPU by default (pass `cuda` to run on the GPU). The first call compiles a kernel,
which takes a few seconds; later calls reuse it.
"""
import sys

import torch

import truestar
from truestar import Tensor, ops


@truestar.torch_op
def rms_norm_residual(x: Tensor["B", 128], r: Tensor["B", 128], w: Tensor[128]) -> Tensor["B", 128]:
    h = x + r
    return h * ops.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w


def reference(x, r, w):
    h = (x + r).double()
    return h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w.double()


def rel(a, b):
    return ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()


def main(device="cpu"):
    torch.manual_seed(0)
    x, r, w = (torch.randn(4, 128, device=device, requires_grad=True),
               torch.randn(4, 128, device=device, requires_grad=True),
               torch.randn(128, device=device, requires_grad=True))
    y = rms_norm_residual(x, r, w)
    y.sum().backward()
    grads = [t.grad.clone() for t in (x, r, w)]

    xd, rd, wd = (t.detach().double().requires_grad_() for t in (x, r, w))
    yd = reference(xd, rd, wd)
    yd.sum().backward()
    errs = {"forward": rel(y.detach(), yd.detach())}
    errs.update({f"grad {n}": rel(g, t.grad) for n, g, t in zip("xrw", grads, (xd, rd, wd))})
    for k, v in errs.items():
        print(f"{k}: relative error vs torch float64 {v:.1e}")
    print("op:", rms_norm_residual.overload)
    torch.library.opcheck(rms_norm_residual.overload, (x.detach(), r.detach(), w.detach()))
    print("torch.library.opcheck: passed")
    return 0 if max(errs.values()) < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "cpu"))
