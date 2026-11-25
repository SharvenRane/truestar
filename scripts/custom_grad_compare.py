"""Wrong hand written backwards: torch.autograd.Function against @fn.grad.

Run: PYTHONPATH=src python scripts/custom_grad_compare.py

For each injected bug (sign, missing term, wrong broadcast, unsummed broadcast) it records what
happens at definition, at the first forward and backward, and with torch.autograd.gradcheck run by
hand, and the same for Truestar's registration check. Prints one JSON document. JAX is not installed here; its behaviour
is taken from its documentation in docs/AUTODIFF.md.
"""

import json
import time
import warnings

import torch

import truestar
from truestar import Tensor, ops
from truestar.frontend import FrontendError


def torch_function(bug):
    class Affine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, b):
            ctx.save_for_backward(x, w)
            return torch.tanh(x * w) + b

        @staticmethod
        def backward(ctx, g):
            x, w = ctx.saved_tensors
            t = torch.tanh(x * w)
            d = g * (1 - t * t)
            gx, gw, gb = d * w, (d * x).sum(0), g.sum(0)
            if bug == "sign":
                gx = -gx
            elif bug == "missing term":
                gw = (g * x).sum(0)          # dropped the 1 - t^2 factor
            elif bug == "wrong broadcast":
                gb = g.mean(0)               # reduced over the batch with a mean, not a sum
            elif bug == "unsummed broadcast":
                gb = g                       # not reduced at all: autograd sums it back silently
            return gx, gw, gb
    return Affine


def run_torch(bug):
    out = {"definition": "accepted"}
    F = torch_function(bug)
    x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, dtype=torch.float64, requires_grad=True)
    try:
        F.apply(x, w, b).sum().backward()
        out["first backward"] = "no error, gradients returned"
    except Exception as e:
        out["first backward"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    t0 = time.perf_counter()
    try:
        torch.autograd.gradcheck(F.apply, (x, w, b))
        out["gradcheck by hand"] = "passed"
    except Exception as e:
        out["gradcheck by hand"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    out["gradcheck seconds"] = round(time.perf_counter() - t0, 3)
    return out


def run_truestar(bug):
    @truestar.function
    def affine(x: Tensor["B", 4], w: Tensor[4], b: Tensor[4]) -> Tensor["B", 4]:
        return ops.tanh(x * w) + b

    rules = {}

    def sign(x, w, b, g):
        t = ops.tanh(x * w)
        d = g * (1.0 - t * t)
        return -(d * w), (d * x).sum(0), g.sum(0)

    def missing(x, w, b, g):
        t = ops.tanh(x * w)
        d = g * (1.0 - t * t)
        return d * w, (g * x).sum(0), g.sum(0)

    def broadcast(x, w, b, g):
        t = ops.tanh(x * w)
        d = g * (1.0 - t * t)
        return d * w, (d * x).sum(0), g.mean(0)

    def unsummed(x, w, b, g):
        t = ops.tanh(x * w)
        d = g * (1.0 - t * t)
        return d * w, (d * x).sum(0), g

    def correct(x, w, b, g):
        t = ops.tanh(x * w)
        d = g * (1.0 - t * t)
        return d * w, (d * x).sum(0), g.sum(0)

    rules = {"sign": sign, "missing term": missing, "wrong broadcast": broadcast, "unsummed broadcast": unsummed, "correct": correct}
    t0 = time.perf_counter()
    try:
        affine.grad(rules[bug])
        res = {"definition": "accepted", "report": str(affine.grad_report())}
    except FrontendError as e:
        d = e.to_dict()
        res = {"definition": f"refused ({d['code']})", "message": d["message"][:200],
               "operands": [o["name"] for o in d["operands"]]}
    res["seconds"] = round(time.perf_counter() - t0, 3)
    return res


def main():
    warnings.simplefilter("ignore")
    bugs = ["sign", "missing term", "wrong broadcast", "unsummed broadcast", "correct"]
    doc = {"torch": torch.__version__, "cases": {b: {"torch.autograd.Function": run_torch(b),
                                                     "truestar @fn.grad": run_truestar(b)} for b in bugs}}
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
