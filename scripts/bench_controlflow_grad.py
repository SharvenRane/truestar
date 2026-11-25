"""Measure gradients through control flow: accuracy, memory per policy, run time, compiles.

    PYTHONPATH=src python scripts/bench_controlflow_grad.py

Prints, for a native RNN loss (B=4, hidden 32, input 16) at T = 8, 32 and 48:
* the worst relative error of the gradients against torch.autograd on the unrolled loop (float64);
* per loop memory policy: copies of the carried state kept alive by the gradient program, and the
  reference interpreter's run time (one program serves every T: compiles are counted);
* the worst finite difference error over the control flow test programs.
"""

import math
import os
import sys
import time

import sympy
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import truestar  # noqa: E402
from truestar import Tensor, ir, ops  # noqa: E402
from truestar.ir.autodiff import grad  # noqa: E402

H, I, B = 32, 16, 4


@truestar.function
def rnn_loss(x: Tensor["B", "T", 16], wx: Tensor[16, 32], wh: Tensor[32, 32]) -> Tensor[()]:
    h = ops.zeros((x.shape[0], 32))
    for t in range(x.shape[1]):
        h = ops.tanh(x[:, t] @ wx + h @ wh)
    return (h * h).mean()


def rows(prog, bindings):
    out = 0
    stack = [prog]
    while stack:
        p = stack.pop()
        for n in p.nodes:
            stack += [v for v in n.attrs.values() if isinstance(v, ir.Program)]
            t = n.output.type
            if n.op == "full" and t.rank == 3:
                tail = [int(sympy.sympify(d).subs(bindings)) for d in t.shape[1:]]
                if tail == [B, H]:
                    out += int(sympy.sympify(t.shape[0]).subs(bindings))
    return out


def main():
    g = torch.Generator().manual_seed(0)
    wx, wh = torch.randn(I, H, generator=g, dtype=torch.float64) * 0.3, torch.randn(H, H, generator=g, dtype=torch.float64) * 0.15
    progs = {}
    for policy in ("store", "recompute", "sqrt"):
        t0 = time.perf_counter()
        progs[policy] = grad(rnn_loss.program, ["wx", "wh"], loop_memory=policy)
        print(f"transform {policy}: {time.perf_counter() - t0:.3f} s, {len(progs[policy].nodes)} top level nodes")
    syms = {sympy.Symbol("B", integer=True, positive=True): B}
    print(f"{'T':>3} {'policy':>9} {'carried copies':>15} {'run ms':>8} {'max rel err vs torch':>21}")
    for T in (8, 32, 48):
        x = torch.randn(B, T, I, generator=g, dtype=torch.float64)
        xs, a, b = x.clone(), wx.clone().requires_grad_(), wh.clone().requires_grad_()
        h = torch.zeros(B, H, dtype=torch.float64)
        for t in range(T):
            h = torch.tanh(xs[:, t] @ a + h @ b)
        want = torch.autograd.grad((h * h).mean(), (a, b))
        bind = dict(syms)
        bind[sympy.Symbol("T", integer=True, positive=True)] = T
        for policy, gp in progs.items():
            ir.run(gp, x, wx, wh)
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                res = ir.run(gp, x, wx, wh)
            ms = (time.perf_counter() - t0) / reps * 1e3
            err = max(((r - w).abs().max() / w.abs().max()).item() for r, w in zip(res[1:], want))
            print(f"{T:>3} {policy:>9} {rows(gp, bind):>15} {ms:>8.1f} {err:>21.1e}   "
                  f"(2*ceil(sqrt T) = {2 * math.ceil(math.sqrt(T))})")
    print(f"compiles of rnn_loss across every T above: {rnn_loss.compile_count}")

    import test_autodiff_cf as t
    from test_autodiff import fd_check
    worst = 0.0
    for fn, sizes in t.FD_CASES:
        prog = t.NATIVE[fn][0].program
        for policy in ("store", "recompute", "sqrt"):
            gp = grad(prog, [v.name for v in prog.inputs], seed="input", loop_memory=policy)
            worst = max(worst, fd_check(prog, t._inputs(fn, **sizes), gp))
    print(f"finite differences, worst relative error over {len(t.FD_CASES)} programs x 3 policies: {worst:.1e}")
    worst = 0.0
    for fn, sizes in t.TORCH_CASES:
        for policy in ("store", "recompute", "sqrt", "auto"):
            got, want = t._vjp(fn, t._inputs(fn, **sizes), policy)
            worst = max(worst, max(t.rel_err(a, b) for a, b in zip(got, want)))
    print(f"torch.autograd on the unrolled programs, worst relative error over {len(t.TORCH_CASES)} cases x 4 "
          f"policies: {worst:.1e}")


if __name__ == "__main__":
    main()
