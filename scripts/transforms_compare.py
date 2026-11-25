"""vmap and forward mode: Truestar against torch.func, on the cases torch.func rejects and on speed.

Run: PYTHONPATH=src python scripts/transforms_compare.py
CPU only, float64 everywhere (the Truestar runs use the float64 reference interpreter and, where it
compiles, the compiled CPU executor in float64). Prints one table per section; docs/TRANSFORMS.md
records the numbers of 2026-11-24.
"""

import statistics
import sys
import time
import warnings

import torch
from torch.func import functional_call, grad as tgrad, hessian as thessian, jacfwd as tjacfwd, vmap as tvmap

import truestar
from truestar import Tensor, ir, ops
from truestar.ir.autodiff import grad
from truestar.ir.forward import hessian, jacfwd
from truestar.ir.vmap import vmap

F64 = torch.float64
torch.manual_seed(0)


def timeit(fn, repeat=21):
    fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)  # the machine is shared: the minimum is the uncontended cost


def rel(a, b):
    a, b = a.to(F64), b.to(F64)
    return ((a - b).abs().max() / max(a.abs().max(), b.abs().max(), 1e-300)).item()


def outcome(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            msgs = [str(x.message).split(".")[0] for x in w if "batching rule" in str(x.message)]
            return "runs" + (f" with warning: {msgs[0]}" if msgs else "")
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"


# ----------------------------------------------------------------------------- 1. cases torch.func rejects

print("## 1. Cases torch.func.vmap rejects or degrades (torch", torch.__version__, ")\n")


@truestar.function
def branchy(x: Tensor[16]) -> Tensor[16]:
    if x.sum() > 0:
        x = ops.relu(x) * 2.0
    else:
        x = ops.tanh(x)
    while (x * x).sum() > 1.0:
        x = x * 0.5
    return x


def torch_branchy(x):
    if x.sum() > 0:
        x = torch.relu(x) * 2.0
    else:
        x = torch.tanh(x)
    while (x * x).sum() > 1.0:
        x = x * 0.5
    return x


xs = (torch.randn(256, 16, dtype=F64) * 3).float().double()
vb = vmap(branchy)
got = ir.run(vb, xs)[0]
loop = torch.stack([torch_branchy(x) for x in xs])
rows = [("if and while on tensor data", outcome(lambda: tvmap(torch_branchy)(xs)),
         f"runs; matches the loop to {rel(got, loop):.1e}")]

drop = lambda x: torch.nn.functional.dropout(x, 0.5, training=True)  # noqa: E731
b = ir.Builder("sample")
p = b.input("p", F64, (5,))
prog_s = b.build([b.op("multinomial", p, seed=3, num_samples=2, replacement=True)])
vs = ir.run(vmap(prog_s), torch.rand(4, 5, dtype=F64) + 0.1)[0]
rows.append(("random op, default flags", outcome(lambda: tvmap(drop)(torch.randn(4, 8))),
             f"runs; per element streams, 4 elements give {len({tuple(r.tolist()) for r in vs})} distinct draws"))
rows.append(("torch._foreach_add (pytorch#200456)",
             outcome(lambda: tvmap(lambda a, c: torch._foreach_add([a], [c])[0])(torch.randn(4, 3), torch.randn(4, 3))),
             "no foreach op in the IR; elementwise add batches by rule"))
q = torch.randn(4, 1, 3, 8)
rows.append(("vmap(grad) through CPU attention",
             outcome(lambda: tvmap(tgrad(lambda a: torch.nn.functional.scaled_dot_product_attention(a, a, a).sum()))(q)),
             "attention batches by rule (one fused op, no fallback)"))
rows.append(("nonzero (data dependent shape)", outcome(lambda: tvmap(torch.nonzero)(torch.randn(4, 3))),
             "not an IR op: shapes are types, refused at import"))
print("| case | torch.func.vmap | truestar vmap |\n|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} |")

# ----------------------------------------------------------------------------- 2. speed: vmap vs a loop

print("\n## 2. vmap against a Python loop (CPU, float64, best of 21)\n")


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Tanh(), torch.nn.Linear(64, 10)).double()

    def forward(self, x, y):
        d = self.m(x) - y
        return (d * d).mean()


mod = Loss()
B = 64
x, y = torch.randn(B, 32, dtype=F64), torch.randn(B, 10, dtype=F64)
prog = ir.import_module(mod, (x[0], y[0]))
names = [n for n, _ in mod.named_parameters()]
t0 = time.perf_counter()
gp = grad(prog, names)
ps = vmap(gp, in_axes={prog.inputs[0].name: 0, prog.inputs[1].name: 0})
t_transform = time.perf_counter() - t0
ts_vmap = timeit(lambda: ir.run(ps, x, y))
ts_loop = timeit(lambda: [ir.run(gp, x[k], y[k]) for k in range(B)])
params = {k: v.detach() for k, v in mod.named_parameters()}
tf = tvmap(tgrad(lambda pp, a, c: functional_call(mod, pp, (a, c))), in_dims=(None, 0, 0))
tf_vmap = timeit(lambda: tf(params, x, y))


def torch_loop():
    out = []
    for k in range(B):
        loss = mod(x[k], y[k])
        out.append(torch.autograd.grad(loss, list(mod.parameters())))
    return out


tf_loop = timeit(torch_loop)
got = ir.run(ps, x, y)
want = tf(params, x, y)
err = max(rel(got[1 + j], want[n]) for j, n in enumerate(names))
rows = [("per sample gradients, MLP 32-64-10, batch 64", ts_loop, ts_vmap, tf_loop, tf_vmap)]
ts_bl = timeit(lambda: [ir.run(branchy.program, xx) for xx in xs])
ts_bv = timeit(lambda: ir.run(vb, xs))
tl = timeit(lambda: [torch_branchy(xx) for xx in xs])
rows.append(("branchy if and while, batch 256", ts_bl, ts_bv, tl, None))
print(f"per sample gradients: transform {t_transform * 1e3:.0f} ms, nodes {len(gp.nodes)} -> {len(ps.nodes)}, "
      f"agreement with torch.func {err:.1e}\n")
print("| workload | truestar loop | truestar vmap | speedup | torch loop | torch.func.vmap | speedup |")
print("|---|---|---|---|---|---|---|")
for name, a, bb, c, d in rows:
    tv = f"{d * 1e3:.2f} ms" if d else "refused"
    sp = f"{c / d:.1f}x" if d else "n/a"
    print(f"| {name} | {a * 1e3:.1f} ms | {bb * 1e3:.2f} ms | {a / bb:.0f}x | {c * 1e3:.2f} ms | {tv} | {sp} |")

try:
    from truestar.compiler import CompiledExecutor
    ex = CompiledExecutor("cpu", "float32")
    cp = ex.prepare(ps)
    x32, y32 = x.float(), y.float()
    tc = timeit(lambda: cp([x32, y32]))
    cerr = max(rel(a, b_) for a, b_ in zip(cp([x32, y32]), got))
    mod32 = Loss().float()
    mod32.load_state_dict({k: v.float() for k, v in mod.state_dict().items()})
    p32 = {k: v.detach() for k, v in mod32.named_parameters()}
    tf32 = tvmap(tgrad(lambda pp, a, c: functional_call(mod32, pp, (a, c))), in_dims=(None, 0, 0))
    t32 = timeit(lambda: tf32(p32, x32, y32))
    print(f"\ncompiled CPU executor (float32) on the vmapped per sample gradient program: {tc * 1e3:.2f} ms "
          f"(torch.func.vmap float32: {t32 * 1e3:.2f} ms), agrees with the float64 reference to {cerr:.1e}")
except Exception as e:  # noqa: BLE001
    print(f"\ncompiled CPU executor: not run ({type(e).__name__}: {str(e).splitlines()[0][:160]})")

# ----------------------------------------------------------------------------- 3. Jacobian and Hessian speed

print("\n## 3. Jacobians and Hessians against torch.func (CPU, float64, best of 21)\n")
net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 16)).double()
x8 = torch.randn(8, 16, dtype=F64)
pj = ir.import_module(net, (x8,))
t0 = time.perf_counter()
jf = jacfwd(pj, [pj.inputs[0].name])
t_jf = time.perf_counter() - t0
r_jf = timeit(lambda: ir.run(jf, x8))
tj = timeit(lambda: tjacfwd(net)(x8))
e_jf = rel(ir.run(jf, x8)[1], tjacfwd(net)(x8))


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = net

    def forward(self, a):
        h = self.net(a)
        return (h * h).sum()


sc = Scalar()
ph = ir.import_module(sc, (x8,))
t0 = time.perf_counter()
hp = hessian(ph, [ph.inputs[0].name])
t_h = time.perf_counter() - t0
r_h = timeit(lambda: ir.run(hp, x8))
th = timeit(lambda: thessian(sc)(x8))
e_h = rel(ir.run(hp, x8)[1], thessian(sc)(x8))
print("| quantity | truestar transform | truestar run | torch.func | agreement | nodes |")
print("|---|---|---|---|---|---|")
print(f"| jacfwd, MLP 16-32-16, input [8, 16] (128 x 128) | {t_jf * 1e3:.0f} ms | {r_jf * 1e3:.2f} ms | "
      f"{tj * 1e3:.2f} ms | {e_jf:.1e} | {len(pj.nodes)} -> {len(jf.nodes)} |")
print(f"| hessian, same MLP, scalar loss (128 x 128) | {t_h * 1e3:.0f} ms | {r_h * 1e3:.2f} ms | "
      f"{th * 1e3:.2f} ms | {e_h:.1e} | {len(ph.nodes)} -> {len(hp.nodes)} |")
sys.stdout.flush()
