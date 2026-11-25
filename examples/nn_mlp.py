"""Train an MLP classifier on synthetic data in native Truestar, next to the same model in PyTorch.

    python examples/nn_mlp.py            # float64 loss curves, about 10 s
    python examples/nn_mlp.py --bench    # also float32: compiled CPU (and CUDA) vs eager

The data: 2048 points in 32 dims, labelled by a fixed random teacher network (10 classes). Both
sides start from Truestar's keyed initialization (copied into the torch model), see the same
batches and use AdamW with warmup and cosine decay and clipping at 1.0. Prints the worst relative
difference between the two loss curves; in float64 it is rounding noise (bar 1e-9).
"""

import argparse
import statistics
import time

import torch

import truestar
from truestar import Tensor, nn, optim

D, H, C, B = 32, 128, 10, 64


def model(dtype):
    return nn.Sequential(nn.Linear(D, H, dtype=dtype), nn.GELU(), nn.Linear(H, H, dtype=dtype), nn.GELU(),
                         nn.Linear(H, C, dtype=dtype))


def make_loss(net, dtype):
    @truestar.native
    def loss(p: net.spec, x: Tensor["B", D, dtype], y: Tensor["B", "i64"]) -> Tensor[dtype]:
        return nn.cross_entropy(net(p, x), y)
    return loss


def data(n_batches, dtype):
    g = torch.Generator().manual_seed(0)
    teacher = torch.randn(D, C, generator=g, dtype=torch.float64)
    xs = torch.randn(n_batches, B, D, generator=g, dtype=torch.float64)
    ys = (torch.tanh(xs) @ teacher).argmax(-1)
    return xs.to(dtype), ys


def torch_model(p, dtype):
    m = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.GELU(), torch.nn.Linear(H, H), torch.nn.GELU(),
                            torch.nn.Linear(H, C)).to(dtype)
    with torch.no_grad():
        for i in (0, 2, 4):
            m[i].weight.copy_(p[str(i)]["w"])
            m[i].bias.copy_(p[str(i)]["b"])
    return m


def run_torch(p, opt, xs, ys, steps, device="cpu"):
    m = torch_model(p, xs.dtype).to(device)
    to = torch.optim.AdamW(m.parameters(), lr=1.0, weight_decay=opt.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(to, lambda i: opt.lr_at(i))
    losses, times = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        x, y = xs[i].to(device), ys[i].to(device)
        to.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        to.step()
        sched.step()
        losses.append(loss.item())
        times.append(time.perf_counter() - t0)
    return losses, statistics.median(times[3:])


def run_truestar(step, p, xs, ys, steps, device="cpu"):
    ts = step.init({k: {kk: t.to(device) for kk, t in v.items()} for k, v in p.items()})
    losses, times = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        ts, r = step(ts, xs[i].to(device), ys[i].to(device))
        losses.append(r.loss)
        times.append(time.perf_counter() - t0)
    return losses, statistics.median(times[3:])


def worst_rel(a, b):
    return max(abs(x - y) / abs(y) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    steps = args.steps
    opt = optim.AdamW(lr=optim.warmup_cosine(3e-3, 10, steps), weight_decay=0.01)

    net = model("f64")
    p = net.init(seed=0)
    xs, ys = data(steps, torch.float64)
    step = truestar.train_step(make_loss(net, "f64"), opt, clip=1.0)
    ours, _ = run_truestar(step, p, xs, ys, steps)
    ref, _ = run_torch(p, opt, xs, ys, steps)
    print(f"float64 ({steps} steps): loss {ours[0]:.4f} -> {ours[-1]:.4f}; worst relative difference to "
          f"PyTorch {worst_rel(ours, ref):.2e} (bar 1e-9)")
    print(f"step program: {len(step.program.nodes)} IR nodes, {net.num_params()} params")

    if args.bench:
        net32 = model("f32")
        p32 = net32.init(seed=0)
        xs32, _ = data(steps, torch.float32)
        loss32 = make_loss(net32, "f32")
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            st = truestar.train_step(loss32, opt, clip=1.0, executor=dev)
            t0 = time.perf_counter()
            ours32, t_ours = run_truestar(st, p32, xs32, ys, steps, dev)
            first = time.perf_counter() - t0
            ref32, t_ref = run_torch(p32, opt, xs32, ys, steps, dev)
            print(f"float32 {dev}: worst relative loss difference {worst_rel(ours32, ref32):.2e}; per step "
                  f"Truestar compiled {t_ours * 1e3:.2f} ms, PyTorch eager {t_ref * 1e3:.2f} ms "
                  f"(PyTorch / Truestar {t_ref / t_ours:.2f}); whole run with compile and first call check "
                  f"{first:.1f} s; median of steps 4 on, preliminary (noisy shared machine)")


if __name__ == "__main__":
    main()
