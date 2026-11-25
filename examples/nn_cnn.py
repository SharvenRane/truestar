"""Train a small CNN with batch norm on generated images in native Truestar, next to PyTorch.

    python examples/nn_cnn.py            # float64 loss curves, about 15 s
    python examples/nn_cnn.py --bench    # also float32: compiled CPU (and CUDA) vs eager

The data: 16x16 single channel images of one of four generated shapes (horizontal bar, vertical
bar, cross, box) at a random place, with noise. The model: conv, batch norm, relu, max pool, conv,
batch norm, relu, average pool, linear. Batch norm running statistics are explicit state: the loss
takes them and returns the new ones, and the step carries them. SGD with Nesterov momentum and
weight decay on both sides. Prints the worst relative difference between the loss curves and
between the final running variances; in float64 both are rounding noise (bar 1e-9).
"""

import argparse
import statistics
import time

import torch

import truestar
from truestar import Tensor, nn, optim

B, S, K = 32, 16, 4


def model(dtype):
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, dtype=dtype), nn.BatchNorm(8, dtype=dtype), nn.ReLU(),
                         nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1, dtype=dtype), nn.BatchNorm(16, dtype=dtype), nn.ReLU(),
                         nn.AvgPool2d(4), nn.Flatten(), nn.Linear(16 * 2 * 2, K, dtype=dtype))


def make_loss(net, dtype):
    @truestar.native
    def loss(p: net.spec, s: net.state_spec, x: Tensor["B", 1, S, S, dtype], y: Tensor["B", "i64"]):
        logits, s2 = net(p, x, training=True, state=s)
        return nn.cross_entropy(logits, y), s2
    return loss


def data(n_batches, dtype):
    g = torch.Generator().manual_seed(1)
    ys = torch.randint(0, K, (n_batches, B), generator=g)
    xs = 0.3 * torch.randn(n_batches, B, 1, S, S, generator=g, dtype=torch.float64)
    at = torch.randint(3, S - 3, (n_batches, B, 2), generator=g)
    for i in range(n_batches):
        for j in range(B):
            r, c = int(at[i, j, 0]), int(at[i, j, 1])
            k = int(ys[i, j])
            img = xs[i, j, 0]
            if k in (0, 2):
                img[r, c - 3:c + 4] += 1.0
            if k in (1, 2):
                img[r - 3:r + 4, c] += 1.0
            if k == 3:
                img[r - 2:r + 3, c - 2] += 1.0
                img[r - 2:r + 3, c + 2] += 1.0
                img[r - 2, c - 2:c + 3] += 1.0
                img[r + 2, c - 2:c + 3] += 1.0
    return xs.to(dtype), ys


def torch_model(p, dtype):
    m = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                            torch.nn.MaxPool2d(2),
                            torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
                            torch.nn.AvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(64, K)).to(dtype)
    with torch.no_grad():
        for i in (0, 4, 9):
            m[i].weight.copy_(p[str(i)]["w"])
            m[i].bias.copy_(p[str(i)]["b"])
        for i in (1, 5):
            m[i].weight.copy_(p[str(i)]["w"])
            m[i].bias.copy_(p[str(i)]["b"])
    return m


OPT = optim.SGD(lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)


def run_torch(p, xs, ys, steps, device="cpu"):
    m = torch_model(p, xs.dtype).to(device)
    to = torch.optim.SGD(m.parameters(), lr=OPT.lr, momentum=OPT.momentum, nesterov=True,
                         weight_decay=OPT.weight_decay)
    losses, times = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        x, y = xs[i].to(device), ys[i].to(device)
        to.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        to.step()
        losses.append(loss.item())
        times.append(time.perf_counter() - t0)
    return losses, statistics.median(times[3:]), m[5].running_var.detach().cpu()


def run_truestar(step, net, p, xs, ys, steps, device="cpu"):
    mv = lambda t: {k: {kk: x.to(device) for kk, x in v.items()} for k, v in t.items()}  # noqa: E731
    ts = step.init(mv(p), mv(net.init_state()))
    losses, times = [], []
    for i in range(steps):
        t0 = time.perf_counter()
        ts, r = step(ts, xs[i].to(device), ys[i].to(device))
        losses.append(r.loss)
        times.append(time.perf_counter() - t0)
    return losses, statistics.median(times[3:]), ts.model_state["5"]["var"].cpu()


def worst_rel(a, b):
    return max(abs(x - y) / abs(y) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    steps = args.steps

    net = model("f64")
    p = net.init(seed=0)
    xs, ys = data(steps, torch.float64)
    step = truestar.train_step(make_loss(net, "f64"), OPT, state="s")
    ours, _, var = run_truestar(step, net, p, xs, ys, steps)
    ref, _, rvar = run_torch(p, xs, ys, steps)
    dv = float(((var - rvar).abs() / rvar.abs()).max())
    print(f"float64 ({steps} steps): loss {ours[0]:.4f} -> {ours[-1]:.4f}; worst relative difference to "
          f"PyTorch {worst_rel(ours, ref):.2e}, running variance {dv:.2e} (bar 1e-9)")
    print(f"step program: {len(step.program.nodes)} IR nodes, {net.num_params()} params")

    if args.bench:
        net32 = model("f32")
        p32 = net32.init(seed=0)
        xs32, _ = data(steps, torch.float32)
        loss32 = make_loss(net32, "f32")
        for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            t0 = time.perf_counter()
            st = truestar.train_step(loss32, OPT, state="s", executor=dev)
            ours32, t_ours, _ = run_truestar(st, net32, p32, xs32, ys, steps, dev)
            first = time.perf_counter() - t0
            ref32, t_ref, _ = run_torch(p32, xs32, ys, steps, dev)
            print(f"float32 {dev}: worst relative loss difference {worst_rel(ours32, ref32):.2e}; per step "
                  f"Truestar compiled {t_ours * 1e3:.2f} ms, PyTorch eager {t_ref * 1e3:.2f} ms "
                  f"(PyTorch / Truestar {t_ref / t_ours:.2f}); whole run with compile and first call check "
                  f"{first:.1f} s; median of steps 4 on, preliminary (noisy shared machine)")


if __name__ == "__main__":
    main()
