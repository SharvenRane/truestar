"""Probe: how far off is the batch norm weight gradient on CPU float16, against float64?

PyTorch only (torchvision for part B). Run it on each torch version to compare:

    python scripts/probe_bn_fp16_backward.py

Context: tests/test_gradcheck.py::test_clean_float16_resnet_reports_clean failed on Linux with
torch 2.14.1 (CPU only): `<model>#0` gradient of layer4.1.bn2.weight, own error 1.13e-01 against
a threshold of 8.8e-03, while the same tensor is about 1e-03 on Windows with torch 2.11, and the
leaf replay of that batch norm alone passed on both.

Part A is the batch norm alone at the test's shape for that layer, [2, 512, 1, 1] in eval mode,
so its weight gradient is a sum over N*H*W = 2 terms per channel. It also prints what pure float16
and float32 accumulation of that sum give. Part B runs the whole resnet18 of the test in float16
and in float64 and compares, at layer4.1.bn2, the forward input, the upstream gradient and the
weight gradient, with oneDNN on and off, to show where the error enters.
"""
import platform

import torch
from torch import nn


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def part_a(shape=(2, 512, 1, 1), seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    up = torch.randn(shape, generator=g, dtype=torch.float64)
    bn = nn.BatchNorm2d(shape[1]).eval().double()
    with torch.no_grad():
        bn.running_mean.copy_(torch.randn(shape[1], generator=g, dtype=torch.float64) * 0.1)
        bn.running_var.copy_(torch.rand(shape[1], generator=g, dtype=torch.float64) + 0.5)
    out = {}
    for dt in (torch.float64, torch.float16):
        m = nn.BatchNorm2d(shape[1]).eval().to(dt)
        m.load_state_dict({k: v.to(dt) if v.is_floating_point() else v for k, v in bn.state_dict().items()})
        xi = x.to(dt).requires_grad_(True)
        m(xi).backward(up.to(dt))
        out[dt] = m.weight.grad.double()
    print(f"A  batch norm alone {list(shape)}: weight grad rel err float16 vs float64 = "
          f"{rel(out[torch.float16], out[torch.float64]):.3e}")
    # The weight gradient is sum over (N, H, W) of up * xhat. Accumulate it in float16 and in
    # float32 from float16 rounded terms, one term at a time, as a naive kernel would.
    xhat = (x - bn.running_mean[None, :, None, None]) / torch.sqrt(
        bn.running_var[None, :, None, None] + bn.eps)
    terms = (up.half().double() * xhat.half().double()).permute(1, 0, 2, 3).reshape(shape[1], -1)
    exact = (up * xhat).permute(1, 0, 2, 3).reshape(shape[1], -1).sum(1)
    for acc in (torch.float16, torch.float32):
        s = torch.zeros(shape[1], dtype=acc)
        for k in range(terms.shape[1]):
            s = (s + terms[:, k].to(acc)).to(acc)
        print(f"A  sum of {terms.shape[1]} terms accumulated in {str(acc)[6:]}: rel err = "
              f"{rel(s.half(), exact):.3e}")


def part_b():
    try:
        import torchvision
    except ImportError:
        print("B  skipped: torchvision not installed")
        return
    torch.manual_seed(0)
    base = torchvision.models.resnet18(num_classes=10).eval()
    x = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 10, generator=torch.Generator().manual_seed(0))

    def run(dt, mkldnn):
        m = torchvision.models.resnet18(num_classes=10).eval()
        m.load_state_dict(base.state_dict())
        m = m.to(dt)
        seen = {}
        bn = m.layer4[1].bn2
        def hook(mod, i, o):
            seen["input"] = i[0].detach().double()
            o.register_hook(lambda g: seen.__setitem__("upstream", g.detach().double()))

        bn.register_forward_hook(hook)
        with torch.backends.mkldnn.flags(enabled=mkldnn):
            y = m(x.to(dt))
            ((y.double() - target.double()) ** 2).mean().backward()
        seen["weight_grad"] = bn.weight.grad.double()
        seen["output"] = y.detach().double()
        return seen

    for mkldnn in (True, False):
        ref = run(torch.float64, mkldnn)
        got = run(torch.float16, mkldnn)
        parts = ", ".join(f"{k} {rel(got[k], ref[k]):.3e}" for k in ("output", "input", "upstream", "weight_grad"))
        print(f"B  resnet18 float16 vs float64, oneDNN {'on' if mkldnn else 'off'}, layer4.1.bn2: {parts}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, python {platform.python_version()}, {platform.system()} "
          f"{platform.machine()}, oneDNN available {torch.backends.mkldnn.is_available()}, "
          f"threads {torch.get_num_threads()}")
    print(f"CPU capability: {torch.backends.cpu.get_cpu_capability()}")
    part_a()
    part_b()
