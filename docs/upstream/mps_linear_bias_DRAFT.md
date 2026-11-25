# DRAFT, not filed: [MPS] `F.linear` with bias returns wrong results on macOS 26

Status: **do not file as a torch bug.** Not reproduced on a physical Mac (2026-11-25: Mac mini M2,
macOS 26.6.2, torch 2.14.1 and 2.15.0, 0 of 184 cases wrong). It shows only on hosted macos-26
runners with a paravirtual GPU, so it belongs to that virtual GPU, not to torch's MPS kernels.

## Summary

On MPS, `torch.nn.functional.linear(x, w, b)` (so every `nn.Linear` with a bias) returns results
that differ from CPU by up to 87% relative error. The same computation without the bias, or written
as `x @ w.T + b`, or as `torch.addmm(b, x, w.T)`, is correct. MPS runs `aten.linear.default` as its
own fused kernel, while CPU decomposes it into addmm, which is why only the fused path is affected.
The same torch build on macOS 15 is correct.

## Environment

* macOS: 26.7.2 (build to fill in), GitHub runner `macos-26` (paravirtual GPU) / physical Mac: TO FILL
* torch: 2.14.1 (also check latest)
* Correct on: macOS 15.8.2, same torch 2.14.1

## Minimal repro (PyTorch only)

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(4, 512)
w = torch.randn(10, 512) / 512 ** 0.5
b = torch.randn(10)

def rel(a, b):
    a, b = a.cpu().double(), b.double()
    return ((a - b).norm() / b.norm()).item()

ref = F.linear(x.double(), w.double(), b.double())
xm, wm, bm = x.to("mps"), w.to("mps"), b.to("mps")
print("linear with bias   ", rel(F.linear(xm, wm, bm), ref))   # wrong on macOS 26
print("linear without bias", rel(F.linear(xm, wm), F.linear(x.double(), w.double())))
print("matmul plus bias   ", rel(xm @ wm.t() + bm, ref))
print("addmm              ", rel(torch.addmm(bm, xm, wm.t()), ref))
```

## Observed (GitHub macos-26 runner, torch 2.14.1)

* `F.linear` with bias: wrong in 40 of 40 shapes tested, max relative error 0.87
  (shapes from (1, 16)x(32, 16) up to (2, 17, 768)x(3072, 768)).
* `F.linear` without bias, `x @ w.T`, `torch.addmm`: all 144 cases correct (relative error at most 5e-7).
* Real models: ResNet18 `fc` relative error 3.2e-2; transformer feed forward layers 8.8e-2 to 1.6e-1.
* Same torch 2.14.1 on macOS 15.8.2: 184 of 184 probe cases correct.

## How it was found

Found with a cross backend checker that replays each module alone on each backend against a float64
CPU reference, then drills down to the ATen op: it named `aten.linear.default` on MPS, an op the CPU
reference never runs (CPU uses `aten.addmm.default`). A single op repro calling
`torch.ops.aten.linear.default` with the saved tensors reproduces it.
