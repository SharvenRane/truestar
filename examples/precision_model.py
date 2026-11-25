"""Example for `truestar compile --precision auto`: an MLP with one layer that 16 bit floats ruin.

    truestar compile examples/precision_model.py --max-flips 1%
    truestar compile examples/precision_model.py --max-flips 1% --format agent --plan plan.json

`OnePassNorm` computes the variance as mean(x * x) - mean(x) ** 2 on sensor readings near 30
with a spread of about 0.3. In float32 that is fine (x * x near 900 resolves steps of 6e-5). In
float16 the steps are 0.5 and in bfloat16 4, so the difference, the variance, is mostly rounding
error. Every other layer is happy in float16, bfloat16 or a quantized kernel. The planner keeps
`norm` in float32 (truestar range names the subtraction, and its measured error says the same)
and makes the rest fast. "fp16 except norms" does not help: `OnePassNorm` is not a torch norm
class, so a rule by class puts it in float16. The weights are trained in code on a synthetic
task for about a second on the CPU. Needs a CUDA GPU.
"""
import torch
from torch import nn


class OnePassNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x * x).mean(-1, keepdim=True) - mean * mean  # cancellation below float32
        return (x - mean) * torch.rsqrt(var.clamp_min(0) + 1e-5)


class Net(nn.Module):
    def __init__(self, d=256, h=1024, classes=16):
        super().__init__()
        self.norm = OnePassNorm()
        self.fc1 = nn.Linear(d, h)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(h, h)
        self.act2 = nn.GELU()
        self.head = nn.Linear(h, classes)

    def forward(self, x):
        return self.head(self.act2(self.fc2(self.act1(self.fc1(self.norm(x))))))


def _data(n, seed, d=256, classes=16):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, d, generator=torch.Generator().manual_seed(7))
    y = torch.randint(0, classes, (n,), generator=g)
    x = centers[y] * 0.35 + torch.randn(n, d, generator=g)
    return 30 + 0.3 * x, y  # readings with a large offset and a small spread, as raw sensor data has


def get_model():
    torch.manual_seed(0)
    m = Net()
    x, y = _data(2048, 0)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(60):
        opt.zero_grad()
        nn.functional.cross_entropy(m(x), y).backward()
        opt.step()
    return m.eval()


def get_inputs():
    return _data(1024, 1)[0]


def get_holdout():
    return [(_data(1024, 2)[0],)]
