"""Constant linear weights stored [K, N] at compile time (compiler.lower.kn_weights).

A strict float32 program (CUDA, TF32 off) stores a 2 dim float32 param whose every use is the
weight of a linear with the same values and shape but strides (1, N): the Triton contraction then
reads the conflict free layout and runs the same fused multiply add chain, so the bits are those
of the [N, K] weight. Weights that are inputs, params with other uses, outputs, TF32 variants and
half precision programs keep the plain layout.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import ir  # noqa: E402
from truestar import kernel as K  # noqa: E402
from truestar.compiler import compile_program  # noqa: E402
from truestar.compiler.lower import Options, kn_weights  # noqa: E402
from truestar.kernel import Graph  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _has_gpu(), reason="needs CUDA and Triton")


class Tied(nn.Module):
    """One weight read by a linear and by an elementwise op: not relaid."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 64)

    def forward(self, x):
        return self.lin(x) + self.lin.weight.sum()


def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 96)).eval()


def test_which_params_are_relaid():
    m = _mlp()
    prog = ir.import_module(m, (torch.randn(8, 256),), name="mlp")
    strict = Options(device="cuda", dtype="float32", allow_tf32=False)
    names = kn_weights(prog, strict)
    weights = {p.name for p in prog.params if prog.param_data[p.name].dim() == 2}
    assert names == weights and len(names) == 2          # both weights, never the biases
    assert kn_weights(prog, Options(device="cuda", dtype="float32", allow_tf32=True)) == set()
    assert kn_weights(prog, Options(device="cuda", dtype="float16")) == set()
    assert kn_weights(prog, Options(device="cpu", dtype="float32")) == set()
    assert kn_weights(prog, Options(device="cuda", dtype="float32", platform="rocm")) == set()
    tied = ir.import_module(Tied().eval(), (torch.randn(4, 64),), name="tied")
    assert kn_weights(tied, strict) == set()


@needs_gpu
@pytest.mark.parametrize("det", [True, False])
def test_relaid_weights_give_the_bits_of_the_plain_layout(det):
    """The compiled MLP stores its weights [K, N] and equals, bit for bit, the same kernels fed
    the [N, K] weights; two runs equal; float32 accurate against float64."""
    m = _mlp(1)
    x = torch.randn(64, 256)
    prog = ir.import_module(m, (x,), name="mlp_kn")
    cp = compile_program(prog, "cuda", "float32", allow_tf32=False, deterministic=det)
    xc = x.cuda()
    out = cp([xc])[0]
    assert torch.equal(out, cp([xc])[0])
    plan = cp.plan
    assert len(plan.kn_params) == 2
    for p in prog.params:
        t = plan.template[plan.lowering.vals[id(p)].slot]
        if p.name in plan.kn_params:
            assert t.stride() == (1, t.shape[0]) and torch.equal(t.cpu(), prog.param_data[p.name])
    # the same math through the kernel layer with [N, K] weights
    w1, b1, w2, b2 = (t.detach().cuda() for t in (m[0].weight, m[0].bias, m[2].weight, m[2].bias))
    g = Graph()
    xi = g.input("x", (K.sym("M"), 256), "float32")
    a, ab = g.input("w1", (512, 256), "float32"), g.input("b1", (512,), "float32")
    (_, k1), = g.fuse({"h": g.relu(xi @ g.transpose(a, 0, 1) + ab)})
    g = Graph()
    hi = g.input("h", (K.sym("M"), 512), "float32")
    c, cb = g.input("w2", (96, 512), "float32"), g.input("b2", (96,), "float32")
    (_, k2), = g.fuse({"y": hi @ g.transpose(c, 0, 1) + cb})
    h = K.compile(k1, "cuda", tune=False, deterministic=det)(x=xc, w1=w1, b1=b1)
    y = K.compile(k2, "cuda", tune=False, deterministic=det)(h=h, w2=w2, b2=b2)
    if det:     # default mode may tune to another split K, which changes bits by design
        assert torch.equal(out, y)
    ref = m.double()(x.double())
    assert ((out.double().cpu() - ref).abs().max() / ref.abs().max()).item() < 1e-5


@needs_gpu
def test_weight_passed_as_an_input_keeps_its_layout():
    class Lin(nn.Module):
        def forward(self, x, w):
            return torch.nn.functional.linear(x, w)
    x, w = torch.randn(16, 128), torch.randn(64, 128)
    prog = ir.import_module(Lin(), (x, w), name="lin_in")
    cp = compile_program(prog, "cuda", "float32", allow_tf32=False, deterministic=True)
    assert cp.plan.kn_params == set()
    out = cp([x.cuda(), w.cuda()])[0]
    assert ((out.double().cpu() - x.double() @ w.double().T).abs().max()).item() < 1e-4
