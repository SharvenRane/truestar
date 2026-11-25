"""Constant tables the importer and autodiff build (interpolation matrices, fold taps, a folded
erfinv, average pooling gradient tables) are stored in their declared dtype, so a float32 program
holding one is written and read back as an artifact (ir/serialize.py refuses float64 data for a
float32 tensor), and its values still match torch."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from truestar import ir, train as T
from truestar.ir import reference as iref, serialize


class Interp(nn.Module):
    def __init__(self, mode, **kw):
        super().__init__()
        self.mode, self.kw = mode, kw

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode=self.mode, **self.kw)


class Fold(nn.Module):
    def forward(self, x):
        return F.fold(x, output_size=(5, 5), kernel_size=(2, 2))


class Erfinv(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("q", torch.linspace(-0.9, 0.9, 6))

    def forward(self, x):
        return x * torch.special.erfinv(self.q)


CASES = [(Interp("bilinear", align_corners=False), (1, 2, 4, 4)),
         (Interp("bicubic", align_corners=True), (1, 2, 4, 4)),
         (Interp("bilinear", antialias=True), (1, 1, 4, 4)),
         (Fold(), (1, 8, 16)),
         (Erfinv(), (3, 6))]


def _float_tables_declared(prog):
    for p in prog.params:
        d = prog.param_data[p.name]
        if d.dtype.is_floating_point and d.dim() > 0:
            assert d.dtype == p.type.dtype, (p.name, d.dtype, p.type)


@pytest.mark.parametrize("m,shape", CASES, ids=["bilinear", "bicubic", "antialias", "fold", "erfinv"])
def test_float32_import_with_a_constant_table_exports(m, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape)
    prog = ir.import_module(m.eval(), (x,))
    _float_tables_declared(prog)
    art = serialize.loads(serialize.dumps(prog))      # refused before: float64 data, float32 value
    got = iref.run(art.program, x)[0]
    want = m(x)
    # float32 rounding of a weighted sum is relative to the inputs, not to an output that cancels
    # to near zero, so the absolute bar scales with the input size
    assert torch.allclose(got.float(), want, rtol=1e-5, atol=1e-5 * float(x.abs().max()))


def test_float64_import_keeps_exact_tables():
    m = Interp("bicubic", align_corners=True).double()
    x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    prog = ir.import_module(m, (x,))
    tabs = [prog.param_data[p.name] for p in prog.params if prog.param_data[p.name].dim() > 0]
    assert tabs and all(t.dtype == torch.float64 for t in tabs)
    assert torch.allclose(iref.run(prog, x)[0], m(x), rtol=1e-12, atol=1e-12)


def test_avg_pool_train_step_exports():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.AvgPool2d(2), nn.Flatten(), nn.Linear(4 * 4 * 4, 3))
    prog = ir.import_module(net, (torch.randn(2, 2, 8, 8),))
    st = T.build_train_step(prog, T.CrossEntropy(), T.SGD(1e-2), params=net)
    _float_tables_declared(st.program)
    serialize.loads(serialize.dumps(st.program))
