"""Float64 islands start only where the model declares float64 (P63), not at values that are float64
by calling convention: a train step's ``lr`` and optimizer ``step`` inputs (Python numbers) and the
optimizer math computed from them stay in the working precision of a float32 program, as before
the islands, so default train steps keep their storage and their bits."""

import torch
from torch import nn

from truestar import ir, train as T
from truestar.compiler import compile_program
from truestar.compiler.lower import convention_values, input_storage_dtypes


def _step(opt):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))
    prog = ir.import_module(net, (torch.randn(4, 16),))
    return T.build_train_step(prog, T.CrossEntropy(), opt, params=net)


def _inputs(st, lr):
    g = torch.Generator().manual_seed(1)
    ins = []
    for i, v in enumerate(st.program.inputs):
        if i == 0:
            ins.append(torch.randn(4, 16, generator=g))
        elif i == 1:
            ins.append(torch.tensor([1, 2, 3, 4]))
        elif v.name == "lr":
            ins.append(torch.tensor(lr, dtype=torch.float64))
        else:
            t = st.initial[v.name]
            ins.append(t.float() if t.is_floating_point() else t)
    return ins


def test_train_step_lr_and_step_are_not_float64_islands():
    for opt in (T.AdamW(1e-3, weight_decay=0.01), T.SGD(1e-2, momentum=0.9)):
        st = _step(opt)
        names = [v.name for v in st.program.inputs]
        assert "lr" in names
        cp = compile_program(st.program, "cpu", "float32")
        assert cp.report.islands == [], cp.report.islands
        dts = dict(zip(names, input_storage_dtypes(st.program, cp.opts)))
        assert dts["lr"] == torch.float32
        assert all(d in (None, torch.float32) for d in dts.values()), dts
        outs = cp(_inputs(st, opt.lr))             # first call checks against float64 pass
        assert all(torch.isfinite(o).all() for o in outs if o.is_floating_point())


def test_model_declared_float64_still_forms_an_island():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("f", torch.linspace(0.1, 1.0, 8))

        def forward(self, pos):
            a = pos.double()[:, None] * self.f.double()    # declared float64 by the model
            return torch.cos(a).float()

    prog = ir.import_module(M(), (torch.arange(4.0),))
    cp = compile_program(prog, "cpu", "float32")
    assert cp.report.islands and all(d == "float64" for _, d, _ in cp.report.islands)
    assert convention_values(prog, cp.opts) == set()
    pos = torch.tensor([10.0, 3000.0, 20000.0, 32000.0])
    out = cp([pos])
    ref = torch.cos(pos.double()[:, None] * torch.linspace(0.1, 1.0, 8).double()).float()
    assert torch.equal(out[0] if isinstance(out, (list, tuple)) else out, ref)
