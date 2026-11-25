"""Op level drilldown: the exact ATen op inside a blamed module is named, and only that one."""

import copy
import json
import os
import runpy
import sys

import pytest
import torch
from torch import nn

import truestar
import truestar.opdrill as opdrill
from truestar.capture import Recorder
from truestar.compare import DIVERGENT, OK
from truestar.testing import DeviceFault, OpFault

# Same accelerator choice as tests/test_check.py: first non CPU backend, or TRUESTAR_TEST_BACKEND.
_ACCEL = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
if os.environ.get("TRUESTAR_TEST_BACKEND"):
    _ACCEL = next(b for b in truestar.available_backends()
                  if b.name == os.environ["TRUESTAR_TEST_BACKEND"])
ACCEL = _ACCEL.name if _ACCEL else None
ACCEL_DEV = _ACCEL.device if _ACCEL else None
ACCEL_TYPE = torch.device(_ACCEL.device).type if _ACCEL else None
needs_accel = pytest.mark.skipif(ACCEL is None, reason="needs a GPU backend (cuda, rocm, mps, xpu)")


def _linear_op() -> str:
    """The ATen op nn.Linear with bias dispatches to on the accelerator. CUDA decomposes it into
    aten.addmm; MPS runs its own fused aten.linear kernel. Tests inject faults into whichever runs."""
    if ACCEL is None:
        return "aten.addmm.default"
    from torch.utils._python_dispatch import TorchDispatchMode

    seen = []

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))

    lin = nn.Linear(4, 3).to(ACCEL_DEV)
    with torch.no_grad(), Rec():
        lin(torch.randn(2, 4, device=ACCEL_DEV))
    for name in ("aten.addmm.default", "aten.linear.default"):
        if name in seen:
            return name
    return seen[0] if seen else "aten.addmm.default"


LINEAR_OP = _linear_op()
FUSED_LINEAR = LINEAR_OP == "aten.linear.default"  # the reference CPU run decomposes linear


class MLP(nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fault = fault if fault is not None else nn.Identity()
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.fc2(self.fault(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        s = self.q(x) @ self.k(x).transpose(-1, -2) / 4.0
        return torch.softmax(s, dim=-1) @ self.v(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)
        self.attn = Attention()
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        return self.out(self.attn(self.norm(x)).mean(dim=1))


class FusedOnDevice(nn.Module):
    """Runs a fused kernel (addmm) only on the accelerator and mm + add elsewhere, like a backend
    with its own linear kernel. The reference never runs addmm."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(8, 16) / 4)
        self.b = nn.Parameter(torch.randn(8))

    def forward(self, x):
        if x.device.type == ACCEL_TYPE:
            return torch.addmm(self.b, x, self.w.t())
        return torch.mm(x, self.w.t()) + self.b


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(16, 16)
        self.fused = FusedOnDevice()
        self.post = nn.ReLU()

    def forward(self, x):
        return self.post(self.fused(torch.tanh(self.pre(x))))


def run(model, x, **kw):
    kw.setdefault("backends", ["cpu", ACCEL])
    return truestar.check(model, x, **kw)


def result(report, name):
    return next(r for r in report.results if r.backend.name == name)


# ----------------------------------------------------------------------------- names the op


@needs_accel
def test_linear_kernel_fault_is_named_as_the_op():
    torch.manual_seed(0)
    with OpFault(LINEAR_OP, ACCEL_TYPE, amount=0.5) as fault:
        rep = run(MLP(), torch.randn(32, 16))
    assert fault.hits > 0
    acc = result(rep, ACCEL)
    assert acc.first_divergent.name == "fc1", rep.to_text()
    op = acc.first_divergent_op
    assert op is not None, rep.to_text()
    assert op.op == LINEAR_OP and op.status == DIVERGENT, rep.to_text()
    assert op.matched is (not FUSED_LINEAR), rep.to_text()
    assert op.module == "fc1#0"
    expected = [[32, 16], [32, 16], [32]] if FUSED_LINEAR else [[32], [32, 16], [16, 32]]
    assert op.input_shapes == expected
    assert abs(op.isolated.rel_err - 0.5) < 1e-3
    assert f"first divergent op: {LINEAR_OP}" in rep.to_text()
    data = json.loads(rep.to_json())
    jr = next(r for r in data["results"] if r["backend"]["name"] == ACCEL)
    assert jr["first_divergent_op"]["op"] == LINEAR_OP
    cpu = result(rep, "cpu")
    assert cpu.status == OK and cpu.first_divergent_op is None, rep.to_text()


@needs_accel
def test_op_in_the_middle_of_parent_code_is_named():
    """softmax runs in the middle of attn (op 17 of 27 on CUDA); ops after it inherit the error but are not
    blamed, and the clean projection ops before it are not blamed either."""
    torch.manual_seed(0)
    with OpFault("aten._softmax.default", ACCEL_TYPE, amount=0.5) as fault:
        rep = run(Block(), torch.randn(4, 6, 16))
    assert fault.hits > 0
    acc = result(rep, ACCEL)
    assert acc.first_divergent.name == "attn", rep.to_text()
    op = acc.first_divergent_op
    assert op is not None and op.op == "aten._softmax.default", rep.to_text()
    assert 1 < op.index < op.total, rep.to_text()


@needs_accel
def test_op_that_only_the_backend_runs_is_named():
    torch.manual_seed(0)
    with OpFault("aten.addmm.default", ACCEL_TYPE, amount=0.5):
        rep = run(Net(), torch.randn(8, 16))
    acc = result(rep, ACCEL)
    # Where nn.Linear lowers to addmm (CUDA), pre is hit too and is the first divergent module.
    # Where it runs a fused linear kernel (MPS), pre is untouched and fused is blamed.
    assert acc.first_divergent.name == ("fused" if FUSED_LINEAR else "pre"), rep.to_text()
    assert acc.first_divergent_op.op == "aten.addmm.default"
    # Without the fault on pre, the fused module is blamed and its backend only addmm is named.
    m = Net()
    m.pre = nn.Identity()
    with OpFault("aten.addmm.default", ACCEL_TYPE, amount=0.5):
        rep = run(m, torch.randn(8, 16))
    acc = result(rep, ACCEL)
    assert acc.first_divergent.name == "fused", rep.to_text()
    op = acc.first_divergent_op
    assert op is not None and op.op == "aten.addmm.default" and not op.matched, rep.to_text()
    assert "does not run in the reference" in rep.to_text()


@needs_accel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_op_fault_is_named(dtype):
    torch.manual_seed(0)
    with OpFault(LINEAR_OP, ACCEL_TYPE, amount=0.5):
        rep = run(MLP().to(dtype), torch.randn(32, 16))
    op = result(rep, ACCEL).first_divergent_op
    assert op is not None and op.op == LINEAR_OP, rep.to_text()
    assert op.input_dtypes == [str(dtype).replace("torch.", "")] * 3


# ----------------------------------------------------------------------------- no false blame


@needs_accel
def test_device_branch_is_not_blamed_on_a_correct_op():
    """DeviceFault multiplies by 1.5 only on the device: aten.mul is right, the code is wrong."""
    rep = run(MLP(fault=DeviceFault(ACCEL_TYPE, "scale", 0.5)), torch.randn(32, 16))
    acc = result(rep, ACCEL)
    assert acc.first_divergent.name == "fault"
    assert acc.first_divergent_op is None, rep.to_text()
    assert any("no single ATen op" in n and "aten.mul.Tensor" in n for n in acc.notes), rep.to_text()


def _clean_cases():
    torch.manual_seed(4)
    cases = [
        (nn.Linear(16, 8), torch.randn(4, 16)),
        (Block(), torch.randn(4, 6, 16)),
        (nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True), torch.randn(2, 5, 32)),
        (nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2)),
         torch.randn(2, 3, 16, 16)),
        (nn.LSTM(8, 8, batch_first=True), torch.randn(2, 5, 8)),
    ]
    try:
        import torchvision

        cases.append((torchvision.models.resnet18(num_classes=10).layer1, torch.randn(2, 64, 16, 16)))
    except ImportError:
        pass
    return cases


@needs_accel
@pytest.mark.parametrize("idx", range(len(_clean_cases())))
def test_clean_module_reports_no_op(idx):
    module, x = _clean_cases()[idx]
    module.eval()
    import copy

    with torch.no_grad(), Recorder(module, keep_inputs=True) as rec:
        copy.deepcopy(module).double()(x.double())
    root = rec.calls[-1]
    thr = truestar.compare.threshold_for(torch.float32, _ACCEL.tf32, None)
    op, notes = opdrill.drill(copy.deepcopy(module).double(), copy.deepcopy(module).to(ACCEL_DEV),
                              root.key, root.args, root.kwargs, ACCEL_DEV, torch.float32, thr,
                              is_cpu=False)
    assert op is None, (op, notes)


# ----------------------------------------------------------------------------- cost


@needs_accel
def test_only_the_blamed_module_is_traced(monkeypatch):
    traced = []
    real = opdrill.trace_ops

    def spy(module, *a, **k):
        traced.append(type(module).__name__)
        return real(module, *a, **k)

    monkeypatch.setattr(opdrill, "trace_ops", spy)
    run(MLP(), torch.randn(8, 16))
    assert traced == []  # clean model: nothing drilled
    with OpFault(LINEAR_OP, ACCEL_TYPE, amount=0.5):
        run(MLP(), torch.randn(8, 16))
    # One module only, never the model: the reference, the backend, and a CPU run in the model
    # dtype that checks the module takes the same code path in both precisions.
    assert traced == ["Linear", "Linear", "Linear"]


# ----------------------------------------------------------------------------- op repro


@needs_accel
def test_op_repro_reproduces_under_the_fault_and_passes_on_cpu(tmp_path, monkeypatch):
    torch.manual_seed(0)
    with OpFault(LINEAR_OP, ACCEL_TYPE, amount=0.5):
        rep = run(MLP(), torch.randn(32, 16), repro_dir=str(tmp_path))
    path = result(rep, ACCEL).op_repro_path
    assert path and os.path.isfile(path), rep.to_text()
    text = open(path, encoding="utf-8").read()
    assert f"torch.ops.{LINEAR_OP}(" in text and len(text.splitlines()) <= 25
    assert "truestar" not in text.split('"""')[-1]  # the script body needs only PyTorch

    def exit_code(argv, fault):
        monkeypatch.setattr(sys, "argv", argv)
        try:
            if fault:
                with OpFault(LINEAR_OP, ACCEL_TYPE, amount=0.5):
                    runpy.run_path(path, run_name="__main__")
            else:
                runpy.run_path(path, run_name="__main__")
        except SystemExit as e:
            return e.code
        return 0

    assert exit_code([path], fault=True) == 1  # the corrupted kernel reproduces
    assert exit_code([path], fault=False) == 0  # the stock kernel is correct
    assert exit_code([path, "cpu"], fault=True) == 0  # CPU control passes


class _DtypeBranchGelu(nn.Module):
    """A "fast GELU" that only float32 inputs take, with a wrong constant. Same op sequence as the
    exact path, so op alignment pairs the two aten.mul calls."""

    def forward(self, x):
        c = 0.7978845608028654
        if x.dtype == torch.float32:
            return 0.05 * x * (1 + torch.tanh(c * (x + 0.044715 * x ** 3)))
        return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * x ** 3)))


class _DtypeBranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.act = _DtypeBranchGelu()
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        return self.out(self.act(self.fc(x)))


def test_dtype_branch_is_not_blamed_on_a_stock_op_on_cpu(tmp_path):
    """The float64 reference and the float32 run take different code paths: no stock op is
    blamed and no op repro is written, on CPU just as on an accelerator."""
    torch.manual_seed(0)
    rep = truestar.check(_DtypeBranchNet(), torch.randn(4, 8), backends=["cpu"],
                         repro_dir=str(tmp_path))
    cpu = result(rep, "cpu")
    assert cpu.first_divergent.name == "act", rep.to_text()
    assert cpu.first_divergent_op is None, rep.to_text()
    assert cpu.op_repro_path is None
    assert not list(tmp_path.rglob("op_repro.py"))
    assert any("own code differs by dtype" in n for n in cpu.notes), rep.to_text()


def test_dtype_independent_module_has_no_code_difference():
    torch.manual_seed(0)
    block = Block().eval()
    x = torch.randn(2, 6, 16)
    ref, _ = opdrill.trace_ops(copy.deepcopy(block).double(), (x.double(),), {})
    low, _ = opdrill.trace_ops(block, (x,), {})
    assert opdrill.code_difference(ref, low, torch.float32) is None
