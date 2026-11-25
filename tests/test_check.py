import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

import truestar
from truestar.check import _decisions
from truestar.compare import DIVERGENT, OK, SUSPECT, diff_tensors
from truestar.testing import DeviceFault

# The accelerator under test: the first non CPU backend here (cuda, rocm, mps or xpu).
# TRUESTAR_TEST_BACKEND overrides it, for example on CI runners.
_ACCEL = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
if os.environ.get("TRUESTAR_TEST_BACKEND"):
    _ACCEL = next(b for b in truestar.available_backends()
                  if b.name == os.environ["TRUESTAR_TEST_BACKEND"])
ACCEL = _ACCEL.name if _ACCEL else None  # backend name, e.g. "cuda", "rocm", "mps"
ACCEL_TYPE = torch.device(_ACCEL.device).type if _ACCEL else None  # tensor device type
needs_cuda = pytest.mark.skipif(ACCEL is None, reason="needs a GPU backend (cuda, rocm, mps, xpu)")


# ----------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self, fault=None):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fault = fault if fault is not None else nn.Identity()
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.fc2(self.fault(self.act(self.fc1(x))))


class TwoHeads(nn.Module):
    """Tuple output and a module called twice."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(8, 8)
        self.a = nn.Linear(8, 3)
        self.b = nn.Linear(8, 2)

    def forward(self, x):
        h = torch.tanh(self.shared(x))
        h = torch.tanh(self.shared(h))
        return self.a(h), {"aux": self.b(h)}


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 32)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True), num_layers=2,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(32, 7)

    def forward(self, ids):
        return self.head(self.enc(self.emb(ids)).mean(dim=1))


class FunctionalBug(nn.Module):
    """Wrong only on CUDA, and in plain torch code between modules (not inside a leaf)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        h = self.fc1(x)
        if h.device.type == ACCEL_TYPE:
            h = h * 1.5
        return self.fc2(h)


class Attention(nn.Module):
    """Parent module with children (projections) AND its own math (softmax attention)."""

    def __init__(self, dim=16, bug_on=None):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.bug_on = bug_on

    def forward(self, x):
        s = self.q(x) @ self.k(x).transpose(-1, -2) / 4.0
        if self.bug_on is not None and x.device.type == self.bug_on:
            s = s * 3.0  # wrong scale only on this backend, in the parent's own code
        return torch.softmax(s, dim=-1) @ self.v(x)


class Block(nn.Module):
    def __init__(self, bug_on=None):
        super().__init__()
        self.norm = nn.LayerNorm(16)
        self.attn = Attention(bug_on=bug_on)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        return self.out(self.attn(self.norm(x)).mean(dim=1))


class InplaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3)
        self.a1 = nn.SiLU(inplace=True)
        self.c2 = nn.Conv2d(8, 8, 3)
        self.a2 = nn.ReLU(inplace=True)
        self.a3 = nn.Hardswish(inplace=True)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        x = self.a3(self.a2(self.c2(self.a1(self.c1(x)))))
        return self.head(x.mean(dim=(2, 3)))


def resnet18():
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    return torchvision.models.resnet18(num_classes=10)


def run(model, x, **kw):
    kw.setdefault("backends", ["cpu", ACCEL] if ACCEL else ["cpu"])
    return truestar.check(model, x, **kw)


def result(report, name):
    return next(r for r in report.results if r.backend.name == name)


# ----------------------------------------------------------------------------- no false alarms


@pytest.mark.parametrize("make,inp", [
    (lambda: MLP(), lambda: torch.randn(32, 16)),
    (lambda: TwoHeads(), lambda: torch.randn(4, 8)),
    (lambda: TinyTransformer(), lambda: torch.randint(0, 50, (4, 12))),
    (lambda: Block(), lambda: torch.randn(4, 6, 16)),
    (lambda: InplaceNet(), lambda: torch.randn(4, 3, 16, 16)),
    (resnet18, lambda: torch.randn(8, 3, 64, 64)),
])
def test_clean_models_report_clean(make, inp):
    torch.manual_seed(1)
    rep = run(make(), inp())
    for r in rep.results:
        assert r.status == OK, rep.to_text()
        assert r.first_divergent is None
        assert r.flips == 0


@needs_cuda
def test_sub_threshold_noise_is_not_flagged():
    rep = run(MLP(fault=DeviceFault(ACCEL_TYPE, "noise", 1e-7)), torch.randn(32, 16))
    assert result(rep, ACCEL).status == OK, rep.to_text()


# ----------------------------------------------------------------------------- finds real faults


@needs_cuda
@pytest.mark.parametrize("kind,amount,expected", [
    ("scale", 0.5, DIVERGENT),
    ("scale", 0.02, SUSPECT),
    ("zero", 0.0, DIVERGENT),
    ("nan", 0.0, DIVERGENT),
    ("rows", 0.5, DIVERGENT),
])
def test_fault_is_blamed_on_the_exact_layer(kind, amount, expected):
    rep = run(MLP(fault=DeviceFault(ACCEL_TYPE, kind, amount, from_row=20)), torch.randn(32, 16))
    cuda, cpu = result(rep, ACCEL), result(rep, "cpu")
    assert cpu.status == OK, rep.to_text()
    assert cuda.status == expected or (expected == SUSPECT and cuda.status in (SUSPECT, DIVERGENT)), rep.to_text()
    assert cuda.first_divergent is not None and cuda.first_divergent.name == "fault", rep.to_text()
    # Downstream layers inherit the error but are not blamed for it.
    fc2 = next(l for l in cuda.layers if l.name == "fc2")
    assert fc2.status == OK, rep.to_text()


@needs_cuda
def test_fault_deep_in_resnet_is_blamed_exactly():
    m = resnet18()
    m.layer3[1].bn2 = nn.Sequential(m.layer3[1].bn2, DeviceFault(ACCEL_TYPE, "scale", 0.5))
    rep = run(m, torch.randn(4, 3, 64, 64))
    cuda = result(rep, ACCEL)
    assert cuda.status == DIVERGENT
    assert cuda.first_divergent.name == "layer3.1.bn2.1", rep.to_text()
    assert [l.name for l in cuda.layers if l.status != OK] == ["layer3.1.bn2.1"], rep.to_text()


@needs_cuda
def test_functional_code_bug_is_blamed_on_the_owning_module():
    rep = run(FunctionalBug(), torch.randn(4, 8))
    cuda = result(rep, ACCEL)
    assert cuda.status == DIVERGENT, rep.to_text()
    assert cuda.first_divergent.name == "<model>" and not cuda.first_divergent.is_leaf, rep.to_text()
    assert [l.name for l in cuda.layers if l.status != OK] == ["<model>"], rep.to_text()


@needs_cuda
def test_bug_in_parent_own_code_is_blamed_on_that_parent():
    rep = run(Block(bug_on=ACCEL_TYPE), torch.randn(4, 6, 16))
    cuda = result(rep, ACCEL)
    assert cuda.status == DIVERGENT, rep.to_text()
    assert [l.name for l in cuda.layers if l.status != OK] == ["attn"], rep.to_text()
    assert result(rep, "cpu").status == OK


def test_pure_containers_are_not_replayed():
    rep = truestar.check(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4),
                         backends=["cpu"])
    root = next(l for l in rep.results[0].layers if l.name == "<model>")
    assert root.replayed is False


# ----------------------------------------------------------------------------- decisions


def test_decision_flips_and_ties():
    ref = torch.tensor([[5.0, 1.0], [1.0, 5.0], [2.0, 2.0 + 1e-9], [3.0, 0.0]], dtype=torch.float64)
    out = torch.tensor([[1.0, 5.0], [1.0, 5.0], [2.0 + 1e-9, 2.0], [3.0, 0.0]])
    n, flips, ties = _decisions(out, ref, threshold=1e-4)
    assert (n, flips, ties) == (4, 2, 1)


def test_decisions_skip_non_classifier_outputs():
    assert _decisions(torch.randn(4, 3, 2), torch.randn(4, 3, 2).double(), 1e-4) == (0, 0, 0)


# ----------------------------------------------------------------------------- repro


@needs_cuda
def test_repro_reproduces_on_backend_and_passes_on_cpu(tmp_path):
    rep = run(MLP(fault=DeviceFault(ACCEL_TYPE, "scale", 0.5)), torch.randn(32, 16),
              repro_dir=str(tmp_path))
    path = result(rep, ACCEL).repro_path
    assert path and os.path.isfile(path)
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    on_cuda = subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)
    on_cpu = subprocess.run([sys.executable, path, "cpu"], capture_output=True, text=True, env=env)
    assert on_cuda.returncode == 1 and "REPRODUCED" in on_cuda.stdout, on_cuda.stdout + on_cuda.stderr
    assert on_cpu.returncode == 0, on_cpu.stdout + on_cpu.stderr


# ----------------------------------------------------------------------------- safety


def test_missing_backend_raises_instead_of_skipping():
    present = {b.name for b in truestar.available_backends()}
    missing = next(n for n in ("mps", "xpu", "cuda") if n not in present)
    with pytest.raises(RuntimeError, match="not available"):
        truestar.check(MLP(), torch.randn(2, 16), backends=[missing])


def test_unavailable_backends_are_listed_as_skipped():
    rep = truestar.check(MLP(), torch.randn(2, 16))
    names = {b.name for b in truestar.available_backends()}
    for n in ("mps", "xpu"):
        if n not in names:
            assert n in [s[0] for s in rep.skipped]


def test_model_is_not_modified():
    m = MLP().train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    run(m, torch.randn(4, 16))
    assert m.training
    assert all(p.device.type == "cpu" for p in m.parameters())
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])


def test_diff_detects_nan_and_shape():
    a = torch.tensor([1.0, float("nan")])
    assert diff_tensors(a, torch.tensor([1.0, 2.0])).status(1e-4) == DIVERGENT
    assert diff_tensors(torch.zeros(2), torch.zeros(3)).status(1e-4) == DIVERGENT
    same_nan = diff_tensors(a, a.clone())
    assert same_nan.status(1e-4) == OK


def test_cli_runs_and_writes_json(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(
        "import torch\n"
        "def get_model():\n    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())\n"
        "def get_inputs():\n    return torch.randn(5, 4)\n"
    )
    from truestar.cli import main

    out = tmp_path / "r.json"
    code = main(["check", str(f), "--backends", "cpu", "--json", str(out), "--no-repro"])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["status"] == "ok" and data["results"][0]["backend"]["name"] == "cpu"


# ----------------------------------------------------------------------------- half precision


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_models_report_clean(dtype):
    torch.manual_seed(2)
    rep = run(resnet18().to(dtype), torch.randn(4, 3, 64, 64))
    for r in rep.results:
        assert r.status == OK, rep.to_text()


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_fault_is_blamed(dtype):
    m = MLP(fault=DeviceFault(ACCEL_TYPE, "scale", 0.5)).to(dtype)
    rep = run(m, torch.randn(32, 16))
    acc = result(rep, ACCEL)
    assert acc.status == DIVERGENT, rep.to_text()
    assert acc.first_divergent.name == "fault", rep.to_text()
    assert result(rep, "cpu").status == OK, rep.to_text()


def test_diff_fast_and_slow_paths_agree():
    torch.manual_seed(3)
    a, b = torch.randn(1000), torch.randn(1000)
    fast = diff_tensors(a, b)
    b2 = b.clone(); b2[0] = float("inf"); a2 = a.clone(); a2[0] = float("inf")
    slow = diff_tensors(a2, b2)  # same inf in both: slow path, finite part nearly the same
    assert slow.nonfinite_mismatch is False
    assert abs(fast.rel_err - slow.rel_err) / fast.rel_err < 0.01
    assert diff_tensors(torch.zeros(3), torch.zeros(3)).rel_err == 0.0
    assert diff_tensors(torch.ones(3), torch.zeros(3)).rel_err == float("inf")


def test_empty_cuda_visible_devices_does_not_crash():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    out = subprocess.run([sys.executable, "-c",
                          "import truestar; print([b.name for b in truestar.available_backends()])"],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "cpu" in out.stdout and "cuda" not in out.stdout


class _SlightlyOff(nn.Module):
    """Off by a relative 5e-4 in float32 only: over the 1e-4 floor, under the divergent bar."""

    def forward(self, x):
        return x * (1 + 5e-4) if x.dtype == torch.float32 else x


class _SuspectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.off = _SlightlyOff()

    def forward(self, x):
        return self.off(self.fc(x))


def test_suspect_module_is_not_called_divergent_in_the_text():
    torch.manual_seed(0)
    rep = truestar.check(_SuspectNet(), torch.randn(4, 8), backends=["cpu"])
    r = rep.results[0]
    assert r.first_divergent.name == "off" and r.first_divergent.status == SUSPECT, rep.to_text()
    text = rep.to_text()
    assert "TS011 first suspect module: off#0" in text, text
    assert "first divergent module" not in text, text


@pytest.mark.skipif(ACCEL not in ("cuda", "rocm"), reason="TF32 exists on CUDA and ROCm only")
def test_tf32_threshold_only_for_modules_with_matmul_or_conv(monkeypatch):
    """With TF32 on, a module with no matmul or convolution keeps the float32 threshold, so an
    error of 1e-3 in it is flagged instead of hiding under the TF32 threshold of 5e-3."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    torch.manual_seed(0)
    rep = truestar.check(MLP(fault=DeviceFault(ACCEL_TYPE, "scale", 1e-3)), torch.randn(32, 16),
                         backends=[ACCEL], drill_ops=False)
    r = result(rep, ACCEL)
    assert r.backend.tf32
    by_name = {l.name: l for l in r.layers}
    assert by_name["fc1"].threshold >= 5e-3  # a matmul: TF32 applies
    assert by_name["fault"].threshold < 5e-3, rep.to_text()
    assert by_name["act"].threshold < 5e-3, rep.to_text()
    assert r.first_divergent is not None and r.first_divergent.name == "fault", rep.to_text()
