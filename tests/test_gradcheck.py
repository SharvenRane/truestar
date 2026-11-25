"""truestar check --backward (P18): per module backward replay against float64."""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import truestar
from truestar import codes
from truestar.cli import main
from truestar.compare import DIVERGENT, OK, SUSPECT
from truestar.gradcheck import _replay_emulated, _tf32, grad_check
from truestar.gradfault import BackwardOpFault, in_backward

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_ACCEL = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
if os.environ.get("TRUESTAR_TEST_BACKEND"):
    _ACCEL = next(b for b in truestar.available_backends()
                  if b.name == os.environ["TRUESTAR_TEST_BACKEND"])
ACCEL = _ACCEL.name if _ACCEL else None
ACCEL_TYPE = torch.device(_ACCEL.device).type if _ACCEL else None
needs_accel = pytest.mark.skipif(ACCEL is None, reason="needs a GPU backend")
BACKENDS = ["cpu", ACCEL] if ACCEL else ["cpu"]


# ----------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        s = self.q(x) @ self.k(x).transpose(-1, -2) / x.shape[-1] ** 0.5
        return torch.softmax(s, dim=-1) @ self.v(x)


class Block(nn.Module):
    """A small transformer block: norm, attention with its own softmax math, MLP, head."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.attn = Attention()
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(x)
        return self.head(x.mean(dim=1))


class LowPrecisionForwardBug(nn.Module):
    """Wrong in forward (and so in backward) whenever it does not run in float64."""

    def forward(self, x):
        return x * (1.0 if x.dtype == torch.float64 else 1.3)


class FwdBugNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.bug = LowPrecisionForwardBug()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        return self.fc2(self.bug(self.fc1(x)))


class SDPA(nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)


class AttnNet(nn.Module):
    def __init__(self, causal=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.attn = SDPA(causal)

    def forward(self, q, k, v):
        return self.attn(q * self.scale, k, v)


def encoder_layer():
    torch.manual_seed(0)
    return nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True)


def resnet18():
    tv = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    return tv.models.resnet18(num_classes=10)


def result(rep, name):
    return next(r for r in rep.results if r.backend.name == name)


def seeded(make):
    torch.manual_seed(0)
    return make()


# ----------------------------------------------------------------------------- no false alarms


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("make,inp", [
    (lambda: seeded(MLP), lambda: torch.randn(32, 16)),
    (lambda: seeded(Block), lambda: torch.randn(4, 12, 32)),
    (encoder_layer, lambda: torch.randn(4, 12, 32)),
    (resnet18, lambda: torch.randn(4, 3, 32, 32)),
], ids=["mlp", "block", "encoder_layer", "resnet18"])
def test_clean_models_report_clean(make, inp, dtype):
    torch.manual_seed(1)
    x = inp()
    rep = grad_check(make().to(dtype), x, backends=BACKENDS)
    assert rep.status == OK, rep.to_text()
    for r in rep.results:
        assert sum(l.replayed for l in r.layers) >= 3, rep.to_text()
        assert r.first_divergent is None and not r.codes, rep.to_text()


def test_clean_float16_resnet_reports_clean():
    rep = grad_check(resnet18().half(), torch.randn(2, 3, 32, 32), backends=BACKENDS)
    assert rep.status == OK, rep.to_text()


# The draw of test_clean_float16_resnet_reports_clean depends on the CPU capability: normal_ for
# float32 has an AVX2 Box-Muller (taken on AVX2 and AVX512) and a scalar one (DEFAULT), which differ
# in the last bit. The scalar draw holds a near tie: one layer4.1 ReLU input is -5.3e-4 in float64
# and +1.2e-4 in float16, so a correctly rounded float16 run of the whole model is 1.1e-1 off
# float64 at layer4.1.bn2.weight. The float16 kernels are bitwise identical on every capability on
# a fixed draw (scripts/probe_cpu_scalar_fp16.py). Seeding as the test above, then hashing the input.
_SCALAR_DRAW = "737795cbf0"
_SCALAR_DRAW_SCRIPT = textwrap.dedent("""
    import hashlib, json, sys
    import torch, torchvision
    from truestar.gradcheck import grad_check
    from truestar.gradfault import BackwardOpFault

    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = {"capability": torch.backends.cpu.get_cpu_capability(),
           "draw": hashlib.sha1(x.numpy().tobytes()).hexdigest()[:10]}

    def summary(rep):
        r = rep.results[0]
        fd = r.first_divergent
        return {"status": rep.status, "text": rep.to_text(), "notes": r.notes,
                "blamed": fd.key if fd else None,
                "op": r.first_divergent_op.op if r.first_divergent_op else None}

    out["clean"] = summary(grad_check(model.half(), x, backends=["cpu"]))
    with BackwardOpFault("aten.threshold_backward.default", "cpu", amount=0.3) as f:
        out["fault"] = summary(grad_check(model.half(), x, backends=["cpu"]))
    out["fault_hits"] = f.hits
    print(json.dumps(out))
""")


@pytest.fixture(scope="module")
def scalar_draw():
    pytest.importorskip("torchvision")
    env = dict(os.environ, ATEN_CPU_CAPABILITY="default", PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, "-c", _SCALAR_DRAW_SCRIPT], capture_output=True,
                       text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["capability"] == "DEFAULT", out["capability"]
    if out["draw"] != _SCALAR_DRAW:
        pytest.skip(f"the scalar normal_ draw on this platform is {out['draw']}, not the "
                    f"{_SCALAR_DRAW} draw that holds the layer4.1 near tie")
    return out


def test_float16_resnet_scalar_cpu_draw_reports_clean(scalar_draw):
    """The draw CI's ubuntu runners failed on (TS180 at <model>#0, layer4.1.bn2.weight 1.13e-01):
    no kernel is wrong, so it must be OK, and the gradient float16 cannot resolve there is checked
    against the correctly rounded run and said so."""
    c = scalar_draw["clean"]
    assert c["status"] == OK, c["text"]
    note = next((n for n in c["notes"] if "correctly rounded run is equally off" in n), None)
    assert note is not None and "<model>#0" in note, c["text"]


def test_float16_resnet_scalar_cpu_draw_still_blames_a_relu_backward_fault(scalar_draw):
    """Checking against the rounded run must not hide a real bug on the same near tie draw."""
    f = scalar_draw["fault"]
    assert scalar_draw["fault_hits"] > 0
    assert f["status"] == DIVERGENT, f["text"]
    assert f["blamed"].endswith("relu#1") and f["op"] == "aten.threshold_backward.default", f["text"]


def test_rounded_run_excuses_only_a_gradient_that_matches_it():
    """Unit check of the rule: off float64 by more than the precision can resolve, OK only when
    the backend lands on the correctly rounded run; 3% away from that run stays divergent."""
    from truestar.gradcheck import _compare, _Replay

    torch.manual_seed(0)
    ref = torch.randn(512, dtype=torch.float64)
    emu = ref.clone()
    emu[:8] = 0.0  # a ReLU mask flip in the rounded run: these channels lose their gradient
    emu_err = float((emu - ref).norm() / ref.norm())
    assert emu_err > 0.1  # over the cap, so _floor drops the emulated floor
    r = _Replay(["param weight"], [ref], [])
    for got, want in ((emu.half(), OK), ((emu * 1.03).half(), DIVERGENT)):
        out, st = _compare(_Replay(["param weight"], [got], []), r, [None], torch.float16,
                           False, False, [emu], [emu_err])
        assert st == want, (out[0].diff, out[0].threshold)
    out, st = _compare(_Replay(["param weight"], [emu.half()], []), r, [None], torch.float16,
                       False, False)
    assert st == DIVERGENT  # without the rounded run there is nothing to excuse it


def _portable_draw(seed, root_fault=0.0):
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts",
                        "probe_fp16_parent_ties.py")
    spec = importlib.util.spec_from_file_location("probe_fp16_parent_ties", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, x = mod.portable_resnet18(seed, root_fault)
    return model.half(), x


def test_float16_parent_near_tie_mask_flip_is_suspect_not_divergent():
    """Portable draw 8 (float64 rand weights, so every CPU gets the same bits): every leaf replay
    is clean, the whole model replay is far off float64 because ReLU masks flip at near ties.
    Owner's rule (2026-11-24): SUSPECT (TS188) with the layer and the flipped count, never TS180."""
    model, x = _portable_draw(8)
    rep = grad_check(model, x, backends=["cpu"], drill_ops=False)
    r = rep.results[0]
    assert r.status == SUSPECT and r.codes == ["TS188"], rep.to_text()
    assert all(l.status == OK for l in r.layers if l.is_leaf and l.replayed)
    fd = r.first_divergent
    assert fd.key == "<model>#0" and fd.tie is not None
    assert fd.tie.total > 0 and all(n.split(".")[-1].startswith("relu#") for n, _ in fd.tie.flips)
    assert fd.tie.max_tie <= fd.tie.band
    msg = next(i.message for i in r.issues if i.code == "TS188")
    assert "near tie mask flip" in msg and "layer" in msg and "<model>." not in msg
    assert fd.as_dict()["near_tie_mask_flip"]["total"] == fd.tie.total
    assert [a["kind"] for a in rep.to_agent()["next_actions"]] == ["accept_near_tie"]


def test_float16_parent_wrong_backward_on_a_tie_draw_stays_divergent():
    """The same draw with a 15% wrong backward in the model's own code (seen only by the whole
    model replay): the backend does not land on the mask forced rounded run, so TS180 stays."""
    model, x = _portable_draw(8, root_fault=0.15)
    rep = grad_check(model, x, backends=["cpu"], drill_ops=False)
    r = rep.results[0]
    assert r.status == DIVERGENT and "TS180" in r.codes, rep.to_text()
    assert r.first_divergent.key == "<model>#0" and r.first_divergent.tie is None
    assert "not explained by the flips" in r.first_divergent.note


def test_mask_forcing_reproduces_the_backend_branch():
    """gradties.ForceMasks gives a ReLU the mask it is told, keeping near tie values near zero."""
    from truestar.gradties import ForceMasks

    x = torch.tensor([-1e-4, 2e-4, -3.0, 4.0], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, False, False, True])
    with ForceMasks([mask]):
        y = torch.relu(x)
    assert torch.equal(y > 0, mask) and float(y.abs().max()) == 4.0
    y.sum().backward()
    assert torch.equal(x.grad, mask.double())


def test_model_is_not_modified():
    m = seeded(MLP)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    grad_check(m, torch.randn(8, 16), backends=["cpu"], steps=3)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())
    assert m.training


def test_reference_upstream_reaches_every_module_with_in_place_ops():
    """resnet18 has in place ReLUs after batch norm: every call gets an upstream gradient."""
    rep = grad_check(resnet18(), torch.randn(2, 3, 32, 32), backends=["cpu"], drill_ops=False)
    r = result(rep, "cpu")
    assert all(l.replayed for l in r.layers if l.is_leaf), [l.key for l in r.layers
                                                            if l.is_leaf and not l.replayed]


def test_model_without_parameters_is_checked():
    rep = grad_check(nn.Sequential(nn.Tanh(), nn.Softmax(dim=-1)), torch.randn(4, 6),
                     backends=["cpu"])
    assert rep.status == OK
    assert sum(l.replayed for l in result(rep, "cpu").layers) == 2  # the Sequential has no own ops


# ----------------------------------------------------------------------------- backward only faults


def _assert_linear_backward_blamed(rep, backend, dtype_name, op="aten.mm.default"):
    r = result(rep, backend)
    assert r.status == DIVERGENT, rep.to_text()
    assert "TS180" in r.codes, rep.to_text()
    fd = r.first_divergent
    # the faulted op is wrong in the backward of both linears; fc2 runs first in backward.
    assert fd.key == "fc2#0" and fd.forward_status == OK, rep.to_text()
    assert r.first_divergent_op is not None, rep.to_text()
    assert r.first_divergent_op.op == op, rep.to_text()
    if op == "aten.mm.default":
        assert r.op_node == "AddmmBackward0", rep.to_text()
        assert r.first_divergent_op.input_dtypes == [dtype_name, dtype_name]
    else:   # MPS: one aten.linear_backward per layer, checked through its CPU form
        assert r.op_node == "LinearBackward0", rep.to_text()
        assert r.first_divergent_op.input_dtypes == [dtype_name] * 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_only_mm_fault_is_blamed_while_forward_check_is_ok_cpu(dtype):
    m, x = seeded(MLP).to(dtype), torch.randn(32, 16)
    with BackwardOpFault("aten.mm.default", "cpu", amount=0.3) as fault:
        rep = grad_check(m, x, backends=["cpu"])
        fwd = truestar.check(m, x, backends=["cpu"])
    assert fault.hits > 0
    assert fwd.status == OK, fwd.to_text()
    _assert_linear_backward_blamed(rep, "cpu", str(dtype).replace("torch.", ""))


@needs_accel
def test_backward_only_mm_fault_is_blamed_on_accelerator_bf16():
    m, x = seeded(MLP).bfloat16(), torch.randn(32, 16)
    # MPS runs a linear layer's backward as one aten.linear_backward op; CUDA as mm
    op = "aten.linear_backward.default" if ACCEL_TYPE == "mps" else "aten.mm.default"
    with BackwardOpFault(op, ACCEL_TYPE, amount=0.3) as fault:
        rep = grad_check(m, x, backends=BACKENDS)
        fwd = truestar.check(m, x, backends=BACKENDS)
    assert fault.hits > 0
    assert fwd.status == OK, fwd.to_text()
    assert result(rep, "cpu").status == OK, rep.to_text()  # the fault is only on the accelerator
    _assert_linear_backward_blamed(rep, ACCEL, "bfloat16", op)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_backward_row_sum_leak_is_blamed_on_attention(backend, dtype):
    """The failure mode of arXiv 2609.34272: a softmax backward that leaks its row sum."""
    dev_type = torch.device(_ACCEL.device).type if backend == ACCEL else "cpu"
    m, x = seeded(Block).to(dtype), torch.randn(4, 16, 32)
    with BackwardOpFault("aten._softmax_backward_data.default", dev_type, amount=0.5,
                         kind="row_leak") as fault:
        rep = grad_check(m, x, backends=[backend])
        fwd = truestar.check(m, x, backends=[backend])
    assert fault.hits > 0
    assert fwd.status == OK, fwd.to_text()
    r = result(rep, backend)
    assert r.first_divergent.key == "attn#0", rep.to_text()
    assert not r.first_divergent.is_leaf  # the softmax is the parent's own code
    assert "TS180" in r.codes and r.first_divergent.forward_status == OK, rep.to_text()
    op = r.first_divergent_op
    assert op is not None and op.op == "aten._softmax_backward_data.default", rep.to_text()
    assert r.op_node == "SoftmaxBackward0"
    # Only the attention module is over threshold: its children (q, k, v) are clean.
    assert [l.key for l in r.layers if l.status != OK] == ["attn#0"], rep.to_text()
    # The gradient norm hardly changes, which is why norm monitoring misses this class of bug.
    assert abs(r.first_divergent.norm_ratio - 1.0) < 0.05, rep.to_text()


def test_forward_bug_is_reported_as_forward_divergent():
    rep = grad_check(seeded(FwdBugNet), torch.randn(8, 16), backends=["cpu"])
    r = result(rep, "cpu")
    assert r.first_divergent.key == "bug#0"
    assert r.first_divergent.forward_status == DIVERGENT
    assert "TS182" in r.codes and "TS180" not in r.codes, rep.to_text()


def test_fault_outside_backward_never_fires():
    m = seeded(MLP)
    with BackwardOpFault("aten.addmm.default", "cpu", amount=0.5) as fault:
        assert not in_backward()
        rep = truestar.check(m, torch.randn(8, 16), backends=["cpu"])
    assert fault.hits == 0 and rep.status == OK


# ----------------------------------------------------------------------------- growth over steps


def test_clean_training_loop_shows_no_drift():
    rep = grad_check(seeded(MLP).bfloat16(), torch.randn(32, 16), backends=BACKENDS, steps=12)
    assert rep.status == OK, rep.to_text()
    for r in rep.results:
        assert r.trend is not None and len(r.trend.total_ratio) == 12
        assert not r.trend.flagged, rep.to_text()
        assert "emulated" in r.trend.control_ratio


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slowly_growing_backward_error_is_flagged_by_the_trend(dtype):
    """An error too small to cross the threshold at the replay, growing a little every backward
    pass, is caught by the training loop (TS183) and not by the single step replay."""
    m, x = seeded(MLP).to(dtype), torch.randn(32, 16)
    with BackwardOpFault("aten.mm.default", "cpu", amount=0.0, ramp=0.002):
        rep = grad_check(m, x, backends=["cpu"], steps=20, drill_ops=False)
    r = result(rep, "cpu")
    assert "TS183" in r.codes, rep.to_text()
    flagged = {f["module"] for f in r.trend.flagged}
    assert "<total>" in flagged and "fc1" in flagged, rep.to_text()
    if dtype == torch.bfloat16:
        assert "TS180" not in r.codes and "TS181" not in r.codes, rep.to_text()


# ----------------------------------------------------------------------------- attention on the GPU


@needs_accel
@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_backward_bf16_realistic_shape(causal):
    """F.scaled_dot_product_attention backward in bf16 at (batch 2, 8 heads, 1024 tokens, 64)."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 8, 1024, 64) for _ in range(3))
    rep = grad_check(AttnNet(causal).bfloat16(), (q, k, v), backends=[ACCEL])
    r = result(rep, ACCEL)
    assert r.status == OK, rep.to_text()
    attn = next(l for l in r.layers if l.key == "attn#0")
    assert [t.label for t in attn.tensors] == ["input 0", "input 1", "input 2"]
    for t in attn.tensors:
        assert t.diff.rel_err < 2e-2 and t.diff.rel_err < t.threshold, rep.to_text()
        assert abs(t.norm_ratio - 1.0) < 1e-2 and t.cosine > 0.999


# ----------------------------------------------------------------------------- noise floor


def test_tf32_rounding():
    x = torch.tensor([1.0, 1.0 + 2 ** -10, 1.0 + 2 ** -12, 1.0 + 2 ** -11 + 2 ** -13, -3.5,
                      float("inf")], dtype=torch.float64)
    y = _tf32(x)
    assert y[0] == 1.0 and y[1] == 1.0 + 2 ** -10  # representable values stay
    assert y[2] == 1.0  # below half a TF32 ulp rounds down
    assert y[3] == 1.0 + 2 ** -10  # above half rounds up
    assert y[4] == -3.5 and torch.isinf(y[5])


def test_emulated_floor_matches_cpu_bf16_and_is_not_zero():
    import copy

    from truestar.capture import Call
    from truestar.gradcheck import _replay_backward, _replay_cpu64

    torch.manual_seed(0)
    lin = nn.Linear(64, 64).double()
    x = torch.randn(32, 64, dtype=torch.float64)
    c = Call("l#0", "l", "Linear", 0, 0, 0, None, True, args=(x,), kwargs={})
    up = [torch.randn(32, 64, dtype=torch.float64)]
    ref = _replay_cpu64(lin, c, up)
    emu_mod = copy.deepcopy(lin)
    with torch.no_grad():
        for p in emu_mod.parameters():
            p.copy_(p.bfloat16().double())
    emu = _replay_emulated(emu_mod, c, up, torch.bfloat16)
    cpu = _replay_backward(copy.deepcopy(lin).bfloat16(), c, up, "cpu", torch.bfloat16)
    rel = lambda a, b: float((a.double() - b).norm() / b.norm())  # noqa: E731
    for e, k, r in zip(emu.grads, cpu.grads, ref.grads):
        ee, ck = rel(e, r), rel(k, r)
        assert 1e-4 < ee < 2e-2, (ee, ck)
        assert ee / 4 < ck < ee * 4, (ee, ck)  # a correct bf16 kernel shows about the same error


# ----------------------------------------------------------------------------- user code, repro


CUSTOM = '''
import torch
from torch import nn


class Square(torch.autograd.Function):
    """x * x whose backward is wrong below float64 (a low precision kernel bug)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * x * (2.0 if x.dtype == torch.float64 else 2.4)


class SquareAct(nn.Module):
    def forward(self, x):
        return Square.apply(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = SquareAct()
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def get_model():
    torch.manual_seed(0)
    return Net()


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(8, 16)
'''

CLEAN = '''
import torch
def get_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2))
def get_inputs():
    torch.manual_seed(1)
    return torch.randn(5, 4)
'''


def _run_cli(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


def test_cli_agent_json_names_custom_backward_and_repro_reproduces(tmp_path, capsys):
    f = tmp_path / "customgrad.py"
    f.write_text(textwrap.dedent(CUSTOM), encoding="utf-8")
    out_dir = tmp_path / "repro"
    code, out = _run_cli(["check", str(f), "--backward", "--backends", "cpu", "--format", "agent",
                          "--repro-dir", str(out_dir)], capsys)
    doc = json.loads(out)  # strict: one JSON document, nothing else on stdout
    assert code == 1 and doc["exit_code"] == 1 and doc["status"] == DIVERGENT
    assert doc["mode"] == "backward" and doc["command"] == "check"
    assert "TS180" in doc["codes"]
    assert all(i["code"] in codes.CODES for i in doc["issues"])
    b = doc["backends"][0]
    bm = b["blamed_module"]
    assert bm["key"] == "act#0" and bm["type"] == "SquareAct" and bm["backward_only"]
    assert bm["gradient"] == "input 0" and bm["forward_status"] == OK
    assert bm["source"]["user_code"] and bm["source"]["file"].endswith("customgrad.py")
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert kinds[0] == "confirm_repro" and "inspect_source" in kinds and "mitigate_fp32" in kinds
    assert "control_repro" not in kinds  # the backend is the CPU itself
    # The repro needs only PyTorch and reproduces on its own.
    repro = b["repro"]["path"]
    env = dict(os.environ, PYTHONPATH="")
    p = subprocess.run([sys.executable, repro], capture_output=True, text=True, env=env,
                       timeout=120)
    assert p.returncode == 1, p.stdout + p.stderr
    assert "REPRODUCED" in p.stdout


def test_cli_text_and_json_clean_model(tmp_path, capsys):
    f = tmp_path / "clean.py"
    f.write_text(CLEAN, encoding="utf-8")
    js = tmp_path / "r.json"
    code, out = _run_cli(["check", str(f), "--backward", "--backends", "cpu", "--no-repro",
                          "--steps", "4", "--json", str(js)], capsys)
    assert code == 0, out
    assert "truestar check --backward" in out and "[OK] cpu" in out
    assert "grad norm ratio vs float64 over 4 SGD steps" in out
    full = json.loads(js.read_text(encoding="utf-8"))
    assert full["mode"] == "backward" and full["status"] == OK
    code, out = _run_cli(["check", str(f), "--backward", "--backends", "cpu", "--format",
                          "agent", "--steps", "4"], capsys)
    doc = json.loads(out)
    assert code == 0 and doc["status"] == OK and doc["next_actions"] == []
    assert len(doc["backends"][0]["trend"]["total_ratio"]) == 4


def test_agent_marks_injected_faults(capsys, tmp_path):
    rep = None
    with BackwardOpFault("aten.mm.default", "cpu", amount=0.3):
        rep = grad_check(seeded(MLP), torch.randn(8, 16), backends=["cpu"])
    doc = rep.to_agent()
    json.dumps(doc, allow_nan=False)
    b = doc["backends"][0]
    assert b["injected_faults"] == ["BackwardOpFault(aten.mm.default)"]
    assert b["blamed_op"]["op"] == "aten.mm.default" and b["blamed_op"]["node"] == "AddmmBackward0"
    assert "injected_fault" in [a["kind"] for a in doc["next_actions"]]


def test_grad_codes_are_registered_and_documented():
    doc = open(os.path.join(SRC, "..", "docs", "GRAD_CHECK.md"), encoding="utf-8").read()
    for c in ("TS180", "TS181", "TS182", "TS183", "TS184"):
        assert c in codes.CODES and c in doc
    assert truestar.grad_check is grad_check


def test_backward_agrees_with_forward_on_check_model_example():
    """examples/check_model.py loses its digits in float32 in the forward pass. The backward check
    must not call that module fwd OK, nor let the emulated float32 floor (which repeats the same
    cancellation) raise its gradient threshold past a 2.6e6 error."""
    import runpy

    ex = runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "examples", "check_model.py"))
    model, x = ex["get_model"](), ex["get_inputs"]()
    fwd = truestar.check(model, x, backends=["cpu"], drill_ops=False)
    assert fwd.results[0].first_divergent.name == "norm"
    rep = grad_check(model, x, backends=["cpu"], drill_ops=False)
    r = rep.results[0]
    assert r.status == DIVERGENT, rep.to_text()
    assert r.first_divergent is not None and r.first_divergent.name == "norm", rep.to_text()
    assert r.first_divergent.forward_status == DIVERGENT, rep.to_text()
    assert r.first_divergent.threshold < 1e-1, rep.to_text()
    assert "TS182" in r.codes
