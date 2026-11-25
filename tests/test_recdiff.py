"""P13: `truestar record` and `truestar diff` name the module and op whose own result changed
between environments, and classify the change (identical, noise, precision, divergent).

Two environments are simulated in one process: an injected op change (truestar.testing.OpFault)
and TF32 toggled on CUDA. The baseline is what people do today: torch.allclose on final logits.
"""

import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

import truestar.recdiff as R
from truestar.cli import main
from truestar.recdiff import diff as D
from truestar.recdiff import store as S
from truestar.testing import OpFault

HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (TF32 and GPU replay)")
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# ----------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class OwnCode(nn.Module):
    """A parent whose own forward code runs an op (the doubling) outside any submodule."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        return self.out(self.fc(x) * 2.0)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Small GPT style decoder from torch.nn parts: token and position embeddings, causal
    attention blocks, final norm and a tied size vocabulary head."""

    def __init__(self, vocab=64, d=64, layers=2, ctx=16):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        t = ids.shape[1]
        h = self.tok(ids) + self.pos(torch.arange(t, device=ids.device))
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=ids.device), 1)
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.ln(h))


def resnet18():
    import torchvision

    torch.manual_seed(0)
    return torchvision.models.resnet18(weights=None).eval(), torch.randn(2, 3, 64, 64)


def encoder():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, dropout=0.0)
    return nn.TransformerEncoder(layer, 2).eval(), torch.randn(2, 16, 64)


def gpt():
    torch.manual_seed(0)
    return GPT().eval(), torch.randint(0, 64, (2, 16))


def mlp():
    torch.manual_seed(0)
    return MLP().eval(), torch.randn(4, 16)


REAL = {"resnet18": resnet18, "encoder": encoder, "gpt": gpt}


@pytest.fixture
def tf32_off():
    """Start with TF32 off everywhere and restore the process settings afterwards."""
    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    _tf32(False)
    yield _tf32
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def _tf32(on: bool) -> None:
    torch.backends.cudnn.allow_tf32 = on
    torch.backends.cuda.matmul.allow_tf32 = on


def codes(rep):
    return rep.codes


# ----------------------------------------------------------------------------- file format


def test_identical_runs_give_identical_bytes_and_round_trip(tmp_path):
    for make in (mlp, resnet18, gpt):
        m, x = make()
        a = R.record(m, x, ops="all")
        b = R.record(m, x, ops="all")
        assert a.raw == b.raw
        p = a.save(str(tmp_path / "a.tsrec"))
        back = R.load(p)
        assert back.raw == a.raw
        c = back.calls[0]
        assert all(torch.equal(t, u) for t, u in zip(back.outputs(c), a.outputs(a.calls[0])))


def test_corrupt_truncated_and_foreign_files_are_refused(tmp_path):
    m, x = mlp()
    raw = bytearray(R.record(m, x).raw)
    cases = {
        "flip": raw[:200] + bytes([raw[200] ^ 1]) + raw[201:],
        "trunc": raw[:-50],
        "magic": b"NOTREC" + raw[6:],
        "version": raw[:5] + b"\x00v9" + raw[8:],
    }
    expect = {"flip": "checksum", "trunc": "checksum", "magic": "bad magic",
              "version": "unsupported recording format version"}
    for name, data in cases.items():
        p = tmp_path / f"{name}.tsrec"
        p.write_bytes(bytes(data))
        with pytest.raises(R.RecordError, match=expect[name]):
            R.load(str(p))


def test_repeated_tensor_arguments_stay_one_object_on_replay():
    """attn(y, y, y) must replay as one tensor: nn.MultiheadAttention takes another kernel
    path when query, key and value are different objects."""
    m, x = gpt()
    a = R.record(m, x)
    c = a.by_key["blocks.0.attn#0"]
    args, kwargs = a.inputs(c)
    assert args[0] is args[1] is args[2]
    assert "mask" not in kwargs or kwargs["attn_mask"] is not args[0]


def test_outputs_only_is_smaller_and_bounded(tmp_path):
    m, x = resnet18()
    full = R.record(m, x)
    small = R.record(m, x, outputs_only=True)
    # Module inputs are mostly earlier outputs (stored once), so the saving is modest here...
    assert len(small.raw) < len(full.raw)
    assert all('"T"' not in json.dumps(c["args"]) or c["name"] == "<model>" for c in small.calls)
    # ...and large with op traces, whose inputs include every weight.
    full_ops, small_ops = R.record(m, x, ops="all"), R.record(m, x, ops="all", outputs_only=True)
    assert len(small_ops.raw) < 0.5 * len(full_ops.raw)
    with pytest.raises(R.RecordTooLarge, match="--outputs-only"):
        R.record(m, x, max_bytes=100_000)


# ----------------------------------------------------------------------------- classification


def test_classify_bands():
    d = lambda r: D.Diff(r, r, False, False)
    assert D.classify(d(0.0), True, 1e-4, 5e-3, True) == D.IDENTICAL
    assert D.classify(d(5e-5), False, 1e-4, None, True) == D.NOISE
    assert D.classify(d(1e-3), False, 1e-4, 5e-3, True) == D.PRECISION_CHANGE
    assert D.classify(d(1e-3), False, 1e-4, None, True) == D.DIVERGED  # nothing explains it
    assert D.classify(d(0.5), False, 1e-4, 5e-3, True) == D.DIVERGED
    assert D.classify(D.Diff(0.0, 0.0, True, False), False, 1e-4, 5e-3, True) == D.DIVERGED


# ----------------------------------------------------------------------------- clean


@pytest.mark.parametrize("name", sorted(REAL))
def test_identical_environments_report_clean(name):
    m, x = REAL[name]()
    a = R.record(m, x, ops="all")
    b = R.record(m, x, ops="all")
    for rep in (R.diff_recordings(a, b), R.diff_replay(a, m)):
        assert rep.status == D.IDENTICAL, rep.to_text()
        assert rep.blamed is None and rep.blamed_op is None
        assert rep.exit_code(strict=True) == 0
        assert not {"TS060", "TS061", "TS062", "TS068"} & set(codes(rep))
        assert rep.baseline["allclose"]


def test_change_within_dtype_tolerance_is_noise_not_a_regression():
    m, x = mlp()
    a = R.record(m, x, ops="all")
    with OpFault("aten.addmm.default", "cpu", amount=1e-6):
        rep = R.diff_replay(a, m)
    assert rep.status == D.NOISE
    assert rep.exit_code(strict=True) == 0
    assert "TS060" not in codes(rep)


# ----------------------------------------------------------------------------- injected op change


@pytest.mark.parametrize("name,module", [("gpt", "blocks.0.mlp.0#0"), ("resnet18", "fc#0"),
                                         ("encoder", "layers.0.linear1#0")])
def test_injected_op_change_is_named_and_divergent(name, module):
    m, x = REAL[name]()
    a = R.record(m, x, ops="all")
    with OpFault("aten.addmm.default", "cpu", amount=0.5) as f:
        b = R.record(m, x, ops="all")
        replay = R.diff_replay(a, m)
    assert f.hits > 0
    for rep in (R.diff_recordings(a, b), replay):
        assert rep.status == D.DIVERGED, rep.to_text()
        assert rep.blamed.key == module and rep.blamed.cls == D.DIVERGED and rep.blamed.own
        assert rep.blamed_op.op == "aten.addmm.default"
        assert rep.blamed_op.cls == D.DIVERGED and rep.blamed_op.source == "recorded_trace"
        assert "TS060" in codes(rep) and rep.exit_code() == 1
        assert rep.baseline["names_module"] is False and rep.baseline["names_op"] is False
        if name == "encoder":
            # Hooks switch off TransformerEncoderLayer's fused path; the plain path never
            # calls aten.addmm, so the plain output (baseline) is unchanged and TS071 says why.
            assert "TS071" in codes(rep) and rep.baseline["allclose"] is True
        else:
            assert "TS071" not in codes(rep) and rep.baseline["allclose"] is False
    # Replay isolates every module, so every faulty Linear owns its change; offline only the
    # first is measured and the modules after it are reported as not isolated (TS068).
    if name == "gpt":
        assert "TS068" in codes(R.diff_recordings(a, b)) and "TS068" not in codes(replay)


def test_parent_own_code_is_blamed_not_its_children():
    torch.manual_seed(0)
    m, x = OwnCode().eval(), torch.randn(3, 8)
    a = R.record(m, x, ops="all")
    with OpFault("aten.mul.Tensor", "cpu", amount=0.5):
        rep = R.diff_replay(a, m)
    assert rep.blamed.key == "<model>#0" and rep.blamed.cls == D.DIVERGED
    assert rep.blamed_op.op == "aten.mul.Tensor"
    assert [m.key for m in rep.modules if m.own and m.cls != D.IDENTICAL] == ["<model>#0"]


def test_outputs_only_recording_still_names_module_and_op():
    m, x = gpt()
    a = R.record(m, x, outputs_only=True, ops="all")
    with OpFault("aten.addmm.default", "cpu", amount=0.5):
        rep = R.diff_replay(a, m)
    assert rep.mode == "replay_outputs_only"
    assert rep.blamed.key == "blocks.0.mlp.0#0" and rep.blamed.method == "hash"
    assert rep.blamed_op.op == "aten.addmm.default"


def test_weights_or_inputs_that_differ_are_not_comparable():
    m, x = mlp()
    a = R.record(m, x)
    torch.manual_seed(1)
    other = MLP().eval()
    rep = R.diff_replay(a, other)
    assert rep.status == "not_comparable" and "TS064" in codes(rep) and rep.exit_code() == 1
    rep = R.diff_recordings(a, R.record(m, x + 1))
    assert rep.status == "not_comparable" and "TS069" in codes(rep)


def test_precision_flags_do_not_explain_a_cpu_recording(tf32_off):
    m, x = mlp()
    a = R.record(m, x)
    tf32_off(True)
    with OpFault("aten.addmm.default", "cpu", amount=1e-3):
        rep = R.diff_replay(a, m)
    assert not rep.precision_explained  # TF32 does not touch CPU kernels
    assert rep.status == D.DIVERGED


# ----------------------------------------------------------------------------- CUDA


@needs_cuda
@pytest.mark.parametrize("name,module,op", [
    ("resnet18", "layer1.0.conv1#0", "aten.convolution.default"),
    ("encoder", "layers.0.self_attn#0", "aten._native_multi_head_attention.default"),
    ("gpt", "blocks.0.attn#0", "aten._native_multi_head_attention.default"),
])
def test_tf32_toggle_is_named_and_classified_as_precision(tf32_off, name, module, op):
    m, x = REAL[name]()
    a = R.record(m, x, device="cuda", ops="all")
    assert R.diff_replay(a, m).status == D.IDENTICAL  # same environment: bitwise clean
    tf32_off(True)
    b = R.record(m, x, device="cuda", ops="all")
    for rep in (R.diff_replay(a, m), R.diff_recordings(a, b)):
        assert rep.status == D.PRECISION_CHANGE, rep.to_text()
        assert rep.precision_explained
        assert rep.blamed.key == module and rep.blamed.cls == D.PRECISION_CHANGE
        assert rep.blamed_op.op == op and rep.blamed_op.cls == D.PRECISION_CHANGE
        assert "TS061" in codes(rep) and "TS060" not in codes(rep)
        assert any(k == "flags.cudnn.allow_tf32" for k, *_ in rep.env_changes)
        assert rep.exit_code() == 0 and rep.exit_code(strict=True) == 1
        # Baseline: allclose on the final logits calls the TF32 change a failure and says
        # nothing about where it comes from.
        assert rep.baseline["allclose"] is False


@needs_cuda
def test_injected_op_change_is_still_divergent_under_tf32(tf32_off):
    m, x = resnet18()
    a = R.record(m, x, device="cuda", ops="all")
    tf32_off(True)
    with OpFault("aten.addmm.default", "cuda", amount=0.5):
        rep = R.diff_replay(a, m)
    assert rep.status == D.DIVERGED
    assert rep.blamed.key == "fc#0" and rep.blamed_op.op == "aten.addmm.default"
    assert rep.counts()[D.PRECISION_CHANGE] > 0  # TF32 convs are still labelled, not hidden


@needs_cuda
def test_without_op_trace_the_op_is_named_against_float64(tf32_off):
    m, x = gpt()
    a = R.record(m, x, device="cuda")  # no --ops
    with OpFault("aten.addmm.default", "cuda", amount=0.5):
        rep = R.diff_replay(a, m)
    assert rep.blamed.key == "blocks.0.mlp.0#0"
    assert rep.blamed_op.op == "aten.addmm.default"
    assert rep.blamed_op.source == "float64_reference"
    kinds = [x["kind"] for x in rep.to_agent()["next_actions"]]
    assert "record_op_trace" in kinds


@needs_cuda
def test_cuda_recording_is_deterministic(tf32_off):
    m, x = resnet18()
    assert R.record(m, x, device="cuda").raw == R.record(m, x, device="cuda").raw


# ----------------------------------------------------------------------------- agent and repro


def test_agent_document_and_next_actions(tmp_path):
    m, x = mlp()
    a = R.record(m, x, ops="all")
    a.save(str(tmp_path / "a.tsrec"))
    with OpFault("aten.addmm.default", "cpu", amount=0.5):
        rep = R.diff_replay(R.load(str(tmp_path / "a.tsrec")), m, repro_dir=str(tmp_path / "r"))
    doc = json.loads(json.dumps(rep.to_agent(), allow_nan=False))
    assert doc["schema_version"] == 1 and doc["command"] == "diff"
    assert doc["status"] == "divergent" and doc["exit_code"] == 1
    assert doc["blamed_module"]["key"] == "fc1#0" and doc["blamed_module"]["class"] == "divergent"
    assert doc["blamed_op"]["op"] == "aten.addmm.default"
    kinds = [x["kind"] for x in doc["next_actions"]]
    assert kinds[0] == "confirm_op_repro" and "report_upstream" in kinds
    assert doc["op_repro"]["path"].endswith("op_repro.py")


def test_op_repro_reproduces_without_truestar(tmp_path):
    """Old environment had the bad op, new one is fine: the repro, run in a plain process,
    shows the op's result moved for the exact recorded inputs (exit code 1)."""
    m, x = mlp()
    with OpFault("aten.addmm.default", "cpu", amount=0.5):
        a = R.record(m, x, ops="all")
    rep = R.diff_replay(a, m, repro_dir=str(tmp_path))
    assert rep.blamed_op.op == "aten.addmm.default"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, rep.op_repro_path, "cpu"], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "relative change" in r.stdout


def test_precision_next_action_names_the_flag():
    rep = D.DiffReport("replay", "a", "b", "m", torch.float32, "cuda:0", {}, {},
                       [("flags.cudnn.allow_tf32", False, True, "precision")], True, [])
    ch = D.ModuleChange("c#0", "c", "Conv2d", True, 0, D.PRECISION_CHANGE, True, "replay",
                        D.Diff(1e-3, 1e-3, False, False), 1e-4, 5e-3)
    ch.own = True
    rep.modules, rep.blamed = [ch], ch
    acts = rep.to_agent()["next_actions"]
    assert acts[0]["kind"] == "restore_precision_flag"
    assert acts[0]["key"] == "flags.cudnn.allow_tf32" and acts[0]["recorded"] is False


# ----------------------------------------------------------------------------- command line


MODEL_FILE = '''
import torch
from torch import nn

def get_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))

def get_inputs():
    torch.manual_seed(1)
    return torch.randn(4, 16)
'''


def test_cli_record_diff_text_json_agent(tmp_path, capsys):
    model = tmp_path / "model.py"
    model.write_text(MODEL_FILE)
    a, b = str(tmp_path / "a.tsrec"), str(tmp_path / "b.tsrec")
    assert main(["record", str(model), "-o", a, "--backend", "cpu", "--ops", "all"]) == 0
    assert main(["record", str(model), "-o", b, "--backend", "cpu", "--ops", "all"]) == 0
    assert open(a, "rb").read() == open(b, "rb").read()
    capsys.readouterr()
    assert main(["diff", a, b]) == 0
    assert "IDENTICAL" in capsys.readouterr().out
    assert main(["diff", a, "--replay", "--format", "agent"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "identical" and doc["next_actions"] == []
    assert main(["diff", a, "--replay", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "identical"
    # usage errors
    assert main(["diff", a]) == 2
    assert main(["diff", a, b, "--replay"]) == 2
    (tmp_path / "bad.tsrec").write_bytes(b"junk" * 20)
    assert main(["diff", str(tmp_path / "bad.tsrec"), "--replay", "--format", "agent"]) == 2
    assert json.loads(capsys.readouterr().out)["codes"] == ["TS065"]
    assert main(["record", str(model), "-o", a, "--backend", "cpu", "--max-mb", "0.0001",
                 "--format", "agent"]) == 2
    assert json.loads(capsys.readouterr().out)["codes"] == ["TS070"]


def test_cli_replay_names_injected_op_in_a_new_process(tmp_path):
    model = tmp_path / "model.py"
    model.write_text(MODEL_FILE)
    a = str(tmp_path / "a.tsrec")
    assert main(["record", str(model), "-o", a, "--backend", "cpu", "--ops", "all"]) == 0
    code = ("import sys; from truestar.testing import OpFault; "
            "f = OpFault('aten.addmm.default', 'cpu', amount=0.5); f.__enter__(); "
            "from truestar.cli import main; "
            f"sys.exit(main(['diff', {a!r}, '--replay', '--format', 'agent', "
            f"'--repro-dir', {str(tmp_path / 'r')!r}]))")
    env = dict(os.environ, PYTHONPATH=SRC)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode == 1, r.stderr
    doc = json.loads(r.stdout)
    assert doc["blamed_module"]["key"] == "0#0"
    assert doc["blamed_op"]["op"] == "aten.addmm.default"
    assert doc["codes"][:1] == ["TS060"] or "TS060" in doc["codes"]
