"""Batch invariance (P14): `truestar invariance` and the invariant reference ops."""

import io
import json
import math
import os
import contextlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

import truestar
from truestar import invariance as inv
from truestar import invariant_ops as iops
from truestar.cli import main

_ACCEL = next((b for b in truestar.available_backends() if b.name != "cpu"), None)
ACCEL = _ACCEL.name if _ACCEL else None
ACCEL_DEV = _ACCEL.device if _ACCEL else None
needs_accel = pytest.mark.skipif(ACCEL is None, reason="needs a GPU backend")
DEVICES = ["cpu"] + ([ACCEL_DEV] if ACCEL else [])
BACKENDS = ["cpu"] + ([ACCEL] if ACCEL else [])


# ----------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self, d=64, h=256, o=10):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(d, h), nn.GELU(), nn.Linear(h, o)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attn(nn.Module):
    def __init__(self, d, h, invariant=False):
        super().__init__()
        L = iops.Linear if invariant else nn.Linear
        self.qkv, self.proj, self.h, self.invariant = L(d, 3 * d), L(d, d), h, invariant

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        sdpa = iops.scaled_dot_product_attention if self.invariant else F.scaled_dot_product_attention
        y = sdpa(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, S, D))


class Block(nn.Module):
    def __init__(self, d, h, invariant=False):
        super().__init__()
        L = iops.Linear if invariant else nn.Linear
        LN = iops.LayerNorm if invariant else nn.LayerNorm
        G = iops.GELU if invariant else nn.GELU
        self.ln1, self.attn, self.ln2 = LN(d), Attn(d, h, invariant), LN(d)
        self.mlp = nn.Sequential(L(d, 4 * d), G(approximate="tanh"), L(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """GPT style decoder built from torch.nn, random weights, greedy decoding in the checker."""

    def __init__(self, vocab=97, d=64, h=4, n=2, ctx=64, invariant=False):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h, invariant) for _ in range(n))
        self.ln = (iops.LayerNorm if invariant else nn.LayerNorm)(d)
        self.head = (iops.Linear if invariant else nn.Linear)(d, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def encoder():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True)
    return nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)


class Scale(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, d))

    def forward(self, x):
        return x * self.w


class BatchMix(nn.Module):
    """Test helper fault: mixes a statistic of the whole batch into every row (a real bug)."""

    def __init__(self, amount=0.1):
        super().__init__()
        self.amount = amount

    def forward(self, x):
        return x + self.amount * x.mean(0, keepdim=True)


class Drift(nn.Module):
    """Test helper fault: different bits on every call (run to run nondeterminism)."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * (1.0 + 1e-6 * (self.calls % 7))


class InvMLP(nn.Module):
    """Built only from batch invariant ops, with an optional injected fault in the middle."""

    def __init__(self, mid=None, d=32, h=64, o=8):
        super().__init__()
        self.fc1, self.scale, self.fc2 = iops.Linear(d, h), Scale(h), iops.Linear(h, o)
        self.mid = mid if mid is not None else nn.Identity()

    def forward(self, x):
        return self.fc2(self.mid(torch.relu(self.scale(self.fc1(x)))))


class BatchOpFault(TorchDispatchMode):
    """Test helper fault: one ATen op whose output depends on the batch size of its input.

    The output is scaled by (1 + amount * (rows - 1)), so a row alone is untouched and the same row
    inside any larger batch changes, by a float level amount or by a large one."""

    def __init__(self, op, amount=1e-6):
        super().__init__()
        self.op, self.amount, self.hits = op, amount, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self.op in (str(func), str(func.overloadpacket)) and isinstance(out, torch.Tensor) \
                and out.is_floating_point() and out.ndim >= 1:
            self.hits += 1
            return out * (1.0 + self.amount * (out.shape[0] - 1))
        return out


def _gpt_inputs():
    torch.manual_seed(1)
    return torch.randint(0, 97, (3, 12))


def _probe_linear_variant(device, d_in, d_out, rows_alone=1, lead=()):
    """Direct op probe, independent of the checker: does F.linear give row 0 other bits inside a
    batch of 2, 7 or 64 on this device?"""
    g = torch.Generator().manual_seed(5)
    w = torch.randn(d_out, d_in, generator=g).to(device)
    b = torch.randn(d_out, generator=g).to(device)
    x1 = torch.randn(1, *lead, d_in, generator=g).to(device)
    ref = F.linear(x1, w, b)
    for n in (2, 7, 64):
        for p in (0, n // 2, n - 1):
            xb = torch.randn(n, *lead, d_in, generator=g).to(device)
            xb[p] = x1[0]
            if not torch.equal(F.linear(xb, w, b)[p:p + 1], ref):
                return True
    return False


def _ops(res):
    return {o["op"] for o in res.non_invariant_ops()}


# ----------------------------------------------------------------------------- row location


def test_regions_cover_batch_major_sequence_major_and_padding():
    c = inv.Context("batch7@3", "batch", 7, 3)
    assert inv._regions((1, 12), (7, 12), c) == [(slice(3, 4), slice(None))]
    # B * S rows of a matmul: batch major or sequence major, both offered
    assert inv._regions((12, 64), (84, 64), c) == [(slice(36, 48), slice(None)),
                                                   (slice(3, None, 7), slice(None))]
    assert inv._regions((12, 64), (80, 64), c) == []  # does not line up: not located
    s = inv.Context("seqpad+5", "seqpad", 1, 0, 5)
    assert inv._regions((1, 12, 8), (1, 17, 8), s) == [(slice(None), slice(0, 12), slice(None))]
    # the strided layout is picked when it holds the row
    row = torch.arange(12.0).view(12, 1)
    seq_major = torch.full((84, 1), -1.0)
    seq_major[3::7] = row
    assert torch.equal(inv._row_part(seq_major, row, c), row)


def test_bits_equal_sees_signed_zero_and_nan():
    assert inv._bits_equal(torch.tensor([float("nan")]), torch.tensor([float("nan")]))
    assert not inv._bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not inv._bits_equal(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))


def test_contexts_place_the_row_and_keep_shared_inputs():
    x = torch.randn(4, 6)
    mask = torch.ones(6)
    inp = inv._parse_inputs((x, mask), None, 0)
    for c in inv.default_contexts():
        (cx, cm), _ = inv.context_inputs(inp, 2, c, seed=0)
        assert cm is mask
        if c.kind in ("batch", "pad"):
            assert cx.shape == (c.size, 6) and torch.equal(cx[c.position], x[2])
        if c.kind == "pad":
            assert int((cx != 0).any(1).sum()) <= 1
    tokens = torch.randint(3, 50, (4, 9))
    t = inv._parse_inputs(tokens, None, 0)
    assert t.lm
    (pad,), _ = inv.context_inputs(t, 1, inv.Context("seqpad+5", "seqpad", 1, 0, 5), 0)
    assert pad.shape == (1, 14) and torch.equal(pad[0, :9], tokens[1])
    (b,), _ = inv.context_inputs(t, 1, inv.Context("batch7@3", "batch", 7, 3), 0)
    assert b.dtype == tokens.dtype and int(b.min()) >= 3 and int(b.max()) <= 49


# ----------------------------------------------------------------------------- acceptance models


LINEAR_OPS = {"aten.addmm.default", "aten.linear.default"}
MPS_SDPA = "aten._scaled_dot_product_attention_math_for_mps.default"


@pytest.mark.parametrize("backend", BACKENDS)
def test_mlp_names_linear_when_linear_is_batch_variant(backend):
    torch.manual_seed(0)
    dev = "cpu" if backend == "cpu" else ACCEL_DEV
    rep = inv.invariance(MLP(), torch.randn(4, 64), backends=[backend], try_flags=False)
    res = rep.results[0]
    variant = _probe_linear_variant(dev, 64, 256) or _probe_linear_variant(dev, 256, 10)
    if not variant:
        pytest.skip(f"F.linear is batch invariant for these shapes on {backend}")
    assert res.status == inv.VARIANT and "TS050" in res.codes
    assert res.first_module.key in ("fc1#0", "fc2#0") and res.first_module.own
    # MPS runs nn.Linear as one aten.linear op; CPU and CUDA decompose it into addmm
    assert res.first_op.op in LINEAR_OPS
    assert _ops(res) <= LINEAR_OPS


@pytest.mark.parametrize("backend", BACKENDS)
def test_transformer_encoder_names_only_real_culprits(backend):
    dev = "cpu" if backend == "cpu" else ACCEL_DEV
    torch.manual_seed(1)
    rep = inv.invariance(encoder(), torch.randn(3, 12, 64), backends=[backend], rows=1,
                         try_flags=False)
    res = rep.results[0]
    variant = _probe_linear_variant(dev, 64, 128, lead=(12,))
    if res.status == inv.INVARIANT:
        assert not variant, "the encoder's linears are batch variant here but nothing was found"
        return
    assert res.status == inv.VARIANT
    # never a data movement op, never an attention kernel that is invariant on its own
    assert _ops(res) and _ops(res) <= LINEAR_OPS | {"aten.mm.default", "aten.bmm.default",
                                                    "aten.baddbmm.default", MPS_SDPA}, _ops(res)
    if variant:
        assert _ops(res) & LINEAR_OPS


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpt_decoder_report_and_tokens(backend):
    torch.manual_seed(0)
    rep = inv.invariance(TinyGPT(), _gpt_inputs(), backends=[backend], rows=1, decode_steps=3,
                         try_flags=False)
    res = rep.results[0]
    names = [c.name for c in rep.contexts]
    assert "seqpad+5" in names and rep.decode_steps == 3
    for r in res.rows:
        if r.context.kind != "seqpad":
            assert len(r.tokens_alone) == 3 and len(r.tokens_context) == 3
    if res.status != inv.INVARIANT:
        assert res.status == inv.VARIANT and res.ops
        allowed = {"aten.addmm.default", "aten.mm.default", "aten.bmm.default",
                   "aten._scaled_dot_product_flash_attention_for_cpu.default",
                   "aten._scaled_dot_product_efficient_attention.default",
                   "aten._scaled_dot_product_flash_attention.default",
                   "aten._scaled_dot_product_cudnn_attention.default", "aten.gelu.default",
                   "aten.linear.default", MPS_SDPA}
        assert _ops(res) <= allowed, _ops(res)
        doc = inv.to_agent(rep)
        kinds = [a["kind"] for a in doc["next_actions"]]
        assert "use_batch_invariant_op" in kinds


@pytest.mark.parametrize("backend", BACKENDS)
def test_model_built_from_invariant_ops_is_clean(backend):
    torch.manual_seed(0)
    rep = inv.invariance(TinyGPT(invariant=True), _gpt_inputs(), backends=[backend], rows=1,
                         decode_steps=2)
    res = rep.results[0]
    assert res.status == inv.INVARIANT, res.failing[:3]
    assert len(res.rows) == len(rep.contexts) and not res.codes
    doc = inv.to_agent(rep)
    assert doc["status"] == "invariant" and doc["exit_code"] == 0 and doc["next_actions"] == []


# ----------------------------------------------------------------------------- injected faults


@pytest.mark.parametrize("backend", BACKENDS)
def test_injected_batch_dependent_op_is_named(backend):
    torch.manual_seed(0)
    model, x = InvMLP(), torch.randn(4, 32)
    clean = inv.invariance(model, x, backends=[backend], rows=1, try_flags=False)
    assert clean.results[0].status == inv.INVARIANT
    with BatchOpFault("aten.mul.Tensor", amount=1e-6) as fault:
        rep = inv.invariance(model, x, backends=[backend], rows=1)
    res = rep.results[0]
    assert fault.hits > 0
    assert res.status == inv.VARIANT and "TS050" in res.codes
    assert res.first_module.key == "scale#0" and res.first_module.own
    assert res.first_op.op == "aten.mul.Tensor" and res.first_op.module == "scale#0"
    assert _ops(res) == {"aten.mul.Tensor"}
    assert res.first_module.source["module"]["file"] == os.path.abspath(__file__)
    # no torch flag can fix a kernel that reads the batch size: reported, verified by rerun
    assert res.flag_trials and not any(t.fixed for t in res.flag_trials)
    doc = inv.to_agent(rep)
    acts = {a["kind"]: a for a in doc["next_actions"]}
    assert acts["use_batch_invariant_op"]["op"] == "aten.mul.Tensor"
    assert acts["flags_do_not_fix"]["verified"] is True


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_batch_dependence_is_an_error_and_named(backend):
    torch.manual_seed(0)
    rep = inv.invariance(InvMLP(mid=BatchMix(1.0)), torch.randn(4, 32), backends=[backend],
                         rows=1, try_flags=False)
    res = rep.results[0]
    assert res.status == inv.DEPENDENT and "TS052" in res.codes
    assert res.first_module.key == "mid#0" and res.first_module.type_name == "BatchMix"
    assert res.first_op.op == "aten.mean.dim" and res.first_op.cls == inv.DIFFERENT
    assert res.argmax_flips > 0 and "TS053" in res.codes
    act = next(a for a in inv.to_agent(rep)["next_actions"] if a["kind"] == "fix_batch_dependence")
    assert act["file"] == os.path.abspath(__file__) and act["line"] > 0


def test_batch_dependence_flips_greedy_tokens():
    torch.manual_seed(0)
    gpt = TinyGPT(invariant=True)
    gpt.blocks[0].mlp.insert(1, BatchMix(5.0))
    rep = inv.invariance(gpt, _gpt_inputs(), backends=["cpu"], rows=1, decode_steps=3,
                         try_flags=False, sizes=(2, 7))
    res = rep.results[0]
    assert res.status == inv.DEPENDENT
    assert res.token_flips > 0 and "TS053" in res.codes
    assert res.first_module.key == "blocks.0.mlp.1#0"
    flipped = [r for r in res.rows if r.token_flip_step is not None]
    assert all(r.context.kind in ("batch", "pad") for r in flipped)


def test_run_to_run_nondeterminism_is_its_own_code():
    torch.manual_seed(0)
    rep = inv.invariance(InvMLP(mid=Drift()), torch.randn(4, 32), backends=["cpu"], rows=1,
                         try_flags=False)
    res = rep.results[0]
    assert "TS051" in res.codes
    assert any(m.key == "mid#0" and m.own and m.context == "repeat" for m in res.modules)
    # every op inside is deterministic: the drift lives in the module's Python state, so no op
    # is blamed and the report says so instead of guessing
    assert not res.ops and "TS054" in res.codes


class ThreadSensitive(nn.Module):
    """Test helper fault: batch dependent only while torch uses more than one CPU thread, like a
    kernel whose split across threads depends on the batch size."""

    def forward(self, x):
        if x.shape[0] > 1 and torch.get_num_threads() > 1:
            return x * (1.0 + 1e-6)
        return x * 1.0


@pytest.mark.skipif(torch.get_num_threads() < 2, reason="needs more than one CPU thread")
def test_a_flag_is_reported_as_a_fix_only_when_rerun_shows_it():
    torch.manual_seed(0)
    threads = torch.get_num_threads()
    rep = inv.invariance(InvMLP(mid=ThreadSensitive()), torch.randn(4, 32), backends=["cpu"],
                         rows=1, sizes=(2, 7))
    res = rep.results[0]
    assert res.status == inv.VARIANT and res.first_module.key == "mid#0"
    trials = {t.name: t for t in res.flag_trials}
    assert set(trials) == {"deterministic", "one_thread", "mkldnn_off", "all"}
    assert trials["one_thread"].fixed and trials["all"].fixed
    assert not trials["deterministic"].fixed and trials["deterministic"].remaining > 0
    assert torch.get_num_threads() == threads  # flags are restored after the trials
    acts = [a for a in inv.to_agent(rep)["next_actions"] if a["kind"] == "set_flags"]
    assert [a["settings"] for a in acts] == [["torch.set_num_threads(1)"]]
    assert acts[0]["verified"] is True


def test_broken_row_substitution_blames_the_wrong_module(monkeypatch):
    """The drilldown must isolate the row. Without the substitution every module after the fault
    sees other inputs and is blamed too, so the report is no longer the one op at fault."""
    torch.manual_seed(0)
    model, x = InvMLP(), torch.randn(4, 32)
    monkeypatch.setattr(inv, "_substitute", lambda ctx_obj, alone_obj, c: _clone_all(ctx_obj))
    with BatchOpFault("aten.mul.Tensor", amount=1e-6):
        rep = inv.invariance(model, x, backends=["cpu"], rows=1, try_flags=False)
    res = rep.results[0]
    assert not (res.first_module and res.first_module.key == "scale#0"
                and res.first_op and res.first_op.op == "aten.mul.Tensor" and len(res.ops) == 1)


def _clone_all(obj):
    from truestar.capture import map_tensors

    return map_tensors(obj, torch.clone)


# ----------------------------------------------------------------------------- agent JSON, CLI

MODEL_FILE = """
import torch
from torch import nn
from truestar import invariant_ops as iops


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, 16))

    def forward(self, x):
        if x.shape[0] > 1:
            return x * self.w * (1.0 + {amount})
        return x * self.w


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.scale, self.fc2 = iops.Linear(8, 16), Scale(), iops.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.scale(self.fc1(x))))


def get_model():
    torch.manual_seed(0)
    return Net()


def get_inputs():
    torch.manual_seed(1)
    return torch.randn(3, 8)
"""


def _cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


def test_cli_agent_json_for_faulty_and_clean(tmp_path):
    bad = tmp_path / "batchy.py"
    bad.write_text(MODEL_FILE.replace("{amount}", "1e-6"))
    code, out = _cli(["invariance", str(bad), "--backends", "cpu", "--sizes", "2,7",
                      "--rows", "1", "--format", "agent"])
    doc = json.loads(out)
    assert code == 1 and doc["exit_code"] == 1 and doc["command"] == "invariance"
    assert doc["schema_version"] == 1 and doc["status"] == "not_invariant"
    assert doc["model"]["file"] == str(bad.resolve()) or os.path.samefile(doc["model"]["file"], bad)
    b = doc["backends"][0]
    assert b["status"] == "batch_variant" and "TS050" in doc["codes"]
    assert b["first_module"]["key"] == "scale#0"
    assert b["first_op"]["op"] == "aten.mul.Tensor"
    assert b["summary"]["pairs"] == len(b["rows"]) and b["summary"]["bitwise"] >= 1
    assert all(set(r) >= {"context", "bitwise", "class", "max_abs", "row"} for r in b["rows"])
    src = b["first_module"]["source"]["user"]
    assert os.path.samefile(src["file"], bad) and src["line"] == 7
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert "use_batch_invariant_op" in kinds and "flags_do_not_fix" in kinds
    json.dumps(doc, allow_nan=False)

    good = tmp_path / "steady.py"
    good.write_text(MODEL_FILE.replace("{amount}", "0.0"))
    code, out = _cli(["invariance", str(good), "--backends", "cpu", "--sizes", "2,7",
                      "--rows", "1", "--format", "agent"])
    doc = json.loads(out)
    assert code == 0 and doc["status"] == "invariant" and doc["codes"] == []


def test_cli_text_and_errors(tmp_path):
    good = tmp_path / "steady2.py"
    good.write_text(MODEL_FILE.replace("{amount}", "1e-6"))
    js = tmp_path / "out.json"
    code, out = _cli(["invariance", str(good), "--backends", "cpu", "--sizes", "2",
                      "--rows", "1", "--no-flags", "--json", str(js)])
    assert code == 1 and "BATCH_VARIANT" in out and "aten.mul.Tensor" in out
    assert json.loads(js.read_text())["backends"][0]["flag_trials"] == []
    code, out = _cli(["invariance", str(tmp_path / "missing.py"), "--format", "agent"])
    doc = json.loads(out)
    assert code == 2 and doc["codes"] == ["TS020"] and doc["command"] == "invariance"
    code, out = _cli(["invariance", str(good), "--backends", "nope", "--format", "agent"])
    assert code == 2 and json.loads(out)["codes"] == ["TS001"]


def test_new_codes_are_registered():
    from truestar import codes

    expected = {"TS050": "batch_variant", "TS051": "nondeterministic", "TS052": "batch_dependent",
                "TS053": "invariance_flips", "TS054": "invariance_unattributed"}
    for c, s in expected.items():
        assert codes.slug(c) == s


# ----------------------------------------------------------------------------- invariant ops


def _row_invariant(fn, make, device, sizes=(2, 7, 64)):
    g = torch.Generator().manual_seed(3)
    x1 = make(1, g).to(device)
    ref = fn(x1)
    for n in sizes:
        for p in (0, n // 2, n - 1):
            xb = make(n, g).to(device)
            xb[p] = x1[0]
            got = fn(xb)[p:p + 1]
            if not inv._bits_equal(got.cpu(), ref.cpu()):
                return False
    return True


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_invariant_ops_are_correct_and_row_invariant(device, dtype):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=dtype)
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    w, b = r(96, 80), r(96)
    lw, lb = r(80), r(80)
    cases = {
        "linear tile": (lambda x: iops.linear(x, w, b), lambda x: F.linear(x, w, b), (12, 80)),
        "linear tree": (lambda x: iops.linear(x, w, b, mode="tree"), lambda x: F.linear(x, w, b), (12, 80)),
        "softmax": (lambda x: iops.softmax(x, -1), lambda x: torch.softmax(x.float(), -1).to(dtype), (12, 80)),
        "layer_norm": (lambda x: iops.layer_norm(x, (80,), lw, lb), lambda x: F.layer_norm(x, (80,), lw, lb), (12, 80)),
        "rms_norm": (lambda x: iops.rms_norm(x, lw), lambda x: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * lw.float()).to(dtype), (12, 80)),
        "tree_sum": (lambda x: iops.tree_sum(x, -1), lambda x: x.float().sum(-1).to(dtype), (12, 80)),
        "mean": (lambda x: iops.mean(x, -1), lambda x: x.float().mean(-1).to(dtype), (12, 80)),
        "gelu": (lambda x: iops.gelu(x, "tanh"), lambda x: F.gelu(x, approximate="tanh"), (12, 80)),
        "silu": (iops.silu, F.silu, (12, 80)),
        "sdpa": (lambda q: iops.scaled_dot_product_attention(q, q, q, is_causal=True),
                 lambda q: F.scaled_dot_product_attention(q, q, q, is_causal=True), (4, 12, 16)),
        "sdpa tile": (lambda q: iops.scaled_dot_product_attention(q, q, q, is_causal=True, mode="tile"),
                      lambda q: F.scaled_dot_product_attention(q, q, q, is_causal=True), (4, 12, 16)),
    }
    for name, (fn, ref_fn, shape) in cases.items():
        make = lambda n, gen, shape=shape: torch.randn(n, *shape, generator=gen).to(dtype)
        x = make(5, g).to(device)
        got, want = fn(x).cpu().double(), ref_fn(x).cpu().double()   # MPS has no float64
        err = float((got - want).norm() / want.norm().clamp_min(1e-30))
        assert err < tol, f"{name}: rel err {err:.2e}"
        assert _row_invariant(fn, make, device), f"{name} is not row invariant on {device} {dtype}"


@pytest.mark.parametrize("device", DEVICES)
def test_tree_attention_is_invariant_to_sequence_padding(device):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, 12, 16, generator=g).to(device)
    pad = torch.randn(1, 4, 5, 16, generator=g).to(device)
    qp = torch.cat([q, pad], dim=2)
    a = iops.scaled_dot_product_attention(q, q, q, is_causal=True)
    b = iops.scaled_dot_product_attention(qp, qp, qp, is_causal=True)[:, :, :12]
    assert inv._bits_equal(a.cpu(), b.cpu())
    # the fixed chunk tree: reduction order depends only on the chunk, so padding is free
    x = torch.randn(3, 70, generator=g).to(device)
    xp = torch.cat([x, torch.zeros(3, 30, device=device)], dim=1)
    assert inv._bits_equal(iops.tree_sum(x).cpu(), iops.tree_sum(xp).cpu())


def test_tile_matmul_keeps_one_layout():
    # A permuted operand is a view for one batch size and a copy for another; the tile matmul
    # must not let that change the GEMM it runs.
    for device in DEVICES:
        qkv = torch.randn(2, 12, 3, 4, 16, device=device)
        q = qkv.permute(2, 0, 3, 1, 4)[0]
        a = iops.bmm(q[:1], q[:1].transpose(-2, -1), mode="tile")
        b = iops.bmm(q, q.transpose(-2, -1), mode="tile")[:1]
        assert inv._bits_equal(a.cpu(), b.cpu())


def test_measure_overhead_reports_every_op():
    rows = iops.measure_overhead("cpu", batch=2, seq=4, dim=32, reps=1)
    assert {r["op"] for r in rows} >= {"linear (tile)", "sdpa causal", "softmax", "layer_norm"}
    assert all(r["default_s"] > 0 and r["invariant_s"] > 0 for r in rows)


class _FloatNoiseLinear(nn.Module):
    """A Linear whose output moves by a float level amount with the batch size, as a split K
    GEMM kernel does: batch variant, but far below any real dependence."""

    def __init__(self, d, h):
        super().__init__()
        self.lin = iops.Linear(d, h)

    def forward(self, x):
        y = self.lin(x)
        return y * (1 + 1e-7 * (x.shape[0] - 1))


class _NoisyInvMLP(InvMLP):
    def __init__(self, mid):
        super().__init__(mid=mid)
        self.fc1 = _FloatNoiseLinear(32, 64)


@pytest.mark.parametrize("backend", BACKENDS)
def test_float_noise_linear_does_not_outrank_a_real_batch_dependence(backend):
    """The early Linear is batch variant at float level; the later BatchMix mixes rows for real.
    Ranked against each module's own noise floor, the real cause is named first."""
    torch.manual_seed(0)
    rep = inv.invariance(_NoisyInvMLP(BatchMix(1.0)), torch.randn(4, 32), backends=[backend],
                         rows=1, try_flags=False)
    res = rep.results[0]
    keys = {m.key for m in res.modules if m.own}
    assert "fc1#0" in keys or "fc1.lin#0" in keys, rep.to_text()
    assert res.first_module.key == "mid#0", rep.to_text()
    assert res.first_module.noise_floor is not None
    assert "first batch dependent module: mid#0" in rep.to_text()
