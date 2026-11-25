"""truestar fix: remedies are found, applied, verified by a rerun check, and costed; broken models
where nothing helps are reported as unresolved."""

import contextlib
import json
import math
import os
import subprocess
import sys
import warnings

import pytest
import torch
from torch import nn

import truestar
import truestar.fix as tf
import truestar.fix.engine
import truestar.fix.output
import truestar.fix.precision
from truestar import codes
from truestar.agent import dumps
from truestar.fix.table import TABLE, Decomposition, candidates
from truestar.testing import DeviceFault, OpFault

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")


# ----------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        torch.manual_seed(0)
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        s = self.q(x) @ self.k(x).transpose(-1, -2) / 4.0
        return torch.softmax(s, dim=-1) @ self.v(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.norm = nn.LayerNorm(16)
        self.attn = Attention()
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        return self.out(self.attn(self.norm(x)).mean(dim=1))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 8, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, groups=2)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.conv2(torch.relu(self.conv(x))))
        return self.head(h.mean((2, 3)))


class Faulty(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(16, 8)
        self.bad = DeviceFault("cuda", "scale", 0.5)
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        return self.out(self.bad(self.fc(x)))


def x16(n=8):
    torch.manual_seed(1)
    return torch.randn(n, 16)


def only(rep):
    assert len(rep.backends) == 1, rep.to_text()
    return rep.backends[0]


@contextlib.contextmanager
def fused_linear_on_cuda(scale: float = 1.0):
    """Make CUDA run aten.linear as its own kernel, like MPS does (CUDA normally decomposes it
    into t plus addmm before any dispatch mode sees it). scale != 1 makes that kernel wrong, a
    kernel level fault that a dispatch mode such as OpFault cannot express."""
    lib = torch.library.Library("aten", "IMPL")

    def kernel(input, weight, bias=None):
        x2 = input.reshape(-1, input.shape[-1])
        y = torch.mm(x2, weight.t())
        if bias is not None:
            y = y + bias
        return (y * scale).reshape(*input.shape[:-1], weight.shape[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lib.impl("linear", kernel, "CUDA")
    try:
        yield
    finally:
        lib._destroy()


# ----------------------------------------------------------------------------- the table (CPU)


def _call_args(op: str):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    if op == "aten.addmm.default":
        return (r(5), r(4, 6), r(6, 5)), {"beta": 0.5, "alpha": 2.0}
    if op == "aten.linear.default":
        return (r(2, 3, 6), r(5, 6), r(5)), {}
    if op == "aten.mm.default":
        return (r(4, 6), r(6, 5)), {}
    if op == "aten.bmm.default":
        return (r(3, 4, 6), r(3, 6, 5)), {}
    if op in ("aten._softmax.default", "aten._log_softmax.default"):
        return (r(3, 7) * 5, -1, False), {}
    if op == "aten.native_layer_norm.default":
        return (r(2, 3, 8), [8], r(8), r(8), 1e-5), {}
    if op == "aten.gelu.default":
        return (r(4, 5),), {"approximate": "tanh"}
    if op == "aten.convolution.default":
        return (r(2, 4, 9, 7), r(6, 2, 3, 3), r(6), [2, 1], [1, 2], [1, 1], False, [0, 0], 2), {}
    raise AssertionError(f"add test inputs for {op}")


def test_every_decomposition_matches_the_op_in_float64():
    checked = 0
    for op, entries in TABLE.items():
        args, kwargs = _call_args(op)
        want = tf.engine.resolve_op(op)(*args, **kwargs)
        want = [want] if isinstance(want, torch.Tensor) else list(want)
        for e in entries:
            if e.unsupported(args, kwargs):
                continue
            got = e(*args, **kwargs)
            got = [got] if isinstance(got, torch.Tensor) else list(got)
            assert len(got) == len(want), (op, e.name)
            for a, b in zip(got, want):
                assert a.shape == b.shape, (op, e.name, a.shape, b.shape)
                assert torch.allclose(a, b, rtol=1e-10, atol=1e-12), (op, e.name)
            checked += 1
    assert checked >= 15


def test_half_only_entries_and_guards():
    x = torch.randn(3, 4)
    e = [d for d in TABLE["aten._softmax.default"] if d.name == "explicit_fp32"][0]
    assert e.unsupported((x, -1, False), {})
    assert e.unsupported((x.half(), -1, False), {}) is None
    big = [d for d in TABLE["aten.mm.default"] if d.name == "mul_sum"][0]
    assert big.unsupported((torch.empty(1024, 1024), torch.empty(1024, 1024)), {})


def test_mps_linear_remedy_is_built_in_and_first():
    c = candidates("aten.linear.default", "mps")
    assert c[0].name == "matmul_add" and "macOS 26" in c[0].note
    lib = candidates("aten.linear.default", "mps", mechanism="library")
    assert "linear_nobias_add" not in [d.name for d in lib]  # would recurse as a kernel
    assert tf.engine.UPSTREAM_DRAFTS[("aten.linear.default", "mps")].endswith(
        "mps_linear_bias_DRAFT.md")
    assert os.path.isfile(os.path.join(ROOT, "docs", "upstream", "mps_linear_bias_DRAFT.md"))


def test_new_codes_are_documented():
    doc = open(os.path.join(ROOT, "docs", "AGENT_INTERFACE.md"), encoding="utf-8").read()
    fixdoc = open(os.path.join(ROOT, "docs", "FIX.md"), encoding="utf-8").read()
    for c in ("TS090", "TS091", "TS092", "TS093", "TS094", "TS095"):
        assert c in codes.CODES and c in doc and c in fixdoc


def test_clean_model_needs_nothing():
    rep = tf.fix(MLP(), x16(), backends=["cpu"], time_reps=0)
    assert rep.status == "clean" and rep.backends == [] and rep.remedies() == []
    assert tf.output.exit_code(rep.status) == 0
    assert rep.to_agent()["next_actions"] == []


def test_cli_lists_the_table():
    out = subprocess.run([sys.executable, "-m", "truestar", "fix", "--list"], capture_output=True,
                         text=True, env=dict(os.environ, PYTHONPATH=SRC), timeout=120)
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)["remedies"]
    assert {"op": "aten.linear.default", "name": "matmul_add"}.items() <= rows[3].items()


# ----------------------------------------------------------------------------- op remedies (CUDA)


@needs_cuda
def test_addmm_fault_is_remedied_verified_and_costed(tmp_path):
    with OpFault("aten.addmm.default", "cuda", amount=0.5) as fault:
        rep = tf.fix(MLP(), x16(), backends=["cuda"], repro_dir=str(tmp_path), time_reps=5)
    assert fault.hits > 0
    b = only(rep)
    assert b.before.status == "divergent" and b.before.op == "aten.addmm.default"
    assert b.outcome == "resolved" and b.after.status == "ok", rep.to_text()
    assert [r.decomposition for r in b.applied] == ["mm_add"]
    won = [a for a in b.attempts if a.won]
    assert len(won) == 1 and won[0].hits > 0 and won[0].op_ok
    assert won[0].op_rel_err < 1e-5 and won[0].module_status == "ok"
    assert b.after.output_rel_err < 1e-5 < b.before.output_rel_err
    assert b.cost is not None and b.cost.baseline_ms > 0 and b.cost.remedied_ms > 0
    assert [i.code for i in b.issues] == ["TS090"]
    assert os.path.isfile(b.upstream["op_repro"])
    doc = json.loads(dumps(rep.to_agent()))  # strict JSON
    assert doc["command"] == "fix" and doc["status"] == "resolved" and doc["exit_code"] == 0
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert kinds == ["apply_remedy", "report_upstream"]
    assert doc["backends"][0]["cost"]["overhead_pct"] is not None
    text = rep.to_text()
    assert "RESOLVED" in text and "mm_add" in text and "cost: forward" in text
    json.loads(rep.to_json())


@needs_cuda
def test_patched_and_apply_fix_the_real_model_and_remove_restores():
    m, x = MLP().cuda(), x16().cuda()
    ref = MLP().double()(x16().double())
    rel = lambda y: float((y.double().cpu() - ref).norm() / ref.norm())
    with OpFault("aten.addmm.default", "cuda", amount=0.5):
        rep = tf.fix(MLP(), x16(), backends=["cuda"], time_reps=0)
        with torch.no_grad():
            assert rel(m(x)) > 0.1
            with tf.patched(rep) as p:
                assert rel(m(x)) < 1e-5
            assert p.hits["aten.addmm.default"] == 2
            h = tf.apply(rep, model=m)
            assert rel(m(x)) < 1e-5 and rel(m(x)) < 1e-5
            assert h.hits["aten.addmm.default"] == 4
            h.remove()
            assert rel(m(x)) > 0.1
            assert not hasattr(m, "_truestar_remedies")
            # A CheckReport gives the unverified first choice, with a warning.
            with pytest.warns(UserWarning, match="without verifying"):
                with tf.patched(rep.before):
                    assert rel(m(x)) < 1e-5


@needs_cuda
def test_genuinely_broken_model_is_reported_unresolved(tmp_path):
    with contextlib.ExitStack() as st:
        for op in ("aten.addmm.default", "aten.mm.default", "aten.bmm.default", "aten.mul.Tensor"):
            st.enter_context(OpFault(op, "cuda", amount=0.5))
        rep = tf.fix(MLP(), x16(), backends=["cuda"], repro_dir=str(tmp_path), time_reps=5)
    b = only(rep)
    assert b.outcome == "unresolved" and b.applied == [] and rep.remedies() == []
    assert b.after.status == "divergent"
    assert len(b.attempts) == 3 and not any(a.won or a.op_ok for a in b.attempts)
    assert all(a.op_rel_err is not None and a.op_rel_err > a.op_threshold for a in b.attempts)
    assert b.cost is None
    assert [i.code for i in b.issues] == ["TS092"]
    assert os.path.isfile(b.upstream["op_repro"]) and b.upstream["op_repro"] in b.issues[0].message
    doc = json.loads(dumps(rep.to_agent()))
    assert doc["status"] == "unresolved" and doc["exit_code"] == 1
    assert [a["kind"] for a in doc["next_actions"]] == ["report_upstream"]
    assert "UNRESOLVED" in rep.to_text()


@needs_cuda
def test_module_level_fault_is_not_applicable():
    rep = tf.fix(Faulty(), x16(), backends=["cuda"], time_reps=0)
    b = only(rep)
    assert b.outcome == "not_applicable" and b.before.op is None
    assert [i.code for i in b.issues] == ["TS093"] and "module bad#0" in b.reason
    assert "inspect_source" in [a["kind"] for a in rep.to_agent()["next_actions"]]


@needs_cuda
def test_wrong_candidate_never_wins_and_the_next_one_is_tried():
    bogus = Decomposition("aten.addmm.default", "bogus", "wrong on purpose",
                          lambda input, mat1, mat2, *, beta=1, alpha=1: torch.mm(mat1, mat2))
    table = {"aten.addmm.default": (bogus,) + TABLE["aten.addmm.default"]}
    with OpFault("aten.addmm.default", "cuda", amount=0.5):
        rep = tf.fix(MLP(), x16(), backends=["cuda"], time_reps=0, table=table)
    b = only(rep)
    assert b.outcome == "resolved"
    assert [a.remedy.decomposition for a in b.attempts] == ["bogus", "mm_add"]
    assert not b.attempts[0].op_ok and b.attempts[1].won


@needs_cuda
def test_remedy_right_for_the_blamed_call_only_fails_the_model_check():
    """Correct for fc1's call (the one blamed, which the op filter replays) and wrong for fc2's:
    only the rerun model check can catch it."""

    def half_right(input, mat1, mat2, *, beta=1, alpha=1):
        y = torch.mm(mat1, mat2) + input
        return y if mat2.shape[1] == 32 else y * 2.0

    sneaky = Decomposition("aten.addmm.default", "half_right", "right for fc1 only", half_right)
    table = {"aten.addmm.default": (sneaky,) + TABLE["aten.addmm.default"]}
    with OpFault("aten.addmm.default", "cuda", amount=0.5):
        rep = tf.fix(MLP(), x16(), backends=["cuda"], time_reps=0, table=table)
    b = only(rep)
    first = b.attempts[0]
    assert first.remedy.decomposition == "half_right" and first.op_ok
    assert first.module_status == "ok" and first.model_status == "divergent" and not first.won
    assert b.outcome == "resolved" and b.applied[0].decomposition == "mm_add"


@needs_cuda
def test_two_bad_ops_take_two_rounds():
    with OpFault("aten.addmm.default", "cuda", 0.5), OpFault("aten._softmax.default", "cuda", 0.5):
        rep = tf.fix(Block(), torch.randn(2, 5, 16), backends=["cuda"], time_reps=3)
    b = only(rep)
    assert b.outcome == "resolved", rep.to_text()
    assert sorted(r.op for r in b.applied) == ["aten._softmax.default", "aten.addmm.default"]


@needs_cuda
def test_convolution_fault_is_remedied_by_im2col():
    torch.manual_seed(2)
    with OpFault("aten.convolution.default", "cuda", 0.5):
        rep = tf.fix(CNN(), torch.randn(2, 3, 12, 10), backends=["cuda"], time_reps=3)
    b = only(rep)
    assert b.outcome == "resolved", rep.to_text()
    assert {(r.op, r.decomposition) for r in b.applied} == {("aten.convolution.default", "im2col")}


@needs_cuda
def test_mps_code_path_fused_linear_faulted_by_opfault():
    """Same path as aten.linear.default on MPS: the backend runs a fused linear the reference
    never runs, the op is blamed, and the built in table entry for MPS clears it."""
    with fused_linear_on_cuda(), OpFault("aten.linear.default", "cuda", 0.5) as fault:
        rep = tf.fix(MLP(), x16(), backends=["cuda"], time_reps=3)
    assert fault.hits > 0
    b = only(rep)
    assert b.before.op == "aten.linear.default"
    assert not rep.before.results[0].first_divergent_op.matched  # fused path: not in reference
    assert b.outcome == "resolved", rep.to_text()
    assert b.applied[0].decomposition == candidates("aten.linear.default", "mps")[0].name


@needs_cuda
def test_library_mechanism_replaces_a_broken_kernel_and_restores_it():
    x = x16().cuda()
    with fused_linear_on_cuda(scale=1.5):
        rep = tf.fix(MLP(), x16(), backends=["cuda"], mechanism="library", time_reps=3)
        b = only(rep)
        assert b.before.op == "aten.linear.default"
        assert b.outcome == "resolved" and b.applied[0].decomposition == "matmul_add"
        lin = MLP().fc1.cuda()
        good = nn.functional.linear(x.cpu().double(), lin.weight.cpu().double(),
                                    lin.bias.cpu().double())
        with torch.no_grad():
            err = lambda: float((lin(x).double().cpu() - good).norm() / good.norm())
            assert err() > 0.1
            h = tf.apply(rep, mechanism="library")
            assert err() < 1e-5 and h.hits["aten.linear.default"] > 0
            h.remove()
            assert err() > 0.1  # the broken kernel is back underneath


@needs_cuda
def test_library_mechanism_cannot_fix_a_dispatch_mode_fault_and_says_so():
    with OpFault("aten.addmm.default", "cuda", 0.5):
        rep = tf.fix(MLP(), x16(), backends=["cuda"], mechanism="library", time_reps=0)
    assert only(rep).outcome == "unresolved"


@needs_cuda
def test_cli_fix_demo_agent_json(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "truestar", "fix", os.path.join(ROOT, "examples", "fix_demo.py"),
         "--backends", "cuda", "--format", "agent", "--reps", "3", "--repro-dir", str(tmp_path),
         "--json", str(tmp_path / "full.json")],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=SRC), timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    doc = json.loads(out.stdout)
    assert doc["status"] == "resolved" and doc["codes"] == ["TS090"]
    assert doc["backends"][0]["remedies"][0]["decomposition"] == "mm_add"
    full = json.loads((tmp_path / "full.json").read_text())
    assert full["backends"][0]["outcome"] == "resolved"


# ----------------------------------------------------------------------------- precision (CPU)


class Outlier(nn.Module):
    """`mid` gets an input where one channel is 400x larger than the rest: per tensor INT8 of that
    input wipes out every other channel, so only `mid` needs float."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.inp = nn.Linear(24, 48)
        self.mid = nn.Linear(48, 48)
        self.out = nn.Linear(48, 10)
        with torch.no_grad():
            self.inp.weight[0] *= 400.0
            self.inp.bias[0] = 0.0
            self.mid.weight[:, 0] /= 400.0

    def forward(self, x):
        return self.out(torch.relu(self.mid(torch.relu(self.inp(x)))))


def _data(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 24, generator=g)


def test_precision_plan_keeps_only_the_sensitive_layer():
    m, ev, cal = Outlier(), _data(400, 1), _data(128, 2)
    plan = tf.select_precision(m, ev, cal, target_flips=2)
    assert plan.flips_before > 10, plan.to_text()
    assert plan.met and plan.flips_after <= 2
    assert plan.keep == ["mid"] and plan.keep_leaves == ["mid"]
    assert plan.compute_fraction == pytest.approx(48 * 48 / (24 * 48 + 48 * 48 + 48 * 10))
    assert plan.issue.code == "TS094"
    # Independent re verification with a fresh simulator.
    sim = tf.Int8Simulator(m, ev, cal)
    assert sim.flips(sim.run(plan.keep_leaves)) == plan.flips_after
    assert sim.flips(sim.run(())) == plan.flips_before
    doc = json.loads(dumps(plan.to_agent()))
    assert doc["codes"] == ["TS094"] and doc["next_actions"][0]["keep"] == ["mid"]
    assert "MET" in plan.to_text()


def test_precision_plan_from_tap_error_data():
    m, ev, cal = Outlier(), _data(400, 1), _data(128, 2)
    taps = {"inp": 0.01, "mid": 0.30, "out": 0.31}  # accumulated, like a tap report
    plan = tf.select_precision(m, ev, cal, target_flips=2, layer_errors=taps)
    assert plan.ranking_source == "layer error data"
    assert plan.ranking[0][0] == "mid" and plan.keep == ["mid"] and plan.met
    assert tf.precision.own_increase(list(taps.items()))[2][1] == pytest.approx(0.01)


def test_precision_unmet_target_is_reported_honestly():
    m, ev, cal = Outlier(), _data(200, 1), _data(64, 2)
    plan = tf.select_precision(m, ev, cal, target_flips=-1)
    assert not plan.met and plan.issue.code == "TS095"
    assert json.loads(dumps(plan.to_agent()))["exit_code"] == 1


def test_cli_int8_plan_agent_json(tmp_path):
    f = tmp_path / "outlier_model.py"
    f.write_text(
        "import sys\n"
        f"sys.path.insert(0, {HERE!r})\n"
        "from test_fix import Outlier, _data\n"
        "def get_model():\n    return Outlier()\n"
        "def get_inputs():\n    return _data(400, 1)\n"
        "def get_calibration():\n    return _data(128, 2)\n")
    out = subprocess.run([sys.executable, "-m", "truestar", "fix", str(f), "--int8",
                          "--target-flips", "2", "--format", "agent"], capture_output=True,
                         text=True, env=dict(os.environ, PYTHONPATH=SRC), timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    doc = json.loads(out.stdout)
    assert doc["codes"] == ["TS094"] and doc["precision_plan"]["keep_leaves"] == ["mid"]
