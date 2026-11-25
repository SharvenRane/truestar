"""Per layer precision planner (truestar.precplan, docs/PRECISION.md, pain row P33).

CPU tests cover the parts that need no GPU (budget, scoring, unit discovery, tie groups). The
planner itself times real kernels, so its tests run on CUDA and skip without it.
"""
import copy
import json
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from truestar.precplan import Budget, PlanError, apply, plan, score
from truestar.precplan import units as U

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


# ----------------------------------------------------------------------------- no GPU needed


def test_cli_precision_names_match_the_table():
    from truestar.precplan.cli import PRECISION_NAMES

    assert PRECISION_NAMES == U.PRECISIONS


def test_budget_counts_and_shares():
    b = Budget.parse("3")
    assert b.allowed(1000, 1000) == 3 and b.allowed(500, 1000) == 1
    s = Budget.parse("0.5%")
    assert s.allowed(2000, 1000) == 10 and s.allowed(100, 1000) == 0
    assert Budget.parse(0).allowed(10 ** 6, 10) == 0
    for bad in ("-1", "1.5", "150%"):
        with pytest.raises(ValueError):
            Budget.parse(bad)


def test_score_counts_flips_and_nonfinite_rows():
    ref = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
    out = torch.tensor([[1.0, 0.0], [1.0, 0.0], [float("nan"), 1.0]])
    s = score([out], [ref])
    assert (s.flips, s.decisions, s.nonfinite_rows) == (2, 3, 1)
    assert s.rel_err == float("inf")
    assert score([ref.float()], [ref]).flips == 0


class _Attn(nn.Module):
    """Attention written in the parent's forward: q @ k needs one dtype."""

    def __init__(self, d=16):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        a = torch.softmax(self.q(x) @ self.k(x).transpose(-1, -2), -1)
        return a @ self.v(x)


class _TieNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(16, 16)
        self.attn = _Attn()
        self.mha = nn.MultiheadAttention(16, 2, batch_first=True)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        h = self.attn(self.inp(x))
        h = h + self.mha(h, h, h, need_weights=False)[0]
        return self.out(h)


def test_units_merge_modules_whose_children_are_never_called():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    w = U.wrap(_TieNet(), "cpu", lambda m: m(x))
    assert "mha" in w.units and "mha.out_proj" not in w.units  # MHA reads out_proj.weight itself
    assert w.merged == ["mha"]
    assert list(w.units) == ["inp", "attn.q", "attn.k", "attn.v", "mha", "out"]  # execution order


def test_tie_groups_found_by_probing():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    w = U.wrap(_TieNet(), "cpu", lambda m: m(x))
    U.find_groups(w, lambda m: m(x))
    tied = [g for g in w.groups if len(g) > 1]
    assert tied == [["attn.q", "attn.k", "attn.v"]]
    # every probe pattern now runs: one dtype per group, alternating
    for a, b in ((torch.float32, torch.float16), (torch.bfloat16, torch.float16)):
        assign = {}
        for i, g in enumerate(w.groups):
            for n in g:
                assign[n] = U.PLAIN[a if i % 2 == 0 else b]
        w.set(assign)
        with torch.no_grad():
            w.model(x)


def test_tie_groups_see_the_mismatch_the_check_is_for():
    """Guard for the guard: without tying, the mixed probe raises."""
    x = torch.randn(2, 5, 16)
    w = U.wrap(_TieNet(), "cpu", lambda m: m(x))
    w.set({"attn.q": "fp32", "attn.k": "fp16"})
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            w.model(x)


def test_planner_refuses_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PlanError, match="TS669"):
        plan(nn.Linear(4, 4), torch.randn(4, 4))


# ----------------------------------------------------------------------------- on the GPU


def _example():
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    import precision_model as pm

    return pm


@pytest.fixture(scope="module")
def example_plan():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    pm = _example()
    m = pm.get_model()
    p = plan(m, pm.get_inputs(), pm.get_holdout(), max_flips="1%", reps=10, rounds=2)
    return pm, m, p


@cuda
def test_planted_sensitive_layer_stays_float32(example_plan):
    pm, m, p = example_plan
    assert p.met and "TS660" in p.codes
    assert p.assign["norm"] == "fp32"
    row = next(u for u in p.units if u.name == "norm")
    assert row.cands["fp16"].status == "range" and row.cands["bf16"].status == "range"
    assert "precision_loss" in row.cands["fp16"].reason and "precision_model.py" in row.cands["fp16"].reason
    assert "TS663" in p.codes
    # everything else leaves float32
    assert all(v != "fp32" for k, v in p.assign.items() if k != "norm")


@cuda
def test_plan_keeps_the_budget_where_naive_baselines_do_not(example_plan):
    pm, m, p = example_plan
    r = p.result
    assert r.cal.flips <= p.allowed_cal and r.hold.flips <= p.allowed_hold
    base = {b.name: b for b in p.baselines}
    assert base["all fp16"].cal.flips > p.allowed_cal          # the norm breaks in float16
    assert base["fp16 except norms"].cal.flips > p.allowed_cal  # a class rule does not see it
    assert r.latency_us < base["all fp32"].latency_us / 1.5    # and the plan is fast


@cuda
def test_plan_explains_every_unit(example_plan):
    pm, m, p = example_plan
    for u in p.units:
        assert u.why.startswith(u.chosen)
        assert u.cands[u.chosen].status == "ok"
    d = p.as_dict()
    json.dumps(d, allow_nan=False)
    assert d["plan"] == p.assign and d["counts"]


@cuda
def test_saved_plan_applies_to_its_model_only(example_plan, tmp_path):
    pm, m, p = example_plan
    path = str(tmp_path / "plan.json")
    p.save(path)
    run = apply(m, path)
    x = pm.get_holdout()[0][0]
    with torch.no_grad():
        ref = copy.deepcopy(m).double()(x.double())
    with torch.no_grad():
        y = run(x)
    s = score([y.reshape(-1, y.shape[-1])], [ref])
    assert s.flips == p.result.hold.flips
    other = copy.deepcopy(m)
    with torch.no_grad():
        other.fc1.weight.mul_(1.001)
    with pytest.raises(PlanError, match="TS666"):
        apply(other, path)


@cuda
def test_held_out_failure_tightens_the_plan():
    """Calibration inputs are ordinary; some held out rows carry a reading of 7e4 that the first
    layer doubles, past float16's 65504 wherever it is stored in float16. A plan verified on
    calibration alone would ship float16; the held out gate must catch it."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 10)).eval()
    with torch.no_grad():
        m[0].weight[:, 0] = 2.0
    cal = torch.randn(2048, 1024)
    hold = torch.randn(2048, 1024)
    hold[:64, 0] = 7e4
    p = plan(m, cal, hold, max_flips=0, precisions=("fp32", "tf32", "fp16"), use_range=False,
             baselines=False, format_loss=False, reps=5, rounds=2)
    first = next(s for s in p.search if s["step"].startswith("held out gate"))
    assert first["flips"] > 0                       # the calibration plan failed held out
    assert p.met and p.result.hold.flips == 0
    assert "TS661" in p.codes and p.tightened >= 1
    assert "fp16" not in p.assign.values()


@cuda
def test_kernel_bug_is_rejected_not_counted_as_format_loss(monkeypatch):
    real = U.make_quant

    def buggy(base, p, name):
        q = real(base, p, name)
        if p == "int4w" and name == "1":
            q.inject = "group_scale_shift"
        return q
    monkeypatch.setattr(U, "make_quant", buggy)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(256, 512), nn.Linear(512, 512), nn.Linear(512, 16)).eval()
    x = torch.randn(256, 256)
    p = plan(m, x, torch.randn(256, 256), max_flips="100%", baselines=False, reps=5, rounds=1)
    c = p.units[1].cands["int4w"]
    assert c.status == "kernel_bug" and "TS460" == c.kernel["code"]
    assert c.format_loss is not None  # verdict (b) still measured, from float64 only
    assert "TS664" in p.codes and p.assign["1"] != "int4w"
    ok = p.units[0].cands["int4w"]
    assert ok.kernel["ok"]


@cuda
def test_unreachable_budget_is_reported_and_plan_is_float32():
    """fp32 itself loses the variance at an offset of 3000 (examples/check_model.py), so no plan
    can keep a zero flip budget."""
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    import check_model as cm

    torch.manual_seed(1)
    x = 3000 + torch.randn(256, 32)
    p = plan(cm.get_model(), x, 3000 + torch.randn(256, 32), max_flips=0, baselines=False,
             format_loss=False, reps=5, rounds=1)
    assert not p.met and "TS662" in p.codes
    assert set(p.assign.values()) == {"fp32"}


@cuda
def test_cli_agent_json():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, "-m", "truestar", "compile",
                        os.path.join(ROOT, "examples", "precision_model.py"), "--max-flips", "1%",
                        "--no-baselines", "--format", "agent"], capture_output=True, text=True,
                       env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    doc = json.loads(r.stdout)
    assert doc["command"] == "compile" and doc["status"] == "ok"
    assert "TS660" in doc["codes"] and doc["plan"]["plan"]["norm"] == "fp32"


def test_cli_usage_errors_are_agent_json():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, "-m", "truestar", "compile",
                        os.path.join(ROOT, "examples", "precision_model.py"), "--precisions",
                        "fp32,fp64", "--format", "agent"], capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode == 2
    doc = json.loads(r.stdout)
    assert doc["codes"] == ["TS021"] and doc["command"] == "compile"
