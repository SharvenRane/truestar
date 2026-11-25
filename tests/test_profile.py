"""`truestar profile` (docs/PROFILE.md): per step GPU time mapped to IR ops and source lines,
every step run accounted for, totals that agree with whole call CUDA events."""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

import truestar  # noqa: E402
from truestar import Tensor, ir, ops  # noqa: E402
from truestar.compiler import CompiledExecutor, compile_program  # noqa: E402
from truestar.compiler import profile as P  # noqa: E402

try:
    import triton  # noqa: F401
    HAS_GPU = torch.cuda.is_available()
except Exception:
    HAS_GPU = False
pytestmark = pytest.mark.skipif(not HAS_GPU, reason="profile times GPU steps")

MODEL = '''
import torch, torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 128)
        self.b = nn.Linear(128, 32)

    def forward(self, x):
        h = torch.relu(self.a(x))      # LINE_A
        return torch.softmax(self.b(h), -1)   # LINE_B

def get_model():
    torch.manual_seed(0)
    return Net()

def get_inputs():
    return (torch.randn(16, 64),)
'''


def test_every_step_accounted_and_mapped_to_lines():
    ex = CompiledExecutor("cuda")

    @truestar.function(executor=ex)
    def f(x: Tensor["B", 64], w: Tensor[64, 64]) -> Tensor["B", 64]:
        h = ops.tanh(x @ w)
        return ops.softmax(h * 2.0, -1)

    x, w = torch.randn(32, 64, device="cuda"), torch.randn(64, 64, device="cuda")
    f(x, w)
    p = P.profile_program(ex.last, [x, w], iters=8, warmup=1)
    kernels = sum(1 for _, s in ex.last.plan.all_steps() if s.kind == "kernel")
    assert p["accounting"]["steps_run"] == p["accounting"]["steps_timed"] == p["accounting"]["steps_expected"] == 8 * kernels
    assert p["accounting"]["unaccounted"] == []
    lines = {(os.path.basename(d["file"]), d["line"]) for d in p["lines"]}
    assert any(name == os.path.basename(__file__) for name, _ in lines)
    assert all(r["lines"] for r in p["steps"])
    assert abs(sum(r["share"] for r in p["steps"]) - 1.0) < 1e-9
    assert any(r["fused"] for r in p["steps"])
    text = P.to_text(p)
    assert "events: " in text and "by source line" in text


def test_loop_steps_are_counted_per_iteration():
    ex = CompiledExecutor("cuda")

    @truestar.function(executor=ex)
    def rnn(x: Tensor["B", "T", 16], w: Tensor[16, 16]) -> Tensor["B", 16]:
        h = ops.zeros((x.shape[0], 16))
        for t in range(x.shape[1]):
            h = ops.tanh(x[:, t] @ w + h)
        return h

    x, w = torch.randn(4, 5, 16, device="cuda"), torch.randn(16, 16, device="cuda")
    rnn(x, w)
    p = P.profile_program(ex.last, [x, w], iters=4, warmup=0)
    body = [r for r in p["steps"] if "/body" in r["region"] and r["kind"] == "kernel"]
    assert body and all(r["calls"] == 5 for r in body)
    assert p["accounting"]["unaccounted"] == []


def test_step_times_sum_to_the_busy_gpu_time():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512)).eval()
    prog = ir.import_module(model, (torch.randn(64, 512),))
    cp = compile_program(prog, "cuda")
    p = P.profile_program(cp, [torch.randn(64, 512, device="cuda")], iters=20, warmup=2, crosscheck=True)
    t = p["totals"]
    assert abs(t["step_gpu_ms"] - t["gpu_busy_ms"]) <= 0.25 * t["gpu_busy_ms"] + 0.01
    c = p["crosscheck"]
    assert c["truestar_kernels_launched"] == 20 * sum(1 for _, s in cp.plan.all_steps() if s.kind == "kernel")
    assert c["truestar_kernels_seen"] + c["dropped"] >= c["truestar_kernels_launched"]


def test_a_lost_event_is_reported_not_dropped(monkeypatch):
    """Tests can fail: an end event that is never recorded must show up as TS510."""
    real = P.StepTimer.stop
    state = {"n": 0}

    def lossy(self, plan, step, env):
        before = len(self.records)
        real(self, plan, step, env)
        if len(self.records) > before and self.target is not None:
            state["n"] += 1
            if state["n"] == 3:        # this run's end event is never recorded
                pl, st, e0, _, h = self.records[-1]
                self.records[-1] = (pl, st, e0, torch.cuda.Event(enable_timing=True), h)

    monkeypatch.setattr(P.StepTimer, "stop", lossy)
    torch.manual_seed(0)
    prog = ir.import_module(nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)).eval(), (torch.randn(2, 8),))
    cp = compile_program(prog, "cuda")
    p = P.profile_program(cp, [torch.randn(2, 8, device="cuda")], iters=3, warmup=0)
    assert len(p["accounting"]["unaccounted"]) == 1
    doc = P.to_agent(p)
    assert doc["status"] == "error" and "TS510" in doc["codes"]


def test_cli_agent_json(tmp_path, capsys):
    from truestar.cli import main
    f = tmp_path / "net.py"
    f.write_text(MODEL)
    code = main(["profile", str(f), "--iters", "5", "--warmup", "1", "--format", "agent"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 0 and doc["command"] == "profile" and doc["status"] == "ok"
    assert doc["schema_version"] == 1 and doc["steps"] and doc["next_actions"]
    lines = {d["line"] for d in doc["lines"] if d["file"].endswith("net.py")}
    src = MODEL.splitlines()
    want = {i + 1 for i, ln in enumerate(src) if "LINE_A" in ln or "LINE_B" in ln}
    assert want <= lines
    code = main(["profile", str(f), "--iters", "3", "--warmup", "0"])
    assert code == 0 and "truestar profile" in capsys.readouterr().out
