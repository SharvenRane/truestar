"""Training spike watch and `truestar replay` (docs/SPIKES.md, ledger row P35).

Three constructed spikes with known causes (tests/spike_models.py) must each get their verdict, a
clean run must raise no shadow flag and no spike, and a replay that cannot reproduce the log must
refuse to give a verdict.
"""

import json
import os
import subprocess
import sys

import pytest

import spike_models as SM
from truestar import codes
from truestar.spikes import ReplayError, replay

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for case in ("clean", "numerics", "data", "optimizer"):
        d = str(tmp_path_factory.mktemp(case))
        _, recs = SM.run(d, case)
        out[case] = (d, recs)
    return out


def test_clean_run_has_no_shadow_flag_and_no_spike(runs):
    d, recs = runs["clean"]
    shadows = [r for r in recs if "shadow" in r]
    assert len(shadows) == 6  # steps 0, 10, ..., 50
    assert not any(r.get("code") for r in recs) and not any(r["spike"] for r in recs)
    assert max(r["shadow"]["loss_error"] for r in shadows) < 1e-2
    assert min(r["shadow"]["grad_cosine"] for r in shadows) > 0.999
    for step in (30, 45, 59):
        rep = replay(d, step)
        assert rep.verdict == "none" and rep.code == "TS696", rep.to_text()


def test_numerics_spike_vanishes_in_float64_and_names_the_norm(runs):
    d, recs = runs["numerics"]
    r45 = recs[45]
    assert r45["spike"] and r45.get("code") == "TS691"  # the watch shadowed the spike on its own
    assert r45["shadow"]["loss_error"] > 1.0
    rep = replay(d, 45)
    assert rep.verdict == "numerics" and rep.code == "TS692", rep.to_text()
    assert rep.first_module == "norm"
    assert rep.lowp[-1] > 5 * rep.baseline and rep.f64[-1] < 2 * rep.baseline
    assert rep.first_gap_step == 45 and max(rep.step_errors[:-1]) < 1e-2


def test_data_spike_survives_float64_and_names_the_corrupted_rows(runs):
    d, _ = runs["data"]
    rep = replay(d, 45)
    assert rep.verdict == "data" and rep.code == "TS693", rep.to_text()
    assert {r for r, _, _ in rep.rows} == {3, 9}
    assert sum(s for _, _, s in rep.rows) > 0.95


def test_learning_rate_spike_is_the_optimizer_not_the_batch(runs):
    d, _ = runs["optimizer"]
    rep = replay(d, 45)
    assert rep.verdict == "optimizer" and rep.code == "TS694", rep.to_text()
    assert "learning rate jumped at step 44" in rep.message
    assert rep.batch_on_checkpoint < 2 * rep.baseline


def test_same_batch_in_bfloat16_does_not_spike(tmp_path):
    """bfloat16 has float32's range: the batch that overflows float16 is fine, and nothing is flagged."""
    d = str(tmp_path / "bf16")
    _, recs = SM.run(d, "numerics", dtype="bfloat16")
    assert not recs[45]["spike"]
    rep = replay(d, 45)
    assert rep.verdict == "none", rep.to_text()


def test_a_replay_that_does_not_reproduce_the_log_gives_no_verdict(runs, tmp_path):
    import shutil

    d = str(tmp_path / "tampered")
    shutil.copytree(runs["numerics"][0], d)
    path = os.path.join(d, "watch.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    rec = json.loads(lines[42])
    rec["loss"] *= 1.5
    lines[42] = json.dumps(rec)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    rep = replay(d, 45)
    assert rep.verdict == "not_reproduced" and rep.code == "TS695" and "step 42" in rep.message, rep.to_text()


def test_missing_batch_and_checkpoint_are_refused(runs, tmp_path):
    import shutil

    d = str(tmp_path / "nobatch")
    shutil.copytree(runs["data"][0], d)
    os.remove(os.path.join(d, "batches", "43.pt"))
    with pytest.raises(ReplayError, match="batch 43 was not saved"):
        replay(d, 45)
    with pytest.raises(ReplayError, match="not in the log"):
        replay(d, 999)


def test_agent_documents(runs):
    for case, code in (("numerics", "TS692"), ("data", "TS693"), ("optimizer", "TS694"), ("clean", "TS696")):
        doc = replay(runs[case][0], 45).to_agent(runs[case][0])
        json.dumps(doc, allow_nan=False)
        assert doc["command"] == "replay" and doc["codes"] == [code] and code in codes.CODES
        assert doc["exit_code"] == (0 if case == "clean" else 1)


def _cli(*argv):
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + HERE, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "truestar", *argv], capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace", timeout=300)


def test_cli(runs, tmp_path):
    p = _cli("replay", runs["numerics"][0], "--step", "45")
    assert p.returncode == 1 and "verdict: NUMERICS [TS692]" in p.stdout and "first split norm" in p.stdout, \
        p.stdout + p.stderr
    p = _cli("replay", runs["data"][0], "--step", "45", "--format", "agent")
    assert p.returncode == 1 and json.loads(p.stdout)["verdict"] == "data", p.stdout + p.stderr
    p = _cli("replay", runs["clean"][0], "--step", "45")
    assert p.returncode == 0, p.stdout + p.stderr
    p = _cli("replay", str(tmp_path), "--step", "3", "--format", "agent")
    assert p.returncode == 2 and json.loads(p.stdout)["issues"][0]["code"] == "TS697", p.stdout + p.stderr
