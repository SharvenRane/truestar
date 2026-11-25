"""truestar conform vllm | sglang: the judge, the battery's bookkeeping and the CLI, without any engine.

The engines need Linux and a CUDA GPU; the real runs are in docs/CONFORM_SERVING.md. Here a CPU driver
stands in for the engine (Truestar's own run in the serving dtype), clean and with planted faults.
"""

import importlib.util
import json
import os

import pytest
import torch

pytest.importorskip("safetensors")

from truestar.cli import main
from truestar.conform import serving_battery as B
from truestar.conform.serving import ServingBundleError, check, to_text
from truestar.conform.serving_model import ServeSpec, logprobs, prompts, weights

TINY = dict(vocab=64, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, ffn=128, n_prompts=3,
            prefix_len=8, suffix_len=6, max_tokens=5, top=3, chunk=4)


def _run(tmp_path, driver=B.CpuEmulatedDriver, **kw):
    spec = ServeSpec(**{**TINY, **kw})
    return B.run_all("cpu-emulated", str(tmp_path / "res"), spec, in_process=True, driver_cls=driver)


class _BrokenDown(B.CpuEmulatedDriver):
    """An engine whose layer 1 down projection is computed 1.5 times too large."""

    def __init__(self, *a):
        super().__init__(*a)
        self.W = dict(self.W)
        self.W["1.down"] = (self.W["1.down"].float() * 1.5).to(self.W["1.down"].dtype)


class _BatchDependent(B.CpuEmulatedDriver):
    """An engine whose numbers move a little when other prompts share the batch."""

    def logprobs(self, ids, batch_size):
        lp = super().logprobs(ids, batch_size)
        return lp + 1e-3 if batch_size > 1 else lp


def test_spec_rebuilds_the_same_weights_and_prompts():
    s = ServeSpec(**TINY)
    a, b = weights(s), weights(s)
    assert all(torch.equal(a[k], b[k]) for k in a) and a["0.q"].dtype == torch.bfloat16
    assert prompts(s) == prompts(s)
    P = prompts(s)
    assert len(P) == 3 and all(p[:8] == P[0][:8] and len(p) == 14 for p in P)  # shared prefix


def test_weights_and_prompts_are_pinned_across_versions():
    # The engine host and the judge run different torch and NumPy builds (torch 2.14 on the GPU host drew
    # other torch.randn values than torch 2.11 here); the frozen NumPy stream must give the same bits.
    s = ServeSpec(**TINY)
    from truestar.conform.serving_model import weights_sha256

    assert weights_sha256(weights(s)) == "73cf5b2a6f5d45605384ddeb4ce05f3fd3ed9dd66bf4dc5848ccc3abaff76e02"
    assert prompts(s)[0] == [22, 22, 47, 40, 59, 31, 16, 40, 54, 51, 15, 29, 6, 60]


def test_serving_dtype_run_is_close_to_float64_but_not_equal():
    s = ServeSpec(**TINY)
    W = weights(s)
    p = prompts(s)[0]
    ref, emu = logprobs(s, W, p), logprobs(s, W, p, torch.bfloat16)
    d = (ref - emu).abs().max().item()
    assert 0 < d < 0.5


def test_clean_engine_conforms_and_is_deterministic(tmp_path):
    rep = check(_run(tmp_path))
    codes = {c for c, _ in rep.issues}
    assert not codes & {"TS1322", "TS1323", "TS1324", "TS1326", "TS1327", "TS1328", "TS1329"}, rep.issues
    assert not rep.finding
    assert all(s.bitwise for s in rep.selfs if s.available)
    assert {(p.setting, p.name) for p in rep.passes} >= {("base", "batch1"), ("base", "batch8"),
                                                        ("prefix_cache", "gen_only"), ("chunked_prefill", "batch1")}
    assert all(p.positions > 0 for p in rep.passes)
    assert rep.tol >= rep.floor
    txt = to_text(rep)
    assert "vs float64" in txt and "batch 1 vs batch 8" in txt and "overall: OK" in txt


def test_planted_weight_fault_is_divergent(tmp_path):
    rep = check(_run(tmp_path, _BrokenDown))
    codes = {c for c, _ in rep.issues}
    assert "TS1323" in codes and rep.finding
    base = next(p for p in rep.passes if (p.setting, p.name) == ("base", "batch1"))
    assert base.over_tol > 0


def test_batch_dependence_is_reported_only_for_the_batch(tmp_path):
    rep = check(_run(tmp_path, _BatchDependent))
    by = {s.key: s for s in rep.selfs}
    assert not by["batch"].bitwise and by["batch"].max_dlogprob == pytest.approx(1e-3, rel=1e-6)
    assert by["repeat"].bitwise and by["chunked_prefill"].bitwise and by["prefix_cache"].bitwise
    codes = [c for c, _ in rep.issues]
    assert "TS1327" in codes and "TS1326" not in codes and "TS1323" not in codes
    assert rep.finding


def test_a_failed_setting_is_reported_not_replaced(tmp_path):
    class _NoChunk(B.CpuEmulatedDriver):
        def __init__(self, model_dir, spec, setting):
            if setting["chunked"]:
                raise ValueError("chunked prefill is not supported")
            super().__init__(model_dir, spec, setting)

    rep = check(_run(tmp_path, _NoChunk))
    assert any(c == "TS1322" and "chunked_prefill" in m for c, m in rep.issues)
    chunk = next(s for s in rep.selfs if s.key == "chunked_prefill")
    assert not chunk.available and "not supported" in chunk.reason


def test_other_weights_are_refused(tmp_path):
    d = _run(tmp_path)
    mp = os.path.join(d, "meta.json")
    meta = json.load(open(mp))
    meta["spec"]["seed"] += 1
    json.dump(meta, open(mp, "w"))
    with pytest.raises(ServingBundleError, match="not the weights the engine"):
        check(d)


def _agent(argv, capsys):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_cli_judges_a_results_dir(tmp_path, capsys):
    d = _run(tmp_path)
    mp = os.path.join(d, "meta.json")
    meta = json.load(open(mp))
    meta["engine"] = "vllm"  # the CPU stand in, filed as an engine run
    json.dump(meta, open(mp, "w"))
    code, doc = _agent(["conform", "vllm", d, "--format", "agent"], capsys)
    assert code == 0 and doc["status"] == "ok" and doc["mode"] == "serving"
    assert doc["vs_float64"] and doc["vs_itself"] and doc["tolerance"]["nats"] > 0
    code, doc = _agent(["conform", "sglang", d, "--format", "agent"], capsys)  # a vllm run is not sglang's
    assert code == 2 and doc["codes"] == ["TS1322"]


def test_cli_refuses_tensorrt_llm_by_name(capsys):
    code, doc = _agent(["conform", "tensorrt-llm", "--format", "agent"], capsys)
    assert code == 2 and doc["codes"] == ["TS1321"] and "TensorRT-LLM" in doc["issues"][0]["message"]


@pytest.mark.skipif(importlib.util.find_spec("vllm") is not None, reason="vllm is installed here")
def test_cli_names_the_missing_engine_and_the_way_round(capsys, tmp_path):
    code, doc = _agent(["conform", "vllm", "--out", str(tmp_path / "o"), "--format", "agent"], capsys)
    assert code == 2 and doc["codes"] == ["TS1320"]
    cmds = [x.get("command", "") for x in doc["next_actions"]]
    assert any("serving_battery run --engine vllm" in c for c in cmds)
    assert not os.path.exists(tmp_path / "o")  # nothing ran, nothing written


def test_cli_missing_dir_is_a_usage_error(tmp_path, capsys):
    code, doc = _agent(["conform", "vllm", str(tmp_path / "nope"), "--format", "agent"], capsys)
    assert code == 2 and doc["codes"] == ["TS1322"]


DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "data", "serving")


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_saved_engine_runs_judge_as_documented(engine):
    """The L4 runs in docs/CONFORM_SERVING.md: within tolerance of float64, only near tie flips, and
    batching, prefix caching (when the cache is read) and chunked prefill change the bits."""
    rep = check(os.path.join(DATA, f"{engine}_bfloat16"), engine)
    codes = {c for c, _ in rep.issues}
    assert not codes & {"TS1322", "TS1323", "TS1324", "TS1326"}
    assert {"TS1325", "TS1327", "TS1328", "TS1329"} <= codes
    by = {s.key: s for s in rep.selfs}
    assert by["repeat"].bitwise and not by["batch"].bitwise
    assert by["prefix_cache_gen"].cached_tokens > 0 and not by["prefix_cache_gen"].bitwise
    assert all(p.over_tol == 0 and p.rms < 0.05 for p in rep.passes)


def test_planted_rope_theta_in_a_real_engine_is_caught():
    rep = check(os.path.join(DATA, "vllm_bfloat16_planted"), "vllm")
    codes = {c for c, _ in rep.issues}
    assert {"TS1323", "TS1324"} <= codes
    base = next(p for p in rep.passes if (p.setting, p.name) == ("base", "batch1"))
    assert base.rms > 0.5 and base.first_real["index"] == 0
    assert "rope_theta" in to_text(rep)


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None or not torch.cuda.is_available(),
                    reason="needs vllm and a CUDA GPU (the real runs are in docs/CONFORM_SERVING.md)")
def test_vllm_real_run(tmp_path):
    d = B.run_all("vllm", str(tmp_path / "v"), ServeSpec(), settings=["base"])
    rep = check(d, "vllm")
    assert not rep.setting_errors and rep.passes


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None or not torch.cuda.is_available(),
                    reason="needs sglang and a CUDA GPU (the real runs are in docs/CONFORM_SERVING.md)")
def test_sglang_real_run(tmp_path):
    d = B.run_all("sglang", str(tmp_path / "s"), ServeSpec(), settings=["base"])
    rep = check(d, "sglang")
    assert not rep.setting_errors and rep.passes
