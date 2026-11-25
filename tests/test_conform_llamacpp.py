"""`truestar conform llamacpp` (docs/CONFORM_LLAMACPP.md).

Everything except the last test runs without llama.cpp: the engine is replaced by a runner that
returns logits Truestar computed, so the verdict logic (derived tolerance, near ties, quantization
apart from engine error, settings) is tested on known answers. The last test runs the real engine
when TRUESTAR_LLAMA_CPP points at a llama.cpp build, and skips with the reason otherwise.
"""

import copy
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

import engine_models as EM
from truestar import codes
from truestar.conform import llamacpp as L
from truestar.interop.safetensors import save_file

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("llamacpp")
    model = EM.build(seed=3)
    hf = d / "hf"
    hf.mkdir()
    c = vars(EM.config()).copy()
    c.pop("rope_scaling")
    c.update(model_type="llama", architectures=["LlamaForCausalLM"])
    (hf / "config.json").write_text(json.dumps(c), encoding="utf-8")
    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in model.state_dict().items()},
              str(hf / "model.safetensors"))
    f32 = EM.to_gguf(model, str(d / "f32.gguf"))
    f16 = EM.to_gguf(model, str(d / "f16.gguf"), float_type="F16")
    theta = EM.to_gguf(model, str(d / "theta.gguf"), theta=500000.0)
    return {"model": model, "hf": str(hf), "f32": f32, "f16": f16, "theta": theta}


PROMPTS = [[1, 40, 7, 99, 12, 250, 3, 18], [5, 6, 7, 8, 9, 10, 11, 12]]


def faithful(noise=0.0, at=None, pick_second=False):
    """A runner that computes what a correct float32 engine computes: the float32 replay of the file
    (optionally with noise at one prompt position, or greedy picking the second best token)."""
    def run(path, s, prompts, n_new):
        w = L.load_gguf(path)
        pl = []
        for i, p in enumerate(prompts):
            lg = L.forward(w, p, dt=torch.float32).to(torch.float32).numpy().copy()
            if noise and (at is None or at[0] == i):
                rows = range(lg.shape[0]) if at is None else [at[1]]
                rng = np.random.default_rng(0)
                for j in rows:
                    lg[j] += noise * np.linalg.norm(lg[j]) / np.sqrt(lg.shape[1]) * rng.standard_normal(lg.shape[1])
            pl.append(lg)
        toks, gls = [], []
        for p in prompts:
            seq, rows = list(p), []
            for step in range(n_new):
                lg = L.forward(w, seq, dt=torch.float32)[-1].to(torch.float32).numpy()
                order = np.argsort(-lg, kind="stable")
                t = int(order[1] if (pick_second and step == 0) else order[0])
                rows.append(lg)
                seq.append(t)
            toks.append(seq[len(p):])
            gls.append(np.stack(rows))
        return L.EngineOutput(pl, toks, gls, {"system": "fake"})
    return run


def check(path, runner, settings=None, source=None, n_new=4):
    return L.conform_llamacpp(path, runner=runner, settings=settings or [L.Setting(4, 512, False)],
                              prompts=PROMPTS, n_new=n_new, source=source)


def test_codes_are_defined():
    for i in range(1310, 1320):
        assert f"TS{i}" in codes.CODES


def test_float64_replay_of_hf_dir_and_gguf_is_the_model(tiny):
    m = copy.deepcopy(tiny["model"]).double()
    ids = torch.tensor([PROMPTS[0]])
    with torch.no_grad():
        ref = m(ids)[0]
    hf = L.forward(L.load_hf_dir(tiny["hf"]), PROMPTS[0])
    gg = L.forward(L.load_gguf(tiny["f32"]), PROMPTS[0])  # q and k rows unpermuted on load
    assert float((hf - ref).norm() / ref.norm()) < 1e-12
    assert float((gg - ref).norm() / ref.norm()) < 1e-8  # eps is stored as float32 in the GGUF


def test_faithful_engine_conforms(tiny):
    rep = check(tiny["f32"], faithful())
    assert not rep.finding, rep.to_text()
    assert "TS1311" not in rep.codes and "TS1312" not in rep.codes
    run = rep.formats[0].runs[0]
    assert run.worst_ratio < 1 and run.positions == sum(len(p) for p in PROMPTS)


def test_f16_weights_are_judged_with_their_rounding(tiny):
    rep = check(tiny["f16"], faithful())
    assert not rep.finding, rep.to_text()


def test_engine_error_at_one_position_is_named(tiny):
    rep = check(tiny["f32"], faithful(noise=0.2, at=(1, 5)))
    assert "TS1311" in rep.codes and rep.finding
    over = rep.formats[0].runs[0].over
    assert [(o["prompt"], o["position"]) for o in over] == [(1, 5)]


def test_greedy_flip_beyond_a_near_tie_is_a_bug(tiny):
    rep = check(tiny["f32"], faithful(pick_second=True))
    g = rep.formats[0].runs[0].greedy
    assert all(x.first_divergent == 0 for x in g)
    real = [x for x in g if not x.near_tie]
    assert real and "TS1312" in rep.codes and rep.finding


def test_greedy_flip_inside_the_rounding_bound_is_ambiguous(tiny, monkeypatch):
    # a bound so wide that every top two pair is a near tie: the same flips are now ambiguous
    real_replay = L.replay

    def wide(*a, **k):
        r = real_replay(*a, **k)
        return L.Replay(r.l64, r.tau, r.bound + 1e6, r.gap)
    monkeypatch.setattr(L, "replay", wide)
    rep = check(tiny["f32"], faithful(pick_second=True))
    assert "TS1313" in rep.codes and "TS1312" not in rep.codes
    assert all(x.near_tie for x in rep.formats[0].runs[0].greedy)


def test_quantization_is_reported_apart_from_engine_error(tiny):
    rep = check(tiny["f16"], faithful(), source=tiny["hf"])
    f = rep.formats[0]
    assert "TS1314" in rep.codes and not rep.finding
    assert 1e-6 < f.quantization["rel_max"] < 1e-2  # f16 weights against the float32 source
    assert f.conversion == []


def test_conversion_fault_is_named(tiny):
    rep = check(tiny["theta"], faithful(), source=tiny["hf"])
    assert "TS1316" in rep.codes and rep.finding
    assert any("theta is 500000" in c for c in rep.formats[0].conversion)
    assert "TS1311" not in rep.codes  # the engine computes the file it was given
    assert any(c == "TS1314" and "conversion fault" in m for c, m in rep.issues)  # not called quantization alone


def test_settings_that_change_bits_but_not_tokens(tiny):
    base = faithful()

    def run(path, s, prompts, n_new):
        out = base(path, s, prompts, n_new)
        if s.threads == 4:
            out.prompt_logits = [x * np.float32(1 + 2e-7) for x in out.prompt_logits]
        return out
    rep = check(tiny["f32"], run, settings=[L.Setting(1, 512, False), L.Setting(4, 512, False)])
    assert "TS1318" in rep.codes and "TS1317" not in rep.codes and not rep.finding
    assert rep.formats[0].settings["groups"] == 2


def test_settings_that_change_tokens(tiny):
    good, flip = faithful(), faithful(pick_second=True)

    def run(path, s, prompts, n_new):
        return flip(path, s, prompts, n_new) if s.flash else good(path, s, prompts, n_new)
    rep = check(tiny["f32"], run, settings=[L.Setting(4, 512, False), L.Setting(4, 512, True)])
    assert "TS1317" in rep.codes
    assert "flash attention on" in rep.formats[0].settings["token_changes"]


def test_driver_output_is_parsed():
    vocab, rows = 5, 2
    lg = np.arange(rows * vocab, dtype="<f4")
    blob = b"TSLC" + struct.pack("<III", 1, vocab, 1) + struct.pack("<II", 1, rows) + \
        struct.pack("<2i", 4, 2) + lg.tobytes()
    v, items = L.parse_output(blob)
    assert v == vocab and items[0][0] == 1 and items[0][1] == [4, 2]
    assert np.array_equal(items[0][2], lg.reshape(rows, vocab))


def _cli(*args, env=None):
    e = dict(os.environ, PYTHONPATH=SRC, CUDA_VISIBLE_DEVICES="-1", **(env or {}))
    if "TRUESTAR_LLAMA_CPP" not in (env or {}):
        e.pop("TRUESTAR_LLAMA_CPP", None)
    return subprocess.run([sys.executable, "-m", "truestar", "conform", "llamacpp", *args, "--format", "agent"],
                          capture_output=True, text=True, env=e, timeout=300)


def test_cli_without_a_build_is_refused_with_the_reason(tiny, tmp_path):
    r = _cli(tiny["f32"], "--llama-cpp", str(tmp_path))
    assert r.returncode == 2, r.stderr
    doc = json.loads(r.stdout)
    assert doc["codes"] == ["TS1310"] and "llama" in doc["issues"][0]["message"] and "not found" in doc["issues"][0]["message"]
    r = _cli(tiny["f32"])
    doc = json.loads(r.stdout)
    assert r.returncode == 2 and doc["codes"] == ["TS1310"] and "--llama-cpp" in doc["issues"][0]["message"]


def test_cli_rejects_an_unknown_architecture(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
    r = _cli(str(d), "--llama-cpp", str(tmp_path))
    assert r.returncode == 2 and json.loads(r.stdout)["codes"] == ["TS1319"]


def test_real_llama_cpp(tiny):
    root = os.environ.get("TRUESTAR_LLAMA_CPP")
    if not root:
        pytest.skip("TRUESTAR_LLAMA_CPP is not set: no llama.cpp build to run (the other tests replace the engine)")
    if sys.platform == "win32":
        pytest.skip("the driver is built with a POSIX C++ compiler; Windows is not supported yet")
    rep = L.conform_llamacpp(tiny["f32"], llama_cpp=root, settings=[L.Setting(1, 512, False), L.Setting(4, 7, True)],
                             prompts=PROMPTS, n_new=4, source=tiny["hf"])
    assert not rep.finding, rep.to_text()
    assert rep.build.get("libllama_sha256")
