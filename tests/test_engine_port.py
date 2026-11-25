"""llama.cpp as a port check candidate (docs/PORT_ENGINES.md, pain row P32).

A tiny Llama style decoder is written to GGUF by tests/engine_models.py, quantized by the engine's
own tool and run in llama.cpp on the CPU. Each fault is injected in the file (settings, layout,
values, tokenizer) or in one engine node through the tap hook, and must be named with the right
blame; every clean variant must report clean.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("llama_cpp")

import engine_models as em  # noqa: E402
from truestar.engines import engine_port as EP  # noqa: E402
from truestar.engines import gguf, llamacpp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("gguf")
    m = em.build()
    f = {"model": m, "dir": str(d)}
    f["f32"] = em.to_gguf(m, str(d / "f32.gguf"))
    f["f16w"] = em.to_gguf(m, str(d / "f16w.gguf"), float_type="F16")
    for q in ("Q8_0", "Q4_K_M"):
        f[q] = str(d / f"{q}.gguf")
        llamacpp.quantize(f["f32"], f[q], q)
    return f


IDS = em.reference_tokenize(em.PROMPT)


def check(files, path, **kw):
    return EP.engine_port_check(files["model"], path, IDS, **kw)


# ----------------------------------------------------------------------------- the format


def test_gguf_round_trip_and_metadata(files):
    g = gguf.read(files["f32"])
    assert g.arch() == "llama" and len(g.tensors) == 3 + 9 * em.LAYERS
    assert g.metadata["llama.attention.head_count_kv"] == em.KV_HEADS
    assert g.metadata["tokenizer.ggml.tokens"] == em.VOCAB
    w = g.tensor("blk.0.ffn_up.weight")
    ref = files["model"].model.layers[0].mlp.up_proj.weight.detach().double().numpy()
    assert np.array_equal(w, ref)  # float32 weights stored as F32: bitwise


def test_decoders_equal_the_gguf_package(files):
    G = pytest.importorskip("gguf")
    seen = set()
    for key in ("Q8_0", "Q4_K_M", "f16w"):
        mine = gguf.read(files[key])
        for t in G.GGUFReader(files[key]).tensors:
            ref = G.quants.dequantize(t.data, t.tensor_type).astype(np.float64).reshape(mine.tensors[t.name].shape)
            assert np.array_equal(mine.tensor(t.name), ref), t.name
            seen.add(mine.tensors[t.name].type_name)
    assert {"Q8_0", "Q4_K", "Q6_K", "F16"} <= seen


def test_corrupt_files_are_refused(tmp_path, files):
    raw = open(files["f32"], "rb").read()
    p = tmp_path / "trunc.gguf"
    p.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(gguf.GGUFError, match="past the end"):
        gguf.read(str(p))
    p.write_bytes(b"GGML" + raw[4:])
    with pytest.raises(gguf.GGUFError, match="magic"):
        gguf.read(str(p))


def test_activation_rounding_model_matches_the_q8_0_rule():
    x = np.linspace(-3, 5, 64, dtype=np.float32)[None]
    y = gguf.round_like_dot_type(x, "Q8_0")
    d = 5 / 127  # amax of the second block
    assert np.allclose(y[0, 32:], np.rint(x[0, 32:] / np.float32(np.float16(d))) * np.float16(d), atol=0)
    assert np.abs(y - x).max() <= 0.5 * float(np.float16(d)) * 1.01


# ----------------------------------------------------------------------------- engine plumbing


def test_engine_tokenizer_equals_the_reference(files):
    assert llamacpp.tokenize(files["f32"], em.PROMPT) == IDS


def test_taps_do_not_change_the_engine_on_cpu(files):
    a = llamacpp.run(files["Q4_K_M"], IDS, tap=True)
    b = llamacpp.run(files["Q4_K_M"], IDS, tap=False)
    assert np.array_equal(a.logits, b.logits)
    assert len(a.taps) > 50 and not b.taps


# ----------------------------------------------------------------------------- clean ports


@pytest.mark.parametrize("key,kw", [("f32", {}), ("f16w", {}), ("Q8_0", {}), ("Q4_K_M", {}),
                                    ("f32", {"kv_type": "f16"}), ("Q4_K_M", {"kv_type": "f16"})])
def test_clean_ports_report_clean(files, key, kw):
    r = check(files, files[key], **kw)
    assert r.status == "ok", r.to_text()
    assert not [c for c in r.codes if c in ("TS640", "TS641", "TS642", "TS643", "TS646", "TS649", "TS657")]
    assert max(s.rel_err / s.threshold for s in r.engine_steps) < 0.5
    assert len(r.engine_steps) == 2 + 15 * em.LAYERS + 1
    assert r.decomposition["conversion"] < EP.CONVERSION_FLOOR  # metadata holds eps and theta as float32


def test_quantization_loss_is_measured_from_float64_only(files):
    q8 = check(files, files["Q8_0"])
    q4 = check(files, files["Q4_K_M"])
    f = check(files, files["f32"])
    assert f.decomposition["quantization"] < 1e-12 and "TS644" not in f.codes and "TS645" not in f.codes
    assert 0 < q8.decomposition["quantization"] < q4.decomposition["quantization"]
    assert q8.decisions_quant.kl_mean < q4.decisions_quant.kl_mean
    # the engine part stays at float32 level even where the quantization loss is large
    assert q4.decomposition["engine"] < 1e-5 < q4.decomposition["quantization"]
    assert {"TS644", "TS645"} & set(q4.codes)
    assert q4.engine["paths"]["Q4_K"] == "activations in Q8_K blocks"
    assert q8.engine["paths"]["Q8_0"] == "activations in Q8_0 blocks"


def test_hugging_face_llama_is_clean(tmp_path):
    tr = pytest.importorskip("transformers")
    cfg = tr.LlamaConfig(vocab_size=em.V, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=50000.0,
                         max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(3)
    m = tr.LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim == 2:
                p.normal_(0, p.shape[1] ** -0.5)
    path = em.to_gguf(m, str(tmp_path / "hf.gguf"))
    r = EP.engine_port_check(m, path, IDS)
    assert r.status == "ok", r.to_text()
    # transformers keeps rotary cos and sin and RMSNorm in float32 even after .double(): float32 level
    assert 1e-9 < r.decomposition["reference_replay"] < EP.REPLAY_FLOAT32
    assert "float32 accurate" in r.engine["reference_note"]


# ----------------------------------------------------------------------------- injected faults


@pytest.mark.parametrize("name,kw,code,blame,step,words", [
    ("theta", {"theta": 500000.0}, "TS641", "settings", "0.q_rope", "rope theta is 500000 in the GGUF"),
    ("eps", {"eps": 1e-6}, "TS641", "settings", "0.attn_norm", "norm eps is 1e-06 in the GGUF"),
    ("scale", {"rope_scale": 4.0}, "TS641", "settings", "0.q_rope", "rope scaling factor is 4"),
    ("transpose", {"transpose": ["blk.0.attn_output.weight"]}, "TS642", "weights", "0.attn_out", "transpose"),
    ("noperm", {"skip_permute": ["blk.1.attn_k.weight"]}, "TS642", "weights", "1.k", "reference row order"),
    ("values", {"scale_weight": {"blk.1.ffn_down.weight": 2.0}}, "TS642", "weights", "1.down", "norm ratio 2"),
])
def test_conversion_faults_are_named(files, tmp_path, name, kw, code, blame, step, words):
    p = em.to_gguf(files["model"], str(tmp_path / f"{name}.gguf"), **kw)
    r = check(files, p)
    assert r.status == "divergent" and code in r.codes and r.blame == blame, r.to_text()
    assert r.first_conversion().key == step
    assert words in r.explanation
    assert "TS640" not in r.codes  # the engine computed the faulty file correctly


def test_a_quantized_file_with_a_fault_names_the_fault_not_the_quantization(files, tmp_path):
    p = em.to_gguf(files["model"], str(tmp_path / "t.gguf"), transpose=["blk.1.attn_output.weight"], eps=1e-6)
    q = str(tmp_path / "tq.gguf")
    llamacpp.quantize(p, q, "Q4_K_M")
    r = check(files, q)
    assert {"TS641", "TS642"} <= set(r.codes) and "TS640" not in r.codes, r.to_text()
    edge = next(w for w in r.weights if w.gguf_name == "blk.1.attn_output.weight")
    assert edge.kind == "transpose" and edge.verdict == "layout" and edge.type_name == "Q4_K"


def test_tokenizer_setting_is_named(files, tmp_path):
    p = em.to_gguf(files["model"], str(tmp_path / "nobos.gguf"), add_bos=False)
    r = check(files, p, prompt=em.PROMPT)
    assert "TS646" in r.codes and r.blame == "tokenizer"
    assert "does not add one" in r.explanation
    assert "TS646" not in check(files, files["f32"], prompt=em.PROMPT).codes


def _bump(name, op, frac):
    def mutate(tap, a):
        if tap.name == name and tap.op == op:
            a = a.copy()
            a.reshape(-1)[7] += frac * np.abs(a).max()
            return a
    return mutate


@pytest.mark.parametrize("key", ["f32", "Q8_0", "Q4_K_M"])
@pytest.mark.parametrize("node,op,step", [("ffn_up-1", "MUL_MAT", "1.up"), ("Kcur-0", "ROPE", "0.k_rope"),
                                          ("kqv_out-1", "CONT", "1.attn"), ("attn_norm-0", "MUL", "0.attn_norm")])
def test_an_engine_fault_is_blamed_on_the_engine_at_its_node(files, key, node, op, step):
    r = check(files, files[key], mutate=_bump(node, op, 0.01))
    fe = r.first_engine()
    assert r.status == "divergent" and r.blame == "engine" and fe is not None, r.to_text()
    assert fe.key == step and fe.node == node
    assert "TS641" not in r.codes and "TS642" not in r.codes
    # later steps read the faulty node as input, so only the faulty node itself is over threshold
    assert [s.key for s in r.engine_steps if s.over] == [step]


def test_small_engine_faults_hidden_under_quantization_loss_are_found(files):
    """A 1e-4 relative error in one node is 100x below the Q4_K_M loss, so the final logits and
    a per layer comparison against the reference cannot see it; the own error replay can."""
    r = check(files, files["Q4_K_M"], mutate=_bump("ffn_gate-0", "MUL_MAT", 2e-4))
    assert r.first_engine().key == "0.gate", r.to_text()
    assert r.decomposition["quantization"] > 100 * r.first_engine().rel_err


def test_the_engine_ignoring_a_setting_is_told_apart_from_the_file():
    taps = [llamacpp.Tap(0, "norm-0", "RMS_NORM", "f32", (256, 16, 1, 1),
                         tuple(np.array([1e-6], np.float32).view(np.int32).tolist()) + (0,) * 15)]
    obs = EP._observed(llamacpp.EngineRun(np.zeros((1, 1)), taps))
    assert obs["eps"] == [pytest.approx(1e-6)]
    s = EP.SettingFinding("norm eps", 1e-5, 1e-5, obs["eps"][0], "engine")
    assert "the engine used norm eps 1e-06 although the GGUF says 1e-05" == s.sentence()


# ----------------------------------------------------------------------------- tests that fail on purpose


def test_without_the_rounding_paths_a_clean_q8_file_false_alarms(files, monkeypatch):
    def f32_only(st, wt, kv):
        yield ("f32", None, None)
    monkeypatch.setattr(EP, "_paths", f32_only)
    r = check(files, files["Q8_0"])
    assert "TS640" in r.codes and r.first_engine().step == "q"


def test_without_transposes_in_the_matcher_the_transposed_weight_is_not_named(files, tmp_path, monkeypatch):
    p = em.to_gguf(files["model"], str(tmp_path / "tr.gguf"), transpose=["blk.0.attn_output.weight"])
    orig = EP._transforms
    monkeypatch.setattr(EP, "_transforms", lambda r, role, cfg: {k: v for k, v in orig(r, role, cfg).items()
                                                                 if "transpose" not in k})
    r = check(files, p)
    assert not any(w.kind == "transpose" for w in r.weights)


# ----------------------------------------------------------------------------- command line


def _ref_file(tmp_path, prompt=True):
    f = tmp_path / "ref.py"
    f.write_text(
        "import sys\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'tests')!r})\n"
        "import torch, engine_models as em\n"
        "def get_model():\n    return em.build()\n"
        "def get_inputs():\n    return torch.tensor([em.reference_tokenize(em.PROMPT)])\n"
        + ("def get_prompt():\n    return em.PROMPT\n" if prompt else ""), encoding="utf-8")
    return str(f)


def _cli(tmp_path, *argv):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "truestar", "port", *argv], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)


def test_cli_agent_json_and_exit_codes(files, tmp_path):
    ref = _ref_file(tmp_path)
    p = _cli(tmp_path, ref, "gguf:" + files["Q8_0"], "--format", "agent")
    assert p.returncode == 0, p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert doc["command"] == "port" and doc["status"] == "ok" and doc["candidate"]["kind"] == "gguf"
    assert doc["engine_verdict"]["first_over_threshold"] is None
    bad = em.to_gguf(files["model"], str(tmp_path / "theta.gguf"), theta=500000.0)
    p = _cli(tmp_path, ref, "gguf:" + bad, "--format", "agent")
    assert p.returncode == 1, p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert "TS641" in doc["codes"] and doc["blame"] == "settings"
    assert any(a["kind"] == "fix_setting" for a in doc["next_actions"])
    p = _cli(tmp_path, ref, "gguf:" + bad)
    assert p.returncode == 1 and "CONVERSION: rope theta is 500000" in p.stdout
    p = _cli(tmp_path, ref, "gguf:" + str(tmp_path / "missing.gguf"), "--format", "agent")
    assert p.returncode == 2 and json.loads(p.stdout)["codes"] == ["TS020"]
