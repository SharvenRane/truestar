"""truestar.models.from_pretrained: Hugging Face decoders built natively from a local folder.

Each family: a tiny transformers model with random weights (built from a config in code, nothing
downloaded), saved with save_pretrained, loaded natively, and compared with transformers in
float64 at 1e-10 relative (the bar of tests/test_hf_models.py, with transformers' float32 casts
kept exact the same way), for the compiled native function, the eager layer, and a prompt plus
decode steps through the KV cache.
"""

import json
import os

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from truestar import models  # noqa: E402
from truestar.interop import safetensors_np as stnp  # noqa: E402
from truestar.ir import reference  # noqa: E402

BAR = 1e-10
VOCAB = 96


def _rel(got, want):
    scale = want.abs().max().item() or 1.0
    return (got.double() - want.double()).abs().max().item() / scale


def _cfg(name):
    t = transformers
    base = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=64)
    if name == "llama":
        return t.LlamaConfig(**base, rope_theta=500000.0, attention_bias=True,
                             rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                           "high_freq_factor": 4.0, "original_max_position_embeddings": 16})
    if name == "llama_tied":
        return t.LlamaConfig(**base, tie_word_embeddings=True, rope_scaling={"rope_type": "linear", "factor": 2.0})
    if name == "mistral":
        return t.MistralConfig(**base, sliding_window=3, head_dim=8)
    if name == "qwen2":
        return t.Qwen2Config(**base, tie_word_embeddings=True)
    if name == "qwen3":
        return t.Qwen3Config(**base, head_dim=16)
    if name == "phi3":
        return t.Phi3Config(**base, sliding_window=4, pad_token_id=0)
    if name == "gemma2":
        return t.Gemma2Config(**base, head_dim=16, query_pre_attn_scalar=12, sliding_window=3,
                              attn_logit_softcapping=5.0, final_logit_softcapping=3.0)
    if name == "gemma3":
        return t.Gemma3TextConfig(**base, head_dim=16, query_pre_attn_scalar=12, sliding_window=3,
                                  sliding_window_pattern=2, rope_local_base_freq=100.0,
                                  rope_scaling={"rope_type": "linear", "factor": 4.0})
    raise KeyError(name)


def _has(name):
    cls = {"qwen3": "Qwen3Config", "gemma3": "Gemma3TextConfig"}.get(name)
    return cls is None or hasattr(transformers, cls)


FAMILIES = [n for n in ("llama", "llama_tied", "mistral", "qwen2", "qwen3", "phi3", "gemma2", "gemma3") if _has(n)]
SKIPPED = sorted(set(("qwen3", "gemma3")) - set(FAMILIES))


def _hf(name, seed=0):
    cfg = _cfg(name)
    attn = "eager" if name.startswith("gemma") else "sdpa"
    torch.manual_seed(seed)
    m = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, torch_dtype=torch.float64)
    m = m.double().eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():   # every weight random, norms and biases included: a swapped name shows
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.2)
    return m


_SAVED = {}


def _saved(name, tmp_path_factory):
    if name not in _SAVED:
        m = _hf(name)
        d = tmp_path_factory.mktemp(name)
        m.save_pretrained(str(d), safe_serialization=True)
        _SAVED[name] = (m, str(d))
    return _SAVED[name]


def _want(m, ids):
    with torch.no_grad(), reference.exact_float32_casts():
        return m(input_ids=ids, use_cache=False).logits


def _ids(b, t, seed=3):
    return torch.randint(0, VOCAB, (b, t), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", FAMILIES)
def test_native_logits_match_transformers_float64(name, tmp_path_factory):
    m, path = _saved(name, tmp_path_factory)
    lm = models.from_pretrained(path)
    assert lm.model.dtype == "f64" and lm.info.cast == {}
    for b, t in ((2, 7), (1, 5)):
        ids = _ids(b, t, seed=b + t)
        want = _want(m, ids)
        got = lm(ids)
        assert got.shape == want.shape
        assert _rel(got, want) < BAR, (name, b, t, _rel(got, want))
        eager = lm.model(lm.params, ids, torch.arange(t))
        assert _rel(eager, want) < BAR, (name, "eager", _rel(eager, want))
    assert lm.fns[0].compile_count == 1   # one compile for both shapes


@pytest.mark.parametrize("name", FAMILIES)
def test_native_decode_steps_match_the_full_forward(name, tmp_path_factory):
    m, path = _saved(name, tmp_path_factory)
    lm = models.from_pretrained(path)
    ids = _ids(2, 8, seed=11)
    want = _want(m, ids)
    cache = lm.init_cache(2, 10)
    logits, cache = lm.step(ids[:, :5], torch.arange(5), cache)
    assert _rel(logits, want[:, :5]) < BAR
    for t in (5, 6, 7):
        logits, cache = lm.step(ids[:, t:t + 1], torch.tensor([t]), cache)
        assert _rel(logits[:, 0], want[:, t]) < BAR, (name, t, _rel(logits[:, 0], want[:, t]))
    assert lm.fns[1].compile_count == 1
    out = lm.generate(ids[:, :4], 3)
    ref = ids[:, :4]
    for _ in range(3):   # greedy by full recomputation in transformers (its generate stops at eos)
        ref = torch.cat([ref, _want(m, ref)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, ref), (name, out, ref)


def test_sharded_folder_loads_the_same(tmp_path):
    m = _hf("llama")
    m.save_pretrained(str(tmp_path), safe_serialization=True, max_shard_size="40KB")
    assert os.path.isfile(tmp_path / stnp.INDEX_NAME)
    ck = stnp.open_checkpoint(str(tmp_path))
    assert len(ck.files) > 2
    lm = models.from_pretrained(str(tmp_path))
    assert lm.info.shards == len(ck.files)
    ids = _ids(2, 6)
    assert _rel(lm(ids), _want(m, ids)) < BAR


def test_broken_index_is_refused_with_every_problem(tmp_path):
    m = _hf("qwen2")
    m.save_pretrained(str(tmp_path), safe_serialization=True, max_shard_size="40KB")
    idx_path = tmp_path / stnp.INDEX_NAME
    idx = json.loads(idx_path.read_text())
    shards = sorted(set(idx["weight_map"].values()))
    os.remove(tmp_path / shards[-1])
    name = next(k for k, v in idx["weight_map"].items() if v == shards[0])
    idx["weight_map"][name] = shards[1]
    idx_path.write_text(json.dumps(idx))
    with pytest.raises(stnp.SafetensorsError) as e:
        models.from_pretrained(str(tmp_path))
    assert e.value.ts_code == "TS1395"
    text = str(e.value)
    assert "not found next to it" in text and f"{name}: the index maps it to {shards[1]}" in text


def test_config_and_weights_that_disagree_are_refused(tmp_path):
    m = _hf("llama")
    m.save_pretrained(str(tmp_path), safe_serialization=True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["intermediate_size"] = 40
    cfg["num_hidden_layers"] = 4
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(models.ModelLoadError) as e:
        models.from_pretrained(str(tmp_path))
    assert e.value.code == "TS1392"
    probs = "\n".join(e.value.problems)
    assert "TS1392 model.layers.3.input_layernorm.weight" in probs
    assert "TS1394 model.layers.0.mlp.gate_proj.weight: [48, 32] stored, [40, 32] expected" in probs


def test_unsupported_settings_and_hub_ids_are_refused(tmp_path):
    with pytest.raises(models.ModelLoadError) as e:
        models.from_pretrained("meta-llama/Llama-3.2-1B")
    assert e.value.code == "TS1397"
    raw = _cfg("llama").to_dict()
    raw["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0}
    with pytest.raises(models.ModelLoadError) as e:
        models.decoder_config(raw)
    assert e.value.code == "TS1391" and "yarn" in str(e.value)
    with pytest.raises(models.ModelLoadError) as e:
        models.decoder_config({"model_type": "gpt2"})
    assert e.value.code == "TS1390"


def test_bf16_checkpoint_widens_exactly(tmp_path):
    m = _hf("llama").to(torch.bfloat16)
    m.save_pretrained(str(tmp_path), safe_serialization=True)
    ck = stnp.open_checkpoint(str(tmp_path))
    assert set(ck.dtypes()) == {"BF16"}
    w = ck["model.layers.0.self_attn.q_proj.weight"]
    ref = m.model.layers[0].self_attn.q_proj.weight.detach()
    assert np.array_equal(w.numpy(), ref.float().numpy())          # NumPy only, no torch in the widening
    assert torch.equal(w.torch(), ref)                              # zero copy view in the stored dtype
    lm = models.from_pretrained(str(tmp_path))                      # float32 by default
    assert lm.model.dtype == "f32" and lm.info.cast.get("BF16->f32")
    lm16 = models.from_pretrained(str(tmp_path), dtype="bf16")
    assert lm16.info.cast == {} and lm16.info.zero_copy == lm16.info.tensors


def test_fp8_tables_match_torch():
    bits = torch.arange(256, dtype=torch.uint8)
    for name, dt in (("F8_E4M3", torch.float8_e4m3fn), ("F8_E5M2", torch.float8_e5m2)):
        want = bits.view(dt).float().numpy()
        got = stnp.widen(bits.numpy(), name)
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
        ok = ~np.isnan(want)
        assert np.array_equal(got[ok], want[ok]), name


def test_reader_does_not_import_torch():
    import subprocess
    import sys
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    code = ("import sys, importlib.util as u; s = u.spec_from_file_location('stnp', r'%s'); m = u.module_from_spec(s); "
            "sys.modules['stnp'] = m; s.loader.exec_module(m); print('torch' in sys.modules)"
            % os.path.join(src, "truestar", "interop", "safetensors_np.py"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_bar_has_teeth(tmp_path_factory):
    """Two same shape weights swapped, or the rotary table off by one float32 step, fail the bar."""
    m, path = _saved("llama", tmp_path_factory)
    ids = _ids(2, 7)
    want = _want(m, ids)
    lm = models.from_pretrained(path)
    p = lm.params
    swapped = dict(p, l1=dict(p["l1"], attn=dict(p["l1"]["attn"], q=p["l1"]["attn"]["o"], o=p["l1"]["attn"]["q"])))
    assert _rel(lm.fns[0](swapped, ids, torch.arange(7)), want) > 1e-3
    inv = p["rope"]["inv"]
    nudged = dict(p, rope={"inv": torch.nextafter(inv, torch.full_like(inv, 2.0))})
    assert _rel(lm.fns[0](nudged, ids, torch.arange(7)), want) > BAR
