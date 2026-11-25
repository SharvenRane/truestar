"""truestar.lora: LoRA and QLoRA on truestar.nn models, checked against Hugging Face Llama in float64."""

import math
import os

import pytest
import torch

import truestar
from truestar import Tensor, codes, lora, nn, ops, optim
from truestar.nn.llama import Llama, from_hf, hf_rope

F64 = torch.float64
transformers = pytest.importorskip("transformers")

V, D = 64, 64


def _hf(net: Llama, dtype=F64):
    cfg = transformers.LlamaConfig(vocab_size=net.vocab, hidden_size=net.dim, intermediate_size=net.ffn,
                                   num_hidden_layers=net.depth, num_attention_heads=net.heads,
                                   num_key_value_heads=net.kv_heads, rms_norm_eps=net.eps, rope_theta=net.base,
                                   tie_word_embeddings=net.tied, max_position_embeddings=64,
                                   attn_implementation="sdpa")
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg).to(dtype).eval()
    _exact_rope(m)
    for mod in m.modules():
        if type(mod).__name__ == "LlamaRMSNorm":
            mod.forward = _rms64(mod)
    return m


def _rms64(mod):
    def forward(x):   # transformers computes the norm in float32; the same formula in the input dtype
        return mod.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + mod.variance_epsilon))
    return forward


def _exact_rope(m):
    rot = m.model.rotary_emb
    hd = m.config.hidden_size // m.config.num_attention_heads
    k = torch.arange(0, hd, 2, dtype=F64)
    inv = torch.exp(k * (-math.log(hf_rope(m.config)[0]) / hd))

    def forward(x, position_ids):   # transformers forces float32 here; the same formula in float64
        emb = position_ids[:, :, None].to(F64) * inv[None, None, :]
        emb = torch.cat((emb, emb), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    rot.forward = forward


def _load_tree(m, p):
    """Load a truestar Llama tree (float weights) into a Hugging Face model."""
    sd = {"model.embed_tokens.weight": p["embed_tokens"]["w"], "model.norm.weight": p["norm"]["w"]}
    i = 0
    while f"l{i}" in p:
        b, pre = p[f"l{i}"], f"model.layers.{i}."
        sd[pre + "input_layernorm.weight"] = b["input_layernorm"]["w"]
        sd[pre + "post_attention_layernorm.weight"] = b["post_attention_layernorm"]["w"]
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[pre + f"self_attn.{n}.weight"] = b["self_attn"][n]["w"]
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[pre + f"mlp.{n}.weight"] = b["mlp"][n]["w"]
        i += 1
    if "lm_head" in p:
        sd["lm_head.weight"] = p["lm_head"]["w"]
    missing, unexpected = m.load_state_dict({k: v.to(F64) for k, v in sd.items()}, strict=False)
    assert not unexpected and all("rotary" in k for k in missing), (missing, unexpected)


class _TorchLoRA(torch.nn.Module):
    """A hand written LoRA linear: the independent reference."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base, self.scale = base, scale
        self.a = torch.nn.Parameter(a.clone())
        self.b = torch.nn.Parameter(b.clone())

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale


def _hf_lora(m, lm, a):
    mods = {}
    for path in lm.adapted:
        layer_i, _, rest = path.partition(".")
        parent_name, _, leaf = rest.rpartition(".")
        blk = m.model.layers[int(layer_i[1:])]
        parent = getattr(blk, parent_name)
        node = a
        for k in path.split("."):
            node = node[k]
        mod = _TorchLoRA(getattr(parent, leaf), node["lora_a"].to(F64), node["lora_b"].to(F64), lm.adapted[path].scale)
        setattr(parent, leaf, mod)
        mods[path] = mod
    for prm in m.parameters():
        prm.requires_grad_(False)
    for mod in mods.values():
        mod.a.requires_grad_(True)
        mod.b.requires_grad_(True)
    return mods


def _net(dtype="f64", tied=False):
    return Llama(vocab=V, dim=D, depth=2, heads=4, kv_heads=2, ffn=128, tied=tied, dtype=dtype)


def _random_b(lm, a, seed=5):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in a.items():
        out[k] = _random_b(lm, v, seed + 1) if isinstance(v, dict) else v
    if "lora_b" in out:
        out["lora_b"] = torch.randn(out["lora_b"].shape, generator=g, dtype=out["lora_b"].dtype) * 0.1
    return out


def _loss_fn(lm, dt="f64"):
    @truestar.native(seed=3)
    def loss(a: lm.spec, p: lm.base_spec, ids: Tensor["B", "T", "i64"], y: Tensor["B", "T", "i64"],
             training: bool = False) -> Tensor[dt]:
        logits = lm(p, a, ids, training=training)
        return nn.cross_entropy(ops.reshape(logits, [-1, V]), ops.reshape(y, [-1]))
    return loss


def _batch(seed=0, B=2, T=12):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    return ids, torch.roll(ids, -1, 1)


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# ----------------------------------------------------------------------------- import


@pytest.mark.parametrize("tied", [False, True])
def test_from_hf_matches_hugging_face_in_float64(tied):
    m = _hf(_net(tied=tied))
    net, p = from_hf(m)
    assert net.tied == tied and net.kv_heads == 2
    ids, _ = _batch()

    @truestar.native
    def fwd(p: net.spec, ids: Tensor["B", "T", "i64"]) -> Tensor["B", "T", V, "f64"]:
        return net(p, ids)

    with torch.no_grad():
        ref = m(ids).logits
    assert _rel(fwd(p, ids), ref) < 1e-12
    assert _rel(net(p, ids), ref) < 1e-12


def test_from_hf_refuses_what_it_cannot_represent():
    m = _hf(_net())
    m.config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="rope_scaling"):
        from_hf(m)


# ----------------------------------------------------------------------------- wrapping


def test_wrap_selects_by_pattern_and_leaves_the_model_alone():
    net = _net()
    before = repr(net.parts["l0"].self_attn.parts)
    lm = lora.wrap(net, targets=("q_proj", "v_proj"), rank=4)
    assert sorted(lm.adapted) == ["l0.self_attn.q_proj", "l0.self_attn.v_proj", "l1.self_attn.q_proj",
                                  "l1.self_attn.v_proj"]
    assert repr(net.parts["l0"].self_attn.parts) == before           # the original config is unchanged
    assert isinstance(net.parts["l0"].self_attn.q_proj, nn.Linear)
    assert sorted(lora.wrap(net, targets=("l1.*",), rank=2).adapted) == [
        "l1.mlp.down_proj", "l1.mlp.gate_proj", "l1.mlp.up_proj", "l1.self_attn.k_proj", "l1.self_attn.o_proj",
        "l1.self_attn.q_proj", "l1.self_attn.v_proj"]
    with pytest.raises(lora.LoRAError, match="TS1371.*l0.self_attn.q_proj"):
        lora.wrap(net, targets=("query",))
    with pytest.raises(lora.LoRAError, match="TS1370"):
        lora.wrap(net, rank=0)
    with pytest.raises(lora.LoRAError, match="TS1376"):
        lora.wrap(net, quant="int3")
    with pytest.raises(lora.LoRAError, match="TS1376.*block"):
        lora.wrap(Llama(V, 48, 1, 4, 2, 96, dtype="f64"), quant="int4:g32")
    a = lm.init(seed=0)
    assert lm.num_adapter_params() == 2 * 4 * (64 + 64) + 2 * 4 * (64 + 32)
    assert all(float(v["lora_b"].abs().max()) == 0 for v in [a["l0"]["self_attn"]["q_proj"]])
    assert torch.equal(lm.init(seed=0)["l1"]["self_attn"]["v_proj"]["lora_a"],
                       a["l1"]["self_attn"]["v_proj"]["lora_a"])            # keyed: same seed, same bits


# ----------------------------------------------------------------------------- LoRA against a hand written torch LoRA


def test_lora_loss_and_gradients_match_hand_written_torch_lora_in_float64():
    net = _net()
    p = net.init(seed=0)
    lm = lora.wrap(net, targets=("q_proj", "v_proj", "down_proj"), rank=4, alpha=8.0)
    a = _random_b(lm, lm.init(seed=1))
    base = lm.prepare_base(p)
    loss = _loss_fn(lm)
    ids, y = _batch()
    val, g = truestar.grad(loss, wrt="a")(a, base, ids, y)

    m = _hf(net)
    _load_tree(m, p)
    mods = _hf_lora(m, lm, a)
    ref = torch.nn.functional.cross_entropy(m(ids).logits.reshape(-1, V), y.reshape(-1))
    ref.backward()
    assert abs(float(val) - float(ref)) <= 1e-13 * abs(float(ref))
    for path, mod in mods.items():
        node = g
        for k in path.split("."):
            node = node[k]
        assert _rel(node["lora_a"], mod.a.grad) < 1e-11, path
        assert _rel(node["lora_b"], mod.b.grad) < 1e-11, path
    # the base tree is an input of the step, never a trained value: only adapters come back
    assert {k for k, _ in lora._leaves(g)} == {k for k, _ in lora._leaves(lm.spec)}


def test_adapters_start_as_the_base_model():
    net = _net()
    p = net.init(seed=0)
    lm = lora.wrap(net, rank=4)
    ids, _ = _batch()
    assert torch.equal(lm(p, lm.init(seed=0), ids), net(p, ids))


# ----------------------------------------------------------------------------- merge


@pytest.mark.parametrize("quant", [None, "int4:g32", "nvfp4"])
def test_merged_forward_equals_adapter_forward_in_float64(quant):
    net = _net()
    p = net.init(seed=0)
    lm = lora.wrap(net, targets=("q_proj", "v_proj", "gate_proj"), rank=8, alpha=16.0, quant=quant)
    base = lm.prepare_base(p)
    a = _random_b(lm, lm.init(seed=2))
    ids, _ = _batch(1)
    err = lora.verify_merge(lm, base, a, ids)
    assert err < 1e-13
    merged = lm.merge(base, a)
    assert {k for k, _ in lora._leaves(merged)} == {k for k, _ in lora._leaves(p)}            # a plain tree for the original model
    w = merged["l0"]["self_attn"]["q_proj"]["w"]
    ad = a["l0"]["self_attn"]["q_proj"]
    w0 = p["l0"]["self_attn"]["q_proj"]["w"] if quant is None else \
        lora.dequantize_leaf(base["l0"]["self_attn"]["q_proj"]["qw"], quant, (D, D))
    assert torch.equal(w, w0.to(F64) + 2.0 * (ad["lora_b"] @ ad["lora_a"]))
    # a wrong merge is caught: drop the scale
    bad = dict(lm.adapted)
    lin = bad["l0.self_attn.q_proj"]
    import dataclasses
    lm.adapted["l0.self_attn.q_proj"] = dataclasses.replace(lin, alpha=lin.alpha * 2)
    try:
        with pytest.raises(lora.LoRAError, match="TS1374"):
            lora.verify_merge(lm, base, a, ids)
    finally:
        lm.adapted["l0.self_attn.q_proj"] = lin


# ----------------------------------------------------------------------------- QLoRA


@pytest.mark.parametrize("quant", ["int8", "int4:g32", "int4:g64", "nvfp4", "mxfp4"])
def test_qlora_gradients_match_float64_of_the_dequantized_model(quant):
    net = _net()
    p = net.init(seed=0)
    lm = lora.wrap(net, targets=("q_proj", "v_proj"), rank=4, alpha=8.0, quant=quant)
    base = lm.prepare_base(p)
    assert len(lm.quantized) == 14                                       # every Linear but the head
    assert lora.verify_decode(lm, base) == 14                            # decoder == format, bit for bit
    a = _random_b(lm, lm.init(seed=1))
    loss = _loss_fn(lm)
    ids, y = _batch(2)
    val, g = truestar.grad(loss, wrt="a")(a, base, ids, y)

    m = _hf(net)
    _load_tree(m, lm.dequantized_base(base))                             # exact float64 of the formats
    mods = _hf_lora(m, lm, a)
    ref = torch.nn.functional.cross_entropy(m(ids).logits.reshape(-1, V), y.reshape(-1))
    ref.backward()
    assert abs(float(val) - float(ref)) <= 1e-13 * abs(float(ref))
    for path, mod in mods.items():
        node = g
        for k in path.split("."):
            node = node[k]
        assert _rel(node["lora_a"], mod.a.grad) < 1e-11, path
        assert _rel(node["lora_b"], mod.b.grad) < 1e-11, path
    # the quantized base really is smaller
    assert lora.tree_bytes(base) < 0.5 * lora.tree_bytes(p)


def test_decode_check_catches_a_wrong_table():
    net = _net()
    lm = lora.wrap(net, quant="int4:g32", rank=2)
    base = lm.prepare_base(net.init(seed=0))
    q = base["l0"]["self_attn"]["q_proj"]["qw"]
    q["lut_hi"] = q["lut_lo"].clone()                                   # a nibble order bug
    with pytest.raises(lora.LoRAError, match="TS1375.*l0.self_attn.q_proj"):
        lora.verify_decode(lm, base)


# ----------------------------------------------------------------------------- training


def test_training_updates_adapters_only_loss_goes_down_and_replays_bitwise():
    net = _net()
    p = net.init(seed=0)
    lm = lora.wrap(net, targets=("q_proj", "v_proj"), rank=4, alpha=8.0, dropout=0.1)
    base = lm.prepare_base(p)
    loss = _loss_fn(lm)
    assert loss.specialize(training=False).program.op_counts().get("random", 0) == 0
    assert loss.specialize(training=True).program.op_counts().get("random", 0) == 4   # one mask per adapter
    ids, y = _batch(3, B=4, T=16)
    ids = ids % 8                                                       # a learnable task: next of a short cycle
    ids = (torch.arange(16)[None, :] + torch.arange(4)[:, None]) % 8
    y = torch.roll(ids, -1, 1)

    def run(n):
        step = truestar.train_step(loss, optim.AdamW(lr=3e-2), params="a", static={"training": True})
        ts = step.init(lm.init(seed=0))
        hist = []
        for _ in range(n):
            ts, r = step(ts, base, ids, y)
            hist.append(r.loss)
        return step, ts, hist

    step, ts, h1 = run(12)
    _, ts2, h2 = run(12)
    assert h1[-1] < 0.8 * h1[0]
    lora.assert_bitwise(ts.params, ts2.params)
    assert h1 == h2
    assert torch.equal(base["l0"]["self_attn"]["q_proj"]["w"], p["l0"]["self_attn"]["q_proj"]["w"])
    assert {k for k, _ in lora._leaves(ts.params)} == {k for k, _ in lora._leaves(lm.spec)}  # only adapters are carried
    w = ts2.params["l0"]["self_attn"]["q_proj"]["lora_a"]
    other = lora._set_tree(ts2.params, "l0.self_attn.q_proj.lora_a", torch.nextafter(w, w + 1))
    with pytest.raises(lora.LoRAError, match="TS1377.*q_proj.lora_a"):
        lora.assert_bitwise(ts.params, other)


def test_adapter_file_round_trip_and_refusals(tmp_path):
    net = _net(dtype="f32")
    lm = lora.wrap(net, rank=4, alpha=8.0)
    a = _random_b(lm, lm.init(seed=0))
    f = lora.save_adapters(str(tmp_path / "a.safetensors"), lm, a)
    b = lora.load_adapters(f, lm)
    lora.assert_bitwise(a, b)
    assert os.path.getsize(f) < 20_000                                   # adapters only, not the 0.3 MB base
    with pytest.raises(lora.LoRAError, match="TS1373.*rank"):
        lora.load_adapters(f, lora.wrap(net, rank=8, alpha=8.0))
    with pytest.raises(lora.LoRAError, match="TS1373.*layout"):
        lora.load_adapters(f, lora.wrap(Llama(V, D, 2, 4, 2, 96), rank=4, alpha=8.0))
    with pytest.raises(lora.LoRAError, match="TS1372.*missing"):
        lm.merge(net.init(seed=0), {})


def test_codes_are_registered():
    for c in ["TS1370", "TS1371", "TS1372", "TS1373", "TS1374", "TS1375", "TS1376", "TS1377"]:
        assert codes.slug(c).startswith(("lora_", "qlora_"))
