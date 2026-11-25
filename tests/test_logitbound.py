"""truestar.logitbound: attention logit bounds from the weights, a constructed input that reaches
them through the real modules, and refusal before an fp16 or fp8 overflow (docs/TRAIN_NUMERICS.md)."""

import math

import pytest
import torch

from truestar import logitbound as L

F64 = torch.float64


def _encoder(scale=1.0, layers=2, seed=0):
    torch.manual_seed(seed)
    layer = torch.nn.TransformerEncoderLayer(64, 4, 128, norm_first=True, batch_first=True, dropout=0.0)
    enc = torch.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    with torch.no_grad():
        for p in enc.layers[0].self_attn.parameters():
            p.mul_(scale)
    return enc.eval()


def _measured_logits(layer, x):
    """Every logit of the layer's self attention for inputs x [B, T, 64], in float64."""
    mha = layer.self_attn
    E, H = mha.embed_dim, mha.num_heads
    with torch.no_grad():
        y = layer.norm1(x).double()
        w, b = mha.in_proj_weight.double(), mha.in_proj_bias.double()
        q = (y @ w[:E].T + b[:E]).view(*y.shape[:2], H, E // H).transpose(1, 2)
        k = (y @ w[E:2 * E].T + b[E:2 * E]).view(*y.shape[:2], H, E // H).transpose(1, 2)
        return q @ k.transpose(-1, -2) / math.sqrt(E // H)


def test_bound_holds_and_is_reached_within_a_few_percent():
    enc = _encoder(seed=1)
    rep = L.check_module(enc, "fp16")
    assert rep.verdict == "safe" and rep.issues()[0].code == "TS1051"
    layer0 = [h for h in rep.heads if h.layer == "layers.0"]
    x = torch.randn(64, 32, 64) * 5
    s = _measured_logits(enc.layers[0], x).abs().amax(dim=(0, 2, 3))
    for h in layer0:
        assert float(s[h.head]) <= h.upper * (1 + 1e-9)
        assert h.reachable <= h.upper * (1 + 1e-6)
        assert h.gap < 1.1, h.line("fp16")


def test_fp16_overflow_is_refused_and_really_happens():
    enc = _encoder(scale=80.0)
    rep = L.check_module(enc, "fp16")
    assert rep.verdict == "overflows"
    with pytest.raises(L.LogitOverflowError) as e:
        L.guard(enc, "fp16")
    assert e.value.code == "TS1053"
    worst = max((h for h in rep.heads if h.layer == "layers.0"), key=lambda h: h.reachable)
    site = L.sites_of(enc)[0]
    dh = 16
    sl = slice(worst.head * dh, (worst.head + 1) * dh)
    zq, zk = L.adversarial_pair(site.wq[sl], site.wk[sl], "layernorm", site.gamma, site.beta, site.bq[sl], site.bk[sl])
    # the constructed pair, run in float16 scores: the logit is infinite
    s = _measured_logits(enc.layers[0], torch.stack([zq, zk]).float()[None])
    assert float(s[0, worst.head, 0, 1].abs()) > 65504
    assert torch.isinf(s[0, worst.head, 0, 1].half())


def test_may_overflow_warns_only():
    enc = _encoder(scale=1.0)
    rep = L.check_module(enc, "fp16")
    h = max(rep.heads, key=lambda h: h.upper)
    assert h.reachable < h.upper
    # a target whose limit lies between the reachable value and the upper bound of the worst head
    L.LIMITS["test_mid"] = 0.5 * (h.reachable + h.upper)
    try:
        r = L.check_module(enc, "test_mid", entries=False)
        assert r.verdict == "may_overflow" and r.issues()[0].code == "TS1052"
        with pytest.warns(RuntimeWarning, match="TS1052"):
            L.guard(enc, "test_mid", entries=False)
    finally:
        del L.LIMITS["test_mid"]


def test_fp8_entries_and_logits():
    enc = _encoder(scale=6.0)
    rep = L.check_module(enc, "fp8_e4m3")
    assert rep.verdict == "overflows"
    assert max(h.reachable for h in rep.heads) > 448


def test_hf_llama_rope_is_bounded_by_the_product_form():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=100)
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg).eval()
    rep = L.check_module(m, "fp16")
    assert {h.method for h in rep.heads} == {"product (rope)"}
    assert len(rep.heads) == 8 and rep.verdict == "safe"
    for h in rep.heads:
        assert 0 < h.reachable <= h.upper and h.gap < 2.0


def test_qk_norm_and_softcap_sites():
    torch.manual_seed(2)
    d, H, dh = 64, 4, 16
    site = L.AttnSite("qk", torch.randn(H * dh, d), torch.randn(H * dh, d), H, H, norm="rmsnorm",
                      gamma=torch.ones(d), q_norm=torch.full((dh,), 2.0), k_norm=torch.full((dh,), 3.0))
    rep = L.check_sites([site], "fp16")
    # scale * max|g_q| * max|g_k| * d_h = 0.25 * 2 * 3 * 16
    assert all(abs(h.upper - 24.0) < 1e-9 and h.method == "qk norm" for h in rep.heads)
    site = L.AttnSite("cap", torch.randn(H * dh, d) * 30, torch.randn(H * dh, d) * 30, H, H, norm="rmsnorm",
                      gamma=torch.ones(d), softcap=50.0)
    rep = L.check_sites([site], "fp16", entries=False)
    assert all(h.upper == 50.0 for h in rep.heads) and rep.verdict == "safe"


def test_no_norm_needs_an_input_bound_and_projection():
    torch.manual_seed(3)
    layer = torch.nn.TransformerEncoderLayer(32, 2, 64, norm_first=False, batch_first=True)
    with pytest.raises(ValueError, match="input bound"):
        L.check_module(layer, "fp16")
    rep = L.check_module(layer, "fp16", input_bound=10.0)
    assert all(h.reachable == 0.0 and h.input_bound == 10.0 for h in rep.heads)
    hist = [(s, 10.0 * 1.01 ** s) for s in range(0, 80, 10)]
    left = L.steps_to_limit(hist, "fp16")
    assert left is not None and abs(70 + left - math.log(65504 / 10) / math.log(1.01)) < 1.0
    assert L.steps_to_limit([(0, 5.0), (10, 4.0)], "fp16") is None
