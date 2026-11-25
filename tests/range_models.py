"""Models for tests/test_rangecheck.py: two real float16 overflow cases rebuilt with synthetic weights,
and clean models. Nothing is downloaded; every weight is seeded.

* ``bge_small_like``: the bge small en v1.5 architecture (BertModel, hidden 384, 12 layers, 12 heads,
  FFN 1536; CLS pooling then L2 normalize). onnxruntime#31626 / PR #32294 measured on the real fp32
  activations that the FFN output LayerNorms of layers 0, 3 and 4 see |x - mean| of 256 to 258, 304 to
  321 and 297 to 304, so Pow(x - mean, 2) passes 65,504 in a decomposed float16 graph and the WebGPU
  run reaches cosine 0.63 to 0.83 against fp32. Here a seeded outlier channel on those three FFN
  outputs (BERT's known outlier dims) is scaled to the same peaks: 258, 318 and 302.
* ``gemma3_like``: a Gemma 3 style decoder (embedding times sqrt(hidden), RMS norms with
  pre and post norms around attention and the MLP, QK norm, GQA with one KV head, GeGLU with tanh
  gelu, tied lm head). onnxruntime#26732: the float16 residual stream overflows; the Hub workaround
  is 18 hand inserted Clip nodes at +-32752. Here seeded post feedforward norm weights on two
  channels make the residual stream grow by about 9,000 per layer and peak near 1.08e5 (over 65,504 from layer 10 on), as reported by users of
  the real model (activations reaching infinity in float16, Unsloth's Gemma 3 notes).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BGE_OUTLIER_DIMS = (308, 77)
BGE_PEAKS = {0: 258.0, 3: 300.0, 4: 290.0}  # layer index: bias scale; measured peaks in the docs


class BgeEmbed(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids):
        h = self.bert(input_ids=input_ids).last_hidden_state
        return F.normalize(h[:, 0], dim=-1)


def bge_small_like(outliers: bool = True, seed: int = 0, layers: int = 12, vocab: int = 1000):
    import pytest

    _tf = pytest.importorskip("transformers")
    BertConfig, BertModel = _tf.BertConfig, _tf.BertModel

    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=vocab, hidden_size=384, num_hidden_layers=layers, num_attention_heads=12,
                     intermediate_size=1536, max_position_embeddings=128, attn_implementation="eager")
    bert = BertModel(cfg, add_pooling_layer=False).eval()
    _trained_norms(bert, seed)
    if outliers:
        g = torch.Generator().manual_seed(seed + 1)
        for li, peak in BGE_PEAKS.items():
            if li >= layers:
                continue
            dense = bert.encoder.layer[li].output.dense
            with torch.no_grad():
                for j, d in enumerate(BGE_OUTLIER_DIMS):
                    # a large bias plus a token dependent part: |x - mean| on that dim reaches ``peak``
                    sign = -1.0 if j == 0 else 1.0
                    dense.bias[d] = sign * peak * (0.99 if j == 0 else 0.55)
                    dense.weight[d] = torch.randn(1536, generator=g) * 0.6
    return BgeEmbed(bert).eval()


def _trained_norms(model, seed):
    """LayerNorm weights and biases away from 1 and 0, as in a trained model (an all zero bias makes
    rows that a norm zeroes look constant, which a real model never has)."""
    g = torch.Generator().manual_seed(seed + 3)
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, nn.LayerNorm):
                mod.weight.copy_(1.0 + 0.1 * torch.randn(mod.weight.shape, generator=g))
                mod.bias.copy_(0.05 * torch.randn(mod.bias.shape, generator=g))


def bge_inputs(batch: int = 4, seq: int = 12, vocab: int = 1000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab, (batch, seq), generator=g)
    ids[:, 0] = 0  # CLS
    return ids


def calibrate_peak(model, ids):
    """Max |x - mean| at the FFN output LayerNorm input of each layer, float64 (for the docs)."""
    m = model.double()
    peaks = {}
    hooks = []
    for i, layer in enumerate(m.bert.encoder.layer):
        def pre(mod, args, i=i):
            x = args[0]
            peaks[i] = float((x - x.mean(-1, keepdim=True)).abs().max())
        hooks.append(layer.output.LayerNorm.register_forward_pre_hook(pre))
    with torch.no_grad():
        m(ids)
    for h in hooks:
        h.remove()
    m.float()
    return peaks


# ----------------------------------------------------------------------------- Gemma 3 style decoder


class GemmaRMSNorm(nn.Module):
    """Gemma's (1 + w) RMS norm, written with F.rms_norm: a fused op whose sum stays in float32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), weight=1.0 + self.weight, eps=self.eps)


def _rotate(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class GemmaAttention(nn.Module):
    def __init__(self, d, heads, kv_heads, head_dim):
        super().__init__()
        self.h, self.kv, self.hd = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(d, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, d, bias=False)
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q = q * cos + _rotate(q) * sin
        k = k * cos + _rotate(k) * sin
        k = k.repeat_interleave(self.h // self.kv, dim=1)
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, self.h * self.hd))


class GemmaMLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaBlock(nn.Module):
    def __init__(self, d, ff, heads, kv_heads, head_dim):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(d)
        self.self_attn = GemmaAttention(d, heads, kv_heads, head_dim)
        self.post_attention_layernorm = GemmaRMSNorm(d)
        self.pre_feedforward_layernorm = GemmaRMSNorm(d)
        self.mlp = GemmaMLP(d, ff)
        self.post_feedforward_layernorm = GemmaRMSNorm(d)

    def forward(self, x, cos, sin):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x


class Gemma3Like(nn.Module):
    def __init__(self, vocab=2048, d=640, layers=18, ff=2048, heads=4, kv_heads=1, head_dim=128, max_len=64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(GemmaBlock(d, ff, heads, kv_heads, head_dim) for _ in range(layers))
        self.norm = GemmaRMSNorm(d)
        self.d = d
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        ang = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
        ang = torch.cat((ang, ang), dim=-1)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, ids):
        T = ids.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        x = self.embed_tokens(ids) * math.sqrt(self.d)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x @ self.embed_tokens.weight.t()


GEMMA_OUTLIER_DIMS = (11, 403)
GEMMA_STEP = 520.0  # post feedforward norm weight on the first outlier dim


def gemma3_like(outliers: bool = True, seed: int = 0, layers: int = 18, d: int = 640, vocab: int = 2048):
    torch.manual_seed(seed)
    m = Gemma3Like(vocab=vocab, d=d, layers=layers, ff=3 * d)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.02)
        m.embed_tokens.weight.normal_(0, 1.0 / math.sqrt(d))
        g = torch.Generator().manual_seed(seed + 7)
        for blk in m.layers:
            for nm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                       "post_feedforward_layernorm"):
                getattr(blk, nm).weight.normal_(0, 0.1, generator=g)
        if outliers:
            for i, blk in enumerate(m.layers):
                # massive activations growing through the middle layers with a stable sign: with
                # up_proj equal to gate_proj the GeGLU product gelu(g) * g is never negative, so a
                # positive down projection row feeds the two dims a positive value every layer, and a
                # large post feedforward norm weight turns it into a fixed step of the residual
                if 3 <= i < layers - 3:
                    blk.mlp.up_proj.weight.copy_(blk.mlp.gate_proj.weight)
                    w = blk.post_feedforward_layernorm.weight
                    for j, c in enumerate(GEMMA_OUTLIER_DIMS):
                        w[c] = GEMMA_STEP if j == 0 else 0.7 * GEMMA_STEP
                        blk.mlp.down_proj.weight[c] = blk.mlp.down_proj.weight[c].abs() * 4.0
    return m.eval()


def gemma_inputs(batch: int = 2, seq: int = 16, vocab: int = 2048, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=g)


# ----------------------------------------------------------------------------- clean models


def resnet_calibrated(name: str = "resnet50", seed: int = 0, size: int = 96):
    """torchvision ResNet with seeded weights and batch norm statistics set from a calibration batch,
    so activations are unit scale as in a trained network (as docs/NPU.md does)."""
    import torchvision

    torch.manual_seed(seed)
    m = getattr(torchvision.models, name)(weights=None, num_classes=100)
    x = torch.randn(16, 3, size, size)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = None
            mod.reset_running_stats()
    m.train()
    with torch.no_grad():
        m(x)
    m.eval()
    with torch.no_grad():
        logits = m(x)
        m.fc.weight.mul_(2.5 / float(logits.std()))
        m.fc.bias.zero_()
    return m


def bert_base(seed: int = 0, layers: int = 12, vocab: int = 1000):
    import pytest

    _tf = pytest.importorskip("transformers")
    BertConfig, BertModel = _tf.BertConfig, _tf.BertModel

    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=vocab, hidden_size=768, num_hidden_layers=layers, num_attention_heads=12,
                     intermediate_size=3072, max_position_embeddings=128, attn_implementation="eager")
    bert = BertModel(cfg, add_pooling_layer=False).eval()
    _trained_norms(bert, seed)
    return BgeEmbed(bert).eval()
