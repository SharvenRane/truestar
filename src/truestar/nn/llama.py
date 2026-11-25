"""A Llama style decoder written with truestar.nn, and a one way importer of Hugging Face weights.

    from truestar.nn.llama import Llama, from_hf

    net = Llama(vocab=256, dim=64, depth=2, heads=4, kv_heads=2, ffn=128)
    p = net.init(seed=0)                           # random weights from the config, nothing downloaded
    net, p = from_hf(hf_model)                     # or a transformers LlamaForCausalLM's weights

The parameter tree uses the Hugging Face module names (``l0.self_attn.q_proj.w``), so a LoRA target
pattern written for a Hugging Face model (``q_proj``, ``v_proj``) selects the same layers here
(docs/LORA.md). Blocks are pre norm with RMSNorm, rotary positions in the rotate half layout,
grouped query attention, a SwiGLU MLP and an untied or tied output head.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..frontend import ops
from .core import Layer, layer
from .layers import Embedding, Linear, RMSNorm
from .layers_more import Rotary


@layer
class LlamaAttention(Layer):
    dim: int
    heads: int
    kv_heads: int
    base: float = 10000.0
    dtype: str = "f32"

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    def children(self):
        kv = self.kv_heads * self.head_dim
        return {"q_proj": Linear(self.dim, self.dim, False, self.dtype),
                "k_proj": Linear(self.dim, kv, False, self.dtype),
                "v_proj": Linear(self.dim, kv, False, self.dtype),
                "o_proj": Linear(self.dim, self.dim, False, self.dtype),
                "rope": Rotary(self.dim // self.heads, self.base)}

    def forward(self, p, x):
        B = x.shape[0]
        T = x.shape[1]
        hd = self.head_dim
        pos = ops.arange(T)
        q = ops.permute(ops.reshape(self.q_proj(p["q_proj"], x), [B, T, self.heads, hd]), [0, 2, 1, 3])
        k = ops.permute(ops.reshape(self.k_proj(p["k_proj"], x), [B, T, self.kv_heads, hd]), [0, 2, 1, 3])
        v = ops.permute(ops.reshape(self.v_proj(p["v_proj"], x), [B, T, self.kv_heads, hd]), [0, 2, 1, 3])
        q = self.rope(q, pos)
        k = self.rope(k, pos)
        rows = ops.reshape(ops.arange(T), [T, 1])
        cols = ops.reshape(ops.arange(T), [1, T])
        a = ops.attention(q, k, v, mask=cols <= rows, enable_gqa=self.kv_heads != self.heads)
        return self.o_proj(p["o_proj"], ops.reshape(ops.permute(a, [0, 2, 1, 3]), [B, T, self.dim]))


@layer
class LlamaMLP(Layer):
    dim: int
    ffn: int
    dtype: str = "f32"

    def children(self):
        return {"gate_proj": Linear(self.dim, self.ffn, False, self.dtype),
                "up_proj": Linear(self.dim, self.ffn, False, self.dtype),
                "down_proj": Linear(self.ffn, self.dim, False, self.dtype)}

    def forward(self, p, x):
        g = self.gate_proj(p["gate_proj"], x)
        return self.down_proj(p["down_proj"], ops.silu(g) * self.up_proj(p["up_proj"], x))


@layer
class LlamaBlock(Layer):
    dim: int
    heads: int
    kv_heads: int
    ffn: int
    eps: float = 1e-6
    base: float = 10000.0
    dtype: str = "f32"

    def children(self):
        return {"input_layernorm": RMSNorm(self.dim, self.eps, self.dtype),
                "self_attn": LlamaAttention(self.dim, self.heads, self.kv_heads, self.base, self.dtype),
                "post_attention_layernorm": RMSNorm(self.dim, self.eps, self.dtype),
                "mlp": LlamaMLP(self.dim, self.ffn, self.dtype)}

    def forward(self, p, x):
        h = x + self.self_attn(p["self_attn"], self.input_layernorm(p["input_layernorm"], x))
        return h + self.mlp(p["mlp"], self.post_attention_layernorm(p["post_attention_layernorm"], h))


@layer
class Llama(Layer):
    """``net(p, ids)``: ids int64 [B, T] to logits [B, T, vocab]."""

    vocab: int
    dim: int
    depth: int
    heads: int
    kv_heads: int
    ffn: int
    eps: float = 1e-6
    base: float = 10000.0
    tied: bool = False
    dtype: str = "f32"

    def children(self):
        out = {"embed_tokens": Embedding(self.vocab, self.dim, 0.02, self.dtype)}
        for i in range(self.depth):
            out[f"l{i}"] = LlamaBlock(self.dim, self.heads, self.kv_heads, self.ffn, self.eps, self.base, self.dtype)
        out["norm"] = RMSNorm(self.dim, self.eps, self.dtype)
        if not self.tied:
            out["lm_head"] = Linear(self.dim, self.vocab, False, self.dtype)
        return out

    def forward(self, p, ids):
        x = self.embed_tokens(p["embed_tokens"], ids)
        for i in range(self.depth):
            k = "l" + str(i)
            x = self.parts[k](p[k], x)
        x = self.norm(p["norm"], x)
        if self.tied:
            return ops.linear(x, p["embed_tokens"]["w"])
        return self.lm_head(p["lm_head"], x)


_DT = {torch.float32: "f32", torch.float64: "f64", torch.bfloat16: "bf16", torch.float16: "f16"}


def hf_rope(cfg):
    """(rope theta, rope scaling or None) of a Hugging Face config, on transformers 4.x
    (``rope_theta``, ``rope_scaling``) and 5.x (one ``rope_parameters`` dict, which ``rope_scaling``
    also returns, with the theta inside and type "default" when there is no scaling)."""
    rp = getattr(cfg, "rope_parameters", None)
    theta = getattr(cfg, "rope_theta", None)
    if theta is None and isinstance(rp, dict):
        theta = rp.get("rope_theta")
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict):
        rest = {k: v for k, v in rs.items() if k != "rope_theta"}
        rs = rest if rest.get("rope_type", rest.get("type", "default")) != "default" else None
    return float(10000.0 if theta is None else theta), rs


def from_hf(model, dtype: Optional[torch.dtype] = None) -> Tuple[Llama, Dict[str, Any]]:
    """The config and parameter tree of a transformers ``LlamaForCausalLM`` (or any model with its
    state dict names and a Llama config: Mistral without a sliding window, Qwen2 without biases is
    refused). Weights are copied, never shared. Refuses rope scaling and attention or MLP biases,
    which this decoder does not have."""
    cfg = model.config
    sd = model.state_dict()
    theta, rs = hf_rope(cfg)
    if rs:
        raise ValueError(f"from_hf: rope_scaling {rs} is not supported by truestar.nn.llama.Llama")
    if getattr(cfg, "attention_bias", False) or getattr(cfg, "mlp_bias", False):
        raise ValueError("from_hf: attention or MLP biases are not supported by truestar.nn.llama.Llama")
    if getattr(cfg, "hidden_act", "silu") != "silu":
        raise ValueError(f"from_hf: hidden_act {cfg.hidden_act!r}, only silu is supported")
    dt = dtype or next(iter(sd.values())).dtype
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    heads = cfg.num_attention_heads
    if getattr(cfg, "head_dim", cfg.hidden_size // heads) != cfg.hidden_size // heads:
        raise ValueError("from_hf: head_dim other than hidden_size / heads is not supported")
    net = Llama(cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, heads,
                getattr(cfg, "num_key_value_heads", None) or heads, cfg.intermediate_size, float(cfg.rms_norm_eps),
                theta, tied, _DT[dt])

    def w(name):
        return sd[name].detach().to(dt).clone()

    p: Dict[str, Any] = {"embed_tokens": {"w": w("model.embed_tokens.weight")}}
    for i in range(net.depth):
        pre = f"model.layers.{i}."
        p[f"l{i}"] = {
            "input_layernorm": {"w": w(pre + "input_layernorm.weight")},
            "self_attn": {n: {"w": w(pre + f"self_attn.{n}.weight")} for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
            "post_attention_layernorm": {"w": w(pre + "post_attention_layernorm.weight")},
            "mlp": {n: {"w": w(pre + f"mlp.{n}.weight")} for n in ("gate_proj", "up_proj", "down_proj")},
        }
    p["norm"] = {"w": w("model.norm.weight")}
    if not tied:
        p["lm_head"] = {"w": w("lm_head.weight")}
    return net, p
