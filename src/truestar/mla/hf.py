"""Hugging Face DeepSeek V3 (transformers 5.x) to ``MLAModel``, and a float64 clean copy of the
Hugging Face model for checks.

Lowered: ``DeepseekV3ForCausalLM`` whose layers all have a dense MLP (``first_k_dense_replace`` at
least the layer count) and ``DeepseekV3Attention``. Refused before anything is built, by stable
code: mixture of experts layers (TS940) and token sparse attention selection (TS1202, DeepSeek V3.2
and V4 indexers) through ``truestar.ir.refusals``; every MLA variant this lowering was not written
for (TS1200), named with the reason. Glm4MoeLiteAttention and YoutuAttention have the same
``forward`` source as DeepseekV3Attention in transformers 5.24.1 but were not run, so they are
refused too.
"""

from __future__ import annotations

import copy
import types
from typing import Optional

import torch
from torch import nn

from .model import MLAConfig, MLAModel, rope_permutation

LOWERED_ATTENTION = ("DeepseekV3Attention",)
STATIC_ROPE = ("default", "linear", "yarn", "llama3")   # inv_freq does not depend on the sequence length


class MLAUnsupported(ValueError):
    """TS1200: an MLA model this lowering does not handle; the message names the layer and why."""
    code = "TS1200"

    def __init__(self, what: str):
        super().__init__(f"[TS1200] truestar.mla cannot lower {what}; nothing was built.")


def _check(hf) -> None:
    from ..ir.refusals import check_module

    check_module(hf, type(hf).__name__)      # TS940 MoE, TS1202 token sparse selection, ...
    hc = hf.config
    if getattr(hf, "model", None) is None or not hasattr(hf.model, "layers"):
        raise MLAUnsupported(f"{type(hf).__name__}: expected a causal LM with .model.layers")
    for i, layer in enumerate(hf.model.layers):
        at = getattr(layer, "self_attn", None)
        name = type(at).__name__
        where = f"model.layers.{i}.self_attn ({name})"
        if name not in LOWERED_ATTENTION:
            why = ("an MLA variant whose code differs from DeepseekV3Attention (not verified here)"
                   if hasattr(at, "kv_b_proj") else "not a multi head latent attention layer")
            raise MLAUnsupported(f"{where}: {why}")
        if at.kv_b_proj.bias is not None:
            raise MLAUnsupported(f"{where}: kv_b_proj has a bias, which the absorbed form does not carry")
        mlp = getattr(layer, "mlp", None)
        if not all(hasattr(mlp, n) for n in ("gate_proj", "up_proj", "down_proj")):
            raise MLAUnsupported(f"model.layers.{i}.mlp ({type(mlp).__name__}): not a dense SwiGLU MLP")
        if at.q_a_layernorm is not None and at.q_a_layernorm.variance_epsilon != at.kv_a_layernorm.variance_epsilon:
            raise MLAUnsupported(f"{where}: the q_a and kv_a norms use different epsilons")
    if hc.num_key_value_heads != hc.num_attention_heads:
        raise MLAUnsupported(f"num_key_value_heads={hc.num_key_value_heads} differs from num_attention_heads="
                             f"{hc.num_attention_heads} (MLA expands one latent into every head)")
    if getattr(hc, "hidden_act", "silu") != "silu":
        raise MLAUnsupported(f"hidden_act={hc.hidden_act!r} (the lowering has SiLU)")
    rope_type = (getattr(hc, "rope_parameters", None) or {}).get("rope_type", "default")
    if rope_type not in STATIC_ROPE:
        raise MLAUnsupported(f"rope_type={rope_type!r}: its frequencies change with the sequence length "
                             f"(static types: {', '.join(STATIC_ROPE)})")


def from_hf(hf) -> MLAModel:
    """An ``MLAModel`` with the weights of a Hugging Face ``DeepseekV3ForCausalLM`` (dense layers);
    interleaved RoPE is folded into the weights as a row permutation (exact)."""
    _check(hf)
    hc = hf.config
    at0 = hf.model.layers[0].self_attn
    rot = hf.model.rotary_emb
    c = MLAConfig(vocab=hc.vocab_size, dim=hc.hidden_size, layers=hc.num_hidden_layers, heads=hc.num_attention_heads,
                  q_rank=hc.q_lora_rank, kv_rank=hc.kv_lora_rank, nope=hc.qk_nope_head_dim, rope=hc.qk_rope_head_dim,
                  v_dim=hc.v_head_dim, ffn=hf.model.layers[0].mlp.gate_proj.out_features,
                  eps=hf.model.layers[0].input_layernorm.variance_epsilon,
                  lat_eps=at0.kv_a_layernorm.variance_epsilon, scaling=float(at0.scaling),
                  rope_scale=float(rot.attention_scaling), bias=bool(hc.attention_bias))
    ffns = {layer.mlp.gate_proj.out_features for layer in hf.model.layers}
    if len(ffns) != 1:
        raise MLAUnsupported(f"MLP widths differ between layers ({sorted(ffns)})")
    m = MLAModel(c).eval().to(next(hf.parameters()).dtype)
    perm = rope_permutation(c.rope) if getattr(hc, "rope_interleave", False) else None

    def q_rows(w):            # [H * (dn + dr), in]: permute each head's rope rows
        if perm is None:
            return w
        w = w.view(c.heads, c.nope + c.rope, -1)
        return torch.cat((w[:, :c.nope], w[:, c.nope:][:, perm]), dim=1).reshape(c.heads * (c.nope + c.rope), -1)

    def kv_rows(w):           # [r + dr, ...]: permute the rope rows
        if perm is None:
            return w
        return torch.cat((w[:c.kv_rank], w[c.kv_rank:][perm]), dim=0)

    with torch.no_grad():
        m.embed.weight.copy_(hf.model.embed_tokens.weight)
        m.norm.copy_(hf.model.norm.weight)
        m.lm_head.weight.copy_(hf.lm_head.weight)
        m.inv_freq = rot.inv_freq.detach().clone().to(m.inv_freq.dtype)
        for blk, layer in zip(m.blocks, hf.model.layers):
            at, mlp = layer.self_attn, layer.mlp
            blk.norm1.copy_(layer.input_layernorm.weight)
            blk.norm2.copy_(layer.post_attention_layernorm.weight)
            if c.q_rank is None:
                blk.q.weight.copy_(q_rows(at.q_proj.weight))
            else:
                blk.q_a.weight.copy_(at.q_a_proj.weight)
                if c.bias:
                    blk.q_a.bias.copy_(at.q_a_proj.bias)
                blk.q_norm.copy_(at.q_a_layernorm.weight)
                blk.q_b.weight.copy_(q_rows(at.q_b_proj.weight))
            blk.kv_a.weight.copy_(kv_rows(at.kv_a_proj_with_mqa.weight))
            if c.bias:
                blk.kv_a.bias.copy_(kv_rows(at.kv_a_proj_with_mqa.bias))
                blk.o.bias.copy_(at.o_proj.bias)
            blk.kv_norm.copy_(at.kv_a_layernorm.weight)
            blk.kv_b.weight.copy_(at.kv_b_proj.weight)
            blk.o.weight.copy_(at.o_proj.weight)
            blk.gate.weight.copy_(mlp.gate_proj.weight)
            blk.up.weight.copy_(mlp.up_proj.weight)
            blk.down.weight.copy_(mlp.down_proj.weight)
    return m


class _Norm64(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight, self.eps = norm.weight, norm.variance_epsilon

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def _rope64(self, x, position_ids):
    f = position_ids.to(torch.float64)[:, :, None] * self.inv_freq.to(torch.float64)[None, None, :]
    e = torch.cat((f, f), dim=-1)
    return (e.cos() * self.attention_scaling).to(x.dtype), (e.sin() * self.attention_scaling).to(x.dtype)


def hf_float64(hf, attn: Optional[str] = "sdpa"):
    """A float64 copy of a Hugging Face DeepSeek V3 model that stays float64 inside: its RMSNorms and
    rotary tables otherwise round to float32 (``.to(torch.float32)``, ``.float()``), and its eager
    attention takes the softmax in float32. Used as the reference of the checks."""
    m = copy.deepcopy(hf).double().eval()
    for name, mod in list(m.named_modules()):
        if type(mod).__name__.endswith("RMSNorm"):
            parent, _, leaf = name.rpartition(".")
            setattr(m.get_submodule(parent) if parent else m, leaf, _Norm64(mod))
    m.model.rotary_emb.forward = types.MethodType(_rope64, m.model.rotary_emb)
    if attn is not None:
        m.config._attn_implementation = attn
    return m


__all__ = ["from_hf", "hf_float64", "MLAUnsupported", "LOWERED_ATTENTION"]
