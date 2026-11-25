"""A tiny Llama style decoder in PyTorch (Hugging Face names and conventions) and a converter that
writes it to GGUF the way a conversion script does, with switches that inject the conversion bugs
people hit: a wrong rope theta, a wrong norm eps, a transposed weight, q and k left unpermuted.

Everything is built in code; nothing is downloaded. Sizes are multiples of 256 so the engine's K
quant types apply to every matrix.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from truestar.engines import gguf

DIM, LAYERS, HEADS, KV_HEADS, FFN = 256, 2, 4, 2, 512
HEAD_DIM = DIM // HEADS

# vocabulary: 3 specials, 256 byte tokens, then word pieces (every prefix of each word, so the
# score ordered pair merging of a sentencepiece style tokenizer reaches the whole word)
WORDS = ["the", "cat", "sat", "on", "mat", "dog", "ran", "to", "a", "red", "big", "and"]
SPACE = "▁"


def vocab() -> List[str]:
    toks = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)] + [SPACE]
    toks += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    seen = set(toks)
    for w in WORDS:
        p = SPACE + w
        for k in range(2, len(p) + 1):
            if p[:k] not in seen:
                toks.append(p[:k])
                seen.add(p[:k])
    while len(toks) % 64:
        toks.append(f"<unused{len(toks)}>")
    return toks


VOCAB = vocab()
V = len(VOCAB)


def token_types() -> List[int]:
    out = []
    for t in VOCAB:
        if t == "<unk>":
            out.append(2)
        elif t in ("<s>", "</s>"):
            out.append(3)
        elif t.startswith("<0x"):
            out.append(6)
        elif t.startswith("<unused"):
            out.append(5)
        else:
            out.append(1)
    return out


def scores() -> List[float]:
    return [0.0 if (t.startswith("<") and len(t) > 1) else float(len(t)) - 10.0 for t in VOCAB]


def reference_tokenize(text: str, add_bos: bool = True) -> List[int]:
    """An independent sentencepiece style tokenizer for this vocabulary: prefix a space, replace
    spaces with the marker, merge the adjacent pair whose joined piece scores highest until none
    joins. Used as the reference's own tokenizer in the tests."""
    idx = {t: i for i, t in enumerate(VOCAB)}
    sc = scores()
    s = SPACE + text.replace(" ", SPACE)
    pieces = list(s)
    while True:
        best, at = None, -1
        for i in range(len(pieces) - 1):
            j = pieces[i] + pieces[i + 1]
            if j in idx and idx[j] >= 3 and (best is None or sc[idx[j]] > best):
                best, at = sc[idx[j]], i
        if at < 0:
            break
        pieces[at:at + 2] = [pieces[at] + pieces[at + 1]]
    ids = [idx[p] if p in idx else idx[f"<0x{ord(p):02X}>"] for p in pieces]
    return ([1] if add_bos else []) + ids


PROMPT = "the big red cat sat on the mat and the dog ran to a cat"


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(DIM, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(HEADS * HEAD_DIM, DIM, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KV_HEADS, HEAD_DIM).transpose(1, 2)
        inv = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
        scale = (self.cfg.rope_scaling or {}).get("factor", 1.0)
        ang = (torch.arange(T, dtype=torch.float64) / scale)[:, None] * inv[None]
        emb = torch.cat([ang, ang], dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = HEADS // KV_HEADS
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        s = q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        o = torch.softmax(s, dim=-1) @ v
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(DIM, FFN, bias=False)
        self.up_proj = nn.Linear(DIM, FFN, bias=False)
        self.down_proj = nn.Linear(FFN, DIM, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(DIM, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(DIM, cfg.rms_norm_eps)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Inner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, DIM)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(LAYERS)])
        self.norm = RMSNorm(DIM, cfg.rms_norm_eps)


class TinyLlama(nn.Module):
    """Returns logits [B, T, V]."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = Inner(cfg)
        self.lm_head = nn.Linear(DIM, V, bias=False)

    def forward(self, ids):
        x = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


def config(theta=10000.0, eps=1e-5, rope_scaling=None):
    return SimpleNamespace(vocab_size=V, hidden_size=DIM, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                           num_key_value_heads=KV_HEADS, intermediate_size=FFN, rms_norm_eps=eps,
                           rope_theta=theta, rope_scaling=rope_scaling, head_dim=HEAD_DIM,
                           tie_word_embeddings=False, max_position_embeddings=512)


def build(seed=0, **cfg_kw) -> TinyLlama:
    torch.manual_seed(seed)
    m = TinyLlama(config(**cfg_kw))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(0.0, 1.0 / math.sqrt(p.shape[1]))
            else:  # trained looking norm weights, not all ones
                p.uniform_(0.6, 1.4)
        m.model.embed_tokens.weight.mul_(1.0)
        m.lm_head.weight.mul_(4.0)  # spread the logits so top 1 is not a coin toss
    return m.eval()


def permute_qk(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Reorder rows of a q or k weight so that rotating adjacent pairs (the engine's llama
    convention) equals rotating the two halves of each head (the reference's): new row 2i of a head
    is old row i, new row 2i + 1 is old row i + head_dim / 2."""
    out, inp = w.shape
    d = out // n_heads
    return w.reshape(n_heads, 2, d // 2, inp).transpose(1, 2).reshape(out, inp)


def to_gguf(model: TinyLlama, path: str, *, theta: Optional[float] = None, eps: Optional[float] = None,
            transpose: Sequence[str] = (), skip_permute: Sequence[str] = (), rope_scale: Optional[float] = None,
            add_bos: bool = True, float_type: str = "F32", heads: Optional[int] = None,
            scale_weight: Optional[Dict[str, float]] = None) -> str:
    """Write the model as a llama architecture GGUF. Fault switches:
    theta / eps: metadata value other than the model's; transpose: GGUF names whose 2 D weight is
    stored transposed; skip_permute: q or k names left in the reference row order;
    scale_weight: multiply a tensor (a converter that applies a factor twice)."""
    cfg = model.config
    L_, H_, KV_ = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D_ = cfg.hidden_size
    HD_ = getattr(cfg, "head_dim", None) or D_ // H_
    sd = {k: v.detach().to(torch.float64) for k, v in model.state_dict().items()}
    names = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
             "lm_head.weight": "output.weight"}
    for i in range(L_):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        names.update({p + "input_layernorm.weight": b + "attn_norm.weight",
                      p + "self_attn.q_proj.weight": b + "attn_q.weight",
                      p + "self_attn.k_proj.weight": b + "attn_k.weight",
                      p + "self_attn.v_proj.weight": b + "attn_v.weight",
                      p + "self_attn.o_proj.weight": b + "attn_output.weight",
                      p + "post_attention_layernorm.weight": b + "ffn_norm.weight",
                      p + "mlp.gate_proj.weight": b + "ffn_gate.weight",
                      p + "mlp.up_proj.weight": b + "ffn_up.weight",
                      p + "mlp.down_proj.weight": b + "ffn_down.weight"})
    tensors = []
    for hf, gn in names.items():
        w = sd[hf]
        if gn.endswith("attn_q.weight") and gn not in skip_permute:
            w = permute_qk(w, H_)
        if gn.endswith("attn_k.weight") and gn not in skip_permute:
            w = permute_qk(w, KV_)
        if gn in transpose:
            w = w.T.contiguous()
        if scale_weight and gn in scale_weight:
            w = w * scale_weight[gn]
        ft = float_type if w.ndim == 2 else "F32"
        tensors.append(gguf.float_tensor(gn, w.numpy(), ft))
    md = {
        "general.architecture": "llama",
        "general.name": "truestar tiny llama",
        "general.file_type": gguf.Meta(0 if float_type == "F32" else 1, gguf.U32),
        "llama.context_length": gguf.Meta(cfg.max_position_embeddings, gguf.U32),
        "llama.embedding_length": gguf.Meta(D_, gguf.U32),
        "llama.block_count": gguf.Meta(L_, gguf.U32),
        "llama.feed_forward_length": gguf.Meta(cfg.intermediate_size, gguf.U32),
        "llama.attention.head_count": gguf.Meta(heads or H_, gguf.U32),
        "llama.attention.head_count_kv": gguf.Meta(KV_, gguf.U32),
        "llama.rope.dimension_count": gguf.Meta(HD_, gguf.U32),
        "llama.rope.freq_base": gguf.Meta(float(theta if theta is not None else cfg.rope_theta), gguf.F32),
        "llama.attention.layer_norm_rms_epsilon": gguf.Meta(float(eps if eps is not None else cfg.rms_norm_eps), gguf.F32),
        "llama.vocab_size": gguf.Meta(cfg.vocab_size, gguf.U32),
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": gguf.Meta(VOCAB, gguf.ARRAY, gguf.STRING),
        "tokenizer.ggml.scores": gguf.Meta(scores(), gguf.ARRAY, gguf.F32),
        "tokenizer.ggml.token_type": gguf.Meta(token_types(), gguf.ARRAY, gguf.I32),
        "tokenizer.ggml.bos_token_id": gguf.Meta(1, gguf.U32),
        "tokenizer.ggml.eos_token_id": gguf.Meta(2, gguf.U32),
        "tokenizer.ggml.unknown_token_id": gguf.Meta(0, gguf.U32),
        "tokenizer.ggml.add_bos_token": add_bos,
        "tokenizer.ggml.add_eos_token": False,
    }
    rs = rope_scale if rope_scale is not None else (cfg.rope_scaling or {}).get("factor")
    if rs is not None and rs != 1.0:
        md["llama.rope.scaling.type"] = "linear"
        md["llama.rope.scaling.factor"] = gguf.Meta(float(rs), gguf.F32)
    gguf.write(path, md, tensors)
    return path
