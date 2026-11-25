"""A small Llama style decoder written three times, for the port check tests (P16).

* `RefLlama`: Hugging Face style names (model.layers.N.self_attn.q_proj ...), RMSNorm as a class,
  separate q, k, v, optional LoRA adapters on q_proj and v_proj (PEFT style base_layer, lora_A,
  lora_B).
* `PortA`: a hand port with other names (blocks.N.attn.wq ...), its own norm class, separate q, k, v.
* `PortB`: an engine style port: fused wqkv and w13 projections, RMSNorm written inline in the
  block with the norm weights as plain parameters.

Each port takes the classic port bugs as flags: eps, rotary pairing (`rope="interleaved"` without
permuting q and k rows rotates the wrong halves), rotary base, attention scale. `port_a(ref)` and
`port_b(ref)` copy the reference weights through an explicit mapping, like a converter script.
Plain torch.nn only, random weights, CPU.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

VOCAB, DIM, HEADS, LAYERS, HIDDEN = 64, 32, 4, 2, 48
HD = DIM // HEADS
EPS, BASE = 1e-5, 10000.0


def rope(S, hd, base, dtype, interleaved=False):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=dtype) / hd))
    t = torch.arange(S, dtype=dtype)
    freqs = torch.outer(t, inv)
    emb = freqs.repeat_interleave(2, dim=-1) if interleaved else torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rotate_pairs(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def attend(q, k, v, scale):
    S = q.shape[-2]
    scores = q @ k.transpose(-2, -1)
    if scale == "sqrt":
        scores = scores / math.sqrt(q.shape[-1])
    elif scale == "linear":
        scores = scores / q.shape[-1]  # the bug: missing square root
    mask = torch.ones(S, S, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


# ----------------------------------------------------------------------------- reference


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))


class LoRALinear(nn.Module):
    def __init__(self, d_in, d_out, r=4, scaling=0.5):
        super().__init__()
        self.base_layer = nn.Linear(d_in, d_out, bias=False)
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        nn.init.normal_(self.lora_B.weight, std=0.05)  # trained adapters are not zero
        self.scaling = scaling

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


class Attention(nn.Module):
    def __init__(self, lora=False):
        super().__init__()
        lin = (lambda: LoRALinear(DIM, DIM)) if lora else (lambda: nn.Linear(DIM, DIM, bias=False))
        self.q_proj = lin()
        self.k_proj = nn.Linear(DIM, DIM, bias=False)
        self.v_proj = lin()
        self.o_proj = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, HEADS, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, S, HEADS, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, S, HEADS, HD).transpose(1, 2)
        cos, sin = rope(S, HD, BASE, x.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = attend(q, k, v, "sqrt")
        return self.o_proj(out.transpose(1, 2).reshape(B, S, DIM))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(DIM, HIDDEN, bias=False)
        self.up_proj = nn.Linear(DIM, HIDDEN, bias=False)
        self.down_proj = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, lora=False):
        super().__init__()
        self.input_layernorm = RMSNorm(DIM)
        self.self_attn = Attention(lora)
        self.post_attention_layernorm = RMSNorm(DIM)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    def __init__(self, lora=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, DIM)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        self.layers = nn.ModuleList(DecoderLayer(lora) for _ in range(LAYERS))
        self.norm = RMSNorm(DIM)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class RefLlama(nn.Module):
    def __init__(self, lora=False):
        super().__init__()
        self.model = LlamaModel(lora)
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        return self.lm_head(self.model(ids))


def reference(seed=0, lora=False, random_norms=False):
    torch.manual_seed(seed)
    m = RefLlama(lora).eval()
    if random_norms:
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, RMSNorm):
                    mod.weight.add_(0.1 * torch.randn_like(mod.weight))
    return m


def inputs(seed=1, batch=2, seq=6):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, seq), generator=g)


# ----------------------------------------------------------------------------- port A


class Norm(nn.Module):
    def __init__(self, dim, eps, cast=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.cast = cast  # the bug: a round trip through a low precision dtype

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.cast is not None:
            n = n.to(self.cast).to(x.dtype)
        return n * self.scale


class AttnA(nn.Module):
    def __init__(self, bugs, lora_unused=False):
        super().__init__()
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.bugs = bugs
        if lora_unused:  # adapter weights loaded, but forward never applies them
            self.q_lora_a = nn.Linear(DIM, 4, bias=False)
            self.q_lora_b = nn.Linear(4, DIM, bias=False)
            self.v_lora_a = nn.Linear(DIM, 4, bias=False)
            self.v_lora_b = nn.Linear(4, DIM, bias=False)

    def forward(self, x):
        b = self.bugs
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, HEADS, HD).transpose(1, 2)
        k = self.wk(x).view(B, S, HEADS, HD).transpose(1, 2)
        v = self.wv(x).view(B, S, HEADS, HD).transpose(1, 2)
        inter = b["rope"] == "interleaved"
        cos, sin = rope(S, HD, b["base"], x.dtype, interleaved=inter)
        rot = rotate_pairs if inter else rotate_half
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        out = attend(q, k, v, b["scale"])
        return self.wo(out.transpose(1, 2).reshape(B, S, DIM))


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, HIDDEN, bias=False)
        self.w3 = nn.Linear(DIM, HIDDEN, bias=False)
        self.w2 = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BlockA(nn.Module):
    def __init__(self, bugs, lora_unused=False):
        super().__init__()
        self.norm1 = Norm(DIM, bugs["eps"], bugs["norm_cast"])
        self.attn = AttnA(bugs, lora_unused)
        self.norm2 = Norm(DIM, EPS)
        self.ffn = FFN()

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        return h + self.ffn(self.norm2(h))


class PortA(nn.Module):
    def __init__(self, bugs, lora_unused=False):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(BlockA(bugs, lora_unused) for _ in range(LAYERS))
        self.norm_f = Norm(DIM, EPS)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm_f(x))


def _bugs(eps=EPS, rope="half", base=BASE, scale="sqrt", norm_cast=None):
    return {"eps": eps, "rope": rope, "base": base, "scale": scale, "norm_cast": norm_cast}


def half_to_interleaved(w):
    """Row permutation that makes interleaved rotary on the permuted weight equal half rotary on
    the original (the inverse of the permute in Hugging Face's Llama conversion script)."""
    perm = []
    for h in range(HEADS):
        for i in range(HD // 2):
            perm += [h * HD + i, h * HD + i + HD // 2]
    return w[perm]


def _q_weight(proj, lora):
    """The weight a converter writes for a reference projection, given what it does with LoRA."""
    if isinstance(proj, LoRALinear):
        w = proj.base_layer.weight
        if lora == "merge":
            w = w + proj.scaling * proj.lora_B.weight @ proj.lora_A.weight
        elif lora == "merge_noscale":  # the bug: alpha / r scaling forgotten
            w = w + proj.lora_B.weight @ proj.lora_A.weight
        return w
    return proj.weight


def port_a(ref, lora="merge", permute_qk=False, first_layer_only=False, **bugs):
    """Convert `ref` to PortA. Bugs go to layer 0 only when first_layer_only is set."""
    b = _bugs(**bugs)
    has_lora = isinstance(ref.model.layers[0].self_attn.q_proj, LoRALinear)
    m = PortA(b, lora_unused=has_lora and lora == "ignore").eval()
    if first_layer_only:
        for blk in m.blocks[1:]:
            blk.attn.bugs = _bugs()
            blk.norm1.eps, blk.norm1.cast = EPS, None
    with torch.no_grad():
        m.tok_emb.weight.copy_(ref.model.embed_tokens.weight)
        for blk, lay in zip(m.blocks, ref.model.layers):
            a = lay.self_attn
            blk.norm1.scale.copy_(lay.input_layernorm.weight)
            blk.norm2.scale.copy_(lay.post_attention_layernorm.weight)
            wq, wk = _q_weight(a.q_proj, lora), a.k_proj.weight
            if permute_qk:
                wq, wk = half_to_interleaved(wq), half_to_interleaved(wk)
            blk.attn.wq.weight.copy_(wq)
            blk.attn.wk.weight.copy_(wk)
            blk.attn.wv.weight.copy_(_q_weight(a.v_proj, lora))
            blk.attn.wo.weight.copy_(a.o_proj.weight)
            if has_lora and lora == "ignore":
                blk.attn.q_lora_a.weight.copy_(a.q_proj.lora_A.weight)
                blk.attn.q_lora_b.weight.copy_(a.q_proj.lora_B.weight)
                blk.attn.v_lora_a.weight.copy_(a.v_proj.lora_A.weight)
                blk.attn.v_lora_b.weight.copy_(a.v_proj.lora_B.weight)
            blk.ffn.w1.weight.copy_(lay.mlp.gate_proj.weight)
            blk.ffn.w3.weight.copy_(lay.mlp.up_proj.weight)
            blk.ffn.w2.weight.copy_(lay.mlp.down_proj.weight)
        m.norm_f.scale.copy_(ref.model.norm.weight)
        m.head.weight.copy_(ref.lm_head.weight)
    return m


# ----------------------------------------------------------------------------- port B


class FusedAttention(nn.Module):
    def __init__(self, bugs):
        super().__init__()
        self.wqkv = nn.Linear(DIM, 3 * DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.bugs = bugs

    def forward(self, x):
        b = self.bugs
        B, S, _ = x.shape
        q, k, v = self.wqkv(x).split([DIM, DIM, DIM], dim=-1)
        q = q.view(B, S, HEADS, HD).transpose(1, 2)
        k = k.view(B, S, HEADS, HD).transpose(1, 2)
        v = v.view(B, S, HEADS, HD).transpose(1, 2)
        inter = b["rope"] == "interleaved"
        cos, sin = rope(S, HD, b["base"], x.dtype, interleaved=inter)
        rot = rotate_pairs if inter else rotate_half
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        out = attend(q, k, v, b["scale"])
        return self.wo(out.transpose(1, 2).reshape(B, S, DIM))


class FusedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13 = nn.Linear(DIM, 2 * HIDDEN, bias=False)
        self.w2 = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x):
        g, u = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * u)


class BlockB(nn.Module):
    def __init__(self, bugs):
        super().__init__()
        self.attn_norm_weight = nn.Parameter(torch.ones(DIM))
        self.ffn_norm_weight = nn.Parameter(torch.ones(DIM))
        self.attention = FusedAttention(bugs)
        self.feed_forward = FusedMLP()
        self.eps = bugs["eps"]

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.attn_norm_weight
        h = x + self.attention(n)
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * self.ffn_norm_weight
        return h + self.feed_forward(n)


class PortB(nn.Module):
    def __init__(self, bugs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList(BlockB(bugs) for _ in range(LAYERS))
        self.norm_weight = nn.Parameter(torch.ones(DIM))
        self.output = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.tok_embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.norm_weight
        return self.output(x)


def port_b(ref, lora="merge", **bugs):
    b = _bugs(**bugs)
    m = PortB(b).eval()
    with torch.no_grad():
        m.tok_embeddings.weight.copy_(ref.model.embed_tokens.weight)
        for blk, lay in zip(m.layers, ref.model.layers):
            a = lay.self_attn
            blk.attn_norm_weight.copy_(lay.input_layernorm.weight)
            blk.ffn_norm_weight.copy_(lay.post_attention_layernorm.weight)
            blk.attention.wqkv.weight.copy_(torch.cat(
                [_q_weight(a.q_proj, lora), a.k_proj.weight, _q_weight(a.v_proj, lora)], 0))
            blk.attention.wo.weight.copy_(a.o_proj.weight)
            blk.feed_forward.w13.weight.copy_(torch.cat(
                [lay.mlp.gate_proj.weight, lay.mlp.up_proj.weight], 0))
            blk.feed_forward.w2.weight.copy_(lay.mlp.down_proj.weight)
        m.norm_weight.copy_(ref.model.norm.weight)
        m.output.weight.copy_(ref.lm_head.weight)
    return m
