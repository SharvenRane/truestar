"""A small GPT style decoder with a KV cache decoding path, for the RL pair check tests (P15).

The same `TinyGPT` serves both paths, as in an RL stack where the trainer and the rollout engine
load the same weights:

* trainer: `model(tokens)` runs the whole sequence teacher forced, rotary positions 0..T-1.
* sampler: `sample_logits(model, tokens, prompt_len)` prefills the prompt into a `KVCache`, then
  decodes one token at a time, feeding the given tokens (the rollout's own samples), and returns
  the logits it used for every scored token.

Injectable sampler bugs: `temperature` (the sampler's logits are divided by it before logprobs,
inside the `Head` module or outside the model), `rope_shift` (the cache reports a position offset
off by that amount once it holds keys, so decode steps rotate with the wrong position), `dtype`
(the sampler runs a copy of the weights in another dtype). `sample_logits_padded` runs the row
left padded inside a batch with a longer row, with a key padding mask. Plain torch.nn, random
weights.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

VOCAB, DIM, HEADS, LAYERS, MAX_LEN = 64, 32, 4, 2, 64


class KVCache:
    def __init__(self, layers: int, rope_shift: int = 0):
        self.k = [None] * layers
        self.v = [None] * layers
        self.rope_shift = rope_shift

    def length(self, layer: int) -> int:
        return 0 if self.k[layer] is None else self.k[layer].shape[2]

    def start(self, layer: int) -> int:
        n = self.length(layer)
        return n + (self.rope_shift if n > 0 else 0)

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]


def _rotate(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, d, h, layer, max_len=MAX_LEN):
        super().__init__()
        self.h, self.hd, self.layer = h, d // h, layer
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, cache=None, key_mask=None):
        B, L, D = x.shape
        start = 0 if cache is None else cache.start(self.layer)
        q = self.q(x).view(B, L, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.hd).transpose(1, 2)
        cos, sin = self.cos[start:start + L], self.sin[start:start + L]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        S = k.shape[2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        mask = torch.ones(L, S, dtype=torch.bool, device=x.device).tril(S - L)
        if key_mask is not None:
            # Padded keys are masked; a padded query still sees itself, so no row is all masked.
            diag = torch.zeros(L, S, dtype=torch.bool, device=x.device)
            diag[torch.arange(L), torch.arange(S - L, S)] = True
            mask = (mask & key_mask[:, None, None, :S]) | diag
        scores = scores.masked_fill(~mask, float("-inf"))
        y = torch.softmax(scores, dim=-1) @ v
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, h, layer):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d), Attention(d, h, layer)
        self.norm2, self.mlp = RMSNorm(d), MLP(d, 2 * d)

    def forward(self, x, cache=None, key_mask=None):
        x = x + self.attn(self.norm1(x), cache, key_mask)
        return x + self.mlp(self.norm2(x))


class Head(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.proj = nn.Linear(d, vocab, bias=False)

    def forward(self, h, temperature=1.0):
        logits = self.proj(h)
        return logits if temperature == 1.0 else logits / temperature


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, d=DIM, h=HEADS, layers=LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, i) for i in range(layers)])
        self.norm = RMSNorm(d)
        self.head = Head(d, vocab)
        with torch.no_grad():  # trained looking norm weights, so a norm bug is visible
            for m in self.modules():
                if isinstance(m, RMSNorm):
                    m.weight.uniform_(0.5, 1.5)

    def forward(self, tokens, cache=None, key_mask=None, temperature=1.0):
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x, cache, key_mask)
        return self.head(self.norm(x), temperature)


class PosAttention(Attention):
    """Hugging Face style: rotary from a position id tensor the model passes in."""

    def forward(self, x, positions, cache=None, key_mask=None):
        B, L, D = x.shape
        q = self.q(x).view(B, L, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.hd).transpose(1, 2)
        cos, sin = self.cos[positions][:, None], self.sin[positions][:, None]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        S = k.shape[2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        mask = torch.ones(L, S, dtype=torch.bool, device=x.device).tril(S - L)
        scores = scores.masked_fill(~mask, float("-inf"))
        return self.o((torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, L, D))


class PosBlock(Block):
    def __init__(self, d, h, layer):
        super().__init__(d, h, layer)
        self.attn = PosAttention(d, h, layer)

    def forward(self, x, positions, cache=None):
        x = x + self.attn(self.norm1(x), positions, cache)
        return x + self.mlp(self.norm2(x))


class PosGPT(TinyGPT):
    """The model builds position ids from the cache offset and passes them to every block."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([PosBlock(DIM, HEADS, i) for i in range(LAYERS)])

    def forward(self, tokens, cache=None, key_mask=None, temperature=1.0):
        B, L = tokens.shape
        start = 0 if cache is None else cache.start(0)
        positions = torch.arange(start, start + L, device=tokens.device).expand(B, L)
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x, positions, cache)
        return self.head(self.norm(x), temperature)


def make_pos_model(seed=0) -> PosGPT:
    torch.manual_seed(seed)
    return PosGPT().eval()


def make_model(seed=0) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT().eval()


def make_tokens(batch=2, length=16, seed=1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, length), generator=g)


def train_logits(model, tokens):
    return model(tokens)


def sample_logits(model, tokens, prompt_len, temperature=1.0, rope_shift=0,
                  outside_temperature=1.0):
    """Prefill then decode one token per step with a KV cache, feeding the rollout's tokens."""
    T = tokens.shape[1]
    cache = KVCache(len(model.blocks), rope_shift)
    out = model(tokens[:, :prompt_len], cache, temperature=temperature)
    logits = [out[:, -1:]]
    for t in range(prompt_len, T - 1):
        logits.append(model(tokens[:, t:t + 1], cache, temperature=temperature))
    z = torch.cat(logits, dim=1)
    return z if outside_temperature == 1.0 else z / outside_temperature


def sample_logits_padded(model, tokens, prompt_len, extra=3, rope_shift=0, seed=5):
    """The rows run left padded by `extra` inside a batch that also holds longer filler rows.

    The sampler batch is [row 0 padded, ..., row B-1 padded, B filler rows], so sampler row b holds
    trainer row b with `extra` pad positions in front. Returns logits for the real rows only."""
    B, T = tokens.shape
    g = torch.Generator().manual_seed(seed)
    filler = torch.randint(0, VOCAB, (B, T + extra), generator=g).to(tokens.device)
    padded = torch.cat([torch.zeros(B, extra, dtype=tokens.dtype, device=tokens.device), tokens], 1)
    batch = torch.cat([padded, filler], 0)
    key_mask = torch.ones(2 * B, T + extra, dtype=torch.bool, device=tokens.device)
    key_mask[:B, :extra] = False
    P = prompt_len + extra
    cache = KVCache(len(model.blocks), rope_shift)
    out = model(batch[:, :P], cache, key_mask)
    logits = [out[:B, -1:]]
    for t in range(P, T + extra - 1):
        logits.append(model(batch[:, t:t + 1], cache, key_mask)[:B])
    return torch.cat(logits, dim=1)


PADDED_EXTRA = 3


def padded_layout(batch):
    return {"rows": list(range(batch)), "left_pad": [PADDED_EXTRA] * batch}
