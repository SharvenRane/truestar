"""Paged KV cache decoding of a Llama style decoder, imported to the Truestar IR and compiled.

Run: ``python examples/llama_paged_decode.py``

A paged cache keeps K and V for every layer in one pool of fixed size blocks shared by all
sequences (``[L, P, HKV, HD]``, P = blocks x block size). A sequence owns a list of blocks, its
block table; a block can sit anywhere in the pool. One decode step

* writes the new K and V into the pool at their slots (``index_copy`` on the pool's slot dim; the
  slot of position p of sequence b is ``table[b, p // block] * block + p % block``, computed on the
  host like a serving engine's slot mapping),
* gathers each sequence's cache back in position order from its block table (``index_select`` with
  ``table[:, :, None] * block + arange(block)``, integer arithmetic in the program), and
* attends with the mask ``arange(S) <= pos``, so slots past a sequence's own position, which may hold
  another sequence's data, are never read.

The same model also has the fixed capacity dense step of examples/llama_kv_decode.py. Both steps are
exported with symbolic batch, token count, block table width and pool size, imported, and run either
by the float64 reference interpreter or compiled (``truestar.compiler``). Weights can be copied from
a Hugging Face ``LlamaForCausalLM`` (``from_hf``), so the decode is checked against ``generate``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Cfg:
    vocab: int = 96
    dim: int = 32
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    layers: int = 2
    ffn: int = 64
    eps: float = 1e-6
    theta: float = 10000.0
    block: int = 4            # tokens per cache block
    rope_f32: bool = False    # rotary angles in float32 (explicit .float(), as Hugging Face computes them)
    rope_f64: bool = False    # rotary angles in float64, cos and sin rounded once to the activations' dtype


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.norm1 = nn.Parameter(torch.ones(c.dim))
        self.norm2 = nn.Parameter(torch.ones(c.dim))
        self.q = nn.Linear(c.dim, c.heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.dim, c.kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.dim, c.kv_heads * c.head_dim, bias=False)
        self.o = nn.Linear(c.heads * c.head_dim, c.dim, bias=False)
        self.gate = nn.Linear(c.dim, c.ffn, bias=False)
        self.up = nn.Linear(c.dim, c.ffn, bias=False)
        self.down = nn.Linear(c.ffn, c.dim, bias=False)


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Llama(nn.Module):
    """RMSNorm, rotary positions, grouped query attention, SwiGLU; untied lm head (Hugging Face
    ``LlamaForCausalLM`` layout)."""

    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.dim)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.Parameter(torch.ones(c.dim))
        self.lm_head = nn.Linear(c.dim, c.vocab, bias=False)
        inv = 1.0 / (c.theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.int64).float() / c.head_dim))
        self.register_buffer("inv_freq", inv)

    # -- shared pieces
    def rope(self, pos):
        """cos and sin for int64 positions of shape [T] (shared) or [B, T] (per sequence), shaped to
        broadcast against [B, H, T, HD]."""
        if self.c.rope_f64:   # angles, cos and sin in float64 (exact positions), then one rounding
            f = pos.double()[..., None] * self.inv_freq.double()
            e = torch.cat((f, f), dim=-1)
            act = self.norm.dtype
            cos, sin = e.cos().to(act), e.sin().to(act)
        elif self.c.rope_f32:   # angles in float32, then cos and sin in the activations' dtype
            f = pos.float()[..., None] * self.inv_freq.float()
            e = torch.cat((f, f), dim=-1)
            act = self.norm.dtype
            cos, sin = e.cos().to(act), e.sin().to(act)
        else:
            f = pos.to(self.inv_freq.dtype)[..., None] * self.inv_freq
            e = torch.cat((f, f), dim=-1)
            cos, sin = e.cos(), e.sin()
        if pos.dim() == 2:
            return cos[:, None], sin[:, None]
        return cos, sin

    def qkv(self, blk, x, cos, sin):
        c = self.c
        b, t, _ = x.shape
        h = rms_norm(x, blk.norm1, c.eps)
        q = blk.q(h).view(b, t, c.heads, c.head_dim).transpose(1, 2)
        k = blk.k(h).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = blk.v(h).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v

    def finish(self, blk, x, a):
        c = self.c
        b, t, _ = x.shape
        x = x + blk.o(a.transpose(1, 2).reshape(b, t, c.heads * c.head_dim))
        y = rms_norm(x, blk.norm2, c.eps)
        return x + blk.down(F.silu(blk.gate(y)) * blk.up(y))

    def head(self, x):
        return self.lm_head(rms_norm(x, self.norm, self.c.eps))

    # -- the two decode steps
    def dense_step(self, ids, pos, kc, vc):
        """Fixed capacity cache ``[L, B, HKV, S, HD]``; ``pos`` int64 [T], the same for every sequence."""
        cos, sin = self.rope(pos)
        x = self.embed(ids)
        s = kc.shape[3]
        keep = (torch.arange(s, device=pos.device)[None, :] <= pos[:, None]).view(1, 1, -1, s)
        ks, vs = [], []
        for i, blk in enumerate(self.blocks):
            q, k, v = self.qkv(blk, x, cos, sin)
            kci = kc[i].index_copy(2, pos, k)
            vci = vc[i].index_copy(2, pos, v)
            x = self.finish(blk, x, F.scaled_dot_product_attention(q, kci, vci, attn_mask=keep, enable_gqa=True))
            ks.append(kci)
            vs.append(vci)
        return self.head(x), torch.stack(ks), torch.stack(vs)

    def paged_step(self, ids, pos, slots, table, kp, vp):
        """Paged cache: pools ``[L, P, HKV, HD]``, ``pos`` int64 [B, T] per sequence, ``slots`` int64
        [B, T] (where each new token's K and V go), ``table`` int64 [B, M] (each sequence's blocks)."""
        c = self.c
        b, t = ids.shape
        m = table.shape[1]
        s = m * c.block
        cos, sin = self.rope(pos)
        x = self.embed(ids)
        read = (table[:, :, None] * c.block + torch.arange(c.block, device=table.device)[None, None, :]).reshape(b * s)
        keep = (torch.arange(s, device=pos.device)[None, None, :] <= pos[:, :, None])[:, None]   # [B, 1, T, S]
        flat = slots.reshape(b * t)
        ks, vs = [], []
        for i, blk in enumerate(self.blocks):
            q, k, v = self.qkv(blk, x, cos, sin)
            kpi = kp[i].index_copy(0, flat, k.transpose(1, 2).reshape(b * t, c.kv_heads, c.head_dim))
            vpi = vp[i].index_copy(0, flat, v.transpose(1, 2).reshape(b * t, c.kv_heads, c.head_dim))
            kci = kpi.index_select(0, read).view(b, s, c.kv_heads, c.head_dim).transpose(1, 2)
            vci = vpi.index_select(0, read).view(b, s, c.kv_heads, c.head_dim).transpose(1, 2)
            x = self.finish(blk, x, F.scaled_dot_product_attention(q, kci, vci, attn_mask=keep, enable_gqa=True))
            ks.append(kpi)
            vs.append(vpi)
        return self.head(x), torch.stack(ks), torch.stack(vs)


class FlatStep(nn.Module):
    """The dense step with its cache as one buffer ``[L * S, B, HKV, HD]`` (layer major, then position)
    written by one ``index_copy`` per layer at rows ``i * S + pos``: a form the compiled plan writes in
    place over an ``update=`` input (each write over the previous one), so a step copies no cache.
    ``rows`` int64 [L, T] holds those rows (the host's slot mapping, as for the paged step)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, pos, rows, kc, vc):
        m = self.model
        c = m.c
        cos, sin = m.rope(pos)
        x = m.embed(ids)
        s = kc.shape[0] // c.layers
        keep = (torch.arange(s, device=pos.device)[None, :] <= pos[:, None]).view(1, 1, -1, s)
        for i, blk in enumerate(m.blocks):
            q, k, v = m.qkv(blk, x, cos, sin)
            kc = kc.index_copy(0, rows[i], k.permute(2, 0, 1, 3))   # [T, B, HKV, HD] at rows of layer i
            vc = vc.index_copy(0, rows[i], v.permute(2, 0, 1, 3))
            shp = (c.layers, s) + tuple(kc.shape[1:])
            kl = kc.view(shp)[i].permute(1, 2, 0, 3)                  # layer i's cache as [B, HKV, S, HD]
            vl = vc.view(shp)[i].permute(1, 2, 0, 3)
            x = m.finish(blk, x, F.scaled_dot_product_attention(q, kl, vl, attn_mask=keep, enable_gqa=True))
        return m.head(x), kc, vc


def import_flat_step(model: "Llama", batch: int = 2, prompt: int = 3, capacity: int = 16,
                     max_capacity: int = 1 << 15):
    """FlatStep exported with symbolic batch, token count and capacity, imported (float64)."""
    from torch.export import Dim

    from truestar import ir

    c = model.c
    B, T = Dim("B", min=1, max=64), Dim("T", min=1, max=512)
    S = Dim("Cap", min=2, max=max_capacity)   # not "S": sympy reads that name as its singleton registry
    kc = torch.zeros(c.layers * capacity, batch, c.kv_heads, c.head_dim, dtype=torch.float64)
    ids = torch.zeros(batch, prompt, dtype=torch.int64)
    pos = torch.arange(prompt)
    rows = pos[None, :] + torch.arange(c.layers)[:, None] * capacity
    return ir.import_module(FlatStep(copy.deepcopy(model)).double(), (ids, pos, rows, kc, kc.clone()),
                            dynamic_shapes=({0: B, 1: T}, {0: T}, {1: T}, {0: S * c.layers, 1: B},
                                            {0: S * c.layers, 1: B}), name="flat_step")


def flat_rows(c: "Cfg", pos: torch.Tensor, capacity: int) -> torch.Tensor:
    """The FlatStep rows input for int64 positions [T]: layer i's rows are pos + i * capacity."""
    return pos[None, :] + torch.arange(c.layers, device=pos.device)[:, None] * capacity


class DenseStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, pos, kc, vc):
        return self.model.dense_step(ids, pos, kc, vc)


class PagedStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, pos, slots, table, kp, vp):
        return self.model.paged_step(ids, pos, slots, table, kp, vp)


def make_model(c: Cfg = Cfg(), seed: int = 0) -> Llama:
    """Random weights scaled so attention and the MLP matter (greedy decoding of a default init
    repeats one token and would not exercise the cache)."""
    torch.manual_seed(seed)
    m = Llama(c).eval()
    with torch.no_grad():
        m.embed.weight.normal_(0.0, 0.3)
        for blk in m.blocks:
            blk.norm1.normal_(1.0, 0.2)
            blk.norm2.normal_(1.0, 0.2)
            for lin in (blk.q, blk.k, blk.v, blk.o, blk.gate, blk.up, blk.down):
                lin.weight.mul_(3.0)
    return m


def from_hf(hf) -> Llama:
    """The same weights as a Hugging Face ``LlamaForCausalLM`` (any config with a head_dim that
    splits evenly); used to check decoding against ``generate``."""
    hc = hf.config
    hd = getattr(hc, "head_dim", None) or hc.hidden_size // hc.num_attention_heads
    c = Cfg(vocab=hc.vocab_size, dim=hc.hidden_size, heads=hc.num_attention_heads, kv_heads=hc.num_key_value_heads,
            head_dim=hd, layers=hc.num_hidden_layers, ffn=hc.intermediate_size, eps=hc.rms_norm_eps,
            theta=getattr(hc, "rope_theta", 10000.0))
    m = Llama(c).eval()
    sd = hf.state_dict()
    with torch.no_grad():
        m.embed.weight.copy_(sd["model.embed_tokens.weight"])
        m.norm.copy_(sd["model.norm.weight"])
        m.lm_head.weight.copy_(sd["lm_head.weight"])
        m.inv_freq.copy_(hf.model.rotary_emb.inv_freq)
        for i, blk in enumerate(m.blocks):
            p = f"model.layers.{i}."
            blk.norm1.copy_(sd[p + "input_layernorm.weight"])
            blk.norm2.copy_(sd[p + "post_attention_layernorm.weight"])
            for mine, theirs in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"),
                                 ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                                 ("down", "mlp.down_proj")):
                getattr(blk, mine).weight.copy_(sd[p + theirs + ".weight"])
    return m


# ----------------------------------------------------------------------------- the host side of a paged cache


class BlockPool:
    """Block allocator for one pool. ``order`` decides which free block a sequence gets next; the
    tests use a shuffled order so no sequence owns neighbouring blocks."""

    def __init__(self, n_blocks: int, block: int, order: Optional[Sequence[int]] = None):
        self.block = block
        self.free = list(order) if order is not None else list(range(n_blocks))
        self.tables: List[List[int]] = []

    def add_sequence(self) -> int:
        self.tables.append([])
        return len(self.tables) - 1

    def reserve(self, seq: int, length: int) -> None:
        """Make sure sequence ``seq`` has blocks for positions [0, length)."""
        need = -(-length // self.block)
        while len(self.tables[seq]) < need:
            if not self.free:
                raise RuntimeError("the block pool is full")
            self.tables[seq].append(self.free.pop(0))

    def slots(self, seq: int, positions: Sequence[int]) -> List[int]:
        t = self.tables[seq]
        return [t[p // self.block] * self.block + p % self.block for p in positions]

    def table(self, seqs: Sequence[int], width: int) -> torch.Tensor:
        """Block tables of ``seqs`` padded to ``width`` blocks (padding repeats block 0; the mask never
        lets a query read past its own position, so padded entries are never used)."""
        rows = []
        for s in seqs:
            t = self.tables[s]
            if len(t) > width:
                raise ValueError(f"sequence {s} has {len(t)} blocks, more than the table width {width}")
            rows.append(t + [t[0] if t else 0] * (width - len(t)))
        return torch.tensor(rows, dtype=torch.int64)


# ----------------------------------------------------------------------------- generation loops

Step = Callable[..., Sequence[torch.Tensor]]


def generate_dense(step: Step, prompt: torch.Tensor, steps: int, c: Cfg, capacity: int,
                   device="cpu", dtype=torch.float64):
    """Greedy decoding through a dense step: returns (tokens [B, steps], last logits per step)."""
    b, t0 = prompt.shape
    kc = torch.zeros(c.layers, b, c.kv_heads, capacity, c.head_dim, dtype=dtype, device=device)
    vc = torch.zeros_like(kc)
    ids, pos = prompt.to(device), torch.arange(t0, device=device)
    toks, lg = [], []
    for s in range(steps):
        logits, kc, vc = step(ids, pos, kc, vc)
        last = logits[:, -1]
        nxt = last.argmax(-1)
        toks.append(nxt)
        lg.append(last)
        ids, pos = nxt[:, None], torch.tensor([t0 + s], device=device)
    return torch.stack(toks, 1), torch.stack(lg)


def generate_paged(step: Step, prompt: torch.Tensor, steps: int, c: Cfg, n_blocks: int, width: int,
                   order=None, device="cpu", dtype=torch.float64, fill=0.0):
    """Greedy decoding through a paged step: the batch's sequences get blocks from one pool as they
    grow. ``fill`` is the pool's initial value (tests fill it with garbage that must never be read)."""
    b, t0 = prompt.shape
    pool = BlockPool(n_blocks, c.block, order)
    seqs = [pool.add_sequence() for _ in range(b)]
    kp = torch.full((c.layers, n_blocks * c.block, c.kv_heads, c.head_dim), fill, dtype=dtype, device=device)
    vp = torch.full_like(kp, fill)
    ids = prompt.to(device)
    positions = [list(range(t0))] * b
    toks, lg = [], []
    for s in range(steps):
        for q in seqs:
            pool.reserve(q, positions[q][-1] + 1)
        pos = torch.tensor(positions, dtype=torch.int64, device=device)
        slots = torch.tensor([pool.slots(q, positions[q]) for q in seqs], dtype=torch.int64, device=device)
        table = pool.table(seqs, width).to(device)
        logits, kp, vp = step(ids, pos, slots, table, kp, vp)
        last = logits[:, -1]
        nxt = last.argmax(-1)
        toks.append(nxt)
        lg.append(last)
        ids = nxt[:, None]
        positions = [[t0 + s]] * b
    return torch.stack(toks, 1), torch.stack(lg)


def import_steps(model: Llama, batch: int = 2, prompt: int = 3, capacity: int = 16, n_blocks: int = 12,
                 width: int = 4, max_capacity: int = 4096, max_tokens: int = 512,
                 trace_dtype: torch.dtype = torch.float64):
    """Both steps exported with symbolic dims and imported to the IR, traced in ``trace_dtype``
    (float64 by default; float32 keeps a model's explicit ``.double()`` visible as a float64 island
    of a float32 program, compiler/lower.py is_island)."""
    from torch.export import Dim

    from truestar import ir

    c = model.c
    m64 = DenseStep(copy.deepcopy(model)).to(trace_dtype)   # .to() changes a module in place
    B, T, S = Dim("B", min=1, max=64), Dim("T", min=1, max=max_tokens), Dim("S", min=2, max=max_capacity)
    kc = torch.zeros(c.layers, batch, c.kv_heads, capacity, c.head_dim, dtype=trace_dtype)
    ids = torch.zeros(batch, prompt, dtype=torch.int64)
    dense = ir.import_module(m64, (ids, torch.arange(prompt), kc, kc.clone()),
                             dynamic_shapes=({0: B, 1: T}, {0: T}, {1: B, 3: S}, {1: B, 3: S}), name="dense_step")
    M, P = Dim("M", min=1, max=256), Dim("P", min=2, max=1 << 20)
    kp = torch.zeros(c.layers, n_blocks * c.block, c.kv_heads, c.head_dim, dtype=trace_dtype)
    pos = torch.arange(prompt).repeat(batch, 1)
    table = torch.arange(batch * width).reshape(batch, width)
    paged = ir.import_module(PagedStep(copy.deepcopy(model)).to(trace_dtype), (ids, pos, pos.clone(), table, kp, kp.clone()),
                             dynamic_shapes=({0: B, 1: T}, {0: B, 1: T}, {0: B, 1: T}, {0: B, 1: M},
                                             {1: P}, {1: P}), name="paged_step")
    return dense, paged


def main():
    from truestar import ir

    c = Cfg()
    model = make_model(c)
    dense, paged = import_steps(model)
    print(f"dense step: {len(dense.nodes)} nodes, paged step: {len(paged.nodes)} nodes")
    prompt = torch.randint(0, c.vocab, (2, 5), generator=torch.Generator().manual_seed(1))
    order = torch.randperm(16, generator=torch.Generator().manual_seed(2)).tolist()   # scattered blocks
    td, ld = generate_dense(lambda *a: ir.run(dense, *a), prompt, 12, c, capacity=24)
    tp, lp = generate_paged(lambda *a: ir.run(paged, *a), prompt, 12, c, n_blocks=16, width=6, order=order,
                            fill=math.pi)
    print("dense tokens:", td.tolist())
    print("paged tokens:", tp.tolist())
    print(f"tokens equal: {torch.equal(td, tp)}, logits bitwise equal: {torch.equal(ld, lp)}")


if __name__ == "__main__":
    main()
