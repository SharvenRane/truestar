"""Speculative decoding that returns exactly the tokens greedy decoding returns.

    python examples/cookbook/speculative_decoding.py        # CPU, about 10 s

A small draft model proposes K tokens; the target model scores all K in one call against its KV
cache and keeps the longest prefix it agrees with, plus its own next token. The output equals plain
greedy decoding of the target only if the target's logits for a position are the same whether that
position was computed alone (one token per call) or inside a verify call of K tokens. This script
does not assume it: it decodes both ways, compares every token and the bits of every logit row the
two paths share, and exits 1 on any difference. Both models run on the compiled CPU executor in
float32, whose reductions do not depend on the number of rows in a call.
Recipe: docs/cookbook/speculative_decoding.md.
"""

import sys

import torch

import truestar
from truestar import Tensor, nn, ops
from truestar.compiler import CompiledExecutor

V, CTX, DIM, HEADS = 32, 64, 32, 4
K = 4          # draft tokens per round
NEW = 24       # tokens to generate


@nn.layer
class TinyLM(nn.Layer):
    vocab: int
    dim: int
    heads: int
    depth: int
    ctx: int
    dtype: str = "f32"

    def children(self):
        out = {"tok": nn.Embedding(self.vocab, self.dim, std=0.5, dtype=self.dtype),
               "pos": nn.Embedding(self.ctx, self.dim, std=0.5, dtype=self.dtype)}
        for i in range(self.depth):
            out[f"b{i}"] = nn.TransformerBlock(self.dim, self.heads, dtype=self.dtype)
        out["norm"] = nn.LayerNorm(self.dim, dtype=self.dtype)
        out["head"] = nn.Linear(self.dim, self.vocab, bias=False, dtype=self.dtype)
        return out

    def forward(self, p, ids, pos, cache):
        x = self.tok(p["tok"], ids) + self.pos(p["pos"], pos)
        new = ()
        for i in range(self.depth):
            k = "b" + str(i)
            x, c = self.parts[k](p[k], x, cache=cache[k], pos=pos)
            new = new + (c,)
        return self.head(p["head"], self.norm(p["norm"], x)), dict(zip(cache.keys(), new))

    def init_cache(self, batch, capacity):
        return {f"b{i}": self.parts[f"b{i}"].attn.init_cache(batch, capacity) for i in range(self.depth)}

    def cache_spec(self):
        return {f"b{i}": self.parts[f"b{i}"].attn.cache_spec() for i in range(self.depth)}


def make_step(m):
    @truestar.native(executor=CompiledExecutor("cpu", "float32", deterministic=True))
    def step(p: m.spec, ids: Tensor["B", "T", "i64"], pos: Tensor["T", "i64"], cache: m.cache_spec()):
        logits, c2 = m(p, ids, pos, cache)
        return logits, ops.argmax(logits, dim=-1), c2     # greedy pick for every position, in the program
    return step


target = TinyLM(V, DIM, HEADS, 2, CTX)
draft = TinyLM(V, DIM, HEADS, 1, CTX)
pt = target.init(seed=1)
# The draft shares the target's embeddings, first block and head, so it agrees often enough to help.
pd = {"tok": pt["tok"], "pos": pt["pos"], "b0": pt["b0"], "norm": pt["norm"], "head": pt["head"]}
t_step, d_step = make_step(target), make_step(draft)
prompt = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])


def greedy():
    """Plain greedy decoding of the target: one token per call. Returns tokens and logit rows."""
    cache = target.init_cache(1, CTX)
    logits, nxt, cache = t_step(pt, prompt, torch.arange(prompt.shape[1]), cache)
    toks, rows = [int(nxt[0, -1])], {prompt.shape[1] - 1: logits[0, -1]}
    while len(toks) < NEW:
        t = prompt.shape[1] + len(toks) - 1
        logits, nxt, cache = t_step(pt, torch.tensor([[toks[-1]]]), torch.tensor([t]), cache)
        toks.append(int(nxt[0, -1]))
        rows[t] = logits[0, -1]
    return toks, rows


def speculative():
    """Draft K tokens, verify them in one target call, keep the agreed prefix plus one target token.
    A rejected draft needs no cache rollback: attention reads only slots <= pos, and the next call
    writes over the stale slots."""
    n = prompt.shape[1]
    tc, dc = target.init_cache(1, CTX), draft.init_cache(1, CTX)
    logits, nxt, tc = t_step(pt, prompt, torch.arange(n), tc)
    _, _, dc = d_step(pd, prompt, torch.arange(n), dc)
    toks, rows, calls, accepted = [int(nxt[0, -1])], {n - 1: logits[0, -1]}, 0, 0
    while len(toks) < NEW:
        start = n + len(toks) - 1                 # position of the last accepted token
        guess, last = [], toks[-1]
        for j in range(K):                        # the draft runs one token at a time
            _, dn, dc = d_step(pd, torch.tensor([[last]]), torch.tensor([start + j]), dc)
            last = int(dn[0, -1])
            guess.append(last)
        ids = torch.tensor([[toks[-1]] + guess[:-1]])
        logits, tn, tc = t_step(pt, ids, torch.arange(start, start + K), tc)   # one verify call
        calls += 1
        for j in range(K):
            rows[start + j] = logits[0, j]
            toks.append(int(tn[0, j]))
            if int(tn[0, j]) != guess[j]:
                break                             # the target's own token replaces the rejected one
            accepted += 1
        # Keep the draft's cache in step with what was accepted: rerun it on the last token.
        _, _, dc = d_step(pd, torch.tensor([[toks[-1]]]), torch.tensor([n + len(toks) - 1]), dc)
    return toks[:NEW], rows, calls, accepted


def main():
    want, want_rows = greedy()
    got, got_rows, calls, accepted = speculative()
    shared = sorted(set(want_rows) & set(got_rows))
    bit_equal = sum(torch.equal(want_rows[t], got_rows[t]) for t in shared)
    worst = max(float((want_rows[t] - got_rows[t]).abs().max()) for t in shared)
    print(f"greedy:      {want}")
    print(f"speculative: {got}")
    print(f"target calls: greedy {NEW}, speculative {calls + 1} ({accepted} of {calls * K} draft tokens accepted)")
    print(f"tokens equal: {want == got}; logit rows bitwise equal: {bit_equal} of {len(shared)} "
          f"(largest difference {worst:.1e})")
    ok = want == got and bit_equal == len(shared)
    print("OK: speculative decoding returned the greedy tokens bit for bit" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
