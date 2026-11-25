"""The fused attention backward (ir/attention_grad.py, compiler/attention_bwd.py; docs/TRAINING.md).

* the float64 reference equals torch.autograd through float64 attention, causal and not;
* the gradient pass keeps plain self attention whole and emits one ``attention_grad``; a masked or
  cross attention keeps the old lowering; the two paths give the same float64 gradients;
* compiled (CPU torch blocks, CUDA Triton kernels) against float64, with odd sequence lengths and
  head dims that are not a power of two; the CUDA kernels give the same bits on every call;
* a compiled training step of a small decoder passes its float64 checks with the fused op.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

from truestar import ir
from truestar import train as T
from truestar.compiler.attention_bwd import _blocked_torch, attention_grad
from truestar.ir.attention_grad import reference_grad

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def _qkvd(shape, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g, dtype=torch.float64).to(device, dtype) for _ in range(4)]


@pytest.mark.parametrize("causal", [False, True])
def test_reference_equals_autograd_in_float64(causal):
    q, k, v, do = _qkvd((2, 3, 7, 5))
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    o = F.scaled_dot_product_attention(*leaves, is_causal=causal, scale=0.3)
    want = torch.autograd.grad(o, leaves, do)
    got = reference_grad(q, k, v, do, 0.3, causal)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-12)


class _Attn(nn.Module):
    def __init__(self, causal=True, mask=False):
        super().__init__()
        self.causal, self.mask = causal, mask

    def forward(self, q, k, v):
        if self.mask:
            m = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
            return F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)


def _grad_prog(mod, shapes):
    xs = [torch.randn(s, dtype=torch.float64) for s in shapes]
    prog = ir.import_module(mod, tuple(xs))
    return prog, ir.grad(prog, [v.name for v in prog.inputs], seed="ones")


@pytest.mark.parametrize("causal", [False, True])
def test_self_attention_is_kept_whole_and_matches_the_lowering(causal, monkeypatch):
    shapes = [(2, 2, 9, 4)] * 3
    _, g = _grad_prog(_Attn(causal), shapes)
    ops = g.op_counts()
    assert ops.get("attention_grad") == 1 and ops.get("attention") == 1 and "softmax" not in ops
    monkeypatch.setenv("TRUESTAR_FUSED_ATTENTION_GRAD", "0")
    _, g0 = _grad_prog(_Attn(causal), shapes)
    assert "attention_grad" not in g0.op_counts() and "softmax" in g0.op_counts()
    xs = _qkvd((2, 2, 9, 4), seed=3)[:3]
    a = ir.run(g, *xs)
    b = ir.run(g0, *xs)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, rtol=1e-12, atol=1e-13)


def test_masked_and_cross_attention_keep_the_lowering():
    _, g = _grad_prog(_Attn(mask=True), [(1, 2, 6, 4)] * 3)
    assert "attention_grad" not in g.op_counts()
    _, g = _grad_prog(_Attn(causal=False), [(1, 2, 6, 4), (1, 2, 5, 4), (1, 2, 5, 4)])
    assert "attention_grad" not in g.op_counts()


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("causal", [False, True])
def test_cpu_blocks_against_float64(causal):
    q, k, v, do = _qkvd((2, 3, 70, 16))
    scale = 0.25
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    got = _blocked_torch(*(t.float() for t in (q, k, v, o, do)), scale, causal)
    for a, b in zip(got, reference_grad(q, k, v, do, scale, causal)):
        assert _rel(a, b) < 1e-5


@gpu
@pytest.mark.parametrize("T_,D,causal", [(512, 64, True), (77, 64, True), (130, 40, False), (64, 128, True),
                                          (200, 80, True)])
def test_triton_kernels_against_float64_and_bitwise_repeatable(T_, D, causal):
    q, k, v, do = _qkvd((2, 3, T_, D))
    scale = 1.0 / math.sqrt(D)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    dev = [t.float().cuda() for t in (q, k, v, o, do)]
    got = attention_grad(*dev, scale, causal)
    again = attention_grad(*dev, scale, causal)
    assert torch.equal(got, again)
    for a, b in zip(got.cpu(), reference_grad(q, k, v, do, scale, causal)):
        assert _rel(a, b) < 2e-6


@gpu
def test_triton_reads_strided_heads():
    """q, k, v as the transposed [B, T, H, D] views a decoder makes."""
    q, k, v, do = (t.float().cuda().transpose(1, 2) for t in _qkvd((2, 100, 3, 32)))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got = attention_grad(q, k, v, o, do, 32 ** -0.5, True)
    want = reference_grad(*(t.double() for t in (q, k, v, do)), 32 ** -0.5, True)
    for a, b in zip(got, want):
        assert _rel(a, b) < 2e-6


@gpu
def test_triton_bfloat16_against_float64():
    q, k, v, do = _qkvd((2, 4, 256, 64))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    dev = [t.to(torch.bfloat16).cuda() for t in (q, k, v, o, do)]
    got = attention_grad(*dev, 0.125, True)
    for a, b in zip(got.cpu(), reference_grad(q, k, v, do, 0.125, True)):
        assert _rel(a, b) < 2e-2


class _Decoder(nn.Module):
    def __init__(self, d=32, heads=4, vocab=64):
        super().__init__()
        self.heads = heads
        self.emb = nn.Embedding(vocab, d)
        self.norm = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        return self.head(x)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=gpu)])
def test_compiled_train_step_with_the_fused_backward(device):
    from truestar.train.compiled import CompiledTrainStep
    torch.manual_seed(0)
    m = _Decoder().eval()
    idx = torch.randint(0, 64, (4, 48))
    prog = ir.import_module(copy.deepcopy(m), (idx[:2],), dynamic_shapes=({0: Dim("B", max=64)},))
    step = T.build_train_step(prog, T.CausalLM(), T.SGD(0.1, momentum=0.9), params=m)
    assert step.program.op_counts().get("attention_grad") == 1
    cs = CompiledTrainStep(step, device, "float32")
    x = idx.to(device)
    losses = [float(cs(x, x, lr=0.1)[0]) for _ in range(3)]     # the first call is checked against float64
    # the same steps in torch float64
    ref = copy.deepcopy(m).double()
    opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9)
    want = []
    for _ in range(3):
        out = ref(idx)
        loss = F.cross_entropy(out[:, :-1].reshape(-1, 64), idx[:, 1:].reshape(-1))
        want.append(float(loss))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert max(abs(a - b) / abs(b) for a, b in zip(losses, want)) < 1e-5


@gpu
def test_graph_step_with_an_embedding_checks_token_ids_after_each_replay():
    """CompiledTrainStep(graph=True) on a model with an embedding: the token id range check is
    recorded in the graph and read after every replay (it used to break the capture with a host
    read). Replays equal plain calls bit for bit, and an out of range id still raises."""
    from truestar.compiler.checks import bounds_error  # noqa: F401  (the error type raised)
    from truestar.train.compiled import CompiledTrainStep
    torch.manual_seed(0)
    m = _Decoder().eval()
    idx = torch.randint(0, 64, (4, 48))
    prog = ir.import_module(copy.deepcopy(m), (idx[:2],), dynamic_shapes=({0: Dim("B", max=64)},))
    step = T.build_train_step(prog, T.CausalLM(), T.SGD(0.1, momentum=0.9), params=m)
    plain = CompiledTrainStep(step, "cuda", "float32")
    graph = CompiledTrainStep(step, "cuda", "float32", graph=True)
    x = idx.cuda()
    for _ in range(5):
        a = plain(x, x, lr=0.1)[0]
        b = graph(x, x, lr=0.1)[0]
        assert torch.equal(a, b)
    assert graph._g is not None and graph._pending
    bad = x.clone()
    bad[0, 0] = 64
    with pytest.raises(Exception, match="index|range"):
        graph(bad, x, lr=0.1)
