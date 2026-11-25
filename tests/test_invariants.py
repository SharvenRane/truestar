"""truestar.invariants: softmax rows sum to one, softmax gradient rows sum to zero, gate rows sum to
one, with dtype derived tolerances, on tensors and as checks compiled into a program; a stale delta
(the FP8 attention backward fault) is named by TS1048 (docs/TRAIN_NUMERICS.md)."""

import importlib.util
import os

import pytest
import torch

import truestar
from truestar import Tensor, invariants as I, ops
from truestar.ir.core import Node
from truestar.train.edit import Editor

HAVE_NUMBA = importlib.util.find_spec("numba") is not None
CUDA = torch.cuda.is_available()


@truestar.native
def attn_loss(q: Tensor["B", 4, 8, 16], k: Tensor["B", 4, 8, 16], v: Tensor["B", 4, 8, 16]) -> Tensor[()]:
    a = ops.attention(q, k, v, is_causal=True)
    s = ops.sigmoid(ops.sum(a, dim=-1))
    w = s / ops.sum(s, dim=-1, keepdim=True)
    return ops.sum(w * w)


def _program():
    # the attention backward lowered to its parts (score matmul, softmax backward, matmuls): the sites
    # these checks instrument. The default fused attention_grad is one kernel, region checked
    # against its float64 definition instead, with no delta of its own to go stale.
    old = os.environ.get("TRUESTAR_FUSED_ATTENTION_GRAD")
    os.environ["TRUESTAR_FUSED_ATTENTION_GRAD"] = "0"
    try:
        return truestar.grad(attn_loss, wrt=("q", "k")).program()
    finally:
        if old is None:
            del os.environ["TRUESTAR_FUSED_ATTENTION_GRAD"]
        else:
            os.environ["TRUESTAR_FUSED_ATTENTION_GRAD"] = old


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 4, 8, 16, generator=g) for _ in range(3)]


def _stale(P, dtype):
    """The attention backward with its delta read from probabilities rounded to ``dtype``, while
    the gradient multiplies by the unrounded ones: the stale delta of arXiv 2609.37852."""
    site = [s for s in I.find(P) if s.kind == "softmax_grad_rows"][0]
    e = Editor(P)
    e.nodes = []
    for node in P.nodes:
        names = {i.name for i in node.inputs if i is not None}
        if node.op == "mul" and {site.operands["p"], site.operands["dp"]} <= names:
            p = [i for i in node.inputs if i.name == site.operands["p"]][0]
            g = [i for i in node.inputs if i.name == site.operands["dp"]][0]
            q = e.op("cast", e.op("cast", p, dtype=dtype), dtype=p.type.dtype, name="stale_p")
            e.nodes.append(Node("mul", (g, q), dict(node.attrs), node.output, "injected stale delta"))
            continue
        e.nodes.append(node)
    return e.build(P.outputs)


def test_sites_are_found_in_an_attention_backward():
    kinds = sorted(s.kind for s in I.find(_program()))
    assert kinds == ["normalized_rows", "prob_rows", "softmax_grad_rows"]


def test_clean_program_passes_and_outputs_are_unchanged():
    P = _program()
    inst = I.instrument(P)
    x = _inputs()
    from truestar.ir.reference import run

    plain = run(P, *x)
    got = I.Sentinels(inst)(*x)
    assert len(got) == len(plain)
    for a, b in zip(got, plain):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
def test_stale_delta_is_ts1048(dtype):
    inst = I.instrument(_stale(_program(), dtype))
    with pytest.raises(I.InvariantError) as e:
        I.Sentinels(inst)(*_inputs())
    assert e.value.code == "TS1048"
    r = e.value.results[0]
    assert r.worst_ratio > 50 and "attention" in r.source


@pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba")
def test_compiled_cpu_program_passes_and_its_stale_delta_fails():
    from truestar.compiler import compile_program

    inst = I.instrument(_program())
    # the check outputs are cancelling sums: the end to end relative check cannot compare them,
    # so the first call check is off here; the region checks still verify every kernel
    run = I.Sentinels(inst, compile_program(inst.program, "cpu", "float32", verify="off"))
    run(*_inputs())
    assert all(r.ok and r.worst_ratio < 0.5 for r in run.last), [r.message() for r in run.last]
    bad = I.instrument(_stale(_program(), torch.bfloat16))
    run = I.Sentinels(bad, compile_program(bad.program, "cpu", "float32", verify="off"), mode="record")
    run(*_inputs())
    assert [r.kind for r in run.last if not r.ok] == ["softmax_grad_rows"]


def test_tensor_checks_follow_the_dtype():
    """A consistent low precision path passes at its own dtype; a stale one does not."""
    g = torch.Generator().manual_seed(3)
    s = torch.randn(64, 128, generator=g) * 3
    dp = torch.randn(64, 128, generator=g)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        p = torch.softmax(s.to(dt).float(), -1).to(dt)
        assert I.prob_rows(p).ok
        pf = p.float()
        delta = (pf * dp).sum(-1, keepdim=True)
        ds = (pf * (dp - delta)).to(dt)
        assert I.softmax_grad_rows(ds, p, dp).ok
    # FP8 probabilities used consistently pass; a delta from the float32 probabilities does not
    p32 = torch.softmax(s, -1)
    p8 = p32.to(torch.float8_e4m3fn).float()
    ok = p8 * (dp - (p8 * dp).sum(-1, keepdim=True))
    assert I.softmax_grad_rows(ok, p8, dp).ok
    stale = p8 * (dp - (p32 * dp).sum(-1, keepdim=True))
    r = I.softmax_grad_rows(stale, p8, dp)
    assert not r.ok and r.code == "TS1048"
    with pytest.raises(I.InvariantError):
        I.softmax_grad_rows(stale, p8, dp, raise_on_fail=True)


def test_gates_and_a_broken_normalization():
    g = torch.Generator().manual_seed(4)
    logits = torch.randn(32, 8, generator=g)
    top = logits.topk(2, -1).values.exp()
    w = top / top.sum(-1, keepdim=True)
    assert I.normalized_rows(w, top).ok
    w_bad = top / (top.sum(-1, keepdim=True) + 1e-3)
    r = I.normalized_rows(w_bad, top)
    assert not r.ok and r.code == "TS1049"
    p = torch.softmax(logits, -1)
    p[5] *= 1.0 + 1e-4
    r = I.prob_rows(p)
    assert not r.ok and r.first_bad_row == (5,)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_vendor_softmax_backward_passes_in_half_precision():
    """torch's CUDA softmax backward in float16 and bfloat16 passes (no false alarm at the derived
    tolerance), over long rows."""
    for dt in (torch.float16, torch.bfloat16):
        s = (torch.randn(256, 2048, device="cuda") * 4).to(dt).requires_grad_()
        p = torch.softmax(s, -1)
        dp = torch.randn_like(p)
        (ds,) = torch.autograd.grad(p, s, dp)
        assert I.prob_rows(p).ok
        r = I.softmax_grad_rows(ds, p, dp)
        assert r.ok, r.message()
