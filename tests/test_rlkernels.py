"""One kernel set for the RL trainer and sampler (P15 part b): truestar.rlkernels.

Acceptance: bitwise equal per token logprobs between the training forward over whole sequences,
the prefill and token by token decoding against a KV cache, at batch sizes 1, 3 and 8, with
padding, on a small Llama style model built in code; TP 1 equal to a one device TP 2 simulation;
`truestar rlpair --kernels rl` verifies it and names a swapped kernel. CUDA only.
"""

import contextlib
import copy
import io
import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from truestar.rlkernels import RLKernelError, RLModeError, available, rl_mode
from truestar.rlkernels import kernels as K
from truestar.rlkernels import llama as L
from truestar.rlkernels import tp as TP

pytestmark = pytest.mark.skipif(not available(), reason="the RL kernel set needs CUDA and Triton")

HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, "..", "examples")
DTYPES = [torch.float32, torch.bfloat16]


def _g(seed=0):
    return torch.Generator(device="cuda").manual_seed(seed)


def rnd(*shape, dtype=torch.float32, seed=0):
    return torch.randn(*shape, device="cuda", generator=_g(seed)).to(dtype)


# ----------------------------------------------------------------------------- acceptance


def _paths(model, rows, target_index, prompt_len):
    """Every path's logprobs for rows[target_index], as {name: tensor [T - 1]}."""
    pl = [min(prompt_len, len(r) - 1) for r in rows]
    t = target_index
    return {
        "train": L.train_logprobs(model, rows)[t],
        "prefill dynamic": L.prefill_logprobs(model, rows)[t],
        "prefill static": L.prefill_logprobs(model, rows, cache="static")[t],
        "decode dynamic": L.decode_logprobs(model, rows, pl)[t],
        "decode static": L.decode_logprobs(model, rows, pl, cache="static")[t],
        "decode static, prefill chunks of 3": L.decode_logprobs(model, rows, pl, cache="static", chunk=3)[t],
    }


def mismatches(model, n_targets=2, sizes=(1, 3, 8), prompt_len=5, seed=1, max_len=70):
    """Run every path at every batch size and batch position; return the ones that differ from
    the training forward of the row alone (an empty list is the acceptance). Rows are 6 to
    `max_len` tokens, so keys cross the attention kernel's 32 key chunks."""
    pool = L.make_rows(16, seed, model.c.vocab, min_len=6, max_len=max_len)
    bad = []
    checked = 0
    for ti in range(n_targets):
        target = pool[ti]
        ref = L.train_logprobs(model, [target])[0]
        others = [r for j, r in enumerate(pool) if j != ti]
        for B in sizes:
            for where in sorted({0, B // 2, B - 1}):
                rows = others[:B - 1]
                rows = rows[:where] + [target] + rows[where:]
                for name, lp in _paths(model, rows, where, prompt_len).items():
                    checked += 1
                    if lp.shape != ref.shape or not torch.equal(lp, ref):
                        d = (lp - ref).abs().max().item() if lp.shape == ref.shape else float("nan")
                        bad.append((ti, B, where, name, d))
    return bad, checked


@pytest.mark.parametrize("dtype", DTYPES)
def test_training_prefill_and_decode_logprobs_are_bitwise_equal(dtype):
    model = L.make_model(device="cuda", dtype=dtype)
    with rl_mode():
        bad, checked = mismatches(model)
    assert checked == 2 * (1 + 3 + 3) * 6
    assert bad == [], bad[:5]


def test_padding_rows_really_are_padded():
    """The batches above mix lengths 6 to 70, so rows are left padded in the sampler and right
    padded in the trainer, and the target rows span two key chunks; check the pool really does."""
    pool = L.make_rows(16, 1, L.Config().vocab, min_len=6, max_len=70)
    assert len({len(r) for r in pool[:8]}) >= 4
    assert min(len(r) for r in pool[:2]) > K.ATT_BN


def test_eager_pytorch_kernels_are_not_bitwise_across_paths():
    """The baseline: the same model and paths on PyTorch's own kernels (cuBLAS, SDPA) are not
    bitwise equal; the differences are rounding level, which is what breaks on policy RL."""
    model = L.make_model(device="cuda")
    bad, checked = mismatches(model, n_targets=1)
    assert bad, "eager kernels happened to be bitwise equal on every path"
    assert max(d for *_, d in bad) < 1e-3


@pytest.mark.parametrize("dtype", DTYPES)
def test_tensor_parallel_2_simulation_equals_tp_1(dtype):
    """SIMULATION on one device: ranks run one after another, all reduce is a fixed tree."""
    model = L.make_model(device="cuda", dtype=dtype)
    rows = L.make_rows(5, 3, model.c.vocab)
    with rl_mode():
        tp1 = L.train_logprobs(model, rows)
        tp2 = L.tp_train_logprobs(model, rows, 2)
    for a, b in zip(tp1, tp2):
        assert torch.equal(a, b)


def test_tensor_parallel_without_the_segment_tree_is_not_bitwise():
    """Control: a rank that reduces its half of K in one sequential pass (as a plain kernel would)
    gives other bits than TP 1, so the fixed segment tree is what makes TP 2 equal."""
    x = rnd(64, 256, seed=1)
    w = rnd(96, 256, seed=2)
    full = K.linear(x, w)
    tree = TP.row_parallel_linear([x[:, :128], x[:, 128:]], w, 2)
    assert torch.equal(tree, full)
    seq = [K.matmul_partials(x[:, r * 128:(r + 1) * 128], w[:, r * 128:(r + 1) * 128].t(), 1)
           for r in range(2)]
    assert not torch.equal(TP.all_reduce(seq), full)


# ----------------------------------------------------------------------------- tests that can fail


def test_unanchored_attention_breaks_left_padding(monkeypatch):
    """Broken kernel: key chunks start at column 0 instead of the first real key. Left padded
    decoding then no longer matches the training forward; the acceptance check catches it."""
    model = L.make_model(device="cuda")
    monkeypatch.setattr(K, "ANCHOR", False)
    with rl_mode():
        bad, _ = mismatches(model, n_targets=1, sizes=(3,))
    assert any(name.startswith(("decode", "prefill")) for *_, name, _ in bad)


def test_unanchored_softmax_breaks_left_padding():
    """Control for the softmax anchor: with leading masked entries the anchored kernel keeps the
    bits, a plain sum from column 0 (eager softmax) does not for some rows."""
    s = rnd(64, 300, seed=4) * 4
    pad = torch.full((64, 37), float("-inf"), device="cuda")
    a = K.log_softmax(s)
    b = K.log_softmax(torch.cat([pad, s], -1))[:, 37:]
    assert torch.equal(a, b)
    e = torch.log_softmax(torch.cat([pad, s], -1), -1)[:, 37:]
    assert (e - a).abs().max().item() < 1e-5


def test_segment_count_that_depends_on_the_batch_would_break_invariance(monkeypatch):
    """Broken dispatch: pick the K split from M (as split K heuristics do). Batch 1 and batch 8
    rows then differ."""
    x = rnd(8, 256, seed=5)
    w = rnd(64, 256, seed=6)
    assert torch.equal(K.linear(x[:1], w), K.linear(x, w)[:1])
    orig = K.linear

    def bad_linear(x_, w_, b=None, segments=None):
        return orig(x_, w_, b, segments=1 if x_.shape[0] == 1 else 4)

    monkeypatch.setattr(K, "linear", bad_linear)
    assert not torch.equal(K.linear(x[:1], w), K.linear(x, w)[:1])


# ----------------------------------------------------------------------------- kernels


@pytest.mark.parametrize("dtype", DTYPES)
def test_kernels_match_float64(dtype):
    """No silent wrong results (never introduce #3): every kernel against float64 PyTorch."""
    tol = 2e-5 if dtype == torch.float32 else 2e-2
    x = rnd(37, 3, 200, dtype=dtype, seed=1)
    w = rnd(72, 200, dtype=dtype, seed=2)
    bias = rnd(72, dtype=dtype, seed=3)

    def rel(a, b):
        return ((a.double() - b).abs().max() / b.abs().max().clamp(min=1e-6)).item()

    assert rel(K.linear(x, w, bias), F.linear(x.double(), w.double(), bias.double())) < tol
    assert rel(K.linear(x, w, segments=1), F.linear(x.double(), w.double())) < tol
    a3, b3 = rnd(4, 1, 33, 40, dtype=dtype, seed=4), rnd(1, 5, 40, 17, dtype=dtype, seed=5)
    assert rel(K.matmul(a3, b3), a3.double() @ b3.double()) < tol
    nw = rnd(200, dtype=dtype, seed=6)
    nb = rnd(200, dtype=dtype, seed=7)
    assert rel(K.rms_norm(x, (200,), nw, 1e-5), F.rms_norm(x.double(), (200,), nw.double(), 1e-5)) < tol
    assert rel(K.layer_norm(x, (200,), nw, nb, 1e-5),
               F.layer_norm(x.double(), (200,), nw.double(), nb.double(), 1e-5)) < tol
    big = rnd(5, 3000, dtype=dtype, seed=8) * 3
    assert rel(K.softmax(big, -1), torch.softmax(big.double(), -1)) < tol
    assert rel(K.log_softmax(big, -1), torch.log_softmax(big.double(), -1)) < tol
    assert rel(K.softmax(x, 1), torch.softmax(x.double(), 1)) < tol
    assert rel(K.sum_last(big, mean=True), big.double().mean(-1)) < tol
    q = rnd(2, 6, 70, 24, dtype=dtype, seed=9)
    k = rnd(2, 3, 90, 24, dtype=dtype, seed=10)
    v = rnd(2, 3, 90, 24, dtype=dtype, seed=11)
    mask = torch.rand(2, 1, 70, 90, device="cuda", generator=_g(12)) > 0.3
    for kw in ({}, {"is_causal": True}, {"attn_mask": mask}, {"scale": 0.3}):
        got = K.attention(q, k, v, enable_gqa=True, **kw)
        want = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), enable_gqa=True, **kw)
        assert rel(got, want) < tol, kw


def test_fully_masked_query_rows_give_zeros_not_nan():
    q, k, v = rnd(1, 2, 4, 16), rnd(1, 2, 6, 16, seed=1), rnd(1, 2, 6, 16, seed=2)
    mask = torch.ones(1, 1, 4, 6, dtype=torch.bool, device="cuda")
    mask[..., 1, :] = False
    v[..., 5, :] = float("nan")       # a never read cache slot holding garbage
    mask[..., 5] = False
    out = K.attention(q, k, v, attn_mask=mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out[:, :, 1], torch.zeros_like(out[:, :, 1]))


def test_row_kernels_are_batch_invariant_and_odd_sizes_work():
    x = rnd(50, 2500, seed=3)
    w = rnd(2500, seed=4)
    full_n = K.rms_norm(x, (2500,), w, 1e-6)
    full_s = K.log_softmax(x)
    full_r = K.sum_last(x)
    for i, n in ((0, 1), (7, 3), (20, 30)):
        assert torch.equal(K.rms_norm(x[i:i + n], (2500,), w, 1e-6), full_n[i:i + n])
        assert torch.equal(K.log_softmax(x[i:i + n]), full_s[i:i + n])
        assert torch.equal(K.sum_last(x[i:i + n]), full_r[i:i + n])
    # K not divisible by the segment count, head dim not a power of two, non contiguous inputs
    a = rnd(9, 30, seed=5)
    b = rnd(30, 11, seed=6).t().contiguous().t()
    assert torch.allclose(K.matmul(a, b, segments=None), a @ b, atol=1e-5)
    q = rnd(2, 2, 9, 20, seed=7).transpose(1, 2).contiguous().transpose(1, 2)
    out = K.attention(q, q, q, is_causal=True)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, q, q, is_causal=True), atol=1e-5)


def _unaligned(t):
    """The same values in a view that starts one element into its buffer (not 16 byte aligned)."""
    buf = torch.empty(t.numel() + 1, device=t.device, dtype=t.dtype)
    buf[1:] = t.reshape(-1)
    return buf[1:].view(t.shape)


def test_unaligned_views_give_the_same_bits():
    """A cache slice or a sliced activation is often not 16 byte aligned. With Triton's default
    alignment specialization the row sum compiled for the unaligned view reduced in another
    order (measured 2026-11-24: sum_last differed); the kernels are compiled without it."""
    x = rnd(64, 4096, seed=8)
    xu = _unaligned(x)
    assert torch.equal(K.sum_last(x), K.sum_last(xu))
    assert torch.equal(K.log_softmax(x), K.log_softmax(xu))
    assert torch.equal(K.rms_norm(x, (4096,), None, 1e-6), K.rms_norm(xu, (4096,), None, 1e-6))
    a, w = rnd(33, 512, seed=9), rnd(256, 512, seed=10)
    assert torch.equal(K.linear(a, w), K.linear(_unaligned(a), w))
    q = rnd(2, 2, 40, 32, seed=11)
    assert torch.equal(K.attention(q, q, q, is_causal=True), K.attention(_unaligned(q), q, q, is_causal=True))


def test_attention_is_invariant_to_cache_layout_and_capacity():
    B, H, Lx, D = 3, 4, 45, 32
    q, k, v = rnd(B, H, Lx, D, seed=1), rnd(B, 2, Lx, D, seed=2), rnd(B, 2, Lx, D, seed=3)
    full = K.attention(q, k, v, is_causal=True, enable_gqa=True)
    for pad, cap in ((0, 45), (5, 64), (33, 100)):
        kc = torch.zeros(B, 2, cap, D, device="cuda")
        vc = torch.zeros_like(kc)
        kc[:, :, pad:pad + Lx], vc[:, :, pad:pad + Lx] = k, v
        for i in range(Lx):
            m = torch.zeros(B, 1, 1, cap, dtype=torch.bool, device="cuda")
            m[..., pad:pad + i + 1] = True
            got = K.attention(q[:, :, i:i + 1], kc, vc, attn_mask=m, enable_gqa=True)
            assert torch.equal(got, full[:, :, i:i + 1]), (pad, cap, i)


def test_kernel_errors_are_loud():
    with pytest.raises(RLKernelError, match="CUDA only"):
        K.linear(torch.randn(2, 4), torch.randn(3, 4))
    with pytest.raises(RLKernelError, match="bool"):
        K.attention(rnd(1, 1, 2, 8), rnd(1, 1, 2, 8), rnd(1, 1, 2, 8), attn_mask=rnd(1, 1, 2, 2))
    with pytest.raises(RLKernelError, match="segments"):
        K.matmul(rnd(2, 6), rnd(6, 3), segments=4)
    with pytest.raises(RLKernelError, match="power of two"):
        K.tree_sum(rnd(3, 2))


# ----------------------------------------------------------------------------- rl_mode


def test_rl_mode_routes_and_refuses():
    x = rnd(4, 8, 16)
    lin = torch.nn.Linear(16, 16, bias=True).cuda()
    with rl_mode() as m:
        lin(x)
        x @ x.transpose(-1, -2)
        x.softmax(-1)
        F.log_softmax(x, dim=1)
        x.sum(-1)
        x.mean(dim=1, keepdim=True)
        x.sum()
        torch.arange(5, device="cuda").sum()           # integer: exact, passes
        x.amax(-1)                                     # order free: passes
        with pytest.raises(RLModeError, match="cumsum"):
            x.cumsum(-1)
        with pytest.raises(RLModeError, match="var"):
            x.var(-1)
        with pytest.raises(RLModeError, match="CUDA only"):
            torch.randn(2, 3) @ torch.randn(3, 2)
    assert m.routed_calls["linear"] == 1 and m.routed_calls["matmul"] == 1
    assert m.routed_calls["softmax"] == 1 and m.routed_calls["log_softmax"] == 1
    assert m.routed_calls["sum"] == 2 and m.routed_calls["mean"] == 1
    with rl_mode(strict=False) as m2:
        x.cumsum(-1)
    assert m2.eager_calls["cumsum"] == 1
    with pytest.raises(ValueError):
        rl_mode(overrides={"conv": "eager"})


def test_rl_mode_matches_eager_outputs_and_gradients():
    """The training forward under rl_mode can train: gradients are the eager op's backward."""
    model = L.make_model(device="cuda").train()
    ids = torch.randint(0, model.c.vocab, (3, 20), device="cuda", generator=_g(3))
    ref_model = copy.deepcopy(model)
    with rl_mode():
        loss = torch.log_softmax(model(ids), -1)[..., 0].mean()
    loss.backward()
    ref = torch.log_softmax(ref_model(ids), -1)[..., 0].mean()
    ref.backward()
    assert abs(loss.item() - ref.item()) < 1e-5
    for (n, p), (_, r) in zip(model.named_parameters(), ref_model.named_parameters()):
        assert p.grad is not None, n
        err = (p.grad - r.grad).abs().max() / r.grad.abs().max().clamp(min=1e-12)
        assert err < 1e-3, (n, err.item())


# ----------------------------------------------------------------------------- truestar rlpair


def _cli(args):
    from truestar.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(args)
    return code, buf.getvalue()


PAIR = os.path.join(EXAMPLES, "rl_kernels_pair.py")


def test_rlpair_verifies_the_rl_kernels_bitwise_with_tp_2():
    code, out = _cli(["rlpair", PAIR, "--device", "cuda", "--kernels", "rl", "--strict", "--tp", "2",
                      "--batch-sizes", "3,8", "--format", "agent"])
    doc = json.loads(out)
    assert code == 0, doc["codes"]
    assert doc["status"] == "identical" and doc["logprobs"]["bitwise"]
    assert doc["kernels"] == "rl" and doc["batch_invariant"] is True
    assert doc["tensor_parallel"] == {"tp": 2, "bitwise": True, "max_abs": 0.0, "simulated": True}
    assert [t["batch_size"] for t in doc["batch_composition"]] == [3, 3, 8, 8]


def test_rlpair_without_the_rl_kernels_is_not_bitwise():
    code, out = _cli(["rlpair", PAIR, "--device", "cuda", "--strict", "--format", "agent"])
    doc = json.loads(out)
    assert code == 1
    assert doc["kernels"] == "default" and not doc["logprobs"]["bitwise"]


def _pair():
    sys.path.insert(0, EXAMPLES)
    import rl_kernels_pair as E

    return E


def test_rlpair_names_a_hand_written_decode_attention():
    from truestar import rlpair as rp

    E = _pair()
    m = E.get_model()
    s = copy.deepcopy(m)
    for b in s.blocks:
        b.attn.decode_impl = "manual"
    rep = rp.pair_check(m, E.get_tokens(), E.get_prompt_len(), E.sample_logits, sample_model=s,
                        layout=E.get_layout(), kernels="rl", batch_sizes=())
    assert not rep.stats.bitwise and rep.exit_code == 1
    assert "TS185" in rep.codes
    k = rep.kernel_swap
    assert k["module"] == "blocks.0.attn#0" and k["sampler_call"] == "blocks.0.attn#1"
    assert k["trainer"] == "attention:rl" and k["sampler"] == "matmul:rl"
    doc = rep.to_agent()
    assert any(a["kind"] == "use_rl_kernels" for a in doc["next_actions"])


def test_rlpair_names_an_eager_kernel_swapped_into_the_sampler():
    from truestar import rlpair as rp

    E = _pair()
    m = E.get_model()
    rep = rp.pair_check(m, E.get_tokens(), E.get_prompt_len(), E.sample_logits,
                        layout=E.get_layout(), kernels="rl", batch_sizes=(),
                        sampler_overrides={"attention": "eager"})
    assert "TS185" in rep.codes and rep.exit_code == 1
    assert rep.kernel_swap["trainer"] == "attention:rl"
    assert rep.kernel_swap["sampler"] == "attention:eager"
    assert rep.kernel_swap["module"].startswith("blocks.0.attn#")


def test_rlpair_reports_a_tp_mismatch():
    from truestar import rlpair as rp

    E = _pair()

    def tp_seq(model, tokens, tp):  # TP 2 on plain kernels: rank partials not on the segment tree
        return L.tp_train_logits(model, tokens, tp) + 1e-6

    rep = rp.pair_check(E.get_model(), E.get_tokens(), E.get_prompt_len(), E.sample_logits,
                        layout=E.get_layout(), kernels="rl", batch_sizes=(), tp=2, tp_fn=tp_seq)
    assert "TS186" in rep.codes and rep.exit_code == 1 and rep.tp["bitwise"] is False
