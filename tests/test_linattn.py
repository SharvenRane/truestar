"""Chunked linear attention and SSM kernels (truestar.linattn, pain P53, docs/LINEAR_ATTENTION.md).

* (a) The float64 reference is the recurrence one step at a time (reference.recurrent).
* (b) Chunk size invariance: the chunked form equals the recurrence for every chunk size in exact
  arithmetic; in float64 every chunk size from 1 to 100 agrees with it to about 1e-15. In float32
  the chunk sizes 16, 32 and 64 give different bits; the tests bound how far apart they are and
  check each is as close to float64 as the verification tolerance demands.
* (c) Deterministic mode: the launch config depends on the dims only, so a 26 SM and a 70 SM GPU
  run the same config; the value block and the forward warps are shown not to change the bits,
  and a row computed alone and in a batch, and two runs, are bitwise equal.
* (d) Gradients: float64 gradients of the chunked form equal those of the recurrence for q, k, v,
  g, beta and the initial state; the Triton op's float32 gradients are within tolerance of float64.
* The first call check catches a deliberately broken kernel (TS1060), refusals carry TS1061 and
  TS1063, and the refusal hook stays closed until an importer lowering is registered.
"""

import pathlib

import pytest
import torch

from truestar import linattn as LA
from truestar.agent import code_table
from truestar.codes import CODES
from truestar.linattn import coverage, op
from truestar.linattn import triton_kernels as TK
from truestar.linattn.chunked import chunked
from truestar.linattn.reference import LinAttnError, recurrent

CUDA = LA.available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA and Triton")


def make(B, T, H, dk, dv, rule, kind, *, dtype=torch.float64, device="cpu", gscale=0.3, seed=0, s0=False):
    gen = torch.Generator().manual_seed(seed)
    f = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)  # noqa: E731
    q = f(B, T, H, dk) * dk ** -0.5
    k = torch.nn.functional.normalize(f(B, T, H, dk), dim=-1)
    v = f(B, T, H, dv)
    if kind == "none":
        g = None
    elif kind == "scalar":
        g = -torch.rand(B, T, H, generator=gen, dtype=torch.float64) * gscale
    else:
        g = -torch.rand(B, T, H, dk, generator=gen, dtype=torch.float64) * gscale
    beta = torch.sigmoid(f(B, T, H)) if rule == "delta" else None
    init = f(B, H, dk, dv) * 0.1 if s0 else None
    mv = lambda x, d=dtype: None if x is None else x.to(device=device, dtype=d)  # noqa: E731
    # decays and beta stay float32 or float64: the models compute them in float32
    gd = torch.float64 if dtype == torch.float64 else torch.float32
    return mv(q), mv(k), mv(v), mv(g, gd), mv(beta, gd), mv(init, gd)


CASES = [(r, kd) for r in ("add", "delta") for kd in ("none", "scalar", "channel")]


# ------------------------------------------------------------------ (a, b) the math, in float64

@pytest.mark.parametrize("rule,kind", CASES)
def test_chunked_equals_recurrence_for_every_chunk_size_float64(rule, kind):
    q, k, v, g, beta, s0 = make(2, 101, 3, 16, 8, rule, kind, s0=True)
    o_ref, s_ref = recurrent(q, k, v, g, beta, rule=rule, scale=0.5, initial_state=s0)
    for c in (1, 2, 3, 7, 16, 64, 100, 128):
        o, s = chunked(q, k, v, g, beta, rule=rule, chunk=c, scale=0.5, initial_state=s0)
        assert (o - o_ref).abs().max() <= 1e-13 * o_ref.abs().max(), (c, rule, kind)
        assert (s - s_ref).abs().max() <= 1e-13 * s_ref.abs().max(), (c, rule, kind)


def test_strong_decay_never_overflows():
    """A log decay of -30 per step: exp(c_i - c_j) is formed only where j <= i, so nothing overflows."""
    for rule in ("add", "delta"):
        for kind in ("scalar", "channel"):
            q, k, v, g, beta, _ = make(1, 70, 2, 16, 16, rule, kind, gscale=60.0)
            o_ref, _ = recurrent(q, k, v, g, beta, rule=rule)
            o, _ = chunked(q, k, v, g, beta, rule=rule, chunk=64)
            assert torch.isfinite(o).all()
            assert (o - o_ref).abs().max() <= 1e-12 * o_ref.abs().max()


def test_lengths_shorter_than_a_chunk_and_one_step():
    for T in (1, 5, 16, 17):
        q, k, v, g, beta, s0 = make(1, T, 2, 8, 4, "delta", "scalar", s0=True)
        o_ref, s_ref = recurrent(q, k, v, g, beta, rule="delta", initial_state=s0)
        o, s = chunked(q, k, v, g, beta, rule="delta", chunk=16, initial_state=s0)
        assert torch.allclose(o, o_ref, rtol=0, atol=1e-13) and torch.allclose(s, s_ref, rtol=0, atol=1e-13)


# ------------------------------------------------------------------ (d) gradients in float64

@pytest.mark.parametrize("rule,kind", CASES)
def test_float64_gradients_equal_the_recurrence(rule, kind):
    ins = make(1, 45, 2, 8, 6, rule, kind, s0=True, seed=3)
    names = ["q", "k", "v", "g", "beta", "s0"]
    do = torch.randn(1, 45, 2, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    dS = torch.randn(1, 2, 8, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(10))
    grads = []
    for fn in (lambda *a, s0: recurrent(*a, rule=rule, initial_state=s0),
               lambda *a, s0: chunked(*a, rule=rule, chunk=16, initial_state=s0)):
        leaves = [None if x is None else x.clone().requires_grad_(True) for x in ins]
        o, S = fn(*leaves[:5], s0=leaves[5])
        torch.autograd.backward([o, S], [do, dS])
        grads.append([None if x is None else x.grad for x in leaves])
    for n, a, b in zip(names, *grads):
        if a is None:
            continue
        assert (a - b).abs().max() <= 1e-11 * max(1.0, b.abs().max().item()), (n, rule, kind)


def test_check_gradients_chunked_float64():
    q, k, v, g, beta, s0 = make(1, 33, 2, 8, 8, "delta", "channel", s0=True)
    res = LA.check_gradients(q, k, v, g, beta, rule="delta", initial_state=s0, impl="chunked", chunk=16)
    tol = res.pop("tol")
    assert set(res) == {"q", "k", "v", "g", "beta", "initial_state"}
    assert max(res.values()) <= tol and max(res.values()) < 1e-11


# ------------------------------------------------------------------ refusals, codes, hook

def test_refusals_have_codes():
    q, k, v, g, beta, _ = make(1, 8, 1, 4, 4, "delta", "scalar")
    with pytest.raises(LinAttnError, match="TS1061"):
        LA.linear_attention(q, k, v, g, None, rule="delta")
    with pytest.raises(LinAttnError, match="TS1061"):
        LA.linear_attention(q, k, v, g, beta, rule="add")
    with pytest.raises(LinAttnError, match="TS1061"):
        LA.linear_attention(q, k, v[:, :4], g, beta, rule="delta")
    with pytest.raises(LinAttnError, match="TS1063"):
        LA.linear_attention(q, k, v, g, beta, rule="delta", chunk=64, deterministic=True)
    with pytest.raises(LinAttnError, match="TS1061"):
        LA.linear_attention(q, k, v, g, beta, rule="delta", impl="triton")   # CPU tensors


def test_codes_registered_documented_and_sorted_numerically():
    for c in ("TS1060", "TS1061", "TS1062", "TS1063"):
        assert c in CODES
    path = pathlib.Path(__file__).resolve().parents[1] / "docs" / "AGENT_INTERFACE.md"
    if path.exists():   # a runner that uploads only src and tests has no docs (test_agent checks them too)
        doc = path.read_text(encoding="utf-8")
        for c in ("TS1060", "TS1061", "TS1062", "TS1063"):
            assert f"| {c} |" in doc
    table = [r["code"] for r in code_table()]
    nums = [int(c[2:]) for c in table]
    assert nums == sorted(nums) and table.index("TS1060") > table.index("TS886")


def test_first_call_check_catches_a_broken_chunked_form(monkeypatch):
    op.reset()
    real = op.chunked

    def broken(*a, **kw):
        o, s = real(*a, **kw)
        o = o.clone()
        o[:, -1] *= 1.001     # the last step 0.1% off
        return o, s

    monkeypatch.setattr(op, "chunked", broken)
    q, k, v, g, beta, _ = make(1, 40, 2, 8, 8, "add", "scalar")
    with pytest.raises(LA.LinAttnMismatchError, match="TS1060"):
        LA.linear_attention(q, k, v, g, rule="add", impl="chunked")
    op.reset()


def test_refusal_hook_open_exactly_for_the_registered_lowerings():
    """hf_linattn registers its lowerings when imported (P60): those classes, and no others, lift
    their refusal. Saved and restored, so the registry other tests see is unchanged."""
    from truestar.ir import hf_linattn
    saved = dict(coverage._LOWERINGS)
    try:
        for cls, (kind, rule, decay, _) in coverage.LAYERS.items():
            assert coverage.kernel_ready(cls), cls
            assert LA.refusal_lifted(kind, cls) == (cls in hf_linattn.SPECS), cls
        assert not LA.refusal_lifted("moe", "Qwen3MoeSparseMoeBlock")
        coverage._LOWERINGS.clear()
        assert not any(LA.refusal_lifted(kind, cls) for cls, (kind, *_r) in coverage.LAYERS.items())
        coverage.register_lowering("Qwen3NextGatedDeltaNet", lambda m: None)
        assert LA.refusal_lifted("linear", "Qwen3NextGatedDeltaNet")
        assert not LA.refusal_lifted("ssm", "Qwen3NextGatedDeltaNet")   # wrong kind
    finally:
        coverage._LOWERINGS.clear()
        coverage._LOWERINGS.update(saved)
    with pytest.raises(KeyError):
        coverage.register_lowering("NotALayer", lambda m: None)


# ------------------------------------------------------------------ Triton kernels on CUDA

TOL_CASES = [(r, kd, dt) for r in ("add", "delta") for kd in ("none", "scalar")
             for dt in (torch.float32, torch.float16, torch.bfloat16)]


@needs_cuda
@pytest.mark.parametrize("rule,kind,dtype", TOL_CASES)
def test_triton_matches_float64(rule, kind, dtype):
    op.reset()
    for (B, T, H, dk, dv) in [(2, 77, 3, 48, 40), (1, 300, 2, 128, 64)]:
        q, k, v, g, beta, s0 = make(B, T, H, dk, dv, rule, kind, dtype=dtype, device="cuda", s0=True)
        o, S = LA.linear_attention(q, k, v, g, beta, rule=rule, initial_state=s0, output_final_state=True)
        last = LA.last_call()
        assert last["impl"] == "triton" and last["checked"]
        assert o.dtype == dtype and S.dtype == torch.float32
        assert last["errors"]["o"] < 1e-5 if dtype == torch.float32 else True


@needs_cuda
def test_value_block_and_warps_do_not_change_bits():
    for rule in ("add", "delta"):
        q, k, v, g, beta, _ = make(2, 200, 2, 64, 64, rule, "scalar", dtype=torch.float32, device="cuda")
        outs = []
        for bv in (16, 32, 64):
            for w in (4, 8):
                cfg = {"CHUNK": 16, "BK": 64, "BV": bv, "num_warps": w}
                outs.append(TK.forward(q, k, v, g, beta, rule=rule, scale=1.0, initial_state=None, cfg=cfg))
        for o, s in outs[1:]:
            assert torch.equal(o, outs[0][0]) and torch.equal(s, outs[0][1]), rule


@needs_cuda
def test_deterministic_bitwise_across_sm_counts_batches_and_runs():
    from truestar.kernel.triton_backend import target_limits
    for rule in ("add", "delta"):
        q, k, v, g, beta, _ = make(3, 150, 2, 32, 32, rule, "scalar", dtype=torch.float32, device="cuda")
        cfgs, outs = [], []
        for sms in (26, 70, 132):
            with target_limits(sms, 101376, 16 << 30):
                cfgs.append(TK.config(3, 2, 32, 32, deterministic=True))
                outs.append(LA.linear_attention(q, k, v, g, beta, rule=rule, deterministic=True))
        assert cfgs[0] == cfgs[1] == cfgs[2]
        assert all(torch.equal(o, outs[0]) for o in outs)
        sl = lambda x: None if x is None else x[1:2]  # noqa: E731
        alone = LA.linear_attention(sl(q), sl(k), sl(v), sl(g), sl(beta), rule=rule, deterministic=True)
        assert torch.equal(alone, outs[0][1:2]), "a row alone and in a batch differ"


@needs_cuda
def test_chunk_sizes_differ_only_by_rounding_float32():
    for rule in ("add", "delta"):
        q, k, v, g, beta, _ = make(2, 512, 2, 64, 64, rule, "scalar", dtype=torch.float32, device="cuda")
        rep = LA.chunk_invariance(q, k, v, g, beta, rule=rule, chunks=(16, 32, 64))
        for c, e in rep["vs_float64"].items():
            assert e["o"] < 2e-5 and e["final_state"] < 2e-5, (rule, c, e)
        for pair, d in rep["pairs"].items():
            assert d["per_element"] < 2e-5, (rule, pair, d)
            assert not d["bitwise_equal"]     # different chunk sizes: different bits, as in other libraries


@needs_cuda
def test_error_does_not_grow_along_the_chunk_chain():
    for rule in ("add", "delta"):
        q, k, v, g, beta, _ = make(1, 4096, 2, 64, 64, rule, "scalar", dtype=torch.float32, device="cuda", gscale=0.05)
        rep = LA.chain_error(q, k, v, g, beta, rule=rule, chunk=16)
        assert rep["worst"] < 2e-5 and rep["last_over_first"] < 4, (rule, rep["worst"], rep["last_over_first"])


@needs_cuda
@pytest.mark.parametrize("rule", ["add", "delta"])
def test_triton_gradients_against_float64(rule):
    q, k, v, g, beta, s0 = make(1, 100, 2, 32, 32, rule, "scalar", dtype=torch.float32, device="cuda", s0=True)
    res = LA.check_gradients(q, k, v, g, beta, rule=rule, initial_state=s0)
    tol = res.pop("tol")
    assert max(res.values()) <= tol, res
    assert LA.last_call()["impl"] == "triton"


@needs_cuda
def test_channel_decay_on_cuda_runs_the_chunked_form_and_says_so():
    q, k, v, g, beta, _ = make(1, 70, 2, 16, 16, "delta", "channel", dtype=torch.float32, device="cuda")
    LA.linear_attention(q, k, v, g, beta, rule="delta")
    assert LA.last_call()["impl"] == "chunked" and LA.last_call()["decay"] == "channel"
    with pytest.raises(LinAttnError, match="TS1061"):
        LA.linear_attention(q, k, v, g, beta, rule="delta", impl="triton")


@needs_cuda
def test_first_call_check_catches_a_broken_kernel(monkeypatch):
    op.reset()
    real = TK.forward

    def broken(*a, **kw):
        o, s = real(*a, **kw)
        o[:, 40:41, 0, 3] += 1e-3 * o.abs().max()    # one element of one step
        return o, s

    monkeypatch.setattr(TK, "forward", broken)
    q, k, v, g, beta, _ = make(1, 64, 2, 32, 32, "delta", "scalar", dtype=torch.float32, device="cuda")
    with pytest.raises(LA.LinAttnMismatchError, match="TS1060"):
        LA.linear_attention(q, k, v, g, beta, rule="delta")
    op.reset()


MPS = torch.backends.mps.is_available()


@pytest.mark.skipif(not MPS, reason="needs an MPS device")
@pytest.mark.parametrize("rule,kind", [("add", "none"), ("add", "scalar"), ("delta", "none"), ("add", "channel")])
def test_mps_runs_the_chunked_form_checked_against_float64_on_the_cpu(rule, kind):
    """MPS has no float64: the first call check runs the float64 recurrence on the CPU (it failed
    with a float64 conversion error on the device before 2026-11-25)."""
    op.reset()
    q, k, v, g, beta, _ = make(2, 64, 2, 16, 8, rule, kind, dtype=torch.float32, seed=3)
    dev = [None if t is None else t.to("mps") for t in (q, k, v, g, beta)]
    o, S = LA.linear_attention(*dev, rule=rule, output_final_state=True)
    info = LA.last_call()
    assert info["impl"] == ("chunked" if kind == "channel" else "metal") and info["checked"]
    assert o.device.type == "mps"
    assert info["errors"]["o"] < 1e-5 and info["errors"]["final_state"] < 1e-5
    o_cpu = LA.linear_attention(q, k, v, g, beta, rule=rule)
    assert torch.allclose(o.cpu(), o_cpu, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not MPS, reason="needs an MPS device")
@pytest.mark.parametrize("rule,kind", [("add", "none"), ("add", "scalar"), ("delta", "none"), ("delta", "scalar")])
@pytest.mark.parametrize("T,dk,dv", [(37, 16, 8), (200, 64, 64), (65, 128, 96)])
def test_metal_kernels_match_float64_bitwise_run_to_run_and_batch_invariant(rule, kind, T, dk, dv):
    op.reset()
    q, k, v, g, beta, s0 = make(3, T, 2, dk, dv, rule, kind, dtype=torch.float32, seed=T + dk, s0=True)
    dev = [None if t is None else t.to("mps") for t in (q, k, v, g, beta, s0)]
    o1, S1 = LA.linear_attention(*dev[:5], rule=rule, initial_state=dev[5], scale=0.7, output_final_state=True)
    info = LA.last_call()
    assert info["impl"] == "metal" and info["checked"]
    o64, s64 = recurrent(*(None if t is None else t.double() for t in (q, k, v, g, beta)), rule=rule,
                         scale=0.7, initial_state=s0.double())
    assert float((o1.cpu().double() - o64).abs().max() / o64.abs().max()) < 1e-5
    assert float((S1.cpu().double() - s64).abs().max() / s64.abs().max()) < 1e-5
    o2, S2 = LA.linear_attention(*dev[:5], rule=rule, initial_state=dev[5], scale=0.7, output_final_state=True)
    assert torch.equal(o1.cpu(), o2.cpu()) and torch.equal(S1.cpu(), S2.cpu())
    one = [None if t is None else t[1:2] for t in dev]
    o3 = LA.linear_attention(*one[:5], rule=rule, initial_state=one[5], scale=0.7)
    assert torch.equal(o1[1:2].cpu(), o3.cpu())


@pytest.mark.skipif(not MPS, reason="needs an MPS device")
def test_metal_kernels_half_inputs_and_gradients():
    op.reset()
    q, k, v, g, beta, _ = make(2, 50, 2, 32, 16, "delta", "scalar", dtype=torch.float32, seed=9)
    dev = [None if t is None else t.to("mps").requires_grad_(True) for t in (q, k, v)] +           [g.to("mps"), beta.to("mps")]
    o = LA.linear_attention(*dev, rule="delta")
    assert LA.last_call()["impl"] == "metal"
    o.square().sum().backward()
    ref = [t.clone().requires_grad_(True) for t in (q, k, v)]
    LA.linear_attention(*ref, g, beta, rule="delta").square().sum().backward()
    for a, b in zip(dev[:3], ref):
        assert torch.allclose(a.grad.cpu(), b.grad, rtol=1e-3, atol=1e-4)
    oh = LA.linear_attention(*(t.detach().half() for t in dev[:3]), dev[3], dev[4], rule="delta")
    assert oh.dtype == torch.float16 and LA.last_call()["checked"]


@pytest.mark.skipif(not MPS, reason="needs an MPS device")
def test_metal_value_block_width_does_not_change_bits(monkeypatch):
    from truestar.linattn import metal_kernels as MKm
    q, k, v, g, beta, _ = make(1, 70, 2, 32, 48, "delta", "scalar", dtype=torch.float32, seed=4)
    dev = [t.to("mps") for t in (q, k, v, g, beta)]
    outs = []
    for cap in ("4", "16", "64"):
        monkeypatch.setenv("TRUESTAR_METAL_LA_BV", cap)
        outs.append(MKm.forward(*dev, rule="delta", scale=1.0, initial_state=None))
    assert all(torch.equal(outs[0][0].cpu(), o.cpu()) and torch.equal(outs[0][1].cpu(), s.cpu()) for o, s in outs[1:])
