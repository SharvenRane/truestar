"""Multi head latent attention (truestar.mla): the absorbed decode proven in float64, imported and
compiled steps with a latent cache, parity with Hugging Face DeepSeek V3, and the refusals."""

import copy

import pytest
import torch

from truestar import ir
from truestar.ir.core import UnsupportedOpError
from truestar.ir.refusals import classify, refusal_for
from truestar.mla import (MLAAbsorptionError, MLAConfig, MLAUnsupported, compile_steps, generate, import_step,
                          make_model, prove_absorption)

CUDA = torch.cuda.is_available()
C = MLAConfig()
_CACHE = {}

try:
    import transformers
    HAS_V3 = hasattr(transformers, "DeepseekV3Config")
except ImportError:   # pragma: no cover
    transformers, HAS_V3 = None, False
needs_v3 = pytest.mark.skipif(not HAS_V3, reason="needs transformers with DeepseekV3 (5.x)")


def model():
    if "m" not in _CACHE:
        _CACHE["m"] = make_model(C)
    return _CACHE["m"]


def _prompt(b=2, t=5, seed=1, vocab=C.vocab):
    return torch.randint(0, vocab, (b, t), generator=torch.Generator().manual_seed(seed))


def _rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max())


def _eager(m, prompt, steps, mode="expand", capacity=16):
    m64 = copy.deepcopy(m).double()
    with torch.no_grad():
        return generate(lambda *a: m64.step(*a, mode=mode), m64.c, prompt, steps, capacity)


# ----------------------------------------------------------------------------- the absorption proof


def test_absorbed_decode_is_proven_equal_in_float64():
    rep = prove_absorption(model(), _prompt(), steps=6)
    assert rep.ok and rep.worst < 1e-13
    assert len(rep.points) == 6 * (2 * C.layers + 1)
    te, le = _eager(model(), _prompt(), 8, "expand")
    ta, la = _eager(model(), _prompt(), 8, "absorb")
    assert torch.equal(te, ta) and _rel(la, le) < 1e-13
    assert len(set(te.reshape(-1).tolist())) > 6      # the decode really moves


@pytest.mark.parametrize("break_", ["heads_rolled", "w_uv_transposed_heads"])
def test_a_wrong_absorption_is_refused_with_ts1201_naming_the_layer(monkeypatch, break_):
    m = copy.deepcopy(model())
    good = m.absorbed

    def bad(blk):
        w_uk, w_uv = good(blk)
        if break_ == "heads_rolled":
            return w_uk.roll(1, 0), w_uv
        return w_uk, w_uv.roll(1, 0)

    monkeypatch.setattr(m, "absorbed", bad)
    with pytest.raises(MLAAbsorptionError) as e:
        prove_absorption(m, _prompt())
    assert e.value.code == "TS1201"
    assert e.value.report.first_bad[:2] == (0, "model.layers.0 attention output")
    assert "[TS1201]" in str(e.value) and "model.layers.0 attention output" in str(e.value)
    with pytest.raises(MLAAbsorptionError):          # nothing is compiled
        compile_steps(m, "cpu")


def test_unknown_mode_is_ts1200():
    with pytest.raises(ValueError, match=r"\[TS1200\] unknown MLA mode 'latent'"):
        compile_steps(model(), "cpu", decode="latent")


# ----------------------------------------------------------------------------- import and compile


@pytest.mark.parametrize("mode", ["expand", "absorb", "fused"])
def test_imported_step_matches_eager_float64_for_prompt_and_decode(mode):
    p = import_step(model(), mode)
    te, le = _eager(model(), _prompt(), 8, "expand")
    tr, lr = generate(lambda *a: ir.run(p, *a), C, _prompt(), 8, 16)
    assert torch.equal(tr, te) and _rel(lr, le) < 1e-13
    # one program for other batch sizes and capacities
    t3, l3 = generate(lambda *a: ir.run(p, *a), C, _prompt(3, 4, seed=5), 5, 24)
    e3, m3 = _eager(model(), _prompt(3, 4, seed=5), 5, "expand", capacity=24)
    assert torch.equal(t3, e3) and _rel(l3, m3) < 1e-13


def test_the_cache_holds_the_latent_not_per_head_k_and_v():
    m = copy.deepcopy(model()).double()
    cc, rc = m.empty_cache(2, 16)
    with torch.no_grad():
        _, cc2, rc2 = m.step(_prompt(), torch.arange(5), cc, rc)
    assert tuple(cc2.shape) == (C.layers, 2, 16, C.kv_rank) and tuple(rc2.shape) == (C.layers, 2, 16, C.rope)
    latent, full = C.cache_bytes(batch=2, tokens=16)
    assert latent == 2 * 16 * C.layers * 4 * (C.kv_rank + C.rope)
    assert full == 2 * 16 * C.layers * 4 * C.heads * (C.nope + C.rope + C.v_dim)
    assert full / latent == pytest.approx(C.heads * 18 / 20)


def _compiled_device():
    if CUDA:
        return "cuda"
    pytest.importorskip("numba")
    return "cpu"


def test_compiled_steps_decode_like_the_float64_reference():
    dev = _compiled_device()
    st = compile_steps(model(), dev, "float32")
    assert st.proof is not None and st.proof.ok
    tc, lc = st.generate(_prompt(), 8, 16, C, device=dev, dtype=torch.float32)
    te, le = _eager(model(), _prompt(), 8, "expand")
    assert torch.equal(tc.cpu(), te)
    assert _rel(lc.cpu(), le) < 1e-5
    for cp in (st.prefill, st.decode):
        assert not cp.report.fallbacks


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_compiled_absorbed_decode_is_bitwise_deterministic_across_two_compiles():
    outs = []
    for _ in range(2):
        st = compile_steps(model(), "cuda", "float32", decode="absorb")
        outs.append(st.generate(_prompt(), 8, 16, C, device="cuda", dtype=torch.float32))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


# ----------------------------------------------------------------------------- torch.empty in the importer


class _Fill(torch.nn.Module):
    def __init__(self, full: bool, dtype=None):
        super().__init__()
        self.full, self.dtype = full, dtype

    def forward(self, a, b):
        out = a.new_empty(a.shape[0], 5, dtype=self.dtype)
        out[:, :3].copy_(a)
        if self.full:
            out[:, 3:].copy_(b)
        return out * 2


def test_empty_written_everywhere_imports_exactly():
    a, b = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)
    p = ir.import_module(_Fill(True), (a, b))
    assert torch.equal(ir.run(p, a, b)[0], _Fill(True)(a, b))


def test_an_entry_of_empty_never_written_reads_as_nan_not_zero():
    a, b = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64)
    out = ir.run(ir.import_module(_Fill(False), (a, b)), a, b)[0]
    assert torch.equal(out[:, :3], 2 * a) and torch.isnan(out[:, 3:]).all()


def test_an_integer_empty_is_refused_with_ts1203():
    a, b = torch.randint(0, 9, (4, 3)), torch.randint(0, 9, (4, 2))
    with pytest.raises(UnsupportedOpError, match=r"TS1203"):
        ir.import_module(_Fill(True, torch.int64), (a, b))


# ----------------------------------------------------------------------------- refusals by class name


def test_token_sparse_selection_is_refused_first():
    for cls in ("DeepseekV32Indexer", "DeepseekV4Indexer", "GlmMoeDsaIndexer", "Qwen4ExpTextQSAIndexer",
                "SomeNewDsaIndexer"):
        assert classify(cls) == "sparse", cls
    assert classify("DeepseekV3Attention") is None
    mods = [("model.layers.1.mlp", "DeepseekV32MoE"), ("model.layers.0.self_attn.indexer", "DeepseekV32Indexer")]
    code, what, cls, path = refusal_for(mods)
    assert (code, cls, path) == ("TS1202", "DeepseekV32Indexer", "model.layers.0.self_attn.indexer")


# ----------------------------------------------------------------------------- Hugging Face DeepSeek V3


def _hf_cfg(**kw):
    base = dict(vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=4, q_lora_rank=24, kv_lora_rank=16, qk_nope_head_dim=8,
                qk_rope_head_dim=4, v_head_dim=6, first_k_dense_replace=2, n_routed_experts=4,
                num_experts_per_tok=2, n_group=1, topk_group=1, max_position_embeddings=64, rope_interleave=True)
    base.update(kw)
    return transformers.DeepseekV3Config(**base)


def _hf(cfg, seed=0):
    torch.manual_seed(seed)
    hf = transformers.DeepseekV3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            if p.dim() == 2:
                p.mul_(30.0)       # std 0.6: attention and the MLP matter for greedy decoding
            else:
                p.normal_(1.0, 0.2)
    return hf


YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16, "mscale": 1.0,
        "mscale_all_dim": 1.0, "rope_theta": 10000.0, "beta_fast": 32, "beta_slow": 1}


@needs_v3
@pytest.mark.parametrize("kw", [{}, {"rope_interleave": False}, {"q_lora_rank": None}, {"attention_bias": True},
                                {"rope_parameters": YARN}], ids=["interleave", "half", "no_q_rank", "bias", "yarn"])
def test_from_hf_matches_deepseek_v3_in_float64(kw):
    from truestar.mla import from_hf, hf_float64

    hf = _hf(_hf_cfg(**kw))
    h64 = hf_float64(hf)
    m = from_hf(hf).double()
    ids = _prompt(2, 9, seed=3, vocab=64)
    with torch.no_grad():
        ref = h64(input_ids=ids, use_cache=False).logits
        cc, rc = m.empty_cache(2, 12)
        for mode in ("expand", "absorb"):
            assert _rel(m.step(ids, torch.arange(9), cc, rc, mode)[0], ref) < 1e-13, mode
        seq, toks = ids[:, :5], []          # greedy by recomputing the whole sequence, no cache
        for _ in range(8):
            nxt = h64(input_ids=seq, use_cache=False).logits[:, -1].argmax(-1)
            toks.append(nxt)
            seq = torch.cat((seq, nxt[:, None]), 1)
    mine, _ = generate(lambda *a: m.step(*a, mode="absorb"), m.c, ids[:, :5], 8, 16)
    assert torch.equal(mine, torch.stack(toks, 1))


@needs_v3
def test_hf_deepseek_v3_imports_through_torch_export_and_matches_float64():
    from truestar.mla import hf_float64

    h64 = hf_float64(_hf(_hf_cfg()))

    class Fwd(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            return self.m(input_ids=ids, use_cache=False).logits

    ids = _prompt(2, 7, seed=4, vocab=64)
    prog = ir.import_module(Fwd(h64), (ids,))       # expand_kv's new_empty used to stop the import here
    assert _rel(ir.run(prog, ids)[0], Fwd(h64)(ids)) < 1e-13


@needs_v3
def test_hf_refusals_moe_sparse_and_other_mla_variants():
    from truestar.mla import from_hf

    with pytest.raises(UnsupportedOpError) as e:
        from_hf(transformers.DeepseekV3ForCausalLM(_hf_cfg(first_k_dense_replace=1)))
    assert e.value.code == "TS940"
    if hasattr(transformers, "DeepseekV32Config"):
        v32 = transformers.DeepseekV32ForCausalLM(transformers.DeepseekV32Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, q_lora_rank=24, kv_lora_rank=16, qk_nope_head_dim=8,
            qk_rope_head_dim=4, v_head_dim=6, first_k_dense_replace=2, n_routed_experts=4, num_experts_per_tok=2,
            n_group=1, topk_group=1, index_topk=4, index_n_heads=2, index_head_dim=8))
        for fn in (lambda: from_hf(v32), lambda: ir.import_module(v32, (torch.zeros(1, 4, dtype=torch.long),))):
            with pytest.raises(UnsupportedOpError) as e:
                fn()
            assert e.value.code == "TS1202" and "DeepseekV32Indexer at model.layers.0.self_attn.indexer" in str(e.value)
    if hasattr(transformers, "DeepseekV2Config"):
        v2 = transformers.DeepseekV2ForCausalLM(transformers.DeepseekV2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=4, q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4,
            v_head_dim=6, first_k_dense_replace=1, n_routed_experts=4))
        with pytest.raises(MLAUnsupported, match=r"\[TS1200\].*DeepseekV2Attention"):
            from_hf(v2)


# ----------------------------------------------------------------------------- fused latent attention (decode="fused")


def test_fused_mode_is_proven_and_imports_one_op_per_layer():
    rep = prove_absorption(model(), _prompt(), mode="fused")
    assert rep.ok and rep.worst < 1e-13
    p = import_step(model(), "fused")
    counts = p.op_counts()
    assert counts.get("mla_attention") == C.layers and counts.get("attention", 0) == 0
    assert counts.get("index_copy") == 2          # every layer's new latents written once, at the end
    te, le = _eager(model(), _prompt(), 8, "expand")
    tr, lr = generate(lambda *a: ir.run(import_step(model(), "expand"), *a), C, _prompt(), 8, 16,
                      decode=lambda *a: ir.run(p, *a))
    assert torch.equal(tr, te) and _rel(lr, le) < 1e-13


def test_fused_compiled_decode_matches_float64():
    dev = _compiled_device()
    st = compile_steps(model(), dev, "float32", decode="fused")
    assert st.decode_mode == "fused" and st.proof.ok
    assert not st.decode.report.fallbacks
    tc, lc = st.generate(_prompt(), 8, 16, C, device=dev, dtype=torch.float32)
    te, le = _eager(model(), _prompt(), 8, "expand")
    assert torch.equal(tc.cpu(), te) and _rel(lc.cpu(), le) < 1e-5


def test_fused_kernel_refuses_shapes_it_does_not_run():
    from truestar.mla.latent import MLAKernelError, latent_attention
    B, H, T, S, dn, dr, r, dv = 1, 2, 1, 4, 8, 4, 2048, 8
    args = (torch.zeros(B, H, T, dn), torch.zeros(B, H, T, dr), torch.zeros(B, S, r), torch.zeros(B, S, dr),
            torch.zeros(B, T, r), torch.zeros(B, T, dr), torch.tensor([2]), torch.zeros(H, dn, r), torch.zeros(H, dv, r))
    with pytest.raises(MLAKernelError, match="TS1460"):
        latent_attention(*args, 1.0)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("B,H,T,S,p0,dn,dr,r,dv", [(2, 4, 1, 16, 9, 8, 4, 16, 6), (3, 8, 1, 192, 150, 32, 16, 128, 32),
                                                  (1, 4, 3, 64, 40, 128, 64, 512, 128), (2, 2, 1, 40, 0, 16, 8, 32, 16),
                                                  (2, 2, 2, 40, 39, 16, 8, 32, 16)])
def test_fused_kernel_matches_float64(B, H, T, S, p0, dn, dr, r, dv):
    from truestar.mla.latent import latent_attention, reference
    g = torch.Generator().manual_seed(B * 100 + S)
    mk = lambda *s: torch.randn(*s, generator=g).cuda()  # noqa: E731
    kv_b = mk(H, dn + dv, r) * 0.2                       # W_uk and W_uv as the strided views the model passes
    args = (mk(B, H, T, dn), mk(B, H, T, dr), mk(B, S, r), mk(B, S, dr), mk(B, T, r), mk(B, T, dr),
            torch.arange(p0, p0 + T).cuda(), kv_b[:, :dn], kv_b[:, dn:])
    out = latent_attention(*args, 0.3)
    ref = reference(*args, 0.3)
    assert _rel(out.double(), ref) < 2e-6
    assert torch.equal(out, latent_attention(*args, 0.3))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_decode_is_bitwise_deterministic_across_two_compiles():
    outs = []
    for _ in range(2):
        st = compile_steps(model(), "cuda", "float32", decode="fused")
        outs.append(st.generate(_prompt(), 8, 16, C, device="cuda", dtype=torch.float32))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
