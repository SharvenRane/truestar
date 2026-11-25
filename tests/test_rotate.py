"""Verified rotations (truestar.rotate): Hadamard matrices, the fold into a Llama style model, the
float64 equivalence proof that must pass before quantization, and the proof naming wrong
placements (a norm left unfolded, a head rotation across RoPE, a rotation across SiLU)."""

import math

import pytest
import torch

from truestar import codes
from truestar.rotate import fold as R
from truestar.rotate import hadamard as H
from truestar.rotate import qerror as QE

transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("n", [2, 16, 128, 12, 48, 80, 576, 960, 1536, 2560, 896])
def test_hadamard_matrices_are_exact_and_the_walsh_transform_matches(n):
    h = H.hadamard(n)
    assert torch.equal(h @ h.T, n * torch.eye(n, dtype=torch.float64))
    assert set(h.unique().tolist()) == {-1.0, 1.0}
    x = torch.randn(3, n, dtype=torch.float64)
    assert float((H.walsh(x, normalize=False) - x @ h).abs().max()) < 1e-12 * n


def test_walsh_of_float16_rows_is_exact_for_powers_of_two():
    x = torch.randn(9, 256).half()
    y = H.walsh(x, normalize=False)
    assert torch.equal(y @ H.hadamard(256) / 256, x.double())
    assert torch.equal(H.walsh(y, normalize=False) / 256, x.double())


def test_orders_without_a_construction_are_refused():
    assert codes.slug("TS1022") == "rotation_unsupported"
    for n in (6, 10, 14, 100):
        with pytest.raises(H.RotationError, match="TS1022"):
            H.hadamard(n)


def tiny(seed=0, tied=True, bias=False):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=2,
                      num_attention_heads=3, num_key_value_heads=1, max_position_embeddings=64,
                      tie_word_embeddings=tied, attention_bias=bias)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.5 * torch.randn_like(p))    # a norm fold that matters
            elif "bias" in n:
                p.copy_(0.1 * torch.randn_like(p))
    return m


IDS = torch.randint(0, 97, (2, 13), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("plan", [R.Plan(), R.Plan(r2=None, r4=None), R.Plan(r1=None),
                                  R.Plan(r1=None, r2=None, r4="hadamard")])
@pytest.mark.parametrize("tied", [True, False])
def test_correct_folds_are_proven_equal_in_float64(plan, tied):
    m = tiny(tied=tied)
    rot = R.apply(m, plan)
    rep = R.prove(m, rot, IDS)
    assert rep.ok and rep.worst < 1e-13 and len(rep.points) == 2 * 10 + 1


def test_attention_biases_fold():
    m = tiny(bias=True)
    assert R.prove(m, R.apply(m), IDS).ok


def test_a_learned_style_orthogonal_matrix_is_proven_like_hadamard():
    m = tiny()
    q, _ = torch.linalg.qr(torch.randn(48, 48, dtype=torch.float64, generator=torch.Generator().manual_seed(3)))
    rep = R.prove(m, R.apply(m, R.Plan(r1=q)), IDS)
    assert rep.ok and rep.orthogonality["r1"] < 1e-14


def test_a_matrix_that_is_not_orthogonal_is_refused():
    assert codes.slug("TS1021") == "rotation_not_orthogonal"
    q = torch.eye(48, dtype=torch.float64)
    q[0, 0] = 1 + 1e-9
    with pytest.raises(R.NotOrthogonalError, match="TS1021"):
        R.apply(tiny(), R.Plan(r1=q))


def test_an_unfolded_norm_fails_and_is_named():
    assert codes.slug("TS1020") == "rotation_equivalence_failed"
    m = tiny()
    with pytest.raises(R.RotationProofError, match="TS1020") as e:
        R.prove(m, R.apply(m, R.Plan(fold_norms=False)), IDS)
    rep = e.value.report
    assert rep.first_bad == "model.layers.0.self_attn.q_proj"
    assert any("input_layernorm" in s for s in rep.suspects)


def test_a_head_rotation_across_rope_fails_at_the_attention_core():
    m = tiny()
    rep = R.prove(m, R.apply(m, R.Plan(extra=["qk_pre_rope"])), IDS, raise_on_error=False)
    names = [p.name for p in rep.points]
    assert rep.first_bad == "model.layers.0.self_attn (attention output, o_proj input)"
    # q and k themselves agree once the head rotation is undone: the fault is RoPE, not the fold
    assert all(p.ok for p in rep.points[: names.index(rep.first_bad)])
    assert any("RoPE" in s for s in rep.suspects)


def test_a_rotation_across_the_activation_fails_at_the_down_input():
    m = tiny()
    rep = R.prove(m, R.apply(m, R.Plan(r4=None, extra=["mlp_across_act"])), IDS, raise_on_error=False)
    assert rep.first_bad == "model.layers.0.mlp.down_proj input"
    assert any("act_fn" in s for s in rep.suspects)


def test_unsupported_layouts_and_steps_are_refused():
    with pytest.raises(H.RotationError, match="TS1022"):
        R.apply(torch.nn.Linear(4, 4))
    with pytest.raises(H.RotationError, match="TS1022"):
        R.apply(tiny(), R.Plan(extra=["r3_everywhere"]))


def test_rotation_lowers_w4a4_error_on_an_activation_outlier():
    """A model whose residual carries one large channel (the pattern of trained LLMs): the
    rotated model's W4A4 per token integer format loss is below the unrotated one's (the format
    a rotation is offered for). Format references only."""
    m = tiny(seed=2, tied=False)
    with torch.no_grad():
        m.model.embed_tokens.weight[:, 5] *= 40
    m64 = R.to_float64(m)
    rot = R.apply(m)
    R.prove(m, rot, IDS)
    ref = QE.logits(m64, IDS)
    base = QE.evaluate(m64, IDS, ["w4a4_int_rtn"], ref_logits=ref)
    rr = QE.evaluate(R.to_float64(rot.model), IDS, ["w4a4_int_rtn"], ref_logits=ref)
    assert rr["act_max_over_rms"] < base["act_max_over_rms"]
    assert rr["formats"]["w4a4_int_rtn"]["kl"] < base["formats"]["w4a4_int_rtn"]["kl"]
    assert math.isclose(rr["ce_float"], base["ce_float"], rel_tol=1e-9)


@pytest.mark.parametrize("spec", ["nvfp4", "mxfp4", "int4:g64", "fp8_e4m3:channel", "fp8_e5m2:tensor",
                                  "int8", "w4a16_int_rtn"])
def test_rotation_before_a_block_or_float_format_is_refused(spec):
    """Rotations measured 1.9x to 5.9x worse on block and float formats (docs/ROTATION.md): a
    rotated model is refused there with TS1024 and the measured result, before any layer is
    swapped; the unrotated model takes the same format, and W4A4 / W8A8 int take the rotated one."""
    m = tiny(seed=2, tied=False)
    rot = R.apply(m)
    r64 = R.to_float64(rot.model)
    before = [n for n, _ in r64.named_modules()]
    with pytest.raises(H.RotationRefusedError, match=r"TS1024.*1\.9x to 5\.9x") as e:
        QE.evaluate(r64, IDS, ["w4a4_int_rtn", spec])
    assert e.value.code == "TS1024" and spec in str(e.value)
    with pytest.raises(H.RotationRefusedError, match="TS1024"):
        QE.format_model(r64, spec)
    assert [n for n, _ in r64.named_modules()] == before        # nothing was swapped in
    QE.check_rotation_allowed(R.to_float64(m), spec)             # the unrotated model takes it
    for ok in ("w4a4_int_rtn", "w8a8_int_rtn"):
        QE.format_model(r64, ok)


def test_a_learned_r1_lowers_its_objective_stays_orthogonal_and_is_proven():
    from truestar.rotate import learned as LR
    m = tiny(seed=4, tied=False)
    with torch.no_grad():
        m.model.embed_tokens.weight[:, 7] *= 30
    log = []
    q = LR.learn_r1(m, IDS, steps=40, lr=1e-2, log=log)
    assert log[-1]["loss"] < log[0]["loss"]
    rep = R.prove(m, R.apply(m, R.Plan(r1=q)), IDS)
    assert rep.ok and rep.orthogonality["r1"] < 1e-12
