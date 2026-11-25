"""RL trainer and sampler pair check (P15): `truestar rlpair` and truestar.rlpair."""

import contextlib
import copy
import functools
import io
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

import rl_models as R  # noqa: E402
from truestar import rlpair as rp  # noqa: E402
from truestar.cli import main  # noqa: E402

P = 6
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def run(sample_fn=R.sample_logits, model=None, **kw):
    model = model if model is not None else R.make_model()
    kw.setdefault("batch_sizes", ())
    return rp.pair_check(model, R.make_tokens(), P, sample_fn, **kw)


# ----------------------------------------------------------------------------- correct pairs


def test_correct_pair_matches_to_rounding():
    rep = run()
    s = rep.stats
    assert rep.status in (rp.IDENTICAL, rp.MATCH)
    assert s.max_abs < 1e-5 and s.mean_abs < 1e-6
    assert s.tokens == 2 * (16 - P) and s.over == 0 and s.outside_clip == 0
    assert abs(s.ratio_max - 1) < 1e-5 and abs(s.ratio_min - 1) < 1e-5
    assert rep.blamed is None and rep.exit_code == 0
    assert "TS190" not in rep.codes and "TS191" not in rep.codes


def test_correct_pair_every_module_maps_and_matches():
    """With localization forced on a correct pair, every module call lines up with the sampler's
    positions and none differs: the position mapping itself is right."""
    rep = run(localize=True)
    assert len(rep.modules) == 29
    assert {m.status for m in rep.modules} <= {rp.M_IDENTICAL, rp.M_NOISE}
    assert all(m.steps == 10 for m in rep.modules)
    assert all(m.total is not None and m.total < 1e-5 for m in rep.modules)
    assert not any(m.input_diffs and any(d.integer for d in m.input_diffs) for m in rep.modules)


def test_padded_batched_sampler_matches_and_maps():
    """The sampler runs the rows left padded by 3 inside a batch of 4 with longer filler rows."""
    rep = run(R.sample_logits_padded, layout=R.padded_layout(2), localize=True)
    assert rep.status in (rp.IDENTICAL, rp.MATCH)
    assert rep.stats.max_abs < 1e-5
    assert rep.layout["sampler_batch"] == 4 and rep.layout["left_pad"] == [3, 3]
    assert {m.status for m in rep.modules} <= {rp.M_IDENTICAL, rp.M_NOISE}
    assert all(m.steps > 0 for m in rep.modules)


def test_wrong_layout_is_not_clean():
    """Without the layout the padded rows are read at the wrong positions and the localization
    sees differences everywhere (the logprob check itself does not depend on the layout)."""
    rep = run(R.sample_logits_padded, layout={"rows": [0, 1], "left_pad": [0, 0]}, localize=True)
    assert any(m.status == rp.M_DIVERGENT for m in rep.modules)


def test_broken_position_mapping_is_caught(monkeypatch):
    """Permanent guard: a chunk mapping shifted by one position makes the correct pair look
    divergent, so the clean result above really depends on the mapping."""
    orig = rp._pairs

    def shifted(kinds, st, g):
        return orig(kinds, rp.Step(st.index, st.offset + 1, st.length), g)

    monkeypatch.setattr(rp, "_pairs", shifted)
    rep = run(localize=True)
    assert any(m.status == rp.M_DIVERGENT for m in rep.modules)


# ----------------------------------------------------------------------------- injected bugs


def test_temperature_in_sampler_module_named_at_head():
    rep = run(functools.partial(R.sample_logits, temperature=0.8))
    assert rep.status == rp.DIVERGENT and "TS190" in rep.codes
    b = rep.blamed
    assert b.key == "head#0" and b.kind == rp.DECODE_PATH
    assert b.first_position == 0 and b.first_step == 0
    assert rep.op is not None and rep.op.op == "aten.div.Tensor" and not rep.op.matched
    assert 0.8 in rep.op.sampler_scalars
    assert rep.stats.max_abs > 0.1 and "TS198" in rep.codes
    assert b.source["user"]["qualname"].endswith("Head")
    assert rep.exit_code == 1


def test_temperature_outside_modules_is_fitted():
    rep = run(functools.partial(R.sample_logits, outside_temperature=0.7))
    assert rep.status == rp.DIVERGENT and "TS193" in rep.codes and rep.blamed is None
    assert rep.temperature["explains"] and abs(rep.temperature["fitted"] - 0.7) < 1e-3
    assert all(m.status in (rp.M_IDENTICAL, rp.M_NOISE) for m in rep.modules)
    acts = rp.next_actions(rep)
    assert acts[0]["kind"] == "match_temperature"


def test_rotary_offset_in_kv_cache_named_at_attention():
    rep = run(functools.partial(R.sample_logits, rope_shift=1))
    assert rep.status == rp.DIVERGENT
    b = rep.blamed
    assert b.key == "blocks.0.attn#0" and b.type_name == "Attention" and b.kind == rp.DECODE_PATH
    assert b.first_position == P and b.first_step == 1  # prefill is right, the first decode is not
    assert b.impl == 0.0  # the module is right on the full sequence
    o = rep.op
    assert o.op == "aten.slice.Tensor" and o.step == 1 and o.position == P
    assert "offset +1" in o.hint and o.sampler_scalars[:3] == [0, P + 1, P + 2]
    assert rp.next_actions(rep)[1]["kind"] == "fix_position_offset"


def test_rotary_offset_named_with_padding():
    rep = run(functools.partial(R.sample_logits_padded, rope_shift=1), layout=R.padded_layout(2))
    assert rep.blamed.key == "blocks.0.attn#0" and rep.blamed.first_position == P
    assert rep.op.op == "aten.slice.Tensor" and "offset +1" in rep.op.hint
    assert "trainer position 6" in rep.op.hint


def test_bf16_sampler_is_precision_with_statistics():
    m = R.make_model()
    rep = run(model=m, sample_model=copy.deepcopy(m).to(torch.bfloat16))
    assert rep.status == rp.PRECISION and "TS191" in rep.codes and "TS190" not in rep.codes
    s = rep.stats
    assert 1e-4 < s.max_abs < 0.1 and s.mean_abs > 0 and s.over > 0
    assert 0.9 < s.ratio_min <= s.ratio_max < 1.1 and s.ratio_std > 0
    assert len(s.worst) == 5 and s.worst[0]["abs_diff"] == pytest.approx(s.max_abs)
    assert {m.status for m in rep.modules} <= {rp.M_PRECISION, rp.M_NOISE, rp.M_IDENTICAL}
    assert any(m.status == rp.M_PRECISION and m.band for m in rep.modules)
    assert rp.next_actions(rep)[0]["kind"] == "match_dtype"
    assert rep.exit_code == 1


def test_bf16_does_not_hide_a_real_bug():
    m = R.make_model()
    rep = run(functools.partial(R.sample_logits, rope_shift=1), model=m,
              sample_model=copy.deepcopy(m).to(torch.bfloat16))
    assert rep.status == rp.DIVERGENT and rep.blamed.key == "blocks.0.attn#0"
    assert rep.op.op == "aten.slice.Tensor"


def test_position_ids_that_differ_are_reported():
    rep = run(functools.partial(R.sample_logits, rope_shift=1), model=R.make_pos_model())
    assert "TS197" in rep.codes
    d = next(m for m in rep.modules if m.input_diffs)
    diff = next(x for x in d.input_diffs if x.integer)
    assert diff.first_position == P and (diff.sampler_value, diff.trainer_value) == (P + 1, P)
    assert diff.path == "args[1]"
    assert rep.blamed.name == rp.ROOT and rep.op.op.startswith("aten.arange")
    assert "offset +1" in rep.op.hint


def test_stale_sampler_weights():
    m = R.make_model()
    s = copy.deepcopy(m)
    with torch.no_grad():
        s.blocks[1].mlp.up.weight.mul_(1.01)
    rep = run(model=m, sample_model=s)
    assert "TS194" in rep.codes and rep.weights["differ"] == ["blocks.1.mlp.up.weight"]
    assert rep.blamed.key == "blocks.1.mlp.up#0" and rep.blamed.kind == rp.IMPLEMENTATION
    assert rep.status == rp.DIVERGENT
    assert any(a["kind"] == "sync_weights" for a in rp.next_actions(rep))


def test_bf16_copy_weights_are_not_stale():
    m = R.make_model()
    assert rp.weight_check(m, copy.deepcopy(m).to(torch.bfloat16))["differ"] == []


def test_sampler_returning_logprobs():
    def lp_fn(model, tokens, prompt_len):
        z = R.sample_logits(model, tokens, prompt_len)
        return torch.log_softmax(z, -1).gather(-1, tokens[:, prompt_len:, None])[..., 0]

    rep = run(lp_fn)
    assert rep.status in (rp.IDENTICAL, rp.MATCH)


# ----------------------------------------------------------------------------- batch composition


def test_batch_composition_reported_honestly_on_cpu():
    rep = run(batch_sizes=(2, 7))
    trials = rep.batch_trials
    assert [(t.size, t.position) for t in trials] == [(2, 0), (2, 1), (7, 0), (7, 6)]
    assert all(t.error == "" for t in trials)
    variant = [t for t in trials if not t.bitwise]
    assert ("TS195" in rep.codes) == bool(variant)
    assert rep.batch_invariant == (not variant)
    for t in variant:
        assert 0 < t.max_abs < 1e-4  # float rounding, not a batch dependence
    # whatever the batch does, the trainer and sampler comparison stays a match
    assert rep.status in (rp.IDENTICAL, rp.MATCH)


def test_batch_mixing_sampler_is_flagged():
    def mixing(model, tokens, prompt_len):
        z = R.sample_logits(model, tokens, prompt_len)
        return z + 1e-2 * z.mean(0, keepdim=True).roll(1, -1)  # leaks other rows into each row

    trials = rp.batch_check(mixing, R.make_model(), R.make_tokens(), P, (2,))
    assert trials and not any(t.bitwise for t in trials) and all(t.max_abs > 1e-4 for t in trials)


@needs_cuda
def test_cuda_pair_and_batch_composition():
    m = R.make_model().cuda()
    rep = rp.pair_check(m, R.make_tokens(), P, R.sample_logits, batch_sizes=(2, 7))
    assert rep.status in (rp.IDENTICAL, rp.MATCH) and rep.stats.max_abs < 1e-5
    assert rep.device.startswith("cuda") and len(rep.batch_trials) == 4
    rep = rp.pair_check(m, R.make_tokens(), P, functools.partial(R.sample_logits, rope_shift=1),
                        batch_sizes=())
    assert rep.blamed.key == "blocks.0.attn#0" and rep.op.op == "aten.slice.Tensor"


# ----------------------------------------------------------------------------- pieces


def test_dim_kinds_and_pairs():
    g = rp.Geometry(Bt=1, Bs=2, T=16, rows=[0], pads=[3])
    st = rp.Step(2, 10, 1)  # sampler position 10 = trainer position 7
    assert rp._dim_kinds((1, 4, 16, 8), [(2, 4, 1, 8)], [st], g) == ["batch", "full", "chunk", "full"]
    assert rp._dim_kinds((1, 4, 16, 16), [(2, 4, 1, 11)], [st], g) == ["batch", "full", "chunk", "prefix"]
    assert rp._dim_kinds((1, 4, 16, 8), [(2, 4, 10, 8)], [st], g) == ["batch", "full", "past", "full"]
    assert rp._dim_kinds((1, 5), [(2, 7)], [st], g) is None
    (ti, si), = rp._pairs(["batch", "chunk"], st, g)
    assert ti == (slice(0, 1), slice(7, 8)) and si == (slice(0, 1), slice(0, 1))
    (ti, si), = rp._pairs(["batch", "prefix"], st, g)
    assert ti == (slice(0, 1), slice(0, 8)) and si == (slice(0, 1), slice(3, 11))


def test_logprob_stats_by_hand():
    a = torch.tensor([[-1.0, -2.0, -3.0]])
    b = torch.tensor([[-1.0, -2.5, -2.9]])
    s = rp.logprob_stats(a, b, None, 0, threshold=0.2, clip=0.2, top=2)
    assert s.max_abs == pytest.approx(0.5) and s.mean_abs == pytest.approx(0.2)
    assert s.over == 1 and s.ratio_max == pytest.approx(torch.exp(torch.tensor(0.5)).item())
    assert s.outside_clip == 1 and s.worst[0]["position"] == 1
    assert not s.bitwise and rp.logprob_stats(a, a.clone()).bitwise


def test_fit_temperature_recovers_value():
    z = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(0))
    tg = torch.randint(0, 32, (2, 5))
    lp = torch.log_softmax(z / 1.3, -1).gather(-1, tg[..., None])[..., 0]
    tau, resid = rp.fit_temperature(z, lp, tg)
    assert abs(tau - 1.3) < 1e-3 and resid < 1e-5


# ----------------------------------------------------------------------------- command line


def _pair_file(tmp_path, name, body):
    p = tmp_path / f"{name}.py"
    p.write_text(
        "import sys, functools\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        "import rl_models as R\n"
        "def get_model():\n    return R.make_model()\n"
        "def get_tokens():\n    return R.make_tokens()\n"
        "def get_prompt_len():\n    return 6\n" + body, encoding="utf-8")
    return str(p)


def _cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(list(args))
    return code, buf.getvalue()


def test_cli_agent_json_rope(tmp_path):
    f = _pair_file(tmp_path, "pair_rope",
                   "sample_logits = functools.partial(R.sample_logits, rope_shift=1)\n")
    code, out = _cli("rlpair", f, "--format", "agent", "--batch-sizes", "")
    doc = json.loads(out)
    assert code == 1 and doc["exit_code"] == 1 and doc["command"] == "rlpair"
    assert doc["status"] == "divergent" and "TS190" in doc["codes"]
    assert doc["blamed_module"]["key"] == "blocks.0.attn#0"
    assert doc["blamed_module"]["first_position"] == 6
    assert doc["blamed_op"]["op"] == "aten.slice.Tensor"
    assert doc["logprobs"]["max_abs_diff"] > 1e-3
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert kinds[:2] == ["inspect_source", "fix_position_offset"]
    assert doc["next_actions"][0]["file"].endswith("rl_models.py")
    json.dumps(doc, allow_nan=False)


def test_cli_clean_exit_codes(tmp_path):
    f = _pair_file(tmp_path, "pair_clean", "sample_logits = R.sample_logits\n")
    code, out = _cli("rlpair", f, "--batch-sizes", "")
    assert code == 0 and "status: MATCH" in out
    code, out = _cli("rlpair", f, "--batch-sizes", "", "--strict", "--format", "agent")
    assert code == 1 and json.loads(out)["strict"] is True  # not bitwise identical


def test_cli_errors(tmp_path):
    code, out = _cli("rlpair", str(tmp_path / "missing.py"), "--format", "agent")
    assert code == 2 and json.loads(out)["codes"] == ["TS020"]
    f = _pair_file(tmp_path, "pair_bad", "sample_logits = lambda m, t, p: m(t)\n")
    code, out = _cli("rlpair", f, "--format", "agent", "--batch-sizes", "")
    doc = json.loads(out)
    assert code == 2 and doc["codes"] == ["TS199"] and doc["command"] == "rlpair"


def test_codes_registered():
    from truestar.codes import CODES

    for c in range(190, 200):
        assert f"TS{c}" in CODES
