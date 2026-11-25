"""P16: `truestar port` names the module where a second implementation computes a different model.

The models live in tests/port_models.py: a Hugging Face style reference and two ports with other
names and layouts (separate versus fused q, k, v; RMSNorm as a class versus inline). Weights are
copied through a converter, so the module map must be inferred from the weights alone.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import port_models as pm  # noqa: E402

from truestar import codes, portcheck  # noqa: E402
from truestar.capture import ROOT  # noqa: E402
from truestar.cli import main  # noqa: E402
from truestar.compare import DIVERGENT, OK  # noqa: E402
from truestar.portcheck import port_check  # noqa: E402

X = pm.inputs()
INFO_CODES = {"TS164", "TS167"}


@pytest.fixture(scope="module")
def ref():
    return pm.reference()


@pytest.fixture(scope="module")
def ref_norms():
    return pm.reference(random_norms=True)


@pytest.fixture(scope="module")
def ref_lora():
    return pm.reference(lora=True)


def _pairs(report):
    return {(p.ref, p.cand) for p in report.module_map.pairs}


def _edge(report, ref_name):
    return [e for e in report.weights.edges if e.ref == ref_name]


def _clean(report):
    assert report.status == OK, report.to_text()
    assert set(report.codes) <= INFO_CODES, report.codes
    assert report.blamed is None
    assert not report.weights.ref_unmatched and not report.weights.cand_unmatched
    assert not report.weights.ref_unused and not report.weights.cand_unused
    assert report.output.rel_err < 1e-12


# ----------------------------------------------------------------------------- correct ports


def test_correct_port_renamed_modules_is_clean(ref):
    r = port_check(ref, pm.port_a(ref), X)
    _clean(r)
    assert {e.kind for e in r.weights.edges} == {"equal"}
    assert len(r.weights.edges) == r.reference_params == r.candidate_params == 21
    pairs = _pairs(r)
    for p in [(ROOT, ROOT), ("model.embed_tokens", "tok_emb"), ("lm_head", "head"),
              ("model.layers.0", "blocks.0"), ("model.layers.1.self_attn", "blocks.1.attn"),
              ("model.layers.0.self_attn.q_proj", "blocks.0.attn.wq"),
              ("model.layers.0.self_attn.v_proj", "blocks.0.attn.wv"),
              ("model.layers.1.mlp.gate_proj", "blocks.1.ffn.w1"),
              ("model.layers.1.mlp.up_proj", "blocks.1.ffn.w3"),
              ("model.layers.0.input_layernorm", "blocks.0.norm1"),
              ("model.norm", "norm_f")]:
        assert p in pairs, p
    # Every norm weight is all ones: identical tensors are paired by first use, not by name.
    assert _edge(r, "model.layers.1.post_attention_layernorm.weight")[0].cand == "blocks.1.norm2.scale"
    assert _edge(r, "model.norm.weight")[0].cand == "norm_f.scale"
    assert _edge(r, "model.layers.0.input_layernorm.weight")[0].cand == "blocks.0.norm1.scale"


def test_correct_port_fused_qkv_and_inline_norm_is_clean(ref):
    r = port_check(ref, pm.port_b(ref), X)
    _clean(r)
    rows = {e.ref: (e.cand, e.rows) for e in r.weights.edges if e.kind == "slice_of_candidate"}
    assert rows["model.layers.0.self_attn.q_proj.weight"] == ("layers.0.attention.wqkv.weight", (0, 32))
    assert rows["model.layers.0.self_attn.k_proj.weight"] == ("layers.0.attention.wqkv.weight", (32, 64))
    assert rows["model.layers.1.self_attn.v_proj.weight"] == ("layers.1.attention.wqkv.weight", (64, 96))
    assert rows["model.layers.0.mlp.up_proj.weight"] == ("layers.0.feed_forward.w13.weight", (48, 96))
    views = {(p.ref, p.cand): (p.view, p.rows) for p in r.module_map.pairs}
    assert views[("model.layers.0.self_attn.k_proj", "layers.0.attention.wqkv")] == ("cand_slice", (32, 64))
    assert views[("model.layers.0.self_attn", "layers.0.attention")] == ("identity", None)
    assert views[("model.layers.0", "layers.0")] == ("identity", None)
    assert r.module_map.inlined["model.layers.0.input_layernorm"] == "layers.0"
    assert r.module_map.inlined["model.layers.1.post_attention_layernorm"] == "layers.1"
    assert r.module_map.inlined["model.norm"] == ROOT
    assert "TS164" in r.codes and "TS167" in r.codes


def test_correct_port_with_trained_norms_is_clean(ref_norms):
    _clean(port_check(ref_norms, pm.port_a(ref_norms), X))
    _clean(port_check(ref_norms, pm.port_b(ref_norms), X))


def test_interleaved_rotary_with_permuted_weights_is_clean(ref):
    """The Hugging Face Llama conversion permutes q and k rows for a different rotary pairing:
    a correct port, matched through a row permutation."""
    r = port_check(ref, pm.port_a(ref, rope="interleaved", permute_qk=True), X)
    _clean(r)
    kinds = {e.ref: e.kind for e in r.weights.edges}
    assert kinds["model.layers.0.self_attn.q_proj.weight"] == "row_perm"
    assert kinds["model.layers.1.self_attn.k_proj.weight"] == "row_perm"
    assert kinds["model.layers.0.self_attn.v_proj.weight"] == "equal"
    views = {(p.ref, p.cand): p.view for p in r.module_map.pairs}
    assert views[("model.layers.0.self_attn.q_proj", "blocks.0.attn.wq")] == "ref_perm"


def test_fused_reference_and_separate_candidate_is_clean(ref_norms):
    r = port_check(pm.port_b(ref_norms), pm.reference(random_norms=True), X)
    _clean(r)
    assert {e.kind for e in r.weights.edges} == {"equal", "slice_of_reference"}
    views = {(p.ref, p.cand): p.view for p in r.module_map.pairs}
    assert views[("layers.0.attention.wqkv", "model.layers.0.self_attn.v_proj")] == "ref_slice"
    assert r.module_map.cand_inlined["model.layers.0.input_layernorm"] == "layers.0"


# ----------------------------------------------------------------------------- injected port bugs

BUGS = [
    # (id, port, bugs, reference module, candidate module, own code, explanation fragments)
    ("eps_class", pm.port_a, {"eps": 1e-6}, "model.layers.0.input_layernorm", "blocks.0.norm1",
     False, ["aten.add.Tensor", "[1e-06] where the reference has [1e-05]"]),
    ("eps_inline", pm.port_b, {"eps": 1e-6}, "model.layers.0", "layers.0", True,
     ["own forward code", "model.layers.0.input_layernorm", "aten.add.Tensor",
      "[1e-06] where the reference has [1e-05]"]),
    ("rotary_half_a", pm.port_a, {"rope": "interleaved"}, "model.layers.0.self_attn",
     "blocks.0.attn", True, ["aten.slice.Tensor", "paired differently"]),
    ("rotary_half_b", pm.port_b, {"rope": "interleaved"}, "model.layers.0.self_attn",
     "layers.0.attention", True, ["aten.slice.Tensor", "paired differently"]),
    ("rotary_base_a", pm.port_a, {"base": 500000.0}, "model.layers.0.self_attn", "blocks.0.attn",
     True, ["aten.pow.Scalar", "constant 500000 instead of 10000"]),
    ("rotary_base_b", pm.port_b, {"base": 500000.0}, "model.layers.0.self_attn",
     "layers.0.attention", True, ["aten.pow.Scalar", "constant 500000 instead of 10000"]),
    ("scale_a", pm.port_a, {"scale": "linear"}, "model.layers.0.self_attn", "blocks.0.attn", True,
     ["aten.div.Tensor", "constant 8 instead of 2.82843", "missing square root"]),
    ("scale_b", pm.port_b, {"scale": "linear"}, "model.layers.0.self_attn", "layers.0.attention",
     True, ["aten.div.Tensor", "missing square root"]),
    ("cast", pm.port_a, {"norm_cast": torch.bfloat16}, "model.layers.0.input_layernorm",
     "blocks.0.norm1", False, ["casts the reference does not: to bfloat16", "round trip through bfloat16"]),
]


@pytest.mark.parametrize("case", BUGS, ids=[b[0] for b in BUGS])
def test_injected_port_bug_is_named_at_the_right_module(ref, case):
    _, port, bugs, ref_mod, cand_mod, own_code, fragments = case
    r = port_check(ref, port(ref, **bugs), X)
    assert r.status == DIVERGENT, r.to_text()
    assert r.codes[0] == "TS160"
    assert r.blamed.pair.ref == ref_mod and r.blamed.pair.cand == cand_mod, r.to_text()
    assert r.blamed.own_code is own_code
    for f in fragments:
        assert f in r.explanation, (f, r.explanation)
    # The weights are a perfect match: nothing but the computation differs.
    assert not r.weights.ref_unmatched and not r.weights.cand_unmatched
    # Baseline: the final logits only say that the outputs differ.
    assert r.output.status(r.output_threshold) == DIVERGENT


def test_bug_in_one_layer_only_names_that_layer(ref):
    r = port_check(ref, pm.port_a(ref, eps=1e-6, first_layer_only=True), X)
    bad = [x for x in r.results if x.status != OK]
    assert [(x.pair.ref, x.pair.cand) for x in bad] == [("model.layers.0.input_layernorm",
                                                         "blocks.0.norm1")]


def test_weight_rounded_to_bfloat16_is_reported(ref):
    cand = pm.port_a(ref)
    with torch.no_grad():
        w = cand.blocks[1].attn.wo.weight
        w.copy_(w.to(torch.bfloat16).to(torch.float32))
    r = port_check(ref, cand, X)
    assert "TS165" in r.codes
    e = _edge(r, "model.layers.1.self_attn.o_proj.weight")[0]
    assert e.kind == "approx" and e.cand == "blocks.1.attn.wo.weight"
    assert "rounded to bfloat16" in e.note
    assert (r.blamed.pair.ref, r.blamed.pair.cand) == ("model.layers.1.self_attn.o_proj",
                                                       "blocks.1.attn.wo")
    assert "rounded to bfloat16" in r.explanation


# ----------------------------------------------------------------------------- adapters (LoRA)


@pytest.mark.parametrize("port,cand_q", [(pm.port_a, "blocks.0.attn.wq"),
                                         (pm.port_b, "layers.0.attention.wqkv")])
def test_dropped_lora_is_reported_as_unmatched_weights(ref_lora, port, cand_q):
    r = port_check(ref_lora, port(ref_lora, lora="drop"), X)
    assert r.status == DIVERGENT
    assert "TS162" in r.codes
    missing = set(r.weights.ref_unmatched)
    assert {"model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.1.self_attn.v_proj.lora_B.weight"} <= missing
    assert len(missing) == 8
    assert (r.blamed.pair.ref, r.blamed.pair.cand) == ("model.layers.0.self_attn.q_proj", cand_q)
    assert "model.layers.0.self_attn.q_proj.lora_A.weight" in r.explanation
    acts = {a["kind"]: a for a in portcheck.next_actions(r)}
    assert "model.layers.0.self_attn.q_proj.lora_B.weight" in acts["load_missing_weights"]["weights"]


def test_lora_loaded_but_never_applied_is_reported_as_unused(ref_lora):
    r = port_check(ref_lora, pm.port_a(ref_lora, lora="ignore"), X)
    assert r.status == DIVERGENT
    assert "TS163" in r.codes
    assert "blocks.0.attn.q_lora_a.weight" in r.weights.cand_unused
    assert (r.blamed.pair.ref, r.blamed.pair.cand) == ("model.layers.0.self_attn", "blocks.0.attn")
    assert "never read" in r.explanation
    acts = {a["kind"]: a for a in portcheck.next_actions(r)}
    assert "blocks.1.attn.v_lora_b.weight" in acts["use_or_remove_weights"]["weights"]


@pytest.mark.parametrize("port", [pm.port_a, pm.port_b])
def test_correctly_merged_lora_is_clean(ref_lora, port):
    r = port_check(ref_lora, port(ref_lora, lora="merge"), X)
    assert r.status == OK, r.to_text()
    assert set(r.codes) <= INFO_CODES
    assert not r.weights.ref_unmatched and not r.weights.cand_unmatched
    merged = [e for e in r.weights.edges if e.merged and e.adapter]
    assert len(merged) == 4  # q and v in two layers
    assert all("0.5 *" in e.note for e in merged)


def test_merged_lora_with_wrong_scale_is_named(ref_lora):
    r = port_check(ref_lora, pm.port_b(ref_lora, lora="merge_noscale"), X)
    assert r.status == DIVERGENT
    assert (r.blamed.pair.ref, r.blamed.pair.cand) == ("model.layers.0.self_attn.q_proj",
                                                       "layers.0.attention.wqkv")
    assert "= reference + 1 * model.layers.0.self_attn.q_proj.lora_B.weight @" in r.explanation


def test_extra_weight_read_by_the_candidate_is_loud(ref):
    class WithBias(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.logit_bias = torch.nn.Parameter(torch.zeros(pm.VOCAB))

        def forward(self, ids):
            return self.inner(ids) + self.logit_bias

    r = port_check(ref, WithBias(pm.port_a(ref)), X)
    assert r.output.rel_err == 0.0
    assert "TS162" in r.codes and r.status == DIVERGENT
    assert r.weights.cand_unmatched == ["logit_bias"]


# ----------------------------------------------------------------------------- the checker can fail


def test_a_wrong_module_map_is_never_silent(ref, monkeypatch):
    """Guard: if the inferred map pairs the wrong modules, a correct port must not report clean."""
    real = portcheck.build_module_map

    def swapped(*a, **k):
        mm = real(*a, **k)
        for p in mm.pairs:
            if p.cand == "blocks.0.attn.wq":
                p.cand = "blocks.0.attn.wk"
        return mm

    monkeypatch.setattr(portcheck, "build_module_map", swapped)
    r = port_check(ref, pm.port_a(ref), X)
    assert r.status == DIVERGENT
    assert r.blamed.pair.ref == "model.layers.0.self_attn.q_proj"


def test_missing_slice_matching_is_loud(ref, monkeypatch):
    """Guard: without fused slice matching the fused port is not reported clean."""
    monkeypatch.setattr(portcheck, "_find_block", lambda *a, **k: None)
    r = port_check(ref, pm.port_b(ref), X)
    assert r.status != OK
    assert "TS162" in r.codes or "TS165" in r.codes


# ----------------------------------------------------------------------------- command line and agent JSON

REF_FILE = """
import sys
sys.path.insert(0, {tests!r})
import port_models as pm
def get_model(): return pm.reference()
def get_inputs(): return pm.inputs()
"""

CAND_FILE = """
import port_models as pm
def get_model(reference): return pm.port_b(reference{bugs})
"""


def _files(tmp_path, bugs=""):
    tests = os.path.dirname(os.path.abspath(__file__))
    r, c = tmp_path / "ref_model.py", tmp_path / "cand_model.py"
    r.write_text(REF_FILE.format(tests=tests), encoding="utf-8")
    c.write_text(CAND_FILE.format(bugs=bugs), encoding="utf-8")
    return str(r), str(c)


def test_cli_agent_json_names_module_and_next_actions(tmp_path, capsys):
    rf, cf = _files(tmp_path, ", eps=1e-6")
    code = main(["port", rf, cf, "--format", "agent"])
    out = capsys.readouterr().out
    doc = json.loads(out)  # exactly one JSON document on stdout
    assert code == 1 and doc["exit_code"] == 1
    assert doc["schema_version"] == 1 and doc["tool"] == "truestar" and doc["command"] == "port"
    assert doc["status"] == "divergent" and doc["codes"][0] == "TS160"
    for i in doc["issues"]:
        assert i["code"] in codes.CODES
    b = doc["blamed"]
    assert b["reference"] == "model.layers.0#0" and b["candidate"] == "layers.0#0"
    assert b["own_code"] is True
    assert "model.layers.0.input_layernorm" in b["inlined_reference_modules"]
    assert b["op"]["op"] == "aten.add.Tensor"
    assert b["source"]["user"]["qualname"].endswith("BlockB")
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert kinds[:3] == ["inspect_source", "inspect_reference", "inspect_op"]
    assert kinds[-1] == "rerun" and "truestar port" in doc["next_actions"][-1]["command"]
    assert doc["weights"]["matched"] == 21 and doc["weights"]["kinds"]["slice_of_candidate"] == 10
    assert any(p["view"] == "cand_slice" for p in doc["module_map"]["pairs"])
    json.dumps(doc, allow_nan=False)


def test_cli_clean_port_exits_zero(tmp_path, capsys):
    rf, cf = _files(tmp_path)
    assert main(["port", rf, cf, "--format", "agent"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "ok" and doc["next_actions"] == [] and doc["blamed"] is None
    assert main(["port", rf, cf]) == 0
    text = capsys.readouterr().out
    assert "status: OK" in text and "slice_of_candidate" in text


def test_cli_text_report_and_json_file(tmp_path, capsys):
    rf, cf = _files(tmp_path, ", scale='linear'")
    jp = tmp_path / "port.json"
    assert main(["port", rf, cf, "--json", str(jp)]) == 1
    text = capsys.readouterr().out
    assert "layers.0.attention" in text and "missing square root" in text
    assert json.loads(jp.read_text(encoding="utf-8"))["blamed"]["candidate"] == "layers.0.attention#0"


def test_cli_bad_file_is_usage_error(tmp_path, capsys):
    rf, _ = _files(tmp_path)
    code = main(["port", rf, str(tmp_path / "missing.py"), "--format", "agent"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 2 and doc["status"] == "error" and doc["codes"] == ["TS020"]
    assert doc["command"] == "port"


def test_port_codes_are_registered():
    for c in ("TS160", "TS161", "TS162", "TS163", "TS164", "TS165", "TS166", "TS167", "TS168",
              "TS169"):
        assert c in codes.CODES
    assert codes.severity("TS160") == "error" and codes.severity("TS164") == "info"
