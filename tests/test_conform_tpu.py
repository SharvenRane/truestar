"""truestar conform tpu: saved battery outputs judged against float64 here (docs/TPU.md, row P44).

No accelerator is needed: the bundles are written by the battery's own writer with outputs built
from the float64 reference (clean), from bfloat16 rounded inputs (the TPU default matmul precision)
and with planted faults.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from truestar.cli import main as cli_main
from truestar.conform import tpu as T
from truestar.conform import tpu_battery as B

CASES = ("mm_f32_default", "mm_bf16", "layernorm_f32", "log_f32", "sum_f32_wide")

PINNED = {  # input bytes are the same on every host: splitmix64 and exact float conversions only
    "mm_f32_default": "109bd40fba8eace0cebbaf6cd62ad396986a90251b36b108ebcfea59ae38fd21",
    "mm_bf16": "37de795375ac82be9c693375dc909e69435a44fcdb6f56f1afc1b6fb32fb7d29",
    "sum_f32_wide": "a05129035434357f021dc6aebce91dab4f1a902f58569e39880b1fc52bdb8480",
    "log_f32": "258df692228eee5177cb2d9ddf85ea957d5d5e260ea04564177d8c133295bf68",
}


def _bf16(x):
    """Round float32 to bfloat16 (nearest even), returned as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def _clean(case):
    ref = B.reference(case)
    out = ref.astype(np.float32)
    return _bf16(out) if B.CASES[case][1] == "bfloat16" else out


def _bundle(tmp_path, results, name="fake_tpu"):
    d = str(tmp_path / "run")
    B.write(d, {name: (results, {"framework": "test"})})
    return d


def _row(out, n=6):
    return {"out": out, "hashes": ["a" * 16] * n}


def test_inputs_are_pinned_and_exact():
    for case, h in PINNED.items():
        assert B.input_hash(B.inputs(case)) == h
    a = B.inputs("mm_bf16")["a"]
    assert np.array_equal(_bf16(a), a)  # bfloat16 cases hold bfloat16 values exactly
    v = B.inputs("sum_f32_wide")["v"]
    assert np.array_equal(v.astype(np.float64).astype(np.float32), v)


def test_clean_outputs_pass(tmp_path):
    d = _bundle(tmp_path, {c: _row(_clean(c)) for c in CASES})
    verdicts, top = T.check(d, cases=list(CASES))
    assert not top
    assert {v.case: v.verdict for v in verdicts} == {c: "ok" for c in CASES}


def test_default_precision_matmul_is_reduced_precision(tmp_path):
    t = B.inputs("mm_f32_default")
    one_pass = (_bf16(t["a"]).astype(np.float64) @ _bf16(t["b"]).astype(np.float64)).astype(np.float32)
    d = _bundle(tmp_path, {"mm_f32_default": _row(one_pass)})
    (v,), top = T.check(d, cases=["mm_f32_default"])
    assert v.verdict == "reduced_precision"
    assert v.metrics["eff_bits"] < T.bits_budget("mm_f32_default")
    assert [c for c, _ in v.issues] == ["TS902"]
    assert "precision=HIGHEST" in v.issues[0][1]  # the backend name says tpu


def test_planted_faults(tmp_path):
    ln = _clean("layernorm_f32").copy()
    ln[3, 7] += 0.5  # one wrong element
    lg = _clean("log_f32").copy()
    lg[::1000] = np.nextafter(lg[::1000], np.float32(np.inf)) * (1 + 2 ** -20)  # 8 ulp off
    res = {"layernorm_f32": _row(ln), "log_f32": _row(lg),
           "mm_bf16": {"out": _clean("mm_bf16"), "hashes": ["a" * 16, "b" * 16, "a" * 16]},
           "sum_f32_wide": {"error": "RuntimeError: no kernel"}}
    d = _bundle(tmp_path, res)
    verdicts, _ = T.check(d, cases=list(CASES[1:]))
    got = {v.case: (v.verdict, [c for c, _ in v.issues]) for v in verdicts}
    assert got["layernorm_f32"] == ("divergent", ["TS901"])  # one wrong element, not reduced precision
    assert got["log_f32"] == ("ulp_high", ["TS905"])
    assert got["mm_bf16"] == ("unstable_ok", ["TS903"])
    assert got["sum_f32_wide"] == ("failed", ["TS906"])


def test_wrong_result_is_divergent(tmp_path):
    d = _bundle(tmp_path, {"mm_f32_default": _row(-_clean("mm_f32_default"))})
    (v,), _ = T.check(d, cases=["mm_f32_default"])
    assert v.verdict == "divergent" and v.issues[0][0] == "TS901"


def test_input_hash_mismatch_is_not_judged(tmp_path):
    d = _bundle(tmp_path, {"log_f32": _row(_clean("log_f32"))})
    p = os.path.join(d, "summary.json")
    s = json.load(open(p, encoding="utf-8"))
    s["input_hashes"]["log_f32"] = "0" * 64
    json.dump(s, open(p, "w", encoding="utf-8"))
    verdicts, top = T.check(d, cases=["log_f32"])
    assert not verdicts and [c for c, _ in top] == ["TS904"]


def test_cli_agent_json_and_usage(tmp_path, capsys):
    d = _bundle(tmp_path, {"log_f32": _row(_clean("log_f32"))})
    s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    assert set(s["backends"]["fake_tpu"]["results"]) == {"log_f32"}
    assert cli_main(["conform", "tpu", d, "--format", "agent"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["mode"] == "tpu" and doc["status"] == "ok"
    assert [c["verdict"] for c in doc["cases"]] == ["ok"]
    assert cli_main(["conform", "tpu", str(tmp_path / "nothing"), "--format", "agent"]) == 2
    assert json.loads(capsys.readouterr().out)["codes"] == ["TS900"]
    assert cli_main(["conform", "tpu", d, "--runtimes", "no_such", "--format", "agent"]) == 2


@pytest.mark.parametrize("case", sorted(B.CASES))
def test_every_case_has_a_budget(case):
    m = {"rel_l2": 0.0, "max_abs": 0.0, "max_abs_over_rms": 0.0, "max_rel": 0.0, "nonfinite": 0, "eff_bits": 60.0, "max_ulp": 0.0}
    verdict, budget, issues = T.judge(case, m, "x")
    assert verdict == "ok" and budget and not issues
