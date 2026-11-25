"""Integer reference (docs/INTREF.md): exact primitives, conventions, matching, NPU explanation,
C golden vectors."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from truestar.intref import fixedpoint as fp
from truestar.intref import golden, npu
from truestar.intref.cases import MUL_DYADIC, cmsis_214, onnxruntime_cases
from truestar.intref.conventions import CONVENTIONS, RULES, documented_conventions
from truestar.intref.match import (ROUNDING_MINORITY_BOUND, distinguishing_case, match_op,
                                   pair_signature, rounding_only_test)
from truestar.intref.ops import conv2d, conv_accumulator, quantize, run_op

RNG = np.random.default_rng(1234)


# ----------------------------------------------------------------------------- scalar transcriptions


def _c_div(a: int, b: int) -> int:  # C integer division, truncation toward zero
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _srdhm(a: int, b: int) -> int:
    if a == b == -(1 << 31):
        return (1 << 31) - 1
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    return _c_div(ab + nudge, 1 << 31)


def _rdbpot(x: int, e: int) -> int:
    mask = (1 << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if rem > thr else 0)


def test_hand_values():
    assert fp.srdhm(-1, 1 << 30) == 0 and fp.srdhm(1, 1 << 30) == 1 and fp.srdhm(-3, 1 << 30) == -1
    assert fp.srdhm(-(1 << 31), -(1 << 31)) == (1 << 31) - 1
    assert fp.rdbpot(3, 1) == 2 and fp.rdbpot(-3, 1) == -2 and fp.rdbpot(-5, 2) == -1
    # single rounding: 5 * 2^30 / 2^32 = 1.25 -> 1; 6 * 2^30 / 2^32 = 1.5 -> 2; -1.5 -> -1 (ties up)
    assert fp.mbqm_single(5, 1 << 30, -1) == 1 and fp.mbqm_single(6, 1 << 30, -1) == 2
    assert fp.mbqm_single(-6, 1 << 30, -1) == -1
    assert fp.mbqm_exact(-6, 1 << 30, -1, "away") == -2 and fp.mbqm_exact(10, 1 << 30, -1, "even") == 2


def test_quantize_multiplier_matches_definition():
    for v in list(10 ** RNG.uniform(-9, 1, 300)) + [1.0, 0.5, 2.0 ** -31, 2.0 ** -33, 0.9999999999]:
        m0, sh = fp.quantize_multiplier(v)
        m0, sh = int(m0), int(sh)
        if v < 2.0 ** -32:
            assert (m0, sh) == (0, 0)
            continue
        assert (1 << 30) <= m0 < (1 << 31)
        assert abs(m0 * 2.0 ** (sh - 31) - v) <= 2.0 ** (sh - 32) * 1.0000001


def test_vectorized_matches_scalar_transcription():
    a = np.concatenate([RNG.integers(-(1 << 31), 1 << 31, 3000), [-(1 << 31), (1 << 31) - 1, 0, -1, 1]])
    b = np.concatenate([RNG.integers(0, 1 << 31, 3000), [-(1 << 31), (1 << 30), 1, 1 << 30, 1 << 30]])
    got = fp.srdhm(a, b)
    assert all(int(g) == _srdhm(int(x), int(y)) for g, x, y in zip(got, a, b))
    x = RNG.integers(-(1 << 31), 1 << 31, 3000)
    e = RNG.integers(0, 31, 3000)
    assert all(int(g) == _rdbpot(int(xi), int(ei)) for g, xi, ei in zip(fp.rdbpot(x, e), x, e))


def test_library_equivalences_measured_not_assumed():
    """gemmlowp and CMSIS-NN high multiplies agree for non negative multipliers (also on exact
    ties); both single rounding spellings agree; exact half up is single rounding."""
    a = np.concatenate([RNG.integers(-(1 << 31), 1 << 31, 200000),
                        RNG.integers(-(1 << 20), 1 << 20, 50000) * 2 + 1])
    b = np.concatenate([RNG.integers(0, 1 << 31, 200000), np.full(50000, 1 << 30)])
    assert np.array_equal(fp.srdhm(a, b), fp.cmsis_dhm_no_sat(a, b))
    x = RNG.integers(-(1 << 31), 1 << 31, 200000)
    m = RNG.integers(1 << 30, 1 << 31, 200000)
    s = RNG.integers(-31, 1, 200000)
    single = fp.mbqm_single(x, m, s)
    assert np.array_equal(single, fp.mbqm_single_cmsis_form(x, m, s))
    assert np.array_equal(single, fp.mbqm_exact(x, m, s, "up"))


def test_rule_pairs_differ_one_way_within_bound():
    """The rounding only test rests on this: two documented rules on the same accumulators differ
    in the minority direction on at most ROUNDING_MINORITY_BOUND of elements, per sign."""
    docs = [r for r in RULES if RULES[r].documented]
    worst = 0.0
    for real in list(10 ** RNG.uniform(-5, 0, 12)) + [0.5, 2.0 ** -7, 0.0036111548234719986]:
        acc = np.arange(1, int(min(256 / real, 1 << 20)), dtype=np.int64)
        outs = {r: RULES[r].requant(acc, real) for r in docs}
        for i, ra in enumerate(docs):
            for rb in docs[i + 1:]:
                d = outs[ra] - outs[rb]
                worst = max(worst, min((d > 0).sum(), (d < 0).sum()) / acc.size)
    assert worst < ROUNDING_MINORITY_BOUND
    assert worst > 0  # the bound is not vacuous: some pairs do differ both ways, rarely


def test_conv_accumulator_exact_against_float64_torch():
    torch = pytest.importorskip("torch")
    x = RNG.integers(-128, 128, (2, 4, 9, 7)).astype(np.int8)
    w = RNG.integers(-127, 128, (6, 2, 3, 3)).astype(np.int8)
    b = RNG.integers(-1000, 1000, 6).astype(np.int32)
    acc = conv_accumulator(x, w, b, -3, 0, stride=2, padding=1, groups=2)
    ref = torch.nn.functional.conv2d(torch.tensor(x.astype(np.float64) + 3), torch.tensor(w.astype(np.float64)),
                                     torch.tensor(b.astype(np.float64)), stride=2, padding=1, groups=2)
    assert np.array_equal(acc, ref.numpy().astype(np.int64))


def test_match_names_generator_on_dyadic_scales():
    """On inputs that reach exact ties, each documented convention is told apart from every one
    that rounds differently; the four classes are double rounding, single half up, half even and
    half away."""
    classes = [{"tflite_ref", "tflite_2_21_py", "cmsis_nn"}, {"tflite_single", "cmsis_nn_single"},
               {"onnx_qlinear", "exact_half_even"}, {"exact_half_away"}]
    case = distinguishing_case("mul", MUL_DYADIC)
    for name in documented_conventions():
        dev = run_op("mul", case, MUL_DYADIC, name)
        v = match_op("mul", case, MUL_DYADIC, dev)
        cls = next(c for c in classes if name in c)
        assert v.code == "TS540" and set(v.matched) - {"npu_fp16_input"} == cls, (name, v.matched)


def test_cmsis_214_per_op_verdict():
    r = cmsis_214()
    s = r["summary"]
    assert "tflite_single" in s["fully_connected_matches"] and "tflite_ref" not in s["fully_connected_matches"]
    assert "tflite_ref" in s["mul_matches"] and "tflite_single" not in s["mul_matches"]
    assert s["one_cmsis_build_matches_both"] is False


def test_indistinguishable_and_no_match_codes():
    case = {"x": np.arange(-128, 128).astype(np.int8)}
    p = dict(x_zp=0, sx=0.5, sy=1.0)  # multiplier 0.5: every rule may tie; use max pool instead
    mp = {"x": RNG.integers(-128, 128, (1, 2, 6, 6)).astype(np.int8)}
    pp = {"kernel": 2, "stride": 2}
    v = match_op("max_pool", mp, pp, run_op("max_pool", mp, pp, None))
    assert v.code == "TS542"
    dev = run_op("requantize", case, p, "tflite_ref") + 1
    assert match_op("requantize", case, p, dev).code == "TS541"


def test_onnxruntime_oracle():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    o = onnxruntime_cases()
    assert "onnx_qlinear" in o["QLinearConv"]["matched"]
    assert "tflite_ref" not in o["QLinearConv"]["matched"]
    assert o["QuantizeLinear"]["matched"] == ["onnx_qlinear"]
    assert o["QLinearAdd"]["code"] == "TS541"  # matches no modelled convention: reported, not hidden


def _conv1_like(rng):
    x = rng.normal(0, 1.5, (2, 3, 24, 24)).astype(np.float32)
    w = rng.integers(-127, 128, (8, 3, 5, 5)).astype(np.int8)
    b = rng.integers(-500, 500, 8).astype(np.int64)
    p = {"x_zp": 126, "sx": 0.0367, "sw": rng.uniform(0.002, 0.004, 8), "sy": 0.03, "y_zp": 0,
         "padding": 2, "out_signed": False}
    return x, w, b, p


def test_rounding_only_test_accepts_rule_changes_and_rejects_input_changes():
    x, w, b, p = _conv1_like(RNG)
    qp = {"scale": p["sx"], "zp": 126, "out_signed": False}
    xq = quantize(x, qp, "onnx_qlinear")
    ctrl = conv2d(xq, w, b, p, "onnx_qlinear")
    for name in documented_conventions():
        sig = pair_signature(conv2d(xq, w, b, p, name), ctrl, min_count=1)
        assert rounding_only_test(sig)["rounding_only_possible"], name
    sig = pair_signature(conv2d(quantize(x, qp, "npu_fp16_input"), w, b, p, "onnx_qlinear"), ctrl)
    assert sig.rate > 0.01
    assert not rounding_only_test(sig)["rounding_only_possible"]


def test_npu_summary_verdict():
    e = npu.from_summary()
    assert e.code == "TS543" and e.best == "npu_fp16_input"
    assert abs(e.signature["rate"] - 0.0444) < 0.0005  # docs/NPU.md: 4.44% of relu elements
    assert e.maxpool["exact"] is True
    docs = [h for h in e.hypotheses if h.documented]
    assert all(h.distance > 5 * e.hypotheses[0].distance for h in docs)


@pytest.mark.skipif(not os.path.exists(os.path.join(npu.cache_dir(), "hub_cache.json")),
                    reason="AI Hub job cache not on this machine (the summary test covers the verdict)")
def test_npu_from_job_cache_matches_summary():
    try:
        e = npu.explain(simulate_inputs=1)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    s = npu.from_summary()
    assert e.signature["differ"] == s.signature["differ"] and e.code == s.code
    assert e.jobs_submitted == 0


def test_golden_header_roundtrip_and_check(tmp_path):
    x, specs = golden.demo_network()
    exact = golden.build_layers(x, specs, "cmsis_nn", "exact")
    port = golden.build_layers(x, specs, "cmsis_nn", "portable")
    assert all(L.tol_lsb == 0 for L in exact)
    assert any(L.tol_lsb > 0 for L in port) and all(L.tol_lsb <= 1 for L in port)
    text = golden.emit_header(port, "cmsis_nn", "portable")
    assert golden.verify_roundtrip(port, text) == []
    parsed = golden.parse_header(text)
    assert parsed["convention"] == "cmsis_nn" and len(parsed["layers"]) == len(specs)
    outs = {L.name: L.expected for L in port}
    assert all(v["bad"] == 0 for v in golden.check_outputs(parsed, outs).values())
    off = dict(outs, fc1=outs["fc1"].astype(np.int64) + 2)
    assert golden.check_outputs(parsed, off)["fc1"]["bad"] == 10
    # a single changed value breaks the checksum
    i = text.index("ts_l3_expected[10] = {") + len("ts_l3_expected[10] = {\n  ")
    j = text.index(",", i)
    bad = text[:i] + str(int(text[i:j]) ^ 1) + text[j:]
    with pytest.raises(ValueError, match="checksum"):
        golden.parse_header(bad)


def test_cli_golden_match_npu(tmp_path, capsys):
    from truestar.cli import main

    out = tmp_path / "g.h"
    assert main(["intref", "golden", "-o", str(out), "--convention", "tflite_ref", "--format", "agent"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["command"] == "intref" and doc["status"] == "ok" and out.exists()
    case = distinguishing_case("mul", MUL_DYADIC)
    dev = run_op("mul", case, MUL_DYADIC, "cmsis_nn_single")
    f = tmp_path / "case.npz"
    np.savez(f, op="mul", params=json.dumps(MUL_DYADIC), device=dev, **case)
    assert main(["intref", "match", str(f), "--format", "agent"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert "cmsis_nn_single" in doc["op"]["matched"] and doc["codes"] == ["TS540"]
    np.savez(f, op="mul", params=json.dumps(MUL_DYADIC), device=dev + 3, **case)
    assert main(["intref", "match", str(f)]) == 1
    capsys.readouterr()
    np.savez(f, op="mul", params=json.dumps(MUL_DYADIC), **case)
    assert main(["intref", "match", str(f), "--format", "agent"]) == 2
    assert json.loads(capsys.readouterr().out)["codes"] == ["TS544"]
    assert main(["intref", "npu", "--summary", "--format", "agent"]) == 1
    assert json.loads(capsys.readouterr().out)["codes"] == ["TS543"]
