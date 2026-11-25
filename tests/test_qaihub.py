"""truestar conform qaihub: saved AI Hub outputs judged here, no network (docs/QAIHUB.md, row P45).

Bundles are written with outputs built from the references themselves: float32 eager (a float32
device), sim:ieee-f16 (correct float16), sim:mali-fp16 (a float16 accumulator, beyond correct
float16), the intended INT8 arithmetic, another intref convention, and planted faults.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
import torch

from truestar.cli import main as cli_main
from truestar.conform import qaihub as Q
from truestar.conform import qaihub_battery as B

PINNED = {  # the same bytes on Windows and Linux (numpy PCG64 random(), exact scaling)
    "float": "b8ca020283ad34700e98352181ce8d1dbe626d7d559262ce62222cc4a6cbe5b5",
    "int8": "db0b0d2af11d330f8bb97371221edd0a7884c1b58fad6d1318e9d9b40fa08917",
}
WEIGHTS = "b8233febfe9174c042db24990c73950d41ea7cd410777b27f817af9127f30221"


@pytest.fixture(scope="module")
def refs():
    arrs = B.inputs("float")
    return {"f32": B.float_reference(arrs, dtype=torch.float32), "f16": B.simulated("sim:ieee-f16", arrs),
            "acc": B.simulated("sim:mali-fp16", arrs), "int8": B.int8_intended()}


def _row(model, unit="npu", **kw):
    return {"model": model, "device": "Test Phone", "runtime": "tflite", "unit": unit, **kw}


def _ai_hub_names(outs, model):
    """AI Hub's TFLite targets rename outputs output_0, output_1, ... in ONNX order."""
    return {f"output_{i}": outs[t].astype(np.float32) for i, t in enumerate(B.tap_names(model))}


def test_inputs_and_weights_are_pinned():
    for m, h in PINNED.items():
        assert B.input_hash(B.inputs(m)) == h
    h = hashlib.sha256()
    for k, v in sorted(B.float_model().state_dict().items()):
        h.update(k.encode())
        h.update(v.numpy().tobytes())
    for L in B.int8_params()["layers"]:
        h.update(L["w"].tobytes())
        h.update(L["bias"].tobytes())
        h.update(np.float64(L["sy"]).tobytes())
    assert h.hexdigest() == WEIGHTS


def test_float_verdicts(tmp_path, refs):
    onnx_names = {B._onnx_name(t): v.astype(np.float32) for t, v in refs["f32"].items()}
    d = Q.write(str(tmp_path / "r"), {
        "cpu": (_row("float", "cpu"), onnx_names),
        "gpu_on_cpu": (_row("float", "gpu", layer_units={"CPU": 61}), _ai_hub_names(refs["f32"], "float")),
        "npu_f16": (_row("float"), _ai_hub_names(refs["f16"], "float")),
        "npu_acc16": (_row("float"), _ai_hub_names(refs["acc"], "float")),
    })
    got = {v.run: v for v in Q.check(d)[0]}
    assert got["cpu"].verdict == "float32" and not got["cpu"].issues
    assert got["gpu_on_cpu"].verdict == "float32"
    assert [c for c, _ in got["gpu_on_cpu"].issues] == ["TS925"] and "CPU 61" in got["gpu_on_cpu"].issues[0][1]
    assert got["npu_f16"].verdict == "float16" and [c for c, _ in got["npu_f16"].issues] == ["TS923"]
    assert all(abs(t.metrics["ratio_to_sim_f16"] - 1.0) < 1e-9 for t in got["npu_f16"].taps)
    assert got["npu_acc16"].verdict == "beyond_float16"
    assert [c for c, _ in got["npu_acc16"].issues] == ["TS924"]


def test_nhwc_taps_are_read_back(tmp_path, refs):
    outs = {t: v.astype(np.float32) for t, v in refs["f16"].items()}
    outs = {B._onnx_name(t): (v.transpose(0, 2, 3, 1) if v.ndim == 4 and t.startswith("conv") else v)
            for t, v in outs.items()}
    d = Q.write(str(tmp_path / "r"), {"nhwc": (_row("float"), outs)})
    (v,), _ = Q.check(d)
    assert v.verdict == "float16"


def test_one_wrong_element_is_beyond_float16(tmp_path, refs):
    outs = {t: v.astype(np.float32).copy() for t, v in refs["f16"].items()}
    outs["conv.stem"][0, 3, 10, 10] += 5.0
    d = Q.write(str(tmp_path / "r"), {"bad": (_row("float"), _ai_hub_names(outs, "float"))})
    (v,), _ = Q.check(d)
    bad = [t.tap for t in v.taps if t.verdict == "beyond_float16"]
    assert bad[0] == "conv.stem" and v.issues[0][0] == "TS924"


def _next_layer(prev, P, j):
    """Layer j of the INT8 MLP computed by intref from a given (dequantized) previous layer output."""
    from truestar.intref import ops as I

    a, L = P["layers"][j - 1], P["layers"][j]
    q = np.rint(prev.astype(np.float64) / a["sy"]).astype(np.int64) + a["zy"]
    p = {"x_zp": a["zy"], "sx": a["sy"], "sw": L["sw"].astype(np.float64), "sy": L["sy"], "y_zp": L["zy"],
         "out_signed": False}
    if L["relu"]:
        p["act_min"] = L["zy"]
    return (I.fully_connected(q, L["w"], L["bias"], p, "onnx_qlinear") - L["zy"]) * L["sy"]


def test_int8_verdicts(tmp_path, refs):
    P = B.int8_params()
    clean = {t: v.astype(np.float32) for t, v in refs["int8"].items()}
    one = {t: v.copy() for t, v in clean.items()}
    one["int8.out"][0, :5] += np.float32(P["layers"][2]["sy"])  # five elements one step up
    three = {t: v.copy() for t, v in clean.items()}
    three["int8.l2"][2, 7] += np.float32(3 * P["layers"][1]["sy"])
    three["int8.out"] = _next_layer(three["int8.l2"], P, 2).astype(np.float32)  # the device's own layer 3
    fp16_in = {t: v.astype(np.float32) for t, v in B.int8_intended(convention="npu_fp16_input").items()}
    d = Q.write(str(tmp_path / "r"), {
        "clean": (_row("int8"), _ai_hub_names(clean, "int8")),
        "one": (_row("int8"), _ai_hub_names(one, "int8")),
        "three": (_row("int8"), _ai_hub_names(three, "int8")),
        "fp16_in": (_row("int8"), _ai_hub_names(fp16_in, "int8")),
    })
    got = {v.run: v for v in Q.check(d)[0]}
    assert got["clean"].verdict == "int8_exact" and not got["clean"].issues
    assert all(t.verdict == "exact" for t in got["clean"].taps)
    assert got["one"].verdict == "int8_one_step" and [c for c, _ in got["one"].issues] == ["TS926"]
    l1, l2, out = got["one"].taps
    assert l1.verdict == l2.verdict == "exact"
    assert out.metrics["max_steps"] == 1 and out.metrics["frac_off"] == pytest.approx(5 / (16 * 10))
    assert got["three"].verdict == "divergent" and got["three"].issues[0][0] == "TS927"
    assert "int8.l2" in got["three"].issues[0][1]
    # layer 3 is replayed from the device's own (wrong) layer 2: judged on its own, it is not off,
    # while the chained comparison against the intended model shows the inherited difference
    l3 = got["three"].taps[2]
    assert l3.verdict == "exact" and l3.metrics["chained_frac_off"] > 0
    codes = [c for c, _ in got["fp16_in"].issues]
    assert "TS926" in codes and "TS928" in codes
    assert got["fp16_in"].taps[0].metrics["best_convention"] == "npu_fp16_input"


def test_failed_missing_and_hash(tmp_path, refs):
    outs = _ai_hub_names(refs["f32"], "float")
    del outs["output_12"]
    d = Q.write(str(tmp_path / "r"), {
        "failed": (_row("int8", error="compile failed: 'tfl.transpose' op has mismatched quantized axes"), None),
        "missing": (_row("float", "cpu"), outs),
    })
    got = {v.run: v for v in Q.check(d)[0]}
    assert got["failed"].verdict == "failed" and got["failed"].issues[0][0] == "TS922"
    assert got["missing"].verdict == "divergent"
    assert "TS929" in [c for c, _ in got["missing"].issues]
    p = os.path.join(d, "summary.json")
    s = json.load(open(p, encoding="utf-8"))
    s["input_hashes"]["float"] = "0" * 64
    json.dump(s, open(p, "w", encoding="utf-8"))
    verdicts, top = Q.check(d)
    assert [c for c, _ in top] == ["TS921"] and [v.run for v in verdicts] == ["failed"]


def test_cli_agent_json_and_usage(tmp_path, capsys, refs):
    d = Q.write(str(tmp_path / "r"), {"cpu": (_row("float", "cpu"), _ai_hub_names(refs["f32"], "float")),
                                      "npu": (_row("float"), _ai_hub_names(refs["f16"], "float"))})
    assert cli_main(["conform", "qaihub", d, "--format", "agent"]) == 0  # TS923 is a warning
    doc = json.loads(capsys.readouterr().out)
    assert doc["mode"] == "qaihub" and doc["status"] == "ok" and doc["codes"] == ["TS923"]
    assert [r["verdict"] for r in doc["runs"]] == ["float32", "float16"]
    assert cli_main(["conform", "qaihub", d, "--runtimes", "cpu", "--format", "agent"]) == 0
    assert [r["run"] for r in json.loads(capsys.readouterr().out)["runs"]] == ["cpu"]
    assert cli_main(["conform", "qaihub", str(tmp_path / "nothing"), "--format", "agent"]) == 2
    assert json.loads(capsys.readouterr().out)["codes"] == ["TS920"]
    assert cli_main(["conform", "qaihub", d, "--runtimes", "no_such", "--format", "agent"]) == 2
    capsys.readouterr()
    s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    s["battery_version"] = 99
    json.dump(s, open(os.path.join(d, "summary.json"), "w", encoding="utf-8"))
    assert cli_main(["conform", "qaihub", d, "--format", "agent"]) == 2


def test_int8_onnx_matches_intended_arithmetic():
    """Host control: ONNX Runtime runs both QDQ forms exactly as intref's ONNX QLinear convention."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    want = B.int8_intended()
    feed = B.inputs("int8")
    steps = B.int8_steps()
    for gemm in (False, True):
        for level in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            so = ort.SessionOptions()
            so.graph_optimization_level = level
            s = ort.InferenceSession(B.int8_onnx_bytes(gemm=gemm), so, providers=["CPUExecutionProvider"])
            names = [o.name for o in s.get_outputs()]
            for n, o in zip(names, s.run(names, feed)):
                t = B.tap_of_output(n)
                assert np.max(np.abs(o - want[t])) / steps[t] < 1e-3, (gemm, level, t)


def test_int8_tflite_runs_the_battery_with_per_channel_weights():
    """P45: AI Hub's ONNX to TFLite converter refuses the per channel QDQ battery, so the TFLite runs
    take a full integer .tflite written directly. Host control: LiteRT's CPU runtime gives every tap
    of it exactly as intref's ONNX QLinear convention computes the battery (uint8 activations
    rewritten as int8 with the zero point shifted by 128, the same grid), and its outputs carry the
    tap names that `match_outputs` reads."""
    pytest.importorskip("ai_edge_litert")
    from ai_edge_litert.interpreter import Interpreter

    from truestar.conform.tflite_graph import TfliteGraph

    data = B.int8_tflite_bytes()
    g = TfliteGraph(data)
    fcs = [op for op in g.ops if op.name == "FULLY_CONNECTED"]
    assert len(fcs) == 3 and "TRANSPOSE" not in {op.name for op in g.ops}
    for op in fcs:
        s, _, axis = g.quant(op.inputs[1])
        assert axis == 0 and s.size == g.shape(op.inputs[1])[0] > 1      # per output channel
    it = Interpreter(model_content=data)
    it.allocate_tensors()
    (inp,) = it.get_input_details()
    assert inp["name"] == "x_q"
    it.set_tensor(inp["index"], B.inputs("int8")["x_q"].astype(np.float32))
    it.invoke()
    got = {d["name"]: it.get_tensor(d["index"]) for d in it.get_output_details()}
    shapes = {t: (B.INT8_ROWS, n) for t, n in zip(B.INT8_TAPS, B.INT8_DIMS[1:])}
    taps = Q.match_outputs(got, "int8", shapes)
    assert sorted(taps) == sorted(B.INT8_TAPS)
    want, steps = B.int8_intended(), B.int8_steps()
    for t in B.INT8_TAPS:
        assert np.max(np.abs(taps[t] - want[t])) / steps[t] < 1e-3, t     # float32 outputs, no step off
    # the file's own float64 meaning is the same arithmetic
    mean = g.evaluate64({g.inputs[0]: B.inputs("int8")["x_q"]}, g.outputs)
    for t, m in zip(B.INT8_TAPS, mean):
        assert np.max(np.abs(m - want[t])) / steps[t] < 1e-6, t
