"""truestar conform: deployed artifacts checked in their own runtimes (docs/CONFORM.md, row P31).

Runs in the ml venv (ONNX Runtime CPU, LiteRT) and in the cv venv (adds ONNX Runtime CUDA and
WebGPU, LiteRT GPU, ExecuTorch); tests for a runtime that is missing here are skipped by name.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import conform_models as cm  # noqa: E402
from truestar.conform import conform, replay  # noqa: E402
from truestar.conform import core as C  # noqa: E402
from truestar.conform import runtimes as R  # noqa: E402
from truestar.conform.artifact import ArtifactError, load as load_artifact  # noqa: E402
from truestar.conform.golden import Bundle, GoldenError  # noqa: E402


def _have(name):
    return R.make(name).unavailable() is None


def _need(*names):
    for n in names:
        why = R.make(n).unavailable()
        if why:
            pytest.skip(f"{n}: {why}")


def _can_export_onnx():
    """The ONNX files here are written by torch.onnx.export(dynamo=True), which imports onnxscript."""
    from truestar.deps import available

    return available("onnxruntime") and available("onnx") and available("onnxscript")


def _need_onnx_export():
    for mod in ("onnxruntime", "onnx", "onnxscript"):
        pytest.importorskip(mod, reason=f"{mod} is not installed (writing and running the test .onnx files needs it)")


def _codes(res):
    return {c for c, _ in res.issues}


@pytest.fixture(scope="module")
def clf():
    return cm.make_classifier(), cm.example()


@pytest.fixture(scope="module")
def onnx_files(tmp_path_factory, clf):
    _need_onnx_export()
    d = tmp_path_factory.mktemp("onnx")
    m, x = clf
    clean = cm.export_onnx(m, x, str(d / "clf.onnx"))
    return {"dir": d, "clean": clean,
            "gelu_tanh": cm.edit_onnx(clean, str(d / "clf_gelu_tanh.onnx"), "gelu_tanh"),
            "det": cm.edit_onnx(clean, str(d / "clf_det.onnx"), "det_node")}


@pytest.fixture(scope="module")
def tflite_files(tmp_path_factory):
    pytest.importorskip("ai_edge_litert")
    d = tmp_path_factory.mktemp("tflite")
    x = cm.example()
    out = {"dir": d}
    for name, kw, faults in [("clean", {}, ()), ("cumsum", {"cumsum": True}, ()),
                             ("eps", {}, ("layernorm_eps_outside",)), ("gelu", {}, ("gelu_tanh",))]:
        out[name] = cm.build_tflite(cm.make_classifier(**kw), x, str(d / f"{name}.tflite"), faults)
    return out


# ----------------------------------------------------------------------------- ONNX


def test_clean_onnx_conforms_and_writes_a_replayable_bundle(onnx_files, clf, tmp_path):
    m, x = clf
    gold = str(tmp_path / "clf.tsgold")
    rep = conform(m, x, onnx_files["clean"], runtimes=["ort:cpu"], golden=gold)
    r = rep.results[0]
    assert rep.status == "ok" and r.status == "ok", rep.to_text()
    assert r.flips == 0 and r.decisions == 8 * 64
    assert r.verdicts["conversion_ok"] and r.verdicts["conversion"] < C.CONVERSION_BAR
    assert r.residency["resident"] and r.outputs[0].beyond_bound == 0
    b = Bundle(gold)
    assert b.manifest["artifact"]["sha256"] == load_artifact(onnx_files["clean"]).sha256
    assert b.manifest["samples"] == 8 and "ort:cpu" in b.manifest["runtimes"]
    assert len(b.manifest["units"]) == 9  # inp, blocks, blocks.0, .fc, .ln, blocks.1, .fc, .ln, head
    # the numpy only checker shipped in the bundle accepts the runtime's outputs and refuses bad ones
    script = tmp_path / "check_outputs.py"
    script.write_bytes(zipfile.ZipFile(gold).read("check_outputs.py"))
    good = {f"s{k}_0": o[0] for k, o in enumerate(r.raw)}
    np.savez(tmp_path / "good.npz", **good)
    np.savez(tmp_path / "bad.npz", **{k: v * 1.01 for k, v in good.items()})
    ok = subprocess.run([sys.executable, str(script), gold, str(tmp_path / "good.npz")], capture_output=True, text=True)
    bad = subprocess.run([sys.executable, str(script), gold, str(tmp_path / "bad.npz")], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert bad.returncode == 1 and "FAIL" in bad.stdout


def test_exporter_decomposition_change_is_a_conversion_fault_at_the_module(onnx_files, clf):
    """gelu decomposed as the tanh approximation: under the float32 output limit at a glance
    (1.2e-4 against 1e-4), but the file's own float64 math names it and the module."""
    m, x = clf
    r = conform(m, x, onnx_files["gelu_tanh"], runtimes=["ort:cpu"]).results[0]
    assert r.status in ("suspect", "divergent")
    assert "TS625" in _codes(r)
    assert r.verdicts["conversion"] > 1e-5 and r.verdicts["runtime"] < 1e-6
    assert r.blamed is not None and r.blamed.module == "blocks.0" and r.blamed.cause == "conversion"
    assert "Tanh" in r.blamed.ops
    assert r.blamed.user_source["class"] == "Block"


def test_int8_rounding_change_flips_decisions_and_is_blamed_on_the_quantizer(tmp_path):
    """Round to nearest replaced by floor in one quantize step (q2)."""
    _need_onnx_export()
    torch.manual_seed(0)
    m, x = cm.QuantClassifier().eval(), cm.example()
    clean = cm.export_onnx(m, x, str(tmp_path / "q.onnx"))
    bad = cm.edit_onnx(clean, str(tmp_path / "q_floor.onnx"), "int8_floor")
    ok = conform(m, x, clean, runtimes=["ort:cpu"]).results[0]
    assert ok.status == "ok" and ok.flips == 0, ok.issues
    r = conform(m, x, bad, runtimes=["ort:cpu"]).results[0]
    assert r.status == "divergent" and "TS624" in _codes(r) and r.flips - r.tie_flips > 5
    assert r.blamed.module == "q2" and r.blamed.cause == "conversion"
    assert "Floor" in r.blamed.ops


def test_ort_gpu_providers_name_the_node_that_fell_back(onnx_files, clf):
    names = [n for n in ("ort:cuda", "ort:webgpu") if _have(n)]
    if not names:
        pytest.skip("no ONNX Runtime GPU provider here (needs the cv venv)")
    m, x = clf
    clean = conform(m, x, onnx_files["clean"], runtimes=names)
    for r in clean.results:
        assert r.status == "ok", (r.runtime, r.issues)  # ort:cuda: TF32 judged as TF32, no alarm
        assert r.residency["resident"] and all(o["trusted"] for o in r.observer)
    det = conform(m, x, onnx_files["det"], runtimes=names)
    for r in det.results:
        assert "TS628" in _codes(r)
        off = r.residency["off_target"]
        assert [p["op"] for p in off] == ["Det"] and off[0]["module"] == "head"
        assert off[0]["where"] == "CPUExecutionProvider"


def test_ort_cuda_tf32_is_reported_as_tf32(onnx_files, clf):
    _need("ort:cuda")
    m, x = clf
    r = conform(m, x, onnx_files["clean"], runtimes=["ort:cuda"]).results[0]
    if not r.info.get("tf32"):
        pytest.skip("ORT CUDA has TF32 off here")
    assert r.precision["observed"] == "tf32" and "TS627" not in _codes(r)


# ----------------------------------------------------------------------------- TFLite


def test_tflite_writer_float64_meaning_matches_the_model(tflite_files):
    art = load_artifact(tflite_files["clean"])
    m, x = cm.make_classifier(), cm.example()
    ref = m.double()(x.double()).detach().numpy()
    got = art.meaning64({"x": x.numpy()}, art.output_names())[0]
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-10
    names = [u.module for u in art.units()]
    assert names[:3] == ["inp", "blocks", "blocks.0"] and "blocks.1.ln" in names


def test_tflite_cpu_runtimes_conform_on_the_clean_file(tflite_files):
    names = ["tflite:interp", "tflite:ref", "tflite:cpu"]
    _need(*names)
    rep = conform(cm.make_classifier(), cm.example(), tflite_files["clean"], runtimes=names)
    for r in rep.results:
        assert r.status == "ok" and not r.issues, (r.runtime, r.issues)
        assert r.residency["resident"]


def test_tflite_layernorm_decomposition_change_is_blamed_on_the_layernorm(tflite_files):
    _need("tflite:interp")
    r = conform(cm.make_classifier(), cm.example(), tflite_files["eps"], runtimes=["tflite:interp"]).results[0]
    assert "TS625" in _codes(r)
    assert r.blamed.module == "blocks.0.ln" and r.blamed.cause == "conversion"
    assert "SQRT" in r.blamed.ops and "DIV" in r.blamed.ops


class _Int8Mlp(torch.nn.Module):
    """Two layers whose module paths are the tensor prefixes of the INT8 file below."""

    def __init__(self, layers):
        super().__init__()
        lin = []
        for L in layers:
            m = torch.nn.Linear(L["w"].shape[1], L["w"].shape[0])
            with torch.no_grad():
                m.weight.copy_(torch.from_numpy(L["w"].astype(np.float64) * L["w_scale"][:, None]))
                m.bias.copy_(torch.from_numpy(L["bias_float"]))
            lin.append(m)
        self.fc = torch.nn.Sequential(lin[0], torch.nn.ReLU())
        self.head = lin[1]

    def forward(self, x):
        return self.head(self.fc(x))


def _int8_mlp(rows=4, dims=(8, 6, 3), seed=0):
    """(model, example, .tflite bytes, layers): a tiny full integer file with per channel int8 weights."""
    pytest.importorskip("ai_edge_litert")
    from truestar.conform.tflite_build import build_int8_mlp

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.uniform(-1, 1, (rows, dims[0]))).float()
    sx, zx = 8.0 / 255, 0     # inputs in [-4, 4]: conform's other samples are normal draws
    layers, s_in, h = [], sx, rng.standard_normal((512, dims[0]))   # calibration draw
    for j, (k, n) in enumerate(zip(dims[:-1], dims[1:])):
        wf = rng.standard_normal((n, k)) / np.sqrt(k)
        sw = (np.abs(wf).max(1) / 127).astype(np.float32)
        wq = np.round(wf / sw[:, None]).astype(np.int8)
        bf = rng.standard_normal(n) * 0.1
        bq = np.round(bf / (np.float64(np.float32(s_in)) * sw)).astype(np.int32)
        relu = j < len(dims) - 2
        h = h @ (wq * sw[:, None].astype(np.float64)).T + bf
        h = np.maximum(h, 0) if relu else h
        lo, hi = min(h.min(), 0.0), max(h.max(), 0.0)
        sy = float(np.float32((hi - lo) * 1.5 / 255))
        layers.append({"name": "fc" if j == 0 else "head", "w": wq, "w_scale": sw, "bias": bq,
                       "bias_float": bq * (np.float64(np.float32(s_in)) * sw), "y_scale": sy,
                       "y_zero_point": int(np.clip(round(-lo * 1.5 / sy) - 128, -128, 127)), "relu": relu})
        s_in = sy
    data = build_int8_mlp(layers, (rows, dims[0]), sx, zx, taps=False)
    return _Int8Mlp(layers), x, data, layers


def test_int8_per_channel_tflite_meaning_is_its_integer_arithmetic():
    """The float64 meaning of a full integer file is the integer arithmetic it asks for, in real units."""
    from truestar.conform.tflite_graph import TfliteGraph

    m, x, data, layers = _int8_mlp()
    g = TfliteGraph(data)
    assert [op.name for op in g.ops] == ["QUANTIZE", "FULLY_CONNECTED", "FULLY_CONNECTED", "DEQUANTIZE"]
    w = g.ops[1].inputs[1]
    s, zp, axis = g.quant(w)
    assert axis == 0 and s.shape == (6,) and np.array_equal(g.constant(w), layers[0]["w"])
    got = g.evaluate64({g.inputs[0]: x.numpy()}, g.outputs)[0]
    sx, zx = 8.0 / 255, 0
    q = np.clip(np.sign(x.numpy() / np.float32(sx)) * np.floor(np.abs(x.numpy() / np.float32(sx)) + 0.5) + zx,
                -128, 127).astype(np.int64)
    s_in, z_in = np.float64(np.float32(sx)), zx
    for L in layers:
        acc = (q - z_in) @ L["w"].astype(np.int64).T + L["bias"]
        r = acc * (s_in * L["w_scale"].astype(np.float64)) / np.float64(np.float32(L["y_scale"]))
        q = np.sign(r) * np.floor(np.abs(r) + 0.5) + L["y_zero_point"]
        q = np.clip(q, L["y_zero_point"] if L["relu"] else -128, 127).astype(np.int64)
        s_in, z_in = np.float64(np.float32(L["y_scale"])), L["y_zero_point"]
    assert np.array_equal(got, (q - z_in) * s_in)


def test_int8_per_channel_tflite_conforms_as_quantized_not_as_a_runtime_fault(tmp_path):
    """P45: a full integer .tflite with per channel weights runs, and conform tells the quantization
    (the file's own math) from the runtime: before, QUANTIZE and DEQUANTIZE had no float64 rule and
    int8 cut outputs were compared as raw codes, so a correct runtime was blamed (TS626, own error inf)."""
    _need("tflite:interp")
    m, x, data, _ = _int8_mlp()
    path = tmp_path / "int8.tflite"
    path.write_bytes(data)
    r = conform(m, x, str(path), runtimes=["tflite:interp"]).results[0]
    codes = _codes(r)
    assert "TS626" not in codes and "TS638" not in codes and "TS636" not in codes, r.issues
    assert r.verdicts["runtime"] < 1e-6                         # the runtime computes the file
    assert r.verdicts["conversion"] > r.verdicts["conversion_bar"] and "TS625" in codes   # quantization
    assert r.blamed is not None and r.blamed.cause == "conversion" and r.blamed.module == "fc"
    assert r.blamed.own_error < 0.05 and r.verdicts["conversion"] < 0.05   # 8 bit steps, no clipping
    # the cut, fed codes on its quantized input, computes exactly the file's arithmetic
    assert abs(r.blamed.conversion_error - r.blamed.own_error) < 1e-6


def test_tflite_ops_outside_the_delegate_are_named(tflite_files):
    names = ["tflite:interp", "tflite:cpu"]
    _need(*names)
    rep = conform(cm.make_classifier(cumsum=True), cm.example(), tflite_files["cumsum"], runtimes=names)
    for r in rep.results:
        assert r.status == "ok"
        assert "TS628" in _codes(r)
        assert [p["op"] for p in r.residency["off_target"]] == ["CUMSUM"], r.residency


def test_tflite_gpu_default_precision_is_float16(tflite_files):
    _need("tflite:gpu", "tflite:gpu-f32")
    rep = conform(cm.make_classifier(), cm.example(), tflite_files["clean"],
                  runtimes=["tflite:gpu", "tflite:gpu-f32"])
    gpu, f32 = rep.results
    assert gpu.precision["observed"] == "float16" and "TS627" in _codes(gpu)
    assert gpu.verdicts["conversion_ok"] and gpu.blamed is None and "TS626" not in _codes(gpu)
    assert f32.status == "ok" and f32.precision["observed"] == "float32"


def test_observer_guard_flags_an_instrumented_run_that_changes_outputs(tflite_files):
    _need("tflite:interp")

    def nudge(outs):
        return [o + np.float32(1e-3) for o in outs]

    m, x = cm.make_classifier(), cm.example()
    clean = conform(m, x, tflite_files["gelu"], runtimes=["tflite:interp"]).results[0]
    assert clean.observer and clean.observer[0]["trusted"] and "TS630" not in _codes(clean)
    bad = conform(m, x, tflite_files["gelu"], runtimes=["tflite:interp"],
                  options={"tflite:interp": {"fault_instrumented": nudge}}).results[0]
    assert "TS630" in _codes(bad) and not bad.observer[0]["trusted"]


def test_observer_check_uses_run_to_run_noise():
    a = [np.zeros((4, 3), np.float32)]
    noisy = [a[0] + 1e-6]
    assert C.observer_check("x", a, noisy, [a[0] + 5e-7], 1e-4)["trusted"]
    ob = C.observer_check("x", a, a, [np.eye(4, 3, dtype=np.float32)], 1e-4)
    assert not ob["trusted"] and ob["decisions_changed"] == 2  # rows 1 and 2 now pick another class


# ----------------------------------------------------------------------------- ExecuTorch, .tsx


def test_pte_names_the_kernels_outside_the_delegate(tmp_path):
    _need("pte:cpu")
    pytest.importorskip("executorch.backends.xnnpack")
    m, x = cm.make_classifier(cumsum=True), cm.example()
    p = cm.build_pte(m, x, str(tmp_path / "m.pte"))
    r = conform(m, x, p).results[0]
    assert r.status == "ok" and "TS628" in _codes(r)
    off = {(o["op"].split("::")[-1].split(".")[0], o["module"]) for o in r.residency["off_target"]}
    assert ("cumsum", "") in off and ("native_layer_norm", "blocks.0.ln") in off


def test_tsx_reference_executor_conforms(tmp_path):
    from truestar.export import export

    m, x = cm.make_classifier(), cm.example()
    p = str(tmp_path / "m.tsx")
    export(m, x, p)
    r = conform(m, x, p, runtimes=["tsx:reference"]).results[0]
    assert r.status == "ok" and r.verdicts["conversion_ok"]


# ----------------------------------------------------------------------------- baseline


def test_replay_identical_then_injected_change_names_versions_and_module(onnx_files, clf, tmp_path, monkeypatch):
    m, x = clf
    gold = str(tmp_path / "base.tsgold")
    conform(m, x, onnx_files["clean"], runtimes=["ort:cpu"], golden=gold)
    res, top = replay(gold, onnx_files["clean"])
    assert not top and res[0].verdict == "identical" and "TS639" in {c for c, _ in res[0].issues}
    assert not res[0].fingerprint_changes

    old = R.OrtRuntime._versions
    monkeypatch.setattr(R.OrtRuntime, "_versions", lambda self: dict(old(self), onnxruntime="9.9.9"))

    def scale(outs):  # an upgrade that changes every result by 1 percent
        return [o * np.float32(1.01) for o in outs]

    res, _ = replay(gold, onnx_files["clean"], options={"ort:cpu": {"fault_outputs": scale}})
    r = res[0]
    assert r.verdict == "regressed" and not r.conformant and "TS633" in {c for c, _ in r.issues}
    assert any(k == "runtime.onnxruntime" and b == "9.9.9" and kd == "version"
               for k, a, b, kd in r.fingerprint_changes)
    assert r.first_changed_module["module"] == "inp"


def test_replay_refuses_other_artifact_and_damaged_bundle(onnx_files, clf, tmp_path):
    m, x = clf
    gold = str(tmp_path / "b.tsgold")
    conform(m, x, onnx_files["clean"], runtimes=["ort:cpu"], golden=gold)
    res, top = replay(gold, onnx_files["gelu_tanh"])
    assert res == [] and top[0][0] == "TS635"
    with zipfile.ZipFile(gold) as z:
        items = {n: z.read(n) for n in z.namelist()}
    name = "expected/s0_0.npy"
    data = bytearray(items[name])
    data[-1] ^= 1
    items[name] = bytes(data)
    bad = str(tmp_path / "bad.tsgold")
    with zipfile.ZipFile(bad, "w") as z:
        for n, v in items.items():
            z.writestr(n, v)
    with pytest.raises(GoldenError, match="damaged"):
        Bundle(bad)


# ----------------------------------------------------------------------------- refusals and CLI


def test_coreml_is_refused_with_the_reason(tmp_path):
    d = tmp_path / "m.mlpackage"
    d.mkdir()
    (d / "Manifest.json").write_text("{}")
    with pytest.raises(ArtifactError, match="macOS"):
        load_artifact(str(d))


@pytest.mark.parametrize("module,package,name,head", [
    ("ai_edge_litert", "ai-edge-litert", "m.tflite", b"\0\0\0\0TFL3" + b"\0" * 8),
    ("onnx", "onnx", "m.onnx", b"\x08\x07" + b"\0" * 14),
    ("executorch", "executorch", "m.pte", b"\0\0\0\0ET12" + b"\0" * 8),
])
def test_missing_runtime_module_is_ts022_with_the_install_command(tmp_path, capsys, monkeypatch,
                                                                    module, package, name, head):
    """A format whose reader is not installed is a usage error that names the module and the pip
    command (TS022), never a raw ModuleNotFoundError, whether or not the module exists here."""
    from truestar.cli import main
    from truestar.deps import MissingDependencyError

    for m in [k for k in sys.modules if k == module or k.startswith(module + ".")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setitem(sys.modules, module, None)  # import now raises ImportError
    art = tmp_path / name
    art.write_bytes(head)
    with pytest.raises(MissingDependencyError) as exc:
        load_artifact(str(art))
    assert exc.value.code == "TS022" and f"pip install {package}" in str(exc.value)
    mf = cm.write_model_file(str(tmp_path / "model.py"))
    code = main(["conform", mf, str(art), "--format", "agent", "--no-golden"])
    doc = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code == 2 and doc["codes"] == ["TS022"], doc
    assert doc["next_actions"][0]["command"] == f"pip install {package}"


def test_cli_agent_json_and_exit_codes(onnx_files, tmp_path, capsys):
    from truestar.cli import main

    mf = cm.write_model_file(str(tmp_path / "model.py"))
    code = main(["conform", mf, onnx_files["gelu_tanh"], "--runtimes", "ort:cpu", "--format", "agent",
                 "--golden", str(tmp_path / "g.tsgold")])
    doc = json.loads(capsys.readouterr().out)
    from truestar import codes

    assert code == 1 and doc["command"] == "conform" and "TS625" in doc["codes"]
    assert all(c in codes.CODES for c in doc["codes"])
    assert doc["runtimes"][0]["blamed"]["module"] == "blocks.0"
    kinds = [a["kind"] for a in doc["next_actions"]]
    assert "fix_conversion" in kinds and "replay_after_upgrade" in kinds
    code = main(["conform", "replay", str(tmp_path / "g.tsgold"), onnx_files["gelu_tanh"], "--format", "agent"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 1 and doc["runtimes"][0]["verdict"] == "identical"  # identical, but not conformant
    code = main(["conform", mf, str(tmp_path / "missing.onnx"), "--format", "agent"])
    assert code == 2 and "TS620" in json.loads(capsys.readouterr().out)["codes"]
    if not _have("ort:cuda"):  # asked for by name and missing: exit 2, nothing passes in its place
        code = main(["conform", mf, onnx_files["clean"], "--runtimes", "ort:cpu,ort:cuda", "--format", "agent",
                     "--no-golden"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 2 and "TS621" in doc["codes"]
    code = main(["conform", mf, onnx_files["clean"], "--runtimes", "tflite:cpu", "--format", "agent"])
    assert code == 2 and "TS021" in json.loads(capsys.readouterr().out.splitlines()[-1])["codes"]


def test_localization_is_needed_for_the_blame(onnx_files, clf, monkeypatch):
    """Guard for the guard: with the cut runs reporting no error, the blame test above fails."""
    m, x = clf

    def blind(self, rt, u):
        return {"module": u.module, "own_error": 0.0, "threshold": 1e-4, "status": "ok",
                "conversion_error": 0.0, "conversion_bar": 1e-7}

    monkeypatch.setattr(C._Localizer, "own", blind)
    r = conform(m, x, onnx_files["gelu_tanh"], runtimes=["ort:cpu"]).results[0]
    assert r.blamed is None and "TS631" in _codes(r)


def test_no_false_alarms_on_clean_files(tmp_path):
    """Clean artifacts of three models on every CPU runtime of their format report ok."""
    models = [("clf", cm.make_classifier()), ("clf_cumsum", cm.make_classifier(cumsum=True, seed=3)),
              ("quant", (torch.manual_seed(2), cm.QuantClassifier().eval())[1])]
    x = cm.example(seed=5)
    checked = 0
    for name, m in models:
        if _have("ort:cpu") and _can_export_onnx():
            p = cm.export_onnx(m, x, str(tmp_path / f"{name}.onnx"))
            r = conform(m, x, p, runtimes=["ort:cpu"], samples=16).results[0]
            assert r.status == "ok" and r.flips == 0, (name, r.issues)
            checked += 1
        if _have("tflite:interp") and name != "quant":
            p = cm.build_tflite(m, x, str(tmp_path / f"{name}.tflite"))
            for r in conform(m, x, p, runtimes=["tflite:interp", "tflite:cpu"], samples=16).results:
                assert r.status == "ok" and r.flips == 0, (name, r.runtime, r.issues)
                checked += 1
    if not checked:
        pytest.skip("no CPU runtime here")


def test_onnx_scope_names_nested_modules_once():
    """name_scopes hold cumulative paths; both readers (conform and the ORT stack backend) must give
    'blocks.0.fc', not the joined 'blocks.blocks.0.blocks.0.fc'."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    from truestar import stacks
    from truestar.conform.artifact import _scope_module

    n = helper.make_node("Gemm", ["x", "w"], ["y"], name="node_linear")
    p = n.metadata_props.add()
    p.key, p.value = "pkg.torch.onnx.name_scopes", "['', 'blocks', 'blocks.0', 'blocks.0.fc', 'linear']"
    assert _scope_module(n) == "blocks.0.fc"
    assert stacks._module_path(n) == "blocks.0.fc"
