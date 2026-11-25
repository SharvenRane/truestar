"""truestar export --to onnx|litert|coreml|executorch (docs/EXPORT.md, "Deploy targets")."""

import json
import os
import sys
import textwrap

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_models as Z  # noqa: E402

from truestar.cli import main as cli_main  # noqa: E402
from truestar.export.deploy import DeployError, export_to  # noqa: E402
from truestar.export.deploy import onnx_writer  # noqa: E402

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


HAS_COREML_RUNTIME = sys.platform == "darwin" and __import__("importlib").util.find_spec("coremltools") is not None


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAS_LITERT = _has("ai_edge_litert")
HAS_ET = _has("executorch.runtime") and _has("executorch.exir")
ALL = ["mlp", "resnet18", "encoder_layer", "decoder"]


@pytest.mark.parametrize("name", ALL)
def test_onnx_written_and_verified_in_onnxruntime(name, tmp_path):
    m, x = Z.ZOO[name]()
    out = str(tmp_path / f"{name}.onnx")
    r = export_to(m, x, out, to="onnx")
    assert r.status == "verified" and r.written and "TS1410" in r.codes
    assert os.path.isfile(out) and os.path.isfile(out + ".truestar.json")
    assert [c["name"] for c in r.checks][-1] == "batch7"  # symbolic batch checked at another size
    assert all(c["passed"] for c in r.checks)
    assert r.conversion["passed"] and r.conversion["error"] < 1e-8
    assert r.used["kernels"] and r.used["residency"]["resident"]
    # the shipped file itself, run fresh, with a batch the checks did not use
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    assert s.get_inputs()[0].shape[0] == "B"
    x3 = tuple(t[:1].repeat(3, *([1] * (t.dim() - 1))) for t in x)
    got = s.run(None, {i.name: t.numpy() for i, t in zip(s.get_inputs(), x3)})[0]
    with torch.no_grad():
        want = m.double()(*[t.double() if t.is_floating_point() else t for t in x3]).numpy()
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-5
    ev = json.load(open(out + ".truestar.json", encoding="utf-8"))
    assert ev["verified"] and ev["program_hash"]


@pytest.mark.skipif(not HAS_LITERT, reason="ai-edge-litert not installed")
@pytest.mark.parametrize("name", ALL)
def test_litert_written_and_verified_in_the_interpreter(name, tmp_path):
    m, x = Z.ZOO[name]()
    r = export_to(m, x, str(tmp_path / f"{name}.tflite"), to="litert")
    assert r.status == "verified" and all(c["passed"] for c in r.checks)
    assert r.conversion["passed"]
    assert any(k.endswith("@XNNPACK") for k in r.used["kernels"])
    if name == "decoder":  # GATHER and SELECT_V2 stay on builtin kernels: reported, not hidden
        assert "TS1416" in r.codes
        assert {p["op"] for p in r.used["residency"]["off_target"]} == {"GATHER", "SELECT_V2"}


@pytest.mark.parametrize("name", ALL)
def test_coreml_written_with_conversion_verdict_and_pending_runtime(name, tmp_path):
    from truestar.export.deploy.coreml_writer import MODEL_PATH, decode

    m, x = Z.ZOO[name]()
    out = str(tmp_path / f"{name}.mlpackage")
    r = export_to(m, x, out, to="coreml")
    if HAS_COREML_RUNTIME:   # macOS with coremltools: the file runs in Core ML and is judged
        assert r.status == "verified" and "TS1413" not in r.codes and r.written, r.to_text()
        assert r.runtime == "coreml:cpu" and all(c["passed"] for c in r.checks)
        assert set(r.used["coreml"]) == {"CPU_ONLY", "CPU_AND_GPU", "ALL"}
    else:
        assert r.status == "pending" and "TS1413" in r.codes and r.written
    assert r.conversion["passed"] and r.conversion["error"] < 1e-8
    man = json.load(open(os.path.join(out, "Manifest.json"), encoding="utf-8"))
    entry = man["itemInfoEntries"][man["rootModelIdentifier"]]
    assert entry["path"] == "com.apple.CoreML/model.mlmodel"
    d = decode(open(os.path.join(out, *MODEL_PATH.split("/")), "rb").read())
    assert d["spec_version"] == 6 and d["opset"] == "CoreML5"
    assert [f["name"] for f in d["inputs"]] and d["block_outputs"]
    # Core ML's MIL parser takes identifiers only: a parameter name like fc1.weight did not load
    import re
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    names = [f["name"] for f in d["inputs"] + d["outputs"]] + d["block_outputs"]
    for op in d["ops"]:
        names += [x for xs in op[1].values() for x in xs] + [op[2]]
    assert names and all(ident.match(n) for n in names), [n for n in names if not ident.match(n)][:5]


@pytest.mark.skipif(not HAS_ET, reason="executorch not installed")
@pytest.mark.parametrize("name", ALL)
def test_executorch_written_and_verified_in_its_runtime(name, tmp_path):
    m, x = Z.ZOO[name]()
    r = export_to(m, x, str(tmp_path / f"{name}.pte"), to="executorch")
    assert r.status == "verified" and all(c["passed"] for c in r.checks)
    assert any(k.startswith("aten::") for k in r.used["kernels"])


def test_wrong_lowering_is_refused_and_nothing_written(tmp_path, monkeypatch):
    """A writer that emits gelu as its tanh approximation: the file's float64 math disagrees."""
    real = onnx_writer.write

    def bad(low):
        import dataclasses

        low.nodes = [dataclasses.replace(n, attrs=dict(n.attrs, approximate="tanh")) if n.op == "gelu" else n
                     for n in low.nodes]
        return real(low)

    monkeypatch.setattr(onnx_writer, "write", bad)
    m, x = Z.mlp()
    out = tmp_path / "m.onnx"
    with pytest.raises(DeployError) as e:
        export_to(m, x, str(out), to="onnx")
    assert e.value.code == "TS1418"
    assert not out.exists() and not (tmp_path / "m.onnx.truestar.json").exists()


def test_runtime_mismatch_refused_unless_forced(tmp_path, monkeypatch):
    """A runtime whose outputs are off by 1e-3 relative: refused; --force writes and says so."""
    from truestar.conform import runtimes as R

    real_open = R.OrtRuntime.open

    def off(self, data, artifact, **kw):
        s = real_open(self, data, artifact, **kw)
        run = s._run
        s._run = lambda feeds: [o * (1 + 1e-3) for o in run(feeds)]
        return s

    monkeypatch.setattr(R.OrtRuntime, "open", off)
    m, x = Z.mlp()
    out = tmp_path / "m.onnx"
    with pytest.raises(DeployError) as e:
        export_to(m, x, str(out), to="onnx")
    assert e.value.code == "TS1414" and not out.exists()
    r = export_to(m, x, str(out), to="onnx", force=True)
    assert r.status == "failed" and "TS1415" in r.codes and out.exists()
    assert json.load(open(str(out) + ".truestar.json", encoding="utf-8"))["verified"] is False


class Cum(nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=1) * 2


def test_op_without_lowering_names_the_node(tmp_path):
    out = tmp_path / "c.onnx"
    with pytest.raises(DeployError) as e:
        export_to(Cum(), torch.randn(3, 4), str(out), to="onnx")
    assert e.value.code == "TS1412" and "cumsum" in str(e.value) and not out.exists()


def test_usage_errors():
    m, x = Z.mlp()
    with pytest.raises(DeployError) as e:
        export_to(m, x, "m.tflite", to="onnx")
    assert e.value.code == "TS1419"
    with pytest.raises(DeployError) as e:
        export_to(m, x, None, to="onnx", runtime="tflite:interp")
    assert e.value.code == "TS1419"


def test_cli_export_to(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(textwrap.dedent("""
        import torch
        from torch import nn
        def get_model():
            torch.manual_seed(0)
            return nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 3))
        def get_inputs():
            return (torch.randn(4, 8),)
    """))
    out = tmp_path / "model.onnx"
    js = tmp_path / "r.json"
    assert cli_main(["export", str(f), "--to", "onnx", "-o", str(out), "--json", str(js)]) == 0
    assert out.exists() and json.load(open(js))["status"] == "verified"
    assert cli_main(["export", str(f), "--to", "onnx", "-o", str(tmp_path / "x.tflite")]) == 2
