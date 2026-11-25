"""Windows ML execution providers as check backends (truestar.winml, research item 6.1).

The fake worker tests run everywhere and pin the wiring: the shared ONNX path reports a session
fallback as TS1120 (never as an EP result), CPU compute nodes as TS1121, and a wrong kernel at the
right module. The real tests run each EP this machine has (RTX 5070 Ti here: NvTensorRtRtx and
WebGPU through their bring your own packages) and skip the rest with the probe's reason.
"""
import sys

import numpy as np
import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import truestar.winml as W  # noqa: E402
from truestar.backends import Backend, BackendUnavailable, resolve_backends  # noqa: E402
from truestar.check import check  # noqa: E402

torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


EP = "NvTensorRTRTXExecutionProvider"


class FakeWorker:
    """ONNX Runtime CPU EP runs reported as runs of a Windows ML EP (provider key "winml")."""

    def __init__(self, fault_op=None, scale=1.5, cpu_ops=(), fallback=False):
        self.fault_op, self.scale, self.cpu_ops, self.fallback = fault_op, scale, set(cpu_ops), fallback
        self.info, self.ep_name, self.providers_seen = {}, EP, []

    def log_tail(self, n=0):
        return ""

    def call(self, op, payload=None):
        from onnx import helper, numpy_helper

        dev = payload.get("provider") == "winml"
        self.providers_seen.append(payload.get("provider"))
        model = onnx.load_from_string(payload["model"])
        if dev and self.fault_op:
            nodes = []
            for n in model.graph.node:
                nodes.append(n)
                if n.op_type == self.fault_op:
                    out = n.output[0]
                    n.output[0] = out + "_raw"
                    model.graph.initializer.append(numpy_helper.from_array(np.array(self.scale, np.float32), out + "_k"))
                    nodes.append(helper.make_node("Mul", [out + "_raw", out + "_k"], [out], name=n.name + "_f"))
            del model.graph.node[:]
            model.graph.node.extend(nodes)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        outs = sess.run(payload.get("outputs"), payload["feeds"])
        nodes = None
        if payload.get("profile"):
            nodes = [(n.name, n.op_type, "CPUExecutionProvider" if (n.op_type in self.cpu_ops or not dev) else EP)
                     for n in model.graph.node]
        provs = ["CPUExecutionProvider"] if (self.fallback or not dev) else [EP, "CPUExecutionProvider"]
        return {"outputs": outs, "output_names": [o.name for o in sess.get_outputs()], "providers": provs,
                "fallback": dev and self.fallback, "ep_error": "", "nodes": nodes, "seconds": 0.0,
                "repeat_equal": True}


@pytest.fixture
def fake(monkeypatch):
    def make(**kw):
        w = FakeWorker(**kw)
        b = Backend(name="winml:nvtrtx", device="winml", description="fake Windows ML EP (CPU EP)",
                    kind="winml", info={"ep": EP})
        monkeypatch.setattr(W, "probe", lambda name: (b, ""))
        monkeypatch.setattr(W, "get_worker", lambda name: w)
        return w

    return make


def test_names_parse():
    assert W.parse_spec("winml:nvtrtx") == ("nvtrtx", 0, {})
    assert W.parse_spec("winml:dml:1+disable_metacommands=1") == ("dml", 1, {"disable_metacommands": "1"})
    for bad in ("winml", "winml:tpu", "winml:dml:x", "winml:dml:0:1", "winml:dml+novalue"):
        with pytest.raises(ValueError):
            W.parse_spec(bad)


def test_clean_model_reports_every_layer(fake):
    w = fake()
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["winml:nvtrtx"])
    r = rep.results[0]
    assert r.status == "ok", r.issues
    assert {l.key for l in r.layers} >= {"fc1#0", "act#0", "fc2#0"}
    assert all(l.propagated is not None for l in r.layers)  # per layer error from the taps
    assert "winml" in w.providers_seen and "webgpu" not in w.providers_seen


def test_wrong_kernel_blamed_at_the_module(fake, tmp_path):
    fake(fault_op="Relu")
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["winml:nvtrtx"], repro_dir=str(tmp_path))
    r = rep.results[0]
    assert r.status == "divergent" and r.first_divergent.key == "act#0"
    assert r.first_divergent_op is not None and "Relu" in r.first_divergent_op.op
    assert r.repro_path and open(r.repro_path, encoding="utf-8").read().count(EP) >= 1


def test_session_fallback_is_ts1120_never_an_ep_result(fake):
    fake(fallback=True)
    r = check(MLP().eval(), torch.randn(4, 16), backends=["winml:nvtrtx"]).results[0]
    assert [i.code for i in r.issues] == ["TS1120"] and r.output is None


def test_cpu_compute_nodes_are_ts1121(fake):
    fake(cpu_ops={"Gemm"})
    r = check(MLP().eval(), torch.randn(4, 16), backends=["winml:nvtrtx"]).results[0]
    msg = next(i.message for i in r.issues if i.code == "TS1121")
    assert "Gemm" in msg


def test_unavailable_ep_is_refused_with_reason(monkeypatch):
    monkeypatch.setattr(W, "probe", lambda name: (None, "QNNExecutionProvider is not available here"))
    with pytest.raises(BackendUnavailable, match="QNNExecutionProvider is not available"):
        resolve_backends(["winml:qnn"])


# ----------------------------------------------------------------------------- real EPs

def _real(name):
    if sys.platform != "win32":
        pytest.skip("Windows ML is Windows only")
    b, why = W.probe(name)
    if b is None:
        pytest.skip(f"{name}: {why}")
    return b


@pytest.mark.parametrize("name", ["winml:nvtrtx", "winml:webgpu", "winml:dml", "winml:openvino",
                                  "winml:qnn", "winml:vitisai", "winml:amdgpu"])
def test_real_ep_clean_model(name):
    b = _real(name)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 10)).eval()
    try:
        r = check(m, torch.randn(8, 64), backends=[name]).results[0]
    finally:
        W.close_workers()
    assert r.status != "divergent", r.issues
    assert r.output is not None and r.output.rel_err < 1e-2
    assert r.extra["residency"]["compute_on_cpu"] == []
    assert any(n.startswith("residency:") for n in r.notes)
    assert b.info["ep"] in b.description

