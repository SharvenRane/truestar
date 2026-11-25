"""WebGPU backend (truestar.webgpu, docs/WEBGPU.md).

Two groups:

* machinery tests that run everywhere ONNX Runtime's CPU EP exists: a fake worker runs the graphs
  on the CPU EP and plays a WebGPU device with an injected faulty kernel, nodes left on the CPU,
  a session that fell back, or taps that change the result. They pin blame, residency, the
  observer guard and the no silent fallback rule without a GPU;
* real WebGPU tests, which need onnxruntime-ep-webgpu (and a WebGPU adapter). They skip with the
  reason in environments without it; on this machine they run under the cv venv.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import truestar.webgpu as tw  # noqa: E402
from truestar.backends import Backend, BackendUnavailable  # noqa: E402
from truestar.check import check  # noqa: E402
from truestar.webgpu import onnx_check  # noqa: E402

torch.manual_seed(0)


# ----------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 8)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.soft(self.fc2(self.act(self.fc1(x))))


class SmallCNN(nn.Module):
    """Batch norm and in place activations: the two things that broke the first exporter."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.act2 = nn.Hardswish(inplace=True)
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return self.head(x.mean(dim=(2, 3)))


def _cnn():
    m = SmallCNN().eval()
    with torch.no_grad():
        for bn in (m.bn1, m.bn2):
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
    return m


# ----------------------------------------------------------------------------- fake worker


class FakeWorker:
    """Runs graphs on ONNX Runtime's CPU EP and reports them as WebGPU runs.

    fault_op: every node of this ONNX op type is followed by a multiply by `scale` on "WebGPU"
    (a wrong kernel, seen the same way in the full graph, an isolated module and a single node).
    cpu_ops: op types reported as placed on the CPU EP. fallback: every WebGPU session reports
    the CPU EP only. tap_shift: graphs with more outputs than the plain model (taps) get their
    first output nudged, like a fusion the taps break."""

    def __init__(self, fault_op=None, scale=1.5, cpu_ops=(), fallback=False, tap_shift=0.0,
                 plain_outputs=1, fault_everywhere=False):
        self.fault_op, self.scale, self.cpu_ops = fault_op, scale, set(cpu_ops)
        self.fault_everywhere = fault_everywhere
        self.fallback, self.tap_shift, self.plain_outputs = fallback, tap_shift, plain_outputs
        self.info = {}
        self.calls = []

    def log_tail(self, n=0):
        return ""

    def _inject(self, model):
        from onnx import helper, numpy_helper

        nodes = []
        for n in model.graph.node:
            nodes.append(n)
            if n.op_type != self.fault_op:
                continue
            out = n.output[0]
            n.output[0] = out + "_raw"
            k = out + "_fault"
            model.graph.initializer.append(numpy_helper.from_array(
                np.array(self.scale, dtype=np.float32), k))
            nodes.append(helper.make_node("Mul", [out + "_raw", k], [out], name=n.name + "_fault"))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        return model

    def call(self, op, payload=None):
        self.calls.append((op, payload.get("provider")))
        model = onnx.load_from_string(payload["model"])
        webgpu = payload.get("provider") == "webgpu"
        if (webgpu or self.fault_everywhere) and self.fault_op:
            model = self._inject(model)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        outs = sess.run(payload.get("outputs"), payload["feeds"])
        if (webgpu and self.tap_shift and len(model.graph.output) > self.plain_outputs
                and len(model.graph.node) > 1):
            outs[0] = (outs[0] * (1 + self.tap_shift)).astype(outs[0].dtype)
        nodes = None
        if payload.get("profile"):
            nodes = [(n.name, n.op_type, "CPUExecutionProvider"
                      if (n.op_type in self.cpu_ops or not webgpu) else "WebGpuExecutionProvider")
                     for n in model.graph.node]
        providers = (["CPUExecutionProvider"] if (self.fallback or not webgpu)
                     else ["WebGpuExecutionProvider", "CPUExecutionProvider"])
        return {"outputs": outs, "output_names": [o.name for o in sess.get_outputs()],
                "providers": providers, "fallback": webgpu and self.fallback, "ep_error": "",
                "nodes": nodes, "seconds": 0.0, "repeat_equal": True}


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def make(**kw):
        w = FakeWorker(**kw)
        holder["w"] = w
        b = Backend(name="webgpu", device="webgpu", description="fake WebGPU (CPU EP)",
                    kind="webgpu")
        monkeypatch.setattr(tw, "probe", lambda name, ep_options=None: (b, ""))
        monkeypatch.setattr(tw, "get_worker", lambda name, ep_options=None: w)
        return w

    return make


# ----------------------------------------------------------------------------- pure functions


def test_backend_names_parse():
    assert tw.parse_spec("webgpu:vulkan:1") == ("Vulkan", 1)
    assert tw.parse_spec("webgpu:d3d12") == ("D3D12", 0)
    assert tw.split_options("webgpu:d3d12+enableMatmulFp32Accumulation=1") == (
        "webgpu:d3d12", {"enableMatmulFp32Accumulation": "1"})
    assert tw.is_webgpu("webgpu+x=1") and tw.is_webgpu("WebGPU") and not tw.is_webgpu("cuda")
    for bad in ("webgpu:opengl", "webgpu:d3d12:x", "webgpu+novalue"):
        with pytest.raises(ValueError):
            tw.parse_spec(bad)


def test_export_keeps_modes_and_inputs():
    """The exporter put batch norm back in training mode, and in place activations overwrote the
    feeds (EfficientNet SiLU looked 67% wrong). Both are guarded here."""
    m = _cnn()
    x = torch.randn(2, 3, 8, 8)
    data, feeds, n_out, _ = onnx_check.export(m, (x,), {}, torch.float32)
    assert not any(mod.training for mod in m.modules())
    np.testing.assert_array_equal(feeds["in0"], x.numpy())
    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    ref = m.double()(x.double()).detach().numpy()
    got = sess.run(None, feeds)[0]
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-5
    # an in place module alone: the feed must be the input, not its result
    act = nn.Hardswish(inplace=True)
    y = torch.randn(4, 8)
    _, feeds, _, _ = onnx_check.export(act, (y,), {}, torch.float32)
    np.testing.assert_array_equal(feeds["in0"], y.numpy())


def test_residency_classifies_cpu_nodes():
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    g = helper.make_graph([helper.make_node("Shape", ["x"], ["s"], name="shape"),
                           helper.make_node("Relu", ["x"], ["r"], name="relu"),
                           helper.make_node("Reshape", ["r", "s"], ["y"], name="reshape")],
                          "g", [x], [y])
    data = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]).SerializeToString()
    where = onnx_check.residency([("shape", "Shape", "CPUExecutionProvider"),
                                  ("relu", "Relu", "CPUExecutionProvider"),
                                  ("reshape", "Reshape", "WebGpuExecutionProvider"),
                                  ("m", "MemcpyToHost", "CPUExecutionProvider")], data)
    assert where["webgpu"] == 1 and where["transfers"] == 1
    assert where["shape_on_cpu"] == [("shape", "Shape")]
    assert where["compute_on_cpu"] == [("relu", "Relu")]


# ----------------------------------------------------------------------------- machinery


def test_clean_model_is_clean(fake):
    w = fake()
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["webgpu"])
    r = rep.results[0]
    assert rep.status == "ok", rep.to_text()
    assert r.extra["residency"]["compute_on_cpu"] == []
    assert any("taps trusted" in n for n in r.notes)
    assert all(p != "cpu" for _, p in w.calls)  # the CPU EP control only runs when needed


def test_faulty_kernel_is_blamed_with_op_and_repro(fake, tmp_path):
    fake(fault_op="Softmax")
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["webgpu"], repro_dir=str(tmp_path))
    r = rep.results[0]
    assert r.status == "divergent", rep.to_text()
    assert r.first_divergent.key == "soft#0"
    assert r.first_divergent_op is not None and "Softmax" in r.first_divergent_op.op
    assert "TS010" in r.codes and "TS563" not in r.codes
    assert os.path.isfile(r.repro_path) and os.path.isfile(r.op_repro_path)
    # the control run on the CPU EP passes (exit 0)
    p = subprocess.run([sys.executable, r.op_repro_path, "cpu"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_fault_in_a_cnn_names_the_layer(fake):
    fake(fault_op="Conv", scale=1.2)
    rep = check(_cnn(), torch.randn(2, 3, 8, 8), backends=["webgpu"])
    r = rep.results[0]
    assert r.first_divergent.key == "conv1#0", rep.to_text()
    assert "Conv" in r.first_divergent_op.op


def test_compute_nodes_on_cpu_are_named(fake):
    fake(cpu_ops={"Gemm"})
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["webgpu"])
    r = rep.results[0]
    assert "TS561" in r.codes and r.status != "ok"
    msg = next(i.message for i in r.issues if i.code == "TS561")
    assert "Gemm" in msg


def test_session_fallback_is_never_reported_as_webgpu(fake):
    fake(fallback=True)
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["webgpu"])
    r = rep.results[0]
    assert "TS560" in r.codes and r.status != "ok"
    assert r.output is None and r.layers == []


def test_observer_effect_marks_taps_untrusted(fake):
    fake(fault_op="Softmax", tap_shift=1e-6)
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["webgpu"])
    r = rep.results[0]
    assert "TS564" in r.codes
    assert any("taps untrusted" in n for n in r.notes)
    # blame still comes from isolated, uninstrumented runs
    assert r.first_divergent.key == "soft#0"


def test_export_differs_is_not_blamed_on_webgpu(fake):
    """A graph that is wrong on the CPU EP too is reported as TS563 (export or ONNX Runtime)."""
    fake(fault_op="Softmax", fault_everywhere=True)
    rep = check(MLP().eval(), torch.randn(4, 16), backends=["webgpu"])
    assert "TS563" in rep.results[0].codes, rep.to_text()


# ----------------------------------------------------------------------------- real WebGPU

_REAL = {}


def _real(name="webgpu"):
    if name not in _REAL:
        b, reason = tw.probe(name)
        tw.close_workers()
        _REAL[name] = reason if b is None else ""
    if _REAL[name]:
        pytest.skip(f"{name} not available: {_REAL[name]}")


def test_real_clean_models():
    _real()
    for model, x in ((MLP().eval(), torch.randn(8, 16)), (_cnn(), torch.randn(2, 3, 16, 16))):
        rep = check(model, x, backends=["webgpu"])
        r = rep.results[0]
        assert rep.status == "ok", rep.to_text()
        assert r.extra["residency"]["compute_on_cpu"] == []
        assert r.extra["residency"]["webgpu"] >= 1
        assert "adapter" in r.backend.description


def test_real_bad_ep_option_is_unavailable_not_cpu():
    """ONNX Runtime falls back to its CPU EP when the WebGPU EP cannot be created; Truestar must
    refuse the backend instead of reporting CPU numbers as WebGPU."""
    _real()
    tw._PROBED.pop("webgpu+dawnBackendType=Bogus", None)
    with pytest.raises(BackendUnavailable) as e:
        check(MLP().eval(), torch.randn(2, 16), backends=["webgpu+dawnBackendType=Bogus"])
    assert "fell back" in str(e.value)


def test_real_forced_cpu_node_is_named():
    _real()
    m = MLP().eval()
    data, _, _, _ = onnx_check.export(m, (torch.randn(4, 16),), {}, torch.float32)
    gemm = next(n.name for n in onnx.load_from_string(data).graph.node if n.op_type == "Gemm")
    rep = check(m, torch.randn(4, 16), backends=[f"webgpu+forceCpuNodeNames={gemm}"])
    r = rep.results[0]
    assert "TS561" in r.codes, rep.to_text()
    assert gemm in next(i.message for i in r.issues if i.code == "TS561")


class Down(nn.Module):
    """A float16 feed forward down projection: 64 tokens, 4096 wide, positive inputs (GELU)."""

    def __init__(self):
        super().__init__()
        self.act = nn.GELU()
        self.down = nn.Linear(4096, 256)

    def forward(self, x):
        return self.down(self.act(x))


def test_real_float16_matmul_accumulation_finding(tmp_path):
    """Measured with onnxruntime-ep-webgpu 0.6.0: the default float16 MatMul accumulates in
    float16 for this shape (M 64, K 4096); enableMatmulFp32Accumulation=1 removes it. If a later
    EP release fixes the default, the first assertion fails and docs/WEBGPU.md needs updating."""
    _real()
    torch.manual_seed(0)
    m = Down().eval().half()
    x = torch.randn(1, 64, 4096) + 1.0
    rep = check(m, x, backends=["webgpu", "webgpu+enableMatmulFp32Accumulation=1"],
                repro_dir=str(tmp_path))
    bad, fixed = rep.results
    assert bad.status != "ok", rep.to_text()
    assert bad.first_divergent.key == "down#0"
    assert "MatMul" in bad.first_divergent_op.op
    assert fixed.status == "ok", rep.to_text()
    from truestar.agent import next_actions

    kinds = [a["kind"] for a in next_actions(rep)]
    assert "try_ep_option" in kinds and "report_upstream" in kinds and "pin_torch" not in kinds
    p = subprocess.run([sys.executable, bad.op_repro_path], capture_output=True, text=True)
    assert p.returncode == 1, p.stdout + p.stderr
    p = subprocess.run([sys.executable, bad.op_repro_path, "cpu"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_real_dawn_probes():
    _real()
    pytest.importorskip("wgpudawn")
    from truestar.webgpu.dawn import run_probes

    doc = run_probes(tw.default_dawn_backend(), 0, "f32")
    assert "error" not in doc, doc
    assert {r["probe"] for r in doc["results"]} == {"matmul", "sum", "sum_tree", "softmax"}
    assert not any(r["divergent"] for r in doc["results"]), doc
