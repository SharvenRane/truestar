"""WebNN as a check backend in a headless browser (truestar.webnn, research item 6.3).

Translation tests run everywhere. The browser tests start Chrome headless (no window) and skip
with the probe's reason when no WebNN capable browser is installed.
"""
import numpy as np
import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import truestar.webnn as W  # noqa: E402
from truestar.check import check  # noqa: E402
from truestar.webgpu.onnx_check import export  # noqa: E402
from truestar.webnn import translate as T  # noqa: E402

torch.manual_seed(0)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, groups=2)
        self.act2 = nn.Hardswish()
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        x = self.act2(self.conv2(x))
        return self.head(x.mean(dim=(2, 3)))


def _cnn():
    m = CNN().eval()
    with torch.no_grad():
        m.bn1.running_mean.uniform_(-0.5, 0.5)
        m.bn1.running_var.uniform_(0.5, 2.0)
    return m


def _mlp():
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.LayerNorm(32), nn.Linear(32, 4), nn.Softmax(-1)).eval()


class CumsumNet(nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=1)


# ----------------------------------------------------------------------------- translation

def test_names_parse():
    assert W.parse_spec("webnn") == "" and W.parse_spec("webnn:GPU") == "gpu"
    for bad in ("webnn:tpu", "webnn:cpu:0", "webgpu"):
        with pytest.raises(ValueError):
            W.parse_spec(bad)


def test_every_node_becomes_one_webnn_op():
    data, feeds, _, _ = export(_cnn(), (torch.randn(2, 3, 16, 16),), {}, torch.float32)
    d = T.translate(data, feeds)
    proto = onnx.load_from_string(data)
    assert len(d["nodes"]) == sum(1 for n in proto.graph.node if n.op_type != "Constant")
    ops = {n["op"] for n in d["nodes"]}
    assert {"conv2d", "maxPool2d", "relu", "gemm"} <= ops
    conv = next(n for n in d["nodes"] if n["op"] == "conv2d")
    assert conv["padding"] == [1, 1, 1, 1] and conv["inputs"][1] in {c["name"] for c in d["consts"]}


def test_untranslatable_op_is_refused():
    data, feeds, _, _ = export(CumsumNet(), (torch.randn(2, 5),), {}, torch.float32)
    with pytest.raises(T.Unsupported, match="CumSum"):
        T.translate(data, feeds)


def test_non_float32_input_is_refused():
    data, feeds, _, _ = export(nn.ReLU(), (torch.randn(2, 3),), {}, torch.float32)
    with pytest.raises(T.Unsupported, match="float32"):
        T.translate(data, {k: v.astype(np.float64) for k, v in feeds.items()})


# ----------------------------------------------------------------------------- headless browser

def _real(name):
    b, why = W.probe(name)
    if b is None:
        pytest.skip(f"{name}: {why}")
    return b


@pytest.fixture(autouse=True)
def _close():
    yield
    W.close_browsers()


@pytest.mark.parametrize("name", ["webnn:cpu", "webnn:gpu", "webnn:npu"])
def test_real_clean_models_with_per_layer_errors(name):
    _real(name)
    for m, x in ((_mlp(), torch.randn(8, 16)), (_cnn(), torch.randn(2, 3, 16, 16))):
        r = check(m, x, backends=[name]).results[0]
        assert r.status != "divergent", r.issues
        assert r.output.rel_err < 1e-5
        assert r.extra["residency"]["device"] == r.extra["residency"]["nodes"] > 0
        assert all(l.isolated is not None or l.propagated is not None for l in r.layers)


def test_real_duplicate_output_operand_runs(tmp_path):
    """A tapped graph names one tensor twice (a module output that is also the model output);
    Chrome 158 lost the WebNN context at build until the page gave the second name an identity."""
    _real("webnn:cpu")
    r = check(nn.Sequential(nn.Linear(4, 3)).eval(), torch.randn(2, 4), backends=["webnn:cpu"]).results[0]
    assert r.status == "ok" and not [i for i in r.issues if i.code == "TS002"]


def test_real_wrong_translation_is_blamed_at_its_module(monkeypatch, tmp_path):
    _real("webnn:cpu")
    monkeypatch.setitem(T.UNARY, "Relu", "sigmoid")  # a wrong kernel for one op
    m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    r = check(m, torch.randn(8, 16), backends=["webnn:cpu"], repro_dir=str(tmp_path)).results[0]
    assert r.status == "divergent" and r.first_divergent.key == "1#0"
    assert r.first_divergent_op is not None and "Relu" in r.first_divergent_op.op
    assert r.repro_path and r.repro_path.endswith(".py")


def test_real_untranslatable_model_is_ts1127_not_run():
    _real("webnn:cpu")
    r = check(CumsumNet().eval(), torch.randn(2, 5), backends=["webnn:cpu"]).results[0]
    assert [i.code for i in r.issues] == ["TS1127"] and r.output is None
