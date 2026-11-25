"""Offline tests for the INT8 agreement and per layer NPU studies (truestar.remote).

qai_hub is never imported: a fake client stands in for Qualcomm AI Hub. It runs the traced model
on the PC, fakes INT8 by rounding every output to a grid, and lets a test inject a prediction flip
or a layer error on one (device, compute unit).
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np
import pytest
import torch
from torch import nn

from truestar.remote import agreement as ag
from truestar.remote import qai_hub_quant as qq

S24, S25 = "Samsung Galaxy S24 (Family)", "Samsung Galaxy S25 (Family)"


# ----------------------------------------------------------------------------- fake AI Hub


class FakeModel:
    def __init__(self, model_id: str, module, quantized: bool, kind: str):
        self.model_id = model_id
        self.module = module
        self.quantized = quantized
        self.kind = kind


class FakeJob:
    def __init__(self, job_id: str, target=None, outputs=None):
        self.job_id = job_id
        self._target = target
        self._outputs = outputs

    def get_target_model(self):
        return self._target

    def download_output_data(self):
        return self._outputs

    def get_status(self):
        return type("S", (), {"message": "ok"})()


class FakeDataset:
    def __init__(self, dataset_id: str, entries):
        self.dataset_id = dataset_id
        self.entries = entries


class FakeHub:
    """Enough of qai_hub for HubSession. `perturb(device, unit, outputs)` edits outputs in place."""

    class QuantizeDtype:
        INT8 = "int8"

    def __init__(self, perturb: Optional[Callable] = None, grid: float = 0.05):
        self.perturb = perturb
        self.grid = grid
        self.calls: List[str] = []
        self.models: Dict[str, FakeModel] = {}
        self.jobs: Dict[str, FakeJob] = {}
        self.datasets: Dict[str, FakeDataset] = {}
        self._n = 0

    def _next(self, p: str) -> str:
        self._n += 1
        return f"{p}{self._n:04d}"

    @staticmethod
    def Device(name):  # noqa: N802  mirrors qai_hub.Device
        return name

    def _register_job(self, job):
        self.jobs[job.job_id] = job
        return job

    def submit_compile_job(self, model, device, options="", input_specs=None, name=None):
        self.calls.append("compile")
        if isinstance(model, FakeModel):
            module, quant = model.module, model.quantized
        else:
            module, quant = model, False
        kind = "onnx" if "onnx" in options else "tflite"
        m = FakeModel(self._next("m"), module, quant, kind)
        self.models[m.model_id] = m
        return self._register_job(FakeJob(self._next("jc"), target=m))

    def submit_quantize_job(self, model, calibration_data, weights_dtype, activations_dtype, name=None):
        self.calls.append("quantize")
        assert isinstance(calibration_data, FakeDataset)
        m = FakeModel(self._next("m"), model.module, True, "onnx")
        self.models[m.model_id] = m
        return self._register_job(FakeJob(self._next("jq"), target=m))

    def upload_dataset(self, data, name=None):
        self.calls.append("upload")
        d = FakeDataset(self._next("d"), data)
        self.datasets[d.dataset_id] = d
        return d

    def get_dataset(self, dataset_id):
        return self.datasets[dataset_id]

    def get_model(self, model_id):
        return self.models[model_id]

    def get_job(self, job_id):
        return self.jobs[job_id]

    def submit_inference_job(self, model, device, inputs, options="", name=None):
        self.calls.append("inference")
        unit = options.split()[-1]
        name0 = sorted(inputs.entries)[0]
        per_sample = []
        with torch.no_grad():
            for a in inputs.entries[name0]:
                out = model.module(torch.from_numpy(a))
                outs = out if isinstance(out, (tuple, list)) else (out,)
                per_sample.append([o.numpy().astype(np.float64) for o in outs])
        k = len(per_sample[0])
        stacked = [np.concatenate([s[i] for s in per_sample], 0) for i in range(k)]
        if model.quantized:
            stacked = [self._quant(t) for t in stacked]
        if self.perturb is not None:
            self.perturb(device, unit, stacked, model.quantized)
        outputs = {f"output_{i}": [stacked[i][j:j + 1].astype(np.float32) for j in range(stacked[i].shape[0])]
                   for i in range(k)}
        return self._register_job(FakeJob(self._next("ji"), outputs=outputs))

    def _quant(self, t):
        step = self.grid if t.ndim == 2 else float(np.sqrt(np.mean(t * t))) / 40 or 1.0
        return np.round(t / step) * step


# ----------------------------------------------------------------------------- small model


class Small(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),
                                  nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8, classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.body(x), 1)) * 4.0


def _data(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 8, 8, generator=g)


def _model(seed=0):
    torch.manual_seed(seed)
    return Small().eval()


# ----------------------------------------------------------------------------- agreement counting


def test_pair_agreement_counts_ties_and_clear_flips():
    ref = np.array([[3.0, 1.0, 0.0],   # agree
                    [2.0, 1.95, 0.0],  # reference near tie, b flips by one step: near tie
                    [1.0, 1.0, 0.0],   # b scores 0 and 1 equal (argmax takes 0), a picks 1: exact tie
                    [5.0, 0.0, 0.0],   # clear flip
                    [0.0, 4.0, 1.0],   # agree
                    [3.0, 0.0, 0.0]])  # b nearly tied after moving 3 units: clear, not a tie
    a = ref.copy()
    a[2] = [0.9, 1.0, 0.0]
    b = ref.copy()
    b[1] = [1.95, 2.0, 0.0]
    b[3] = [0.0, 5.0, 0.0]
    b[5] = [2.0, 2.05, 0.0]
    ra = ag.RunLogits("a", a, "float64")
    rb = ag.RunLogits("b", b, "int8", step=0.05)
    p = ag.pair_agreement(ra, rb)
    assert (p.n, p.agree, p.disagree) == (6, 2, 4)
    assert p.flipped == [1, 2, 3, 5]
    assert p.exact_tie == 1          # row 2
    assert p.near_tie == 2           # rows 1 and 2: swing 0.1 <= 2 steps
    assert p.clear == 2              # rows 3 and 5
    assert ag.pair_agreement(rb, ra).as_dict()["near_tie"] == 2  # symmetric


def test_agreement_matrix_pairs_and_accuracy():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(64, 10)) * 3
    labels = base.argmax(1)
    other = base.copy()
    for i in (3, 7, 11):  # move 3 predictions by a lot
        other[i, (labels[i] + 1) % 10] += 20
    runs = [ag.RunLogits("ref", base), ag.RunLogits("x", other, "float32")]
    pairs = ag.agreement_matrix(runs)
    assert len(pairs) == 1 and pairs[0].agree == 61 and pairs[0].clear == 3
    assert runs[0].accuracy(labels) == 1.0
    assert runs[1].accuracy(labels) == pytest.approx(61 / 64)
    txt = ag.agreement_text(runs, pairs, labels)
    assert "61/64 agree" in txt and "95.31%" in txt


def test_estimate_step_on_grid_and_float():
    rng = np.random.default_rng(1)
    q = np.round(rng.normal(size=1000) / 0.037) * 0.037
    assert ag.estimate_step(q) == pytest.approx(0.037, rel=1e-6)
    assert ag.estimate_step(rng.normal(size=1000)) is None
    # a float32 round trip of a dequantized grid is still recognised
    assert ag.estimate_step(q.astype(np.float32)) == pytest.approx(0.037, rel=1e-5)


def test_int8_tie_tolerance_uses_estimated_step():
    q = np.round(np.random.default_rng(2).normal(size=(50, 5)) / 0.1) * 0.1
    r = ag.RunLogits("q", q, "int8")
    assert r.step == pytest.approx(0.1, rel=1e-6)
    assert np.allclose(r.tie_tolerance(), 0.2)


# ----------------------------------------------------------------------------- outputs


def test_outputs_in_natural_order_and_channel_last():
    out = {f"output_{i}": [np.full((1, 2), i, dtype=np.float32), np.full((1, 2), i, dtype=np.float32)]
           for i in range(12)}
    arrs = qq.outputs_in_order(out)
    assert [int(a[0, 0]) for a in arrs] == list(range(12))  # a plain sort gives 0, 1, 10, 11, 2, ...
    assert arrs[0].shape == (2, 2)
    nchw = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    assert np.array_equal(qq.align(np.transpose(nchw, (0, 2, 3, 1)), nchw.shape), nchw)


def test_match_outputs_falls_back_to_shape_order():
    a, b = np.zeros((2, 10)), np.ones((2, 3, 4, 4))
    got = qq.match_outputs([b, a], [(2, 10), (2, 3, 4, 4)])
    assert got[0].shape == (2, 10) and got[1].shape == (2, 3, 4, 4)


# ----------------------------------------------------------------------------- taps


def test_tap_wrapper_returns_intermediates_safe_from_inplace():
    m = _model()
    x = _data(4)
    w = qq.TapWrapper(m, ["body.2", "body.4"]).eval()
    with torch.no_grad():
        out, t1, t2 = w(x)
        h = m.body[0](x).relu()
        conv2 = m.body[2](h)
    assert torch.allclose(out, m(x))
    # body.3 is ReLU(inplace=True): without the clone the tap would be the post ReLU tensor
    assert torch.allclose(t1, conv2) and (t1 < 0).any()
    assert t2.shape == (4, 8, 1, 1)
    with pytest.raises(ValueError):
        qq.TapWrapper(m, ["nope"])
    with pytest.raises(ValueError):
        qq.TapWrapper(m, ["body.0"] * (qq.MAX_TAPS + 1))


def test_default_taps_resnet18_block_level():
    import torchvision

    m = torchvision.models.resnet18(num_classes=10).eval()
    taps = qq.default_taps(m, (torch.randn(1, 3, 64, 64),))
    assert taps == ["relu", "maxpool", "layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0",
                    "layer3.1", "layer4.0", "layer4.1", "avgpool"]
    assert len(qq.default_taps(m, (torch.randn(1, 3, 64, 64),), max_taps=4)) == 4


# ----------------------------------------------------------------------------- int8 study with fake hub


def _flipper(flips):
    """Perturb: on (device, unit) in flips, swap the top two logits of the given inputs."""
    def perturb(device, unit, outs, quantized):
        for i in flips.get((device, unit), []):
            row = outs[0][i]
            order = np.argsort(row)
            row[order[-1]], row[order[-2]] = row[order[-2]], row[order[-1]]
    return perturb


def test_int8_study_counts_injected_flips_with_few_jobs(tmp_path):
    hub = FakeHub(perturb=_flipper({(S24, "npu"): [1, 5, 9], (S25, "npu"): [5]}))
    sess = qq.HubSession(hub=hub, cache_dir=str(tmp_path))
    m, x, calib = _model(), _data(40), _data(16, seed=1)
    labels = torch.from_numpy(qq.reference_outputs(m, x)[0].argmax(1))
    st = qq.int8_agreement_study(m, x, labels, calib, devices=[S24, S25],
                                 units=[(S24, "npu"), (S24, "cpu"), (S25, "npu")], session=sess)
    names = [r.name for r in st.runs]
    assert names == ["float64 reference", "int8 S24 npu", "int8 S24 cpu", "int8 S25 npu"]
    pairs = {(p.a, p.b): p for p in st.pairs}
    npu_cpu = pairs[("int8 S24 npu", "int8 S24 cpu")]
    s25_cpu = pairs[("int8 S24 cpu", "int8 S25 npu")]
    assert npu_cpu.disagree == 3 and npu_cpu.flipped == [1, 5, 9]
    assert s25_cpu.disagree == 1 and s25_cpu.flipped == [5]
    assert pairs[("int8 S24 npu", "int8 S25 npu")].flipped == [1, 9]
    for r in st.runs[1:]:
        assert r.step == pytest.approx(0.05, rel=1e-4)  # fake INT8 output grid found from the data
    # one ONNX compile, one quantize, one TFLite compile shared by both phones, one inference per run
    assert hub.calls.count("compile") == 2 and hub.calls.count("quantize") == 1
    assert hub.calls.count("inference") == 3 and hub.calls.count("upload") == 2
    assert all(v for k, v in st.jobs.items() if k != "notes")
    assert "accuracy on the batch" in st.to_text()


def test_int8_study_rerun_is_cached(tmp_path):
    m, x, calib = _model(), _data(24), _data(8, seed=1)
    labels = torch.zeros(24, dtype=torch.long)
    hub = FakeHub()
    qq.int8_agreement_study(m, x, labels, calib, devices=[S24], session=qq.HubSession(hub, cache_dir=str(tmp_path)))
    first = list(hub.calls)
    # a fresh session reading the same cache submits nothing at all
    s2 = qq.HubSession(hub, cache_dir=str(tmp_path))
    qq.int8_agreement_study(m, x, labels, calib, devices=[S24], session=s2)
    assert hub.calls == first and s2.log.submitted == [] and len(s2.log.reused) >= 5
    # outputs deleted locally: they are downloaded again from the cached job, still no new job
    for f in os.listdir(tmp_path / "outputs"):
        os.remove(tmp_path / "outputs" / f)
    s3 = qq.HubSession(hub, cache_dir=str(tmp_path))
    qq.int8_agreement_study(m, x, labels, calib, devices=[S24], session=s3)
    assert hub.calls == first
    # a changed weight is a new model: new compiles, new quantize, new inference
    m2 = _model()
    with torch.no_grad():
        m2.fc.bias[0] += 0.5
    s4 = qq.HubSession(hub, cache_dir=str(tmp_path))
    qq.int8_agreement_study(m2, x, labels, calib, devices=[S24], session=s4)
    kinds = [k for k, _ in s4.log.submitted]
    assert kinds.count("compile") == 2 and kinds.count("quantize") == 1 and kinds.count("inference") == 2
    assert "dataset" not in kinds  # same inputs: datasets reused


def test_fingerprint_sensitive_to_weights_specs_and_tag():
    m = _model()
    specs = {"input_0": ((1, 3, 8, 8), "float32")}
    f = qq.model_fingerprint(m, specs)
    assert f == qq.model_fingerprint(_model(), specs)
    assert f != qq.model_fingerprint(m, {"input_0": ((1, 3, 16, 16), "float32")})
    assert f != qq.model_fingerprint(m, specs, "|taps:a")
    with torch.no_grad():
        m.body[0].weight[0, 0, 0, 0] += 1e-6
    assert f != qq.model_fingerprint(m, specs)


# ----------------------------------------------------------------------------- per layer with fake hub


def _layer_fault(start: int, rel: float, quant_lsb: int = 0):
    """From tap index `start` on (tap outputs are 1..k, output 0 is the logits), add error on NPU."""
    def perturb(device, unit, outs, quantized):
        if unit != "npu":
            return
        rng = np.random.default_rng(0)
        for i in range(1, len(outs)):
            t = outs[i]
            if i < start:
                continue
            if quantized:
                step = ag.estimate_step(t) or 1.0
                mask = rng.random(t.shape) < 0.3
                t += mask * quant_lsb * step
            else:
                t += rel * np.sqrt(np.mean(t * t)) * rng.normal(size=t.shape)
        outs[0] += (rel if start < len(outs) else 0) * np.abs(outs[0]).mean()
    return perturb


def test_layer_blame_names_first_float_layer_over_threshold(tmp_path):
    m, x = _model(), _data(6)
    taps = ["body.0", "body.1", "body.2", "body.3", "body.4"]
    hub = FakeHub(perturb=_layer_fault(start=3, rel=3e-2))
    st = qq.layer_blame_study(m, x, taps=taps, device=S24, session=qq.HubSession(hub, cache_dir=str(tmp_path)))
    assert st.blame.first is not None and st.blame.first.name == "body.2"  # output index 3 = third tap
    assert [r.crossed for r in st.blame.rows[:2]] == [False, False]
    assert hub.calls.count("compile") == 1 and hub.calls.count("inference") == 2
    assert "first layer over threshold: body.2" in st.to_text()


def test_layer_blame_clean_reports_nothing(tmp_path):
    m, x = _model(), _data(6)
    st = qq.layer_blame_study(m, x, taps=["body.1", "body.3"], device=S24,
                              session=qq.HubSession(FakeHub(), cache_dir=str(tmp_path)))
    assert st.blame.first is None
    assert all(r.rel_vs_control == 0.0 for r in st.blame.rows)


def test_layer_blame_int8_threshold_is_one_lsb(tmp_path):
    m, x, calib = _model(), _data(6), _data(8, seed=1)
    taps = ["body.0", "body.2", "body.4"]
    # 30% of elements off by one LSB everywhere: rounding, not a finding
    one = qq.layer_blame_study(m, x, taps=taps, device=S24, quantized=True, calib_x=calib,
                               session=qq.HubSession(FakeHub(perturb=_layer_fault(1, 0.0, quant_lsb=1)),
                                                     cache_dir=str(tmp_path / "a")))
    assert one.blame.first is None
    assert all(r.step is not None for r in one.blame.rows[:-1])
    # from the second tap on, 30% of elements off by three LSB: blamed there
    three = qq.layer_blame_study(m, x, taps=taps, device=S24, quantized=True, calib_x=calib,
                                 session=qq.HubSession(FakeHub(perturb=_layer_fault(2, 0.0, quant_lsb=3)),
                                                       cache_dir=str(tmp_path / "b")))
    assert three.blame.first is not None and three.blame.first.name == "body.2"
    assert three.blame.rows[1].frac_over_1lsb > 0.2


def test_layer_blame_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ag.layer_blame(["a"], [np.zeros(2)], [], [np.zeros(2)], False)


# ----------------------------------------------------------------------------- synthetic task


def test_synthetic_task_is_deterministic_and_meaningful():
    t1 = __import__("truestar.remote.synthetic", fromlist=["build_task"]).build_task(
        num_classes=10, image_size=64, n_eval=64, n_calib=16, n_train_per_class=8)
    t2 = __import__("truestar.remote.synthetic", fromlist=["build_task"]).build_task(
        num_classes=10, image_size=64, n_eval=64, n_calib=16, n_train_per_class=8)
    assert torch.equal(t1.eval_x, t2.eval_x) and torch.equal(t1.eval_y, t2.eval_y)
    for a, b in zip(t1.model.state_dict().values(), t2.model.state_dict().values()):
        assert torch.equal(a, b)
    with torch.no_grad():
        lo = t1.model.double()(t1.eval_x.double())
    pred = lo.argmax(1)
    assert pred.unique().numel() >= 8           # not collapsed onto one class
    assert 1.0 < float(lo.std()) < 5.0          # logits of a few units
    assert float((pred == t1.eval_y).double().mean()) > 0.4


def test_no_qai_hub_import():
    assert "qai_hub" not in sys.modules
