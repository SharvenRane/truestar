"""`truestar range` (docs/RANGE.md): low precision range prediction, remedies and their verification.

Ground truth comes from PyTorch's own float16 CPU kernels (truestar.rangecheck.groundtruth), never
from the emulator under test. Models: tests/range_models.py (synthetic weights, nothing downloaded).
"""

import json
import math
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import range_models as rm  # noqa: E402

from truestar import ir  # noqa: E402
from truestar.ir.reference import bind_symbols  # noqa: E402
from truestar.rangecheck import FORMATS, TARGETS, range_check  # noqa: E402
from truestar.rangecheck.analysis import _passes  # noqa: E402
from truestar.rangecheck.engine import compare, emulate, observe  # noqa: E402
from truestar.rangecheck.groundtruth import chained_run, ground_truth, match  # noqa: E402
from truestar.rangecheck.interval import concrete_shapes, input_box, mag, propagate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


@pytest.fixture(autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def _origins(rep):
    return [s for s in rep.sites + rep.info if s.kind in ("overflow", "hidden_overflow") and s.origin]


# ----------------------------------------------------------------------------- formats


def test_formats_and_twin_rounding():
    f = FORMATS["fp16"]
    x = torch.tensor([1.0 + 2 ** -12, 7e4, 3e-9, 1e6], dtype=torch.float64)
    r = f.round(x)
    assert r[0] == 1.0 and math.isinf(r[1]) and r[2] == 0.0
    t = f.round_mantissa(x)  # same significand bits, no range limit
    assert t[0] == 1.0 and math.isfinite(t[1]) and t[2] != 0.0 and abs(t[3] / 1e6 - 1) < 2 ** -11
    assert f.max == 65504.0 and FORMATS["fp8-e4m3"].max == 448.0 and FORMATS["fp8-e5m2"].max == 57344.0
    assert f.headroom_bits(65504.0 / 4) == pytest.approx(2.0)
    from truestar.rangecheck.cli import TARGET_NAMES

    assert tuple(sorted(TARGETS)) == TARGET_NAMES


# ----------------------------------------------------------------------------- bge small (onnxruntime#31626, PR #32294)


@pytest.fixture(scope="module")
def bge():
    return rm.bge_small_like(), rm.bge_inputs()


def test_bge_small_decomposed_layernorm_overflow_found_and_remedied(bge):
    model, ids = bge
    rep = range_check(model, (ids,), target="webgpu-f16")
    sites = _origins(rep)
    mods = sorted(s.where.module for s in sites)
    assert mods == [f"bert.encoder.layer.{i}.output.LayerNorm" for i in (0, 3, 4)], mods
    assert all(s.hidden == "(x - mean)^2" and s.headroom_bits < 0 for s in sites)
    assert all(s.where.file == "modeling_bert.py" and s.where.line for s in sites)
    assert rep.status == "remedied" and "TS302" in rep.codes and "TS307" in rep.codes
    assert rep.target_run.cosine_min < 0.9  # quietly wrong: finite, no NaN
    assert rep.target_run.nonfinite == 0
    assert rep.chosen.remedy == "rescale" and rep.chosen.exact_rel_err <= 1e-12
    assert rep.chosen.result.cosine_min >= 0.9999 and rep.chosen.result.real_flips == 0


def test_bge_small_matches_pytorch_float16_ground_truth(bge):
    model, ids = bge
    rep = range_check(model, (ids,), target="webgpu-f16", remedies=False)
    gt = ground_truth(model, (ids,), decompose_norms=True)
    m = match(_origins(rep), [g for g in gt.sites if g.inputs_in_range])
    assert m["true_sites"] == 3 and m["found"] == 3 and m["missed"] == 0 and m["false_positives"] == 0, m
    # the emulator and PyTorch's own chained float16 run agree on the damage
    base = chained_run(model, (ids,), decompose_norms=True)
    assert abs(base["cosine_mean"] - rep.target_run.cosine_mean) < 0.01


def test_bge_small_fused_float16_is_clean(bge):
    model, ids = bge
    rep = range_check(model, (ids,), target="fp16")
    assert rep.status == "ok" and not rep.sites, rep.to_text()
    assert any(s.kind == "hidden_in_fused" for s in rep.info)  # reported, not alarmed
    assert ground_truth(model, (ids,)).sites == []


# ----------------------------------------------------------------------------- Gemma 3 style (onnxruntime#26732)


@pytest.fixture(scope="module")
def gemma():
    return rm.gemma3_like(), rm.gemma_inputs()


def test_gemma_residual_overflow_found_rescaled_and_verified(gemma):
    model, ids = gemma
    rep = range_check(model, (ids,), target="fp16")
    first = [s for s in rep.sites if s.kind == "overflow" and s.origin]
    assert [s.where.module for s in first] == ["layers.10"] and first[0].op == "add"
    assert first[0].where.code.startswith("x = x + self.post_feedforward_layernorm")
    assert any(not s.origin for s in rep.sites)  # downstream residual values carried out of range
    assert rep.target_run.nonfinite > 0
    assert rep.status == "remedied" and rep.chosen.remedy == "rescale"
    assert rep.chosen.exact_rel_err <= 1e-12 and rep.chosen.cost["scale_exponent"] == -2
    assert rep.chosen.result.cosine_min >= 0.9999 and rep.chosen.result.real_flips == 0
    # the Hub workaround (clip at +-32752) is tried and fails verification on this model
    clamp = next(a for a in rep.attempts if a.remedy == "clamp")
    assert not clamp.passed and clamp.result.real_flips > 0
    gt = ground_truth(model, (ids,))
    m = match(_origins(rep), [g for g in gt.sites if g.inputs_in_range])
    assert m["true_sites"] == 1 and m["found"] == 1 and m["false_positives"] == 0, m
    assert chained_run(model, (ids,))["nonfinite"] > 0


def test_gemma_on_the_neural_engine_and_in_bfloat16(gemma):
    model, ids = gemma
    rep = range_check(model, (ids,), target="ane", remedies=False)
    assert rep.status == "overflow" and "TS301" in rep.codes
    rep = range_check(model, (ids,), target="bf16")
    assert rep.status == "ok" and not rep.sites


def test_wrong_remedy_fails_verification(gemma):
    """Tests can fail: a remedy that changes the answer must not pass."""
    model, ids = gemma
    prog = ir.import_module(model, (ids,))
    tgt = TARGETS["fp16"]
    _, _, refs = observe(prog, [(ids,)], tgt, local=False)
    twin = [emulate(prog, (ids,), tgt, "twin").outputs]
    tc = compare(refs, twin, twin)
    site = next(n for n in prog.nodes if n.output.name == "add_109")
    bad = emulate(prog, (ids,), tgt, "target", clamps={id(site): 1.0}).outputs
    assert not _passes(compare(refs, [bad], twin), tc)


# ----------------------------------------------------------------------------- clean models


def test_bert_base_is_clean_in_float16_and_decomposed():
    model, ids = rm.bert_base(), rm.bge_inputs()
    for target in ("fp16", "webgpu-f16"):
        rep = range_check(model, (ids,), target=target)
        assert rep.status == "ok" and not rep.sites, rep.to_text()
        assert "TS300" in rep.codes
    assert ground_truth(model, (ids,), decompose_norms=True).sites == []


def test_resnet50_is_clean_in_float16():
    model = rm.resnet_calibrated("resnet50")
    x = torch.randn(2, 3, 96, 96, generator=torch.Generator().manual_seed(3))
    rep = range_check(model, (x,), target="fp16")
    assert rep.status == "ok" and not rep.sites, rep.to_text()
    assert ground_truth(model, (x,)).sites == []


# ----------------------------------------------------------------------------- underflow and precision loss


class TinyLogits(nn.Module):
    """Logits near 1e-5, the random init MobileNetV3 case of docs/NPU.md."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 10)
        with torch.no_grad():
            self.fc2.weight.mul_(1e-5)
            self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_underflowing_logits_found_and_kept_in_float32():
    torch.manual_seed(0)
    rep = range_check(TinyLogits().eval(), (torch.randn(16, 16),), target="fp16")
    assert [s.kind for s in rep.sites] == ["underflow"] and rep.sites[0].where.module == "fc2"
    assert rep.status == "remedied" and rep.chosen.remedy == "fp32"


def test_cancellation_is_a_precision_warning_not_a_range_error():
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    import check_model

    x = torch.randn(8, 32, generator=torch.Generator().manual_seed(0)) + 60.0
    rep = range_check(check_model.get_model(), (x,), target="fp16")
    assert rep.status == "at_risk" and [s.kind for s in rep.sites] == ["precision_loss"]
    s = rep.sites[0]
    assert s.op == "sub" and s.where.file == "check_model.py" and s.where.line == 20


# ----------------------------------------------------------------------------- static bounds are sound


@pytest.mark.parametrize("name", ["bge", "gemma", "resnet18", "example"])
def test_static_bounds_contain_every_observed_value(name):
    """Never introduce #3: a bound that is too small would 'prove' an overflowing value safe."""
    if name == "bge":
        model, x = rm.bge_small_like(layers=5), rm.bge_inputs()
    elif name == "gemma":
        model, x = rm.gemma3_like(layers=8), rm.gemma_inputs(seq=8)
    elif name == "resnet18":
        model, x = rm.resnet_calibrated("resnet18", size=64), torch.randn(2, 3, 64, 64)
    else:
        sys.path.insert(0, os.path.join(ROOT, "examples"))
        import range_model

        model, (x,) = range_model.get_model(), range_model.get_inputs()
    prog = ir.import_module(model, (x,))
    obs, leaf, _ = observe(prog, [(x,)], TARGETS["fp16"], local=False)
    b = propagate(prog, input_box([(x,)]), concrete_shapes(prog, bind_symbols(prog, (x,))))
    checked = 0
    for n in prog.nodes:
        if not n.output.type.dtype.is_floating_point:
            continue
        bound = mag(b.of(n.output))
        assert obs[id(n)].out.peak <= bound * (1 + 1e-9) + 1e-12, (n.op, n.output.name, obs[id(n)].out.peak, bound)
        for h, st in obs[id(n)].hidden.items():
            hb = b.hidden.get(id(n), {}).get(h)
            if hb is not None:
                assert st.peak <= mag(hb) * (1 + 1e-9) + 1e-12, (n.op, h, st.peak, hb)
        checked += 1
    assert checked > 20


# ----------------------------------------------------------------------------- command line


def test_cli_agent_json_on_the_example(tmp_path):
    e = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2")
    p = subprocess.run([sys.executable, "-m", "truestar", "range", os.path.join(ROOT, "examples", "range_model.py"),
                        "--target", "fp16", "--format", "agent"], cwd=tmp_path, env=e, capture_output=True,
                       text=True, encoding="utf-8", timeout=300)
    assert p.returncode == 1, p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert doc["command"] == "range" and doc["schema_version"] == 1 and doc["status"] == "remedied"
    assert doc["codes"][:1] == ["TS301"] and "TS307" in doc["codes"]
    first = doc["sites"][0]
    assert first["module"] == "blocks.9" and first["origin"] and first["file"] == "range_model.py"
    assert first["headroom_bits"] < 0 and doc["chosen_remedy"] == "rescale"
    assert any(a["kind"] == "apply_rescale" for a in doc["next_actions"])
    p = subprocess.run([sys.executable, "-m", "truestar", "range", "missing.py", "--format", "agent"],
                       cwd=tmp_path, env=e, capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert p.returncode == 2 and json.loads(p.stdout)["codes"] == ["TS020"]


def test_rescale_is_exact_in_float64_including_norm_eps():
    """The rescale must change nothing in exact arithmetic, also where eps is not negligible."""
    from truestar.ir.core import Builder
    from truestar.ir.reference import run
    from truestar.rangecheck.remedy import apply_rescale, find_region

    torch.manual_seed(0)
    b = Builder("eps")
    x = b.input("x", torch.float32, (4, 8))
    w = b.param("w", torch.randn(8, 8) * 3000.0)
    h = b.op("linear", x, w, None)
    r = b.op("add", h, b.op("linear", x, b.param("w2", torch.randn(8, 8)), None))
    y = b.op("layer_norm", r, b.param("g", torch.ones(8)), b.param("bb", torch.zeros(8)),
             normalized_shape=[8], eps=1e7)  # eps comparable to the variance
    prog = b.build([y])
    xin = torch.randn(4, 8)
    peaks = {id(h): 1e5, id(r): 1e5}  # both out of range, so both are carried scaled
    region = find_region(prog, {id(r): 1e5 / 32752.0}, peaks, 32752.0)
    new = apply_rescale(prog, region)
    assert region.k == 2 and region.norm_exits
    a, c = run(prog, xin)[0], run(new, xin)[0]
    assert float((a - c).abs().max() / a.abs().max()) < 1e-12
