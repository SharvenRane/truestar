"""Simulated device numerics (docs/DEVICES.md): cited cases reproduced, clean models clean, no
silent fallback, sweep and special value probe."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from truestar.backends import BackendUnavailable, resolve_backends
from truestar.check import check
from truestar.devsim import PROFILES, get_profile, simulate
from truestar.devsim.profiles import DOCUMENTED, GUESS, HYPOTHESIS, MEASURED

SIMS = sorted(PROFILES)


# ----------------------------------------------------------------------------- profiles are cited


def test_every_profile_rule_is_cited_and_labelled():
    for p in PROFILES.values():
        assert p.rules, p.name
        for r in p.rules:
            assert r.status in (DOCUMENTED, MEASURED, HYPOTHESIS, GUESS), (p.name, r)
            assert r.what and r.source, (p.name, r)
            if r.status in (DOCUMENTED, MEASURED):
                assert r.source != "no public source", (p.name, r)


def test_ideal_profile_drops_device_rules():
    ane = get_profile("sim:ane").ideal()
    assert ane.storage == torch.float16 and ane.acc == "wide" and not ane.tile
    assert not (ane.nan_to_inf or ane.ftz or ane.saturate or ane.overrides)


# ----------------------------------------------------------------------------- cited cases


def test_ane_softplus_cliff_coremltools_2687():
    """Issue: 2000 point sweep over [-15, 15], first point with reference > 5 and output < 1 is
    x = 10.4077 (output 0, reference 10.4077); fine sweep: 10.394 gives 10.39, 10.395 gives 0."""
    x = torch.linspace(-15, 15, 2000)
    ref = F.softplus(x.double())
    with simulate("sim:ane"):
        y = F.softplus(x)
        fine = F.softplus(torch.tensor([10.394, 10.395]))
    first = ((ref > 5) & (y < 1)).nonzero()[0].item()
    assert x[first].item() == pytest.approx(10.4077, abs=1e-4) and y[first].item() == 0.0
    assert fine[0].item() == pytest.approx(10.39, abs=0.01) and fine[1].item() == 0.0
    with simulate("sim:ieee-f16"):
        assert not ((ref > 5) & (F.softplus(x) < 1)).any()  # correct float16 has no cliff


def test_ane_logsumexp_cliff_coremltools_2690():
    """Issue: C = 32, cliff at x = 7.6313, reference 11.0971, NE output 0."""
    xs = torch.linspace(-15, 15, 2000)
    inp = xs.view(-1, 1).expand(-1, 32).contiguous()
    ref = torch.logsumexp(inp.double(), 1)
    with simulate("sim:ane"):
        out = torch.logsumexp(inp, 1)
    first = ((ref > 5) & (out < 1)).nonzero()[0].item()
    assert xs[first].item() == pytest.approx(7.6313, abs=1e-4)
    assert ref[first].item() == pytest.approx(11.0971, abs=1e-4) and out[first].item() == 0.0
    with simulate("sim:ieee-f16"):
        assert not ((ref > 5) & (torch.logsumexp(inp, 1) < 1)).any()


def _reduce_l2_input():
    norms = torch.linspace(32, 400, 784, dtype=torch.float64)
    d = torch.randn(384, 784, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    d = d / d.norm(dim=0) * norms
    return d.view(1, 384, 1, 28, 28).float(), norms


def test_npu_reduce_l2_saturation_openvino_39703():
    """Issue: [1,384,1,28,28] over axis 1, true norm up to 400: NPU max 255.9, max rel err above
    256 is 0.36, below 256 is 5.9e-4; ReduceL2(x/2)*2 gives 400.0."""
    x, norms = _reduce_l2_input()
    with simulate("sim:openvino-npu-f16"):
        out = torch.linalg.vector_norm(x, dim=1).double().flatten()
        fixed = (torch.linalg.vector_norm(x / 2, dim=1) * 2).double().flatten()
    rel = (out - norms).abs() / norms
    assert out.max().item() == pytest.approx(255.9, abs=0.05)
    assert rel[norms > 256].max().item() == pytest.approx(0.36, abs=0.005)
    assert rel[norms <= 256].max().item() < 1e-3
    assert fixed.max().item() == 400.0
    with simulate("sim:ieee-f16"):
        assert torch.linalg.vector_norm(x, dim=1).max().item() == 400.0


@pytest.mark.parametrize("n,want", [(2048, 2048.0), (2049, 2048.0), (4096, 2048.0)])
def test_openvino_gpu_f16_gemm_openvino_22613(n, want):
    """Issue: [1,1,n] x [1,n,1] of ones on the GPU in default f16: 2048.0 at n = 2049."""
    a, b = torch.ones(1, 1, n), torch.ones(1, n, 1)
    with simulate("sim:openvino-gpu-f16"):
        assert (a @ b).item() == want
    with simulate("sim:ieee-f16"):  # float32 class accumulator: only output rounding
        assert (a @ b).item() == float(torch.tensor(float(n)).half())


def test_decomposed_layer_norm_overflow_onnxsim_1742():
    """onnxsim#1742: the decomposed LayerNorm's Pow(x, 2) overflows float16 once |x| passes 256
    (ViT activations reached 1418). A fused LayerNorm in correct float16 survives."""
    x = torch.randn(4, 64) * 50
    x[0, 0] = 1418.0
    ref = F.layer_norm(x.double(), (64,))
    with torch.no_grad(), simulate("sim:webgpu-f16"):
        dec = F.layer_norm(x, (64,))
    with torch.no_grad(), simulate("sim:ieee-f16"):
        fused = F.layer_norm(x, (64,))
    assert torch.isfinite(ref).all() and (fused.double() - ref).abs().max() < 1e-2
    assert (dec[0].double() - ref[0]).abs().max() > 0.5  # the overflowing row is destroyed


# ----------------------------------------------------------------------------- check integration


class _Act(nn.Module):
    def forward(self, x):
        return F.softplus(x)


class _SP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(16, 32), _Act(), nn.Linear(32, 10)
        with torch.no_grad():
            self.fc1.weight.mul_(8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_check_blames_the_softplus_on_ane_only():
    torch.manual_seed(0)
    rep = check(_SP(), torch.randn(64, 16), backends=["cpu", "sim:ieee-f16", "sim:ane"])
    by = {r.backend.name: r for r in rep.results}
    assert by["cpu"].status == "ok" and by["sim:ieee-f16"].status == "ok"
    ane = by["sim:ane"]
    assert ane.status == "divergent" and "TS010" in ane.codes and "TS520" in ane.codes
    assert ane.first_divergent.key == "act#0"
    assert ane.first_divergent_op.op == "aten.softplus.default"


def test_check_blame_depends_on_the_rule():
    """Tests can fail: without the softplus formulation (the ideal of sim:ane) nothing is blamed."""
    torch.manual_seed(0)
    from truestar.devsim import profiles

    ideal = profiles.PROFILES["sim:ane"].ideal()
    profiles.PROFILES["sim:test-ane-ideal"] = ideal.__class__(**{**ideal.__dict__,
                                                                 "name": "sim:test-ane-ideal"})
    try:
        rep = check(_SP(), torch.randn(64, 16), backends=["sim:test-ane-ideal"])
        assert rep.results[0].status == "ok"
    finally:
        del profiles.PROFILES["sim:test-ane-ideal"]


def _clean_models():
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 10))
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                        nn.Conv2d(8, 8, 3, groups=2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                        nn.Linear(8, 10)).eval()
    attn = nn.Sequential(nn.LayerNorm(32), nn.MultiheadAttention(32, 4, batch_first=True))

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm, self.mha, self.head = nn.LayerNorm(32), attn[1], nn.Linear(32, 10)

        def forward(self, x):
            h = self.norm(x)
            q = h.view(h.shape[0], h.shape[1], 4, 8).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, q, q).transpose(1, 2).reshape(h.shape)
            return self.head(o.mean(1))

    return [(mlp, torch.randn(16, 32)), (cnn, torch.randn(2, 3, 12, 12)),
            (Attn(), torch.randn(4, 6, 32))]


@pytest.mark.parametrize("sim", SIMS)
def test_clean_models_are_clean_on_every_sim(sim):
    for model, x in _clean_models():
        rep = check(model, x, backends=[sim], drill_ops=False)
        r = rep.results[0]
        if sim == "sim:hexagon" and any(isinstance(m, nn.GELU) for m in model.modules()):
            # the Snapdragon GELU's errors lean up on the real device too (tests/test_hexagon_sim.py):
            # TS961 there is the device's finding, the only one allowed
            assert set(r.codes) <= {"TS520", "TS1140", "TS961"}, (sim, r.codes, r.notes)
            continue
        assert r.status == "ok", (sim, type(model).__name__, r.codes, r.notes)


def test_sim_backends_never_run_silently_elsewhere():
    with pytest.raises(BackendUnavailable, match="truestar check and truestar probe only"):
        resolve_backends(["sim:ane"])
    with pytest.raises(BackendUnavailable, match="unknown simulated backend"):
        resolve_backends(["sim:nope"], allow_sim=True)
    chosen, _ = resolve_backends(None, allow_sim=True)
    assert not any(b.sim for b in chosen)  # never added unless asked for


def test_in_place_and_views_under_simulation():
    x = torch.tensor([1.0001, -2.0, float("nan")])
    v = x[:2]
    with simulate("sim:ane"):
        v.relu_()
        y = x + 0
    assert x[0].item() == 1.0 and x[1].item() == 0.0  # rounded to float16, written through the view
    assert y[2].item() == math.inf  # NaN becomes +inf on the ANE


def test_emulate_precision_with_a_profile():
    from truestar import ir
    from truestar.train import EmulatedReferenceExecutor

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4)).eval()
    x = torch.randn(3, 8)
    prog = ir.import_module(model, (x,))

    ane = EmulatedReferenceExecutor(profile="sim:ane").prepare(prog)([x])[0]
    with torch.no_grad(), simulate(get_profile("sim:ane").ideal()):
        eager = model(x)
    assert torch.equal(ane.half().double(), ane.double())  # every value is a float16 value
    assert (ane.double() - eager.double()).abs().max() < 2e-3
    xn = x.clone()
    xn[0, 0] = float("nan")
    out = EmulatedReferenceExecutor(profile="sim:ane").prepare(prog)([xn])[0]
    assert not torch.isnan(out).any()  # ANE never produces NaN
    neuron = EmulatedReferenceExecutor(profile="sim:neuron-bf16").transform(prog)
    low = [n for n in neuron.nodes if n.op == "cast" and n.attrs.get("storage") == torch.bfloat16]
    assert len(low) >= 4  # x, both weights and the hidden activation feed linear in bfloat16


# ----------------------------------------------------------------------------- sweep and probe


class _Dot(nn.Module):
    def forward(self, a, b):
        return a @ b


def test_sweep_finds_the_f16_accumulator_cliff():
    from truestar.devsim.sweep import ladder, sweep

    build = lambda n: (_Dot(), (torch.ones(1, 1, n), torch.ones(1, n, 1)), None)
    rep = sweep(build, "n", 512, 8192, ["cpu", "sim:ieee-f16", "sim:openvino-gpu-f16"])
    by = {r.backend: r for r in rep.results}
    assert by["cpu"].verdict == "clean" and by["sim:ieee-f16"].verdict == "clean"
    gpu = by["sim:openvino-gpu-f16"]
    # 2048 against n is within the float16 floor (5e-3) up to n = 2058
    assert (gpu.verdict, gpu.last_ok, gpu.first_bad) == ("cliff", 2058, 2059)
    assert gpu.module == "<model>#0" and gpu.op == "aten.bmm.default" and "TS521" in gpu.codes
    assert 2048 in ladder(512, 8192) and 2049 in ladder(512, 8192)
    assert rep.runs <= len(ladder(512, 8192)) + 2 * (math.ceil(math.log2(2048)) + 2)


def test_sweep_input_errors():
    from truestar.devsim.sweep import parse_sweep

    assert parse_sweep("n:512..8192") == ("n", 512, 8192)
    for bad in ("n512..8192", "n:8192..512", "1n:2..4", "n:a..b"):
        with pytest.raises(ValueError):
            parse_sweep(bad)


def test_probe_specials_tables():
    from truestar.devsim.specials import probe_specials

    rep = probe_specials(["cpu", "sim:ane", "sim:webgpu-f16", "sim:openvino-npu-f16",
                          "sim:ieee-f16"])
    dep = {}
    for r in rep.rows:
        if r.verdict == "departure":
            dep.setdefault(r.backend, set()).add(r.case)
    assert "cpu" not in dep and "sim:ieee-f16" not in dep  # IEEE in their own precision
    assert {"add_nan", "relu_specials", "sum_nan", "softmax_nan"} <= dep["sim:ane"]  # NaN to +inf
    assert {"subnormal_pass", "subnormal_product"} <= dep["sim:webgpu-f16"]  # flush to zero
    assert "cast_to_f16_overflow" in dep["sim:openvino-npu-f16"]  # saturation
    soft = [r for r in rep.rows if r.backend == "sim:ane" and r.case == "softmax_nan"]
    assert [r.got for r in soft] == [1.0, 0.0]  # all mass on the NaN lane (arXiv 2606.22283)
    impl = [r for r in rep.rows if r.case == "cast_to_int_specials"]
    assert impl and all(r.verdict in ("impl_defined", "impl_disagree") for r in impl)
    assert rep.status == "divergent" and "TS523" in rep.codes()


def test_fix_refuses_simulated_backends():
    from truestar.fix.engine import fix

    with pytest.raises(BackendUnavailable, match="does not support simulated backends"):
        fix(_SP(), torch.randn(4, 16), backends=["sim:ane"])
