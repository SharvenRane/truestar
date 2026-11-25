"""sim:hexagon against real Snapdragon NPU outputs (docs/DEVICES.md section 7, P59).

The fixture holds device taps of the Qualcomm AI Hub battery (docs/QAIHUB.md): the S26 (Hexagon
v81, TFLite NPU) MLP and encoder outputs, two conv stem channels and the conv head, and the X Elite
(v73, QNN DLC) GELU tap. Every expected number below is a measurement of that data, not a target
the simulation was tuned to: the GELU curve was fitted on half of the first tap's points and is
scored here on the other half and on taps it never saw.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from truestar.check import check
from truestar.conform import qaihub_battery as B
from truestar.devsim import get_profile, simulate
from truestar.devsim import hexagon as HX

DATA = os.path.join(os.path.dirname(__file__), "data", "hexagon_npu_taps.npz")


def r16(v):
    return np.asarray(v, np.float64).astype(np.float16).astype(np.float64)


def rel_l2(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


@pytest.fixture(scope="module")
def dev():
    z = np.load(DATA)
    return {k: z[k].astype(np.float64) for k in z.files}


@pytest.fixture(scope="module")
def arrs():
    return B.inputs("float")


def _sim(sub, x, profile):
    with torch.no_grad(), simulate(profile):
        out = sub(*[torch.from_numpy(v) for v in x])
    return [r16(o.double().numpy()) for o in out]


def _fused_pre(arrs):
    fc1 = B.float_model().mlp.fc1
    return r16(arrs["x_mlp"]) @ r16(fc1.weight.detach().numpy()).T + r16(fc1.bias.detach().numpy())


def _held_out(x):
    order = np.argsort(x.ravel(), kind="stable")
    test = np.ones(x.size, bool)
    test[order[::2]] = False  # the fit used these points (scripts/hexagon_sim_fit.py)
    return test.reshape(x.shape)


# ----------------------------------------------------------------------------- the fitted curve


@pytest.mark.parametrize("key", ["s26_fc1_gelu", "xelite_fc1_gelu"])
def test_gelu_curve_on_held_out_points(dev, arrs, key):
    x = _fused_pre(arrs)
    test = _held_out(x)
    want = dev[key]
    ours = r16(HX.gelu(torch.from_numpy(x)).numpy())
    exact = r16(F.gelu(torch.from_numpy(r16(x))).numpy())  # what sim:ieee-f16 computes
    # measured: 2.73e-4 and 89.4% (S26); exact GELU 2.71e-3 and 6.7%
    assert rel_l2(ours[test], want[test]) < 3.0e-4
    assert rel_l2(exact[test], want[test]) > 2.5e-3
    assert np.mean(ours[test] == want[test]) > 0.88
    assert np.mean(exact[test] == want[test]) < 0.08


def test_gelu_reads_the_unrounded_accumulator(dev, arrs):
    x = _fused_pre(arrs)
    test = _held_out(x)
    want = dev["s26_fc1_gelu"]
    fused = r16(HX.gelu(torch.from_numpy(x)).numpy())
    unfused = r16(HX.gelu(torch.from_numpy(r16(x))).numpy())
    # measured: 89.4% against 72.5% bit equal; the fusion is what the data says, not a choice
    assert np.mean(fused[test] == want[test]) > np.mean(unfused[test] == want[test]) + 0.1


def test_gelu_clamps_and_special_values():
    x = torch.tensor([-1e4, -3.1, -3.03, 0.0, 3.1, 1e4, -float("inf"), float("inf"), float("nan")],
                     dtype=torch.float64)
    y = HX.gelu(x)
    assert y[:3].tolist() == [0.0, 0.0, 0.0]  # device: every input at or below -3.03 gave 0
    assert y[3].item() == 0.0
    assert y[4].item() == 3.1 and y[5].item() == 1e4
    assert y[6].item() == 0.0 and y[7].item() == float("inf") and torch.isnan(y[8])
    assert 3.0 < HX.NEG_ZERO < 3.03 and 3.0 <= HX.POS_ONE < 3.04


# ----------------------------------------------------------------------------- the battery, end to end


def test_mlp_and_encoder_taps_closer_to_the_device(dev, arrs):
    m = B.float_model()
    before = _sim(m.mlp, [arrs["x_mlp"]], "sim:ieee-f16")
    after = _sim(m.mlp, [arrs["x_mlp"]], "sim:hexagon")
    # measured ratios (sim:ieee-f16 error over sim:hexagon error): 10.9x, 7.8x
    for i, key, ratio in ((0, "s26_fc1_gelu", 8.0), (1, "s26_fc2_relu", 5.0)):
        assert rel_l2(before[i], dev[key]) > ratio * rel_l2(after[i], dev[key]), key
    # the encoder's GELU was never in the fit and runs on a 3d input (view, addmm, view): 1.84x
    enc_b = _sim(m.enc, [arrs["x_enc"]], "sim:ieee-f16")[3]
    enc_a = _sim(m.enc, [arrs["x_enc"]], "sim:hexagon")[3]
    assert rel_l2(enc_b, dev["s26_enc_out"]) > 1.5 * rel_l2(enc_a, dev["s26_enc_out"])


def test_convolutions_stay_bit_exact(dev, arrs):
    m = B.float_model()
    ieee = _sim(m.conv, [arrs["x_conv"]], "sim:ieee-f16")
    hexa = _sim(m.conv, [arrs["x_conv"]], "sim:hexagon")
    for a, b in zip(ieee, hexa):
        assert np.array_equal(a, b)
    assert np.mean(hexa[0][:, :2] == dev["s26_conv_stem_c01"]) == 1.0  # measured 100%
    assert np.mean(hexa[5] == dev["s26_conv_out"]) >= 0.8  # measured 8 of 10


# ----------------------------------------------------------------------------- fusion semantics


def test_fusion_only_after_a_linear_layer_and_never_stale():
    torch.manual_seed(0)
    lin = nn.Linear(16, 32)
    conv = nn.Conv1d(4, 4, 1)
    x = torch.randn(8, 16) * 2
    with torch.no_grad():
        wide = (r16(x.numpy()) @ r16(lin.weight.numpy()).T + r16(lin.bias.numpy()))
        notes = set()
        with simulate("sim:hexagon", notes):
            y = F.gelu(lin(x))
        assert np.array_equal(y.double().numpy(), r16(HX.gelu(torch.from_numpy(wide)).numpy()))
        assert any(n.startswith("TS1140: aten.gelu: the curve fitted") and "fused" in n for n in notes)
        with simulate("sim:hexagon"):
            h = lin(x)
            h.add_(1.0)  # an in place write: the kept accumulator is stale and must not be used
            y2 = F.gelu(h)
        with simulate("sim:hexagon"):
            y3 = F.gelu(lin(x) + 1.0)
        assert torch.equal(y2, y3)
        notes = set()
        with simulate("sim:hexagon", notes):
            F.gelu(conv(torch.randn(2, 4, 5)))
            F.gelu(lin(x), approximate="tanh")
        assert any("not a linear layer's output" in n for n in notes)
        assert any(n.startswith("TS1141: aten.gelu(approximate='tanh')") for n in notes)


def test_profile_rules_and_ideal():
    p = get_profile("sim:hexagon")
    assert p.storage == torch.float16 and p.acc == "wide"
    assert {"gelu_hexagon", "fuse_linear_gelu"} <= p.overrides
    assert not p.ideal().overrides  # the ideal is correct float16, the noise floor


def test_check_names_the_fitted_op_and_the_unmeasured_one():
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 32)

        def forward(self, x):
            return torch.softmax(F.gelu(self.fc(x)), dim=-1)

    rep = check(Block().eval(), torch.randn(4, 32), backends=["sim:hexagon"], drill_ops=False)
    r = rep.results[0]
    assert {"TS520", "TS1140", "TS1141"} <= set(r.codes), r.codes
    assert r.status == "ok", (r.codes, r.notes)



def test_gelu_lean_is_the_devices(dev):
    """The device GELU's errors lean up (TS961's test on the real S26 tap); the profile predicts the
    same lean and correct float16 shows none, so TS961 on sim:hexagon is not a false alarm."""
    from truestar import bias as BI

    ref = torch.from_numpy(B.float_reference()["mlp.fc1_gelu"])
    floor = 5e-3 * BI.BIAS_FRACTION
    means = {}
    for name, out in (("device", torch.from_numpy(dev["s26_fc1_gelu"])),
                      ("sim:hexagon", torch.from_numpy(B.simulated("sim:hexagon")["mlp.fc1_gelu"])),
                      ("sim:ieee-f16", torch.from_numpy(B.simulated("sim:ieee-f16")["mlp.fc1_gelu"]))):
        d = BI.lean_test("fc1_gelu", out, ref, floor, 1.0).as_dict()
        means[name] = (d["biased"], next(x["mean"] for x in d["leans"] if x["direction"] == "up"))
    # measured: device +6.53e-4, sim:hexagon +6.59e-4, sim:ieee-f16 -5.9e-6 (not biased)
    assert means["device"][0] and means["sim:hexagon"][0] and not means["sim:ieee-f16"][0]
    assert abs(means["sim:hexagon"][1] - means["device"][1]) < 0.1 * means["device"][1]


def test_clean_models_stay_clean():
    """Without GELU nothing is flagged; with GELU the only finding is the device's own lean."""
    torch.manual_seed(0)
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 8, 3),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))
    enc = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, activation="relu", batch_first=True)
    for model, x in ((cnn, torch.randn(2, 3, 12, 12)), (enc, torch.randn(2, 6, 32))):
        r = check(model.eval(), x, backends=["sim:hexagon"], drill_ops=False).results[0]
        assert r.status == "ok", (type(model).__name__, r.codes, r.notes)
    mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 4))
    r = check(mlp.eval(), torch.randn(16, 32), backends=["sim:hexagon"], drill_ops=False).results[0]
    assert r.status in ("ok", "suspect") and set(r.codes) <= {"TS520", "TS1140", "TS961"}, r.codes
