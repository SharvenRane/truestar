"""QAIRT range modes as conventions (truestar.intref.qairt, research item 6.5).

The tf and symmetric cases are the worked examples of the QAIRT quantization page; the others are
properties every encoding must have (zero exact, range covered, minimum range) and the device hook
on constructed device results.
"""
import numpy as np
import pytest
import torch
from torch import nn

from truestar.intref import qairt as Q


def test_doc_example_codes_and_range():
    # QAIRT doc: values [-1.8, -1.0, 0, 0.5] -> min -1.803922, max 0.496078, step 0.009020,
    # offset 200 (as a code), codes [0, 89, 200, 255]
    e = Q.encoding(np.array([-1.8, -1.0, 0.0, 0.5]), 8, "tf")
    assert e.min == pytest.approx(-1.803922, abs=1e-6)
    assert e.max == pytest.approx(0.496078, abs=1e-6)
    assert e.scale == pytest.approx(0.009020, abs=1e-6)
    assert e.zero_point == 200 and e.offset == -200
    assert e.quantize([-1.8, -1.0, 0.0, 0.5]).tolist() == [0, 89, 200, 255]
    assert e.dequantize([0, 89, 200, 255]) == pytest.approx([-1.8039, -1.0011, 0.0, 0.4961], abs=1e-4)


def test_doc_example_zero_shift_and_one_sided_ranges():
    e = Q.encoding(np.array([-5.1, 5.1]), 8, "tf")
    assert (e.min, e.max) == (pytest.approx(-5.12), pytest.approx(5.08))
    assert e.quantize([0.0]).tolist() == [128]
    pos = Q.encoding(np.array([5.0, 10.0]), 8, "tf")
    assert (pos.min, pos.max, pos.offset) == (0.0, pytest.approx(10.0), 0)
    neg = Q.encoding(np.array([-20.0, -6.0]), 8, "tf")
    assert (neg.min, neg.max) == (pytest.approx(-20.0), pytest.approx(0.0, abs=1e-12))
    tiny = Q.encoding(np.array([0.0, 0.0]), 8, "tf")
    assert tiny.max - tiny.min == pytest.approx(Q.MIN_RANGE)


@pytest.mark.parametrize("mode", Q.MODES)
@pytest.mark.parametrize("bw", [8, 16])
@pytest.mark.parametrize("ties", Q.TIES)
def test_every_mode_keeps_zero_exact(mode, bw, ties):
    rng = np.random.default_rng(bw)
    x = np.concatenate([rng.normal(0.3, 1.0, 4000), [7.5]])  # one long tail value
    e = Q.encoding(x, bw, mode, ties=ties)
    assert e.offset <= 0 and e.offset >= -e.levels
    assert e.fake([0.0])[0] == 0.0
    assert e.max - e.min >= Q.MIN_RANGE * (1 - 1e-12)
    if mode in ("tf", "symmetric", "adjusted"):  # these never clip the calibration data by design
        assert e.max >= x.max() - e.scale / 2 - 1e-12 and e.min <= x.min() + e.scale / 2 + 1e-12
    if mode == "adjusted":  # covers the true range exactly, no inward shift at all
        assert e.max >= x.max() and e.min <= min(x.min(), 0.0)
    if mode == "symmetric":
        assert e.offset == -(1 << (bw - 1))


def test_enhanced_clips_a_long_tail_and_lowers_error():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 20000), [40.0]])
    tf, enh = Q.encoding(x, 8, "tf"), Q.encoding(x, 8, "enhanced")
    assert enh.max < tf.max
    err = lambda e: float(np.mean((e.fake(x) - x) ** 2))
    assert err(enh) < err(tf)


def test_tie_rules_differ_only_on_half_steps():
    e_away = Q.from_range(-1.0, 1.0, 8, ties="half_away")
    e_even = Q.from_range(-1.0, 1.0, 8, ties="half_even")
    assert e_away.scale == e_even.scale
    halves = e_away.min + (np.arange(10, 20) + 0.5) * e_away.scale
    assert not np.array_equal(e_away.quantize(halves), e_even.quantize(halves))
    mids = e_away.min + (np.arange(10, 20) + 0.25) * e_away.scale
    assert np.array_equal(e_away.quantize(mids), e_even.quantize(mids))


def test_half_away_is_exact_near_one_half():
    # floor(|v| + 0.5) rounds 0.49999999999999994 up; the exact rule must not
    assert Q._round(np.array([0.49999999999999994, -0.49999999999999994, 0.5, -0.5, 2.5]),
                    "half_away").tolist() == [0.0, -0.0, 1.0, -1.0, 3.0]


def test_signed_codes_are_unsigned_minus_half_range():
    e = Q.encoding(np.array([-1.8, 0.5]), 8, "tf", signed=True)
    u = Q.encoding(np.array([-1.8, 0.5]), 8, "tf")
    x = np.linspace(-1.8, 0.5, 33)
    assert np.array_equal(e.quantize(x), u.quantize(x) - 128)
    assert np.allclose(e.dequantize(e.quantize(x)), u.fake(x))
    assert e.zero_point == 200 - 128


def test_per_channel_is_symmetric_with_fixed_offset():
    w = np.random.default_rng(1).normal(size=(3, 4, 2, 2))
    per = Q.encoding(w, 8, "symmetric", axis=0, signed=True)
    assert len(per) == 3 and all(p.offset == -128 for p in per)
    with pytest.raises(ValueError):
        Q.encoding(w, 8, "tf", axis=0)


@pytest.mark.parametrize("bad", [np.array([]), np.array([0.0, np.nan]), np.array([np.inf, 1.0])])
def test_bad_calibration_refused(bad):
    with pytest.raises(ValueError):
        Q.encoding(bad, 8, "tf")


def test_unknown_mode_and_ties_refused():
    with pytest.raises(ValueError):
        Q.encoding(np.ones(3), 8, "minmax")
    with pytest.raises(ValueError):
        Q.from_range(0, 1, 8, ties="up")


# ----------------------------------------------------------------------------- device hook

def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.2, 1.0, n), [6.0, -3.0]])


@pytest.mark.parametrize("mode", ["tf", "symmetric", "adjusted", "enhanced"])
def test_attribute_names_the_mode_a_device_used(mode):
    x = _data()
    dev = Q.encoding(x, 8, mode).fake(x)  # a device that quantized with this mode
    a = Q.attribute(x, dev)
    assert a.code == "TS1130", a.verdict
    assert a.best.startswith(mode + "/") or mode in a.verdict


def test_attribute_one_step_is_not_a_kernel_bug():
    x = _data()
    e = Q.encoding(x, 8, "tf")
    q = e.quantize(x)
    q[::97] += 1  # one step off on a few elements, same grid
    q = np.clip(q, 0, 255)
    a = Q.attribute(x, e.dequantize(q))
    assert a.code == "TS1131" and a.best.startswith("tf/")


def test_attribute_big_errors_on_the_grid_are_unexplained():
    x = _data()
    e = Q.encoding(x, 8, "tf")
    q = np.clip(e.quantize(x) + np.where(np.arange(x.size) % 50 == 0, 5, 0), 0, 255)
    a = Q.attribute(x, e.dequantize(q))
    assert a.code == "TS1132" and "kernel" in a.verdict


def test_attribute_foreign_grid_and_off_grid():
    x = _data()
    other = Q.from_range(-10.0, 10.0, 8)  # an encoding no mode picks from this data
    assert Q.attribute(x, other.fake(x)).code == "TS1132"
    assert Q.attribute(x, x + 1e-3 * np.sin(np.arange(x.size))).code == "TS1133"


def test_attribute_int16():
    x = _data(20000, 3)
    e = Q.encoding(x, 16, "tf")
    a = Q.attribute(x, e.fake(x), bw=16)
    assert a.code == "TS1130" and a.best.startswith("tf/")


def test_attribute_codes_with_hint():
    x = _data()
    e = Q.encoding(x, 8, "symmetric")
    a = Q.attribute(x, e.quantize(x), device_codes=True, encoding_hint=e.as_dict())
    assert a.code == "TS1130" and a.best.startswith("symmetric/")


def test_compare_to_device_maps_taps():
    x = _data()
    out = Q.compare_to_device({"conv1": x, "fc": x * 2}, {"conv1": Q.encoding(x, 8, "tf").fake(x)})
    assert set(out) == {"conv1"} and out["conv1"]["code"] == "TS1130"


# ----------------------------------------------------------------------------- simulation

def test_simulate_quantizes_every_tap_on_its_grid():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    cal = [(torch.randn(32, 8),) for _ in range(4)]
    x = cal[0][0][:5]  # inside the calibration range (outside it QNN clamps, as simulated)
    sim = Q.simulate(m, cal, (x,), Q.SimConfig())
    for name, t in sim["taps"].items():
        enc = sim["encodings"]["activations"][name]
        k = (t.numpy() - enc["min"]) / enc["scale"]
        assert np.abs(k - np.rint(k)).max() < 1e-6, name
    with torch.no_grad():
        ref = m.double()(x.double())
    rel8 = float((sim["output"] - ref).norm() / ref.norm())
    sim16 = Q.simulate(m, cal, (x,), Q.SimConfig(act_bw=16, param_bw=16))
    rel16 = float((sim16["output"] - ref).norm() / ref.norm())
    assert 0 < rel16 < 1e-3 < rel8 < 0.05  # int16 is about 256 times closer than int8
    # clamping: an input far outside the calibration range is cut at the encoding max
    far = Q.simulate(m, cal, (x * 100,), Q.SimConfig())
    assert float(far["taps"]["0"].abs().max()) <= max(abs(sim["encodings"]["activations"]["0"]["min"]),
                                                       sim["encodings"]["activations"]["0"]["max"]) + 1e-9


def test_simulate_float16_fallback_module_stays_off_the_int_grid():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    cal = [(torch.randn(32, 8),)]
    sim = Q.simulate(m, cal, (torch.randn(5, 8),), Q.SimConfig(float16_modules=("2",), per_channel=True))
    t = sim["taps"]["2"].numpy()
    assert np.array_equal(t, t.astype(np.float16).astype(np.float64))
    assert isinstance(sim["encodings"]["params"]["0.weight"], list)
