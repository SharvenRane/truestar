"""truestar.nn initializers added 2026-11-25: Xavier, Kaiming, truncated normal and per parameter
rules, against torch.nn.init's definitions, keyed like every other draw."""

import math

import pytest
import torch

from truestar import nn
from truestar.nn.core import param_key

F64 = torch.float64


def _ks(a, b):
    """Two sample Kolmogorov Smirnov statistic."""
    a, b = torch.sort(a.flatten().double())[0], torch.sort(b.flatten().double())[0]
    grid = torch.cat([a, b])
    fa = torch.searchsorted(a, grid, right=True).double() / a.numel()
    fb = torch.searchsorted(b, grid, right=True).double() / b.numel()
    return float((fa - fb).abs().max())


def _torch_draw(fn, shape, seed=0):
    torch.manual_seed(seed)
    w = torch.empty(shape, dtype=F64)
    fn(w)
    return w


def test_fans_and_gain_match_torch():
    for shape in [(7,), (5, 3), (8, 4, 3, 3), (2, 6, 5)]:
        if len(shape) >= 2:
            assert nn.fans(shape) == torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))
    for nl, prm in [("linear", None), ("tanh", None), ("relu", None), ("leaky_relu", None), ("leaky_relu", 0.2),
                    ("selu", None), ("conv2d", None), ("sigmoid", None)]:
        assert nn.gain(nl, prm) == pytest.approx(torch.nn.init.calculate_gain(nl, prm), rel=1e-15)


SCHEMES = {
    "xavier_uniform": (nn.xavier_uniform(2.0), lambda w: torch.nn.init.xavier_uniform_(w, gain=2.0)),
    "xavier_normal": (nn.xavier_normal(), torch.nn.init.xavier_normal_),
    "kaiming_uniform_relu": (nn.kaiming_uniform(nonlinearity="relu"),
                             lambda w: torch.nn.init.kaiming_uniform_(w, nonlinearity="relu")),
    "kaiming_normal_fan_out": (nn.kaiming_normal(a=0.1, mode="fan_out"),
                               lambda w: torch.nn.init.kaiming_normal_(w, a=0.1, mode="fan_out")),
    "trunc_normal": (nn.trunc_normal(0.02, -0.04, 0.04), lambda w: torch.nn.init.trunc_normal_(w, 0, 0.02, -0.04, 0.04)),
    "trunc_normal_wide": (nn.trunc_normal(1.0), lambda w: torch.nn.init.trunc_normal_(w)),
}


@pytest.mark.parametrize("name", list(SCHEMES))
def test_scheme_has_torch_init_distribution(name):
    ours, theirs = SCHEMES[name]
    shape = (384, 96, 3)          # fan_in 288, fan_out 1152
    w = ours.draw(shape, F64, param_key(0, "w"), "w")
    t = _torch_draw(theirs, shape)
    assert _ks(w, t) < 0.01, (name, _ks(w, t))       # 110592 draws each: 1% is about 5 sigma
    assert abs(float(w.std()) / float(t.std()) - 1) < 0.01
    if ours.kind.endswith("uniform"):
        b = ours.resolve(shape).scale
        assert float(w.abs().max()) < b and float(w.abs().max()) > 0.99 * b
    if ours.kind == "trunc_normal":
        assert float(w.min()) >= ours.lo and float(w.max()) <= ours.hi


def test_kaiming_uniform_sqrt5_is_the_torch_nn_default_bit_for_bit():
    """torch.nn.Linear's reset_parameters is kaiming_uniform_(a=sqrt(5)), a bound of 1/sqrt(fan_in):
    the same Init as nn.fan_in_uniform, so the same key gives the same bits."""
    shape = (16, 40)
    k = param_key(3, "w")
    a = nn.kaiming_uniform(a=math.sqrt(5)).draw(shape, torch.float32, k, "w")
    b = nn.fan_in_uniform(40).draw(shape, torch.float32, k, "w")
    assert abs(nn.kaiming_uniform(a=math.sqrt(5)).resolve(shape).scale - 1 / math.sqrt(40)) < 1e-16
    assert torch.equal(a, b)


def test_float32_trunc_normal_is_the_float64_draw_rounded():
    k = param_key(1, "w")
    a = nn.trunc_normal(0.02).draw((64, 64), torch.float32, k, "w")
    b = nn.trunc_normal(0.02).draw((64, 64), F64, k, "w")
    assert torch.equal(a, b.float())


def test_rules_override_one_parameter_and_leave_every_other_bit_alone():
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    base = net.init(seed=5)
    over = net.init(seed=5, rules={"0.w": nn.kaiming_normal(nonlinearity="relu"), "*.b": nn.zeros})
    assert not torch.equal(base["0"]["w"], over["0"]["w"])
    assert torch.equal(base["2"]["w"], over["2"]["w"])
    assert float(over["0"]["b"].abs().max()) == 0.0 and float(over["2"]["b"].abs().max()) == 0.0
    fn = net.init(seed=5, rules=lambda path, prm: nn.xavier_uniform() if len(prm.shape) == 2 else None)
    assert torch.equal(fn["0"]["b"], base["0"]["b"])
    assert float(fn["2"]["w"].abs().max()) < math.sqrt(6 / 20)
    assert torch.equal(net.init(seed=5, rules={"0.w": nn.xavier_uniform()})["0"]["w"],
                       net.init(seed=5, rules={"0.w": nn.xavier_uniform()})["0"]["w"])


def test_bad_inits_are_refused():
    with pytest.raises(ValueError):
        nn.trunc_normal(1.0, 2.0, -2.0)
    with pytest.raises(ValueError):
        nn.kaiming_normal(mode="fan")
    with pytest.raises(ValueError):
        nn.gain("gelu")
    with pytest.raises(TypeError):
        nn.Linear(2, 2).init(rules={"w": 0.1})
