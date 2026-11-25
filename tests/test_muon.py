"""truestar.optim.Muon: torch.optim.Muon's update, its Newton Schulz orthogonality measured against
float64 (intended distance and rounding, with dtype derived tolerances), eager and compiled from one
definition, faults named by TS1040 (docs/OPTIM.md "Muon", docs/TRAIN_NUMERICS.md)."""

import importlib.util
import math

import pytest
import torch

import truestar
from truestar import Tensor, nn, optim
from truestar.optim import muon as M

F64 = torch.float64
HAVE_NUMBA = importlib.util.find_spec("numba") is not None
CUDA = torch.cuda.is_available()


def test_eager_update_matches_torch_optim_muon():
    """Same buffer, Nesterov blend, decay and lr ratio as torch.optim.Muon; the bf16 iterations
    differ only by rounding (torch normalizes in bf16, Truestar in the parameter dtype), so the
    float32 parameters stay within a small fraction of the total change over 10 steps."""
    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(48, 32, generator=g)
    opt = optim.Muon(lr=0.02, weight_decay=0.1)
    params = {"w": w0.clone()}
    st = opt.init(params)
    tw = torch.nn.Parameter(w0.clone())
    to = torch.optim.Muon([tw], lr=0.02, weight_decay=0.1)
    for i in range(10):
        gr = torch.randn(48, 32, generator=g)
        params, st = opt.update(params, {"w": gr}, st, step=i)
        tw.grad = gr.clone()
        to.step()
    moved = float((tw.detach() - w0).norm())
    assert float((params["w"] - tw.detach()).norm()) < 0.05 * moved
    # torch's buffer is the same average
    assert torch.allclose(st["m"]["w"], to.state[tw]["momentum_buffer"], atol=1e-6)


def test_orthogonality_parts_and_tolerance_across_shapes_and_dtypes():
    """The rounding of a correct iteration stays below the typical estimate (so far below the
    4x tolerance) and below the worst case bound, for bf16, f16, f32 and f64, wide, tall, square,
    low rank plus noise and log spaced spectra; the intended part is the coefficients' own choice."""
    torch.manual_seed(1)
    for shape in [(64, 200), (200, 64), (128, 128), (32, 300)]:
        for kind in ("randn", "lowrank", "decay"):
            if kind == "randn":
                u = torch.randn(*shape)
            elif kind == "lowrank":
                u = torch.randn(shape[0], 8) @ torch.randn(8, shape[1]) + 1e-3 * torch.randn(*shape)
            else:
                k = min(shape)
                U, _ = torch.linalg.qr(torch.randn(shape[0], k))
                V, _ = torch.linalg.qr(torch.randn(shape[1], k))
                u = (U * torch.logspace(0, -3, k)) @ V.T
            for dt in ("bf16", "f16", "f32", "f64"):
                o = optim.Muon(ns_dtype=dt)
                r = M.orthogonality_report(u, o.orthogonalize(u), dtype=o.ns_torch_dtype)
                assert r.ok, r.line()
                assert r.rounding <= 1.5 * r.estimate, r.line()
                assert r.rounding <= r.bound, r.line()
    u = torch.randn(64, 200)
    r = M.orthogonality_report(u, optim.Muon().orthogonalize(u), dtype=torch.bfloat16)
    assert 0.15 < r.intended < 0.3 and r.intended_sv_range[0] > 0.6 and r.intended_sv_range[1] < 1.2


def test_convergent_coefficients_reach_the_polar_factor():
    u = torch.randn(64, 200)
    o = optim.Muon(ns_dtype="f32", ns_coefficients=M.CONVERGENT_COEFFICIENTS, ns_steps=30)
    r = M.orthogonality_report(u, o.orthogonalize(u), o.ns_coefficients, 30, dtype=torch.float32)
    assert r.ok and r.ortho_error < 1e-5 and r.intended < 1e-12


def test_torch_muon_iteration_is_measured_too():
    """torch.optim.Muon's own bf16 iteration passes the same verdict (and is reported)."""
    from torch.optim._muon import _zeropower_via_newtonschulz

    u = torch.randn(96, 160)
    r = M.orthogonality_report(u, _zeropower_via_newtonschulz(u, M.DEFAULT_COEFFICIENTS, 5, 1e-7),
                               dtype=torch.bfloat16, name="torch.optim.Muon")
    assert r.ok and r.rounding < 0.03, r.line()


@pytest.mark.parametrize("fault", ["f32 coefficient typo", "4 steps for 5", "6 steps for 5", "f32 one entry",
                                   "sign of b"])
def test_faults_are_named_ts1040(fault):
    u = torch.randn(64, 200)
    if fault == "f32 coefficient typo":
        o, dt = optim.Muon(ns_dtype="f32", ns_coefficients=(3.4445, -4.7750, 2.0135)).orthogonalize(u), torch.float32
    elif fault == "4 steps for 5":
        o, dt = optim.Muon(ns_steps=4).orthogonalize(u), torch.bfloat16
    elif fault == "6 steps for 5":
        o, dt = optim.Muon(ns_steps=6).orthogonalize(u), torch.bfloat16
    elif fault == "f32 one entry":
        o, dt = optim.Muon(ns_dtype="f32").orthogonalize(u).clone(), torch.float32
        o[3, 7] += 1e-3
    else:
        o, dt = optim.Muon(ns_coefficients=(3.4445, 4.7750, 2.0315)).orthogonalize(u), torch.bfloat16
    r = M.orthogonality_report(u, o, M.DEFAULT_COEFFICIENTS, 5, dtype=dt)
    assert not r.ok, r.line()
    assert r.issue().code == "TS1040"


def test_check_raises_ts1040_for_a_wrong_optimizer_config():
    p = {"w": torch.randn(32, 48)}
    good = optim.Muon()
    st = good.init(p)
    reps = good.check(p, {"w": torch.randn(32, 48)}, st, 0, raise_on_fail=True)
    assert reps[0].ok and reps[0].issue().code == "TS1041"
    bad = optim.Muon(ns_steps=3)
    reps = bad.check(p, {"w": torch.randn(32, 48)}, st, 0)
    # checked against its own coefficients and steps, a 3 step config is correct
    assert reps[0].ok


def _mlp64():
    return nn.Sequential(nn.Linear(6, 16, dtype="f64"), nn.Activation("tanh"), nn.Linear(16, 4, dtype="f64"))


NET = _mlp64()


@truestar.native
def mse64(p: NET.spec, x: Tensor["B", 6, "f64"], y: Tensor["B", 4, "f64"]) -> Tensor["f64"]:
    return nn.mse(NET(p, x), y)


def test_compiled_step_is_the_eager_definition_in_float64():
    """With the iteration in float64, the reference executor's step program and the eager update
    agree to 1e-12 over 6 steps (2D leaves Muon, biases AdamW)."""
    opt = optim.Muon(lr=0.05, weight_decay=0.01, ns_dtype="f64", adam_lr=1e-2)
    p = NET.init(seed=2)
    step = truestar.train_step(mse64, opt)
    ts = step.init(p)
    q, st = p, opt.init(p)
    grad = truestar.grad(mse64, wrt="p")
    gen = torch.Generator().manual_seed(5)
    for i in range(6):
        x, y = torch.randn(8, 6, generator=gen, dtype=F64), torch.randn(8, 4, generator=gen, dtype=F64)
        ts, _ = step(ts, x, y)
        _, g = grad(q, x, y)
        q, st = opt.update(q, g, st, step=i)
    for k in ("0", "2"):
        for kk in ("w", "b"):
            assert float((ts.params[k][kk] - q[k][kk]).abs().max()) < 1e-12


def test_non_matrix_leaves_are_adamw():
    """A bias under Muon follows torch.optim.AdamW with betas (0.9, 0.95)."""
    b0 = torch.randn(7)
    opt = optim.Muon(lr=0.02, adam_lr=3e-3, weight_decay=0.05)
    p, st = {"b": b0.clone()}, None
    st = opt.init(p)
    tb = torch.nn.Parameter(b0.clone().double())
    to = torch.optim.AdamW([tb], lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
    p = {"b": b0.clone().double()}
    st = opt.init(p)
    g = torch.Generator().manual_seed(1)
    for i in range(8):
        gr = torch.randn(7, generator=g, dtype=F64)
        p, st = opt.update(p, {"b": gr}, st, step=i)
        tb.grad = gr.clone()
        to.step()
    assert float((p["b"] - tb.detach()).abs().max()) < 1e-12


def _f32_net():
    return nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 8))


NET32 = _f32_net()


@truestar.native
def ce32(p: NET32.spec, x: Tensor["B", 16], y: Tensor["B", "i64"]) -> Tensor[()]:
    return nn.cross_entropy(NET32(p, x), y)


def _compiled_updates_are_orthogonal(executor):
    opt = optim.Muon(lr=0.02, weight_decay=0.01, adam_lr=3e-3)
    p = {k: {kk: t.float() for kk, t in v.items()} for k, v in NET32.init(seed=0).items()}
    dev = "cuda" if executor == "cuda" else "cpu"
    step = truestar.train_step(ce32, opt, executor=executor)
    ts = step.init({k: {kk: t.to(dev) for kk, t in v.items()} for k, v in p.items()})
    gen = torch.Generator().manual_seed(0)
    x, y = torch.randn(64, 16, generator=gen), torch.randint(0, 8, (64,), generator=gen)
    before = {k: v["w"].clone().cpu() for k, v in ts.params.items()}
    ts2, _ = step(ts, x.to(dev), y.to(dev))
    grad = truestar.grad(ce32, wrt="p")
    _, g = grad(p, x, y)
    for k in before:
        u, _ = opt.ns_input(g[k]["w"].float(), torch.zeros_like(before[k]))
        o = opt.recover_update(before[k], ts2.params[k]["w"].cpu(), 0)
        r = M.orthogonality_report(u, o, opt.ns_coefficients, opt.ns_steps, opt.eps, opt.ns_torch_dtype,
                                   name=f"compiled {k}", extra_error=opt.recover_resolution(before[k], 0))
        assert r.ok, r.line()


@pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba")
def test_compiled_cpu_updates_pass_the_float64_verdict():
    _compiled_updates_are_orthogonal("cpu")


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_compiled_cuda_updates_pass_the_float64_verdict():
    _compiled_updates_are_orthogonal("cuda")


def test_config_is_checked():
    with pytest.raises(ValueError):
        optim.Muon(ns_dtype="int8")
    with pytest.raises(ValueError):
        optim.Muon(adjust_lr="other")
    with pytest.raises(ValueError):
        optim.Muon(momentum=1.0)
    assert math.isclose(M.lr_ratio(512, 128, "original"), 2.0)
    assert math.isclose(M.lr_ratio(128, 512, "match_rms_adamw"), 0.2 * math.sqrt(512))
