"""truestar.optim: optimizers against torch.optim, eager and compiled from one definition, the
expm1 bias correction against the 1 - b**t cancellation, schedules and clipping (docs/OPTIM.md)."""

import math

import pytest
import torch

import truestar
from truestar import Tensor, nn, ops, optim

F64 = torch.float64


def _mlp():
    return nn.Sequential(nn.Linear(5, 7, dtype="f64"), nn.Activation("tanh"), nn.Linear(7, 3, dtype="f64"))


NET = _mlp()


@truestar.native
def mse_loss(p: NET.spec, x: Tensor["B", 5, "f64"], y: Tensor["B", 3, "f64"]) -> Tensor["f64"]:
    return nn.mse(NET(p, x), y)


def _torch_twin(p):
    m = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)).double()
    with torch.no_grad():
        for i in ("0", "2"):
            m[int(i)].weight.copy_(p[i]["w"])
            m[int(i)].bias.copy_(p[i]["b"])
    return m


def _batches(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, 5, generator=g, dtype=F64), torch.randn(8, 3, generator=g, dtype=F64)) for _ in range(n)]


CASES = {
    "sgd": (optim.SGD(lr=0.1), lambda ps: torch.optim.SGD(ps, lr=0.1)),
    "momentum_wd": (optim.SGD(lr=0.05, momentum=0.9, weight_decay=0.01),
                    lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, weight_decay=0.01)),
    "nesterov": (optim.SGD(lr=0.05, momentum=0.8, nesterov=True),
                 lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.8, nesterov=True)),
    "adam_l2": (optim.Adam(lr=3e-3, b1=0.8, b2=0.99, weight_decay=0.05),
                lambda ps: torch.optim.Adam(ps, lr=3e-3, betas=(0.8, 0.99), weight_decay=0.05)),
    "adamw": (optim.AdamW(lr=1e-2, weight_decay=0.1), lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.1)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_compiled_step_matches_torch_optim_for_30_steps(name):
    ours, theirs = CASES[name]
    p = NET.init(seed=4)
    step = truestar.train_step(mse_loss, ours)
    ts = step.init(p)
    m = _torch_twin(p)
    to = theirs(m.parameters())
    worst = 0.0
    for x, y in _batches(30):
        ts, r = step(ts, x, y)
        to.zero_grad()
        loss = torch.nn.functional.mse_loss(m(x), y)
        loss.backward()
        to.step()
        worst = max(worst, abs(r.loss - loss.item()) / loss.item())
        for i in ("0", "2"):
            for k, t in (("w", m[int(i)].weight), ("b", m[int(i)].bias)):
                worst = max(worst, float((ts.params[i][k] - t.detach()).abs().max() / t.detach().abs().max()))
    assert worst < 1e-12, worst


def test_eager_update_is_the_same_definition_as_the_compiled_one():
    opt = optim.AdamW(lr=optim.warmup_cosine(1e-2, 2, 6), weight_decay=0.1)
    p = NET.init(seed=1)
    step = truestar.train_step(mse_loss, opt)
    ts = step.init(p)
    q, st = p, opt.init(p)
    grad = truestar.grad(mse_loss, wrt="p")
    for i, (x, y) in enumerate(_batches(6)):
        ts, _ = step(ts, x, y)
        _, g = grad(q, x, y)
        q, st = opt.update(q, g, st, step=i)
    for k in ("0", "2"):
        for kk in ("w", "b"):
            assert float((ts.params[k][kk] - q[k][kk]).abs().max()) < 1e-15
            assert float((ts.opt["v"][k][kk] - st["v"][k][kk]).abs().max()) < 1e-18
    assert p["0"]["w"] is not q["0"]["w"]   # pure: the input tree is never written


def _bias_correction_program():
    """-expm1(t ln b) and the naive 1 - exp(t ln b), as one program over a vector of step counts."""
    @truestar.native
    def bc(t: Tensor["N"]):
        x = t * math.log(0.999)
        return -ops.expm1(x), 1.0 - ops.exp(x)
    return bc


HAVE_NUMBA = __import__("importlib").util.find_spec("numba") is not None


@pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba (the compiled CPU executor computes in real float32)")
def test_expm1_bias_correction_removes_the_float32_cancellation():
    """Computed in float32 by the compiled CPU executor (its first call check is off here so the
    naive formula's raw float32 numbers come back): 1 - 0.999^t loses about 1e-5 relative at t = 1,
    -expm1(t ln 0.999) stays within a few float32 ulps at every t."""
    from truestar.compiler import CompiledExecutor

    bc = _bias_correction_program()
    bc32 = truestar.native(bc.python, executor=CompiledExecutor("cpu", "float32", verify="off"))
    t = torch.tensor([1.0, 2.0, 3.0, 10.0, 100.0, 1000.0, 10000.0, 1e6])
    ours, naive = bc32(t)
    assert ours.dtype == torch.float32
    exact = torch.tensor([-math.expm1(float(v) * math.log(0.999)) for v in t], dtype=F64)
    r_ours = float(((ours.double() - exact).abs() / exact).max())
    r_naive = float(((naive.double() - exact).abs() / exact).max())
    assert r_ours < 4 * 2 ** -24, r_ours           # a few float32 ulps
    assert r_naive > 5e-6, r_naive                 # the cancellation of 1 - b^t (docs/MEMORY.md Limits)


@pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba (compiled CPU)")
def test_compiled_float32_adamw_step_passes_its_first_call_check():
    """The kernel check that refused a compiled AdamW step (docs/MEMORY.md Limits) now passes: the
    compiled executor compares the first call with the float64 reference and raises on a mismatch."""
    net = nn.Sequential(nn.Linear(5, 4), nn.Activation("tanh"), nn.Linear(4, 3))

    @truestar.native
    def loss(p: net.spec, x: Tensor["B", 5], y: Tensor["B", 3]) -> Tensor[()]:
        return nn.mse(net(p, x), y)

    step = truestar.train_step(loss, optim.AdamW(lr=1e-3), executor="cpu")
    ts = step.init(net.init(0))
    for x, y in _batches(3):
        ts, r = step(ts, x.float(), y.float())
    assert math.isfinite(r.loss)
    assert step.executor.last.report.fallbacks == []


def test_schedules():
    s = optim.warmup_cosine(1.0, warmup=4, total=12, final=0.1)
    assert [s(i) for i in range(4)] == [0.0, 0.25, 0.5, 0.75]
    assert s(4) == 1.0 and abs(s(8) - 0.55) < 1e-15 and s(12) == 0.1 and s(50) == 0.1
    lin = optim.warmup_linear(2.0, 2, 6)
    assert [lin(i) for i in range(7)] == [0.0, 1.0, 2.0, 1.5, 1.0, 0.5, 0.0]
    assert optim.SGD(lr=0.3).lr_at(99) == 0.3
    # a schedule matches torch's LambdaLR around the same function step for step (lr is a host value,
    # so the step program is the same for every learning rate)
    m = torch.nn.Linear(2, 2)
    to = torch.optim.SGD(m.parameters(), lr=1.0)
    sch = torch.optim.lr_scheduler.LambdaLR(to, s)
    for i in range(14):
        assert to.param_groups[0]["lr"] == s(i)
        to.step()
        sch.step()


def test_clip_by_global_norm_matches_torch():
    gs = [torch.randn(4, 3, dtype=F64), torch.randn(5, dtype=F64) * 3]
    clipped, norm = optim.clip_by_global_norm(gs, 1.0)
    ts = [g.clone().requires_grad_() for g in gs]
    for t, g in zip(ts, gs):
        t.grad = g.clone()
    tn = torch.nn.utils.clip_grad_norm_(ts, 1.0)
    assert abs(float(norm) - float(tn)) < 1e-14
    for a, t in zip(clipped, ts):
        assert float((a - t.grad).abs().max()) < 1e-15
    small, _ = optim.clip_by_global_norm([g * 1e-3 for g in gs], 1.0)
    assert all(torch.equal(a, g * 1e-3) for a, g in zip(small, gs))


def test_bad_configs_are_refused():
    with pytest.raises(ValueError, match="b2"):
        optim.Adam(b2=1.0)
    with pytest.raises(ValueError, match="Nesterov"):
        optim.SGD(lr=0.1, nesterov=True)
    with pytest.raises(ValueError, match="lr"):
        optim.SGD(lr=-1.0)
