"""truestar.optim additions: Lion, RMSprop, Adagrad against torch.optim (or the paper for Lion),
compiled and eager from one definition, float32 compiled against the float64 eager update."""

import pytest
import torch

import truestar
from truestar import Tensor, nn, optim

F64 = torch.float64


def _net(dt):
    return nn.Sequential(nn.Linear(5, 7, dtype=dt), nn.Activation("tanh"), nn.Linear(7, 3, dtype=dt))


NET = _net("f64")
NET32 = _net("f32")


@truestar.native
def mse_loss(p: NET.spec, x: Tensor["B", 5, "f64"], y: Tensor["B", 3, "f64"]) -> Tensor["f64"]:
    return nn.mse(NET(p, x), y)


@truestar.native
def mse_loss32(p: NET32.spec, x: Tensor["B", 5, "f32"], y: Tensor["B", 3, "f32"]) -> Tensor["f32"]:
    return nn.mse(NET32(p, x), y)


def _torch_twin(p):
    m = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)).double()
    with torch.no_grad():
        for i in (0, 2):
            m[i].weight.copy_(p[str(i)]["w"])
            m[i].bias.copy_(p[str(i)]["b"])
    return m


def _batches(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, 5, generator=g, dtype=F64), torch.randn(8, 3, generator=g, dtype=F64)) for _ in range(n)]


class _LionRef(torch.optim.Optimizer):
    """Lion written straight from Algorithm 2 of the paper, float64, for the comparison."""

    def __init__(self, params, lr, betas, weight_decay):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            b1, b2 = grp["betas"]
            for p in grp["params"]:
                st = self.state[p]
                m = st.setdefault("m", torch.zeros_like(p))
                c = b1 * m + (1 - b1) * p.grad
                p.mul_(1 - grp["lr"] * grp["weight_decay"])
                p.add_(torch.sign(c), alpha=-grp["lr"])
                st["m"] = b2 * m + (1 - b2) * p.grad


CASES = {
    "lion": (optim.Lion(lr=3e-3, weight_decay=0.1), lambda ps: _LionRef(ps, 3e-3, (0.9, 0.99), 0.1)),
    "lion_betas": (optim.Lion(lr=1e-3, b1=0.8, b2=0.95), lambda ps: _LionRef(ps, 1e-3, (0.8, 0.95), 0.0)),
    "rmsprop": (optim.RMSprop(lr=1e-2), lambda ps: torch.optim.RMSprop(ps, lr=1e-2)),
    "rmsprop_centered_momentum_wd": (
        optim.RMSprop(lr=5e-3, alpha=0.9, momentum=0.8, centered=True, weight_decay=0.01),
        lambda ps: torch.optim.RMSprop(ps, lr=5e-3, alpha=0.9, momentum=0.8, centered=True, weight_decay=0.01)),
    "adagrad": (optim.Adagrad(lr=5e-2), lambda ps: torch.optim.Adagrad(ps, lr=5e-2)),
    "adagrad_decay_init_wd": (
        optim.Adagrad(lr=5e-2, lr_decay=0.01, initial_accumulator_value=0.1, weight_decay=0.02),
        lambda ps: torch.optim.Adagrad(ps, lr=5e-2, lr_decay=0.01, initial_accumulator_value=0.1,
                                       weight_decay=0.02)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_compiled_step_matches_reference_for_30_steps(name):
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
        for i in (0, 2):
            for k, t in (("w", m[i].weight), ("b", m[i].bias)):
                worst = max(worst, float((ts.params[str(i)][k] - t.detach()).abs().max() / t.detach().abs().max()))
    assert worst < 1e-12, (name, worst)


@pytest.mark.parametrize("name", list(CASES))
def test_eager_update_is_the_compiled_one(name):
    opt = CASES[name][0]
    p = NET.init(seed=2)
    step = truestar.train_step(mse_loss, opt)
    ts = step.init(p)
    q, st = p, opt.init(p)
    for k, (x, y) in enumerate(_batches(5, seed=3)):
        ts, _ = step(ts, x, y)
        _, g = truestar.grad(mse_loss, wrt="p")(q, x, y)
        q, st = opt.update(q, g, st, step=k)
    for i in ("0", "2"):
        for w in ("w", "b"):
            assert torch.allclose(ts.params[i][w], q[i][w], rtol=1e-13, atol=1e-15)


HAVE_NUMBA = __import__("importlib").util.find_spec("numba") is not None


@pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba (the compiled CPU executor computes in real float32)")
@pytest.mark.parametrize("name", ["lion", "rmsprop_centered_momentum_wd", "adagrad_decay_init_wd"])
def test_float32_compiled_step_tracks_the_float64_update(name):
    """20 compiled float32 steps against the same 20 steps in float64: the float32 run stays within
    a few hundred float32 ulps of float64 (Lion within its sign flips: none on these batches)."""
    opt = CASES[name][0]
    p64 = NET.init(seed=5)
    p32 = {i: {w: t.float() for w, t in d.items()} for i, d in p64.items()}
    s64, s32 = truestar.train_step(mse_loss, opt), truestar.train_step(mse_loss32, opt, executor="cpu")
    t64, t32 = s64.init(p64), s32.init(p32)
    for x, y in _batches(20, seed=6):
        t64, _ = s64(t64, x, y)
        t32, _ = s32(t32, x.float(), y.float())
    assert s32.executor.last.report.fallbacks == []
    for i in ("0", "2"):
        for w in ("w", "b"):
            assert t32.params[i][w].dtype == torch.float32
            ref = t64.params[i][w]
            err = float((t32.params[i][w].double() - ref).abs().max() / ref.abs().max())
            assert err < 1e-4, (name, i, w, err)


def test_lion_sign_of_zero_is_zero_and_slot_count():
    opt = optim.Lion(lr=0.5)
    p = {"w": torch.tensor([1.0, 2.0, 3.0], dtype=F64)}
    g = {"w": torch.tensor([0.0, -1.0, 2.0], dtype=F64)}
    q, st = opt.update(p, g, opt.init(p), step=0)
    assert q["w"].tolist() == [1.0, 2.5, 2.5]
    assert set(st) == {"m"}
    assert optim.RMSprop(centered=True, momentum=0.5).slots == ("v", "a", "buf")


def test_bad_configs_are_refused():
    with pytest.raises(ValueError):
        optim.Lion(b1=1.0)
    with pytest.raises(ValueError):
        optim.RMSprop(alpha=1.5)
    with pytest.raises(ValueError):
        optim.RMSprop(momentum=1.0)
    with pytest.raises(ValueError):
        optim.Adagrad(lr_decay=-1.0)


# ----------------------------------------------------------------------------- schedules


def _torch_lrs(make, n, lr=1.0):
    """The learning rate torch's scheduler gives update k (0 based), k < n."""
    import warnings

    p = torch.nn.Parameter(torch.zeros(1))
    o = torch.optim.SGD([p], lr=lr)
    s = make(o)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            out.append(s.get_last_lr()[0])
            o.step()
            s.step()
    return out


LS = torch.optim.lr_scheduler

SCHEDULES = {
    "step": (optim.step_decay(0.3, every=7, gamma=0.5), lambda o: LS.StepLR(o, 7, 0.5), 0.3),
    "multistep": (optim.multistep(0.2, [30, 10, 45], gamma=0.3), lambda o: LS.MultiStepLR(o, [10, 30, 45], 0.3), 0.2),
    "exponential": (optim.exponential(0.1, 0.97), lambda o: LS.ExponentialLR(o, 0.97), 0.1),
    "polynomial": (optim.polynomial(0.1, 40, power=2.0), lambda o: LS.PolynomialLR(o, 40, 2.0), 0.1),
    "restarts": (optim.cosine_restarts(0.1, 9, final=1e-3), lambda o: LS.CosineAnnealingWarmRestarts(o, 9, 1, 1e-3), 0.1),
    "restarts_mult": (optim.cosine_restarts(0.1, 5, mult=2, final=1e-3),
                      lambda o: LS.CosineAnnealingWarmRestarts(o, 5, 2, 1e-3), 0.1),
    "one_cycle": (optim.one_cycle(0.4, 60), lambda o: LS.OneCycleLR(o, 0.4, total_steps=60), 0.4),
    "one_cycle_linear_3": (optim.one_cycle(0.4, 60, pct_start=0.25, anneal="linear", three_phase=True),
                           lambda o: LS.OneCycleLR(o, 0.4, total_steps=60, pct_start=0.25, anneal_strategy="linear",
                                                   three_phase=True), 0.4),
    "warmup_then_step": (optim.warmup(optim.step_decay(0.3, 10, 0.5), 5, start=0.1),
                         lambda o: LS.SequentialLR(o, [LS.LinearLR(o, 0.1, 1.0, 5), LS.StepLR(o, 10, 0.5)], [5]), 0.3),
}


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_schedules_match_torch_lr_scheduler(name):
    ours, make, lr = SCHEDULES[name]
    n = 60 if "one_cycle" in name else 100
    theirs = _torch_lrs(make, n, lr)
    for k, t in enumerate(theirs):
        assert abs(ours(k) - t) <= 1e-12 * max(abs(t), 1e-3), (name, k, ours(k), t)


def test_a_new_schedule_drives_the_compiled_step_without_recompiling():
    sched = optim.warmup(optim.cosine_restarts(1e-2, 4), 3)
    step = truestar.train_step(mse_loss, optim.SGD(lr=sched))
    ts = step.init(NET.init(seed=0))
    before = truestar.compile_count() if hasattr(truestar, "compile_count") else None
    lrs = []
    for x, y in _batches(10):
        ts, r = step(ts, x, y)
        lrs.append(r.lr)
    assert lrs == [sched(k) for k in range(10)]
    if before is not None:
        assert truestar.compile_count() == before
    assert step.config()["optimizer"]["lr"]["schedule"] == "Warmup"


def test_bad_schedules_are_refused():
    for bad in (lambda: optim.step_decay(0.1, 0), lambda: optim.cosine_restarts(0.1, 5, mult=0),
                lambda: optim.one_cycle(0.1, 10, anneal="exp"), lambda: optim.warmup(0.1, 0),
                lambda: optim.polynomial(0.1, 0)):
        with pytest.raises(ValueError):
            bad()


# ----------------------------------------------------------------------------- EMA of the weights


def test_ema_in_the_compiled_step_matches_torch_averaged_model(tmp_path):
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    p = NET.init(seed=7)
    step = truestar.train_step(mse_loss, optim.EMA(optim.AdamW(lr=1e-2, weight_decay=0.1), decay=0.9))
    ts = step.init(p)
    m = _torch_twin(p)
    to = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
    avg = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(0.9))
    avg.update_parameters(m)          # the first call copies: the average starts at the initial weights
    for x, y in _batches(20, seed=8):
        ts, _ = step(ts, x, y)
        to.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        to.step()
        avg.update_parameters(m)
    worst = 0.0
    for i in (0, 2):
        for k, t in (("w", avg.module[i].weight), ("b", avg.module[i].bias)):
            t = t.detach()
            worst = max(worst, float((ts.opt["ema"][str(i)][k] - t).abs().max() / t.abs().max()))
            assert not torch.equal(ts.opt["ema"][str(i)][k], ts.params[str(i)][k])
    assert worst < 1e-12, worst
    path = step.save(str(tmp_path / "s.safetensors"), ts)
    back = step.load(path)
    assert torch.equal(back.opt["ema"]["0"]["w"], ts.opt["ema"]["0"]["w"])


def test_ema_warmup_and_eager_update_are_one_definition():
    opt = optim.EMA(optim.SGD(lr=0.1, momentum=0.9), decay=0.999, warmup=True)
    p = NET.init(seed=9)
    step = truestar.train_step(mse_loss, opt)
    ts = step.init(p)
    q, st = p, opt.init(p)
    ref = {i: {w: t.clone() for w, t in d.items()} for i, d in p.items()}
    for k, (x, y) in enumerate(_batches(6, seed=10)):
        ts, _ = step(ts, x, y)
        _, g = truestar.grad(mse_loss, wrt="p")(q, x, y)
        q, st = opt.update(q, g, st, step=k)
        t = k + 1
        d = min(0.999, (1 + t) / (10 + t))
        ref = {i: {w: ref[i][w] + (q[i][w] - ref[i][w]) * (1 - d) for w in ref[i]} for i in ref}
    for i in ("0", "2"):
        for w in ("w", "b"):
            assert torch.allclose(ts.opt["ema"][i][w], st["ema"][i][w], rtol=1e-13, atol=1e-15)
            assert torch.allclose(st["ema"][i][w], ref[i][w], rtol=1e-13, atol=1e-15)
    assert set(st) == {"buf", "ema"}
    assert step.config()["optimizer"]["inner"]["optimizer"] == "SGD"


def test_ema_refuses_bad_configs():
    with pytest.raises(ValueError):
        optim.EMA(optim.SGD(), decay=1.0)
    with pytest.raises(ValueError):
        optim.EMA(optim.EMA(optim.SGD()))


# ----------------------------------------------------------------------------- clipping variants


def _grads(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(4, 3, generator=g, dtype=F64) * 5, torch.randn(7, generator=g, dtype=F64)]


@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_clip_by_global_norm_norm_types_match_torch(norm_type):
    gs = _grads(1)
    ps = [torch.nn.Parameter(torch.zeros_like(t)) for t in gs]
    for p, g in zip(ps, gs):
        p.grad = g.clone()
    want = torch.nn.utils.clip_grad_norm_(ps, 2.0, norm_type=norm_type)
    out, norm = optim.clip_by_global_norm(gs, 2.0, norm_type=norm_type)
    assert abs(float(norm) - float(want)) <= 1e-14 * float(want)
    for o, p in zip(out, ps):
        assert torch.allclose(o, p.grad, rtol=1e-15, atol=0)


def test_clip_by_value_matches_torch():
    gs = _grads(2)
    ps = [torch.nn.Parameter(torch.zeros_like(t)) for t in gs]
    for p, g in zip(ps, gs):
        p.grad = g.clone()
    torch.nn.utils.clip_grad_value_(ps, 0.7)
    for o, p in zip(optim.clip_by_value(gs, 0.7), ps):
        assert torch.equal(o, p.grad)


@pytest.mark.parametrize("kw,torch_clip", [
    ({"clip": 0.05, "clip_norm_type": float("inf")},
     lambda ps: torch.nn.utils.clip_grad_norm_(ps, 0.05, norm_type=float("inf"))),
    ({"clip_value": 0.02}, lambda ps: torch.nn.utils.clip_grad_value_(ps, 0.02)),
])
def test_compiled_step_with_clipping_variants_matches_torch(kw, torch_clip):
    p = NET.init(seed=11)
    step = truestar.train_step(mse_loss, optim.SGD(lr=0.5, momentum=0.9), **kw)
    ts = step.init(p)
    m = _torch_twin(p)
    to = torch.optim.SGD(m.parameters(), lr=0.5, momentum=0.9)
    worst = 0.0
    for x, y in _batches(15, seed=12):
        ts, r = step(ts, x, y)
        to.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        torch_clip(list(m.parameters()))
        to.step()
        for i in (0, 2):
            for k, t in (("w", m[i].weight), ("b", m[i].bias)):
                worst = max(worst, float((ts.params[str(i)][k] - t.detach()).abs().max() / t.detach().abs().max()))
    assert worst < 1e-12, worst
    assert "clip_norm_type" in step.config()


def test_clipping_options_are_checked():
    from truestar.fitting import FitError

    with pytest.raises(FitError):
        truestar.train_step(mse_loss, optim.SGD(), clip=1.0, clip_value=1.0)
    with pytest.raises(FitError):
        truestar.train_step(mse_loss, optim.SGD(), clip=1.0, clip_norm_type=3.0)
    with pytest.raises(FitError):
        truestar.train_step(mse_loss, optim.SGD(), clip_value=-1.0)
