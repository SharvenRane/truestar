"""truestar.train_step and truestar.fit: one IR program per step from a native loss, against
PyTorch training loops in float64; model state, dropout replay, checkpoints, mixed precision and
non finite steps (docs/OPTIM.md, docs/NN.md)."""

import math

import pytest
import torch
import torch.nn.functional as F

import truestar
from truestar import Tensor, nn, ops, optim
from truestar.fitting import FitError, NonFiniteStep

F64 = torch.float64

MLP = nn.Sequential(nn.Linear(6, 16, dtype="f64"), nn.GELU(), nn.Dropout(0.25), nn.Linear(16, 4, dtype="f64"))


@truestar.native(seed=5)
def mlp_loss(p: MLP.spec, x: Tensor["B", 6, "f64"], y: Tensor["B", "i64"], training: bool = False):
    logits = MLP(p, x, training=training)
    return nn.cross_entropy(logits, y), {"acc": nn.accuracy(logits, y)}


def batches(n, b=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(b, 6, generator=g, dtype=F64)
        out.append((x, (x[:, 0] > 0).long() + 2 * (x[:, 1] > 0).long()))
    return out


def torch_mlp(p):
    m = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)).double()
    with torch.no_grad():
        m[0].weight.copy_(p["0"]["w"]); m[0].bias.copy_(p["0"]["b"])     # noqa: E702
        m[2].weight.copy_(p["3"]["w"]); m[2].bias.copy_(p["3"]["b"])     # noqa: E702
    return m


def test_fit_matches_a_pytorch_loop_with_schedule_and_clipping():
    opt = optim.AdamW(lr=optim.warmup_cosine(5e-2, 5, 25, final=1e-3), weight_decay=0.05)
    step = truestar.train_step(mlp_loss, opt, clip=0.3)
    p0 = MLP.init(seed=2)
    ts, hist = truestar.fit(step, step.init(p0), batches(25))
    m = torch_mlp(p0)
    to = torch.optim.AdamW(m.parameters(), lr=1.0, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(to, opt.lr_at)
    worst = 0.0
    for r, (x, y) in zip(hist, batches(25)):
        to.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.3)
        to.step()
        sch.step()
        worst = max(worst, abs(r.loss - loss.item()) / loss.item(), abs(r.grad_norm - float(norm)) / float(norm))
        assert r.lr == opt.lr_at(r.step)
    worst = max(worst, float((ts.params["3"]["w"] - m[2].weight.detach()).abs().max()))
    assert worst < 1e-12, worst
    assert ts.step == 25 and len(hist) == 25
    assert 0.0 <= float(hist[-1].aux["acc"]) <= 1.0 and hist[-1].loss < hist[0].loss


CNN = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1, dtype="f64"), nn.BatchNorm(4, dtype="f64"), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * 4 * 4, 3, dtype="f64"))


@truestar.native
def cnn_loss(p: CNN.spec, s: CNN.state_spec, x: Tensor["B", 2, 8, 8, "f64"], y: Tensor["B", "i64"]):
    logits, s2 = CNN(p, x, training=True, state=s)
    return nn.cross_entropy(logits, y), s2


def test_batch_norm_state_is_carried_by_the_step_like_torch_running_stats():
    step = truestar.train_step(cnn_loss, optim.SGD(lr=0.1, momentum=0.9, nesterov=True), state="s")
    p0 = CNN.init(seed=1)
    ts = step.init(p0, CNN.init_state())
    m = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                            torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(64, 3)).double()
    with torch.no_grad():
        m[0].weight.copy_(p0["0"]["w"]); m[0].bias.copy_(p0["0"]["b"])     # noqa: E702
        m[5].weight.copy_(p0["5"]["w"]); m[5].bias.copy_(p0["5"]["b"])     # noqa: E702
    to = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    for _ in range(8):
        x = torch.randn(5, 2, 8, 8, generator=g, dtype=F64) * 2 + 0.5
        y = torch.randint(0, 3, (5,), generator=g)
        ts, r = step(ts, x, y)
        to.zero_grad()
        loss = F.cross_entropy(m(x), y)
        loss.backward()
        to.step()
        worst = max(worst, abs(r.loss - loss.item()) / loss.item())
    worst = max(worst, float((ts.model_state["1"]["mean"] - m[1].running_mean).abs().max()),
                float((ts.model_state["1"]["var"] - m[1].running_var).abs().max()),
                float((ts.params["5"]["w"] - m[5].weight.detach()).abs().max()))
    assert worst < 1e-12, worst
    with pytest.raises(FitError, match="model state"):
        step.init(p0)


def test_dropout_replays_bit_for_bit_and_resume_equals_a_straight_run(tmp_path):
    opt = optim.Adam(lr=1e-2)
    step = truestar.train_step(mlp_loss, opt, static={"training": True})
    assert step.program.op_counts().get("random", 0) >= 1
    data = batches(6)
    straight, h1 = truestar.fit(step, step.init(MLP.init(0)), data)
    again, h2 = truestar.fit(step, step.init(MLP.init(0)), data)
    assert [r.loss for r in h1] == [r.loss for r in h2]                 # keyed masks: same bits
    half, _ = truestar.fit(step, step.init(MLP.init(0)), data[:3], checkpoint=str(tmp_path / "c.safetensors"))
    resumed = step.load(str(tmp_path / "c.safetensors"))
    assert resumed.step == 3
    resumed, h3 = truestar.fit(step, resumed, data[3:])
    assert [r.loss for r in h3] == [r.loss for r in h1[3:]]
    for k in ("0", "3"):
        for kk in ("w", "b"):
            assert torch.equal(resumed.params[k][kk], straight.params[k][kk])
            assert torch.equal(resumed.opt["v"][k][kk], straight.opt["v"][k][kk])
    # training on and off are two different programs; off has no dropout in it
    off = truestar.train_step(mlp_loss, opt)
    assert off.program.op_counts().get("random", 0) == 0


def test_a_checkpoint_of_another_program_is_refused(tmp_path):
    a = truestar.train_step(mlp_loss, optim.Adam(lr=1e-2))
    b = truestar.train_step(mlp_loss, optim.Adam(lr=1e-2, b2=0.99))
    path = a.save(str(tmp_path / "a.safetensors"), a.init(MLP.init(0)))
    with pytest.raises(FitError, match="another step program") as ei:
        b.load(path)
    assert ei.value.code == "TS764" and str(ei.value).startswith("[TS764]")
    from truestar.interop.safetensors import save_file
    bad = str(tmp_path / "bad.safetensors")
    save_file({"x": torch.zeros(2)}, bad)
    with pytest.raises(FitError, match="not a truestar training checkpoint"):
        a.load(bad)


def test_one_program_serves_every_batch_size():
    class Counting(truestar.frontend.ReferenceExecutor):
        prepares = 0

        def prepare(self, program):
            Counting.prepares += 1
            return super().prepare(program)

    step = truestar.train_step(mlp_loss, optim.SGD(lr=0.1), executor=Counting())
    ts = step.init(MLP.init(0))
    for b in (3, 7, 12):
        ts, _ = step(ts, *batches(1, b=b)[0])
    assert Counting.prepares == 1 and ts.step == 3


NET32 = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 4))


@truestar.native
def loss32(p: NET32.spec, x: Tensor["B", 6], y: Tensor["B", "i64"]) -> Tensor[()]:
    return nn.cross_entropy(NET32(p, x), y)


def test_bfloat16_autocast_trains_close_to_float32():
    data = [(x.float(), y) for x, y in batches(15)]
    full = truestar.train_step(loss32, optim.AdamW(lr=1e-2))
    half = truestar.train_step(loss32, optim.AdamW(lr=1e-2), precision="bf16")
    assert half.program.op_counts().get("cast", 0) > full.program.op_counts().get("cast", 0)
    _, h32 = truestar.fit(full, full.init(NET32.init(0)), data)
    ts16, h16 = truestar.fit(half, half.init(NET32.init(0)), data)
    assert ts16.params["0"]["w"].dtype == torch.float32                  # master weights stay float32
    rel = max(abs(a.loss - b.loss) / b.loss for a, b in zip(h16, h32))
    assert 1e-6 < rel < 5e-2, rel        # bfloat16 rounding shows, and the curve follows float32


def test_float16_loss_scaling_skips_an_overflowing_step_and_backs_off():
    step = truestar.train_step(loss32, optim.SGD(lr=0.1), precision="f16",
                               loss_scale=truestar.train.LossScaler(init_scale=2.0 ** 30, growth_interval=3))
    ts = step.init(NET32.init(0))
    before = ts.params["0"]["w"].clone()
    x, y = batches(1)[0]
    skips = 0
    for _ in range(30):
        ts, r = step(ts, x.float(), y)
        if not r.skipped:
            break
        skips += 1
        assert torch.equal(ts.params["0"]["w"], before)                  # an overflowing step is not applied
        assert float(ts.amp["scale"]) == 2.0 ** (30 - skips)              # and the scale halves
    assert 1 <= skips < 30 and not r.skipped
    assert not torch.equal(ts.params["0"]["w"], before)
    # afterwards the scale follows GradScaler's rule exactly: halve on a skip, double after 3 finite steps
    scale, tracker = float(ts.amp["scale"]), 1
    for _ in range(8):
        ts, r = step(ts, x.float(), y)
        if r.skipped:
            scale, tracker = scale * 0.5, 0
        else:
            tracker += 1
            if tracker == 3:
                scale, tracker = scale * 2.0, 0
        assert float(ts.amp["scale"]) == scale and float(ts.amp["growth"]) == tracker


@truestar.native
def sqrt_loss(p: {"w": Tensor[3, 1, "f64"]}, x: Tensor["B", 3, "f64"]) -> Tensor["f64"]:
    return ops.sum(ops.sqrt(x @ p["w"]))


def test_a_non_finite_step_is_refused_and_names_the_op():
    step = truestar.train_step(sqrt_loss, optim.SGD(lr=0.1))
    ts = step.init({"w": torch.tensor([[1.0], [-2.0], [0.5]], dtype=F64)})
    with pytest.raises(NonFiniteStep) as ei:
        step(ts, torch.tensor([[1.0, 1.0, 1.0]], dtype=F64))
    assert "sqrt" in str(ei.value) and "test_fit.py" in str(ei.value) and ei.value.code == "TS763"
    assert ts.step == 0                                                   # the old state is untouched


def test_errors_say_what_to_change():
    with pytest.raises(FitError, match="no tensor parameter 'params'"):
        truestar.train_step(loss32, optim.SGD(lr=0.1), params="params")
    with pytest.raises(FitError, match="truestar.optim"):
        truestar.train_step(loss32, torch.optim.SGD)
    with pytest.raises(FitError, match="precision"):
        truestar.train_step(loss32, optim.SGD(lr=0.1), precision="fp8")
    with pytest.raises(FitError, match="must return \\(loss, new s\\)"):
        @truestar.native
        def bad(p: CNN.spec, s: CNN.state_spec, x: Tensor["B", 2, 8, 8, "f64"]) -> Tensor["f64"]:
            return ops.sum(CNN(p, x, training=True, state=s)[0])
        truestar.train_step(bad, optim.SGD(lr=0.1), state="s")
