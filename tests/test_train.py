"""Training on the IR (truestar.train) checked against PyTorch.

1. losses against torch.nn.functional in float64, values and gradients;
2. optimizers against torch.optim step for step: an MLP, a small CNN and a TransformerEncoderLayer
   imported with torch.export, 20 steps per optimizer, losses and every parameter;
3. the training step is one program, the trainer's schedule, NaN attribution and checkpoints;
4. mixed precision against torch.autocast with torch.amp.GradScaler, including when steps are skipped;
5. data parallel training on 2 gloo ranks against 1 rank on the concatenated batch;
6. tests that show the checks fail on broken optimizer rules.
"""

import copy
import io
import math
import os
import time
import warnings
from functools import lru_cache

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

from truestar import ir
from truestar import train as T
from truestar.ir import autodiff, grad_rules, serialize
from truestar.train import ops as train_ops
from truestar.train.step import loss_and_grad

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

F64 = torch.float64
BATCH = Dim("B", min=1, max=64)

# Bar for float64 training against torch.optim. Both sides do the same arithmetic in a slightly
# different order (fused addcmul, lerp, sum orders), so each step differs by a few units in the last
# place, about 1e-16 relative; 20 steps of a contracting optimizer keep that below 1e-14 (measured
# worst 2.6e-15 below, in the transformer). A wrong rule (a missing bias correction, momentum or
# decay term) changes an update by 1e-3 or more relative. 1e-10 is the bar the IR tests use.
TRAIN_TOL = 1e-10


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = max(a.abs().max().item(), b.abs().max().item()) if a.numel() else 0.0
    if scale == 0.0:
        return 0.0
    return (a.double() - b.double()).abs().max().item() / scale


RESULTS = {}  # filled by the tests, printed at the end (pytest -s) for docs/TRAINING.md


# ----------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.body(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 8, 3, stride=2)
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        return self.fc(F.gelu(self.c2(x)).flatten(1))


class TinyLM(nn.Module):
    """A TransformerEncoderLayer with a vocabulary head, trained with the causal LM loss."""

    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True)
        self.head = nn.Linear(16, 11)

    def forward(self, x):
        return self.head(self.layer(x))


def _mlp_data(g, n):
    x = torch.randn(n, 6, generator=g, dtype=F64)
    return x, torch.randn(n, 3, generator=g, dtype=F64)


def _cnn_data(g, n):
    return torch.randn(n, 3, 12, 12, generator=g, dtype=F64), torch.randint(0, 5, (n,), generator=g)


def _lm_data(g, n):
    return torch.randn(n, 6, 16, generator=g, dtype=F64), torch.randint(0, 11, (n, 6), generator=g)


MODELS = {
    # name: (constructor, example input shape, IR loss, torch loss, data)
    "mlp": (MLP, (6,), lambda: T.MSE(), lambda out, y: F.mse_loss(out, y), _mlp_data),
    "cnn": (CNN, (3, 12, 12), lambda: T.CrossEntropy(label_smoothing=0.1),
            lambda out, y: F.cross_entropy(out, y, label_smoothing=0.1), _cnn_data),
    "transformer": (TinyLM, (6, 16), lambda: T.CausalLM(),
                    lambda out, y: F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]), y[:, 1:].reshape(-1)),
                    _lm_data),
}


@lru_cache(maxsize=None)
def imported(name):
    """(float64 module, its IR program from torch.export). Import a copy: the IR shares parameter
    storage with the module it came from, and torch.optim updates its module in place."""
    ctor, shape = MODELS[name][:2]
    torch.manual_seed(0)
    model = ctor().double().eval()
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, *shape, dtype=F64),), dynamic_shapes=({0: BATCH},))
    return model, prog


def lion_reference(params, state, lr, betas, weight_decay):
    """Lion as written in the paper (Chen et al. 2023, Algorithm 2), in plain torch. torch.optim has
    no Lion, so this is the reference."""
    b1, b2 = betas
    with torch.no_grad():
        for p in params:
            m = state.setdefault(p, torch.zeros_like(p))
            c = b1 * m + (1 - b1) * p.grad
            p.mul_(1 - lr * weight_decay)
            p.add_(torch.sign(c), alpha=-lr)
            m.mul_(b2).add_(p.grad, alpha=1 - b2)


OPTIMIZERS = {
    # name: (IR optimizer, torch optimizer factory or None for Lion)
    "sgd_momentum": (lambda: T.SGD(0.05, momentum=0.9, dampening=0.1, weight_decay=1e-3),
                     lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, dampening=0.1, weight_decay=1e-3)),
    "sgd_nesterov": (lambda: T.SGD(0.05, momentum=0.9, weight_decay=1e-3, nesterov=True),
                     lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, weight_decay=1e-3, nesterov=True)),
    "adamw": (lambda: T.AdamW(3e-3, betas=(0.9, 0.99), weight_decay=0.05),
              lambda ps: torch.optim.AdamW(ps, lr=3e-3, betas=(0.9, 0.99), weight_decay=0.05)),
    "lion": (lambda: T.Lion(3e-4, betas=(0.9, 0.99), weight_decay=0.1), None),
}


def key_bias_slices(name, model):
    """Entries whose exact gradient is zero, so both sides move them by rounding noise only.

    The key bias of attention adds q . b_k to every score of a row, and softmax ignores a constant
    shift of a row, so the gradient of in_proj_bias[d:2d] is exactly zero in exact arithmetic
    (measured: 1e-18 against 1e-2 for the query bias). Sign based Lion moves those entries by
    +-lr depending on the sign of the noise, so no two implementations agree on them. They are
    compared separately (see test_key_bias_gradient_is_zero)."""
    if name != "transformer":
        return {}
    d = model.layer.self_attn.embed_dim
    return {"layer.self_attn.in_proj_bias": slice(d, 2 * d)}


def compare_training(name, opt_name, steps=20, clip=None, ir_opt=None, torch_opt=None):
    """Train the same model with the IR step program and with torch, step for step. Returns the
    worst relative error over losses and every parameter after every step."""
    model, prog = imported(name)
    loss_ir, loss_t, data = MODELS[name][2], MODELS[name][3], MODELS[name][4]
    mk_ir, mk_torch = OPTIMIZERS[opt_name] if opt_name else (ir_opt, torch_opt)
    step = T.build_train_step(prog, loss_ir(), mk_ir(), model, clip_grad_norm=clip)
    trainer = T.Trainer(step)
    ref = copy.deepcopy(model).train()
    params = dict(ref.named_parameters())
    opt = mk_torch(ref.parameters()) if mk_torch else None
    lion_state = {}
    skip = key_bias_slices(name, model)
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    losses = []
    for k in range(steps):
        x, y = data(g, 4)
        r = trainer.train_step(x, y)
        ref.zero_grad()
        want = loss_t(ref(x), y)
        want.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(ref.parameters(), clip)
        if opt is not None:
            opt.step()
        else:
            o = step.optimizer
            lion_reference(list(ref.parameters()), lion_state, o.lr, o.betas, o.weight_decay)
        worst = max(worst, abs(r.loss - want.item()) / abs(want.item()))
        for n, p in params.items():
            ours, theirs = trainer.state[n], p.detach()
            if n in skip:
                keep = torch.ones(ours.shape[0], dtype=torch.bool)
                keep[skip[n]] = False
                ours, theirs = ours[keep], theirs[keep]
            worst = max(worst, rel_err(ours, theirs))
        losses.append(r.loss)
    return worst, losses, step


# ----------------------------------------------------------------------------- 1. losses


def _logits_program(shape, dtype=F64):
    b = ir.Builder("logits")
    x = b.input("x", dtype, shape)
    return b.build([b.op("mul", x, b.param("one", torch.tensor(1.0, dtype=F64)))])


@pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_cross_entropy_matches_torch(smoothing, reduction):
    prog = _logits_program(("N", 7))
    lp, targets = T.attach_loss(prog, T.CrossEntropy(label_smoothing=smoothing, reduction=reduction))
    gp = T.grad(lp, ["x"])
    assert targets == ["target"]
    gen = torch.Generator().manual_seed(3)
    for n in (1, 5):  # one program, two batch sizes
        x = torch.randn(n, 7, generator=gen, dtype=F64) * 3
        y = torch.randint(0, 7, (n,), generator=gen)
        y[0] = -100  # ignored row
        if n == 1:
            y[0] = 2
        loss, gx = ir.run(gp, x, y)
        xr = x.clone().requires_grad_(True)
        want = F.cross_entropy(xr, y, label_smoothing=smoothing, reduction=reduction)
        (gw,) = torch.autograd.grad(want, xr)
        assert abs(loss.item() - want.item()) <= 1e-14 * abs(want.item())
        assert rel_err(gx, gw) < 1e-14


def test_cross_entropy_is_stable_for_huge_logits():
    """The log softmax shift: logits of 1e4 overflow exp() in float64 without it."""
    lp, _ = T.attach_loss(_logits_program((3, 4)), T.CrossEntropy())
    gp = T.grad(lp, ["x"])
    x = torch.tensor([[1e4, -1e4, 0.0, 5e3], [-3e4, 2e4, 2e4, 0.0], [1.0, 2.0, 3.0, 4.0]], dtype=F64)
    y = torch.tensor([0, 2, 3])
    loss, gx = ir.run(gp, x, y)
    xr = x.clone().requires_grad_(True)
    want = F.cross_entropy(xr, y)
    (gw,) = torch.autograd.grad(want, xr)
    assert math.isfinite(loss.item()) and torch.isfinite(gx).all()
    assert abs(loss.item() - want.item()) <= 1e-14 * abs(want.item())
    assert rel_err(gx, gw) < 1e-14


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_mse_matches_torch(reduction):
    lp, _ = T.attach_loss(_logits_program(("N", 3)), T.MSE(reduction=reduction))
    gp = T.grad(lp, ["x"])
    gen = torch.Generator().manual_seed(4)
    x, y = torch.randn(6, 3, generator=gen, dtype=F64), torch.randn(6, 3, generator=gen, dtype=F64)
    loss, gx = ir.run(gp, x, y)
    xr = x.clone().requires_grad_(True)
    want = F.mse_loss(xr, y, reduction=reduction)
    (gw,) = torch.autograd.grad(want, xr)
    assert abs(loss.item() - want.item()) <= 1e-14 * abs(want.item())
    assert rel_err(gx, gw) < 1e-14


def test_causal_lm_loss_matches_torch():
    lp, targets = T.attach_loss(_logits_program(("B", 5, 9)), T.CausalLM(label_smoothing=0.05))
    assert targets == ["tokens"]
    gp = T.grad(lp, ["x"])
    gen = torch.Generator().manual_seed(5)
    for b in (1, 3):
        x = torch.randn(b, 5, 9, generator=gen, dtype=F64)
        tok = torch.randint(0, 9, (b, 5), generator=gen)
        tok[0, 3] = -100  # padding is ignored, as with F.cross_entropy(ignore_index=-100)
        loss, gx = ir.run(gp, x, tok)
        xr = x.clone().requires_grad_(True)
        want = F.cross_entropy(xr[:, :-1].reshape(-1, 9), tok[:, 1:].reshape(-1), label_smoothing=0.05)
        (gw,) = torch.autograd.grad(want, xr)
        assert abs(loss.item() - want.item()) <= 1e-14 * abs(want.item())
        assert rel_err(gx, gw) < 1e-14
        assert gx[:, -1].abs().max().item() == 0.0  # the last position predicts nothing


def _fd_rel_err(prog, x, h=1e-3):
    """Fourth order central differences of the reference interpreter against the gradient program,
    for the scalar sum(v * f(x)) (the bar and its derivation are those of tests/test_autodiff.py)."""
    gen = torch.Generator().manual_seed(9)
    out = ir.run(prog, x)[0]
    v = torch.randn(out.shape, generator=gen, dtype=F64)
    gp = T.grad(prog, ["x"], seed="input")
    analytic = ir.run(gp, x, v)[1]
    fd = torch.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.numel()):
        vals = []
        for k in (-2, -1, 1, 2):
            xp = flat.clone()
            xp[i] += k * h
            vals.append((ir.run(prog, xp.reshape(x.shape))[0] * v).sum().item())
        fd.reshape(-1)[i] = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
    return rel_err(analytic, fd)


def test_new_op_gradients_match_finite_differences():
    gen = torch.Generator().manual_seed(8)
    b = ir.Builder("lsm")
    x = b.input("x", F64, (3, 5))
    prog = b.build([b.op("log_softmax", x, dim=1)])
    assert _fd_rel_err(prog, torch.randn(3, 5, generator=gen, dtype=F64)) < 1e-8
    b = ir.Builder("lsm0")
    x = b.input("x", F64, (4, 3))
    prog = b.build([b.op("log_softmax", x, dim=0)])
    assert _fd_rel_err(prog, torch.randn(4, 3, generator=gen, dtype=F64)) < 1e-8
    # cast float64 -> float32 -> float64 times tanh: the cotangent passes through the casts unchanged
    b = ir.Builder("cast")
    x = b.input("x", F64, (6,))
    c = b.op("cast", b.op("tanh", b.op("cast", x, dtype=torch.float32)), dtype=F64)
    gp = T.grad(b.build([c]), ["x"], seed="ones")
    assert gp.op_counts()["cast"] == 4  # two forward casts, two backward casts back to the input dtype
    xv = torch.linspace(-2, 2, 6, dtype=F64)
    want = 1 - torch.tanh(xv.float().double()) ** 2
    assert rel_err(ir.run(gp, xv)[1], want) < 1e-7  # tanh of the float32 rounded input


def test_global_grad_rule_table_is_unchanged(monkeypatch):
    """The training rules are installed only during grad calls made by truestar.train, so the
    autodiff tests that enumerate the global table keep seeing exactly their own ops."""
    b = ir.Builder("lsm")
    x = b.input("x", F64, (2, 3))
    prog = b.build([b.op("sum", b.op("log_softmax", x, dim=1))])
    # since the op coverage work (ir/grad_rules_extra.py) cast and log_softmax are global rules with
    # the same formulas: a grad call by truestar.train then installs nothing and changes nothing
    assert "cast" in grad_rules.GRAD_RULES and "log_softmax" in grad_rules.GRAD_RULES
    before = dict(grad_rules.GRAD_RULES)
    T.grad(prog, ["x"])
    assert grad_rules.GRAD_RULES == before
    # without the global rules, the training rules are still only present during the call
    monkeypatch.delitem(grad_rules.GRAD_RULES, "cast")
    monkeypatch.delitem(grad_rules.GRAD_RULES, "log_softmax")
    with pytest.raises(ir.UnsupportedOpError, match="log_softmax"):
        autodiff.grad(prog, ["x"])  # loud, names the op
    T.grad(prog, ["x"])
    assert "cast" not in grad_rules.GRAD_RULES and "log_softmax" not in grad_rules.GRAD_RULES


def test_cast_rounds_and_overflows_like_float16():
    b = ir.Builder("c")
    x = b.input("x", torch.float32, (4,))
    prog = b.build([b.op("cast", x, dtype=torch.float16)])
    xv = torch.tensor([1.0 + 2 ** -12, 70000.0, -1e6, 3.0], dtype=torch.float32)
    (out,) = ir.run(prog, xv)
    assert out.dtype == F64
    assert torch.equal(out, xv.half().double())
    assert out[1].item() == math.inf and out[2].item() == -math.inf


# ----------------------------------------------------------------------------- 2. optimizers vs torch


@pytest.mark.parametrize("opt_name", sorted(OPTIMIZERS))
@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_training_matches_torch_step_for_step(model_name, opt_name):
    worst, losses, step = compare_training(model_name, opt_name)
    RESULTS[f"{model_name} {opt_name}"] = worst
    assert worst < TRAIN_TOL, f"{model_name} {opt_name}: worst relative error {worst:.2e}"


def test_adam_l2_amsgrad_and_clipping_match_torch():
    worst, _, _ = compare_training(
        "cnn", None, clip=0.5,
        ir_opt=lambda: T.Adam(2e-3, weight_decay=0.01, amsgrad=True),
        torch_opt=lambda ps: torch.optim.Adam(ps, lr=2e-3, weight_decay=0.01, amsgrad=True))
    RESULTS["cnn adam amsgrad L2 clip 0.5"] = worst
    assert worst < TRAIN_TOL
    worst, _, _ = compare_training(
        "mlp", None, clip=0.1, ir_opt=lambda: T.SGD(0.1),
        torch_opt=lambda ps: torch.optim.SGD(ps, lr=0.1))
    RESULTS["mlp plain sgd clip 0.1"] = worst
    assert worst < TRAIN_TOL


def test_key_bias_gradient_is_zero():
    """Justifies key_bias_slices: our gradient of the attention key bias is rounding noise."""
    model, prog = imported("transformer")
    names = [n for n, _ in model.named_parameters()]
    gp, _, nf = loss_and_grad(prog, T.CausalLM(), names)
    g = torch.Generator().manual_seed(1)
    x, y = _lm_data(g, 4)
    outs = ir.run(gp, x, y, *[prog.param_data[n] for n in names])
    gb = outs[nf + names.index("layer.self_attn.in_proj_bias")]
    assert gb[16:32].abs().max().item() < 1e-15 * gb[:16].abs().max().item() * 1e2
    assert gb[:16].abs().max().item() > 1e-4


def test_update_program_is_standalone_ir():
    """The optimizer alone is a pure IR program: (lr, params, grads, state) -> (params, state)."""
    types = {"w": ir.TensorType(F64, (3, 2)), "b": ir.TensorType(F64, (2,))}
    opt = T.AdamW(1e-2, weight_decay=0.1)
    prog = T.update_program(opt, types)
    names = [v.name for v in prog.inputs]
    assert names[:5] == ["lr", "w", "b", "w.grad", "b.grad"]
    assert names[5:] == ["step", "w.exp_avg", "w.exp_avg_sq", "b.exp_avg", "b.exp_avg_sq"]
    assert set(prog.op_counts()) <= {"add", "sub", "mul", "div", "sqrt", "exp", "expm1", "neg"}
    gen = torch.Generator().manual_seed(6)
    w, bb = torch.randn(3, 2, generator=gen, dtype=F64), torch.randn(2, generator=gen, dtype=F64)
    tw, tb = w.clone().requires_grad_(True), bb.clone().requires_grad_(True)
    topt = torch.optim.AdamW([tw, tb], lr=1e-2, weight_decay=0.1)
    state = [torch.tensor(0.0, dtype=F64)] + [torch.zeros(3, 2, dtype=F64)] * 2 + [torch.zeros(2, dtype=F64)] * 2
    for k in range(5):
        gw, gb = torch.randn(3, 2, generator=gen, dtype=F64), torch.randn(2, generator=gen, dtype=F64)
        out = ir.run(prog, torch.tensor(1e-2, dtype=F64), w, bb, gw, gb, *state)
        w, bb, state = out[0], out[1], out[2:]
        tw.grad, tb.grad = gw.clone(), gb.clone()
        topt.step()
        assert rel_err(w, tw.detach()) < 1e-14 and rel_err(bb, tb.detach()) < 1e-14
    assert state[0].item() == 5.0


# ----------------------------------------------------------------------------- 3. step and trainer


class CountingExecutor:
    def __init__(self):
        self.prepared, self.calls = 0, 0

    def prepare(self, program):
        self.prepared += 1

        def run(inputs):
            self.calls += 1
            return ir.run(program, *inputs)
        return run


def test_one_program_one_call_per_step():
    model, prog = imported("mlp")
    step = T.build_train_step(prog, T.MSE(), T.AdamW(1e-3), model, clip_grad_norm=1.0)
    p = step.program
    names = [v.name for v in p.inputs]
    assert names[:3] == step.batch_inputs + ["lr"] and step.batch_inputs[1] == "target"
    assert names[3:3 + len(step.param_names)] == step.param_names
    assert names[3 + len(step.param_names):] == step.state_names
    assert len(p.outputs) == 3 + len(step.param_names) + len(step.state_names)
    counts = p.op_counts()
    assert counts["linear"] == 3 and counts["sqrt"] >= len(step.param_names)  # forward and Adam in one program
    ex = CountingExecutor()
    tr = T.Trainer(step, executor=ex)
    g = torch.Generator().manual_seed(1)
    for n in (4, 1, 7):  # one program for every batch size
        tr.train_step(*_mlp_data(g, n))
    assert ex.prepared == 1 and ex.calls == 3
    serialize.loads(serialize.dumps(p))  # the step program is an ordinary artifact


def test_warmup_cosine_matches_torch_schedulers():
    sched = T.WarmupCosine(0.1, warmup_steps=5, total_steps=30, min_lr=1e-3, warmup_start_factor=0.1)
    w = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    ref = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=5),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25, eta_min=1e-3)], milestones=[5])
    for k in range(30):
        assert abs(sched(k) - opt.param_groups[0]["lr"]) < 1e-12, k
        opt.step()
        ref.step()
    assert sched(30) == 1e-3


def test_trainer_with_schedule_and_logging_matches_torch():
    model, prog = imported("mlp")
    sched = T.WarmupCosine(1e-2, warmup_steps=3, total_steps=12, warmup_start_factor=0.25)
    log = io.StringIO()
    tr = T.Trainer(T.build_train_step(prog, T.MSE(), T.AdamW(1e-2), model), lr=sched, log_every=4, stream=log)
    ref = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    lam = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: sched(k) / 1e-2)
    g = torch.Generator().manual_seed(2)
    for k in range(12):
        x, y = _mlp_data(g, 8)
        r = tr.train_step(x, y)
        assert r.lr == pytest.approx(opt.param_groups[0]["lr"], rel=1e-15)
        opt.zero_grad()
        F.mse_loss(ref(x), y).backward()
        opt.step()
        lam.step()
    for n, p in ref.named_parameters():
        assert rel_err(tr.state[n], p.detach()) < TRAIN_TOL
    lines = log.getvalue().splitlines()
    assert len(lines) == 3 and lines[0].startswith("step 3 lr ") and "loss" in lines[0]


class SqrtNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc(x)
        return torch.sqrt(h)  # MARK:sqrt_line  NaN for negative h


def test_nonfinite_loss_names_the_first_op_and_skips_the_step():
    torch.manual_seed(0)
    model = SqrtNet().double()
    prog = ir.import_module(copy.deepcopy(model), (torch.rand(2, 4, dtype=F64),), dynamic_shapes=({0: BATCH},))
    with torch.no_grad():  # a bias that keeps h positive for inputs in [0, 1)
        prog.param_data["fc.bias"].fill_(10.0)
    tr = T.Trainer(T.build_train_step(prog, T.MSE(), T.SGD(1e-3), ["fc.weight", "fc.bias"]))
    gen = torch.Generator().manual_seed(0)
    tr.train_step(torch.rand(3, 4, generator=gen, dtype=F64), torch.rand(3, 4, generator=gen, dtype=F64))
    before = {k: v.clone() for k, v in tr.state.items()}
    bad = torch.rand(3, 4, generator=gen, dtype=F64)
    bad[1] = -1e3  # drives h negative in row 1
    with pytest.raises(T.NonFiniteError) as info:
        tr.train_step(bad, torch.rand(3, 4, generator=gen, dtype=F64))
    rep = info.value.report
    assert rep["kind"] == "node" and rep["op"] == "sqrt", rep
    assert rep["nan"] > 0 and rep["step"] == 1
    line = next(i for i, l in enumerate(open(__file__, encoding="utf-8"), 1) if "MARK:sqrt" + "_line" in l)
    assert f"test_train.py:{line}" in rep["source"], rep["source"]
    assert "the step was not applied" in str(info.value)
    assert tr.step_count == 1 and all(torch.equal(before[k], tr.state[k]) for k in before)
    # a NaN already in the batch is reported as the input, not blamed on an op
    nan_in = torch.rand(3, 4, generator=gen, dtype=F64)
    nan_in[0, 0] = math.nan
    with pytest.raises(T.NonFiniteError) as info:
        tr.train_step(nan_in, torch.rand(3, 4, generator=gen, dtype=F64))
    assert info.value.report["kind"] == "input" and info.value.report["value"] == prog.inputs[0].name


def _lm_trainer(opt=None):
    model, prog = imported("transformer")
    sched = T.WarmupCosine(1e-3, warmup_steps=2, total_steps=8)
    return T.Trainer(T.build_train_step(prog, T.CausalLM(), opt or T.AdamW(1e-3), model, clip_grad_norm=1.0),
                     lr=sched)


def test_checkpoint_resume_is_bit_identical(tmp_path):
    g = torch.Generator().manual_seed(7)
    batches = [_lm_data(g, 3) for _ in range(6)]
    straight = _lm_trainer()
    for b in batches:
        straight.train_step(*b)
    first = _lm_trainer()
    for b in batches[:3]:
        first.train_step(*b)
    path = str(tmp_path / "ckpt.tsx")
    first.save(path)
    assert open(path, "rb").read() == first.checkpoint_bytes()  # same state, same bytes
    resumed = _lm_trainer()
    resumed.load(path)
    assert resumed.step_count == 3
    for b in batches[3:]:
        resumed.train_step(*b)
    for n in straight.state:
        assert torch.equal(straight.state[n], resumed.state[n]), n
    assert straight.history == resumed.history
    assert straight.checkpoint_bytes() == resumed.checkpoint_bytes()
    # a checkpoint of another step program is refused, naming why
    other = _lm_trainer(T.AdamW(1e-3, weight_decay=0.5))
    with pytest.raises(ir.IRError, match="different training step program"):
        other.load(path)


# ----------------------------------------------------------------------------- 4. mixed precision


class AmpMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))

    def forward(self, x):
        return self.body(x)


@lru_cache(maxsize=None)
def amp_model():
    torch.manual_seed(0)
    model = AmpMLP().eval()  # float32
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(4, 16),), dynamic_shapes=({0: BATCH},))
    return model, prog


def _amp_data(g):
    return torch.randn(32, 16, generator=g) * 3, torch.randn(32, 4, generator=g) * 10


def test_autocast_policy_structure():
    model, prog = amp_model()
    ap = T.autocast(prog, T.AutocastPolicy(torch.float16))
    by_op = {}
    for n in ap.nodes:
        by_op.setdefault(n.op, []).append(n)
    assert len(by_op["cast"]) == 7  # the input, three weights and three biases; activations stay float16
    for n in by_op["linear"]:
        assert all(x.type.dtype == torch.float16 for x in n.inputs) and n.output.type.dtype == torch.float16
    assert {n.output.type.dtype for n in by_op["tanh"] + by_op["relu"]} == {torch.float16}
    weights = [p for p in prog.params if p.name.endswith("weight")]
    assert all(p.type.dtype == torch.float32 for p in weights)  # master weights stay float32
    with pytest.raises(ir.IRError, match="float16 or bfloat16"):
        T.AutocastPolicy(torch.float32)


def _ulps(a, b, dtype):
    return ((a - b).abs() / (b.abs().clamp(min=torch.finfo(dtype).tiny) * torch.finfo(dtype).eps)).max().item()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_amp_gradients_match_torch_autocast(dtype):
    """Loss and the scaled gradients of every master weight against torch.autocast("cpu") plus
    autograd, from the same weights and batch, at 10 points along a torch AdamW trajectory.
    Gradients of half precision ops land on the half precision grid in both, so entries can be
    compared bit for bit."""
    model, prog = amp_model()
    names = [n for n, _ in model.named_parameters()]
    gp, _, nf = loss_and_grad(prog, T.MSE(), names, T.AutocastPolicy(dtype), scaled=True)
    run = T.EmulatedReferenceExecutor().prepare(gp)
    eps = torch.finfo(dtype).eps
    m = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    same = total = 0
    worst_loss = worst_fp32 = worst_rel = 0.0
    for k in range(10):
        x, y = _amp_data(g)
        with torch.autocast("cpu", dtype=dtype):
            loss = F.mse_loss(m(x).float(), y)
        want = torch.autograd.grad(loss * 1024.0, list(m.parameters()), retain_graph=True)
        outs = run([x, y] + [p.detach() for p in m.parameters()] + [torch.tensor(1024.0)])
        worst_loss = max(worst_loss, abs(outs[1].item() - loss.item()) / loss.item())
        fp32 = F.mse_loss(m(x), y).item()
        worst_fp32 = max(worst_fp32, abs(fp32 - loss.item()) / loss.item())
        for ours, theirs in zip(outs[nf:], want):
            same += int((ours.float() == theirs).sum())
            total += theirs.numel()
            worst_rel = max(worst_rel, rel_err(ours, theirs))
        opt.zero_grad()
        loss.backward()
        opt.step()
    frac = same / total
    RESULTS[f"amp {dtype} gradient entries bit identical"] = frac
    RESULTS[f"amp {dtype} gradient rel"] = worst_rel
    RESULTS[f"amp {dtype} loss rel"] = worst_loss
    RESULTS[f"amp {dtype} plain float32 loss vs autocast rel"] = worst_fp32
    # The forward rounds where torch rounds: the loss agrees to float32 rounding (a few ulps of the
    # float32 mean), while plain float32 differs from autocast by about 1e-5 or more.
    assert worst_loss < 1e-6 < worst_fp32
    # The rest differ where torch's half precision matmul accumulates in float32 in its own order and
    # we accumulate exactly: a result near a rounding boundary rounds the other way (seen on 1 of 2048
    # entries of a 256 long dot product), and that one ulp spreads through the backward pass. Never by
    # more than one low precision epsilon of the tensor's largest entry. Plain float32 gradients share
    # no entry with autocast's.
    assert frac > 0.98, frac
    assert worst_rel < eps, worst_rel


def _load_torch_state(tr, model, opt, scaler, steps_taken, opt_name):
    for n, p in model.named_parameters():
        tr.state[n] = p.detach().clone()
        st = opt.state.get(p, {})
        if opt_name == "sgd":
            buf = st.get("momentum_buffer")
            tr.state[f"{n}.momentum_buffer"] = buf.clone() if buf is not None else torch.zeros_like(p)
        else:
            tr.state[f"{n}.exp_avg"] = st["exp_avg"].clone() if st else torch.zeros_like(p)
            tr.state[f"{n}.exp_avg_sq"] = st["exp_avg_sq"].clone() if st else torch.zeros_like(p)
    tr.state["step"] = torch.tensor(float(steps_taken), dtype=F64)
    if scaler.is_enabled():
        tr.state["amp.scale"] = torch.tensor(scaler.get_scale(), dtype=torch.float32)
        gt = scaler._growth_tracker
        tr.state["amp.growth_tracker"] = torch.tensor(float(gt.item()) if gt is not None else 0.0, dtype=F64)


AMP_OPTS = {
    "sgd": (lambda: T.SGD(0.05, momentum=0.9), lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9)),
    "adamw": (lambda: T.AdamW(1e-3, weight_decay=0.01), lambda ps: torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.01)),
}


@pytest.mark.parametrize("opt_name", sorted(AMP_OPTS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_amp_training_matches_torch_amp_with_grad_scaler(dtype, opt_name):
    """20 steps against torch.autocast + torch.amp.GradScaler (growth_interval 3, so the scale grows
    into overflow again and again). Two runs:

    * local: every step starts from torch's exact params, optimizer state and scaler state, so the
      skip decision, the new scale and the update are compared without drift;
    * free: both train on their own for 20 steps; the skip pattern and the scale sequence must be
      identical."""
    model, prog = amp_model()
    mk_ir, mk_torch = AMP_OPTS[opt_name]
    use_scaler = dtype == torch.float16
    scaler_cfg = T.LossScaler(growth_interval=3) if use_scaler else None
    step = T.build_train_step(prog, T.MSE(), mk_ir(), model, amp=T.AutocastPolicy(dtype), scaler=scaler_cfg)
    executor = T.EmulatedReferenceExecutor()

    def torch_step(m, opt, scaler, x, y, grads=None):
        """One torch.amp step. With ``grads``, torch's own scaled gradients are computed (for the
        overflow decision) and then replaced by ``grads`` before GradScaler unscales and steps."""
        with torch.autocast("cpu", dtype=dtype):
            loss = F.mse_loss(m(x).float(), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        overflow = any(not torch.isfinite(p.grad).all() for p in m.parameters())
        if grads is not None:
            for p, gr in zip(m.parameters(), grads):
                p.grad = gr.float().clone()
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        return loss.item(), use_scaler and scaler.get_scale() < before, overflow

    # local: from torch's exact state every step
    names = step.param_names
    gp, _, nf = loss_and_grad(prog, T.MSE(), names, T.AutocastPolicy(dtype), scaled=use_scaler)
    grads_of = executor.prepare(gp)
    tr = T.Trainer(step, executor=executor)
    m = copy.deepcopy(model).train()
    opt = mk_torch(m.parameters())
    scaler = torch.amp.GradScaler("cpu", growth_interval=3, enabled=use_scaler)
    g = torch.Generator().manual_seed(1)
    taken, worst_update, worst_param, skips = 0, 0.0, 0.0, []
    for k in range(20):
        x, y = _amp_data(g)
        _load_torch_state(tr, m, opt, scaler, taken, opt_name)
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        extra = [torch.tensor(scaler.get_scale())] if use_scaler else []
        ours_g = grads_of([x, y] + [before[n] for n in names] + extra)[nf:]
        ours_overflow = any(not torch.isfinite(t).all() for t in ours_g)
        r = tr.train_step(x, y)
        _, skipped, overflow = torch_step(m, opt, scaler, x, y, grads=ours_g)
        # skip exactly when torch's own gradients overflow
        assert ours_overflow == overflow == r.skipped == skipped, (k, ours_overflow, overflow, r.skipped, skipped)
        if use_scaler:
            assert tr.state["amp.scale"].item() == scaler.get_scale()
            assert tr.state["amp.growth_tracker"].item() == scaler._growth_tracker.item()
        taken += not skipped
        skips.append(skipped)
        for n, p in m.named_parameters():
            if skipped:
                assert torch.equal(tr.state[n], before[n]), n
            else:
                worst_update = max(worst_update, rel_err(tr.state[n] - before[n], p.detach() - before[n]))
                worst_param = max(worst_param, rel_err(tr.state[n], p.detach()))
    if use_scaler:
        assert 3 <= sum(skips) <= 10, skips  # the scale really overflowed, more than once
    RESULTS[f"amp {dtype} {opt_name} local update rel"] = worst_update
    RESULTS[f"amp {dtype} {opt_name} skipped steps of 20"] = sum(skips)
    RESULTS[f"amp {dtype} {opt_name} local param rel"] = worst_param
    # Same gradients on both sides (ours, handed to torch's GradScaler), so this compares the
    # unscale, the float32 master weight update, the skip and the scale arithmetic. Both sides are
    # float32 with different rounding points (one per IR op here, one per fused kernel in torch).
    # The new params agree to a few float32 ulps (bar 1e-6, 8 ulps). The update (new minus old
    # param) loses the leading digits of a param about 300 times larger than an Adam step, so one
    # float32 ulp of the param is about 1.5e-5 of the largest update: bar 1e-4. The gradients
    # themselves are compared in test_amp_gradients_match_torch_autocast.
    assert worst_param < 1e-6, f"worst param error {worst_param:.2e}"
    assert worst_update < 1e-4, f"worst update error {worst_update:.2e}"

    # free
    tr = T.Trainer(step, executor=executor)
    m = copy.deepcopy(model).train()
    opt = mk_torch(m.parameters())
    scaler = torch.amp.GradScaler("cpu", growth_interval=3, enabled=use_scaler)
    g = torch.Generator().manual_seed(1)
    worst_param = 0.0
    for k in range(20):
        x, y = _amp_data(g)
        r = tr.train_step(x, y)
        loss, skipped, _ = torch_step(m, opt, scaler, x, y)
        assert r.skipped == skipped, f"free run step {k}: we skip {r.skipped}, torch skips {skipped}"
        if use_scaler:
            assert r.scale == scaler.get_scale()
        worst_param = max(worst_param, max(rel_err(tr.state[n], p.detach()) for n, p in m.named_parameters()))
    RESULTS[f"amp {dtype} {opt_name} free run param rel"] = worst_param
    if opt_name == "adamw":
        # AdamW at lr 1e-3 is not chaotic here: the trajectories stay together (measured 6e-5 for
        # float16, 8e-6 for bfloat16). SGD with momentum 0.9 at lr 0.05 amplifies the rare one ulp
        # gradient differences (as two torch runs with different summation orders would), so only
        # its skip pattern and scales are compared over the free run.
        assert worst_param < 1e-3


def test_amp_scaler_arithmetic_on_forced_overflow():
    """An Inf gradient skips the step: params and state stay, the scale halves, the counter resets."""
    model, prog = amp_model()
    step = T.build_train_step(prog, T.MSE(), T.SGD(0.05, momentum=0.9), model, amp=T.AutocastPolicy(torch.float16),
                              scaler=T.LossScaler(init_scale=2.0 ** 24, growth_interval=2))
    tr = T.Trainer(step, executor=T.EmulatedReferenceExecutor())
    g = torch.Generator().manual_seed(1)
    start = {k: v.clone() for k, v in tr.state.items()}
    r = tr.train_step(*_amp_data(g))
    assert r.skipped and r.scale == 2.0 ** 23
    for k in step.carried:
        if k != "amp.scale":
            assert torch.equal(tr.state[k], start[k]), k
    scales = [r.scale]
    while len(scales) < 16:
        scales.append(tr.train_step(*_amp_data(g)).scale)
    assert scales[-1] < 2.0 ** 16 and any(b > a for a, b in zip(scales, scales[1:]))  # it also grew back


# ----------------------------------------------------------------------------- 5. data parallel


def _run_dp(tmp_path, step, batches, steps_for_rank, timeout=30.0, deadline=180.0):
    import torch.multiprocessing as mp

    import train_workers

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=train_workers.dp_worker,
                         args=(r, 2, str(tmp_path), step, batches, steps_for_rank, timeout)) for r in range(2)]
    for p in procs:
        p.start()
    t0 = time.time()
    try:
        while any(p.is_alive() for p in procs) and time.time() - t0 < deadline:
            time.sleep(0.1)
    finally:
        for p in procs:  # never leave a worker behind
            if p.is_alive():
                p.kill()
            p.join(timeout=10)
    return [p.exitcode for p in procs]


def train_workers_path():
    import train_workers

    return os.path.abspath(train_workers.__file__)


dp_ok = pytest.mark.skipif(not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
                           reason="needs torch.distributed with gloo")


@dp_ok
def test_data_parallel_two_ranks_match_one_rank_on_the_whole_batch(tmp_path):
    model, prog = imported("mlp")
    opt = T.AdamW(3e-3, weight_decay=0.05)
    dp = T.build_train_step(prog, T.MSE(), opt, model, clip_grad_norm=0.5, data_parallel=True)
    assert dp.program.op_counts()["all_reduce"] == 1  # one bucket, one collective per step
    single = T.build_train_step(prog, T.MSE(), opt, model, clip_grad_norm=0.5)
    g = torch.Generator().manual_seed(11)
    batches = [_mlp_data(g, 8) for _ in range(6)]
    codes = _run_dp(tmp_path, dp, batches, [6, 6])
    assert codes == [0, 0], codes
    one = T.Trainer(single)
    for b in batches:
        one.train_step(*b)
    worst = 0.0
    for r in range(2):
        res = torch.load(str(tmp_path / f"result_rank{r}.pt"), weights_only=False)
        for h1, h2 in zip(res["history"], one.history):
            worst = max(worst, abs(h1["loss"] - h2["loss"]) / h2["loss"], abs(h1["grad_norm"] - h2["grad_norm"]) / h2["grad_norm"])
        for n in single.param_names + single.state_names:
            worst = max(worst, rel_err(res["state"][n], one.state[n]))
    RESULTS["data parallel 2 ranks vs 1"] = worst
    # mean of two half batch means against one mean: the same sum split in two, about 1e-16
    assert worst < 1e-13, worst
    a = torch.load(str(tmp_path / "result_rank0.pt"), weights_only=False)["state"]
    b = torch.load(str(tmp_path / "result_rank1.pt"), weights_only=False)["state"]
    assert all(torch.equal(a[n], b[n]) for n in a)  # ranks stay bit identical to each other


@dp_ok
def test_data_parallel_rank_that_runs_out_of_data_is_diagnosed(tmp_path):
    from truestar.diag.collectives import diagnose

    model, prog = imported("mlp")
    dp = T.build_train_step(prog, T.MSE(), T.SGD(0.05), model, data_parallel=True)
    g = torch.Generator().manual_seed(12)
    batches = [_mlp_data(g, 4) for _ in range(3)]
    codes = _run_dp(tmp_path, dp, batches, [3, 2], timeout=2.0, deadline=90.0)
    # The waiting rank ends by the watchdog, or by gloo's own error when it sees the finished peer
    # go away first (Linux reports a closed peer at once; Windows blocks until the watchdog), as
    # in tests/test_diag_collectives.py. Either way it must fail and the diagnosis must be right.
    assert codes[0] not in (0, None) and codes[1] == 0, codes
    d = diagnose(str(tmp_path / "rec"))
    assert d.kind == "behind" and d.rank == 1 and d.seq == 2, d.to_text()
    text = d.to_text()
    assert "rank 1 finished after 2 collectives" in text and "all_reduce" in text, text
    # the recorded site is the user's line that ran the step, not the kernel inside truestar
    line = next(i for i, l in enumerate(open(train_workers_path(), encoding="utf-8"), 1) if "MARK:" + "dp_step" in l)
    assert os.path.basename(d.file) == "train_workers.py" and d.line == line, (d.file, d.line)


# ----------------------------------------------------------------------------- 6. broken rules fail


def test_broken_bias_correction_is_caught(monkeypatch):
    """Adam without bias correction (b^t replaced by 0, so c1 = c2 = 1) must fail the comparison."""
    # since 2026-11-24 the corrections are -expm1(t ln b) (T.Adam._one_minus_pow); b^t = 0 means c = 1
    monkeypatch.setattr(T.Adam, "_one_minus_pow", staticmethod(
        lambda e, base, t, name: e.op("add", e.op("mul", t, e.const(0.0, t.type.dtype)), e.const(1.0, t.type.dtype),
                                      name=name)))
    worst, _, _ = compare_training("mlp", "adamw", steps=3)
    assert worst > 1e-3, worst


def test_broken_nesterov_is_caught(monkeypatch):
    """SGD that ignores the Nesterov flag must fail the nesterov comparison."""
    orig = T.SGD.emit

    def no_nesterov(self, *a):
        saved, self.nesterov = self.nesterov, False
        try:
            return orig(self, *a)
        finally:
            self.nesterov = saved
    monkeypatch.setattr(T.SGD, "emit", no_nesterov)
    worst, _, _ = compare_training("cnn", "sgd_nesterov", steps=3)
    assert worst > 1e-3, worst


def test_zz_print_results():
    for k, v in sorted(RESULTS.items()):
        print(f"RESULT {k}: {v:.3g}")
