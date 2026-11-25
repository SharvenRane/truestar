"""Compiled mixed precision training steps (docs/TRAINING.md, Mixed precision, compiled; TS1480).

* an autocast step compiles with ``mixed=True``: the policy's casts, contractions and activations
  run in bfloat16 or float16 (float32 accumulation), the master weights, optimizer state, loss and
  loss scale stay float32 islands, nothing falls back;
* the end to end check holds each float32 output to a bar derived for the declared mixed precision
  (4x an independent evaluation rounded to the declared dtypes, at least the float32 bar), which is
  far tighter than the 16 bit working bar: master weights rounded to 16 bits and a missing unscale
  are refused (TS1480), the declared rounding is not (no false alarm over many verified steps);
* the compiled step follows torch.autocast with torch.amp.GradScaler, including skipped steps,
  and is bitwise identical run to run.

The CPU rows run the numba kernels; the same tests run on CUDA when a GPU is present.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

from truestar import ir
from truestar import train as T
from truestar.compiler.runtime import ProgramMismatchError
from truestar.train.compiled import CompiledTrainStep

pytest.importorskip("numba")
CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
LOW = [torch.bfloat16, torch.float16]


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 10)).eval()


def _step(model, opt, dtype, scaler="auto"):
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 32),), dynamic_shapes=({0: Dim("B", max=128)},))
    return T.build_train_step(prog, T.CrossEntropy(), opt, params=model, amp=T.AutocastPolicy(dtype), scaler=scaler)


def _batch(n=16, seed=1, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 32, generator=g) * scale, torch.randint(0, 10, (n,), generator=g)


def _short(dtype):
    return str(dtype).replace("torch.", "")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", LOW, ids=_short)
def test_mixed_step_keeps_float32_masters_and_runs_low_precision_contractions(device, dtype):
    cs = CompiledTrainStep(_step(_mlp(), T.SGD(1e-2, momentum=0.9), dtype), device)
    assert cs.cp.opts.mixed and cs.cp.opts.dtype == _short(dtype)
    kinds = [s.kind for _, s in cs.cp.plan.all_steps()]
    assert "fallback" not in kinds, cs.cp.report
    for n, t in cs.carried_state().items():
        assert t.dtype == torch.float32, (n, t.dtype)   # params, momentum, step, loss scale: islands
    # the linear kernels read the policy's low precision casts of the master weights
    casts = [n for _, s in cs.cp.plan.all_steps() if s.kind == "kernel" for n in s.nodes if n.op == "cast"]
    assert any(n.output.type.dtype == dtype and n.inputs[0].name == "0.weight" for n in casts)
    x, y = _batch()
    loss, norm, nonfinite = cs(x.to(device), y.to(device), lr=1e-2)
    assert loss.dtype == torch.float32 and not bool(nonfinite)
    bars = [b for b in cs.cp.mixed_bounds if b]
    # the master weights are held far tighter than the 16 bit working bar the check used before
    new = [b for b in bars if b["output"].endswith(("_new", "_kept")) and "momentum" not in b["output"]]
    assert new and all(b["bar"] < cs.cp.end_to_end_tolerance() / 10 for b in new), new


def _inputs(step, x, y, device, lr=1e-2):
    b = {step.batch_inputs[0]: x.to(device), step.batch_inputs[1]: y.to(device)}
    out = []
    for v in step.program.inputs:
        if v.name in b:
            out.append(b[v.name])
        elif v.name == "lr":
            out.append(torch.tensor(lr, dtype=torch.float32, device=device))
        else:
            t = step.initial[v.name]
            out.append(t.to(device, torch.float32 if t.dtype.is_floating_point else t.dtype))
    return out


def _corrupt(cp, fn):
    orig = cp.plan.run

    def run(*a, **kw):
        return fn(orig(*a, **kw))
    cp.plan.run = run
    return orig


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", LOW, ids=_short)
def test_master_weights_rounded_to_16_bits_are_refused(device, dtype):
    """A fault after every region check passed: the updated master weight comes back rounded to
    the working dtype (a wrong cast, or an island stored in 16 bits). Its error is below the 16 bit
    working bar the check used before (so it was accepted), and far over the declared precision bar."""
    from truestar.compiler import compile_program
    step = _step(_mlp(), T.SGD(1e-2, momentum=0.9), dtype)
    cp = compile_program(step.program, device, _short(dtype), mixed=True)
    k = [v.name for v in step.program.outputs].index("2.weight_new" if dtype == torch.bfloat16 else "2.weight_kept")

    def wrong_cast(outs):
        outs[k] = outs[k].to(dtype).to(outs[k].dtype)
        return outs
    orig = _corrupt(cp, wrong_cast)
    x, y = _batch()
    with pytest.raises(ProgramMismatchError, match=r"TS1480[\s\S]*2\.weight") as e:
        cp(_inputs(step, x, y, device))
    assert e.value.error < cp.end_to_end_tolerance()     # the working precision bar passed it
    cp.plan.run = orig
    cp(_inputs(step, x, y, device))
    assert cp.verified


@pytest.mark.parametrize("device", DEVICES)
def test_missing_unscale_is_refused(device):
    """float16 with loss scaling: the momentum (the first step's gradient) and the update come back
    as if the gradients were not divided by the scale. Refused, whatever the scale (here 2, so the
    outputs are only 2x off)."""
    from truestar.compiler import compile_program
    step = _step(_mlp(), T.SGD(1e-2, momentum=0.9), torch.float16, scaler=T.LossScaler(init_scale=2.0))
    cp = compile_program(step.program, device, "float16", mixed=True)
    names = [v.name for v in step.program.outputs]
    w, m = names.index("2.weight_kept"), names.index("2.weight.momentum_buffer_kept")
    w0 = step.initial["2.weight"].to(device)

    def no_unscale(outs):
        outs[m] = outs[m] * 2.0
        outs[w] = w0 - (w0 - outs[w]) * 2.0
        return outs
    orig = _corrupt(cp, no_unscale)
    x, y = _batch()
    with pytest.raises(ProgramMismatchError, match="TS1480"):
        cp(_inputs(step, x, y, device))
    cp.plan.run = orig
    cp(_inputs(step, x, y, device))
    assert cp.verified


def _torch_amp_run(model, dtype, steps, device, scaler_kw, data):
    """torch.autocast + torch.amp.GradScaler, SGD momentum, float32 master weights."""
    m = copy.deepcopy(model).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)
    sc = torch.amp.GradScaler(device, enabled=dtype == torch.float16, **scaler_kw)
    losses, skips = [], []
    for x, y in data[:steps]:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=dtype):
            out = m(x)
        loss = F.cross_entropy(out.float(), y)
        opt.zero_grad()
        sc.scale(loss).backward()
        before = sc.get_scale() if dtype == torch.float16 else None
        sc.step(opt)
        sc.update()
        skips.append(dtype == torch.float16 and sc.get_scale() < before)
        losses.append(loss.item())
    return m, losses, skips, sc.get_scale() if dtype == torch.float16 else None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", LOW, ids=_short)
def test_compiled_mixed_step_follows_torch_amp_and_is_bitwise_run_to_run(device, dtype):
    """20 steps against torch.autocast + GradScaler (growth_interval 3 and an input scaled into
    float16 overflow, so the scale grows and steps are skipped), every call verified."""
    model = _mlp()
    scaler_kw = {"growth_interval": 3, "init_scale": 2.0 ** 15}
    big = 3e3 if dtype == torch.float16 else 1.0   # one batch overflows float16 (a skipped step)
    data = [_batch(seed=k, scale=(big if k == 7 else 1.0)) for k in range(20)]
    sc = T.LossScaler(growth_interval=3, init_scale=2.0 ** 15) if dtype == torch.float16 else None
    runs = []
    for _ in range(2):
        cs = CompiledTrainStep(_step(model, T.SGD(1e-2, momentum=0.9), dtype, scaler=sc), device, verify="all")
        losses, skips = [], []
        for x, y in data:
            loss, _, nonfinite = cs(x.to(device), y.to(device), lr=1e-2)
            losses.append(loss.item())
            skips.append(bool(nonfinite))
        runs.append((cs, losses, skips))
    (a, la, sa), (b, lb, sb) = runs
    for n, t in a.carried_state().items():
        assert torch.equal(t.view(torch.int32) if t.dtype == torch.float32 else t,
                           b.carried_state()[n].view(torch.int32) if t.dtype == torch.float32 else b.carried_state()[n]), n
    assert la == lb
    tm, tl, ts, tscale = _torch_amp_run(model, dtype, 20, device, scaler_kw, data)
    if dtype == torch.float16:
        assert sa == ts and any(sa), (sa, ts)
        assert a.carried_state()["amp.scale"].item() == tscale
    for k, (u, v) in enumerate(zip(la, tl)):
        assert abs(u - v) <= 3e-2 * max(1.0, abs(v)), (k, u, v)
    got = a.params()
    for n, p in tm.named_parameters():
        p = p.detach().float()
        err = ((got[n].float() - p).abs().max() / p.abs().max()).item()
        assert err < 2e-2, (n, err)


# ----------------------------------------------------------------------------- stochastic rounding


def _sr_program(n):
    from truestar.ir.core import Program
    e = T.Editor(Program("sr", [], [], {}, [], [], {}), "sr")
    v = e.input("v", torch.float32, (n,))
    key = e.input("key", torch.int64, ())
    return e.build([T.amp.stochastic_round(e, v, key, "test.sr")])


def _bf16_neighbours(v):
    """lo (|v| rounded toward zero to bfloat16, with v's sign) and the next bfloat16 away from zero."""
    b = v.to(torch.bfloat16).float()
    lo = torch.where(b.abs() > v.abs(), torch.nextafter(b.to(torch.bfloat16), torch.zeros_like(b).to(torch.bfloat16)).float(), b)
    hi = torch.where(lo == v, lo, torch.nextafter(lo.to(torch.bfloat16),
                                                    (lo * 2 + torch.sign(v)).to(torch.bfloat16)).float())
    return lo, hi


def test_stochastic_round_is_a_neighbour_unbiased_and_bitwise_across_backends():
    from truestar.compiler import compile_program
    g = torch.Generator().manual_seed(0)
    v = torch.cat([torch.randn(2000, generator=g) * 10.0 ** torch.randint(-6, 6, (2000,), generator=g),
                   torch.tensor([0.0, -0.0, 1.0, -2.0, 0.5, 2.0 ** -126, 3.0, 1.0 + 2.0 ** -9, -(1.0 + 2.0 ** -9)])])
    prog = _sr_program(v.numel())
    cp = compile_program(prog, "cpu", "bfloat16", mixed=True)
    lo, hi = _bf16_neighbours(v)
    ups = torch.zeros_like(v)
    for k in range(64):
        key = torch.tensor(k * 7919 + 1, dtype=torch.int64)
        (r,) = ir.run(prog, v, key)
        (c,) = cp([v, key])
        assert c.dtype == torch.bfloat16
        assert torch.equal(c.float().view(torch.int32), r.float().view(torch.int32)), k   # bitwise = reference
        rf = r.float()
        assert bool(((rf == lo) | (rf == hi)).all())
        ups += (rf == hi) & (hi != lo)
    exact = lo == v
    assert bool((ups[exact] == 0).all())                     # bfloat16 numbers stay as they are
    # unbiased: P(up) = (|v| - |lo|) / (|hi| - |lo|); 64 draws per value, so check the mean over values
    gap = (hi - lo).abs()
    p = torch.where(exact, torch.zeros_like(v), (v - lo).abs() / torch.where(exact, torch.ones_like(gap), gap))
    m = ~exact
    assert abs((ups[m] / 64 - p[m]).mean().item()) < 0.01
    # 1 + 2^-9 sits a quarter of the way to 1 + 2^-7: up about 25% of the time
    i = v.numel() - 2
    assert 0.05 < ups[i].item() / 64 < 0.5


@pytest.mark.parametrize("device", DEVICES)
def test_bfloat16_weights_with_keyed_stochastic_rounding(device):
    """weights='bfloat16_sr': bfloat16 params, no float32 copy, float32 momentum; every call
    verified; two runs bitwise equal; the loss follows the float32 master run."""
    model = _mlp()
    data = [_batch(seed=k) for k in range(12)]

    def run(weights):
        prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 32),), dynamic_shapes=({0: Dim("B", max=128)},))
        st = T.build_train_step(prog, T.CrossEntropy(), T.SGD(5e-2, momentum=0.9), params=model,
                                amp=T.AutocastPolicy(torch.bfloat16), weights=weights, sr_seed=7)
        cs = CompiledTrainStep(st, device, verify="all")
        losses = [cs(x.to(device), y.to(device), lr=5e-2)[0].item() for x, y in data]
        return cs, losses

    a, la = run("bfloat16_sr")
    b, lb = run("bfloat16_sr")
    m, lm = run("float32")
    assert a.carried_state()["0.weight"].dtype == torch.bfloat16
    assert a.carried_state()["0.weight_f32.momentum_buffer"].dtype == torch.float32
    for n, t in a.carried_state().items():
        u = b.carried_state()[n]
        assert torch.equal(t.view(torch.int16) if t.dtype == torch.bfloat16 else t, 
                           u.view(torch.int16) if u.dtype == torch.bfloat16 else u), n
    assert la == lb
    assert la[-1] < la[0]
    for k, (u, v) in enumerate(zip(la, lm)):
        assert abs(u - v) <= 3e-2 * max(1.0, abs(v)), (k, u, v)
