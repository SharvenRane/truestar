"""Static memory planner, buffer reuse, donation and verified recomputation (docs/MEMORY.md).

GPU tests compare the plan (made before any launch) with torch's allocator statistics of a real
call: ``requested_bytes`` must equal the planned bytes exactly, ``max_memory_allocated`` may exceed
the 512 byte rounded plan only by cached blocks the allocator does not split (under 1 MiB each)."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

from truestar import codes, ir
from truestar.compiler import memory as M

CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
gpu = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def test_parse_and_format_bytes():
    assert M.parse_bytes("6 GiB") == 6 * 2 ** 30
    assert M.parse_bytes("512MB") == 512 * 10 ** 6
    assert M.parse_bytes("1.5k") == 1500
    assert M.parse_bytes(4096) == 4096
    with pytest.raises(ValueError):
        M.parse_bytes("lots")
    assert M.fmt_bytes(3 * 2 ** 20) == "3.00 MiB"
    assert M.block_bytes(1, "cuda") == 512 and M.block_bytes(513, "cuda") == 1024 and M.block_bytes(0, "cuda") == 0


def test_codes_registered():
    for c in ("TS420", "TS421"):
        assert c in codes.CODES
    assert M.MemoryBudgetError.code == "TS420"


# ----------------------------------------------------------------------------- models


class Attention(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Residual(nn.Module):
    """An elementwise kernel reading a dead buffer: a candidate for writing over it."""

    def __init__(self, d=256):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)

    def forward(self, x):
        h = self.b(torch.relu(self.a(x)))
        g = h.t().reshape(-1)              # a copy (the transpose is not contiguous)
        return torch.tanh(g * 2.0 + 1.0)   # elementwise in the dead copy: written over it


def _models():
    torch.manual_seed(0)
    return {
        "mlp": (nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10),
                              nn.Softmax(-1)), (64,)),
        "transformer": (nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True), (16, 64)),
        "attention": (Attention(), (32, 64)),
        "residual": (Residual(), (256,)),
    }


def _prog(name, max_b=64):
    model, shape = _models()[name]
    model = model.eval()
    return model, shape, ir.import_module(copy.deepcopy(model), (torch.randn(2, *shape),),
                                          dynamic_shapes=({0: Dim("B", max=max_b)},))


def _measure(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    req0 = torch.cuda.memory_stats().get("requested_bytes.all.current", 0)
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    reqp = torch.cuda.memory_stats().get("requested_bytes.all.peak", 0) - req0
    return peak, reqp, out


# ----------------------------------------------------------------------------- plan vs measured


@gpu
@pytest.mark.parametrize("name", ["mlp", "transformer", "attention", "residual"])
@pytest.mark.parametrize("reuse", [True, False])
def test_planned_peak_equals_measured(name, reuse):
    from truestar.compiler import compile_program
    _, shape, prog = _prog(name)
    cp = compile_program(prog, "cuda", "float32", memory_reuse=reuse, graphs="off")  # measures the ordinary call's own allocations
    for b in (1, 7, 33):
        x = torch.randn(b, *shape, device="cuda")
        planned = cp.memory_plan({"B": b})              # before any launch at this batch size
        requested = cp.memory_plan({"B": b}, rounding=False)
        assert planned.notes == []
        cp([x])
        cp([x])
        peak, req, _ = _measure(lambda: cp([x]))
        assert req == requested.peak_bytes, (name, b, req, requested.peak_bytes)
        assert planned.peak_bytes <= peak <= planned.peak_bytes + 2 ** 20, (name, b, peak, planned.peak_bytes)
        if not reuse:
            assert planned.peak_bytes == planned.hold_peak_bytes


@gpu
@pytest.mark.parametrize("name", ["mlp", "transformer", "attention", "residual"])
def test_reuse_gives_the_same_bits(name):
    from truestar.compiler import compile_program
    _, shape, prog = _prog(name)
    reuse = compile_program(prog, "cuda", "float32")
    hold = compile_program(prog, "cuda", "float32", memory_reuse=False)
    x = torch.randn(9, *shape, device="cuda")
    for _ in range(2):                    # the checked first call and a plain call
        a, b = reuse([x]), hold([x])
        assert all(torch.equal(p, q) for p, q in zip(a, b))
    mp = reuse.memory_plan({"B": 9})
    assert mp.peak_bytes <= mp.hold_peak_bytes
    if name in ("mlp", "transformer", "attention"):
        assert mp.peak_bytes < mp.hold_peak_bytes


@gpu
def test_in_place_reuse_is_planned_and_happens(monkeypatch):
    from truestar.compiler import compile_program, runtime as R
    _, shape, prog = _prog("residual")
    cp = compile_program(prog, "cuda", "float32", graphs="off")  # measures the ordinary call's own allocations
    mp = cp.memory_plan({"B": 64})
    assert mp.inplace_steps >= 1
    assert mp.peak_bytes <= mp.no_inplace_peak_bytes
    x = torch.randn(64, *shape, device="cuda")
    ref = compile_program(prog, "cuda", "float32", memory_reuse=False)([x])
    cp([x])
    calls = []
    real = R.KernelStep.run_into
    monkeypatch.setattr(R.KernelStep, "run_into", lambda self, env, rt, j: (calls.append(self.name),
                                                                             real(self, env, rt, j))[1])
    _, req, out = _measure(lambda: cp([x]))
    assert len(calls) == mp.inplace_steps
    assert req == cp.memory_plan({"B": 64}, rounding=False).peak_bytes
    assert all(torch.equal(a, b) for a, b in zip(out, ref))


def test_elementwise_inputs_rejects_reductions_and_shifted_reads():
    from truestar import kernel as K
    x = K.Buffer("x", (8, 16), "float32")
    i, j = K.axis("i", 8), K.axis("j", 16)
    good = K.kernel("ew", [K.output("o", (i, j), K.tanh(x[i, j]) * 2.0)])
    assert M.elementwise_inputs(_Fake(good)) == [0]
    jj = K.axis("jj", 16)
    red = K.kernel("red", [K.output("o", (i, j), x[i, j] - K.reduce_max(jj, x[i, jj]))])
    assert M.elementwise_inputs(_Fake(red)) == []
    xt = K.Buffer("xt", (16, 16), "float32")
    a, b = K.axis("a", 16), K.axis("b", 16)
    transposed = K.kernel("t", [K.output("o", (a, b), xt[b, a])])
    assert M.elementwise_inputs(_Fake(transposed)) == []


class _Fake:
    def __init__(self, k):
        self.kernel = k


# ----------------------------------------------------------------------------- budgets


@gpu
def test_budget_error_names_the_line_and_the_largest_batch_that_fits():
    from truestar.compiler import compile_program
    _, shape, prog = _prog("transformer", max_b=256)
    probe = compile_program(prog, "cuda", "float32")
    want = probe.memory_plan({"B": 100}).total_bytes
    with pytest.raises(M.MemoryBudgetError) as e:
        compile_program(prog, "cuda", "float32", memory_budget=want)     # planned at B=256, the declared max
    err = e.value
    assert "[TS420]" in str(err) and "Nothing was launched" in str(err)
    assert "peak at " in str(err) and ".py:" in str(err)                   # the source line at the peak
    fit = err.fits["B"]
    assert probe.memory_plan({"B": fit}).total_bytes <= want < probe.memory_plan({"B": fit + 1}).total_bytes
    assert fit >= 100
    cp = compile_program(prog, "cuda", "float32", memory_budget=want, memory_dims={"B": fit})
    cp([torch.randn(fit, *shape, device="cuda")])


@gpu
def test_unbounded_symbol_is_checked_before_the_first_launch():
    from truestar.compiler import compile_program
    b = ir.Builder("unbounded")
    x = b.input("x", torch.float32, ("B", 64))          # B has no upper bound
    y = b.op("tanh", b.op("linear", x, b.param("w", torch.randn(256, 64)), None))
    prog = b.build([y])
    shape = (64,)
    cp = compile_program(prog, "cuda", "float32", memory_budget="1 MiB")
    cp([torch.randn(4, *shape, device="cuda")])
    with pytest.raises(M.MemoryBudgetError):
        cp([torch.randn(4096, *shape, device="cuda")])
    launched = {s.ck.name: list(s.ck._fast) for _, s in cp.plan.all_steps() if s.kind == "kernel"}
    assert all(all(sig[0][0][0] == 4 for sig in sigs) for sigs in launched.values()), launched


def test_recompute_needs_a_budget():
    if not CUDA:
        pytest.skip("needs CUDA")
    from truestar.compiler import compile_program
    _, _, prog = _prog("mlp")
    with pytest.raises(ValueError):
        compile_program(prog, "cuda", "float32", recompute=True)


# ----------------------------------------------------------------------------- donation


def _adamw_step():
    from truestar import train as T
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 512), nn.Tanh(), nn.Linear(512, 64)).eval()
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 128),), dynamic_shapes=({0: Dim("B", max=64)},))
    return T.build_train_step(prog, T.MSE(), T.SGD(1e-3, momentum=0.9, weight_decay=0.1), params=model)


def _step_inputs(step, b=32):
    from truestar.ir.core import eval_dim, symbol
    g = torch.Generator().manual_seed(1)
    vals = []
    for v in step.program.inputs:
        if v.name in step.initial:
            t = step.initial[v.name].clone()
        elif v.name == "lr":
            t = torch.tensor(1e-3, dtype=torch.float64)
        else:
            t = torch.randn(tuple(eval_dim(d, {symbol("B"): b}) for d in v.type.shape), generator=g)
        # each in its working form: float32, lr and step included (float64 by convention only, no island)
        vals.append(t.to("cuda", torch.float32) if t.dtype.is_floating_point else t.cuda())
    return vals


@gpu
def test_donated_params_and_momentum_are_updated_in_place_with_the_same_bits():
    from truestar.compiler import compile_program
    step = _adamw_step()
    cp = compile_program(step.program, "cuda", "float32")
    base = _step_inputs(step)
    ref = cp([t.clone() for t in base])                                  # checked call, nothing donated
    kept = cp.memory_plan({"B": 32})
    don = cp.memory_plan({"B": 32}, donate=True)
    assert don.peak_bytes < kept.peak_bytes
    for _ in range(2):
        ins = [t.clone() for t in base]
        torch.cuda.synchronize()
        _, req, out = _measure(lambda: cp(ins, donate=True))
        assert ins == []                                                  # the executor took them
        assert all(torch.equal(a, b) for a, b in zip(out, ref))
    assert req == cp.memory_plan({"B": 32}, donate=True, rounding=False).peak_bytes


@gpu
def test_donation_never_writes_over_a_tensor_the_caller_still_holds():
    from truestar.compiler import compile_program
    step = _adamw_step()
    cp = compile_program(step.program, "cuda", "float32")
    base = _step_inputs(step)
    cp([t.clone() for t in base])
    held = [t.clone() for t in base]
    snapshot = [t.clone() for t in held]
    cp(list(held), donate=True)            # a copy of the list: the caller keeps every tensor
    assert all(torch.equal(a, b) for a, b in zip(held, snapshot))


# ----------------------------------------------------------------------------- recompute


class TinyLM(nn.Module):
    def __init__(self, d=64, layers=3, vocab=96):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True)
                                     for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _lm():
    from truestar import train as T
    torch.manual_seed(0)
    model = TinyLM().eval()
    names = [n for n, _ in model.named_parameters()]
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 32, 64),), dynamic_shapes=({0: Dim("B", max=16)},))
    gp, _, _ = T.step.loss_and_grad(prog, T.CausalLM(), names)
    x = torch.randn(8, 32, 64, device="cuda")
    y = torch.randint(0, 96, (8, 32), device="cuda")
    params = [dict(model.named_parameters())[n].detach().cuda() for n in names]
    return gp, [x, y] + params


@gpu
def test_recompute_fits_the_budget_with_bitwise_equal_gradients():
    from truestar.compiler import compile_program
    gp, inputs = _lm()
    base = compile_program(gp, "cuda", "float32")
    ref = base(inputs)
    mp0 = base.memory_plan({"B": 8})
    # 55% of the peak: since the relu gradient reads its output (2026-11-24) the peak without
    # recompute is 16% lower (3.47 against 4.12 MiB), and 55% of it (1.91 MiB) is a tighter absolute
    # budget than the 50% of the old peak (2.06 MiB) this test asked for before
    budget = mp0.resident_bytes + int(mp0.peak_bytes * 0.55)
    cp = compile_program(gp, "cuda", "float32", memory_budget=budget, memory_dims={"B": 8}, recompute=True)
    assert cp.recompute_report and cp.report.counts()["recompute"] >= len(cp.recompute_report) // 2
    mp = cp.memory_plan({"B": 8})
    assert mp.total_bytes <= budget
    out = cp(inputs)                       # check run: every recomputed value compared bit for bit
    assert all(torch.equal(a, b) for a, b in zip(out, ref))
    cp(inputs)
    _, req, out = _measure(lambda: cp(inputs))
    assert req == cp.memory_plan({"B": 8}, rounding=False).peak_bytes
    assert all(torch.equal(a, b) for a, b in zip(out, ref))
    assert "recomputed" in mp.to_text()


@gpu
def test_recompute_check_catches_a_corrupted_recompute():
    """The class of pytorch#201158: storage reuse changes a value that is recomputed or cached.
    Here one input of a recomputed kernel is changed just before the recompute."""
    from truestar.compiler import compile_program, runtime as R
    gp, inputs = _lm()
    mp0 = compile_program(gp, "cuda", "float32").memory_plan({"B": 8})
    cp = compile_program(gp, "cuda", "float32", memory_budget=mp0.resident_bytes + int(mp0.peak_bytes * 0.55),
                         memory_dims={"B": 8}, recompute=True)
    victim = next(s for s in cp.plan.steps if isinstance(s, R.RecomputeStep) and s.producer.kind == "kernel")
    run = victim.producer.run

    def corrupted(env, rt):
        s = victim.producer.in_slots[0]
        env[s] = env[s] * 1.5 + 0.25        # what a stale or reused buffer would hand the recompute
        run(env, rt)
    victim.run = corrupted
    with pytest.raises(R.RecomputeMismatchError) as e:
        cp(inputs)
    assert "[TS421]" in str(e.value) and e.value.stage == "recompute"


@gpu
def test_every_planned_in_place_write_happens(monkeypatch):
    """The plan and the runtime agree on every in place write: on a plain call of a training program
    the number of in place launches equals the plan's count. (An expand view, which does not cover
    its storage, was once planned in place; the guard refused it and the plan was off.)"""
    from truestar.compiler import compile_program, runtime as R
    gp, inputs = _lm()
    cp = compile_program(gp, "cuda", "float32", graphs="off")  # measures the ordinary call's own allocations
    planned = cp.memory_plan({"B": 8}).inplace_steps
    assert planned > 0
    cp(inputs)
    calls = []
    real = R.KernelStep.run_into
    monkeypatch.setattr(R.KernelStep, "run_into", lambda self, env, rt, j: (calls.append(self.name),
                                                                         real(self, env, rt, j))[1])
    cp(inputs)
    assert len(calls) == planned


@gpu
def test_each_tf32_variant_has_its_own_plan_and_budget_check():
    """A program that follows PyTorch's TF32 setting builds a variant per setting; each variant is
    planned, budget checked and measured on its own."""
    from truestar.compiler import compile_program
    _, shape, prog = _prog("mlp")
    probe = compile_program(prog, "cuda", "float32", allow_tf32=False)
    budget = probe.memory_plan({"B": 64}).total_bytes + 2 ** 20
    cp = compile_program(prog, "cuda", "float32", memory_budget=budget, memory_dims={"B": 64})
    x = torch.randn(33, *shape, device="cuda")
    prev = torch.backends.cuda.matmul.allow_tf32
    plans = {}
    try:
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            cp([x])
            assert bool(cp.opts.allow_tf32) == tf32
            assert cp.memory_plan_at_compile is not None          # this variant's own budget check
            plan = cp.plan
            cp([x])
            _, req, _ = _measure(lambda: cp([x]))
            assert req == cp.memory_plan({"B": 33}, rounding=False).peak_bytes
            plans[tf32] = plan
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    assert plans[False] is not plans[True]


def test_plan_without_data_covers_index_steps():
    # The dry run feeds index steps (gather, from cross entropy) a stand in runtime; it once lacked
    # the `node` the index range check records, so memory_plan raised AttributeError on any
    # program with a gather.
    import truestar
    from truestar import Tensor
    from truestar import nn as tnn
    from truestar.compiler import compile_program

    net = tnn.Linear(16, 4)

    @truestar.native
    def loss(p: net.spec, x: Tensor["B", 16], y: Tensor["B", "i64"]) -> Tensor[()]:
        return tnn.cross_entropy(net(p, x), y)

    prog = loss.program
    assert any(n.op == "gather" for n in prog.nodes)
    plan = compile_program(prog, "cpu", "float32").memory_plan({"B": 8})
    assert plan.peak_bytes > 0
