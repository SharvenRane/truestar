"""Compiled training steps (docs/TRAINING.md, Compiled; ledger row P37).

* the ops a training step adds (erf from the exact gelu gradient, log_softmax from cross entropy,
  expm1 from Adam's bias correction) are Truestar kernels, not fallbacks;
* the relu gradient reads the relu output, so the relu input is not kept for the backward pass;
* a compiled AdamW step passes its checks (it failed before: 1 - exp(t ln b2) cancels in float32)
  and matches torch.optim.AdamW; a wiring fault after the checks is still refused end to end;
* ``CompiledTrainStep``: donated params and state keep their buffers, a CUDA graph replay is bit
  for bit equal to plain calls, a batch of another shape is refused (TS781).
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

from truestar import ir
from truestar import train as T
from truestar.compiler import compile_program
from truestar.compiler.runtime import ProgramMismatchError
from truestar.train.compiled import CompiledTrainStep, GraphInputError

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 10)).eval()


def _step(model, opt):
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 32),), dynamic_shapes=({0: Dim("B", max=128)},))
    return T.build_train_step(prog, T.CrossEntropy(), opt, params=model)


def _batch(n=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 32, generator=g), torch.randint(0, 10, (n,), generator=g)


def test_expm1_op_reference_gradient_and_forward_rule():
    x = torch.linspace(-3, 2, 11, dtype=torch.float64)

    class M(nn.Module):
        def forward(self, x):
            return torch.expm1(x) * x

    prog = ir.import_module(M(), (x,))
    assert any(n.op == "expm1" for n in prog.nodes)
    (y,) = ir.run(prog, x)
    assert torch.allclose(y, torch.expm1(x) * x, rtol=1e-15, atol=0)
    gp = ir.grad(prog, [prog.inputs[0].name])
    xt = x.clone().requires_grad_()
    (torch.expm1(xt) * xt).backward(torch.ones_like(x))
    outs = ir.run(gp, x, torch.ones_like(x))
    assert torch.allclose(outs[-1], xt.grad, rtol=1e-14, atol=1e-15)
    # tiny arguments: expm1 keeps its digits where exp(x) - 1 cancels
    (t,) = ir.run(ir.import_module(M(), (torch.tensor([1e-10], dtype=torch.float64),)),
                  torch.tensor([1e-10], dtype=torch.float64))
    want = math.expm1(1e-10) * 1e-10
    assert abs(t.item() - want) / want < 1e-15


def test_relu_gradient_reads_the_output_not_the_input():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    prog = ir.import_module(model, (torch.randn(2, 8),))
    lp, _, _ = T.step.loss_and_grad(prog, T.MSE(), [n for n, _ in model.named_parameters()])
    relu = [n for n in lp.nodes if n.op == "relu"]
    assert len(relu) == 1
    relu_in, relu_out = relu[0].inputs[0], relu[0].output
    readers_of_input = [n.op for n in lp.nodes if any(i is relu_in for i in n.inputs)]
    assert readers_of_input == ["relu"], readers_of_input
    assert any(n.op == "gt" and n.inputs[0] is relu_out for n in lp.nodes)


@gpu
@pytest.mark.parametrize("opt", ["sgd", "adamw"])
def test_compiled_step_has_no_fallbacks_and_matches_torch(opt):
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _mlp()
    topt = T.SGD(1e-2, momentum=0.9) if opt == "sgd" else T.AdamW(1e-2, weight_decay=0.01)
    step = _step(model, topt)
    cs = CompiledTrainStep(step, "cuda", "float32")
    kinds = [s.kind for _, s in cs.cp.plan.all_steps()]
    assert "fallback" not in kinds, cs.cp.report
    ops = {n.op for _, s in cs.cp.plan.all_steps() if s.kind == "kernel" for n in s.nodes}
    assert {"erf", "log_softmax"} <= ops
    if opt == "adamw":
        assert "expm1" in ops
    ref = copy.deepcopy(model).double()
    o = (torch.optim.SGD(ref.parameters(), lr=1e-2, momentum=0.9) if opt == "sgd"
         else torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.01))
    for k in range(4):
        x, y = _batch(seed=k)
        loss = cs(x.cuda(), y.cuda(), lr=1e-2)[0]
        rl = F.cross_entropy(ref(x.double()), y)
        o.zero_grad()
        rl.backward()
        o.step()
        assert abs(loss.item() - rl.item()) < 1e-4 * max(1.0, abs(rl.item()))
    got = cs.params()
    for n, p in ref.named_parameters():
        err = ((got[n].double().cpu() - p.detach()).abs().max() / p.detach().abs().max()).item()
        assert err < 1e-3, (n, err)


@gpu
def test_end_to_end_still_refuses_a_wiring_fault_on_an_ill_conditioned_step():
    """With accept_ill_conditioned=True, an output corrupted by 1% after every region check passed (a
    wiring fault) is still refused: the allowance is relative to an independent float32 evaluation,
    which the corruption is far beyond."""
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _mlp()
    step = _step(model, T.AdamW(1e-3))
    cp = compile_program(step.program, "cuda", "float32", accept_ill_conditioned=True)
    x, y = _batch()
    names = [v.name for v in step.program.inputs]

    def inputs():
        b = {step.batch_inputs[0]: x.cuda(), step.batch_inputs[1]: y.cuda()}
        return [b[n] if n in b else torch.tensor(1e-3, device="cuda") if n == "lr"
                else step.initial[n].to("cuda", torch.float32 if step.initial[n].dtype.is_floating_point
                                        else step.initial[n].dtype) for n in names]

    orig = cp.plan.run
    k = [v.name for v in step.program.outputs].index("0.weight_new")

    def corrupt(*a, **kw):
        outs = orig(*a, **kw)
        outs[k] = outs[k] * 1.01
        return outs

    cp.plan.run = corrupt
    with pytest.raises(ProgramMismatchError, match="0.weight_new"):
        cp(inputs())
    cp.plan.run = orig
    cp(inputs())       # the same step, wired right, passes (accepted outputs are listed)
    assert cp.verified
    for c in cp.conditioning:
        assert c["error"] <= 4.0 * c["float32_reference_error"]


@gpu
def test_donated_state_keeps_its_buffers_and_graph_replay_is_bitwise():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _mlp()
    plain = CompiledTrainStep(_step(model, T.SGD(1e-2, momentum=0.9)), "cuda", "float32")
    graph = CompiledTrainStep(_step(model, T.SGD(1e-2, momentum=0.9)), "cuda", "float32", graph=True, warmup=2)
    batches = [_batch(seed=k) for k in range(6)]
    ptrs = None
    for k, (x, y) in enumerate(batches):
        a = plain(x.cuda(), y.cuda(), lr=1e-2)
        b = graph(x.cuda(), y.cuda(), lr=1e-2)
        assert torch.equal(a[0], b[0]), k
        if k == 1:
            ptrs = {n: t.data_ptr() for n, t in plain.carried_state().items()}
    # every weight and its momentum is written over its own buffer (donated input first)
    st = plain.carried_state()
    for n in ("0.weight", "0.weight.momentum_buffer", "2.weight.momentum_buffer"):
        assert st[n].data_ptr() == ptrs[n], n
    assert graph._g is not None and "step" in graph.copied_back
    for n, t in plain.carried_state().items():
        assert torch.equal(t, graph.carried_state()[n]), n
    with pytest.raises(GraphInputError, match="TS781"):
        x, y = _batch(n=8)
        graph(x.cuda(), y.cuda(), lr=1e-2)


@gpu
def test_split_weight_gradient_is_deterministic_and_checked():
    """conv2d_weight_grad cut into row chunks (conv_rules.weight_grad_splits) and summed in a fixed
    order: two compiles and two calls give the same bits, and the executor's float64 checks pass."""
    from truestar.compiler import conv_rules
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert conv_rules.weight_grad_splits(64 * 64 * 9, 56, 56) > 1
    assert conv_rules.weight_grad_splits(512 * 512 * 9, 7, 7) == 1
    torch.manual_seed(0)
    conv = nn.Conv2d(16, 16, 3, 1, 1, bias=False)
    x = torch.randn(2, 16, 32, 32)
    prog = ir.import_module(conv, (x,))
    gp = ir.grad(prog, [prog.params[0].name], seed="input")
    assert conv_rules.weight_grad_splits(16 * 16 * 9, 32, 32) > 1
    xs, g = x.cuda(), torch.randn(2, 16, 32, 32, device="cuda")
    a = compile_program(gp, "cuda", "float32")
    b = compile_program(gp, "cuda", "float32", verify="all")
    ra, rb, ra2 = a([xs, g])[-1], b([xs, g])[-1], a([xs, g])[-1]
    assert torch.equal(ra, rb) and torch.equal(ra, ra2)
    assert sum(1 for _, s in a.plan.all_steps() if s.kind == "kernel") >= 3   # conv, partial sums, total
    ref = torch.nn.grad.conv2d_weight(x.double(), conv.weight.shape, g.double().cpu(), 1, 1)
    assert ((ra.double().cpu() - ref).abs().max() / ref.abs().max()).item() < 1e-5


@pytest.mark.parametrize("opt", ["sgd", "adamw"])
def test_param_group_step_equals_the_per_param_step(opt):
    """param_group=True (one flat buffer per carried value, the update one kernel per value) computes
    the same step as one update per param: float64 reference interpreter, 3 steps, every param and
    moment, and the optimizer part collapses to a few nodes."""
    model = _mlp().double()
    mk = (lambda: T.SGD(1e-2, momentum=0.9)) if opt == "sgd" else (lambda: T.AdamW(1e-2, weight_decay=0.01))
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 32, dtype=torch.float64),),
                            dynamic_shapes=({0: Dim("B", max=128)},))
    a = T.build_train_step(prog, T.CrossEntropy(), mk(), params=model, clip_grad_norm=1.0)
    b = T.build_train_step(prog, T.CrossEntropy(), mk(), params=model, clip_grad_norm=1.0, param_group=True)
    assert b.param_names == ["params.flat"] and set(b.layout) == set(a.param_names)
    sa, sb = dict(a.initial), dict(b.initial)
    for k in range(3):
        x, y = _batch(seed=k)
        x = x.double()
        outs = []
        for s, st in ((a, sa), (b, sb)):
            ins = [x if v.name == s.batch_inputs[0] else y if v.name == s.batch_inputs[1]
                   else torch.tensor(1e-2, dtype=torch.float64) if v.name == "lr" else st[v.name]
                   for v in s.program.inputs]
            o = ir.run(s.program, *ins)
            st.update(zip(s.carried, o[3:]))
            outs.append(o)
        assert abs(outs[0][0].item() - outs[1][0].item()) < 1e-14
        assert abs(outs[0][1].item() - outs[1][1].item()) < 1e-12 * max(1.0, outs[0][1].item())
    ub = b.unflatten(sb)
    for n, t in sa.items():
        assert torch.allclose(ub[n], t, rtol=1e-12, atol=1e-15), n
    upd = lambda s: sum(1 for n in s.program.nodes if "update" in (n.source or "") or "step count" in (n.source or ""))  # noqa
    assert upd(b) * 3 < upd(a)     # 6 params: the per param part shrinks 6x, the bias corrections stay


@gpu
def test_ill_conditioned_adamw_first_step_passes_by_its_derived_bound_and_ts782_still_works():
    """The benchmark MLP's first AdamW step is ill conditioned (each update divides by |g|): eager
    float32 PyTorch is 6.8e-4 and 2.6e-3 from float64 on two weights, over the end to end bar. By
    default (end_to_end="auto": CompiledTrainStep donates its state) the outputs are judged by the
    bound derived from the step and it runs (docs/TOLERANCE.md). With the fixed bar alone
    (end_to_end="bar") and accept_ill_conditioned=True the older allowance still runs, each accepted
    output listed and named in a TS782 warning."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.GELU(),
                          nn.Linear(2048, 1000)).eval()
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 1024),), dynamic_shapes=({0: Dim("B", max=128)},))
    step = T.build_train_step(prog, T.CrossEntropy(), T.AdamW(1e-3, weight_decay=0.01), params=model)
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 1024, generator=g), torch.randint(0, 1000, (64,), generator=g)
    cs = CompiledTrainStep(step, "cuda", "float32")
    cs(x.cuda(), y.cuda(), lr=1e-3)
    assert cs.cp.verified and not cs.cp.conditioning
    from truestar.compiler.runtime import IllConditionedWarning
    cp = compile_program(step.program, "cuda", "float32", accept_ill_conditioned=True, end_to_end="bar",
                         graphs="off")
    b = {step.batch_inputs[0]: x.cuda(), step.batch_inputs[1]: y.cuda()}
    ins = [b[v.name] if v.name in b else torch.tensor(1e-3, device="cuda") if v.name == "lr"
           else step.initial[v.name].to("cuda", torch.float32 if step.initial[v.name].dtype.is_floating_point
                                         else step.initial[v.name].dtype) for v in step.program.inputs]
    with pytest.warns(IllConditionedWarning, match="TS782") as rec:
        cp(ins)
    assert cp.verified and cp.conditioning
    text = " ".join(str(w.message) for w in rec)
    for c in cp.conditioning:
        assert c["error"] > c["tolerance"] and c["error"] <= 4.0 * c["float32_reference_error"]
        assert f"%{c['output']}" in text
