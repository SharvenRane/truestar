"""The end to end check of training steps: derived bounds accept correct float32 steps, and still refuse
wrong ones (docs/TOLERANCE.md, End to end; docs/TRAINING.md)."""

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from truestar import ir, train as T
from truestar.compiler import compile_program
from truestar.compiler import runtime as R

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(64, 256), nn.Linear(256, 32)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# ----------------------------------------------------------------------------- injected bugs


class AdamWWrongDecaySign(T.AdamW):
    """Decoupled weight decay with the wrong sign: p (1 + lr wd) instead of p (1 - lr wd)."""

    def emit(self, e, params, grads, state, lr):
        wd = self.weight_decay
        self.weight_decay = -wd
        try:
            return super().emit(e, params, grads, state, lr)
        finally:
            self.weight_decay = wd


class AdamWNoBiasCorrection1(T.AdamW):
    """Step size lr instead of lr / (1 - b1^t)."""

    def _one_minus_pow(self, e, base, t, name):
        if name == "adam_bias_correction1":
            return e.const(1.0, t.type.dtype)
        return super()._one_minus_pow(e, base, t, name)


class AdamWNoBiasCorrection2(T.AdamW):
    """sqrt(v) instead of sqrt(v) / sqrt(1 - b2^t)."""

    def _one_minus_pow(self, e, base, t, name):
        if name == "adam_bias_correction2":
            return e.const(1.0, t.type.dtype)
        return super()._one_minus_pow(e, base, t, name)


def _over_n_minus_1(Base, n):
    class Off(Base):
        """Every gradient times n / (n - 1): a mean taken over n - 1 instead of n."""

        def emit(self, e, params, grads, state, lr):
            grads = [e.op("mul", g, e.const(n / (n - 1), g.type.dtype)) for g in grads]
            return super().emit(e, params, grads, state, lr)
    return Off


def _grads_through(Base, dtype):
    class Cast(Base):
        """Every gradient rounded through ``dtype`` and back: a wrong cast in the step."""

        def emit(self, e, params, grads, state, lr):
            grads = [e.op("cast", e.op("cast", g, dtype=dtype), dtype=g.type.dtype) for g in grads]
            return super().emit(e, params, grads, state, lr)
    return Cast


# ----------------------------------------------------------------------------- the check


def _step(opt, batch=64):
    torch.manual_seed(0)
    m = MLP().eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 64, generator=g)
    y = torch.randint(0, 32, (batch,), generator=g)
    prog = ir.import_module(copy.deepcopy(m), (x,), name="mlp_train")
    st = T.build_train_step(prog, T.CrossEntropy(), opt, params=m)
    lst = []
    for i, n in enumerate(v.name for v in st.program.inputs):
        if i == 0:
            lst.append(x.cuda())
        elif i == 1:
            lst.append(y.cuda())
        elif n == "lr":
            lst.append(torch.tensor(opt.lr, dtype=torch.float32, device="cuda"))
        else:
            t = st.initial[n]
            lst.append(t.to("cuda", torch.float32) if t.is_floating_point() else t.cuda())
    return st, lst


def strict_verdict(cp, inputs, outs):
    """end_to_end="derived": every float output by its derived bound alone."""
    derived = cp._derived_verdict(inputs, outs) or {}
    return {v.name: bool(derived[i].ok) for i, v in enumerate(cp.program.outputs) if i in derived}


def verdict(cp, inputs, outs):
    """The end to end rule of CompiledProgram._verified_run for these outputs: an output passes within
    the fixed bar, or over it within the bound derived from the program. Returns {output: passed}."""
    refs = R.iref.run(cp.program, *[t.detach() for t in inputs], device=torch.device("cuda"))
    tol = cp.end_to_end_tolerance()
    derived = None
    res = {}
    for i, (v, o, r) in enumerate(zip(cp.program.outputs, outs, refs)):
        if not v.type.dtype.is_floating_point:
            res[v.name] = bool(torch.equal(o.cpu(), r.cpu()))
            continue
        if R._compare(v.name, o, r, tol).ok:
            res[v.name] = True
            continue
        if derived is None:
            derived = cp._derived_verdict(inputs, outs) or {}
        d = derived.get(i)
        res[v.name] = bool(d is not None and d.ok)
    return res


def run_step(st, lst):
    cp = compile_program(st.program, "cuda", "float32", verify="off", graphs="off")
    return cp, cp([t.clone() for t in lst])


@pytest.mark.parametrize("opt", [T.AdamW(1e-3, weight_decay=0.01), T.SGD(1e-2, momentum=0.9)],
                         ids=["adamw", "sgd"])
def test_a_correct_step_passes(opt):
    st, lst = _step(opt)
    cp, outs = run_step(st, lst)
    assert all(verdict(cp, lst, outs).values())
    assert all(strict_verdict(cp, lst, outs).values())


def test_strict_mode_refuses_the_wrong_decay_sign_through_the_compiled_call(monkeypatch):
    """end_to_end="derived" end to end: a program whose compiled result has the wrong decay sign is
    refused at its first call. The compiled step of the right program is swapped for the wrong one's
    results, as a code generator bug would."""
    from truestar.compiler.runtime import ProgramMismatchError
    good, lst = _step(T.AdamW(1e-3, weight_decay=0.01))
    bad, _ = _step(AdamWWrongDecaySign(1e-3, weight_decay=0.01))
    _, wrong = run_step(bad, lst)
    cp = compile_program(good.program, "cuda", "float32", end_to_end="derived", graphs="off")
    real = cp.plan.run
    monkeypatch.setattr(cp.plan, "run", lambda args, rt, own=False: [w.clone() for w in wrong])
    with pytest.raises(ProgramMismatchError, match="derived from the program"):
        cp([t.clone() for t in lst])
    monkeypatch.setattr(cp.plan, "run", real)
    ok = compile_program(good.program, "cuda", "float32", end_to_end="derived", graphs="off")
    ok([t.clone() for t in lst])                  # the right result passes the strict check


BUGS = [
    (AdamWWrongDecaySign, T.AdamW, "fc1.weight_new"),
    (AdamWNoBiasCorrection1, T.AdamW, "fc1.weight_new"),
    (AdamWNoBiasCorrection2, T.AdamW, "fc1.weight_new"),
    ("n-1", T.AdamW, "fc1.weight_exp_avg"),
    ("n-1", T.SGD, "fc1.weight_momentum"),
    ("cast", T.AdamW, "fc1.weight_exp_avg"),
    ("cast", T.SGD, "fc1.weight_momentum"),
]
BUG_IDS = ["adamw_wrong_decay_sign", "adamw_no_bias_correction1", "adamw_no_bias_correction2",
           "adamw_grad_over_n_minus_1", "sgd_grad_over_n_minus_1", "adamw_grad_cast_float16",
           "sgd_grad_cast_float16"]


def _bug_case(bug, base):
    kw = dict(weight_decay=0.01) if base is T.AdamW else dict(momentum=0.9)
    lr = 1e-3 if base is T.AdamW else 1e-2
    good, lst = _step(base(lr, **kw))
    cls = (_over_n_minus_1(base, 64) if bug == "n-1" else _grads_through(base, torch.float16) if bug == "cast"
           else bug)
    bad, _ = _step(cls(lr, **kw))
    _, wrong = run_step(bad, lst)                  # the wrong step's results, judged as the right step's
    return good, lst, wrong


@pytest.mark.parametrize("bug,base,watch", BUGS, ids=BUG_IDS)
def test_an_injected_bug_is_refused(bug, base, watch):
    good, lst, outs = _bug_case(bug, base)
    cp_good = compile_program(good.program, "cuda", "float32", verify="off", graphs="off")
    s = strict_verdict(cp_good, lst, outs)         # the derived bound alone: refused, every bug
    assert watch in s and not s[watch], s
    v = verdict(cp_good, lst, outs)                # end_to_end="bar": the bar, then the derived bound over it
    if bug is AdamWWrongDecaySign or bug == "cast":
        assert v[watch]     # within the fixed bar (decay sign 2e-5, bar 5.3e-4): only the strict verdict refuses it
    else:
        assert not v[watch], v


@pytest.mark.parametrize("bug,base,watch", BUGS, ids=BUG_IDS)
def test_a_training_loop_call_refuses_every_injected_bug_by_default(bug, base, watch, monkeypatch):
    """The default (end_to_end="auto") through the real first call of a training loop, cp(inputs,
    update=upd): the compiled step's results are swapped for the wrong step's, as a code generator bug
    would make them, and the call is refused; the right results pass the same call."""
    from truestar.compiler.runtime import ProgramMismatchError
    good, lst, wrong = _bug_case(bug, base)
    upd = {3 + k: 3 + k for k in range(len(good.program.outputs) - 3)}
    cp = compile_program(good.program, "cuda", "float32", graphs="off")
    monkeypatch.setattr(cp.plan, "run", lambda args, rt, own=False: [w.clone() for w in wrong])
    with pytest.raises(ProgramMismatchError, match="derived from the program"):
        cp([t.clone() for t in lst], update=upd)
    ok = compile_program(good.program, "cuda", "float32")
    ok([t.clone() for t in lst], update=upd)


def test_a_kernel_with_no_tensor_input_runs():
    """sqrt of a constant is a kernel with no tensor input: its launch takes the current CUDA device."""
    from truestar import kernel as K
    from truestar.kernel import Graph
    g = Graph()
    x = g.input("x", (4,), "float32")
    (_, kern), = g.fuse({"y": x * 0.0 + 1.0})
    ck = K.compile(kern, "cuda")
    assert torch.equal(ck(x=torch.zeros(4, device="cuda")), torch.ones(4, device="cuda"))
    st, lst = _step(AdamWNoBiasCorrection2(1e-3, weight_decay=0.01))
    run_step(st, lst)                     # its sqrt(1.0) step crashed before (no device)


def test_bool_outputs_have_no_rounding_in_the_derivation():
    from truestar import tolerance as TL
    x = torch.randn(8, device="cuda")
    rep = TL.close_report(lambda t: (t * 2.0, torch.isfinite(t).all()), (x,), (x * 2.0, torch.isfinite(x).all()))
    assert rep.ok


def test_an_ordinary_training_loop_with_update_graphs_automatically_with_the_same_bits():
    """cp(inputs, update=upd) every step, the state updated in place: from the second step the call
    replays from a CUDA graph, and every step's loss, params and optimizer state are bitwise those of
    the same loop with graphs off."""
    for opt in (T.SGD(1e-2, momentum=0.9), T.AdamW(1e-3, weight_decay=0.01)):
        st, lst = _step(opt)
        n_out = len(st.program.outputs)
        upd = {3 + k: 3 + k for k in range(n_out - 3)}
        runs = {}
        for g in ("off", "auto"):
            cp = compile_program(st.program, "cuda", "float32", graphs=g)
            ins = [t.clone() for t in lst]
            hist = []
            for _ in range(5):
                outs = cp(ins, update=upd)
                hist.append([o.clone() for o in outs])
            runs[g] = (hist, cp)
        (off, _), (auto, cp_auto) = runs["off"], runs["auto"]
        assert cp_auto.auto_graph_replays >= 3, cp_auto.auto_graph_refusals
        for a, b in zip(off, auto):
            assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_other_state_tensors_fall_back_to_an_ordinary_call():
    st, lst = _step(T.SGD(1e-2, momentum=0.9))
    upd = {3 + k: 3 + k for k in range(len(st.program.outputs) - 3)}
    cp = compile_program(st.program, "cuda", "float32")
    ins = [t.clone() for t in lst]
    for _ in range(3):
        cp(ins, update=upd)
    other = [t.clone() for t in ins]             # the same values, other tensors: TS802 in the graph
    ref = compile_program(st.program, "cuda", "float32", graphs="off")
    want = ref([t.clone() for t in ins], update=upd)
    got = cp(other, update=upd)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert any("TS802" in why for why in cp.auto_graph_refusals.values())


# ----------------------------------------------------------------------------- formulas that cancel


class OneMinusExp(nn.Module):
    """sqrt(1 - b2^t) written by hand: 1 - exp(t ln b2) cancels for a small t (the bias correction of
    Adam before it used -expm1)."""

    def forward(self, t):
        return torch.sqrt(1.0 - torch.exp(t * -0.0010005003335835344))


def test_a_hand_written_one_minus_exp_passes_and_is_not_called_a_code_generator_bug():
    prog = ir.import_module(OneMinusExp(), (torch.ones(1, dtype=torch.float64),))
    cp = compile_program(prog, "cuda", "float32")
    for t in (1.0, 2.0, 7.0, 50.0):
        x = torch.tensor([t], device="cuda")
        got = cp([x])[0].double().cpu()
        want = torch.sqrt(1 - torch.exp(torch.tensor([t], dtype=torch.float64) * -0.0010005003335835344))
        assert ((got - want).abs() / want).item() < 1e-4       # the formula's own float32 error, 2.3e-5 at t = 1


def _kernel_and_derived(sign):
    from truestar import kernel as K
    from truestar.kernel import Graph, reference, verify
    from truestar.kernel.compiler import _derived_runs
    g = Graph()
    t = g.input("t", (1,), "float32")
    (_, kern), = g.fuse({"y": g.sqrt(1.0 - g.exp(t * -0.0010005003335835344))})
    ck = K.compile(kern, "cuda")
    x = torch.ones(1, device="cuda")
    out = ck(t=x)
    if sign:                                       # a wrong kernel: exp(-x) as exp(x), so exp(x) - 1
        out = torch.sqrt(torch.exp(x * 0.0010005003335835344) - 1.0)
    inputs = {b.name: x for b in ck.ck.inputs}
    np_in = {n: v.double().cpu().numpy() for n, v in inputs.items()}
    refs = reference.run_reference(ck.ck, np_in)
    oname = ck.ck.outputs[0].name
    return verify, ck, {oname: out.double().cpu().numpy()}, refs, lambda: _derived_runs(ck.ck, np_in, refs)


def test_the_kernel_check_accepts_the_cancelling_formula_and_refuses_a_wrong_kernel():
    verify, ck, outs, refs, derived = _kernel_and_derived(False)
    verify.check_outputs(ck.ck, "cuda", outs, refs, 2e-5, None, False, derived)       # passes
    verify, ck, outs, refs, derived = _kernel_and_derived(True)
    with pytest.raises(verify.KernelMismatchError):
        verify.check_outputs(ck.ck, "cuda", outs, refs, 2e-5, None, False, derived)
