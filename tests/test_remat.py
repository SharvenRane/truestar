"""Min cut recompute and host offload (truestar.compiler.remat, docs/MEMORY.md, P55)."""

import os
import sys

import pytest
import torch

from truestar import codes
from truestar.compiler import remat as RM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
gpu = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def test_codes_exist():
    for c in ("TS1110", "TS1111", "TS1112", "TS1113"):
        assert c in codes.CODES


def test_min_cut_stores_the_cheap_side():
    """Chain a -> b -> c, backward reads c. Recomputing c from a needs b too.
    With recompute cheap (lambda small) the cut keeps only the small value a and recomputes b and c;
    with recompute dear everything backward reads is stored (c only, b and a are not needed)."""
    nodes = [1, 2, 3]
    store = {1: 10.0, 2: 1000.0, 3: 1000.0}
    work = {1: 5.0, 2: 5.0, 3: 5.0}
    reads = {2: [1], 3: [2]}
    needed = {3}
    assert RM.min_cut(nodes, store, work, set(), reads, needed, 1e-6) == {1, 2, 3} or \
        RM.min_cut(nodes, store, work, set(), reads, needed, 1e-6) == {2, 3}
    cheap = RM.min_cut(nodes, store, work, {1}, reads, needed, 1e-6)
    assert cheap == {2, 3}                 # a fixed (must be stored), b and c replayed from it
    assert RM.min_cut(nodes, store, work, set(), reads, needed, 1e6) == set()


def test_min_cut_respects_values_that_cannot_be_rerun():
    nodes = [1, 2]
    store = {1: 100.0, 2: 100.0}
    work = {1: 1.0, 2: 1.0}
    got = RM.min_cut(nodes, store, work, {2}, {2: [1]}, {2}, 1e-6)
    assert 2 not in got                     # a library step's value is stored, never replayed


def test_prune_keeps_only_what_backward_reaches():
    reads = {3: [2], 2: [1], 5: [4]}
    assert RM._prune({1, 2, 3, 4, 5}, reads, {3}) == {1, 2, 3}


def test_offload_refused_off_cuda():
    from test_memory import TinyLM  # noqa: F401  (same small model as the recompute tests)
    import copy

    from torch.export import Dim

    from truestar import ir, train as T
    from truestar.compiler import compile_program
    torch.manual_seed(0)
    model = TinyLM().eval()
    names = [n for n, _ in model.named_parameters()]
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(2, 32, 64),), dynamic_shapes=({0: Dim("B", max=16)},))
    gp, _, _ = T.step.loss_and_grad(prog, T.CausalLM(), names)
    with pytest.raises(Exception, match="TS1113"):
        compile_program(gp, "cpu", "float32", memory_budget=1, memory_dims={"B": 8}, recompute="offload")


def test_unknown_mode_refused():
    from test_memory import _prog
    from truestar.compiler import compile_program
    _, _, prog = _prog("mlp")
    with pytest.raises(ValueError, match="mincut"):
        compile_program(prog, "cpu", "float32", memory_budget="1 GiB", memory_dims={"B": 4}, recompute="optimal")


def _budget(frac):
    from test_memory import _lm
    from truestar.compiler import compile_program
    gp, inputs = _lm()
    base = compile_program(gp, "cuda", "float32")
    ref = base(inputs)
    mp0 = base.memory_plan({"B": 8})
    return gp, inputs, ref, mp0.resident_bytes + int(mp0.peak_bytes * frac)


@gpu
# The budget is a fraction of the plain program's peak. The compiled training work (relu gradient
# read from its output) lowered that peak from 4.32 MB to 3.64 MB. Since the relu gradient's zero is
# a scalar (2026-11-25) it fuses into the matmul that produces the cotangent: the plain peak is
# 3.53 MB, and the greedy offload pass around the new peak kernel fits 0.7 of it (0.9 and 0.7 fit,
# 0.8, 0.75 and 0.65 to 0.35 do not); mincut+offload fits 0.45, still tighter than mincut's 0.5.
# With the fused attention backward (attention_grad, 2026-11-25) the plain peak is 3.34 MB and the
# greedy offload pass fits 0.8 and above (L4: 0.75 to 0.35 do not); mincut 0.5, mincut+offload 0.45.
@pytest.mark.parametrize("mode,frac", [("mincut", 0.5), ("offload", 0.8), ("mincut+offload", 0.45)])
def test_mode_fits_the_budget_bitwise(mode, frac):
    from test_memory import _measure
    from truestar.compiler import compile_program
    gp, inputs, ref, budget = _budget(frac)
    cp = compile_program(gp, "cuda", "float32", memory_budget=budget, memory_dims={"B": 8}, recompute=mode)
    assert cp.recompute_report
    kinds = cp.report.counts()
    assert kinds["recompute"] + kinds["reload"] >= 1
    assert all(r["code"] in ("TS1110", "TS1111") for r in cp.recompute_report)
    mp = cp.memory_plan({"B": 8})
    assert mp.total_bytes <= budget
    out = cp(inputs)              # check run: every recomputed and reloaded value compared bit for bit
    assert all(torch.equal(a, b) for a, b in zip(out, ref))
    cp(inputs)
    _, req, out = _measure(lambda: cp(inputs))
    assert req == cp.memory_plan({"B": 8}, rounding=False).peak_bytes      # planned equals measured
    assert all(torch.equal(a, b) for a, b in zip(out, ref))


@gpu
def test_mincut_recomputes_no_more_work_than_needed_to_fit():
    """Every value the refined cut recomputes is needed: storing any one recomputed step again
    breaks the budget (that is what the refinement checks), so the report is not empty and fits."""
    from truestar.compiler import compile_program
    gp, inputs, ref, budget = _budget(0.75)
    cp = compile_program(gp, "cuda", "float32", memory_budget=budget, memory_dims={"B": 8}, recompute="mincut")
    s = cp.remat_summary
    assert s["fits"] and s["planned_peak"] <= s["target"] and s["cuts"] >= 2


@gpu
def test_offload_check_catches_a_corrupted_reload():
    from truestar.compiler import compile_program, runtime as R
    gp, inputs, ref, budget = _budget(0.8)
    cp = compile_program(gp, "cuda", "float32", memory_budget=budget, memory_dims={"B": 8}, recompute="offload")
    victim = next(s for s in cp.plan.steps if isinstance(s, R.ReloadStep))
    run = victim.run

    def corrupted(env, rt):
        run(env, rt)
        victim.sync(env)
        env[victim.slot] = env[victim.slot] + 1.0      # what a reused host block would hand back
    victim.run = corrupted
    with pytest.raises(R.RecomputeMismatchError) as e:
        cp(inputs)
    assert "[TS1112]" in str(e.value) and e.value.stage == "offload"
