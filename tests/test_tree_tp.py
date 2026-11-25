"""Tensor parallel steps bitwise equal to the one device step (truestar.parallel.tree, P55)."""

import os
import sys
import types

import pytest
import torch

from truestar import codes
from truestar.parallel.layout import Layout
from truestar.parallel import tree as TR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parallel_models as PM  # noqa: E402

MODEL_FILE = """import sys
sys.path.insert(0, {here!r})
import parallel_models as PM


def get_model():
    return PM.tiny()


def get_inputs():
    return PM.tiny_inputs()
"""


def _example():
    """The pre norm transformer of tests/parallel_models.py (the one examples/parallel_model.py
    shows) with its tp plan: q, k, v column, o row, gate and up column:2, down row."""
    return types.SimpleNamespace(get_model=PM.tiny, get_inputs=PM.tiny_inputs,
                                 get_parallel_plan=lambda layout: PM.tiny_plan())


def test_codes_exist():
    for c in ("TS1100", "TS1101", "TS1102", "TS1103"):
        assert c in codes.CODES


def test_pairwise_tree_of_subtrees_is_the_whole_tree():
    """The property the cross rank finish rests on: for G a power of two, summing aligned blocks
    of G/tp by the tree and then the tp roots by the tree is bitwise the tree over all G."""
    g = torch.Generator().manual_seed(3)
    for G in (2, 4, 8, 16):
        x = (torch.randn(G, 257, generator=g) * torch.logspace(-8, 8, G).unsqueeze(1)).float()
        whole = TR.ptree(x, 0)
        for tp in (1, 2, 4, 8, 16):
            if G % tp:
                continue
            roots = torch.stack([TR.ptree(x[r * (G // tp):(r + 1) * (G // tp)], 0) for r in range(tp)])
            assert TR.bits_equal(TR.ptree(roots, 0), whole)[0]
    # and a left to right sum is not the tree (the reason rank order sums differ)
    x = torch.tensor([1.0, 1e8, -1e8, 1.0], dtype=torch.float32).reshape(4, 1)
    seq = ((x[0] + x[1]) + x[2]) + x[3]
    assert not TR.bits_equal(seq, TR.ptree(x, 0))[0]


def test_bits_equal_sees_signed_zero():
    assert not TR.bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))[0]
    assert TR.bits_equal(torch.tensor([1.5]), torch.tensor([1.5]))[0]


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("leaf", ["exact", "tiled"])
@pytest.mark.parametrize("opt", ["adamw", "sgd"])
def test_tp_step_is_bitwise_the_one_device_step(tp, leaf, opt):
    pm = _example()
    r = TR.tree_check(pm.get_model(), pm.get_inputs(), tp=tp, plan=pm.get_parallel_plan(Layout(tp=tp)), leaf=leaf,
                      optimizer=opt, clip=1.0)
    assert r.bitwise and r.codes[0] == "TS1100", r.to_text()
    assert r.compared == tp * (2 + 2 * len(list(pm.get_model().parameters())))
    # the plain step (F.linear, rank order sums) is not: the check would catch the tree being off
    assert r.plain_differing and "TS1103" in r.codes


def test_breaking_the_tree_is_caught(monkeypatch):
    """Guard for the guard: a cross rank finish in rank order (left to right) instead of the tree."""
    pm = _example()
    real = TR.TreeSpec.gather

    def gather_then_sequential(self, t, what):
        g = real(self, t, what)
        if g.shape[0] < 3:
            return g
        acc = g[0]
        for x in g[1:-1]:
            acc = acc + x
        return torch.stack([acc, g[-1]])          # ((a + b) + c) + d, as the plain sum does
    monkeypatch.setattr(TR.TreeSpec, "gather", gather_then_sequential)
    r = TR.tree_check(pm.get_model(), pm.get_inputs(), tp=4, plan=pm.get_parallel_plan(Layout(tp=4)),
                      plain=False)
    assert not r.bitwise and r.codes == ["TS1101"]


def test_tp_that_does_not_split_the_tree_is_refused():
    pm = _example()
    with pytest.raises(TR.TreeLayoutError, match="TS1102"):
        TR.install(pm.get_model(), _styles(pm.get_parallel_plan(Layout(tp=2))), tp=3)   # widths 16, 24: no factor 3
    # cap 1: one group per axis, which tp=2 cannot split
    r = TR.tree_check(pm.get_model(), pm.get_inputs(), tp=2, plan=pm.get_parallel_plan(Layout(tp=2)), cap=1,
                      plain=False)
    assert r.codes == ["TS1102"]


def test_thread_ranks_refuse_cuda_models():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pm = _example()
    with pytest.raises(ValueError, match="CPU only"):
        TR._run_threads(pm.get_model().cuda(), [], None, None, {}, 2, {})


def test_gloo_processes_cpu(tmp_path):
    pm = _example()
    mf = tmp_path / "tree_model.py"
    mf.write_text(MODEL_FILE.format(here=HERE), encoding="utf-8")
    r = TR.tree_check(pm.get_model(), pm.get_inputs(), tp=2, plan=pm.get_parallel_plan(Layout(tp=2)), ranks="gloo",
                      model_file=str(mf), plain=False)
    assert r.bitwise, r.to_text()


# ----------------------------------------------------------------------------- beyond 2 ranks (P66)


def _styles(plan):
    from truestar.parallel.layout import parse_style

    return {k: parse_style(v) for k, v in plan.items()}


def _wide(d, ff):
    """The tiny transformer at other widths, gated by tanh: torch 2.11's CPU silu, gelu and sigmoid
    give other bits on a strided view (the gate chunk) than on a contiguous tensor for some widths,
    so a silu gate makes even tp=2 differ for ff=48 (outside the tree's linear layers; reported)."""
    torch.manual_seed(0)
    return PM.Tiny(d=d, ff=ff, head_dim=4, act=torch.tanh)


@pytest.mark.parametrize("tp,d,ff", [(3, 24, 48), (6, 24, 48), (8, 32, 64), (4, 32, 64)])
def test_tp_with_any_factor_is_bitwise_the_one_device_step(tp, d, ff):
    """tp = W * 2^a: an odd factor W cuts every split axis into W blocks, in the one device step too."""
    r = TR.tree_check(_wide(d, ff), PM.tiny_inputs(), tp=tp, plan=PM.tiny_plan(), clip=1.0)
    assert r.bitwise and r.codes[0] == "TS1100", r.to_text()
    assert r.plain_differing and "TS1103" in r.codes


def test_odd_ways_tree_is_the_block_tree():
    """The property tp=3 rests on: the pairwise tree over W * 2^k groups sums each block of 2^k first,
    so rank subtree roots (tp = W * 2^a) finished by the tree give the whole tree."""
    g = torch.Generator().manual_seed(5)
    for W in (1, 3, 5, 7):
        for G in (1, 2, 4, 8):
            x = (torch.randn(W * G, 33, generator=g) * torch.logspace(-8, 8, W * G).unsqueeze(1)).float()
            whole = TR.ptree(x, 0)
            for a in range(4):
                tp = W * 2 ** a
                if G % 2 ** a:
                    continue
                per = W * G // tp
                roots = torch.stack([TR.ptree(x[r * per:(r + 1) * per], 0) for r in range(tp)])
                assert TR.bits_equal(TR.ptree(roots, 0), whole)[0], (W, G, tp)


def test_tp_three_on_widths_without_a_factor_three_is_refused_by_name():
    with pytest.raises(TR.TreeLayoutError, match=r"TS1102.*16 output features per part do not split into 3"):
        TR.install(PM.tiny(), _styles(PM.tiny_plan()), tp=3)
    # a one device run cannot be told to use another tp's blocks
    with pytest.raises(TR.TreeLayoutError, match="TS1102"):
        TR.install(_wide(24, 48), _styles(PM.tiny_plan()), tp=6, ways=1)


def test_power_of_two_tp_keeps_the_tree_it_had():
    """W = 1 for powers of two: the one device tree is the one before odd factors existed (same bits)."""
    from truestar.parallel.runtime import MSE

    y, plan = torch.zeros(8, 4, 6), _styles(PM.tiny_plan())
    a = TR.run_step(_wide(32, 64), [PM.tiny_inputs()], y, MSE(), plan, clip=1.0)
    b = TR.run_step(_wide(32, 64), [PM.tiny_inputs()], y, MSE(), plan, clip=1.0, ways=1)
    c = TR.run_step(_wide(24, 48), [PM.tiny_inputs()], y, MSE(), plan, clip=1.0, ways=3)
    assert all(TR.bits_equal(a.updated[k], b.updated[k])[0] for k in a.updated)
    assert c.error is None and c.norm is not None
    assert TR.axis_groups(64, 8) == TR.group_count(64, 8) == 8 and TR.axis_groups(48, 8, 3) == 24
    assert TR.axis_groups(64, 8, 3) == 0 and TR.odd_part(12) == 3 and TR.odd_part(8) == 1


def test_gloo_processes_cpu_three_ranks(tmp_path):
    mf = tmp_path / "tree_model3.py"
    mf.write_text(MODEL_FILE.format(here=HERE).replace("PM.tiny()", "PM.Tiny(d=24, ff=48, head_dim=4, act=PM.torch.tanh)"),
                  encoding="utf-8")   # each process loads the parent's weights into this model
    r = TR.tree_check(_wide(24, 48), PM.tiny_inputs(), tp=3, plan=PM.tiny_plan(), ranks="gloo", model_file=str(mf),
                      plain=False)
    assert r.bitwise, r.to_text()
    assert r.devices == ["cpu"] * 3, r.devices


@pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.device_count() >= 2), reason="needs 2 or more GPUs")
@pytest.mark.parametrize("tp", [2, 4])
def test_gloo_processes_each_on_its_own_gpu(tmp_path, tp):
    """Rank r runs on cuda:r (before, every rank process used cuda:0) and stays bitwise equal."""
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    mf = tmp_path / "tree_model_gpu.py"
    mf.write_text(MODEL_FILE.format(here=HERE).replace("PM.tiny()", "PM.Tiny(d=32, ff=64, head_dim=4, act=PM.torch.tanh)"),
                  encoding="utf-8")
    r = TR.tree_check(_wide(32, 64), PM.tiny_inputs(), tp=tp, plan=PM.tiny_plan(), ranks="gloo", model_file=str(mf),
                      device="cuda", plain=False)
    assert r.bitwise, r.to_text()
    assert [d.split(" ")[0] for d in r.devices] == [f"cuda:{i}" for i in range(tp)], r.devices
