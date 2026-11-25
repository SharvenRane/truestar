"""Automatic CUDA graph replay of ordinary compiled calls (docs/COMPILER.md, Automatic graphs)."""

import pytest
import torch
from torch import nn

from truestar import ir
from truestar.compiler import compile_program

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def mlp_program():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 16), nn.Softmax(-1)).double()
    B = torch.export.Dim("B", min=1, max=64)
    return ir.import_module(m, (torch.randn(4, 32, dtype=torch.float64),), dynamic_shapes=({0: B},))


def x(b=4, seed=1):
    return torch.randn(b, 32, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed))


def test_the_second_call_at_a_signature_replays_with_the_same_bits():
    prog = mlp_program()
    off = compile_program(prog, "cuda", "float32", graphs="off")
    on = compile_program(prog, "cuda", "float32")
    for b in (4, 9):
        ref = [off([x(b)])[0] for _ in range(3)]
        got = [on([x(b)])[0] for _ in range(4)]          # ordinary, recorded, replayed, replayed
        assert all(torch.equal(r, ref[0]) for r in ref)
        assert all(torch.equal(g, ref[0]) for g in got)
    assert on.auto_graph_replays == 4 and off.auto_graph_replays == 0
    assert on.auto_graph_refusals == {}


def test_outputs_are_fresh_tensors_and_inputs_are_read_each_call():
    cp = compile_program(mlp_program(), "cuda", "float32")
    a, b = x(4, 1), x(4, 2)
    cp([a])
    ya = cp([a])                                          # recorded
    yb = cp([b])                                          # replayed with other values
    assert not torch.equal(ya[0], yb[0])
    assert torch.equal(cp([a])[0], ya[0])
    assert ya[0].data_ptr() != cp([a])[0].data_ptr()


def test_opt_outs(monkeypatch):
    prog = mlp_program()
    cp = compile_program(prog, "cuda", "float32", graphs="off")
    for _ in range(3):
        cp([x()])
    assert cp.auto_graph_replays == 0
    monkeypatch.setenv("TRUESTAR_GRAPHS", "off")
    cp = compile_program(prog, "cuda", "float32")
    for _ in range(3):
        cp([x()])
    assert cp.auto_graph_replays == 0
    with pytest.raises(ValueError):
        compile_program(prog, "cuda", "float32", graphs="always")


@pytest.mark.parametrize("kw", [{"memory_budget": "1 GiB"}, {"sentinels": True, "sentinel_read": "deferred"}])
def test_memory_control_and_deferred_sentinels_stay_ordinary(kw):
    cp = compile_program(mlp_program(), "cuda", "float32", **kw)
    for _ in range(3):
        cp([x()])
    assert cp.auto_graph_replays == 0


def test_cpu_inputs_and_verify_all_stay_ordinary(monkeypatch):
    cp = compile_program(mlp_program(), "cuda", "float32")
    for _ in range(3):
        cp([x().cpu()])
    assert cp.auto_graph_replays == 0
    monkeypatch.setenv("TRUESTAR_VERIFY", "all")
    cp = compile_program(mlp_program(), "cuda", "float32")
    for _ in range(3):
        cp([x()])
    assert cp.auto_graph_replays == 0


def test_random_draws_keep_following_the_call_count():
    """Draws keyed by the seed and the call count must be fresh at every call: the random step reads
    its key on the host, so its recording is refused and every call stays ordinary."""
    import truestar
    from truestar import Tensor, ops
    from truestar.compiler import CompiledExecutor

    def body(v: Tensor["B", 16]):
        return v + ops.randn(v.shape)

    f = truestar.function(seed=5, executor=CompiledExecutor("cuda"))(body)
    g = truestar.function(seed=5)(body)
    v = torch.zeros(3, 16)
    got = [f(v.cuda()).cpu() for _ in range(4)]
    ref = [g(v) for _ in range(4)]
    assert all(torch.equal(a.to(r.dtype), r) for a, r in zip(got, ref))
    assert not torch.equal(got[0], got[1])


def test_more_signatures_than_the_cap_run_as_ordinary_calls():
    cp = compile_program(mlp_program(), "cuda", "float32")
    cp.AUTO_GRAPHS_MAX = 2
    for b in (1, 2, 3):
        cp([x(b)])
        cp([x(b)])
    assert len(cp._auto._entries) == 2
    assert any("signatures" in why for why in cp.auto_graph_refusals.values())


def test_profile_measures_the_ordinary_call_and_restores_the_setting():
    from truestar.compiler.profile import profile_program
    cp = compile_program(mlp_program(), "cuda", "float32")
    prof = profile_program(cp, [x()], iters=3, warmup=1)
    assert prof and cp.auto_graphs is True
