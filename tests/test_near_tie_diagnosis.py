"""An end to end mismatch caused by a comparison decided on a near tie is named as ill conditioning.

A float comparison whose operands are within rounding of each other (a relu mask of an input
at 1e-8 of its rms) can go either way in a correct float32 run, and everything downstream of it
changes. scripts/bench_memory.py's language model hit one: layers.1 linear1 at [11, 23, 895] is
-1.28e-08 in float64 and +5.96e-08 in eager PyTorch float32 on the CPU, and both that and the
compiled CUDA program end 0.00754 off float64 on that layer's weight gradient (bar 0.00153). The end
to end check then has to say it is the tie, with the numbers, and not suggest a wiring fault; and it
must not say so when the decisions do not explain the error.

The tests nudge one compiled value across the tie by 2^-29 (below float32's unit roundoff of a
value of size 1), which every region check accepts as rounding, as a real float32 run would.
"""

import os
import tempfile

import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import ir  # noqa: E402
from truestar.compiler import ProgramMismatchError, compile_program  # noqa: E402
from truestar.ir.core import Builder  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


DEVICES = ["cpu"] + (["cuda"] if _has_gpu() else [])
TIE = 2.0 ** -30          # the float64 value of h[0, 0]: positive, far below float32 rounding of h


def tie_program(wiring_output: bool = False):
    """h = x @ w is its own kernel (a later contraction reads relu(h)), the mask gt(h, 0) is decided
    in a later kernel on the h that kernel received, as a relu's gradient is."""
    g = torch.Generator().manual_seed(0)
    b = Builder("tie")
    x = b.input("x", torch.float32, ("B", 8))
    u = b.input("u", torch.float32, ("B", 8))
    w = torch.randn(8, 8, generator=g)
    w[0, 0] = 1.0
    h = b.op("matmul", x, b.param("w", w))
    z = b.op("matmul", b.op("relu", h), b.param("w2", torch.randn(8, 8, generator=g)))
    mask = b.op("gt", h, b.param("zero", torch.zeros(())))
    y = b.op("where", mask, b.op("add", u, z), b.op("full_like", u, value=0.0, dtype=None))
    outs = [b.op("tanh", y)]
    if wiring_output:
        outs.append(b.op("select", b.op("permute", z, dims=[1, 0]), dim=0, index=3))
    xv = torch.randn(4, 8, generator=g)
    xv[0] = 0.0
    xv[0, 0] = TIE                        # h[0, 0] = TIE * w[0, 0] exactly, in float32 and float64
    uv = torch.randn(4, 8, generator=g)
    uv[0, 0] = 2.0
    return b.build(outs), [xv, uv]


def nudge_across_the_tie(cp):
    """The compiled h lands on the other side of 0 at [0, 0], by 2^-29 (a rounding size difference)."""
    step = next(s for s in cp.plan.steps if s.ops() == ["matmul"])
    real = step.run

    def run(env, rt):
        real(env, rt)
        env[step.out] = env[step.out].clone()
        env[step.out][0, 0] -= 2 * TIE

    step.run = run


@pytest.mark.parametrize("device", DEVICES)
def test_a_near_tie_decision_is_named_as_ill_conditioning(device):
    prog, inputs = tie_program()
    cp = compile_program(prog, device)
    cp(inputs)
    assert cp.verified                    # control: without the nudge the program is verified
    cp = compile_program(prog, device)
    nudge_across_the_tie(cp)
    with pytest.raises(ProgramMismatchError) as e:
        cp(inputs)
    err = e.value
    msg = str(err)
    assert err.stage == "end_to_end" and err.ill_conditioned
    assert "ill conditioned in float32 at a near tie" in msg
    assert "%gt = gt(%matmul, %zero) on 1 element(s), first at [0, 0]" in msg
    assert "9.31e-10 against 0" in msg                       # the float64 operand, 2^-30
    assert "wired wrongly" not in msg
    [tie] = err.near_ties
    assert tie["value"] == "gt" and tie["at"] == [0, 0] and tie["lhs_value"] == TIE and tie["count"] == 1
    # the program is still refused, and checked again on the next call
    assert not cp.verified


def test_a_wiring_fault_next_to_a_near_tie_is_not_called_ill_conditioning():
    prog, inputs = tie_program(wiring_output=True)
    cp = compile_program(prog, "cpu")
    cp(inputs)
    assert cp.verified
    cp = compile_program(prog, "cpu")
    nudge_across_the_tie(cp)
    step = next(s for s in cp.plan.steps if s.kind == "layout" and s.nodes[0].op == "permute")
    real = step.fn
    step.fn = lambda attrs, args, shape: real(attrs, [args[0].flip(0)], shape)   # moves data wrongly
    with pytest.raises(ProgramMismatchError) as e:
        cp(inputs)
    err = e.value
    msg = str(err)
    assert err.stage == "end_to_end" and not err.ill_conditioned and err.near_ties == []
    assert "but the float64 reference taking them still differs" in msg
    assert "wired wrongly" in msg and "ill conditioned" not in msg


def test_a_wiring_fault_that_only_moves_a_mask_is_not_called_a_near_tie():
    """Replaying this run's decisions explains this output exactly (the fault only reaches the
    comparison), so the replay alone would call it ill conditioning; the decisions are taken on values
    far from the tie, and that is what refuses the label."""
    g = torch.Generator().manual_seed(1)
    b = Builder("mask_wiring")
    x = b.input("x", torch.float32, ("B", 8))
    h = b.op("matmul", x, b.param("w", torch.randn(8, 8, generator=g)))
    u = b.input("u", torch.float32, (8, "B"))
    ht = b.op("permute", h, dims=[1, 0])
    mask = b.op("gt", ht, b.param("zero", torch.zeros(())))
    y = b.op("where", mask, u, b.op("full_like", u, value=0.0, dtype=None))
    prog = b.build([y])
    inputs = [torch.randn(4, 8, generator=g), torch.randn(8, 4, generator=g)]
    cp = compile_program(prog, "cpu")
    cp(inputs)
    assert cp.verified
    cp = compile_program(prog, "cpu")
    step = next(s for s in cp.plan.steps if s.kind == "layout" and s.nodes[0].op == "permute")
    real = step.fn
    step.fn = lambda attrs, args, shape: real(attrs, [args[0].flip(0)], shape)
    with pytest.raises(ProgramMismatchError) as e:
        cp(inputs)
    err = e.value
    msg = str(err)
    assert err.stage == "end_to_end" and not err.ill_conditioned
    assert "and explain this output, but not all on a near tie" in msg
    assert "wired wrongly" in msg


def test_reference_overrides_replace_a_decision():
    prog, inputs = tie_program()
    (ref,), named = ir.reference.run(prog, *inputs, return_all=True)
    mask = next(v for n in prog.nodes for v in n.outputs if n.op == "gt")
    flipped = named[mask.name].clone()
    flipped[0, 0] = ~flipped[0, 0]
    (got,) = ir.reference.run(prog, *inputs, overrides={id(mask): flipped})
    diff = (got != ref).nonzero().tolist()
    assert diff == [[0, 0]] and got[0, 0] == 0.0          # tanh(0): the where took the other branch
    with pytest.raises(ValueError):
        ir.reference.run(prog, *inputs, device="meta", overrides={id(mask): flipped})
