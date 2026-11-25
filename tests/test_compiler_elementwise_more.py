"""Kernel lowerings added for rotary positions and Hugging Face models (2026-11-24): cos, sin,
log, abs, reciprocal, cast and pow with a constant exponent. Each compiled program is checked
against the float64 reference on its first call; here also that nothing took the fallback."""

import pytest
import torch

from truestar import ir
from truestar.compiler import compile_program

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _prog():
    b = ir.Builder("elementwise_more")
    x = b.input("x", torch.float32, ("B", 16))
    m = b.input("m", torch.bool, ("B", 16))
    i = b.input("i", torch.int64, ("B", 16))
    two = b.param("two", torch.tensor(2.0, dtype=torch.float64))
    half = b.param("half", torch.tensor(0.5, dtype=torch.float64))
    neg1 = b.param("neg1", torch.tensor(-1.0, dtype=torch.float64))
    three = b.param("three", torch.tensor(3.0, dtype=torch.float64))
    ax = b.op("abs", x)
    outs = [
        b.op("cos", x), b.op("sin", x), b.op("log", b.op("add", ax, half)), ax,
        b.op("reciprocal", b.op("add", ax, half)),
        b.op("pow", x, two), b.op("pow", ax, half), b.op("pow", b.op("add", ax, half), neg1), b.op("pow", x, three),
        b.op("add", b.op("cast", m, dtype=torch.float32), b.op("cast", i, dtype=torch.float32)),
        b.op("cast", x, dtype=torch.float64),
    ]
    return b.build(outs)


@pytest.mark.parametrize("device", DEVICES)
def test_new_rules_compile_to_kernels_and_match(device):
    prog = _prog()
    cp = compile_program(prog, device, "float32")
    for batch in (3, 8):
        g = torch.Generator().manual_seed(batch)
        x = torch.randn(batch, 16, generator=g) * 3
        m = torch.rand(batch, 16, generator=g) < 0.5
        i = torch.randint(-50, 50, (batch, 16), generator=g)
        outs = cp([x, m, i])
        want = ir.run(prog, x.double(), m, i)
        for o, w in zip(outs, want):
            assert ((o.double().cpu() - w).abs().max() / w.abs().max()).item() < 1e-5
    assert not cp.report.fallbacks, cp.report.fallbacks


@pytest.mark.parametrize("device", DEVICES)
def test_cos_and_sin_of_large_angles_stay_accurate(device):
    """cos and sin of float32 angles up to 32k radians (rotary positions times frequencies): a fast
    approximate GPU sine would miss the float32 bar here (the Triton kernels use libdevice). The
    angles are inputs, so the reference sees the same float32 numbers."""
    b = ir.Builder("angles")
    a = b.input("a", torch.float32, ("T", 32))
    prog = b.build([b.op("cos", a), b.op("sin", a)])
    g = torch.Generator().manual_seed(0)
    ang = torch.rand(512, 32, generator=g) * 32768
    outs = compile_program(prog, device, "float32")([ang])
    want = ir.run(prog, ang.double())
    for o, w in zip(outs, want):
        assert (o.double().cpu() - w).abs().max().item() < 2e-6


def test_a_rounding_cast_is_not_a_copy():
    """A cast to float16 inside a float32 executor rounds in the reference, so it takes the fallback
    instead of a kernel that would copy."""
    b = ir.Builder("cast16")
    x = b.input("x", torch.float32, ("B", 4))
    prog = b.build([b.op("cast", x, dtype=torch.float16)])
    cp = compile_program(prog, "cpu", "float32")
    x = torch.randn(3, 4) * 1000
    out = cp([x])[0]
    assert torch.equal(out.double(), x.half().double())
    (fb,) = cp.report.fallbacks
    assert "rounds" in fb["reason"]


def test_pow_with_a_tensor_exponent_takes_the_named_fallback():
    # every constant exponent has a kernel since the op coverage work (tests/test_compiler_coverage.py)
    b = ir.Builder("pow_other")
    x = b.input("x", torch.float32, ("B", 4))
    e = b.input("e", torch.float32, ("B", 4))
    prog = b.build([b.op("pow", b.op("abs", x), e)])
    cp = compile_program(prog, "cpu", "float32")
    cp([torch.randn(2, 4), torch.randn(2, 4)])
    (fb,) = cp.report.fallbacks
    assert fb["ops"] == ["pow"] and "exponent" in fb["reason"]


@pytest.mark.parametrize("device", DEVICES)
def test_integer_index_arithmetic_runs_on_the_device_exactly(device):
    """Slot and block index arithmetic (a paged cache's ``table * block + offset``) stays integer on
    the device instead of a float64 round trip through the CPU."""
    b = ir.Builder("slots")
    table = b.input("table", torch.int64, ("B", "M"))
    pos = b.input("pos", torch.int64, ("B", "T"))
    bs = b.param("bs", torch.tensor(16))
    off = b.op("remainder", pos, bs)
    slots = b.op("add", b.op("mul", table, bs), b.op("sub", b.op("neg", b.op("neg", table)), table))
    prog = b.build([off, slots, b.op("maximum", pos, b.op("abs", b.op("neg", pos)))])
    cp = compile_program(prog, device, "float32")
    g = torch.Generator().manual_seed(0)
    t = torch.randint(0, 1 << 40, (3, 5), generator=g)      # beyond float32's exact integers
    p = torch.randint(-100, 3000, (3, 7), generator=g)
    outs = cp([t, p])
    want = ir.run(prog, t, p)
    for o, w in zip(outs, want):
        assert o.dtype == torch.int64 and torch.equal(o.cpu(), w)
    assert not cp.report.fallbacks
    assert {e["impl"] for e in cp.report.library} >= {"torch integer remainder", "torch integer mul"}
