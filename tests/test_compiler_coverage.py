"""Compiler paths added with the op coverage work (compiler/coverage_rules.py, docs/COMPILER.md).

Each program is compiled for float32 on the CPU and (when present) CUDA. The first call is checked
by the compiler against the float64 reference interpreter (kernel, group and end to end); here also
that the outputs agree with the reference at two batch sizes, that nothing took the float64 CPU
fallback, and which steps the ops became.
"""

import pytest
import torch

from truestar import ir
from truestar.compiler import compile_program

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
F32, F64 = torch.float32, torch.float64


def _check(prog, make_inputs, device, tol=1e-5, batches=(2, 5)):
    cp = compile_program(prog, device, "float32")
    for batch in batches:
        xs = make_inputs(batch)
        outs = cp([x.to(device) for x in xs])
        want = ir.run(prog, *[x.double() if x.is_floating_point() else x for x in xs])
        for o, w in zip(outs, want):
            if not w.dtype.is_floating_point:
                assert torch.equal(o.cpu(), w)
                continue
            scale = w.abs().max().item() or 1.0
            assert (o.double().cpu() - w).abs().max().item() / scale < tol
    assert not cp.report.fallbacks, cp.report.fallbacks
    return cp


def _r(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("device", DEVICES)
def test_masks_padding_flip_and_scatters_are_kernels(device):
    b = ir.Builder("movement")
    x = b.input("x", F32, ("B", 6, 7))
    s = b.input("s", F32, ("B", 3, 7))
    r = b.input("r", F32, ("B", 6))
    outs = [
        b.op("triu", x, diagonal=1), b.op("tril", x, diagonal=-2),
        b.op("pad", x, pads=[2, 1, 0, 3], value=-1.5),         # after only on one dim, both on the other
        b.op("pad", x, pads=[-1, 2], value=0.0),               # a crop and a pad
        b.op("flip", x, dims=[1, 2]),
        b.op("slice_scatter", x, s, dim=1, start=2, end=5, step=1),
        b.op("select_scatter", x, r, dim=2, index=-1),
    ]
    prog = b.build(outs)
    cp = _check(prog, lambda n: [_r(n, 6, 7, seed=n), _r(n, 3, 7, seed=n + 1), _r(n, 6, seed=n + 2)], device)
    assert not cp.report.library
    assert {"triu", "tril", "pad", "flip", "slice_scatter", "select_scatter"} <= set(cp.report.kernel_ops())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("c", [0.75, -1.3, 2.0, 5.0, -3.0, 0.0, 17.0])
def test_pow_with_any_constant_exponent(device, c):
    b = ir.Builder("pow_const")
    x = b.input("x", F32, ("B", 8))
    e = b.param("e", torch.tensor(c, dtype=F64))
    base = b.op("add", b.op("abs", x), b.param("h", torch.tensor(0.25, dtype=F64)))
    prog = b.build([b.op("pow", base, e), b.op("pow", x, e) if c.is_integer() else b.op("pow", base, e)])
    _check(prog, lambda n: [_r(n, 8, seed=n)], device, tol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_pow_of_a_constant_base_is_a_kernel(device):
    """c ** x for a positive constant c (rotary inverse frequencies: 10000 ** (2i / d)) is a kernel,
    exp(x ln c), not a float64 CPU fallback."""
    b = ir.Builder("pow_base")
    x = b.input("x", F32, ("B", 16))
    base = b.param("c", torch.tensor(10000.0, dtype=F64))
    prog = b.build([b.op("pow", base, b.op("mul", x, b.param("s", torch.tensor(-0.125, dtype=F64))))])
    cp = _check(prog, lambda n: [_r(n, 16, seed=n)], device, tol=2e-5)
    assert not cp.report.fallbacks and "pow" in cp.report.kernel_ops()


@pytest.mark.parametrize("device", DEVICES)
def test_pow_of_zero_and_negative_bases_follows_the_reference(device):
    b = ir.Builder("pow_edges")
    x = b.input("x", F32, (6,))
    prog = b.build([b.op("pow", x, b.param("e", torch.tensor(0.75, dtype=F64))),
                    b.op("pow", x, b.param("f", torch.tensor(-0.75, dtype=F64)))])
    cp = compile_program(prog, device, "float32")
    xv = torch.tensor([-2.0, -0.0, 0.0, 1e-30, 0.5, 3.0])
    got = [o.cpu() for o in cp([xv.to(device)])]
    want = ir.run(prog, xv.double())
    for g, w in zip(got, want):
        assert torch.equal(torch.isnan(g), torch.isnan(w))
        fin = torch.isfinite(w)
        assert torch.equal(torch.isinf(g), torch.isinf(w))
        assert torch.allclose(g[fin].double(), w[fin], rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_norms_compile_through_their_lowerings(device):
    b = ir.Builder("norms")
    x = b.input("x", F32, ("B", 6, 4, 3))
    w = b.param("w", torch.rand(6, dtype=F64) + 0.5)
    c = b.param("c", torch.randn(6, dtype=F64))
    v = b.param("v", torch.rand(3, dtype=F64) + 0.5)
    outs = [b.op("group_norm", x, w, c, num_groups=3, eps=1e-5),
            b.op("rms_norm", x, v, normalized_shape=[3], eps=1e-6),
            b.op("vector_norm", x, ord=2.0, dims=[1, 2], keepdim=False),
            b.op("vector_norm", x, ord=3.0, dims=[3], keepdim=True),
            b.op("vector_norm", x, ord=float("inf"), dims=None, keepdim=False)]
    prog = b.build(outs)
    cp = _check(prog, lambda n: [_r(n, 6, 4, 3, seed=n) * 2], device, tol=2e-5)
    assert not cp.report.library


@pytest.mark.parametrize("device", DEVICES)
def test_integer_results_run_on_the_device_and_match_exactly(device):
    """Casts to integer and bool dtypes, argmax and argmin (with ties: the first index) and integer
    sums are steps on the device, checked with zero tolerance, not the CPU reference."""
    b = ir.Builder("ints")
    x = b.input("x", F32, ("B", 9))
    t = b.op("mul", x, b.param("k", torch.tensor(4.0, dtype=F64)))
    i = b.op("cast", t, dtype=torch.int64)
    outs = [i, b.op("cast", x, dtype=torch.bool), b.op("argmax", x, dim=1, keepdim=False),
            b.op("argmin", x, dim=1, keepdim=True), b.op("sum", i, dims=[1], keepdim=False)]
    prog = b.build(outs)

    def inputs(n):
        v = torch.round(_r(n, 9, seed=n) * 2) / 2      # repeated values: argmax ties
        v[0, 3] = 0.0
        return [v]
    cp = _check(prog, inputs, device)
    steps = {tuple(st["ops"]): st.get("impl", "") for st in cp.report.library}
    from truestar.compiler import gather_native
    be = gather_native.backend_for(device)      # argmax and argmin: Truestar's kernel where there is one
    assert steps[("argmax",)] == (f"truestar argmax ({be})" if be else "torch integer argmax")
    assert steps[("cast",)].startswith("torch integer")


def test_a_wrong_mask_kernel_is_refused_on_first_call(monkeypatch):
    """The first call check catches a broken new rule: triu that keeps the diagonal below."""
    from truestar.compiler import coverage_rules, lower

    orig = lower.RULES["triu"]

    def wrong(L, node):
        node.attrs["diagonal"] = node.attrs.get("diagonal", 0) - 1     # the rule reads it once
        try:
            return orig(L, node)
        finally:
            node.attrs["diagonal"] += 1
    monkeypatch.setitem(lower.RULES, "triu", wrong)
    b = ir.Builder("bad_triu")
    x = b.input("x", F32, (4, 4))
    prog = b.build([b.op("triu", x, diagonal=0)])
    cp = compile_program(prog, "cpu", "float32")
    with pytest.raises(Exception):
        cp([_r(4, 4)])
    assert coverage_rules is not None


@pytest.mark.parametrize("device", DEVICES)
def test_float_maximum_and_minimum_are_kernels_and_propagate_nan(device):
    b = ir.Builder("extremes")
    x = b.input("x", F32, (6,))
    y = b.input("y", F32, (6,))
    prog = b.build([b.op("maximum", x, y), b.op("minimum", x, y)])
    cp = compile_program(prog, device, "float32")
    nan = float("nan")
    xv = torch.tensor([1.0, -2.0, nan, 3.0, 0.0, nan])
    yv = torch.tensor([0.5, 4.0, 1.0, nan, -0.0, nan])
    got = [o.cpu() for o in cp([xv.to(device), yv.to(device)])]
    want = ir.run(prog, xv.double(), yv.double())
    for g, w in zip(got, want):
        assert torch.equal(torch.isnan(g), torch.isnan(w))
        assert torch.equal(g[~torch.isnan(g)].double(), w[~torch.isnan(w)])
    assert not cp.report.fallbacks and not cp.report.library
