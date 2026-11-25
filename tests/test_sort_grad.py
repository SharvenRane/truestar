"""Gradients of sort, topk and the gathers (scatter style backward through ``scatter_add``):
imported models against torch.autograd in float64, ties, determinism and the compiled path.
Rule by rule finite difference cases are in tests/test_autodiff.py."""

import pytest
import torch
from torch import nn

from truestar import ir
from truestar.ir import reference
from truestar.ir.autodiff import grad

F64 = torch.float64
CUDA = torch.cuda.is_available()


def rel(a, b):
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-300)
    return (a.double().cpu() - b.double().cpu()).abs().max().item() / scale


class Sampler(nn.Module):
    """sort, topk (both directions) and a gather with the sort indices, as a sampler's logits path."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(9, dtype=F64))

    def forward(self, x):
        v, i = torch.sort(x * self.w, dim=-1, descending=True, stable=True)
        top = torch.topk(x, self.k, dim=-1).values
        low = torch.topk(x, 2, dim=-1, largest=False).values
        g = torch.gather(x * x, -1, i)
        return torch.cat([v * torch.softmax(v, dim=-1), top, low, g], dim=-1)


def _import(model, x):
    B = torch.export.Dim("B", min=1, max=64)
    return ir.import_module(model, (x,), dynamic_shapes=({0: B},))


def _torch_grads(model, x, v, topk_by_stable_sort=False):
    x = x.clone().requires_grad_(True)
    if topk_by_stable_sort:  # PyTorch does not promise which tied entry torch.topk returns
        orig = torch.topk

        def stable_topk(t, k, dim=-1, largest=True):
            s = torch.sort(t, dim=dim, descending=largest, stable=True)
            return torch.return_types.topk((s.values.narrow(dim, 0, k), s.indices.narrow(dim, 0, k)))
        torch.topk = stable_topk
        try:
            out = model(x)
        finally:
            torch.topk = orig
    else:
        out = model(x)
    return torch.autograd.grad(out, [x, model.w], v)


@pytest.mark.parametrize("ties", [False, True])
def test_imported_sampler_gradient_matches_torch_autograd(ties):
    torch.manual_seed(0)
    model = Sampler()
    x0 = torch.randn(2, 9, dtype=F64)
    prog = _import(model, x0)
    gp = grad(prog, [prog.inputs[0].name, "w"], seed="input")
    for b in (2, 5):  # one program, two batch sizes
        g = torch.Generator().manual_seed(b)
        if ties:   # few distinct values: ties inside sort, at the topk boundary and in the gather
            x = torch.randint(-2, 3, (b, 9), generator=g).to(F64) * 0.5
        else:
            x = torch.randn(b, 9, generator=g, dtype=F64)
        v = torch.randn(b, 9 + 3 + 2 + 9, generator=g, dtype=F64)
        out, gx, gw = ir.run(gp, x, v)
        want_x, want_w = _torch_grads(model, x, v, topk_by_stable_sort=ties)
        assert rel(out, model(x)) < 1e-14
        assert rel(gx, want_x) < 1e-12, rel(gx, want_x)
        assert rel(gw, want_w) < 1e-12, rel(gw, want_w)


def test_ties_resolve_to_the_first_position_and_never_split():
    """Equal entries: the cotangent of sorted position r goes to the entry the stable sort put there,
    whole (no splitting between ties, unlike amax)."""
    b = ir.Builder("ties")
    x = b.input("x", F64, (1, 5))
    vals, _ = b.op_multi("topk", x, k=2, dim=-1, largest=True)
    gp = grad(b.build([vals]), ["x"], seed="input")
    xv = torch.tensor([[1.0, 3.0, 3.0, 3.0, 0.0]], dtype=F64)
    _, gx = ir.run(gp, xv, torch.tensor([[10.0, 20.0]], dtype=F64))
    assert torch.equal(gx, torch.tensor([[0.0, 10.0, 20.0, 0.0, 0.0]], dtype=F64))


def test_gradient_program_is_deterministic():
    torch.manual_seed(1)
    model = Sampler()
    x = torch.randint(-2, 3, (4, 9)).to(F64)
    prog = _import(model, x)
    gp = grad(prog, [prog.inputs[0].name, "w"], seed="ones")
    first = ir.run(gp, x)
    for _ in range(3):
        again = ir.run(gp, x)
        assert all(torch.equal(a, c) for a, c in zip(first, again))


def test_scatter_add_reference_matches_its_definition_and_orders_repeats():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(3, 4, 5, generator=g, dtype=F64)
    idx = torch.randint(0, 4, (3, 6, 2), generator=g)
    src = torch.randn(3, 6, 2, generator=g, dtype=F64)
    got = reference.IMPLS["scatter_add"]({"dim": 1}, [x, idx, src], None)
    want = x.clone()
    for a in range(3):
        for k in range(6):
            for c in range(2):
                want[a, idx[a, k, c], c] += src[a, k, c]
    assert torch.equal(got, want)   # same additions in the same order: bitwise
    with pytest.raises(ir.IRError, match="out of range"):
        reference.IMPLS["scatter_add"]({"dim": 1}, [x, idx + 4, src], None)


def test_broken_sort_rule_is_caught(monkeypatch):
    """A sort rule that gathers instead of scattering (the inverse permutation applied the wrong way)
    is wrong whenever the permutation is not an involution; the torch.autograd comparison fails."""
    from truestar.ir import grad_rules

    def wrong(e, n, g):
        (x,) = n.inputs
        d = n.attrs.get("dim", -1) % x.type.rank
        return [lambda: e.op("gather", g, n.extra_outputs[0], dim=d)]

    monkeypatch.setitem(grad_rules.GRAD_RULES, "sort", wrong)
    torch.manual_seed(0)
    model = Sampler()
    x = torch.randn(3, 9, dtype=F64)
    prog = _import(model, x)
    gp = grad(prog, [prog.inputs[0].name, "w"], seed="input")
    v = torch.randn(3, 23, dtype=F64)
    _, gx, _ = ir.run(gp, x, v)
    want_x, _ = _torch_grads(model, x, v)
    assert rel(gx, want_x) > 1e-2


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_compiled_gradient_program_matches_the_reference(device):
    """The gradient program through the compiler: checked against the float64 reference on the first
    call (raises otherwise), then compared here, at two batch sizes, with ties."""
    from truestar.compiler import compile_program

    torch.manual_seed(0)
    model = Sampler()
    prog = _import(model, torch.randn(2, 9, dtype=F64))
    gp = grad(prog, [prog.inputs[0].name, "w"], seed="input")
    cp = compile_program(gp, device, "float32")
    assert not cp.report.fallbacks          # sort, topk, gather, scatter_add run on the device
    for b in (3, 7):
        g = torch.Generator().manual_seed(10 + b)
        x = torch.randint(-2, 3, (b, 9), generator=g).to(F64) * 0.5
        v = torch.randn(b, 23, generator=g, dtype=F64)
        outs = cp([x.float(), v.float()])
        want = ir.run(gp, x, v)
        for o, w in zip(outs, want):
            assert rel(o, w) < 1e-5
        again = cp([x.float(), v.float()])
        assert all(torch.equal(a, c) for a, c in zip(outs, again))   # deterministic run to run


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_compiled_sort_backward_is_exact_data_movement():
    """d/dx sum(sort(x) * v) is v put back in x's order: every entry is a copy, so the float32
    compiled result equals the reference exactly."""
    from truestar.compiler import compile_program

    b = ir.Builder("sortgrad")
    x = b.input("x", torch.float32, ("B", 50))
    vals, _ = b.op_multi("sort", x, dim=-1, descending=True)
    gp = grad(b.build([vals]), ["x"], seed="input")
    cp = compile_program(gp, "cuda", "float32")
    g = torch.Generator().manual_seed(0)
    xs = torch.randint(-5, 5, (6, 50), generator=g).float()
    vs = torch.randn(6, 50, generator=g)
    _, gx = cp([xs, vs])
    _, want = ir.run(gp, xs.double(), vs.double())
    assert torch.equal(gx.cpu().double(), want)


def test_a_wrong_device_sort_is_caught_by_the_first_call_check(monkeypatch):
    """An IndexStep that ignores ``descending`` computes a valid sort, just the wrong one; the region
    check against the reference must stop it before a result is returned."""
    from truestar.compiler import ProgramMismatchError, compile_program
    from truestar.compiler import runtime

    def wrong(rt, a, args):
        s = torch.sort(args[0], dim=a.get("dim", -1), stable=True)
        return [s.values, s.indices]

    monkeypatch.setitem(runtime.INDEX_IMPLS, "sort", wrong)
    b = ir.Builder("desc")
    x = b.input("x", torch.float32, ("B", 8))
    vals, idx = b.op_multi("sort", x, dim=-1, descending=True)
    cp = compile_program(b.build([vals, idx]), "cuda" if CUDA else "cpu", "float32")
    with pytest.raises(ProgramMismatchError):
        cp([torch.randn(3, 8)])
