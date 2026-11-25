"""Truestar's own gather and argmax kernels (compiler/gather_native.py).

The kernels are checked here against torch's ops on every layout the compiler hands them (single
and several index tensors, adjacent or not, broadcast, negative entries, strided inputs, every
dtype), against the float64 reference inside compiled programs (embedding, the cross entropy class
pick, index_select padding, argmax, the adaptive max pool), and for the two properties the torch
step had: an out of range index is an error naming the op, and the first call check refuses a
wrong kernel.
"""

import pytest
import torch

from truestar import ir
from truestar.compiler import compile_program, gather_native as GN

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
F32, F64 = torch.float32, torch.float64


def _backend(device):
    b = GN.backend_for(device)
    if b is None:
        pytest.skip(f"no Truestar gather kernel for {device} here")
    return b


def _g(seed):
    return torch.Generator().manual_seed(seed)


def _gather(op, attrs, args, device):
    got = GN.gather(op, attrs, [None if a is None else a.to(device) for a in args], _backend(device))
    assert got is not None
    out, flag, _ = got
    return out.cpu(), int(flag.cpu()[0])


def _x(shape, dtype, seed=0):
    x = torch.randn(*shape, generator=_g(seed), dtype=F64)
    if dtype == torch.bool:
        return x > 0
    if not dtype.is_floating_point:
        return (x * 100).to(dtype)
    return x.to(dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [F32, F64, torch.float16, torch.bfloat16, torch.int64, torch.bool])
def test_index_select_and_gather_equal_torch_bitwise(device, dtype):
    x = _x((4, 7, 5), dtype)
    if dtype.is_floating_point:
        x.view(-1)[3] = float("nan")       # a NaN is copied as it is
    for d in range(3):
        i = torch.randint(-x.shape[d], x.shape[d], (6,), generator=_g(d))
        got, flag = _gather("index_select", {"dim": d}, [x, i], device)
        _same(got, torch.index_select(x, d, i % x.shape[d]))
        assert flag == 0
        shape = list(x.shape)
        shape[d] = 3
        shape[(d + 1) % 3] -= 1                       # gather's index may be smaller off its dim
        gi = torch.randint(0, x.shape[d], shape, generator=_g(10 + d))
        got, flag = _gather("gather", {"dim": d}, [x, gi], device)
        _same(got, torch.gather(x, d, gi))
        assert flag == 0


def _same(got, want):
    assert got.shape == want.shape and got.dtype == want.dtype
    if want.dtype.is_floating_point:
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
    else:
        assert torch.equal(got, want)


@pytest.mark.parametrize("device", DEVICES)
def test_advanced_indexing_every_placement_equals_torch(device):
    x = _x((3, 5, 6, 4), F32, seed=1)
    a = torch.randint(-5, 5, (2, 3), generator=_g(1))
    a3 = torch.randint(-3, 3, (2, 3), generator=_g(6))
    b = torch.randint(0, 6, (3,), generator=_g(2)).to(torch.int32)
    c = torch.randint(0, 4, (2, 1), generator=_g(3))
    cases = [
        [None, a],                       # one tensor after a full slice (embedding is [a] on dim 0)
        [torch.randint(0, 3, (4, 2), generator=_g(4))],
        [None, a, b],                    # two adjacent tensors, broadcast
        [None, None, b, c],              # adjacent at the end, broadcast (the adaptive max pool's taps)
        [a3[:, :1], None, b],             # not adjacent: the broadcast dims go first
        [torch.tensor([0, 2]), None, None, c[:, 0]],
        [a3[:, :1], b[:1], torch.randint(0, 6, (2, 3), generator=_g(5))],   # three tensors
    ]
    for idx in cases:
        want = x[tuple(slice(None) if i is None else i.long() for i in idx)]
        got, flag = _gather("index", {}, [x] + idx, device)
        assert got.shape == want.shape and torch.equal(got, want), idx
        assert flag == 0
    xt = x.permute(2, 0, 3, 1)               # a strided input is read through its strides
    want = xt[:, a3]
    got, _ = _gather("index", {}, [xt, None, a3], device)
    assert torch.equal(got, want)


@pytest.mark.parametrize("device", DEVICES)
def test_an_out_of_range_index_sets_the_flag_and_reads_inside(device):
    x = _x((5, 3), F32)
    for op, attrs, i in [("index_select", {"dim": 0}, torch.tensor([0, 5])),
                         ("index", {}, torch.tensor([-6, 1])),
                         ("gather", {"dim": 0}, torch.tensor([[0, 9, 1]]))]:
        args = [x, i] if op != "gather" else [x, i]
        got, flag = _gather(op, attrs, args, device)
        assert flag == 1, op
        assert torch.isfinite(got).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [F32, F64, torch.float16, torch.bfloat16])
def test_argmax_argmin_first_extreme_and_nan(device, dtype):
    be = _backend(device)
    if be == "numba" and dtype not in (F32, F64):
        pytest.skip("the CPU kernel takes float32 and float64")
    x = torch.round(_x((6, 9, 700), F64, seed=2) * 2).to(dtype)   # ties everywhere, a row over one block
    x[1, 2, 650] = float("nan")
    x[1, 2, 660] = float("nan")
    x[2, :, 5] = float("-inf")
    x[3, 4, :] = float("-inf")
    x[4, 5, :] = float("inf")
    from truestar.ir import reference as R
    for op in ("argmax", "argmin"):
        for dim, keep in [(None, False), (None, True), (0, False), (1, True), (2, False), (-1, True)]:
            attrs = {"dim": dim, "keepdim": keep}
            want = R.IMPLS[op](attrs, [x.double()], None)
            got = GN.arg_extreme(op, attrs, x.to(device), None, be)
            assert got is not None
            assert torch.equal(got.cpu(), want), (op, dim, keep)
        xt = x.transpose(0, 2)
        want = R.IMPLS[op]({"dim": 1, "keepdim": False}, [xt.double()], None)
        assert torch.equal(GN.arg_extreme(op, {"dim": 1, "keepdim": False}, xt.to(device), None, be).cpu(), want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [F32, F64, torch.float16, torch.bfloat16])
def test_scatter_add_sums_in_index_order_and_repeats_bitwise(device, dtype):
    """Repeated indices are summed in index order: the result equals a plain loop in that order
    (float64 exactly; narrower types within one rounding per add) and is the same bits every run."""
    be = _backend(device)
    if be == "numba" and dtype not in (F32, F64):
        pytest.skip("the CPU kernel takes float32 and float64")
    for d, xs, ish in [(0, (6, 5), (40, 5)), (1, (3, 7, 4), (3, 30, 2)), (0, (9,), (25,))]:
        x = _x(xs, dtype, seed=d)
        i = torch.randint(-xs[d], xs[d], ish, generator=_g(d))
        src = _x(tuple(ish[k] + (1 if k != d else 0) for k in range(len(ish))), dtype, seed=d + 7)
        got = GN.scatter_add({"dim": d}, [t.to(device) for t in (x, i, src)], be)
        assert got is not None
        out, flag, _ = got
        assert int(flag.cpu()[0]) == 0
        want = x.double().clone()
        for pos in torch.cartesian_prod(*[torch.arange(n) for n in ish]).reshape(-1, len(ish)).tolist():
            tgt = list(pos)
            tgt[d] = int(i[tuple(pos)]) % xs[d]
            want[tuple(tgt)] += src[tuple(pos)].double()
        tol = {F64: 1e-12, F32: 1e-5, torch.float16: 2e-2, torch.bfloat16: 1e-1}[dtype]
        assert (out.cpu().double() - want).abs().max().item() <= tol * (want.abs().max().item() + 1)
        again = GN.scatter_add({"dim": d}, [t.to(device) for t in (x, i, src)], be)[0]
        assert torch.equal(out.view(torch.int16 if out.element_size() == 2 else out.dtype),
                           again.view(torch.int16 if out.element_size() == 2 else out.dtype))
    bad = GN.scatter_add({"dim": 0}, [t.to(device) for t in (_x((4,), dtype), torch.tensor([0, 4]), _x((2,), dtype))], be)
    assert int(bad[1].cpu()[0]) == 1


# ----------------------------------------------------------------------------- compiled programs


def _check(prog, inputs, device, tol=1e-5):
    cp = compile_program(prog, device, "float32")
    for xs in inputs:
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


def _impls(cp):
    return {st["ops"][0]: st.get("impl", "") for st in cp.report.library}


@pytest.mark.parametrize("device", DEVICES)
def test_compiled_reads_and_argmax_run_truestar_kernels(device):
    be = _backend(device)
    b = ir.Builder("reads")
    ids = b.input("ids", torch.int64, ("B", 5))
    y = b.input("y", torch.int64, ("B", 1))
    w = b.param("w", torch.randn(11, 8, generator=_g(0), dtype=F64))
    emb = b.op("index", w, ids)                                     # [B, 5, 8]
    logits = b.op("sum", emb, dims=[1], keepdim=False)               # [B, 8]
    lsm = b.op("log_softmax", logits, dim=1)
    pick = b.op("gather", lsm, y, dim=1)                             # the cross entropy class pick
    flat = b.input("flat", torch.int64, ("N",))
    sel = b.op("index_select", w, flat, dim=0)                       # [N, 8]
    am = b.op("argmax", logits, dim=1, keepdim=False)
    prog = b.build([pick, sel, am])

    def inputs(n, seed):
        return [torch.randint(-11, 11, (n, 5), generator=_g(seed)), torch.randint(0, 8, (n, 1), generator=_g(seed + 1)),
                torch.randint(-11, 11, (n + 3,), generator=_g(seed + 2))]
    cp = _check(prog, [inputs(2, 0), inputs(6, 3)], device)
    impls = _impls(cp)
    for op in ("index", "gather", "index_select", "argmax"):
        assert impls[op] == f"truestar {op} ({be})", impls


@pytest.mark.parametrize("device", DEVICES)
def test_compiled_adaptive_max_pool_and_index_padding_run_truestar_kernels(device):
    _backend(device)
    for i, (mod, shape) in enumerate([(torch.nn.AdaptiveMaxPool2d((3, 4)), (2, 3, 7, 9)),
                                      (torch.nn.ReflectionPad2d((2, 1, 1, 2)), (2, 3, 7, 9)),
                                      (torch.nn.ReplicationPad1d(3), (2, 3, 7)),
                                      (torch.nn.Upsample(scale_factor=2, mode="nearest"), (2, 3, 5, 4)),
                                      (torch.nn.Embedding(10, 4, padding_idx=1), None),
                                      (torch.nn.EmbeddingBag(10, 4, mode="max"), None)]):
        mod = mod.double()
        x = torch.randint(0, 10, (2, 5), generator=_g(i)) if shape is None else             torch.randn(*shape, generator=_g(i), dtype=F64)
        prog = ir.import_module(mod, (x,))
        bigger = torch.randint(0, 10, (2, 5), generator=_g(9)) if shape is None else x * 3
        xs = [[x if shape is None else x.float()], [bigger if shape is None else bigger.float()]]
        cp = _check(prog, xs, device)
        reads = {st.get("impl", "") for st in cp.report.library if st["ops"][0] in GN.GATHER_OPS + GN.ARG_OPS + GN.SCATTER_OPS}
        assert reads and all(i.startswith("truestar") for i in reads), (mod, reads)


@pytest.mark.parametrize("device", DEVICES)
def test_an_out_of_range_index_in_a_compiled_program_names_the_op(device):
    _backend(device)
    b = ir.Builder("emb")
    ids = b.input("ids", torch.int64, ("B",))
    w = b.param("w", torch.randn(4, 3, dtype=F64))
    prog = b.build([b.op("index", w, ids)])
    cp = compile_program(prog, device, "float32")
    cp([torch.tensor([0, 3]).to(device)])
    with pytest.raises(ir.IRError, match="index: an index is out of range"):
        cp([torch.tensor([1, 4]).to(device)])


@pytest.mark.parametrize("device", DEVICES)
def test_a_wrong_gather_kernel_is_refused_on_first_call(device, monkeypatch):
    """Tests can fail: a kernel that reads the next row is caught by the first call check."""
    _backend(device)
    orig = GN.Plan.index_tensors

    def shifted(self, idx):
        return [torch.clamp(t + 1, max=self.n[0] - 1) for t in orig(self, idx)]
    monkeypatch.setattr(GN.Plan, "index_tensors", shifted)
    GN._PLANS.clear()
    b = ir.Builder("emb")
    ids = b.input("ids", torch.int64, ("B",))
    w = b.param("w", torch.randn(6, 3, dtype=F64))
    prog = b.build([b.op("index", w, ids)])
    cp = compile_program(prog, device, "float32")
    with pytest.raises(Exception, match="(?i)index"):
        cp([torch.tensor([0, 2, 4]).to(device)])
