"""Op coverage tests for the IR (ops added 2026-11-24).

Each case is a small module exported with torch.export, imported in both graph forms (the default
export graph and the core ATen graph), run by the float64 reference interpreter and compared with
the same module run by PyTorch in float64. Every case has a symbolic batch and is checked at a
second batch size from the same import. Cases also assert that the new IR op really appears in the
program, so a case cannot pass by testing something else.
"""

import copy
import math
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

from truestar import ir
from truestar.ir import reference, shapes

warnings.filterwarnings("ignore", category=FutureWarning)

TOL = 1e-10


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300)).item()


def f64(model: nn.Module, *args):
    """The module run by PyTorch in float64 (integer and bool inputs stay as they are)."""
    with torch.no_grad():
        out = copy.deepcopy(model).double()(*[a.double() if a.is_floating_point() else a for a in args])
    return list(out) if isinstance(out, (tuple, list)) else [out]


class _Fn(nn.Module):
    def __init__(self, fn, **parts):
        super().__init__()
        self.fn = fn
        for k, v in parts.items():
            if isinstance(v, nn.Module):
                setattr(self, k, v)
            elif isinstance(v, nn.Parameter):
                self.register_parameter(k, v)
            else:
                self.register_buffer(k, v)


class _Fn1(_Fn):
    def forward(self, x):
        return self.fn(self, x)


class _Fn2(_Fn):
    def forward(self, x, y):
        return self.fn(self, x, y)


class _Fn3(_Fn):
    def forward(self, x, y, z):
        return self.fn(self, x, y, z)


def Fn(fn, **parts):
    """A module whose forward is ``fn(self, *inputs)``; submodules, params and buffers are keywords.
    (torch.export needs a fixed number of named inputs, so the arity picks the class.)"""
    n = fn.__code__.co_argcount - 1
    return {1: _Fn1, 2: _Fn2, 3: _Fn3}[n](fn, **parts)


def _g(seed):
    return torch.Generator().manual_seed(seed)


def randn(*shape, seed=0):
    return torch.randn(*shape, generator=_g(seed))


def randomize(model: nn.Module, seed: int = 0) -> nn.Module:
    """Non trivial affine parameters for every norm, so a wrong formula cannot hide behind 1 and 0."""
    with torch.no_grad():
        for m in model.modules():
            for name in ("weight", "bias"):
                p = getattr(m, name, None)
                if isinstance(p, torch.Tensor) and isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.RMSNorm,
                                                                  nn.InstanceNorm2d)):
                    p.copy_((1.0 if name == "weight" else 0.0) + 0.3 * torch.randn(p.shape, generator=_g(seed)))
    return model.eval()


def check(model, args, dyn=None, other=(), expect=(), graphs=(False, True)):
    """Import in each graph form, assert the expected IR ops appear (export graph), and compare with
    PyTorch float64 at the example inputs and at every extra input set in ``other``."""
    model = model.eval()
    progs = []
    for decompose in graphs:
        prog = ir.import_module(model, args, dynamic_shapes=dyn, decompose=decompose)
        if not decompose:
            ops = prog.op_counts()
            missing = [e for e in expect if e not in ops]
            assert not missing, f"expected IR ops {missing} in the program, got {sorted(ops)}"
        for a in (args,) + tuple(other):
            got = ir.run(prog, *a)
            want = f64(model, *a)
            assert len(got) == len(want)
            for g, w in zip(got, want):
                assert tuple(g.shape) == tuple(w.shape), f"decompose={decompose}: {g.shape} vs {w.shape}"
                if w.is_floating_point():
                    err = rel_err(g, w)
                    assert err < TOL, f"decompose={decompose}, batch {a[0].shape[0]}: relative error {err:.3e}"
                else:
                    assert g.dtype == w.dtype and torch.equal(g, w), f"decompose={decompose}: integer result differs"
        progs.append(prog)
    return progs


B = Dim("B", min=1, max=64)
T = Dim("T", min=2, max=64)


def batch_case(fn, shape, expect, parts=None, seed=0, more=None):
    """A case with one float input whose dim 0 is the symbolic batch, checked at batch 2 and 5."""
    model = Fn(fn, **(parts or {}))
    x2 = randn(2, *shape, seed=seed)
    x5 = randn(5, *shape, seed=seed + 1)
    if more is not None:
        x2, x5 = more(x2), more(x5)
    return model, (x2,), ({0: B},), [(x5,)], expect


# ----------------------------------------------------------------------------- cases

CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


@case("avg_pool2d_pad_count_include")
def _():
    return batch_case(lambda m, x: m.p(x), (3, 11, 9), ["avg_pool2d"], {"p": nn.AvgPool2d(3, 1, 1)})


@case("avg_pool2d_ceil_exclude_pad")
def _():
    return batch_case(lambda m, x: m.p(x), (3, 10, 9), ["avg_pool2d"],
                      {"p": nn.AvgPool2d(3, 2, 1, ceil_mode=True, count_include_pad=False)})


@case("avg_pool2d_ceil_include_pad")
def _():
    return batch_case(lambda m, x: m.p(x), (3, 10, 8), ["avg_pool2d"], {"p": nn.AvgPool2d(3, 2, 1, ceil_mode=True)})


@case("avg_pool2d_divisor_override")
def _():
    return batch_case(lambda m, x: (m.p(x), F.avg_pool2d(x, (2, 3), (1, 2))), (3, 7, 9), ["avg_pool2d"],
                      {"p": nn.AvgPool2d(2, divisor_override=3)})


@case("adaptive_pools_uneven_bins")
def _():
    def fn(m, x):
        return (F.adaptive_avg_pool2d(x, (4, 5)), F.adaptive_max_pool2d(x, (4, 5)),
                F.adaptive_max_pool2d(x, 3), F.adaptive_avg_pool1d(x.flatten(2), 7))
    return batch_case(fn, (3, 11, 9), ["adaptive_avg_pool2d", "adaptive_max_pool2d"])


@case("group_norm")
def _():
    parts = {"a": randomize(nn.GroupNorm(3, 6)), "b": nn.GroupNorm(2, 6, affine=False)}
    return batch_case(lambda m, x: m.a(x) + m.b(x), (6, 5, 4), ["group_norm"], parts)


@case("group_norm_3d_input")
def _():
    return batch_case(lambda m, x: m.a(x), (6, 7), ["group_norm"], {"a": randomize(nn.GroupNorm(2, 6))})


@case("instance_norm")
def _():
    parts = {"a": randomize(nn.InstanceNorm2d(6, affine=True)), "b": nn.InstanceNorm2d(6)}
    # the core ATen graph turns instance norm into batch norm with input statistics on [1, N*C, H, W]
    return batch_case(lambda m, x: m.a(x) * m.b(x), (6, 5, 4), ["group_norm"], parts)


@case("layer_norm_variants")
def _():
    parts = {"a": nn.LayerNorm(8, elementwise_affine=False), "b": randomize(nn.LayerNorm(8, bias=False)),
             "c": randomize(nn.LayerNorm([5, 8]))}
    return batch_case(lambda m, x: m.a(x) + m.b(x) * m.c(x), (5, 8), ["layer_norm"], parts)


@case("rms_norm_module")
def _():
    # explicit eps: with eps=None PyTorch uses the epsilon of the run dtype, which differs between
    # the float32 model and its float64 copy (test_rms_norm_default_eps covers eps=None)
    parts = {"a": randomize(nn.RMSNorm(8, eps=1e-6)), "b": randomize(nn.RMSNorm([5, 8], eps=1e-5)),
             "c": nn.RMSNorm(8, eps=1e-4, elementwise_affine=False)}
    return batch_case(lambda m, x: m.a(x) + m.b(x) - m.c(x), (5, 8), ["rms_norm"], parts)


@case("rms_norm_composite")
def _():
    w = nn.Parameter(1 + 0.3 * randn(8, seed=4))

    def fn(m, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * m.w
    return batch_case(fn, (5, 8), ["pow", "rsqrt"], {"w": w})


@case("gelu_tanh")
def _():
    return batch_case(lambda m, x: F.gelu(x, approximate="tanh") - F.gelu(x), (5, 8), ["gelu"])


@case("embedding_tied")
def _():
    emb = nn.Embedding(20, 8, padding_idx=0)
    model = Fn(lambda m, ids: F.linear(m.e(ids), m.e.weight), e=emb)
    ids2 = torch.randint(0, 20, (2, 5), generator=_g(0))
    ids5 = torch.randint(0, 20, (5, 5), generator=_g(1))
    return model, (ids2,), ({0: B},), [(ids5,)], ["index"]


@case("index_select_gather")
def _():
    def fn(m, x):
        a = x.index_select(1, m.i)[:, :, :4] + torch.index_select(x, 2, m.j)[:, :, :4]
        k = ((x.abs() * 7).to(torch.int64) % x.shape[2])[..., :3]  # data dependent index, fixed shape
        return a, torch.gather(x, 2, k), x.gather(1, k[:, :2] % x.shape[1])
    parts = {"i": torch.tensor([3, 0, 0, 1]), "j": torch.tensor([4, 2, 1, 0, 3])}
    return batch_case(fn, (4, 5), ["index_select", "gather", "cast", "remainder"], parts)


@case("index_tensor_advanced")
def _():
    def fn(m, x):
        table = m.table[m.idx]                       # buffer table indexed by a 2D buffer index
        adjacent = x[:, m.i]                         # [B, 3, 5, 6]
        split = x[:, m.i, :, m.j]                    # non adjacent indices: broadcast dims go first
        neg = x[:, :, m.neg]                         # negative indices wrap
        return table, adjacent, split, neg
    parts = {"table": randn(7, 3, seed=5), "idx": torch.tensor([[0, 6], [3, 3], [5, 1]]),
             "i": torch.tensor([3, 0, 1]), "j": torch.tensor([5, 2, 0]), "neg": torch.tensor([-1, 0, -5])}
    return batch_case(fn, (4, 5, 6), ["index"], parts)


@case("attention_masks")
def _():
    def fn(m, q, k, v):
        L = q.shape[-2]
        causal_bool = torch.ones(L, L, dtype=torch.bool).tril()
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_bool)
        b = F.scaled_dot_product_attention(q, k, v, attn_mask=m.bias)
        c = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        pad = torch.where(m.keep, 0.0, float("-inf"))[None, :]
        d = F.scaled_dot_product_attention(q, k, v, attn_mask=pad, scale=0.25)
        return a + b + c + d
    parts = {"bias": randn(1, 1, 6, 6, seed=7), "keep": torch.tensor([True, True, False, True, True, False])}
    q2, k2, v2 = (randn(2, 2, 6, 4, seed=s) for s in (1, 2, 3))
    q5, k5, v5 = (randn(5, 2, 6, 4, seed=s) for s in (4, 5, 6))
    dyn = ({0: B}, {0: B}, {0: B})
    return Fn(fn, **parts), (q2, k2, v2), dyn, [(q5, k5, v5)], ["attention", "tril"]


@case("where_masked_fill")
def _():
    def fn(m, x):
        a = x.masked_fill(x > 0.5, -1e4)
        b = x.masked_fill(x < -0.5, m.v)
        c = torch.where(x > 0, x, 0.0) + torch.where(x > 0, 1.0, x) + torch.where(x > 0, 2.0, -3.0)
        return a + b + c
    return batch_case(fn, (5, 8), ["where"], {"v": torch.tensor(7.0)})


@case("arange_symbolic_seq")
def _():
    def fn(m, x):
        pos = torch.arange(x.shape[1])                       # symbolic length T
        even = torch.arange(0, x.shape[1], 2)                # ceil(T / 2)
        return x + pos.float()[None, :, None], even.float() * 2
    model = Fn(fn)
    x = randn(2, 7, 3)
    dyn = ({0: B, 1: Dim("T", min=3, max=64)},)  # torch.export needs T >= 3 for arange(0, T, 2)
    return model, (x,), dyn, [(randn(5, 4, 3, seed=1),), (randn(3, 9, 3, seed=2),)], ["arange", "cast"]


@case("cumsum")
def _():
    return batch_case(lambda m, x: (torch.cumsum(x, 1), torch.cumsum((x > 0).long(), 2), x.cumsum(-1)),
                      (5, 8), ["cumsum"])


@case("triu_tril_masks")
def _():
    def fn(m, x):
        L = x.shape[-1]
        mask = torch.full((L, L), float("-inf")).triu(1)
        return x.triu(1) + x.tril(-1) + torch.tril(x, 2) + mask.exp()
    return batch_case(fn, (6, 6), ["triu", "tril"])


@case("shape_ops")
def _():
    def fn(m, x):
        a = x.unsqueeze(1).squeeze(1).unsqueeze(-1).expand(-1, -1, -1, 3)
        b = torch.swapaxes(x, 1, 2).movedim(1, 2)
        c = torch.stack(list(x.unbind(1)), dim=2) + x.transpose(1, 2).view_as(x.transpose(1, 2))
        d = m.w.expand_as(x)
        return a.sum(-1) + b + c.transpose(1, 2) + d
    return batch_case(fn, (4, 5), ["expand", "permute", "cat", "select"], {"w": randn(5, seed=3)})


@case("split_chunk")
def _():
    def fn(m, x):
        a, b, c = x.chunk(3, dim=2)                 # 7 -> 3, 3, 1
        d = x.split(2, dim=1)                       # 4 -> 2, 2
        e = torch.split(x, [1, 2, 1], dim=1)
        return a + b, c * 2, d[0] - d[1], e[0] + e[2] + e[1].sum(1, keepdim=True)
    return batch_case(fn, (4, 7), ["slice"])


@case("roll_pad")
def _():
    def fn(m, x):
        a = torch.roll(x, (1, -2), (1, 2))
        b = F.pad(x, (1, 2, 0, 1), value=0.5)
        c = F.pad(x, (-1, 1, 2, -2))                 # negative pads crop
        return a, b, c, torch.roll(x, 3, 2)
    return batch_case(fn, (4, 6), ["pad", "cat"])


@case("roll_flat")
def _():
    model = Fn(lambda m, x: torch.roll(x, 5) + torch.roll(x, -2))
    return model, (randn(3, 4),), None, [], ["cat"]


@case("einsum")
def _():
    def fn(m, x, y):
        return (torch.einsum("bij,bjk->bik", x, y),
                torch.einsum("b i j, b k j -> b i k", x, x),
                torch.einsum("bij->bj", x),
                torch.einsum("bij->bji", x),
                torch.einsum("bij,j->bi", x, m.v),
                torch.einsum("bi,bj->bij", x[:, 0], x[:, 1]),
                torch.einsum("ij,bjk,kl->bil", m.a, y, m.c),
                torch.einsum("bij,bij->b", x, x),
                torch.einsum("bij,bjk", x, y))       # implicit output: sorted free indices
    parts = {"v": randn(5, seed=1), "a": randn(3, 5, seed=2), "c": randn(6, 2, seed=3)}
    x2, y2 = randn(2, 4, 5, seed=4), randn(2, 5, 6, seed=5)
    x5, y5 = randn(5, 4, 5, seed=6), randn(5, 5, 6, seed=7)
    return Fn(fn, **parts), (x2, y2), ({0: B}, {0: B}), [(x5, y5)], ["matmul"]


@case("rotary_positions")
def _():
    def fn(m, x):
        T_, D = x.shape[-2], x.shape[-1]
        inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=x.dtype) / D))  # x.dtype: float64 in the reference run
        f = torch.outer(torch.arange(T_, dtype=x.dtype), inv)
        e = torch.cat((f, f), dim=-1)
        x1, x2 = x.chunk(2, dim=-1)
        return x * e.cos() + torch.cat((-x2, x1), dim=-1) * e.sin()
    model = Fn(fn)
    # the frequencies depend on no input and no weight: their pow is computed once at import
    # (ir/importer_vision.py); pow of an input is covered by the elementwise case below
    return model, (randn(2, 3, 7, 8),), ({0: B, 2: T},), [(randn(5, 3, 4, 8, seed=1),)], ["cos", "sin"]


@case("elementwise_math")
def _():
    def fn(m, x):
        p = x.abs() + 0.5
        return (p.pow(x), torch.pow(2.0, x), p.log(), p.reciprocal(), x.abs(), torch.erf(x),
                torch.fmod(x * 5, 1.5), torch.remainder(x * 5, 1.5), torch.fmod(m.i, 3), m.i % 3,
                torch.maximum(x, x * 0.5), torch.minimum(x, m.w),
                x.clamp_min(-0.2), x.clamp_max(0.3))
    parts = {"i": torch.tensor([-7, -3, 0, 4, 8]), "w": randn(8, seed=9)}
    return batch_case(fn, (5, 8), ["pow", "log", "reciprocal", "abs", "erf", "fmod", "remainder", "maximum",
                                   "minimum", "clamp"], parts)


@case("vector_norm")
def _():
    def fn(m, x):
        return (torch.linalg.vector_norm(x, 2, -1, True), torch.linalg.vector_norm(x, 1, (1, 2)),
                torch.linalg.vector_norm(x, float("inf"), -1), torch.linalg.vector_norm(x, 3.0, 1),
                F.normalize(x, dim=-1), x.norm(dim=-1))
    return batch_case(fn, (5, 8), ["vector_norm"])


@case("cast")
def _():
    def fn(m, x):
        y = x * 3
        return y.to(torch.int64), y.bool(), (x > 0).float() + x, m.i.type_as(x) * x, (x > 0).to(torch.int32)
    return batch_case(fn, (5, 8), ["cast"], {"i": torch.arange(8)})


@case("creation")
def _():
    def fn(m, x):
        z = x.new_zeros((x.shape[0], 8)) + x.new_ones(8) + x.new_full((1, 8), 2.5)
        return x + z.unsqueeze(1).sum(1, keepdim=True) + torch.zeros_like(x) + torch.ones_like(x) * torch.ones(8)
    return batch_case(fn, (5, 8), ["full", "full_like"])


@case("fill_through_slices")
def _():
    def fn(m, x):
        mask = x.new_zeros((6, 6))                  # Swin's shifted window mask pattern
        mask[0:2, 0:3] = 1.0
        mask[2:, 3:] = 2.0
        mask[:, 5] = 3.0
        mask[4] = mask[4] + 0.5
        return x + mask
    return batch_case(fn, (6, 6), ["slice_scatter", "select_scatter"])


@case("zero_and_copy_through_slices")
def _():
    def fn(m, x):
        b = m.bias.clone()
        b[2:4].zero_()                               # Swin V2's qkv bias pattern
        y = F.linear(x, m.w, b)
        z = y.clone()
        z[:, :, 0] = y[:, :, 1] * 2
        z[:, 1:3] += 1.0
        return y, z
    parts = {"w": nn.Parameter(randn(6, 8, seed=2)), "bias": nn.Parameter(randn(6, seed=3))}
    return batch_case(fn, (5, 8), ["slice_scatter", "select_scatter"], parts)


@pytest.mark.parametrize("name", list(CASES))
def test_op_matches_pytorch_float64(name):
    torch.manual_seed(0)
    model, args, dyn, other, expect = CASES[name]()
    progs = check(model, args, dyn, other, expect)
    if dyn is not None:
        assert "B" in str(progs[0].inputs[0].type) or "T" in str(progs[0].inputs[0].type)


def test_new_ops_have_symbolic_shape_rules():
    """The rules keep symbolic dims (batch B, sequence T) instead of needing numbers."""
    b, t = ir.symbol("B"), ir.symbol("T")
    f = lambda *s: ir.TensorType(torch.float32, s)  # noqa: E731
    i64 = lambda *s: ir.TensorType(torch.int64, s)  # noqa: E731
    assert shapes.infer("avg_pool2d", [f(b, 3, 11, 9)], dict(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]))\
        .shape == (b, 3, 6, 5)
    assert shapes.infer("index", [f(20, 8), i64(b, t)], {}).shape == (b, t, 8)
    assert shapes.infer("index", [f(b, 4, 5, 6), None, i64(3), None, i64(3)], {}).shape == (3, b, 5)
    assert shapes.infer("gather", [f(b, t, 5), i64(b, t, 2)], dict(dim=2)).shape == (b, t, 2)
    assert shapes.infer("index_select", [f(b, t), i64(4)], dict(dim=1)).shape == (b, 4)
    assert shapes.infer("pad", [f(b, t, 4)], dict(pads=[1, 2, 0, 3])).shape == (b, t + 3, 7)
    assert shapes.infer("arange", [], dict(start=0, end=t, step=2, dtype=torch.int64)).shape == (sympy_floor(t + 1, 2),)
    assert shapes.infer("cumsum", [i64(b, t)], dict(dim=1)).dtype == torch.int64
    assert shapes.infer("tril", [f(b, t, t)], dict(diagonal=0)).shape == (b, t, t)
    assert shapes.infer("rms_norm", [f(b, t, 8), f(8)], dict(normalized_shape=[8], eps=1e-6)).shape == (b, t, 8)
    assert shapes.infer("group_norm", [f(b, 6, t), None, None], dict(num_groups=3, eps=1e-5)).shape == (b, 6, t)
    assert shapes.infer("vector_norm", [f(b, t, 8)], dict(ord=2.0, dims=[-1], keepdim=True)).shape == (b, t, 1)
    assert shapes.infer("slice_scatter", [f(b, 6), f(b, 2)], dict(dim=1, start=1, end=3, step=1)).shape == (b, 6)
    assert shapes.infer("cast", [f(b, t)], dict(dtype=torch.bool)).dtype == torch.bool
    with pytest.raises(ir.ShapeError, match="gather"):
        shapes.infer("gather", [f(b, 5), i64(b, 5, 1)], dict(dim=1))
    with pytest.raises(ir.ShapeError, match="slice_scatter"):
        shapes.infer("slice_scatter", [f(b, 6), f(b, 3)], dict(dim=1, start=1, end=3, step=1))
    with pytest.raises(ir.UnsupportedOpError, match="boolean mask"):
        shapes.infer("index", [f(b, 5), None, ir.TensorType(torch.bool, (5,))], {})


def sympy_floor(t, k):
    import sympy
    return sympy.floor(t / k)


# ----------------------------------------------------------------------------- loud refusals


def _refused(fn, args, dyn=None, match=None):
    with pytest.raises(ir.UnsupportedOpError) as e:
        ir.import_module(Fn(fn).eval(), args, dynamic_shapes=dyn)
    if match:
        for m in match:
            assert m in str(e.value), str(e.value)
    return str(e.value)


def test_view_read_after_its_storage_changed_is_refused():
    def fn(m, x):
        mask = x.new_zeros((4, 4))
        early = mask[0:2]
        mask[1:3] = 1.0           # changes rows 1 and 2, which ``early`` also sees in PyTorch
        return x + early.sum()
    msg = _refused(fn, (randn(3, 4, 4),), match=["aten.fill_", "reads", "afterwards"])
    assert "test_ir_ops.py" in msg


def test_einsum_diagonal_is_refused_naming_the_op():
    _refused(lambda m, x: torch.einsum("bii->bi", x), (randn(2, 3, 3),), match=["aten.einsum.default", "repeated"])


def test_roll_along_symbolic_dim_is_refused():
    _refused(lambda m, x: torch.roll(x, 1, 0), (randn(4, 3),), ({0: B},), match=["aten.roll.default", "symbolic"])


def test_rms_norm_default_eps_is_the_float32_epsilon():
    prog = ir.import_module(nn.RMSNorm(8).eval(), (randn(2, 8),))
    (node,) = [n for n in prog.nodes if n.op == "rms_norm"]
    assert node.attrs["eps"] == torch.finfo(torch.float32).eps


def test_reflect_pad_imports_and_over_a_symbolic_dim_is_refused_naming_the_mode():
    # reflect, replicate and circular padding import since the op coverage work (2026-11-24,
    # tests/test_op_coverage.py); a symbolic padded dim is still refused, naming the mode
    x = randn(2, 3, 5)
    m = Fn(lambda m, x: F.pad(x, (1, 1), mode="reflect")).eval()
    assert torch.equal(ir.run(ir.import_module(m, (x,)), x)[0], F.pad(x, (1, 1), mode="reflect"))
    L = torch.export.Dim("L", min=3, max=16)
    with pytest.raises(ir.UnsupportedOpError) as e:
        ir.import_module(m, (x,), dynamic_shapes=({2: L},))
    assert "reflect" in str(e.value)


def test_strided_view_that_is_not_a_reshape_is_refused():
    _refused(lambda m, x: torch.as_strided(x, (2, 2), (1, 2)) * 1.0, (randn(4, 4),), match=["as_strided"])


# ----------------------------------------------------------------------------- tests that prove the tests work


@pytest.mark.parametrize("op, broken, case_name", [
    ("group_norm", "unbiased", "group_norm"),
    ("avg_pool2d", "always_count_pad", "avg_pool2d_ceil_exclude_pad"),
    ("index", "off_by_one", "embedding_tied"),
])
def test_broken_new_reference_op_is_caught(monkeypatch, op, broken, case_name):
    real = reference.IMPLS[op]

    def wrong(attrs, args, shape):
        if broken == "unbiased":  # variance with n - 1 instead of n
            x = args[0]
            n = x.reshape(x.shape[0], attrs["num_groups"], -1).shape[-1]
            return real({**attrs, "eps": attrs["eps"]}, args, shape) * math.sqrt((n - 1) / n)
        if broken == "always_count_pad":
            return real({**attrs, "count_include_pad": True}, args, shape)
        return real(attrs, [args[0].roll(1, 0)] + list(args[1:]), shape)

    monkeypatch.setitem(reference.IMPLS, op, wrong)
    with pytest.raises(AssertionError, match="relative error"):
        model, args, dyn, other, expect = CASES[case_name]()
        check(model, args, dyn, other, expect, graphs=(False,))


# ----------------------------------------------------------------------------- a decoder only transformer


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.2 * torch.randn(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class DecoderBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 4 * d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.heads, -1).transpose(1, 2) for z in (q, k, v))
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        y = self.norm2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))  # SwiGLU


class TinyDecoder(nn.Module):
    """Embedding, rotary positions, causal attention, RMSNorm, SwiGLU, and an lm head tied to the embedding."""

    def __init__(self, vocab=50, d=32, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([DecoderBlock(d, heads) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.embed.weight
        self.head_dim = d // heads

    def forward(self, ids):
        t = ids.shape[1]
        dtype = self.embed.weight.dtype  # float32 when exported, float64 in the reference run
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=dtype) / self.head_dim))
        freqs = torch.outer(torch.arange(t, dtype=dtype), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.lm_head(self.norm(x))


@pytest.mark.parametrize("decompose", [False, True], ids=["export_ir", "core_aten"])
def test_decoder_transformer_matches_pytorch_float64(decompose):
    torch.manual_seed(0)
    model = TinyDecoder().eval()
    ids = torch.randint(0, 50, (2, 8), generator=_g(0))
    prog = ir.import_module(model, (ids,), dynamic_shapes=({0: B, 1: T},), decompose=decompose)
    assert str(prog.inputs[0].type) == "i64[B, T]"
    assert str(prog.outputs[0].type) == "f32[B, T, 50]"
    names = [p.name for p in prog.params]
    assert sum("embed" in n or "lm_head" in n for n in names) == 1, names  # the tied weight is stored once
    for shape in ((2, 8), (3, 5), (1, 13)):  # two more sequence lengths and batch sizes from one import
        x = torch.randint(0, 50, shape, generator=_g(sum(shape)))
        got = ir.run(prog, x)[0]
        want = f64(model, x)[0]
        assert got.shape == want.shape
        err = rel_err(got, want)
        assert err < TOL, f"{shape}: relative error {err:.3e}"
