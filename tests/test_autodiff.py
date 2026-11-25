"""Tests for reverse mode autodiff as a compiler pass (truestar.ir.autodiff).

Three independent checks, from the smallest unit up:

1. every gradient rule against finite differences of the float64 reference interpreter;
2. whole imported models against torch.autograd in float64;
3. a training loop against torch.optim.SGD, step for step.

Plus structure tests (fan out, broadcasting, dead code, loud failures) and tests that prove the
checks can fail.
"""

import copy
import math
import warnings

import pytest
import torch
from torch import nn
from torch.export import Dim

from truestar import ir
from truestar.ir import autodiff, grad_rules, reference, shapes
from truestar.ir.autodiff import grad

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

F64 = torch.float64

# Finite difference bar.
#
# The derivative of phi(x) = <v, f(x)> along one input entry is estimated with the fourth order
# central difference (phi(x-2h) - 8 phi(x-h) + 8 phi(x+h) - phi(x+2h)) / 12h, h = 1e-3, in float64.
# Its two error terms are
#   truncation: h^4 / 30 * |phi^(5)| = 3e-14 * |phi^(5)|,
#   rounding:   about 1.5 * eps * |phi| / h = 3e-13 * |phi|, with eps = 2.2e-16.
# The test programs take O(1) inputs and produce O(1) to O(100) values, with fifth derivatives up to
# about 1e3 for the sharpest ops (division, rsqrt and the norms, softmax), so the estimate is good
# to 1e-12 up to 1e-10 relative to the largest gradient entry. Measured on 2026-11-24: the worst
# case over all rule checks is 1.6e-10 (see docs/AUTODIFF.md). The bar is 1e-8, about 60 times above
# the worst measured value. A wrong rule is off by a constant factor, a sign or a missing term, an
# error of order 1e-2 to 1, so it cannot hide below this bar.
FD_STEP = 1e-3
FD_TOL = 1e-8

# Bar for whole models against torch.autograd, both in float64. The two sides run the same
# mathematics with different kernels and summation orders, so they agree to rounding: measured
# worst case 8.6e-15 (mobilenet_v3_small). The stated target is 1e-8; 1e-10 is asserted, the same bar the
# forward tests in test_ir.py use.
AUTOGRAD_TOL = 1e-10


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = max(a.abs().max().item(), b.abs().max().item()) if a.numel() else 0.0
    if scale == 0.0:
        return 0.0
    return (a - b).abs().max().item() / scale


def float_inputs(prog):
    return [v.name for v in prog.inputs if v.type.dtype.is_floating_point]


def fd_check(prog, inputs, gprog=None, seed=0):
    """Worst relative error between the grad program's vector Jacobian products and finite
    differences of the reference interpreter, over every float input."""
    wrt = float_inputs(prog)
    gprog = gprog or grad(prog, wrt, seed="input")
    outs = ir.run(prog, *inputs)
    gen = torch.Generator().manual_seed(seed)
    vs = [torch.randn(o.shape, generator=gen, dtype=F64) for o in outs]
    res = ir.run(gprog, *inputs, *vs)
    assert len(res) == len(outs) + len(wrt)
    for a, b in zip(res, outs):  # the transformed program still returns the original outputs
        assert rel_err(a, b) < 1e-14
    grads = res[len(outs):]

    def phi(args):
        return sum((o * v).sum().item() for o, v in zip(ir.run(prog, *args), vs))

    worst = 0.0
    names = [v.name for v in prog.inputs]
    for name, g in zip(wrt, grads):
        k = names.index(name)
        base = inputs[k]
        assert tuple(g.shape) == tuple(base.shape)
        fd = torch.zeros_like(base)
        flat = fd.reshape(-1)
        for i in range(base.numel()):
            vals = []
            for mult in (-2, -1, 1, 2):
                x = base.clone()
                x.reshape(-1)[i] += mult * FD_STEP
                args = list(inputs)
                args[k] = x
                vals.append(phi(args))
            flat[i] = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * FD_STEP)
        worst = max(worst, rel_err(g, fd))
    return worst


def away_from(x: torch.Tensor, kinks, margin=0.05) -> torch.Tensor:
    """Move entries off the points where a piecewise op is not differentiable. Finite differences
    straddling a kink are meaningless, so inputs keep a margin of 25 times the largest step."""
    for k in kinks:
        near = (x - k).abs() < margin
        x = torch.where(near, torch.full_like(x, k + 2 * margin), x)
    return x


def distinct(shape, gen) -> torch.Tensor:
    """Values that all differ by at least 0.05, so every max pool window has a clear winner."""
    n = math.prod(shape)
    return (torch.randperm(n, generator=gen).to(F64) * 0.05 - n * 0.025).reshape(shape)


# ----------------------------------------------------------------------------- rule cases

CASES = {}


def case(name, shapes_, build, prep=None, dtypes=None):
    CASES[name] = (shapes_, build, prep, dtypes)


def make(name, seed=0):
    shapes_, build, prep, dtypes = CASES[name]
    b = ir.Builder(name)
    dtypes = dtypes or [F64] * len(shapes_)
    vals = [b.input(f"x{i}", dt, s) for i, (s, dt) in enumerate(zip(shapes_, dtypes))]
    outs = build(b, *vals)
    prog = b.build(outs if isinstance(outs, list) else [outs])
    gen = torch.Generator().manual_seed(seed)
    inputs = []
    for s, dt in zip(shapes_, dtypes):
        if dt == torch.bool:
            inputs.append(torch.rand(s, generator=gen) < 0.6)
        else:
            inputs.append(torch.randn(s, generator=gen, dtype=F64))
    if prep is not None:
        inputs = prep(inputs, gen)
    return prog, inputs


def positive(i=0):
    def prep(inputs, gen):
        inputs[i] = inputs[i].abs() + 0.5
        return inputs
    return prep


def kinks(points, scale=1.0, i=0):
    def prep(inputs, gen):
        inputs[i] = away_from(inputs[i] * scale, points)
        return inputs
    return prep


# elementwise arithmetic, with broadcasting on both sides and a 0 dim operand
case("add_broadcast", [(3, 1, 4), (2, 1)], lambda b, x, y: b.op("add", x, y))
case("sub_broadcast", [(2, 4), (3, 1, 4)], lambda b, x, y: b.op("sub", x, y))
case("mul_broadcast", [(3, 1, 4), (2, 4)], lambda b, x, y: b.op("mul", x, y))
case("mul_scalar", [(3, 4), ()], lambda b, x, y: b.op("mul", x, y))
case("div_broadcast", [(3, 1, 4), (2, 1)], lambda b, x, y: b.op("div", x, y), positive(1))
case("neg", [(3, 4)], lambda b, x: b.op("neg", x))
case("exp", [(3, 4)], lambda b, x: b.op("exp", x))
case("sqrt", [(3, 4)], lambda b, x: b.op("sqrt", x), positive())
case("rsqrt", [(3, 4)], lambda b, x: b.op("rsqrt", x), positive())
case("tanh", [(3, 4)], lambda b, x: b.op("tanh", x))
case("sigmoid", [(3, 4)], lambda b, x: b.op("sigmoid", x))
case("erf", [(3, 4)], lambda b, x: b.op("erf", x))
case("expm1", [(3, 4)], lambda b, x: b.op("expm1", x))
case("relu", [(4, 5)], lambda b, x: b.op("relu", x), kinks([0.0]))
case("silu", [(3, 4)], lambda b, x: b.op("silu", x))
case("gelu_exact", [(3, 4)], lambda b, x: b.op("gelu", x, approximate="none"))
case("gelu_tanh", [(3, 4)], lambda b, x: b.op("gelu", x, approximate="tanh"))
case("hardsigmoid", [(5, 6)], lambda b, x: b.op("hardsigmoid", x), kinks([-3.0, 3.0], scale=3.0))
case("hardswish", [(5, 6)], lambda b, x: b.op("hardswish", x), kinks([-3.0, 3.0], scale=3.0))
case("clamp_both", [(5, 6)], lambda b, x: b.op("clamp", x, min=-0.5, max=0.7), kinks([-0.5, 0.7]))
case("clamp_min", [(4, 5)], lambda b, x: b.op("clamp", x, min=0.2, max=None), kinks([0.2]))
case("clamp_max", [(4, 5)], lambda b, x: b.op("clamp", x, min=None, max=0.0), kinks([0.0]))
case("clamp_strict", [(5, 6)], lambda b, x: b.op("clamp", x, min=-0.5, max=0.7, strict=True), kinks([-0.5, 0.7]))
case("where_broadcast", [(3, 4), (3, 4), (4,)], lambda b, c, x, y: b.op("where", c, x, y),
     dtypes=[torch.bool, F64, F64])

# reductions
case("sum_all", [(3, 4)], lambda b, x: b.op("sum", x, dims=None, keepdim=False))
case("sum_dim", [(3, 4, 2)], lambda b, x: b.op("sum", x, dims=[1], keepdim=False))
case("sum_keepdim", [(3, 4, 2)], lambda b, x: b.op("sum", x, dims=[0, -1], keepdim=True))
case("mean_all", [(3, 4)], lambda b, x: b.op("mean", x, dims=None, keepdim=False))
case("mean_dims", [(3, 4, 2)], lambda b, x: b.op("mean", x, dims=[-2, 0], keepdim=False))
case("mean_keepdim", [(3, 4, 2)], lambda b, x: b.op("mean", x, dims=[2], keepdim=True))
case("softmax_last", [(3, 5)], lambda b, x: b.op("softmax", x, dim=-1))
case("softmax_mid", [(2, 4, 3)], lambda b, x: b.op("softmax", x, dim=1))

# linear algebra
case("matmul", [(2, 3, 4), (2, 4, 5)], lambda b, x, y: b.op("matmul", x, y))
case("matmul_broadcast_left", [(3, 4), (2, 4, 5)], lambda b, x, y: b.op("matmul", x, y))
case("matmul_broadcast_both", [(2, 1, 3, 4), (3, 4, 2)], lambda b, x, y: b.op("matmul", x, y))
case("linear", [(2, 3, 5), (4, 5), (4,)], lambda b, x, w, c: b.op("linear", x, w, c))
case("linear_no_bias", [(3, 5), (4, 5)], lambda b, x, w: b.op("linear", x, w, None))
case("linear_rank1", [(5,), (4, 5), (4,)], lambda b, x, w, c: b.op("linear", x, w, c))


def conv(stride=1, padding=0, dilation=1, groups=1, bias=True):
    def build(b, x, w, *c):
        return b.op("conv2d", x, w, c[0] if bias else None, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)
    return build


case("conv2d_basic", [(2, 3, 6, 5), (4, 3, 3, 3), (4,)], conv(padding=1))
# stride 2 with one leftover row and column the forward conv never reads
case("conv2d_stride", [(2, 2, 8, 8), (3, 2, 3, 3), (3,)], conv(stride=2, padding=1))
case("conv2d_stride3_leftover", [(1, 2, 9, 8), (2, 2, 2, 3)], conv(stride=3, bias=False))
case("conv2d_dilated", [(1, 2, 8, 7), (3, 2, 3, 2), (3,)], conv(dilation=2, padding=[2, 1]))
case("conv2d_groups", [(2, 4, 6, 6), (6, 2, 3, 3), (6,)], conv(groups=2, padding=1))
case("conv2d_depthwise_mixed", [(2, 3, 7, 8), (3, 1, 2, 3)],
     conv(stride=[2, 1], padding=[1, 0], dilation=[1, 2], groups=3, bias=False))
case("conv2d_1x1", [(2, 3, 4, 4), (5, 3, 1, 1), (5,)], conv())
case("conv2d_big_padding", [(1, 2, 4, 4), (2, 2, 3, 3)], conv(padding=3, stride=2, bias=False))
case("conv2d_groups_stride_dilation", [(2, 4, 9, 9), (4, 2, 3, 3), (4,)],
     conv(stride=2, padding=2, dilation=2, groups=2))


# normalization
def bn_prep(inputs, gen):
    inputs[4] = inputs[4].abs() + 0.5  # running_var
    return inputs


case("batch_norm", [(3, 4, 3, 2), (4,), (4,), (4,), (4,)],
     lambda b, x, w, c, m, v: b.op("batch_norm", x, w, c, m, v, eps=1e-5), bn_prep)
case("batch_norm_rank2", [(5, 4), (4,), (4,), (4,), (4,)],
     lambda b, x, w, c, m, v: b.op("batch_norm", x, w, c, m, v, eps=1e-3), bn_prep)
case("batch_norm_no_affine", [(3, 4, 2, 2), (4,), (4,)],
     lambda b, x, m, v: b.op("batch_norm", x, None, None, m, v, eps=1e-5),
     lambda inputs, gen: [inputs[0], inputs[1], inputs[2].abs() + 0.5])
case("layer_norm", [(2, 3, 5), (5,), (5,)],
     lambda b, x, w, c: b.op("layer_norm", x, w, c, normalized_shape=[5], eps=1e-5))
case("layer_norm_two_dims", [(2, 3, 5), (3, 5), (3, 5)],
     lambda b, x, w, c: b.op("layer_norm", x, w, c, normalized_shape=[3, 5], eps=1e-5))
case("layer_norm_no_affine", [(4, 6)],
     lambda b, x: b.op("layer_norm", x, None, None, normalized_shape=[6], eps=1e-2))


# attention
def mask_prep(inputs, gen):
    m = inputs[3].clone()
    m[..., torch.arange(min(m.shape[-2:])), torch.arange(min(m.shape[-2:]))] = True  # no fully masked row
    inputs[3] = m
    return inputs


# chunked linear attention and SSM scans (linear_attention, P60): the add rule with one decay per
# head (Mamba 2 SSD), the delta rule (Gated DeltaNet) and one decay per key channel; g <= 0 and
# beta in (0, 1) as in the models


def _linattn_prep(inputs, gen):
    inputs[3] = -inputs[3].abs() * 0.3 - 0.05
    if len(inputs) > 4:
        inputs[4] = torch.sigmoid(inputs[4])
    return inputs


case("linear_attention_ssd", [(2, 5, 2, 3), (2, 5, 2, 3), (2, 5, 2, 4), (2, 5, 2)],
     lambda b, q, k, v, g: b.op("linear_attention", q, k, v, g, None, rule="add", scale=1.0), _linattn_prep)
case("linear_attention_delta", [(2, 5, 2, 3), (2, 5, 2, 3), (2, 5, 2, 4), (2, 5, 2), (2, 5, 2)],
     lambda b, q, k, v, g, beta: b.op("linear_attention", q, k, v, g, beta, rule="delta", scale=0.5), _linattn_prep)
case("linear_attention_channel", [(1, 4, 2, 3), (1, 4, 2, 3), (1, 4, 2, 2), (1, 4, 2, 3)],
     lambda b, q, k, v, g: b.op("linear_attention", q, k, v, g, None, rule="add", scale=1.0), _linattn_prep)
case("attention", [(2, 3, 4, 6), (2, 3, 5, 6), (2, 3, 5, 4)],
     lambda b, q, k, v: b.op("attention", q, k, v, None, scale=None, is_causal=False))
case("attention_bool_mask", [(2, 4, 6), (2, 5, 6), (2, 5, 3), (4, 5)],
     lambda b, q, k, v, m: b.op("attention", q, k, v, m, scale=None, is_causal=False),
     mask_prep, dtypes=[F64, F64, F64, torch.bool])
case("attention_float_mask", [(2, 4, 6), (2, 5, 6), (2, 5, 3), (4, 5)],
     lambda b, q, k, v, m: b.op("attention", q, k, v, m, scale=None, is_causal=False))
case("attention_causal_scale", [(2, 4, 6), (2, 4, 6), (2, 4, 3)],
     lambda b, q, k, v: b.op("attention", q, k, v, None, scale=0.37, is_causal=True))
# the fused attention backward (ir/attention_grad.py): reverse and forward mode through its lowering
# (o is the attention's own output, as the gradient pass emits it: attention_grad does not read it)
case("attention_grad_causal", [(2, 4, 6)] * 4,
     lambda b, q, k, v, do: b.op("attention_grad", q, k, v,
                                 b.op("attention", q, k, v, None, scale=0.41, is_causal=True), do,
                                 scale=0.41, is_causal=True))


# pooling
def pool_prep(inputs, gen):
    return [distinct(tuple(inputs[0].shape), gen)]


def pool(**attrs):
    attrs.setdefault("padding", 0)
    attrs.setdefault("dilation", 1)
    attrs.setdefault("ceil_mode", False)
    return lambda b, x: b.op("max_pool2d", x, **attrs)


case("max_pool2d_overlap", [(2, 2, 7, 7)], pool(kernel_size=3, stride=2, padding=1), pool_prep)
case("max_pool2d_plain", [(1, 3, 6, 6)], pool(kernel_size=2, stride=2), pool_prep)
case("max_pool2d_mixed", [(1, 2, 8, 9)], pool(kernel_size=[3, 2], stride=[2, 1], padding=[1, 1], dilation=[1, 2]),
     pool_prep)
case("max_pool2d_ceil", [(1, 2, 7, 7)], pool(kernel_size=2, stride=2, ceil_mode=True), pool_prep)
case("adaptive_avg_pool2d_overlap", [(2, 3, 7, 5)], lambda b, x: b.op("adaptive_avg_pool2d", x, output_size=[3, 2]))
case("adaptive_avg_pool2d_global", [(2, 3, 4, 5)], lambda b, x: b.op("adaptive_avg_pool2d", x, output_size=[1, 1]))
case("adaptive_avg_pool2d_rank3", [(3, 6, 6)], lambda b, x: b.op("adaptive_avg_pool2d", x, output_size=[4, 3]))


# the gradient ops the conv2d and max_pool2d rules emit (2026-11-24); their own rules give second
# order gradients
case("conv2d_input_grad_stride", [(2, 4, 4, 4), (4, 3, 3, 3)],
     lambda b, g, w: b.op("conv2d_input_grad", g, w, input_size=[8, 8], stride=2, padding=1, dilation=1,
                          groups=1))
case("conv2d_input_grad_groups_dilated", [(1, 6, 5, 8), (6, 2, 3, 2)],
     lambda b, g, w: b.op("conv2d_input_grad", g, w, input_size=[9, 8], stride=[2, 1], padding=[2, 1],
                          dilation=[2, 2], groups=2))
case("conv2d_weight_grad_stride", [(2, 3, 8, 8), (2, 4, 4, 4)],
     lambda b, x, g: b.op("conv2d_weight_grad", x, g, kernel_size=[3, 3], stride=2, padding=1, dilation=1,
                          groups=1))
case("conv2d_weight_grad_depthwise", [(2, 3, 7, 8), (2, 3, 4, 4)],
     lambda b, x, g: b.op("conv2d_weight_grad", x, g, kernel_size=[2, 3], stride=[2, 1], padding=[1, 0],
                          dilation=[1, 2], groups=3))
# x only moves which tap is selected, so its exact gradient is zero; finite differences of a zero
# gradient are rounding noise, so x is a constant here and the zero is checked on its own below
case("max_pool2d_grad_overlap", [(2, 2, 4, 4)],
     lambda b, g: b.op("max_pool2d_grad", b.param("x", distinct((2, 2, 7, 7), torch.Generator().manual_seed(3))),
                       g, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))


def test_max_pool2d_grad_has_a_zero_gradient_in_x():
    b = ir.Builder("mpg")
    x = b.input("x", F64, (1, 2, 7, 7))
    g = b.input("g", F64, (1, 2, 4, 4))
    prog = b.build([b.op("max_pool2d_grad", x, g, kernel_size=3, stride=2, padding=1, dilation=1,
                         ceil_mode=False)])
    gen = torch.Generator().manual_seed(0)
    xv, gv = distinct((1, 2, 7, 7), gen), torch.randn((1, 2, 4, 4), generator=gen, dtype=F64)
    v = torch.randn((1, 2, 7, 7), generator=gen, dtype=F64)
    gx = ir.run(grad(prog, ["x"], seed="input"), xv, gv, v)[1]
    assert torch.equal(gx, torch.zeros_like(gx))
    for i in range(xv.numel()):        # a step that keeps every window's winner changes nothing
        x2 = xv.clone()
        x2.reshape(-1)[i] += 1e-3
        assert torch.equal(ir.run(prog, x2, gv)[0], ir.run(prog, xv, gv)[0])

# shape ops
case("reshape", [(2, 3, 4)], lambda b, x: b.op("reshape", x, shape=[4, -1]))
case("permute", [(2, 3, 4)], lambda b, x: b.op("permute", x, dims=[2, 0, 1]))
case("expand", [(3, 1, 4)], lambda b, x: b.op("expand", x, shape=[2, 3, 5, 4]))
case("select", [(3, 4, 2)], lambda b, x: b.op("select", x, dim=1, index=2))
case("select_negative", [(3, 4)], lambda b, x: b.op("select", x, dim=-1, index=-1))
case("slice", [(3, 7)], lambda b, x: b.op("slice", x, dim=1, start=2, end=5, step=1))
case("slice_step", [(3, 8)], lambda b, x: b.op("slice", x, dim=1, start=1, end=None, step=3))
case("slice_negative_start", [(6, 2)], lambda b, x: b.op("slice", x, dim=0, start=-4, end=-1, step=2))
case("cat", [(2, 3), (2, 1), (2, 4)], lambda b, x, y, z: b.op("cat", x, y, z, dim=1))
case("cat_single", [(2, 3)], lambda b, x: b.op("cat", x, dim=0))

# decoder LLM ops (2026-11-24)
case("flip", [(3, 4, 2)], lambda b, x: b.op("flip", x, dims=[0, 2]))


def _index_copy_case(b, x, src):
    idx = b.param("idx", torch.tensor([3, 0]))  # two of five rows of dim 1, in a non sorted order
    return b.op("index_copy", x, idx, src, dim=1)


case("index_copy", [(2, 5, 3), (2, 2, 3)], _index_copy_case)

# control flow support (2026-11-24): a select at a run time index (autodiff_cf.py)
case("select_dynamic", [(2, 5, 3)],
     lambda b, x: b.op("select_dynamic", x, b.param("i", torch.tensor(3)), dim=1))
case("select_dynamic_negative", [(4, 3)],
     lambda b, x: b.op("select_dynamic", x, b.param("i", torch.tensor(-1)), dim=0))


def _distinct_prep(inputs, gen):
    inputs[0] = distinct(tuple(inputs[0].shape), gen)
    return inputs


def _tie_prep(inputs, gen):
    x = distinct(tuple(inputs[0].shape), gen)
    x[:, 1] = x[:, 0]  # every row of dim 1 ties at its max or not: the cotangent splits evenly
    inputs[0] = x
    return inputs


case("amax_dims", [(2, 3, 4)], lambda b, x: b.op("amax", x, dims=[0, 2], keepdim=False), _distinct_prep)
case("amin_all_keepdim", [(3, 4)], lambda b, x: b.op("amin", x, dims=None, keepdim=True), _distinct_prep)
case("amax_ties", [(3, 2, 2)], lambda b, x: b.op("amax", x, dims=[1], keepdim=True), _tie_prep)


# sort, topk and the gathers (2026-11-24): scatter style backward through scatter_add. Values are
# distinct by 0.05 so no finite difference step reorders them (ties are checked against
# torch.autograd in test_sort_topk_gradients_with_ties_follow_the_stable_order).
case("sort_last_dim", [(3, 6)], lambda b, x: b.op_multi("sort", x, dim=-1, descending=False)[0], _distinct_prep)
case("sort_descending_dim0", [(5, 3)], lambda b, x: b.op_multi("sort", x, dim=0, descending=True)[0],
     _distinct_prep)
case("topk_largest", [(2, 7)], lambda b, x: b.op_multi("topk", x, k=3, dim=-1, largest=True)[0], _distinct_prep)
case("topk_smallest_middle_dim", [(2, 6, 3)], lambda b, x: b.op_multi("topk", x, k=2, dim=1, largest=False)[0],
     _distinct_prep)


def _sort_both_outputs(b, x, y):
    """The indices output reused to gather another input: gradients reach y through gather and x
    through sort."""
    v, i = b.op_multi("sort", x, dim=-1, descending=True)
    return b.op("add", v, b.op("gather", y, i, dim=-1))


case("sort_indices_feed_gather", [(2, 5), (2, 5)], _sort_both_outputs, _distinct_prep)


def _gather_case(b, x):
    idx = b.param("idx", torch.tensor([[0, 2, 2, 1], [3, 3, 3, 0]]))  # repeated indices add up
    return b.op("gather", x, idx, dim=1)


case("gather_repeats", [(2, 4)], _gather_case)


def _index_select_case(b, x):
    idx = b.param("idx", torch.tensor([2, 0, 2]))
    return b.op("index_select", x, idx, dim=1)


case("index_select_repeats", [(3, 4, 2)], _index_select_case)


def _embedding_case(b, w):
    idx = b.param("idx", torch.tensor([1, 4, 1, 0]))
    return b.op("index_select", w, idx, dim=0)


case("index_select_embedding", [(5, 3)], _embedding_case)


def _scatter_add_case(b, x, src):
    idx = b.param("idx", torch.tensor([[0, 3, 0], [2, 2, 1]]))
    return b.op("scatter_add", x, idx, src, dim=1)


case("scatter_add_repeats", [(2, 4), (2, 3)], _scatter_add_case)


# ----------------------------------------------------------------------------- rules added with the op
# coverage work (2026-11-24, ir/grad_rules_extra.py, docs/MODEL_COVERAGE.md)

case("cos", [(3, 4)], lambda b, x: b.op("cos", x))
case("sin", [(3, 4)], lambda b, x: b.op("sin", x))
case("log", [(3, 4)], lambda b, x: b.op("log", x), positive())
case("reciprocal", [(3, 4)], lambda b, x: b.op("reciprocal", x), positive())
case("abs", [(4, 5)], lambda b, x: b.op("abs", x), kinks([0.0]))
case("cast_f32_roundtrip", [(3, 4)], lambda b, x: b.op("cast", b.op("cast", x, dtype=torch.float32), dtype=F64))
case("log_softmax", [(3, 5)], lambda b, x: b.op("log_softmax", x, dim=-1))
case("pow_both", [(3, 4), (4,)], lambda b, x, y: b.op("pow", x, y), positive(0))
case("pow_scalar_exponent", [(3, 4), ()], lambda b, x, y: b.op("pow", x, y), positive(0))


def _separate(i, j):
    """Keep inputs[i] (the larger shape) 0.1 away from inputs[j], so no entry sits at a tie."""
    def prep(inputs, gen):
        inputs[i] = torch.where((inputs[i] - inputs[j]).abs() < 0.1, inputs[i] + 0.3, inputs[i])
        return inputs
    return prep


case("maximum_broadcast", [(3, 4), (4,)], lambda b, x, y: b.op("maximum", x, y), _separate(0, 1))
case("minimum", [(3, 4), (3, 4)], lambda b, x, y: b.op("minimum", x, y), _separate(0, 1))


def _fmod_prep(inputs, gen):
    inputs[0] = inputs[0] * 3.0
    inputs[1] = inputs[1].abs() + 0.7
    q = inputs[0] / inputs[1]
    near = (q - q.round()).abs() < 0.05     # keep a / b off the integers, where the op jumps
    inputs[0] = torch.where(near, inputs[0] + 0.2 * inputs[1], inputs[0])
    return inputs


case("fmod", [(3, 4), (4,)], lambda b, x, y: b.op("fmod", x, y), _fmod_prep)
case("remainder", [(3, 4), (4,)], lambda b, x, y: b.op("remainder", x, y), _fmod_prep)
case("cumsum", [(3, 5)], lambda b, x: b.op("cumsum", x, dim=1))
case("triu", [(2, 4, 5)], lambda b, x: b.op("triu", x, diagonal=1))
case("tril", [(4, 4)], lambda b, x: b.op("tril", x, diagonal=-1))
case("pad", [(2, 3, 4)], lambda b, x: b.op("pad", x, pads=[1, 2, -1, 1], value=0.5))


def _slice_scatter_case(b, x, src):
    return b.op("slice_scatter", x, src, dim=1, start=1, end=6, step=2)


case("slice_scatter", [(2, 6), (2, 3)], _slice_scatter_case)
case("select_scatter", [(3, 4), (3,)], lambda b, x, src: b.op("select_scatter", x, src, dim=1, index=2))


def _index_single_case(b, w):
    idx = b.param("idx", torch.tensor([[1, 4, 1], [0, -1, 2]]))     # repeats and a negative index
    return b.op("index", w, idx)


def _index_two_case(b, x):
    i = b.param("i", torch.tensor([[0], [2]]))
    j = b.param("j", torch.tensor([1, 3, 1]))                      # broadcast to [2, 3]
    return b.op("index", x, None, i, j)


case("index_embedding", [(5, 3)], _index_single_case)
case("index_two_adjacent", [(2, 3, 4)], _index_two_case)
case("avg_pool2d_pad", [(2, 2, 6, 5)],
     lambda b, x: b.op("avg_pool2d", x, kernel_size=[3, 2], stride=[2, 1], padding=[1, 1], ceil_mode=False,
                       count_include_pad=True, divisor_override=None))
case("avg_pool2d_ceil_exclude_pad", [(1, 2, 7, 7)],
     lambda b, x: b.op("avg_pool2d", x, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], ceil_mode=True,
                       count_include_pad=False, divisor_override=None))
case("group_norm", [(2, 6, 3, 2), (6,), (6,)],
     lambda b, x, w, c: b.op("group_norm", x, w, c, num_groups=3, eps=1e-5))
def _not_parallel(inputs, gen):
    """make() draws the input and the cotangent from the same seed, so for a one input program they
    are equal; a norm without weights is scale invariant, so its gradient along x is zero and the
    check would be relative to nothing. A nonlinear change keeps them apart."""
    inputs[0] = inputs[0] + 0.5 * inputs[0] * inputs[0]
    return inputs


case("group_norm_instance", [(2, 3, 9)], lambda b, x: b.op("group_norm", x, None, None, num_groups=3, eps=1e-5),
     _not_parallel)
case("rms_norm", [(2, 3, 5), (5,)], lambda b, x, w: b.op("rms_norm", x, w, normalized_shape=[5], eps=1e-6))
case("vector_norm_2", [(3, 4)], lambda b, x: b.op("vector_norm", x, ord=2.0, dims=[1], keepdim=False))
case("vector_norm_3_all", [(3, 4)], lambda b, x: b.op("vector_norm", x, ord=3.0, dims=None, keepdim=True),
     kinks([0.0]))
def _distinct_positive(inputs, gen):
    x = distinct(tuple(inputs[0].shape), gen)
    inputs[0] = x - x.min() + 0.5      # distinct and positive, so |x| has no ties either
    return inputs


case("vector_norm_inf", [(3, 4)], lambda b, x: b.op("vector_norm", x, ord=math.inf, dims=[0], keepdim=False),
     _distinct_positive)
case("adaptive_max_pool2d", [(2, 2, 7, 5)], lambda b, x: b.op("adaptive_max_pool2d", x, output_size=[3, 2]),
     _distinct_prep)


# structure: fan out and a chain mixing several rules
def fan_out(b, x):
    sq = b.op("mul", x, x)
    ex = b.op("exp", x)
    return b.op("add", b.op("add", sq, b.op("mul", ex, x)), b.op("tanh", sq))


case("fan_out", [(3, 4)], fan_out)


def chain(b, x, w):
    h = b.op("linear", x, w, None)
    h = b.op("layer_norm", h, None, None, normalized_shape=[4], eps=1e-5)
    h = b.op("gelu", h, approximate="none")
    p = b.op("softmax", h, dim=-1)
    return [b.op("sum", b.op("mul", p, h), dims=[1], keepdim=False), b.op("mean", h, dims=None, keepdim=False)]


case("chain_two_outputs", [(3, 5), (4, 5)], chain)


def custom_vjp_case(b, x, w):
    """custom_vjp (ir/custom_vjp.py): the rule inlines the bwd region; this checks the inlining.
    Wrong user rules are refused at registration, tests/test_custom_grad.py."""
    def region(name, types):
        r = ir.Builder(name)
        return r, [r.input(f"a{i}", F64, t) for i, t in enumerate(types)]
    fb, (fx, fw) = region("fwd", [(3, 4), (4,)])
    fwd = fb.build([fb.op("mul", fx, fb.op("tanh", fw))])
    bb, (bx, bw, _out, bg) = region("bwd", [(3, 4), (4,), (3, 4), (3, 4)])
    t = bb.op("tanh", bw)
    one = bb.op("full", shape=[4], value=1.0, dtype=F64)
    dw = bb.op("mul", bb.op("sum", bb.op("mul", bg, bx), dims=[0], keepdim=False),
               bb.op("sub", one, bb.op("mul", t, t)))
    bwd = bb.build([bb.op("mul", bg, t), dw])
    return b.op("custom_vjp", x, w, fwd=fwd, bwd=bwd, name="x_tanh_w")


case("custom_vjp", [(3, 4), (4,)], custom_vjp_case)

FD_RESULTS = {}


@pytest.mark.parametrize("name", sorted(CASES))
def test_rule_matches_finite_differences(name):
    prog, inputs = make(name)
    err = fd_check(prog, inputs)
    FD_RESULTS[name] = err
    assert err < FD_TOL, f"{name}: gradient differs from finite differences by {err:.2e}"


def test_every_differentiable_op_has_a_finite_difference_case():
    covered = set()
    for name in CASES:
        covered |= set(make(name)[0].op_counts())
    differentiable = set(grad_rules.GRAD_RULES) | set(grad_rules.LOWERINGS)
    assert differentiable - covered == set()


# The ops docs/IR.md lists (47) plus erf, which grad_rules.py adds. Each must be handled explicitly.
IR_OPS = """conv2d linear matmul batch_norm layer_norm attention relu gelu hardswish hardsigmoid silu
sigmoid tanh clamp add sub mul div neg exp sqrt rsqrt where eq ne lt le gt ge logical_and logical_or
logical_not mean sum any softmax max_pool2d adaptive_avg_pool2d reshape permute expand select slice
cat full full_like arange erf""".split()


def test_every_ir_op_is_classified():
    table = autodiff.rule_table()
    assert len(IR_OPS) == 48
    assert [op for op in IR_OPS if op not in table] == []
    assert {op for op, kind in table.items() if kind == "lowered"} == {
        "batch_norm", "layer_norm", "attention", "group_norm", "rms_norm", "vector_norm", "adaptive_max_pool2d",
        "attention_grad"}
    assert {op for op, kind in table.items() if kind == "no gradient"} == set(grad_rules.NONDIFF)
    # the added op is a real IR op: shape rule and reference kernel
    assert "erf" in shapes.RULES and "erf" in reference.IMPLS


# ----------------------------------------------------------------------------- structure


def test_fan_out_accumulates():
    b = ir.Builder("fan")
    x = b.input("x", F64, (5,))
    y = b.op("add", b.op("add", x, x), b.op("mul", x, x))  # dy/dx = 2 + 2x
    gp = grad(b.build([y]), ["x"], seed="ones")
    xv = torch.randn(5, dtype=F64)
    _, g = ir.run(gp, xv)
    assert rel_err(g, 2 + 2 * xv) < 1e-15


def test_seed_modes():
    b = ir.Builder("seeds")
    x = b.input("x", F64, (3, 4))
    y = b.op("mul", x, x)
    s = b.op("sum", y, dims=None, keepdim=False)
    xv = torch.randn(3, 4, dtype=F64)

    scalar = grad(b.build([s]), ["x"])  # rank 0 output: seeded with 1, no new input
    assert [v.name for v in scalar.inputs] == ["x"]
    assert rel_err(ir.run(scalar, xv)[1], 2 * xv) < 1e-15

    tensor = grad(b.build([y]), ["x"])  # non scalar output: cotangent input
    assert [v.name for v in tensor.inputs] == ["x", "d_mul"]
    assert tensor.inputs[1].type.same_as(y.type)
    v = torch.randn(3, 4, dtype=F64)
    assert rel_err(ir.run(tensor, xv, v)[1], 2 * xv * v) < 1e-15

    ones = grad(b.build([y]), ["x"], seed="ones")
    assert len(ones.inputs) == 1
    assert rel_err(ir.run(ones, xv)[1], 2 * xv) < 1e-15

    both = grad(b.build([y, s]), ["x"], of=[1])  # only the second output is differentiated
    assert len(both.inputs) == 1 and len(both.outputs) == 3
    assert rel_err(ir.run(both, xv)[2], 2 * xv) < 1e-15


def test_dead_code_gets_no_backward():
    b = ir.Builder("dead")
    x = b.input("x", F64, (4,))
    w = b.input("w", F64, (4,))
    a = b.op("exp", x)
    out0 = b.op("tanh", a)
    c = b.op("sigmoid", w)
    out1 = b.op("mul", c, c)
    unused = b.op("relu", x)
    prog = b.build([out0, out1])

    gp = grad(prog, ["x"], of=[0])
    backward = gp.nodes[len(prog.nodes):]
    # tanh costs sub, mul, mul and exp costs one mul; sigmoid, the other mul and relu cost nothing
    assert [n.op for n in backward] == ["mul", "sub", "mul", "mul"]
    dead = {id(v) for v in (w, c, out1, unused)}
    assert not any(id(v) in dead for n in backward for v in n.inputs if v is not None)

    # w does not reach output 0: its gradient is zeros and no rule runs
    gw = grad(prog, ["w"], of=[0])
    assert [n.op for n in gw.nodes[len(prog.nodes):]] == ["full"]
    res = ir.run(gw, torch.randn(4, dtype=F64), torch.randn(4, dtype=F64), torch.randn(4, dtype=F64))
    assert torch.equal(res[2], torch.zeros(4, dtype=F64))


def test_only_requested_operands_get_backward():
    b = ir.Builder("lin")
    x = b.input("x", F64, (3, 5))
    w = b.input("w", F64, (4, 5))
    c = b.input("c", F64, (4,))
    prog = b.build([b.op("linear", x, w, c)])
    assert [n.op for n in grad(prog, ["x"]).nodes[1:]] == ["matmul"]
    assert [n.op for n in grad(prog, ["c"]).nodes[1:]] == ["sum"]
    assert [n.op for n in grad(prog, ["w"]).nodes[1:]] == ["permute", "matmul"]


def test_values_behind_non_differentiable_ops_get_no_backward():
    b = ir.Builder("mask")
    x = b.input("x", F64, (6,))
    y = b.input("y", F64, (6,))
    zero = b.param("zero", torch.tensor(0.0, dtype=F64))
    out = b.op("where", b.op("gt", b.op("exp", x), zero), y, b.op("full_like", y, value=0.0, dtype=None))
    prog = b.build([out])
    gp = grad(prog, ["x", "y"], seed="ones")
    # x only reaches the output through a comparison, so nothing is emitted for exp
    assert "exp" not in [n.op for n in gp.nodes[len(prog.nodes):]]
    res = ir.run(gp, torch.randn(6, dtype=F64), torch.randn(6, dtype=F64))
    assert torch.equal(res[1], torch.zeros(6, dtype=F64)) and torch.equal(res[2], torch.ones(6, dtype=F64))


def test_lowered_forward_matches_the_fused_reference_ops():
    """batch_norm, layer_norm and attention are rewritten into primitives when differentiated. The
    rewritten forward must give what the fused reference kernel gives."""
    for name in ("batch_norm", "layer_norm", "attention_bool_mask", "attention_causal_scale"):
        prog, inputs = make(name)
        gp = grad(prog, float_inputs(prog), seed="ones")
        lowered_op = name.split("_")[0] if name.startswith("attention") else name
        assert lowered_op in prog.op_counts() and lowered_op not in gp.op_counts()
        assert rel_err(ir.run(gp, *inputs)[0], ir.run(prog, *inputs)[0]) < 1e-15
    # not differentiated through: the fused op stays
    prog, inputs = make("layer_norm")
    b = ir.Builder("keep")
    x = b.input("x", F64, (2, 3, 5))
    z = b.input("z", F64, (2, 3, 5))
    out = b.op("add", b.op("layer_norm", x, None, None, normalized_shape=[5], eps=1e-5), z)
    assert grad(b.build([out]), ["z"]).op_counts().get("layer_norm") == 1


def test_grad_program_is_a_plain_verified_program():
    prog, inputs = make("chain_two_outputs")
    gp = grad(prog, float_inputs(prog))
    gp.verify()
    assert gp.outputs[:2] == prog.outputs
    for w, g in zip(prog.inputs, gp.outputs[2:]):
        assert g.type.same_as(w.type)
    text = str(gp)
    assert "program grad_chain_two_outputs" in text and "%d_" in text
    # the original program is untouched
    assert "layer_norm" in prog.op_counts() and len(prog.inputs) == 2
    prog.verify()


def test_grad_of_grad_gives_hessian_vector_products():
    """The result is a plain program, so the pass applies to it again."""
    b = ir.Builder("f")
    x = b.input("x", F64, (5,))
    w = b.input("w", F64, (4, 5))
    h = b.op("tanh", b.op("linear", x, w, None))
    s = b.op("sum", b.op("mul", h, b.op("sigmoid", h)), dims=None, keepdim=False)
    first = grad(b.build([s]), ["x"])           # outputs: s, ds/dx
    second = grad(first, ["x"], of=[1])         # new input: a vector v; new output: H v
    assert len(second.inputs) == 3 and len(second.outputs) == 3
    gen = torch.Generator().manual_seed(0)
    xv, wv, v = (torch.randn(shape, generator=gen, dtype=F64) for shape in ((5,), (4, 5), (5,)))
    got = ir.run(second, xv, wv, v)

    xr = xv.clone().requires_grad_(True)
    hid = torch.tanh(wv @ xr)
    (g,) = torch.autograd.grad((hid * torch.sigmoid(hid)).sum(), xr, create_graph=True)
    (hv,) = torch.autograd.grad(g, xr, grad_outputs=v)
    assert rel_err(got[1], g.detach()) < 1e-14
    assert rel_err(got[2], hv) < 1e-13


# ----------------------------------------------------------------------------- symbolic dims


def symbolic_program():
    b = ir.Builder("symbolic")
    x = b.input("x", F64, ("B", 1, 4))
    y = b.input("y", F64, (3, 1))
    w = b.input("w", F64, (4,))
    s = b.op("mul", b.op("add", x, y), w)                       # [B, 3, 4], x broadcast over dim 1
    m = b.op("mean", s, dims=[0, 2], keepdim=False)             # count 4 B is symbolic
    first = b.op("select", s, dim=0, index=0)                   # select along the symbolic dim
    rest = b.op("slice", s, dim=0, start=1, end=None, step=1)   # [B - 1, 3, 4]
    back = b.op("cat", b.op("reshape", first, shape=[1, 3, 4]), rest, dim=0)
    e = b.op("expand", b.op("reshape", w, shape=[1, 1, 4]), shape=[b.symbol("B"), 3, 4])
    total = b.op("sum", b.op("mul", back, e), dims=[1], keepdim=False)
    return b.build([m, total, b.op("tanh", s)])


def test_symbolic_batch_one_program_two_sizes():
    prog = symbolic_program()
    gp = grad(prog, ["x", "y", "w"], seed="input")
    assert [str(g.type) for g in gp.outputs[3:]] == ["f64[B, 1, 4]", "f64[3, 1]", "f64[4]"]
    for batch in (2, 5):
        gen = torch.Generator().manual_seed(batch)
        inputs = [torch.randn(s, generator=gen, dtype=F64) for s in ((batch, 1, 4), (3, 1), (4,))]
        assert fd_check(prog, inputs, gprog=gp) < FD_TOL


def test_symbolic_limits_fail_loudly():
    # cat along a symbolic dim: an operand behind a symbolic sized one starts at a symbolic offset,
    # which the IR's slice cannot express yet
    b = ir.Builder("cat_sym")
    x = b.input("x", F64, ("B", 3))
    y = b.input("y", F64, (2, 3))
    prog = b.build([b.op("cat", x, y, dim=0)])
    for name in ("x", "y"):
        with pytest.raises(ir.UnsupportedOpError, match="gradient of cat along a symbolic dim"):
            grad(prog, [name])
    b = ir.Builder("cat_sym_ok")
    x = b.input("x", F64, ("B", 3))
    y = b.input("y", F64, (2, 3))
    ok = b.build([b.op("cat", y, x, dim=0)])  # fixed sizes first, the symbolic one last: supported
    assert fd_check(ok, [torch.randn(3, 3, dtype=F64), torch.randn(2, 3, dtype=F64)]) < FD_TOL

    b = ir.Builder("conv_sym")
    x = b.input("x", F64, ("B", 3, "H", 8))
    w = b.input("w", F64, (4, 3, 3, 3))
    prog = b.build([b.op("conv2d", x, w, None, stride=1, padding=1, dilation=1, groups=1)])
    with pytest.raises(ir.UnsupportedOpError, match="gradient of conv2d needs fixed"):
        grad(prog, ["w"])


# ----------------------------------------------------------------------------- loud failures


def test_op_without_a_rule_fails_naming_the_op(monkeypatch):
    monkeypatch.setitem(shapes.RULES, "mystery", shapes.RULES["neg"])
    monkeypatch.setitem(shapes.RULES, "riddle", shapes.RULES["neg"])
    b = ir.Builder("unknown")
    x = b.input("x", F64, (3,))
    w = b.input("w", F64, (3,))
    h = b.op("mystery", b.op("exp", x), source="aten.mystery.default (node %m), model.py:12")
    out = b.op("add", b.op("riddle", h), w)
    side = b.op("mystery", w)
    prog = b.build([out, side])
    with pytest.raises(ir.UnsupportedOpError) as err:
        grad(prog, ["x"], of=[0])
    msg = str(err.value)
    assert "no gradient rule for IR op 'mystery'" in msg and "no gradient rule for IR op 'riddle'" in msg
    assert "model.py:12" in msg and "2 op(s)" in msg
    # an op with no rule that is off the gradient path is not a problem
    gp = grad(prog, ["w"], of=[0])
    assert [n.op for n in gp.nodes[len(prog.nodes):]] == []


def test_bad_requests_fail_clearly():
    b = ir.Builder("bad")
    x = b.input("x", F64, (3,))
    i = b.input("i", torch.int64, (3,))
    prog = b.build([b.op("mul", x, i), b.op("gt", x, x)])
    with pytest.raises(ir.IRError, match="'nope' is not an input or param"):
        grad(prog, ["nope"])
    with pytest.raises(ir.IRError, match="not floating point"):
        grad(prog, ["i"], of=[0])
    with pytest.raises(ir.IRError, match="not floating point"):
        grad(prog, ["x"])  # output 1 is bool
    with pytest.raises(ir.IRError, match="listed twice"):
        grad(prog, ["x", "x"], of=[0])
    with pytest.raises(ir.IRError, match="has 2 outputs"):
        grad(prog, ["x"], of=[2])
    with pytest.raises(ir.IRError, match="seed must be"):
        grad(prog, ["x"], of=[0], seed="zeros")
    with pytest.raises(ir.IRError, match="wrt is empty"):
        grad(prog, [], of=[0])


# ----------------------------------------------------------------------------- whole models vs torch.autograd

BATCH = ({0: Dim("B", min=1, max=64)},)


def randomize_norms(model: nn.Module, seed: int = 0) -> nn.Module:
    """Non trivial statistics and affine params for every norm, so a wrong norm gradient shows."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) + 0.5)
            if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)) and m.weight is not None:
                m.weight.copy_(1 + 0.2 * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=g))
    return model


class ExactHardsigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


def exact_hardsigmoid(model: nn.Module) -> nn.Module:
    """Swap nn.Hardsigmoid for the same function written with clamp, in the torch reference only.

    PyTorch's hardsigmoid backward kernel multiplies by 1/6 rounded to float32 even for float64
    tensors (0.1666666716337204, torch 2.11), a relative error of 3.0e-8 in every gradient that
    passes through it. With the stock module mobilenet_v3_small differs from our gradients by
    3.2e-8; with the kernel avoided the two agree to rounding. See
    test_hardsigmoid_gradient_is_exact_where_torch_rounds.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Hardsigmoid):
            setattr(model, name, ExactHardsigmoid())
        else:
            exact_hardsigmoid(child)
    return model


def test_hardsigmoid_gradient_is_exact_where_torch_rounds():
    b = ir.Builder("hs")
    x = b.input("x", F64, (3,))
    gp = grad(b.build([b.op("hardsigmoid", x)]), ["x"], seed="ones")
    xv = torch.tensor([0.5, -1.0, 2.0], dtype=F64)
    ours = ir.run(gp, xv)[1]
    assert rel_err(ours, torch.full((3,), 1.0 / 6.0, dtype=F64)) < 1e-15
    xr = xv.clone().requires_grad_(True)
    (stock,) = torch.autograd.grad(nn.functional.hardsigmoid(xr).sum(), xr)
    # measured 2.98e-8 on torch 2.11; not asserted to be nonzero, so a fixed PyTorch still passes
    assert rel_err(stock, ours) < 1e-7
    xr = xv.clone().requires_grad_(True)
    (exact,) = torch.autograd.grad(ExactHardsigmoid()(xr).sum(), xr)
    assert rel_err(exact, ours) < 1e-15


def model_grad_error(model, shape, decompose, batches):
    """Worst relative error, over the input and every parameter, between the gradients from one
    transformed program (symbolic batch) and torch.autograd in float64, at each batch size."""
    model = model.eval()
    prog = ir.import_module(model, (torch.randn(2, *shape),), dynamic_shapes=BATCH, decompose=decompose)
    names = [n for n, _ in model.named_parameters()]
    assert [n for n in names if n not in prog.param_data] == []  # every parameter is differentiated
    gp = grad(prog, [prog.inputs[0].name] + names)
    assert "B" in str(gp.inputs[1].type)  # the cotangent input keeps the symbolic batch
    worst = 0.0
    for batch in batches:
        gen = torch.Generator().manual_seed(100 + batch)
        x = torch.randn((batch, *shape), generator=gen, dtype=F64)
        ref = exact_hardsigmoid(copy.deepcopy(model).double())
        xr = x.clone().requires_grad_(True)
        y = ref(xr)
        v = torch.randn(y.shape, generator=gen, dtype=F64)
        params = dict(ref.named_parameters())
        want = torch.autograd.grad(y, [xr] + [params[n] for n in names], grad_outputs=v)
        got = ir.run(gp, x, v)
        worst = max(worst, rel_err(got[0], y.detach()))
        for g, w in zip(got[1:], want):
            assert tuple(g.shape) == tuple(w.shape)
            worst = max(worst, rel_err(g, w))
    return worst, gp


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 12),
                         nn.ReLU(), nn.Linear(12, 4)), (8,)


def _resnet18():
    import torchvision
    torch.manual_seed(0)
    return randomize_norms(torchvision.models.resnet18()), (3, 64, 64)


def _encoder_layer_relu():
    torch.manual_seed(0)
    return randomize_norms(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)), (7, 32)


def _encoder_layer_gelu_prenorm():
    torch.manual_seed(0)
    return randomize_norms(nn.TransformerEncoderLayer(32, 4, 64, batch_first=True, activation="gelu",
                                                      norm_first=True)), (7, 32)


def _mobilenet_v3_small():
    import torchvision
    torch.manual_seed(0)
    return randomize_norms(torchvision.models.mobilenet_v3_small()), (3, 64, 64)


class Zoo(nn.Module):
    """Features the three named models do not reach: grouped, dilated and depthwise conv, ceil mode
    pooling, adaptive pooling to a non divisor size, silu, hardswish, relu6, tanh gelu, sigmoid,
    slicing with a step, cat, division, causal attention and a mean over the batch."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 3, padding=2, dilation=2, groups=2)
        self.dw = nn.Conv2d(8, 8, (3, 2), stride=(1, 2), groups=8, bias=False)
        self.pool = nn.MaxPool2d(3, 2, padding=1, ceil_mode=True)
        self.avg = nn.AdaptiveAvgPool2d((3, 2))
        self.fc = nn.Linear(8 * 3 * 2, 24)
        self.norm = nn.LayerNorm(8)
        self.out = nn.Linear(24 + 12, 5)

    def forward(self, x):
        h = nn.functional.silu(self.bn(self.c1(x)))
        h = nn.functional.hardswish(self.c2(h)) + h
        h = nn.functional.relu6(self.dw(h))
        h = self.avg(self.pool(h))
        h = nn.functional.gelu(self.fc(h.flatten(1)), approximate="tanh")
        t = self.norm(h.reshape(-1, 3, 8))
        t = nn.functional.scaled_dot_product_attention(t, t, t, is_causal=True)
        t = torch.sigmoid(t).reshape(-1, 24)
        t = t / (1.0 + t * t).sqrt() - t.mean(dim=0, keepdim=True)
        return self.out(torch.cat([h, t[:, ::2]], dim=1)).softmax(dim=-1)


def _zoo():
    torch.manual_seed(0)
    return randomize_norms(Zoo()), (3, 19, 17)


MODELS = {"mlp": _mlp, "resnet18": _resnet18, "encoder_layer_relu": _encoder_layer_relu,
          "encoder_layer_gelu_prenorm": _encoder_layer_gelu_prenorm, "mobilenet_v3_small": _mobilenet_v3_small,
          "zoo": _zoo}
MODEL_RESULTS = {}


@pytest.mark.parametrize("decompose", [False, True], ids=["export_graph", "core_aten"])
@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_gradients_match_torch_autograd(name, decompose):
    """Input and every parameter, one transformed program, batch sizes 1 and 3 (symbolic batch)."""
    model, shape = MODELS[name]()
    err, _ = model_grad_error(model, shape, decompose, batches=(1, 3))
    MODEL_RESULTS[(name, decompose)] = err
    assert err < AUTOGRAD_TOL, f"{name}: gradients differ from torch.autograd by {err:.2e}"


def test_model_gradients_also_pass_finite_differences():
    """An end to end check that does not lean on torch.autograd: a few random directions through the
    whole transformer layer, directional derivative against the vector Jacobian product."""
    model, shape = _encoder_layer_gelu_prenorm()
    model = model.eval()
    prog = ir.import_module(model, (torch.randn(2, *shape),), dynamic_shapes=BATCH)
    names = [n for n, _ in model.named_parameters()]
    gp = grad(prog, [prog.inputs[0].name] + names)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn((2, *shape), generator=gen, dtype=F64)
    v = torch.randn((2, *shape), generator=gen, dtype=F64)
    grads = ir.run(gp, x, v)[1:]
    saved = dict(prog.param_data)
    try:
        for trial in range(3):
            # a unit length direction over the input and all parameters together, so the fifth
            # derivative along it stays O(1) and the finite difference bar above applies
            dx = torch.randn(x.shape, generator=gen, dtype=F64)
            dps = {n: torch.randn(saved[n].shape, generator=gen, dtype=F64) for n in names}
            norm = math.sqrt(dx.pow(2).sum().item() + sum(d.pow(2).sum().item() for d in dps.values()))
            dx = dx / norm
            dps = {n: d / norm for n, d in dps.items()}
            analytic = (grads[0] * dx).sum().item() + sum((g * dps[n]).sum().item() for g, n in zip(grads[1:], names))

            def phi(t):
                for n in names:
                    prog.param_data[n] = saved[n].double() + t * dps[n]
                return (ir.run(prog, x + t * dx)[0] * v).sum().item()

            h = FD_STEP
            numeric = (phi(-2 * h) - 8 * phi(-h) + 8 * phi(h) - phi(2 * h)) / (12 * h)
            assert abs(numeric - analytic) / abs(numeric) < FD_TOL
    finally:
        prog.param_data.update(saved)


# ----------------------------------------------------------------------------- training demo


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 3))

    def forward(self, x, y):
        d = self.body(x) - y
        return (d * d).mean()


def test_sgd_training_matches_torch_step_for_step():
    """An MLP with its loss, imported from torch.export, trained with gradients from the pass and
    the reference interpreter. Loss and every parameter must track torch.optim.SGD in float64."""
    torch.manual_seed(0)
    model = Regressor().double()
    batch = Dim("B", min=1, max=64)
    # import a copy: the IR shares parameter storage with the module it was exported from, and the
    # torch optimizer below updates its module in place
    prog = ir.import_module(copy.deepcopy(model), (torch.randn(4, 6, dtype=F64), torch.randn(4, 3, dtype=F64)),
                            dynamic_shapes=({0: batch}, {0: batch}))
    names = [n for n, _ in model.named_parameters()]
    step = grad(prog, names)  # outputs: loss, then one gradient per parameter
    assert len(step.inputs) == 2 and step.outputs[0].type.rank == 0

    lr = 0.1
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1)
    teacher = torch.randn(6, 3, generator=gen, dtype=F64)
    x_fixed = torch.randn(16, 6, generator=gen, dtype=F64)
    batches = [(x_fixed, torch.tanh(x_fixed @ teacher))] * 8
    for size in (1, 5):  # the same transformed program at other batch sizes
        xb = torch.randn(size, 6, generator=gen, dtype=F64)
        batches.append((xb, torch.tanh(xb @ teacher)))

    losses = []
    for i, (x, y) in enumerate(batches):
        out = ir.run(step, x, y)
        loss, grads = out[0], out[1:]
        for n, g in zip(names, grads):
            step.param_data[n] = step.param_data[n] - lr * g

        opt.zero_grad()
        want = model(x, y)
        want.backward()
        opt.step()

        assert abs(loss.item() - want.item()) <= 1e-13 * abs(want.item()), f"step {i}: loss differs"
        for n, p in model.named_parameters():
            assert rel_err(step.param_data[n], p.detach()) < 1e-13, f"step {i}: {n} differs"
        losses.append(loss.item())

    fixed = losses[:8]
    assert all(b < a for a, b in zip(fixed, fixed[1:])), f"loss did not decrease: {fixed}"
    assert fixed[-1] < 0.9 * fixed[0]


# ----------------------------------------------------------------------------- tests that prove the tests work


def _broken_tanh(e, n, g):
    y = n.output
    return [lambda: e.op("mul", g, e.op("sub", e.const(1.0, y), y))]  # 1 - y, should be 1 - y^2


def test_broken_rule_is_caught_by_finite_differences(monkeypatch):
    prog, inputs = make("tanh")
    assert fd_check(prog, inputs) < FD_TOL
    monkeypatch.setitem(grad_rules.GRAD_RULES, "tanh", _broken_tanh)
    assert fd_check(prog, inputs) > 1e-2


def test_broken_rule_is_caught_by_autograd_comparison(monkeypatch):
    model, shape = _mlp()
    monkeypatch.setitem(grad_rules.GRAD_RULES, "tanh", _broken_tanh)
    err, _ = model_grad_error(model, shape, decompose=False, batches=(2,))
    assert err > 1e-2


def test_subtle_break_is_caught(monkeypatch):
    """A small error, not a gross one: the layer norm lowering with eps dropped from the variance.
    The forward value moves by about 1e-5, far above both bars."""
    real = grad_rules.LOWERINGS["layer_norm"]

    def no_eps(e, n):
        node = ir.Node(n.op, n.inputs, dict(n.attrs, eps=0.0), n.output, n.source)
        return real(e, node)

    monkeypatch.setitem(grad_rules.LOWERINGS, "layer_norm", no_eps)
    model, shape = _encoder_layer_relu()
    err, _ = model_grad_error(model, shape, decompose=False, batches=(2,))
    assert err > 1e-7


def test_wrong_shape_from_a_rule_is_an_internal_error(monkeypatch):
    """The pass checks every cotangent against the shape of the value it belongs to."""
    monkeypatch.setattr(autodiff._Emitter, "sum_to", lambda self, g, t: g)
    prog, _ = make("add_broadcast")
    with pytest.raises(ir.IRError, match="internal error: the gradient rule of add produced"):
        grad(prog, ["x0", "x1"])


def test_zz_report_measured_errors(capsys):
    """Prints the measured errors when TRUESTAR_SHOW_ERRORS=1. Never fails; last in this file."""
    import os
    if os.environ.get("TRUESTAR_SHOW_ERRORS") != "1":
        return
    with capsys.disabled():
        if FD_RESULTS:
            worst = max(FD_RESULTS, key=FD_RESULTS.get)
            print(f"\nfinite differences: {len(FD_RESULTS)} cases, worst {FD_RESULTS[worst]:.2e} ({worst})")
        for (name, decompose), err in sorted(MODEL_RESULTS.items()):
            print(f"autograd: {name} {'core_aten' if decompose else 'export_graph'} {err:.2e}")
