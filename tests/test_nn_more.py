"""The layers added with the op coverage work (truestar/nn/layers_more.py, docs/NN.md).

Each layer three ways against torch.nn (or, for rotary and ALiBi, against transformers' Llama rotary
code and the ALiBi paper's definition), all in float64: eagerly, compiled as a native function, and
its gradient through truestar.grad against torch.autograd. Bar 1e-12 relative (TOL of test_nn.py).
"""

import math

import pytest
import torch
import torch.nn.functional as F

import truestar
from test_nn import TOL, grads_vs_torch, rel
from truestar import Tensor, nn, ops

F64 = torch.float64


def _r(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


def _check(layer, p, x, torch_fn, shape):
    """Eager and compiled values, and the gradient of sum(tanh(y)) for every param."""
    if not layer.has_params:
        @truestar.native
        def g(x: Tensor[tuple(shape) + ("f64",)]) -> Tensor["f64"]:
            return ops.sum(ops.tanh(layer(x)))
        want = torch_fn(p, x)
        assert rel(layer(x), want) < TOL
        assert rel(g(x), torch.tanh(want).sum()) < TOL
        return

    @truestar.native
    def f(p: layer.spec, x: Tensor[tuple(shape) + ("f64",)]) -> Tensor["f64"]:
        return ops.sum(ops.tanh(layer(p, x)))

    want = torch_fn(p, x)
    assert rel(layer(p, x), want) < TOL
    assert rel(f(p, x), torch.tanh(want).sum()) < TOL
    if p:
        assert grads_vs_torch(f, lambda q, x: torch.tanh(torch_fn(q, x)).sum(), p, x) < TOL


def test_group_norm_and_instance_norm():
    gn = nn.GroupNorm(3, 6, dtype="f64")
    p = gn.init(seed=1)
    p = {"w": p["w"] + _r(6, seed=2) * 0.3, "b": _r(6, seed=3)}
    _check(gn, p, _r(2, 6, 4, 5) * 2, lambda q, x: F.group_norm(x, 3, q["w"], q["b"], 1e-5), ("B", 6, 4, 5))
    inst = nn.InstanceNorm(4, affine=True, dtype="f64")
    q = {"w": _r(4, seed=4).abs() + 0.5, "b": _r(4, seed=5)}
    ref = torch.nn.InstanceNorm2d(4, affine=True).double()
    with torch.no_grad():
        ref.weight.copy_(q["w"])
        ref.bias.copy_(q["b"])
    x = _r(3, 4, 5, 6, seed=6)
    _check(inst, q, x, lambda q2, x: F.instance_norm(x, weight=q2["w"], bias=q2["b"], eps=1e-5), ("B", 4, 5, 6))
    assert rel(inst(q, x), ref(x)) < TOL


@pytest.mark.parametrize("groups,stride,pad", [(1, 1, 1), (2, 2, 0), (4, 1, 3)])
def test_conv1d_grouped_and_depthwise(groups, stride, pad):
    conv = nn.Conv1d(4, 8, 3, stride=stride, padding=pad, groups=groups, dtype="f64")
    p = conv.init(seed=7)
    _check(conv, p, _r(2, 4, 11, seed=8),
           lambda q, x: F.conv1d(x, q["w"], q["b"], stride=stride, padding=pad, groups=groups), ("B", 4, 11))


@pytest.mark.parametrize("groups", [1, 4])
def test_depthwise_and_grouped_conv2d(groups):
    conv = nn.Conv2d(4, 8, 3, stride=2, padding=1, groups=groups, dtype="f64")
    p = conv.init(seed=9)
    _check(conv, p, _r(2, 4, 9, 9, seed=10),
           lambda q, x: F.conv2d(x, q["w"], q["b"], stride=2, padding=1, groups=groups), ("B", 4, 9, 9))


@pytest.mark.parametrize("stride,pad,out_pad,groups", [(1, 0, 0, 1), (2, 1, 1, 1), (2, 1, 0, 2), (3, 2, 2, 1)])
def test_conv_transpose2d(stride, pad, out_pad, groups):
    conv = nn.ConvTranspose2d(4, 6, 3, stride=stride, padding=pad, output_padding=out_pad, groups=groups, dtype="f64")
    p = conv.init(seed=11)
    ref = torch.nn.ConvTranspose2d(4, 6, 3, stride=stride, padding=pad, output_padding=out_pad, groups=groups).double()
    assert tuple(ref.weight.shape) == tuple(p["w"].shape)
    _check(conv, p, _r(2, 4, 5, 6, seed=12),
           lambda q, x: F.conv_transpose2d(x, q["w"], q["b"], stride=stride, padding=pad, output_padding=out_pad,
                                           groups=groups), ("B", 4, 5, 6))


@pytest.mark.parametrize("mode,align,scale", [("nearest", False, 2), ("nearest", False, 3), ("bilinear", False, 2),
                                              ("bilinear", True, 2), ("bilinear", False, 3)])
def test_upsample(mode, align, scale):
    up = nn.Upsample(scale, mode, align)
    x = _r(2, 3, 4, 5, seed=13)
    kw = {} if mode == "nearest" else {"align_corners": align}
    ref = torch.nn.Upsample(scale_factor=scale, mode=mode, **kw)
    _check(up, {}, x, lambda q, x: ref(x), ("B", 3, 4, 5))
    # and its gradient with respect to x
    @truestar.native
    def f(x: Tensor["B", 3, 4, 5, "f64"]) -> Tensor["f64"]:
        return ops.sum(ops.tanh(up(x)) * ops.tanh(up(x)))
    val, g = truestar.grad(f, wrt="x")(x)
    xt = x.clone().requires_grad_()
    (torch.tanh(ref(xt)) ** 2).sum().backward()
    assert rel(g, xt.grad) < TOL


def test_pixel_shuffle():
    ps = nn.PixelShuffle(2)
    _check(ps, {}, _r(2, 8, 3, 4, seed=14), lambda q, x: F.pixel_shuffle(x, 2), ("B", 8, 3, 4))


_TORCH_NAMES = {"w_ih": "weight_ih_l0", "w_hh": "weight_hh_l0", "b_ih": "bias_ih_l0", "b_hh": "bias_hh_l0"}


def _copy_rnn(ref, p):
    with torch.no_grad():
        for k, v in _TORCH_NAMES.items():
            getattr(ref, v).copy_(p[k])


@pytest.mark.parametrize("kind", ["LSTM", "GRU"])
def test_recurrent_layers_over_a_symbolic_length(kind):
    layer_ = getattr(nn, kind)(4, 5, dtype="f64")
    p = layer_.init(seed=15)
    ref = getattr(torch.nn, kind)(4, 5, batch_first=True).double()
    _copy_rnn(ref, p)

    def torch_fn(q, x):
        out = torch.func.functional_call(ref, {_TORCH_NAMES[k]: v for k, v in q.items()}, (x,))
        return out[0]

    @truestar.native
    def f(p: layer_.spec, x: Tensor["B", "T", 4, "f64"]) -> Tensor["f64"]:
        out = layer_(p, x)
        return ops.sum(ops.tanh(out[0]))

    for steps in (1, 3, 7):                        # one compile, every length
        x = _r(2, steps, 4, seed=steps)
        want = torch_fn(p, x)
        assert rel(layer_(p, x)[0], want) < TOL
        assert rel(f(p, x), torch.tanh(want).sum()) < TOL
    assert grads_vs_torch(f, lambda q, x: torch.tanh(torch_fn(q, x)).sum(), p, _r(3, 6, 4, seed=16)) < TOL
    assert "loop(" in str(f.program)        # an IR loop over the symbolic T, not an unrolled one


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_embedding_bag(mode):
    bag = nn.EmbeddingBag(10, 4, mode=mode, dtype="f64")
    p = bag.init(seed=17)
    ids = torch.tensor([[1, 4, 4], [9, 0, 2]])
    ref = torch.nn.EmbeddingBag(10, 4, mode=mode).double()

    def torch_fn(q, ids):
        return torch.func.functional_call(ref, {"weight": q["w"]}, (ids,))

    @truestar.native
    def f(p: bag.spec, ids: Tensor["B", 3, "i64"]) -> Tensor["f64"]:
        return ops.sum(ops.tanh(bag(p, ids)))

    assert rel(bag(p, ids), torch_fn(p, ids)) < TOL
    assert rel(f(p, ids), torch.tanh(torch_fn(p, ids)).sum()) < TOL
    assert grads_vs_torch(f, lambda q, ids: torch.tanh(torch_fn(q, ids)).sum(), p, ids) < TOL


def test_rotary_matches_llama_rotary_code():
    transformers = pytest.importorskip("transformers")
    from transformers.models.llama import modeling_llama as ml
    cfg = transformers.LlamaConfig(hidden_size=32, num_attention_heads=4, max_position_embeddings=64, rope_theta=500.0)
    emb = ml.LlamaRotaryEmbedding(cfg)
    rot = nn.Rotary(8, base=500.0)
    x = _r(2, 4, 6, 8, seed=18)
    pos = torch.tensor([0, 1, 2, 5, 9, 33])
    with torch.no_grad():
        cos, sin = emb(x.float(), pos[None])
        # the Llama code computes its frequencies and angles in float32; the same formula in float64
        inv = 1.0 / (500.0 ** (torch.arange(0, 8, 2, dtype=F64) / 8))
        assert (inv - emb.inv_freq.double()).abs().max() < 1e-7
        ang = pos[:, None].double() * inv[None]
        cos64 = torch.cat([ang, ang], -1).cos()
        sin64 = torch.cat([ang, ang], -1).sin()
        want, _ = ml.apply_rotary_pos_emb(x, x, cos64[None], sin64[None])
    assert rel(rot(x, pos), want) < TOL
    assert (cos.double()[0] - cos64).abs().max() < 1e-6      # same angles as the Llama module (float32)

    @truestar.native
    def f(x: Tensor["B", 4, "T", 8, "f64"], pos: Tensor["T", "i64"]) -> Tensor["f64"]:
        return ops.sum(ops.tanh(rot(x, pos)))
    assert rel(f(x, pos), torch.tanh(want).sum()) < TOL


def test_alibi_matches_the_paper_definition():
    def slopes(n):   # Press et al. 2022: geometric with ratio 2^(-8/n), non powers of two interleaved
        def p2(k):
            s = 2 ** (-(2 ** -(math.log2(k) - 3)))
            return [s * s ** i for i in range(k)]
        if math.log2(n).is_integer():
            return p2(n)
        c = 2 ** math.floor(math.log2(n))
        return p2(c) + slopes(2 * c)[0::2][:n - c]
    for heads in (4, 8, 6, 12):
        al = nn.ALiBi(heads)
        bias = al(5, 7, "f64")
        m = torch.tensor(slopes(heads), dtype=F64)
        want = m[:, None, None] * (torch.arange(7)[None, None, :] - torch.arange(5)[None, :, None]).double()
        assert rel(bias, want) < TOL, heads


@pytest.mark.parametrize("act", ["silu", "gelu"])
def test_swiglu_and_geglu(act):
    mlp = (nn.SwiGLU if act == "silu" else nn.GEGLU)(6, 10, dtype="f64")
    p = mlp.init(seed=19)
    a = F.silu if act == "silu" else F.gelu
    _check(mlp, p, _r(3, 6, seed=20), lambda q, x: F.linear(a(F.linear(x, q["w1"])) * F.linear(x, q["w3"]), q["w2"]),
           ("B", 6))
