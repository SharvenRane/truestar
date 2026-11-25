"""The compiled executor: tensor IR lowered to Truestar kernels, on CPU and GPU.

Every result is compared with float64 (PyTorch in float64 for imported models, the IR reference
interpreter for hand built programs), on top of the checks every compiled program runs on its
first inputs. Tests at the end break the compiler on purpose and assert the checks catch it.
"""

import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

import truestar  # noqa: E402
from truestar import Tensor, ir, ops  # noqa: E402
from truestar import kernel as K  # noqa: E402
from truestar.compiler import (STATS, CompiledExecutor, FallbackNotAllowed, ProgramMismatchError,  # noqa: E402
                               compile_program, lower, runtime)
from truestar.frontend import set_default_executor  # noqa: E402
from truestar.ir.core import Builder, IRError, Node, Program, TensorType, UnsupportedOpError, Value  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


HAS_GPU = _has_gpu()
DEVICES = ["cpu"] + (["cuda"] if HAS_GPU else [])
CONFIGS = [(d, t) for d in DEVICES for t in ("float32", "float16")]
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and Triton")
TOL = {"float32": 2e-4, "float16": 3e-2}

g = torch.Generator().manual_seed(0)


def rnd(*shape, scale=1.0):
    return torch.randn(*shape, generator=g) * scale


def rel_err(out, ref):
    out, ref = out.detach().double().cpu(), ref.detach().double().cpu()
    assert out.shape == ref.shape, (out.shape, ref.shape)
    return (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


def check_program(prog, inputs, device, dtype, want=None, **kw):
    """Compile, run, and compare every output with float64. Returns the compiled program."""
    cp = compile_program(prog, device, dtype, **kw)
    outs = cp(list(inputs))
    want = want if want is not None else ir.run(prog, *inputs)
    assert len(outs) == len(want)
    for o, w, v in zip(outs, want, prog.outputs):
        assert o.device.type == device
        if v.type.dtype.is_floating_point:
            assert o.dtype == getattr(torch, dtype)
            assert rel_err(o, w) < TOL[dtype], f"{prog.name} %{v.name}: {rel_err(o, w):.3e}"
        else:
            assert o.dtype == v.type.dtype
            assert torch.equal(o.cpu(), w.cpu())
    assert cp.verified
    return cp


def import_model(model, example, batch_max=64):
    B = torch.export.Dim("B", min=1, max=batch_max)
    return ir.import_module(model.eval(), (example,), dynamic_shapes=({0: B},))


def randomize_norms(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.5, generator=g)
                m.running_var.uniform_(0.5, 2.0, generator=g)
                m.weight.uniform_(0.5, 1.5, generator=g)
                m.bias.normal_(0, 0.5, generator=g)
            if isinstance(m, nn.LayerNorm):
                m.weight.uniform_(0.5, 1.5, generator=g)
                m.bias.normal_(0, 0.5, generator=g)
    return model


# ----------------------------------------------------------------------------- imported models


@pytest.mark.parametrize("device,dtype", CONFIGS)
def test_mlp_matches_float64(device, dtype):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 8),
                          nn.Softmax(-1)).eval()
    prog = import_model(model, torch.randn(2, 16))
    cp = compile_program(prog, device, dtype)
    for batch in (5, 1, 9):
        x = rnd(batch, 16)
        y = cp([x])[0]
        assert y.dtype == getattr(torch, dtype) and y.device.type == device
        assert rel_err(y, model.double()(x.double())) < TOL[dtype]
        model.float()
    rep = cp.report
    assert rep.fallbacks == [] and rep.library == []
    assert sorted(rep.kernel_ops()) == ["gelu", "linear", "linear", "linear", "relu", "softmax"]
    assert cp.verified and cp.max_error < TOL[dtype]
    if device == "cuda":   # elementwise ops ride along as the epilogue of the matrix product
        assert [k["shape"] for k in rep.kernels] == ["contraction", "contraction", "contraction", "tile"]
        assert rep.kernels[0]["ops"] == ["linear", "gelu"]


@pytest.mark.parametrize("device,dtype", CONFIGS)
@pytest.mark.parametrize("variant", ["relu_post", "gelu_pre"])
def test_transformer_encoder_layer_matches_float64(device, dtype, variant):
    torch.manual_seed(1)
    layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True,
                                       activation="relu" if variant == "relu_post" else "gelu",
                                       norm_first=variant == "gelu_pre")
    layer = randomize_norms(layer).eval()
    prog = import_model(layer, torch.randn(2, 5, 32))
    cp = compile_program(prog, device, dtype)
    for batch in (3, 1):
        x = rnd(batch, 5, 32)
        assert rel_err(cp([x])[0], layer.double()(x.double())) < TOL[dtype]
        layer.float()
    rep = cp.report
    assert rep.fallbacks == [] and rep.library == []
    for op in ("linear", "attention", "layer_norm", "add"):
        assert op in rep.kernel_ops()
    shapes = [k["shape"] for k in rep.kernels]
    if device == "cuda":
        assert shapes.count("attention") == 1, shapes      # scores, softmax and product in one kernel
    else:
        assert set(shapes) == {"loops"}


RESNET_CONFIGS = [c for c in CONFIGS if c != ("cpu", "float16")]


@pytest.mark.parametrize("device,dtype", RESNET_CONFIGS)
def test_resnet18_matches_float64(device, dtype):
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(2)
    model = randomize_norms(torchvision.models.resnet18()).eval()
    prog = import_model(model, torch.randn(2, 3, 64, 64))
    cp = compile_program(prog, device, dtype)
    for batch in (2, 1):
        x = rnd(batch, 3, 64, 64)
        assert rel_err(cp([x])[0], model.double()(x.double())) < TOL[dtype]
        model.float()
    rep = cp.report
    assert rep.fallbacks == []
    # since 2026-11-24 convolution and pooling are Truestar kernels: no torch library kernel is left
    # (it was 22: 20 conv2d, max_pool2d, adaptive_avg_pool2d)
    assert rep.library == []
    kops = rep.kernel_ops()
    assert (kops.count("conv2d"), kops.count("max_pool2d"), kops.count("adaptive_avg_pool2d")) == (20, 1, 1)
    assert (kops.count("batch_norm"), kops.count("relu"), kops.count("add"), kops.count("linear")) == (20, 17, 8, 1)
    # every batch norm is the epilogue of its conv: on the GPU 20 conv kernels, 2 pooling kernels and
    # the linear; the CPU loop nest also takes each downsample conv into its block's residual kernel
    assert len(rep.kernels) == (23 if device == "cuda" else 20)
    if device == "cuda":
        shapes = sorted(e["shape"] for e in rep.kernels)
        assert shapes == ["contraction"] + ["conv"] * 20 + ["window"] * 2
    text = rep.to_text()
    assert "library kernels (torch): 0" in text and "fallbacks (float64 reference implementation on the CPU): 0" in text


# ----------------------------------------------------------------------------- every lowering rule


def _scalar(b, value):
    return b.param("c", torch.tensor(value, dtype=torch.float64))


def elementwise_zoo():
    b = Builder("elementwise_zoo")
    x = b.input("x", torch.float32, ("B", 6, 8))
    y = b.input("y", torch.float32, ("B", 1, 8))
    outs = []
    for op in ("neg", "exp", "tanh", "relu", "sigmoid", "silu", "hardsigmoid", "hardswish"):
        outs.append(b.op(op, x))
    pos = b.op("add", b.op("mul", x, x), _scalar(b, 0.5))
    outs += [b.op("sqrt", pos), b.op("rsqrt", pos)]
    outs.append(b.op("clamp", x, min=-0.5, max=0.75))
    outs.append(b.op("gelu", x, approximate="none"))
    outs.append(b.op("gelu", x, approximate="tanh"))
    for op in ("add", "sub", "mul"):
        outs.append(b.op(op, x, y))
    outs.append(b.op("div", x, b.op("add", b.op("mul", y, y), _scalar(b, 1.0))))
    lt, ge = b.op("lt", x, y), b.op("ge", x, _scalar(b, 0.0))
    eq, ne = b.op("eq", b.op("relu", x), _scalar(b, 0.0)), b.op("ne", b.op("relu", y), _scalar(b, 0.0))
    both = b.op("logical_and", lt, ge)
    either = b.op("logical_or", b.op("logical_not", lt), eq)
    outs.append(b.op("where", both, x, y))
    outs.append(b.op("where", either, b.op("neg", x), _scalar(b, 2.0)))
    outs += [b.op("gt", x, y), b.op("le", x, y), both, either, ne]          # bool outputs
    return b.build(outs)


def reduction_zoo():
    b = Builder("reduction_zoo")
    x = b.input("x", torch.float32, ("B", 6, 8))
    outs = [b.op("sum", x, dims=[2], keepdim=False), b.op("sum", x, dims=[1], keepdim=True),
            b.op("sum", x, dims=[1, 2], keepdim=False), b.op("sum", x, dims=[], keepdim=False),
            b.op("mean", x, dims=[0], keepdim=False), b.op("mean", x, dims=[0, 2], keepdim=True),
            b.op("mean", x, dims=[], keepdim=False),
            b.op("any", b.op("gt", x, _scalar(b, 1.5)), dims=[2], keepdim=False),
            b.op("any", b.op("gt", x, _scalar(b, 2.5)), dims=[], keepdim=False),
            b.op("softmax", x, dim=-1), b.op("softmax", x, dim=1),
            b.op("layer_norm", x, None, None, normalized_shape=[8], eps=1e-5)]
    w, bias = b.param("w", rnd(8)), b.param("bias", rnd(8))
    outs.append(b.op("layer_norm", b.op("mul", x, _scalar(b, 3.0)), w, bias, normalized_shape=[8], eps=1e-3))
    return b.build(outs)


def linear_algebra_zoo():
    b = Builder("linear_algebra_zoo")
    x = b.input("x", torch.float32, ("B", 6, 8))
    m = b.input("m", torch.float32, (8, 5))
    stats = b.input("stats", torch.float32, (4, 6))           # batch norm on non constant statistics
    outs = [b.op("matmul", x, m)]
    outs.append(b.op("matmul", b.op("permute", x, dims=[0, 2, 1]), x))             # batched, both dynamic
    outs.append(b.op("linear", x, b.param("lw", rnd(7, 8)), b.param("lb", rnd(7))))
    outs.append(b.op("linear", x, b.param("lw2", rnd(3, 8)), None))
    bn = [b.param(n, t) for n, t in (("bw", rnd(6)), ("bb", rnd(6)), ("bm", rnd(6)), ("bv", torch.rand(6, generator=g) + 0.5))]
    outs.append(b.op("batch_norm", x, *bn, eps=1e-5))                               # constants: folded
    var = b.op("add", b.op("mul", b.op("select", stats, dim=0, index=3), b.op("select", stats, dim=0, index=3)),
               _scalar(b, 0.5))
    outs.append(b.op("batch_norm", x, b.op("select", stats, dim=0, index=0), b.op("select", stats, dim=0, index=1),
                     b.op("select", stats, dim=0, index=2), var, eps=1e-5))
    return b.build(outs)


def layout_zoo():
    b = Builder("layout_zoo")
    x = b.input("x", torch.float32, ("B", 6, 8))
    h = b.op("relu", x)
    outs = [b.op("slice", h, dim=2, start=1, end=7, step=2),
            b.op("cat", b.op("tanh", x), h, dim=1),
            b.op("expand", b.op("sum", h, dims=[1], keepdim=True), shape=[x.type.shape[0], 6, 8]),
            b.op("select", b.op("permute", h, dims=[2, 0, 1]), dim=0, index=3),
            b.op("reshape", h, shape=[x.type.shape[0], 48]),
            b.op("mul", b.op("full", shape=[x.type.shape[0], 6, 8], value=1.5, dtype=torch.float32), h),
            b.op("add", b.op("full_like", h, value=-2.0), h),
            b.op("mul", b.op("arange", start=0, end=8, step=1, dtype=torch.float32), h),
            b.op("reshape", x, shape=[x.type.shape[0], 6, 8])]          # a view of an input: must not alias
    return b.build(outs)


ZOOS = {"elementwise": (elementwise_zoo, lambda B: [rnd(B, 6, 8), rnd(B, 1, 8)]),
        "reductions": (reduction_zoo, lambda B: [rnd(B, 6, 8)]),
        "linear_algebra": (linear_algebra_zoo, lambda B: [rnd(B, 6, 8), rnd(8, 5), rnd(4, 6)]),
        "layout": (layout_zoo, lambda B: [rnd(B, 6, 8)])}


@pytest.mark.parametrize("device,dtype", RESNET_CONFIGS)     # half precision on the CPU is covered by the models
@pytest.mark.parametrize("zoo", sorted(ZOOS))
def test_every_lowering_rule_matches_the_reference(device, dtype, zoo):
    build, make = ZOOS[zoo]
    prog = build()
    cp = check_program(prog, make(4), device, dtype)
    assert cp.report.fallbacks == [], cp.report.to_text()
    outs = cp(make(1))                       # another batch size through the same compiled program
    assert len(outs) == len(prog.outputs)


def native_want(fn, *inputs):
    return fn.python(*[t.double() if t.is_floating_point() else t for t in inputs])


def attention_program(kind):
    b = Builder(f"attention_{kind}")
    q = b.input("q", torch.float32, ("B", 2, 5, 8))
    k = b.input("k", torch.float32, ("B", 2, 7, 8))
    v = b.input("v", torch.float32, ("B", 2, 7, 6))
    mask, attrs = None, dict(scale=None, is_causal=False)
    if kind == "bool_mask":
        keep = torch.rand(5, 7, generator=g) > 0.3
        keep[:, 0] = True
        mask = b.param("mask", keep)
    elif kind == "float_mask":
        mask = b.param("mask", rnd(5, 7))
    elif kind == "scaled":
        attrs["scale"] = 0.3
    elif kind == "causal":
        attrs["is_causal"] = True
    return b.build([b.op("attention", q, k, v, mask, **attrs)])


@pytest.mark.parametrize("device,dtype", RESNET_CONFIGS)
@pytest.mark.parametrize("kind", ["plain", "causal", "bool_mask", "float_mask", "scaled"])
def test_attention_variants(device, dtype, kind):
    prog = attention_program(kind)
    cp = check_program(prog, [rnd(3, 2, 5, 8), rnd(3, 2, 7, 8), rnd(3, 2, 7, 6)], device, dtype)
    assert cp.report.fallbacks == []
    if device == "cuda":
        assert [k["shape"] for k in cp.report.kernels] == ["attention"]


@pytest.mark.parametrize("device", DEVICES)
def test_attention_written_by_hand_fuses_without_a_template(device):
    """Differential attention, (softmax(q1 k1) - c softmax(q2 k2)) v, written with plain ops in the
    native front end. No rule knows this layer; on the GPU it still becomes one attention kernel."""
    ex = CompiledExecutor(device)

    @truestar.function(executor=ex)
    def differential(q1: Tensor["B", 2, "S", 8], k1: Tensor["B", 2, "S", 8], q2: Tensor["B", 2, "S", 8],
                     k2: Tensor["B", 2, "S", 8], v: Tensor["B", 2, "S", 16]) -> Tensor["B", 2, "S", 16]:
        a1 = ops.softmax(q1 @ k1.transpose(-1, -2) * 0.35, dim=-1)
        a2 = ops.softmax(q2 @ k2.transpose(-1, -2) * 0.35, dim=-1)
        return (a1 - 0.6 * a2) @ v

    for batch, seq in ((2, 5), (1, 9), (3, 33)):
        args = [rnd(batch, 2, seq, 8) for _ in range(4)] + [rnd(batch, 2, seq, 16)]
        assert rel_err(differential(*args), native_want(differential, *args)) < TOL["float32"]
    rep = ex.last.report
    assert ex.compile_count == 1 and rep.fallbacks == []
    if device == "cuda":
        assert [k["shape"] for k in rep.kernels] == ["attention"], rep.to_text()
        assert sorted(rep.kernels[0]["ops"]) == ["matmul", "matmul", "matmul", "mul", "mul", "mul", "softmax",
                                                 "softmax", "sub"]


# ----------------------------------------------------------------------------- control flow


def make_rnn(executor):
    @truestar.function(executor=executor)
    def rnn(x: Tensor["B", "T", 16], wx: Tensor[16, 32], wh: Tensor[32, 32]) -> Tensor["B", 32]:
        h = ops.zeros((x.shape[0], 32))
        for t in range(x.shape[1]):          # a loop over a symbolic length
            h = ops.tanh(x[:, t] @ wx + h @ wh)
        if h.sum() > 0:                      # a branch on tensor data
            h = ops.relu(h)
        else:
            h = -h * 0.5
        return h / x.shape[1]
    return rnn


@pytest.mark.parametrize("device,dtype", CONFIGS)
def test_native_loop_and_branch(device, dtype):
    ex = CompiledExecutor(device, dtype)
    rnn = make_rnn(ex)
    wx, wh = rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2)
    taken = set()
    for batch, steps in ((4, 9), (1, 1), (7, 17), (2, 3)):
        x = rnd(batch, steps, 16)
        for sign in (1.0, -1.0):
            got = rnn(sign * x, wx, wh)
            want = ir.run(rnn.program, sign * x, wx, wh)[0]
            assert rel_err(got, want) < TOL[dtype]
            taken.add(bool(ir.run(rnn.program, sign * x, wx, wh, return_all=True)[1]["gt"]))
    assert taken == {True, False}            # both branches really ran
    assert ex.compile_count == 1
    rep = ex.last.report
    assert rep.fallbacks == [] and rep.counts()["loop"] == 1 and rep.counts()["cond"] == 1
    assert {"matmul", "tanh", "sum", "gt", "relu", "neg", "div"} <= set(rep.kernel_ops())
    body = [k for k in rep.kernels if "body" in k["region"]]
    assert body and all(k["region"] == "%h/body" for k in body)


@pytest.mark.parametrize("device,dtype", CONFIGS)
def test_native_while_loop(device, dtype):
    ex = CompiledExecutor(device, dtype)

    @truestar.function(executor=ex)
    def newton_sqrt(a: Tensor["B"]) -> Tensor["B"]:
        x = a * 0.5 + 0.5
        n = 0
        while ops.any(ops.abs(x * x - a) > 0.05):
            x = 0.5 * (x + a / x)
            n = n + 1
        return x

    for batch in (6, 1, 33):
        a = torch.rand(batch, generator=g) * 10 + 0.5
        got = newton_sqrt(a)
        # The loop stops once |x*x - a| <= 0.05 (tested in the run dtype), so check that rule rather
        # than the exact square root: for a small a, stopping within 0.05 can be several percent
        # away from sqrt(a) and still correct.
        eps = torch.finfo(getattr(torch, dtype)).eps
        residual = (got.cpu().double() ** 2 - a.double()).abs().max().item()
        assert residual <= 0.05 + 8 * eps * a.max().item(), residual
        assert rel_err(got, a.double().sqrt()) < 0.05 / (2 * a.min().item() ** 0.5) + 1e-2
    assert ex.compile_count == 1
    rep = ex.last.report
    assert rep.fallbacks == []
    assert [h["ops"] for h in rep.host] == [["add"]]         # the integer iteration counter, on the host
    assert {"mul", "sub", "where", "gt", "any", "div"} <= set(rep.kernel_ops())


@pytest.mark.parametrize("device", DEVICES)
def test_cond_with_two_results_and_nested_loop(device):
    ex = CompiledExecutor(device)

    @truestar.function(executor=ex)
    def f(x: Tensor["B", 8], w: Tensor[8, 8]):
        y = x @ w
        if y.mean() > 0.0:
            a = ops.relu(y)
            for i in range(x.shape[0]):
                a = a * 0.5 + ops.sigmoid(a)
            b = a.sum(dim=1)
        else:
            a = ops.tanh(y)
            b = -a.sum(dim=1)
        return a, b

    w = rnd(8, 8)
    for batch in (3, 5):
        for shift in (2.0, -2.0):
            x = rnd(batch, 8) * 0.1 + shift * torch.linalg.pinv(w).sum(0)
            got = f(x, w)
            want = ir.run(f.program, x, w)
            for o, r in zip(got, want):
                assert rel_err(o, r) < TOL["float32"]
    assert ex.compile_count == 1 and ex.last.report.fallbacks == []


# ----------------------------------------------------------------------------- compile counting


@pytest.mark.parametrize("device", DEVICES)
def test_symbolic_batch_compiles_once(device):
    torch.manual_seed(3)
    layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True).eval()
    prog = import_model(layer, torch.randn(2, 5, 32))
    before = dict(STATS)
    cp = compile_program(prog, device)
    after_compile = dict(STATS)
    assert after_compile["programs"] == before["programs"] + 1
    cp([rnd(1, 5, 32)])
    cache = (K.stats.misses, K.stats.disk_hits, K.stats.memory_hits)
    for batch in (1, 7, 64):
        x = rnd(batch, 5, 32)
        assert rel_err(cp([x])[0], layer.double()(x.double())) < TOL["float32"]
        layer.float()
    assert dict(STATS) == after_compile                       # no program, region or kernel was compiled again
    assert (K.stats.misses, K.stats.disk_hits, K.stats.memory_hits) == cache   # the kernel cache was not even asked
    with pytest.raises(ir.ShapeError):
        cp([rnd(65, 5, 32)])                                  # outside the declared range: an error, not a recompile


@pytest.mark.parametrize("device", DEVICES)
def test_loop_lengths_and_branches_compile_once(device):
    ex = CompiledExecutor(device)
    rnn = make_rnn(ex)
    wx, wh = rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2)
    rnn(rnd(2, 2, 16), wx, wh)
    rnn(-rnd(2, 2, 16), wx, wh)
    stats = dict(STATS)
    for batch, steps in ((1, 1), (7, 5), (64, 12), (3, 40)):
        rnn(rnd(batch, steps, 16), wx, wh)
    assert dict(STATS) == stats and ex.compile_count == 1 and rnn.compile_count == 1


def test_an_edit_recompiles_only_the_kernels_that_changed():
    def build(eps):
        b = Builder("edit")
        x = b.input("x", torch.float32, ("B", 12))
        h = b.op("relu", b.op("linear", x, b.param("w", torch.ones(12, 12)), None))
        h = b.op("layer_norm", h, None, None, normalized_shape=[12], eps=eps)
        return b.build([b.op("softmax", h, dim=-1)])

    first = compile_program(build(1e-5), "cpu")
    assert first.num_kernels == 3
    base = K.stats.misses
    compile_program(build(1e-5), "cpu")
    assert K.stats.misses == base                             # same program: nothing generated
    compile_program(build(1e-3), "cpu")
    assert K.stats.misses == base + 1                         # only the layer norm kernel changed


def test_launch_configs_and_tuning_keys_do_not_follow_the_batch():
    """Block sizes are Triton compile time constants. For a symbolic batch neither the default
    config, nor a tuning candidate, nor the autotuner's cache key may depend on the batch size, or a
    new batch size would compile a new Triton variant of a kernel Truestar compiled once."""
    from truestar.kernel import ir as kir, tune
    from truestar.kernel import triton_backend as tb
    B, N = K.sym("B"), K.sym("N")
    x = K.buffer("x", (B, 512))
    i, j = K.axis("i", B), K.axis("j", 512)
    pointwise = tb.plan(kir.canonicalize(K.kernel("p", [K.output("y", (i, j), K.tanh(x[i, j]))]))[0])
    reduced = tb.plan(kir.canonicalize(K.kernel("r", [K.output("y", (i, j), K.softmax_weight(x[i, j], over=j))]))[0])
    for plan in (pointwise, reduced):
        assert tb.batch_symbols(plan) == ["s0"]
        configs = [tb.default_config(plan, {"s0": b}) for b in (1, 7, 64, 4096)]
        assert all(c == configs[0] for c in configs)
        cands = [tb.candidates(plan, {"s0": b}) for b in (1, 7, 64, 4096)]
        assert all(c == cands[0] for c in cands)
        assert all(tb.check_config(plan, {"s0": 64}, c) is None for c in cands[0])
        keys = {tune.tune_key("k", {"s0": b}, ["float32"], "gpu", ignore=tb.batch_symbols(plan)) for b in (1, 7, 64)}
        assert len(keys) == 1
    # a static row count still follows the rows, and a symbolic lane axis still has its buckets
    xs = K.buffer("x", (4, N))
    a0, a1 = K.axis("i", 4), K.axis("j", N)
    lanes = tb.plan(kir.canonicalize(K.kernel("s", [K.output("y", (a0, a1), K.tanh(xs[a0, a1]))]))[0])
    assert tb.batch_symbols(lanes) == []
    assert tb.default_config(lanes, {"s0": 64})["BR"] == 4
    assert tune.tune_key("k", {"s0": 64}, ["float32"], "gpu") != tune.tune_key("k", {"s0": 1024}, ["float32"], "gpu")


# ----------------------------------------------------------------------------- fallbacks are listed


def wide_layer_norm():
    b = Builder("wide_layer_norm")
    x = b.input("x", torch.float32, ("B", 4, 6))
    h = b.op("layer_norm", b.op("tanh", x), None, None, normalized_shape=[4, 6], eps=1e-5)
    return b.build([b.op("mul", h, h)])


@pytest.mark.parametrize("device,dtype", CONFIGS)
def test_fallback_is_correct_and_reported(device, dtype):
    if device == "cpu":
        pytest.skip("the CPU loop nest accepts this op; the GPU has no loop shape for it")
    prog = wide_layer_norm()
    cp = check_program(prog, [rnd(3, 4, 6)], device, dtype)
    falls = cp.report.fallbacks
    assert [f["ops"] for f in falls] == [["layer_norm"]]
    assert "more than the last dim" in falls[0]["reason"]
    assert falls[0]["path"] == "float64 reference implementation on the CPU"
    assert "FALLBACK layer_norm %layer_norm" in cp.report.to_text()
    assert cp.report.to_dict()["fallbacks"][0]["name"] == "%layer_norm"
    assert set(cp.report.kernel_ops()) == {"tanh", "mul"}     # the rest still runs in kernels
    with pytest.raises(FallbackNotAllowed, match="layer_norm"):
        compile_program(prog, device, dtype, allow_fallback=False)


@pytest.mark.parametrize("device", DEVICES)
def test_op_without_a_rule_falls_back_per_op_and_is_listed(device, monkeypatch):
    monkeypatch.delitem(lower.RULES, "tanh")
    b = Builder("no_tanh_rule")
    x = b.input("x", torch.float32, ("B", 8))
    prog = b.build([b.op("relu", b.op("tanh", b.op("mul", x, x)))])
    cp = check_program(prog, [rnd(5, 8)], device, "float32")
    assert [(f["ops"], f["reason"]) for f in cp.report.fallbacks] == [(["tanh"], "tanh has no kernel lowering yet")]
    assert sorted(cp.report.kernel_ops()) == ["mul", "relu"]
    assert cp.report.counts()["fallback"] == 1


def test_integer_tensor_work_is_listed_not_hidden():
    b = Builder("ints")
    x = b.input("x", torch.float32, ("B", 8))
    idx = b.op("arange", start=0, end=8, step=1, dtype=torch.int64)
    bumped = b.op("add", idx, idx)                            # integer tensor arithmetic: no kernel type
    prog = b.build([b.op("mul", x, bumped)])
    cp = check_program(prog, [rnd(3, 8)], "cpu", "float32")
    # since 2026-11-24 integer arithmetic runs on the device as a declared library step (exact, checked
    # with zero tolerance), still listed in the report by its op and implementation
    assert cp.report.fallbacks == []
    assert [(e["ops"], e["impl"]) for e in cp.report.library] == [(["add"], "torch integer add")]
    assert cp.report.counts()["adapter"] == 1
    # integer cumsum joined that list with the op coverage work: a listed device step, not a fallback
    b = Builder("ints2")
    x = b.input("x", torch.float32, ("B", 8))
    idx = b.op("arange", start=0, end=8, step=1, dtype=torch.int64)
    prog = b.build([b.op("mul", x, b.op("cumsum", idx, dim=0))])
    cp = check_program(prog, [rnd(3, 8)], "cpu", "float32")
    assert cp.report.fallbacks == []
    assert [(e["ops"], e["impl"]) for e in cp.report.library] == [(["cumsum"], "torch integer cumsum")]
    # an integer op outside that list still takes the named fallback
    b = Builder("ints3")
    x = b.input("x", torch.float32, ("B", 8))
    idx = b.op("arange", start=0, end=8, step=1, dtype=torch.int64)
    prog = b.build([b.op("mul", x, b.op("flip", idx, dims=[0]))])
    cp = check_program(prog, [rnd(3, 8)], "cpu", "float32")
    assert [f["ops"] for f in cp.report.fallbacks] == [["flip"]]


def test_op_with_no_lowering_and_no_reference_fails_loudly():
    x = Value("x", TensorType(torch.float32, (4,)), "input")
    y = Value("y", TensorType(torch.float32, (4,)), "op")
    prog = Program("unknown", [x], [], {}, [Node("frobnicate", (x,), {}, y)], [y])
    with pytest.raises(UnsupportedOpError, match="frobnicate"):
        compile_program(prog, "cpu")


def test_no_silent_device_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never falls back"):
        CompiledExecutor("cuda")
    with pytest.raises(ValueError, match="unknown device"):
        CompiledExecutor("mps")
    with pytest.raises(ValueError, match="float64 is the reference"):
        CompiledExecutor("cpu", dtype="float64")


# ----------------------------------------------------------------------------- the two seams


@pytest.mark.parametrize("device,dtype", CONFIGS)
def test_executor_seam_of_native_functions(device, dtype):
    ex = CompiledExecutor(device, dtype)

    @truestar.function(executor=ex)
    def layer(x: Tensor["B", 16], w: Tensor[16, 8], b: Tensor[8]) -> Tensor["B", 8]:
        return ops.softmax(ops.gelu(x @ w + b), dim=-1)

    assert layer.executor.name == f"compiled:{device}:{dtype}"
    w, b = rnd(16, 8), rnd(8)
    for batch in (1, 4, 9):
        x = rnd(batch, 16)
        got = layer(x, w, b)
        assert got.dtype == getattr(torch, dtype) and got.device.type == device
        assert rel_err(got, native_want(layer, x, w, b)) < TOL[dtype]
    assert ex.compile_count == 1
    with pytest.raises(truestar.frontend.InputError):
        layer(rnd(3, 15), w, b)


def test_default_executor_can_be_the_compiled_one():
    old = set_default_executor(CompiledExecutor("cpu"))
    try:
        @truestar.function
        def f(x: Tensor["B", 4]):
            return ops.relu(x) * 2.0 + 1.0

        x = rnd(3, 4)
        got = f(x)
        assert got.dtype == torch.float32 and rel_err(got, torch.relu(x.double()) * 2 + 1) < 1e-6
    finally:
        set_default_executor(old)


@pytest.mark.parametrize("name", ["cpu", "cpu:float16"] + (["cuda", "cuda:float16"] if HAS_GPU else []))
def test_export_runner_seam(tmp_path, name):
    from truestar import export
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(12, 24), nn.SiLU(), nn.Linear(24, 6)).eval()
    path = str(tmp_path / "mlp.tsx")
    export.export(model, (torch.randn(2, 12),), path)
    reference = export.load(path)
    compiled = export.load(path, executor=name)
    for batch in (2, 7, 1):
        x = rnd(batch, 12)
        got, want = compiled(x)[0], reference(x)[0]
        assert got.dtype == torch.float32                      # the runner casts to the declared dtype
        assert got.device.type == name.split(":")[0]
        assert rel_err(got, want) < TOL["float16" if "16" in name else "float32"]
    from truestar.export import runner
    cp = runner.EXECUTORS[name].compiled(compiled.program)
    assert cp.calls == 3 and cp.verified and cp.report.fallbacks == []
    with pytest.raises(ValueError, match="unknown executor"):
        export.load(path, executor="tpu")


def test_outputs_never_alias_inputs_or_weights():
    b = Builder("alias")
    x = b.input("x", torch.float32, ("B", 6))
    w = b.param("w", torch.arange(6.0))
    prog = b.build([b.op("reshape", x, shape=[x.type.shape[0], 3, 2]), b.op("expand", w, shape=[2, 6])])
    cp = compile_program(prog, "cpu")
    x0 = rnd(2, 6)
    keep = x0.clone()
    out_x, out_w = cp([x0])
    out_x.zero_()
    out_w.fill_(7.0)
    assert torch.equal(x0, keep)
    assert torch.equal(cp([x0])[1], torch.arange(6.0).expand(2, 6))


def test_report_is_machine_readable():
    cp = compile_program(wide_layer_norm(), "cpu")
    d = cp.report.to_dict()
    assert d["schema"] == "truestar.compiler.report/1" and d["device"] == "cpu" and d["dtype"] == "float32"
    assert d["counts"]["kernel"] == cp.num_kernels and isinstance(d["entries"], list)
    import json
    json.dumps(d)


# ----------------------------------------------------------------------------- the checks catch a broken compiler
#
# Each test breaks one part of the compiler and asserts that the built in check raises, names the
# region and the op, and that the same program is fine once the break is removed.


def _small_mlp():
    b = Builder("guarded")
    x = b.input("x", torch.float32, ("B", 8))
    h = b.op("linear", x, b.param("w1", rnd(16, 8)), b.param("b1", rnd(16)))
    h = b.op("relu", h)
    h = b.op("linear", h, b.param("w2", rnd(4, 16)), None)
    return b.build([b.op("softmax", h, dim=-1)])


def _leaky(L, node):
    """A wrong relu: max(x, 0.1 x)."""
    x = L.operand(node.inputs[0])
    return L.new(L.shape_of(node.output), "f", lambda idx, lo: K.maximum(lo(x, idx), lo(x, idx) * 0.1), (x,),
                 node=node)


@pytest.mark.parametrize("device,dtype", CONFIGS)
def test_a_broken_lowering_rule_is_caught_and_named(device, dtype, monkeypatch):
    prog = _small_mlp()
    x = rnd(5, 8)
    check_program(prog, [x], device, dtype)                    # sound compiler: passes
    monkeypatch.setitem(lower.RULES, "relu", _leaky)
    cp = compile_program(prog, device, dtype)
    with pytest.raises(ProgramMismatchError) as e:
        cp([x])
    err = e.value
    assert err.op == "relu" and err.program == "guarded" and err.region == "guarded"
    assert err.stage == "region" and "relu" in err.ops
    assert "relu (%relu" in str(err) and "was not returned" in str(err)
    assert not cp.verified
    with pytest.raises(ProgramMismatchError):                  # a retry is checked again, not waved through
        cp([x])


@pytest.mark.parametrize("device", DEVICES)
def test_a_broken_rule_in_a_fused_region_is_isolated_to_its_op(device, monkeypatch):
    def bad_tanh(L, node):
        x = L.operand(node.inputs[0])
        return L.new(L.shape_of(node.output), "f", lambda idx, lo: K.tanh(lo(x, idx) * 1.05), (x,), node=node)

    b = Builder("fused")
    x = b.input("x", torch.float32, ("B", 8))
    h = b.op("mul", b.op("sigmoid", b.op("tanh", b.op("add", x, x))), x)
    prog = b.build([h])
    monkeypatch.setitem(lower.RULES, "tanh", bad_tanh)
    cp = compile_program(prog, device)
    assert cp.num_kernels == 1 and cp.report.kernels[0]["ops"] == ["add", "tanh", "sigmoid", "mul"]
    with pytest.raises(ProgramMismatchError) as e:
        cp([rnd(4, 8)])
    assert e.value.op == "tanh" and e.value.ops == ["add", "tanh", "sigmoid", "mul"]
    assert "wrong by itself" in str(e.value)


@pytest.mark.parametrize("device", DEVICES)
def test_a_broken_kernel_code_generator_is_caught(device):
    prog = _small_mlp()
    x = rnd(5, 8)
    wrong = (lambda v: f"tl.exp(({v}) * 0.97)") if device == "cuda" else (lambda v: f"math.exp(({v}) * 0.97)")
    cp = compile_program(prog, device, _break={"exp": wrong})
    with pytest.raises(ProgramMismatchError) as e:
        cp([x])
    assert e.value.op == "softmax" and e.value.stage == "kernel" and e.value.region == "guarded"
    assert isinstance(e.value.__cause__, K.KernelMismatchError)
    check_program(prog, [x], device, "float32")


@pytest.mark.parametrize("device", DEVICES)
def test_a_broken_rule_inside_a_loop_body_names_the_region(device, monkeypatch):
    def bad_matmul(L, node):
        lv = lower._matmul_lv(L, node, L.operand(node.inputs[0]), L.operand(node.inputs[1]),
                              L.shape_of(node.output))
        fn = lv.fn
        lv.fn = lambda idx, lo: fn(idx, lo) * 1.01
        return lv

    monkeypatch.setitem(lower.RULES, "matmul", bad_matmul)
    rnn = make_rnn(CompiledExecutor(device))
    with pytest.raises(ProgramMismatchError) as e:
        rnn(rnd(3, 4, 16), rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2))
    assert e.value.op == "matmul" and e.value.region == "rnn/loop %h/body" and e.value.program == "rnn"


@pytest.mark.parametrize("device", DEVICES)
def test_a_branch_is_checked_when_it_first_runs(device, monkeypatch):
    monkeypatch.setitem(lower.RULES, "neg", lambda L, node: _leaky(L, node))   # only the else branch negates
    ex = CompiledExecutor(device)
    rnn = make_rnn(ex)
    wx, wh = rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2)
    x = rnd(3, 4, 16)
    sign = 1.0 if bool(ir.run(rnn.program, x, wx, wh, return_all=True)[1]["gt"]) else -1.0
    rnn(sign * x, wx, wh)                                      # then branch: verified, the program is "verified"
    assert ex.last.verified
    with pytest.raises(ProgramMismatchError) as e:             # first run of the else branch, on a later call
        rnn(-sign * x, wx, wh)
    assert e.value.op == "neg" and e.value.region == "rnn/cond %h_1/else_branch"


@needs_gpu
def test_a_broken_library_kernel_is_caught(monkeypatch):
    b = Builder("conv")
    x = b.input("x", torch.float32, ("B", 3, 8, 8))
    h = b.op("conv2d", x, b.param("w", rnd(4, 3, 3, 3)), None, stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1)
    prog = b.build([b.op("relu", h)])
    inputs = [rnd(2, 3, 8, 8)]
    for device in DEVICES:     # conv2d is a Truestar kernel by default; the library runs it when asked
        check_program(prog, inputs, device, "float32", library=("conv2d",))
    real = runtime.F.conv2d
    monkeypatch.setattr(runtime.F, "conv2d", lambda *a, **k: real(*a, **k) * 1.001)
    for device in DEVICES:
        with pytest.raises(ProgramMismatchError) as e:
            compile_program(prog, device, library=("conv2d",))(inputs)
        assert e.value.op == "conv2d" and e.value.stage == "region"


def test_wrong_wiring_between_regions_is_caught_end_to_end():
    prog = layout_zoo()
    inputs = [rnd(4, 6, 8)]
    cp = compile_program(prog, "cpu")
    step = next(s for s in cp.plan.steps if s.kind == "layout" and s.nodes[0].op == "permute")
    real = step.fn
    step.fn = lambda attrs, args, shape: real(attrs, [args[0].flip(0)], shape)   # a layout step that moves data wrongly
    with pytest.raises(ProgramMismatchError) as e:
        cp(inputs)
    assert e.value.stage == "end_to_end" and e.value.op == "select"
    step.fn = real
    cp(inputs)
    assert cp.verified


def test_without_the_check_the_wrong_result_would_ship(monkeypatch):
    """Control for the tests above: the same broken rule returns a wrong answer when verification is
    turned off, so it is the built in check that catches it and nothing else."""
    prog = _small_mlp()
    x = rnd(5, 8)
    want = ir.run(prog, x)[0]
    monkeypatch.setitem(lower.RULES, "relu", _leaky)
    monkeypatch.setenv("TRUESTAR_VERIFY", "0")
    got = compile_program(prog, "cpu")([x])[0]
    assert rel_err(got, want) > 1e-3
    monkeypatch.setenv("TRUESTAR_VERIFY", "first")
    with pytest.raises(ProgramMismatchError):
        compile_program(prog, "cpu")([x])


def test_verify_all_checks_every_call(monkeypatch):
    prog = _small_mlp()
    cp = compile_program(prog, "cpu", verify="all")
    cp([rnd(3, 8)])
    spent = cp.verify_seconds
    cp([rnd(3, 8)])
    assert cp.verify_seconds > spent
    quiet = compile_program(prog, "cpu")
    quiet([rnd(3, 8)])
    spent = quiet.verify_seconds
    quiet([rnd(3, 8)])
    assert quiet.verify_seconds == spent


def test_a_loop_that_cannot_end_in_this_precision_says_so(monkeypatch):
    monkeypatch.setattr(runtime, "STALL_CHECK_FROM", 8)
    monkeypatch.setattr(runtime, "STALL_CHECK_EVERY", 8)

    @truestar.function(executor=CompiledExecutor("cpu", verify="off"))
    def stuck(x: Tensor["B"]) -> Tensor["B"]:
        while ops.any(x > 0.0):
            x = ops.relu(x)
        return x

    with pytest.raises(IRError, match="cannot terminate in float32"):
        stuck(torch.ones(3))
