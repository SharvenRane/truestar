"""Convolution and pooling as Truestar kernels: strided loads in the kernel IR, the conv (implicit
GEMM) and window loop shapes, the compiler rules and the gradient ops.

Every result is compared with float64 (PyTorch in float64, or the IR reference interpreter), on
top of the check every kernel and every compiled program runs on its first inputs. The last tests
break a rule or the code generator on purpose and assert the checks catch it.
"""

import math
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import ir  # noqa: E402
from truestar import kernel as K  # noqa: E402
from truestar.compiler import ProgramMismatchError, compile_program, lower  # noqa: E402
from truestar.compiler import conv_rules  # noqa: E402
from truestar.ir.autodiff import grad  # noqa: E402
from truestar.ir.core import Builder  # noqa: E402
from truestar.kernel import triton_backend as tb  # noqa: E402
from truestar.kernel import triton_conv  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


HAS_GPU = _has_gpu()
DEVICES = ["cpu"] + (["cuda"] if HAS_GPU else [])
CONFIGS = [(d, t) for d in DEVICES for t in ("float32", "float16")] + \
    ([("cuda", "bfloat16")] if HAS_GPU else [])
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and Triton")
TOL = {"float32": 2e-4, "float16": 3e-2, "bfloat16": 6e-2}
F64 = torch.float64
gen = torch.Generator().manual_seed(0)


def rnd(*shape, scale=1.0):
    return torch.randn(*shape, generator=gen) * scale


def rel_err(out, ref):
    out, ref = out.detach().double().cpu(), ref.detach().double().cpu()
    assert out.shape == ref.shape, (out.shape, ref.shape)
    return (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


def check(prog, inputs, device, dtype, **kw):
    cp = compile_program(prog, device, dtype, **kw)
    outs = cp(list(inputs))
    want = ir.run(prog, *[x.double() for x in inputs])
    for o, w in zip(outs, want):
        assert rel_err(o, w) < TOL[dtype], f"{prog.name}: {rel_err(o, w):.3e}"
    assert cp.verified
    return cp


# ----------------------------------------------------------------------------- kernel IR


def test_strided_index_builds_and_prints():
    i, u = K.axis("i", 4), K.axis("u", 3)
    s = K.strided([(i, 2), (u, 1)], -1)
    assert isinstance(s, K.Strided) and repr(s) == "(2*i + u - 1)"
    assert K.strided([(i, 1)]) == i                     # a plain axis stays an axis
    x = K.buffer("x", (8,))
    with pytest.raises(K.IRError, match="pad"):
        K.Load(x, (s,))                                  # may fall outside: a pad value is required
    ld = K.Load(x, (s,), pad=0.0)
    assert ld.pad == 0.0 and set(K.ir.free_axes(ld)) == {i, u}
    inside = K.Load(x, (K.strided([(i, 1), (u, 1)]),), pad=5.0)   # 0..5 always inside 8
    assert inside.pad is None
    assert K.Load(x, (K.strided([(i, 1)], 0, 2),), pad=0.0).pad == 0.0   # exact division may fail


def _conv_kernel(stride, pad, dil, C=5, O=7, H=9, W=8, KH=3, KW=2):
    N = K.sym("N")
    OH = (H + 2 * pad - dil * (KH - 1) - 1) // stride + 1
    OW = (W + 2 * pad - dil * (KW - 1) - 1) // stride + 1
    x, w = K.buffer("x", (N, C, H, W)), K.buffer("w", (O, C, KH, KW))
    n, o, i, j = K.axis("n", N), K.axis("o", O), K.axis("i", OH), K.axis("j", OW)
    c, u, v = K.axis("c", C), K.axis("u", KH), K.axis("v", KW)
    xl = K.Load(x, (n, c, K.strided([(i, stride), (u, dil)], -pad), K.strided([(j, stride), (v, dil)], -pad)),
                pad=0.0)
    e = K.reduce_sum(c, K.reduce_sum(u, K.reduce_sum(v, xl * w[o, c, u, v])))
    return K.kernel("conv", [K.output("y", (n, o, i, j), K.relu(e))])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stride,pad,dil", [(1, 1, 1), (2, 1, 1), (2, 2, 2), (3, 0, 1)])
def test_conv_kernel_matches_float64(device, stride, pad, dil):
    k = _conv_kernel(stride, pad, dil)
    if device == "cuda":
        assert tb.plan(k).kind == "conv"
    f = K.compile(k, device)
    x, w = rnd(3, 5, 9, 8).to(device), rnd(7, 5, 3, 2).to(device)
    y = f(x=x, w=w)                  # the kernel layer checks it against its float64 reference too
    ref = F.relu(F.conv2d(x.double().cpu(), w.double().cpu(), stride=stride, padding=pad, dilation=dil))
    assert rel_err(y, ref) < 2e-5


@pytest.mark.parametrize("device", DEVICES)
def test_exact_division_reads_only_where_it_divides(device):
    """y[h] = g[(h + 1) / 2] where exact: the gather form of a stride 2 transposed read."""
    g_ = K.buffer("g", (5,))
    h = K.axis("h", 10)
    k = K.kernel("up", [K.output("y", (h,), K.Load(g_, (K.strided([(h, 1)], 1, 2),), pad=-3.0))])
    gv = torch.arange(5, dtype=torch.float32).to(device) + 10
    y = K.compile(k, device)(g=gv).cpu()
    want = torch.tensor([10 + (t + 1) // 2 if (t + 1) % 2 == 0 and (t + 1) // 2 < 5 else -3.0
                         for t in range(10)])
    assert torch.equal(y, want)


@needs_gpu
def test_window_shape_takes_pooling_and_refuses_gemm_shaped_sums():
    N = K.sym("N")
    x = K.buffer("x", (N, 4, 9, 9))
    n, c, i, j, u, v = (K.axis(a, s) for a, s in (("n", N), ("c", 4), ("i", 4), ("j", 4), ("u", 3), ("v", 3)))
    ld = K.Load(x, (n, c, K.strided([(i, 2), (u, 1)], -1), K.strided([(j, 2), (v, 1)], -1)), pad=-math.inf)
    pool = K.kernel("pool", [K.output("y", (n, c, i, j), K.reduce_max(u, K.reduce_max(v, ld)))])
    assert tb.plan(pool).kind == "window"
    # a plain matmul written with a chain of sums is no convolution: no strided load, no new shape
    a, b = K.buffer("a", (8, 4, 4)), K.buffer("b", (4, 4, 16))
    m, nn_, r1, r2 = K.axis("m", 8), K.axis("n", 16), K.axis("r1", 4), K.axis("r2", 4)
    mm = K.kernel("mm", [K.output("y", (m, nn_), K.reduce_sum(r1, K.reduce_sum(r2, a[m, r1, r2] * b[r1, r2, nn_])))])
    with pytest.raises(K.Unschedulable):
        tb.plan(mm)
    with pytest.raises(K.Unschedulable, match="GEMM shaped"):
        triton_conv.plan_window(_two_convs())


def _two_convs():
    k1 = _conv_kernel(1, 1, 1)
    k2 = _conv_kernel(1, 1, 1)
    e1 = k1.outputs[0].expr
    e2 = K.ir.substitute(k2.outputs[0].expr, {})
    return K.kernel("two", [K.output("y", k1.outputs[0].axes, e1 + e2)])


# ----------------------------------------------------------------------------- compiler


def conv_prog(shape, wshape, bias=True, bn=False, act="relu", name="conv", **attrs):
    attrs = {"stride": 1, "padding": 0, "dilation": 1, "groups": 1, **attrs}
    b = Builder(name)
    x = b.input("x", torch.float32, ("B",) + tuple(shape[1:]))
    w = b.param("w", rnd(*wshape, scale=1.0 / math.sqrt(wshape[1] * wshape[2] * wshape[3])))
    bb = b.param("b", rnd(wshape[0])) if bias else None
    y = b.op("conv2d", x, w, bb, **attrs)
    if bn:
        o = wshape[0]
        y = b.op("batch_norm", y, b.param("g", rnd(o).abs() + 0.5), b.param("bt", rnd(o)),
                 b.param("m", rnd(o, scale=0.1)), b.param("vr", rnd(o).abs() + 0.5), training=False,
                 momentum=0.1, eps=1e-5)
    if act:
        y = b.op(act, y)
    return b.build([y])


CONV_CASES = {
    "3x3_pad": ((2, 6, 10, 9), (8, 6, 3, 3), dict(padding=1)),
    "stride2": ((2, 4, 11, 10), (8, 4, 3, 3), dict(stride=2, padding=1)),
    "7x7_stride2": ((1, 3, 20, 20), (16, 3, 7, 7), dict(stride=2, padding=3)),
    "dilated_mixed": ((2, 4, 12, 11), (6, 4, 3, 2), dict(stride=[2, 1], padding=[2, 1], dilation=[2, 2])),
    "1x1": ((2, 16, 7, 7), (32, 16, 1, 1), {}),
    "1x1_stride2": ((2, 16, 8, 8), (32, 16, 1, 1), dict(stride=2)),
    "groups": ((2, 8, 9, 9), (12, 4, 3, 3), dict(groups=2, padding=1)),
    "depthwise": ((2, 6, 10, 10), (6, 1, 3, 3), dict(groups=6, padding=1)),
    "depthwise_5x5_stride2": ((2, 6, 11, 11), (6, 1, 5, 5), dict(groups=6, padding=2, stride=2)),
    "depthwise_multiplier": ((2, 3, 8, 8), (6, 1, 3, 3), dict(groups=3, padding=1)),
}


@pytest.mark.parametrize("device,dtype", CONFIGS)
@pytest.mark.parametrize("case", sorted(CONV_CASES))
def test_conv2d_compiles_to_truestar_kernels(device, dtype, case):
    shape, wshape, attrs = CONV_CASES[case]
    prog = conv_prog(shape, wshape, bn=True, **attrs)
    cp = check(prog, [rnd(*shape)], device, dtype)
    rep = cp.report
    assert rep.library == [] and rep.fallbacks == []
    assert len(rep.kernels) == 1                     # conv, batch norm and relu: one kernel
    assert set(rep.kernels[0]["ops"]) == {"conv2d", "batch_norm", "relu"}
    if device == "cuda":
        want = "window" if attrs.get("groups", 1) > 1 else "conv"
        assert rep.kernels[0]["shape"] == want


@needs_gpu
def test_channels_last_input_and_symbolic_batch():
    """NHWC memory: kernels read any strides, so a channels_last input needs no copy; one compile
    serves batch 1, 3 and 5."""
    prog = conv_prog((2, 8, 9, 9), (16, 8, 3, 3), padding=1)
    cp = compile_program(prog, "cuda")
    for bs in (1, 3, 5):
        x = rnd(bs, 8, 9, 9).cuda().to(memory_format=torch.channels_last)
        assert not x.is_contiguous()
        out = cp([x])[0]
        assert rel_err(out, ir.run(prog, x.double().cpu())[0]) < 2e-4


POOL_CASES = {
    "max_3x3_s2_p1": ("max_pool2d", (2, 3, 11, 11), dict(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1],
                                                          dilation=[1, 1], ceil_mode=False)),
    "max_ceil_dilated": ("max_pool2d", (1, 2, 10, 9), dict(kernel_size=[2, 3], stride=[2, 2], padding=[1, 1],
                                                            dilation=[2, 1], ceil_mode=True)),
    "avg_include_pad": ("avg_pool2d", (2, 3, 9, 9), dict(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1],
                                                          ceil_mode=False, count_include_pad=True,
                                                          divisor_override=None)),
    "avg_exclude_pad_ceil": ("avg_pool2d", (2, 3, 10, 9), dict(kernel_size=[3, 2], stride=[2, 2], padding=[1, 1],
                                                               ceil_mode=True, count_include_pad=False,
                                                               divisor_override=None)),
    "adaptive_global": ("adaptive_avg_pool2d", (2, 5, 7, 7), dict(output_size=[1, 1])),
    "adaptive_uneven": ("adaptive_avg_pool2d", (2, 3, 7, 5), dict(output_size=[3, 2])),
}


@pytest.mark.parametrize("device,dtype", CONFIGS)
@pytest.mark.parametrize("case", sorted(POOL_CASES))
def test_pooling_compiles_to_truestar_kernels(device, dtype, case):
    op, shape, attrs = POOL_CASES[case]
    b = Builder(case)
    x = b.input("x", torch.float32, ("B",) + shape[1:])
    prog = b.build([b.op(op, b.op("tanh", x), **attrs)])
    cp = check(prog, [rnd(*shape)], device, dtype)
    assert cp.report.library == [] and cp.report.fallbacks == []
    if device == "cuda":
        assert [e["shape"] for e in cp.report.kernels if op in e["ops"]] == ["window"]


def test_library_kernels_only_when_asked():
    prog = conv_prog((2, 4, 8, 8), (4, 4, 3, 3), padding=1)
    for device in DEVICES:
        cp = check(prog, [rnd(2, 4, 8, 8)], device, "float32", library=("conv2d",))
        lib = cp.report.library
        assert [e["ops"] for e in lib] == [["conv2d"]] and lib[0]["reason"] == "asked for with library="
        assert "LIBRARY conv2d" in cp.report.to_text()
    with pytest.raises(ValueError, match="library="):
        compile_program(prog, "cpu", library=("relu",))


def test_a_rule_that_cannot_take_an_op_uses_the_library_and_says_why():
    b = Builder("sym_conv")
    x = b.input("x", torch.float32, ("B", 3, "H", 8))
    prog = b.build([b.op("conv2d", x, b.param("w", rnd(4, 3, 3, 3)), None, stride=1, padding=1, dilation=1,
                         groups=1)])
    cp = compile_program(prog, "cpu")
    lib = cp.report.library
    assert len(lib) == 1 and "fixed sizes" in lib[0]["reason"]
    out = cp([rnd(2, 3, 6, 8)])[0]
    assert out.shape == (2, 4, 6, 8)


# ----------------------------------------------------------------------------- gradients


GRAD_CASES = {
    "conv2d_input_grad": (lambda b, g, w: b.op("conv2d_input_grad", g, w, input_size=[9, 8], stride=[2, 1],
                                                padding=[2, 1], dilation=[2, 2], groups=2),
                          [(2, 6, 5, 8), (6, 2, 3, 2)]),
    "conv2d_input_grad_s2": (lambda b, g, w: b.op("conv2d_input_grad", g, w, input_size=[16, 16], stride=2,
                                                   padding=1, dilation=1, groups=1),
                             [(2, 8, 8, 8), (8, 6, 3, 3)]),
    "conv2d_input_grad_s2_k2": (lambda b, g, w: b.op("conv2d_input_grad", g, w, input_size=[16, 16], stride=2,
                                                      padding=0, dilation=1, groups=1),
                                [(2, 8, 8, 8), (8, 6, 2, 2)]),
    "conv2d_input_grad_s2_k4_p2": (lambda b, g, w: b.op("conv2d_input_grad", g, w, input_size=[16, 14], stride=2,
                                                         padding=2, dilation=1, groups=1),
                                   [(2, 8, 9, 8), (8, 6, 4, 4)]),
    "conv2d_weight_grad_wide": (lambda b, x, g: b.op("conv2d_weight_grad", x, g, kernel_size=[3, 3], stride=1,
                                                      padding=1, dilation=1, groups=1),
                                [(4, 128, 8, 8), (4, 4, 8, 8)]),
    "conv2d_weight_grad": (lambda b, x, g: b.op("conv2d_weight_grad", x, g, kernel_size=[3, 3], stride=2,
                                                 padding=1, dilation=1, groups=1),
                           [(2, 6, 16, 16), (2, 8, 8, 8)]),
    "conv2d_weight_grad_depthwise": (lambda b, x, g: b.op("conv2d_weight_grad", x, g, kernel_size=[2, 3],
                                                           stride=[2, 1], padding=[1, 0], dilation=[1, 2],
                                                           groups=3),
                                     [(2, 3, 7, 8), (2, 3, 4, 4)]),
    "max_pool2d_grad": (lambda b, x, g: b.op("max_pool2d_grad", x, g, kernel_size=3, stride=2, padding=1,
                                              dilation=1, ceil_mode=False),
                        [(2, 3, 9, 9), (2, 3, 5, 5)]),
}


@pytest.mark.parametrize("device,dtype", [c for c in CONFIGS if c[1] != "bfloat16"])
@pytest.mark.parametrize("case", sorted(GRAD_CASES))
def test_gradient_ops_compile_to_truestar_kernels(device, dtype, case):
    build, shapes = GRAD_CASES[case]
    b = Builder(case)
    vals = [b.input(f"x{i}", torch.float32, s) for i, s in enumerate(shapes)]
    prog = b.build([build(b, *vals)])
    inputs = [rnd(*s) for s in shapes]
    if case == "max_pool2d_grad":          # a clear winner in every window, so float16 picks the same tap
        n = math.prod(shapes[0])
        inputs[0] = (torch.randperm(n, generator=gen).float() * (4.0 / n)).reshape(shapes[0])
    cp = check(prog, inputs, device, dtype)
    assert cp.report.library == [] and cp.report.fallbacks == []


@needs_gpu
@pytest.mark.parametrize("case", ["conv2d_weight_grad_wide", "conv2d_weight_grad", "conv2d_input_grad_s2",
                                  "conv2d_input_grad_s2_k4_p2"])
def test_gradient_kernels_are_bitwise_deterministic(case):
    """The split weight gradient adds its partial sums in a fixed order and the parity input
    gradient drops only zero terms: two calls give the same bits, and a fresh compile too."""
    build, shapes = GRAD_CASES[case]
    b = Builder(case)
    vals = [b.input(f"x{i}", torch.float32, s) for i, s in enumerate(shapes)]
    prog = b.build([build(b, *vals)])
    inputs = [rnd(*s).cuda() for s in shapes]
    cp = compile_program(prog, "cuda")
    a1 = cp(list(inputs))[0].clone()
    a2 = cp(list(inputs))[0].clone()
    a3 = compile_program(prog, "cuda")(list(inputs))[0]
    assert torch.equal(a1, a2) and torch.equal(a1, a3)
    want = ir.run(prog, *[x.double().cpu() for x in inputs])[0]
    assert rel_err(a1, want) < TOL["float32"]


def test_training_conv_splits_depend_on_shapes_only():
    assert conv_rules.channels_last_splits(64 * 128, 32, 28, 28) == conv_rules.channels_last_splits(64 * 128, 32, 28, 28)
    sn, sr = conv_rules.channels_last_splits(64 * 128, 32, 28, 28)
    assert sn * sr <= conv_rules.WG_MAX_PARTS and 32 % sn == 0 and 28 % sr == 0
    assert conv_rules.channels_last_splits(512 * 512 * 9, 32, 7, 7) == (1, 1)


@needs_gpu
def test_resnet_block_backward_compiles_without_library_kernels():
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet18(num_classes=10).eval()
    B = torch.export.Dim("B", min=1, max=16)
    prog = ir.import_module(model.layer1, (torch.randn(2, 64, 16, 16),), dynamic_shapes=({0: B},))
    names = [n for n, _ in model.layer1.named_parameters() if ".conv" in n]
    gp = grad(prog, [prog.inputs[0].name] + names, seed="input")
    counts = gp.op_counts()
    assert counts["conv2d_input_grad"] == 4 and counts["conv2d_weight_grad"] == 4
    cp = compile_program(gp, "cuda")
    x, v = rnd(2, 64, 16, 16), rnd(2, 64, 16, 16)
    outs = cp([x, v])
    m = model.layer1.double()
    xr = x.double().requires_grad_()
    want = torch.autograd.grad(m(xr), [xr] + [dict(m.named_parameters())[n] for n in names], v.double())
    for o, w in zip(outs[1:], want):
        assert rel_err(o, w) < 2e-4
    assert cp.report.library == [] and cp.report.fallbacks == []


# ----------------------------------------------------------------------------- the checks catch breaks


def test_a_broken_conv_rule_is_caught(monkeypatch):
    """A conv rule that forgets the padding offset on one axis computes a shifted convolution."""
    prog = conv_prog((2, 4, 8, 8), (4, 4, 3, 3), padding=1, bias=False, act=None, name="shifted")
    real = conv_rules.aff

    def shifted(parts, offset=0, exact=1):
        return real(parts, 0 if offset == -1 else offset, exact)

    monkeypatch.setattr(conv_rules, "aff", shifted)
    for device in DEVICES:
        with pytest.raises(ProgramMismatchError) as e:
            compile_program(prog, device)([rnd(2, 4, 8, 8)])
        assert e.value.op == "conv2d"


@needs_gpu
def test_a_strided_load_that_ignores_exact_division_is_caught(monkeypatch):
    """Code generation that drops the exact division reads g at the undivided index: the kernel
    layer's float64 check catches it."""
    real = triton_conv.emit_strided_load

    def sloppy(em, n, ctx, raw=False):
        idx = tuple(K.Strided(a.terms, a.offset, 1) if isinstance(a, K.Strided) else a for a in n.index)
        tampered = K.Load(n.buffer, idx, n.pad if n.pad is not None else 0.0)
        return real(em, tampered, ctx, raw)

    monkeypatch.setattr(triton_conv, "emit_strided_load", sloppy)
    g_ = K.buffer("g", (5,))
    h = K.axis("h", 10)
    k = K.kernel("up_broken", [K.output("y", (h,), K.Load(g_, (K.strided([(h, 1)], 1, 2),), pad=0.0))])
    with pytest.raises(K.KernelMismatchError):
        K.compile(k, "cuda")(g=torch.arange(5, dtype=torch.float32, device="cuda") + 1)


def test_operand_marks_are_undone_when_a_rule_gives_up():
    """A conv rule that fails after marking its input must not leave the input materialized."""
    b = Builder("undo")
    x = b.input("x", torch.float32, ("B", 3, "H", 8))
    t = b.op("tanh", x)
    prog = b.build([b.op("conv2d", t, b.param("w", rnd(4, 3, 3, 3)), None, stride=1, padding=1, dilation=1,
                         groups=1)])
    L = lower.Lowering(prog, lower.Options("cpu"), lower.DimTable(["B", "H"]), "undo")
    L.lower_nodes(_Handlers())
    assert not any(lv.force_mat for lv in L.lvs)


class _Handlers:
    library_ops = lower.LIBRARY_OPS

    def library(self, L, node, reason=""):
        for v in node.inputs:
            if v is not None:
                L.need(v)
        L.leaf(node.output)

    def fallback(self, L, node, reason):
        raise AssertionError(reason)


# ----------------------------------------------------------------------------- sentinels


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("groups", [1, 4])          # conv shape (implicit GEMM) and window shape (depthwise)
def test_a_nan_made_inside_a_conv_kernel_is_named(device, groups):
    """An infinite input times a zero weight is NaN inside the convolution's sum: the conv kernel's
    sentinel names conv2d, the user's line and the first bad element, like any other kernel."""
    from truestar.compiler import SentinelError
    b = Builder("convnan")
    x = b.input("x", torch.float32, ("B", 4, 8, 8))
    w = torch.rand(8 if groups == 1 else 4, 4 // groups, 3, 3) + 0.1
    w[:, :, 1, 1] = 0.0                                  # the centre tap multiplies the infinity by 0
    c = b.op("conv2d", x, b.param("w", w), None, stride=1, padding=1, dilation=1, groups=groups,
             source="net.py:12 in forward: y = self.conv(x)")
    prog = b.build([b.op("relu", c, source="net.py:13 in forward: return y.relu()")])
    cp = compile_program(prog, device, sentinels="nan")
    assert cp.report.library == []
    if device == "cuda":
        assert [e["shape"] for e in cp.report.kernels] == ["conv" if groups == 1 else "window"]
    xs = torch.rand(2, 4, 8, 8)
    cp([xs])                                             # clean: nothing raised
    xs[1, 0, 3, 5] = float("inf")
    with pytest.raises(SentinelError) as e:
        cp([xs])
    err = e.value
    assert err.check == "nan" and err.op == "conv2d" and err.line == 12
    assert err.index[0] == 1 and tuple(err.index[2:]) == (3, 5)


@needs_gpu
def test_max_pool_padding_never_trips_the_inf_sentinel():
    """Max pooling reads minus infinity outside the input (its pad); only elements really read are
    checked, so a clean pooled model reports nothing with every check armed."""
    b = Builder("poolinf")
    x = b.input("x", torch.float32, ("B", 3, 9, 9))
    t = b.op("tanh", x)
    p = b.op("max_pool2d", t, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False)
    prog = b.build([b.op("adaptive_avg_pool2d", p, output_size=[1, 1])])
    cp = compile_program(prog, "cuda", sentinels="all")
    cp([rnd(2, 3, 9, 9)])
    cp([rnd(5, 3, 9, 9)])
    assert cp.report.library == [] and cp.verified
