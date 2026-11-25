"""Sentinels, the compiled in runtime checks (docs/COMPILER.md, Sentinels; kernel/checks.py, compiler/checks.py).

Each check class catches an injected fault and names the op, the user's line and the first
offending element; clean models with every check armed report nothing; the checks are off by
default; and breaking the flag code makes these tests fail.
"""

import inspect
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
from truestar.compiler import SentinelError, CompiledExecutor, compile_program  # noqa: E402
from truestar.ir.core import Builder  # noqa: E402
from truestar.kernel import checks as KC  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _has_numba():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


HAS_GPU = _has_gpu()
DEVICES = (["cpu"] if _has_numba() else []) + (["cuda"] if HAS_GPU else [])
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and Triton")
HERE = os.path.basename(__file__)


def line_of(fn, text):
    src, start = inspect.getsourcelines(fn)
    for k, ln in enumerate(src):
        if text in ln:
            return start + k
    raise AssertionError(text)


# ----------------------------------------------------------------------------- kernel layer


@pytest.mark.parametrize("device", DEVICES)
def test_kernel_flags_fire_per_class_and_stay_clean(device):
    M, N = K.sym("M"), K.sym("N")
    x, y = K.buffer("x", (M, N)), K.buffer("y", (M, N))
    i, j = K.axis("i", M), K.axis("j", N)
    k = K.kernel("k", [K.output("o", (i, j), K.log(x[i, j]) / y[i, j] + K.exp(x[i, j] * 100.0))])
    ck = K.compile(k, device, checks="nan,inf,div")
    xs = torch.rand(37, 70, device=device) * 0.5 + 0.1
    ys = torch.rand(37, 70, device=device) + 0.5
    flags = torch.zeros(KC.WORDS, dtype=torch.int32, device=device)

    def run(a, b):
        flags.zero_()
        ck(x=a, y=b, _chk=flags)
        return {c for c in KC.CLASSES if flags[KC.SLOT[c]].item()}

    assert run(xs, ys) == set()
    zero = ys.clone()
    zero[5, 9] = 0
    assert run(xs, zero) >= {"div"}
    neg = xs.clone()
    neg[3, 3] = -1.0
    assert run(neg, ys) == {"nan"}
    big = xs.clone()
    big[30, 69] = 2.0          # exp(200) is beyond float32
    assert "inf" in run(big, ys)


@needs_gpu
def test_kernel_overflow_on_half_precision_store():
    M = K.sym("M")
    x = K.buffer("x", (M, 64), "float16")
    i, j = K.axis("i", M), K.axis("j", 64)
    ck = K.compile(K.kernel("sq", [K.output("o", (i, j), x[i, j] * x[i, j])]), "cuda", checks="overflow,inf")
    flags = torch.zeros(KC.WORDS, dtype=torch.int32, device="cuda")
    a = torch.ones(9, 64, device="cuda", dtype=torch.float16)
    ck(x=a, _chk=flags)
    assert flags.tolist() == [0, 0, 0, 0]
    a[4, 7] = 300.0                       # 90000 > 65504: finite in float32, Inf in float16
    ck(x=a, _chk=flags)
    assert flags[KC.SLOT["overflow"]].item() == 1 and flags[KC.SLOT["inf"]].item() == 0


def test_kernel_checks_are_off_by_default_and_change_the_cache_key():
    x = K.buffer("x", (8,))
    i = K.axis("i", 8)
    k = K.kernel("d", [K.output("o", (i,), 1.0 / x[i])])
    plain = K.compile(k, "cuda" if HAS_GPU else "cpu")
    checked = K.compile(k, "cuda" if HAS_GPU else "cpu", checks=True)

    def own(src: str) -> str:
        """The kernel's own code: the CPU backend puts the same helper header (with the _divc
        check helper) in front of every kernel, armed or not."""
        if not HAS_GPU:
            from truestar.kernel.cpu_backend import HEADER
            assert src.startswith(HEADER)
            return src[len(HEADER):]
        return src

    assert "chk" not in own(plain.source) and "chk" in own(checked.source)
    assert plain.key != checked.key
    assert KC.parse(True) == {"nan", "div"} and KC.parse("all") == set(KC.CLASSES)
    assert KC.parse(None) == set() and KC.parse("off") == set()
    with pytest.raises(ValueError):
        KC.parse("nan,typo")


def test_infinity_analysis_keeps_a_masked_softmax_armed():
    S = K.sym("S")
    x = K.buffer("x", (S, S))
    i, j = K.axis("i", S), K.axis("j", S)
    s = K.where(K.index(j) <= K.index(i), x[i, j], -math.inf)
    w = K.softmax_weight(s, over=j)
    assert KC.may_be_inf(w, {}) == frozenset()            # exp(-inf) is 0: finite on purpose
    assert KC.may_be_inf(s, {}) == {KC.NEG}               # the masked scores themselves are not
    assert KC.may_be_inf(x[i, j] * 2.0, {"x": frozenset({KC.POS})}) == {KC.POS, KC.NEG}
    assert not KC.may_be_zero(K.const(3.0)) and KC.may_be_zero(x[i, j])


# ----------------------------------------------------------------------------- compiled programs


@pytest.mark.parametrize("device", DEVICES)
def test_division_by_zero_named_with_line_and_index(device):
    ex = CompiledExecutor(device, sentinels=True)

    @truestar.function(executor=ex)
    def norm(x: Tensor["B", 16], w: Tensor[16, 16]) -> Tensor["B", 16]:
        h = x @ w
        y = ops.log(h * h + 1.0)
        return y / x.sum(-1, keepdim=True)

    x, w = torch.rand(6, 16) + 0.5, torch.randn(16, 16)   # positive rows: no cancellation in the sum
    assert norm(x, w).shape == (6, 16)
    x[4] = 0.0
    with pytest.raises(SentinelError) as e:
        norm(x, w)
    err = e.value
    assert err.code == "TS503" and err.check == "div" and err.op == "div"
    assert err.line == line_of(norm.python, "return y / x.sum") and err.file.endswith(HERE)
    assert err.index == (4, 0) and err.count == 16
    assert "log" in err.fused_ops and "div" in err.fused_ops
    assert "TS503" in str(err) and "not returned" in str(err)


@pytest.mark.parametrize("device", DEVICES)
def test_nan_origin_is_the_op_not_its_readers(device):
    ex = CompiledExecutor(device, sentinels="nan")

    @truestar.function(executor=ex)
    def f(x: Tensor["B", 32]) -> Tensor["B", 32]:
        a = ops.sqrt(x - 0.5)
        b = ops.softmax(a, -1)
        return b * 2.0 + 1.0

    x = torch.rand(5, 32) + 0.6
    f(x)
    x[2, 17] = 0.1
    with pytest.raises(SentinelError) as e:
        f(x)
    err = e.value
    assert err.check == "nan" and err.code == "TS500" and err.op == "sqrt"
    assert err.line == line_of(f.python, "ops.sqrt(x - 0.5)")
    assert err.index == (2, 17) and err.count == 1


@pytest.mark.parametrize("device", DEVICES)
def test_inf_and_input_nan(device):
    ex = CompiledExecutor(device, sentinels="nan,inf")

    @truestar.function(executor=ex)
    def g(x: Tensor["B", 8]) -> Tensor["B", 8]:
        return ops.exp(x * 100.0) + 1.0

    x = torch.rand(3, 8) * 0.1
    g(x)
    x[1, 2] = 1.0                          # exp(100) is beyond float32
    with pytest.raises(SentinelError) as e:
        g(x)
    assert e.value.check == "inf" and e.value.op == "exp" and e.value.index == (1, 2)
    x[1, 2] = float("nan")
    with pytest.raises(SentinelError) as e:
        g(x)
    assert e.value.check == "nan" and e.value.op == "input" and e.value.value == "x" and e.value.index == (1, 2)


@needs_gpu
def test_overflow_in_float16_named():
    ex = CompiledExecutor("cuda", "float16", sentinels="all")

    @truestar.function(executor=ex)
    def sq(x: Tensor["B", 64]) -> Tensor["B", 64]:
        return x * x * 4.0

    x = torch.ones(4, 64)
    sq(x)
    x[3, 9] = 200.0                       # 160000 does not fit float16
    with pytest.raises(SentinelError) as e:
        sq(x)
    err = e.value
    assert err.check == "overflow" and err.code == "TS502"
    assert err.index == (3, 9) and err.line == line_of(sq.python, "x * x * 4.0")
    assert abs(err.bad_value - 160000.0) < 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_nan_inside_a_loop_names_the_iteration(device):
    ex = CompiledExecutor(device, sentinels=True)

    @truestar.function(executor=ex)
    def rnn(x: Tensor["B", "T", 8], w: Tensor[8, 8]) -> Tensor["B", 8]:
        h = ops.zeros((x.shape[0], 8))
        for t in range(x.shape[1]):
            h = ops.sqrt(x[:, t] @ w + h * 0.5)
        return h

    x, w = torch.rand(2, 6, 8) * 0.1, torch.rand(8, 8) * 0.1
    rnn(x, w)
    x[1, 4] = -50.0                        # a negative square root argument at step 4, NaN after it
    with pytest.raises(SentinelError) as e:
        rnn(x, w)
    err = e.value
    assert err.iterations == [4]
    assert err.op == "sqrt" and err.line == line_of(rnn.python, "h = ops.sqrt(") and err.index[0] == 1


def test_bounds_are_always_checked_with_line_and_index():
    device = "cuda" if HAS_GPU else ("cpu" if DEVICES else None)
    if device is None:
        pytest.skip("needs a compiled backend")
    b = Builder("table")
    x = b.input("x", torch.float32, (10, 4))
    idx = b.input("idx", torch.int64, ("N",))
    y = b.op("index_select", x, idx, dim=0, source="lookup.py:12 in f: rows = table[idx]")
    prog = b.build([y])
    cp = compile_program(prog, device)        # no checks asked: bounds are on anyway
    cp([torch.randn(10, 4), torch.tensor([1, 2, 3])])
    with pytest.raises(SentinelError) as e:
        cp([torch.randn(10, 4), torch.tensor([1, 12, 3, 40])])
    err = e.value
    assert isinstance(err, ir.IRError) and err.code == "TS504"
    assert err.file == "lookup.py" and err.line == 12 and err.index == (1,) and err.bad_value == 12 and err.count == 2


def _transformer():
    torch.manual_seed(3)
    layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True).eval()
    return layer


@pytest.mark.parametrize("device", DEVICES)
def test_no_false_alarm_on_clean_models_with_every_check(device):
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8), nn.Softmax(-1)).eval()
    B = torch.export.Dim("B", min=1, max=64)
    p1 = ir.import_module(mlp, (torch.randn(2, 16),), dynamic_shapes=({0: B},))
    cp = compile_program(p1, device, sentinels="all")
    for n in (1, 7):
        cp([torch.randn(n, 16) * 3])
    layer = _transformer()
    mask = nn.Transformer.generate_square_subsequent_mask(6)         # a float -inf causal mask
    p2 = ir.import_module(layer, (torch.randn(2, 6, 32),), dynamic_shapes=({0: B},))
    cp2 = compile_program(p2, device, sentinels="all")
    cp2([torch.randn(3, 6, 32)])
    assert mask.isinf().any()


@needs_gpu
def test_resnet18_clean_with_every_check():
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18().eval()
    prog = ir.import_module(model, (torch.randn(1, 3, 64, 64),))
    cp = compile_program(prog, "cuda", sentinels="all")
    cp([torch.randn(1, 3, 64, 64)])
    armed = [k for k in cp.report.kernels if k.get("checks")]
    assert len(armed) == len(cp.report.kernels) > 0


@needs_gpu
def test_deliberate_infinity_disarms_only_inf_on_its_readers():
    b = Builder("masked")
    x = b.input("x", torch.float32, ("B", 8))
    m = b.input("m", torch.bool, ("B", 8))
    ninf = b.op("full", shape=[], value=-math.inf, dtype=torch.float32)
    masked = b.op("where", m, x, ninf)
    out = b.op("add", masked, b.op("mul", x, x))
    sm = b.op("softmax", masked, dim=-1)
    prog = b.build([out, sm])
    cp = compile_program(prog, "cuda", sentinels="all")
    xs = torch.randn(3, 8)
    ms = torch.rand(3, 8) > 0.3
    ms[:, 0] = True
    cp([xs, ms])                                   # -inf in the output on purpose: no alarm
    disarmed = [k for k in cp.report.kernels if k.get("checks_disarmed")]
    assert disarmed and all("nan" in k["checks"] for k in disarmed)
    xs[0, 0] = float("nan")
    with pytest.raises(SentinelError) as e:
        cp([xs, ms])
    assert e.value.check == "nan"


@needs_gpu
def test_breaking_the_flag_code_fails(monkeypatch):
    """Tests can fail: with the store checks emitting nothing, a NaN goes unreported."""
    from truestar.kernel import triton_backend as tb
    monkeypatch.setattr(tb, "_store_checks", lambda *a, **k: None)
    K.clear_memory_cache() if hasattr(K, "clear_memory_cache") else None
    from truestar.kernel import compiler as kc
    kc.clear_memory_cache()
    M = K.sym("M")
    x = K.buffer("x", (M, 8))
    i, j = K.axis("i", M), K.axis("j", 8)
    ck = K.compile(K.kernel("brk", [K.output("o", (i, j), K.sqrt(x[i, j] - 7.0))]), "cuda", checks="nan")
    flags = torch.zeros(KC.WORDS, dtype=torch.int32, device="cuda")
    os.environ["TRUESTAR_VERIFY"] = "0"
    try:
        ck(x=torch.zeros(3, 8, device="cuda"), _chk=flags)
    finally:
        os.environ.pop("TRUESTAR_VERIFY")
    assert flags.sum().item() == 0       # the broken generator misses it; the real one flags it
    kc.clear_memory_cache()


@needs_gpu
def test_deferred_read_raises_at_a_later_call_or_read_sentinels():
    ex = CompiledExecutor("cuda", sentinels=True, sentinel_read="deferred")

    @truestar.function(executor=ex)
    def f(x: Tensor["B", 16]) -> Tensor["B", 16]:
        return ops.sqrt(x) * 3.0

    good = torch.rand(4, 16, device="cuda")
    f(good)                                   # first call: verified and read at once
    f(good)
    bad = good.clone()
    bad[1, 2] = -1.0
    f(bad)                                    # no host read: returns (the NaN is in the result)
    with pytest.raises(SentinelError) as e:
        ex.last.read_sentinels()
    assert e.value.op == "sqrt" and e.value.index == (1, 2) and "deferred read" in str(e.value)
    f(bad)
    with pytest.raises(SentinelError):
        torch.cuda.synchronize()
        f(good)                               # the next call reads the earlier flags
    f(good)
    ex.last.read_sentinels()                     # clean calls: nothing raised


@needs_gpu
def test_library_output_checked_where_a_kernel_loads_it():
    """A convolution's output is read only by a kernel, so the kernel checks it as it loads it (no
    extra pass); a value the kernel then hides (a clamp) is still reported, at the convolution."""
    b = Builder("convmin")
    x = b.input("x", torch.float32, ("B", 3, 8, 8))
    w = b.param("w", torch.ones(4, 3, 3, 3))
    c = b.op("conv2d", x, w, None, stride=1, padding=1, dilation=1, groups=1,
             source="net.py:7 in forward: y = self.conv(x)")
    y = b.op("clamp", c, min=-1.0, max=1.0, source="net.py:8 in forward: return y.clamp(-1, 1)")
    prog = b.build([y])
    # conv2d is a Truestar kernel by default since the convolution kernels; the library runs it on request
    cp = compile_program(prog, "cuda", sentinels="nan,inf", library=("conv2d",))
    conv = [s for _, s in cp.plan.all_steps() if s.kind == "library"]
    kern = [s for _, s in cp.plan.all_steps() if s.kind == "kernel"]
    assert conv and conv[0].trig is None and kern and kern[0].check_loads
    xs = torch.rand(2, 3, 8, 8)
    cp([xs])
    xs[1, :2, 4, 4] = 3e38                   # two taps of 3e38: the sum overflows float32 in the convolution
    with pytest.raises(SentinelError) as e:
        cp([xs])
    assert e.value.check == "inf" and e.value.op == "conv2d" and e.value.line == 7 and e.value.index[0] == 1


@needs_gpu
def test_every_compiled_variant_carries_the_sentinels():
    """A program that follows PyTorch's TF32 and determinism settings compiles one variant per
    setting; each one is armed and names the fault."""
    b = Builder("variants")
    x = b.input("x", torch.float32, ("B", 32))
    w = b.param("w", torch.ones(32, 32))
    y = b.op("sqrt", b.op("matmul", x, w), source="v.py:3 in f: return (x @ w).sqrt()")
    cp = compile_program(b.build([y]), "cuda", sentinels="nan")
    xs = torch.ones(4, 32)
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            bad = xs.clone()
            bad[2] = -1.0
            with pytest.raises(SentinelError) as e:
                cp([bad])
            assert e.value.op == "sqrt" and e.value.index[0] == 2 and e.value.line == 3
            assert cp.checks is not None and cp.opts.allow_tf32 == tf32
        saved = cp._variants[(False, False)]            # the variant left behind keeps its own sentinels
        assert saved["checks"] is not None and saved["checks"] is not cp.checks
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@needs_gpu
def test_sentinels_leave_the_device_memory_plan_unchanged():
    """The flag words are pinned host memory, and with the planner on a watched torch step output
    is reduced at once, so the planned and the measured device peak do not move."""
    torchvision = pytest.importorskip("torchvision")
    from truestar.compiler.memory import plan_memory
    model = torchvision.models.resnet18().eval()
    prog = ir.import_module(model, (torch.randn(2, 3, 64, 64),))
    x = torch.randn(2, 3, 64, 64, device="cuda")
    peaks, plans = {}, {}
    for name, s in (("off", None), ("all", "all")):
        cp = compile_program(prog, "cuda", sentinels=s)
        cp([x])
        plans[name] = plan_memory(cp, [x])
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = cp([x])
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - base
        del out, cp
    assert plans["all"].peak_bytes == plans["off"].peak_bytes
    assert plans["all"].inplace_steps == plans["off"].inplace_steps
    assert peaks["all"] <= peaks["off"] + 64 * 1024      # a few 0 dim trigger tensors at most


@needs_gpu
def test_sentinels_name_the_op_when_the_planner_reuses_buffers():
    ex = CompiledExecutor("cuda", sentinels=True)

    @truestar.function(executor=ex)
    def chain(x: Tensor["B", 64]) -> Tensor["B", 64]:
        a = ops.exp(x) * 0.5
        b = a + 1.0
        c = ops.log(b - 2.0)
        return c * 2.0

    x = torch.rand(8, 64, device="cuda") + 2.0
    chain(x)
    x[5, 7] = -3.0
    with pytest.raises(SentinelError) as e:
        chain(x)
    assert e.value.op == "log" and e.value.index == (5, 7) and e.value.line == line_of(chain.python, "ops.log(b - 2.0)")


@needs_gpu
@pytest.mark.parametrize("read", ["call", "deferred"])
def test_a_sentinel_program_is_freed_without_the_garbage_collector(read):
    """Sentinels leave no global state behind: once the last reference to a compiled program goes,
    the program and its device buffers are gone at once, with the garbage collector off. (They
    used to hold the program strongly, a cycle that kept a resnet18's weights alive until a
    collection, and a memory tracker that lists live CUDA tensors then saw them instead of the
    model's own parameters.)"""
    import gc
    import weakref
    from torch._subclasses.fake_tensor import is_fake
    b = Builder("freed")
    x = b.input("x", torch.float32, ("B", 3, 8, 8))
    w = b.param("w", torch.randn(64, 3, 3, 3))
    c = b.op("conv2d", x, w, None, stride=1, padding=1, dilation=1, groups=1)
    y = b.op("clamp", c, min=-1.0, max=1.0)
    prog = b.build([y, c])                   # c is a program output: a watched torch step too
    gc.collect()
    gc.disable()
    try:
        # library=("conv2d",): the convolution runs as a cuDNN torch step, whose output is also a
        # program output, so it is watched at the end of the call (not checked at a kernel load)
        cp = compile_program(prog, "cuda", sentinels="all", sentinel_read=read, library=("conv2d",))
        cp([torch.rand(2, 3, 8, 8)])
        cp([torch.rand(2, 3, 8, 8)])
        if read == "deferred":
            cp.read_sentinels()
        lib = [s for _, s in cp.plan.all_steps() if s.kind == "library"]
        assert lib and lib[0].trig is not None       # the watched torch step this test is about
        ref = weakref.ref(cp)
        ptrs = {t.untyped_storage().data_ptr() for t in cp.plan.template if isinstance(t, torch.Tensor) and t.is_cuda}
        del cp, lib
        assert ref() is None                 # freed by reference counting alone
        # real device tensors only: the fake tensors a torch.compile elsewhere in the process left
        # in its graphs report a CUDA device but own no memory (their data pointer raises)
        live = {o.untyped_storage().data_ptr() for o in gc.get_objects()
                if isinstance(o, torch.Tensor) and o.is_cuda and not is_fake(o)}
        assert not (ptrs & live)
    finally:
        gc.enable()
