"""P17: native functions registered as PyTorch custom ops (truestar.customop)."""

import json
import os
import subprocess
import sys
import textwrap
import warnings

import pytest
import torch
import torch.nn.functional as F

import truestar
from truestar import Tensor, ops
from truestar import customop
from truestar.customop import executor as cexec
from truestar.customop import op as cop

HAS_GPU = torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA")
DEVICES = ["cpu"] + (["cuda"] if HAS_GPU else [])
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


# ----------------------------------------------------------------------------- the three functions


@truestar.torch_op
def gelu_mlp(x: Tensor["B", 64], w1: Tensor[256, 64], b1: Tensor[256], w2: Tensor[64, 256]) -> Tensor["B", 64]:
    h = ops.gelu(ops.linear(x, w1, b1), approximate="tanh")
    return h @ w2.T + x


@truestar.torch_op
def softcap_attention(q: Tensor["B", "H", "S", 32], k: Tensor["B", "H", "S", 32],
                      v: Tensor["B", "H", "S", 32]) -> Tensor["B", "H", "S", 32]:
    s = (q @ k.transpose(-1, -2)) * 0.17677669529663687
    s = 20.0 * ops.tanh(s / 20.0)          # the custom score modification: tanh softcapping
    return ops.softmax(s, dim=-1) @ v


@truestar.torch_op
def rms_norm_residual(x: Tensor["B", 128], r: Tensor["B", 128], w: Tensor[128]) -> Tensor["B", 128]:
    h = x + r
    return h * ops.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w


def t_gelu_mlp(x, w1, b1, w2):
    return F.gelu(F.linear(x, w1, b1), approximate="tanh") @ w2.T + x


def t_softcap_attention(q, k, v):
    s = (q @ k.transpose(-1, -2)) * 0.17677669529663687
    s = 20.0 * torch.tanh(s / 20.0)
    return torch.softmax(s, dim=-1) @ v


def t_rms_norm_residual(x, r, w):
    h = x + r
    return h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w


CASES = {
    "gelu_mlp": (gelu_mlp, t_gelu_mlp, lambda B: [(B, 64), (256, 64), (256,), (64, 256)]),
    "softcap_attention": (softcap_attention, t_softcap_attention, lambda B: [(B, 2, 16, 32)] * 3),
    "rms_norm_residual": (rms_norm_residual, t_rms_norm_residual, lambda B: [(B, 128), (B, 128), (128,)]),
}


def make(shapes, device, seed=0, grad=True):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*s, generator=g).to(device).requires_grad_(grad) for s in shapes]


def rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def grads_of(fn, xs, seed=1):
    for x in xs:
        x.grad = None
    y = fn(*xs)
    g = torch.Generator().manual_seed(seed)
    cot = torch.randn(y.shape, generator=g).to(y.device)
    (y * cot).sum().backward()
    out = [x.grad.clone() for x in xs]
    for x in xs:
        x.grad = None
    return y.detach(), out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(CASES))
def test_forward_and_gradients_match_torch(case, device):
    op, ref, shapes = CASES[case]
    for B in (1, 3, 5):
        xs = make(shapes(B), device, seed=B)
        y, g = grads_of(op, xs)
        # float64 torch is the yardstick; the same math in float32 torch is shown next to it
        xs64 = [x.detach().double().requires_grad_() for x in xs]
        y64, g64 = grads_of(ref, xs64)
        y32, g32 = grads_of(ref, xs)
        assert y.dtype == torch.float32 and y.device.type == device
        assert rel(y, y64) < 1e-5, (case, B, rel(y, y64), rel(y32, y64))
        for a, b, c in zip(g, g64, g32):
            assert a.shape == b.shape
            assert rel(a, b) < 1e-4, (case, B, rel(a, b), rel(c, b))
    assert op.plan(device).path == "kernel"
    assert op.plan(device, backward=True).path == "kernel"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(CASES))
def test_opcheck_passes(case, device):
    op, _, shapes = CASES[case]
    for B in (2, 3):
        torch.library.opcheck(op.overload, tuple(make(shapes(B), device, seed=B)))
    # also without requires_grad (the faketensor and schema tests on plain inputs)
    torch.library.opcheck(op.overload, tuple(make(shapes(4), device, grad=False)))


def test_opcheck_fails_when_the_fake_implementation_is_wrong(monkeypatch):
    real = cop._fake_outputs

    def broken(program, tensors, who):
        outs = real(program, tensors, who)
        return [o.new_empty([*o.shape[:-1], o.shape[-1] + 1]) for o in outs]   # one column too many

    monkeypatch.setattr(cop, "_fake_outputs", broken)
    xs = make(CASES["rms_norm_residual"][2](3), "cpu")
    with pytest.raises(Exception) as e:
        torch.library.opcheck(rms_norm_residual.overload, tuple(xs))
    assert "fake" in str(e.value).lower() or "shape" in str(e.value).lower()


def test_wrong_gradient_from_a_plugged_executor_is_caught():
    """An executor returning a wrong backward is refused by the program level float64 check."""
    gprog = rms_norm_residual.grad_program

    def factory(program, device_type):
        if program is not gprog:
            return None

        def run(inputs):
            outs = cexec._reference_runner(program)(inputs)
            return [o * 1.01 for o in outs]
        return run

    customop.register_executor("broken", factory)
    try:
        xs = make(CASES["rms_norm_residual"][2](2), "cpu")
        y = rms_norm_residual(*xs)
        with pytest.raises(customop.ProgramMismatchError, match="differs from the float64 reference"):
            y.sum().backward()
    finally:
        customop.unregister_executor("broken")


def test_executor_seam_takes_over_and_can_be_removed():
    seen = []

    def factory(program, device_type):
        if program is not gelu_mlp.program:
            return None
        ref = cexec._reference_runner(program)

        def run(inputs):
            seen.append(tuple(inputs[0].shape))
            return ref(inputs)
        return run

    customop.register_executor("counting", factory)
    try:
        xs = make(CASES["gelu_mlp"][2](3), "cpu", grad=False)
        y = gelu_mlp(*xs)
        assert seen == [(3, 64)]
        assert gelu_mlp.plan("cpu").path == "counting"
        assert rel(y, t_gelu_mlp(*[x.double() for x in xs])) < 1e-5
    finally:
        customop.unregister_executor("counting")
    assert gelu_mlp.plan("cpu").path == "kernel"


# ----------------------------------------------------------------------------- torch.compile


class _Counter:
    def __init__(self):
        self.graphs = 0

    def __call__(self, gm, example_inputs):
        self.graphs += 1
        return gm.forward


@pytest.mark.parametrize("case", list(CASES))
def test_zero_graph_breaks_under_torch_compile(case):
    op, ref, shapes = CASES[case]

    def model(*xs):
        return op(*xs).relu() * 2.0 + 1.0

    xs = make(shapes(3), "cpu", grad=False)
    ex = torch._dynamo.explain(model)(*xs)
    assert ex.graph_break_count == 0 and ex.graph_count == 1, ex.break_reasons
    torch._dynamo.reset()
    counter = _Counter()
    cm = torch.compile(model, backend=counter, fullgraph=True, dynamic=True)
    # batch sizes distinct from every fixed dim, so Dynamo's duck sizing cannot alias B with H = 2
    for B in (3, 5, 7):
        xs = make(shapes(B), "cpu", grad=False, seed=B)
        assert rel(cm(*xs), model(*xs)) < 1e-6
    assert counter.graphs == 1


@pytest.mark.parametrize("case", list(CASES))
def test_same_source_eager_and_compiled_with_backward(case):
    """The same function, eager and under torch.compile (AOTAutograd), gives the same values and
    gradients; the backward custom op is traced through its own fake implementation."""
    op, _, shapes = CASES[case]

    def model(*xs):
        return op(*xs) * 0.5

    torch._dynamo.reset()
    cm = torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=True)
    for B in (2, 5):
        xs = make(shapes(B), "cpu", seed=B)
        y_e, g_e = grads_of(model, xs)
        y_c, g_c = grads_of(cm, xs)
        assert rel(y_c, y_e) < 1e-6
        for a, b in zip(g_c, g_e):
            assert rel(a, b) < 1e-6


@needs_gpu
def test_inductor_on_cuda_fullgraph_with_backward():
    op, _, shapes = CASES["softcap_attention"]

    def model(q, k, v):
        return op(q, k, v).sum(-1)

    torch._dynamo.reset()
    try:
        cm = torch.compile(model, backend="inductor", fullgraph=True)
        xs = make(shapes(2), "cuda")
        y_e, g_e = grads_of(model, xs)
        y_c, g_c = grads_of(cm, xs)
        assert rel(y_c, y_e) < 1e-5
        for a, b in zip(g_c, g_e):
            assert rel(a, b) < 1e-5
    finally:
        torch._dynamo.reset()   # drop the compiled graphs (and the fake tensors they keep) for later tests


def test_fake_implementation_keeps_the_batch_symbolic():
    from torch.fx.experimental.proxy_tensor import make_fx

    xs = make([(6, 128), (6, 128), (128,)], "cpu", grad=False)
    gm = make_fx(lambda x, r, w: rms_norm_residual.overload(x, r, w), tracing_mode="symbolic")(*xs)
    node = next(n for n in gm.graph.nodes if n.target is rms_norm_residual.overload)
    shape = node.meta["val"].shape
    assert isinstance(shape[0], torch.SymInt) and shape[1] == 128
    assert node.meta["val"].dtype == torch.float32


def test_shape_and_dtype_errors_in_eager_name_the_argument():
    xs = make([(3, 128), (4, 128), (128,)], "cpu", grad=False)
    with pytest.raises(Exception, match="dim 0"):
        rms_norm_residual(*xs)
    with pytest.raises(Exception, match="dtype"):
        rms_norm_residual(xs[0].double(), xs[0].double(), xs[2])


# ----------------------------------------------------------------------------- slow path, no backward


def test_reference_path_is_reported_and_correct():
    @truestar.torch_op
    def exact_gelu_scale(x: Tensor["B", 16], s: Tensor[16]) -> Tensor["B", 16]:
        return ops.gelu(x) * s

    xs = make([(3, 16), (16,)], "cpu")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y, g = grads_of(exact_gelu_scale, xs)
    slow = [x for x in w if issubclass(x.category, customop.SlowPathWarning)]
    assert slow and "erf" in str(slow[0].message)
    p = exact_gelu_scale.plan("cpu")
    assert p.path == "reference" and "erf" in p.reason
    xs64 = [x.detach().double().requires_grad_() for x in xs]
    y64, g64 = grads_of(lambda x, s: F.gelu(x) * s, xs64)
    assert y.dtype == torch.float32 and rel(y, y64) < 1e-6
    for a, b in zip(g, g64):
        assert rel(a, b) < 1e-6
    torch.library.opcheck(exact_gelu_scale.overload, tuple(make([(2, 16), (16,)], "cpu")))


def test_kernels_false_forces_reference_and_agrees_with_kernel_path():
    plain = truestar.torch_op(rms_norm_residual.native, name="rms_ref_only", kernels=False)
    xs = make(CASES["rms_norm_residual"][2](4), "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", customop.SlowPathWarning)
        y1, g1 = grads_of(plain, xs)
    y2, g2 = grads_of(rms_norm_residual, xs)
    assert plain.plan("cpu").path == "reference"
    assert rel(y1, y2) < 1e-5
    for a, b in zip(g1, g2):
        assert rel(a, b) < 1e-4


def test_control_flow_op_runs_forward_and_backward():
    """Since 2026-11-24 the backward program goes through cond (autodiff_cf.py): a control flow op
    has gradients on both sides of its branch, and still compiles without a break."""
    @truestar.torch_op
    def branchy(x: Tensor["B", 8]) -> Tensor["B", 8]:
        if x.sum() > 0:
            x = ops.relu(x) * x
        else:
            x = -x * 0.5
        return x

    assert branchy.backward_op is not None and branchy.grad_error is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", customop.SlowPathWarning)
        for x in (torch.rand(4, 8) + 0.1, -torch.rand(4, 8) - 0.1):
            y = branchy(x)
            expected = torch.relu(x) * x if x.sum() > 0 else -x * 0.5
            assert torch.allclose(y, expected)
            xg, xr = x.clone().requires_grad_(), x.clone().requires_grad_()
            (branchy(xg) * torch.arange(8.0)).sum().backward()
            ((torch.relu(xr) * xr if xr.sum() > 0 else -xr * 0.5) * torch.arange(8.0)).sum().backward()
            assert torch.allclose(xg.grad, xr.grad, atol=1e-6)
        ex = torch._dynamo.explain(lambda a: branchy(a) + 1)(torch.randn(4, 8))
        assert ex.graph_break_count == 0


def test_multiple_outputs_and_names():
    @truestar.torch_op(name="split_stats")
    def stats(x: Tensor["B", 32]):
        return x.mean(-1, keepdim=True), x * 2.0

    assert stats.qualname == "truestar::split_stats"
    assert stats.schema == "(Tensor x) -> (Tensor, Tensor)"
    x = torch.randn(5, 32, requires_grad=True)
    m, d = stats(x)
    (m.sum() + d.sum()).backward()
    assert torch.allclose(x.grad, torch.full_like(x, 1 / 32 + 2.0))
    torch.library.opcheck(stats.overload, (torch.randn(3, 32, requires_grad=True),))
    # a second registration of the same name gets a new one instead of clobbering the first
    again = truestar.torch_op(stats.native, name="split_stats")
    assert again.qualname == "truestar::split_stats_2"


# ----------------------------------------------------------------------------- cold start, no toolchain


def _clean_env():
    """PATH reduced to the Python folder and the OS folder: no MSVC, gcc, clang or nvcc."""
    env = dict(os.environ)
    py = os.path.dirname(sys.executable)
    keep = [py, os.path.join(py, "Scripts")]
    if os.name == "nt":
        keep.append(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
    else:
        keep += ["/usr/bin", "/bin"]
    env["PATH"] = os.pathsep.join(keep)
    env["PYTHONPATH"] = SRC
    for k in ("CUDA_HOME", "CUDA_PATH", "CC", "CXX"):
        env.pop(k, None)
    return env


def test_cold_registration_under_2_s_and_never_invokes_a_compiler(tmp_path):
    # Runners may have gcc or clang installed; what must hold is that registering and calling the
    # op never runs one. Every process launch and every torch C++ extension build is recorded.
    code = textwrap.dedent("""
        import json, os, subprocess, time
        import torch
        import torch.utils.cpp_extension as cpp
        COMPILERS = {"cl", "nvcc", "gcc", "g++", "clang", "clang++", "cc", "c++", "ld"}
        calls = []
        real_init = subprocess.Popen.__init__
        def spy_init(self, args, *a, **k):
            argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
            first = os.fsdecode(argv[0]).split()[0] if argv else ""
            name = os.path.basename(first).lower()
            if name.endswith(".exe"):
                name = name[:-4]
            if name in COMPILERS:
                calls.append(name)
            return real_init(self, args, *a, **k)
        subprocess.Popen.__init__ = spy_init
        real_system = os.system
        os.system = lambda cmd: (calls.append("os.system: " + str(cmd)), real_system(cmd))[1]
        for fn in ("load", "load_inline"):
            setattr(cpp, fn, lambda *a, _fn=fn, **k: calls.append("cpp_extension." + _fn))
        t0 = time.perf_counter()
        import truestar
        from truestar import Tensor, ops
        t1 = time.perf_counter()

        @truestar.torch_op
        def rms(x: Tensor["B", 128], r: Tensor["B", 128], w: Tensor[128]) -> Tensor["B", 128]:
            h = x + r
            return h * ops.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w
        t2 = time.perf_counter()
        xs = [torch.randn(4, 128, requires_grad=True), torch.randn(4, 128), torch.randn(128, requires_grad=True)]
        y = rms(*xs); y.sum().backward()
        ref = (xs[0] + xs[1]) * torch.rsqrt(((xs[0] + xs[1]) ** 2).mean(-1, keepdim=True) + 1e-6) * xs[2]
        print(json.dumps({"import": t1 - t0, "register": t2 - t1,
                          "compiler_calls": calls,
                          "err": float((y - ref).abs().max()), "path": rms.plan("cpu").path}))
    """)
    script = tmp_path / "cold.py"
    script.write_text(code, encoding="utf-8")
    out = subprocess.run([sys.executable, str(script)], env=_clean_env(), capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["compiler_calls"] == [], r
    assert r["register"] < 2.0, r
    assert r["err"] < 1e-4 and r["path"] == "kernel"
