"""Region and end to end checks of the compiler with the float64 reference on the GPU.

The compiler's own checks (docs/COMPILER.md, Verification, layers 2 and 3) evaluate the tensor IR
reference implementations on the program's GPU when the check is big enough. These tests hold the
GPU path to the CPU path: the same reference values (up to float64 rounding), the same verdicts,
the same blamed region, op, stage and message on deliberately broken compilers, the same schedule
(a branch is checked when it first runs) and a CPU fallback when there is no room or no op.
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
from truestar.compiler import CompiledExecutor, ProgramMismatchError, compile_program, lower, runtime  # noqa: E402
from truestar.ir import reference as iref  # noqa: E402
from truestar.ir.core import Builder  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs CUDA and Triton")

g = torch.Generator().manual_seed(7)


def rnd(*shape, scale=1.0):
    return torch.randn(*shape, generator=g) * scale


def rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    if a.dtype == torch.bool or b.dtype == torch.bool or not a.is_floating_point():
        return 0.0 if torch.equal(a.cpu(), b.cpu()) else math.inf
    fin = torch.isfinite(b)
    if not torch.equal(torch.isfinite(a), fin) or not torch.equal(a[~fin], b[~fin]):
        return math.inf
    if not fin.any():
        return 0.0
    return (a[fin] - b[fin]).abs().max().item() / max(b[fin].abs().max().item(), 1e-300)


@pytest.fixture
def on(monkeypatch):
    """on("gpu") / on("cpu"): where every check's float64 reference runs."""
    def set_device(where):
        monkeypatch.setenv("TRUESTAR_VERIFY_DEVICE", where)
    return set_device


@pytest.fixture
def twin(monkeypatch):
    """Runs the CPU reference next to every GPU reference of a region check and records the worst
    relative difference per value, and counts region checks that ran on the GPU."""
    seen = {"values": 0, "worst": 0.0, "gpu_checks": 0}
    real = runtime.GroupCheck.reference

    def both(self, env, rt, device=None, drop=False):
        local = real(self, env, rt, device, drop)
        if device is not None:
            seen["gpu_checks"] += 1
            cpu = real(self, env, rt)
            for vid, t in local.items():
                seen["values"] += 1
                seen["worst"] = max(seen["worst"], rel(t, cpu[vid]))
        return local

    monkeypatch.setattr(runtime.GroupCheck, "reference", both)
    return seen


# ----------------------------------------------------------------------------- programs


def mlp_program():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 96), nn.ReLU(), nn.Linear(96, 10),
                      nn.Softmax(-1)).eval()
    return ir.import_module(m, (torch.randn(2, 64),),
                            dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},)), (9, 64)


def transformer_program():
    torch.manual_seed(0)
    m = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True).eval()
    return ir.import_module(m, (torch.randn(2, 16, 32),),
                            dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},)), (3, 16, 32)


def resnet_program():
    import torchvision
    torch.manual_seed(0)
    m = torchvision.models.resnet18().eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.5, generator=g)
                mod.running_var.uniform_(0.5, 2.0, generator=g)
    return ir.import_module(m, (torch.randn(2, 3, 64, 64),),
                            dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},)), (2, 3, 64, 64)


def attention_program(kind):
    b = Builder(f"attention_{kind}")
    q = b.input("q", torch.float32, ("B", 2, 33, 16))
    k = b.input("k", torch.float32, ("B", 2, 40, 16))
    v = b.input("v", torch.float32, ("B", 2, 40, 8))
    mask, attrs = None, dict(scale=None, is_causal=kind == "causal")
    if kind == "bool_mask":
        keep = torch.rand(33, 40, generator=g) > 0.3
        keep[:, 0] = True
        mask = b.param("mask", keep)
    elif kind == "float_mask":
        mask = b.param("mask", rnd(33, 40))
    return b.build([b.op("attention", q, k, v, mask, **attrs)])


def make_rnn(executor):
    @truestar.function(executor=executor)
    def rnn(x: Tensor["B", "T", 16], wx: Tensor[16, 32], wh: Tensor[32, 32]) -> Tensor["B", 32]:
        h = ops.zeros((x.shape[0], 32))
        for t in range(x.shape[1]):
            h = ops.tanh(x[:, t] @ wx + h @ wh)
        if h.sum() > 0:
            h = ops.relu(h)
        else:
            h = -h
        return h
    return rnn


MODELS = {
    "mlp": lambda: (mlp_program()[0], [rnd(*mlp_program()[1])]),
    "transformer": lambda: (transformer_program()[0], [rnd(*transformer_program()[1])]),
    "resnet18": lambda: (resnet_program()[0], [rnd(*resnet_program()[1])]),
    "attention_causal": lambda: (attention_program("causal"), [rnd(2, 2, 33, 16), rnd(2, 2, 40, 16), rnd(2, 2, 40, 8)]),
    "attention_bool_mask": lambda: (attention_program("bool_mask"),
                                    [rnd(2, 2, 33, 16), rnd(2, 2, 40, 16), rnd(2, 2, 40, 8)]),
    "attention_float_mask": lambda: (attention_program("float_mask"),
                                     [rnd(2, 2, 33, 16), rnd(2, 2, 40, 16), rnd(2, 2, 40, 8)]),
}


# ----------------------------------------------------------------------------- same reference, same report


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_gpu_reference_matches_cpu_reference_and_reports_match(name, dtype, on, twin):
    prog, inputs = MODELS[name]()
    on("gpu")
    gpu = compile_program(prog, "cuda", dtype)
    out_gpu = gpu(inputs)
    assert gpu.verified and gpu.end_to_end_device.startswith("cuda")
    assert twin["gpu_checks"] > 0 and twin["values"] > 0
    assert twin["worst"] < 1e-12, twin                  # float64 rounding only, every region value
    on("cpu")
    cpu = compile_program(prog, "cuda", dtype)
    out_cpu = cpu(inputs)
    assert cpu.verified and cpu.end_to_end_device == "cpu"
    assert all(torch.equal(a, b) for a, b in zip(out_gpu, out_cpu))      # the same compiled program
    # the same errors against float64, up to float64 rounding of the reference
    assert abs(gpu.max_error - cpu.max_error) <= 1e-12 * max(1.0, cpu.max_error)
    assert abs(gpu.max_region_error - cpu.max_region_error) <= 1e-12 * max(1.0, cpu.max_region_error)
    assert gpu.report.to_dict() == cpu.report.to_dict() or gpu.report.to_text().splitlines()[1:] == \
        cpu.report.to_text().splitlines()[1:]


def test_native_loop_and_branch_on_the_gpu_reference(on, twin):
    on("gpu")
    ex = CompiledExecutor("cuda")
    rnn = make_rnn(ex)
    wx, wh = rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2)
    x = rnd(3, 5, 16)
    for sign in (1.0, -1.0):                     # both branches, each checked on its first run
        got = rnn(sign * x, wx, wh)
        want = ir.run(rnn.program, sign * x, wx, wh)[0]
        assert rel(got, want) < 1e-4
    assert twin["gpu_checks"] > 0 and twin["worst"] < 1e-12
    assert ex.last.end_to_end_device.startswith("cuda")


def test_interpreter_on_the_gpu_equals_the_cpu_with_control_flow():
    """ir.run(device=cuda): loops, branches, a while loop and return_all, against the CPU run."""
    @truestar.function
    def newton(x: Tensor["B"], w: Tensor["B", 8]) -> Tensor["B"]:
        y = x
        while ops.any(ops.abs(y * y - x) > 1e-9):
            y = 0.5 * (y + x / y)
        if (w.sum() > 0):
            y = y + w.mean(-1)
        else:
            y = y - w.mean(-1)
        return y

    x = torch.rand(6, generator=g, dtype=torch.float64) + 0.5
    w = rnd(6, 8)
    cpu, cpu_all = ir.run(newton.program, x, w, return_all=True)
    gpu, gpu_all = ir.run(newton.program, x, w, return_all=True, device="cuda")
    assert gpu[0].device.type == "cuda"
    assert rel(gpu[0], cpu[0]) < 1e-14
    assert set(gpu_all) == set(cpu_all) and all(rel(gpu_all[k], cpu_all[k]) < 1e-14 for k in cpu_all)
    rnn = make_rnn(None)
    args = (rnd(2, 7, 16), rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2))
    assert rel(ir.run(rnn.program, *args, device="cuda")[0], ir.run(rnn.program, *args)[0]) < 1e-14


# ----------------------------------------------------------------------------- attention in row blocks


@pytest.mark.parametrize("kind", ["plain", "causal", "bool_mask", "float_mask", "broadcast_mask", "scaled"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attention_in_row_blocks_equals_the_whole_matrix(kind, device, monkeypatch):
    q, k, v = rnd(2, 3, 37, 8).double(), rnd(2, 3, 29, 8).double(), rnd(2, 3, 29, 5).double()
    mask = None
    attrs = dict(scale=0.4 if kind == "scaled" else None, is_causal=kind == "causal")
    if kind == "bool_mask":
        mask = torch.rand(37, 29, generator=g) > 0.3
        mask[:, 0] = True
    elif kind == "float_mask":
        mask = rnd(2, 1, 37, 29).double()
    elif kind == "broadcast_mask":
        mask = rnd(1, 29).double()
    args = [t.to(device) if t is not None else None for t in (q, k, v, mask)]
    with torch.device(device):                   # as call_on_device runs it
        whole = iref._attention(attrs, args, (2, 3, 37, 5))
    for block in (1, 2, 5, 36):                 # rows per block: one, two, uneven last block
        monkeypatch.setattr(iref, "DEVICE_CHUNK_ELEMENTS", block * 2 * 3 * 29)
        with torch.device(device):
            rows = iref._attention_rows(attrs, args, (2, 3, 37, 5))
        assert rows.device == whole.device
        assert rel(rows, whole) < 1e-14, (kind, block)


def test_a_wrong_attention_is_caught_when_checked_in_row_blocks(on, monkeypatch):
    """The blocked reference is not weaker: an output wrong only in late query rows is caught, on
    the GPU with blocks of four rows exactly as on the CPU with the whole matrix."""
    real = runtime.KernelStep.run

    def late_rows_wrong(self, env, rt):
        real(self, env, rt)
        if "attention" in self.ops():
            env[self.out][:, :, 30:, :] *= 1.01       # after the kernel layer's own check passed

    monkeypatch.setattr(runtime.KernelStep, "run", late_rows_wrong)
    monkeypatch.setattr(iref, "DEVICE_CHUNK_ELEMENTS", 4 * 2 * 40)
    inputs = [rnd(2, 2, 33, 16), rnd(2, 2, 40, 16), rnd(2, 2, 40, 8)]
    reports = {}
    for where in ("cpu", "gpu"):
        on(where)
        reports[where] = _failure(lambda: compile_program(attention_program("causal"), "cuda")(inputs))
    assert reports["gpu"]["op"] == "attention" and reports["gpu"]["region"] == "attention_causal"
    for key in ("op", "region", "stage", "ops", "value", "tolerance", "message"):
        assert reports["gpu"][key] == reports["cpu"][key], key


# ----------------------------------------------------------------------------- broken compilers, same blame


def _small_mlp():
    b = Builder("guarded")
    x = b.input("x", torch.float32, ("B", 8))
    h = b.op("linear", x, b.param("w1", rnd(16, 8)), b.param("b1", rnd(16)))
    h = b.op("relu", h)
    h = b.op("linear", h, b.param("w2", rnd(4, 16)), None)
    return b.build([b.op("softmax", h, dim=-1)])


def _leaky(L, node):
    x = L.operand(node.inputs[0])
    return L.new(L.shape_of(node.output), "f", lambda idx, lo: K.maximum(lo(x, idx), lo(x, idx) * 0.1), (x,),
                 node=node)


def _bad_tanh(L, node):
    x = L.operand(node.inputs[0])
    return L.new(L.shape_of(node.output), "f", lambda idx, lo: K.tanh(lo(x, idx) * 1.05), (x,), node=node)


def _bad_matmul(L, node):
    lv = lower._matmul_lv(L, node, L.operand(node.inputs[0]), L.operand(node.inputs[1]), L.shape_of(node.output))
    fn = lv.fn
    lv.fn = lambda idx, lo: fn(idx, lo) * 1.01
    return lv


def _fused():
    b = Builder("fused")
    x = b.input("x", torch.float32, ("B", 8))
    return b.build([b.op("mul", b.op("sigmoid", b.op("tanh", b.op("add", x, x))), x)])


def _conv():
    b = Builder("conv")
    x = b.input("x", torch.float32, ("B", 3, 8, 8))
    h = b.op("conv2d", x, b.param("w", rnd(4, 3, 3, 3)), None, stride=[1, 1], padding=[1, 1], dilation=[1, 1],
             groups=1)
    return b.build([b.op("relu", h)])


def _failure(run):
    with pytest.raises(ProgramMismatchError) as e:
        run()
    err = e.value
    return {"op": err.op, "region": err.region, "program": err.program, "stage": err.stage, "ops": err.ops,
            "value": err.value, "message": str(err), "error": err.error, "tolerance": err.tolerance}


FAULTS = {
    "relu_rule": ("relu", _leaky, lambda: _small_mlp(), (5, 8)),
    "tanh_in_fused_region": ("tanh", _bad_tanh, lambda: _fused(), (4, 8)),
    "relu_after_resnet_conv": ("relu", _leaky, lambda: resnet_program()[0], (2, 3, 64, 64)),
    "relu_in_transformer": ("relu", _leaky, lambda: transformer_program()[0], (3, 16, 32)),
}


@pytest.mark.parametrize("fault", list(FAULTS))
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_a_broken_rule_is_blamed_the_same_on_the_gpu_and_the_cpu(fault, dtype, on, twin, monkeypatch):
    op, rule, make, shape = FAULTS[fault]
    if fault == "tanh_in_fused_region" and dtype == "float16":
        # A 5% tanh error behind a sigmoid moves the output by about 1.5e-3, inside the float16
        # tolerance (2e-3) on either device; measured, not assumed. The float32 case catches it.
        pytest.skip("below the float16 tolerance on the CPU reference too")
    prog = make()
    x = rnd(*shape)
    monkeypatch.setitem(lower.RULES, op, rule)
    reports = {}
    for where in ("cpu", "gpu"):
        on(where)
        cp = compile_program(prog, "cuda", dtype)
        reports[where] = _failure(lambda: cp([x]))
        assert not cp.verified
        assert _failure(lambda: cp([x]))["op"] == reports[where]["op"]      # a retry is checked again
    assert twin["gpu_checks"] > 0                                           # the GPU path really ran
    cpu, gpu = reports["cpu"], reports["gpu"]
    assert gpu["op"] == op and gpu["stage"] == "region"
    for key in ("op", "region", "program", "stage", "ops", "value", "tolerance", "message"):
        assert gpu[key] == cpu[key], (key, gpu[key], cpu[key])
    assert abs(gpu["error"] - cpu["error"]) <= 1e-9 * cpu["error"]


def test_a_broken_rule_inside_a_loop_body_is_blamed_the_same(on, twin, monkeypatch):
    monkeypatch.setitem(lower.RULES, "matmul", _bad_matmul)
    reports = {}
    for where in ("cpu", "gpu"):
        on(where)
        rnn = make_rnn(CompiledExecutor("cuda"))
        reports[where] = _failure(lambda: rnn(rnd(3, 4, 16), rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2)))
    assert twin["gpu_checks"] > 0
    assert reports["gpu"]["op"] == "matmul" and reports["gpu"]["region"] == "rnn/loop %h/body"
    for key in ("op", "region", "program", "stage", "ops", "value", "tolerance"):
        assert reports["gpu"][key] == reports["cpu"][key], key


def test_a_branch_is_checked_on_the_gpu_when_it_first_runs(on, twin, monkeypatch):
    on("gpu")
    monkeypatch.setitem(lower.RULES, "neg", _leaky)        # only the else branch negates
    ex = CompiledExecutor("cuda")
    rnn = make_rnn(ex)
    wx, wh = rnd(16, 32, scale=0.3), rnd(32, 32, scale=0.2)
    x = rnd(3, 4, 16)
    sign = 1.0 if bool(ir.run(rnn.program, x, wx, wh, return_all=True)[1]["gt"]) else -1.0
    rnn(sign * x, wx, wh)
    assert ex.last.verified
    checks = twin["gpu_checks"]
    with pytest.raises(ProgramMismatchError) as e:
        rnn(-sign * x, wx, wh)
    assert e.value.op == "neg" and e.value.region == "rnn/cond %h_1/else_branch"
    assert twin["gpu_checks"] > checks


def test_a_broken_library_kernel_is_blamed_the_same(on, twin, monkeypatch):
    prog = _conv()
    inputs = [rnd(2, 3, 8, 8)]
    real = runtime.F.conv2d
    monkeypatch.setattr(runtime.F, "conv2d", lambda *a, **k: real(*a, **k) * 1.001)
    reports = {}
    for where in ("cpu", "gpu"):
        on(where)
        reports[where] = _failure(lambda: compile_program(prog, "cuda", library=("conv2d",))(inputs))
    assert twin["gpu_checks"] > 0
    assert reports["gpu"]["op"] == "conv2d" and reports["gpu"]["stage"] == "region"
    for key in ("op", "region", "stage", "ops", "value", "tolerance", "message"):
        assert reports["gpu"][key] == reports["cpu"][key], key


def test_wrong_wiring_is_caught_end_to_end_on_the_gpu(on):
    b = Builder("layout")
    x = b.input("x", torch.float32, ("B", 6, 8))
    h = b.op("permute", b.op("tanh", x), dims=[0, 2, 1])
    prog = b.build([b.op("mul", b.op("select", h, dim=1, index=2), b.op("select", h, dim=1, index=2))])
    inputs = [rnd(4, 6, 8)]
    reports = {}
    for where in ("cpu", "gpu"):
        on(where)
        cp = compile_program(prog, "cuda")
        step = next(s for s in cp.plan.steps if s.kind == "layout" and s.nodes[0].op == "permute")
        real = step.fn
        step.fn = lambda attrs, args, shape, real=real: real(attrs, [args[0].flip(0)], shape)
        reports[where] = _failure(lambda: cp(inputs))
        assert cp.end_to_end_device is None or cp.end_to_end_device == ("cpu" if where == "cpu" else
                                                                        cp.end_to_end_device)
        step.fn = real
        cp(inputs)
        assert cp.verified
        assert cp.end_to_end_device.startswith("cuda" if where == "gpu" else "cpu")
    assert reports["gpu"]["stage"] == "end_to_end"
    for key in ("op", "region", "stage", "value", "tolerance", "message"):
        assert reports["gpu"][key] == reports["cpu"][key], key


def test_without_any_check_the_wrong_result_would_ship(monkeypatch):
    """Control: the faults above are caught by the checks, not by something else."""
    prog = _small_mlp()
    x = rnd(5, 8)
    want = ir.run(prog, x)[0]
    monkeypatch.setitem(lower.RULES, "relu", _leaky)
    monkeypatch.setenv("TRUESTAR_VERIFY", "0")
    assert rel(compile_program(prog, "cuda")([x])[0], want) > 1e-3


# ----------------------------------------------------------------------------- where the check runs


def test_auto_puts_big_checks_on_the_gpu_and_small_ones_on_the_cpu(on, twin):
    on("auto")
    prog, shape = mlp_program()
    cp = compile_program(prog, "cuda")
    cp([rnd(*shape)])                                   # 9 x 128 values: well below 2^20 elements of work
    assert twin["gpu_checks"] == 0 and cp.end_to_end_device == "cpu"
    b = Builder("wide")
    x = b.input("x", torch.float32, ("B", 1024))
    prog = b.build([b.op("tanh", b.op("linear", x, b.param("w", rnd(1024, 1024, scale=0.03)), None))])
    cp = compile_program(prog, "cuda")
    cp([rnd(8, 1024)])                                  # 8 x 1024 x 1024 multiply adds
    assert twin["gpu_checks"] > 0 and cp.end_to_end_device.startswith("cuda")
    assert twin["worst"] < 1e-12


def test_a_check_over_the_memory_budget_stays_on_the_cpu_and_still_catches(on, twin, monkeypatch):
    on("gpu")
    monkeypatch.setenv("TRUESTAR_VERIFY_GPU_MB", "0.00001")    # about 10 bytes: nothing fits
    prog = _small_mlp()
    x = rnd(5, 8)
    cp = compile_program(prog, "cuda")
    cp([x])
    assert cp.verified and twin["gpu_checks"] == 0
    monkeypatch.setitem(lower.RULES, "relu", _leaky)
    err = _failure(lambda: compile_program(prog, "cuda")([x]))
    assert err["op"] == "relu" and twin["gpu_checks"] == 0


def test_an_op_torch_lacks_on_the_device_falls_back_for_that_node_only(on, twin, monkeypatch):
    on("gpu")
    real = iref.IMPLS["tanh"]

    def cpu_only(attrs, args, shape):
        if args[0].device.type != "cpu":
            raise NotImplementedError("tanh not implemented for 'Double' on this device (simulated)")
        return real(attrs, args, shape)

    monkeypatch.setitem(iref.IMPLS, "tanh", cpu_only)
    monkeypatch.setattr(iref, "device_cpu_fallbacks", set())
    prog = _fused()
    cp = compile_program(prog, "cuda")
    cp([rnd(4, 8)])
    assert cp.verified and twin["gpu_checks"] > 0 and twin["worst"] < 1e-12
    assert iref.device_cpu_fallbacks == {"tanh"}


def test_a_real_error_on_the_device_is_not_hidden_as_a_fallback(monkeypatch):
    def broken(attrs, args, shape):
        raise RuntimeError("an unrelated failure")

    monkeypatch.setitem(iref.IMPLS, "tanh", broken)
    with pytest.raises(RuntimeError, match="unrelated"):
        iref.call_on_device("tanh", {}, [torch.zeros(3, dtype=torch.float64, device="cuda")], (3,),
                            torch.device("cuda"))


def test_cpu_programs_never_use_the_gpu_reference(on, twin):
    on("gpu")
    prog, shape = mlp_program()
    cp = compile_program(prog, "cpu")
    cp([rnd(*shape)])
    assert cp.verified and twin["gpu_checks"] == 0 and cp.end_to_end_device == "cpu"
