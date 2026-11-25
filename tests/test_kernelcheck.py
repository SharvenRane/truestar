"""`truestar kernel` (R1): many input variants, derived tolerances, a named property per failure,
against a single input allclose."""

import json
import os
import subprocess
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from truestar import kernelbench
from truestar.kernelcheck import check_kernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
EX = os.path.join(ROOT, "examples")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CASES = {c.name: c for c in kernelbench.cases()}


def _inputs(case, draw=0, device="cpu"):
    g = torch.Generator().manual_seed(2000 + draw)
    return tuple(t.to(device) for t in case.make_inputs(case.dims, case.dtype, g))


def _check(name, draw=0):
    c = CASES[name]
    return check_kernel(c.reference, c.candidate, _inputs(c, draw))


@pytest.mark.parametrize("name", ["mm_blocked_fp16", "mm_split_k_fp32", "softmax_online_fp16",
                                  "attn_causal_bf16", "ln_welford_offset_fp32",
                                  "rowsum_sequential_fp32", "gelu_torch_fp32", "silu_unfused_bf16"])
def test_correct_kernels_pass_every_variant(name):
    rep = _check(name)
    assert rep.status == "ok", rep.to_text()
    assert len(rep.variants) >= 10


@pytest.mark.parametrize("name,code,kind", [
    ("mm_column_tail_even_shape", "TS200", "shape"),  # right at 256 columns, wrong at 257
    ("mm_contiguity_fp32", "TS204", "stride"),  # right on contiguous inputs only
    ("softmax_no_max_fp32", "TS200", "family"),  # right until the logits are large
    ("rowsum_racy_fp32", None, None),  # a lost update: wrong values or different bits
])
def test_bugs_a_single_allclose_misses_are_named(name, code, kind):
    c = CASES[name]
    x = _inputs(c)
    out, ref = c.candidate(*x), c.reference(*x)
    if name != "rowsum_racy_fp32":
        assert torch.allclose(out.double(), ref.double(), rtol=1e-2, atol=1e-2)  # baseline passes
    rep = check_kernel(c.reference, c.candidate, x)
    assert rep.status != "ok", rep.to_text()
    if code:
        assert rep.first_failure.code == code and rep.first_failure.kind == kind, rep.to_text()


def test_rates_against_single_input_allclose():
    """On every CPU case, one draw: no correct kernel rejected and no bug accepted (the two below
    the per element bound, a float16 scale of 1e-3 and the bfloat16 tail tile scaled by 1 + 2^-7,
    are caught by the lean verdict); the KernelBench style allclose accepts at least a third."""
    ff = fp = base_fp = n_bad = 0
    for c in kernelbench.cases():
        x = _inputs(c)
        rep = check_kernel(c.reference, c.candidate, x)
        out, ref = c.candidate(*x), c.reference(*x)
        base = bool(torch.allclose(out.double(), ref.double(), rtol=1e-2, atol=1e-2))
        if c.correct:
            ff += rep.status != "ok"
        else:
            n_bad += 1
            fp += rep.status == "ok"
            base_fp += base
    assert ff == 0
    assert fp == 0
    assert base_fp >= n_bad // 3


def test_shape_failure_and_input_mutation_codes():
    ref = lambda x: x * 2  # noqa: E731

    def picky(x):
        if x.shape[0] != 16:
            raise ValueError("only 16 rows")
        return x * 2

    def writes(x):
        x.mul_(2)
        return x.clone()

    x = torch.randn(16, 40)
    rep = check_kernel(ref, picky, (x,))
    assert "TS201" in rep.codes
    rep = check_kernel(ref, writes, (x,))
    assert rep.first_failure.code == "TS206"


def test_wrong_integer_output_is_reported_not_raised():
    """An argmax kernel whose index is off by one (held out METR level 1 tasks 51, 52): the check
    used to raise TypeError while formatting the message."""
    ref = lambda x: torch.argmax(x, dim=1)  # noqa: E731
    off = lambda x: (torch.argmax(x, dim=1) + 1) % x.shape[1]  # noqa: E731
    rep = check_kernel(ref, off, (torch.randn(16, 40),))
    assert rep.status != "ok" and rep.first_failure.code == "TS200"
    assert "integer or boolean values differ" in rep.first_failure.message


class _Float32Fault(TorchDispatchMode):
    """aten.rsqrt off by 1% in float32 only (float64 stays exact): a wrong backend op."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if str(func.overloadpacket) == "aten.rsqrt" and out.dtype == torch.float32:
            return out * 1.01
        return out


def test_op_blame_names_the_wrong_aten_op():
    ref = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)  # noqa: E731
    x = torch.randn(32, 128)
    with _Float32Fault():
        rep = check_kernel(ref, ref, (x,), shapes=False)
    assert rep.status == "divergent"
    assert rep.blamed_op is not None and rep.blamed_op.op == "aten.rsqrt.default", rep.to_text()


def test_cli_agent_json_and_text():
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-m", "truestar", "kernel",
                        os.path.join(EX, "kernel_reference.py"),
                        os.path.join(EX, "kernel_candidate.py"), "--device", "cpu",
                        "--format", "agent"], capture_output=True, text=True, env=env,
                       timeout=300)
    assert p.returncode == 1, p.stderr
    doc = json.loads(p.stdout)
    assert doc["command"] == "kernel" and doc["status"] == "divergent"
    assert doc["codes"][0] == "TS200"
    assert doc["first_failure"]["kind"] == "shape"
    assert any(a["kind"] == "inspect_candidate" and a["file"].endswith("kernel_candidate.py")
               for a in doc["next_actions"])
    assert "right on its own inputs" in doc["op_note"]


@needs_cuda
@pytest.mark.parametrize("name,code", [("triton_mm_fp16", None), ("triton_softmax_fp32", None),
                                       ("triton_mm_floor_grid_fp16", "TS205"),
                                       ("triton_mm_rowmajor", "TS204"),
                                       ("triton_rowsum_atomic", "TS203"),
                                       ("triton_softmax_edge", "TS200")])
def test_triton_kernels_on_the_gpu(name, code):
    from truestar import kernelbench_triton

    c = next(c for c in kernelbench_triton.cases() if c.name == name)
    rep = check_kernel(c.reference, c.candidate, _inputs(c, device="cuda"))
    if code is None:
        assert rep.status == "ok", rep.to_text()
    else:
        assert code in rep.codes, rep.to_text()


def test_lean_verdict_catches_the_bf16_tail_tile_scale_and_no_correct_kernel():
    """--lean (TS1300, the TS960 statistic): the bfloat16 last row tile scaled by 1 + 2^-7, which
    the per element bound passes, is caught; no correct CPU case fires."""
    fired = []
    for c in kernelbench.cases():
        if not (c.correct or c.name in ("mm_last_row_tile_scale_bf16", "mm_scale_1e-3_fp16")):
            continue
        rep = check_kernel(c.reference, c.candidate, _inputs(c), lean=True, drill_ops=False)
        if "TS1300" in rep.codes:
            fired.append(c.name)
    assert "mm_last_row_tile_scale_bf16" in fired
    assert [n for n in fired if CASES[n].correct] == []


def test_nan_dropped_by_maximum_number_is_its_own_verdict_and_named():
    """A row max that drops NaN (fmaxf, IEEE maximumNumber) while the reference propagates it:
    TS202 suspect with the rule named, not a wrong number; a NaN turned into 0 is not named."""
    ref = lambda x: torch.amax(x, dim=1)  # noqa: E731
    drop = lambda x: torch.amax(torch.nan_to_num(x, nan=float("-inf")), dim=1)  # noqa: E731
    zero = lambda x: torch.amax(torch.nan_to_num(x, nan=0.0), dim=1)  # noqa: E731
    x = torch.randn(32, 64, generator=torch.Generator().manual_seed(0)) - 8.0  # maxima < 0
    rep = check_kernel(ref, drop, (x,), shapes=False, strides=False)
    assert rep.status == "suspect" and rep.numerics == "ok" and "TS202" in rep.codes
    assert "maximumNumber" in next(v.message for v in rep.variants if v.code == "TS202")
    rep = check_kernel(ref, zero, (x,), shapes=False, strides=False)
    v = next(v for v in rep.variants if v.code == "TS202")
    assert "maximumNumber" not in v.message


def test_argmax_that_drops_nan_is_the_nan_verdict():
    """Held out KernelBench and METR argmax kernels skip NaN (fmaxf) where torch.argmax returns
    the NaN's index: an integer output, so only the maximumNumber match can tell it from a wrong
    index. A real off by one index stays divergent."""
    ref = lambda x: torch.argmax(x, dim=1)  # noqa: E731
    drop = lambda x: torch.argmax(torch.nan_to_num(x, nan=float("-inf")), dim=1)  # noqa: E731
    off = lambda x: (torch.argmax(x, dim=1) + 1) % x.shape[1]  # noqa: E731
    x = torch.randn(16, 40, generator=torch.Generator().manual_seed(1))
    rep = check_kernel(ref, drop, (x,), shapes=False, strides=False)
    assert rep.numerics == "ok" and "TS202" in rep.codes, rep.to_text()
    assert check_kernel(ref, off, (x,), shapes=False, strides=False).numerics == "divergent"


def _kernel_from_source(tmp_path, name, text):
    import importlib.util

    p = tmp_path / f"{name}.py"
    p.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel


def test_lean_from_reduced_precision_in_the_source_is_its_own_verdict(tmp_path):
    """A float32 matmul whose errors lean (scaled by 1 + 2^-20, inside the size bound): TS1301,
    suspect, when its source asks for TF32; TS1300, divergent, when it does not."""
    body = "def kernel(a, b):\n    return (a @ b) * (1 + 2 ** -20)\n"
    tf32 = _kernel_from_source(tmp_path, "cand_tf32",
                               "# torch.backends.cuda.matmul.allow_tf32 = True\n" + body)
    plain = _kernel_from_source(tmp_path, "cand_plain", body)
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(256, 512, generator=g), torch.randn(512, 128, generator=g))
    ref = lambda a, b: a @ b  # noqa: E731
    rep = check_kernel(ref, tf32, x, shapes=False, strides=False, families=False)
    assert "TS1301" in rep.codes and rep.numerics == "ok", rep.to_text()
    rep = check_kernel(ref, plain, x, shapes=False, strides=False, families=False)
    assert "TS1300" in rep.codes and rep.numerics == "divergent", rep.to_text()


def test_reference_out_of_memory_is_raised_not_a_skipped_variant():
    """Held out runs: a float64 reference that ran out of GPU memory on the value families was
    read as "the reference rejects this input" and skipped, so a wrong kernel passed."""
    def ref(x):
        if float(x.abs().max()) > 50:  # the large values family
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 242.00 MiB")
        return x * 2

    with pytest.raises(torch.OutOfMemoryError):
        check_kernel(ref, lambda x: x * 2, (torch.randn(8, 8),), shapes=False, strides=False)
