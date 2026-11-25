"""The compiled executor follows PyTorch's float32 matmul precision unless it is pinned.

PyTorch decides TF32 for float32 matmuls with ``torch.backends.cuda.matmul.allow_tf32``,
``torch.backends.cuda.matmul.fp32_precision`` or ``torch.set_float32_matmul_precision``. The
compiled executor reads the same setting at every call (``allow_tf32=None``, the default) and runs
the variant compiled for it, each verified against float64 with its own tolerance; ``allow_tf32``
True or False pins the choice whatever PyTorch says.
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

import truestar  # noqa: E402
from truestar import Tensor, ir, ops  # noqa: E402
from truestar.compiler import STATS, CompiledExecutor, compile_program  # noqa: E402
from truestar.compiler.runtime import torch_allows_tf32  # noqa: E402


def _has_gpu():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


HAS_GPU = _has_gpu()
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA and Triton")


@pytest.fixture
def torch_precision():
    """Restore PyTorch's matmul precision after the test, whichever API the test used."""
    m = torch.backends.cuda.matmul
    saved = (m.allow_tf32, torch.get_float32_matmul_precision(), getattr(m, "fp32_precision", None))
    yield
    # Restore every API, legacy last: this clears torch's "mixed APIs" state, in which reading
    # allow_tf32 or get_float32_matmul_precision() raises (it broke 74 later tests once).
    torch.set_float32_matmul_precision(saved[1])
    m.allow_tf32 = saved[0]
    if saved[2] is not None:
        m.fp32_precision = saved[2]
    assert (m.allow_tf32, torch.get_float32_matmul_precision()) == saved[:2]


SETTERS = {
    "allow_tf32": (lambda on: setattr(torch.backends.cuda.matmul, "allow_tf32", on)),
    "fp32_precision": (lambda on: setattr(torch.backends.cuda.matmul, "fp32_precision",
                                          "tf32" if on else "ieee")),
    "set_float32_matmul_precision": (lambda on: torch.set_float32_matmul_precision(
        "high" if on else "highest")),
    "set_float32_matmul_precision medium": (lambda on: torch.set_float32_matmul_precision(
        "medium" if on else "highest")),
}


@pytest.mark.parametrize("api", sorted(SETTERS))
def test_reader_agrees_with_what_torch_runs(api, torch_precision):
    """torch_allows_tf32 matches every way of setting the precision, including mixed APIs, where
    reading allow_tf32 itself raises."""
    if not hasattr(torch.backends.cuda.matmul, "fp32_precision") and api == "fp32_precision":
        pytest.skip("torch without fp32_precision")
    for on in (True, False, True):
        SETTERS[api](on)
        assert torch_allows_tf32() is on
    # mixing: the legacy flag after the new API, then the new API after the legacy flag
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch_allows_tf32() is False
    SETTERS[api](True)
    assert torch_allows_tf32() is True


def _mlp_program():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 128)).eval()
    B = torch.export.Dim("B", min=1, max=512)
    return model, ir.import_module(model, (torch.randn(4, 256),), dynamic_shapes=({0: B},))


def _err(out, ref):
    out, ref = out.detach().double().cpu(), ref.detach().double().cpu()
    return ((out - ref).abs().max() / ref.abs().max()).item()


@needs_gpu
@pytest.mark.parametrize("api", ["allow_tf32", "set_float32_matmul_precision"])
def test_default_follows_torch_both_ways(api, torch_precision):
    model, prog = _mlp_program()
    x = torch.randn(300, 256)
    ref = model.double()(x.double())
    SETTERS[api](False)
    cp = compile_program(prog, "cuda", "float32")
    assert cp.follows_tf32 and cp.opts.allow_tf32 is False
    strict = cp([x])[0]
    assert cp.report.fp32_precision == "ieee" and "follows torch" in cp.report.to_text()
    e_strict = _err(strict, ref)
    assert e_strict < 1e-5                       # IEEE level
    programs = STATS["programs"]
    SETTERS[api](True)
    tf32 = cp([x])[0]
    assert cp.opts.allow_tf32 is True and cp.report.fp32_precision == "tf32"
    assert STATS["programs"] == programs + 1     # the TF32 variant compiled once, on first use
    e_tf32 = _err(tf32, ref)
    assert 1e-5 < e_tf32 < 5e-3                  # TF32 level, and verified with the TF32 tolerance
    assert cp.verified
    SETTERS[api](False)
    again = cp([x])[0]
    assert torch.equal(again, strict)            # back to the IEEE variant, no new compile
    SETTERS[api](True)
    assert torch.equal(cp([x])[0], tf32)
    assert STATS["programs"] == programs + 1


@needs_gpu
@pytest.mark.parametrize("pinned", [False, True])
def test_pinned_setting_ignores_torch(pinned, torch_precision):
    model, prog = _mlp_program()
    x = torch.randn(64, 256)
    ref = model.double()(x.double())
    outs = []
    for torch_on in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = torch_on
        cp = compile_program(prog, "cuda", "float32", allow_tf32=pinned)
        assert not cp.follows_tf32 and cp.opts.allow_tf32 is pinned
        assert "pinned" in cp.report.to_text()
        outs.append(cp([x])[0])
    assert torch.equal(outs[0], outs[1])
    assert (_err(outs[0], ref) > 1e-5) == pinned


@needs_gpu
def test_native_function_follows_torch(torch_precision):
    ex = CompiledExecutor("cuda")

    @truestar.function(executor=ex)
    def f(x: Tensor["B", 384], w: Tensor[384, 256]) -> Tensor["B", 256]:
        return ops.relu(x @ w)

    torch.manual_seed(1)
    x, w = torch.randn(96, 384), torch.randn(384, 256)
    ref = torch.relu(x.double() @ w.double())
    torch.backends.cuda.matmul.allow_tf32 = False
    e0 = _err(f(x, w), ref)
    torch.backends.cuda.matmul.allow_tf32 = True
    e1 = _err(f(x, w), ref)
    assert e0 < 1e-5 < e1 < 5e-3
    assert ex.compile_count == 1                 # one program, two precision variants inside it


@needs_gpu
def test_float16_programs_ignore_the_flag(torch_precision):
    _, prog = _mlp_program()
    cp = compile_program(prog, "cuda", "float16")
    x = torch.randn(32, 256)
    torch.backends.cuda.matmul.allow_tf32 = False
    a = cp([x])[0]
    programs = STATS["programs"]
    torch.backends.cuda.matmul.allow_tf32 = True
    b = cp([x])[0]
    assert torch.equal(a, b) and STATS["programs"] == programs


@needs_gpu
def test_deterministic_follows_torch_and_keeps_rows_batch_invariant():
    """Split K makes a row's rounding depend on its batch; with torch's deterministic flag on the
    compiled program switches to its variant without split K, where rows are bitwise equal."""
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU()).eval()
    B = torch.export.Dim("B", min=1, max=512)
    prog = ir.import_module(model, (torch.randn(4, 4096),), dynamic_shapes=({0: B},))
    cp = compile_program(prog, "cuda", "float32", allow_tf32=False)
    assert cp.follows_deterministic and not cp.opts.deterministic
    x = torch.randn(300, 4096)

    def rows():
        return [cp([x[:m]])[0][:7] for m in (7, 64, 300)]

    prev = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        fast = rows()
        assert not all(torch.equal(r, fast[0]) for r in fast)
        torch.use_deterministic_algorithms(True)
        det = rows()
        assert cp.opts.deterministic
        assert all(torch.equal(r, det[0]) for r in det)
        torch.use_deterministic_algorithms(False)
        assert torch.equal(cp([x[:7]])[0][:7], fast[0])      # back to the split K variant
    finally:
        torch.use_deterministic_algorithms(prev)
    pinned = compile_program(prog, "cuda", "float32", allow_tf32=False, deterministic=True)
    assert not pinned.follows_deterministic
    prow = [pinned([x[:m]])[0][:7] for m in (7, 64, 300)]
    assert all(torch.equal(r, det[0]) for r in prow)
