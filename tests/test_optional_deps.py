"""Optional dependencies: without numba the CPU kernel paths fail with TS022, or fall back where
that is the documented behaviour, never with a raw ModuleNotFoundError from a generated file."""

import json
import os
import sys
import warnings

import pytest
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

import truestar  # noqa: E402
from truestar import Tensor, codes, customop, deps, ops  # noqa: E402
from truestar import kernel as K  # noqa: E402
from truestar.kernel import compiler  # noqa: E402


@pytest.fixture
def no_numba(monkeypatch):
    """Simulate a machine without numba: `import numba` raises ModuleNotFoundError."""
    monkeypatch.setitem(sys.modules, "numba", None)
    assert not deps.available("numba")


def _kernel(c=1.5):
    M = K.sym("M")
    x = K.buffer("x", (M,))
    i = K.axis("i", M)
    return K.kernel("scale", [K.output("y", (i,), x[i] * c + 0.25)])


def _assert_ts022(err):
    assert isinstance(err, ModuleNotFoundError) and isinstance(err, ImportError)
    assert err.code == "TS022" and err.module == "numba" and err.extra == "cpu-kernels"
    assert 'pip install "truestar[cpu-kernels]"' in str(err)
    doc = err.to_dict()
    json.dumps(doc, allow_nan=False)
    assert doc["code"] == "TS022" and doc["slug"] == "optional_dependency_missing"
    assert doc["severity"] == "error"
    assert doc["next_actions"][0] == {"kind": "install_extra",
                                      "description": "Install the optional module 'numba' (the "
                                                     "'cpu-kernels' extra).",
                                      "command": 'pip install "truestar[cpu-kernels]"'}
    return doc


def test_code_is_registered():
    assert codes.slug("TS022") == "optional_dependency_missing"
    assert codes.severity("TS022") == codes.ERROR


def test_numba_present_here():
    # The rest of the suite relies on numba; the simulation below is what tests its absence.
    assert deps.available("numba")


def test_kernel_compile_for_cpu_names_the_extra(no_numba):
    with pytest.raises(deps.MissingDependencyError) as ei:
        K.compile(_kernel(), "cpu")
    _assert_ts022(ei.value)


def test_generated_file_import_is_translated(no_numba, monkeypatch, tmp_path):
    # Even if the up front check is bypassed, the generated module's own `import numba` must not
    # surface as a raw ModuleNotFoundError.
    monkeypatch.setattr(deps, "require", lambda *a, **k: None)
    monkeypatch.setattr(compiler, "_memory", {})
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path))
    with pytest.raises(deps.MissingDependencyError) as ei:
        K.compile(_kernel(2.75), "cpu")
    _assert_ts022(ei.value)
    assert ei.value.__cause__ is None
    assert list((tmp_path / "numba").glob("k_*.py"))  # the file was written, then refused


def test_other_import_errors_are_not_relabelled(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "_memory", {})
    monkeypatch.setenv("TRUESTAR_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "math", None)  # generated files import math before numba
    with pytest.raises(ModuleNotFoundError) as ei:
        K.compile(_kernel(3.125), "cpu")
    assert not isinstance(ei.value, deps.MissingDependencyError)


def test_compiled_executor_on_cpu_refuses_with_the_reference_as_the_way_out(no_numba):
    from truestar.compiler import CompiledExecutor
    with pytest.raises(deps.MissingDependencyError) as ei:
        CompiledExecutor("cpu")
    doc = _assert_ts022(ei.value)
    assert [a["kind"] for a in doc["next_actions"]] == ["install_extra", "use_reference"]
    assert "reference interpreter" in str(ei.value)


def test_custom_op_falls_back_to_the_reference_with_the_reason(no_numba):
    @truestar.torch_op(name="deps_scale_shift_nonumba")
    def scale_shift(x: Tensor["B", 8], s: Tensor[8]) -> Tensor["B", 8]:
        return ops.relu(x) * s + 1.0

    x, s = torch.randn(3, 8), torch.randn(8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = scale_shift(x, s)
    slow = [m for m in w if issubclass(m.category, customop.SlowPathWarning)]
    assert slow, "the fallback must warn (never introduce #1)"
    assert "TS022" in str(slow[0].message) and "truestar[cpu-kernels]" in str(slow[0].message)
    plan = scale_shift.plan("cpu")
    assert plan.path == "reference" and "TS022" in plan.reason
    want = torch.relu(x.double()) * s.double() + 1.0
    assert y.dtype == torch.float32 and float((y.double() - want).abs().max()) < 1e-6


def test_run_cli_with_cpu_executor_exits_2_with_the_hint(no_numba, tmp_path, capsys):
    from torch import nn

    from truestar import export
    from truestar.export import runner
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3)).eval()
    path = str(tmp_path / "m.tsx")
    export.export(model, (torch.randn(2, 4),), path)
    inp = str(tmp_path / "x.pt")
    torch.save(torch.randn(2, 4), inp)
    assert runner.main([path, inp, "--executor", "cpu"]) == 2
    err = capsys.readouterr().err
    assert "TS022" in err and 'pip install "truestar[cpu-kernels]"' in err
    assert runner.main([path, inp]) == 0  # the reference executor needs no numba


def test_truestar_run_forwards_the_executor(no_numba, tmp_path, capsys):
    from torch import nn

    from truestar import export
    from truestar.cli import main
    torch.manual_seed(1)
    path = str(tmp_path / "m.tsx")
    export.export(nn.Sequential(nn.Linear(4, 3)).eval(), (torch.randn(2, 4),), path)
    inp = str(tmp_path / "x.pt")
    torch.save(torch.randn(2, 4), inp)
    assert main(["run", path, inp, "--executor", "cpu"]) == 2
    assert "TS022" in capsys.readouterr().err
    assert main(["run", path, inp]) == 0
