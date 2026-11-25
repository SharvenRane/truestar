"""Compilers and runtimes as check backends (docs/BACKENDS.md).

Every stack test injects one real fault into one stack at a known module and requires the check to
name that module and op, and requires the same model on the unmodified stack to be clean. Stacks
that are not installed here are skipped by name (TensorRT and the ORT CUDA provider live in a
separate environment on the development machine; scripts/stack_study.py runs them all)."""

import json
import os
import subprocess
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stack_models as S  # noqa: E402

from truestar import codes  # noqa: E402
from truestar.backends import BackendUnavailable, resolve_backends  # noqa: E402
from truestar.check import check  # noqa: E402
from truestar.stacks import inductor_kernels, probe  # noqa: E402

warnings.filterwarnings("ignore")


def _need(name):
    b, why = probe(name)
    if b is None:
        pytest.skip(f"{name} not available here: {why}")
    return b


# ----------------------------------------------------------------------------- naming, discovery


def test_unknown_stack_is_refused_with_reason():
    with pytest.raises(BackendUnavailable) as e:
        resolve_backends(["ort:tpu"], allow_stacks=True)
    assert "ort:tpu" in str(e.value) and "stacks are" in str(e.value)


def test_missing_stack_says_why():
    b, why = probe("trt:fp32")
    if b is not None:
        pytest.skip("TensorRT is installed here")
    with pytest.raises(BackendUnavailable) as e:
        resolve_backends(["cpu", "trt:fp32"], allow_stacks=True)
    assert "trt:fp32: " + why in str(e.value)


def test_stacks_refused_outside_check():
    # Other commands would run the model in eager PyTorch under the stack's name: refused.
    for name in ("ort:cpu", "cuda+inductor", "trt:fp16"):
        with pytest.raises(BackendUnavailable) as e:
            resolve_backends([name])
        assert "truestar check only" in str(e.value)


def test_stacks_refused_in_training_mode():
    _need("ort:cpu")
    with pytest.raises(BackendUnavailable) as e:
        check(S.Net().eval(), torch.randn(8, 16), backends=["ort:cpu"], train=True)
    assert "training mode" in str(e.value)


def test_sim_and_stack_in_one_check():
    _need("ort:cpu")
    from truestar.devsim import PROFILES

    sim = sorted(PROFILES)[0]
    torch.manual_seed(0)
    rep = check(S.Net().eval(), torch.randn(8, 16), backends=["cpu", sim, "ort:cpu"])
    names = [r.backend.name for r in rep.results]
    assert names[0] == "cpu" and names[2] == "ort:cpu" and names[1].startswith("sim:")
    assert rep.results[2].status == "ok" and rep.results[2].stack is not None


def test_stack_codes_are_registered_and_documented():
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "AGENT_INTERFACE.md"),
               encoding="utf-8").read()
    for c, slug in [("TS400", "stack_cpu_fallback"), ("TS401", "stack_provider_missing"),
                    ("TS402", "stack_fusion_only"), ("TS403", "stack_graph_breaks")]:
        assert codes.slug(c) == slug and c in doc


def test_inductor_kernel_origins_are_parsed():
    src = (
        "# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]\n"
        "triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_l', '''\n"
        "'''\n"
        "            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.addmm]\n"
        "            extern_kernels.mm(arg2_1, buf0, out=buf1)\n"
        "# Topologically Sorted Source Nodes: [sigmoid, mul], Original ATen: [aten.sigmoid, aten.mul]\n"
        "triton_poi_fused_mul_sigmoid_1 = async_compile.triton('triton_poi_fused_mul_sigmoid_1', '''\n")
    ks = inductor_kernels([src])
    assert [k["kernel"] for k in ks] == ["triton_per_fused_native_layer_norm_0", "extern_kernels.mm",
                                        "triton_poi_fused_mul_sigmoid_1"]
    assert ks[1]["extern"] and ks[2]["aten"] == ["aten.sigmoid", "aten.mul"]


# ----------------------------------------------------------------------------- ONNX Runtime CPU


def test_ort_cpu_clean_model_needs_one_compile():
    _need("ort:cpu")
    torch.manual_seed(0)
    r = check(S.Net("gelu").eval(), torch.randn(8, 16), backends=["ort:cpu"]).results[0]
    assert r.status == "ok" and r.codes == []
    assert r.stack["compiles"] == 1 and r.stack["localization"].startswith("not needed")
    assert r.output.rel_err < 1e-5


def test_ort_cpu_exporter_fault_named_at_module_and_op(tmp_path):
    _need("ort:cpu")
    torch.manual_seed(0)
    rep = check(S.GeluMLP().eval(), torch.randn(32, 16) * 2, backends=["ort:cpu"],
                stack_options={"ort:cpu": S.onnx_gelu_ignores_approximate()},
                repro_dir=str(tmp_path))
    r = rep.results[0]
    assert r.status != "ok" and "TS011" in r.codes
    assert r.first_divergent.key == "act1#0"
    assert r.first_divergent_op.op == "aten.gelu.default"
    assert any("eagerly on cpu" in n and "matches" in n for n in r.notes)
    # Localization compiled modules in order until the first bad one, not every module.
    assert r.stack["compiles"] <= 7
    # Repros run with the plain stack (the injected translation is not carried) and as control.
    for script in (r.repro_path, r.op_repro_path):
        for extra in ([], ["eager"]):
            p = subprocess.run([sys.executable, script] + extra, capture_output=True, text=True)
            assert p.returncode == 0, p.stdout + p.stderr


def test_ort_cpu_agent_json_and_cli(tmp_path, capsys):
    _need("ort:cpu")
    from truestar.cli import main

    f = tmp_path / "mlp.py"
    f.write_text(
        "import torch\n"
        "def get_model():\n"
        "    torch.manual_seed(0)\n"
        "    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 3))\n"
        "def get_inputs():\n"
        "    return torch.randn(4, 8)\n", encoding="utf-8")
    code = main(["check", str(f), "--backends", "cpu,ort:cpu", "--no-repro", "--format", "agent"])
    doc = json.loads(capsys.readouterr().out)
    json.dumps(doc, allow_nan=False)
    assert code == 0 and doc["status"] == "ok"
    ort = [b for b in doc["backends"] if b["name"] == "ort:cpu"][0]
    assert ort["stack"]["kind"] == "ort" and ort["stack"]["compiles"] == 1


# ----------------------------------------------------------------------------- Inductor on CUDA


def test_inductor_clean_model():
    _need("cuda+inductor")
    torch.manual_seed(0)
    r = check(S.Net().eval(), torch.randn(8, 16), backends=["cuda+inductor"]).results[0]
    assert r.status == "ok" and r.stack["localization"].startswith("not needed")


def test_inductor_miscompile_traced_to_fused_kernel():
    _need("cuda+inductor")
    torch.manual_seed(0)
    model = S.Net().eval()
    with S.inductor_wrong_sigmoid():
        r = check(model, torch.randn(8, 16), backends=["cuda+inductor"]).results[0]
    assert r.status != "ok"
    assert r.first_divergent.key == "blocks.0#0"
    op = r.first_divergent_op
    assert op.op == "aten.sigmoid.default"
    assert op.kernel is not None and "sigmoid" in op.kernel["kernel"]
    assert "aten.sigmoid" in op.kernel["aten"] and not op.kernel["extern"]
    assert any("not in the device kernel" in n for n in r.notes)
    # After the fault is removed the same model is clean again (no faulty kernel cached).
    r2 = check(model, torch.randn(8, 16), backends=["cuda+inductor"]).results[0]
    assert r2.status == "ok"


# ----------------------------------------------------------------------------- ORT CUDA, TensorRT


def test_ort_cuda_silent_cpu_fallback_reported():
    _need("ort:cuda")
    torch.manual_seed(0)
    r = check(S.DetNet().eval(), torch.randn(4, 6, 8), backends=["ort:cuda"]).results[0]
    assert "TS400" in r.codes and r.status == "suspect"
    msg = [i.message for i in r.issues if i.code == "TS400"][0]
    assert "Det" in msg and "module det" in msg


def test_ort_cuda_clean_model_has_no_fallback():
    _need("ort:cuda")
    torch.manual_seed(0)
    r = check(S.Net("gelu").eval(), torch.randn(8, 16), backends=["ort:cuda"]).results[0]
    assert r.status == "ok" and "TS400" not in r.codes


def test_trt_fp32_precision_fault_named():
    _need("trt:fp32")
    torch.manual_seed(0)
    model = S.ShiftNorm().eval()
    r = check(model, torch.randn(8, 16), backends=["trt:fp32"],
              stack_options={"trt:fp32": S.trt_norm_in_fp16()}).results[0]
    assert r.status == "divergent" and r.first_divergent.key == "norm#0"
    assert r.first_divergent_op.op == "aten.native_layer_norm.default"
    clean = check(model, torch.randn(8, 16), backends=["trt:fp32"]).results[0]
    assert clean.status == "ok"


def test_trt_fp16_overflow_named_and_blamed_on_precision():
    _need("trt:fp16")
    torch.manual_seed(0)
    r = check(S.RmsNet().eval(), torch.randn(64, 64), backends=["trt:fp16"]).results[0]
    assert r.status == "divergent" and r.first_divergent.key == "norm#0"
    assert r.first_divergent_op.op == "aten.pow.Tensor_Scalar"
    assert r.first_divergent_op.isolated.nonfinite_mismatch
    assert any("float16 itself cannot hold" in n for n in r.notes)
    clean = check(S.RmsNet().eval(), torch.randn(64, 64), backends=["trt:fp32"]).results[0]
    assert clean.status == "ok"
