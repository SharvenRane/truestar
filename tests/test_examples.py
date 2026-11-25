"""Every example in examples/ runs, gives the result its docstring promises, and stays under a minute.

Each example runs through the real command line in a fresh process, from a temporary directory,
so repros, recordings and artifacts never land in the checkout.
"""

import os
import re
import shutil
import subprocess
import sys
import time

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
EXAMPLES = os.path.join(ROOT, "examples")
BUDGET_S = 60.0
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def run(tmp_path, *argv, env=None, module=True):
    """Run `python -m truestar argv` (or `python argv` with module=False) in tmp_path."""
    e = dict(os.environ, PYTHONPATH=SRC, PYTHONIOENCODING="utf-8", **(env or {}))
    cmd = [sys.executable] + (["-m", "truestar"] if module else []) + list(argv)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=tmp_path, env=e, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    took = time.perf_counter() - t0
    assert took < BUDGET_S, f"{argv} took {took:.1f} s"
    return p


def ex(name):
    return os.path.join(EXAMPLES, name)


def test_every_example_is_tested():
    tested = {"cpu_check.py", "fault_demo.py", "check_model.py", "backward_model.py",
              "record_diff_model.py", "invariance_model.py", "port_reference.py",
              "port_candidate.py", "export_model.py", "native_function.py", "custom_op.py",
              "fix_demo.py", "llama_kv_decode.py", "rl_kernels_pair.py", "llama_paged_decode.py",
              "range_model.py", "kernel_reference.py", "kernel_candidate.py", "device_softplus.py",
              "size_cliff.py", "parallel_model.py", "spike_demo.py", "precision_model.py", "nn_mlp.py",
              "nn_cnn.py", "nn_gpt.py", "llama_kvquant_decode.py"}
    assert {f for f in os.listdir(EXAMPLES) if f.endswith(".py")} == tested


def test_cpu_check_is_clean(tmp_path):
    p = run(tmp_path, "check", ex("cpu_check.py"), "--backends", "cpu")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "overall: OK" in p.stdout


def test_check_blames_the_one_pass_variance(tmp_path):
    p = run(tmp_path, "check", ex("check_model.py"), "--backends", "cpu")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "first divergent module: norm#0 (OnePassNorm)" in p.stdout
    assert "check_model.py:17" in p.stdout
    assert "first divergent op: aten.sub.Tensor" in p.stdout
    repro = tmp_path / "truestar_repro" / "cpu_norm_0" / "repro.py"
    assert repro.is_file()


@needs_cuda
def test_check_on_cuda_too(tmp_path):
    p = run(tmp_path, "check", ex("check_model.py"), "--backends", "cpu,cuda", "--no-repro")
    assert p.returncode == 1, p.stdout + p.stderr
    assert p.stdout.count("first divergent module: norm#0 (OnePassNorm)") == 2


@needs_cuda
def test_fault_demo_blames_the_cuda_only_layer(tmp_path):
    pytest.importorskip("torchvision")
    p = run(tmp_path, "check", ex("fault_demo.py"), "--backends", "cpu,cuda", "--no-repro")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "[OK] cpu" in p.stdout and "[DIVERGENT] cuda" in p.stdout
    assert "layer3.1.bn2.1" in p.stdout


def test_backward_check_finds_what_the_forward_check_cannot(tmp_path):
    fwd = run(tmp_path, "check", ex("backward_model.py"), "--backends", "cpu", "--no-repro")
    assert fwd.returncode == 0, fwd.stdout + fwd.stderr
    p = run(tmp_path, "check", ex("backward_model.py"), "--backward", "--backends", "cpu")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "TS180 first divergent backward: module act#0 (ScaledTanh)" in p.stdout
    assert "so the backward itself is wrong" in p.stdout
    assert (tmp_path / "truestar_repro" / "cpu_act_0_backward" / "grad_repro.py").is_file()


def test_record_then_diff(tmp_path):
    p = run(tmp_path, "record", ex("record_diff_model.py"), "-o", "before.tsrec",
            "--backend", "cpu", "--ops", "all")
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "before.tsrec").is_file()
    same = run(tmp_path, "diff", "before.tsrec", "--replay")
    assert same.returncode == 0, same.stdout + same.stderr
    assert "Verdict: IDENTICAL" in same.stdout
    moved = run(tmp_path, "diff", "before.tsrec", "--replay",
                env={"TRUESTAR_EXAMPLE_UPGRADE": "1"})
    assert moved.returncode == 1, moved.stdout + moved.stderr
    assert "Verdict: DIVERGENT" in moved.stdout
    assert "First module whose own result changed: norm#0 (Norm)" in moved.stdout


def test_invariance_names_the_batch_dependent_module(tmp_path):
    p = run(tmp_path, "invariance", ex("invariance_model.py"), "--backends", "cpu")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "first batch dependent module: norm#0 (Standardize)" in p.stdout
    assert "op: aten.mean.dim in norm#0" in p.stdout and "[TS052]" in p.stdout


def test_port_names_the_module_and_the_constant(tmp_path):
    p = run(tmp_path, "port", ex("port_reference.py"), ex("port_candidate.py"))
    assert p.returncode == 1, p.stdout + p.stderr
    assert "DIVERGENT: candidate blocks.0.norm (Norm)" in p.stdout
    assert "constant 1e-06 instead of 1e-05" in p.stdout


def test_kernel_names_the_shape_a_single_allclose_misses(tmp_path):
    p = run(tmp_path, "kernel", ex("kernel_reference.py"), ex("kernel_candidate.py"),
            "--device", "cpu")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "[TS200] wrong result beyond the derived tolerance on d1=1025 (was 1024)" in p.stdout
    assert "right on its own inputs" in p.stdout


def test_export_then_run(tmp_path):
    p = run(tmp_path, "export", ex("export_model.py"), "-o", "small_cnn.tsx")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "check second_batch: shapes [[7, 3, 16, 16]]" in p.stdout
    mk = run(tmp_path, ex("export_model.py"), "inputs.pt", module=False)
    assert mk.returncode == 0, mk.stderr
    r = run(tmp_path, "run", "small_cnn.tsx", "inputs.pt", "-o", "outputs.pt")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "output 0: float32[3, 10]" in r.stdout
    # The artifact's output matches the model it came from, without its code.
    sys.path.insert(0, EXAMPLES)
    try:
        import export_model
    finally:
        sys.path.remove(EXAMPLES)
    outs = torch.load(tmp_path / "outputs.pt")
    out = outs[0] if isinstance(outs, (list, tuple)) else outs
    with torch.no_grad():
        want = export_model.get_model()(torch.load(tmp_path / "inputs.pt"))
    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-5)
    info = run(tmp_path, "run", "small_cnn.tsx", "--info")
    assert info.returncode == 0 and "B in [1, inf]" in info.stdout  # unbounded since 2026-11-24


def test_native_function(tmp_path):
    p = run(tmp_path, ex("native_function.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "rnn(x: f32[B, T, 16]" in p.stdout
    assert "shape bug caught at decoration: shape mismatch in matmul" in p.stdout


def _worst(stdout: str) -> float:
    return float(re.search(r"worst relative difference to PyTorch ([0-9.e+-]+)", stdout).group(1))


def test_nn_mlp_trains_like_pytorch(tmp_path):
    p = run(tmp_path, ex("nn_mlp.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _worst(p.stdout) < 1e-9, p.stdout


def test_nn_cnn_trains_like_pytorch_with_batch_norm_state(tmp_path):
    p = run(tmp_path, ex("nn_cnn.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _worst(p.stdout) < 1e-9, p.stdout
    assert float(re.search(r"running variance ([0-9.e+-]+)", p.stdout).group(1)) < 1e-9


def test_nn_gpt_trains_like_pytorch_and_decodes_with_one_compile(tmp_path):
    p = run(tmp_path, ex("nn_gpt.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _worst(p.stdout) < 1e-9, p.stdout
    assert "1 compile for prompt and 15 decode steps; equals full recomputation: True" in p.stdout


def test_custom_op_on_cpu(tmp_path):
    p = run(tmp_path, ex("custom_op.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "torch.library.opcheck: passed" in p.stdout


@needs_cuda
def test_custom_op_on_cuda(tmp_path):
    p = run(tmp_path, ex("custom_op.py"), "cuda", module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "torch.library.opcheck: passed" in p.stdout


@needs_cuda
def test_fix_demo_finds_and_verifies_a_remedy(tmp_path):
    p = run(tmp_path, "fix", ex("fix_demo.py"), "--backends", "cuda", "--format", "agent")
    import json

    doc = json.loads(p.stdout)
    assert p.returncode == 0, p.stdout + p.stderr
    text = json.dumps(doc)
    assert "aten.addmm" in text and "mm_add" in text, text[:2000]


def test_llama_kv_decode_runs(tmp_path):
    p = run(tmp_path, ex("llama_kv_decode.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr


@needs_cuda
def test_rlpair_on_the_rl_kernels_is_bitwise(tmp_path):
    p = run(tmp_path, "rlpair", ex("rl_kernels_pair.py"), "--device", "cuda", "--kernels", "rl",
            "--strict", "--tp", "2")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "status: IDENTICAL" in p.stdout
    assert "TP 1 vs TP 2 bitwise" in p.stdout
    p = run(tmp_path, "rlpair", ex("rl_kernels_pair.py"), "--device", "cuda", "--strict")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "bitwise no" in p.stdout


def test_llama_paged_decode_runs(tmp_path):
    p = run(tmp_path, ex("llama_paged_decode.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "tokens equal: True, logits bitwise equal: True" in p.stdout, p.stdout


def test_range_example_finds_the_residual_overflow_and_rescales(tmp_path):
    p = run(tmp_path, "range", ex("range_model.py"), "--target", "fp16")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "status: REMEDIED" in p.stdout
    assert "TS301 add %add_9 in blocks.9 (Block) overflows" in p.stdout
    assert "range_model.py:33" in p.stdout
    assert "chosen: rescale" in p.stdout


@needs_cuda
def test_precision_example_keeps_the_planted_norm_in_float32(tmp_path):
    p = run(tmp_path, "compile", ex("precision_model.py"), "--max-flips", "1%", "--plan", "plan.json")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "plan (met)" in p.stdout and "[TS660]" in p.stdout
    assert "norm (OnePassNorm): fp32" in p.stdout
    assert "precision_loss at norm (OnePassNorm), aten.sub.Tensor, precision_model.py:22" in p.stdout
    assert (tmp_path / "plan.json").exists()


def test_device_softplus_on_the_simulated_ane(tmp_path):
    p = run(tmp_path, "check", ex("device_softplus.py"), "--backends", "cpu,sim:ieee-f16,sim:ane")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "[OK] cpu" in p.stdout and "[OK] sim:ieee-f16" in p.stdout
    assert "[DIVERGENT] sim:ane" in p.stdout
    assert "first divergent module: act#0 (Act)" in p.stdout
    assert "first divergent op: aten.softplus.default" in p.stdout
    repro = tmp_path / "truestar_repro" / "sim_ane_act_0"
    assert run(tmp_path, str(repro / "repro.py"), module=False).returncode == 1  # the simulation
    assert run(tmp_path, str(repro / "repro.py"), "cpu", module=False).returncode == 0  # control
    assert run(tmp_path, str(repro / "op_repro.py"), module=False).returncode == 1


def test_size_cliff_sweep(tmp_path):
    p = run(tmp_path, "check", ex("size_cliff.py"), "--backends",
            "cpu,sim:ieee-f16,sim:openvino-gpu-f16", "--sweep", "n:512..8192")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "[OK] cpu: clean at every size" in p.stdout
    assert "[OK] sim:ieee-f16: clean at every size" in p.stdout
    assert "sim:openvino-gpu-f16: clean up to n=2058; SUSPECT from n=2059" in p.stdout
    assert "op aten.bmm.default" in p.stdout


def test_probe_specials_cli(tmp_path):
    p = run(tmp_path, "probe", "--specials", "--backends", "cpu,sim:ane", "--format", "agent")
    import json

    doc = json.loads(p.stdout)
    assert p.returncode == 1 and doc["status"] == "divergent" and "TS523" in doc["codes"]
    ane = [r for r in doc["probe"]["rows"] if r["backend"] == "sim:ane" and r["case"] == "add_nan"]
    assert ane and ane[0]["got"] == "inf"


def test_parallel_layout_is_clean_and_names_an_injected_missing_sync(tmp_path):
    p = run(tmp_path, "parallel", ex("parallel_model.py"), "--layout", "tp=2,dp=2,ga=2,sp=1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "overall: OK" in p.stdout
    p = run(tmp_path, "parallel", ex("parallel_model.py"), "--layout", "tp=2,dp=2,ga=2,sp=1",
            "--inject", "missing_tp_sync")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "TS684 [blocks.0.norm1]" in p.stdout and "their sum over tp equals the single device gradient" in p.stdout


def test_spike_demo_replay_says_numerics(tmp_path):
    p = run(tmp_path, ex("spike_demo.py"), "run", module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "step 45:" in p.stdout and "TS691" in p.stdout
    p = run(tmp_path, "replay", "run", "--step", "45")
    assert p.returncode == 1, p.stdout + p.stderr
    assert "verdict: NUMERICS [TS692] the spike vanishes in float64" in p.stdout and "first split norm" in p.stdout


@needs_cuda
def test_llama_kvquant_decode_runs(tmp_path):
    p = run(tmp_path, ex("llama_kvquant_decode.py"), module=False)
    assert p.returncode == 0, p.stdout + p.stderr
    for fmt in ("int8", "fp8"):
        line = next(l for l in p.stdout.splitlines() if l.startswith(fmt + ":"))
        assert "fallbacks 0" in line
        assert int(line.split("%")[0].split()[-1]) >= 90      # greedy tokens equal to float32
