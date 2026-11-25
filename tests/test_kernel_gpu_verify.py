"""Kernel verification with the float64 reference on the GPU.

The reference interpreter and the comparison run on the kernel's own CUDA device. These tests pin
down that this is the same check as the NumPy one: the same results to float64 rounding, the same
faults caught, every new shape signature still checked.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="truestar_kcache_"))

from truestar import kernel as K  # noqa: E402
from truestar.kernel import Graph, compiler, reference, verify  # noqa: E402


def _has_triton():
    try:
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_triton(), reason="needs CUDA and Triton")
DEVICES = ["gpu", "auto", "cpu"]  # TRUESTAR_VERIFY_DEVICE: GPU always, by size, NumPy always


def rnd(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to("cuda", dtype)


def _cases():
    """(kernel, inputs) pairs covering every IR node kind the reference evaluates."""
    out = []
    M, N = K.sym("M"), K.sym("N")
    a, b = K.buffer("a", (M, N)), K.buffer("b", (M, N))
    i, j = K.axis("i", M), K.axis("j", N)
    e = K.where((a[i, j] >= 0.0) & (b[i, j] <= 0.5), K.exp(a[i, j]) / (K.abs_(b[i, j]) + 0.5),
                K.maximum(K.sqrt(K.abs_(a[i, j])), K.rsqrt(K.abs_(b[i, j]) + 0.5)))
    e = e + K.sigmoid(b[i, j]) - K.tanh(a[i, j]) * K.log(K.abs_(a[i, j]) + 0.1)
    out.append((K.kernel("pw", [K.output("y", (i, j), e)]),
                {"a": rnd(37, 53, seed=1), "b": rnd(37, 53, seed=2)}))
    x = K.buffer("x", (M, N))
    r = K.axis("r", N)
    out.append((K.kernel("red", [K.output("s", (i,), K.reduce_sum(r, x[i, r] * 2.0)),
                                 K.output("m", (i,), K.reduce_max(r, x[i, r])),
                                 K.output("n", (i,), K.reduce_min(r, x[i, r]))]),
                {"x": rnd(9, 3001, seed=3)}))
    g = Graph()
    ga, gb = g.input("a", (40, 30)), g.input("b", (30, 20))
    k = g.fuse({"y": ga @ gb + 1.0}, "cuda")[0][1]
    out.append((k, {"a": rnd(40, 30, seed=4), "b": rnd(30, 20, seed=5)}))
    g = Graph()
    B, H, S = 2, 3, K.sym("S")
    q, kk, v = (g.input(n, (B, H, S, 32), "float16") for n in "qkv")
    s = g.pointwise(lambda s_, b_, h_, i_, j_: K.where(j_ <= i_, s_, K.NEG_INF),
                    (q @ kk.T) * (32 ** -0.5), with_index=True)
    k = g.fuse({"o": g.softmax(s) @ v}, "cuda")[0][1]
    out.append((k, {n: rnd(2, 3, 71, 32, dtype=torch.float16, seed=6 + t)
                    for t, n in enumerate("qkv")}))
    return out


def test_gpu_reference_equals_numpy_reference():
    reference.cpu_fallbacks.clear()
    for k, inputs in _cases():
        g = reference.run_reference_torch(k, inputs, torch.device("cuda"))
        c = reference.run_reference(k, {n: v.double().cpu().numpy() for n, v in inputs.items()})
        for name in c:
            assert g[name].dtype == torch.float64 and g[name].device.type == "cuda"
            chk = verify.compare_numpy(name, g[name].cpu().numpy(), c[name], 1e-12)
            assert chk.ok, (k.name, name, chk)
    assert not reference.cpu_fallbacks  # every op ran on the GPU in float64


@pytest.mark.parametrize("cap", [100, 4600, 9000, 200000])
def test_gpu_reference_chunked_equals_whole(monkeypatch, cap):
    """Blocks of one index, blocks of several with a partial last block, and blocks over an
    earlier axis: all equal the unchunked evaluation and the NumPy reference."""
    k, inputs = _cases()[-1]
    whole = reference.run_reference_torch(k, inputs, torch.device("cuda"))["o"]
    c = reference.run_reference(k, {n: v.double().cpu().numpy() for n, v in inputs.items()})["o"]
    monkeypatch.setattr(reference, "GPU_CHUNK_ELEMENTS", cap)
    chunked = reference.run_reference_torch(k, inputs, torch.device("cuda"))["o"]
    assert (whole - chunked).abs().max().item() < 1e-12
    assert abs(chunked.cpu().numpy() - c).max() < 1e-12


def test_op_missing_on_gpu_falls_back_for_that_op_only(monkeypatch):
    k, inputs = _cases()[0]
    want = reference.run_reference_torch(k, inputs, torch.device("cuda"))["y"]
    table = reference._t_unary()

    def no_tanh(x):
        raise NotImplementedError("tanh not on this device in float64 (simulated)")
    table["tanh"] = no_tanh
    monkeypatch.setattr(reference, "_t_unary", lambda: table)
    reference.cpu_fallbacks.clear()
    got = reference.run_reference_torch(k, inputs, torch.device("cuda"))["y"]
    assert reference.cpu_fallbacks == {"tanh"}
    assert got.device.type == "cuda"
    assert (got - want).abs().max().item() < 1e-12
    reference.cpu_fallbacks.clear()


def test_compare_torch_matches_compare_numpy():
    cases = [([1.0, 2.0, np.nan], [1.0, 2.0, np.nan], 1e-6),
             ([1.0, np.nan, np.nan], [1.0, 2.0, np.nan], 1e-6),
             ([1.0, 2.0, 3.0], [1.0, 2.0, np.nan], 1e-6),
             ([1.0, 2.1, np.nan], [1.0, 2.0, np.nan], 1e-3),
             ([np.inf, 1.0], [np.inf, 1.0], 1e-6),
             ([-np.inf, 1.0], [np.inf, 1.0], 1e-6),
             ([np.inf, 1.0], [5.0, 1.0], 1e-6),
             ([1e-9, 2e-9], [0.0, 0.0], 1e-3),
             ([1.0, 2.0], [1.0, 2.0, 3.0], 1e-6)]
    for out, ref, tol in cases:
        a = verify.compare_numpy("o", np.array(out), np.array(ref), tol)
        b = verify.compare("o", torch.tensor(out, dtype=torch.float64, device="cuda"),
                           torch.tensor(ref, dtype=torch.float64, device="cuda"), tol)
        assert a.ok == b.ok and a.detail == b.detail, (out, ref, a, b)
        assert a.error == pytest.approx(b.error, rel=1e-12) or a.error == b.error


def test_cuda_kernels_are_checked_on_the_gpu(monkeypatch):
    def no_cpu(*a, **kw):
        raise AssertionError("the NumPy reference ran")
    monkeypatch.setenv("TRUESTAR_VERIFY_DEVICE", "gpu")
    monkeypatch.setattr(reference, "run_reference", no_cpu)
    k, inputs = _cases()[1]
    fn = K.compile(k, "cuda")
    fn(**inputs)  # verified on the GPU, the CPU reference is never called
    assert fn.verify_seconds > 0
    # default mode: a big check goes to the GPU, a small one stays on the CPU
    monkeypatch.delenv("TRUESTAR_VERIFY_DEVICE", raising=False)
    M = K.sym("M")
    x, w = K.buffer("x", (M, 512)), K.buffer("w", (512, 512))
    i, j, r = K.axis("i", M), K.axis("j", 512), K.axis("r", 512)
    big = K.kernel("big", [K.output("y", (i, j), K.reduce_sum(r, x[i, r] * w[r, j]) + 0.125)])
    assert reference.work_estimate(big, {"M": 256}) == 256 * 512 * 512 >= reference.GPU_MIN_WORK
    K.compile(big, "cuda")(x=rnd(256, 512), w=rnd(512, 512, seed=1))
    with pytest.raises(AssertionError, match="NumPy reference ran"):
        K.compile(big, "cuda")(x=rnd(2, 512), w=rnd(512, 512, seed=1))


@pytest.mark.parametrize("device", DEVICES)
def test_broken_kernels_caught_with_either_reference(monkeypatch, device):
    monkeypatch.setenv("TRUESTAR_VERIFY_DEVICE", device)
    M = K.sym("M")
    x = K.buffer("x", (M,))
    i = K.axis("i", M)
    k = K.kernel("chain", [K.output("y", (i,), K.sigmoid(x[i] * {"cpu": 3.25, "gpu": 3.5, "auto": 3.75}[device])
                                    + 1.0)])
    fn = K.compile(k, "cuda", _break={"sigmoid": lambda e: f"(tl.sigmoid({e}) * 1.01)"})
    with pytest.raises(K.KernelMismatchError, match="kernel 'chain'"):
        fn(x=rnd(1000, seed=20))
    g = Graph()
    a, b = g.input("a", (40, 30)), g.input("b", (30, 20))
    k = g.fuse({"y": a @ b + {"cpu": 1.0, "gpu": 2.0, "auto": 3.0}[device]}, "cuda")[0][1]
    fn = K.compile(k, "cuda", _break={"add": lambda x_, y_: f"({x_}) + ({y_}) * 1.02"})
    with pytest.raises(K.KernelMismatchError):
        fn(a=rnd(40, 30), b=rnd(30, 20))
    # NaN where the reference has none
    k = K.kernel("nanny", [K.output("y", (i,), K.exp(x[i]) + {"cpu": 0.5, "gpu": 0.75, "auto": 0.875}[device])])
    fn = K.compile(k, "cuda", _break={"exp": lambda e: f"tl.where(({e}) > 1.5, float('nan'), "
                                                       f"tl.exp({e}))"})
    with pytest.raises(K.KernelMismatchError, match="NaN"):
        fn(x=rnd(500, seed=3))


@pytest.mark.parametrize("device", DEVICES)
def test_kernel_wrong_only_at_a_larger_batch_is_caught(monkeypatch, device):
    """A binary that is right for the first 64 rows and wrong beyond them (a second row block that
    a small batch never reaches). Verified once at batch 7, once at 64: both pass. The first call
    at batch 100 must still be checked and must fail, and so must every later new batch size."""
    monkeypatch.setenv("TRUESTAR_VERIFY_DEVICE", device)
    monkeypatch.setenv("TRUESTAR_VERIFY", "first")
    M = K.sym("M")
    x, w = K.buffer("x", (M, 48)), K.buffer("w", (48, 32))
    i, j, r = K.axis("i", M), K.axis("j", 32), K.axis("r", 48)
    scale = {"cpu": 1.0, "gpu": 1.5, "auto": 1.25}[device]
    k = K.kernel("rowblock", [K.output("y", (i, j), K.reduce_sum(r, x[i, r] * w[r, j]) * scale)])
    fn = K.compile(k, "cuda")
    real = compiler.CompiledKernel._run_triton

    def wrong_past_64(self, got, L):
        outs = real(self, got, L)
        for t in outs.values():
            if t.shape[0] > 64:
                t[64:] *= 1.001  # tiny, but 50x the float32 tolerance
        return outs
    monkeypatch.setattr(compiler.CompiledKernel, "_run_triton", wrong_past_64)
    wt = rnd(48, 32, seed=1)
    fn(x=rnd(7, 48, seed=2), w=wt)
    fn(x=rnd(64, 48, seed=3), w=wt)
    with pytest.raises(K.KernelMismatchError, match="rowblock"):
        fn(x=rnd(100, 48, seed=4), w=wt)
    with pytest.raises(K.KernelMismatchError, match="rowblock"):
        fn(x=rnd(4096, 48, seed=5), w=wt)
    fn(x=rnd(33, 48, seed=6), w=wt)  # rows below 64 stay right, and still pass
