"""DLPack exchange (truestar.interop.dlpack): zero copy, checked before PyTorch sees the memory."""

import subprocess
import sys

import numpy as np
import pytest
import torch

import truestar
from truestar import Tensor, ops
from truestar.frontend import InputError
from truestar.interop import DLPackError, from_dlpack, inspect_capsule, to_dlpack

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64, np.int64, np.int32, np.uint8, np.bool_])
def test_numpy_zero_copy(dtype):
    a = (np.arange(24).reshape(2, 3, 4) % 3).astype(dtype)
    t = from_dlpack(a)
    assert t.data_ptr() == a.ctypes.data
    assert t.dtype == torch.from_numpy(a).dtype and tuple(t.shape) == a.shape
    a.flat[5] = 1
    assert t.reshape(-1)[5].item() == 1  # same memory


def test_strided_view_keeps_its_strides_without_copy():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    v = a[:, ::2]
    info = inspect_capsule(v.__dlpack__(max_version=(1, 0)))
    assert not info.compact and info.strides == (6, 2)
    t = from_dlpack(v)
    assert t.stride() == (6, 2) and t.data_ptr() == v.ctypes.data
    assert torch.equal(t, torch.from_numpy(np.ascontiguousarray(v)))


def test_negative_strides_refused_instead_of_aborting():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    with pytest.raises(DLPackError) as ei:
        from_dlpack(a[:, ::-1])
    assert ei.value.code == "dlpack_negative_strides" and "ascontiguousarray" in ei.value.suggestion


def test_stock_torch_from_dlpack_aborts_on_negative_strides():
    """The upstream behaviour the check above protects against (torch 2.11 on this machine)."""
    code = ("import numpy as np, torch\n"
            "a = np.arange(12, dtype=np.float32).reshape(3, 4)\n"
            "try:\n    torch.from_dlpack(a[:, ::-1])\nexcept Exception as e:\n    print('raised', type(e).__name__)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        pytest.skip(f"this torch handles it: {r.stdout.strip()}")
    assert "raised" not in r.stdout  # the process died, no Python exception


def test_readonly_memory():
    a = np.ones((2, 2), dtype=np.float32)
    a.flags.writeable = False
    with pytest.raises(DLPackError) as ei:
        from_dlpack(a)
    assert ei.value.code == "dlpack_readonly"
    assert from_dlpack(a, allow_readonly=True).data_ptr() == a.ctypes.data


def test_expected_type_errors_name_both_types():
    a = np.zeros((3, 8), dtype=np.float64)
    with pytest.raises(DLPackError, match=r"x is float64\[3, 8\] but is declared f32\[B, 8\]"):
        from_dlpack(a, name="x", expect=Tensor["B", 8])
    with pytest.raises(DLPackError, match="dim 1 is 8, declared 4"):
        from_dlpack(a.astype(np.float32), name="x", expect=Tensor["B", 4])
    with pytest.raises(DLPackError, match="rank 2"):
        from_dlpack(a.astype(np.float32), name="x", expect=Tensor[8])
    assert from_dlpack(a.astype(np.float32), expect=Tensor["B", 8]).shape == (3, 8)


def test_producer_refusals_and_non_arrays():
    with pytest.raises(DLPackError) as ei:
        from_dlpack(np.ones(3, dtype=">f4"))
    assert ei.value.code == "dlpack_refused_by_producer" and "byte order" in str(ei.value)
    with pytest.raises(DLPackError) as ei:
        from_dlpack([1.0, 2.0])
    assert ei.value.code == "dlpack_not_supported"
    cap = np.ones(3, dtype=np.float32).__dlpack__()
    torch.from_dlpack(cap)
    with pytest.raises(DLPackError, match="already consumed"):
        inspect_capsule(cap)


def test_cross_device_needs_copy_true():
    a = np.ones(4, dtype=np.float32)
    with pytest.raises(DLPackError) as ei:
        from_dlpack(a, device="cuda")
    assert ei.value.code == "dlpack_device_mismatch"
    if torch.cuda.is_available():
        assert from_dlpack(a, device="cuda", copy=True).device.type == "cuda"


def test_native_function_takes_numpy_inputs():
    @truestar.function
    def layer(x: Tensor["B", 4], w: Tensor[4, 3]) -> Tensor["B", 3]:
        return ops.tanh(x @ w)

    x = np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)
    w = np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)
    y = layer(x, w)
    assert torch.allclose(y.float(), torch.tanh(torch.from_numpy(x) @ torch.from_numpy(w)), atol=1e-6)
    with pytest.raises(InputError) as ei:
        layer(x.astype(np.float64), w)
    assert ei.value.code == "dlpack_dtype_mismatch" and "x is float64[5, 4]" in str(ei.value)
    # the output goes back to NumPy without a copy
    out = np.from_dlpack(y)
    assert out.ctypes.data == y.data_ptr()


def test_to_dlpack_refusals_and_round_trip():
    t = torch.arange(6.0)
    back = torch.from_dlpack(to_dlpack(t))
    assert back.data_ptr() == t.data_ptr()
    with pytest.raises(DLPackError) as ei:
        to_dlpack(torch.ones(2, requires_grad=True))
    assert ei.value.code == "dlpack_requires_grad"
    with pytest.raises(DLPackError) as ei:
        to_dlpack(torch.ones(2, dtype=torch.complex64).conj())
    assert ei.value.code == "dlpack_lazy_view"


class Producer:
    """A foreign CUDA producer: records the stream the consumer asks for."""

    def __init__(self, t):
        self.t = t
        self.streams = []

    def __dlpack_device__(self):
        return self.t.__dlpack_device__()

    def __dlpack__(self, stream=None, max_version=None):
        self.streams.append(stream)
        return self.t.__dlpack__(stream=stream, max_version=max_version)


@cuda
def test_cuda_zero_copy_and_stream_handles():
    x = torch.arange(1024.0, device="cuda")
    p = Producer(x)
    t = from_dlpack(p)
    assert t.data_ptr() == x.data_ptr() and t.device == x.device
    assert p.streams[-1] == 1  # the legacy default stream, never the ambiguous 0
    side = torch.cuda.Stream()
    from_dlpack(p, stream=side)
    assert p.streams[-1] == side.cuda_stream
    with torch.cuda.stream(side):
        from_dlpack(p)
    assert p.streams[-1] == side.cuda_stream  # the current stream by default
    from_dlpack(p, stream=-1)
    assert p.streams[-1] == -1
    with pytest.raises(DLPackError, match="ambiguous"):
        from_dlpack(p, stream=0)
    with pytest.raises(DLPackError) as ei:
        from_dlpack(p, device="cpu")
    assert ei.value.code == "dlpack_device_mismatch"
    with pytest.raises(DLPackError, match="no streams"):
        from_dlpack(np.ones(2, dtype=np.float32), stream=side)


def _stream_trial(make_producer, hog_a, hog_stream, a, n_hog=8):
    """One producer and consumer exchange while a hog stream keeps the GPU busy. Returns whether
    the consumer saw the finished value."""
    producer_stream, consumer_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(hog_stream):
        h = hog_a
        for _ in range(n_hog):
            h = torch.tanh(h @ hog_a * 1e-4)
    with torch.cuda.stream(producer_stream):
        y = a
        for _ in range(20):                      # long pending work on the producer stream
            y = torch.tanh(y @ a * 1e-3)
        t = from_dlpack(make_producer(y), stream=consumer_stream)   # the consumer stream must wait
    with torch.cuda.stream(consumer_stream):
        got = t.sum()
    torch.cuda.synchronize()
    return t.data_ptr() == y.data_ptr() and torch.allclose(got, y.sum(), rtol=1e-5, atol=1e-30)


@cuda
@pytest.mark.parametrize("producer", ["torch tensor", "foreign producer"])
def test_cuda_consumer_stream_waits_for_the_producer_under_load(producer):
    """Under GPU load the first version failed this for a torch tensor (17 of 40 trials read the value
    before the producer finished): its pass through path skipped the wait."""
    make = (lambda y: y) if producer == "torch tensor" else Producer
    hog_stream = torch.cuda.Stream()
    hog_a = torch.randn(4096, 4096, device="cuda")
    a = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    bad = sum(not _stream_trial(make, hog_a, hog_stream, a) for _ in range(20))
    assert bad == 0, f"{bad} of 20 exchanges read the tensor before the producer stream finished"


@cuda
def test_torch_tensor_stream_handling():
    x = torch.ones(8, device="cuda")
    side = torch.cuda.Stream()
    assert from_dlpack(x) is x                        # same stream: nothing to order
    assert from_dlpack(x, stream=side) is x           # ordered and recorded, still the same tensor
    assert from_dlpack(x, stream=-1) is x
    with pytest.raises(DLPackError, match="ambiguous"):
        from_dlpack(x, stream=0)
    with pytest.raises(DLPackError, match="no streams"):
        from_dlpack(torch.ones(2), stream=side)
