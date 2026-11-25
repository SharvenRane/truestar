"""DLPack exchange, checked before anything is shared.

``from_dlpack(obj)`` takes any object with ``__dlpack__`` (NumPy, CuPy, JAX, PyTorch, ...) and returns
a torch tensor on the same memory, with no copy. Before the memory is handed to PyTorch the capsule
itself is read (``inspect_capsule``) and checked:

* dtype: a type PyTorch can hold (no vector lanes); with ``expect=`` it must equal the declared dtype;
* layout: no negative strides (PyTorch 2.11 and 2.12 abort the whole process on one, a C++ exception that
  escapes ``torch.from_dlpack``; 2.14.1 and later raise an unclear overflow error), strides reported when the array is not compact;
* shape: with ``expect=`` rank and every fixed dim must match, both types named in the error;
* device: the producer's device must be the one asked for (``device=``); a copy across devices is
  made only with ``copy=True``, never silently;
* read only memory (DLPack 1.0 flag): refused unless ``allow_readonly=True``, because the tensor
  PyTorch returns is writable and a write would change memory its owner declared read only;
* zero copy: after import the tensor's data pointer must equal the capsule's.

Streams (CUDA): the consumer stream is passed to the producer as the standard says, so the producer
orders its pending work before ours. By default it is the current torch stream on the producer's
device (the legacy default stream is sent as 1, since 0 is not allowed); ``stream=-1`` tells the
producer not to synchronize; an int or a ``torch.cuda.Stream`` names another stream.

``to_dlpack(tensor)`` is the export side: the same checks on a torch tensor, then its capsule.
Truestar functions return torch tensors, which other libraries take with their own ``from_dlpack``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

# --------------------------------------------------------------------------- the C structs (dlpack.h)


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


class _DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [("version", _DLPackVersion), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p),
                ("flags", ctypes.c_uint64), ("dl_tensor", _DLTensor)]


_FLAG_READ_ONLY = 1 << 0
_FLAG_IS_COPIED = 1 << 1

_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_GetPointer.restype = ctypes.c_void_p
_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_GetName = ctypes.pythonapi.PyCapsule_GetName
_GetName.restype = ctypes.c_char_p
_GetName.argtypes = [ctypes.py_object]
_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_IsValid.restype = ctypes.c_int
_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

DEVICE_TYPES = {1: "cpu", 2: "cuda", 3: "cuda_host", 4: "opencl", 7: "vulkan", 8: "metal", 9: "vpi",
                10: "rocm", 11: "rocm_host", 12: "ext_dev", 13: "cuda_managed", 14: "oneapi", 15: "webgpu",
                16: "hexagon", 17: "maia"}

# (DLDataTypeCode, bits) -> torch dtype. Codes: 0 int, 1 uint, 2 float, 4 bfloat, 5 complex, 6 bool.
_DTYPES = {(0, 8): torch.int8, (0, 16): torch.int16, (0, 32): torch.int32, (0, 64): torch.int64,
           (1, 8): torch.uint8, (1, 16): torch.uint16, (1, 32): torch.uint32, (1, 64): torch.uint64,
           (2, 16): torch.float16, (2, 32): torch.float32, (2, 64): torch.float64, (4, 16): torch.bfloat16,
           (5, 64): torch.complex64, (5, 128): torch.complex128, (6, 8): torch.bool}
_CODE_NAMES = {0: "int", 1: "uint", 2: "float", 3: "opaque handle", 4: "bfloat", 5: "complex", 6: "bool",
               7: "float8 e3m4", 8: "float8 e4m3", 9: "float8 e4m3b11fnuz", 10: "float8 e4m3fn",
               11: "float8 e4m3fnuz", 12: "float8 e5m2", 13: "float8 e5m2fnuz", 14: "float8 e8m0fnu"}


class DLPackError(ValueError):
    """An exchange that would be wrong, unsafe or a hidden copy. ``code`` is stable."""

    def __init__(self, code: str, message: str, suggestion: str = ""):
        self.code = code
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{message}" + (f"\n  fix: {suggestion}" if suggestion else ""))

    def to_dict(self):
        return {"code": self.code, "message": self.message, "suggestion": self.suggestion}


@dataclass(frozen=True)
class DLPackInfo:
    """What a capsule says, read without consuming it."""

    versioned: bool
    version: Optional[Tuple[int, int]]
    device: str
    device_id: int
    dtype: Optional[torch.dtype]
    dtype_desc: str
    shape: Tuple[int, ...]
    strides: Optional[Tuple[int, ...]]    # in elements; None means compact row major
    byte_offset: int
    data: int
    readonly: bool
    copied: bool

    @property
    def compact(self) -> bool:
        return self.strides is None or self.strides == _compact_strides(self.shape)

    @property
    def negative_strides(self) -> bool:
        return self.strides is not None and any(s < 0 for s, n in zip(self.strides, self.shape) if n > 1)

    def __str__(self) -> str:
        dev = self.device if self.device == "cpu" else f"{self.device}:{self.device_id}"
        st = "compact" if self.compact else f"strides {list(self.strides)}"
        ro = ", read only" if self.readonly else ""
        return f"{self.dtype_desc}{list(self.shape)} on {dev}, {st}{ro}"


def _compact_strides(shape) -> Tuple[int, ...]:
    out, acc = [], 1
    for n in reversed(shape):
        out.append(acc)
        acc *= max(int(n), 1)
    return tuple(reversed(out))


def inspect_capsule(capsule) -> DLPackInfo:
    """Read a DLPack capsule (legacy ``dltensor`` or 1.0 ``dltensor_versioned``) without consuming it."""
    name = _GetName(capsule) if type(capsule).__name__ == "PyCapsule" else None
    if name not in (b"dltensor", b"dltensor_versioned"):
        used = name in (b"used_dltensor", b"used_dltensor_versioned")
        raise DLPackError("dlpack_bad_capsule", "the capsule was already consumed" if used else
                          f"not a DLPack capsule ({type(capsule).__name__}"
                          + (f" named {name.decode(errors='replace')!r})" if name else ")"),
                          "ask the producer for a new one with obj.__dlpack__()")
    ptr = _GetPointer(capsule, name)
    versioned = name == b"dltensor_versioned"
    if versioned:
        m = _DLManagedTensorVersioned.from_address(ptr)
        t, flags, version = m.dl_tensor, int(m.flags), (int(m.version.major), int(m.version.minor))
    else:
        m = _DLManagedTensor.from_address(ptr)
        t, flags, version = m.dl_tensor, 0, None
    nd = int(t.ndim)
    shape = tuple(int(t.shape[i]) for i in range(nd))
    strides = tuple(int(t.strides[i]) for i in range(nd)) if nd and bool(t.strides) else None
    code, bits, lanes = int(t.dtype.code), int(t.dtype.bits), int(t.dtype.lanes)
    dtype = _DTYPES.get((code, bits)) if lanes == 1 else None
    desc = (str(dtype).replace("torch.", "") if dtype is not None else
            f"{_CODE_NAMES.get(code, f'code {code}')}{bits}" + (f"x{lanes}" if lanes != 1 else ""))
    return DLPackInfo(versioned, version, DEVICE_TYPES.get(int(t.device.device_type), f"device type {t.device.device_type}"),
                      int(t.device.device_id), dtype, desc, shape, strides, int(t.byte_offset), int(t.data or 0),
                      bool(flags & _FLAG_READ_ONLY), bool(flags & _FLAG_IS_COPIED))


# --------------------------------------------------------------------------- import


def _consumer_stream(stream, dev_type: str, dev_id: int):
    if dev_type not in ("cuda", "rocm"):
        if stream not in (None,):
            raise DLPackError("dlpack_bad_stream", f"a stream was given but the producer is on {dev_type}, "
                                                   "which has no streams", "leave stream=None")
        return None
    if isinstance(stream, int):
        if stream == 0 and dev_type == "cuda":
            raise DLPackError("dlpack_bad_stream", "stream 0 is ambiguous for CUDA and not allowed by DLPack",
                              "pass 1 (legacy default), 2 (per thread default) or a stream handle")
        return stream
    if stream is None:
        if not torch.cuda.is_available():
            raise DLPackError("dlpack_device_unavailable", f"the array is on {dev_type}:{dev_id} but this PyTorch "
                                                           "has no CUDA device", "move it to the CPU first")
        s = torch.cuda.current_stream(dev_id)
    elif isinstance(stream, torch.cuda.Stream):
        s = stream
    else:
        raise DLPackError("dlpack_bad_stream", f"stream must be None, an int or a torch.cuda.Stream, got "
                                               f"{type(stream).__name__}")
    h = int(s.cuda_stream)
    if dev_type == "cuda" and h == 0:
        return 1  # the legacy default stream; 0 is ambiguous in DLPack
    return h


def _stream_object(handle: int, dev: int) -> "torch.cuda.Stream":
    """The torch stream for a DLPack CUDA stream handle (1: legacy default, 2: per thread default)."""
    if handle == 1:
        return torch.cuda.default_stream(dev)
    cur = torch.cuda.current_stream(dev)
    if int(cur.cuda_stream) == handle:
        return cur
    return torch.cuda.ExternalStream(handle, device=dev)


def _record(t: torch.Tensor, consumer) -> None:
    """Tell the caching allocator the consumer stream uses t's memory, so a free on the producer
    stream does not hand the block to new work while the consumer's kernels are still pending."""
    try:
        t.record_stream(consumer)
    except RuntimeError:
        pass  # memory not owned by this process's caching allocator (a foreign producer keeps it alive)


def _sync_consumer(t: torch.Tensor, handle: Optional[int], dev: int) -> None:
    """Make the consumer stream wait for the producer's current stream, then record it on t.
    handle -1 means the consumer asked for no synchronization."""
    if handle is None or handle == -1:
        return
    producer = torch.cuda.current_stream(dev)
    consumer = _stream_object(handle, dev)
    if int(consumer.cuda_stream) == int(producer.cuda_stream) or (handle == 1 and producer.cuda_stream == 0):
        return  # same stream: already ordered
    consumer.wait_stream(producer)
    _record(t, consumer)


def _expect_type(expect):
    if expect is None:
        return None
    to_type = getattr(expect, "to_type", None)
    return to_type() if to_type is not None else expect


def _check_expected(info: DLPackInfo, expect, name: str) -> None:
    t = _expect_type(expect)
    if t is None:
        return
    if info.dtype != t.dtype:
        raise DLPackError("dlpack_dtype_mismatch", f"{name} is {info.dtype_desc}{list(info.shape)} but is declared {t}",
                          f"convert it on the producer side, for example x.astype(...), before passing it")
    if len(info.shape) != t.rank:
        raise DLPackError("dlpack_shape_mismatch", f"{name} has rank {len(info.shape)} {list(info.shape)} but is "
                                                   f"declared {t}")
    for i, (d, n) in enumerate(zip(t.shape, info.shape)):
        if isinstance(d, int) and d != n:
            raise DLPackError("dlpack_shape_mismatch", f"{name} has shape {list(info.shape)} but is declared {t} "
                                                       f"(dim {i} is {n}, declared {d})")


def import_with_info(obj: Any, *, name: str = "tensor", expect=None, device=None, stream=None,
                     allow_readonly: bool = False, copy: bool = False) -> Tuple[torch.Tensor, Optional[DLPackInfo]]:
    """``from_dlpack`` that also returns what the capsule said (None for a torch tensor passed in)."""
    want = None if device is None else torch.device(device)
    if isinstance(obj, torch.Tensor):
        info = None
        t = obj
        tt = _expect_type(expect)
        if tt is not None and t.dtype != tt.dtype:
            raise DLPackError("dlpack_dtype_mismatch", f"{name} is {str(t.dtype).replace('torch.', '')} but is "
                                                       f"declared {tt}")
        # No capsule is made for a torch tensor, so do what a producer's __dlpack__(stream=...) must
        # do: order the consumer stream after the producer's current stream. Skipping this (as the
        # first version did) let the consumer read the tensor before its pending kernels finished.
        if t.is_cuda:
            dev = t.device.index if t.device.index is not None else torch.cuda.current_device()
            _sync_consumer(t, _consumer_stream(stream, "cuda", dev), dev)
        elif stream is not None:
            _consumer_stream(stream, "cpu", 0)  # raises: a CPU tensor has no stream
    else:
        if not hasattr(obj, "__dlpack__"):
            if type(obj).__name__ == "PyCapsule":
                cap = obj
                info = inspect_capsule(cap)
                dev_type = info.device
                s = None  # a bare capsule was made earlier; its producer already chose a stream
            else:
                raise DLPackError("dlpack_not_supported", f"{name} is a {type(obj).__name__}, which has no "
                                                          "__dlpack__ method",
                                  "pass a torch tensor or an array from a library that supports DLPack "
                                  "(NumPy 1.22+, CuPy, JAX)")
        else:
            dev_code, dev_id = obj.__dlpack_device__()
            dev_type = DEVICE_TYPES.get(int(dev_code), f"device type {dev_code}")
            s = _consumer_stream(stream, dev_type, int(dev_id))
            try:
                try:
                    cap = obj.__dlpack__(stream=s, max_version=(1, 0))
                except TypeError:  # a producer from before DLPack 1.0 (no max_version keyword)
                    cap = obj.__dlpack__(stream=s)
            except BufferError as e:
                raise DLPackError("dlpack_refused_by_producer", f"the producer of {name} ({type(obj).__name__}) "
                                                                f"refused to export it: {e}",
                                  "make a native byte order, detached copy on the producer side") from None
            info = inspect_capsule(cap)
        if info.dtype is None:
            raise DLPackError("dlpack_unsupported_dtype", f"{name} has dtype {info.dtype_desc}, which PyTorch "
                                                          "cannot hold", "cast it on the producer side")
        if info.negative_strides:
            raise DLPackError("dlpack_negative_strides", f"{name} has negative strides {list(info.strides)} "
                                                         f"(a reversed view, {info}); PyTorch cannot represent it "
                                                         "(torch.from_dlpack aborts the process on torch 2.12 and older)",
                              "pass a compact copy, for example numpy.ascontiguousarray(x)")
        if info.readonly and not allow_readonly:
            raise DLPackError("dlpack_readonly", f"{name} is read only memory ({info}); the PyTorch tensor would be "
                                                 "writable", "pass allow_readonly=True if nothing will write to it, "
                                                             "or pass a writable copy")
        _check_expected(info, expect, name)
        if want is not None and not copy and not _same_device(want, info.device, info.device_id):
            raise DLPackError("dlpack_device_mismatch", f"{name} is on {info.device}:{info.device_id} but "
                                                        f"{want} was asked for; that needs a copy",
                              "pass copy=True to allow the copy, or move it on the producer side")
        t = torch.from_dlpack(cap)
        if t.numel() and t.data_ptr() != info.data + info.byte_offset:
            raise DLPackError("dlpack_copied", f"{name} was copied on import (pointer {t.data_ptr():#x} vs "
                                               f"{info.data + info.byte_offset:#x})")
        if t.is_cuda and s not in (None, -1):
            _record(t, _stream_object(s, t.device.index or 0))
    if want is not None and t.device != want and not (want.index is None and t.device.type == want.type):
        if not copy:
            raise DLPackError("dlpack_device_mismatch", f"{name} is on {t.device} but {want} was asked for; "
                                                        "that needs a copy", "pass copy=True to allow the copy")
        t = t.to(want)
    return t, info


def _same_device(want: torch.device, dev_type: str, dev_id: int) -> bool:
    if want.type == "cpu":
        return dev_type == "cpu"
    if want.type == "cuda":
        return dev_type == "cuda" and (want.index is None or want.index == dev_id)
    return False


def from_dlpack(obj: Any, *, name: str = "tensor", expect=None, device=None, stream=None,
                allow_readonly: bool = False, copy: bool = False) -> torch.Tensor:
    """A torch tensor sharing ``obj``'s memory, after the checks in the module docstring.

    ``expect`` is a ``Tensor[...]`` annotation or an IR ``TensorType``: dtype, rank and fixed dims must
    match (symbolic dims are bound later by the function call). ``device`` asks for a device and
    refuses a cross device copy unless ``copy=True``. ``stream`` is the consumer stream (CUDA).
    Raises DLPackError with a stable ``code``."""
    return import_with_info(obj, name=name, expect=expect, device=device, stream=stream,
                            allow_readonly=allow_readonly, copy=copy)[0]


# --------------------------------------------------------------------------- export


def to_dlpack(t: torch.Tensor, *, stream=None, versioned: bool = True):
    """The DLPack capsule of a torch tensor, for a consumer that takes capsules.

    ``stream`` is the consumer's stream (CUDA); the producer (PyTorch) makes it wait for pending work.
    Refuses what DLPack cannot carry, with the fix: a tensor that requires grad (detach it, the
    gradient would be lost silently), a conjugate or negative view (resolve it), sparse layouts."""
    if not isinstance(t, torch.Tensor):
        raise DLPackError("dlpack_not_supported", f"to_dlpack takes a torch tensor, got {type(t).__name__}")
    if t.requires_grad:
        raise DLPackError("dlpack_requires_grad", "the tensor requires grad; DLPack carries no autograd history",
                          "export t.detach() if the gradient is not needed")
    if t.layout != torch.strided:
        raise DLPackError("dlpack_layout", f"layout {t.layout} cannot be exported", "export t.to_dense()")
    if t.is_conj() or t.is_neg():
        raise DLPackError("dlpack_lazy_view", "the tensor is a lazy conjugate or negative view",
                          "export t.resolve_conj().resolve_neg()")
    if t.device.type == "cuda" and stream is None:
        stream = _consumer_stream(None, "cuda", t.device.index or 0)
    elif isinstance(stream, torch.cuda.Stream):
        stream = _consumer_stream(stream, "cuda", t.device.index or 0)
    if versioned:
        try:
            return t.__dlpack__(stream=stream, max_version=(1, 0))
        except TypeError:
            pass
    return t.__dlpack__(stream=stream)
