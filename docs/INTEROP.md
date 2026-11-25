# Interop: DLPack and safetensors

Author: Sharven Rane. First written 2026-11-24.

The design asks for zero copy exchange through DLPack and for loading safetensors weights
directly. Both are here, and both check before they share or load anything. Code in
`src/truestar/interop/`, tests in `tests/test_dlpack.py` (18) and `tests/test_safetensors.py` (15),
measurements by `scripts/dlpack_bench.py` and `scripts/safetensors_bench.py`. No new dependency: the
safetensors format is read and written by Truestar itself; the package is used only in tests and
benchmarks, for comparison.

## DLPack

```python
from truestar.interop import from_dlpack, to_dlpack
t = from_dlpack(numpy_or_cupy_or_jax_array)            # a torch tensor on the same memory
t = from_dlpack(x, name="x", expect=Tensor["B", 128])   # dtype, rank and fixed dims checked first
y = layer(numpy_x, numpy_w)                             # native functions take DLPack arrays directly
np.from_dlpack(y)                                       # outputs are torch tensors: zero copy back
```

Before PyTorch sees the memory, Truestar reads the capsule itself (`inspect_capsule`, a ctypes view
of the DLPack 0.x and 1.0 structs, without consuming it) and refuses with a stable `code` and a fix:

| case | `torch.from_dlpack` (torch 2.11, NumPy 2.2.6) | `truestar.interop.from_dlpack` |
|---|---|---|
| reversed view (negative stride) | **the process dies** on torch 2.11 to 2.12 (exit code 0xC0000409: a C++ exception escapes; pytorch#188023); a RuntimeError about storage size overflow from 2.14.1 | `dlpack_negative_strides`, fix `numpy.ascontiguousarray(x)` |
| read only array | accepted; the returned tensor is writable, so a write changes memory its owner declared read only | `dlpack_readonly`; `allow_readonly=True` to accept |
| big endian array | `BufferError: DLPack only supports native byte order.` | the same reason, wrapped, with a fix |
| CUDA array asked for on the CPU | `RuntimeError: Unsupported device in DLTensor.` (from NumPy as consumer) | `dlpack_device_mismatch`: names both devices; `copy=True` makes the copy explicit |
| a list | `RuntimeError: from_dlpack received an invalid capsule ...` | `dlpack_not_supported`: names the type |
| wrong dtype or shape for a typed input | accepted, fails later inside the function | `dlpack_dtype_mismatch` or `dlpack_shape_mismatch` naming both types, before the call |

After import the tensor's data pointer is compared with the capsule's; a copy would be an error
(`dlpack_copied`). Measured: zero copy for NumPy on the CPU and for a CUDA producer (pointer equality,
tests at every supported dtype).

**Streams.** For a CUDA producer the consumer stream is passed to `__dlpack__` as the array API
standard says: by default the current torch stream on the producer's device, sent as 1 for the
legacy default stream (0 is ambiguous and refused), `stream=-1` for no synchronization, or any
`torch.cuda.Stream` or handle. A CPU producer with a stream is an error. A torch tensor is passed
through without a capsule, so Truestar does the producer's part itself: the consumer stream waits on
the stream current at the call (the producer's), and `record_stream` tells the caching allocator the
consumer uses the memory, so a free on the producer stream cannot hand the block to new work while
the consumer's kernels are pending. Imported CUDA tensors get the same `record_stream`. The first
version skipped the wait on the pass through path (fixed 2026-11-24, see the status log): under GPU
load 15 to 17 of 20 to 40 exchanges read the tensor before the producer finished. The permanent test
runs 20 exchanges per producer kind while a hog stream keeps the GPU busy and requires 0 failures; a
second test records the stream each call hands a foreign producer.

**Cost** (`scripts/dlpack_bench.py`, median of 2000 imports of a 1024 x 1024 float32 array):

| producer | torch.from_dlpack | Truestar |
|---|---|---|
| NumPy | 1.1 us | 4.6 us |
| a foreign CUDA producer | 18.0 us | 23.6 us |
| a torch CUDA tensor | 18.2 us | 0.3 us (a torch tensor is passed through, only checked) |

The extra 3.5 to 5.5 us is the capsule read in Python.

`to_dlpack(t, stream=...)` is the export side for consumers that take capsules. It refuses a tensor
that requires grad (DLPack carries no autograd history; `t.detach()` says it on purpose), a lazy
conjugate or negative view and sparse layouts.

## safetensors

```python
from truestar.interop import load_file, save_file, load_weights, save_weights
w = load_weights("mlp.safetensors", into=mlp)        # a native function: fn(x, **w)
w = load_weights("model.safetensors", into=module)   # an nn.Module: checked, then loaded
w = load_weights("model.safetensors", into=program)  # an IR program: w.program has the new params
save_weights("mlp.safetensors", weights, check=mlp)  # also records reference outputs
```

**Reading** validates the whole header before any tensor exists and lists every problem at once:
duplicate names (Python's `json` keeps the last silently; the format forbids them), unknown dtypes,
byte lengths that disagree with the shape, overlaps, holes, trailing or missing bytes, a header over
100 MB. Tensors are views of a copy on write memory map: nothing is read until touched and no write
reaches the file. A tensor whose offset is not aligned to its dtype is copied and named in
`LoadReport.copied`.

**Writing** orders tensors by dtype size then name and pads the header to 8 bytes, the same layout
as the package: the bytes are identical to `safetensors.torch.save_file` for the same tensors and
metadata (tested). Writes are atomic (temporary file, then rename).

**Loading into a model** checks names, dtypes and shapes against what the weights go into and
lists every missing, unexpected, mistyped and misshapen tensor in one error; nothing is assigned
when anything is wrong. For a native function, symbolic dims must bind to one value across all
weights (`w2: dim 1 is H = 7, but b1 has H = 6`), and the parameters not in the file are reported
as its data inputs. For a module there is no silent dtype cast (`cast=True` allows it), and a tied
weight (an lm head sharing the embedding) is filled from the one name the file has.

**Reference outputs.** Shape checks cannot see two tensors of one shape swapped or a square weight
stored transposed, and `load_state_dict(strict=True)` accepts both. `save_weights(..., check=model)`
stores in `__metadata__` (key `truestar.check`) the model's outputs on inputs generated from a
formula (a Weyl sequence in float64, the same bits on every machine, no random generator) and
`load_weights` reruns them; a mismatch is refused (`safetensors_check_failed`) with the likely
causes. Up to 4096 values per output are stored (evenly strided). Tolerance by the output's
declared dtype: 1e-9 float64, 1e-4 float32, 1e-2 float16, 6e-2 bfloat16. Other readers ignore the
metadata, so the file stays a plain safetensors file.

### Measured (2026-11-24, Windows 11, NVMe, torch 2.11, safetensors 0.8.0, warm file cache)

`scripts/safetensors_bench.py` on the cached Hugging Face Qwen2.5 0.5B checkpoint (988,097,824
bytes, 290 bfloat16 tensors), median of 5. "Read every byte" sums every tensor's bytes, so lazy and
eager loaders do the same work.

| loader | seconds |
|---|---|
| Truestar `read_header` (full validation, 290 entries) | 0.001 |
| Truestar `load_file`, memory map, lazy | 0.0025 |
| Truestar `load_file` + read every byte | 0.82 |
| `safetensors.torch.load_file` + read every byte | 0.79 |
| Truestar `load_file(mmap=False)` + read every byte | 0.97 |
| `torch.load(weights_only=True)` of the same tensors + read every byte | 0.80 |
| `torch.load(mmap=True)` + read every byte | 0.98 |
| Truestar `load_file(device="cuda")` | 0.40 |
| `safetensors` `load_file(device="cuda")` | 0.43 |

Against the package: 4% slower when every byte is read on the CPU (target: within 10%), 7% faster to
CUDA. Every tensor is bitwise equal to the package's. Cold cache numbers were not measured (the OS
file cache was not dropped).

GPT2 (548 MB hub file) into `transformers` `GPT2LMHeadModel` (4.48): the hub file names tensors
without the `transformer.` prefix and holds 12 `h.N.attn.bias` buffers the module has no slot for, so
the benchmark renames and drops them, saves with a reference record (10 tokens, logits `[1, 10,
50257]`), then `load_weights` reruns it: 0.20 s including the forward, record error 1.1e-6
(float32, tolerance 1e-4), `lm_head.weight` filled from the tied `transformer.wte.weight`, logits
identical to `from_pretrained` (max difference 0). With `h.0.attn.c_proj.weight` (768 x 768) stored
transposed, `load_state_dict` accepts the file; `load_weights` refuses it, relative error 0.51.

Tests that prove the tests work (manual, restored): with duplicate detection removed and the record
rerun returning early, 3 tests fail (the every problem test, the swapped tensors test, the
transposed module weight test).

## Limits (2026-11-24)

* DLPack: only CPU and CUDA were run; ROCm stream numbering follows the standard but is untested.
  CuPy and JAX are not installed, so their producers were imitated by a wrapper object around a
  torch tensor. Vector lanes and the float8 DLPack codes are refused as unsupported dtypes.
* safetensors: big endian hosts are not handled (the format is little endian). On Windows a file
  cannot be deleted while a tensor from its memory map lives. The reference record for an IR program
  is not run (functions and modules only). `load_weights` into the native decoder of docs/LLM.md
  with Hugging Face names (the R15 measure in the research notes) was not done: its weights are
  function parameters with other names, and a name map is not built.
* Load time is a warm cache measurement on one machine.
