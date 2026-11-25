# AMD GPUs (ROCm, HIP)

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/kernel/rocm.py`,
`src/truestar/kernel/rocm_cases.py`; tests `tests/test_rocm.py`; scripts
`scripts/rocm_compile_matrix.py`, `scripts/rocm_validate.py`; manual workflow
`.github/workflows/rocm.yml`. Ledger row P43.

**Status: compiles, never run.** No AMD GPU has run any of this yet. Every loop shape compiles to
AMDGPU code objects for gfx942 (MI300X), gfx1100 (RX 7900 XTX) and gfx1103 (Radeon 780M) through
Triton's AMD backend, on a machine without an AMD GPU. Whether the results are right, and how fast
they are, is unknown until `scripts/rocm_validate.py` has run on the hardware (section 5).

## 1. Two ways in

The GPU backend emits Triton source, and Triton has an AMD backend that lowers the same source to
AMDGPU. Nothing in the kernel IR, the scheduler or the code generator is NVIDIA specific apart from
launch defaults, TF32 and resource limits, which this page covers.

**On an AMD GPU** (a ROCm build of torch, `torch.version.hip` set). torch exposes the GPU as
`"cuda"`, so tensors are `"cuda"` tensors.

```python
from truestar import kernel as K
fn = K.compile(kernel, "rocm")          # "cuda" means the same GPU on a ROCm build
from truestar.compiler import CompiledExecutor, compile_program
cp = compile_program(program, "rocm", "float16")          # or CompiledExecutor("rocm", dtype=...)
```

```
truestar check model.py --backends cpu,rocm,rocm+inductor
```

`truestar check` already listed the GPU as `rocm` on a ROCm build (`backends.py`); `rocm+inductor`
is `torch.compile` with Inductor on it. On a torch without HIP, `"rocm"` raises `RocmUnavailable`
(TS883) in `K.compile` and the compiled executor, `rocm+inductor` is refused with the reason, and
`--backends rocm` is TS001. Nothing falls back to another device.

**Ahead of time, without an AMD GPU.** `truestar.kernel.rocm.build` plans a launch with a named
target's CUs and LDS, compiles every Triton function of it (the split K second pass too) with
`triton.compile(ASTSource(...), target=GPUTarget("hip", arch, wave))`, and checks its LDS and
scratch from the code object's metadata. It needs only Triton with its AMD backend: the Windows
Triton 3.7.1 wheel in the ml venv and the Linux `triton==3.7.1` wheel both carry it, with no ROCm
installed.

```python
from truestar.kernel import rocm
b = rocm.build(kernel, "gfx942", {"x": (64, 1024), "w": (4096, 1024), "b": (4096,)})
b.plan_kind, b.config, b.functions[0].lds, b.functions[0].vgprs, b.notes
```

```
PYTHONPATH=src python scripts/rocm_compile_matrix.py --targets gfx942,gfx1100,gfx1103
PYTHONPATH=src python scripts/rocm_compile_matrix.py --candidates      # every tuning candidate
```

Every result carries TS885: compiled and not run, so nothing about its values is verified.

## 2. Targets

| arch | product the table describes | wave | CUs | LDS per workgroup | matrix cores | TF32 dot | scratch per lane |
|---|---|---|---|---|---|---|---|
| gfx942 | Instinct MI300X (CDNA3) | 64 | 304 | 64 KiB | MFMA | yes (xf32) | 131056 B |
| gfx1100 | Radeon RX 7900 XTX (RDNA3) | 32 | 96 | 64 KiB | WMMA | no | 262136 B |
| gfx1103 | Radeon 780M iGPU (RDNA3) | 32 | 12 | 64 KiB | WMMA | no | 262136 B |

Sources: CU counts, wave sizes and memory from the ROCm GPU hardware specifications table
(rocm.docs.amd.com, reference/gpu-arch-specs). RDNA3 has 128 KiB of LDS per WGP, but one workgroup
addresses 64 KiB, as on gfx942 (gfx950 raised it to 160 KiB, llvm PR #116309). Largest scratch per
wave from LLVM's `getMaxWaveScratchSize` (llvm PR #77625): a 13 bit field in units of 256 dwords
before GFX11, a 15 bit field in units of 64 dwords on GFX11, divided by the wave size. TF32: Triton
3.7.1's AMD backend allows `input_precision="tf32"` on gfx942 only. Assumptions, not measured: 2048
resident lanes per CU for the scratch reservation check when no GPU is attached, and 8 GiB of memory
for the 780M, whose memory is shared with the CPU. On hardware every value is read from the device
instead (`rocm.current_target()`: `gcnArchName`, `warp_size`, `multi_processor_count`,
`shared_memory_per_block`, `total_memory`, `max_threads_per_multi_processor`), and an arch the
table does not know is described from the device (TS880 is only for ahead of time builds).

## 3. What is handled, and how

* **CUs are the SMs of the SM aware defaults.** `triton_backend.device_limits()` reads the CU
  count on a ROCm build (torch's `multi_processor_count`) and `triton_backend.target_limits(cus,
  lds, memory)` sets them for an ahead of time build. Thresholds written for 70 SMs scale by
  `cus / 70`; below `FEW_SMS` (48) the few SM rules apply, so the 12 CU 780M follows the rules
  measured on a 26 SM laptop GPU. None of this was measured on an AMD GPU.
* **LDS.** The shared memory budget of the pipelined operand loads is the target's LDS per
  workgroup minus 3 KiB, so `_fit_shared` trims stages, then BK, to fit 64 KiB. A compiled config
  whose LDS is above the target's limit is refused with `LdsError` (TS882) before any launch.
* **Wave size 64 and 32.** Triton compiles for the target's wave size (`GPUTarget(..., wave)`), and
  `num_warps` counts waves, so a 4 wave program has 256 lanes on gfx942 and 128 on RDNA3. Ladder
  steps never exceed 1024 lanes per program. Measured at compile time: the defaults derived from
  the NVIDIA measurements spill registers on wave32 RDNA3 in 11 of the 13 dot cases on each of
  gfx1100 and gfx1103 (every linear and attention case and the float32 conv; 188 to 3076 bytes of
  scratch per lane), in none on wave64 gfx942, and in none of the tile and window kernels.
* **Relief ladder (TS886).** When the default config spills on an AMD GPU, the launch walks
  `rocm.relief_ladder`: twice the waves (up to 1024 lanes), at most 2 stages, half the BK, 1 stage,
  half the BK again, half the BN, half the BM, each step keeping the previous ones, and uses the
  first config that compiles without spilling (else the one that spills least). `SPLIT_K` never
  changes, so a deterministic kernel keeps its summation order. The ahead of time build walks the
  same ladder; on hardware it runs in `CompiledKernel._prepare` from Triton's HIP
  `LOCAL_SIZE_BYTES`. With it all 57 default launches of the case matrix compile without spills.
  Only defaults are relieved; a tuned, cached or forced config is used as it is.
* **MFMA and WMMA.** Dot kernels (contraction, attention, conv) on MFMA targets compile with
  `matrix_instr_nonkdim=16` (ROCm's guide to optimizing Triton kernels: mfma 16 x 16 usually beats
  32 x 32 on MI300X GEMMs); RDNA3 WMMA has only 16 x 16, which Triton picks itself. Truestar's dot
  tiles are already at least 16 on every side (`check_config`), which both instruction families
  need. float32 dots on RDNA3 have no matrix instruction and run on the vector ALUs, like the strict
  float32 class on NVIDIA.
* **TF32 (TS884).** TF32 is a permission. On gfx942 an `allow_tf32` float32 dot compiles with
  `input_precision="tf32"` (xf32); on RDNA3 the permission is dropped and the dot stays strict IEEE
  float32, which is more exact, never less, and the plan's operand class is `ieee`.
* **Resources (TS700).** On a ROCm build `check_resources` reads the same attribute Triton's HIP
  driver reports (`HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`) and compares it with the target's scratch
  per lane and, for every resident lane of every CU, with a quarter of the GPU's memory. Ahead of
  time builds read `.private_segment_fixed_size`, `.vgpr_spill_count` and
  `.group_segment_fixed_size` from the AMDGPU code object.
* **Float64 verification on first call.** Unchanged: every kernel is checked against the float64
  reference on its first call per shape, dtype and config; on a ROCm build the reference runs on the
  GPU in float64 through torch (docs/KERNEL_IR.md, Verification), or on the CPU below the work
  threshold. The compiled executor's region and end to end checks are the same code.
* **Deterministic reductions.** No Truestar kernel uses atomics; split K adds its partials in a
  fixed order in a second kernel; `deterministic=True` never picks a split from M or the CU count (its K chunks follow K alone, docs/KERNEL_IR.md; tested on
  304 and 12 CUs at plan level). Library ops (`library=`) go through torch, so MIOpen's choices
  follow `torch.use_deterministic_algorithms`.

## 4. What compiles (2026-11-24, compile only)

`scripts/rocm_compile_matrix.py` on the case matrix of `rocm_cases.py`: 19 cases (pointwise and
softmax in float32 and float16, a 16384 wide looped row sum, linear layers in strict float32, TF32,
float16 and bfloat16, a small M split K linear with and without determinism, causal attention with
head dim 64 and 128 in float16, attention in float32, 3 x 3 convolutions in float16 and float32 at
stride 1 and 2, and a 3 x 3 stride 2 max pool) times 3 targets: 57 of 57 default launches compile,
none spills after relief, every LDS within 64 KiB. Wall time 49 s for all 57 on the ml venv
(Windows, Triton 3.7.1), most of it the attention kernels on RDNA3 (up to 9.4 s each). The same
matrix runs on Linux (the CPU server, Triton 3.7.1 wheel, no ROCm): see the status log.

## 5. Running it on an AMD GPU

`scripts/rocm_validate.py` runs everything that needs the hardware and writes
`<out>/summary.json`: the environment and `current_target()`, the compile matrix for the attached
arch, the kernel tests on the GPU (tests/test_rocm.py with its hardware group, the kernel, tuning,
SM aware, split K and conv tests), the bit level fingerprint (`scripts/laptop_fingerprint.py`,
float32, float16, bfloat16; `truestar fingerprint` too once the CLI has it) and a short
`scripts/bench_matmul.py` run in float32 and float16 against `F.linear`. Exit 0 all passed, 1 a
step failed, 2 no ROCm.

On AMD Developer Cloud (an MI300X GPU droplet; steps from AMD's getting started article and the
DigitalOcean ROCm image docs, not yet followed by us):

1. Create a GPU droplet with one MI300X and a ROCm software image, with your SSH key; log in as
   root at its public IP.
2. Check the GPU and torch: `rocm-smi`, then `python3 -c "import torch; print(torch.__version__,
   torch.version.hip, torch.cuda.get_device_name(0))"`. The ROCm image may run its stack inside a
   container (`docker exec -it rocm bash`); use the Python where torch prints a HIP version. If
   there is none, install the ROCm wheel of torch that pytorch.org lists for the image's ROCm
   version, and Triton comes with it (`pytorch-triton-rocm`).
3. Copy the repo (a git bundle: `git bundle create truestar.bundle wt/rocm` here, `scp` it, then
   `git clone truestar.bundle truestar`), and `pip install numpy pytest numba`.
4. `cd truestar && PYTHONPATH=src python scripts/rocm_validate.py --out rocm-results` (about an hour
   for the full set; `--quick` for tests/test_rocm.py and test_kernel.py only).
5. Copy `rocm-results/` back and diff `fingerprint.json` against an NVIDIA machine's file.
6. Destroy the droplet: a powered off droplet is still billed.

`.github/workflows/rocm.yml` runs the same script by hand (`workflow_dispatch`) on a self hosted
runner labelled `self-hosted, linux, rocm`; no such runner exists yet.

## 6. Untested until hardware

* Every result: values, the float64 checks on the GPU, determinism, speed. The hardware group of
  tests/test_rocm.py (every case against float64, batch independence of a deterministic linear,
  the compiled executor on `"rocm"`, `current_target()`) skips with the reason here.
* The relief ladder on hardware (`CompiledKernel._relieve`), which reads Triton's HIP driver. Its
  control flow ran once on an NVIDIA GPU with the AMD target faked and the default reported as
  spilling (tests/test_rocm.py): the relieved config was launched and verified against float64.
* The SM aware rules and the relief ladder's choices as speed: chosen by compile metadata, not
  timed. The tuner's candidates are timed on hardware as on NVIDIA.
* `rocm+inductor`, CUDA graph capture (hipGraph through torch), the sentinels, the memory planner
  and `truestar profile` on ROCm.
* Archs other than the three in the table (gfx90a, gfx950, gfx1101, gfx1102, gfx12): described from
  the device on hardware, never compiled here.
