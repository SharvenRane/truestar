# Cross chip fingerprint

`truestar fingerprint` hashes, bit for bit, what the compiled executor, eager PyTorch and
`torch.compile` give for four models on this GPU. `truestar fingerprint compare A B` says which
settings and versions differ between two such files and which results are bitwise equal. This page
states what has been measured on three GPUs, under exactly which conditions, and what has not.

## What can be claimed (2026-11-24)

> On an NVIDIA RTX 5070 Ti (Blackwell, compute capability 12.0, 70 SMs, Windows, driver 610.47,
> CUDA 12.8) and a Tesla T4 (Turing, compute capability 7.5, 40 SMs, Linux, driver 580.178.04,
> CUDA 13.0), both with PyTorch 2.11.0 and Triton 3.7.1, Truestar's compiled float32 outputs with
> `deterministic=True` and TF32 off were bitwise identical across the two GPUs for an MLP, a
> TransformerEncoderLayer and a two layer Llama style decode step, at batch sizes 1, 7, 64 and 256
> (12 of 12 rows), and identical run to run on each GPU. At batch 256 the transformer's feed
> forward contraction was autotuned (contractions are config independent without split K);
> attention and reductions keep a shape only default config in deterministic mode. Eager PyTorch
> and `torch.compile` differed between the two GPUs in every one of those 12 rows.
>
> The same claim does not hold for: float16, bfloat16 and TF32 (the two architectures compute them
> with different instructions, and the T4 has no TF32); resnet18 (its convolutions are cuDNN calls
> inside the compiled program); a different Triton version (Triton 3.6.0 on the same T4 changed the
> transformer and decode step bits); AMD, Apple and Intel GPUs, Ampere, Ada and Hopper, which have
> not been run. Truestar does not claim bitwise equality with PyTorch or with float64; its compiled
> results are checked against float64 within a stated bound.

The tagline "the same answer on every chip" is a goal. This table is the claim.

## The three GPUs

| GPU | architecture | SMs | driver | CUDA | torch | Triton | cuDNN | OS | files |
|---|---|---|---|---|---|---|---|---|---|
| RTX 5070 Ti (desktop) | Blackwell, cc 12.0 | 70 | 610.47 | 12.8 | 2.11.0+cu128 | 3.7.1 (triton-windows) | 9.19 | Windows 11 | [fp2_rtx5070ti.json](cross_chip/fp2_rtx5070ti.json) |
| RTX 5060 Laptop | Blackwell, cc 12.0 | 26 | not recorded | 12.8 | 2.11.0+cu128 | 3.7.1 | 9.19 | Windows 11 | [fp_laptop_5060.json](cross_chip/fp_laptop_5060.json) (first script, legacy cases only) |
| Tesla T4 (Kaggle, private kernel) | Turing, cc 7.5 | 40 | 580.178.04 | 13.0 | 2.11.0+cu130 | 3.6.0 (image default) and 3.7.1 (installed for the run) | 9.27 | Linux | [fp2_t4_triton360.json](cross_chip/fp2_t4_triton360.json), [fp2_t4_triton371.json](cross_chip/fp2_t4_triton371.json) |

The desktop and T4 files come from commit a73d121 (the T4 ran the same source, embedded in the
kernel, commit passed in as `TRUESTAR_GIT_COMMIT`). Both started with an empty kernel and tuning
cache (`--fresh-cache`). The laptop was not available again: it is compared only through the three
legacy cases of the first script (below).

## Results: deterministic mode (`truestar_det`), RTX 5070 Ti against T4

Rows equal out of 4 (batch 1, 7, 64, 256). The other columns are recorded for comparison and carry
no claim.

| case | dtype | det, T4 with Triton 3.7.1 | det, T4 with Triton 3.6.0 | det, T4 3.6.0 against T4 3.7.1 | compiled default mode | eager | torch.compile |
|---|---|---|---|---|---|---|---|
| mlp | float32 | **4 of 4** | 4 of 4 | 4 of 4 | 4 of 4 | 0 of 4 | 0 of 4 |
| mlp | tf32 | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| mlp | float16 | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| mlp | bfloat16 | 1 of 4 | 1 of 4 | 4 of 4 | 1 of 4 | 0 of 4 | 0 of 4 |
| transformer | float32 | **4 of 4** | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 |
| transformer | tf32 | 0 of 4 | 0 of 4 | 0 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| transformer | float16 | 0 of 4 | 0 of 4 | 1 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| transformer | bfloat16 | 0 of 4 | 0 of 4 | 2 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| llama_step | float32 | **4 of 4** | 1 of 4 | 1 of 4 | 4 of 4 | 0 of 4 | 0 of 4 |
| llama_step | tf32 | 0 of 4 | 0 of 4 | 1 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| llama_step | float16 | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| llama_step | bfloat16 | 1 of 4 | 1 of 4 | 4 of 4 | 1 of 4 | 0 of 4 | 0 of 4 |
| resnet18 | float32 | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| resnet18 | tf32 | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| resnet18 | float16 | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |
| resnet18 | bfloat16 | 0 of 4 | 0 of 4 | 4 of 4 | 0 of 4 | 0 of 4 | 0 of 4 |

float32 deterministic output hashes (first 8 hex digits of the first output, batch 1, 7, 64, 256):

| case | RTX 5070 Ti | T4, Triton 3.7.1 | T4, Triton 3.6.0 |
|---|---|---|---|
| mlp | e0410306 540f247b e7490981 9b43470f | e0410306 540f247b e7490981 9b43470f | e0410306 540f247b e7490981 9b43470f |
| transformer | fe0354ed 65d1c54d 1b5dc59c b17c8f0e | fe0354ed 65d1c54d 1b5dc59c b17c8f0e | 84ecddc4 6cf16010 8d46ae7a 0fa8edb5 |
| llama_step | 704b824b 7507ce50 53b3677e 280aae12 | 704b824b 7507ce50 53b3677e 280aae12 | 704b824b 4373ecc7 cf53819d 4b29f03e |
| resnet18 | f9718816 1bc3fec4 aa7df107 a2502830 | 8dcda438 26b7d6f9 cb4a7a36 4c7f29a8 | 8dcda438 26b7d6f9 cb4a7a36 4c7f29a8 |

Largest error against float64 (max abs error over max abs reference, worst batch, deterministic
mode, identical on both GPUs where the bits are identical): float32 1.4e-6 (mlp), 3.1e-7
(transformer), 1.0e-6 (llama_step), 1.0e-6 and 1.4e-6 (resnet18 on the 5070 Ti and the T4).

Run to run: every row of every arm gave the same bits on two calls, on both GPUs; a second process
on the T4 (`fp_t4_again`, float32 deterministic, mlp, transformer, llama_step) gave the same 12
hashes as the first.

## Why the other rows differ

Measured, with the cause where it was isolated, and labelled where it is inferred.

| rows | what differs | cause |
|---|---|---|
| float32 transformer and llama_step, T4 Triton 3.6.0 against 3.7.1 | same GPU, same source, only Triton changed | Triton's code generation for the attention and reduction kernels changed between 3.6.0 and 3.7.1 (measured: switching Triton alone changes the bits; the MLP, whose kernels are contractions and pointwise tiles, did not change) |
| resnet18, every dtype | the 5070 Ti against the T4 | its convolutions and pooling are library calls (`conv2d`, `max_pool2d`, `adaptive_avg_pool2d` in the compile report), so cuDNN chooses the algorithm per GPU; not a Truestar kernel. Inferred from the report, not isolated per op |
| tf32, every case | the T4 computes these in IEEE float32 | the T4 has no TF32 hardware: on the T4 every tf32 hash equals the float32 hash and the error is float32 sized (1.4e-6, not 2e-3). The compile report still says `fp32_precision: tf32` there; see "Found" below |
| float16, bfloat16 | same Triton, other architecture | inferred: the half precision dot products run on tensor core instructions of another shape on Turing (and Turing has no bfloat16 tensor core instruction), which accumulate in another order. Not isolated per kernel. bfloat16 at batch 1 is equal on mlp and llama_step |
| eager, torch.compile | every row | cuBLAS, cuDNN and Inductor choose kernels per GPU and per batch (row 0 also changes with the batch size in eager float32 on both GPUs) |
| legacy cases on the T4 | not comparable | the first script drew weights and inputs with PyTorch's CPU generator, whose bits on the Kaggle Linux CPU differ from the Windows desktop (below) |

## Found on the way

* **PyTorch's seeded CPU draws are not portable.** The first T4 run (commit 335fc92) used
  `torch.manual_seed` with the default initializers and `torch.randn` with a seeded CPU generator:
  every case's weights and inputs had other bits on the Kaggle host than on the desktop (TS744 on all
  7 cases), so nothing could be compared. The cases now draw with a splitmix64 counter hash and exact
  float arithmetic (`models.portable_uniform`), and `models.GOLDEN` pins the weights and inputs hashes;
  both GPUs matched them (`golden_match` true for the 4 cases on both). A machine whose bits differ gets
  TS750 instead of a silent comparison of different inputs.
* **Deterministic mode tuned attention by timing.** Before commit 3e48656 the attention of the
  TransformerEncoderLayer at batch 256 was tuned from BN 32 to BN 64 in deterministic mode; that
  changed its bits and made row 0 differ between batch 64 and 256. Configs whose bits depend on them
  are now pinned in deterministic mode (docs/KERNEL_IR.md 6b).
* **Asking for TF32 on a GPU without it.** On the T4, `allow_tf32=True` and PyTorch's TF32 flags give
  IEEE float32 results (equal bits), while the compile report says `tf32`. Not fixed in this change.

## The 5060 Laptop (first script)

The first cross chip script ran three cases at batch 4 in float32 on the 5070 Ti and the 5060
Laptop. The fingerprint keeps them as `legacy_mlp`, `legacy_transformer` and `legacy_resnet18`, built
exactly as before, and `compare` reads the old files. On current code the 5070 Ti still gives the same
hashes as the first run for all three cases (deterministic and eager), and those equal the laptop's
except eager on the transformer layer, as in the first run. So the laptop agrees with the desktop for
these three legacy cases (consumer Blackwell, Windows, same software), and the new cases have not been
run on it.

## Run it

```
truestar fingerprint --json mine.json --fresh-cache
truestar fingerprint compare docs/cross_chip/fp2_rtx5070ti.json mine.json
```

Filters: `--cases mlp,transformer,resnet18,llama_step` (plus the three `legacy_*`), `--dtypes
float32,tf32,float16,bfloat16`, `--batches 1,7,64,256`, `--arms
truestar_det,truestar_default,eager,inductor`. `--data-dir` keeps the weights and inputs files
(written once, checked against their manifest on every load, TS740 when changed). `--format agent`
gives one JSON document. The full matrix took 78 s on an idle 5070 Ti and 882 s to 979 s on the T4.

The JSON records the git commit and dirty flag, Truestar, torch, Triton, CUDA runtime, driver, cuDNN,
numpy, Python and platform, GPU name, compute capability, SM count and memory, the tuning mode and
budget, `CUBLAS_WORKSPACE_CONFIG`, per arm the deterministic algorithm, cuDNN and TF32 switches, and
for Truestar arms every launch config with its source (`default`, `tuned`, `cache`, `reused`,
`pinned`), the library ops and the compile time. Per row: sha256 of every output, of a second call,
of row 0, and the error against float64.

`compare` exits 1 only when a deterministic row differs (TS743); every other difference is listed
with its key: environment (TS747), settings (TS745), arms without a claim (TS746), rows in one file
only (TS748), different inputs (TS744). Python: `truestar.fingerprint.run(...)`,
`truestar.fingerprint.compare_files(a, b)`.

## What is checked where

* On one machine (tests/test_fingerprint.py): run to run stability in process and from the command
  line in a fresh process, batch invariance of deterministic MLP rows, pinned configs, the data hashes
  against `GOLDEN`, every `compare` code and exit status.
* Only across machines: bitwise equality of deterministic rows across GPUs, drivers and Triton
  versions. `test_deterministic_rows_match_another_gpu` runs it when `TRUESTAR_FP_OTHER` names a
  fingerprint from another GPU; the table above is that comparison run by hand on the files in
  docs/cross_chip, and `test_committed_results_say_what_cross_chip_md_says` checks that the files still say what this page says.

## Not run

Ampere (sm_80, sm_86), Ada (sm_89), Hopper (sm_90), datacenter Blackwell (sm_100), AMD ROCm, Apple MPS,
Intel XPU, CPU rows, other drivers on the same GPU, Triton other than 3.6.0 and 3.7.1, PyTorch other
than 2.11.0, the new cases on the 5060 Laptop, models with control flow, training steps.
