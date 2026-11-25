# Order probe: which order does a backend add in?

No library documents the order in which it adds the terms of a sum, the K dimension of a matmul or a
softmax denominator, yet that order is why two correct backends, or one library on two GPUs, give
float32 results that differ in the last bits. `truestar orderprobe` recovers the order from outputs
alone, says it in plain words, and proves it by replaying it bit for bit. The idea of probing the
accumulation tree with crafted inputs is from Xie, Gao, Wang and Xue, "Revealing Floating-Point
Accumulation Orders in Software/Hardware Implementations" (USENIX ATC 2025); the method below and its
code are Truestar's own.

```
truestar orderprobe --op linear --n 512 --rows 128 --cols 128 --backends torch:cuda,truestar:cuda
truestar orderprobe --case transformer --backends torch:cuda --json gpu_a.json
truestar orderprobe compare gpu_a.json gpu_b.json
```

Ops: `sum` (`x.sum(-1)` over a [rows, n] input), `linear` (nn.Linear's call, x W^T + b with a zero
bias, the cuBLAS path of a model), `matmul` (x @ W), `softmax` (the denominator of `softmax(x, -1)`).
Backends: `torch:cpu`, `torch:cuda`, `compile:cpu`, `compile:cuda` (torch.compile), `truestar:cpu`,
`truestar:cuda` (a Truestar kernel), `ort:cpu`, `ort:cuda` (ONNX Runtime). The shape matters (cuBLAS
and the CPU GEMM pick kernels by shape), so probe the shape the model runs. Python:
`truestar.orderprobe.probe(op, backend, n, rows, cols)` returns an `OrderResult` (tree, words,
accumulator bits, verified, codes); `probe(Target(...))` probes any callable.

## How the order is read

An order is a tree: each node adds its children and rounds once to the accumulator's precision (p
significand bits). IEEE addition commutes, so the unordered tree decides the result.

* **Sums and matmuls.** +M at term a, -M at term b, a unit u at the other terms of a set S, zeros
  elsewhere, with M = u 2^(p + 1 + ceil(log2 n)). A unit that meets +M or -M before they cancel is
  absorbed (it is under half an ulp of M, and so is any partial sum of units); units that join after
  the cancellation survive. The result counts the terms of S outside the subtree where a and b meet,
  so one row per pair gives that subtree's size. For a pivot a the other terms fall into levels by
  size along a's path to the root; each level is probed again alone. A matmul uses a weight of ones,
  so every product is exact.
* **Softmax denominators** (exp is positive, so no cancelling pair): logit 0 at a (exp gives 1
  after the max is subtracted), log(d) at b and c with d = 0.75 2^-p, minus infinity elsewhere. One d
  meeting 1 is absorbed, two d added first survive, so `softmax[a] < 1` exactly when b and c meet
  before either meets a. Grouping and ordering the levels uses the same triplet test.
* **Accumulator width.** If no float32 tree fits (nothing is absorbed), the probe retries at 53 bits.
* **Rows.** Probe rows are batched only when the same data in every row gives the same result in every
  row (4 random trials); otherwise (CPU GEMM micro kernels: the edge rows take another path) each
  probe runs alone in row 0 and that row's order is reported.
* **Repeatability.** Every batch runs twice; a different answer is **TS963** (atomics or a run
  dependent split) and no tree is claimed.

**Proof.** The tree is replayed with one round to nearest per node, in exact arithmetic (two sum and
a midpoint correction), on 64 random rows, and must give the backend's bits (**TS962**). Otherwise
**TS964**. For a torch matmul the replay is repeated with random weights under two product models
(fused multiply add per added product, or products rounded first) and the matching one is named.
Softmax trees are checked by the triplet answers only (the backend's exp is not replayed). Two
backends with different trees on the same reduction get **TS965**, with the number of random rows on
which the two trees give different float32 results.

The words come from the tree: sequential, pairwise (down to blocks), split into contiguous chunks,
interleaved accumulators (term i to accumulator i mod k), a balanced tree over interleaved halves
(the shape of a warp shuffle reduction), a fused node (several terms with one rounding), each level
described recursively.

## Measured on this PC (2026-11-24, RTX 5070 Ti, 70 SMs, torch 2.11.0+cu128, Triton 3.7.1, Windows, TF32 off)

`truestar orderprobe --case transformer --backends torch:cuda,truestar:cuda` probes the reductions of
the fingerprint's TransformerEncoderLayer(128, 4, 512) at batch 4 (128 rows). Every tree below
replayed bit for bit (TS962); cuBLAS matched the fused multiply add product model.

| reduction | eager PyTorch (cuBLAS, CUDA kernels) | Truestar compiled kernel |
|---|---|---|
| in_proj, out_proj, linear1 (K = 128) | split K: 2 contiguous halves of 64, each sequential, then one add | sequential over 128 |
| linear2 (K = 512) | split K: 4 contiguous chunks of 128, each sequential, the 4 partials added in order | 2 halves of 256, each sequential, then one add |
| attention softmax (32 keys, 512 rows) | a balanced tree over interleaved halves (i meets i + 16, + 8, ...) | 8 chunks of 4, sequential inside, then interleaved halves |

Others on this PC: `torch:cpu` sum of 512 is 32 interleaved accumulators (8 wide vectors times 4),
each sequential, then combined; `torch:cpu` linear with K = 512 at 128 x 128 is 3 chunks of 192 (last
128); `truestar:cpu` (numba) is sequential with a float64 accumulator (53 bits found by the probe, as
docs/KERNEL_IR.md states); `torch:cuda` sum of 512 is 128 interleaved accumulators of 4, then chunks
of 4, then a shuffle tree over 32.

## Measured on a Mac mini M2 (2026-11-25, 10 core GPU, macOS 26.6.2, torch 2.14.1)

Backends `torch:mps`, `compile:mps` and `truestar:mps` (a Truestar Metal kernel), 64 rows, 64
columns. Every sum, linear and matmul tree replayed bit for bit (TS962); softmax trees are read from
the triplet answers only. Raw: <research>/mac_real/orderprobe_mps*.txt and the JSON
files next to them.

| reduction | MPS eager | Truestar Metal | torch CPU (M2) |
|---|---|---|---|
| sum of 64 | 32 interleaved accumulators of 2, then a balanced tree | sequential | 16 interleaved accumulators, then 4, then sequential |
| sum of 512 | 256 interleaved accumulators of 2, then 32 blocks of 8 sequential, pairwise | a shuffle tree (i meets i + 256, + 128, ...) | 16 interleaved accumulators of pairwise blocks |
| sum of 4096 | 64 chunks of 64 (shuffle tree each), then 32 accumulators, then pairwise | 256 interleaved accumulators of 16, then a shuffle tree | 16 accumulators of 16 chunks of 16, then 4 |
| linear and matmul, K 64 and 512 | sequential | sequential | sequential |
| linear and matmul, K 4096 | **split K**: 16 chunks of 256, each sequential, the partials added in order | sequential | sequential |
| softmax denominator, 64 and 512 | shuffle tree | shuffle tree | 4 interleaved accumulators, then a shuffle tree |

`compile:mps` (Inductor's Metal code) sums 512 terms and softmax denominators as a pairwise tree
over contiguous halves, another order than eager MPS; its linear is eager's (sequential). So on the
Mac the same matmul gives MPS eager, Truestar and CPU bits that differ from K 4096 on (TS965), and a
row sum differs between eager MPS, torch.compile and the CPU at every length probed. Truestar's
Metal contraction and the M2's CPU GEMM both sum sequentially at every K probed.

Two probe fixes came from the Mac: outputs are copied to the CPU before the float64 conversion (the
device has none), and a 4096 term sequential order (4096 tree levels) overflowed the tree helpers'
recursion on Python 3.12; deep probes now run on a 64 MB stack thread with a raised limit. The
softmax probe at 4096 terms was killed by the 16 GB machine's memory limit and is not reported.

## The README row: eager PyTorch differs between two GPUs on the transformer layer

README_DRAFT.md ("Same model, two GPUs") and docs/QUALITY_BAR.md (P41) show eager PyTorch giving other
bits on the RTX 5060 Laptop (26 SMs) and on a Tesla T4 than on the RTX 5070 Ti (70 SMs) for the
transformer layer, with the same torch, driver and cuDNN. On the 5070 Ti, every Linear of that layer
runs cuBLAS with split K (2 chunks at K = 128, 4 at K = 512): the output has few tiles (128 rows), so
the K dimension is split to keep the SMs busy. The split factor is the kind of choice a heuristic makes
from the SM count, and a different split is a different tree, which gives other bits for correct
math. That a 26 SM GPU picks a different split is **inferred, not measured here**: the other GPUs
were not available to this run. To settle it, run on each GPU

```
truestar orderprobe --case transformer --backends torch:cuda --json <gpu>.json
truestar orderprobe compare rtx5070ti.json <gpu>.json
```

`compare` names each reduction whose order differs (TS965) and how many of 64 random rows the two
trees round differently, so the README row becomes "linear2: 4 chunks on 70 SMs, k chunks on 26".
Truestar's compiled outputs had equal hashes on both GPUs (docs/CROSS_CHIP.md); the Truestar column
above is one kernel compiled with `truestar.kernel.compile` at default options on this PC, not the
compiled executor's deterministic mode, and its tree on the other GPUs was not probed either.

## Not covered

The attention score and value products (batched matmuls over heads) and layer norms (not plain sums)
of the case; fused multiply add placement beyond the two product models; accumulators other than 24
or 53 bits (a tensor core's internal width); float16 and bfloat16 probes are best effort (bfloat16
outputs hold counts up to 256 exactly); `compile:` and `ort:` backends were not run on this PC.
