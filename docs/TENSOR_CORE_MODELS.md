# Tensor core models: other GPUs' matrix units as simulated devices

Author: Sharven Rane. 2026-11-24.

Truestar owns one GPU (an RTX 5070 Ti, sm_120). A matmul on a V100, A100, H100 or B200 does not
round like a matmul on that card: the matrix units sum a group of products at once, align them to
the largest exponent with a fixed number of bits, truncate what falls off, and truncate the
result. This page describes a bit accurate software model of that inner product, one profile per
generation, and how the sm_120 profile was checked against the real card: **32,665,225 random dot
products, every one bitwise equal**, across fp16, bf16, tf32, E4M3 and E5M2 inputs, Triton
`tl.dot`, cuBLAS and `torch._scaled_mm`. A second generation, an NVIDIA L4 (Ada, sm_89) rented
for two short cloud runs, confirmed the paper's Ada parameters for fp16, bf16 and tf32 bit for bit
and contradicted its Ada fp8 model (section 4.5).

Code: `src/truestar/devsim/tensorcore.py` (the model), the `sim:*` tensor core profiles in
`src/truestar/devsim/profiles.py`, measurement in `scripts/tc_probe_sm120.py` and
`scripts/tc_validate_sm120.sh`, raw results in `docs/data/tensor_core_sm120.jsonl`, tests in
`tests/test_tensorcore_model.py`.

Sources: Khattak and Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004 (v4,
read 2026-11-24), and Fasi, Higham, Mikaitis and Pranesh, "Numerical behavior of NVIDIA tensor
cores", PeerJ Computer Science 7:e330 (2021). The model is written from their published
description; none of their code was used.

## 1. Use

```
truestar check model.py --backends cpu,sim:h100-fp8,sim:sm120-fp8
truestar backends --sims          # every profile, every rule with source and status
```

From Python:

```python
from truestar.devsim import simulate
from truestar.devsim.tensorcore import unit_for, unit_matmul

with simulate("sim:h100-bf16"):
    y = model(x)                    # mm, addmm, bmm, baddbmm, mv, dot go through the H100 unit

d = unit_matmul(a, b, c, unit_for("h100", "e4m3"), "e4m3")   # D = A B + C, float64 numpy
```

`unit_matmul` takes numpy arrays whose values are already representable in the input format
(and `c` in the accumulator format), with any leading batch shape, and returns float64 values that
are exactly the unit's fp32 (or fp16) results. 256 x 256 x 256 in fp16 takes about 0.3 s on a CPU.

A tensor core profile changes only the matmul ops listed above. Their float inputs are rounded to
the profile's input format (fp8 profiles first scale each operand by a power of two so its largest
magnitude fits the format, then undo the scale exactly on the output), the product runs through
the unit model, the result is the fp32 accumulator. Everything else in the model runs in its own
dtype. The profile's ideal (the noise floor `truestar check` calibrates against) keeps the same
input rounding and sums exactly, rounding once to nearest even, so a finding on a tensor core
profile is about the unit's truncation and grouping, not about fp16 or fp8 inputs as such.

## 2. The model

One fused group of `group` products plus the accumulator `c`:

1. every product `a_i b_i` is exact and stays unnormalized: its significand (up to two integer
   bits) keeps the exponent `ea + eb`. So 1.5 x 1.5 = 2.25 has exponent 0 while 1 x 2.25 has
   exponent 1, and they align differently;
2. `E` is the largest exponent among the nonzero products and a nonzero `c`, raised to the unit's
   exponent floor when it has one;
3. every term is put on the grid `2^(E - frac_bits)`; bits below it are dropped from the
   magnitude (truncation toward zero, sign magnitude);
4. the truncated terms are added exactly (the hardware has enough carry bits);
5. the sum is rounded once to the accumulator: truncation for fp32 output, round to nearest even
   for fp16 output. A zero result is +0.

A dot product longer than one group runs the groups in K order, the rounded result of one group
being the `c` of the next, starting from zero (or from C). Hopper and B200 fp8 through `mma.sync`
add a variant: the 32 products go through the fp16 unit as two halves of alternating pairs, and c
is added afterwards with round to nearest even.

Inputs or outputs holding inf or NaN follow IEEE float64 arithmetic for the outputs they reach.
The hardware's special value rules were not modelled.

## 3. Profiles

`frac_bits` is the alignment width below the leading integer bits (23 plus the paper's extra
alignment bits). Status follows the devsim convention: measured (a paper or Truestar measured it on
the device), documented, hypothesis, guess.

| profile | unit | group | frac bits | final | exponent floor | source, status |
|---|---|---|---|---|---|---|
| `sim:v100-fp16` | V100 fp16 | 4 | 23 | truncate | none | paper 4.1.1, Table 3 (measured by the paper) |
| `sim:a100-fp16`, `sim:a100-bf16` | A100, A2, A30 | 8 | 24 | truncate | -132 | paper 4.1.2, 4.1.3 (measured by the paper) |
| `sim:a100-tf32` | A100 tf32 | 4 | 24 | truncate | -132 | paper 4.1.2 (measured by the paper) |
| `sim:ada-fp16`, `sim:ada-bf16` | Ada (L4, L40S, RTX 40 series) | 8 | 24 | truncate | -132 (bf16) | paper 4.1.4, 4.1.5; **Truestar fitted on an L4, section 4.5** |
| `sim:ada-tf32` | Ada tf32 | 4 | 24 | truncate | -132 | paper; **Truestar fitted on an L4** |
| `sim:ada-fp8` | L40S, RTX 1000, E4M3 via QMMA | 16 | 13 | truncate | none | paper 4.1.4, 4.1.5 (measured by the paper); **not reproduced on an L4 through Triton** (section 4.5) |
| `sim:h100-fp16`, `sim:h100-bf16` | H100, H200 | 16 | 25 | truncate | -133 | paper 4.1.6 (measured by the paper) |
| `sim:h100-tf32` | H100 tf32 (m16n8k8) | 8 | 25 | truncate | -133 | paper 4.1.6 (measured by the paper) |
| `sim:h100-fp8` | H100 E4M3 via wgmma (the path cuBLAS and CUTLASS use) | 32 | **13** | truncate | none | paper 4.1.6 (measured by the paper) |
| `sim:h100-fp8-mma` | H100 and B200 E4M3 via mma.sync: fp16 unit, two interleaved halves, late c with round to nearest even | 16 per half | 25 | truncate, then late c rounded to nearest | -133 | paper 4.1.6, 4.1.7 (measured); which half runs first (hypothesis, Fig. 5c) |
| `sim:b200-fp16`, `sim:b200-bf16` | B200 | 16 | 25 | truncate | -133 | paper 4.1.7 (measured by the paper) |
| `sim:b200-tf32` | B200 tf32 | 8 | 25 | truncate | -133 | paper 4.1.7 (measured by the paper) |
| `sim:b200-fp8` | B200 E4M3 via tcgen05 | 32 | 25 | truncate | none | paper 4.1.7 (measured by the paper) |
| `sim:sm120-fp16`, `sim:sm120-bf16` | RTX 50 series, RTX PRO 6000 (sm_120) | 16 | 25 | truncate | -133 (bf16) | **Truestar measured, section 4** |
| `sim:sm120-tf32` | sm_120 tf32 | 8 | 25 | truncate | -133 | **Truestar measured** |
| `sim:sm120-fp8` | sm_120 E4M3 (E5M2 also measured) via mma.sync, QMMA | 32 | 25 | truncate | none | **Truestar measured** |

Rules shared by every tensor core profile, with their status:

* products exact and unnormalized; sign magnitude truncation at alignment; c summed with the
  products (paper, measured; Truestar measured for sm_120);
* groups chained in K order from zero, no split K, no fp32 promotion step: one warp's
  instruction chain (guess for what a given library does; see TS1097 and section 4.3);
* a zero result is +0 (Truestar measured on sm_120; assumed elsewhere);
* addmm and baddbmm add the bias after the product in fp32 with round to nearest even (guess);
* tf32 inputs rounded to nearest even from float32 (guess; the hardware ignores the low bits, and
  what cuBLAS does before that was not checked);
* fp8 per tensor power of two scaling (a Truestar choice, not device behaviour);
* inf and NaN follow IEEE float64 (guess).

**Every generation other than sm_120 and Ada's 16 bit and tf32 paths is exactly as good as the
paper's description.** Truestar has not run a single dot product on a V100, A100, H100 or B200. The models reproduce the
paper's own probe vectors (tests below), which shows the code implements the description; it does
not show the description is complete. Using one of those profiles in `truestar check` carries
TS1094.

The paper's B200 and RTX PRO 6000 parameters are the same as what Truestar measured on the RTX
5070 Ti, so the sm_120 measurement is also indirect evidence that the paper's method yields
complete models for that family. It says nothing direct about H100's 13 bit fp8 path, which has
no counterpart on sm_120.

## 4. Validation on the RTX 5070 Ti (sm_120)

Torch 2.11.0+cu128, Triton 3.7.1, Windows, 2026-11-24. Every number below is in
`docs/data/tensor_core_sm120.jsonl`; `sh scripts/tc_validate_sm120.sh <python>` reproduces it in
about ten minutes of mostly CPU time (each GPU call is under a second).

### 4.1 Method

* **GPU paths.** A Triton kernel that loads a 16 x K tile of A, a K x 16 tile of B and a 16 x 16
  tile of C, and calls `tl.dot(a, b, acc)` over K in one tile, so the compiled code is one chain of
  MMA instructions with C as the starting accumulator. cuBLAS through
  `torch.mm(a, b, out_dtype=torch.float32)` (and `allow_tf32` for tf32). fp8 through
  `torch._scaled_mm` with unit scales, with and without `use_fast_accum`.
* **Probes** in the paper's style: unnormalized products (2.25 as 1.5 x 1.5 against 1 x 2.25),
  `1 - 2^-26` for the truncation direction, 1 + 2^-24 + 2^-24 with the second small term moved
  through positions 2 to 63 for the group size, zero times a large number.
* **Fit.** A designed ensemble (normal inputs; normal with nonzero C; exponents spread over the
  format's range; nearly cancelling pairs; for bf16 and tf32 products all below 2^-133) of 49,152
  to 81,920 dot products at K = 32, scored against every parameter set in a grid: group 4, 8, 16, 32;
  fraction bits 22 to 27 (12, 13, 14, 23 to 26 for fp8); truncation sign magnitude or two's
  complement; final truncation or round to nearest; exponent floor none, -132, -133, -134.
* **Validation.** About 10^6 random dot products per row of the table below from the same five
  ensemble kinds, a new seed, compared as fp32 bit patterns (so +0 and -0 count as different).

### 4.2 Fitted parameters

Each fit had exactly one parameter set with every output equal; the next best is shown.

| inputs | outputs in fit | best set (all equal) | next best |
|---|---|---|---|
| fp16 | 65,536 | group 16, 25 fraction bits, sign magnitude cut, truncate | 50,296 (26 bits) |
| bf16 | 81,920 | group 16, 25 bits, sign magnitude, truncate, floor -133 | 81,901 (floor -134) |
| tf32 | 81,920 | group 8, 25 bits, sign magnitude, truncate, floor -133 | 81,910 (floor -134) |
| E4M3 | 65,536 | group 32, 25 bits, sign magnitude, truncate | 62,242 (26 bits) |
| E5M2 | 65,536 | group 32, 25 bits, sign magnitude, truncate | 60,553 (26 bits) |
| fp16, fp16 accumulator | 49,152 | group 16, 25 bits, round to nearest even directly from the truncated sum | 49,151 (two's complement cut) |

The feature probes agree: 67 of 67 equal, the group boundary between positions 15 and 16, and
`1 - 2^-26` returns 1 (a two's complement cut would return `1 - 2^-24`).

What the fit added to the paper: a zero result is always +0, even for `-0 + -0` and for a negative
sum truncated to zero (first validation had 41 bf16 mismatches out of about 200,000 in the tiny
ensemble, all of them this); and the fp16 accumulator path rounds the truncated sum straight to
fp16 with round to nearest even, not through an fp32 truncation first as the paper's Fig. 2 and 5
draw it (49,149 against 49,152 for that variant).

### 4.3 Bitwise agreement

| inputs | path | K | equal / dot products |
|---|---|---|---|
| fp16 | Triton tl.dot | 16, 64, 256 | 999,424 / 999,424 each |
| bf16 | Triton tl.dot | 16, 64, 256 | 999,680 / 999,680 each |
| tf32 | Triton tl.dot (input_precision tf32) | 16, 64, 256 | 999,680 / 999,680 each |
| E4M3 | Triton tl.dot | 32, 64, 256 | 999,424 / 999,424 each |
| E5M2 | Triton tl.dot | 32, 64, 256 | 999,424 / 999,424 each |
| fp16 with fp16 accumulator | Triton tl.dot | 32, 256 | 999,936 / 999,936 each |
| fp16 | cuBLAS, side 496 | 16, 256, 1024 | 984,064 / 984,064 each |
| bf16 | cuBLAS, side 448 | 16, 256, 1024 | 1,003,520 / 1,003,520 each |
| tf32 | cuBLAS, side 448 | 256 | 1,003,520 / 1,003,520 |
| tf32 | cuBLAS, side 448, modelled as split K in 2 | 16 | 1,003,520 / 1,003,520 |
| bf16 | cuBLAS, side 447 (unaligned), modelled with groups of 8 | 256 | 999,045 / 999,045 |
| fp16 | cuBLAS, side 447 (unaligned) | 256 | 799,236 / 799,236 |
| E4M3 | `_scaled_mm`, fast accumulation off and on | 32, 256, 1024 | 984,064 / 984,064 each (6 rows) |

Total 32,665,225 of 32,665,225. The unit model is exact on this card. Two rows needed the
library's schedule, not the unit, to be modelled, and they show why TS1097 exists:

* cuBLAS tf32 at K = 16 (side 448) splits K into two slices of 8, runs each slice from zero and
  adds the two fp32 results with round to nearest even. The unit model alone gives 661,297 of
  1,003,520; with that split, all of them.
* cuBLAS bf16 with an unaligned leading dimension (447) uses an m16n8k8 instruction, so the group
  is 8 products, not 16: 867,522 of 999,045 with group 16 at K = 16, all equal with group 8. fp16
  at the same shape kept groups of 16.

`torch._scaled_mm` on sm_120 with `use_fast_accum` on or off gave identical bits up to K = 1024,
matching a plain chain of 32 product groups: at these sizes neither setting promoted the
accumulator.

### 4.4 A bug found on the way

PyTorch's CPU cast from float64 to float16 goes through float32, which can round twice:
`0x1.f420008cp+3` becomes `0x1.f4p+3` instead of `0x1.f44p+3`. The tensor core model rounds with
its own exact routine (`tensorcore.round_to`, checked against numpy's correct cast on 2 x 10^6
values). The rest of devsim still rounds through torch; docs/DEVICES.md already lists that double
rounding as a limit.

### 4.5 A second generation: NVIDIA L4 (Ada, sm_89)

Two runs of `scripts/tc_probe_sm120.py fitall` on a Modal L4 (Linux, torch 2.11.0+cu130 with its
bundled Triton), the same designed ensembles and grid as section 4.2, through Triton `tl.dot`.
Results in `docs/data/tensor_core_l4.jsonl`.

| inputs | outputs in fit | result |
|---|---|---|
| fp16 | 65,536 | all equal for exactly one set: group 8, 24 fraction bits, sign magnitude, truncate (next best 35,881) |
| bf16 | 81,920 | all equal for exactly one set: group 8, 24 bits, floor -132 (next best 81,868 with floor -133) |
| tf32 | 81,920 | all equal for exactly one set: group 4, 24 bits, floor -132 (next best 81,870) |
| E4M3 | 65,536 | no set is exact; best is the paper's Ada model (group 16, 13 bits) at 52,880 |
| E5M2 | 65,536 | no set is exact; best is the paper's Ada model at 55,801 |

The 16 bit and tf32 results are an independent confirmation of the paper's A100 and Ada unit on a
GPU the paper did not test (it measured L40S and RTX 1000), so `sim:ada-fp16`, `sim:ada-bf16` and
`sim:ada-tf32` are marked validated. These were fits of 65,536 to 81,920 dot products, not the
10^6 per row of section 4.3.

fp8 is open. The grid covered groups of 8, 16 and 32 with 12, 13, 14 and 23 to 26 fraction bits,
so neither the paper's QMMA model nor an fp16 unit path (Hopper's `mma.sync` behaviour) explains
what Triton produces on sm_89; this may be Triton's lowering (it may split the K = 32 tile or
convert operands differently from the paper's `mma.sync` kernels) rather than the unit. Until it
is resolved, `sim:ada-fp8` follows the paper and carries a measured rule saying it does not match
an L4 through Triton.

## 5. Codes

| code | severity | meaning |
|---|---|---|
| TS1094 | info | the tensor core profile's generation was never measured by Truestar; it follows the published description only |
| TS1095 | warning | a matmul format the profile's unit has no path for; accumulated exactly instead, named |
| TS1096 | info | ops that run on matrix units on the device (convolution, fused attention) computed exactly on a tensor core profile, named |
| TS1097 | info | the profile chains the unit's groups in K order from zero; a library may split K, use another instruction shape or promote the accumulator |

## 6. Limits

* Only sm_120 is validated on 10^6 dot products per path, and Ada's fp16, bf16 and tf32 units by
  a fit on an L4. The paper's other generations are reproduced from its text and figures,
  including its probe vectors, and nothing more. Ada fp8 through Triton does not match.
* The schedule is one instruction chain in K order. Real GEMM libraries choose split K, instruction
  shapes and (for Hopper fp8 without fast accumulation) periodic fp32 promotion by shape; none of
  that is modelled per library (TS1097). Section 4.3 shows two such choices on sm_120.
* Convolution, fused attention and `_scaled_mm` inside a model are not routed through the unit
  (TS1096). `matmul`, `linear` and `einsum` are, because they reach ATen as mm, addmm or bmm.
* Which half of the Hopper `mma.sync` fp8 split runs first is a hypothesis read off a figure.
* FP6, FP4 and block scaled formats are not modelled. Neither are fp64 tensor cores (IEEE FMA
  per the paper) nor non NVIDIA matrix units.
* Special values (inf, NaN) and overflow inside the unit follow IEEE float64 by assumption.
* fp8 profiles use one per tensor power of two scale; a model with its own scales (as with
  `torch._scaled_mm`) is not read.
* Forward only, like the rest of devsim.
