# Core ML: placement and numerics of saved runs

Author: Sharven Rane. First written 2026-11-25. Code: `src/truestar/conform/coreml.py`.
Pain ledger row: P61 (docs/QUALITY_BAR.md).

Core ML takes a `compute_units` hint (CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL) and decides per
operation where to run. Nothing in the prediction API says where an operation ran, whether a
request for the Neural Engine was honored, or how precise the arithmetic was. `truestar conform
coreml` answers those from saved runs of a fixed battery.

## 1. Use

On a Mac with coremltools (collect), then anywhere with torch (check):

```
python -m truestar.conform.coreml collect DIR     # converts and runs the battery, about 2 minutes on an M2
truestar conform coreml DIR [--format agent]      # verdict per run, exit 1 on a finding
```

The battery: an MLP (1024, 2048, 2048, 16, batch 128), a conv net (three 3 x 3 convolutions,
8 x 3 x 128 x 128), a transformer encoder layer (d 256, 8 heads, 128 tokens, batch 4) and the MLP
with int8 weights (Core ML `linear_symmetric`). Weights and inputs are portable draws
(`truestar.fingerprint.models.portable_*`), so the check side rebuilds them bit for bit on any OS
and refuses a bundle whose input hash differs (TS1166). Each model is converted at FLOAT32 and
FLOAT16 compute precision and run with each of the four compute units: 32 runs.

`collect` saves per run the output, two predictions compared bitwise, the median latency and Core
ML's compute plan (`MLComputePlan`: per operation the preferred device and the devices that support
it). `check` recomputes float64, float32, `sim:ieee-f16` (float16 storage, wide accumulation) and
`sim:mali-fp16` (a float16 accumulator) references itself. The int8 runs are judged against the
float64 model with the dequantized weights, so weight rounding is not counted as error.

## 2. Verdicts and codes

Accuracy, max relative error against float64: float32 when within 8x the float32 prediction (or
1e-5), float16 when within 4x `sim:ieee-f16`, beyond_float16 above (TS1163, with the ratio to the
float16 accumulator model). A FLOAT32 program at float16 level is TS1162.

Placement: a run asking for the GPU or the Neural Engine whose plan puts operations elsewhere is
TS1160 when the requested unit supports those operations (Core ML chose otherwise) and TS1161 when
it supports none of them at the program's precision. TS1167 is a run whose two predictions differ
in bits; TS1164 a failed conversion or run; TS1165 a run with no plan saved; TS1168 an unreadable
bundle (usage error, exit 2).

## 3. Measured (2026-11-25, Mac mini M2, 10 core GPU, macOS 26.6.2, coremltools 9.0, torch 2.14.1)

`truestar conform coreml` on the 32 run bundle (<research>/mac_real/coreml/):

| model | precision | CPU_ONLY | CPU_AND_GPU | CPU_AND_NE | ALL |
|---|---|---|---|---|---|
| mlp | FLOAT32 | 7.3e-07 CPU | 1.2e-06 GPU | 7.3e-07 CPU, TS1161 | 1.2e-06 GPU |
| mlp | FLOAT16 | 6.6e-03 CPU, 11.2x sim, TS1163 | 6.2e-04 GPU, 1.05x | 5.7e-03 NE, 9.6x, TS1163 | 6.2e-04 GPU |
| mlp int8 weights | FLOAT16 | 5.5e-03 CPU, 9.2x, TS1163 | 6.5e-04 GPU, 1.09x | 5.2e-03 NE, 8.7x, TS1163 | 6.5e-04 GPU |
| conv net | FLOAT16 | 1.5e-03 CPU, 2.3x | 6.5e-04 GPU, 1.00x | 1.8e-03 NE, 2.7x | 1.8e-03 NE |
| encoder | FLOAT16 | 1.8e-03 CPU, 1.1x | 1.2e-03 GPU, 0.74x | 1.9e-03 NE, 1.2x | 1.9e-03 NE |

What it shows:

- **The GPU is correct float16** (0.74x to 1.09x `sim:ieee-f16` on every model). The CPU and the
  Neural Engine are not on the MLP: 8.7x to 11.2x correct float16, between correct float16 and a
  float16 accumulator (0.50x to 0.79x `sim:mali-fp16`), so their long dot products (2048 terms)
  accumulate at reduced precision. On shorter reductions (conv, 288 and 576 terms; encoder) they
  stay within 4x.
- **A FLOAT32 program on CPU_AND_NE runs entirely on the CPU** (TS1161 on all four models): the
  Neural Engine supports none of the operations at FLOAT32. The prediction succeeds with no
  warning, and on the large conv net in the study it took 19 ms against 6.5 ms on CPU_AND_GPU.
- **Small models run on the CPU whatever is asked.** In the separate study
  (<research>/mac_real/coreml/table.md) an MLP of 512 wide at batch 8 and a conv net
  on 32 x 32 inputs were planned on the CPU for CPU_AND_GPU and ALL too, though the GPU supports
  every operation (TS1160 in this command's terms).
- **ALL is a per operation choice**: on the 4096 wide MLP at FLOAT16 four operations went to the
  Neural Engine and one to the GPU.
- Every run gave the same bits twice (no TS1167).

## 4. Limits

The compute plan is Core ML's plan for the model on this machine, read through coremltools; it is
not a trace of the run. Latency per unit agrees with it in every case measured (the Neural Engine
runs 1.7x to 5.1x faster than the GPU on the large FLOAT16 models of the study). One Mac, one macOS; other chips
(M3, M4, A series) can plan differently. The thresholds are those of `truestar conform qaihub`.
