# Bitwise reproducible results across GPUs

Script: [examples/cookbook/cross_gpu_bitwise.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/cross_gpu_bitwise.py)

`truestar fingerprint --json fp.json`, run on a GPU, hashes the raw output bytes of a few fixed
models (the same weight and input hashes on every machine) for Truestar's compiled executor in
deterministic mode, with eager PyTorch recorded as the baseline. Comparing two files needs no GPU.
The script reads the two committed in `docs/cross_chip/`; pass your own as arguments.

```python
from truestar.fingerprint import compare_files, load, match_table

docs = [("A", load("fp_a.json")), ("B", load("fp_b.json"))]
rows = match_table(docs, arm="truestar_det")       # per case, dtype and batch: hash on A, on B, equal
compare_files("fp_a.json", "fp_b.json").env_diff   # what differs between the two machines
```

Output:

```
A: NVIDIA GeForce RTX 5070 Ti (cc 12.0, 70 SMs, Triton 3.7.1)
B: Tesla T4 (cc 7.5, 40 SMs, Triton 3.7.1)
environment differences: gpu, capability, sm_count, gpu_memory_mib, driver, cuda_runtime, cudnn, torch, torchvision, numpy, python, platform, cpu, git.commit

case         batch  Truestar compiled, A / B     equal  eager PyTorch equal
llama_step       1  704b824b,320b06aa,dd90a4dd,e543a6a5,3ea1f944 / 704b824b,320b06aa,dd90a4dd,e543a6a5,3ea1f944 True   False
mlp              1  e0410306 / e0410306          True   False
...
transformer    256  b17c8f0e / b17c8f0e          True   False

float32, 12 rows: Truestar compiled bitwise equal on 12, eager PyTorch on 0
```

To add a machine: `truestar fingerprint --json mine.json` on it, then
`truestar fingerprint compare mine.json docs/cross_chip/fp2_rtx5070ti.json`, or this script with
both paths.

## What the claim is

Measured on these two GPUs only ([CROSS_CHIP.md](../CROSS_CHIP.md)): float32, TF32 off,
`deterministic=True`, PyTorch 2.11 and Triton 3.7.1 on both, an MLP, a TransformerEncoderLayer and a
two layer Llama style decode step at batch 1, 7, 64 and 256. It does not hold for float16,
bfloat16 or TF32, for resnet18 (its convolutions are cuDNN calls), or for another Triton version
(3.6.0 on the same T4 changed bits). Ampere, Ada, Hopper, AMD, Apple and Intel GPUs have not been
run. Truestar does not claim bitwise equality with float64; its compiled results are checked
against float64 within a stated bound.

To make your own native program follow the same rules, compile it with
`CompiledExecutor("cuda", "float32", deterministic=True)`.
