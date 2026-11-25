# Getting started

Every command on this page runs today, on CPU, with the files in `examples/`, and each finishes in
under 10 s on a desktop CPU. tests/test_examples.py runs the same commands and checks their output.
Run them from the repository root.

## Install

Install PyTorch (2.11 to 2.15, the range tested) the way pytorch.org says for your machine, then
Truestar from a checkout:

```
pip install .
truestar backends
```

`truestar backends` prints the backends PyTorch can use here as JSON. Without installing, run
`PYTHONPATH=src python -m truestar` wherever this page says `truestar`.

## The model file

Every command takes a Python file that defines `get_model()`, returning an `nn.Module`, and
`get_inputs()`, returning a tensor or a tuple of forward arguments. `get_kwargs()` is optional. Use
random weights: Truestar compares implementations, so trained weights are not needed.

```python
import torch
from torch import nn

def get_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()

def get_inputs():
    torch.manual_seed(1)
    return torch.randn(8, 16)
```

## 1. Check a model on every backend

```
truestar check examples/cpu_check.py
truestar check examples/check_model.py --backends cpu
```

The first model is clean: `overall: OK`, exit code 0. The second has a layer norm that computes the
variance as `mean(x * x) - mean(x) ** 2` on inputs near 3000, which cancels to rounding error in
float32. The report blames `norm#0 (OnePassNorm)` with the file and line of the class, names
`aten.sub.Tensor` as the first op that goes wrong, and writes two scripts that reproduce it with
plain PyTorch:

```
python truestar_repro/cpu_norm_0/repro.py        # exit 1: reproduced
python truestar_repro/cpu_norm_0/op_repro.py     # the single op, exit 1
```

On a machine with a GPU, `--backends cpu,cuda` checks both. `examples/fault_demo.py` (needs
torchvision and CUDA) is a ResNet18 with a layer that is wrong only on CUDA; CPU reports OK and CUDA
blames `layer3.1.bn2.1`. Use `--format agent` for JSON, `--json report.json` to also save the full
report, `--no-repro` to skip the scripts.

## 2. Check the backward pass

```
truestar check examples/backward_model.py --backends cpu
truestar check examples/backward_model.py --backward --backends cpu
```

`ScaledTanh` has a correct forward and a hand written backward that forgets a square. The forward
check reports OK; the backward check reports TS180, blames `act#0 (ScaledTanh)`, says its forward
matches so the backward itself is wrong, and writes `truestar_repro/cpu_act_0_backward/grad_repro.py`.
Add `--steps 20` to also run SGD steps and watch the gradient norm drift (docs/GRAD_CHECK.md).

## 3. Record, upgrade, diff

```
truestar record examples/record_diff_model.py -o before.tsrec --backend cpu --ops all
truestar diff before.tsrec --replay
TRUESTAR_EXAMPLE_UPGRADE=1 truestar diff before.tsrec --replay
```

`record` stores every module's inputs and outputs and the environment fingerprint. The first diff
replays every module alone here and reports `IDENTICAL` (exit 0). The environment variable in the
second diff stands in for an upgrade: it makes `Norm` use another variance divisor, so the diff
reports `DIVERGENT` with `norm#0 (Norm)` as the first module whose own result changed (exit 1).
After a real upgrade of torch, CUDA, cuDNN or the driver, run the same `diff` with no variable; a
change explained by a precision flag such as TF32 is reported as a precision change, not a bug.
Windows PowerShell: `$env:TRUESTAR_EXAMPLE_UPGRADE = "1"` before the command.

## 4. Batch invariance

```
truestar invariance examples/invariance_model.py --backends cpu
```

Each example row runs alone and inside batches of 2, 7 and 64 at several positions. `Standardize`
subtracts the batch mean, so the row's answer depends on its neighbours: TS052, first batch
dependent module `norm#0 (Standardize)`, op `aten.mean.dim`, and argmax flips counted. The command
also reruns with torch flag sets and reports that none fixes it. Exit code 1.

## 5. Check a port

```
truestar port examples/port_reference.py examples/port_candidate.py
```

The candidate renames every module and hardcodes eps 1e-6 where the reference uses 1e-5. Modules
are paired by weight content, so names do not matter. The report names `blocks.0.norm (Norm)` and
the op: `constant 1e-06 instead of 1e-05`. `port_candidate.py` defines `get_model(reference)`, which
receives the reference model and copies its weights, the way a converter script does.

## 6. Export and run

```
truestar export examples/export_model.py -o small_cnn.tsx
python examples/export_model.py inputs.pt
truestar run small_cnn.tsx inputs.pt -o outputs.pt
truestar run small_cnn.tsx --info
```

`export` writes the artifact only after it matches PyTorch in float64 at the example batch (2) and
at a second batch (7); otherwise it exits 1 and writes nothing. The second line writes a batch of 3
inputs, which `run` accepts because the batch dim stays symbolic. `--info` prints the signature and
the stored verification evidence.

## 7. Native functions and custom ops

```
python examples/native_function.py
python examples/custom_op.py
```

`native_function.py` compiles a typed RNN with a loop over a symbolic length and a branch on tensor
data from its source (docs/FRONTEND.md), runs it at three batch sizes and lengths, and shows a shape
bug caught when the decorator runs. `custom_op.py` registers a typed RMSNorm as a PyTorch custom op
with `truestar.torch_op`, checks forward and gradients against PyTorch float64 and runs
`torch.library.opcheck` (docs/CUSTOM_OPS.md). Pass `cuda` to run it on the GPU.

## Next

* Agents and JSON: [AGENTS.md](../AGENTS.md), [AGENT_INTERFACE.md](AGENT_INTERFACE.md).
* Every doc: [index.md](index.md).
