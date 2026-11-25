# Verify a model against float64

Script: [examples/cookbook/verify_float64.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/verify_float64.py)

A native model compiled for the CPU in float32 is checked on its first call against Truestar's
float64 reference interpreter before the result is returned. There is nothing to call: the check
is part of the executor.

```python
from truestar import Tensor, nn
from truestar.compiler import CompiledExecutor

net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 10))
ex = CompiledExecutor("cpu", dtype="float32")         # "cuda" on a GPU; float16 and bfloat16 too

@truestar.native(executor=ex)
def forward(p: net.spec, x: Tensor["B", 32]) -> Tensor["B", 10]:
    return net(p, x)

y = forward(net.init(seed=0), x)
ex.last.report, ex.last.verified, ex.last.max_error
```

Output:

```
compile report: forward on cpu, float32 (compiled in 0.36 s)
  Truestar kernels: 4 covering 6 IR ops (4 loops)
  library kernels (torch): 0
  fallbacks (float64 reference implementation on the CPU): 0
  ...
  steps:
    kernel   forward:%linear [loops] linear
    kernel   forward:%layer_norm [loops] gelu + layer_norm
    kernel   forward:%silu [loops] linear + silu
    kernel   forward:%linear_2 [loops] linear

verified on the first call: True; end to end error vs float64 7.5e-08
by hand: relative L2 error of the float32 result against eager float64: 6.3e-08
batch 5 ran on the same compiled program (compiles: 1), output (5, 10)
```

## What is checked

Three levels, all on by default (`TRUESTAR_VERIFY=first`, [COMPILER.md](../COMPILER.md)):

1. every kernel against the float64 interpreter of its own kernel IR, on its first call per input
   shape signature;
2. every fused region and library step against the float64 reference of the IR ops it covers,
   fed the inputs that step really received, so each region's own error is measured;
3. the whole program against `ir.run`, which covers layout steps and the wiring between regions.

A mismatch raises before the result is returned, with the op, the source line and the error
against its bound. An output that is ill conditioned in float32 (a correct float32 evaluation is
itself far from float64) is refused by default and says so (TS782).

For a PyTorch model file, the same question across every backend on the machine is
`truestar check model.py` ([GETTING_STARTED.md](../GETTING_STARTED.md)).

## Limits

* Verification runs on the first call per shape signature (`TRUESTAR_VERIFY=first`); later calls
  at that signature are not rechecked unless you set `TRUESTAR_VERIFY=all` (or `CompiledExecutor(..., verify="all")`).
* The float64 reference is the IR's meaning. A model that casts explicitly to float32 is rounded at
  those casts in the reference too.
