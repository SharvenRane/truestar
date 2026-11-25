# TPU and other remote accelerators: a fixed battery, judged from files (`truestar conform tpu`, row P44)

Author: Sharven Rane. First written 2026-11-24.

A TPU computes a float32 `jnp.matmul` with one bfloat16 pass per product unless the caller asks for
`precision=HIGHEST` (JAX's DEFAULT precision). The result looks like float32 and is not. Nobody with
a model on a TPU has the TPU next to a float64 reference, so the check has to work from files.

Code: `src/truestar/conform/tpu_battery.py` (the battery, self contained, numpy at import) and
`src/truestar/conform/tpu.py` (the judge). Tests: `tests/test_conform_tpu.py`. Codes TS900 to TS906.

## Use

On the machine with the accelerator (a Kaggle TPU kernel is one file: this module plus an entry):

```
python tpu_battery.py OUT_DIR                  # JAX on its default device, then on its CPU
python tpu_battery.py OUT_DIR --runner torch   # PyTorch CPU, CUDA, CUDA with TF32
```

Anywhere, no accelerator needed:

```
truestar conform tpu OUT_DIR                          # every backend in the directory
truestar conform tpu OUT_DIR --runtimes jax_tpu --format agent
```

Exit codes: 0 every case within budget, 1 a finding (divergent, reduced precision, failed), 2 the
directory cannot be read (TS900).

## What is checked

The battery has 18 cases: float32 matmul at JAX precision default, high and highest; bfloat16
matmul with bfloat16 output and with float32 accumulation; softmax; layer norm; an MLP (float32 at
default and highest, bfloat16); one post norm transformer encoder layer (default and highest);
sums over 2^24 and 2^22 floats (uniform and over 40 binades); exp, tanh, log, rsqrt over 2^20 points.

Inputs come from a splitmix64 counter hash in uint64 arithmetic and exact float conversions (24 bit
integers over 2^23 for float32, 7 bit integers over 2^6 for bfloat16, power of two scales), so they
are the same bytes on every host. The run records an input hash per case; the judge regenerates the
inputs, refuses a case whose hash differs (TS904), and computes the float64 reference itself.
Each case is called 5 times from one compile and once from a second compile; differing bits are
TS903.

Budgets:

| case kind | budget | below it |
|---|---|---|
| float32 matmul, sum | effective bits (-log2 of worst error over abs products, or over sum abs) at least 24 - log2(n)/2 - 3 (probabilistic bound lambda sqrt(n) u, lambda 8, Higham and Mary 2019) | TS902 if rel L2 is still within 3e-2, else TS901 |
| other float32 | rel L2 at most 1e-5; elementwise functions also 4 ulp | TS902 / TS901; ulp only: TS905 |
| bfloat16 | rel L2 at most 3e-2 | TS901 |
| any | no element off by more than rms(ref)/16, no non finite value | TS901 |

## Measurements

Local runs (JAX CPU, PyTorch CPU, RTX 5070 Ti with TF32 off and on) and the TPU run status are in
`<research>/tpu/README.md`. Locally: TF32 matmuls are flagged TS902 at 14.3
effective bits against a budget of 16 (float32 paths give 23 to 24); XLA CPU's tanh is flagged
TS905 at 4.5 ulp; everything else is within budget and bitwise stable.

## Limits

* The TPU run itself is queued on Kaggle (batch TPU sessions wait 10 to 24 hours); no TPU number is
  in this document yet.
* The judge covers the fixed battery only, not a user's model: it says how a device computes these
  ops, which is what a model on that device inherits. A user model on a TPU is not checked.
* torch_xla runs three cases only, when the image has it.
