# Round once elementwise math

Author: Sharven Rane
Date: 2026-11-24

Round once math is an opt in mode in which `exp`, `log`, `tanh`, `erf`, `rsqrt` and `sigmoid` give
the same bits on every device, and those bits are the correctly rounded result. It is research item
1.7 of the November research list (idea from arXiv 2510.09180: correct rounding for the basic
operations is what makes results bit identical across hardware; no code or coefficients were taken
from it or from any math library).

```python
from truestar import kernel as K
from truestar.kernel import roundonce as R

y = R.tanh(x)                        # torch tensor in (float16, bfloat16, float32), same dtype out
y = R.erf(x, dtype=torch.float16)    # round once to another output format
fn = K.compile(kernel, "cuda", math="round_once")        # Triton; "cpu" (numba) and "metal" work the same
fn = graph.compile(outputs, "cuda", math="round_once")    # K.Graph passes it to every kernel
```

## Why

Native transcendental functions are not correctly rounded and are not the same across devices.
Measured over all 2^32 float32 inputs on this machine (torch 2.11, Windows, AMD Zen 5 CPU, RTX 5070
Ti), torch's CPU and CUDA float32 results differ in these many inputs:

| function | CPU vs CUDA bits differ | CPU max ulp | CUDA max ulp | CPU not correctly rounded | CUDA not correctly rounded |
|---|---|---|---|---|---|
| exp | 159,926,624 | 0.568 | 1.931 | 2,990,185 | 159,912,805 |
| log | 70,101,668 | 0.974 | 0.864 | 12,460,319 | 72,516,687 |
| tanh | 39,591,514 | 0.513 | 1.815 | 164,766 | 39,501,004 |
| erf | 351,232,384 | 0.784 | 1.037 | 120,059,490 | 370,931,198 |
| rsqrt | 700,499,800 | 1.490 | 1.514 | 556,013,448 | 463,314,104 |
| sigmoid | 53,794,175 | 2,097,151 (*) | 2,097,151 (*) | 170,335,222 | 187,336,683 |

(*) sigmoid of x below about -87.3: both return 0 where the true result is a float32 subnormal up
to 2^21 subnormal steps away. Away from that range the error is ordinary.

How it was measured (`scripts/roundonce_measure.py native`): every float32 bit pattern in chunks of
2^24, torch's own function on the CPU (22 threads) and on CUDA; bits compared with any NaN equal to
any NaN. The ulp error is measured against round once's float64 value without its exception table
(below), whose relative error is at most 9.6 times 2^-53 in the samples, so the ulp numbers are good
to about 1e-8 ulp. "Not correctly rounded" counts inputs whose result differs from the round once
result, which is the correctly rounded one (the exhaustive sweep below).

In float16 and bfloat16 (all 65,536 inputs each, against mpmath) native torch is mostly right, with
two exceptions: torch's CPU `erf` in half precision misses the correctly rounded result for 6,586
float16 and 28,690 bfloat16 inputs (CUDA: 0), and CPU and CUDA disagree on those same inputs. The
other native counts are 0 to 8 inputs per function and format.

## How it works

Each function is a fixed sequence of IEEE 754 float64 operations that are correctly rounded by the
standard (add, subtract, multiply, divide, square root, compare, select) or exact (floor, negate,
absolute value, sign bit, scaling by a power of two built from its bits, splitting a value into
mantissa and exponent). No fused multiply add anywhere. The float64 result is then rounded once to
the output format. Any device that runs the same sequence with IEEE float64 gets the same bits.

The algorithms, written for Truestar (constants derived in the module from exact rational
arithmetic with `fractions` and `decimal`: ln 2 by Python's decimal logarithm, pi by Machin's
formula, series coefficients as exact fractions rounded once to float64):

* **exp**: clamp to [-800, 800] (both limits are beyond float64's range, so they are exact), k =
  floor(x / ln 2 + 1/2), r = (x - k ln2_hi) - k ln2_lo with ln 2 split so that k ln2_hi is exact
  (32 significant bits), Taylor series of degree 13 for e^r on |r| <= ln2 / 2 (remainder below
  2^-57), then two exact power of two scalings 2^floor(k/2) and 2^(k - floor(k/2)).
* **log**: x = m 2^e with m in [1, 2) from the bits, m halved above sqrt 2, s = (m - 1)/(m + 1)
  (both exact for inputs of at most 24 bits), log m = 2 atanh(s) by its series to s^25 (|s| <=
  0.1716), result e ln2_hi + (e ln2_lo + log m).
* **tanh**: on a = |x| clamped at 20: below 1/2, E/(E + 2) with E = expm1(2a) by its Taylor series
  (20 terms); above, (1 - t)/(1 + t) with t = exp(-2a); the sign is copied from the sign bit, so
  tanh(-0) = -0.
* **erf**: on a = |x| below 4, erf(a) = 2/sqrt(pi) a e^(-a^2) sum_n (2a^2)^n / (2n+1)!!, a series
  with only positive terms (no cancellation), nested to 62 levels (tail below 2^-61 of the sum at
  a = 4); a^2 is exact for 24 bit inputs. For |x| >= 4 the result is +-1: erfc(4) = 1.5e-8 is below
  half an ulp of 1 in float32 (2^-25 = 3.0e-8), so every supported format rounds erf there to +-1.
  This is an analytic argument, not a sample.
* **sigmoid**: 1/(1 + exp(-x)). Near 0, sigmoid(x) = 1/2 + x/4 - x^3/48 + ...: for a float32 x the
  part A = 1/2 + x/4 is exact in float64 and is often exactly a float32 rounding midpoint, while the
  true value sits x^3/48 away, far below float64's resolution (the first sweep found 66 such
  inputs). For 2^-26 <= |x| < 2^-14 the result is A + c with c = -x^3/48, and when that sum rounds
  back to A it is moved one float64 ulp towards c, which puts it on the true value's side of every
  midpoint near A.
* **rsqrt**: 1 / sqrt(x), two correctly rounded operations.
* **sqrt** in round once kernels is the IEEE float32 square root (`tl.sqrt_rn`; on the CPU the
  float64 square root of the float32 value rounded to float32, which is the same thing). `neg`,
  `abs` and `not` are exact anyway.

Six float32 inputs have a float64 value closer to a float32 rounding midpoint than the algorithm's
error: five for log (0.0117943827, 9.4726362, 58037908, 1.2783784e23, 5.4983061e28) and one for
sigmoid (-0.0011178852). The exhaustive sweep found them and mpmath settled them; they are stored
with their exact answer in `roundonce.EXCEPTIONS` and selected by exact comparison. None of them is
a float16 or bfloat16 value.

Every NaN result is the format's canonical quiet NaN (0x7FC00000 in float32, 0x7E00 in float16,
0x7FC0 in bfloat16). Without that, NaN bits did differ: the cast of a NaN from float64 to float16
kept payload bits on one device and not the other (all 2,046 float16 NaN inputs), and LLVM may
replace one NaN by another in a float select, so the Triton path sets the canonical NaN on the
integer bits.

### One description, three evaluators

The algorithm is written once against a small set of primitives (`where`, `eq`, `ne`, `div`,
`sqrt`, `floor`, `abs`, `signbit`, `pow2`, `frexp1`) and Python arithmetic. It runs as:

* the torch evaluator (`R.exp`, `R.value64`), on the CPU or a GPU, one tensor op per step;
* Triton source: `@triton.jit` helpers `_ro_exp` and so on, emitted into the kernel module by
  `K.compile(..., math="round_once")`. Those modules set `TS_FP_FUSION = False` and
  `triton_backend.prepare` launches them with `enable_fp_fusion=False`; with fusion on, a probe of
  `a*x + b` style float64 code differed from torch in 97 of 2^20 inputs, with it off in 0;
* numba source for the 'cpu' kernel backend (`fastmath=False`).
* Metal source for the 'metal' backend. Apple GPUs have no float64, so each float64 operation is
  a call into `kernel/softf64_metal.py`: IEEE binary64 on the 64 bit integer holding its bits (add,
  subtract, multiply by 32 bit limbs, divide by long division, square root digit by digit on a 128
  bit radicand, each rounded to nearest even with subnormal results; compare, floor, the exact
  float32 conversions). In round once mode Metal also routes `sqrt` through it, since Metal's
  precise float32 sqrt is not documented as correctly rounded. Nothing here depends on the GPU's
  float rounding or its subnormal flush (the M2 flushes float32 subnormals in arithmetic).

The kernel IR's float32 arithmetic around these functions is unchanged: a round once function takes
the kernel's float32 value, computes in float64 and returns the correctly rounded float32. A kernel
whose output is float16 or bfloat16 then rounds that float32 once more when it stores (see "What
does not work").

### Pitfalls found on the way

* **torch's CPU float64 sqrt is not correctly rounded** on this build (torch 2.11, Windows, AVX512):
  931 of 65,536 random inputs in [0, 1) differ from Python's `math.sqrt` (1.4 percent); CUDA and
  NumPy: 0. The CPU evaluator takes its square root from NumPy. `tests/test_roundonce.py` checks it.
* **torch on CUDA divides by a Python scalar as a multiply by its reciprocal**: `a / 3.0` differed
  from the correctly rounded quotient in 350,825 of 2^20 inputs. Every division here is tensor by
  tensor.
* **numba keeps `float(np.float32(x))` as float32**, so a following `math.sqrt` ran in float32:
  8,357 of the float16 inputs were off by one ulp in rsqrt until the argument became
  `np.float64(np.float32(x))`.
* **The kernel layer stores -0 as +0**: its broadcast adds a block of zeros (`v + tl.zeros`), which
  turns -0 into +0 in both math modes. The round once functions themselves return -0 for tanh(-0)
  and erf(-0) (the torch evaluator keeps it); kernel outputs do not. This is existing kernel layer
  behavior, not fixed here.

## What is proven and what is sampled

| claim | how | result |
|---|---|---|
| float16, all 65,536 inputs, each function: correctly rounded | exhaustive, against mpmath at 200 bits (2000 near a midpoint), rounded by integer arithmetic | 0 mismatches for all six functions, on the CPU and on CUDA |
| bfloat16, all 65,536 inputs, each function: correctly rounded | same | 0 mismatches for all six |
| float16 and bfloat16: CPU and CUDA give the same bits | exhaustive, raw bits | 0 differences |
| float32: float64 error of the algorithm | stratified sample: 160 random mantissas in each of the 255 exponents of both signs, plus 1,201 neighbours of each threshold (overflow, underflow, subnormal, branch points, 0, 1, sqrt 2), about 100,000 inputs per function, against mpmath | max relative error in units of 2^-53: exp 1.51, log 2.72, tanh 3.19, erf 9.64 (|x| < 4), rsqrt 1.49, sigmoid 2.00 |
| float32, all 2^32 inputs: correctly rounded | exhaustive sweep on CUDA, conditional on the error bound B = 16 times the sampled maximum (at least 2^-50): every input whose float64 value lies within B of a float32 rounding midpoint is flagged and checked against mpmath | 0 mismatches for all six after the six exceptions (table below) |
| float32: CPU and CUDA give the same bits | 2^28 random bit patterns per function, raw bits | 0 differences |
| float32: kernels give the torch evaluator's bits | Triton (CUDA) and numba (CPU) kernels with a float32 output, against the torch evaluator: all float16 inputs, all bfloat16 inputs, 2^22 random float32 inputs | 0 differences for all six, Triton vs numba also 0 |
| Apple M2 GPU (Metal, soft float64) gives the CPU evaluator's bits | the soft float64 operations against NumPy float64 on 200,484 operand pairs (edge values, random bit patterns, random normals); the kernels against `R.apply` on all float16 and bfloat16 inputs; then an exhaustive sweep: every one of the 2^32 float32 bit patterns per function through the Metal kernel against the torch CPU evaluator on the same Mac (`ro_sweep.py`, <research>/mac_real/scripts/) | 0 differences in every operation and for all six functions over all 2^32 inputs (NaN bits included, -0 and +0 equal); the round once sqrt equals torch's correctly rounded sqrt on 2^20 inputs |
| two machines give the same bits | `scripts/roundonce_hashes.py` on this desktop (Windows, Zen 5, RTX 5070 Ti, torch 2.11 cu128) and on a Modal L4 (Linux, its x86 CPU, torch 2.11 cu130): SHA256 of the torch CPU, torch CUDA and Triton kernel outputs over all float16, all bfloat16 and 2^22 seeded float32 inputs | 72 of 72 hashes equal (docs/data/roundonce_hashes_*.json) |

The float32 sweep in numbers (`scripts/roundonce_measure.py sweep`; "live" excludes NaN inputs,
results that are 0 or below 2^-1000 in float64, and erf at |x| >= 4):

| function | B in units of 2^-53 | live inputs | flagged | mismatches after the exceptions | GPU seconds |
|---|---|---|---|---|---|
| exp | 24.1 | 2,287,909,764 | 35 | 0 | 66.7 |
| log | 43.5 | 2,139,095,038 | 247 | 0 (5 before) | 41.4 |
| tanh | 51.1 | 4,278,190,080 | 36 | 0 | 29.9 |
| erf | 154.3 | 2,164,260,862 | 1,218 | 0 | 40.3 |
| rsqrt | 23.9 | 2,139,095,039 | 256 | 0 | 10.5 |
| sigmoid | 32.0 | 3,282,913,645 | 574 | 0 (66 before the small x path, 1 before the exception) | 24.1 |

So the float32 claim is: correctly rounded for every float32 input, provided the float64 error of
the algorithm stays below B at every input. That bound is measured on a sample, with a margin of 16,
not proven. A float64 error bound proven by hand per function would turn the sweep into a proof; it
is not done.

## Speed

`scripts/roundonce_measure.py speed`, RTX 5070 Ti, float32 in and out, billions of elements per
second (median of 5 rounds of 20 launches; this consumer GPU runs float64 at 1/64 of float32):

| function | n | torch native | Truestar default | Truestar round once | torch evaluator |
|---|---|---|---|---|---|
| exp | 2^24 | 93.8 | 92.5 | 7.47 | 0.52 |
| exp | 2^26 | 94.7 | 93.3 | 7.77 | |
| log | 2^26 | 94.0 | 92.4 | 7.53 | |
| tanh | 2^26 | 88.2 | 89.8 | 2.83 | |
| erf | 2^26 | 82.2 | 82.6 | 1.06 | |
| rsqrt | 2^26 | 95.0 | 87.0 | 14.98 | |
| sigmoid | 2^26 | 94.9 | 93.8 | 4.82 | |

Round once kernels are 6 (rsqrt) to 77 (erf) times slower than native here: native elementwise ops
are memory bound, the float64 sequences are compute bound at the 1/64 rate. On a GPU with full rate
float64 (data center parts) the gap is far smaller; not measured. The torch evaluator, one kernel
per step, is 10 to 70 times slower again and is meant for checking, not for speed. All numbers per
size are in docs/data/roundonce_results.json.

## What does not work, or was not done

* **Stores in float16 or bfloat16 round twice.** A kernel computes in float32, so a round once
  function's correctly rounded float32 is rounded again when stored as float16. Over all float16
  inputs that differs from the directly rounded result for exp 2, log 1, erf 2 and sigmoid 7 inputs
  (tanh and rsqrt 0; bfloat16 0 for all). The bits are still the same on every device; they are not
  always the correctly rounded float16. The torch evaluator rounds float64 directly to the output
  format and has no such case.
* **Refused, never silently native** (the codes are in docs/AGENT_INTERFACE.md): a backend other
  than Triton, numba and Metal (TS1089), an unknown `math` value (TS1090), float64 buffers or outputs and non float dtypes (TS1091), ops
  without a round once form such as cos and sin (TS1092), and the attention loop shape, whose online
  softmax computes exp itself (TS1093). Tile shaped softmax kernels (`K.softmax_weight` without a
  following contraction) do work, through the IR's own exp nodes.
* **ROCm and other GPUs**: 'rocm' compiles the same Triton source, but whether its launches
  accept and honor `enable_fp_fusion=False` was not checked, and nothing here was run on AMD
  hardware. The cross device evidence is two NVIDIA GPUs of different generations (Blackwell
  desktop, Ada L4), an Apple M2 GPU (exhaustive float32 sweep, above) and three CPUs on three
  operating systems.
* **Only these six functions.** cos, sin, pow, and the float32 arithmetic around them are not
  covered. Bitwise results for a whole model also need fixed reduction orders (research item 1.4);
  this item covers the elementwise functions only. Triton's float32 division is not checked here.
* **float32 correct rounding rests on a sampled error bound** (above), not on a hand proof.
* **No exhaustive CPU vs CUDA comparison of the round once float32 path**: 2^28 random inputs per
  function, not all 2^32 (the torch evaluator on the CPU would need about 30 minutes for all).

## Files

* `src/truestar/kernel/roundonce.py`: constants, algorithms, the three evaluators, refusals.
* `src/truestar/kernel/triton_backend.py`, `cpu_backend.py`, `compiler.py`, `graph.py`: the `math`
  option and code generation.
* `tests/test_roundonce.py`, `tests/roundonce_ref.py` (the mpmath reference).
* `scripts/roundonce_measure.py` (every number above), `scripts/roundonce_hashes.py` (machine
  comparison), docs/data/roundonce_results.json and docs/data/roundonce_hashes_*.json.
