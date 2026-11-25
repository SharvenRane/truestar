# Quantize safely

Script: [examples/cookbook/quantize_safely.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/quantize_safely.py)

Quantizing raises two separate questions, and `truestar.quant` answers them separately
([QUANT.md](../QUANT.md)):

* **(b) the format.** How far do int8 weights, or fp8 weights and activations, move each layer and
  the predictions? Computed in float64 with the exact codes and scales, so no kernel is involved.
* **(a) the kernel.** Does the matmul that runs the quantized layer compute what those codes and
  scales say, within a bound derived from float64?

```python
from truestar.quant import formats as F
from truestar.quant.model import decisions
from truestar.quant.verify import check_matmul, layer_format_loss

qw = F.quantize_weight(w, "int8")                 # or "fp8_e4m3:channel", "int4:g128", "nvfp4", "mxfp4"
layer_format_loss(x_in, w, qw)                    # (b) one layer, float64
decisions(ref_logits, quantized_logits, tie)      # (b) predictions changed, near ties apart
chk = check_matmul(x, qw, my_kernel(x, qw))       # (a) your kernel's output against float64
chk.ok, chk.ratio, chk.code                       # TS460 when the error is over the bound
```

The kernels in the script are short CPU functions: a correct int8 kernel, the same kernel with
its per channel scale stored in bfloat16 (a common bug), and an fp8 e4m3 W8A8 kernel that
quantizes its activations per row. Output:

```
(b) the format alone, in float64
  int8               layer error 0: 3.9e-03, 2: 3.9e-03, 4: 3.9e-03; output 5.9e-03; top 1 changed 1 of 512 (0 near ties, 1 real)
  fp8_e4m3:channel   layer error 0: 3.6e-02, 2: 3.6e-02, 4: 3.5e-02; output 5.3e-02; top 1 changed 19 of 512 (0 near ties, 19 real)

(a) the kernel, against the float64 reference of the same codes and scales
  int8 kernel                      ok True  error / bound     0.01
  int8 kernel, scale in bfloat16   ok False error / bound   100.13  TS460
  fp8 e4m3 W8A8 kernel             ok True  error / bound     0.00  activation codes off the format: 0
```

Reading it: the format costs int8 one prediction in 512 and fp8 W8A8 nineteen. Those are real
format loss, not kernel bugs and not near ties, and whether they are acceptable is your call. The
bfloat16 scale is caught at 100 times its bound, while the correct kernels sit far below theirs.
For fp8 the check also compares the activation codes and scales the kernel produced with the
format's exact rounding rule (`quantizer_mismatches`, TS461).

A near tie here is a row whose float64 top two logits are within 4 times the plain float32
model's own rounding on that row, the rule `truestar.quant.model.compare` uses.

On CUDA, `truestar.quant.kernels.quantized_matmul` runs Truestar's own Triton kernels for every
format, and `verify.verify_kernel(x, qw)` runs and checks them in one call, with a diagnosis on
failure.

## Limits

* The CUDA kernels need float16 or bfloat16 activations and refuse CPU tensors; on the CPU,
  check your own kernel with `check_matmul` as here.
* `model.quantize_model` and `model.compare` take PyTorch modules (every `nn.Linear` except the
  head); for a native model, quantize the parameter tree as the script does.
