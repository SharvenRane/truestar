# Phone NPUs: INT8 agreement and per layer blame

Measured 2026-11-24 on hosted phones through Qualcomm AI Hub (qai-hub 0.59, torch 2.11, TFLite
runtime). Code: `src/truestar/remote/qai_hub_quant.py` (studies, job cache),
`src/truestar/remote/agreement.py` (counting and blame, no network), `src/truestar/remote/synthetic.py`
(model and data). Rerun everything with `PYTHONPATH=src python scripts/npu_study.py --out study.json`.

## Why

arXiv 2609.16085 reports that one INT8 model with fixed scales agrees on only 958 to 965 of 1000
predictions across CPUs with different integer kernels, and on 939 of 1000 on a vendor NPU, while
aggregate accuracy looks unchanged. Only a per input comparison shows it (pain ledger P2). This
study repeats that measurement on real Snapdragon phones and then asks where in the network the
NPU leaves the intended INT8 arithmetic.

## Model and data (no downloads)

`truestar.remote.synthetic.build_task()` with defaults (seed 0):

* 20 classes. Each class prototype is a seeded 3x6x6 random tensor upsampled bilinearly to
  128x128. A sample is its prototype plus smooth noise (8x8 upsampled) with a per sample strength
  drawn from 0.25 to 1.0, plus pixel noise with standard deviation 0.3. Some inputs are easy, some
  sit near a boundary.
* torchvision ResNet18 with seeded conv weights and seeded batch norm affine parameters (gamma in
  0.5 to 1.5, beta in minus 0.1 to 0.1). Every batch norm's running statistics are set from a 64
  sample calibration batch (one train mode pass with momentum None), so activations are unit scale
  everywhere, as in a trained network.
* The classifier is a nearest class mean head on the pooled 512 dim features of a separate training
  draw (16 per class), scaled so logits have a standard deviation of 2.5 (range about minus 20 to 11).
* 256 evaluation inputs, 64 calibration inputs. float64 accuracy on the evaluation batch: 80.08%
  (205/256), all 20 classes predicted.

Weights and data are bit identical across runs, which keeps the job cache valid.

## Pipeline and job count

INT8: TorchScript to ONNX compile, AI Hub quantize job (w8a8, the 64 calibration inputs), one
TFLite compile, then one inference job per (device, compute unit) carrying all 256 inputs. The same
TFLite file ran on both phones. Floats: one TFLite compile, NPU and CPU inference on the S24.

Every compile, quantize, dataset upload and inference is cached in `~/.cache/truestar/qai_hub/`
(override with `TRUESTAR_CACHE`), keyed by a hash of model structure plus every parameter and
buffer, input specs, device and options; inference outputs are stored as .npz. The whole study used
18 AI Hub jobs and 3 dataset uploads. A second full run submitted 0 jobs and took 25 s.

| step | job ids |
|---|---|
| ONNX compile, quantize, INT8 TFLite compile | jgj6lx41p, jgzq7ek4g, jp46wokq5 |
| INT8 inference S24 NPU, S24 CPU, S25 NPU, S25 CPU | jprmyq2vg, jpe0v1r0g, j571dw9l5, j5qx2k1np |
| float TFLite compile, S24 NPU, S24 CPU | jgzq7kozg, jgdyvlekg, jgo7vqox5 |
| float taps: compile, NPU, CPU | jp46w0615, jpx8128l5, jgkwlxevg |
| INT8 taps: ONNX compile, quantize, TFLite compile, NPU, CPU | jpvy1k1mp, jgzqxdx6g, j571jrvr5, jp0810ee5, j5w0d6nmp |

Devices: "Samsung Galaxy S24 (Family)" (Snapdragon 8 Gen 3) and "Samsung Galaxy S25 (Family)"
(Snapdragon 8 Elite). Host runs use ONNX Runtime 1.29 on the x86 PC on the same QDQ model: once with
every Quantize/Dequantize pair kept (simulated INT8, the intended arithmetic) and once fully
optimized (integer kernels).

## Result 1: INT8 agreement, 256 inputs

Accuracy on the batch (top 1 vs label):

| run | accuracy |
|---|---|
| float64 reference | 80.08% (205) |
| float S24 NPU, float S24 CPU | 80.08% (205) each |
| INT8 S24 NPU | 79.30% (203) |
| INT8 S24 CPU, S25 NPU, S25 CPU | 79.69% (204) each |
| INT8 host simulated | 80.08% (205) |
| INT8 host integer kernels | 80.47% (206) |

Every INT8 run is within 2 inputs of the float accuracy. Per input agreement tells a different story:

| pair | agree / 256 | differ | near ties | exact ties | clear |
|---|---|---|---|---|---|
| S24 CPU vs S25 CPU (INT8) | 256 | 0 | 0 | 0 | 0 |
| host simulated vs host kernels | 252 | 4 | 4 | 4 | 0 |
| S24 CPU vs host simulated | 251 | 5 | 5 | 3 | 0 |
| S24 NPU vs S25 NPU | 245 | 11 | 10 | 6 | 1 |
| S24 NPU vs S24 CPU | 242 | 14 | 9 | 5 | 5 |
| S25 NPU vs S25 CPU | 242 | 14 | 10 | 4 | 4 |
| S24 NPU vs host simulated | 237 | 19 | 13 | 8 | 6 |
| float64 reference vs INT8 S24 NPU | 242 | 14 | 5 | 4 | 9 |
| float64 reference vs INT8 S24 CPU | 240 | 16 | 3 | 2 | 13 |
| float S24 NPU vs float S24 CPU | 256 | 0 | 0 | 0 | 0 |
| float64 reference vs float S24 NPU | 256 | 0 | 0 | 0 | 0 |

How ties are counted: when two runs pick different classes, the swing is how far the logit gap
between those two classes moved. It is a near tie if the swing is within both runs' rounding
tolerance (2 output steps per INT8 run; the INT8 output step is 0.0925, found from the data). An
exact tie means one run scored the two classes identically and argmax took the lower index. The
INT8 output step is coarse relative to the logits, so exact ties are common.

What this says:

* **Finding: the NPU does not compute the same INT8 model as the CPU.** On the same phone, with
  the same TFLite file and the same scales, 14 of 256 predictions (5.5%) differ between NPU and CPU on
  both phones, 5 and 4 of them clear flips. That is 945 per 1000, in line with the paper's 939 for a
  vendor NPU. Accuracy moves by at most 0.4 points, so an accuracy check would pass it.
* The two NPU generations also disagree with each other on 11 inputs, almost all near ties.
* The phone CPUs are bit for bit consistent across generations (256/256, identical logits) and are
  within 5 near ties of the host simulation. Host integer kernels vs host simulation: 4 exact ties.
  The CPU path is close to the intended arithmetic; ties are where it differs.
* float TFLite on the NPU (float16) agrees with float64 on all 256 inputs. The flips come from the
  INT8 path, not from the NPU as such.
* INT8 vs float64 flips are mostly clear (9 to 13 per device): that is quantization error itself,
  and expected.

Noise vs finding: with 256 inputs, 14 disagreements has a rough 95% interval of 8 to 23, so the
NPU vs CPU rate (3% to 9%) is clearly above the CPU vs CPU rate (0) and the host kernels vs
simulation rate (1.6%). The exact counts per pair are not stable at this sample size; the ordering
is.

## Result 2: per layer blame on the S24 NPU

The model is wrapped (`TapWrapper`) so 11 block level module outputs are extra outputs: `relu`
(after conv1 and bn1), `maxpool`, the 8 BasicBlocks `layer1.0` to `layer4.1`, `avgpool`, plus the
logits. Taps are chosen by `default_taps`: children of top level containers and top level modules
that are not a bare conv or norm, because tapping a conv before its batch norm would stop the
compiler from folding them and change the model under test. At most `MAX_TAPS = 12` extra outputs:
each is one more tensor to download per input and one more point where the compiler must
materialize a tensor. 16 inputs per run (the taps are about 0.4 MB per input).

Thresholds: float runs use the float16 floor (5e-3 relative L2). INT8 runs use one LSB rms: the
relative error of being off by one quantization step on every element (step / rms of the control
tensor, step found from the data). Taps come from one full run, so a tap's error includes what it
inherited; the first tap over threshold is where the divergence becomes larger than rounding.

**Float model, NPU vs CPU control:** no tap over threshold. The error grows smoothly from 4.0e-4 at
`relu` to 4.4e-3 at `layer4.1` and is 1.4e-3 at the logits, which is float16 accumulation. The CPU
control matches float64 to 4e-6 at every tap. Clean, but `layer4.1` sits at 0.88 of the float16
floor, so a deeper network could cross it without any bug.

**INT8 model, NPU vs CPU control:** first tap over threshold is `layer2.0`. The more telling view is
each unit against the host simulation of the same tapped QDQ model (no extra jobs):

| tap | NPU elements that differ | NPU max | CPU elements that differ | CPU max |
|---|---|---|---|---|
| relu (conv1, bn1) | 4.44% (all 1 LSB but a few) | 3 LSB | 0.00% | 1 LSB |
| maxpool | 6.76% | 2 LSB | 0.00% | 1 LSB |
| layer1.0 | 22.6% | 4 LSB | 0.05% | 2 LSB |
| layer1.1 | 35.1% | 6 LSB | 0.39% | 3 LSB |

Relative error vs the host simulation, first crossing: NPU at `layer2.0` (5.96e-2 vs threshold
5.1e-2), CPU at `layer4.0` (4.7e-2 vs 3.3e-2). ONNX Runtime's own integer kernels match the S24 CPU
element counts exactly at these taps.

What this says: the NPU leaves the intended INT8 arithmetic at the very first conv, where 4.4% of
elements land one step away. The likely cause is a different rounding or requantization rule in
the NPU kernels; that is an inference from the 1 LSB pattern, not something AI Hub exposes.
Update (docs/INTREF.md, `truestar intref npu`, same saved jobs, no new ones): the differences go
both ways inside every channel, which no requantization rounding rule produces, so it is not a
rounding convention; quantizing the float input in float16 reproduces 3.9% to 4.3% with the same
growth by output level. Those
one step differences compound through the residual blocks until they decide near tie predictions.
No single layer is broken; `layer2.0` is where the accumulated drift first exceeds one step rms, so
it is a threshold crossing, not the culprit. The culprit is the per element rounding that starts at
`relu`.

## Limits

* One model family, one seed, 256 inputs. Counts are within the noise band given above.
* TFLite only. QNN context binaries or ONNX Runtime QNN would compile differently and are untested.
* The tie tolerance for INT8 uses the output step found from the data; if a runtime returned float
  logits from an INT8 model, the step would be missing and every disagreement would count as clear.
* Tapping block outputs forces those tensors to exist, so the tapped and untapped compiled graphs
  are not identical. Whether the taps change the NPU's rounding was not measured.
* The per layer numbers include inherited error: this does not replay one layer alone with
  reference inputs the way `truestar check` does on local backends, since that would cost one
  compile per layer.
* AI Hub warns that compiling from TorchScript is deprecated; the code still uses
  `torch.jit.trace` like the float backend. It will need to move to `torch.export` when that path
  is removed.
