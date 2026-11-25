# Compare two chips

Script: [examples/cookbook/compare_chips.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/compare_chips.py)

A simulated device (`sim:*`) is a profile of how a chip does float arithmetic: storage dtype,
accumulator, special values, op formulations. Every rule carries its source and a label
(documented, measured, hypothesis or guess; `truestar backends --sims` prints them). It runs on the
CPU, so you can ask what a model gives on a chip before anyone owns one.

```python
from truestar import nn
from truestar.devsim import simulate

ref = net(p64, x.double())                    # float64 reference
with simulate("sim:ane"):                     # Apple Neural Engine through Core ML
    ane = net(p32, x)
with simulate("sim:mali-fp16"):               # Arm Mali through ncnn's Vulkan defaults
    mali = net(p32, x)
```

The layers of a native model run eagerly through PyTorch's ops outside a compiled function, and
`simulate` applies the profile to every one of them. Output:

```
device             rel error vs float64  top 1 changed  profile
sim:ieee-f16                    6.6e-04         1 of 512  textbook float16
sim:ane                         9.2e-04         0 of 512  Apple Neural Engine through Core ML
sim:mali-fp16                   4.6e-03         1 of 512  Arm Mali GPU through ncnn Vulkan defaults

sim:ane against sim:mali-fp16: relative difference 4.6e-03, 1 of 512 predictions differ between the two chips
```

`sim:ieee-f16` is what correct float16 arithmetic gives (float16 storage, a float32 class
accumulator), so it separates the cost of float16 from the cost of the device: here the Mali
profile's float16 accumulator is about 7 times further from float64 than textbook float16.

For a PyTorch model file, the same profiles are backends of `truestar check`, with module blame,
op drilldown and repro scripts, and `--sweep` finds the size where a model starts to break:

```
truestar check examples/device_softplus.py --backends cpu,sim:ieee-f16,sim:ane
truestar check examples/size_cliff.py --backends cpu,sim:ieee-f16,sim:openvino-gpu-f16 --sweep n:512..8192
```

Profiles: `sim:ane`, `sim:webgpu-f16`, `sim:mali-fp16`, `sim:neuron-bf16`, `sim:mi350-tf32`,
`sim:openvino-gpu-f16`, `sim:openvino-npu-f16`, and bit accurate tensor core models such as
`sim:a100-bf16`, `sim:h100-fp8`, `sim:sm120-fp16` ([DEVICES.md](../DEVICES.md),
[TENSOR_CORE_MODELS.md](../TENSOR_CORE_MODELS.md)).

## Limits

* A simulation is the profile, not the device. Results carry TS520 in `truestar check` so nobody
  mistakes them for a device run, and rules labelled guess or hypothesis are only as good as their
  label.
* Inside a compiled native function the program runs on Truestar's own kernels; `simulate` applies
  to eager calls only.
