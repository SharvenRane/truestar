# Switching to Truestar

Author: Sharven Rane. First written 2026-11-24.

This page takes a typical PyTorch training loop and inference script and writes each one in
Truestar's own API: `truestar.nn` for layers, `truestar.optim` and `truestar.train_step` for
training, `@truestar.native` and the compiled executor for running, `truestar.check` and the CLI
for checking. Section 4 brings an existing PyTorch or Hugging Face model in once through the
importer, after which the PyTorch class is no longer needed. Every Truestar snippet here is
cut from a script under `examples/switching/` that `tests/test_cookbook.py` runs on the CPU.

Short recipes for single tasks are in the [cookbook](cookbook/index.md).

## The idea in four lines

* A layer is an immutable config (`nn.Linear(16, 64)`). Its parameters are a plain tree of
  tensors that you hold (`p = net.init(seed=0)`), drawn from a keyed generator, so the same seed
  gives the same bits on every machine.
* A model, a loss or a forward pass is a native function: `@truestar.native` translates its
  source to Truestar IR before any data exists. Shapes are types, so a shape error names the line
  at import time.
* Training mode is a compile time argument. `training=True` and `training=False` are two
  programs; the evaluation program contains no dropout op at all.
* A training step is one program: forward, backward, clipping and the optimizer, with every piece
  of state (parameters, moments, step count) an explicit value in a `TrainState`.

## 1. A training loop

PyTorch:

```python
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.GELU(), torch.nn.Dropout(0.1),
                            torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 5))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_then_cosine)
model.train()
for x, y in loader:
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, "run.pt")
model.eval()
with torch.no_grad():
    acc = (model(xv).argmax(-1) == yv).float().mean()
```

Truestar ([examples/switching/train_native.py](https://github.com/SharvenRane/truestar/blob/main/examples/switching/train_native.py)):

```python
import truestar
from truestar import Tensor, nn, optim
from truestar.compiler import CompiledExecutor

net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(), nn.Linear(H, C))

@truestar.native
def loss(p: net.spec, x: Tensor["B", D], y: Tensor["B", "i64"], training: bool = False) -> Tensor[()]:
    return nn.cross_entropy(net(p, x, training=training), y)

@truestar.native
def accuracy(p: net.spec, x: Tensor["B", D], y: Tensor["B", "i64"]) -> Tensor[()]:
    return nn.accuracy(net(p, x), y)

opt = optim.AdamW(lr=optim.warmup_cosine(3e-3, warmup=6, total=STEPS), weight_decay=0.01)
step = truestar.train_step(loss, opt, clip=1.0, static={"training": True},
                           executor=CompiledExecutor("cpu", "float32"))
ts = step.init(net.init(seed=0))          # TrainState: params, optimizer slots, step count
ts, history = truestar.fit(step, ts, batches, log_every=20, checkpoint="run.safetensors")
ts = step.load("run.safetensors")         # resume later from here, bit for bit
acc = float(accuracy(ts.params, xv, yv))  # evaluation: a separate program, no dropout in it
```

Output on the CPU:

```
step      0  loss 1.622505  grad norm 0.2791  lr 0
step     20  loss 1.297778  grad norm 0.4028  lr 0.00253
step     40  loss 0.897585  grad norm 0.5362  lr 0.000906
loss 1.623 -> 0.973; validation accuracy 0.771
```

What went away: `zero_grad`, `backward`, `step` and the scheduler's `step` are one call;
`model.train()` and `model.eval()` are the static `training` argument, so forgetting one cannot
leave dropout on at evaluation; the checkpoint holds the whole `TrainState`, so a resumed run
continues the uninterrupted one bit for bit
([cookbook: deterministic training](cookbook/deterministic_training.md)). By hand, without `fit`:
`ts, r = step(ts, x, y)` with `r.loss`, `r.grad_norm`, `r.lr`. More in [OPTIM.md](OPTIM.md) and
[NN.md](NN.md).

## 2. An inference script

PyTorch:

```python
model = Net()
model.load_state_dict(safetensors.torch.load_file("weights.safetensors"))
model.eval()
with torch.no_grad():
    labels = model(x).argmax(-1)
```

Truestar ([examples/switching/infer_native.py](https://github.com/SharvenRane/truestar/blob/main/examples/switching/infer_native.py)):

```python
from truestar.compiler import CompiledExecutor
from truestar.interop.safetensors import load_file

net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, C))
ex = CompiledExecutor("cpu", "float32")          # or "cuda"

@truestar.native(executor=ex)
def predict(p: net.spec, x: Tensor["B", D]) -> Tensor["B", "i64"]:
    return net(p, x).argmax(-1)

p = unflatten(load_file("weights.safetensors", mmap=False))   # {"0": {"w", "b"}, "2": {"w", "b"}}
labels = predict(p, x)                           # any batch size, one compiled program
```

```
batch   1: predictions [4]; equal to eager True
batch   7: predictions [3, 3, 3, 4, 3, 4, 0]; equal to eager True
batch 256: predictions [0, 0, 0, 0, 4, 3, 4, 4] ...; equal to eager True
compiles: 1; verified against float64 on the first call: True
```

The first call compiles the program, checks every kernel, every fused region and the whole program
against Truestar's float64 reference interpreter, and only then returns
([cookbook: verify against float64](cookbook/verify_float64.md)). There is no `no_grad`: a native
function computes no gradient unless you ask for one with `truestar.grad`. Parameter names in the
file are the tree paths joined by dots (`0.w`, `0.b`, `2.w`); the script has the two short helpers
that flatten and unflatten the tree.

## 3. PyTorch calls and their Truestar equivalents

| PyTorch | Truestar | Notes |
|---|---|---|
| `torch.nn.Linear(16, 64)` | `nn.Linear(16, 64)` | a config; weights from `layer.init(seed)` |
| `class Net(torch.nn.Module)` with `__init__` and `forward(self, x)` | `@nn.layer class Net(nn.Layer)` with `children()` and `forward(self, p, x)` | [NN.md](NN.md) |
| `model(x)` | `net(p, x)` eagerly, or inside a `@truestar.native` function | |
| `model.parameters()`, `state_dict()` | the tree `p`; `net.num_params()` | |
| `torch.manual_seed(0)` before building a model | `net.init(seed=0)` | keyed per parameter path: inserting a layer does not move the others |
| `model.train()`, `model.eval()` | a static `training: bool` argument | two programs, not a flag read at run time |
| `torch.nn.functional.cross_entropy` | `nn.cross_entropy(logits, y, ignore_index=..., label_smoothing=...)` | also `nn.mse`, `nn.accuracy` |
| `torch.nn.Dropout(0.1)` | `nn.Dropout(0.1)` | masks keyed by seed, step and call site ([RANDOM.md](RANDOM.md)) |
| `torch.nn.BatchNorm2d` | `nn.BatchNorm(c)`: `y, s2 = bn(p, s, x, training=...)` | running statistics are a second tree, returned, never mutated |
| `torch.optim.AdamW(model.parameters(), lr)` | `optim.AdamW(lr=...)` | also `SGD`, `Adam`, `Muon` ([OPTIM.md](OPTIM.md)) |
| `LambdaLR` warmup and cosine | `optim.warmup_cosine(peak, warmup, total)` | evaluated on the host, never recompiles |
| `clip_grad_norm_(params, 1.0)` | `truestar.train_step(..., clip=1.0)` | |
| `zero_grad(); loss.backward(); opt.step()` | `ts, r = step(ts, x, y)` | one compiled program |
| `torch.autograd.grad(loss, params)` | `value, g = truestar.grad(loss, wrt="p")(p, x, y)` | `g` has the structure of `p` |
| `torch.autocast("cuda", torch.bfloat16)` in training | `truestar.train_step(..., precision="bf16")` | `"f16"` adds dynamic loss scaling |
| `model.to("cuda")` | `CompiledExecutor("cuda")`, or `executor="cuda"` on `train_step` | |
| `torch.compile(model)` | `@truestar.native(executor=CompiledExecutor(...))` | checked against float64 on the first call per shape signature ([COMPILER.md](COMPILER.md)) |
| `torch.use_deterministic_algorithms(True)` | `CompiledExecutor(..., deterministic=True)` | [cookbook: bitwise across GPUs](cookbook/cross_gpu_bitwise.md) |
| `torch.save(state)`, `torch.load` | `truestar.fit(..., checkpoint=path)`, `step.save(path, ts)`, `step.load(path)` | safetensors underneath |
| `safetensors.torch.save_file`, `load_file` | `truestar.interop.safetensors.save_file`, `load_file` | [INTEROP.md](INTEROP.md) |
| `torch.cuda.max_memory_allocated()` after a run | `ex.last.memory_plan({"B": 64})` before it | [cookbook: memory planning](cookbook/memory_recompute.md) |
| `torch.utils.checkpoint` | `CompiledExecutor(..., memory_budget=..., recompute=True)` | chosen by the planner, checked bitwise |
| `torch.export.export`, then a converter | `truestar export model.py -o model.tsx` | written only if it matches the model ([EXPORT.md](EXPORT.md)) |
| `torch.testing.assert_close(out, ref, rtol=..., atol=...)` | `truestar.assert_close(fn, inputs, out)` | tolerance derived from float64 ([TOLERANCE.md](TOLERANCE.md)) |
| comparing two devices by hand | `truestar check model.py --backends cpu,cuda` | names the module and the op ([GETTING_STARTED.md](GETTING_STARTED.md)) |

## 4. Bring an existing model in once, then keep working in Truestar

Two ways in, depending on how much of the model you want to own.

**Rewrite the architecture in `truestar.nn`, copy the weights once.** For a model you wrote, this
is the full switch: the layers in [NN.md](NN.md) match `torch.nn` in float64 at 1e-12 relative
(values and gradients, tested), and their parameter layouts are torch's (`Linear` weight
`[d_out, d_in]`, conv weight `[c_out, c_in / groups, k, k]`, LSTM gate order), so the copy is a
renaming: `p["0"]["w"] = sd["0.weight"]`. `examples/nn_mlp.py`, `nn_cnn.py` and `nn_gpt.py` do the
reverse copy to show both sides train to the same loss curve.

**Import the model as it is.** For a model you do not want to rewrite (a Hugging Face model, a
torchvision model), the importer turns it into a Truestar program once. From then on the program
file is the model: it is trained, compiled, checked and deployed by Truestar, and the PyTorch class
is not needed again
([examples/switching/import_once.py](https://github.com/SharvenRane/truestar/blob/main/examples/switching/import_once.py)):

```python
from torch.export import Dim
from truestar import export, ir
from truestar import train as T
from truestar.ir import serialize

# once, where the PyTorch class exists
prog = ir.import_module(model, (x,), dynamic_shapes=({0: Dim("B", min=1, max=4096)},))
ir.run(prog, x)                       # the float64 reference interpreter; compare with model(x)
serialize.save(prog, "classifier.tsx")

# from then on, no PyTorch class
prog = serialize.load("classifier.tsx").program
step = T.build_train_step(prog, T.CrossEntropy(), T.AdamW(3e-3, weight_decay=0.01),
                          params=[v.name for v in prog.params], clip_grad_norm=1.0)
trainer = T.Trainer(step)
r = trainer.train_step(x, y)          # one program: forward, loss, backward, clip, AdamW
trained = dataclasses.replace(prog, param_data={**prog.param_data, **trainer.params()})
serialize.save(trained, "classifier_tuned.tsx")
run = export.load("classifier_tuned.tsx", executor="cpu")    # compiled, checked on its first call
```

A Hugging Face model goes through the same call. The script imports a small random weight Llama
(transformers 4.48) with the batch symbolic:

```
imported Classifier: 4 IR nodes, 6 params; max difference to the module in float64 4.4e-16
fine tuned 30 steps: loss 1.692 -> 0.771; compiled run of the trained file on 3 rows: (3, 5)
Hugging Face Llama (4.48.0): 154 IR nodes; max difference to transformers in float64 1.1e-16 (with its float32 rotary rounding left in: 3.6e-08)
```

Llama's rotary embedding casts to float32 inside a float64 model. Truestar's reference computes
such casts exactly, so the program is compared with transformers under
`ir.reference.exact_float32_casts()`, which does the same on the PyTorch side; without it the
difference is transformers' own float32 rounding.

What the importer covers is measured in [MODEL_COVERAGE.md](MODEL_COVERAGE.md): 170 of 170 models
of its list (the torchvision zoo, 15 Hugging Face configs, 92 `torch.nn` layers and losses), and on
transformers 5.24, 26 of 28 current model families. Mixture of experts, chunked linear attention
and chunked state space layers are refused by name (TS940, TS941, TS942) instead of imported
wrongly; for linear attention layers, see [the cookbook recipe](cookbook/linear_attention.md). An
op the importer cannot express exactly is refused with the op, the setting and your source line
(TS860, [IR.md](IR.md)).

## 5. Checking what you switched

Switching is a change of numerics, so check it the way Truestar checks everything else:

* `truestar check model.py` compares a PyTorch model on every backend here with a float64 CPU
  reference and names the first module and ATen op that diverge
  ([GETTING_STARTED.md](GETTING_STARTED.md)).
* A native program is checked by its executor: every kernel, region and the whole program against
  float64 on its first call per shape signature ([cookbook](cookbook/verify_float64.md)).
* `truestar conform model.py model.tsx` runs the file you ship in its runtime and judges it against
  the model in float64 ([cookbook](cookbook/catch_lying_runtime.md)).

## Limits (2026-11-24)

* The native front end translates a subset of Python: no data dependent Python inside a layer
  except through IR control flow (`if` and `for` on tensors become `cond` and `loop`), no
  comprehensions ([FRONTEND.md](FRONTEND.md), [NN.md](NN.md) Limits).
* `truestar.nn` has no attention dropout and no bidirectional or stacked LSTM in one layer; Rotary
  and ALiBi are standalone layers.
* The importer's training path (`truestar.train`) is a different API from the native one
  (`truestar.train_step`); an imported program does not become a `truestar.nn` tree.
* `truestar export` takes a PyTorch module; a native function is saved with
  `truestar.ir.serialize.save(fn.program, path)` and has no conform model to judge it against yet.
* `load_file` memory maps by default; on Windows the file stays open while the tensors live, so
  pass `mmap=False` to delete or overwrite it right away.
