# Deterministic training

Script: [examples/cookbook/deterministic_training.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/deterministic_training.py)

Two runs from the same seed give the same bits, and a run stopped, saved and resumed continues the
uninterrupted one bit for bit, dropout included.

```python
net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, C))

@truestar.native
def loss(p: net.spec, x: Tensor["B", D], y: Tensor["B", "i64"], training: bool = False) -> Tensor[()]:
    return nn.cross_entropy(net(p, x, training=training), y)

opt = optim.AdamW(lr=optim.warmup_cosine(1e-2, warmup=4, total=STEPS), weight_decay=0.01)
step = truestar.train_step(loss, opt, clip=1.0, static={"training": True},
                           executor=CompiledExecutor("cpu", "float32"))
ts, hist = truestar.fit(step, step.init(net.init(seed=0)), batches, steps=10, checkpoint="half.safetensors")
resumed = step.load("half.safetensors")       # in a new process, a new step object: the same bits
```

Output:

```
run 1 params be307ad9dce191da, run 2 params be307ad9dce191da: identical True
loss 1.4518 -> 0.9501 over 20 steps
stopped at step 10, resumed from the checkpoint: params be307ad9dce191da, equal to the uninterrupted run True
```

## Why it holds

* Every value that changes during training is in the `TrainState`: parameters, optimizer moments,
  the step count. There is no hidden state in modules or optimizer objects, so the checkpoint is
  the whole run.
* Initial weights come from `net.init(seed)`, drawn at a key made from the seed and the
  parameter's path; no global generator.
* A dropout mask is drawn at a key made from the function's seed, its call number and the call site
  (`loss/Sequential#0/Dropout#0/dropout#0`), so a resumed step draws the mask the uninterrupted
  run drew ([RANDOM.md](../RANDOM.md)).
* The compiled CPU kernels reduce in a fixed order.

## Limits

* The check here is on one machine. Across GPUs, see
  [bitwise reproducible results across GPUs](cross_gpu_bitwise.md), which states exactly which
  conditions have been measured.
* The data order is yours: `fit` consumes the batches you give it.
