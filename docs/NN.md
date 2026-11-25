# truestar.nn: layers written natively

Author: Sharven Rane. First written 2026-11-24. Pain row: P36 (docs/QUALITY_BAR.md).
Code: `src/truestar/nn/` (`core.py`, `layers.py`, `losses.py`), tests `tests/test_nn.py`,
examples `examples/nn_mlp.py`, `examples/nn_cnn.py`, `examples/nn_gpt.py`. Optimizers and the
training step: docs/OPTIM.md.

## Use

```python
import torch
import truestar
from truestar import Tensor, nn

net = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3))
p = net.init(seed=0)                     # {"0": {"w", "b"}, "3": {"w", "b"}}: a plain tree of tensors

@truestar.native
def loss(p: net.spec, x: Tensor["B", 16], y: Tensor["B", "i64"], training: bool = False) -> Tensor[()]:
    return nn.cross_entropy(net(p, x, training=training), y)

loss(p, torch.randn(8, 16), torch.randint(0, 3, (8,)))        # compiled once, any batch size
value, g = truestar.grad(loss, wrt="p")(p, x, y)              # g has the structure of p
net(p, x)                                                     # the same layer, eagerly, through PyTorch
```

`@truestar.native` is the Truestar name of the decorator (`@truestar.function` stays as an alias).

## The model, in four rules

1. **A layer is an immutable config.** Every layer is a frozen dataclass holding sizes, flags and a
   dtype. It owns no tensors, is hashable and compares by value (`nn.Linear(3, 2) == nn.Linear(3, 2)`).
2. **Parameters are a tree the caller holds.** `layer.init(seed)` returns nested dicts of tensors;
   `layer.spec` is the same tree of `Tensor[...]` specs, used as the annotation of a native
   function's parameter. A layer with parameters is called `layer(p, x)`; one without (activations,
   pools, dropout) is called `layer(x)`.
3. **State is a second tree, in and out.** A layer with state (batch norm) is called
   `y, s2 = layer(p, s, x, training=...)`; `layer.state_spec` and `layer.init_state()` describe and
   create it. Nothing is ever updated in place: the input tree is unchanged after the call.
4. **Training is a compile time argument.** `training=True` and `training=False` are two programs.
   The evaluation program contains no random op and no running statistic update at all (a test
   counts the ops), instead of a flag read at run time.

Random draws (dropout) are keyed: inside a native function the key comes from the function's seed,
its call number and the call site (`loss/Sequential#0/Dropout#0/dropout#0`), so a step replays bit
for bit on every backend (docs/RANDOM.md). Eagerly, pass `key=`.

## How a layer call compiles

A layer's `forward` is written in the native subset (`truestar.ops` plus static Python on the
config). Outside a native function `layer(p, x)` runs that method as Python through PyTorch. Inside
one, the translator sees an object with `__truestar_inline__` and inlines the method translated
from its source, exactly like a call to another native function: nothing is traced, a shape error
names the line in `layers.py` and the operand types at decoration time, and the IR autodiff
differentiates the result. A user layer is written the same way:

```python
@nn.layer
class GPT(nn.Layer):
    vocab: int
    dim: int
    depth: int

    def children(self):
        out = {"tok": nn.Embedding(self.vocab, self.dim)}
        for i in range(self.depth):
            out[f"b{i}"] = nn.TransformerBlock(self.dim, 4)
        out["head"] = nn.Linear(self.dim, self.vocab, bias=False)
        return out

    def forward(self, p, ids):
        x = self.tok(p["tok"], ids)
        for i in range(self.depth):
            k = "b" + str(i)
            x = self.parts[k](p[k], x)
        return self.head(p["head"], x)
```

`children()` names the sublayers; the parameter tree nests the same way. Inside `forward` a child
is `self.parts["name"]` or the attribute `self.name`. `params()` declares a layer's own tensors as
`Param(shape, init)`; `state()` its state.

## Layers

| Layer | Call | Notes |
|---|---|---|
| `Linear(d_in, d_out, bias=True)` | `lin(p, x)` | W `[d_out, d_in]`, W and b uniform on +-1/sqrt(d_in) (torch.nn's default bound) |
| `Embedding(count, dim, std=1.0)` | `emb(p, ids)` | ids of any shape; `emb.attend(p, x)` for a tied head |
| `Conv2d(c_in, c_out, kernel, stride, padding, groups, bias)` | `conv(p, x)` | `[N, C, H, W]` |
| `MaxPool2d(kernel, stride, padding)`, `AvgPool2d(kernel)`, `GlobalAvgPool()`, `Flatten(start=1)` | `pool(x)` | `AvgPool2d` takes non overlapping windows and needs H and W multiples of the kernel (refused at decoration otherwise) |
| `LayerNorm(dim, eps)`, `RMSNorm(dim, eps)` | `norm(p, x)` | over the last dim |
| `BatchNorm(channels, momentum, eps)` | `y, s2 = bn(p, s, x, training=...)` | running mean and unbiased variance as state, torch's update rule |
| `Activation(kind)`, `ReLU()`, `GELU(approximate)`, `SiLU()`, `Tanh()` | `act(x)` | kinds relu, gelu, gelu_tanh, silu, tanh, sigmoid, none |
| `Dropout(p)` | `drop(x, training=...)` | keyed mask; identity and no op in the program when not training |
| `Sequential(*layers)` | `seq(p, x, training=False, state=None)` | tree keyed by position for layers with params; returns `(y, new_state)` when a layer has state |
| `MLP(dim, hidden, d_out, act, dropout)` | `mlp(p, x, training=...)` | |
| `MultiHeadAttention(dim, heads, kv_heads, causal, bias)` | `att(p, x, mask=None)`, `att(p, x, cache=c, pos=pos)` | padding mask `bool[B, S]`; `kv_heads` < heads is grouped query attention; a KV cache call returns `(y, new_cache)` |
| `TransformerBlock(dim, heads, hidden, kv_heads, causal, norm, act, dropout)` | `blk(p, x, mask=None, training=False, cache=None, pos=None)` | pre norm, "layer" or "rms" |

Added with the op coverage work (2026-11-24, `nn/layers_more.py`, tests `tests/test_nn_more.py`):

| Layer | Call | Notes |
|---|---|---|
| `GroupNorm(groups, channels, eps, affine)`, `InstanceNorm(channels, eps, affine)` | `norm(p, x)` | IR `group_norm`; instance norm is one group per channel (torch's InstanceNorm without running statistics) |
| `Conv1d(c_in, c_out, kernel, stride, padding, dilation, groups, bias)` | `conv(p, x)` | `[N, C, L]` through a 2D convolution over `[N, C, 1, L]`; depthwise and grouped 2D convolution are `Conv2d(groups=...)` |
| `ConvTranspose2d(c_in, c_out, kernel, stride, padding, output_padding, groups, dilation, bias)` | `conv(p, x)` | weight `[c_in, c_out / groups, k, k]` as torch; the new op `ops.conv_transpose2d` (IR `conv2d_input_grad`) |
| `Upsample(scale, mode, align_corners)` | `up(x)` | nearest or bilinear by an integer scale, source indices from `arange` and two `index_select` reads per axis |
| `PixelShuffle(factor)` | `ps(x)` | reshape and permute |
| `LSTM(d_in, hidden, bias)`, `GRU(d_in, hidden, bias)` | `out, h, c = lstm(p, x)`, `out, h = gru(p, x)` | batch first, torch's gate order and parameter layout; the time steps are an IR loop, so one compile serves every sequence length |
| `EmbeddingBag(count, dim, mode)` | `bag(p, ids, per_sample_weights=None)` | sum, mean or max over `[B, L]` bags |
| `Rotary(head_dim, base)` | `rot(x, pos)` | RoPE in Llama's rotate half layout, frequencies `base^(-2i/D)` |
| `ALiBi(heads)` | `alibi(q_len, k_len, dtype)` | additive bias `m_h (j - i)` with the paper's slopes (powers of two and the interleaved rule for other head counts) |
| `GatedMLP(dim, hidden, act, bias)`, `SwiGLU(dim, hidden)`, `GEGLU(dim, hidden)` | `mlp(p, x)` | `w2(act(w1 x) * w3 x)` |

Each is checked eagerly, compiled as a native function and through `truestar.grad` against torch.nn
(or `torch.nn.functional`) in float64 at 1e-12 relative: GroupNorm and InstanceNorm2d, Conv1d at
three stride, padding and group settings, Conv2d grouped and depthwise, ConvTranspose2d at four
settings including output padding and groups, Upsample nearest at scales 2 and 3 and bilinear with
and without align corners, PixelShuffle, LSTM and GRU at sequence lengths 1, 3 and 7 from one
compile (the program holds an IR `loop`), EmbeddingBag in all three modes. Rotary is checked
against transformers' `LlamaRotaryEmbedding` frequencies and `apply_rotary_pos_emb` (the same
formula in float64; the Llama module computes in float32, within 1e-6), ALiBi against the paper's
slope formula for 4, 8, 6 and 12 heads, SwiGLU and GEGLU against the torch composition.

Losses (inlined native functions, eager or compiled): `nn.mse(pred, target)`,
`nn.cross_entropy(logits, targets, ignore_index=None, label_smoothing=0.0)` (one `log_softmax` op,
one `gather`, the counted rows computed in the program), `nn.accuracy(logits, targets)`.

Added 2026-11-25 (`tests/test_nn_losses.py`; value and `truestar.grad` gradient equal to
torch.nn.functional within 1e-13 and 1e-11 relative in float64, eager and compiled):

| Loss | torch equivalent |
|---|---|
| `nn.binary_cross_entropy_with_logits(logits, target, pos_weight=None, reduction="mean")` | `F.binary_cross_entropy_with_logits` |
| `nn.binary_cross_entropy(probs, target, reduction)` | `F.binary_cross_entropy` (logs clamped at -100) |
| `nn.l1`, `nn.smooth_l1(beta=1.0)`, `nn.huber(delta=1.0)` | `F.l1_loss`, `F.smooth_l1_loss`, `F.huber_loss` |
| `nn.nll(logp, targets, ignore_index=None)` | `F.nll_loss` |
| `nn.kl_div(logq, target, reduction="batchmean", log_target=False)` | `F.kl_div` |
| `nn.cosine_similarity(a, b, dim=-1, eps=1e-8)` | `F.cosine_similarity` |
| `nn.log1p(x)`, `nn.softplus(x)` | `torch.log1p`, `F.softplus` |

`reduction` is "mean", "sum" or "none". `nn.log1p` is the importer's composite written in the
native subset (a 2 atanh series on [-1/2, 1/2]): compiled float32 within 2.4e-7 relative of
`torch.log1p` from 1e-12 to 1e4, where `log(1 + x)` in float32 is off by up to 100%. So the compiled
float32 BCE of a confident correct prediction (loss about `exp(-|x|)`, x from 5 to 60) stays within
1e-6 relative of float64; `log(1 + exp(-|x|))` in float32 returns 0 for more than half of them.

**KV cache.** `att.init_cache(batch, capacity)` gives `{"k", "v"}` of `[B, kv_heads, S, head_dim]`
and `att.cache_spec("B", "S")` its annotation. A call with `cache` and `pos` (int64 `[T]`, the
absolute positions of the T new tokens) writes the new keys and values at `pos` with `index_copy`
and attends to slots `<= pos`, so unwritten slots never take part. The prompt and every decode step
are one compiled program for any batch, length, position and capacity (`examples/nn_gpt.py`: one
compile for the prompt and 15 steps; `tests/test_nn.py`: equal to the full forward to 1e-12).

**Causal masks** are built explicitly from `arange` (`cols <= rows`), not with `is_causal`, so the
attention gradient exists for every symbolic length (the IR's attention rule needs fixed lengths
for its own causal flag).

## Initialization

Every parameter is drawn from `truestar.rng` at key `fold(seed, path)`, path being its place in the
tree (`"blocks.0.attn.q.w"`). Consequences, all tested: the same seed gives the same bits on every
machine and device (the draw equals the `python` reference generator); inserting a layer does not
change the initial values of the others; nothing depends on construction order or a global
generator. Scales: `fan_in_uniform` (torch.nn's default bound) for linear and conv, `normal(std)` for
embeddings, ones and zeros for norms. `nn.normal`, `nn.uniform`, `nn.constant` build others.

Schemes added 2026-11-25 (`tests/test_nn_init.py`): `nn.xavier_uniform(gain)`,
`nn.xavier_normal(gain)`, `nn.kaiming_uniform(a, mode, nonlinearity)`,
`nn.kaiming_normal(a, mode, nonlinearity)` (bound or deviation from the parameter's fans at draw
time, `nn.fans` and `nn.gain` equal to torch's `_calculate_fan_in_and_fan_out` and
`calculate_gain`), and `nn.trunc_normal(std, lo, hi)` (torch's inverse CDF method, in float64 from
keyed uniforms; a float32 parameter is the float64 draw rounded). Each has torch.nn.init's
distribution: two sample KS statistic below 0.01 and standard deviation within 1% on 110592
draws. `kaiming_uniform(a=sqrt(5))` is torch.nn's default Linear bound and gives the same bits as
`fan_in_uniform`. Override any parameter with `layer.init(seed, rules={"*.w": nn.xavier_uniform(),
"*.b": nn.zeros})` (first matching `fnmatch` pattern) or `rules=lambda path, prm: ...`; every
parameter keeps its key, so an override changes no other parameter's bits (tested).

## Summary and counting

`print(nn.summary(net))` lists every parameter and state entry (path, shape, dtype, count) from the
config alone, nothing drawn, with totals and bytes; `.params`, `.state`, `.bytes` and `.by_child()`
give the numbers. `nn.count(tree)` is `(elements, bytes)` of any tree of tensors (parameters,
optimizer slots). Counts equal torch.nn's `sum(p.numel())` for a conv, batch norm and linear stack
and for `TransformerBlock` against `torch.nn.TransformerEncoderLayer` (`tests/test_nn_summary.py`).

## Compared with torch.nn and Flax

The same two layer classifier, trained one step, in each.

torch.nn:

```python
model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 3))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()                       # hidden flag on every module; forgetting model.eval() later is a silent bug
loss = torch.nn.functional.cross_entropy(model(x), y)
opt.zero_grad(); loss.backward(); opt.step()     # gradients and moments live inside the objects
```

Truestar:

```python
net = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3))

@truestar.native
def loss(p: net.spec, x: Tensor["B", 16], y: Tensor["B", "i64"], training: bool = False) -> Tensor[()]:
    return nn.cross_entropy(net(p, x, training=training), y)

step = truestar.train_step(loss, optim.AdamW(lr=1e-3), static={"training": True})
ts = step.init(net.init(seed=0))
ts, r = step(ts, x, y)              # one compiled program: forward, backward, AdamW
```

What is simpler than torch.nn: there is no hidden mutable state (parameters, buffers, gradients,
optimizer moments, the training flag and the random generator are all explicit values), so there is
no `.train()`/`.eval()` confusion, no `zero_grad`, no in place batch norm buffers that
`torch.func` needs replacing for, and no `deepcopy` to snapshot a model: a state is a tree. Shape
errors inside a layer are reported before any data exists, with the line in the layer's source.

Flax NNX keeps parameters inside module objects and switches modes with `model.train()` and
`model.eval()`, which set `deterministic` and `use_running_average` on every nested module (or with
separate train and eval views of the same weights); randomness comes from an `nnx.Rngs` object
passed into the module and the step ([Flax: Linen to NNX](https://flax.readthedocs.io/en/stable/migrating/linen_to_nnx.html),
[Model Views](https://flax.readthedocs.io/en/stable/guides/view.html), [MNIST tutorial](https://flax.readthedocs.io/en/stable/mnist_tutorial.html)).
Equinox makes the model itself a pytree, threads `key=` arguments through every random layer,
needs a separate `State` object built with `make_with_state` for batch norm, and an inference flag
for evaluation. Truestar has fewer concepts than both: a config, a parameter tree, an optional state
tree, a static `training` argument. Keys are derived, never threaded; there are no variable
collections, no `Rngs` object, no filtering of a module into arrays and statics (a config is
static, a tree is data, by construction). What Flax and Equinox have that this does not: shared
submodules (weight tying is done by reusing a tree entry, `Embedding.attend`), arbitrary Python
inside a layer (here the native subset), and a large layer zoo.

## Measured (2026-11-24, torch 2.11, CPU unless noted)

`tests/test_nn.py`, 16 tests, all against torch.nn or torch.nn.functional in float64, relative
error bar 1e-12 for values and gradients (1e-10 for attention gradients):

* Linear, Embedding, two conv layers with stride, padding and groups, max and average pools,
  Flatten, LayerNorm, RMSNorm, GlobalAvgPool: values and every parameter's gradient.
* BatchNorm in training and evaluation mode: output, running mean and variance against
  `BatchNorm2d`; the input state is unchanged; the evaluation program has one `batch_norm` op and
  the training program none (it is the primitive batch statistics form).
* MultiHeadAttention: causal, causal with a padding mask and grouped query heads (2 KV heads),
  padding mask with multi query (1 KV head), against `scaled_dot_product_attention`.
* TransformerBlock (RMS norm, SiLU, GQA) decoding 5 + 4 tokens through the KV cache equals the full
  forward, one compile.
* Dropout masks: the same call number gives the same mask, the next call a new one, the evaluation
  program has no random op.
* Cross entropy with `ignore_index` and label smoothing against `F.cross_entropy` (value and
  gradient); logits of 1e4 give a finite loss.
* `ops.expm1` against `torch.expm1` in float64 on 15 points from -30 to 40 including 1e-300:
  under 4e-16 relative, gradient equal to `exp`.

Tests can fail: a causal mask written `cols < rows` failed 3 tests, a batch norm that returns the
old variance failed 2 (here and in test_fit.py); restored.

## Front end additions

* `ops.conv2d`, `ops.max_pool2d`, `ops.avg_pool2d`, `ops.adaptive_avg_pool2d`, `ops.log_softmax`
  (the IR op with its own gradient rule) and `ops.expm1` (a composite: a degree 17 series on
  [-1/2, 1/2], `exp(x) - 1` outside, below 4e-16 relative in float64).
* Calls to objects with `__truestar_inline__` (layers, `nn.inline` functions) are inlined.
* `dict(...)` is a static builtin (layers build their returned state trees with it).
* `truestar.grad` differentiates `cast` and `log_softmax` too (the training gradient rules).
* `truestar.frontend.function.emit_call(builder, fn, *args)` translates a native function into an
  existing builder or editor; the training step uses it to emit the optimizer.

## Limits (2026-11-24)

* The reference executor returns float64 values; a float32 tree fed back into a float32 signature
  must be cast (`train_step` does this for everything it carries; a hand written decode loop casts
  its cache, `examples/nn_gpt.py`).
* No attention dropout (the IR attention op has none). Rotary and ALiBi are standalone layers
  (`Rotary`, `ALiBi`), not yet options of `MultiHeadAttention`. LSTM and GRU are one layer and one
  direction (stack them for more), and collect their outputs with a one hot sum (O(T^2) work).
* A query row with every key masked gives NaN, as in PyTorch.
* Layer forwards are the native subset: no data dependent Python inside a layer except through IR
  control flow, no comprehensions.
* Mixed precision cannot rewrite a loss with loops or branches yet (TS766).
