# Kernel IR

Author: Sharven Rane. First written 2026-11-24. Code: `src/truestar/kernel/`.

The kernel IR is the loop level below the tensor IR. A tensor level lowering pass produces kernels
in this IR; this layer fuses them, generates GPU code (Triton for NVIDIA, Metal Shading Language
for Apple GPUs, docs/METAL.md) or CPU code (numba), caches the compiled result by a hash of the IR,
and checks every compiled kernel against a float64 reference.

## 1. Concepts

* **Size**: an `int`, or a `Sym` (symbolic size) bound from input shapes at call time. A kernel
  compiled with symbolic sizes is reused for every concrete size; nothing recompiles when a
  symbolic dim changes, apart from the Triton block size buckets noted under limits.
* **Axis**: a named iteration axis with a size. Output axes are iterated in parallel; reduction
  axes are bound by a `Reduce`.
* **Buffer**: a named input with a shape (sizes) and a dtype (`float16`, `bfloat16`, `float32`,
  `float64`). Any strides are accepted at run time.
* **Expr**: an immutable expression tree that compares structurally. Equal subtrees are shared,
  which gives common subexpression elimination for free and a stable text form for hashing.
* **Output**: `name[axes] = expr` with a dtype. The expr may only use its own axes freely; every
  other axis must be bound by a reduction.
* **Kernel**: a list of outputs. Its inputs are the buffers its expressions load.

A kernel means: for every point of the output axes, evaluate the expression. Reductions may nest
and may appear anywhere in an expression, so a softmax, a layer norm, a matmul with an epilogue
and a whole attention block are each one kernel.

## 2. Nodes

| Node | Meaning |
|---|---|
| `Const(v)` | float or bool constant |
| `Dim(size)` | the value of a size (an int, or a symbol bound at call time) |
| `Index(axis)` | the current index along an axis, as a number |
| `Load(buffer, index, pad=None)` | element of an input; each index entry is an `Axis`, an `int` (0 for a broadcast dim) or a strided index (below) |
| `Unary(op, x)` | `neg exp log sqrt rsqrt sigmoid tanh abs not erf cos sin expm1` (`expm1` added 2026-11-24: `np.expm1`, `torch.expm1`, `math.expm1` in numba, `libdevice.expm1` in Triton) |
| `Binary(op, a, b)` | `add sub mul div maximum minimum`, comparisons `lt le gt ge eq ne`, logic `and or` |
| `Where(c, a, b)` | select |
| `Reduce(op, axis, body)` | `sum max min` of body over axis |

### Strided indices (2026-11-24)

Convolution and pooling read `x[i * stride + tap * dilation - pad]`, and the gradient of a strided
convolution reads `g[(h + pad - tap * dilation) / stride]` only where that division is exact. A load
index entry can therefore be a `Strided` index:

```
(sum over terms of coeff * (((axis * mul) // div) % mod) + offset) / exact
```

built with `K.strided([(i, 2), (u, 1)], -1)` (terms as `(axis, coeff)` pairs or `K.Term(axis, coeff,
mul, div, mod)`), `offset` and `exact` (1 means no division). `div` and `mod` serve grouped
convolution (`(o // out_per_group) * in_per_group + c`) and adaptive pooling (window start
`(i * H) // OH`). The index is valid when it divides exactly and falls inside the buffer dim. A load
whose strided index may be invalid must give `pad=`, the value it reads there (0 for zero padding,
minus infinity for max pooling); building one without it raises `IRError`. An index that is provably
valid for every axis value (static sizes) needs no pad and gets no mask. Axes are never negative,
so floor division and modulo of a term mean the same in every backend; the whole numerator can be
negative, and every backend first checks it is not before dividing.

The references gather with these rules (NumPy and torch, `reference.load_strided`), the numba
backend emits `(buf[...] if valid else pad)`, and Triton masks the load with `other=pad`. The
references also sum a chain of nested sums over a product (a convolution sums over channels and
both kernel taps) as one einsum, so the check of a batch 64 convolution never builds the unfolded
input.

Python operators build nodes: `+ - * /`, unary `-`, `< <= > >=`, `& | ~`. `==` stays structural
equality; use `K.eq(a, b)` for an elementwise comparison. Using an expression in `if` raises, so
data dependent control flow cannot hide in a kernel.

## 3. Builder API

```python
import torch
from truestar import kernel as K

M, N = K.sym("M"), K.sym("N")
x = K.buffer("x", (M, N), "float32")
i, j = K.axis("i", M), K.axis("j", N)
y = K.output("y", (i, j), K.softmax_weight(x[i, j] * 0.5, over=j))
fn = K.compile(K.kernel("softmax", [y]), "cuda")      # or "cpu"
out = fn(x=torch.randn(64, 1000, device="cuda"))
```

`K.softmax_weight(score, over=j)` writes the canonical softmax form
`exp(s - max_r s) / sum_r exp(s - max_r s)`; the attention scheduler recognizes that form.

The graph layer (`K.Graph`) builds kernels from tensor ops and fuses them:

```python
g = K.Graph()
B, H, S, D = 2, 16, K.sym("S"), 64
q, k, v = (g.input(n, (B, H, S, D), "float16") for n in "qkv")
s = (q @ k.T) * D ** -0.5
s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i, s, K.NEG_INF), s, with_index=True)
fn = g.compile({"o": g.softmax(s) @ v}, "cuda")      # one fused attention kernel
o = fn(q=q_t, k=k_t, v=v_t)["o"]
```

Graph ops: `input`, `+ - * /` with NumPy broadcasting, `exp log sqrt rsqrt sigmoid tanh relu abs
maximum minimum scale where mask`, `pointwise(fn, *xs, with_index=...)` for any elementwise
Python function of IR expressions, `sum max min mean` over an axis, `softmax`, `layer_norm`,
`matmul` (`@`, batched, with batch broadcasting) and `transpose` / `.T`.

## 4. Lowering contract (for the tensor IR)

Each tensor op lowers to an index function: given one IR index per output dim (an `Axis`, or `0`
for a broadcast dim of size 1), return the IR expression for that element. `Graph.lower` in
`graph.py` is the reference implementation of this contract and the tensor IR can reuse it op
for op:

* elementwise ops call their operands' index functions at the broadcast index;
* a reduction over dim `d` creates a fresh reduction axis `r`, calls its operand at the index with
  `r` inserted at `d`, and wraps the result in `Reduce`;
* `matmul` is `Reduce("sum", r, a[..., i, r] * b[..., r, j])`;
* `softmax` and `layer_norm` expand to their reductions (see `graph.py`);
* a value that is not inlined becomes a `Buffer` and is read with `Load`.

Reduction axes must be deterministic per op instance (`graph.py` names them after the node id) so
that the same producer inlined twice yields equal subtrees.

Fusion is the choice of which values to inline. `Graph.fuse` inlines everything, asks the backend
scheduler whether the kernel fits a loop shape, and if not materializes the nearest reduction
producer and asks again. The tensor IR can call `triton_backend.plan(kernel)` the same way; it
raises `Unschedulable` with the reason.

## 5. GPU loop shapes

The Triton scheduler (`triton_backend.plan`) chooses a shape from the IR structure only. Op names
never matter, so a new layer written with the plain ops above is fused if its structure fits.

| Shape | IR structure | Code |
|---|---|---|
| tile | no reductions, or reductions that all run over axes of one size and each give one value per row | 2D block of rows by lanes; whole row in registers up to 4096, looped above |
| contraction | one `sum` of a product whose factors split the output axes (matmul), any elementwise prologue on the factors and epilogue on the result | tiled `tl.dot` with grouped program order |
| attention | `sum_j W[i, j] * V[j, d]` where `W` is a linear combination of softmaxes over `j`, each softmax score any elementwise function of one dot product, indices and loads | one online softmax pass with one running max, sum and accumulator per softmax term |
| conv (2026-11-24) | a chain of two or more nested sums over a product `a * b` with a strided load, whose output axes split into M axes (only in `a`), N axes (only in `b`) and batch axes (in both); any elementwise epilogue | implicit GEMM: M axes flattened into rows (batch times output pixels), N into columns (output channels), the summed axes into K. No unfolded input is written; zero padding is the load mask |
| window (2026-11-24) | a kernel with a strided load whose reductions all have static sizes (at most 4096 steps per output element) and none is GEMM shaped | each program owns a block of output elements; every reduction is a loop inside it, nested as written. Pooling, depthwise and grouped convolution, the gather form of the max pooling gradient |

Conv details (code in `kernel/triton_conv.py`, a kernel family of its own; it uses the shared
emitter and launch ABI of `triton_backend.py` but none of the contraction code):

* **Tap loop.** When the outermost summed axis is wide (channels, at least 16) and the others are
  fixed kernel taps (at most 64), K runs as one loop over (tap, channel block): per step only
  scalars are divided, the tap's spatial offsets and padding masks are per row of the tile, and the
  channel slab is contiguous K. Otherwise (the 3 channel stem, the weight gradient, whose sum runs
  over batch and pixels) K is one flat range, decomposed per block. On resnet18 layers the tap loop
  plus storing constant weights with the output channels contiguous (docs/COMPILER.md) took a batch 64 float16 3 x 3 conv from about 3x slower than cuDNN to
  0.6x to 1.5x of cuDNN conv plus relu.
* The conv and window shapes are tried after tile, contraction and attention and accept only
  kernels with a strided load, so no kernel that compiled before changes shape. The window shape
  refuses GEMM shaped sums, so fusion splits two convolutions into two conv kernels instead of
  running both element by element.
* Launch configs: conv `BM BN BK GROUP num_warps num_stages` (default by operand class, as the
  contraction: half, TF32 or strict float32), window `BLOCK num_warps`. The conv's M (batch times
  pixels) is part of the tune key, so a config is tuned per power of two bucket of it; window rows
  are batch like and are not.

Attention details:

* Score modifications are plain expressions: masks, softcapping, position biases, per head terms.
* Masks written as index comparisons (`j <= i`, `j > i - W`, combined with `&`) inside
  `where(cond, s, -inf)` are analysed, and key blocks that are fully masked for a query block are
  skipped. Causal attention therefore does half the work, with no block mask from the user.
* Several softmax terms fuse into one kernel that loads each key and value block once. Differential
  attention, `(softmax(Q1 K1) - lambda * softmax(Q2 K2)) V`, is two terms. FlexAttention's
  `score_mod` sees one score at a time and cannot express it in one call.
* Epilogues such as a gate `o * sigmoid(g)` fuse after the normalization.
* Schedule (2026-11-24, each rule measured on its own, see the status log in QUALITY_BAR.md):
  query blocks run longest first (LPT: the last query block, which sees the most keys, starts
  first, with the batch index fastest so the first wave is all heavy blocks); key blocks that no
  analysed mask can touch run in their own loop with every index mask and bounds check removed,
  and only the diagonal, window edge and ragged tail blocks pay for masks (SPLIT); softmax uses
  `exp2` with log2(e) folded into the score's constants (`dot * 0.125` becomes
  `dot * 0.1803...`); the accumulator, running max and sum stay in float32. Operands that are
  plain loads go to `tl.dot` without an elementwise op in between, and pad masks are dropped when
  a head dim equals its block width. LPT and SPLIT are launch switches the autotuner can turn off.
* Split KV decode (`KV_CHUNK`, 2026-11-24): a launch of up to 16 query rows (one token of a decode
  per batch and head) runs one 16 row tile and, when its batch times heads leaves the SMs idle,
  splits the keys into chunks of `KV_CHUNK` (a power of two, at least BN) along grid axis 2. Each
  program stores its float32 running max, sum and accumulator to a workspace laid out [chunks,
  batch, query rows, terms, BD + 2]; a second generated kernel (`kern_splitkv`) merges the chunks
  in the fixed order 0, 1, 2, ... (a chunk whose keys are all masked stores max -inf and sum 0 and
  adds nothing) and applies the same epilogue. The default chunk (`triton_backend.kv_chunk`) is the
  smallest power of two that keeps about 4 waves of programs on the GPU's SMs, and a decode splits
  only when one program would otherwise walk at least 2048 keys in a row (waves times context):
  below that the second launch costs more than it saves. A decode that does not split, and every
  prefill, keeps the blocks it had before split KV. Multi term (differential) attention splits every term. The workspace is allocated
  per call like the split K one, so CUDA graph capture works. Paged KV caches and grouped query
  heads reach the kernel already gathered and expanded (docs/LLM.md), so they split the same way.

The Metal backend (`metal_backend.py`, docs/METAL.md) uses this scheduler's structure analysis and
maps tile kernels to a pointwise or a threadgroup per row shape, contractions to SIMD group matrix
tiles, and attention, conv and window kernels to one thread per output element with serial loops.

The CPU backend (`cpu_backend.py`) accepts every kernel: a numba loop nest per output, the
outermost loop in parallel, each reduction placed at the shallowest loop depth where its axes are
bound. It is the correctness backend, not a fast one.

## 6. Cache

`compile` canonicalizes the kernel (axes, buffers, symbols and outputs renamed by order of first
appearance), then hashes the canonical text together with the backend, the options and
`CODEGEN_VERSION`. Consequences:

* Kernels that differ only in names share one binary.
* An edit to one layer changes only the hashes of kernels whose IR changed; the rest hit the cache.
* In process hits cost a hash (graph build, fusion and cache hit for an attention block: 1.2 ms).
* Generated modules live in `TRUESTAR_CACHE_DIR` (default `~/.cache/truestar/kernels`). Triton
  and numba key their own binary caches on that source, so a new process skips codegen and the
  backend compiler.

## 6a. Launch configs and autotuning

Each GPU loop shape has a launch config, a flat dict chosen per call signature:

| shape | config keys |
|---|---|
| tile, pointwise | `BR BC num_warps` |
| tile, single pass reduction | `BR num_warps` (BC is the padded row) |
| tile, looped reduction (rows above 4096) | `BC num_warps` |
| contraction | `BM BN BK GROUP num_warps num_stages KTAIL SPLIT_K` |
| attention | `BM BN num_warps num_stages LPT SPLIT KV_CHUNK` |

`triton_backend.default_config` is the fixed heuristic, `candidates` the search space (default
first; contraction candidates are pruned by shape to at most 5 tiles plus split K neighbours),
`check_config` the validator. `compile(..., tune=None)` (additive
option; `Graph.compile` passes it through) picks the mode: `True` tunes every launch, `False` never
does, `None` follows `TRUESTAR_TUNE` (`auto` by default: tune launches whose work is big enough to
matter, `triton_backend.worth_tuning`; `0` off; `1` on).

* A search runs once per (kernel structure, shape bucket, input dtypes, device name, Triton
  version). The shape bucket rounds each symbolic size up to a power of two, and the row count of
  a contraction (batch times tokens in a linear layer) up to a power of four, so nearby batch
  sizes share one search. The kernel structure is the cache key with float constants blanked, so
  editing a scale reuses the tuned config while the new binary is still compiled and verified.
* Contractions also look up a shape key: the operand expressions and sizes without the epilogue
  (`CompiledKernel.tune_shape_key`). A config tuned for `linear + gelu` is used, without a search,
  by `linear + relu` with the same weights shape, in this process or a later one.
* Every candidate, the default included, compiles in one thread pool (`TRUESTAR_TUNE_WORKERS`),
  so a search costs about one Triton compile of wall time plus the timing; then candidates are
  timed one at a time with `triton.testing.do_bench`, best three timed again for longer.
  `TRUESTAR_TUNE_BUDGET` (default 30 s) bounds the compile phase; unfinished candidates are
  skipped and reported. A failing candidate (for example out of shared memory) is skipped. The
  record stores `compile_seconds` and `timing_seconds`.
* `auto` tunes a launch when its work reaches `WORTH_TUNING`: 2^24 elements for tile kernels,
  2^28 for attention and 2^32 multiply adds for contractions (since 2026-11-24; the contraction
  defaults came within 2.5% of the tuned configs on the transformer layer, see the status log).
* A launch not worth tuning normally takes the default config. When that default would be a new
  Triton variant while a config this kernel already compiled (at another batch size) is rated at
  most `REUSE_SLACK` (1.5) times the default by `triton_backend.estimate` (waves of programs times
  the work of one program, with a load overhead for thin tiles), that config is reused, so a new
  batch size compiles nothing; only for launches below `REUSE_MAX_WORK` (2^28 multiply adds),
  since a larger one repays its own compile. It is still verified at the new shape.
* The winner is written to `<cache>/triton/tune/<key>.json` with a digest and the timings. A later
  process reads it and does no search. A file that does not parse, has another version or key,
  fails its digest, or holds a config `check_config` rejects raises a `TuneCacheWarning` and is
  tuned again and rewritten.
* `CompiledKernel.launch_configs()` lists the config and its source (`default`, `tuned`, `cache`,
  `shape cache`, `reused`) per call signature, `CompiledKernel.tune_seconds` and `K.tune_stats` report the cost.
* Device limits (`triton_backend.device_limits`): the SM count, shared memory per program (opt in)
  and total memory of the GPU the inputs are on, read once per device from
  `torch.cuda.get_device_properties`; without a visible GPU, the values of the RTX 5070 Ti the
  defaults were measured on (70 SMs, 101376 bytes, 16 GB). Program count thresholds written for 70
  SMs are scaled by `sms / 70` (`_waves`), so a 70 SM GPU gets exactly the measured configs (a
  frozen table of 168 cases in tests/data/contraction_defaults_70sm.json). Strict float32 on fewer
  than `FEW_SMS` (48) SMs follows its own rules (`_few_sm_ieee`), measured on an RTX 5060 Laptop GPU
  with 26 SMs: about 4 waves of programs instead of about 1, so up to 16 rows get 16 x 128 x 32 tiles
  with 2 warps and 4 stages and split K to 4 waves in chunks of at least 128, 17 to 32 rows 32 x 64
  x 32, and 128 x 256 and 128 x 128 tiles only from 5 and 2 waves. float16, bfloat16 and TF32 below
  `FEW_SMS` (`_few_sm_half`, `_few_sm_tf32`, same measurements) keep the 70 SM rules except: 64 x 128
  tiles for large products (half from 1024 rows, K of 1024 and 5 waves; TF32 from 256 rows with N
  and K of at least 1024),
  and split K to 4 waves where few programs run over a long K (half: under 2 waves of 32 x 64 tiles
  with K of at least 2048; TF32: under 2 waves of 64 x 64 tiles above 32 rows, and to 8 waves up to
  32 rows). Without split K (deterministic mode) those split rules give way to the 70 SM ones. Only
  26 and 70 SMs were measured. The shared memory budget of the pipelined operand
  loads is the device's limit minus 3 KiB (96 KiB on sm_120, as before), and a split K workspace
  may take at most 1/16 of the GPU's memory.
* A compiled config that needs more local memory (spilled registers, Triton's `n_spills`) than the
  GPU allows is refused before its first launch with `ConfigResourceError` (TS700): above 512 KiB
  per thread (the CUDA limit) or, for every resident thread of every SM, above a quarter of the
  GPU's memory. The tuner records the reason and skips the candidate; a default or forced config
  raises.
* Contraction defaults (`triton_backend._contraction_default`) pick tiles by operand class and by
  how many programs a tile gives for the GPU's SMs: `half` (float16, bfloat16), `tf32`
  (float32 compiled with `allow_tf32`, recorded on the plan as `tf32`) and `ieee` (strict float32 on
  FMA units). Program counts include the batch axes (a projection imported from `[B, T, C]` is a
  contraction with batch axes), and 128 row tiles need more than 64 rows. Small M gets 16 or 32 row
  tiles with a long BK so enough programs run, or split K (below); large half and TF32 problems get
  64 x 64 tiles; large strict float32 gets 128 x 256 x 16 with 8 warps, since an FMA dot gains from
  more outputs per thread. `KTAIL` (on by default) runs whole BK slabs without a
  K mask and one masked tail slab: with K a runtime size the K mask has a constancy of one element,
  which kept the operand loads from vectorizing. `tune.SEARCH_SPACE` is part of the tune key, so a
  change of defaults or candidates searches again instead of reusing configs from the old space.
* Split K (`SPLIT_K` above 1): each program reduces one chunk of whole BK slabs (grid axis 2)
  and stores a float32 partial to a workspace; a second generated kernel (`kern_splitk`) adds the
  partials in the fixed order 0, 1, ..., SPLIT_K - 1 and applies the epilogue. Strict float32
  defaults use it when wide tiles give too few programs for the SMs (small M, long K), with
  64 x 128 x 16 tiles (16 x 64 x 64 for M up to 16) and chunks of at least 256 elements of K, and
  two chunks when 128 x 128 tiles fill less than one wave (on fewer than 48 SMs, see device limits
  above). Both kernels of a new split launch
  compile at once in two threads. The
  launch path allocates the workspace per call (`SPLIT_K * batch * M * N` float32) from the torch
  caching allocator, so CUDA graph capture and streams work as for any output.
* Verification covers the config actually used: the verified set is keyed on the config, so a
  tuned config is checked against float64 on its first call even when the default was checked
  before. The tuner never skips or weakens verification.

### Contraction rows and the cuBLAS launch config (2026-11-24)

A contraction whose right factor has no batch axis folds the batch axes into its rows
(`ContractionPlan.fold`): one program grid over every row, the row index split into the batch
indices and `m` with `//` and `%`. `BM` then counts flattened rows. For float32 contraction kernels
compiled with `library=True` (the compiled executor does, outside deterministic mode) the launch
config `{"LIB": 1}` runs the kernel as one cuBLAS call through torch when the epilogue is none, a
bias, relu of a bias, a residual or both; it is chosen per shape bucket by timing (cached as
`<base key>:library`) and verified like any launch config. See `kernel/library.py` and
docs/COMPILER.md.

## 6b. Determinism

* **Run to run:** every Truestar kernel gives the same bits for the same inputs, shapes and launch
  config. No kernel uses atomics; split K adds its partials in a fixed order in a second kernel.
* **Across launch configs and batch sizes:** without split K, a contraction's result is bitwise
  independent of the launch config (BM, BN, BK, warps, stages, KTAIL), measured over 72 configs for
  IEEE float32 and TF32 dots: each output element is accumulated over K in the same order in every
  tile shape. So a row's result does not depend on the batch it was computed in, nor on which
  config tuning picked. Split K changes the order (partials of chunks), and its chunk count follows
  M, so with split K a row computed in a batch of 7 and in a batch of 300 can differ in the last
  bits (tests/test_contraction_split.py shows both).
* **`compile(..., deterministic=True)`** (and `CompiledExecutor`/`compile_program` with
  `deterministic=True`, or `None`, the default, which follows
  `torch.use_deterministic_algorithms`) never chooses a split from M or the SM count. Where K is
  one chain (K up to 1024, TF32, half precision, a symbolic K) `SPLIT_K` stays 1 and
  `check_config` rejects a split config, also one read back from the tuning cache. A strict
  float32 contraction with a constant K above 1024 sums K in fixed chunks instead
  (`triton_backend.det_k_chunk`, since 2026-11-25): chunk c covers elements [512 c, 512 (c + 1)),
  each chunk is one fused multiply add chain in K order from zero, and the chunk sums are added
  left to right before the epilogue. The chunks follow K alone, so the result is still
  independent of the batch size, the tile, the SM count and the GPU; whether the chunks run in one
  program or in up to one program each (then added by `kern_splitk` in the same order) is a launch
  choice with the same bits (tests/test_contraction_split.py checks configs, batches, layouts and
  a float64 emulation of the order).
* **Across GPUs of different sizes:** SM aware defaults change tiles and the number of programs
  along K, never the accumulation order in deterministic mode, so a deterministic compiled result
  does not depend on the GPU's SM count. Measured: the compiled executor gave bitwise equal outputs on an RTX
  5070 Ti (70 SMs) and an RTX 5060 Laptop GPU (26 SMs) for an MLP, a TransformerEncoderLayer and
  resnet18 (scripts/laptop_fingerprint.py), before and after the change.
* **Launch configs in deterministic mode.** Tuning by timing picks the fastest config, and that pick
  differs per GPU and can differ per run. For a contraction (without split K) or a pointwise tile
  the pick cannot change the bits, so these are still tuned. For attention and for reductions
  (`tile_single`, `tile_looped`) the block sizes and warps set the order of the sums and of the
  online softmax rescaling, so a kernel compiled deterministic keeps their default config, which
  depends only on the shapes: it is neither tuned nor read from the tuning cache, and
  `launch_configs()` reports its source as `pinned` (`triton_backend.bits_follow_config`). Measured
  2026-11-24 on the TransformerEncoderLayer of `truestar fingerprint` at batch 256: before the pin the
  attention was tuned from BN 32 to BN 64 in deterministic mode, which changed its bits and made row
  0 differ between batch 64 and batch 256; with the pin row 0 is equal at batch 1, 64 and 256.
  Order keeping tuning (P49): such a kernel is still tuned over the parameters that cannot change
  its order, from its default config: `LPT` (which program runs which tile) and `num_stages` (how
  far ahead loads are issued) for attention, `BR` (independent rows per program) for `tile_single`;
  `tile_looped` has none and stays pinned (`triton_backend.ORDER_FREE`, `order_key`,
  `order_candidates`). A config read from the tuning cache whose `order_key` differs from the
  default's is refused, and the pick is run against the default on the tuning inputs: unless the
  outputs are bitwise equal the default is kept (the record's `order_check`). The source is then
  `order tuned`. `scripts/order_classes.py` measured the split: over 18 cases (attention causal, full
  and decode, softmax and layer norm, float16 and float32, every config of a grid) these
  parameters never changed a bit, and block sizes, warps, `SPLIT` and `KV_CHUNK` did (warps change
  the tree of `tl.sum`). On the RTX 5060 Laptop the float32 attention prefills ran 1.23x faster
  (`num_stages` 2 to 1); the other cases are within the timing noise (about 5 percent, from same
  config pairs) (`scripts/bench_detautotune.py`).
  Half precision and TF32 contractions were also bitwise equal over their whole tuning grid
  (8 configs each, RTX 5070 Ti). What this does not cover across GPUs: other architectures compile
  the same Triton source to other instructions (docs/CROSS_CHIP.md).
* **Split KV decode:** run to run deterministic in both modes (fixed merge order, no atomics). In
  deterministic mode the chunk is `DET_KV_CHUNK` (256 keys) from a context of 2048 keys, on
  every GPU: the chunking depends on the context length only, never on the SM count, the batch or
  the launch blocks, so a 26 SM and a 70 SM GPU, and a row decoded alone or in a batch, give the
  same bits (tests/test_splitkv.py). `check_config` rejects any other chunk for a deterministic
  attention. A decode and a prefill of the same row are different launches and can differ in the
  last bits (the prefill does not split).

### Order free sums (2026-11-24)

`K.order_free_sum(axis, body)` (IR op `osum`) and `compile(..., order_free=True)` give row
reductions whose float32 bits do not depend on the order, block size, split or SM count, so such a
kernel is tuned even in deterministic mode. About 1.3x to 2x slower per sum. Contractions and
attention keep ordered sums. Details and measurements: docs/ORDER_FREE_SUM.md.

### Round once elementwise math (2026-11-24)

`K.compile(k, backend, math="round_once")` (also `Graph.compile`) makes `exp`, `log`, `tanh`,
`erf`, `rsqrt` and `sigmoid` give the same, correctly rounded float32 bits on every device: each is a
fixed sequence of correctly rounded float64 operations rounded once, emitted as `_ro_<op>` helpers
for Triton (launched with `enable_fp_fusion=False`) and numba; `sqrt` becomes the IEEE float32
square root. Correct rounding is proven over all float16 and bfloat16 inputs and checked over all
2^32 float32 inputs with a measured error bound; the cost on this GPU is 6 to 77 times the native
op. Metal, float64 buffers, cos and sin, and the attention loop shape are refused (TS1089 to
TS1093). Every compiled kernel is still checked against float64 on its first call. Details,
numbers and limits: docs/ROUND_ONCE.md.

## 7. Verification

Every compiled kernel is run against `reference.run_reference`, a float64 NumPy interpreter of the
IR, on its first call for each input shape and dtype signature. A mismatch raises
`KernelMismatchError` naming the kernel, the backend, the output, the error and the tolerance, and
prints the kernel's IR. `TRUESTAR_VERIFY=all` checks every call; `TRUESTAR_VERIFY=0` turns it off
for debugging only.

* Error measure, per element (since 2026-11-24): `|out_i - ref_i| / max(|ref_i|, floor)`, worst
  element per output, against the tolerance; the error message names that element. The floor is a
  fraction of the output's rms, capped at its largest magnitude and never below 1e-6: 1/8 of the
  rms for a kernel with no reduction, 2x the rms for a kernel with one (a sum that cancels to near
  zero carries the rounding of its terms, which is on the scale of the output's rms, not its own),
  and 4x the rms in the compiled executor's region and end to end checks, which do not see the
  kernel's reduction length. Before, every element was divided by the output's largest magnitude
  (`max |out - ref| / max |ref|`), so a small element could be wrong by 100% and pass: softmax
  probabilities away from the peak, a low magnitude region, a tail of small values. With the cap
  the new measure is never looser than the old one. NaN and infinity must appear at exactly the
  same places as in the reference.
* How the floors were chosen: every one of the 5583 checks the kernel, tuning, compiler, executor,
  frontend, training and LLM op tests run (`kernel/verify.compare_numpy` and `compare_torch`) was
  recorded with the smallest floor it would need to pass. Of the 5455 checks on correct kernels the
  largest need was 0.011 x rms without a reduction (floor 0.125: 11x margin), 0.52 x rms with one
  (floor 2: 3.8x) and 1.29 x rms in region checks (a float16 ReLU after a convolution; floor 4:
  3.1x). The other 128 checks are the tests' deliberately broken kernels, which fail under both
  measures. So the gain is large where an output has a wide spread of magnitudes (an elementwise
  kernel, a peaky reduction output) and modest (about largest / (2 rms), 1.7x on a typical
  Gaussian output) for a reduction with an even output.
* Measured (tests/test_kernel_verify_elements.py): an `exp` kernel wrong only on its small results
  (3x below 0.5 in float16 and bfloat16, 1.5x below 0.02 in float32) and a row sum 0.5% wrong on
  every row but the 4 that are 1000x larger: 8 compiled runs on CPU and CUDA, all 8 pass the old
  measure, all 8 fail the new one; the correct kernels pass both.
* Tolerance: by the lowest precision among inputs and outputs (`float32` 2e-5, `float16` 2e-3,
  `bfloat16` 1.6e-2, TF32 when allowed 2e-3), times `max(1, log2(longest reduction) / 4)`.
* float32 dot products use IEEE precision unless `compile(..., allow_tf32=True)`. The kernel
  layer takes the setting explicitly; the compiled executor follows PyTorch's (docs/COMPILER.md).
* The reference evaluates large outputs in chunks over leading axes, so a 2048 token attention
  check needs about 64 MB per intermediate array (256 MB on the GPU, see below).

### Where the reference runs

A CUDA kernel whose check is big enough is checked on its own GPU (rule below):
`reference.run_reference_torch` evaluates the same IR
node by node with the torch float64 op that computes the same function as the NumPy op (einsum
for a sum of products, NaN propagating `maximum`, `minimum`, `amax` and `amin`, `torch.erf` for
`erf`), and `verify.compare_torch` applies the comparison rules above step for step on the device.
Nothing is computed in lower precision: inputs and outputs are widened to float64 on the GPU first.
The two references differ only by float64 rounding (summation order, libm last place), measured at
up to 9.6e-16 relative over 801 outputs of 158 kernels in the kernel, tuning and compiler tests,
at least six orders of
magnitude below the smallest tolerance (1e-9 for float64 kernels). If torch cannot run a node's op
on the device in float64, that node alone is evaluated with the NumPy function on the CPU, the
result is moved back, and the op's name goes into `reference.cpu_fallbacks`; on torch 2.11 with
CUDA 12.8 no op needs this. Kernels whose inputs are not all on one CUDA device (the numba CPU
backend) keep the NumPy reference.

* `TRUESTAR_VERIFY_DEVICE=auto` (default): the GPU when `reference.work_estimate` (the largest
  output, or the largest iteration space of a reduction) is at least `GPU_MIN_WORK` = 2^20
  elements, else NumPy. Below that NumPy takes a few milliseconds, while the GPU's first float64
  use in a process costs 0.1 to 0.2 s (CUDA loading each float64 kernel and cuBLAS once).
  `gpu` uses the device for every CUDA kernel; `cpu` uses NumPy for every kernel.
* On the device, large outputs are evaluated in blocks of their leading axes within 2^25 float64
  elements (256 MB) per intermediate array; the last chunked axis is cut into blocks as wide as
  fit: an S8192 causal attention check (B1 H2 D128) takes 1.1 s on the GPU; the CPU reference
  took 20.8 to 26.0 s per S8192 row in the kernel benchmark of 2026-11-24.

### Sliced float64 reference for large contractions (2026-11-24, opt in)

`TRUESTAR_VERIFY_F64=sliced` computes the device reference of a large two operand contraction on the
int8 tensor cores with a per entry bound proven at or below native float64's, and checks its first
call per shape against a CPU float64 matmul (docs/SLICED_F64.md). Measured 2.05x faster for the
float64 check of a 4096 cubed matmul on an L4; slower than native below 2048.

### The schedule is unchanged: every new shape signature is checked

A cheaper schedule was considered and rejected: verify a kernel once per dtype at a representative
size and once at its first distinct size, then trust the same code at every later batch size. It
is not as strong, so it would break "never introduce" #3. The generated code being identical does
not make its behaviour at a new size identical: masks at the last partial tile, a second row block
or a second program along the grid axis, 32 bit offset arithmetic and Triton specializing a
variant for a new power of two bucket are all reached first at some size that neither of the two
checked sizes covered. `tests/test_kernel_gpu_verify.py` holds such a kernel (right for 64 rows,
wrong past them, checked first at 7 and 64): any schedule that stops after two sizes passes it at
100 rows. So every new input shape signature (and every new launch config) is still checked; the
GPU reference makes each check cheap enough that this costs little.

### When verification runs, and what that does not catch

Verification is per process and per shape: a kernel is checked on its first call for each input
shape, dtype and launch config signature, and the record of what was checked lives in process
memory (`kernel.compiler._verified`), so a new process checks again. The compiled executor's region
and end to end checks run on the first run of each program and region (docs/COMPILER.md). What this
catches: a code generator or lowering bug that shows on the inputs of that first call, at every
shape seen (masks at a partial tile, a second row block, 32 bit offsets are reached first at some
size, and each new size is checked). What it does not catch:

* data dependent faults on later calls with the same shapes: an overflow or a fully masked row that
  only real data reaches, a NaN that only appears later, values outside the range the first inputs
  covered;
* races or nondeterminism that the first call happens not to show;
* faults on small elements smaller than the floor above (below 1/8 of the rms for elementwise
  kernels, 2x the rms for reductions, on the scale of the tolerance), and anything within the
  tolerance itself.

`TRUESTAR_VERIFY=all` checks every call (the float64 reference then runs on each one);
`truestar kernel` (docs/KERNEL_CHECK.md) runs a kernel on many input variants with a derived
tolerance, for the faults a first call check cannot see.

## 8. Limits (2026-11-24)

* GPU outputs of one kernel must share their axes; offsets are 32 bit, so tensors stay below 2^31
  elements.
* Attention head dims up to 256; each softmax score must contain exactly one dot product.
* Tile kernels need every reduction in the kernel to run over axes of the same size; other kernels
  are split by fusion.
* Autotuning searches a small fixed candidate list per loop shape, not an open space; the default
  configs were measured on one GPU (RTX 5070 Ti, sm_120). A tuned config is per power of two
  shape bucket, so a size far from the bucket's top may not get its own best config.
* Block sizes are Triton compile time constants, so a symbolic reduced or lane axis that crosses a
  power of two bucket compiles a new Triton variant even though Truestar's cache key is unchanged.
  Batch like symbols do not: since the tensor IR was wired in (docs/COMPILER.md), symbolic sizes are
  excluded from Triton's own argument specialization (on the value 1 and on divisibility by 16),
  the tile block row count does not follow a symbolic row count (in the defaults and in the tuning
  candidates), and the autotuner's key leaves out tile row symbols and the batch symbols of
  contraction and attention kernels. One tuned config, and one Triton variant, then serves every
  batch size; the price is that it is the config that won at the batch size that was tuned first.
  Strides and pointers keep Triton's default specialization.
* SPLIT duplicates the attention loop body (two or three loops), which makes a cold attention
  compile slower than with one loop (see the status log).
* The CPU backend widens float16 and bfloat16 to float32 and computes in float64 inside its loops;
  its attention is a plain loop nest (quadratic work per output element).
* NaN propagation through `max` and `min` reductions on the GPU follows the hardware and can differ
  from the reference; verification reports it as a NaN mismatch.
* No autodiff yet: kernels are forward only.
* No scan node: a recurrence over a carried state (linear attention, SSM scans) cannot be written in
  this IR. The chunked linear attention and SSD kernels are a kernel family of their own in
  `truestar.linattn` (docs/LINEAR_ATTENTION.md), checked against a float64 recurrence the same way.
* Conv kernels never split K: the summation order would change with the launch config. Deep layers
  with few output pixels (resnet18's 7 x 7 stage at batch 64) then run few programs with a long K and
  trail cuDNN (1.5x in float16, 2.2x in strict float32, 2026-11-24 measurement).
* The window shape loops over reductions element by element: right for pooling and depthwise
  convolution, slow for a dense convolution (which is why it refuses GEMM shaped sums).
