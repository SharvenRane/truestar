# Export (pain P7)

Author: Sharven Rane. First written 2026-11-24.

Pain P7: export bakes in Python branches, freezes the batch size and fails on stock CNNs, while
TorchScript is deprecated and ExecuTorch is not ready to replace it. Truestar's answer is "export is
compile": the artifact that was verified is the artifact that ships, and an artifact that was not
verified is never written.

Code: `src/truestar/ir/serialize.py` (file format), `src/truestar/export/` (exporter and runner),
tests in `tests/test_export.py`, scripts `scripts/onnx_branch_compare.py` and `scripts/export_zoo.py`.

## Use

```
truestar export model.py -o model.tsx     # model.py defines get_model() and get_inputs()
truestar run model.tsx inputs.pt -o outputs.pt
truestar run model.tsx --info             # signature, program hash, verification evidence
```

```python
from truestar.export import export, load
res = export(model.eval(), torch.randn(2, 3, 224, 224), "model.tsx")   # raises if anything disagrees
outs = load("model.tsx")(torch.randn(7, 3, 224, 224))                    # no model class needed
```

Exit codes of `truestar export`: 0 written, 1 refused (nothing written), 2 usage error.

Since 2026-11-24 the example call may use keyword arguments (`get_kwargs()` in the model file,
`kwargs=` in Python) and nested inputs and outputs, more dims may be symbolic, and models with
`torch.cond`, `while_loop` and `scan` export:

```
truestar export rnn_model.py -o rnn.tsx --static-batch --dynamic xs:0=T --dynamic xs:1=B --dynamic h0:0=B
```

```python
res = export(model, (batch, extras), "m.tsx", kwargs={"scale": s})     # nested arguments and keywords
out = load("m.tsx")(batch, extras, scale=s)                             # the model's own output structure
res = export(rnn, (xs, h0), "rnn.tsx", dynamic_batch=False,
             dynamic_dims={"xs": {0: "T", 1: "B"}, "h0": {0: "B"}})    # each symbol verified at a second size
```

## What export does

1. **Symbolic batch by default.** Dim 0 of every input tensor becomes the symbol `B` in `[1, inf]`
   (before 2026-11-24: `[1, 1024]`, a workaround for an importer crash on unbounded dims;
   `--max-batch` sets a bound, `--static-batch` freezes it). `dynamic_dims` (`--dynamic x:1=T`)
   makes more dims symbolic; each gets its own check at another size (`second_T`), so a length
   baked into the graph is caught like a batch size. A dim torch.export specializes is refused. An example batch of 1 is
   rejected because torch.export specializes size 1 dims, which would freeze the batch silently.
2. **Import** through the existing torch.export importer (docs/IR.md). Unknown ops are listed all at
   once with the user's source line.
3. **Serialize, then parse the bytes back.** Verification runs on the program parsed from the
   artifact bytes, not on the in memory import, so a serializer bug cannot hide.
4. **Verify against PyTorch in float64**, the IR reference interpreter against the user's module
   itself (not against the exported graph), on
   * the example inputs, and
   * a second batch of a different size (7 by default, `--second-batch`), built from the example
     rows in reverse order, cycled, with noise of 5% of the input std on float inputs. A batch size
     or a branch baked into the graph shows up here.

   The bar is a relative error of 1e-9 against each output's largest magnitude (measured agreement
   is about 1e-15). Any NaN or inf mismatch, shape mismatch or integer difference fails.
5. **Refuse or write.** If any check fails, `VerificationError` lists every check and nothing is
   written. If all pass, the same program bytes are written (atomically, temp file plus rename) with
   the evidence, and the written file is read back and compared byte for byte.

Evidence stored in the artifact under `metadata.truestar_export`: per check the input shapes,
dtypes, sha256 of the inputs, max abs and max relative error; the tolerance; the batch symbol and
range; the program hash it verified; torch, truestar and Python versions; UTC date.

## File format (version 1)

One file, no new dependencies, all integers little endian:

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `TRUESTAR` |
| 8 | 4 | u32 format version (1) |
| 12 | 4 | u32 reserved, must be 0 |
| 16 | 8 | u64 header length H |
| 24 | H | header, UTF 8 JSON with sorted keys and no whitespace |
| | | zero padding to a multiple of 64 |
| | D | data section: raw little endian tensor bytes, each tensor 64 byte aligned |
| end minus 32 | 32 | sha256 of every byte before it |

Header: `format`, `format_version`, `program_hash`, `data_nbytes`, `program`, `metadata`. The program
holds the symbols with their ranges, inputs and params with dtype and shape (symbolic dims as an
expression tree over `Add`, `Mul`, `Pow`, `Mod`, `floor`, `ceiling`, `Max`, `Min`, never through
`eval`), every node as op name, input names, attributes, output and source line, and the outputs.
Each param records its data's own dtype, shape, offset, byte length and sha256.

**Generic ops.** An attribute value is a tagged JSON value: `none`, `bool`, `int`, `float` (exact
hex string, so `-0.0`, inf and NaN survive), `str`, `list`, `tuple`, `dict`, `dtype`, `dim`
(symbolic expression), `value` (a reference to an SSA value), `program` (a nested program, for
control flow regions) and `tensor`. A new op whose attributes use these types serializes with no
code change; `test_new_ops_serialize_without_code_changes` writes an op the IR does not define and
reads it back. Anything else (a set, a `torch.device`, a `torch.Size`, an arbitrary object) raises
`SerializeError` naming the attribute, the op and the node. A new structural field on `Node` or
`Program` is also refused, so it can never be dropped silently; it needs a format version bump.

**Identity.** `program_hash` is the sha256 of the canonical program JSON plus the data section. It
excludes the metadata, so the same model exported twice (on different days, from different example
inputs) has the same program hash; the evidence differs. `dumps` of the same program gives the same
bytes, and reading then writing an artifact gives the same bytes.

**Reading** checks, in order, and raises `ArtifactError` on the first failure: magic, format
version (a newer version says "upgrade truestar", an unknown one is never guessed at), reserved
field, file checksum (naming the damaged tensors when the header still parses), header format,
data length, op names (an op this truestar does not define is named), the program hash, param dtypes,
and finally `Program.verify()` (SSA order plus every shape rule).

## Runner

`truestar.export.runner` loads the artifact and runs it through a named executor. The only
executor today is `reference`: the float64 IR reference interpreter, outputs cast to the declared
dtypes (`cast_outputs=False` keeps float64). The compiled executor plugs in with
`register_executor(name, fn)` without changing callers. Inputs are checked for count, exact dtype,
shape and symbol range before anything runs.

`test_artifact_runs_in_a_fresh_process_without_the_model_class` proves the standalone claim: the
model's source file is deleted, a fresh Python process blocks importing the model module and
torchvision, replaces `torch.export.export`, `torch.export.load` and `torch.jit.load` with functions
that raise, loads the artifact, runs batch 5 (exported at batch 2) and matches eager PyTorch.

## Native front end programs

Programs written with `@truestar.function` (docs/FRONTEND.md) carry real control flow: `cond` with
`then_branch` and `else_branch` regions, `loop` with a `body` region and a symbolic `trip_count`,
and nodes with several outputs (`extra_outputs`). These serialize through the same generic path:
regions are `program` attributes encoded recursively, extra outputs are stored next to the output,
and `trip_count` is a `dim` expression tree. Tests round trip a function with a data dependent
`if` that returns two values from its `cond`, and a loop over `T - 1` steps with a data dependent
`if` inside its body, compare the reference outputs bit for bit on both branches and at several
`T`, and run the saved loop program in a fresh process.

Symbolic expressions are stored as a tree of whitelisted sympy functions rather than as a string,
so loading a file never calls `eval` (sympy's string parser does). The printed form (`T - 1`) is
not stored; it is regenerated from the tree.

## Control flow and structured inputs (2026-11-24)

**Branches are verified on both sides.** `torch.cond` imports as an IR `cond` (docs/IR.md). Every
check records which side each `cond` took. A side no check takes is not left unverified: its IR
region runs on the operands that reached the `cond` and is compared, in float64, with torch's own
captured subgraph for that side. A disagreement refuses the export. The evidence holds the
coverage (`branches`: where, how often each side ran, the forced check and its error), and a
`cond` that never ran in any check is a warning. Real output (`Gate` returns
`torch.cond(x.mean() > 0, relu(lin(x)), -x * 0.5)`, exported from positive inputs only):

```
  batch symbol B in [1, inf]
  check example: shapes [[2, 3]], max rel error 0.00e+00 (bar 1e-09) PASS
  check second_batch: shapes [[7, 3]], max rel error 0.00e+00 (bar 1e-09) PASS
  branch cond (node %cond), t9.py:11 in forward: return torch.cond(x.mean() > 0, ...): then taken 2,
    else taken 0, else side never taken: checked against torch's branch, max rel error 0.00e+00
```

A permanent test breaks the reference `neg` (used only by the side never taken): the export is
refused naming that branch and nothing is written.

**Loops.** `while_loop` and `scan` import as IR `loop`; a scan RNN exports with a symbolic length
and batch (`f32[T, B, 4]`), checked at the example, at another `B` and at another `T` (0.7 s).

**Structured inputs and outputs.** Inputs may be tuples, lists, dicts, NamedTuples and dataclasses
of tensors, with keyword arguments; dataclasses are registered with torch.export by Truestar, the
user registers nothing. The input and output structures are stored in the evidence (`io`, plain
JSON, so a fresh process rebuilds them without importing user code: a NamedTuple or dataclass whose
module is not loaded comes back as a stand in with the same name and fields). `load(path)` called
with tensors in program order returns the flat list as before; called the way the model was
exported it returns the model's own structure (`Runner.call`). A mismatch names the leaf:
`batch.x: is torch.float64 but the artifact declares f32[B, 4]`, `scale: missing keyword argument
scale; unexpected keyword argument scal`, `extras: expected 2 items, got 1`. `None` entries in an
output stay `None`. `truestar run model.tsx inputs.pt` passes a dict in `inputs.pt` as keyword
arguments and prints each output by its path.

Tests: tests/test_export_structured.py.

## Python branches: truestar versus torch.onnx.export

`scripts/onnx_branch_compare.py` exports two small models with both ONNX exporters and with
truestar, then runs the ONNX files in onnxruntime (CPU) on inputs that take the other side of the
branch or use another batch size.

```python
import torch
from torch import nn

class ValueBranch(nn.Module):          # a Python if on a tensor value
    def forward(self, x):
        if x.mean() > 0:
            return torch.relu(x)
        return -x

class BatchBranch(nn.Module):          # a Python if on the batch size
    def forward(self, x):
        if x.shape[0] > 4:
            return x * 2
        return x + 1
```

Real output (2026-11-24, CPU):

```
torch 2.11.0+cu128, onnx 1.17.0, onnxruntime 1.20.1

[ValueBranch] torch.onnx.export(dynamo=False, dynamic batch=False) at x=3*ones(2, 3)
  export: wrote a file
  warning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as 
  onnxruntime x=-3*ones(2, 3), other side: WRONG, max abs error 3
  onnxruntime x=3*ones(7, 3), batch 7: raised InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: onnx::Relu_0 for the following indices

[BatchBranch] torch.onnx.export(dynamo=False, dynamic batch=True) at x=3*ones(2, 3)
  export: wrote a file
  warning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as 
  onnxruntime x=3*ones(2, 3), same batch: matches eager
  onnxruntime x=3*ones(7, 3), batch 7: WRONG, max abs error 2

[ValueBranch] torch.onnx.export(dynamo=True, dynamic batch=False) at x=3*ones(2, 3)
  export: raised TorchExportError: Failed to export the model with torch.export. This is step 1/3 of exporting the model to ONNX. Next steps: | - Modify the model code for `torch.export.export` to succeed. Refer to https://pytorch.org/docs/stable/generated/exportdb/index.html for more information. | - Debug `torch.export.export` and submit a PR to PyTorch.

[BatchBranch] torch.onnx.export(dynamo=True, dynamic batch=True) at x=3*ones(2, 3)
  export: wrote a file
  onnxruntime x=3*ones(2, 3), same batch: matches eager
  onnxruntime x=3*ones(7, 3), batch 7: WRONG, max abs error 2

[ValueBranch] truestar export at x=3*ones(2, 3), dynamic batch
  export refused: ValueBranch has a Python branch on a tensor's value, which torch.export cannot capture; exporting it would bake in whichever side the example inputs took.
    branch: onnx_branch_compare.py:27 in forward: if x.mean() > 0:
  No artifact was written. Fix: compute both sides and pick with torch.where, or use torch.cond (torch.export captures it; truestar imports it once IR control flow ops land).

[BatchBranch] truestar export at x=3*ones(2, 3), dynamic batch
  export refused: BatchBranch has code that only works for some batch sizes, so torch.export cannot keep the batch dim symbolic; exporting it would freeze the batch or bake in a branch.
    branch: onnx_branch_compare.py:36 in forward: if x.shape[0] > 4:  (guard: L['x'].size()[0] <= 4)
    torch.export: Not all values of B = L['x'].size()[0] in the specified range B <= 1024 satisfy the generated guard 2 <= L['x'].size()[0] and L['x'].size()[0] <= 4
  No artifact was written. Fix: make that code batch independent, export a fixed batch (--static-batch, dynamic_batch=False), or limit --max-batch to the range torch.export suggests above.
```

What this shows:

* **TorchScript ONNX exporter (`dynamo=False`)** traces one side of each branch. For `ValueBranch`
  it writes a file that is silently wrong on the other side (max abs error 3) and, with no
  `dynamic_axes`, rejects batch 7 at run time. For `BatchBranch` it writes a file with a symbolic
  batch that is wrong at batch 7 (max abs error 2). The only signal is a TracerWarning at export.
* **torch.export based ONNX exporter (`dynamo=True`, the default in torch 2.11)** refuses
  `ValueBranch` with torch.export's error. For `BatchBranch` it is worse than the tracer: it writes
  a file whose input is declared with a symbolic `B`, whose graph is a single `Add` (the `x + 1` side
  baked in), with no warning about the branch at all. torch.export alone refuses the same model with
  the same dynamic shape (checked with both `strict=False` and `strict=True`). The ONNX capture
  code (`torch/onnx/_internal/exporter/_capture_strategies.py`) catches that refusal, applies
  torch.export's suggested fix (`B = Dim('B', max=4)`) and exports again; the ONNX file then
  declares `x: [B, 3]` with no range, so the restriction to batch 2 to 4 is lost and batch 7 runs
  the wrong side. Inspected with `onnx.load`: input `x: [B, 3]`, nodes `['Add']`.
* **truestar export** refuses both, names the line of the branch, and writes nothing. A branch
  captured with `torch.cond` exports since 2026-11-24 with both sides verified (section above);
  `test_captured_branch_is_exported_correctly_or_refused` now requires the export and both sides
  to run correctly.

## torchvision classifiers

`scripts/export_zoo.py` exports every torchvision classifier up to 100M params with random weights
(64x64 inputs, 224 for ViT, Swin and MaxViT, 299 for Inception), each through the full export path
(float64 checks at batch 2 and 7), then loads the artifact and runs batch 3 through the runner
against eager float32. Result on 2026-11-24 (torch 2.11, torchvision 0.26, CPU):

**Rerun on 2026-11-24 after the op coverage work: 62 of 63 swept models export and round trip, 1
passes with a weak check, none is refused; 17 above 100M params were not swept.** googlenet now
passes too (2026-11-25, see below), so 63 of 63 export and round trip with an input dependent check. Worst float64
relative error in any check: 6.0e-15. Runner float32 output against eager float32: at most 1.7e-6.

| Status | Models |
|---|---|
| PASS (62) | the 46 below plus densenet 121, 161, 169, 201; inception_v3; shufflenet_v2 x0_5, x1_0, x1_5, x2_0; maxvit_t; swin t, s, b; swin_v2 t, s, b |
| WEAK (1) | googlenet, unchanged (see below); PASS since 2026-11-25: float64 4.9e-14, runner float32 against eager 1.6e-5, with measured batch norm statistics |
| REFUSED (0) | |

The first sweep, on 2026-11-24 before that work:

**46 of 63 swept models export and round trip, 1 passes with a weak check, 16 are refused with the
missing ops named; 17 above 100M params were not swept.** Worst float64 relative error in any
check: 6.4e-15. Runner float32 output against eager float32: at most 1.4e-6.

| Result | Models |
|---|---|
| PASS (46) | alexnet; convnext tiny, small, base; efficientnet b0 to b7, v2_s, v2_m; mnasnet 0_5, 0_75, 1_0, 1_3; mobilenet_v2, v3_small, v3_large; regnet_x 400mf, 800mf, 1_6gf, 3_2gf, 8gf, 16gf; regnet_y 400mf, 800mf, 1_6gf, 3_2gf, 8gf, 16gf; resnet 18, 34, 50, 101, 152; resnext50_32x4d, resnext101_32x8d, resnext101_64x4d; squeezenet 1_0, 1_1; vit_b_16, vit_b_32; wide_resnet50_2 |
| WEAK (1) | googlenet: with random weights at 64x64 its outputs are the same for every input row (to 1.5e-11 relative), so the check passes but proves little; the evidence says so |
| REFUSED (16) | densenet 121, 161, 169, 201 and inception_v3: `aten.avg_pool2d`; shufflenet_v2 (4): `aten.chunk`; maxvit_t: `avg_pool2d`, `chunk`, `einsum`, `index.Tensor`, `swapaxes`; swin t, s, b: `fill_`, `index.Tensor`, `masked_fill`, `new_zeros`, `pad`, `roll`; swin_v2 t, s, b: the swin list plus `clamp_min`, `expand_as`, `linalg_vector_norm`, `zero_` |
| not swept (17) | over 100M params: vgg (8), vit_l_16, vit_l_32, vit_h_14, convnext_large, efficientnet_v2_l, regnet_x_32gf, regnet_y_32gf, regnet_y_128gf, wide_resnet101_2 |

Every refusal is the importer saying which op is missing, with its source line, and nothing is
written. The op coverage work (docs/IR.md) added every op on that list; Swin's shifted window mask
(`mask[a:b, c:d] = k` on a fresh tensor) and Swin V2's bias edit are imported by rewriting the in
place change through slices as `slice_scatter`. The same 16 models also import and match PyTorch
in float64 from the core ATen graph (worst 1.5e-15 for densenet121, inception_v3, shufflenet_v2_x1_0,
maxvit_t, swin_t, swin_v2_t, checked once by hand).

Two findings from the sweep changed the exporter:

* torchvision zero initializes the ViT head, so with random weights every output is exactly 0 and
  any comparison passes. The first sweep counted that as a pass. The exporter now records a
  warning in the evidence (`warnings`, also printed) when every output is zero or when the outputs
  do not depend on the input row within the tolerance. The sweep gives zero initialized weight
  matrices random values before exporting.
* googlenet with random weights is input independent at 64x64 and is now reported as WEAK rather
  than PASS. The cause is its init, not the exporter: torchvision draws its convolutions with
  std 0.01, and with the default batch norm statistics (mean 0, variance 1) the activations shrink to
  zero through its 57 convolutions, so every row gives exactly the same output. The sweep now
  measures batch norm statistics on random inputs (train mode, cumulative average, buffers only) for
  a model whose outputs do not depend on the input row (googlenet was the only model with that warning in
  the 2026-11-24 rerun; the other 62 were not rerun), and googlenet then passes (`tests/test_export.py::test_googlenet_exports_with_an_input_dependent_check`).

The tests export and round trip resnet18, mobilenet_v3_small, mobilenet_v2, efficientnet_b0,
squeezenet1_1 and mnasnet0_5 with randomized batch norm statistics.

## Tests that prove the tests work

Manual checks on 2026-11-24, each reverted after the run:

* Skipping the file checksum check in `loads` failed 2 tests (the corrupted weight and corrupted
  header cases).
* Skipping the format version check failed `test_unknown_versions_are_loud`.
* Removing the second batch check from the exporter failed 3 tests, including
  `test_disagreement_only_at_another_batch_is_caught`, where the program is right at batch 2 and
  wrong at every other batch.

* Not storing `extra_outputs` failed `test_native_data_dependent_if_round_trips` (the second
  result of the `cond` was then undefined).

Permanent tests that break the system on purpose: a reference relu that is off by 0.1% makes
export refuse and write nothing; a reference linear that is wrong only above batch 2 passes the
example check and fails the second batch check; a header edit or a weight edit with a refreshed
file checksum is still caught by the program hash.

## Deploy targets: `--to onnx|litert|coreml|executorch` (2026-11-24)

The checker becomes a deploy tool: the deployable file is written from the verified Truestar
program, run in its own runtime, judged against Truestar's float64 reference, and refused when it
fails. Code: `src/truestar/export/deploy/` (lower.py, onnx_writer.py, tflite_writer.py,
coreml_writer.py, pte_writer.py, `__init__.py`). Tests: `tests/test_export_deploy.py`, models in
`tests/deploy_models.py`. Codes TS1410 to TS1419.

```
truestar export model.py --to onnx -o model.onnx          # symbolic batch B, verified at batch 7 too
truestar export model.py --to litert -o model.tflite      # static shapes, NHWC inside
truestar export model.py --to coreml -o model.mlpackage   # runtime check pending (needs a Mac)
truestar export model.py --to executorch -o model.pte     # portable kernels
truestar export model.py --to onnx -o m.onnx --runtime ort:cuda --force --json report.json
```

```python
from truestar.export.deploy import export_to
r = export_to(model, example, "model.onnx", to="onnx")   # DeployError (with .code) and nothing written on failure
print(r.to_text()); r.used["kernels"]; r.used["residency"]
```

Exit codes: 0 written (verified, or runtime check pending), 1 refused (nothing written) or written
with `--force` after a failed check, 2 usage error or missing optional package (TS022).

**What happens.**

1. The model is imported and verified against PyTorch in float64 exactly as for `.tsx` (example
   and second batch). A failure is TS1411.
2. Input independent nodes are folded once in float64 (a causal mask from `ones(t, t).tril()`,
   positions from `arange`), then the target writer emits the file. Truestar writes every format
   from its own IR: ONNX node by node (opset 17), LiteRT flatbuffers through the schema in the
   LiteRT wheel, Core ML as a hand encoded protobuf (Model spec 6, an ML Program in opset CoreML5,
   weights as immediate values, in the `.mlpackage` layout; no coremltools, which has no Windows
   wheel). ExecuTorch programs can only be produced by ExecuTorch's own serializer, so the IR is
   turned back into a module of plain torch ops (`IRModule`, one op per IR node) and exported
   through `executorch.exir`. An op without an exact form in the target is refused with the node
   and source line (TS1412); nothing is approximated.
3. The written bytes run in the runtime (`ort:cpu`, `tflite:interp`, `pte:cpu` by default, any
   runtime of that format from `truestar conform runtimes` with `--runtime`) on the example, three
   more samples (floats drawn around the example, integer ids redrawn in the example's range) and,
   for ONNX with a symbolic batch, batch 7. Each output is judged against Truestar's float64
   reference (`ir.run` of the verified program) with the float32 threshold `truestar check` uses
   (relative L2, at least 10x what a float32 PyTorch run of the model shows) and the per element
   bound of docs/TOLERANCE.md. The bound is derived on `IRModule`, not on the user's module: on
   `nn.TransformerEncoderLayer` the user's module calls the fused `aten._transformer_encoder_layer_fwd`,
   which has no error rule, and the bound derived there was broken by PyTorch's own float32 run (8
   elements, up to 17x). A bound that Truestar's own float32 run of the program breaks is dropped
   for that output, with a note.
4. The file's own math in float64: the ONNX reference evaluator on a float64 copy, Truestar's
   TFLite float64 rules (extended with every op the writer emits), and for Core ML the file decoded
   from its bytes and its MIL program evaluated in float64. A disagreement beyond the conversion bar
   is TS1418 and refuses the file before any runtime result counts. ExecuTorch has no float64
   evaluator; there only the runtime check judges.
5. What the runtime actually ran: ONNX Runtime from one profiled session (kernels after its graph
   fusions, their providers and element types, and whether profiling changed the outputs), LiteRT
   from the execution plan (XNNPACK partitions and every node left on a builtin kernel), ExecuTorch
   from the program's kernel and delegate calls. Nodes off the requested path are TS1416; fused or
   reduced precision execution is TS1417.
6. A failed runtime check (TS1414) or conversion check (TS1418) writes nothing unless `--force`
   (TS1415, the evidence says `"verified": false`). Otherwise the file is written atomically and
   read back byte for byte, with the evidence next to it in `<file>.truestar.json`. Core ML is
   written with TS1413 (runtime verification pending).

**Measured (2026-11-24, CPU, torch 2.11, onnx 1.17, onnxruntime 1.20.1, ai-edge-litert 2.4.0,
executorch 1.2.0 in a scratch location).** Random weights, randomized norm statistics. Worst
relative L2 error over the checks of the runtime output against Truestar float64 (bar 1e-4 in
every case, 0 elements beyond the derived bound in every case); in brackets the file's own float64
math against Truestar float64.

| model | onnx (ort:cpu) | litert (tflite:interp) | coreml | executorch (pte:cpu) |
|---|---|---|---|---|
| MLP 16-32-32-10, relu and gelu, batch 4 | 1.5e-7 [3.4e-9] | 1.8e-7 [2.7e-16] | pending [2.5e-16] | 1.9e-7 |
| resnet18, 64x64, batch 2 | 4.0e-7 [1.3e-12] | 8.4e-7 [4.3e-8] | pending [1.3e-12] | 9.1e-7 |
| TransformerEncoderLayer d32 h4, batch 2 x 8 | 1.1e-7 [1.2e-9] | 1.0e-7 [1.2e-9] | pending [1.2e-9] | 1.1e-7 |
| decoder (embeddings, 2 causal blocks, head), ids 2 x 8 | 1.6e-7 [1.6e-9] | 1.5e-7 [9.9e-10] | pending [9.9e-10] | 1.5e-7 |

Each export takes 0.2 to 4.2 s. The ONNX gelu is written as `0.5 x (1 + erf(x / sqrt 2))` with
float32 constants, which is where its 3.4e-9 conversion error comes from; the 4.3e-8 of resnet18 in
LiteRT is batch norm written as one multiply and add whose constants are rounded to float32 once.

What the runtimes did with the files, as reported:

* ONNX Runtime CPU rewrote the graphs: `FusedGemm` and its own `Gelu` kernel for the MLP (the erf
  decomposition was recognized), `FusedMatMul` and `BiasGelu` for the decoder, `ReorderOutput`
  (its NCHWc layout) for resnet18. Reported as TS1417; the outputs still pass.
* LiteRT XNNPACK left `PADV2` (the -inf padding of resnet18's max pool), `STRIDED_SLICE` with a
  shrink axis (the encoder's q, k, v selects), `GATHER` (the embedding) and `SELECT_V2` (the causal
  mask) on builtin kernels: TS1416 names them. The first writer padded convolutions with `PADV2`
  too, which XNNPACK also refused (5 nodes on builtin kernels, 6 partitions); the writer now uses
  `PAD` for zero padding (1 node left, 2 partitions).
* ExecuTorch ran every program on portable kernels (`aten::addmm.out`, `aten::convolution.out`,
  `aten::native_layer_norm.out` ...), no delegate.

**Tests that prove the tests work** (manual, reverted): accepting every runtime output failed
`test_runtime_mismatch_refused_unless_forced`; accepting every conversion error failed
`test_wrong_lowering_is_refused_and_nothing_written`. Permanent fault tests: a writer that emits
gelu as its tanh approximation is refused with TS1418 and writes nothing; a runtime whose outputs are
1e-3 off is refused with TS1414, and with `force=True` the file is written with TS1415 and
`"verified": false`.

**Tried and not shipped.** ExecuTorch with the XNNPACK delegate: the partitioner imports torchao,
and torchao 0.15 (the build next to executorch 1.2) refuses torch 2.11's C++ extensions and the
import then fails inside executorch (IndexError); no delegated file could be verified, so the option
is not offered. ExecuTorch 1.5.1 installs but its runtime fails on every input with torch 2.11
("tensor does not have a device"): the ExecuTorch package must match the torch build.

**Limits.** Core ML runtime verification needs macOS with coremltools; elsewhere it stays pending
(TS1413). On a Mac the written .mlpackage is compiled and run by Core ML itself with CPU_ONLY (the
verified, float32 run, judged against the float64 reference like every other runtime), and
CPU_AND_GPU and ALL are run too, with each unit's planned devices from MLComputePlan and its
difference from the CPU run in `used["coreml"]`. First run on a Mac mini M2 (macOS 26.6.2,
2026-11-25): every file failed to load, because MIL value names must be identifiers and parameter
names like `fc1.weight` were written as is; the writer now maps every name to an identifier. After
that, all four models of tests/deploy_models.py load and verify on all three units (errors 1.0e-7
to 7.4e-7 against a 1e-4 bar); resnet18 runs on the GPU with CPU_AND_GPU and ALL (4.1e-6 from the
CPU run), the MLP, encoder layer and decoder stay on the CPU on every unit. Core ML inputs that are int64 in the model are int32 in the file (Core
ML multiarrays have no int64). LiteRT, Core ML and ExecuTorch files have static shapes; ONNX keeps
the batch symbolic when the model allows it (else the batch is frozen with a note). Deploy targets
take positional tensor inputs; control flow (`cond`, `loop`) and training programs are refused.
Depthwise and grouped convolutions are refused for LiteRT. Reading a `.pte` back needs ExecuTorch's
`flatc` binary, which the Windows wheel does not ship, so the plan is read from the program object
at write time.

## Limits

* The default execution is the float64 reference interpreter: correct, slow, CPU only. The artifact format is
  ready for a compiled executor. Since 2026-11-24 one exists: `load(path, executor="cuda")`
  (or `"cpu"`, `"cuda:float16"`, `"cpu:float16"`) runs the artifact through Truestar kernels,
  see docs/COMPILER.md.
* Structured outputs that are not tensors, `None`, tuples, lists, dicts, NamedTuples or dataclasses
  (a registered custom container other than a mapping with string keys) are refused. A Hugging
  Face style mapping output comes back as a dict.
* Only dim 0 is symbolic automatically; other dims through `dynamic_dims` / `--dynamic`, by
  top level argument name (a dim inside a nested argument is not addressable yet).
* `torch.map` and `associative_scan` are refused. A loop's trip count is not forced the way a
  branch side is: a body is checked at the lengths the checks reach.
* The never taken side of a branch is compared with torch's captured subgraph, not with the eager
  module, and only on the operands the other side saw first.
* Verification covers the given example inputs and one generated batch. It proves the artifact
  matches PyTorch on those inputs, not on every input.
* The second batch is generated from the example rows; a model whose input domain is narrow (token
  ids, masks) gets the rows unchanged, floats get small noise.
* Artifacts are written and read on little endian hosts only (every current target).
