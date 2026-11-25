# Agent interface

Much ML debugging is now done by AI coding agents. This page is the contract they can rely on:
a versioned JSON document from `truestar check --format agent`, stable codes for every finding, and an
MCP server (`truestar mcp`) that exposes the same checks as tools.

## Command line

```
truestar check model.py --format agent                 # JSON on stdout, nothing else
truestar check model.py --backends cpu,cuda --format agent --repro-dir out/
truestar backends                                      # backends on this machine, as JSON
truestar codes                                         # every code below, as JSON
truestar mcp                                           # MCP server on stdin/stdout
```

`model.py` defines `get_model()` (an `nn.Module`) and `get_inputs()` (a tensor or a tuple of forward
arguments), and optionally `get_kwargs()`.

Exit codes are the same in every format:

| exit | meaning |
|---|---|
| 0 | every backend matches the float64 CPU reference |
| 1 | at least one backend is divergent or suspect |
| 2 | usage error: bad arguments, model file invalid, requested backend missing, reference failed |

With `--format agent`, stdout holds exactly one JSON document even on exit 2, and anything the user's
model prints is sent to stderr instead. The default output stays the human text report.

## Schema (schema_version 1)

Compatibility rules: `schema_version` is bumped only when a field is removed or changes meaning.
New fields may appear at any time, so ignore fields you do not know. The JSON is strict: there is
never `NaN` or `Infinity`. A number that is not finite or was not measured is `null`, and the
`nonfinite_mismatch` and `shape_mismatch` flags next to it say why.

Top level:

| field | type | meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `tool`, `command` | str | `"truestar"`, `"check"` |
| `truestar_version`, `torch_version` | str | versions that produced the document |
| `status` | str | `ok`, `suspect`, `divergent`, or `error` (exit 2) |
| `exit_code` | int | the process exit code, 0, 1 or 2 |
| `model` | object | `name`, `file` (absolute path), `dtype` |
| `reference` | str or null | `"CPU float64"`, or the degraded reference (see TS003) |
| `codes` | list of str | every distinct code in the document, in order of appearance |
| `issues` | list | `{code, slug, severity, backend, message}` for each finding |
| `backends` | list | one entry per backend checked, below |
| `skipped` | list | `{name, reason, code: "TS001", severity: "info"}` for backends not present |
| `next_actions` | list | what to do next, below; empty when status is `ok` |
| `available` | list of str | only on TS001 errors: backends that do exist here |
| `stderr_tail` | str | only from the MCP `check_model` tool, when the check wrote to stderr |

Each entry of `backends`:

| field | type | meaning |
|---|---|---|
| `name`, `device`, `description` | str | for example `cuda`, `cuda:0`, `NVIDIA GeForce RTX 5070 Ti, CUDA 12.8` |
| `status` | str | `ok`, `suspect` or `divergent` |
| `codes` | list of str | codes for this backend |
| `output` | object or null | final output: `rel_err`, `max_abs`, `nonfinite_mismatch`, `shape_mismatch` |
| `decisions` | object | `total`, `flips`, `near_tie_flips` (per input argmax changes, classifier outputs only) |
| `blamed_module` | object or null | the first module whose own error is over threshold, below |
| `modules_checked` | int | modules replayed alone on this backend |
| `modules_over_threshold_count` | int | how many modules are over threshold |
| `modules_over_threshold` | list | first 5: `key`, `type`, `status`, `rel_err` (own), `accumulated_rel_err`, `threshold` |
| `repro` | object or null | `path`, `command`, `control_command`, `device` |
| `error` | str or null | the exception when the model failed on this backend (TS002) |
| `stack` | object or null | compiler and runtime backends only (docs/BACKENDS.md): `kind` (`inductor`, `ort`, `trt`), `option`, `compute_dtype`, `engine_seconds`, `localization` (how the mismatch was localized, or why it was not needed), `localize_seconds`, `compiles`, `op_compiles`, and when present `providers`, `graph_breaks`, `blamed_kernels` (Inductor: each kernel of the blamed module with its ATen ops) or `blamed_engine_layers` (TensorRT layer names) |

For a stack backend, `modules_checked` counts the whole model engine plus each module compiled alone
during localization, and `blamed_op.kernel` (Inductor) is the fused kernel that computes the op:
`kernel`, `aten` (its ATen ops), `extern` (a library call such as `extern_kernels.mm`). The repro's
`control_command` runs the same inputs through eager PyTorch on CPU.

`blamed_module`:

| field | type | meaning |
|---|---|---|
| `name`, `key`, `type` | str | module path in the model (`layer3.1.bn2`), call key (`layer3.1.bn2#0`), class name |
| `leaf` | bool | true for a module with no children |
| `own_code` | bool | true when the error is in this module's own forward code, outside its submodules |
| `status` | str | `divergent` or `suspect` |
| `rel_err`, `threshold`, `ratio` | float or null | the module's own relative L2 error (replayed alone with reference inputs), its threshold, and their ratio |
| `nonfinite_mismatch`, `shape_mismatch` | bool | NaN/Inf where the reference has none, or a shape change |
| `error` | str or null | the exception if the module raised when replayed alone (TS014) |
| `source` | object or null | where the module's class is defined: `file`, `line`, `qualname`, `user_code` |
| `user_source` | object or null | the nearest module, itself or an enclosing one, defined in user code (not torch or another installed library), plus its `module` path |

For a blamed `nn.Linear` inside the user's `MLP`, `source` points into torch and `user_source` points
at `class MLP`. Both come from `inspect.getsourcefile` and `inspect.getsourcelines`; either can be null
for classes built at run time.

Each entry of `next_actions` has `kind`, `description` and `command` (a shell command or null), and may
carry `backend`, `file`, `line`, `current` or `repro_dir`. Kinds:

| kind | when |
|---|---|
| `confirm_repro` | a repro was written: run it, exit 1 confirms the divergence on its own |
| `control_repro` | run the same repro on CPU, exit 0 shows the kernel on the backend is at fault |
| `inspect_source` | `file` and `line` of the user class to read |
| `pin_torch` | the blamed module is a library class: test other torch releases, `current` is the installed one |
| `report_upstream` | library class with a repro: report with the `repro_dir` folder |
| `injected_fault` | the blamed module is a `truestar.testing` fault injector |
| `inspect_failure` | the model raised on this backend (TS002) |
| `wrap_in_module` | the output diverges but no module is to blame (TS012) |
| `check_control_flow` | different modules ran on the backend (TS004) |
| `check_flips` | predictions changed (TS013) |
| `note_reference_degraded` | the float64 reference could not run (TS003) |
| `use_available_backends` | a requested backend is missing (TS001) |
| `fix_model_file` | the model file is invalid (TS020) |

## Codes

Codes never change meaning once released. They appear in the agent JSON, in the full `--json` report
(`codes` and `issues`) and in the text report, next to the line they explain.

| code | severity | slug | meaning |
|---|---|---|---|
| TS001 | error | backend_unavailable | a requested backend is not on this machine (exit 2); `info` when only listed in `skipped` |
| TS002 | error | model_failed_on_backend | the model raised on this backend |
| TS003 | warning | reference_degraded | the float64 reference failed, so CPU bugs cannot be detected |
| TS004 | warning | module_set_differs | different modules ran on the backend than on the reference |
| TS005 | error | reference_failed | the model could not run on the CPU reference at all (exit 2) |
| TS010 | error | divergence | a module's own output is wrong, not just imprecise |
| TS011 | warning | suspect | a module's own error is over threshold but under the divergent cutoff |
| TS012 | error | output_divergent_unattributed | the output diverges but no single module is to blame |
| TS013 | warning | prediction_flips | per input predictions changed beyond near ties |
| TS014 | error | module_replay_failed | a module raised when replayed alone on the backend |
| TS020 | error | model_file_invalid | model file missing, failing to import, or lacking get_model() / get_inputs() (exit 2) |
| TS021 | error | usage_error | invalid command line or tool arguments (exit 2) |
| TS022 | error | optional_dependency_missing | a requested path needs an optional module that is not installed (numba for CPU kernels: `pip install "truestar[cpu-kernels]"`; onnx, ai-edge-litert or executorch for `truestar conform` on .onnx, .tflite or .pte); raised as `truestar.deps.MissingDependencyError` |
| TS400 | warning | stack_cpu_fallback | a GPU runtime (ORT CUDA) ran floating point nodes on the CPU with no error; the message names each node and its module |
| TS401 | error | stack_provider_missing | the requested execution provider did not load, so the whole model ran on the CPU |
| TS402 | error | stack_fusion_only | the compiled whole model diverges but every module compiled alone matches (fusion across modules, or code between them) |
| TS403 | info | stack_graph_breaks | torch.compile split the model into several graphs; code between them ran eagerly |
| TS560 | error | webgpu_session_on_cpu | WebGPU backend: ONNX Runtime could not create the WebGPU EP and ran the graph on its CPU EP; nothing is reported as WebGPU (docs/WEBGPU.md) |
| TS561 | warning | webgpu_nodes_on_cpu | WebGPU backend: compute nodes ran on the CPU EP, named; the result is WebGPU plus CPU |
| TS562 | error | onnx_export_failed | WebGPU backend: the model or a module could not be exported to ONNX, so it was not checked there |
| TS563 | error | onnx_export_differs | WebGPU backend: the exported graph is already off on ONNX Runtime's CPU EP, so the export changes the result, not WebGPU |
| TS564 | warning | observer_effect | WebGPU backend: the tapped or profiled run gave a different final output than the plain run; per module numbers from it are untrusted |
| TS565 | warning | webgpu_nondeterministic | WebGPU backend: two identical runs gave different outputs |
| TS566 | error | dawn_probe_divergent | `truestar.webgpu.dawn`: a single op compute shader on Dawn is off beyond the bar for its precision; reserved, not emitted yet (the `webgpu/dawn.py` docstring promises it) |
| TS420 | error | memory_budget_exceeded | compiler: a program's planned peak memory exceeds `memory_budget`; raised as `truestar.compiler.memory.MemoryBudgetError` before any launch, naming the line at the peak and the largest batch that fits (docs/MEMORY.md) |
| TS421 | error | recompute_not_bitwise | compiler: a recomputed value is not bitwise equal to the stored one; raised as `truestar.compiler.runtime.RecomputeMismatchError`, result not returned (docs/MEMORY.md) |
| TS160 | error | port_divergence | `truestar port`: a candidate module computes differently from its mapped reference module on the same inputs |
| TS161 | warning | port_suspect | `truestar port`: a candidate module's own difference is above float32 noise but below the divergent cutoff |
| TS162 | error | weights_unmatched | `truestar port`: a weight one forward reads has no counterpart in the other model (a dropped adapter) |
| TS163 | error | weights_unused | `truestar port`: a weight present in both models is never read by one forward |
| TS164 | info | weights_layout_changed | `truestar port`: weights matched through a transpose, permutation, fused slice or merged adapter |
| TS165 | warning | weight_values_differ | `truestar port`: weights matched by shape or dtype cast only, values differ |
| TS166 | error | port_output_divergent_unattributed | `truestar port`: outputs differ but no mapped module is to blame |
| TS167 | info | modules_unmapped | `truestar port`: modules with no counterpart (inlined or unmatched) |
| TS168 | error | port_replay_failed | `truestar port`: a candidate module raised when replayed with the reference inputs |
| TS169 | warning | weights_dead | `truestar port`: a weight in one model only that its forward never reads |

`truestar port reference.py candidate.py --format agent` uses the same envelope with
`"command": "port"`; its fields are documented in [PORT_CHECK.md](PORT_CHECK.md).
| TS090 | info | remedy_verified | `truestar fix`: a remedy was applied and the rerun check is clean |
| TS091 | warning | remedy_partial | `truestar fix`: remedies lower the error or flips measurably, check not clean |
| TS092 | error | remedy_unresolved | `truestar fix`: no remedy cleared the blamed module; report upstream |
| TS093 | error | remedy_not_applicable | `truestar fix`: no single ATen op is to blame, so an op remedy cannot apply |
| TS094 | info | precision_plan_verified | `truestar fix --int8`: kept layers bring simulated flips under the target |
| TS095 | warning | precision_plan_unmet | `truestar fix --int8`: the flip target was not reached in simulation |

## Example

`truestar check faulty.py --backends cpu,cuda --format agent` on a model whose `Scaler` module is
wrong only on CUDA (shortened: the cpu backend entry and `skipped` are cut):

```json
{
  "schema_version": 1,
  "tool": "truestar",
  "command": "check",
  "truestar_version": "0.1.0",
  "torch_version": "2.11.0+cu128",
  "status": "divergent",
  "exit_code": 1,
  "model": {"name": "faulty", "file": "/work/faulty.py", "dtype": "float32"},
  "reference": "CPU float64",
  "codes": ["TS010"],
  "issues": [
    {"code": "TS010", "slug": "divergence", "severity": "error", "backend": "cuda",
     "message": "module bad#0 (Scaler) own error 5.00e-01 vs threshold 5.0e-03"}
  ],
  "backends": [
    {
      "name": "cuda",
      "device": "cuda:0",
      "description": "NVIDIA GeForce RTX 5070 Ti, CUDA 12.8",
      "status": "divergent",
      "codes": ["TS010"],
      "output": {"rel_err": 0.47, "max_abs": 0.61, "nonfinite_mismatch": false, "shape_mismatch": false},
      "decisions": {"total": 16, "flips": 0, "near_tie_flips": 0},
      "blamed_module": {
        "key": "bad#0", "name": "bad", "type": "Scaler", "leaf": true, "own_code": false,
        "status": "divergent", "rel_err": 0.5, "threshold": 0.005, "ratio": 100.0,
        "nonfinite_mismatch": false, "shape_mismatch": false, "error": null,
        "source": {"file": "/work/faulty.py", "line": 6, "qualname": "truestar_user_model.Scaler", "user_code": true},
        "user_source": {"file": "/work/faulty.py", "line": 6, "qualname": "truestar_user_model.Scaler", "user_code": true, "module": "bad"}
      },
      "modules_checked": 4,
      "modules_over_threshold_count": 1,
      "modules_over_threshold": [
        {"key": "bad#0", "type": "Scaler", "status": "divergent", "rel_err": 0.5,
         "accumulated_rel_err": 0.5, "threshold": 0.005}
      ],
      "repro": {
        "path": "/work/truestar_repro/cuda_bad_0/repro.py",
        "command": "\"/usr/bin/python3\" \"/work/truestar_repro/cuda_bad_0/repro.py\"",
        "control_command": "\"/usr/bin/python3\" \"/work/truestar_repro/cuda_bad_0/repro.py\" cpu",
        "device": "cuda:0"
      },
      "error": null
    }
  ],
  "next_actions": [
    {"kind": "confirm_repro", "backend": "cuda",
     "description": "Run the repro on cuda; exit code 1 confirms the divergence on its own.",
     "command": "\"/usr/bin/python3\" \"/work/truestar_repro/cuda_bad_0/repro.py\""},
    {"kind": "control_repro", "backend": "cuda",
     "description": "Run the same repro on CPU; exit code 0 shows the module itself is fine and the cuda kernel is at fault.",
     "command": "\"/usr/bin/python3\" \"/work/truestar_repro/cuda_bad_0/repro.py\" cpu"},
    {"kind": "inspect_source", "backend": "cuda", "file": "/work/faulty.py", "line": 6, "command": null,
     "description": "The blamed module bad#0 is your class truestar_user_model.Scaler. Check its forward for code that behaves differently on cuda."}
  ]
}
```

## MCP server

`truestar mcp` speaks the Model Context Protocol over stdio (newline delimited `JSON-RPC 2.0`). It uses
the official `mcp` Python package when that is installed and otherwise a builtin server with no
dependencies; `--impl sdk` or `--impl builtin` forces one. The sdk transport needs mcp 1.x; with mcp 2.x
installed, auto picks builtin and `--impl sdk` exits 2 naming the installed version. Both expose the same
tools:

| tool | arguments | returns |
|---|---|---|
| `list_backends` | none | `{backends: [...], torch_version}` |
| `check_model` | `model_path` (required), `backends` (list), `repro_dir`, `write_repro` (default true), `timeout_s` | the agent JSON above |
| `run_repro` | `repro_path` (required), `device`, `timeout_s` | `{exit_code, reproduced, passed, stdout, stderr, command}` |
| `grad_check` | `model_path` (required), `backends`, `steps` (default 0), `lr`, `repro_dir`, `write_repro`, `timeout_s` | the `--backward` agent JSON (docs/GRAD_CHECK.md) |
| `port_check` | `reference_path`, `candidate_path` (both required), `timeout_s` | the `port` agent JSON (docs/PORT_CHECK.md) |
| `invariance_check` | `model_path` (required), `backends`, `sizes` (list of ints, each at least 2), `rows`, `decode`, `seq_pad`, `try_flags` (default true), `timeout_s` | the `invariance` agent JSON (docs/INVARIANCE.md) |
| `record` | `model_path`, `output_path` (both required), `backend`, `ops` (`none`, `all` or module names), `outputs_only`, `max_mb`, `timeout_s` | the `record` agent JSON: path, size, sha256, fingerprint |
| `diff` | `recorded_path` (required), `current_path` (omit to replay here), `model_path`, `backend`, `repro_dir`, `write_repro`, `strict`, `timeout_s` | the `diff` agent JSON (docs/RECORD_DIFF.md) |
| `verify_kernel` | `reference_path`, `candidate_path` (both required), `device`, `dtypes` (list), `shapes` (default true), `timeout_s` | the `kernel` agent JSON (docs/KERNEL_CHECK.md) |
| `export_model` | `model_path`, `output_path` (both required), `static_batch`, `max_batch`, `second_batch`, `tolerance`, `decompose`, `timeout_s` | `status` `written` or `refused`, `artifact` (`path`, `bytes`, `format_version`, `program_hash`, `evidence`), `stdout`, `stderr`, `next_actions` |

Results carry the document both as `structuredContent` and as JSON text in `content`. `isError` is
true only when the tool could not do its job (exit 2, bad arguments, a crashed repro), never because a
divergence was found or an export was refused. `check_model`, `grad_check` and `diff` write repros
to `truestar_repro/` next to the model file (for `diff`, next to the recording) by default. Every
tool runs the CLI in a child process, so a crash or a print in the model cannot break the protocol
stream, and the server itself never imports torch.

Tools only run or overwrite files Truestar wrote: `run_repro` runs only `repro.py` files written by
`truestar check`; `diff` reads only files that start with the `.tsrec` magic (a replay imports the
model file a recording names, so a foreign file is refused before anything runs); `record` and
`export_model` refuse to overwrite an existing file that is not a recording or an artifact. Model
files are the one exception by design: the tools import the `model_path` the agent passes, as the
CLI does.

The `export_model` document uses the common envelope with `"command": "export"`. `truestar export`
itself has no `--format agent`; the tool reads the artifact header (docs/EXPORT.md, file format) to
fill `artifact`, and a refusal comes with `next_actions` kind `fix_export` and the reason in
`stderr`. After a successful export, `next_actions` holds `run_artifact` with a `truestar run`
command.

Register it with an agent, for example in an MCP client config:

```json
{"mcpServers": {"truestar": {"command": "python", "args": ["-m", "truestar", "mcp"]}}}
```

## Op level fields (added 2026-11-24)

Each backend object also carries `blamed_op` (the exact ATen op inside the blamed module: `op`,
`module`, `index`, `total`, `inputs`, `rel_err`, `threshold`, `ratio`, `runs_in_reference`) and
`op_repro` (same shape as `repro`, pointing at `op_repro.py`). Both are null when op drilldown found
nothing. When an op repro exists, `next_actions` starts with `confirm_op_repro`. These fields are
additive, so `schema_version` stays 1.

## Invariance (added 2026-11-24)

`truestar invariance model.py --format agent` prints a document of the same envelope with
`command: "invariance"`; its fields, next action kinds and codes are in docs/INVARIANCE.md.

| code | severity | slug | meaning |
|---|---|---|---|
| TS050 | warning | batch_variant | a row is not bitwise identical inside other batches, positions or padding |
| TS051 | warning | nondeterministic | the same row run twice alone gives different bits |
| TS052 | error | batch_dependent | a row's output depends on other rows beyond float tolerance |
| TS053 | warning | invariance_flips | the non invariance flips argmax predictions or greedy tokens |
| TS054 | warning | invariance_unattributed | not invariant, but no single module or op was named |
## Record and diff (added 2026-11-24)

`truestar record model.py -o run.tsrec --format agent` and `truestar diff run.tsrec --replay --format
agent` use the same envelope with `command` set to `record` or `diff`, codes TS060 to TS071 and their
own `next_actions` kinds (`pin_version`, `report_upstream` with an op repro, `restore_precision_flag`
and others). The fields are listed in docs/RECORD_DIFF.md. These are additive, so `schema_version`
stays 1.

| code | severity | slug | meaning |
|---|---|---|---|
| TS060 | error | env_divergence | a module's own output changed between environments beyond any precision explanation |
| TS061 | warning | env_precision_change | changed within the band of a precision setting that differs (TF32) |
| TS062 | info | env_numerical_noise | changed within the dtype band |
| TS063 | info | env_changed | the environment fingerprints differ |
| TS064 | error | module_state_differs | different parameters or buffers than recorded |
| TS065 | error | record_invalid | the recording is missing, truncated, modified or of an unknown version (exit 2) |
| TS066 | warning | call_set_differs | different module calls ran than recorded |
| TS067 | warning | replay_failed | a recorded module call could not be replayed |
| TS068 | info | isolation_limited | changed modules saw different inputs; replay isolates every module |
| TS069 | error | inputs_differ | the two recordings fed different model inputs |
| TS070 | error | record_too_large | the recording would exceed its size limit (exit 2) |
| TS071 | warning | instrumented_path_differs | recording hooks changed the model's code path |
## Backward check (added 2026-11-24)

`truestar check model.py --backward [--steps N] --format agent` emits the same envelope with
`"mode": "backward"` and per backend `blamed_module.gradient`, `norm_ratio`, `cosine`,
`forward_status`, `backward_only`, `blamed_op.node`, `grad_norm_ratio`, `trend` and
`injected_faults`. Full field list and method: docs/GRAD_CHECK.md. Additive, so `schema_version`
stays 1. Codes:

| code | severity | slug | meaning |
|---|---|---|---|
| TS180 | error | grad_divergence | a module's own backward is wrong while its own forward matches |
| TS181 | warning | grad_suspect | own gradient error over threshold, under the divergent cutoff |
| TS182 | error | grad_forward_divergent | gradients diverge and the module's own forward diverges too |
| TS183 | warning | grad_norm_drift | gradient norm drifts from a correct run over the training steps |
| TS184 | error | backward_replay_failed | the module raised when its backward was replayed alone |
| TS188 | warning | grad_near_tie_mask_flip | a parent is off float64 only through ReLU masks that flip at near ties (every leaf clean, a rounded run with the backend's masks matches) |
## `truestar fix` documents (added 2026-11-24)

`truestar fix model.py --format agent` prints one document with the same envelope (`schema_version`
1, `tool`, `truestar_version`, `torch_version`, `status`, `exit_code`, `model`, `codes`,
`issues`, `next_actions`) and `command` set to `"fix"`. `status` is `clean`, `resolved`,
`improved`, `not_applicable` or `unresolved`; exit code 0 for clean and resolved, else 1.
Each entry of `backends` has `outcome`, `reason`, `before` and `after` (status, output error,
flips, blamed module and op with their own errors), `remedies` (op, device type, decomposition),
`attempts` (every candidate with its op level error, rerun check status and how many calls it
replaced), `cost` (`baseline_ms`, `remedied_ms`, `overhead_pct`, `reps`) and `upstream` (paths
of `op_repro`, `repro` and a drafted report when one exists). `next_actions` kinds:
`apply_remedy`, `apply_partial_remedy`, `report_upstream`, `inspect_source`. With `--int8`
the document carries `precision_plan` instead of `backends`, with next action
`keep_layers_in_float` or `raise_target_or_retrain`. Details: [FIX.md](FIX.md).
## RL pair check (added 2026-11-24)

`truestar rlpair pair.py --format agent` emits the same envelope with `"command": "rlpair"`:
`logprobs` (statistics, importance ratio, worst positions), `blamed_module` (with `kind`,
`first_position`, `first_step`, `input_diffs`, `source`), `blamed_op` (with `sampler_scalars`,
`trainer_scalars` and a `hint`), `temperature`, `weights`, `batch_composition`, `batch_invariant`
and `next_actions`. Fields and method: docs/RL_PAIR.md. Additive, so `schema_version` stays 1.

| code | severity | slug | meaning |
|---|---|---|---|
| TS190 | error | rl_module_divergence | a module computes differently in the sampler than in the trainer for the same positions |
| TS191 | warning | rl_precision_mismatch | logprobs differ only as far as the sampler dtype explains |
| TS192 | info | rl_logprob_noise | logprobs differ only at rounding level |
| TS193 | error | rl_mismatch_outside_modules | every module matches; the gap is outside the modules (temperature fitted) |
| TS194 | error | rl_sampler_weights_differ | sampler parameters are not the trainer's rounded to the sampler dtype |
| TS195 | warning | rl_sampler_batch_variant | the sampler's logprobs for a row change with its batch |
| TS196 | info | rl_modules_unmapped | module calls that could not be lined up with sampler positions |
| TS197 | error | rl_inputs_differ | a module gets integer inputs (positions, masks) that differ at the same position |
| TS198 | warning | rl_ratio_outside_clip | importance ratios outside the clip range |
| TS185 | error | rl_kernel_swapped | with `--kernels rl`, a module runs another kernel for one op in the sampler than in the trainer; the op is named |
| TS186 | error | rl_tp_mismatch | trainer logprobs at TP 1 and the simulated TP size are not bitwise equal |
| TS187 | error | rl_kernels_not_bitwise | with `--kernels rl`, same kernels in every module but logprobs not bitwise equal |
| TS199 | error | rl_pair_invalid | the pair file or the trainer and sampler outputs are invalid (exit 2) |
## Sentinels and `truestar profile` (added 2026-11-24)

Compiled programs with `sentinels=` (docs/COMPILER.md, Sentinels) raise `truestar.compiler.SentinelError`;
`SentinelError.to_dict()` gives `code`, `check`, `op`, `value`, `source`, `file`, `line`, `index`,
`bad_value`, `count`, `numel`, `step`, `fused_ops`, `region`, `iterations`. Index bounds are checked
in every compiled program and raise the same error with TS504.

`truestar profile model.py --format agent` prints one document with the usual envelope and
`"command": "profile"`: `totals`, `accounting` (`steps_run`, `steps_timed`, `unaccounted`),
`first_call`, `steps` (per step `kind`, `shape`, `ops`, `fused`, `lines`, `calls`, `gpu_ms`, `share`,
`bytes`, `gb_per_s`, `flops`, `tflop_per_s`), `lines` and, with `--crosscheck`, `crosscheck`.
Fields and method: [PROFILE.md](PROFILE.md). Exit code 0, or 1 when a step run has no valid time.

| code | severity | slug | meaning |
|---|---|---|---|
| TS500 | error | sentinel_nan | a compiled in check found NaN; op, line and first offending element named |
| TS501 | error | sentinel_inf | an infinity no infinite constant explains (register overflow, log(0), an infinite input) |
| TS502 | error | sentinel_overflow | a float16 or bfloat16 value finite in float32 became Inf on store |
| TS503 | error | sentinel_div_by_zero | a division had a zero denominator (float or integer) |
| TS504 | error | sentinel_index_bounds | an index out of bounds or repeated where it must be unique; nothing was read or written out of bounds |
| TS510 | error | profile_events_unaccounted | a step run has no valid GPU time; the profile says it is incomplete |
| TS511 | warning | profiler_dropped_events | torch.profiler saw fewer kernel launches than the runtime made |
| TS512 | info | profile_fallback_time | steps run through the float64 fallback; their time is listed |
| TS800 | error | graph_capture_refused | a compiled program could not be recorded into a CUDA graph; nothing ran in its place |
| TS801 | error | graph_replay_mismatch | the recorded graph gave other bits than the verified call; refused |
| TS802 | error | graph_bound_input_swapped | an input bound by address was not the tensor the graph was recorded with |

## `truestar range` documents (added 2026-11-24)

`truestar range model.py --target fp16 --format agent` emits the same envelope with `"command":
"range"`. `status` is `ok` (exit 0), `at_risk`, `overflow` or `remedied` (exit 1); usage errors are
exit 2 with TS020 or TS021. Fields: `target` (name, format, max, min_normal, significand_bits,
decomposed, nan_to_inf), `headroom_bits`, `calibration_inputs`, `counts` (float values, proven safe
statically, safe on the calibration inputs only), `twin` and `target_run` (rel_err, cosine_min,
cosine_mean, nonfinite, flips, real_flips, rows), `sites` and `info` (each: kind, value, op, hidden,
aten, module, module_type, file, line, code, peak, headroom_bits, static_peak,
static_headroom_bits, proven_safe_statically, share_below_normal, precision_bits, origin, effect,
remedy), `remedies` (remedy, passed, result, detail, cost, exact_rel_err) and `chosen_remedy`.
`next_actions` kinds: `inspect_source`, `apply_rescale`, `apply_fp32`, `apply_clamp`,
`keep_float32`, `add_calibration`. Method: docs/RANGE.md. Additive, so `schema_version` stays 1.

| code | severity | slug | meaning |
|---|---|---|---|
| TS300 | info | range_clean | every value fits and the emulated target matches its range unlimited twin |
| TS301 | error | range_overflow | an op's output exceeds the format's maximum on a calibration input |
| TS302 | error | range_hidden_overflow | an intermediate inside a composite op the target runs decomposed exceeds the range |
| TS303 | warning | range_near_overflow | a value within the headroom of the maximum, not proven safe by its static bound |
| TS304 | warning | range_underflow | most nonzero values of an output fall below the smallest normal value |
| TS305 | warning | range_precision_loss | an op replayed alone with rounded inputs keeps fewer than 2 correct bits |
| TS306 | error | range_target_diverges | the emulated target is worse than its twin (non finite, flips, larger error) |
| TS307 | info | range_remedy_verified | a remedy passed the same test as the twin in emulation |
| TS308 | error | range_remedy_unresolved | no remedy passed, or no site explains the failure |
| TS309 | info | range_event_benign | saturated constants, overflow inside fused kernels, or events without effect |

## Precision manifest in `truestar check` (added 2026-11-24)

Each entry of `backends` (and of `results` in the full `--json` report) carries `precision`:
`dtype`, `ops` (for `linear`, `bmm` and `conv2d`: `input_bits` the kernel really used,
`dtype_bits`, `effective` such as `float32`, `tf32` or `bf16 inputs`, `subnormals` `kept` or
`flushed`, and for 16 bit dtypes `accumulator`) and `flags` (the settings meant to control it).
Measured with crafted inputs on the backend, see docs/RANGE.md. Additive, so `schema_version`
stays 1.
## Kernel check (added 2026-11-24)

`truestar kernel ref.py cand.py --format agent` emits the same envelope with `"command": "kernel"`:
`reference`, `candidate` (with `source`), `device`, `variants` (one entry per input variant: `name`,
`kind` among base, shape, family, stride, dtype, repeat; `status`, `code`, `message`, `worst_ulps`,
`worst_ratio`, `location`), `variant_counts`, `first_failure`, `blamed_op` (`op`, `index`, `total`,
`message`) or `op_note`, `seconds`. `next_actions` kinds: `inspect_candidate` (file and line of the
candidate, the code and variant), `inspect_op`, `make_deterministic`, `rerun`. Method and limits:
docs/KERNEL_CHECK.md; the tolerance: docs/TOLERANCE.md. Additive, so `schema_version` stays 1.

| code | severity | slug | meaning |
|---|---|---|---|
| TS200 | error | kernel_divergence | the output is further from float64 than a correct implementation can be (derived tolerance) |
| TS201 | error | kernel_shape_failure | the candidate raises on a shape the reference accepts |
| TS202 | warning | kernel_nonfinite_mismatch | NaN or infinity in the inputs is handled differently from the float64 reference while every finite element is right; a verdict of its own, not a wrong number |
| TS203 | warning | kernel_nondeterministic | the same inputs give different bits from run to run |
| TS204 | error | kernel_stride_dependent | wrong on non contiguous inputs while contiguous inputs are right |
| TS205 | error | kernel_unwritten_output | output elements never written (NaN from the allocator shows through) |
| TS206 | error | kernel_mutates_inputs | the candidate writes into its inputs |
| TS207 | error | autotune_config_rejected | `truestar.verified_autotune`: a config gives a wrong result and was removed |
| TS540 | info | intref_convention_matched | `truestar intref`: the device output equals the integer reference under the named convention(s) bit for bit (docs/INTREF.md) |
| TS541 | error | intref_no_convention | the device output matches no known rounding convention; closest one and its rate given |
| TS542 | warning | intref_indistinguishable | every convention gives the same output on these inputs; distinguishing inputs needed |
| TS543 | warning | intref_not_rounding | device and control differ both ways inside channels: the accumulators differ, not the rounding |
| TS544 | error | intref_invalid_case | the case file or saved results are missing or invalid (exit 2) |
| TS545 | error | intref_golden_invalid | a golden vector header failed its parse back check |

## Simulated devices, size sweeps and special values (added 2026-11-24)

`--backends sim:NAME` adds a simulated device to `truestar check` (docs/DEVICES.md). Its entry in
`backends` has the same fields as a real one, and every simulated backend carries TS520.
`truestar check model.py --sweep NAME:LO..HI --format agent` emits the envelope with `command` set
to `check --sweep`, a `sweep` object (`ladder`, `checks_run`, and per backend `verdict`, `last_ok`,
`first_bad`, `module`, `op`, `checked`) and `run_repro` next actions. `truestar probe --specials
--format agent` emits `command: "probe --specials"` and a `probe` object with one row per case,
element and backend (`float64`, `ieee`, `got`, `verdict`). Additive, so `schema_version` stays 1.

| code | severity | slug | meaning |
|---|---|---|---|
| TS520 | info | simulated_backend | the backend is a CPU simulation of a device's numerics; confirm on the device |
| TS521 | error | size_cliff | a size sweep found the first size where a backend diverges |
| TS522 | error | diverges_at_smallest_size | the backend already diverges at the smallest size of the sweep |
| TS523 | warning | special_value_departure | NaN, inf, a subnormal, minus zero or a cast departs from IEEE in the backend's own precision, or backends disagree on an implementation defined case |
| TS524 | info | special_value_precision | a special value case differs from float64 only as far as the precision explains |
| TS525 | info | simulation_approximated | ops computed exactly on a simulated backend because its accumulator is not modelled for them |
| TS1094 | info | tensor_core_unvalidated | a simulated tensor core profile (sim:h100-fp16, ...) whose generation Truestar never measured; it follows the published description only (docs/TENSOR_CORE_MODELS.md) |
| TS1095 | warning | tensor_core_format_unmodelled | a matmul format the profile's matrix unit model has no path for; accumulated exactly instead and named |
| TS1096 | info | tensor_core_op_unmodelled | ops that run on matrix units on the device (convolution, fused attention) computed exactly on a tensor core profile; named |
| TS1097 | info | tensor_core_schedule_assumed | the tensor core profile chains unit groups in K order from zero; a library may split K or change instruction shape |
| TS460 | error | quant_kernel_error | a quantized matmul kernel differs from the exact float64 product of its dequantized operands beyond its rounding bound (a kernel bug, docs/QUANT.md) |
| TS461 | error | quant_quantizer_error | a run time activation quantizer wrote codes or scales off the format's rounding rule (a kernel bug) |
| TS462 | warning | quant_format_flips | the format alone, in float64 with no kernel involved, changes decisions beyond near ties (expected format loss) |
| TS463 | info | quant_format_loss | format loss measured from float64 references only; no decision changed beyond near ties |
| TS464 | error | quant_format_unsupported | the format cannot run on this GPU, Triton version or shape; nothing falls back; reserved, not emitted yet: format refusals currently raise `FormatError` with no code |

## Parallel layouts (added 2026-11-24)

`truestar parallel model.py --layout tp=2,dp=2,ga=2,sp=1 --format agent` (docs/PARALLEL.md) emits
the envelope with `command: "parallel"`, `layout` (tp, dp, pp, ga, sp, seq_dim), `plan` (module to
style), `plan_source`, `ranks` (threads, gloo or devices), `simulated` (false with `--ranks devices`),
`devices` (one line per real rank: host, device, GPU model, backend), `loss` (`reference`,
`parallel`), `compared` and `worst_error` per stage, `threshold`, `blamed_module`,
`blamed_collective`, `source` (file and line of the blamed module's class), `issues` (each with
`module`, `collective` and `error`), `injected_faults` and `timings_s`. Exit 0 when the layout
computes the single device step, 1 for a finding, 2 for a layout that cannot run (TS680).

| code | severity | slug | meaning |
|---|---|---|---|
| TS680 | error | parallel_layout_invalid | the layout or plan cannot run as given (exit 2) |
| TS681 | error | parallel_forward_divergence | a module's output under the layout differs while every earlier one matches, or a weight piece is not the plan's split |
| TS682 | error | parallel_loss_normalization | outputs match but the loss differs: per micro batch or per replica means instead of the global count |
| TS683 | error | parallel_gradient_scale | every gradient is off by one factor (dp, ga or an inverse): a scaling collective or the accumulation |
| TS684 | error | parallel_missing_gradient_sync | a replicated weight's gradients differ between ranks: its all reduce is missing |
| TS685 | error | parallel_backward_collective | a module passes a wrong input gradient back: its backward collective |
| TS686 | error | parallel_parameter_gradient | a module's parameter gradients differ with no single factor or missing sync to explain it |
| TS687 | error | parallel_update_divergence | gradients match, the weights after the optimizer step do not (clipping on a local norm) |
| TS688 | error | parallel_replica_drift | a replicated weight differs between ranks after the step |
| TS689 | error | parallel_rank_failed | a rank raised under the layout; module and line named |
| TS690 | info | parallel_simulated | ranks were simulated on this CPU; no GPU interconnect involved |
| TS820 | info | parallel_real_ranks | ranks ran as processes on real devices; every rank's host, device and backend named |
| TS821 | error | parallel_devices_unavailable | fewer devices than ranks, or the backend is missing (exit 2) |
| TS822 | error | parallel_rendezvous_failed | the ranks could not join; rank 0's address named (exit 2) |
| TS823 | error | data_parallel_step_unsupported | a data parallel or accumulated native step setting that would not compute the single device step, refused |
| TS824 | error | data_parallel_replicas_diverged | replicas hold different parameters after a native data parallel step |
| TS825 | warning | data_parallel_not_bitwise | bitwise equality asked for where the reduction order or the devices do not allow it |

## Training spike replay (added 2026-11-24)

`truestar replay RUN --step N --format agent` (docs/SPIKES.md) emits the envelope with
`command: "replay"`, `run_dir`, `step`, `checkpoint`, `dtype`, `verdict` (numerics, data,
optimizer, none, not_reproduced), `baseline_loss`, `losses` (`logged`, `replayed`, `float64`, one
per step from the checkpoint to N), `batch_on_checkpoint_float64`, `first_module`,
`module_errors`, `step_errors` (each step's loss against the same weights in float64),
`first_gap_step`, `rows` (row, loss, share, for data) and `seconds`. Exit 0 for no spike, 1 for a
verdict or a replay that does not reproduce the log, 2 when the run cannot replay the step. The
watch writes TS691 into `watch.jsonl` on a shadowed step that is far from float64.

| code | severity | slug | meaning |
|---|---|---|---|
| TS691 | warning | spike_shadow_gap | a shadowed step's low precision loss or gradient is far from the same step in float64 |
| TS692 | error | spike_numerics | the replayed spike vanishes in float64: rounding or overflow; first module named |
| TS693 | error | spike_data | the spike survives float64 and the batch is high on the checkpoint weights; rows named |
| TS694 | error | spike_optimizer | the spike survives float64, the batch is normal on the checkpoint weights: the updates caused it |
| TS695 | error | spike_not_reproduced | the replay does not reproduce the logged losses; no verdict |
| TS696 | info | spike_none | no spike at the replayed step |
| TS697 | error | replay_invalid | the run directory cannot replay the step (exit 2) |
## Deployed artifact conformance (added 2026-11-24)

`truestar conform model.py ARTIFACT --format agent` emits the envelope with `"command": "conform"`:
`artifact` (`path`, `format`, `sha256`, `bytes`, `nodes`), `samples`, `modules_mapped`, `golden`
(the bundle path or null), `runtimes` (one entry per runtime: `runtime`, `status` among ok,
suspect, divergent, unavailable, failed; `codes`, `issues`, `outputs` (per output `status`,
`rel_err`, `max_abs`, `threshold`, `beyond_bound`, `total`), `decisions`, `flips`, `tie_flips`,
`precision` (`declared`, `observed`), `verdicts` (`conversion`: the file in float64 against the
model in float64, `runtime`: the runtime against the file, `conversion_bar`, `conversion_ok`),
`residency` (`requested`, `known`, `method`, `nodes`, `resident`, `instrumented`, `off_target`:
`node`, `op`, `module`, `where`), `observer` (per instrumented run: `run`, `noise`, `effect`,
`trusted`, `decisions_changed`), `blamed` (`module`, `own_error`, `threshold`, `cause`
conversion or runtime, `conversion_error`, `ops`, `nodes`, `user_source`, `path`),
`units_checked`, `fingerprint`, `seconds`). `next_actions` kinds: `fix_conversion`,
`report_runtime`, `inspect_fallback`, `force_precision`, `replay_after_upgrade` (with `command`).
`truestar conform replay BUNDLE ARTIFACT --format agent` has `"mode": "replay"` and per runtime
`verdict` (identical, changed, regressed, unavailable, failed), `conformant`, `max_change`,
`rel_change`, `decisions_moved`, `fingerprint_changes` (`key`, `recorded`, `current`, `kind`),
`first_changed_module`, `modules`. Method: docs/CONFORM.md. Additive, so `schema_version` stays 1.

| code | severity | slug | meaning |
|---|---|---|---|
| TS620 | error | conform_artifact_unsupported | the artifact cannot be read or its format cannot run here (Core ML off macOS), reason given (exit 2) |
| TS621 | error | conform_runtime_unavailable | a runtime cannot run here; nothing else runs in its place |
| TS622 | error | conform_divergent | the artifact's output in this runtime diverges from the model in float64 |
| TS623 | warning | conform_suspect | the output error is above its calibrated limit, below the divergent cutoff |
| TS624 | error | conform_decision_flips | per row decisions differ from the float64 model beyond near ties |
| TS625 | error | conform_conversion_fault | the file itself, in float64, computes something else than the model (module named when localized) |
| TS626 | error | conform_runtime_fault | the file is right in float64 at the blamed module and the runtime computes it wrongly |
| TS627 | warning | conform_lower_precision | the error is what a correct float16 (or TF32) implementation shows |
| TS628 | warning | conform_off_target_nodes | nodes ran off the requested device or delegate; each one named with its module |
| TS629 | warning | conform_residency_unknown | the runtime does not say where nodes ran |
| TS630 | warning | conform_observer_effect | an instrumented run changed the outputs beyond run to run noise; its per node data is untrusted |
| TS631 | warning | conform_not_localized | wrong output but no single mapped module at fault, or no module names in the format |
| TS632 | error | conform_golden_invalid | a golden bundle is missing, damaged or of an unknown version |
| TS633 | error | conform_baseline_regressed | rerun: outputs moved and are now beyond tolerance; fingerprint changes and first changed module named |
| TS634 | warning | conform_baseline_changed | rerun: outputs moved but still conform |
| TS635 | error | conform_artifact_mismatch | the artifact is not the one the bundle was written for |
| TS636 | error | conform_runtime_failed | the runtime could not load or run the artifact |
| TS637 | error | conform_inputs_mismatch | the model's inputs do not fit the artifact's signature |
| TS638 | info | conform_meaning_unavailable | the file's float64 meaning could not be evaluated, so conversion and runtime faults are not told apart |
| TS639 | info | conform_baseline_identical | rerun: outputs bit for bit identical to the recording; fingerprint changes listed |
| TS760 | error | fit_bad_loss | the loss cannot make a training step: not native, no such parameter tree, or the first result is not a float scalar |
| TS761 | error | fit_state_mismatch | model state declared but the loss does not return (loss, new state) shaped like it, or no initial state |
| TS762 | error | fit_bad_config | an optimizer, clipping, precision or executor setting is invalid |
| TS763 | error | fit_nonfinite_step | NaN or Inf in the loss or a gradient; the step was not applied; the first non finite op is named |
| TS764 | error | fit_checkpoint_other_program | the checkpoint was written by another step program; its configuration is shown |
| TS765 | error | fit_checkpoint_mismatch | not a training checkpoint, or tensors missing, extra, misshapen or mistyped (all listed) |
| TS766 | error | fit_mixed_precision_unsupported | mixed precision could not be applied to the loss program (loops and branches) |
| TS780 | error | train_graph_capture_refused | a compiled training step cannot be replayed from one CUDA graph (a carried value is not written over its own buffer, or the first replay differs bit for bit from a plain call); state restored, nothing replayed (docs/TRAINING.md, Compiled) |
| TS781 | error | train_graph_input_mismatch | a batch given to a captured training step differs in shape, dtype or device from the captured one; not replayed |
| TS782 | warning | ill_conditioned_output_accepted | `accept_ill_conditioned=True`: a float32 output over its end to end bar was accepted because an independent float32 evaluation is as far off float64 (within 4x); each output named with its numbers (docs/COMPILER.md) |
| TS860 | error | import_setting_unsupported | an ATen op the importer lowers was called with a setting its lowering cannot express exactly; the op, setting and user line are named (docs/IR.md) |
| TS861 | error | import_size_not_fixed | a lowering needs a fixed size that is symbolic (a recurrent layer's sequence length, an interpolated, padded or shuffled dim, a 3D depth) |
| TS940 | error | moe_routing_unsupported | the model has a mixture of experts layer Truestar does not lower; the message names the class, path and why (transposed or biased experts, clamped gate, per expert modules, expert capacity dropping). Mixtral, Qwen3 MoE, OLMoE and DeepSeek V3 style routed top k experts are lowered (`truestar.moe`, docs/MOE.md); refused by name before any op is imported |
| TS941 | error | linear_attention_unsupported | the model has a chunked linear attention or delta rule layer Truestar does not lower (Kimi Delta Attention, lightning attention, mLSTM, a Gated DeltaNet without the transformers 5 module function); refused by name with the reason. Qwen3.5's Gated DeltaNet is lowered (TS1150) |
| TS942 | error | chunked_ssm_unsupported | the model has a chunked selective state space layer Truestar does not lower (Mamba 2 SSD on a transformers without `mamba2_chunk_scan`); refused by name with the reason. Mamba 2, Falcon H1, Bamba and Zamba 2 are lowered on transformers 5 |
| TS1180 | error | moe_expert_index_invalid | the grouped experts got an expert index outside 0..E-1 (docs/MOE.md) |
| TS1181 | error | moe_experts_mismatch | grouped experts differ from float64 beyond the kernel tolerance (first call per signature) |
| TS1182 | warning | moe_router_near_tie | a token's k-th and (k+1)-th router scores are within the float64 derived rounding bound; layer, token, experts, gap, bound named (`truestar.moe.router_ties`) |
| TS1183 | error | moe_router_flip | the working run picked another expert set at a token that is not a near tie |
| TS1184 | info | moe_router_unmeasured | a covered layer's router rule is not one the report bounds; listed, not skipped |
| TS1200 | error | mla_variant_unsupported | an MLA model has a feature the latent cache lowering was not written for (attention code that differs from DeepseekV3Attention, a kv_b_proj bias, fewer KV heads than query heads, a sequence length dependent RoPE, an activation other than SiLU, an unknown mode); the layer and reason are named and nothing is built (`MLAUnsupported`) |
| TS1201 | error | mla_absorption_mismatch | the absorbed MLA decode does not compute the expanded decode in float64 within 1e-9 relative; the first step and layer are named and nothing is compiled (`MLAAbsorptionError`) |
| TS1202 | error | sparse_attention_selection_refused | the model keeps a top k of keys per query with an indexer (DeepSeek Sparse Attention in V3.2 and V4 and nine relatives); refused by class name before export, ahead of MoE |
| TS1203 | error | uninitialized_int_buffer | an integer or bool `torch.empty` buffer is not imported (unwritten float entries are NaN in the IR; no integer value can mark them) |
| TS1260 | error | detection_postprocess_unsupported | the eval forward of a torchvision detector (Faster, Mask and Keypoint R-CNN, RetinaNet, FCOS, SSD) selects boxes by value; refused by class and module before torch.export, naming the parts that import (backbone with FPN and the RPN or detection head, the second stage on pooled RoI features; docs/MODEL_COVERAGE.md) |
| TS1310 | error | llamacpp_unavailable | no usable llama.cpp build or driver; the reason is named (`truestar conform llamacpp`, docs/CONFORM_LLAMACPP.md) |
| TS1311 | error | llamacpp_logits_divergent | engine logits beyond the derived tolerance of the file's float64 replay |
| TS1312 | error | llamacpp_greedy_divergent | greedy token differs where the top two are not a near tie |
| TS1313 | info | llamacpp_greedy_near_tie | greedy differs only at a near tie (ambiguous) |
| TS1314 | info | llamacpp_quantization_loss | quantization alone, in float64 |
| TS1315 | warning | llamacpp_quantization_flips | quantization alone changes decisions |
| TS1316 | error | llamacpp_conversion_fault | the GGUF does not hold the source model |
| TS1317 | warning | llamacpp_settings_change_tokens | greedy tokens change with engine settings |
| TS1318 | info | llamacpp_settings_change_bits | logits bits change with settings, tokens do not |
| TS1319 | error | llamacpp_model_unsupported | architecture or tensor type not covered, malformed file, or llama.cpp's converter or quantizer failed |
| TS1320 | error | serving_engine_unavailable | the engine is not installed here or there is no CUDA GPU (exit 2) (`truestar conform vllm`, `truestar conform sglang`, docs/CONFORM_SERVING.md) |
| TS1321 | error | serving_engine_refused | named but not supported in this release (TensorRT-LLM) (exit 2) |
| TS1322 | error | serving_run_failed | a setting failed in the engine, or DIR cannot be judged |
| TS1323 | error | serving_logprob_divergent | log probabilities off float64 beyond the derived tolerance |
| TS1324 | error | serving_decision_divergent | a top 1 or greedy token differs from float64 beyond a near tie |
| TS1325 | info | serving_near_tie | differs from float64 only at near ties |
| TS1326 | warning | serving_repeat_differs | the same request twice gave different results |
| TS1327 | warning | serving_batch_differs | batch 1 vs batch 8 differ |
| TS1328 | warning | serving_prefix_cache_differs | prefix caching changes the results |
| TS1329 | warning | serving_chunked_prefill_differs | chunked prefill changes the results |
| TS1150 | error | linattn_lowering_call_unsupported | a lowered Mamba 2 SSD or Gated DeltaNet layer (`truestar.ir.hf_linattn`) was called with an initial recurrent state or asked for its final state (generation with a cache); import the forward without a cache (docs/LINEAR_ATTENTION.md section 10) |
| TS640 | error | engine_op_divergent | `truestar port ref.py gguf:m.gguf`: an engine graph node differs from the float64 replay of its step on the engine's own inputs, with the GGUF's weights, settings and allowed rounding (an engine bug, docs/PORT_ENGINES.md) |
| TS641 | error | engine_setting_mismatch | a setting (rope theta, rope scaling, norm eps, head counts, dims) in the GGUF, or as the engine used it, differs from the reference |
| TS642 | error | engine_weight_layout | a GGUF tensor holds a reference weight transposed, without the rotary row permutation its architecture needs, or with other values |
| TS643 | error | engine_weight_missing | a reference weight has no GGUF tensor, or a GGUF tensor matches no reference weight |
| TS644 | info | engine_quant_loss | quantization loss from float64 replays only; no decision changed beyond near ties |
| TS645 | warning | engine_quant_flips | the quantized weights alone (float64 replay, no engine) change top 1 decisions beyond near ties (expected format loss) |
| TS646 | error | engine_tokenizer_mismatch | the engine tokenizes the reference prompt into other ids (BOS added or missing, other pieces) |
| TS647 | error | engine_unavailable | the engine library, device or model load is not available; nothing runs elsewhere |
| TS648 | error | engine_tap_unreadable | an engine intermediate tensor needed for the replay is missing or unreadable; no verdict is guessed |
| TS649 | warning | engine_observer_effect | tapped and untapped engine logits differ, so per node data may not describe the untouched engine |
| TS650 | error | engine_reference_not_replayable | the float64 architecture replay does not reproduce the reference, or the architecture is not replayed |
| TS651 | error | engine_type_unsupported | a GGUF tensor type Truestar does not decode exactly, or a quantization type the engine lacks |
| TS652 | error | framework_unavailable | `truestar port ref.py jax:cand.py` / `keras:cand.py`: the framework or backend is missing or failed to start |
| TS653 | error | framework_port_divergent | a candidate layer in another framework, run alone on the reference layer's inputs, computes differently |
| TS654 | warning | framework_port_suspect | a candidate layer differs by more than its threshold but less than the divergent cutoff |
| TS655 | error | framework_weights_unmatched | reference weights without a counterpart in the candidate framework, or candidate weights matching nothing |
| TS656 | error | framework_backend_divergent | the same multi backend model diverges on one backend while another matches the reference |
| TS657 | error | engine_output_divergent_unattributed | the engine's final output differs beyond the explained loss but no step, setting or weight is to blame |
| TS658 | info | framework_lower_precision | candidate modules compute below float64 (Keras forces 64 bit results to 32 bit on its jax and torch backends); float32 noise thresholds apply |
| TS700 | error | kernel_config_resources | a GPU launch config needs more local memory (spilled registers) than this GPU allows; refused before launch as `truestar.kernel.triton_backend.ConfigResourceError`, skipped by the tuner (docs/KERNEL_IR.md, 6a) |
| TS840 | error | metal_unavailable | Metal kernels were requested where they cannot run (not macOS, no MPS device, or a torch without `torch.mps.compile_shader`); no fallback (docs/METAL.md) |
| TS841 | error | metal_float64_unsupported | a kernel or model needs float64, which Metal does not have; refused before any launch |
| TS842 | error | metal_checks_unsupported | compiled in checks (sentinels) were requested for a Metal kernel; not generated for Metal yet |
| TS843 | error | metal_compile_failed | the Metal compiler rejected Truestar's generated source (a Truestar bug); names the kernel and the source file |
| TS844 | info | metal_simdgroup_unavailable | SIMD group matrices are not usable on this device; contraction kernels use threadgroup tiles with plain multiply adds |
| TS845 | error | metal_size_limit | a Metal launch needs a size, stride or thread count beyond 32 bits |
| TS880 | error | rocm_target_unknown | an AMD target the table does not know was named (`truestar.kernel.rocm.UnknownTarget`); on that GPU the target is read from the device (docs/ROCM.md) |
| TS881 | error | rocm_compile_failed | Triton's AMD backend could not compile a kernel for an AMD target (`RocmCompileError`, names kernel, loop shape, function, config, target) |
| TS882 | error | rocm_lds_over_limit | a compiled launch config needs more LDS than one workgroup has on the AMD target (`LdsError`), refused before any launch |
| TS883 | error | rocm_unavailable | ROCm was asked for (`K.compile(..., "rocm")`, `CompiledExecutor("rocm")`) but torch is not a ROCm build or sees no AMD GPU (`RocmUnavailable`); nothing falls back |
| TS884 | info | rocm_tf32_unavailable | TF32 was allowed on an AMD target without a TF32 dot (every RDNA3 arch); float32 dots run strict (warning `RocmNote`) |
| TS885 | info | rocm_compiled_not_run | an ahead of time AMD build (`rocm.build`) compiled and was not run; its results are unverified until it runs on the GPU |
| TS886 | info | rocm_default_relieved | the default launch config spilled registers on the AMD GPU; the first spill free config of the relief ladder is used and verified like any other |
| TS1060 | error | linattn_kernel_mismatch | a chunked linear attention or SSM kernel (`truestar.linattn`) disagrees with the float64 sequential recurrence on its first call check (`LinAttnMismatchError`, docs/LINEAR_ATTENTION.md) |
| TS1061 | error | linattn_call_refused | `linear_attention` refused a call: shapes, rule, beta, device, dtype or launch config it does not run (`LinAttnError`) |
| TS1062 | warning | linattn_chain_error_growth | `chain_error`: the last quarter of chunks is more than 4x further from float64 than the first quarter |
| TS1063 | error | linattn_deterministic_chunk | deterministic mode pins the chunk size; another chunk was asked for |
## `truestar compile --precision auto` documents (added 2026-11-24)

`truestar compile model.py --precision auto --max-flips N --format agent` emits the envelope with
`"command": "compile"`. `status` is `ok` (exit 0), `budget_not_met` or `kernel_rejected` (exit 1);
usage errors are exit 2 with TS020, TS021 or TS669 (no CUDA GPU). The `plan` object (also written
by `--plan plan.json`, the file `truestar.precplan.apply` reads) has: `plan` (unit name to
precision), `counts`, `max_flips`, `allowed_flips_calibration`, `allowed_flips_held_out`, `met`,
`tightened_rounds`, `result` and `baselines` (each: name, calibration and held_out with flips,
decisions, rel_err, nonfinite_rows; latency_us, latency_range_us, eager_us, predicted_us),
`units` (name, module_type, kind, group, order, calls, chosen, why, and per candidate precision:
status, reason, time_us, err, flips, kernel, format_loss), `groups` (tie groups), `search` (every
verification step), `verification_runs`, `seconds`, `fingerprint`. `next_actions` kinds:
`apply_plan`, `raise_budget`, `report_kernel_bug`. Method: docs/PRECISION.md. Additive, so
`schema_version` stays 1.

| code | severity | slug | meaning |
|---|---|---|---|
| TS660 | info | precplan_verified | the plan keeps flips against float64 within budget on the calibration and the held out inputs |
| TS661 | warning | precision_plan_tightened | a plan within budget on calibration failed the held out inputs; the budget was tightened until both held |
| TS662 | error | precision_budget_unreachable | no plan keeps the budget, not even float32 everywhere; the plan is float32 everywhere |
| TS663 | info | precision_range_excluded | truestar range keeps the named units out of a 16 bit dtype (overflow, near overflow or lost digits) |
| TS664 | error | precision_kernel_rejected | a quantized kernel failed its exact check on a layer's own inputs; removed as a kernel bug, never counted as format loss |
| TS665 | info | precision_candidate_unavailable | a precision cannot run for a layer here (shape or kernel limits); nothing is substituted |
| TS666 | error | precision_plan_mismatch | a saved plan was applied to a model with other units or weights |
| TS667 | info | precision_units_tied | code between units mixes dtypes in an op that does not promote, so they share one activation dtype |
| TS668 | warning | precision_range_unavailable | the range check could not run on this model; range safety rests on measurement and verification |
| TS669 | error | precision_needs_cuda | the planner times real kernels and needs a CUDA GPU |
| TS670 | warning | precision_baseline_faster | a uniform precision baseline keeps the budget and measured faster than the plan |
| TS740 | error | fingerprint_data_mismatch | `truestar fingerprint`: a saved weights or inputs file does not match its manifest (exit 2, docs/CROSS_CHIP.md) |
| TS741 | error | fingerprint_not_run_to_run | deterministic mode gave different bits on two calls on this GPU (exit 1) |
| TS742 | warning | fingerprint_arm_failed | an arm or row raised (for example a dtype this GPU lacks); recorded, no hash |
| TS743 | error | fingerprint_claim_differs | `fingerprint compare`: a deterministic row differs between the two files (exit 1) |
| TS744 | error | fingerprint_inputs_differ | the two files started from different weights or inputs; not comparable |
| TS745 | info | fingerprint_settings_differ | the two files ran with different settings, each named |
| TS746 | info | fingerprint_unclaimed_differs | rows of arms without a claim (eager, torch.compile, compiled default) differ |
| TS747 | info | fingerprint_env_differs | different GPU, driver or software versions, each named |
| TS748 | info | fingerprint_row_missing | some rows are in one file only |
| TS749 | error | fingerprint_bad_file | a file given to compare is not a fingerprint (exit 2) |
| TS750 | warning | fingerprint_data_not_golden | a case's weights or inputs built here differ from the pinned hashes; not comparable |
| TS701 | error | kv_quant_unsupported | a quantized KV cache format, layout or shape the decode kernel cannot run, raised as `truestar.kvquant.KVQuantError` (docs/KV_QUANT.md); nothing falls back |
| TS702 | warning | prefix_cache_divergent | a cached prefix run gives other bits than one full prefill; first token and layer named (`truestar prefixcheck`, docs/PREFIX_CHECK.md) |
| TS703 | error | prefix_cache_wrong | against a float64 reference the cached prefix run is less accurate than the full prefill beyond the tolerance |
| TS704 | info | prefix_cache_bitwise | the cached prefix run gives the full prefill's bits |
| TS960 | error | kernel_errors_lean | a compiled Truestar kernel's errors are within the size tolerance but lean one way; raised as `truestar.kernel.KernelBiasError` (docs/BIAS.md); `TRUESTAR_VERIFY_BIAS=warn` reports without refusing |
| TS961 | warning | module_errors_lean | `truestar check`: a module's own errors on a backend lean one way beyond unbiased round to nearest, within the size threshold (docs/BIAS.md) |
| TS962 | info | order_recovered | `truestar orderprobe`: the summation order of a reduction was recovered and replayed bit for bit (docs/ORDERPROBE.md) |
| TS963 | warning | order_not_repeatable | `truestar orderprobe`: the same probe input gave different results across runs (atomics or a run dependent split) |
| TS964 | warning | order_unresolved | `truestar orderprobe`: no recovered order explains the backend's result |
| TS965 | info | orders_differ | `truestar orderprobe`: two backends sum the same reduction in different orders |
| TS966 | warning | tf32_inputs_lean | a kernel compiled with `allow_tf32` leans toward zero because the TF32 tensor cores truncate float32 operands; a `KernelBiasWarning`, not refused (docs/BIAS.md) |
| TS1100 | info | tp_tree_bitwise | parallel tree: the tp step through the canonical reduction tree is bitwise equal to the one device step (docs/PARALLEL.md) |
| TS1101 | error | tp_tree_differs | parallel tree: a value of the tp step differs bitwise from the one device step; the first ones are named |
| TS1102 | error | tp_tree_unsplittable | parallel tree: tp is not a power of two dividing a split axis' group count; nothing was approximated |
| TS1103 | info | tp_plain_differs | parallel tree: for contrast, the plain tp step (F.linear, rank order sums) differs from its one device step |
| TS1110 | info | remat_mincut_plan | compiler: a value recomputed by the min cut plan, in `recompute_report` with its work (docs/MEMORY.md) |
| TS1111 | info | remat_offload_plan | compiler: a value offloaded to pinned host memory and reloaded at positions fixed at compile time |
| TS1112 | error | offload_reload_mismatch | compiler: a reloaded value is not bitwise equal to the offloaded one; raised as `RecomputeMismatchError` (stage offload), result not returned |
| TS1113 | error | offload_unavailable | compiler: host offload asked for a program not compiled for CUDA |
| TS1210 | error | data_parallel_unequal_replica_batches | the replicas of a data parallel step hold batches with different row counts; the mean over replicas would not be the batch mean, refused on every replica with the counts and the rows to pad or drop (docs/TRAINING.md) |
| TS1211 | error | reduce_scatter_uneven | a reduce scatter's dim does not split into equal pieces over its group; refused on every backend (docs/PARALLEL.md) |
| TS1000 | error | fp8_variant_mismatch | an FP8 artifact's variant (OCP or FNUZ) is not the target GPU's native FP8: the same bits decode to other values there (docs/QUANT.md) |
| TS1001 | warning | fp8_variant_ambiguous_name | an FP8 dtype name does not say OCP or FNUZ |
| TS1002 | info | fp8_variant_inferred | an unnamed FP8 tensor's variant was inferred from codes only one variant allows |
| TS1003 | error | fp8_variant_bits_conflict | an FP8 tensor named one variant holds codes only the other variant uses as values |
| TS1004 | warning | fp8_variant_undetermined | neither name nor codes decide OCP or FNUZ |
| TS1005 | info | fp8_target_unknown | the target GPU's native FP8 variant is not known; no comparison made |
| TS1006 | error | sround_bits_differ | a stochastic rounding kernel's codes differ from the keyed reference |
| TS1007 | error | sround_biased | keyed stochastic rounding failed the statistical test (mean error or variance not that of unbiased rounding) |
| TS1008 | error | mx_scale_differs | an MX quantizer's shared block scales differ from the declared scale rule |
| TS1009 | error | mx_element_differs | an MX quantizer's element codes are not round to nearest even of the input over its block scale |
| TS1080 | error | order_free_sum_refused | an order free sum on terms it cannot take (not float32, or more than 2**24 - 1 per row), raised as `truestar.kernel.ordersum.OrderFreeSumError` (docs/ORDER_FREE_SUM.md) |
| TS1081 | error | order_free_sum_backend_unsupported | a kernel with an order free sum compiled for a backend that does not generate them; no current backend raises it (Metal generates them since 2026-11-25) |
| TS1085 | error | sliced_f64_mismatch | the sliced float64 matmul (verification only) disagreed with a CPU float64 matmul beyond both proven bounds, raised as `truestar.kernel.f64emu.SlicedF64Mismatch` (docs/SLICED_F64.md) |
| TS1086 | error | sliced_f64_unavailable | the sliced float64 matmul cannot run (no CUDA, non finite operands), `SlicedF64Error`; verification keeps the native float64 reference |
| TS1098 | warning | local_bound_look_closer | two stacks differ on an operator above the probable but within the worst case rounding bound (`truestar.opbounds.compare`, docs/TOLERANCE.md) |
| TS1099 | error | local_bound_not_rounding | two stacks differ on an operator, same inputs, above twice the worst case rounding bound: not rounding (`truestar.opbounds.compare`) |
| TS1089 | error | round_once_backend_unsupported | `K.compile(..., math="round_once")` on a backend that does not generate it (Triton, numba and Metal do); refused, raised as `truestar.kernel.roundonce.RoundOnceError` (docs/ROUND_ONCE.md) |
| TS1090 | error | math_mode_unknown | the `math` compile option is not `default` or `round_once` |
| TS1091 | error | round_once_dtype_unsupported | a round once function or kernel input or output is not float32, float16 or bfloat16 |
| TS1092 | error | round_once_op_unsupported | a round once kernel uses an op without a round once form (cos, sin), or a function name round once does not cover |
| TS1093 | error | round_once_shape_unsupported | the attention loop shape computes its softmax exp itself; no round once form yet |
| TS1020 | error | rotation_equivalence_failed | a rotated model (`truestar.rotate.fold`) does not compute the original in float64 within the stated bound; the first differing point and the folding steps before it are named, raised as `RotationProofError` (docs/ROTATION.md); nothing is quantized |
| TS1021 | error | rotation_not_orthogonal | a rotation matrix given to the rotation pass is not orthogonal to float64 accuracy (`NotOrthogonalError`) |
| TS1022 | error | rotation_unsupported | no Hadamard construction for the size, an unknown model layout or step (`truestar.rotate.hadamard.RotationError`); nothing falls back to a random rotation |
| TS1023 | error | walsh_kernel_mismatch | the fast Walsh transform kernel of online rotations disagrees with the float64 transform on its first call (`WalshKernelError`) |
| TS1024 | error | rotation_hurts_format | a rotated model was given to a block or float format (int4 grouped, nvfp4, mxfp4, mxfp8, fp8), which rotations made 1.9x to 5.9x worse, or weight only integers (1.6x to 2.4x worse; measured, docs/ROTATION.md); only W4A4 and W8A8 per token integer formats take a rotated model (`RotationRefusedError`) |

Training numerics (docs/TRAIN_NUMERICS.md):

| code | severity | slug | meaning |
|---|---|---|---|
| TS1040 | error | muon_orthogonality_exceeded | a Muon update is off the float64 result of its own Newton Schulz iteration by more than its dtype's rounding, raised as `truestar.optim.MuonOrthogonalityError` |
| TS1041 | info | muon_orthogonality | a Muon update's intended distance from orthogonal and its rounding, measured against float64 |
| TS1042 | error | lowp_gemm_kernel_error | a low precision training GEMM is off the float64 product of its dequantized operands beyond float32 accumulation, raised as `truestar.lpgemm.LowPrecisionKernelError` in strict mode |
| TS1043 | warning | lowp_weight_inconsistent | the input gradient multiplies by other weight values than the forward (per channel or 1D block scales) |
| TS1044 | warning | lowp_gradient_over_budget | a layer's weight gradient error against float64 exceeds the plan's budget, so it stays in higher precision; reserved, not emitted yet |
| TS1045 | info | lowp_format_unavailable | a low precision training format has no quantizer or kernel here (MXFP8 on main); nothing is substituted; reserved, not emitted yet |
| TS1047 | error | invariant_prob_rows | softmax rows do not sum to one within the dtype derived tolerance |
| TS1048 | error | invariant_softmax_grad_rows | softmax gradient rows do not sum to zero (stale or requantized delta, or a wrong kernel), raised as `truestar.invariants.InvariantError` |
| TS1049 | error | invariant_normalized_rows | x / sum(x) rows (router gates) do not sum to one |
| TS1051 | info | logit_bound_safe | proven from the weights: no valid input makes an attention logit leave the target format |
| TS1052 | warning | logit_bound_may_overflow | the weight based upper bound is outside the target format; no constructed input reaches it |
| TS1053 | error | logit_overflow_reachable | a constructed valid input overflows the target format; `truestar.logitbound.guard` refuses, as `LogitOverflowError` |

Saved accelerator battery outputs (`truestar conform tpu RESULTS_DIR`, docs/TPU.md):

| code | severity | slug | meaning |
|---|---|---|---|
| TS900 | error | tpu_bundle_invalid | the battery output directory cannot be read, is from another battery version, or a file does not match its case |
| TS901 | error | tpu_case_divergent | a case's saved output is outside even the bfloat16 budget against float64 |
| TS902 | warning | tpu_float32_reduced_precision | a float32 case computed at bfloat16 or TF32 level (TPU default matmul precision) |
| TS903 | warning | tpu_not_bitwise_stable | identical calls of the same compiled case gave different bits |
| TS904 | error | tpu_inputs_mismatch | the run's input hash differs from the regenerated inputs; outputs not judged |
| TS905 | warning | tpu_elementwise_ulp_high | an elementwise function is float32 accurate overall but its worst element is over 4 ulp off |
| TS906 | error | tpu_case_failed | a case raised on the accelerator and has no output |

## `truestar conform qaihub` documents (added 2026-11-24)

`truestar conform qaihub RESULTS_DIR [--runtimes RUN,...] --format agent` emits the envelope with
`"command": "conform"`, `"mode": "qaihub"`, `results_dir` and `runs` (each: run, model, device,
runtime, unit, verdict, layer_units, taps with verdict and metrics, issues). Exit 1 when any error
code below is present, 2 for an unreadable directory (TS920). Method: docs/QAIHUB.md.

| code | severity | slug | meaning |
|---|---|---|---|
| TS920 | error | qaihub_bundle_invalid | the results directory cannot be read, is from another battery version, or a file does not match its run |
| TS921 | error | qaihub_inputs_mismatch | a model's recorded input hash differs from the regenerated inputs; its runs are not judged |
| TS922 | error | qaihub_run_failed | a compile or inference job failed; the run has no output |
| TS923 | warning | qaihub_float32_ran_float16 | a float32 model ran at float16 level, within 4x of the sim:ieee-f16 prediction on every tap |
| TS924 | error | qaihub_beyond_float16 | a tap's error is more than 4x what correct float16 (sim:ieee-f16) gives |
| TS925 | info | qaihub_float32_on_accelerator | an NPU or GPU run is float32 accurate on every tap (ran on the CPU or in float32) |
| TS926 | warning | qaihub_int8_one_step | an INT8 layer replayed from the device's own input has elements one step from ONNX QLinear |
| TS927 | error | qaihub_int8_divergent | an INT8 layer replayed from the device's own input has an element more than one step off |
| TS928 | info | qaihub_int8_convention | an INT8 layer matches another truestar.intref convention better than ONNX QLinear |
| TS929 | error | qaihub_output_missing | a tap is missing from a run's outputs or has the wrong size |

## Device breadth: Windows ML EPs, LiteRT NPU, WebNN, QAIRT range modes (added 2026-11-24)

`--backends winml:<ep>` and `webnn[:cpu|gpu|npu]` fill the usual `truestar check` backend fields
(docs/BACKENDS.md); the LiteRT NPU runtime is `tflite:npu` in `truestar conform` (docs/CONFORM.md);
the QAIRT range mode hook is `truestar.intref.qairt.attribute` (docs/INTREF.md). Ledger row P56.

| code | severity | slug | meaning |
|---|---|---|---|
| TS1120 | error | winml_session_on_cpu | a Windows ML EP could not be created for the graph and ONNX Runtime ran it on the CPU EP; nothing is reported as that EP |
| TS1121 | warning | winml_nodes_on_cpu | compute nodes ran on the CPU EP, not on the requested Windows ML EP, named |
| TS1122 | warning | winml_nondeterministic | two identical runs on a Windows ML EP gave different outputs |
| TS1123 | info | winml_ep_host_runtime | a catalog EP was registered into an ONNX Runtime other than the one Windows ML ships; its version is named |
| TS1124 | warning | litert_npu_partial | LiteRT put only part of the graph on the NPU; operators left outside are named |
| TS1125 | error | litert_npu_silent_cpu | NPU requested, no NPU compiler or dispatch library loaded: the whole model ran on the CPU fallback while is_fully_accelerated() said True |
| TS1126 | info | litert_npu_static_partition | an ahead of time compiled LiteRT file read without an NPU: dispatch partitions and the operators outside them |
| TS1127 | error | webnn_unsupported_op | an ONNX node has no WebNN translation; the graph was not run on WebNN |
| TS1128 | warning | webnn_nondeterministic | two identical WebNN runs gave different outputs |
| TS1129 | error | webnn_context_failed | the browser could not create a WebNN context or build the graph for the device type |
| TS1130 | info | qairt_mode_reproduced | one QAIRT range mode and tie rule reproduces a device's quantized tensor bit for bit |
| TS1131 | warning | qairt_one_step_in_mode | the device uses one range mode's grid but some elements differ by exactly one step (tie or input rounding, not a kernel bug) |
| TS1132 | error | qairt_mode_unexplained | no range mode and tie rule explains the device tensor (other encoding, or more than a step off: look at the kernel) |
| TS1133 | warning | qairt_not_on_grid | the device values are not on one quantization grid; nothing to attribute |
| TS1160 | warning | coreml_silent_fallback | a Core ML run asked for the GPU or Neural Engine, the compute plan put operations elsewhere that the requested unit supports |
| TS1161 | warning | coreml_unit_cannot_run_precision | the requested unit supports none of the operations at the program's precision (Neural Engine with FLOAT32): all ran elsewhere |
| TS1162 | warning | coreml_float32_at_float16 | a FLOAT32 Core ML program gave float16 level error |
| TS1163 | error | coreml_beyond_float16 | a Core ML run's error is above correct float16 (sim:ieee-f16); ratio to a float16 accumulator given |
| TS1164 | error | coreml_run_failed | a battery model failed to convert or run on Core ML, or its output has the wrong size |
| TS1165 | info | coreml_no_compute_plan | no compute plan saved for a run; placement not judged |
| TS1166 | error | coreml_input_hash_mismatch | a run's battery input hash is not this Truestar's; not judged |
| TS1167 | warning | coreml_nondeterministic | two identical Core ML predictions gave different bits |
| TS1168 | error | coreml_bundle_unreadable | the Core ML results directory is missing, unreadable or of another battery version |
| TS705 | error | spec_rows_variant | the target's logits depend on the query row count: speculative decoding would change its tokens (`truestar.specdec.check_rows`, docs/SPEC_DECODE.md) |
| TS706 | info | spec_rows_invariant | one row and the verify width give the same bits: speculative greedy decoding gives plain greedy's tokens |

## sim:hexagon fitted to real Snapdragon outputs (added 2026-11-25)

`sim:hexagon` is a `truestar check` backend like every `sim:*` profile (TS520 on every result). Two
codes say which of its ops come from the device measurement and which do not (docs/DEVICES.md
section 7, ledger row P59):

| code | severity | name | meaning |
|---|---|---|---|
| TS1140 | info | hexagon_fitted_op | an op ran with the behaviour fitted to real Snapdragon NPU outputs (GELU: the device's approximation, fused with the linear layer before it; the message says whether the input was a linear layer's accumulator) |
| TS1141 | info | hexagon_op_unmeasured | an op whose Hexagon behaviour is not modelled ran as correct float16 (GELU with `approximate='tanh'`; softmax and attention, which the device computes 1.7x to 2.0x further from float64) |

## Persistent compile cache (added 2026-11-24)

A verified first call writes a program record; a later call with the same key runs unchecked and is
accepted only when it launched the recorded kernel binaries and returned the recorded output bits
(docs/COMPILER.md). Opt out with TRUESTAR_COMPILE_CACHE=0.

| code | severity | name | meaning |
|---|---|---|---|
| TS1350 | info | compile_cache_hit | a first call was accepted from a compile cache record made by a verified call on the same inputs: the same kernel binaries ran and the outputs are bitwise equal to the recorded ones |
| TS1351 | warning | compile_cache_entry_unreadable | a compile cache entry failed to parse, has another format version or a wrong digest; it was ignored, deleted and is rebuilt |
| TS1352 | info | compile_cache_trimmed | the compile cache was over its size cap (TRUESTAR_CACHE_MAX_MB); the least recently used entries were deleted |
| TS1353 | warning | compile_cache_write_failed | a compile cache record could not be written (read only or full disk); the call itself succeeded and the next process verifies again |
| TS1354 | info | compile_cache_replay_mismatch | a compile cache record existed for these inputs but the unchecked call launched other kernel binaries or returned other bits; the call was verified in full instead |

## Dynamic shapes (added 2026-11-24)

One compile per declared range of a symbolic dim, checked per shape bucket (docs/DYNAMIC_SHAPES.md).
A call outside its range is refused, never recompiled.

| code | severity | name | meaning |
|---|---|---|---|
| TS1420 | error | dynamic_shape_out_of_range | a call's symbolic dim is outside its declared range (or its shapes do not fit the program): refused, never recompiled |
| TS1421 | error | dynamic_bitwise_needs_deterministic | Bitwise equality with the static compile was asked for without deterministic launches; split K and tuned reduction blocks would make a shape's bits depend on the GPU and history |
| TS1422 | error | dynamic_buckets_invalid | a bucket list is not increasing, starts below the dim's minimum or does not cover its declared maximum |
| TS1423 | error | dynamic_bitwise_library_op | Bitwise equality with the static compile was asked for on a program that runs library ops (cuDNN chooses algorithms by shape), which Truestar cannot pin |
| TS1424 | error | dynamic_declaration_invalid | the dynamic dims declaration does not match the model or program (unknown dim, missing examples, no symbolic dim, or a range torch.export changed) |
| TS1425 | error | dynamic_conv_symbolic_channels | a convolution sums over a symbolic channel dim whose range includes counts below 16: the static compile of such a count loops over a flat K, the symbolic one over taps and channel blocks, so the summation order (and the bits) would differ |
| TS1426 | error | dynamic_static_bits_differ | At a shape the dynamic program's result is not bitwise the static compile's (found by prove); the elements are counted |
| TS1427 | error | dynamic_prove_needs_module | prove compiles the static program per shape from the torch module; an IR program has no static source |

## Native from_pretrained (added 2026-11-24)

truestar.models loads a local Hugging Face decoder checkpoint (safetensors, sharded or not) into
Truestar's own modules without transformers (docs/MODELS.md). A family or setting it does not
implement is refused by name.

| code | severity | name | meaning |
|---|---|---|---|
| TS1390 | error | native_family_unknown | config.json names a model_type with no native Truestar implementation; import the model through truestar.ir.import_module instead |
| TS1391 | error | native_setting_unimplemented | config.json asks for a setting the native decoder does not implement (a rope scaling other than default, linear or llama3, Qwen sliding windows, multimodal Gemma 3); refused instead of ignored |
| TS1392 | error | checkpoint_tensor_missing | the native model needs a tensor the folder does not hold (every missing one is listed) |
| TS1393 | error | checkpoint_tensor_unexpected | the folder holds a tensor the native model does not use; nothing is dropped silently |
| TS1394 | error | checkpoint_shape_mismatch | a stored tensor's shape differs from what config.json implies for its place in the native model |
| TS1395 | error | sharded_index_inconsistent | a sharded checkpoint's index and its shards disagree: a missing shard, a name in the wrong shard or in two, a tensor the index does not list. Nothing was loaded |
| TS1396 | info | checkpoint_dtype_converted | Weights were converted from the stored dtype to the requested one (counted per conversion); without a conversion the leaves are views of the memory map |
| TS1397 | error | pretrained_not_local | from_pretrained was given something that is not a local folder; it never downloads from the Hub |

## Data loader and checkpoint v2 (added 2026-11-24)

Every error raised by truestar.data and by checkpoint loading carries one of these codes in .code
and starts its message with it (docs/DATA.md). truestar.data.DATA_CODES is the view of this range of
codes.CODES.

| code | severity | name | meaning |
|---|---|---|---|
| TS1380 | error | data_bad_config | a loader setting is invalid: batch size below 1, world size below 1, rank outside the world, or an unknown worker mode |
| TS1381 | error | data_source_invalid | a data source cannot be read: arrays of different lengths, an empty folder, a missing file, or an index out of range |
| TS1382 | error | data_state_mismatch | a saved loader position was written with other settings (seed, batch size, shuffle, drop_last or rank) than the loader it is restored into, so the resume would not be exact |
| TS1383 | error | data_dataset_changed | the dataset differs from the one the saved loader position was taken on (other length or fingerprint) |
| TS1384 | error | data_worker_failed | a worker raised while loading a batch; the message names the epoch, the batch and its sample indices |
| TS1385 | error | data_collate_failed | the samples of a batch cannot be stacked (different shapes, dtypes or structure); the message names the field |
| TS1386 | error | checkpoint_corrupt | a checkpoint file is not a truestar checkpoint or fails its checksum |
| TS1387 | error | checkpoint_version_unsupported | the checkpoint was written in a checkpoint version this truestar does not read |
| TS1388 | error | checkpoint_mismatch | the checkpoint was written by another training step (program, param or state names, dtypes or shapes differ) |
| TS1389 | error | resume_not_exact | the resume cannot be exact: the checkpoint lacks the RNG seed or loader position the resumed run needs, or holds a different seed |

## LoRA and QLoRA (added 2026-11-24)

LoRA and QLoRA fine tuning on truestar.nn and train_step (docs/LORA.md): adapters, adapter files,
merging and the quantized base weights each refuse a mismatch by code.

| code | severity | name | meaning |
|---|---|---|---|
| TS1370 | error | lora_bad_config | a LoRA configuration Truestar refuses (rank below 1, dropout outside [0, 1), a model that is not a truestar.nn layer); docs/LORA.md |
| TS1371 | error | lora_no_target | no Linear layer of the model matches the LoRA target patterns; the message lists the Linear layer paths |
| TS1372 | error | lora_adapter_mismatch | an adapter tree does not fit the wrapped model: every missing, extra or misshapen adapter tensor is named |
| TS1373 | error | lora_adapter_file_mismatch | an adapter file was written for another LoRA configuration or another base model layout; what differs is named |
| TS1374 | error | lora_merge_mismatch | the merged model's float64 output differs from the adapter model's beyond float64 rounding: the merge is wrong |
| TS1375 | error | qlora_decode_mismatch | a quantized base weight decoded the way the training program decodes it is not bit identical to the format's exact float64 dequantization |
| TS1376 | error | qlora_format_unsupported | a QLoRA base weight format that is not supported (int8, int4:g32/64/128, nvfp4, mxfp4 are), or a layer whose input size is not a multiple of the format's block |
| TS1377 | error | lora_not_bitwise | two runs of the same LoRA training from the same seed and batches gave adapters that are not bit identical |

## truestar serve (added 2026-11-24)

Truestar's own LLM server, an OpenAI compatible subset whose answers do not change with batching,
prefix caching or chunked prefill; truestar serve --check proves it bit for bit (docs/SERVE.md).

| code | severity | name | meaning |
|---|---|---|---|
| TS1400 | info | serve_bitwise_invariant | a request gave the same tokens and logprobs bit for bit alone, batched with 7 others, with the prefix cache on and off and with chunked prefill on and off (truestar serve --check) |
| TS1401 | error | serve_invariance_broken | a request's tokens or logprobs changed with the batch it ran in, the prefix cache or chunked prefill; the scenario and the first token that differs are named |
| TS1402 | error | serve_reference_mismatch | the float32 server's greedy tokens differ from the float64 reference server's at a token that is not a near tie, or the reference server differs from a forward with no cache |
| TS1403 | error | serve_capacity_exceeded | a request was refused: prompt plus max_tokens exceed the model length, the KV block pool, or (without chunked prefill) the tokens per step |
| TS1404 | error | serve_bad_request | a request was refused: a malformed field, or a parameter the server does not implement asked for something other than its default (n, echo, penalties, logit_bias, ...); never ignored silently |
| TS1405 | info | serve_seed_assigned | a sampled request came without a seed; the server drew one and returned it, so the request can be replayed exactly |
| TS1406 | warning | serve_compile_fallback | the compiled serving step runs some ops outside Truestar kernels (fallbacks); the guarantee check still decides whether answers are invariant |
| TS1407 | error | serve_executor_unavailable | the requested serving executor (cuda, or cpu without numba) cannot run on this machine |
| TS1408 | info | serve_reference_agrees | the float32 server's greedy tokens equal the float64 reference server's, and the reference server equals a plain forward with no KV cache |
| TS1409 | warning | serve_baseline_not_invariant | the eager PyTorch baseline gave a request other bits batched than alone (measured by scripts/bench_serve.py); shown for contrast with truestar serve |

## Deploy targets for truestar export (added 2026-11-24)

truestar export --to onnx, litert, coreml or executorch writes a deployable file and runs it in its
own runtime; a file that fails its runtime check is refused unless --force (docs/EXPORT.md, Deploy
targets).

| code | severity | name | meaning |
|---|---|---|---|
| TS1410 | info | deploy_file_verified | the deployable file was written and its outputs in its own runtime agree with Truestar's float64 reference within the derived tolerance |
| TS1411 | error | deploy_program_unverified | the model's Truestar program could not be verified against PyTorch in float64, so no deployable file was written |
| TS1412 | error | deploy_no_lowering | an op or attribute has no exact lowering for the target format; the node and its source line are named and nothing was written |
| TS1413 | warning | deploy_runtime_pending | the target runtime cannot run on this machine (Core ML outside macOS and iOS, or a runtime that is not installed): the file was written, its runtime verification is pending |
| TS1414 | error | deploy_runtime_mismatch | the written file's outputs in its runtime are outside the derived tolerance of Truestar's float64 reference; the file was refused |
| TS1415 | warning | deploy_forced | the file failed its check and was written only because --force was given; its evidence records it as unverified |
| TS1416 | warning | deploy_silent_fallback | Nodes of the file did not run on the requested delegate or provider (named with their op); the runtime fell back without saying so |
| TS1417 | info | deploy_runtime_rewrote | the runtime did not run the ops as written: it fused them into its own kernels, or its error is that of a lower precision than float32 |
| TS1418 | error | deploy_conversion_mismatch | the written file's own math, evaluated in float64, differs from the verified program: the lowering is wrong, before any runtime ran. The file was refused |
| TS1419 | error | deploy_usage | a deploy export was asked for something it does not do (unknown target, wrong extension, a runtime of another format, nested or keyword inputs) |

## Portable tensor cores (added 2026-11-24)

compile(..., portable_tensor_cores=True) runs float16, bfloat16 and TF32 dot products as exact
tensor core blocks with all rounding in a fixed order, so the bits are the same on every covered
NVIDIA GPU (docs/PORTABLE_TC.md). Where that cannot apply it is refused, never silently replaced by
native tensor cores.

| code | severity | name | meaning |
|---|---|---|---|
| TS1330 | error | portable_tc_backend_unsupported | Portable tensor cores were asked for on a backend that does not generate them (only CUDA Triton does); refused, never a silent native fallback |
| TS1331 | error | portable_tc_device_unproven | Portable tensor cores were asked for on a GPU the exactness proof does not cover (AMD, or NVIDIA below sm_80) |
| TS1332 | error | portable_tc_block_unsupported | a dot product's K block is outside 16 to 256 columns, where an exact tensor core block's integer sums could exceed 2**24 |
| TS1333 | error | portable_tc_library_ops | library= ops run through vendor kernels whose tensor core rounding differs by GPU, so they cannot be combined with portable tensor cores |

## Kernel check lean and precision verdicts (TS1300, TS1301)

truestar kernel judges a candidate whose errors are within the derived tolerance but lean one way as
its own verdict, on by default (docs/KERNEL_CHECK.md); a lean from reduced precision the source asks
for is a separate warning.

| code | severity | name | meaning |
|---|---|---|---|
| TS1300 | error | kernel_check_errors_lean | `truestar kernel`: the candidate's errors against float64 are within the derived tolerance but lean one way (the TS960 lean test: mean signed error beyond what unbiased round to nearest allows, and above 1/64 of the size tolerance), on the example input |
| TS1301 | warning | kernel_reduced_precision | `truestar kernel`: the candidate's errors lean one way (TS1300's test) and its source asks for reduced precision (TF32 matmuls, float16 or bfloat16 compute): a precision verdict of its own, not a wrong number |

## Fused MLA decode (TS1460)

The compiled MLA decode runs the latent attention as one fused kernel by default (docs/MLA.md); a
shape the fused kernel was not written for is refused by name.

| code | severity | name | meaning |
|---|---|---|---|
| TS1460 | error | mla_fused_kernel_shape_unsupported | the fused MLA latent attention kernel was given shapes it does not run (a latent rank above 1024, a head dim above 256, mismatched operands, or tensors off CUDA); the shapes are named and decode='absorb' is the alternative |

## Compiled mixed precision (TS1480 to TS1482)

compile(..., mixed=True) keeps every value in its declared dtype: 16 bit values in the working
precision, float32 and float64 ones in float32 islands; the first call is judged by bars derived for
the declared precision (docs/TRAINING.md, Mixed precision, compiled). The derived tolerance of the
float32 path does not judge mixed programs.

| code | severity | name | meaning |
|---|---|---|---|
| TS1480 | error | mixed_precision_mismatch | a compiled mixed precision program (16 bit working precision, float32 islands) has a float32 output beyond the end to end bar derived for its declared precision: 4x the larger of how far an independent evaluation rounded to the declared dtypes is from float64 and how far one ulp of every 16 bit value moves the output. A wrong cast, a 16 bit master weight or a missing unscale lands here; the result is not returned |
| TS1481 | info | mixed_precision_near_tie | a mixed precision output over its bar was accepted because every difference is explained by branch decisions on near ties (a relu input within rounding of 0): the float64 reference taking this run's decisions matches every output within its bar. Listed in mixed_bounds |
| TS1482 | info | mixed_precision_ill_conditioned | an output that the declared precision itself leaves more than 0.1 from float64 (a cancelling bias gradient behind batch norm); its bar follows that distance and it is listed as ill conditioned in mixed_bounds |
