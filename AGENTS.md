# Using Truestar from a coding agent

Truestar checks whether a PyTorch model gives the same answer on every backend on this machine
(CPU, CUDA, ROCm, MPS, XPU). It compares each against a float64 CPU reference, names the exact module
and ATen op that diverge, and writes a standalone repro script. Other commands check gradients,
upgrades, batch invariance, ports and exports the same way.

## Run it

1. Write a model file that defines `get_model()` (returns an `nn.Module`) and `get_inputs()` (returns
   a tensor or a tuple of forward arguments). `get_kwargs()` is optional. Examples for every command
   are in `examples/`.
2. Run:

   ```
   truestar check model.py --format agent
   ```

   Without an install, use `PYTHONPATH=src python -m truestar check model.py --format agent`.
   Add `--backends cpu,cuda` to choose backends, `--repro-dir DIR` to choose where repros go, or
   `--no-repro` to skip them. Compilers and runtimes are backends too: `cuda+inductor`,
   `cpu+inductor`, `ort:cpu`, `ort:cuda`, `trt:fp32`, `trt:tf32`, `trt:fp16`, and `metal` (Truestar's
   own compiler on a Mac's GPU, [docs/METAL.md](docs/METAL.md))
   ([docs/BACKENDS.md](docs/BACKENDS.md)); they are never picked by default.
   Add `--backends cpu,cuda` to choose backends (`webgpu` is opt in, docs/WEBGPU.md),
   `--repro-dir DIR` to choose where repros go, or
   `--no-repro` to skip them.
3. Parse stdout as JSON. It holds exactly one document, even on errors. Act on `next_actions` in order:
   each has a `kind`, a `description` and, when there is one, a shell `command` to run.

The same contract holds for these commands, each with `--format agent`:

| question | command |
|---|---|
| is the backward pass right | `truestar check model.py --backward --format agent` |
| what did an upgrade change | `truestar record model.py -o run.tsrec --format agent`, later `truestar diff run.tsrec --replay --format agent` |
| does a row depend on its batch | `truestar invariance model.py --format agent` |
| does a port compute the same model | `truestar port reference.py candidate.py --format agent` |
| where does compiled GPU time go, per line | `truestar profile model.py --format agent` |
| does llama.cpp (or a Flax, Keras port) run the same model | `truestar port reference.py gguf:model.gguf --format agent` |
| where does it overflow in float16 (or bf16, fp8) | `truestar range model.py --target fp16 --format agent` |
| which precision per layer is fastest within a flip budget | `truestar compile model.py --precision auto --max-flips 0.1% --format agent` |
| is a kernel right on every input | `truestar kernel ref.py cand.py --format agent` |
| does a parallel layout compute the single device step | `truestar parallel model.py --layout tp=2,dp=2 --format agent` (simulated ranks; `--ranks devices` for one process per real GPU) |
| was a loss spike numerics, data or the optimizer | `truestar replay run_dir --step N --format agent` (run written by `truestar.spikes.SpikeWatch`) |
| do compiled results have the same bits on another GPU | `truestar fingerprint --json fp.json --format agent`, then `truestar fingerprint compare A.json B.json --format agent` ([docs/CROSS_CHIP.md](docs/CROSS_CHIP.md)) |
| does vLLM or SGLang serve the same numbers as float64, and the same numbers as itself under batching, prefix caching and chunked prefill | `truestar conform vllm RESULTS_DIR --format agent` (DIR written on a GPU host by `python -m truestar.conform.serving_battery run --engine vllm --out DIR`; `sglang` the same) ([docs/CONFORM_SERVING.md](docs/CONFORM_SERVING.md)) |
| does the deployed .onnx, .tflite, .pte or .tsx file compute the model, in its own runtime | `truestar conform model.py model.tflite --format agent`, after an upgrade `truestar conform replay model.tflite.tsgold model.tflite --format agent` ([docs/CONFORM.md](docs/CONFORM.md)) |
| does an LLM server give the same answer batched, cached and chunked | `truestar serve --check --device cuda --format agent` (Truestar's own server: `truestar serve`, OpenAI compatible subset, [docs/SERVE.md](docs/SERVE.md)) |

`truestar export model.py -o model.tsx` has no JSON mode: exit 0 means written, 1 refused (nothing
written, reason on stderr), 2 usage error. Other commands: `truestar backends` (JSON list of
backends here), `truestar codes` (JSON list of codes).

## MCP

`truestar mcp` runs an MCP server on stdio. Tools: `list_backends`, `check_model`, `run_repro`,
`grad_check`, `port_check`, `invariance_check`, `record`, `diff`, `export_model`, `verify_kernel`.
Each returns the
same JSON document as the matching command, as `structuredContent` and as text. `isError` is true
only when the tool could not run (exit 2, bad arguments), never because something diverged.
Register it with:

```json
{"mcpServers": {"truestar": {"command": "python", "args": ["-m", "truestar", "mcp"]}}}
```

Tools only run or overwrite files Truestar wrote (repros, recordings, artifacts); they do import the
model files you pass. Arguments and results per tool: [docs/AGENT_INTERFACE.md](docs/AGENT_INTERFACE.md).

## Exit codes

| exit | meaning |
|---|---|
| 0 | every backend matches the reference (`diff`: identical, noise or explained precision change) |
| 1 | a finding: divergent or suspect backend, a batch dependent row, a different port, a divergent change |
| 2 | usage error: bad arguments, invalid model file, missing backend, or the reference failed |

## Reading the result

* Schema: [docs/AGENT_INTERFACE.md](docs/AGENT_INTERFACE.md). The field `schema_version` (now 1) only
  changes when a field is removed or changes meaning; ignore fields you do not know.
* `backends[i].blamed_module` is the first module whose own error, replayed alone with reference
  inputs, is over threshold. `source` is where its class is defined; `user_source` is the nearest
  enclosing class in the user's code, with `file` and `line`. `blamed_op` is the ATen op inside it.
* Codes are stable: `TS001` backend unavailable, `TS002` model failed on a backend, `TS010`
  divergence, `TS011` suspect, `TS013` prediction flips. The full table is in docs/AGENT_INTERFACE.md
  (`truestar codes` prints it).
* A repro exits 1 when the divergence reproduces and 0 when the output matches. Run it once on the
  divergent backend and once with `cpu` as the argument (the control) before reporting a bug.

## Rules for agents working in this repo

* Python: `PYTHONPATH=src python -m pytest -q -p no:cacheprovider` runs the suite; it must stay green.
* Read docs/QUALITY_BAR.md before adding a feature: every feature needs a pain ledger row and must not
  bring in a false alarm or a silent fallback.
* Never put code, data or weights from a private source into this repo.
* Docs show only commands that run today; tests/test_examples.py runs every example.
