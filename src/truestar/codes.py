"""Stable machine readable codes for every finding truestar reports.

Codes never change meaning once released. New findings get new codes. They appear in the text
report, in the full JSON report and in the agent JSON (docs/AGENT_INTERFACE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

INFO, WARNING, ERROR = "info", "warning", "error"

# code: (severity, slug, one line meaning)
CODES: Dict[str, tuple] = {
    "TS001": (ERROR, "backend_unavailable",
              "A backend was requested or expected but is not available on this machine."),
    "TS002": (ERROR, "model_failed_on_backend",
              "The model raised an exception when run on this backend."),
    "TS003": (WARNING, "reference_degraded",
              "The float64 CPU reference could not run, so the reference is the CPU run in the "
              "model's own dtype and CPU bugs cannot be detected."),
    "TS004": (WARNING, "module_set_differs",
              "The modules that ran on this backend differ from the reference run."),
    "TS005": (ERROR, "reference_failed",
              "The model could not run on the CPU reference at all."),
    "TS010": (ERROR, "divergence",
              "A module's own output diverges from the reference: the result is wrong, not "
              "just imprecise."),
    "TS011": (WARNING, "suspect",
              "A module's own error is above its threshold but below the divergent cutoff."),
    "TS012": (ERROR, "output_divergent_unattributed",
              "The final output diverges but no single module is to blame (code outside any "
              "module, or work that only runs on this backend)."),
    "TS013": (WARNING, "prediction_flips",
              "Some per input predictions changed versus the reference, beyond near ties."),
    "TS014": (ERROR, "module_replay_failed",
              "A module raised an exception when replayed alone on this backend."),
    # compilers and runtimes as check backends (docs/BACKENDS.md)
    "TS400": (WARNING, "stack_cpu_fallback",
              "A runtime asked for a GPU ran floating point nodes on the CPU with no error (ONNX "
              "Runtime provider fallback); the result is partly the CPU provider's."),
    "TS401": (ERROR, "stack_provider_missing",
              "The requested execution provider did not load, so the runtime silently ran the "
              "whole model on the CPU."),
    "TS402": (ERROR, "stack_fusion_only",
              "The compiled whole model diverges but every module compiled alone matches: the "
              "fault needs the modules compiled together (fusion across modules) or lives in "
              "code between them."),
    "TS403": (INFO, "stack_graph_breaks",
              "torch.compile split the model into several graphs; code between graphs ran "
              "eagerly, as it does for the user."),
    "TS180": (ERROR, "grad_divergence",
              "A module's own backward gives wrong gradients (input or parameter) while its own "
              "forward matches the reference: the backward kernel is wrong."),
    "TS181": (WARNING, "grad_suspect",
              "A module's own gradient error is above its threshold but below the divergent "
              "cutoff."),
    "TS188": (WARNING, "grad_near_tie_mask_flip",
              "A parent module's gradients are off float64 only because ReLU masks flip at near "
              "ties: every leaf is clean and a correctly rounded run with the backend's masks "
              "matches (suspect, not a backward bug)."),
    "TS182": (ERROR, "grad_forward_divergent",
              "A module's gradients diverge and so does its own forward, so the gradient error "
              "may follow from the forward one (run the forward check first)."),
    "TS183": (WARNING, "grad_norm_drift",
              "Over repeated training steps the gradient norm on this backend drifts away from "
              "the float64 reference, more than CPU in the same precision does."),
    "TS184": (ERROR, "backward_replay_failed",
              "A module raised an exception when its backward was replayed alone on this "
              "backend."),
    # truestar fix (docs/FIX.md)
    "TS200": (ERROR, "kernel_divergence",
              "`truestar kernel`: the candidate's output is further from float64 than a correct "
              "implementation can be (derived tolerance) on some input variant."),
    "TS201": (ERROR, "kernel_shape_failure",
              "`truestar kernel`: the candidate raises on an input shape the reference accepts."),
    "TS202": (WARNING, "kernel_nonfinite_mismatch",
              "`truestar kernel`: NaN or infinity in the inputs is handled differently from the "
              "float64 reference (dropped, spread, or turned into finite values) while every "
              "finite element is right; a verdict of its own, not a wrong number."),
    "TS203": (WARNING, "kernel_nondeterministic",
              "`truestar kernel`: the same inputs give different bits from run to run."),
    "TS204": (ERROR, "kernel_stride_dependent",
              "`truestar kernel`: wrong on non contiguous inputs while contiguous inputs are right."),
    "TS205": (ERROR, "kernel_unwritten_output",
              "`truestar kernel`: some output elements are never written (NaN from the allocator "
              "shows through)."),
    "TS206": (ERROR, "kernel_mutates_inputs",
              "`truestar kernel`: the candidate writes into its input tensors."),
    "TS207": (ERROR, "autotune_config_rejected",
              "`truestar.verified_autotune`: an autotune config gives a wrong result and was removed "
              "from the tuning set."),
    "TS090": (INFO, "remedy_verified",
              "A remedy was applied and the rerun check on this backend is clean."),
    "TS091": (WARNING, "remedy_partial",
              "Applied remedies measurably lower the error or the flips, but the rerun check is "
              "not clean."),
    "TS092": (ERROR, "remedy_unresolved",
              "No remedy in the table made the blamed module clean; report the op upstream."),
    "TS093": (ERROR, "remedy_not_applicable",
              "No single ATen op is to blame (module code, a device branch, a model that fails), "
              "so an op remedy cannot apply."),
    "TS094": (INFO, "precision_plan_verified",
              "Keeping the listed layers in higher precision brings simulated INT8 flips under "
              "the target."),
    "TS095": (WARNING, "precision_plan_unmet",
              "No set of layers kept in higher precision reached the flip target in simulation."),
    # truestar range (docs/RANGE.md)
    "TS300": (INFO, "range_clean",
              "Every value fits the target format on the calibration inputs and the emulated target "
              "matches its range unlimited twin."),
    "TS301": (ERROR, "range_overflow",
              "An op's output exceeds the target format's largest finite value on a calibration "
              "input (float64 peak named with its margin in bits)."),
    "TS302": (ERROR, "range_hidden_overflow",
              "An intermediate inside a composite op that the target runs decomposed (x minus mean "
              "squared, mean of squares, attention scores) exceeds the format's range."),
    "TS303": (WARNING, "range_near_overflow",
              "A value peaks within the headroom of the format's maximum and its static bound does "
              "not prove it safe, so inputs beyond the calibration set may overflow."),
    "TS304": (WARNING, "range_underflow",
              "Most nonzero values of an op's output fall below the format's smallest normal value "
              "(they flush to zero or keep few bits)."),
    "TS305": (WARNING, "range_precision_loss",
              "An op replayed alone with inputs rounded to the format keeps fewer than 2 correct "
              "bits (cancellation)."),
    "TS306": (ERROR, "range_target_diverges",
              "The emulated target run is worse than its range unlimited twin: non finite values, "
              "decision flips or a larger output error, so range, not rounding, changes the answer."),
    "TS307": (INFO, "range_remedy_verified",
              "A proposed remedy (rescale, float32 region or clamp) was applied in emulation and the "
              "target run then passes the same test as its twin."),
    "TS308": (ERROR, "range_remedy_unresolved",
              "No remedy made the emulated target run pass, or the failure could not be tied to a "
              "range site."),
    "TS309": (INFO, "range_event_benign",
              "Range events that do not change the result: constants saturated to the format's "
              "maximum, overflow inside fused kernels with float32 internals, or overflow without "
              "effect on the calibration outputs."),
    "TS020": (ERROR, "model_file_invalid",
              "The model file is missing, fails to import, or lacks get_model() / get_inputs()."),
    "TS021": (ERROR, "usage_error",
              "The command line or tool arguments are invalid."),
    "TS022": (ERROR, "optional_dependency_missing",
              "A path that was asked for needs an optional module that is not installed (for "
              "example numba for CPU kernels); the message names the pip extra that installs it."),
    # compiler memory planner and recompute (P23, docs/MEMORY.md)
    "TS420": (ERROR, "memory_budget_exceeded",
              "A compiled program's planned peak memory exceeds its memory budget; refused before any "
              "kernel launch, naming the source line at the peak and the largest batch that fits."),
    "TS421": (ERROR, "recompute_not_bitwise",
              "A recomputed value is not bitwise equal to the stored value it replaces; the result "
              "was not returned."),
    # compiled training steps (P37, docs/TRAINING.md, Compiled)
    "TS780": (ERROR, "train_graph_capture_refused",
              "A compiled training step cannot be replayed from one CUDA graph: a carried param or "
              "state value is not written over its own buffer, or the first replay differs bit for "
              "bit from a plain call. The state was restored; nothing was replayed."),
    "TS781": (ERROR, "train_graph_input_mismatch",
              "A batch given to a captured training step differs in shape, dtype or device from the "
              "captured one; the step was not replayed."),
    "TS782": (WARNING, "ill_conditioned_output_accepted",
              "A compiled float32 output over its end to end float64 bar was accepted because the caller "
              "passed accept_ill_conditioned=True and an independent float32 evaluation of the same "
              "program is as far off float64 (within 4x); each output is named with its numbers."),
    # truestar port (P16, docs/PORT_CHECK.md)
    "TS160": (ERROR, "port_divergence",
              "A candidate module computes differently from its mapped reference module on the "
              "same inputs: the port computes a different model."),
    "TS161": (WARNING, "port_suspect",
              "A candidate module's own difference is above float32 noise but below the "
              "divergent cutoff."),
    "TS162": (ERROR, "weights_unmatched",
              "A weight one forward pass reads has no counterpart in the other model (for "
              "example an adapter dropped on conversion)."),
    "TS163": (ERROR, "weights_unused",
              "A weight present in both models is never read by one of the forward passes."),
    "TS164": (INFO, "weights_layout_changed",
              "Weights matched through a transpose, permutation, fused slice or merged adapter."),
    "TS165": (WARNING, "weight_values_differ",
              "Weights matched by shape or after a dtype cast only: their values differ."),
    "TS166": (ERROR, "port_output_divergent_unattributed",
              "The final outputs differ but no mapped module is to blame."),
    "TS167": (INFO, "modules_unmapped",
              "Some modules have no counterpart (inlined into a bigger module, or unmatched)."),
    "TS168": (ERROR, "port_replay_failed",
              "A candidate module raised when replayed with the reference module's inputs."),
    "TS169": (WARNING, "weights_dead",
              "A weight exists in one model only and its forward never reads it."),
    # truestar rlpair (P15, docs/RL_PAIR.md)
    "TS190": (ERROR, "rl_module_divergence",
              "A module computes differently in the RL sampler than in the trainer for the same "
              "tokens and positions: in the incremental (KV cache) path, or in the sampler's own "
              "implementation."),
    "TS191": (WARNING, "rl_precision_mismatch",
              "Trainer and sampler logprobs differ only as far as the sampler's dtype explains; "
              "every module stays within its emulated dtype band."),
    "TS192": (INFO, "rl_logprob_noise",
              "Trainer and sampler logprobs differ, but only at rounding level."),
    "TS193": (ERROR, "rl_mismatch_outside_modules",
              "Logprobs differ although every module computes the same: the gap is in code "
              "outside the modules (for example a sampling temperature applied before logprobs)."),
    "TS194": (ERROR, "rl_sampler_weights_differ",
              "The sampler's parameters are not the trainer's rounded to the sampler dtype "
              "(stale or partial weight sync)."),
    "TS195": (WARNING, "rl_sampler_batch_variant",
              "The sampler's logprobs for a row change with the other rows in its batch."),
    "TS196": (INFO, "rl_modules_unmapped",
              "Some module calls could not be lined up with sampler positions and were not "
              "compared."),
    "TS197": (ERROR, "rl_inputs_differ",
              "A module receives integer inputs (positions, masks, ids) from the sampler that "
              "differ from the trainer's at the same position."),
    "TS198": (WARNING, "rl_ratio_outside_clip",
              "Some tokens have a trainer over sampler importance ratio outside the clip range."),
    "TS185": (ERROR, "rl_kernel_swapped",
              "With --kernels rl, a module runs another kernel for one op in the sampler than in the "
              "trainer (another implementation, another reduction order); the op is named."),
    "TS186": (ERROR, "rl_tp_mismatch",
              "Trainer logprobs at TP 1 and at the simulated TP size are not bitwise equal."),
    "TS187": (ERROR, "rl_kernels_not_bitwise",
              "With --kernels rl, logprobs are not bitwise equal although every module ran the same "
              "kernels on both sides: an op outside the kernel set or the inputs differ."),
    "TS199": (ERROR, "rl_pair_invalid",
              "The pair file or the trainer and sampler outputs are invalid (shapes, layout, "
              "a run that raised)."),
    # truestar invariance (docs/INVARIANCE.md)
    "TS050": (WARNING, "batch_variant",
              "A row's output is not bitwise identical when the row runs inside other batches, "
              "at another position or with padding (summation order depends on the batch)."),
    "TS051": (WARNING, "nondeterministic",
              "The same row run twice alone gives different bits."),
    "TS052": (ERROR, "batch_dependent",
              "A row's output depends on the other rows beyond float tolerance."),
    "TS053": (WARNING, "invariance_flips",
              "The non invariance changes argmax predictions or greedy decoded tokens."),
    "TS054": (WARNING, "invariance_unattributed",
              "The output is not batch invariant but no single module or op was named."),
    # truestar record / truestar diff (docs/RECORD_DIFF.md)
    "TS060": (ERROR, "env_divergence",
              "A module's own output changed between the recorded and the current environment "
              "beyond what the dtype tolerance or a precision setting explains."),
    "TS061": (WARNING, "env_precision_change",
              "A module's own output changed within the tolerance of a precision setting that "
              "differs between the environments (for example TF32)."),
    "TS062": (INFO, "env_numerical_noise",
              "A module's own output changed, but only within the dtype tolerance."),
    "TS063": (INFO, "env_changed",
              "The environment fingerprints differ (versions, precision or algorithm flags)."),
    "TS064": (ERROR, "module_state_differs",
              "A module's parameters or buffers differ from the recording, so a different "
              "model is being compared."),
    "TS065": (ERROR, "record_invalid",
              "The recording file is missing, truncated, modified or of an unknown version."),
    "TS066": (WARNING, "call_set_differs",
              "Different module calls ran than in the recording."),
    "TS067": (WARNING, "replay_failed",
              "A recorded module call could not be replayed (it raised, or an input could not "
              "be stored)."),
    "TS068": (INFO, "isolation_limited",
              "Some changed modules saw different inputs than recorded, so their own behaviour "
              "was not measured; replay isolates every module."),
    "TS069": (ERROR, "inputs_differ",
              "The two recordings fed the model different inputs, so they are not comparable."),
    "TS070": (ERROR, "record_too_large",
              "The recording would exceed its size limit."),
    # simulated devices, size sweeps and special values (docs/DEVICES.md)
    "TS520": (INFO, "simulated_backend",
              "This backend is a simulation of a device's numerics on the CPU (sim:...): no device "
              "ran. Confirm a finding on the device before reporting it upstream."),
    "TS521": (ERROR, "size_cliff",
              "A size sweep found the first size at which a backend diverges; it is clean at the "
              "size below."),
    "TS522": (ERROR, "diverges_at_smallest_size",
              "A size sweep found the backend already diverging at the smallest size of the range."),
    "TS523": (WARNING, "special_value_departure",
              "A backend handles NaN, infinity, a subnormal, minus zero or a cast differently from "
              "IEEE 754 arithmetic in its own precision, or backends disagree on a case IEEE leaves "
              "to the implementation (a float to int cast of NaN, for example)."),
    "TS524": (INFO, "special_value_precision",
              "A special value case differs from the float64 reference only as far as the "
              "backend's precision explains (for example a float16 overflow to infinity)."),
    "TS525": (INFO, "simulation_approximated",
              "Some ops on a simulated backend were computed exactly and rounded once because "
              "the profile's accumulator is not modelled for them; they are named."),
    "TS071": (WARNING, "instrumented_path_differs",
              "Recording hooks changed the model's code path (they switch off fused fast paths), "
              "so per module results describe the instrumented path; the final output check "
              "uses the plain path."),
    # sentinels, compiled in runtime checks (docs/COMPILER.md, Sentinels): raised as truestar.compiler.SentinelError
    "TS500": (ERROR, "sentinel_nan",
              "A sentinel (compiled in runtime check) found NaN: the op that produced it, its line and the first "
              "offending element are named; the results of the call were not returned."),
    "TS501": (ERROR, "sentinel_inf",
              "A sentinel (compiled in runtime check) found an infinity that no infinite constant explains (an overflow "
              "of the register precision, log(0), an infinite input)."),
    "TS502": (ERROR, "sentinel_overflow",
              "A float16 or bfloat16 value was finite in float32 and became Inf when stored."),
    "TS503": (ERROR, "sentinel_div_by_zero",
              "A division had a zero denominator (float or integer)."),
    "TS504": (ERROR, "sentinel_index_bounds",
              "An index was out of bounds (or repeated where it must be unique); index bounds are "
              "always checked and nothing was read or written out of bounds."),
    # truestar profile (docs/PROFILE.md)
    "TS510": (ERROR, "profile_events_unaccounted",
              "Some step runs have no valid GPU time; the profile is incomplete and says so."),
    "TS511": (WARNING, "profiler_dropped_events",
              "torch.profiler saw fewer kernel launches than the runtime made; the per step event "
              "times still account for every launch."),
    "TS512": (INFO, "profile_fallback_time",
              "Steps of the profiled program run through the float64 fallback; their time is listed."),
    # CUDA graph replay of compiled programs and the cuBLAS contraction choice (P38, docs/COMPILER.md)
    "TS800": (ERROR, "graph_capture_refused",
              "A compiled program could not be recorded into a CUDA graph (a step syncs with the host, or "
              "TRUESTAR_VERIFY=all checks every call); nothing ran in its place."),
    "TS801": (ERROR, "graph_replay_mismatch",
              "The recorded CUDA graph gave other bits than the verified call at the same signature; "
              "the graph was refused."),
    "TS802": (ERROR, "graph_bound_input_swapped",
              "An input bound by address (static= or update=) was not the tensor the graph was recorded with."),
    # truestar intref (docs/INTREF.md)
    "TS540": (INFO, "intref_convention_matched",
              "A device's integer output equals the integer reference under the named rounding "
              "convention(s) bit for bit."),
    "TS541": (ERROR, "intref_no_convention",
              "A device's integer output matches no known rounding convention on the same integer "
              "inputs; the closest one and its match rate are given."),
    "TS542": (WARNING, "intref_indistinguishable",
              "Every convention gives the same output on these inputs, so they cannot say which "
              "one the device uses; distinguishing inputs are needed."),
    "TS543": (WARNING, "intref_not_rounding",
              "A device differs from its control in both directions inside channels, which no "
              "requantization rule difference produces: the accumulators differ (cause before "
              "the requantizer, for example input quantization)."),
    "TS544": (ERROR, "intref_invalid_case",
              "The integer reference case or saved results are missing or invalid."),
    "TS545": (ERROR, "intref_golden_invalid",
              "A golden vector header failed its parse back check (lengths, checksums, values)."),

    # quantized kernels (docs/QUANT.md)
    "TS460": (ERROR, "quant_kernel_error",
              "A quantized matmul kernel differs from the exact float64 product of its dequantized "
              "operands by more than rounding allows: a kernel bug, not format loss."),
    "TS461": (ERROR, "quant_quantizer_error",
              "A run time activation quantizer wrote codes or scales that differ from the "
              "format's rounding rule: a kernel bug, not format loss."),
    "TS462": (WARNING, "quant_format_flips",
              "The format itself (float64 reference, no kernel involved) changes decisions "
              "beyond near ties: expected format loss, reported per format."),
    "TS463": (INFO, "quant_format_loss",
              "Format loss measured from float64 references only (loss delta, KL, flips); no "
              "decision changed beyond near ties."),
    "TS464": (ERROR, "quant_format_unsupported",
              "A quantized format was asked for that this GPU, Triton version or shape cannot "
              "run; nothing falls back to another path."),

    # WebGPU backend (truestar.webgpu, docs/WEBGPU.md)
    "TS560": (ERROR, "webgpu_session_on_cpu",
              "ONNX Runtime could not create the WebGPU EP for this graph and ran it on its CPU "
              "EP instead; nothing from this run is reported as a WebGPU result."),
    "TS561": (WARNING, "webgpu_nodes_on_cpu",
              "Some compute nodes of the graph ran on ONNX Runtime's CPU EP, not on WebGPU; they "
              "are named, and the result is WebGPU plus CPU."),
    "TS562": (ERROR, "onnx_export_failed",
              "The model (or a module) could not be exported to ONNX, so it was not checked on "
              "WebGPU; this is not a WebGPU finding."),
    "TS563": (ERROR, "onnx_export_differs",
              "The exported ONNX graph already differs from the float64 reference on ONNX "
              "Runtime's CPU EP: the export (or the CPU EP) changes the result, not WebGPU."),
    "TS564": (WARNING, "observer_effect",
              "The instrumented run (per module taps or profiling) gave a different final output "
              "than the plain run, so per module numbers from it are marked untrusted."),
    "TS565": (WARNING, "webgpu_nondeterministic",
              "Two identical runs on the WebGPU backend gave different outputs."),
    "TS566": (ERROR, "dawn_probe_divergent",
              "A single op compute shader run directly on Dawn differs from the float64 "
              "reference beyond the bar for its precision."),

    # parallel layouts against the single device step (docs/PARALLEL.md)
    "TS680": (ERROR, "parallel_layout_invalid",
              "The layout or the tensor parallel plan cannot run as given (unknown key, pipeline "
              "stages, a width or batch that does not divide); nothing was approximated."),
    "TS681": (ERROR, "parallel_forward_divergence",
              "Under the layout a module's output differs from the single device step while every "
              "module before it matches, or a rank holds a weight piece that is not the plan's split."),
    "TS682": (ERROR, "parallel_loss_normalization",
              "Every module output matches the single device step but the loss differs: the loss is "
              "normalized per micro batch or per replica instead of by the global count."),
    "TS683": (ERROR, "parallel_gradient_scale",
              "Every gradient is the single device gradient times one factor (dp, ga or their "
              "inverses) while activations match: a scaling collective or the accumulation."),
    "TS684": (ERROR, "parallel_missing_gradient_sync",
              "A replicated weight's gradients differ between ranks: the all reduce that sums its "
              "partial gradients (sequence parallel weights, a replica group) is missing."),
    "TS685": (ERROR, "parallel_backward_collective",
              "A module passes a wrong input gradient back while every later module's output "
              "gradient matches: the backward collective of that module is missing or wrong."),
    "TS686": (ERROR, "parallel_parameter_gradient",
              "A module's parameter gradients differ while its output and output gradient match, and "
              "no single factor or missing sync explains it."),
    "TS687": (ERROR, "parallel_update_divergence",
              "Every gradient matches but the weights after the optimizer step differ (gradient "
              "clipping on a local norm, an optimizer applied to shards differently)."),
    "TS688": (ERROR, "parallel_replica_drift",
              "A replicated weight differs between ranks after the step."),
    "TS689": (ERROR, "parallel_rank_failed",
              "A rank raised under the layout (shape error from a wrong split, a model that reads a "
              "sharded size, a collective that never completed); the module and line are named."),
    "TS690": (INFO, "parallel_simulated",
              "Ranks were simulated on this machine's CPU (threads or gloo processes): the layout's "
              "arithmetic is real, a GPU interconnect is not involved."),
    # real multi GPU ranks: layouts, data parallel and sharded state (docs/PARALLEL.md, docs/TRAINING.md)
    "TS820": (INFO, "parallel_real_ranks",
              "Ranks ran as separate processes on real devices joined by torch.distributed; the "
              "report names every rank's host, device, GPU model and backend."),
    "TS821": (ERROR, "parallel_devices_unavailable",
              "The ranks need more devices than this machine has, or the requested backend (NCCL, "
              "gloo) is missing; two ranks are never put on one GPU and nothing runs elsewhere."),
    "TS822": (ERROR, "parallel_rendezvous_failed",
              "The ranks could not join: rank 0's address did not answer, a firewall blocked the "
              "port, or a rank never arrived before the timeout. The address is named."),
    "TS823": (ERROR, "data_parallel_step_unsupported",
              "A data parallel training step setting would not compute the single device step (model "
              "state such as batch norm statistics, random ops drawn per replica, a batch that does not "
              "split evenly, loss scaling), so it is refused instead of run."),
    "TS824": (ERROR, "data_parallel_replicas_diverged",
              "After a data parallel step the replicas hold different parameters (their fingerprints "
              "differ), so they no longer train one model; the first differing tensor is named."),
    "TS825": (WARNING, "data_parallel_not_bitwise",
              "Bitwise equality with the single device step was asked for, but the reduction order or "
              "the devices do not allow it (native all reduce over 3 or more ranks, different GPU "
              "models); results are compared within tolerance instead."),
    # training spikes: float64 shadow and replay (docs/SPIKES.md)
    "TS691": (WARNING, "spike_shadow_gap",
              "On a shadowed training step the low precision loss or gradient is far from the same step "
              "in float64 (loss error above its running bar, gradient cosine below 0.9, or non finite)."),
    "TS692": (ERROR, "spike_numerics",
              "Replayed from a checkpoint, the spike happens in the run's precision and vanishes in "
              "float64: rounding or overflow caused it; the first module over its error bar is named."),
    "TS693": (ERROR, "spike_data",
              "The spike survives float64 and the batch is already high on the checkpoint weights: the "
              "batch caused it; the rows carrying the loss are named."),
    "TS694": (ERROR, "spike_optimizer",
              "The spike survives float64 but the batch is normal on the checkpoint weights: the updates "
              "since the checkpoint (learning rate, optimizer state) caused it."),
    "TS695": (ERROR, "spike_not_reproduced",
              "The replay does not reproduce the logged losses, so no verdict is given (nondeterministic "
              "run or state that was not saved)."),
    "TS696": (INFO, "spike_none",
              "The replayed step is within the spike factor of its baseline: no spike to explain."),
    "TS697": (ERROR, "replay_invalid",
              "The run directory cannot replay the step: no log, no checkpoint at or before it, or a "
              "missing batch."),

    # deployed artifact conformance (docs/CONFORM.md)
    "TS620": (ERROR, "conform_artifact_unsupported",
              "The artifact cannot be read, or its format cannot run on this machine; the reason is given."),
    "TS621": (ERROR, "conform_runtime_unavailable",
              "A runtime was asked for (or is a default for the format) but cannot run here; nothing "
              "else runs in its place."),
    "TS622": (ERROR, "conform_divergent",
              "The deployed artifact's output in this runtime diverges from the PyTorch model in float64."),
    "TS623": (WARNING, "conform_suspect",
              "The artifact's output error is above its calibrated limit but below the divergent cutoff."),
    "TS624": (ERROR, "conform_decision_flips",
              "Per row decisions of the artifact differ from the float64 model beyond near ties."),
    "TS625": (ERROR, "conform_conversion_fault",
              "The file itself, evaluated in float64, computes something else than the model: the "
              "conversion changed the math (named module when localized)."),
    "TS626": (ERROR, "conform_runtime_fault",
              "The file is right in float64 at the blamed module, and the runtime computes it wrongly."),
    "TS627": (WARNING, "conform_lower_precision",
              "The runtime's error is what a correct float16 implementation shows: it computes the "
              "float32 file in float16."),
    "TS628": (WARNING, "conform_off_target_nodes",
              "Some nodes ran off the requested device (provider fallback, outside the delegate or "
              "accelerator); each one is named."),
    "TS629": (WARNING, "conform_residency_unknown",
              "The runtime does not say where nodes ran, so silent fallback is not ruled out."),
    "TS630": (WARNING, "conform_observer_effect",
              "An instrumented run (profiling, preserved tensors, extra outputs) changed the outputs "
              "beyond run to run noise; its per node data is marked untrusted."),
    "TS631": (WARNING, "conform_not_localized",
              "The output is wrong but no single mapped module is at fault, or the format keeps no "
              "module names."),
    "TS632": (ERROR, "conform_golden_invalid",
              "A golden bundle is missing, damaged (a file's sha256 differs) or of an unknown version."),
    "TS633": (ERROR, "conform_baseline_regressed",
              "Rerun after a change: outputs moved and are now beyond the float64 tolerance; the "
              "fingerprint changes and the first changed module are named."),
    "TS634": (WARNING, "conform_baseline_changed",
              "Rerun after a change: outputs moved but still conform; the fingerprint changes and the "
              "first changed module are named."),
    "TS635": (ERROR, "conform_artifact_mismatch",
              "The artifact given is not the one the golden bundle was written for (sha256 differs)."),
    "TS636": (ERROR, "conform_runtime_failed",
              "The runtime could not load or run the artifact; its error is given."),
    "TS637": (ERROR, "conform_inputs_mismatch",
              "The model's inputs do not fit the artifact's signature (count, shape or dtype)."),
    "TS638": (INFO, "conform_meaning_unavailable",
              "The file's float64 meaning could not be evaluated (an op without a float64 rule), so "
              "conversion and runtime faults are not told apart."),
    "TS639": (INFO, "conform_baseline_identical",
              "Rerun after a change: outputs bit for bit identical to the recording; fingerprint "
              "changes are listed."),
    # truestar.nn, truestar.optim and train_step (docs/NN.md, docs/OPTIM.md)
    "TS760": (ERROR, "fit_bad_loss",
              "The loss function cannot make a training step: it is not a native function, has no "
              "parameter tree of that name, or its first result is not a float scalar."),
    "TS761": (ERROR, "fit_state_mismatch",
              "Model state was declared but the loss does not return (loss, new state) shaped like the "
              "state, or no initial state was given."),
    "TS762": (ERROR, "fit_bad_config",
              "An optimizer, clipping norm, precision or executor setting is invalid; the message names "
              "the accepted values."),
    "TS763": (ERROR, "fit_nonfinite_step",
              "A NaN or Inf in the loss or a gradient: the step was not applied, and the first op "
              "whose result is not finite (its operands finite) is named with the user's line."),
    "TS764": (ERROR, "fit_checkpoint_other_program",
              "A checkpoint was written by another step program (model, loss, optimizer settings or "
              "precision differ); its configuration is shown."),
    "TS765": (ERROR, "fit_checkpoint_mismatch",
              "A file is not a training checkpoint, or its tensors are missing, extra, misshapen or of "
              "another dtype; every problem is listed at once."),
    "TS766": (ERROR, "fit_mixed_precision_unsupported",
              "Mixed precision could not be applied to the loss program (for example it has a loop or "
              "a branch, which the autocast pass does not rewrite yet)."),

    # op coverage (docs/MODEL_COVERAGE.md, ir/importer_extra.py, scripts/op_coverage.py)
    "TS860": (ERROR, "import_setting_unsupported",
              "An ATen op the importer lowers was called with a setting the lowering cannot express "
              "exactly (avg_pool3d divisor_override, embedding_bag include_last_offset, recurrent "
              "projections or training dropout, output_padding not below the stride, reflect padding "
              "wider than the dim); the message names the op, the setting and the user's line."),
    "TS861": (ERROR, "import_size_not_fixed",
              "A lowering needs a fixed size that is symbolic here (a recurrent layer's sequence "
              "length, which is unrolled; an interpolated, padded or shuffled dim; a 3D depth)."),

    # model families refused by name (coverage refresh, docs/MODEL_COVERAGE.md, ir/refusals.py)
    "TS940": (ERROR, "moe_routing_unsupported",
              "The model has a mixture of experts layer Truestar does not lower (truestar.moe lowers routed "
              "top k experts stored as stacked gate_up_proj and down_proj with act(gate) * up, as in Mixtral, "
              "Qwen3 MoE, OLMoE and DeepSeek V3): nothing is imported; the message names the layer class, "
              "its path and why (transposed or biased experts, a clamped gate, per expert modules, expert "
              "capacity dropping)."),
    "TS941": (ERROR, "linear_attention_unsupported",
              "The model has a chunked linear attention or delta rule layer that Truestar does not lower "
              "(Kimi Delta Attention, lightning attention, mLSTM, a Gated DeltaNet on a transformers "
              "without the module function the lowering replaces): nothing is imported; the message names "
              "the layer class, its path and why it was not lowered."),
    "TS942": (ERROR, "chunked_ssm_unsupported",
              "The model has a chunked selective state space layer that Truestar does not lower (Mamba 2 "
              "SSD on a transformers without mamba2_chunk_scan, or an SSD class with no lowering): nothing "
              "is imported (the Mamba 1 sequential scan is supported); the message names the layer class, "
              "its path and why it was not lowered."),
    # Mamba 2 SSD and Gated DeltaNet layers lowered to linear_attention (ir/hf_linattn.py, P60)
    "TS1150": (ERROR, "linattn_lowering_call_unsupported",
               "A lowered Mamba 2 SSD or Gated DeltaNet layer was called with what the lowering does not "
               "cover: an initial recurrent state or a request for the final state (generation with a "
               "cache). Import the forward without a cache."),

    # vision and image generation coverage (docs/MODEL_COVERAGE.md, ir/refusals.py)
    "TS1260": (ERROR, "detection_postprocess_unsupported",
               "The model is a torchvision detector in eval mode (Faster, Mask or Keypoint R-CNN, RetinaNet, "
               "FCOS, SSD): proposal selection and post processing (top k, score threshold, non maximum "
               "suppression, RoI align on the kept boxes) return a number of boxes that depends on the "
               "values, so nothing is imported; the message names the module class and path. The backbone "
               "with its heads and the second stage on pooled features import on their own."),

    # engines and other frameworks as port candidates (docs/PORT_ENGINES.md)
    "TS640": (ERROR, "engine_op_divergent",
              "An engine graph node differs from the float64 replay of the same step on the "
              "engine's own inputs, with the GGUF's weights and settings and the engine's allowed "
              "rounding: an engine bug, not quantization or settings."),
    "TS641": (ERROR, "engine_setting_mismatch",
              "A model setting (rope theta, rope scaling, norm eps, head counts, dims) in the GGUF, "
              "or as the engine actually used it, differs from the reference."),
    "TS642": (ERROR, "engine_weight_layout",
              "A GGUF tensor holds a reference weight in another layout than the architecture "
              "needs (transposed, rows not permuted for its rotary convention) or with other values."),
    "TS643": (ERROR, "engine_weight_missing",
              "A reference weight has no GGUF tensor, or a GGUF tensor matches no reference weight."),
    "TS644": (INFO, "engine_quant_loss",
              "Quantization loss measured from float64 replays only (no engine involved): top 1 "
              "agreement, KL and per layer change; no decision changed beyond near ties."),
    "TS645": (WARNING, "engine_quant_flips",
              "The quantized weights themselves (float64 replay, no engine involved) change top 1 "
              "decisions beyond near ties: expected format loss, reported per type."),
    "TS646": (ERROR, "engine_tokenizer_mismatch",
              "The engine tokenizes the reference prompt into other ids than the reference "
              "(BOS added or missing, other pieces)."),
    "TS647": (ERROR, "engine_unavailable",
              "The engine library, the requested device or the model load is not available; "
              "nothing runs elsewhere instead."),
    "TS648": (ERROR, "engine_tap_unreadable",
              "An engine intermediate tensor needed for the replay could not be found or read "
              "(another graph naming or struct layout); no verdict is guessed."),
    "TS649": (WARNING, "engine_observer_effect",
              "The engine's logits with tensor taps differ from the untapped run, so per node data "
              "may not describe the untouched engine."),
    "TS650": (ERROR, "engine_reference_not_replayable",
              "Truestar's float64 replay of the architecture does not reproduce the reference "
              "model, or the GGUF architecture is not one Truestar replays."),
    "TS651": (ERROR, "engine_type_unsupported",
              "A GGUF tensor type that Truestar cannot decode exactly, or a quantization type the "
              "engine does not have."),
    "TS652": (ERROR, "framework_unavailable",
              "A candidate framework or backend (JAX, Flax, Keras with a given backend) is not "
              "installed or failed to start; nothing runs elsewhere instead."),
    "TS653": (ERROR, "framework_port_divergent",
              "A candidate layer in another framework, run alone on the reference layer's inputs, "
              "computes differently from the reference layer."),
    "TS654": (WARNING, "framework_port_suspect",
              "A candidate layer in another framework differs from the reference by more than "
              "float64 noise but less than a divergence."),
    "TS655": (ERROR, "framework_weights_unmatched",
              "Reference weights with no counterpart in the candidate framework's model, or "
              "candidate weights matching no reference weight."),
    "TS656": (ERROR, "framework_backend_divergent",
              "The same multi backend model gives another answer on one of its backends while "
              "another backend matches the reference."),
    "TS658": (INFO, "framework_lower_precision",
              "Candidate modules return float32 where the reference runs float64 (for example Keras forces "
              "64 bit results to 32 bit on its jax and torch backends); their thresholds use float32 noise."),
    "TS657": (ERROR, "engine_output_divergent_unattributed",
              "The final output differs beyond the explained quantization loss but no step, "
              "setting or weight is to blame."),

    "TS700": (ERROR, "kernel_config_resources",
              "A GPU launch config needs more local memory (spilled registers) than this GPU "
              "allows; it is refused before launch instead of failing inside the CUDA context. "
              "The tuner skips it; a default or forced config raises."),

    # Metal backend of the kernel layer and the compiled executor (docs/METAL.md)
    "TS840": (ERROR, "metal_unavailable",
              "Metal kernels were requested on a machine that cannot run them (not macOS, no MPS "
              "device, or a torch without torch.mps.compile_shader); nothing falls back to another device."),
    "TS841": (ERROR, "metal_float64_unsupported",
              "A kernel or model needs float64 arithmetic, which Metal does not have; it is refused "
              "before any launch."),
    "TS842": (ERROR, "metal_checks_unsupported",
              "Compiled in checks (sentinels) were requested for a Metal kernel; they are not generated "
              "for Metal yet."),
    "TS843": (ERROR, "metal_compile_failed",
              "The Metal compiler rejected Truestar's generated source for a kernel (a Truestar bug); "
              "the message names the kernel and the source file."),
    "TS844": (INFO, "metal_simdgroup_unavailable",
              "SIMD group matrices did not compile or did not compute an 8 x 8 product correctly on this "
              "device, so contraction kernels use threadgroup tiles with plain multiply adds."),
    "TS845": (ERROR, "metal_size_limit",
              "A Metal launch needs a size, stride or thread count beyond the 32 bit range its kernels use."),

    # AMD GPUs, ROCm and HIP (docs/ROCM.md)
    "TS880": (ERROR, "rocm_target_unknown",
              "An AMD target was named that the target table does not know; on a machine with that "
              "GPU the target is read from the device instead."),
    "TS881": (ERROR, "rocm_compile_failed",
              "Triton's AMD backend could not compile a kernel for an AMD target; the message names "
              "the kernel, its loop shape, the function, the launch config and the target."),
    "TS882": (ERROR, "rocm_lds_over_limit",
              "A compiled launch config needs more LDS (shared memory) than one workgroup has on the "
              "AMD target; it is refused before any launch."),
    "TS883": (ERROR, "rocm_unavailable",
              "ROCm was asked for but torch here is not a ROCm build or sees no AMD GPU; nothing "
              "falls back to another device."),
    "TS884": (INFO, "rocm_tf32_unavailable",
              "TF32 was allowed but the AMD target has no TF32 (xf32) dot in Triton, so float32 dots "
              "run as strict IEEE float32."),
    "TS885": (INFO, "rocm_compiled_not_run",
              "A kernel was compiled for an AMD target without running it; nothing about its "
              "results is verified until it runs on that GPU."),
    "TS886": (INFO, "rocm_default_relieved",
              "The default launch config spilled registers on the AMD GPU, so the first config of "
              "the relief ladder that does not spill (more waves, fewer stages, smaller tiles) is "
              "used; it is verified like any other config."),

    # chunked linear attention and SSM kernels (docs/LINEAR_ATTENTION.md, truestar.linattn, P53)
    "TS1060": (ERROR, "linattn_kernel_mismatch",
               "A chunked linear attention or state space kernel disagrees with the float64 "
               "sequential recurrence on its first call check (LinAttnMismatchError)."),
    "TS1061": (ERROR, "linattn_call_refused",
               "linear_attention refused a call: bad shapes, rule or beta, a device or dtype the "
               "requested implementation does not run, or a launch config it does not compile."),
    "TS1062": (WARNING, "linattn_chain_error_growth",
               "chain_error: the error against float64 of the last quarter of chunks is more than 4x "
               "that of the first quarter, so the state carries a growing error along the chunk chain."),
    "TS1063": (ERROR, "linattn_deterministic_chunk",
               "Deterministic mode pins the chunk size (the reduction order); a different chunk or a "
               "config without the pinned chunk was asked for and refused."),

    # precision planner (docs/PRECISION.md)
    "TS660": (INFO, "precplan_verified",
              "The per layer precision plan keeps decision flips against float64 within the "
              "budget on the calibration inputs and on the held out inputs."),
    "TS661": (WARNING, "precision_plan_tightened",
              "A plan within budget on the calibration inputs flipped too many held out "
              "decisions, so the calibration budget was tightened until both held."),
    "TS662": (ERROR, "precision_budget_unreachable",
              "No plan keeps the flip budget, not even float32 everywhere; the plan is float32 "
              "everywhere and the budget is reported as not met."),
    "TS663": (INFO, "precision_range_excluded",
              "truestar range found values that leave a 16 bit dtype's range or lose their "
              "digits in it, so those units never get that activation dtype."),
    "TS664": (ERROR, "precision_kernel_rejected",
              "A quantized kernel failed its exact check against the dequantized float64 "
              "product on a layer's own inputs; the candidate is removed as a kernel bug, never "
              "counted as format loss."),
    "TS665": (INFO, "precision_candidate_unavailable",
              "A precision cannot run for this layer on this GPU (shape or kernel limits); it is "
              "named and nothing is substituted for it."),
    "TS666": (ERROR, "precision_plan_mismatch",
              "A saved plan was applied to a model whose units or weights differ from the model "
              "it was made for."),
    "TS667": (INFO, "precision_units_tied",
              "Code between units mixes dtypes in an op that does not promote, so those units "
              "share one activation dtype (or stay float32)."),
    "TS668": (WARNING, "precision_range_unavailable",
              "The range check could not run on this model; range safety rests on measurement "
              "and verification only."),
    "TS669": (ERROR, "precision_needs_cuda",
              "The planner times real kernels and needs a CUDA GPU; nothing is planned on the "
              "CPU."),
    "TS670": (WARNING, "precision_baseline_faster",
              "A uniform precision baseline keeps the budget and measured faster than the plan."),

    # cross chip fingerprint (docs/CROSS_CHIP.md)
    "TS740": (ERROR, "fingerprint_data_mismatch",
              "A saved fingerprint weights or inputs file no longer matches its manifest; nothing ran."),
    "TS741": (ERROR, "fingerprint_not_run_to_run",
              "In deterministic mode two calls of the compiled program gave different bits on this GPU."),
    "TS742": (WARNING, "fingerprint_arm_failed",
              "A fingerprint arm or row raised (for example an unsupported dtype on this GPU); the error "
              "is recorded and that row has no hash."),
    "TS743": (ERROR, "fingerprint_claim_differs",
              "A deterministic mode row has different bits in the two fingerprints: the cross chip claim "
              "does not hold for it under the recorded conditions."),
    "TS744": (ERROR, "fingerprint_inputs_differ",
              "The two fingerprints started from different weights or inputs, so their outputs are not "
              "comparable."),
    "TS745": (INFO, "fingerprint_settings_differ",
              "The two fingerprints ran with different settings (TF32, deterministic algorithms, tuning, "
              "cache); each one is named."),
    "TS746": (INFO, "fingerprint_unclaimed_differs",
              "Rows of arms that carry no claim (eager, torch.compile, compiled default mode) differ "
              "between the two fingerprints; recorded, not a finding."),
    "TS747": (INFO, "fingerprint_env_differs",
              "The two fingerprints come from different GPUs, drivers or software versions; each one is "
              "named."),
    "TS748": (INFO, "fingerprint_row_missing",
              "Some fingerprint rows are present in one file only."),
    "TS749": (ERROR, "fingerprint_bad_file",
              "A file given to fingerprint compare is not a fingerprint."),
    "TS750": (WARNING, "fingerprint_data_not_golden",
              "The weights or inputs a fingerprint case built on this machine differ from the pinned "
              "hashes, so its outputs cannot be compared with other machines'."),

    "TS702": (WARNING, "prefix_cache_divergent",
              "A run with a cached prefix gives other bits than one full prefill of the whole sequence; "
              "the first token and layer that differ are named (truestar prefixcheck, docs/PREFIX_CHECK.md)."),
    "TS703": (ERROR, "prefix_cache_wrong",
              "Against a float64 reference, the cached prefix run is less accurate than the full prefill by "
              "more than the tolerance: not only rounded differently."),
    "TS704": (INFO, "prefix_cache_bitwise",
              "The cached prefix run gives the full prefill's bits."),

    # truestar serve: the LLM server whose answers do not change with batching (docs/SERVE.md)
    "TS1400": (INFO, "serve_bitwise_invariant",
               "A request gave the same tokens and logprobs bit for bit alone, batched with 7 others, with the "
               "prefix cache on and off and with chunked prefill on and off (truestar serve --check)."),
    "TS1401": (ERROR, "serve_invariance_broken",
               "A request's tokens or logprobs changed with the batch it ran in, the prefix cache or chunked "
               "prefill; the scenario and the first token that differs are named."),
    "TS1402": (ERROR, "serve_reference_mismatch",
               "The float32 server's greedy tokens differ from the float64 reference server's at a token that "
               "is not a near tie, or the reference server differs from a forward with no cache."),
    "TS1403": (ERROR, "serve_capacity_exceeded",
               "A request was refused: prompt plus max_tokens exceed the model length, the KV block pool, or "
               "(without chunked prefill) the tokens per step."),
    "TS1404": (ERROR, "serve_bad_request",
               "A request was refused: a malformed field, or a parameter the server does not implement asked for "
               "something other than its default (n, echo, penalties, logit_bias, ...); never ignored silently."),
    "TS1405": (INFO, "serve_seed_assigned",
               "A sampled request came without a seed; the server drew one and returned it, so the request can "
               "be replayed exactly."),
    "TS1406": (WARNING, "serve_compile_fallback",
               "The compiled serving step runs some ops outside Truestar kernels (fallbacks); the guarantee check "
               "still decides whether answers are invariant."),
    "TS1407": (ERROR, "serve_executor_unavailable",
               "The requested serving executor (cuda, or cpu without numba) cannot run on this machine."),
    "TS1408": (INFO, "serve_reference_agrees",
               "The float32 server's greedy tokens equal the float64 reference server's, and the reference server "
               "equals a plain forward with no KV cache."),
    "TS1409": (WARNING, "serve_baseline_not_invariant",
               "The eager PyTorch baseline gave a request other bits batched than alone (measured by "
               "scripts/bench_serve.py); shown for contrast with truestar serve."),

    # low precision formats: FP8 variants, MX, keyed stochastic rounding (docs/QUANT.md)
    "TS1000": (ERROR, "fp8_variant_mismatch",
               "An FP8 artifact's variant (OCP or FNUZ) is not the target GPU's native FP8: the "
               "same bits decode to other values there (a normal E4M3 code is worth half under "
               "FNUZ)."),
    "TS1001": (WARNING, "fp8_variant_ambiguous_name",
               "An FP8 dtype name does not say whether it is OCP FP8 or FNUZ FP8."),
    "TS1002": (INFO, "fp8_variant_inferred",
               "An FP8 tensor's variant was not named; it was inferred from codes that only one "
               "variant allows."),
    "TS1003": (ERROR, "fp8_variant_bits_conflict",
               "An FP8 tensor is named one variant but holds codes that only the other variant "
               "uses as values."),
    "TS1004": (WARNING, "fp8_variant_undetermined",
               "Neither an FP8 tensor's name nor its codes decide whether it is OCP or FNUZ."),
    "TS1005": (INFO, "fp8_target_unknown",
               "The target GPU's native FP8 variant is not known to Truestar; no variant "
               "comparison was made."),
    "TS1006": (ERROR, "sround_bits_differ",
               "A stochastic rounding kernel's codes differ from the keyed reference, which knows "
               "every random draw."),
    "TS1007": (ERROR, "sround_biased",
               "Keyed stochastic rounding failed the statistical test: its mean error or variance "
               "over many draws is not that of unbiased rounding."),
    "TS1008": (ERROR, "mx_scale_differs",
               "An MX quantizer's shared block scales differ from the declared scale rule."),
    "TS1009": (ERROR, "mx_element_differs",
               "An MX quantizer's element codes are not round to nearest even of the input over "
               "its own block scale."),

    # training numerics: Muon, low precision GEMM training, invariant sentinels, attention logit bound
    # (docs/TRAIN_NUMERICS.md, ledger row P52)
    "TS1040": (ERROR, "muon_orthogonality_exceeded",
               "A Muon update is farther from the float64 result of its Newton Schulz iteration than the "
               "rounding of its dtype allows: the iteration or its kernel is wrong, not just imprecise."),
    "TS1041": (INFO, "muon_orthogonality",
               "The orthogonality of a Muon update against float64: the iteration's intended distance from "
               "the polar factor and the rounding its dtype adds, each measured."),
    "TS1042": (ERROR, "lowp_gemm_kernel_error",
               "A low precision training GEMM (forward, input gradient or weight gradient) is off the "
               "float64 product of its own dequantized operands by more than float32 accumulation allows."),
    "TS1043": (WARNING, "lowp_weight_inconsistent",
               "The input gradient of a low precision linear multiplies by other weight values than the "
               "forward did (per channel or 1D block scales of W and W^T differ), so the gradient is the "
               "gradient of another weight."),
    "TS1044": (WARNING, "lowp_gradient_over_budget",
               "A layer's weight gradient in the low precision recipe is farther from float64 than the "
               "budget, so the plan keeps it in higher precision."),
    "TS1045": (INFO, "lowp_format_unavailable",
               "A low precision training format is not available here (not in truestar.quant, or no "
               "kernel for this GPU); it is named and nothing is substituted for it."),
    "TS1047": (ERROR, "invariant_prob_rows",
               "Rows of a softmax output do not sum to one within the tolerance derived from their dtype "
               "and length."),
    "TS1048": (ERROR, "invariant_softmax_grad_rows",
               "Rows of a softmax input gradient do not sum to zero (to delta (1 - sum p)) within the "
               "dtype derived tolerance: the gradient used other probabilities than its delta (a stale or "
               "requantized delta) or the kernel is wrong."),
    "TS1049": (ERROR, "invariant_normalized_rows",
               "Rows normalized as x / sum(x) (router gates) do not sum to one within the dtype derived "
               "tolerance."),
    "TS1051": (INFO, "logit_bound_safe",
               "Proven from the weights: no input the norm in front can produce makes an attention logit "
               "leave the target format's finite range."),
    "TS1052": (WARNING, "logit_bound_may_overflow",
               "The weight based upper bound of an attention logit is outside the target format's range, "
               "but no constructed input reaches it yet."),
    "TS1053": (ERROR, "logit_overflow_reachable",
               "A valid input (built from the weights and run through the real modules) gives an attention "
               "logit outside the target format's range; running attention in that format is refused."),

    # canonical reduction tree for tensor parallel (truestar.parallel.tree, docs/PARALLEL.md, P55)
    "TS1100": (INFO, "tp_tree_bitwise",
               "The tensor parallel step through the canonical reduction tree is bitwise equal to the one "
               "device step: loss, output, every gradient and every updated weight on every rank."),
    "TS1101": (ERROR, "tp_tree_differs",
               "A value of the tensor parallel step through the canonical tree differs bitwise from the one "
               "device step; the first ones are named."),
    "TS1102": (ERROR, "tp_tree_unsplittable",
               "The canonical tree cannot split this layout or layer (a split axis that does not cut into "
               "tp's odd factor of equal blocks, or tp's power of two part not dividing a block's group "
               "count); nothing was approximated."),
    "TS1103": (INFO, "tp_plain_differs",
               "For contrast, the plain tensor parallel step (F.linear, rank order sums) differs from its one "
               "device step; recorded, not a finding about the tree."),
    # min cut recompute and host offload (truestar.compiler.remat, docs/MEMORY.md, P55)
    "TS1110": (INFO, "remat_mincut_plan",
               "Values to recompute were chosen by a minimum cut over the forward values backward needs; "
               "the plan lists them with their recompute work."),
    "TS1111": (INFO, "remat_offload_plan",
               "Values stored across the forward and backward parts are copied to pinned host memory and "
               "back at positions fixed at compile time."),
    "TS1112": (ERROR, "offload_reload_mismatch",
               "A value reloaded from host memory is not bitwise equal to the value that was offloaded; the "
               "result was not returned."),
    "TS1113": (ERROR, "offload_unavailable",
               "Host offload was asked for a program that does not run on CUDA (it needs pinned host memory "
               "and a copy stream)."),
    # persistent compile cache (truestar.compile_cache, docs/COMPILER.md, Compile cache)
    "TS1350": (INFO, "compile_cache_hit",
               "A first call was accepted from a compile cache record made by a verified call on the same "
               "inputs: the same kernel binaries ran and the outputs are bitwise equal to the recorded ones."),
    "TS1351": (WARNING, "compile_cache_entry_unreadable",
               "A compile cache entry failed to parse, has another format version or a wrong digest; it was "
               "ignored, deleted and is rebuilt."),
    "TS1352": (INFO, "compile_cache_trimmed",
               "The compile cache was over its size cap (TRUESTAR_CACHE_MAX_MB); the least recently used "
               "entries were deleted."),
    "TS1353": (WARNING, "compile_cache_write_failed",
               "A compile cache record could not be written (read only or full disk); the call itself "
               "succeeded and the next process verifies again."),
    "TS1354": (INFO, "compile_cache_replay_mismatch",
               "A compile cache record existed for these inputs but the unchecked call launched other kernel "
               "binaries or returned other bits; the call was verified in full instead."),

    "TS705": (ERROR, "spec_rows_variant",
              "The target's logits for a row depend on how many query rows the call has: speculative "
              "decoding with it would not give the tokens of plain greedy decoding (truestar.specdec)."),
    "TS706": (INFO, "spec_rows_invariant",
              "The target gives the same bits with one query row and with the verify width: speculative "
              "greedy decoding returns plain greedy decoding's tokens."),
    "TS701": (ERROR, "kv_quant_unsupported",
              "A quantized KV cache was asked for in a format, layout or shape the decode kernel "
              "cannot run (truestar.kvquant); nothing falls back to another path."),

    # lean test on rounding errors and the summation order probe (docs/BIAS.md, docs/ORDERPROBE.md);
    # TS960 to TS979 are reserved for them
    "TS960": (ERROR, "kernel_errors_lean",
              "A compiled Truestar kernel's errors against float64 are within the size tolerance but "
              "lean one way (mean signed error beyond what unbiased round to nearest allows, and above "
              "1/64 of the tolerance); the kernel was refused."),
    "TS961": (WARNING, "module_errors_lean",
              "A module's own errors on this backend are within the size threshold but lean one way "
              "(toward zero, or toward one infinity) beyond what unbiased round to nearest allows; "
              "training adds such a push up over many steps."),
    "TS962": (INFO, "order_recovered",
              "The order probe recovered the summation order a backend uses for a reduction, and the "
              "recovered order replayed in exact arithmetic predicts the backend's result bit for bit."),
    "TS963": (WARNING, "order_not_repeatable",
              "The same probe input gave different results across runs on this backend: the reduction "
              "order is not fixed (atomics or a run dependent split)."),
    "TS964": (WARNING, "order_unresolved",
              "The order probe could not explain a backend's result with any order it recovers "
              "(fused multiply adds with other rounding, wider accumulators, or a non tree method)."),
    "TS965": (INFO, "orders_differ",
              "Two backends sum the same reduction in different orders, which is why their float "
              "results can differ bit for bit while both are correct."),
    "TS966": (WARNING, "tf32_inputs_lean",
              "A Truestar kernel compiled with allow_tf32 leans toward zero: the TF32 tensor cores "
              "truncate float32 operands instead of rounding them. Reported, not refused (the caller "
              "chose TF32); compile in strict float32 where a lean matters, such as training."),

    # saved Qualcomm AI Hub outputs of the fixed battery, judged here (truestar conform qaihub,
    # docs/QAIHUB.md)
    "TS920": (ERROR, "qaihub_bundle_invalid",
              "The AI Hub results directory cannot be read, is from another battery version, or a "
              "file in it does not match its run."),
    "TS921": (ERROR, "qaihub_inputs_mismatch",
              "A model's recorded input hash differs from the inputs this battery regenerates, so its "
              "runs are not judged."),
    "TS922": (ERROR, "qaihub_run_failed",
              "A compile or inference job failed, so the run has no output to judge."),
    "TS923": (WARNING, "qaihub_float32_ran_float16",
              "A float32 model ran at float16 level on the device: every tap within 4x of the "
              "sim:ieee-f16 prediction, so correct float16 arithmetic, not float32."),
    "TS924": (ERROR, "qaihub_beyond_float16",
              "A tap's error against float64 is more than 4x what correct float16 arithmetic "
              "(sim:ieee-f16) gives on the same model and inputs."),
    "TS925": (INFO, "qaihub_float32_on_accelerator",
              "A run asked for the NPU or GPU but every tap is float32 accurate: it ran on the CPU or "
              "in float32 (the per layer placement is named)."),
    "TS926": (WARNING, "qaihub_int8_one_step",
              "An INT8 layer, replayed from the device's own input, has elements one step from the "
              "intended arithmetic (ONNX QLinear)."),
    "TS927": (ERROR, "qaihub_int8_divergent",
              "An INT8 layer, replayed from the device's own input, has an element more than one "
              "step from the intended arithmetic."),
    "TS928": (INFO, "qaihub_int8_convention",
              "An INT8 layer matches another truestar.intref convention better than ONNX QLinear "
              "(the convention is named)."),
    "TS929": (ERROR, "qaihub_output_missing",
              "A tap the battery defines is missing from a run's outputs or has the wrong size."),

    # saved accelerator outputs of the fixed battery, judged here (truestar conform tpu, docs/TPU.md)
    "TS900": (ERROR, "tpu_bundle_invalid",
              "The battery output directory cannot be read, is from another battery version, or a "
              "file in it does not match its case."),
    "TS901": (ERROR, "tpu_case_divergent",
              "A battery case's saved output is outside even the bfloat16 budget against the "
              "float64 reference: the result is wrong, not just imprecise."),
    "TS902": (WARNING, "tpu_float32_reduced_precision",
              "A float32 battery case computed at bfloat16 or TF32 level (on a TPU usually the "
              "default matmul precision: one bfloat16 pass per product)."),
    "TS903": (WARNING, "tpu_not_bitwise_stable",
              "Identical calls of the same compiled case gave different bits."),
    "TS904": (ERROR, "tpu_inputs_mismatch",
              "The run's recorded input hash differs from the inputs this battery regenerates, so "
              "its outputs are not judged."),
    "TS905": (WARNING, "tpu_elementwise_ulp_high",
              "An elementwise function is float32 accurate overall but its worst element is more "
              "than 4 ulp off the correctly rounded value."),
    "TS906": (ERROR, "tpu_case_failed",
              "A battery case raised on the accelerator, so it has no output to judge."),

    # device breadth: Windows ML EPs, LiteRT NPU, WebNN, QAIRT range modes (ledger row P56)
    "TS1120": (ERROR, "winml_session_on_cpu",
               "ONNX Runtime could not create the requested Windows ML execution provider for this "
               "graph and ran it on its CPU EP; nothing from this run is reported as that EP's."),
    "TS1121": (WARNING, "winml_nodes_on_cpu",
               "Some compute nodes ran on ONNX Runtime's CPU EP, not on the requested Windows ML "
               "execution provider; they are named, and the result is that EP plus CPU."),
    "TS1122": (WARNING, "winml_nondeterministic",
               "Two identical runs on a Windows ML execution provider gave different outputs."),
    "TS1123": (INFO, "winml_ep_host_runtime",
               "A Windows ML catalog EP was registered into an ONNX Runtime build other than the one "
               "Windows ML ships; the EP's own kernels ran, the host runtime's version is named."),
    "TS1124": (WARNING, "litert_npu_partial",
               "LiteRT put only part of the graph on the NPU; the operators left outside the NPU "
               "partitions ran on the CPU or GPU fallback and are named."),
    "TS1125": (ERROR, "litert_npu_silent_cpu",
               "The NPU was requested but LiteRT never loaded an NPU compiler or dispatch library, "
               "so the whole model ran on its CPU fallback while is_fully_accelerated() said True."),
    "TS1126": (INFO, "litert_npu_static_partition",
               "An ahead of time compiled LiteRT file was read without an NPU: its NPU dispatch "
               "partitions and the operators outside them are listed from the file alone."),
    "TS1127": (ERROR, "webnn_unsupported_op",
               "An ONNX node has no WebNN translation in Truestar, so the graph was not run on "
               "WebNN; nothing is replaced by another backend."),
    "TS1128": (WARNING, "webnn_nondeterministic",
               "Two identical WebNN runs in the browser gave different outputs."),
    "TS1129": (ERROR, "webnn_context_failed",
               "The browser could not create a WebNN context or build the graph for the requested "
               "device type."),
    "TS1130": (INFO, "qairt_mode_reproduced",
               "A device's quantized tensor is reproduced bit for bit by one QAIRT range mode and "
               "tie rule (named)."),
    "TS1131": (WARNING, "qairt_one_step_in_mode",
               "A device's quantized tensor uses one QAIRT range mode's grid but some elements differ "
               "by exactly one step: a tie or input rounding difference, not a kernel bug."),
    "TS1132": (ERROR, "qairt_mode_unexplained",
               "No QAIRT range mode and tie rule explains a device's quantized tensor: the encoding "
               "came from elsewhere, or values differ by more than a step (look at the kernel)."),
    "TS1133": (WARNING, "qairt_not_on_grid",
               "The device values are not on one quantization grid, so no range mode can be "
               "attributed."),

    # sim:hexagon fitted to real Snapdragon outputs (docs/DEVICES.md section 7)
    "TS1140": (INFO, "hexagon_fitted_op",
               "sim:hexagon computed an op with the behaviour fitted to real Snapdragon NPU outputs "
               "(GELU: the device's approximation, fused with the linear layer before it); the fit, "
               "its range and its held out error are in docs/DEVICES.md section 7."),
    "TS1141": (INFO, "hexagon_op_unmeasured",
               "sim:hexagon computed an op whose device behaviour Truestar has not modelled as "
               "correct float16 (named, with what is known: softmax and attention are 1.7x to 2.0x "
               "further from float64 on the device)."),
    # dynamic shapes: one compile per range, checks per bucket (docs/DYNAMIC_SHAPES.md)
    "TS1420": (ERROR, "dynamic_shape_out_of_range",
               "A call's symbolic dim is outside its declared range (or its shapes do not fit the "
               "program): refused, never recompiled."),
    "TS1421": (ERROR, "dynamic_bitwise_needs_deterministic",
               "Bitwise equality with the static compile was asked for without deterministic launches; "
               "split K and tuned reduction blocks would make a shape's bits depend on the GPU and history."),
    "TS1422": (ERROR, "dynamic_buckets_invalid",
               "A bucket list is not increasing, starts below the dim's minimum or does not cover its "
               "declared maximum."),
    "TS1423": (ERROR, "dynamic_bitwise_library_op",
               "Bitwise equality with the static compile was asked for on a program that runs library "
               "ops (cuDNN chooses algorithms by shape), which Truestar cannot pin."),
    "TS1424": (ERROR, "dynamic_declaration_invalid",
               "The dynamic dims declaration does not match the model or program (unknown dim, missing "
               "examples, no symbolic dim, or a range torch.export changed)."),
    "TS1425": (ERROR, "dynamic_conv_symbolic_channels",
               "A convolution sums over a symbolic channel dim whose range includes counts below 16: the "
               "static compile of such a count loops over a flat K, the symbolic one over taps and channel "
               "blocks, so the summation order (and the bits) would differ."),
    "TS1426": (ERROR, "dynamic_static_bits_differ",
               "At a shape the dynamic program's result is not bitwise the static compile's (found by "
               "prove); the elements are counted."),
    "TS1427": (ERROR, "dynamic_prove_needs_module",
               "prove compiles the static program per shape from the torch module; an IR program has no "
               "static source."),
    # Core ML battery: placement and accuracy of saved runs (docs/COREML.md)
    "TS1160": (WARNING, "coreml_silent_fallback",
               "A Core ML run asked for the GPU or the Neural Engine, but Core ML's compute plan put "
               "operations on another unit that the requested one supports; nothing in the API reports it."),
    "TS1161": (WARNING, "coreml_unit_cannot_run_precision",
               "A Core ML run asked for a unit that supports none of the program's operations at its "
               "precision (the Neural Engine with a FLOAT32 program), so every operation ran elsewhere."),
    "TS1162": (WARNING, "coreml_float32_at_float16",
               "A FLOAT32 Core ML program gave an error at float16 level, not float32."),
    "TS1163": (ERROR, "coreml_beyond_float16",
               "A Core ML run's error is above correct float16 arithmetic (sim:ieee-f16) by more "
               "than the allowed factor; the ratio to a float16 accumulator (sim:mali-fp16) is given."),
    "TS1164": (ERROR, "coreml_run_failed",
               "A battery model failed to convert or run on Core ML, or its output has the wrong size."),
    "TS1165": (INFO, "coreml_no_compute_plan",
               "No Core ML compute plan was saved for a run, so its placement is not judged."),
    "TS1166": (ERROR, "coreml_input_hash_mismatch",
               "A run's battery input hash is not this Truestar's, so the run is not judged."),
    "TS1167": (WARNING, "coreml_nondeterministic",
               "Two identical Core ML predictions gave different bits."),
    "TS1168": (ERROR, "coreml_bundle_unreadable",
               "The Core ML results directory is missing, unreadable or of another battery version."),

    # order free sums (docs/ORDER_FREE_SUM.md)
    "TS1080": (ERROR, "order_free_sum_refused",
               "An order free sum was asked for on terms it cannot take: not float32, or more than "
               "2**24 - 1 terms per row, where the exact int64 total could overflow."),
    "TS1081": (ERROR, "order_free_sum_backend_unsupported",
               "A kernel with an order free sum was compiled for a backend that does not generate "
               "them; nothing falls back to a plain sum. Every kernel backend generates them since "
               "2026-11-25 (Metal included), so no current backend raises it."),

    # sliced float64 matmul on int8 tensor cores, verification only (docs/SLICED_F64.md)
    "TS1085": (ERROR, "sliced_f64_mismatch",
               "The sliced float64 matmul used by verification disagreed with a CPU float64 matmul "
               "beyond both proven bounds on its first call for a shape; it is a bug and is never used."),
    "TS1086": (ERROR, "sliced_f64_unavailable",
               "The sliced float64 matmul cannot run here (no CUDA tensors, non finite operands or "
               "unsupported shapes); verification keeps the native float64 reference."),

    # operator local bounds for comparing two stacks (docs/TOLERANCE.md)
    "TS1098": (WARNING, "local_bound_look_closer",
               "Two stacks differ on an operator by more than the probable rounding bound but within "
               "the worst case bound for the declared precision: rounding is possible, look closer."),
    "TS1099": (ERROR, "local_bound_not_rounding",
               "Two stacks differ on an operator, on the same inputs, by more than twice the worst "
               "case rounding bound for the declared precision: the difference is not rounding."),

    # tensor core models (docs/TENSOR_CORE_MODELS.md)
    "TS1094": (INFO, "tensor_core_unvalidated",
               "A simulated tensor core profile whose generation Truestar has not measured: its "
               "unit model follows a published description and was not compared with a device."),
    "TS1095": (WARNING, "tensor_core_format_unmodelled",
               "A matmul's input or accumulator format has no path in the profile's matrix unit "
               "model; it was accumulated exactly and rounded once instead, and is named."),
    "TS1096": (INFO, "tensor_core_op_unmodelled",
               "Ops that run on matrix units on the device (convolution, fused attention) were "
               "computed exactly on the tensor core profile; they are named."),
    "TS1097": (INFO, "tensor_core_schedule_assumed",
               "The tensor core profile chains the unit's groups in K order from zero, one warp's "
               "instruction chain; a library may split K, use another instruction shape or promote "
               "the accumulator, so the device can differ where the unit model is exact."),

    # portable tensor cores: exact tensor core blocks (docs/PORTABLE_TC.md)
    "TS1330": (ERROR, "portable_tc_backend_unsupported",
               "Portable tensor cores were asked for on a backend that does not generate them "
               "(only CUDA Triton does); refused, never a silent native fallback."),
    "TS1331": (ERROR, "portable_tc_device_unproven",
               "Portable tensor cores were asked for on a GPU the exactness proof does not cover "
               "(AMD, or NVIDIA below sm_80)."),
    "TS1332": (ERROR, "portable_tc_block_unsupported",
               "A dot product's K block is outside 16 to 256 columns, where an exact tensor core "
               "block's integer sums could exceed 2**24."),
    "TS1333": (ERROR, "portable_tc_library_ops",
               "library= ops run through vendor kernels whose tensor core rounding differs by GPU, "
               "so they cannot be combined with portable tensor cores."),

    # round once elementwise math (docs/ROUND_ONCE.md)
    "TS1089": (ERROR, "round_once_backend_unsupported",
               "Round once elementwise math was asked for on a backend that does not generate it "
               "(the Triton, numba and Metal backends do); refused, never a silent native fallback."),
    "TS1090": (ERROR, "math_mode_unknown",
               "The math compile option names a mode Truestar does not have (use default or "
               "round_once)."),
    "TS1091": (ERROR, "round_once_dtype_unsupported",
               "A round once function or kernel was given a dtype it does not take: inputs and "
               "outputs must be float32, float16 or bfloat16."),
    "TS1092": (ERROR, "round_once_op_unsupported",
               "A kernel compiled with round once math uses an op that has no round once form (it "
               "covers exp, log, tanh, erf, rsqrt, sigmoid; sqrt, abs and neg are exact natively)."),
    "TS1093": (ERROR, "round_once_shape_unsupported",
               "The kernel's loop shape evaluates a transcendental itself (the attention online "
               "softmax) and has no round once form yet."),

    # verified rotations before quantization (docs/ROTATION.md)
    "TS1020": (ERROR, "rotation_equivalence_failed",
               "A rotated model does not compute the original model in float64 within the stated "
               "bound; the first point that differs and the folding steps before it are named, and "
               "nothing is quantized."),
    "TS1021": (ERROR, "rotation_not_orthogonal",
               "A rotation matrix given to the rotation pass is not orthogonal to float64 accuracy."),
    "TS1022": (ERROR, "rotation_unsupported",
               "A rotation was asked for a size with no Hadamard construction here, an unknown model "
               "layout or an unknown step; nothing falls back to a random rotation."),
    "TS1023": (ERROR, "walsh_kernel_mismatch",
               "The fast Walsh transform kernel (online rotations) disagrees with the float64 "
               "transform on its first call."),
    "TS1024": (ERROR, "rotation_hurts_format",
               "A rotated model was given to a format a rotation makes worse: block and float formats "
               "(int4 grouped, nvfp4, mxfp4, mxfp8, fp8 at any granularity) lost 1.9x to 5.9x more KL "
               "with every rotation measured, weight only integers 1.6x to 2.4x (docs/ROTATION.md). "
               "Rotations "
               "are offered for per token integer activation formats only (W4A4, W8A8 int)."),

    # mixture of experts (truestar.moe, docs/MOE.md)
    "TS1180": (ERROR, "moe_expert_index_invalid",
               "The grouped experts were given an expert index outside 0..E-1; the message counts the slots."),
    "TS1181": (ERROR, "moe_experts_mismatch",
               "The grouped experts (gather per expert, one matmul pair per expert, a fixed slot order "
               "combine) differ from the same math in float64 beyond the kernel tolerance; checked on the "
               "first call per shape, dtype and device."),
    "TS1182": (WARNING, "moe_router_near_tie",
               "A token's k-th and (k+1)-th router scores differ by less than the float64 derived rounding "
               "bound of the working precision: which experts it reaches is not determined there. The "
               "report names the layer, the token, the two experts, the gap and the bound."),
    "TS1183": (ERROR, "moe_router_flip",
               "The working precision run sent a token to a different set of experts than float64 although "
               "the choice is not a near tie: a routing divergence beyond rounding."),
    "TS1184": (INFO, "moe_router_unmeasured",
               "A covered MoE layer's router uses a scoring rule the near tie report does not bound (not "
               "softmax top k, not DeepSeek's sigmoid plus bias); it is listed, not skipped."),
    # multi head latent attention with a latent KV cache (docs/MLA.md)
    "TS1200": (ERROR, "mla_variant_unsupported",
               "An MLA model has a feature the latent cache lowering was not written for (a variant "
               "whose attention code differs from DeepSeek V3's, a kv_b_proj bias, fewer KV heads than "
               "query heads, a sequence length dependent RoPE, an unknown mode); the layer and the "
               "reason are named and nothing is built."),
    "TS1201": (ERROR, "mla_absorption_mismatch",
               "The absorbed MLA decode (W_uk folded into the query, W_uv after the attention) does "
               "not compute the expanded decode in float64 within the stated bound; the first step "
               "and layer that differ are named, and nothing is compiled."),
    "TS1202": (ERROR, "sparse_attention_selection_refused",
               "The model selects a top k of keys per query with an indexer (DeepSeek Sparse "
               "Attention and its relatives); the class and module path are named, nothing is "
               "imported, and no dense or approximate fallback is built."),
    "TS1203": (ERROR, "uninitialized_int_buffer",
               "A torch.empty buffer of an integer or bool dtype was not imported: unwritten float "
               "entries are NaN in the IR (as under PyTorch's deterministic mode), and no integer "
               "value can mark an unwritten entry."),
    # data parallel and reduce scatter beyond 2 ranks (docs/PARALLEL.md, docs/TRAINING.md)
    # fused MLA latent attention (docs/MLA.md, decode="fused")
    "TS1460": (ERROR, "mla_fused_kernel_shape_unsupported",
               "The fused MLA latent attention kernel was given shapes it does not run (a latent rank "
               "above 1024, a head dim above 256, mismatched operands, or tensors off CUDA); the shapes "
               "are named and decode='absorb' is the alternative."),
    "TS1210": (ERROR, "data_parallel_unequal_replica_batches",
               "The replicas of a data parallel step hold batches with different row counts: the mean over "
               "replicas would not be the batch mean. Refused on every replica before any compute; the "
               "counts and how many rows to pad or drop are named."),
    "TS1211": (ERROR, "reduce_scatter_uneven",
               "A reduce scatter's dim does not split into equal pieces over its group; refused on every "
               "backend (simulated ranks, fixed order and gloo), naming the dim and the group."),
    # llama.cpp conformance (`truestar conform llamacpp`, docs/CONFORM_LLAMACPP.md)
    "TS1310": (ERROR, "llamacpp_unavailable",
               "No usable llama.cpp build: the directory, libllama, its headers, a C++ compiler for "
               "Truestar's driver, or the engine run itself is missing or failed; the reason is named "
               "and nothing runs in its place."),
    "TS1311": (ERROR, "llamacpp_logits_divergent",
               "llama.cpp's logits differ from the float64 replay of the GGUF's own decoded weights "
               "beyond the derived tolerance (10 times the larger of float32 noise and the rounding "
               "the engine is allowed to do) at some position: an engine error, not quantization."),
    "TS1312": (ERROR, "llamacpp_greedy_divergent",
               "Greedy decoding in llama.cpp picks another token than the float64 replay of the same "
               "file, at a step where the replay's top two logits are not a near tie."),
    "TS1313": (INFO, "llamacpp_greedy_near_tie",
               "Greedy decoding in llama.cpp differs from the float64 replay only at a near tie (top "
               "two logits within twice the rounding bound): ambiguous, not a bug."),
    "TS1314": (INFO, "llamacpp_quantization_loss",
               "What quantization alone changes: the GGUF's decoded weights against the source weights, "
               "both replayed in float64, reported apart from the engine's error."),
    "TS1315": (WARNING, "llamacpp_quantization_flips",
               "Quantization alone changes decisions: teacher forced top 1 flips beyond near ties, or "
               "greedy steps, between the source weights and the GGUF's decoded weights in float64."),
    "TS1316": (ERROR, "llamacpp_conversion_fault",
               "The GGUF does not hold the source model: a setting differs by name, or a tensor is in "
               "another layout or has other values than its type's rounding explains."),
    "TS1317": (WARNING, "llamacpp_settings_change_tokens",
               "llama.cpp's greedy tokens change with engine settings (threads, batch, flash "
               "attention); each change says whether it is at a near tie."),
    "TS1318": (INFO, "llamacpp_settings_change_bits",
               "llama.cpp's logits bits differ between engine settings while the greedy tokens stay "
               "the same."),
    "TS1319": (ERROR, "llamacpp_model_unsupported",
               "The check cannot run on this model: an architecture or tensor type the float64 replay "
               "does not cover, a malformed file, or llama.cpp's own converter or quantizer failed."),
    # GPU serving engines (truestar conform vllm | sglang, docs/CONFORM_SERVING.md)
    "TS1320": (ERROR, "serving_engine_unavailable",
               "The serving engine cannot run here (its package is not installed or there is no CUDA GPU); "
               "run the battery where it can and judge the saved results directory."),
    "TS1321": (ERROR, "serving_engine_refused",
               "This serving engine is named but not supported in this release (TensorRT-LLM); nothing "
               "falls back to another engine."),
    "TS1322": (ERROR, "serving_run_failed",
               "An engine setting failed to run, or a serving results directory cannot be judged (missing "
               "files, other weights or prompts than its spec rebuilds)."),
    "TS1323": (ERROR, "serving_logprob_divergent",
               "Engine log probabilities differ from the float64 reference of the same weights by more than "
               "the tolerance a correct run in the serving dtype sets."),
    "TS1324": (ERROR, "serving_decision_divergent",
               "The engine's top 1 or greedy token differs from the float64 argmax given the same tokens, and "
               "the float64 margin between them is larger than a near tie."),
    "TS1325": (INFO, "serving_near_tie",
               "The engine's top 1 or greedy token differs from float64 only at a near tie (margin within "
               "twice the tolerance); either choice is correct."),
    "TS1326": (WARNING, "serving_repeat_differs",
               "The same request sent twice to the same engine gave different results."),
    "TS1327": (WARNING, "serving_batch_differs",
               "A prompt's results change when it is batched with other prompts (batch 1 vs batch 8)."),
    "TS1328": (WARNING, "serving_prefix_cache_differs",
               "Turning prefix caching on changes a prompt's results."),
    "TS1329": (WARNING, "serving_chunked_prefill_differs",
               "Turning chunked prefill on changes a prompt's results."),
    # data loader and checkpoint format (docs/DATA.md)
    "TS1380": (ERROR, "data_bad_config",
               "A loader setting is invalid: batch size below 1, world size below 1, rank outside the world, "
               "or an unknown worker mode."),
    "TS1381": (ERROR, "data_source_invalid",
               "A data source cannot be read: arrays of different lengths, an empty folder, a missing file, "
               "or an index out of range."),
    "TS1382": (ERROR, "data_state_mismatch",
               "A saved loader position was written with other settings (seed, batch size, shuffle, "
               "drop_last or rank) than the loader it is restored into, so the resume would not be exact."),
    "TS1383": (ERROR, "data_dataset_changed",
               "The dataset differs from the one the saved loader position was taken on (other length or "
               "fingerprint)."),
    "TS1384": (ERROR, "data_worker_failed",
               "A worker raised while loading a batch; the message names the epoch, the batch and its "
               "sample indices."),
    "TS1385": (ERROR, "data_collate_failed",
               "The samples of a batch cannot be stacked (different shapes, dtypes or structure); the "
               "message names the field."),
    "TS1386": (ERROR, "checkpoint_corrupt",
               "A checkpoint file is not a truestar checkpoint or fails its checksum."),
    "TS1387": (ERROR, "checkpoint_version_unsupported",
               "The checkpoint was written in a checkpoint version this truestar does not read."),
    "TS1388": (ERROR, "checkpoint_mismatch",
               "The checkpoint was written by another training step (program, param or state names, "
               "dtypes or shapes differ)."),
    "TS1389": (ERROR, "resume_not_exact",
               "The resume cannot be exact: the checkpoint lacks the RNG seed or loader position the "
               "resumed run needs, or holds a different seed."),
    # native Hugging Face checkpoints (docs/MODELS.md)
    "TS1390": (ERROR, "native_family_unknown",
               "config.json names a model_type with no native Truestar implementation; import the "
               "model through truestar.ir.import_module instead."),
    "TS1391": (ERROR, "native_setting_unimplemented",
               "config.json asks for a setting the native decoder does not implement (a rope scaling "
               "other than default, linear or llama3, Qwen sliding windows, multimodal Gemma 3); "
               "refused instead of ignored."),
    "TS1392": (ERROR, "checkpoint_tensor_missing",
               "The native model needs a tensor the folder does not hold (every missing one is listed)."),
    "TS1393": (ERROR, "checkpoint_tensor_unexpected",
               "The folder holds a tensor the native model does not use; nothing is dropped silently."),
    "TS1394": (ERROR, "checkpoint_shape_mismatch",
               "A stored tensor's shape differs from what config.json implies for its place in the "
               "native model."),
    "TS1395": (ERROR, "sharded_index_inconsistent",
               "A sharded checkpoint's index and its shards disagree: a missing shard, a name in the "
               "wrong shard or in two, a tensor the index does not list. Nothing was loaded."),
    "TS1396": (INFO, "checkpoint_dtype_converted",
               "Weights were converted from the stored dtype to the requested one (counted per "
               "conversion); without a conversion the leaves are views of the memory map."),
    "TS1397": (ERROR, "pretrained_not_local",
               "from_pretrained was given something that is not a local folder; it never downloads "
               "from the Hub."),
    "TS1370": (ERROR, "lora_bad_config",
               "A LoRA configuration Truestar refuses (rank below 1, dropout outside [0, 1), a model "
               "that is not a truestar.nn layer); docs/LORA.md."),
    "TS1371": (ERROR, "lora_no_target",
               "No Linear layer of the model matches the LoRA target patterns; the message lists the "
               "Linear layer paths."),
    "TS1372": (ERROR, "lora_adapter_mismatch",
               "An adapter tree does not fit the wrapped model: every missing, extra or misshapen "
               "adapter tensor is named."),
    "TS1373": (ERROR, "lora_adapter_file_mismatch",
               "An adapter file was written for another LoRA configuration or another base model "
               "layout; what differs is named."),
    "TS1374": (ERROR, "lora_merge_mismatch",
               "The merged model's float64 output differs from the adapter model's beyond float64 "
               "rounding: the merge is wrong."),
    "TS1375": (ERROR, "qlora_decode_mismatch",
               "A quantized base weight decoded the way the training program decodes it is not bit "
               "identical to the format's exact float64 dequantization."),
    "TS1376": (ERROR, "qlora_format_unsupported",
               "A QLoRA base weight format that is not supported (int8, int4:g32/64/128, nvfp4, mxfp4 "
               "are), or a layer whose input size is not a multiple of the format's block."),
    "TS1377": (ERROR, "lora_not_bitwise",
               "Two runs of the same LoRA training from the same seed and batches gave adapters that "
               "are not bit identical."),
    # truestar export --to: deployable files verified in their own runtime (docs/EXPORT.md)
    "TS1410": (INFO, "deploy_file_verified",
               "The deployable file was written and its outputs in its own runtime agree with Truestar's "
               "float64 reference within the derived tolerance."),
    "TS1411": (ERROR, "deploy_program_unverified",
               "The model's Truestar program could not be verified against PyTorch in float64, so no "
               "deployable file was written."),
    "TS1412": (ERROR, "deploy_no_lowering",
               "An op or attribute has no exact lowering for the target format; the node and its source "
               "line are named and nothing was written."),
    "TS1413": (WARNING, "deploy_runtime_pending",
               "The target runtime cannot run on this machine (Core ML outside macOS and iOS, or a runtime "
               "that is not installed): the file was written, its runtime verification is pending."),
    "TS1414": (ERROR, "deploy_runtime_mismatch",
               "The written file's outputs in its runtime are outside the derived tolerance of Truestar's "
               "float64 reference; the file was refused."),
    "TS1415": (WARNING, "deploy_forced",
               "The file failed its check and was written only because --force was given; its evidence "
               "records it as unverified."),
    "TS1416": (WARNING, "deploy_silent_fallback",
               "Nodes of the file did not run on the requested delegate or provider (named with their op); "
               "the runtime fell back without saying so."),
    "TS1417": (INFO, "deploy_runtime_rewrote",
               "The runtime did not run the ops as written: it fused them into its own kernels, or its error "
               "is that of a lower precision than float32."),
    "TS1418": (ERROR, "deploy_conversion_mismatch",
               "The written file's own math, evaluated in float64, differs from the verified program: the "
               "lowering is wrong, before any runtime ran. The file was refused."),
    "TS1419": (ERROR, "deploy_usage",
               "A deploy export was asked for something it does not do (unknown target, wrong extension, "
               "a runtime of another format, nested or keyword inputs)."),
    "TS1300": (ERROR, "kernel_check_errors_lean",
               "`truestar kernel`: the candidate's errors against float64 are within the "
               "derived tolerance but lean one way (the TS960 lean test: mean signed error beyond "
               "what unbiased round to nearest allows, and above 1/64 of the size tolerance), on "
               "the example input."),
    "TS1301": (WARNING, "kernel_reduced_precision",
               "`truestar kernel`: the candidate's errors lean one way (TS1300's test) and its "
               "source asks for reduced precision (TF32 matmuls, float16 or bfloat16 compute): a "
               "precision verdict of its own, not a wrong number."),

    # compiled mixed precision training (compile_program mixed=True, docs/TRAINING.md)
    "TS1480": (ERROR, "mixed_precision_mismatch",
               "A compiled mixed precision program (16 bit working precision, float32 islands) has a float32 "
               "output beyond the end to end bar derived for its declared precision: 4x the larger of how far "
               "an independent evaluation rounded to the declared dtypes is from float64 and how far one ulp "
               "of every 16 bit value moves the output. A wrong cast, a 16 bit master weight or a missing "
               "unscale lands here; the result is not returned."),
    "TS1481": (INFO, "mixed_precision_near_tie",
               "A mixed precision output over its bar was accepted because every difference is explained by "
               "branch decisions on near ties (a relu input within rounding of 0): the float64 reference "
               "taking this run's decisions matches every output within its bar. Listed in mixed_bounds."),
    "TS1482": (INFO, "mixed_precision_ill_conditioned",
               "An output that the declared precision itself leaves more than 0.1 from float64 (a cancelling "
               "bias gradient behind batch norm); its bar follows that distance and it is listed as ill "
               "conditioned in mixed_bounds."),
}


def severity(code: str) -> str:
    return CODES[code][0]


def slug(code: str) -> str:
    return CODES[code][1]


@dataclass
class Issue:
    code: str
    message: str
    backend: str = ""  # empty for report level issues

    def __post_init__(self) -> None:
        if self.code not in CODES:
            raise ValueError(f"unknown truestar code {self.code}")

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "slug": slug(self.code),
            "severity": severity(self.code),
            "backend": self.backend or None,
            "message": self.message,
        }
