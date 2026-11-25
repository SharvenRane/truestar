"""Phone NPUs through Qualcomm AI Hub: compile a model for a real hosted device, run it there,
and compare the outputs with Truestar's float64 CPU reference.

Every NPU run is paired with a CPU run of the same compiled model on the same phone. If the CPU
control matches but the NPU does not, the divergence belongs to the NPU path; if both diverge, the
conversion itself changed the model.

Needs `pip install qai-hub` and an AI Hub API token (qai-hub configure --api_token ...).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..check import _decisions
from ..compare import DIVERGENT, OK, SUSPECT, Diff, diff_tensors, threshold_for, worst
from ..capture import flatten_tensors
from .qai_hub_quant import _natural

# Hexagon NPUs run float models in float16; compare at float16 precision.
NPU_DTYPE = torch.float16
# float16 has about 3 decimal digits: absolute differences below this on unit scale values are noise.
FP16_RESOLUTION = 2.0 ** -10


@dataclass
class RemoteRun:
    compute_unit: str
    status: str = OK
    output: Optional[Diff] = None
    decisions: int = 0
    flips: int = 0
    tie_flips: int = 0
    compile_job: str = ""
    inference_job: str = ""
    error: str = ""
    inconclusive: str = ""


@dataclass
class QaiHubReport:
    device: str
    runtime: str
    runs: List[RemoteRun] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return worst(*[r.status for r in self.runs]) if self.runs else OK

    def to_text(self) -> str:
        lines = [f"truestar check on {self.device} via Qualcomm AI Hub ({self.runtime}), reference CPU float64"]
        for n in self.notes:
            lines.append(f"  note: {n}")
        for r in self.runs:
            lines.append(f"[{r.status.upper()}] {r.compute_unit}")
            if r.error:
                lines.append(f"  failed: {r.error}")
            if r.inconclusive:
                lines.append(f"  inconclusive: {r.inconclusive}")
            if r.output is not None:
                lines.append(f"  final output: relative error {r.output.rel_err:.1e}"
                             + ("  (NaN/Inf mismatch)" if r.output.nonfinite_mismatch else ""))
            if r.decisions:
                lines.append(f"  decisions: {r.flips}/{r.decisions} predictions changed"
                             + (f" ({r.tie_flips} near ties)" if r.flips else ""))
            if r.compile_job:
                lines.append(f"  jobs: compile {r.compile_job}, inference {r.inference_job}")
        lines.append(f"overall: {self.status.upper()}")
        return "\n".join(lines)


def check_qai_hub(
    model: nn.Module,
    inputs: Sequence[torch.Tensor],
    device: str = "Samsung Galaxy S24 (Family)",
    runtime: str = "tflite",
    compute_units: Sequence[str] = ("cpu", "npu"),
    timeout_s: int = 1800,
) -> QaiHubReport:
    import qai_hub as hub

    args = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    base = copy.deepcopy(model).cpu().float().eval()
    with torch.no_grad():
        ref = flatten_tensors(copy.deepcopy(base).double()(*[a.double() if a.is_floating_point() else a
                                                              for a in args]))
    traced = torch.jit.trace(base, args)
    names = [f"input_{i}" for i in range(len(args))]
    specs = {n: (tuple(a.shape), "float32" if a.is_floating_point() else "int32") for n, a in zip(names, args)}
    feed = {n: [a.numpy().astype(np.float32 if a.is_floating_point() else np.int32)]
            for n, a in zip(names, args)}
    rep = QaiHubReport(device=device, runtime=runtime)
    hub_device = hub.Device(device)
    for unit in compute_units:
        run = RemoteRun(compute_unit=unit)
        try:
            cj = hub.submit_compile_job(
                model=traced, device=hub_device, input_specs=specs,
                options=f"--target_runtime {runtime} --compute_unit {unit}")
            run.compile_job = cj.job_id
            target = cj.get_target_model()
            if target is None:
                raise RuntimeError(f"compile failed: {cj.get_status().message}")
            ij = hub.submit_inference_job(model=target, device=hub_device, inputs=feed,
                                          options=f"--compute_unit {unit}")
            run.inference_job = ij.job_id
            out = ij.download_output_data()
            if out is None:
                raise RuntimeError(f"inference failed: {ij.get_status().message}")
            got = [torch.from_numpy(np.asarray(out[k][0])) for k in sorted(out, key=_natural)]
            run.output = _best_match(got, ref[0])
            thr = threshold_for(NPU_DTYPE if unit == "npu" else torch.float32, False, None)
            run.status = run.output.status(thr)
            ref_scale = float(ref[0].abs().max()) if ref[0].numel() else 0.0
            if (unit == "npu" and run.status != OK and not run.output.nonfinite_mismatch
                    and not run.output.shape_mismatch and run.output.max_abs <= FP16_RESOLUTION
                    and ref_scale <= FP16_RESOLUTION * 10):
                run.status = OK
                run.inconclusive = (
                    f"reference outputs are tiny (max {ref_scale:.1e}) and every difference is below "
                    f"float16 resolution ({FP16_RESOLUTION:.0e}), so this run cannot show a bug either "
                    "way. Use trained weights or inputs that give outputs of normal size.")
            if got:
                o = _reshape_like(got, ref[0])
                if o is not None:
                    run.decisions, run.flips, run.tie_flips = _decisions(o, ref[0], thr)
                    if run.flips - run.tie_flips > 0:
                        run.status = worst(run.status, SUSPECT)
        except Exception as exc:  # noqa: BLE001
            run.status = DIVERGENT
            run.error = f"{type(exc).__name__}: {exc}"
        rep.runs.append(run)
    by = {r.compute_unit: r for r in rep.runs}
    if "cpu" in by and "npu" in by and by["cpu"].status == OK and by["npu"].status != OK:
        rep.notes.append("The CPU control on the same phone matches, so the divergence is in the NPU path.")
    if "cpu" in by and by["cpu"].status != OK:
        rep.notes.append("The CPU control already diverges: the model changed during conversion.")
    return rep


def _reshape_like(got: List[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    for g in got:
        if g.numel() == ref.numel():
            return g.reshape(ref.shape)
    return None


def _best_match(got: List[torch.Tensor], ref: torch.Tensor) -> Diff:
    """Runtimes may rename or reorder outputs; compare the first output with the reference's size."""
    g = _reshape_like(got, ref)
    if g is None:
        return Diff(float("inf"), float("inf"), False, True,
                    note=f"no output with {ref.numel()} elements; got {[tuple(x.shape) for x in got]}")
    return diff_tensors(g, ref)
