"""The compiled executor, behind the two execution seams.

    from truestar.compiler import CompiledExecutor

    @truestar.function(executor=CompiledExecutor("cuda", dtype="float16"))
    def layer(x: Tensor["B", 128], w: Tensor[128, 64]) -> Tensor["B", 64]: ...

    from truestar import export
    runner = export.load("model.tsx", executor="cuda")     # or "cpu", "cuda:float16", "cpu:float16"

Outputs come back on the executor's device in its float precision. There is no silent fallback to
another device: asking for "cuda" on a machine without CUDA raises, asking for "rocm" without a ROCm
build of torch and an AMD GPU raises RocmUnavailable (TS883), and asking for "cpu" without
numba raises truestar.deps.MissingDependencyError (TS022, install the `cpu-kernels` extra).
"metal" runs Metal kernels on the MPS device of a Mac (docs/METAL.md); outputs are MPS tensors, and
asking for it elsewhere raises TS840.

"rocm" is the AMD spelling of the GPU device: torch exposes a ROCm GPU as "cuda", so the program
runs on torch "cuda" tensors, compiled for the attached AMD GPU (docs/ROCM.md). On a ROCm build
"cuda" means the same GPU.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..frontend.function import Executor
from ..ir.core import Program
from .lower import Options
from .runtime import CompiledProgram

DEVICES = ("cuda", "rocm", "cpu", "metal")
DTYPES = ("float32", "float16", "bfloat16")


def _options(device: str, dtype: str, allow_tf32: Optional[bool], hidden: Dict[str, Any],
             deterministic: Optional[bool] = None, memory_reuse: bool = True, sentinels=None,
             sentinel_read: str = "call", library=(), accept_ill_conditioned: bool = False,
             portable_tensor_cores: bool = False, mixed: bool = False) -> Options:
    device = str(device)
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; the compiled executor runs on 'cuda', 'rocm', 'cpu' or 'metal'")
    platform = ""
    if device == "rocm":
        from ..kernel import rocm
        rocm.require_hip("the compiled executor on 'rocm'")
        device, platform = "cuda", "rocm"     # torch spells an AMD GPU "cuda"
    elif device == "cuda" and getattr(torch.version, "hip", None):
        platform = "rocm"
    dtype = str(dtype).replace("torch.", "")
    dtype = {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "half": "float16"}.get(dtype, dtype)
    if dtype not in DTYPES:
        raise ValueError(f"unsupported precision {dtype!r}; use one of {', '.join(DTYPES)} "
                         f"(float64 is the reference interpreter)")
    if device == "cuda" and not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        raise RuntimeError("the compiled executor was asked for device 'cuda' but CUDA is not available; "
                           "it never falls back to the CPU silently. Use device='cpu'.")
    if device == "metal":
        from ..kernel import metal_backend
        metal_backend.require("the compiled executor on 'metal'")
    if device == "cpu":
        from ..deps import require
        require("numba", "The compiled executor on 'cpu'",
                fallback="run through the float64 reference interpreter instead (the default "
                         "executor, 'reference'): correct and slow")
    if mixed and dtype == "float32":
        raise ValueError("mixed=True keeps float32 values in float32 islands of a 16 bit program; compile it "
                         "with dtype='bfloat16' or 'float16' (in float32 every value is already float32 or wider)")
    bad = set(hidden) - {"_break"}
    if bad:
        raise TypeError(f"unknown options: {', '.join(sorted(bad))}")
    library = tuple(library or ())
    from .lower import LIBRARY_OPS
    unknown = [op for op in library if op not in LIBRARY_OPS]
    if unknown:
        raise ValueError(f"library= names ops without a torch library kernel: {', '.join(unknown)}; "
                         f"choose from {', '.join(LIBRARY_OPS)}")
    from ..kernel.checks import parse
    if sentinels is None:
        import os
        sentinels = os.environ.get("TRUESTAR_SENTINELS") or None
    if sentinel_read not in ("call", "deferred"):
        raise ValueError(f"sentinel_read must be 'call' or 'deferred', not {sentinel_read!r}")
    if portable_tensor_cores:
        from ..kernel import tcexact
        if device != "cuda" or platform:
            raise tcexact.PortableTensorCoreError(
                "TS1330" if device != "cuda" else "TS1331",
                f"portable tensor cores run on NVIDIA CUDA GPUs (sm_80 and newer) only, not on "
                f"{platform or device!r}")
        if library:
            raise tcexact.PortableTensorCoreError(
                "TS1333", f"library={list(library)} runs those ops through vendor kernels whose "
                          f"tensor core rounding differs by GPU; drop library= with portable tensor cores")
        tcexact.check_device()
        deterministic = True   # a row's bits must not depend on the batch size either
    return Options(device, dtype, None if allow_tf32 is None else bool(allow_tf32),
                   None if deterministic is None else bool(deterministic), dict(hidden), bool(memory_reuse),
                   library=library, checks=parse(sentinels), sentinel_read=sentinel_read, platform=platform,
                   accept_ill_conditioned=bool(accept_ill_conditioned),
                   portable_tensor_cores=bool(portable_tensor_cores), mixed=bool(mixed))


def compile_program(program: Program, device: str = "cuda", dtype: str = "float32", *,
                    verify: Optional[str] = None, allow_tf32: Optional[bool] = None, allow_fallback: bool = True,
                    deterministic: Optional[bool] = None, memory_budget=None,
                    memory_dims: Optional[Dict[str, int]] = None, recompute: bool = False,
                    memory_reuse: bool = True, sentinels=None, sentinel_read: str = "call",
                    library: Sequence[str] = (), accept_ill_conditioned: bool = False, graphs: str = "auto",
                    portable_tensor_cores: bool = False, end_to_end: str = "auto", mixed: bool = False,
                    **hidden) -> CompiledProgram:
    """Compile an IR program. The result is called with a list of input tensors and returns a list of
    outputs; ``.report`` is the compile report.

    ``allow_tf32``: None (default) follows PyTorch's float32 matmul precision
    (``torch.backends.cuda.matmul.allow_tf32``, ``fp32_precision`` or
    ``torch.set_float32_matmul_precision``) at every call, as eager PyTorch does; True or False pins it.

    ``deterministic``: None (default) follows ``torch.are_deterministic_algorithms_enabled()`` at
    every call; True makes contraction results independent of the batch size (no split K). Runs are
    deterministic in both modes.

    Memory (docs/MEMORY.md): ``memory_budget`` (bytes or "6 GiB") refuses, before any launch, a program
    whose planned peak exceeds it (TS420), planned at ``memory_dims`` or the largest declared symbol
    values; ``recompute=True`` first recomputes values instead of storing them until the plan fits,
    verified bitwise (TS421); ``recompute="mincut"`` chooses them by a minimum cut, ``"offload"`` copies
    stored values to pinned host memory and back on a fixed schedule (TS1112), ``"mincut+offload"`` does
    both (compiler/remat.py). Each TF32 or deterministic variant is planned (and recomputed) on
    its own. ``memory_reuse=False`` holds every value to the end of the call (the executor before the
    planner, kept for measurement).

    ``library`` names ops (``conv2d``, ``max_pool2d``, ``avg_pool2d``, ``adaptive_avg_pool2d``) to run
    through torch (cuDNN) instead of Truestar kernels; they are declared in the report and checked
    like kernels.

    ``sentinels``: compiled in runtime checks (docs/COMPILER.md, Sentinels): None follows
    TRUESTAR_SENTINELS (default off), True is nan and div, "all" or a list of nan, inf, overflow,
    div. Every compiled variant carries them; their flag words are pinned host memory, outside the
    device memory plan. ``sentinel_read``: "call" or "deferred".

    ``accept_ill_conditioned`` (float32 only, default False): the end to end check refuses any output
    over its float64 bar, and its error names how far an independent float32 evaluation of the same
    program (the reference interpreter in float32) is from float64. With True, an output within 4x of
    that distance is accepted, listed in ``conditioning`` and reported by a TS782 warning. Use it for
    ill conditioned steps such as AdamW's first update (docs/TRAINING.md).

    ``end_to_end``: how the first call's end to end check judges each float output against float64.
    "bar": within a fixed bar (BASE_TOL times 4 sqrt(kernels)), or over it within the bound derived
    from the program itself (truestar.tolerance: error model runs propagated op by op,
    docs/TOLERANCE.md); the derivation runs only when an output is over the bar. "derived": every float
    output by its derived bound alone, also one within the bar: catches errors smaller than the bar
    (a wrong weight decay sign) at the cost of the derivation (about ten float64 runs of the program)
    on the first call. "auto" (default): "derived" for a call that writes the caller's state in place
    (``update=`` or ``donate=True``, as a training loop calls its step: an error there carries into
    every later step), "bar" otherwise.

    ``mixed`` (bfloat16 or float16 only): every value keeps its declared dtype. Values declared in
    the working precision (the low precision casts and contractions of an autocast step) are computed
    in it with float32 accumulation; values declared float32 or float64 (master weights, gradients
    after their cast back, optimizer state, the loss, the loss scale) are float32 islands. Without it
    the program's dtype is the one holding the most elements and every value is computed in the
    working precision (docs/TRAINING.md, Mixed precision, compiled).

    ``graphs``: "auto" (default) replays a CUDA program from a CUDA graph from its second ordinary
    call at the same input shapes, the graph recorded after a verified call and checked bitwise
    against it, so the outputs keep their bits (docs/COMPILER.md, Automatic graphs); "off" (or
    TRUESTAR_GRAPHS=off) keeps every call an ordinary call.

    ``portable_tensor_cores`` (CUDA, sm_80 and newer): float16, bfloat16 and TF32 dot products run
    as exact tensor core blocks with all rounding in a fixed order, so the result bits are the same
    on Ampere, Ada, Hopper and Blackwell (docs/PORTABLE_TC.md). Implies deterministic. Refused with
    TS1330 to TS1332 where it cannot apply."""
    if graphs not in ("auto", "off"):
        raise ValueError(f"graphs must be 'auto' or 'off', not {graphs!r}")
    if end_to_end not in ("auto", "bar", "derived"):
        raise ValueError(f"end_to_end must be 'auto', 'bar' or 'derived', not {end_to_end!r}")
    cp = CompiledProgram(program, _options(device, dtype, allow_tf32, hidden, deterministic, memory_reuse,
                                           sentinels, sentinel_read, library, accept_ill_conditioned,
                                           portable_tensor_cores, mixed),
                         verify, allow_fallback, memory_budget, memory_dims, recompute)
    if graphs == "off":
        cp.auto_graphs = False
    cp.end_to_end = end_to_end
    return cp


class CompiledExecutor(Executor):
    """Runs programs through Truestar kernels on ``device`` in ``dtype``.

    ``verify``: None follows TRUESTAR_VERIFY (default: check the first run), "all" checks every run,
    "off" is for debugging only. ``allow_fallback=False`` refuses programs that need the fallback path.
    ``allow_tf32``: None (default) follows PyTorch's float32 matmul precision at every call; True or
    False pins it; ``deterministic`` likewise follows ``torch.are_deterministic_algorithms_enabled()``
    (see compile_program).
    ``sentinels``: compiled in runtime checks, raising SentinelError with the op, the line and the first
    offending element (True: nan and div; "all"; or a list of nan, inf, overflow, div). Off by default.
    """

    name = "compiled"

    def __init__(self, device: str = "cuda", dtype: str = "float32", *, verify: Optional[str] = None,
                 allow_tf32: Optional[bool] = None, allow_fallback: bool = True,
                 deterministic: Optional[bool] = None, memory_budget=None,
                 memory_dims: Optional[Dict[str, int]] = None, recompute: bool = False, memory_reuse: bool = True,
                 sentinels=None, sentinel_read: str = "call", library: Sequence[str] = (),
                 accept_ill_conditioned: bool = False, **hidden):
        self.options = _options(device, dtype, allow_tf32, hidden, deterministic, memory_reuse, sentinels,
                                sentinel_read, library, accept_ill_conditioned)
        self.memory = dict(memory_budget=memory_budget, memory_dims=memory_dims, recompute=recompute)
        self.device = self.options.platform or self.options.device
        self.dtype = self.options.dtype
        self.verify = verify
        self.allow_fallback = allow_fallback
        self.compile_count = 0
        self.programs: List[CompiledProgram] = []
        self.name = f"compiled:{self.device}:{self.dtype}"

    def prepare(self, program: Program) -> CompiledProgram:
        cp = CompiledProgram(program, self.options, self.verify, self.allow_fallback, **self.memory)
        self.compile_count += 1
        self.programs.append(cp)
        return cp

    @property
    def last(self) -> CompiledProgram:
        return self.programs[-1]


# ----------------------------------------------------------------------------- export runner seam

_EXPORT_CACHE: Dict[Tuple[int, str, str], Tuple[Program, CompiledProgram]] = {}


def export_executor(device: str, dtype: str = "float32") -> Callable[[Program, Sequence[torch.Tensor]], List]:
    """An executor function for ``truestar.export.runner.register_executor``: compiles each program
    once per device and precision and reuses it for every later call."""
    def run(program: Program, inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        key = (id(program), device, dtype)
        hit = _EXPORT_CACHE.get(key)
        if hit is None or hit[0] is not program:
            hit = (program, compile_program(program, device, dtype))
            _EXPORT_CACHE[key] = hit
        return hit[1](list(inputs))

    run.compiled = lambda program: _EXPORT_CACHE[(id(program), device, dtype)][1]
    return run


def register_export_executors() -> List[str]:
    """Register "cuda", "cpu", "metal" and their ":float16" forms with the export runner. Idempotent."""
    from ..export import runner
    names = []
    for device in DEVICES:
        for dtype in ("float32", "float16"):
            name = device if dtype == "float32" else f"{device}:{dtype}"
            if name not in runner.EXECUTORS:
                runner.register_executor(name, export_executor(device, dtype))
            names.append(name)
    return names
