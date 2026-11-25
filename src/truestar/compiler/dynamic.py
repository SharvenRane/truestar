"""Dynamic shapes: one compile serves every batch size and sequence length in declared ranges,
kernels can be checked once per shape bucket, and each result is bitwise the static compile's.

Author: Sharven Rane. 2026-11-24. docs/DYNAMIC_SHAPES.md.

The compiler already keeps symbolic dims symbolic (docs/COMPILER.md, Control flow and symbolic
dims): one Truestar compile, kernels that bind their sizes at call time, tails masked inside the
kernel. What this module adds:

* a declaration API: dims with ranges at import (``axes`` and ``dims``) or a torch.export
  ``dynamic_shapes`` spec whose ranges are read back from the export;
* shape buckets (powers of two or a user list) as an opt in unit of verification: the kernel layer
  checks each kernel binary once per bucket instead of once per shape (a shape in the bucket that
  runs another Triton binary is checked on its own, kernel/compiler.py ``verify_group``);
* the bitwise promise: the dynamic program's result at a shape is bitwise the static compile's at
  that shape. It holds by construction only when no launch choice that changes reduction order
  depends on anything but the exact sizes, which the compile pins (deterministic mode, no library
  ops); where it cannot hold the compile refuses with a code (TS1421, TS1423). ``prove`` compiles
  the static program at given shapes and compares bits (TS1426 on a difference).

Codes TS1420 to TS1429 (codes.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..ir import reference as iref
from ..ir.core import IRError, Program, symbol
from ..kernel import compiler as kcomp
from .executor import compile_program


class DynamicShapeError(IRError):
    """A refusal of the dynamic shape layer; ``code`` is one of TS1420 to TS1429."""

    def __init__(self, code: str, msg: str):
        super().__init__(f"[{code}] {msg}")
        self.code = code


BucketSpec = Union[str, Sequence[int]]


def _pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


@dataclass
class Buckets:
    """Bucket policy per symbol: ``"pow2"`` (the bucket of n is the next power of two, which is
    also the block size the GPU kernels choose for a reduced axis, so a bucket never spans two
    reduction layouts) or an increasing list of upper bounds that covers the symbol's range."""
    rule: Dict[str, BucketSpec]
    ranges: Dict[str, Tuple[int, Optional[int]]]

    @classmethod
    def make(cls, spec: Union[BucketSpec, Dict[str, BucketSpec]], ranges) -> "Buckets":
        if isinstance(spec, str) or not isinstance(spec, dict):
            spec = {s: spec for s in ranges}
        unknown = sorted(set(spec) - set(ranges))
        if unknown:
            raise DynamicShapeError("TS1424", f"buckets name {unknown}, which are not symbolic dims of the "
                                              f"program (declared: {sorted(ranges)})")
        rule = {}
        for s in ranges:
            r = spec.get(s, "pow2")
            if isinstance(r, str):
                if r != "pow2":
                    raise DynamicShapeError("TS1422", f"bucket rule {r!r} for {s}: use 'pow2' or a list of "
                                                      f"upper bounds")
                rule[s] = "pow2"
                continue
            ub = [int(v) for v in r]
            lo, hi = ranges[s]
            if not ub or any(b <= a for a, b in zip(ub, ub[1:])) or ub[0] < lo:
                raise DynamicShapeError("TS1422", f"buckets {ub} for {s} must be increasing upper bounds "
                                                  f"starting at or above the minimum {lo}")
            if hi is None or ub[-1] < hi:
                raise DynamicShapeError("TS1422", f"buckets {ub} for {s} do not cover its declared range "
                                                  f"[{lo}, {'unbounded' if hi is None else hi}]: a shape "
                                                  f"above {ub[-1]} would have no bucket")
            rule[s] = tuple(ub)
        return cls(rule, dict(ranges))

    def of(self, name: str, n: int) -> int:
        r = self.rule[name]
        if r == "pow2":
            return _pow2(n)
        for b in r:
            if n <= b:
                return b
        raise AssertionError("unreachable: the range check runs first")


@dataclass
class DynamicStats:
    calls: int = 0
    compiles: int = 0                # Truestar compiles (one per DynamicProgram)
    compile_seconds: float = 0.0     # Truestar's own compile (lowering, fusion, code generation)
    first_call_seconds: Dict[Tuple, float] = field(default_factory=dict)   # per bucket, Triton and checks
    shapes: Dict[Tuple, Tuple] = field(default_factory=dict)              # shape key -> bucket key
    kernel_checks_skipped: int = 0   # kernel checks the bucket made unnecessary (same binary)

    @property
    def buckets(self) -> List[Tuple]:
        return sorted(set(self.shapes.values()))


def _bit_differences(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose bits differ (a NaN equals the same NaN; +0 and -0 differ)."""
    a, b = a.detach().contiguous(), b.detach().to(a.device).contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(int(a.numel()), 1)
    if a.dtype.is_floating_point:
        iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        a, b = a.view(iv), b.view(iv)
    return int((a != b).sum().item())


def _ranges(prog: Program) -> Dict[str, Tuple[int, Optional[int]]]:
    return {str(s): (int(lo), None if hi is None else int(hi)) for s, (lo, hi) in prog.symbols.items()}


def torch_dynamic_shapes(example: Sequence[torch.Tensor], axes: Sequence[Optional[Dict[int, str]]],
                         dims: Dict[str, Tuple[int, Optional[int]]]):
    """torch.export's ``dynamic_shapes`` from ``axes`` (per input, {axis: dim name}) and ``dims``
    ({name: (min, max)}, max None for unbounded). One Dim object per name, shared by every input
    axis that names it."""
    if len(axes) != len(example):
        raise DynamicShapeError("TS1424", f"axes has {len(axes)} entries for {len(example)} inputs")
    named = {}
    for nm, (lo, hi) in dims.items():
        named[nm] = torch.export.Dim(nm, min=int(lo), max=int(hi)) if hi is not None else \
            torch.export.Dim(nm, min=int(lo))
    out = []
    for i, a in enumerate(axes):
        if not a:
            out.append(None)
            continue
        miss = sorted(set(a.values()) - set(named))
        if miss:
            raise DynamicShapeError("TS1424", f"input {i} names dims {miss} that dims does not declare")
        out.append({int(ax): named[nm] for ax, nm in a.items()})
    return tuple(out)


def compile_dynamic(src: Union[Program, torch.nn.Module], example: Optional[Sequence[torch.Tensor]] = None, *,
                    axes: Optional[Sequence[Optional[Dict[int, str]]]] = None,
                    dims: Optional[Dict[str, Tuple[int, Optional[int]]]] = None,
                    dynamic_shapes: Any = None, device: str = "cuda", dtype: str = "float32",
                    buckets: Union[BucketSpec, Dict[str, BucketSpec]] = "pow2", verify: str = "shape",
                    bitwise_static: bool = True, deterministic: Optional[bool] = None,
                    library: Sequence[str] = (), **kw) -> "DynamicProgram":
    """Compile once for every value of the declared symbolic dims.

    ``src`` is an IR program (its symbols and ranges are used as declared) or a torch module with
    ``example`` inputs and either ``axes`` plus ``dims`` or a torch.export ``dynamic_shapes`` spec.
    ``verify``: ``"shape"`` (default, the compiler's rule) checks each kernel once per shape,
    ``"bucket"`` each kernel binary once per bucket (opt in: on an L4 it skipped 78 of the encoder
    sweep's kernel checks but saved no measurable time, docs/DYNAMIC_SHAPES.md), ``"off"`` never. ``bitwise_static`` (default) pins every launch
    choice that could make a result differ from the static compile of the same shape and refuses
    options that cannot be pinned. Other keywords go to ``compile_program``."""
    if verify not in ("bucket", "shape", "off"):
        raise ValueError(f"verify must be 'bucket', 'shape' or 'off', not {verify!r}")
    if bitwise_static:
        if deterministic is False:
            raise DynamicShapeError("TS1421", "bitwise_static needs deterministic launches: without them "
                                              "a contraction splits K by the row count and reductions take "
                                              "tuned (timed) block sizes, so a shape's bits would depend on "
                                              "the GPU, the tuning cache and the launch history; pass "
                                              "deterministic=None or True, or bitwise_static=False")
        if library:
            raise DynamicShapeError("TS1423", f"bitwise_static cannot hold for library ops {list(library)}: "
                                              f"cuDNN chooses its algorithm by shape and by benchmark, "
                                              f"outside Truestar's control")
        deterministic = True
    module = None
    if isinstance(src, Program):
        prog = src
    elif isinstance(src, torch.nn.Module):
        if example is None:
            raise DynamicShapeError("TS1424", "a torch module needs example inputs")
        from ..ir.importer import import_module
        if dynamic_shapes is None:
            if axes is None or dims is None:
                raise DynamicShapeError("TS1424", "declare dims with axes and dims, or pass a torch.export "
                                                  "dynamic_shapes spec")
            dynamic_shapes = torch_dynamic_shapes(example, axes, dims)
        module = src
        prog = import_module(src.eval(), tuple(example), dynamic_shapes=dynamic_shapes)
    else:
        raise TypeError(f"compile_dynamic takes a Program or a torch.nn.Module, not {type(src).__name__}")
    ranges = _ranges(prog)
    if not ranges:
        raise DynamicShapeError("TS1424", "the program has no symbolic dims: every shape is static, use "
                                          "compile_program")
    if dims:
        for nm, (lo, hi) in dims.items():
            if nm in ranges and ranges[nm] != (int(lo), None if hi is None else int(hi)):
                # torch.export narrows or widens a range it can prove (a Dim min of 1 becomes 2 when
                # the model specializes size 1): the declared range must be the one that holds
                raise DynamicShapeError("TS1424", f"dim {nm} was declared {lo}..{hi} but torch.export "
                                                  f"derived {ranges[nm][0]}..{ranges[nm][1]}; declare that "
                                                  f"range")
    bk = Buckets.make(buckets, ranges)
    t0 = time.perf_counter()
    cp = compile_program(prog, device, dtype, deterministic=deterministic, library=library,
                         verify="off" if verify == "off" else None, **kw)
    if bitwise_static:
        _refuse_order_changes(cp, ranges)
    dp = DynamicProgram(cp, prog, bk, verify, module, dict(device=device, dtype=dtype, deterministic=deterministic,
                                                           library=library, **kw))
    dp.stats.compiles = 1
    dp.stats.compile_seconds = time.perf_counter() - t0
    return dp


def _least(kernel_sym: str, ranges) -> int:
    """Smallest value of the dim expression a kernel symbol stands for (lower.py names one kernel
    symbol per distinct expression), with every program symbol at its declared minimum (dims are
    sums and products of positive sizes, so that is the least value)."""
    import sympy
    from . import lower
    expr = sympy.sympify(lower._SIZE_TAKEN.get(kernel_sym, kernel_sym))
    return int(expr.subs({x: ranges.get(str(x), (1, None))[0] for x in expr.free_symbols}))


def _refuse_order_changes(cp, ranges) -> None:
    """TS1425 for the one loop shape choice that reads whether a size is symbolic and changes the
    summation order: a convolution's tap loop (kernel/triton_conv.py tap_mode) is taken for a
    symbolic channel count but only for a static count of at least 16. Every other launch choice
    of a deterministic program follows the exact sizes, which the static compile sees too."""
    from ..kernel import triton_conv
    for _, st in cp.plan.all_steps():
        art = getattr(getattr(st, "ck", None), "art", None)
        p = getattr(art, "plan", None)
        if p is None or getattr(p, "kind", None) != "conv":
            continue
        ch = p.chain[0]
        if isinstance(ch.size, int) or not triton_conv.tap_mode(p):
            continue
        lo = _least(ch.size.name, ranges)
        if lo < 16:
            raise DynamicShapeError("TS1425", f"kernel {st.name}: the convolution sums over the symbolic channel "
                                              f"dim {ch.size} (minimum {lo}); a static compile below 16 channels "
                                              f"sums in another order. Declare a minimum of 16 or keep the "
                                              f"channels static")


class DynamicProgram:
    """A compiled program over symbolic dims, run per call at the bucket of its shapes."""

    def __init__(self, cp, prog: Program, buckets: Buckets, verify: str, module, opts: Dict):
        self.cp, self.program, self.buckets, self.verify = cp, prog, buckets, verify
        self.module, self.opts = module, opts
        self.stats = DynamicStats()
        self._sym = [str(s) for s in prog.symbols]

    @property
    def report(self):
        return self.cp.report

    def bind(self, inputs: Sequence[torch.Tensor]) -> Dict[str, int]:
        """Symbol values of these inputs; TS1420 when one is outside its declared range."""
        try:
            b = iref.bind_symbols(self.program, list(inputs))
        except IRError as e:
            raise DynamicShapeError("TS1420", f"input shapes {[tuple(t.shape) for t in inputs]} do not fit "
                                              f"the program: {e}") from e
        vals = {s: int(b[symbol(s)]) for s in self._sym}
        for s, v in vals.items():
            lo, hi = self.buckets.ranges[s]
            if v < lo or (hi is not None and v > hi):
                raise DynamicShapeError("TS1420", f"{s} = {v} is outside its declared range "
                                                  f"[{lo}, {'unbounded' if hi is None else hi}]; it is "
                                                  f"refused, never recompiled")
        return vals

    def bucket_of(self, vals: Dict[str, int]) -> Tuple:
        return tuple((s, self.buckets.of(s, vals[s])) for s in self._sym)

    def __call__(self, inputs: Sequence[torch.Tensor], **kw) -> List[torch.Tensor]:
        shape_key = tuple(tuple(t.shape) for t in inputs)
        bkey = self.stats.shapes.get(shape_key)
        if bkey is None:
            bkey = self.bucket_of(self.bind(inputs))
            self.stats.shapes[shape_key] = bkey
        first = bkey not in self.stats.first_call_seconds
        self.stats.calls += 1
        token = kcomp.verify_group.set((id(self), bkey)) if self.verify == "bucket" else None
        skipped0 = kcomp.group_skips[0]
        t0 = time.perf_counter()
        try:
            out = self.cp(inputs, **kw)
        finally:
            if token is not None:
                kcomp.verify_group.reset(token)
        if first:
            if out and out[0].is_cuda:
                torch.cuda.synchronize(out[0].device)
            self.stats.first_call_seconds[bkey] = time.perf_counter() - t0
        self.stats.kernel_checks_skipped += kcomp.group_skips[0] - skipped0
        return out

    def prove(self, shapes: Sequence[Sequence[torch.Tensor]]) -> List[Dict]:
        """Compile the static program at each given input set and compare bits with this program's
        result; raises TS1426 at the first difference. Needs the torch module this program was
        imported from (TS1427 otherwise)."""
        if self.module is None:
            raise DynamicShapeError("TS1427", "prove needs the torch module (a program given as IR has no "
                                              "static source to compile per shape)")
        from ..ir.importer import import_module
        rows = []
        for inputs in shapes:
            got = self(list(inputs))
            opts = {k: v for k, v in self.opts.items() if k not in ("device", "dtype")}
            st = compile_program(import_module(self.module.eval(), tuple(inputs)), self.opts["device"],
                                 self.opts["dtype"], **opts)
            want = st(list(inputs))
            diffs = [_bit_differences(a, b) for a, b in zip(got, want)]
            row = {"shapes": [tuple(t.shape) for t in inputs], "differing": sum(diffs),
                   "elements": sum(int(a.numel()) for a in got)}
            rows.append(row)
            if row["differing"]:
                raise DynamicShapeError("TS1426", f"at input shapes {row['shapes']} the dynamic program differs "
                                                  f"from the static compile in {row['differing']} of "
                                                  f"{row['elements']} elements")
        return rows
