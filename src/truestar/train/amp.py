"""Mixed precision as program transformations.

Two separate passes:

* :func:`autocast` is the policy. It rewrites a forward program so the ops in ``low_ops`` (by
  default ``linear``, ``matmul``, ``conv2d`` and ``attention``, the ops torch's CPU autocast runs in
  half precision) read float16 or bfloat16 casts of their float operands, and the ops in
  ``fp32_ops`` read float32 casts. Every other op runs in the dtype its operands promote to, which is
  what autocast does for ops on neither list (``relu`` of a float16 tensor is float16, ``add`` of
  float16 and float32 is float32). The master weights stay float32 program inputs; the cast of a
  weight is emitted once and shared by every reader, like autocast's weight cache. The gradient of
  ``cast`` casts the cotangent back, so gradients of the master weights arrive in float32.
  Dynamic loss scaling is built by the training step (``build_train_step(amp=...)``) from the
  :class:`LossScaler` settings.

* :func:`emulate_precision` is for the reference interpreter only. The interpreter computes every
  float in float64, whatever dtype a value is declared with. This pass adds a ``cast`` to the
  declared dtype after every op whose result is declared float32, float16 or bfloat16, so the
  float64 run rounds where a backend computing in the declared dtypes rounds, and overflows where it
  overflows. :class:`EmulatedReferenceExecutor` applies it. A compiled backend computes in the
  declared dtypes and does not need it. With ``profile`` (a ``sim:*`` device profile from
  :mod:`truestar.devsim`) the rounding follows that device instead: every float result is rounded
  to the device's storage dtype under its rules (flush to zero, saturation, NaN to +inf), and the
  float inputs of its low precision ops (Neuron's matmuls in bfloat16) are rounded first. The IR
  path models storage and special values only; accumulation width, tiles and the device's own op
  formulations are modelled by the eager simulation behind ``truestar check --backends sim:*``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence

import torch

from ..ir.core import IRError, Node, Program, Value
from .edit import Editor, copy_node, retype

LOW_DTYPES = (torch.float16, torch.bfloat16)
ROUNDED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
# ops whose result is exactly an operand entry, so rounding them changes nothing
_EXACT_OPS = frozenset({"cast", "reshape", "permute", "expand", "select", "slice", "cat", "select_dynamic",
                        "relu", "where", "full", "full_like", "neg"})


@dataclass(frozen=True)
class AutocastPolicy:
    """Which ops run in low precision. The default matches torch.autocast on CPU as measured on
    torch 2.11: linear, matmul, conv2d and scaled dot product attention return float16 or bfloat16;
    softmax, log_softmax, layer_norm, gelu, tanh and relu follow their input; mse_loss and
    cross_entropy run in float32 (the training step casts model outputs to float32 before the loss).
    For CUDA's lists pass ``fp32_ops={"softmax", "log_softmax", "layer_norm", ...}``."""

    dtype: torch.dtype = torch.float16
    low_ops: FrozenSet[str] = frozenset({"linear", "matmul", "conv2d", "attention"})
    fp32_ops: FrozenSet[str] = frozenset()

    def __post_init__(self):
        if self.dtype not in LOW_DTYPES:
            raise IRError(f"autocast dtype must be float16 or bfloat16, got {self.dtype}")
        both = set(self.low_ops) & set(self.fp32_ops)
        if both:
            raise IRError(f"ops {sorted(both)} are in both low_ops and fp32_ops")

    @classmethod
    def cuda(cls, dtype: torch.dtype = torch.bfloat16) -> "AutocastPolicy":
        """torch.autocast("cuda")'s lists for the ops the IR has: linear, matmul, conv2d and attention
        in ``dtype``; softmax, log_softmax, the norms (batch norm too: its float32 statistics, as
        cuDNN keeps them), exp, log, pow, rsqrt and sum in float32
        (torch 2.11; scripts/bench_train_amp.py prints the live lists next to its rows)."""
        return cls(dtype, fp32_ops=frozenset({"softmax", "log_softmax", "layer_norm", "group_norm", "rms_norm", "batch_norm",
                                               "exp", "log", "pow", "rsqrt", "sum"}))

    def config(self) -> dict:
        return {"dtype": str(self.dtype).replace("torch.", ""), "low_ops": sorted(self.low_ops),
                "fp32_ops": sorted(self.fp32_ops)}


@dataclass(frozen=True)
class LossScaler:
    """Dynamic loss scaling with the semantics and defaults of ``torch.amp.GradScaler``.

    The loss is multiplied by ``scale`` before the backward pass and the gradients are multiplied by
    ``1 / scale`` after it. If any gradient entry is NaN or infinite the step is skipped (params and
    optimizer state keep their values), the scale is multiplied by ``backoff_factor`` and the growth
    counter resets. After ``growth_interval`` consecutive finite steps the scale is multiplied by
    ``growth_factor`` (if the result is finite in float32) and the counter resets."""

    init_scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True

    def config(self) -> dict:
        return asdict(self)


def autocast(prog: Program, policy: AutocastPolicy = AutocastPolicy(), name: Optional[str] = None) -> Program:
    """Apply the mixed precision policy to a forward program (see the module docstring)."""
    cache: Dict[tuple, Value] = {}

    def cast(e: Editor, x: Optional[Value], dtype: torch.dtype) -> Optional[Value]:
        if x is None or not x.type.dtype.is_floating_point or x.type.dtype == dtype:
            return x
        key = (id(x), dtype)
        if key not in cache:
            tag = str(dtype).replace("torch.", "")
            cache[key] = e.op("cast", x, dtype=dtype, name=f"{x.name}_{tag}",
                              source=f"truestar.train autocast {tag}")
        return cache[key]

    def per_node(e: Editor, node: Node, ins: List[Optional[Value]]) -> Value:
        if node.op in policy.low_ops:
            ins = [cast(e, x, policy.dtype) for x in ins]
        elif node.op in policy.fp32_ops:
            ins = [cast(e, x, torch.float32) if x is not None and x.type.dtype in LOW_DTYPES else x for x in ins]
        return copy_node(e, node, ins)

    return retype(prog, per_node, name or f"autocast_{prog.name}")


def _group(n: Node) -> Optional[str]:
    """The kernel a node belongs to when the autodiff pass emitted it: all nodes of one gradient rule
    application (or one lowering) share their source string. Other nodes are their own kernel."""
    s = n.source or ""
    return s if s.startswith(("grad of ", "lowered ")) else None


def _group_escapes(prog: Program) -> set:
    """Values that leave their kernel: read by a node of another kernel, or a program output."""
    out = {id(v) for v in prog.outputs}
    group_of = {id(n.output): _group(n) for n in prog.nodes}
    for n in prog.nodes:
        g = _group(n)
        for x in n.inputs:
            if x is None or id(x) not in group_of:
                continue
            gx = group_of[id(x)]
            if gx is None or gx != g:
                out.add(id(x))
    return out


def emulate_precision(prog: Program, dtypes: Sequence[torch.dtype] = ROUNDED_DTYPES,
                      kernel_groups: bool = True, profile=None, range_only: bool = False,
                      rounding: str = "nearest") -> Program:
    """Round after every op whose result is declared in one of ``dtypes`` (reference interpreter
    only, see the module docstring). The rounded value keeps the original name and identity, so the
    program's outputs and every reader see the rounded value.

    ``kernel_groups``: the nodes the autodiff pass emits for one gradient rule (``1 - y^2`` times
    the cotangent for tanh, say) stand for one backward kernel in eager PyTorch, which computes in
    float32 internally and rounds once. With ``kernel_groups`` only values that leave such a group
    are rounded. Without it every node rounds.

    ``profile``: a :class:`truestar.devsim.Profile` or its name (``"sim:ane"``). Every float result
    rounds to the profile's storage dtype (the declared dtype when it has none) under its rules,
    and the float inputs of its ``ir_low_ops`` round to its low precision dtype. Without a profile
    the rounding is IEEE round to nearest even in the declared dtype.

    ``range_only``: the inserted casts do not round; a value stays exact unless it is beyond the
    declared dtype's range, where it becomes infinite as rounding to that dtype would make it. With
    ``dtypes=LOW_DTYPES`` this is the float64 reference of a mixed precision program that overflows
    where its declared float16 values overflow (so a loss scaler skips the steps torch and the
    compiled program skip) and is exact otherwise: the end to end reference of ``mixed=True``."""
    if profile is not None:
        return _emulate_profile(prog, profile, kernel_groups)
    nodes: List[Node] = []
    names = {v.name for v in prog.inputs + prog.params} | {o.name for n in prog.nodes for o in n.outputs}
    escapes = _group_escapes(prog) if kernel_groups else None
    k = 0
    for n in prog.nodes:
        dt = n.output.type.dtype
        if (rounding == "toward_zero" and n.op == "cast" and not n.attrs.get("sr") and dt.is_floating_point
                and torch.finfo(dt).bits < 32):
            # the program's own 16 bit casts (the policy's, a gradient cast back) round toward zero too;
            # the casts inside stochastic rounding keep round to nearest, which their algorithm needs
            nodes.append(Node("cast", n.inputs, {**n.attrs, "toward_zero": True}, n.output, n.source))
            continue
        if n.extra_outputs or n.op in _EXACT_OPS or dt not in dtypes or (
                escapes is not None and id(n.output) not in escapes):
            nodes.append(n)
            continue
        raw_name = f"{n.output.name}__unrounded"
        while raw_name in names:
            k += 1
            raw_name = f"{n.output.name}__unrounded{k}"
        names.add(raw_name)
        raw = Value(raw_name, n.output.type, "op")
        nodes.append(Node(n.op, n.inputs, dict(n.attrs), raw, n.source))
        attrs = {"dtype": dt, "range_only": True} if range_only else {"dtype": dt}
        if rounding == "toward_zero":
            attrs["toward_zero"] = True
        nodes.append(Node("cast", (raw,), attrs, n.output, n.source))
    out = Program(prog.name, list(prog.inputs), list(prog.params), dict(prog.param_data), nodes,
                  list(prog.outputs), dict(prog.symbols))
    out.verify()
    return out


def _emulate_profile(prog: Program, profile, kernel_groups: bool) -> Program:
    from ..devsim import get_profile

    p = get_profile(profile) if isinstance(profile, str) else profile
    names = {v.name for v in prog.inputs + prog.params} | {o.name for n in prog.nodes for o in n.outputs}
    escapes = _group_escapes(prog) if kernel_groups else None
    counter = [0]

    def fresh(base: str) -> str:
        name = base
        while name in names:
            counter[0] += 1
            name = f"{base}{counter[0]}"
        names.add(name)
        return name

    nodes: List[Node] = []
    low_cache: Dict[int, Value] = {}
    for n in prog.nodes:
        ins = list(n.inputs)
        if n.op in p.ir_low_ops:
            for i, x in enumerate(ins):
                if x is None or not x.type.dtype.is_floating_point:
                    continue
                if id(x) not in low_cache:
                    v = Value(fresh(f"{x.name}__{p.name[4:]}_in"), x.type, "op")
                    nodes.append(Node("cast", (x,), {"dtype": x.type.dtype, "profile": p.name,
                                                     "storage": p.low_dtype}, v, n.source))
                    low_cache[id(x)] = v
                ins[i] = low_cache[id(x)]
        dt = n.output.type.dtype
        if n.extra_outputs or n.op in _EXACT_OPS or not dt.is_floating_point or (
                escapes is not None and id(n.output) not in escapes):
            nodes.append(Node(n.op, tuple(ins), dict(n.attrs), n.output, n.source) if ins != list(n.inputs) else n)
            continue
        raw = Value(fresh(f"{n.output.name}__unrounded"), n.output.type, "op")
        nodes.append(Node(n.op, tuple(ins), dict(n.attrs), raw, n.source))
        nodes.append(Node("cast", (raw,), {"dtype": dt, "profile": p.name}, n.output, n.source))
    out = Program(prog.name, list(prog.inputs), list(prog.params), dict(prog.param_data), nodes,
                  list(prog.outputs), dict(prog.symbols))
    out.verify()
    return out


class EmulatedReferenceExecutor:
    """The reference interpreter on ``emulate_precision(program)``: float64 arithmetic rounded to
    each value's declared dtype after every op."""

    name = "reference_emulated"

    def __init__(self, dtypes: Sequence[torch.dtype] = ROUNDED_DTYPES, profile=None):
        self.dtypes = tuple(dtypes)
        self.profile = profile
        if profile is not None:
            self.name = f"reference_{profile if isinstance(profile, str) else profile.name}"

    def transform(self, program: Program) -> Program:
        return emulate_precision(program, self.dtypes, profile=self.profile)

    def prepare(self, program: Program) -> Callable[[Sequence[torch.Tensor]], List[torch.Tensor]]:
        from ..ir import reference

        emulated = self.transform(program)

        def run(inputs):
            return reference.run(emulated, *inputs)
        return run


# ----------------------------------------------------------------------------- stochastic rounding

# x * SR_UP for a bfloat16 number x = m 2^k (m in [1, 2), 8 significant bits) is x plus between
# 0.53 and 1.07 of its spacing, exact in float32 (8 + 13 bits), so rounding it to bfloat16 gives the
# next bfloat16 number above x; the difference is the spacing, also exact.
SR_UP = 1.0 + 2.0 ** -8 + 2.0 ** -12


def stochastic_round(e: Editor, v: Value, key: Value, site_name: str, dtype: torch.dtype = torch.bfloat16) -> Value:
    """``v`` (float32) rounded to ``dtype`` (bfloat16) stochastically: to the neighbour above |v|
    with probability (|v| - lo) / spacing, where lo is |v| rounded toward zero, else to lo. The draw
    is the keyed counter based generator (truestar.rng, docs/RANDOM.md) at ``site_name`` under the
    int64 ``key``, so the result is a pure function of (v, key, site, index): the same bits on every
    backend and every run. Built only from ops IEEE rounds exactly: casts to bfloat16 (round to
    nearest even), float32 products with a 21 bit result, Sterbenz differences, compares and
    selects. The uniform draw has 24 bits and the dropped part 16, so the probability is exact
    (unbiased: E[result] = v). Normal numbers only: below bfloat16's smallest normal (1.2e-38) it
    rounds to nearest, and NaN passes through."""
    if dtype != torch.bfloat16:
        raise IRError(f"stochastic rounding is defined for bfloat16, not {dtype}")
    from .. import rng
    f32 = torch.float32
    if v.type.dtype != f32:
        raise IRError(f"stochastic rounding takes a float32 value, %{v.name} is {v.type.dtype}")

    def rne(x):
        return e.op("cast", e.op("cast", x, dtype=dtype, sr=True), dtype=f32)

    def above(x):     # spacing of bfloat16 numbers just above x (x >= 0, a bfloat16 number)
        # the cast rounds up on purpose: "directed" exempts its kernel from the lean test (TS960)
        up = e.op("cast", e.op("cast", e.op("mul", x, e.const(SR_UP, f32)), dtype=dtype, directed=True, sr=True), dtype=f32)
        return e.op("sub", up, x)

    a = e.op("abs", v)
    t = rne(a)
    up_t = above(t)
    pow2 = e.op("eq", t, e.op("mul", up_t, e.const(128.0, f32)))
    below = e.op("where", pow2, e.op("mul", up_t, e.const(0.5, f32)), up_t)
    lo = e.op("where", e.op("gt", t, a), e.op("sub", t, below), t)
    step = above(lo)
    shape = list(v.type.shape)
    u = e.op("random", key, dist="uniform", shape=shape, dtype=f32, site=rng.site_id(site_name),
             site_name=site_name)
    mag = e.op("where", e.op("lt", e.op("mul", u, step), e.op("sub", a, lo)), e.op("add", lo, step), lo)
    signed = e.op("where", e.op("lt", v, e.const(0.0, f32)), e.op("neg", mag), mag)
    # a zero result keeps the sign of v (-0.0 stays -0.0), as round to nearest does
    zero = e.const(0.0, f32)
    signed = e.op("where", e.op("eq", mag, zero), e.op("mul", v, zero), signed)
    return e.op("cast", signed, dtype=dtype, name=f"{v.name}_sr", sr=True)
