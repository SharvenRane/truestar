"""Units of a precision plan: the modules that get a precision, and how each one runs in it.

A unit is a leaf module (no children) or a module whose forward reads its children's weights
directly instead of calling them (nn.MultiheadAttention reads `out_proj.weight`): such a child is
never called, so it cannot be planned alone and its parent becomes one unit. This is found by
running the model, not from a list of module types.

Each unit runs in one precision:

| precision | weights | activations in and out | compute |
|---|---|---|---|
| fp32 | float32 | float32 | strict float32 (TF32 off for matmul and convolution) |
| tf32 | float32 | float32 | TF32 tensor cores for this unit's matmuls and convolutions |
| bf16, fp16 | that dtype | that dtype | that dtype, float32 accumulation where the library does it |
| int8w, int4w | INT8 per channel, INT4 per group | float16 | Truestar weight only kernels (truestar.quant) |
| fp8 | FP8 E4M3 per channel | float16, quantized per token to E4M3 inside | Truestar FP8 kernel |

A unit casts its floating inputs to its activation dtype (nothing when they already are) and
returns its output in that dtype. Code between units (residual adds, attention written in a
parent's forward) runs in whatever dtype reaches it, with PyTorch's promotion: float16 plus
float32 is float32. Where that code mixes dtypes in an op that does not promote (a matmul of a
float16 query with a float32 key), the units under that module are tied to one activation dtype
(a tie group), found by probing runs.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

PRECISIONS = ("fp32", "tf32", "bf16", "fp16", "int8w", "fp8", "int4w")
STORAGE = {"fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16,
           "fp16": torch.float16, "int8w": torch.float16, "fp8": torch.float16,
           "int4w": torch.float16}
STORAGES = (torch.float32, torch.bfloat16, torch.float16)
PLAIN = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
# most accurate first; used to break timing ties in favour of the more accurate candidate
ACCURACY_RANK = {"fp32": 0, "tf32": 1, "fp16": 2, "bf16": 3, "int8w": 4, "fp8": 5, "int4w": 6}
QUANT = ("int8w", "fp8", "int4w")
NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.RMSNorm, nn.BatchNorm1d, nn.BatchNorm2d,
              nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
              nn.LocalResponseNorm)
_CONV = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
         nn.ConvTranspose3d)


class PlanError(RuntimeError):
    """The planner cannot build or run a plan for this model (the message says why)."""


def kind_of(mod: nn.Module) -> str:
    if type(mod) is nn.Linear or (isinstance(mod, nn.Linear) and not mod._modules):
        return "linear"
    if isinstance(mod, _CONV):
        return "conv"
    if any(isinstance(m, (nn.Linear, nn.MultiheadAttention) + _CONV) for m in mod.modules()):
        return "matmul"  # an atomic module holding matmuls (attention)
    return "other"


def candidates_for(kind: str) -> Tuple[str, ...]:
    if kind == "linear":
        return PRECISIONS
    if kind in ("conv", "matmul"):
        return ("fp32", "tf32", "bf16", "fp16")
    return ("fp32", "bf16", "fp16")


def int4_spec(in_features: int) -> Optional[str]:
    for g in (128, 64, 32):
        if in_features % g == 0:
            return f"int4:g{g}"
    return None


def quant_spec(p: str, in_features: int) -> Optional[str]:
    if p == "int8w":
        return "int8"
    if p == "fp8":
        return "fp8_e4m3:channel"
    if p == "int4w":
        return int4_spec(in_features)
    raise ValueError(p)


def make_quant(base: nn.Linear, p: str, name: str) -> nn.Module:
    """The Truestar quantized kernel layer for precision p (tests patch this to inject bugs)."""
    from ..quant.model import QuantLinear

    spec = quant_spec(p, base.in_features)
    if spec is None:
        raise PlanError(f"{p}: in_features {base.in_features} is not a multiple of 32")
    q = QuantLinear(base, spec, name, verify=True)
    q.raise_on_error = False  # the planner reads the verdict and rejects the candidate itself
    return q


def _cast(obj: Any, dt: torch.dtype) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(dt) if obj.is_floating_point() and obj.dtype != dt else obj
    if isinstance(obj, tuple):
        return tuple(_cast(o, dt) for o in obj)
    if isinstance(obj, list):
        return [_cast(o, dt) for o in obj]
    if isinstance(obj, dict):
        return {k: _cast(v, dt) for k, v in obj.items()}
    return obj


@dataclass
class ArgSpec:
    """What a unit was called with: float tensors by shape only (timing makes fresh ones),
    everything else kept as it was."""

    args: Any
    kwargs: Any
    float_numel: int


def _spec(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return ("__float__", tuple(obj.shape), float(obj.detach().float().std()) if obj.numel() > 1 else 1.0)
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(_spec(o) for o in obj)
    if isinstance(obj, list):
        return [_spec(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _spec(v) for k, v in obj.items()}
    return obj


def _numel(obj: Any) -> int:
    if isinstance(obj, tuple) and len(obj) == 3 and obj[0] == "__float__":
        n = 1
        for s in obj[1]:
            n *= s
        return n
    if isinstance(obj, (tuple, list)):
        return sum(_numel(o) for o in obj)
    if isinstance(obj, dict):
        return sum(_numel(v) for v in obj.values())
    return 0


def materialize(obj: Any, dt: torch.dtype, device, gen: torch.Generator) -> Any:
    if isinstance(obj, tuple) and len(obj) == 3 and obj[0] == "__float__":
        std = obj[2] if obj[2] == obj[2] and obj[2] > 0 else 1.0
        return (torch.randn(obj[1], generator=gen, device=device) * std).to(dt)
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(materialize(o, dt, device, gen) for o in obj)
    if isinstance(obj, list):
        return [materialize(o, dt, device, gen) for o in obj]
    if isinstance(obj, dict):
        return {k: materialize(v, dt, device, gen) for k, v in obj.items()}
    return obj


@contextlib.contextmanager
def strict_fp32():
    """Strict float32 for everything outside tf32 units (PyTorch's default runs convolutions in
    TF32, which would make the fp32 baseline quietly TF32)."""
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m
        torch.backends.cudnn.allow_tf32 = c


class Unit(nn.Module):
    """Runs the wrapped module in its active precision. Variants are built once and kept."""

    def __init__(self, name: str, base: nn.Module):
        super().__init__()
        self.unit_name = name
        self.base = base
        self.kind = kind_of(base)
        self.variants: Dict[str, nn.Module] = {}  # a plain dict: not registered as submodules
        self.active = "fp32"
        self.calls = 0
        self.order: Optional[int] = None
        self.record: Optional[Callable[["Unit"], int]] = None
        self.spec: Optional[ArgSpec] = None
        self.failed: Dict[str, str] = {}

    def __getattr__(self, name: str):
        # a parent that reads a child's weights (MultiheadAttention reads out_proj.weight) still
        # finds them; the child is then never called and discovery merges it into the parent
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("base")
            if base is None:
                raise
            return getattr(base, name)

    def variant(self, p: str) -> nn.Module:
        v = self.variants.get(p)
        if v is None:
            if p in ("fp32", "tf32"):
                v = self.base
            elif p in ("bf16", "fp16"):
                v = copy.deepcopy(self.base).to(STORAGE[p]).eval()
            else:
                if self.kind != "linear":
                    raise PlanError(f"{self.unit_name}: {p} needs a Linear layer")
                v = make_quant(self.base, p, self.unit_name)
            self.variants[p] = v
        return v

    def forward(self, *args, **kwargs):
        self.calls += 1
        if self.record is not None and self.order is None:
            self.order = self.record(self)
            self.spec = ArgSpec(_spec(args), _spec(kwargs), _numel(_spec(args)) + _numel(_spec(kwargs)))
        p = self.active
        dt = STORAGE[p]
        args, kwargs = _cast(args, dt), _cast(kwargs, dt)
        mod = self.variant(p)
        if p == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                return mod(*args, **kwargs)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        return mod(*args, **kwargs)

    def kernel_checks(self, p: str) -> List[Any]:
        v = self.variants.get(p)
        return list(getattr(v, "checks", []) or [])


def _set(root: nn.Module, name: str, mod: nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, leaf, mod)


@dataclass
class Wrapped:
    """A copy of the model with every unit wrapped, plus the tie groups."""

    model: nn.Module
    units: Dict[str, Unit]                     # by name, in execution order after `record`
    groups: List[List[str]] = field(default_factory=list)
    group_of: Dict[str, int] = field(default_factory=dict)
    fp32_only_groups: Dict[int, str] = field(default_factory=dict)
    merged: List[str] = field(default_factory=list)   # modules made atomic (children never called)
    tie_reasons: Dict[int, str] = field(default_factory=dict)

    def set(self, assign: Dict[str, str]) -> None:
        for n, u in self.units.items():
            u.active = assign.get(n, "fp32")

    def ordered(self) -> List[str]:
        return sorted(self.units, key=lambda n: (self.units[n].order is None, self.units[n].order or 0))


def _discover(model: nn.Module, atomic: Sequence[str]) -> List[str]:
    names: List[str] = []

    def walk(mod: nn.Module, name: str):
        kids = list(mod.named_children())
        if not kids or name in atomic:
            names.append(name)
            return
        for cn, c in kids:
            walk(c, f"{name}.{cn}" if name else cn)

    walk(model, "")
    return names


def wrap(model: nn.Module, device: str, run: Callable[[nn.Module], Any]) -> Wrapped:
    """Wrap a float32 copy of model on device. `run(model)` runs one forward on the probe inputs.
    Units whose module is never called (the parent reads their weights) are merged into the
    parent until every unit runs."""
    atomic: List[str] = []
    for _ in range(64):
        m = copy.deepcopy(model).to(device).float().eval()
        names = _discover(m, atomic)
        if names == [""]:
            raise PlanError("the model has no submodules to plan; wrap its layers in nn.Module "
                            "children so they can get their own precision")
        units: Dict[str, Unit] = {}
        for n in names:
            u = Unit(n, m.get_submodule(n))
            units[n] = u
            _set(m, n, u)
        counter = [0]

        def rec(u: Unit) -> int:
            counter[0] += 1
            return counter[0]

        for u in units.values():
            u.record = rec
        with torch.no_grad(), strict_fp32():
            run(m)
        for u in units.values():
            u.record = None
        dead = [n for n, u in units.items() if u.calls == 0]
        if not dead:
            w = Wrapped(m, {n: units[n] for n in sorted(units, key=lambda n: units[n].order)},
                        merged=list(atomic))
            for n in w.units:
                w.units[n].calls = 0
            return w
        parents = sorted({n.rpartition(".")[0] for n in dead})
        new = [p for p in parents if p not in atomic and p != ""]
        if not new:
            raise PlanError("units never called even after merging into their parents: "
                            + ", ".join(dead[:6]))
        atomic.extend(new)
    raise PlanError("unit discovery did not converge")


def _units_under(w: Wrapped, module: str) -> List[str]:
    if module == "":
        return list(w.units)
    return [n for n in w.units if n == module or n.startswith(module + ".")]


def find_groups(w: Wrapped, run: Callable[[nn.Module], Any]) -> None:
    """Tie the units whose glue code mixes dtypes in an op that does not promote. Probes run the
    model with alternating activation dtypes; an error inside module M ties every unit under M."""
    parent = {n: n for n in w.units}

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    reasons: Dict[str, str] = {}
    fp32_only: Dict[str, str] = {}
    order = w.ordered()
    stack: List[str] = []
    hooks = []
    unit_mods = {id(u) for u in w.units.values()}
    for name, mod in w.model.named_modules():
        if id(mod) in unit_mods or any(id(mod) == id(u.base) for u in w.units.values()):
            continue
        if name and any(name == n or name.startswith(n + ".") for n in w.units):
            continue  # inside a unit

        def pre(m, a, name=name):
            stack.append(name)

        def post(m, a, o, name=name):
            if stack and stack[-1] == name:
                stack.pop()
        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))
    pairs = [(torch.float32, torch.float16), (torch.float16, torch.float32),
             (torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)]
    try:
        for a, b in pairs:
            for _ in range(len(w.units) + 2):
                roots: Dict[str, int] = {}
                assign = {}
                for n in order:
                    r = root(n)
                    if r in fp32_only:
                        assign[n] = "fp32"
                        continue
                    k = roots.setdefault(r, len(roots))
                    assign[n] = PLAIN[a if k % 2 == 0 else b]
                w.set(assign)
                stack.clear()
                try:
                    with torch.no_grad(), strict_fp32():
                        run(w.model)
                    break
                except (RuntimeError, TypeError) as exc:
                    where = stack[-1] if stack else ""
                    members = _units_under(w, where)
                    rs = {root(n) for n in members}
                    msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                    if len(rs) <= 1:
                        if not members:
                            raise PlanError(f"probe run failed outside every unit: {msg}") from exc
                        r = root(members[0])
                        if r in fp32_only:
                            raise PlanError(f"probe run fails in {where or '<model>'} with every "
                                            f"unit there in float32: {msg}") from exc
                        fp32_only[r] = f"{where or '<model>'} mixes a low precision unit output with "\
                                       f"float32 values it owns ({msg})"
                        continue
                    first = members[0]
                    for n in members[1:]:
                        parent[root(n)] = root(first)
                    reasons[root(first)] = f"{where or '<model>'}: {msg}"
            else:
                raise PlanError("probe runs with mixed activation dtypes keep failing after tying "
                                "units; the model cannot be planned per layer")
    finally:
        for h in hooks:
            h.remove()
        w.set({})
    groups: Dict[str, List[str]] = {}
    for n in order:
        groups.setdefault(root(n), []).append(n)
    w.groups = list(groups.values())
    w.group_of = {n: i for i, g in enumerate(w.groups) for n in g}
    for i, g in enumerate(w.groups):
        r = root(g[0])
        if r in fp32_only:
            w.fp32_only_groups[i] = fp32_only[r]
        if len(g) > 1:
            w.tie_reasons[i] = reasons.get(r, "")
