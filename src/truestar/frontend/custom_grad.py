"""User written gradients for native functions, verified when they are registered.

    @truestar.function
    def softplus(x: Tensor["N"]) -> Tensor["N"]:
        return ops.where(x > 20.0, x, ops.log(1.0 + ops.exp(ops.minimum(x, 20.0))))

    @softplus.grad
    def _(x, g):                  # the primal's argument names, plus g (the output cotangent)
        return g * ops.sigmoid(x)  # and optionally out (the primal output)

The rule is compiled like any native function, typed from the primal: each argument named like a
primal tensor parameter gets that parameter's type, ``g`` and ``out`` get the output type. It
returns one gradient per floating point tensor parameter, in the primal's order (a bare tensor
when there is one); the number 0 means a zero gradient. A gradient of the wrong type is refused at
once, naming the parameter.

Then the rule is run against the primal on generated inputs, at two sizes of every symbolic dim:

* against reverse mode autodiff of the primal (``truestar.ir.grad``), entry by entry, when every op
  of the primal has a gradient rule;
* against finite differences of the primal, always: for each floating point input, three random
  directions ``v``, ``<g, f(x + h v) - f(x - h v)> / 2h`` with a fourth order stencil, compared with
  ``<rule_x, v>``. A direction where steps ``h`` and ``h/2`` disagree (a kink) is drawn again.

A rule that disagrees is refused with ``CustomGradientError`` naming every wrong input gradient and
its error. ``unverified="reason"`` skips the numeric check for a rule that differs on purpose (a
straight through estimator); the reason is kept in the report.

Inside other native functions a call to such a function becomes one IR ``custom_vjp`` node, so the
rule is what ``truestar.ir.grad`` and ``truestar.torch_op`` differentiate.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..ir import reference
from ..ir.core import Builder, IRError, Program, TensorType, UnsupportedOpError, eval_dim, symbol
from .errors import FrontendError, FrontendShapeError, InputError
from .translate import FunctionSource, Scope, T, Translator, is_number, is_tensor

AUTODIFF_BAR = 1e-8   # entrywise, relative to the largest entry: correct rules agree to about 1e-14
FD_BAR = 1e-6         # directional: the fourth order stencil at h = 1e-3 is good to about 1e-11
FD_STEP = 1e-3
DIRECTIONS = 3
DEFAULT_DOMAIN = (-2.0, 2.0)
RESERVED = ("g", "out")


class CustomGradientError(FrontendError):
    """A user written gradient disagrees with the function it belongs to (or cannot be checked)."""

    code = "wrong_gradient"


@dataclass
class InputCheck:
    name: str
    autodiff_error: Optional[float]
    fd_error: Optional[float]
    fd_directions: int
    wrong: bool


@dataclass
class GradReport:
    """What registration checked. ``to_dict()`` is JSON compatible."""

    function: str
    verified: bool
    unverified_reason: Optional[str]
    autodiff: Optional[str]          # None when used; otherwise why autodiff of the primal was unavailable
    sizes: List[Dict[str, int]]
    inputs: List[InputCheck] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": self.function, "verified": self.verified, "unverified_reason": self.unverified_reason,
            "autodiff_unavailable": self.autodiff, "sizes": self.sizes, "seconds": round(self.seconds, 4),
            "bars": {"autodiff": AUTODIFF_BAR, "finite_differences": FD_BAR},
            "inputs": [vars(c) for c in self.inputs],
        }

    def __str__(self) -> str:
        if not self.verified:
            return f"custom gradient of {self.function}: not verified ({self.unverified_reason})"
        parts = []
        for c in self.inputs:
            ad = "n/a" if c.autodiff_error is None else f"{c.autodiff_error:.1e}"
            fd = "n/a" if c.fd_error is None else f"{c.fd_error:.1e}"
            parts.append(f"{c.name}: autodiff {ad}, finite differences {fd}")
        return (f"custom gradient of {self.function}: verified in {self.seconds:.2f} s at "
                f"{self.sizes} ({'; '.join(parts)})")


class CustomGrad:
    """A rule attached to a native function (one per function)."""

    def __init__(self, primal, fn: Callable, unverified: Optional[str], domain, sizes, seed: int):
        self.primal = primal
        self.fn = fn
        self.src = FunctionSource(fn)
        self.unverified = unverified
        self.domain = dict(domain or {})
        self.sizes = sizes
        self.seed = seed
        self.reports: Dict[Tuple, GradReport] = {}
        sig = inspect.signature(fn)
        self.params = list(sig.parameters)
        allowed = set(primal.specs) | set(primal.static_params) | set(RESERVED)
        for n in RESERVED:
            if n in primal.specs or n in primal.static_params:
                raise self._err(f"{primal.name} has a parameter named {n!r}, which the gradient rule uses for "
                                f"{'the output cotangent' if n == 'g' else 'the primal output'}; rename it",
                                code="bad_gradient_signature")
        bad = [n for n in self.params if n not in allowed]
        if bad:
            raise self._err(f"the gradient rule takes {bad}, which are not parameters of {primal.name}; it may take "
                            f"{sorted(set(primal.specs) | set(primal.static_params))} by name, plus g (the output "
                            "cotangent) and out (the primal output)", code="bad_gradient_signature")
        if "g" not in self.params:
            raise self._err("the gradient rule must take g, the cotangent of the output", code="bad_gradient_signature")
        for p in sig.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise self._err("*args and **kwargs are not supported in a gradient rule",
                                code="bad_gradient_signature")
        for k in self.domain:
            if k not in primal.specs:
                raise self._err(f"domain names {k!r}, which is not a tensor parameter of {primal.name}",
                                code="bad_gradient_signature")

    def _err(self, msg, cls=CustomGradientError, **kw):
        return self.src.error(self.src.tree, msg, cls, **kw)

    # -------------------------------------------------------------- regions

    def regions(self, in_types: List[Tuple[str, TensorType]], static: Dict[str, Any],
                symbols: Dict[str, Tuple[int, Optional[int]]]) -> Tuple[Program, Program]:
        """The fwd and bwd regions for tensor inputs of these types (the declared ones at the top
        level, the caller's at a call site)."""
        fwd, outs = translate_region(self.primal.src, f"{self.primal.name}_fwd", in_types, static, symbols, [])
        if len(outs) != 1:
            raise self.primal.src.error(self.primal.src.tree, f"{self.primal.name} returns {len(outs)} values; "
                                        "a function with a custom gradient returns one tensor",
                                        CustomGradientError, code="bad_gradient_signature")
        out_t = fwd.outputs[0].type
        if not out_t.dtype.is_floating_point:
            raise self.primal.src.error(self.primal.src.tree, f"{self.primal.name} returns {out_t}; a custom "
                                        "gradient needs a floating point output", CustomGradientError,
                                        code="bad_gradient_signature")
        float_names = [n for n, t in in_types if t.dtype.is_floating_point]
        bwd, gouts = translate_region(self.src, f"{self.primal.name}_bwd", in_types + [("out", out_t), ("g", out_t)],
                                      static, symbols, self.params, zeros_for=[t for n, t in in_types
                                                                               if t.dtype.is_floating_point])
        if len(gouts) != len(float_names):
            raise self._err(f"the gradient rule returns {len(gouts)} value(s); {self.primal.name} has "
                            f"{len(float_names)} floating point tensor parameter(s) {float_names}, so it must "
                            f"return {len(float_names)} in that order", FrontendShapeError, code="wrong_gradient_count")
        types = dict(in_types)
        for n, v in zip(float_names, bwd.outputs):
            if not v.type.same_as(types[n]):
                hint = None
                if v.type.rank > types[n].rank:
                    k = v.type.rank - types[n].rank
                    hint = (f"the gradient of {n} must have {n}'s shape: sum the cotangent over the broadcast dims, "
                            f"for example .sum({0 if k == 1 else tuple(range(k))})")
                raise self._err(f"the gradient returned for {n} has type {v.type} but {n} is {types[n]}",
                                FrontendShapeError, code="wrong_gradient_shape",
                                operands=[(f"gradient of {n}", v.type), (n, types[n])], suggestion=hint)
        return fwd, bwd

    # -------------------------------------------------------------- verification

    def verify(self, fwd: Program, bwd: Program, symbols: Dict[str, Tuple[int, Optional[int]]],
               key: Tuple) -> GradReport:
        t0 = time.perf_counter()
        report = GradReport(self.primal.name, self.unverified is None, self.unverified, None, [])
        if self.unverified is not None:
            self.reports[key] = report
            return report
        names = [v.name for v in fwd.inputs]
        float_idx = [i for i, v in enumerate(fwd.inputs) if v.type.dtype.is_floating_point]
        try:
            from ..ir.autodiff import grad
            ad = grad(fwd, wrt=[names[i] for i in float_idx], seed="input", name=f"{fwd.name}_autodiff")
        except UnsupportedOpError as e:
            ad = None
            report.autodiff = str(e).splitlines()[1].strip() if len(str(e).splitlines()) > 1 else str(e)
        gen = torch.Generator().manual_seed(self.seed)
        worst_ad = {i: None for i in float_idx}
        worst_fd = {i: None for i in float_idx}
        n_dirs = {i: 0 for i in float_idx}
        for sizes in self._size_sets(fwd, symbols):
            report.sizes.append(sizes)
            bind = {symbol(k): v for k, v in sizes.items()}
            xs = [self._draw(v.type, bind, gen, self.domain.get(v.name, DEFAULT_DOMAIN)) for v in fwd.inputs]
            (y,) = reference.run(fwd, *xs)
            if not torch.isfinite(y).all():
                raise self._err(f"cannot check the rule: {self.primal.name} is not finite on the generated inputs "
                                f"(uniform in {DEFAULT_DOMAIN} unless domain= says otherwise, sizes {sizes})",
                                code="gradient_unverifiable",
                                suggestion="pass domain={'x': (lo, hi)} to @fn.grad for inputs with a restricted domain")
            g = torch.randn(y.shape, generator=gen, dtype=torch.float64)
            user = reference.run(bwd, *xs, y, g)
            if ad is not None:
                ref = reference.run(ad, *xs, g)[1:]
                for k, i in enumerate(float_idx):
                    err = _rel(user[k], ref[k])
                    worst_ad[i] = err if worst_ad[i] is None else max(worst_ad[i], err)
            for k, i in enumerate(float_idx):
                for _ in range(DIRECTIONS):
                    err = self._fd(fwd, xs, i, user[k], g, gen)
                    if err is None:
                        continue
                    n_dirs[i] += 1
                    worst_fd[i] = err if worst_fd[i] is None else max(worst_fd[i], err)
        wrong = []
        for i in float_idx:
            bad = (worst_ad[i] is not None and worst_ad[i] > AUTODIFF_BAR) or \
                  (worst_fd[i] is not None and worst_fd[i] > FD_BAR)
            report.inputs.append(InputCheck(names[i], worst_ad[i], worst_fd[i], n_dirs[i], bad))
            if bad:
                wrong.append(report.inputs[-1])
            elif worst_ad[i] is None and n_dirs[i] == 0:
                raise self._err(f"cannot check the gradient of {names[i]}: every finite difference direction "
                                "crossed a kink and autodiff of the primal is unavailable "
                                f"({report.autodiff})", code="gradient_unverifiable",
                                suggestion="pass domain= to keep the inputs away from the kink, or "
                                           "unverified=\"reason\" if the rule is meant to differ")
        report.seconds = time.perf_counter() - t0
        if wrong:
            lines = []
            for c in wrong:
                parts = []
                if c.autodiff_error is not None:
                    parts.append(f"relative error {c.autodiff_error:.1e} against autodiff of the forward "
                                 f"(bar {AUTODIFF_BAR:.0e})")
                if c.fd_error is not None:
                    parts.append(f"{c.fd_error:.1e} against finite differences (bar {FD_BAR:.0e})")
                lines.append(f"{c.name} (" + ", ".join(parts) + ")")
            first = wrong[0].name
            raise self._err(f"the custom gradient of {self.primal.name} is wrong for " + " and ".join(lines)
                            + f", checked at sizes {report.sizes}",
                            operands=[(f"gradient of {c.name}", next(v.type for v in fwd.inputs if v.name == c.name))
                                      for c in wrong],
                            suggestion=f"fix the rule for {first}; if it is meant to differ (a straight through "
                                       f"estimator), register it with @{self.primal.name}.grad(unverified=\"why\")")
        self.reports[key] = report
        return report

    def _size_sets(self, fwd: Program, symbols) -> List[Dict[str, int]]:
        if self.sizes is not None:
            return [dict(s) for s in self.sizes]
        used = sorted({str(s) for v in fwd.inputs for d in v.type.shape if not isinstance(d, int)
                       for s in getattr(d, "free_symbols", ())})
        if not used:
            return [{}]
        out = []
        for want in (3, 2):
            s = {}
            for name in used:
                lo, hi = symbols.get(name, (1, None))
                s[name] = max(lo, want if hi is None else min(hi, want))
            if s not in out:
                out.append(s)
        return out

    @staticmethod
    def _draw(t: TensorType, bind, gen, domain) -> torch.Tensor:
        shape = [int(eval_dim(d, bind)) for d in t.shape]
        if t.dtype.is_floating_point:
            lo, hi = domain
            return torch.rand(shape, generator=gen, dtype=torch.float64) * (hi - lo) + lo
        if t.dtype == torch.bool:
            return torch.rand(shape, generator=gen) > 0.5
        lo, hi = domain if domain is not DEFAULT_DOMAIN else (0, 2)
        return torch.randint(int(lo), int(hi), shape, generator=gen, dtype=t.dtype)

    @staticmethod
    def _fd(fwd: Program, xs, i, user_grad, g, gen) -> Optional[float]:
        """Relative error of ``<user_grad, v>`` against a finite difference along a random ``v``;
        None when the difference is unreliable here (steps h and h/2 disagree: a kink)."""
        v = torch.randn(xs[i].shape, generator=gen, dtype=torch.float64)

        def phi(t):
            ys = list(xs)
            ys[i] = xs[i] + t * v
            return float((reference.run(fwd, *ys)[0] * g).sum())

        def stencil(h):
            return (8 * (phi(h) - phi(-h)) - (phi(2 * h) - phi(-2 * h))) / (12 * h)

        fd1, fd2 = stencil(FD_STEP), stencil(FD_STEP / 2)
        dot = float((user_grad * v).sum())
        typical = float(user_grad.norm()) * float(v.norm()) / math.sqrt(max(1, v.numel()))
        scale = max(abs(fd1), abs(dot), typical, 1e-300)
        if abs(fd1 - fd2) > FD_BAR * 0.1 * max(abs(fd1), typical, 1e-300):
            return None
        return abs(fd1 - dot) / scale


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return 0.0 if torch.equal(a, b) else float("inf")
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-300)
    return float((a - b).abs().max()) / scale


def translate_region(src: FunctionSource, name: str, in_types: List[Tuple[str, TensorType]],
                     static: Dict[str, Any], symbols, accepted: Sequence[str],
                     zeros_for: Sequence[TensorType] = ()) -> Tuple[Program, List[Any]]:
    """Translate ``src``'s body into a closed program whose inputs have ``in_types``.

    Every entry of ``in_types`` becomes an input (in order) whether or not the body reads it, so the
    region's inputs line up with the node's operands. ``accepted`` limits which names the body sees
    (empty: all). A returned number at position k becomes a constant of type ``zeros_for[k]``."""
    b = Builder(name)
    b.symbols = dict(symbols)
    scope = Scope(b)
    for n, t in in_types:
        v = b.input(n, t.dtype, t.shape)
        if not accepted or n in accepted:
            scope.vars[n] = T(v, b)
    for n, v in static.items():
        if not accepted or n in accepted:
            scope.vars[n] = v
    tr = Translator(src)
    r = tr.block(src.tree.body, scope)
    if r is None:
        raise src.error(src.tree, f"{src.name} never returns a value", code="no_return")
    values = list(r.value) if isinstance(r.value, tuple) else [r.value]
    outs = []
    for k, v in enumerate(values):
        if not is_tensor(v):
            if is_number(v) and k < len(zeros_for):
                t = zeros_for[k]
                v = tr.emit(scope, "full", [], r.node, shape=list(t.shape), value=float(v), dtype=t.dtype)
            elif is_number(v):
                v = tr.const(scope, v, None, r.node)
            else:
                raise tr.err(r.node, f"return value {k} is {v!r}; return tensors", code="type_error")
        outs.append(v)
    return b.build([o.v for o in outs]), outs


def wrap_program(primal: Program, fwd: Program, bwd: Program, fn_name: str, source: str) -> Program:
    """The top level program of a function with a custom gradient: one ``custom_vjp`` node."""
    b = Builder(primal.name)
    b.symbols = dict(primal.symbols)
    ins = [b.input(v.name, v.type.dtype, v.type.shape) for v in primal.inputs]
    y = b.op("custom_vjp", *ins, name=fn_name, fwd=fwd, bwd=bwd, source=source)
    return b.build([y])


def emit_call(tr: Translator, fn, bound: Dict[str, Any], e, scope: Scope):
    """A call to a function with a custom gradient inside another native function: one node."""
    cg: CustomGrad = fn._custom_grad
    in_types, args, static = [], [], {}
    for pname in fn.py_signature.parameters:
        v = bound[pname]
        if pname in fn.specs:
            v = scope.adopt(v)
            args.append(v)
            in_types.append((pname, v.type))
        else:
            static[pname] = v
    fwd, bwd = cg.regions(in_types, static, scope.b.symbols)
    return tr.emit(scope, "custom_vjp", args, e, name=fn.name, fwd=fwd, bwd=bwd)
