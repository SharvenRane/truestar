"""Register an ``@truestar.function`` as a real PyTorch custom op, with no C++ toolchain.

    import truestar
    from truestar import Tensor, ops

    @truestar.torch_op
    def rmsnorm(x: Tensor["B", 128], w: Tensor[128]) -> Tensor["B", 128]:
        return x * ops.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w

    y = rmsnorm(x, w)                       # torch.ops.truestar.rmsnorm under the hood
    torch.compile(model, fullgraph=True)    # no graph break: the op is opaque to Dynamo

What registration does (see docs/CUSTOM_OPS.md):

* the schema comes from the typed signature (one ``Tensor`` argument per ``Tensor[...]`` parameter);
* the fake (meta) implementation is derived from the IR shape rules: every symbolic dim of the
  signature is bound to the incoming size (a torch SymInt under torch.compile), every fixed dim and
  every repeated symbol is checked with ``torch._check``, and each output size is the IR's dim
  expression evaluated over those sizes, so a symbolic batch stays symbolic;
* the autograd formula is the IR program from ``truestar.ir.grad``, itself registered as a second
  custom op (``<name>_backward``) with its own derived fake implementation, so AOTAutograd can trace
  the backward under torch.compile;
* the real implementation runs through ``executor.execute``: a registered executor, the kernel
  layer, or the float64 reference interpreter with a ``SlowPathWarning``.
"""

from __future__ import annotations

import inspect
import keyword
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import sympy
import torch

from ..frontend.function import NativeFunction, function as _native_function
from ..ir.core import IRError, Program, UnsupportedOpError, Value
from . import executor as _exec

NAMESPACE = "truestar"
_REGISTERED: Dict[str, "TorchOp"] = {}


class CustomOpError(Exception):
    """A native function cannot be registered as a torch op, with the reason."""


# ----------------------------------------------------------------------------- shape rules on SymInts


def _eval_dim(d, env: Dict[str, Any]):
    """Evaluate an IR dim (int or sympy expression) over sizes that may be torch SymInts."""
    if isinstance(d, int):
        return d
    if isinstance(d, sympy.Integer):
        return int(d)
    if isinstance(d, sympy.Symbol):
        return env[str(d)]
    if isinstance(d, sympy.Add):
        acc = 0
        for a in d.args:
            acc = acc + _eval_dim(a, env)
        return acc
    if isinstance(d, sympy.Mul):
        acc = 1
        for a in d.args:
            acc = acc * _eval_dim(a, env)
        return acc
    if isinstance(d, sympy.Pow) and isinstance(d.exp, sympy.Integer) and int(d.exp) > 0:
        base = _eval_dim(d.base, env)
        acc = 1
        for _ in range(int(d.exp)):
            acc = acc * base
        return acc
    if isinstance(d, sympy.floor):
        num, den = d.args[0].as_numer_denom()
        return _eval_dim(num, env) // _eval_dim(den, env)
    if isinstance(d, sympy.ceiling):
        num, den = d.args[0].as_numer_denom()
        return -((-_eval_dim(num, env)) // _eval_dim(den, env))
    raise CustomOpError(f"cannot evaluate the dim expression {d} on torch sizes")


def _bind(program: Program, tensors: Sequence[torch.Tensor], who: str) -> Dict[str, Any]:
    """Bind every plain symbol of the program's inputs to the incoming sizes; check the rest.

    Works on real tensors and on fake tensors whose sizes are SymInts: equalities are stated with
    ``torch._check`` so torch.compile records them as facts instead of guarding on a value."""
    if len(tensors) != len(program.inputs):
        raise TypeError(f"{who} takes {len(program.inputs)} tensors, got {len(tensors)}")
    env: Dict[str, Any] = {}
    for v, t in zip(program.inputs, tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{who}: {v.name} must be a tensor {v.type}, got {type(t).__name__}")
        if t.dim() != v.type.rank:
            raise ValueError(f"{who}: {v.name} has rank {t.dim()}, the signature declares {v.type}")
        if t.dtype != v.type.dtype:
            raise ValueError(f"{who}: {v.name} has dtype {t.dtype}, the signature declares {v.type}")
        for d, n in zip(v.type.shape, t.shape):
            if isinstance(d, sympy.Symbol) and str(d) not in env:
                env[str(d)] = n
    for v, t in zip(program.inputs, tensors):
        for i, (d, n) in enumerate(zip(v.type.shape, t.shape)):
            want = _eval_dim(d, env)
            torch._check(n == want, lambda v=v, i=i, d=d, n=n: (
                f"{who}: {v.name} dim {i} is {n}, the signature {v.type} needs {d}"))
    for s, (lo, hi) in program.symbols.items():
        if s in env:
            n = env[s]
            torch._check(n >= lo, lambda s=s, n=n, lo=lo: f"{who}: symbol {s} = {n} is below its minimum {lo}")
            if hi is not None:
                torch._check(n <= hi, lambda s=s, n=n, hi=hi: f"{who}: symbol {s} = {n} is above its maximum {hi}")
    return env


def _fake_outputs(program: Program, tensors: Sequence[torch.Tensor], who: str) -> List[torch.Tensor]:
    env = _bind(program, tensors, who)
    dev = tensors[0].device if tensors else torch.device("cpu")
    outs = []
    for v in program.outputs:
        shape = [_eval_dim(d, env) for d in v.type.shape]
        outs.append(tensors[0].new_empty(shape, dtype=v.type.dtype, device=dev) if tensors
                    else torch.empty(shape, dtype=v.type.dtype, device=dev))
    return outs


# ----------------------------------------------------------------------------- program helpers


def _prune(program: Program, keep: Sequence[int], name: str) -> Program:
    """A copy of ``program`` with only the outputs ``keep`` and only the nodes they need."""
    outputs = [program.outputs[i] for i in keep]
    needed = {id(v) for v in outputs}
    nodes = []
    for n in reversed(program.nodes):
        if any(id(o) in needed for o in n.outputs):
            nodes.append(n)
            needed.update(id(x) for x in n.inputs if x is not None)
    nodes.reverse()
    params = [p for p in program.params if id(p) in needed]
    out = Program(name, list(program.inputs), params, {p.name: program.param_data[p.name] for p in params},
                  nodes, outputs, dict(program.symbols))
    out.verify()
    return out


def _schema_name(n: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", n) or keyword.iskeyword(n) or n in ("self", "out"):
        return f"{n}_"
    return n


def _exists(qualname: str) -> bool:
    try:
        return torch._C._dispatch_find_schema_or_throw(qualname, "") is not None
    except RuntimeError:
        return False


def _unique_name(base: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", base)
    if base[0].isdigit():
        base = "_" + base
    name, k = base, 1
    while name in _REGISTERED or name.endswith("_backward") or _exists(f"{NAMESPACE}::{name}"):
        k += 1
        name = f"{base}_{k}"
    return name


def _schema(arg_names: Sequence[str], n_out: int) -> str:
    args = ", ".join(f"Tensor {_schema_name(a)}" for a in arg_names)
    ret = "Tensor" if n_out == 1 else "(" + ", ".join(["Tensor"] * n_out) + ")"
    return f"({args}) -> {ret}"


# ----------------------------------------------------------------------------- the op


class TorchOp:
    """A native function registered as ``torch.ops.truestar.<name>``. Call it like the function."""

    def __init__(self, fn: NativeFunction, name: Optional[str] = None, static: Optional[Dict[str, Any]] = None,
                 kernels: bool = True, autograd: bool = True):
        t0 = time.perf_counter()
        self.native = fn
        compiled = fn.specialize(**(static or {}))
        self.program: Program = compiled.program
        self.signature = compiled.signature
        self.returns_tuple = compiled.signature.returns_tuple
        self.allow_kernel = kernels
        if compiled.signature.in_trees:
            raise CustomOpError(f"{fn.name} takes structured arguments ({', '.join(compiled.signature.in_trees)}); a "
                                "torch op takes tensors, so pass each tensor as its own parameter")
        self.arg_names = [n for n, _ in compiled.signature.params]
        n_out = len(self.program.outputs)
        if n_out == 0:
            raise CustomOpError(f"{fn.name} returns nothing")
        if any(p.kind != "input" for p in self.program.inputs):
            raise CustomOpError(f"{fn.name}: unexpected program inputs")
        self.name = _unique_name(name or fn.name)
        self.qualname = f"{NAMESPACE}::{self.name}"
        self.schema = _schema(self.arg_names, n_out)

        # the backward program: gradients of every float output wrt every float input
        self.grad_program: Optional[Program] = None
        self.grad_error: Optional[str] = None
        self.wrt = [i for i, v in enumerate(self.program.inputs) if v.type.dtype.is_floating_point]
        self.of = [i for i, v in enumerate(self.program.outputs) if v.type.dtype.is_floating_point]
        if autograd and self.wrt and self.of:
            from ..ir.autodiff import grad
            try:
                g = grad(self.program, wrt=[self.program.inputs[i].name for i in self.wrt], of=self.of,
                         seed="input", name=f"{self.program.name}_backward")
                n_fwd = len(self.program.outputs)
                self.grad_program = _prune(g, list(range(n_fwd, len(g.outputs))), g.name)
            except (UnsupportedOpError, IRError) as e:
                self.grad_error = str(e)
        elif not autograd:
            self.grad_error = "autograd=False was passed at registration"
        else:
            self.grad_error = "no floating point input or output to differentiate"

        self._register()
        _REGISTERED[self.name] = self
        self.register_seconds = time.perf_counter() - t0

    # -------------------------------------------------------------- registration

    def _register(self):
        prog, who, allow = self.program, self.qualname, self.allow_kernel
        returns_one = len(prog.outputs) == 1

        def impl(*tensors):
            _bind(prog, tensors, who)
            outs = _exec.execute(prog, [t.detach() for t in tensors], allow)
            return outs[0] if returns_one else tuple(outs)

        def fake(*tensors):
            outs = _fake_outputs(prog, tensors, who)
            return outs[0] if returns_one else tuple(outs)

        self.op = torch.library.custom_op(self.qualname, impl, mutates_args=(), schema=self.schema)
        self.op.register_fake(fake)
        self.overload = getattr(getattr(torch.ops, NAMESPACE), self.name).default

        if self.grad_program is not None:
            gprog = self.grad_program
            bname = f"{self.qualname}_backward"
            n_in = len(prog.inputs)
            bargs = self.arg_names + [f"d_out{i}" for i in self.of]
            bschema = _schema(bargs, len(gprog.outputs))
            b_one = len(gprog.outputs) == 1

            def bimpl(*tensors):
                _bind(gprog, tensors, bname)
                outs = _exec.execute(gprog, [t.detach() for t in tensors], allow)
                return outs[0] if b_one else tuple(outs)

            def bfake(*tensors):
                outs = _fake_outputs(gprog, tensors, bname)
                return outs[0] if b_one else tuple(outs)

            self.backward_op = torch.library.custom_op(bname, bimpl, mutates_args=(), schema=bschema)
            self.backward_op.register_fake(bfake)
            bover = getattr(getattr(torch.ops, NAMESPACE), f"{self.name}_backward").default
            self.backward_overload = bover
            wrt, of, outs_decl = self.wrt, self.of, prog.outputs

            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(*inputs)

            def backward(ctx, *grads):
                saved = ctx.saved_tensors
                cot = []
                for i in of:
                    gi = grads[i]
                    if gi is None:
                        ref = saved[0]
                        shape = [_eval_dim(d, _bind(prog, saved, who)) for d in outs_decl[i].type.shape]
                        gi = ref.new_zeros(shape, dtype=outs_decl[i].type.dtype)
                    cot.append(gi.contiguous())
                res = bover(*saved, *cot)
                res = (res,) if b_one else tuple(res)
                full = [None] * n_in
                for k, i in enumerate(wrt):
                    full[i] = res[k]
                return tuple(full)

            self.op.register_autograd(backward, setup_context=setup_context)
        else:
            self.backward_op = None
            self.backward_overload = None
            reason = self.grad_error

            def setup_context(ctx, inputs, output):
                pass

            def backward(ctx, *grads):
                raise RuntimeError(f"{who} has no backward: {reason}")

            self.op.register_autograd(backward, setup_context=setup_context)

    # -------------------------------------------------------------- use

    def __call__(self, *args, **kwargs):
        if kwargs:
            bound = inspect.signature(self.native.python).bind_partial(*args, **kwargs)
            args = tuple(bound.arguments[n] for n in self.arg_names)
        return self.overload(*args)

    def plan(self, device_type: str = "cpu", backward: bool = False) -> "_exec.Plan":
        """The execution plan the forward (or backward) program gets on a device type."""
        prog = self.grad_program if backward else self.program
        if prog is None:
            raise CustomOpError(f"{self.qualname} has no backward: {self.grad_error}")
        return _exec.plan(prog, device_type, self.allow_kernel)

    def report(self) -> Dict[str, Any]:
        out = {"op": self.qualname, "schema": self.schema, "signature": str(self.signature),
               "register_seconds": round(self.register_seconds, 4),
               "backward": f"{self.qualname}_backward" if self.backward_op is not None else None,
               "backward_error": self.grad_error, "paths": {}}
        for (pid, dev, _), p in list(_exec._PLANS.items()):
            for label, prog in (("forward", self.program), ("backward", self.grad_program)):
                if prog is not None and pid == id(prog):
                    out["paths"][f"{label}/{dev}"] = {"path": p.path, "reason": p.reason, "calls": p.calls}
        return out

    def __repr__(self) -> str:
        return f"<truestar.torch_op {self.qualname}{self.schema}>"


def torch_op(fn=None, *, name: Optional[str] = None, static: Optional[Dict[str, Any]] = None,
             dims: Optional[Dict[str, Any]] = None, kernels: bool = True, autograd: bool = True):
    """Register a native function as a PyTorch custom op. Works as ``torch_op(f)``, ``@torch_op`` or
    ``@torch_op(name=..., static=..., dims=...)``, on a plain typed function or an existing
    ``@truestar.function``. ``static`` fixes the values of non tensor parameters. ``kernels=False``
    forces the reference interpreter (reported as slow)."""
    def wrap(f):
        nf = f if isinstance(f, NativeFunction) else _native_function(f, dims=dims)
        return TorchOp(nf, name=name, static=static, kernels=kernels, autograd=autograd)
    return wrap(fn) if fn is not None else wrap
