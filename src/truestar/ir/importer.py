"""Import ``torch.export`` programs (ATen IR) into the Truestar IR.

Both the default export graph (``torch.export.export``) and the core ATen graph
(``ExportedProgram.run_decompositions()``) are accepted. Many ATen ops lower to one IR op (``view``,
``flatten``, ``unsqueeze`` all become ``reshape``); a few lower to two (``addmm`` becomes
``matmul`` then ``add``). Ops with no data effect in inference (``contiguous``, ``clone``,
``dropout`` with ``train=False``) map to their input.

Rules the importer never breaks:

* an op it does not know is an error naming the op and the user's source line; all unknown ops
  are listed at once, before any work, and nothing is ever skipped;
* an op setting it does not support (attention dropout, transposed conv; training mode batch norm
  and dropout unless ``training=True`` is passed) is an error, never a silent approximation;
* with ``training=True`` the buffers a training graph mutates (batch norm running statistics,
  ``num_batches_tracked``) become explicit state: program inputs after the user inputs and program
  outputs after the user outputs, listed in ``Program.state``; dropout becomes a keyed ``random``
  op reading the program input ``Program.key`` (docs/RANDOM.md);
* every output type is derived by the IR's own shape rules, then compared with the shape
  torch.export recorded; any disagreement is an error (it would be a bug in a shape rule).
"""

from __future__ import annotations

import math
import operator
import re
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import sympy
import torch

from .core import (Dim, IRError, Node, Program, ShapeError, TensorType, UnsupportedOpError, Value,
                   dims_equal, norm_dim, symbol)
from .shapes import infer


class _Missing:
    """A secondary output of an ATen op the IR does not model (saved mean, pool indices)."""

    def __init__(self, reason: str):
        self.reason = reason


_TORCH_DIR = __import__("os").path.dirname(torch.__file__).replace("\\", "/").lower() + "/"



import contextlib as _contextlib


@_contextlib.contextmanager
def plain_einsum():
    """Trace einsum without opt_einsum. When the optional opt_einsum package is installed (jax
    installs it), torch.einsum picks a contraction path from the example shapes at trace time, which
    specializes dims marked symbolic, so the same model would import differently depending on what
    else is installed. The traced graph must not depend on that."""
    be = getattr(torch.backends, "opt_einsum", None)
    if be is None or not hasattr(be, "enabled"):
        yield
        return
    old = be.enabled
    be.enabled = False
    try:
        yield
    finally:
        be.enabled = old

def source_line(node: torch.fx.Node) -> str:
    """The user's source line that produced this node, from the export stack trace.

    Frames inside PyTorch itself (nn.Conv2d.forward and so on) are skipped, so the line points at the
    user's model code, with the innermost module path when torch.export recorded one.
    """
    trace = node.meta.get("stack_trace") or ""
    frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)', trace)
    if not frames:
        return ""
    user = [f for f in frames if not f[0].startswith("<")  # <eval_with_key>: code torch generated
            and not f[0].replace("\\", "/").lower().startswith(_TORCH_DIR)]
    path, line, fn, code = (user or frames)[-1]
    path = path.replace("\\", "/").split("/")[-1]
    out = f"{path}:{line} in {fn}: {code.strip()}"
    stack = node.meta.get("nn_module_stack")
    if stack:
        fqn, cls = list(stack.values())[-1]
        cls = getattr(cls, "__name__", str(cls)).split(".")[-1].strip("'>")
        if fqn:
            out += f" (module {fqn}: {cls})"
    return out


def _describe(node: torch.fx.Node) -> str:
    where = source_line(node)
    return f"{node.target} (node %{node.name})" + (f", {where}" if where else "")


def _plain_sympy(expr):
    """torch's integer division and modulo functions (FloorDiv, PythonMod, ...) as plain sympy
    floor and Mod, the forms the IR's own shape rules produce and the serializer can encode."""
    from torch.utils._sympy import functions as tf

    floor_div = tuple(getattr(tf, n) for n in ("FloorDiv", "CleanDiv") if hasattr(tf, n))
    mods = tuple(getattr(tf, n) for n in ("PythonMod", "Mod") if hasattr(tf, n))
    if floor_div:
        expr = expr.replace(lambda e: isinstance(e, floor_div), lambda e: sympy.floor(e.args[0] / e.args[1]))
    if mods:
        expr = expr.replace(lambda e: isinstance(e, mods), lambda e: sympy.Mod(e.args[0], e.args[1]))
    return expr


class _Ctx:
    def __init__(self, ep, name: str, check_against_export: bool):
        self.ep = ep
        self.check = check_against_export
        self.inputs: List[Value] = []
        self.params: List[Value] = []
        self.param_data: Dict[str, torch.Tensor] = {}
        self.weights: set = set()           # names of the params that are model parameters (trainable)
        self.nodes: List[Node] = []
        self.env: Dict[str, Any] = {}
        self.names: set = set()
        self.sym_map: Dict[sympy.Symbol, sympy.Symbol] = {}  # torch symbol -> IR symbol
        self.node: Optional[torch.fx.Node] = None
        self.n_const = 0
        # training import (docs/RANDOM.md, docs/IR.md): explicit state and the random key
        self.training = False
        self.state_targets: List[str] = []                 # mutated buffers, in first use order
        self.state_in: Dict[str, Value] = {}
        self.state_out: Dict[str, Value] = {}
        self.state_of: Dict[int, str] = {}                 # id(input Value) -> buffer target
        self.key: Optional[Value] = None
        self.sites: Dict[str, int] = {}

    def random_key(self) -> Value:
        """The program input every random op of a training graph reads (created on first use)."""
        if self.key is None:
            self.key = Value(self.fresh("rng_key"), TensorType(torch.int64, ()), "input")
        return self.key

    def site_name(self, kind: str) -> str:
        """``<module path>/<kind>#<k>``: the innermost module that ran the op, k counting its ops."""
        stack = self.node.meta.get("nn_module_stack") if self.node is not None else None
        path = list(stack.values())[-1][0] if stack else ""
        base = f"{path}/{kind}" if path else kind
        k = self.sites.get(base, 0)
        self.sites[base] = k + 1
        return f"{base}#{k}"

    # values -----------------------------------------------------------------

    def fresh(self, base: str) -> str:
        name, i = base, 0
        while name in self.names:
            i += 1
            name = f"{base}_{i}"
        self.names.add(name)
        return name

    def val(self, x: Any) -> Any:
        if isinstance(x, torch.fx.Node):
            v = self.env[x.name]
            if isinstance(v, _Missing):
                raise UnsupportedOpError(f"{_describe(self.node)} uses {v.reason}")
            return v
        if isinstance(x, (list, tuple)):
            return [self.val(e) for e in x]
        return x

    def tensor(self, x: Any) -> Optional[Value]:
        v = self.val(x)
        if v is not None and not isinstance(v, Value):
            raise UnsupportedOpError(f"{_describe(self.node)}: expected a tensor operand, got {v!r}")
        return v

    def dim(self, x: Any) -> Dim:
        v = self.val(x)
        if isinstance(v, Value):
            raise UnsupportedOpError(f"{_describe(self.node)}: a tensor is used as a size")
        return norm_dim(v)

    def dims(self, xs: Sequence[Any]) -> List[Dim]:
        return [self.dim(x) for x in xs]

    def const(self, value: Number, like: Value) -> Value:
        """A scalar operand as a 0 dim param, typed the way PyTorch types a Python scalar."""
        lt = like.type.dtype
        if isinstance(value, bool):
            dtype = lt
        elif isinstance(value, int):
            dtype = lt if lt != torch.bool else torch.int64
        else:
            dtype = lt if lt.is_floating_point else torch.get_default_dtype()
        name = self.fresh(f"const{self.n_const}")
        self.n_const += 1
        v = Value(name, TensorType(dtype, ()), "param")
        self.params.append(v)
        # The type follows PyTorch's scalar typing, but the stored value keeps the exact Python
        # number (float64), so the float64 reference does not inherit a float32 rounding.
        self.param_data[name] = torch.tensor(value, dtype=torch.float64 if dtype.is_floating_point else dtype)
        return v

    def operand(self, x: Any, like: Value) -> Value:
        v = self.val(x)
        if isinstance(v, Value):
            return v
        if isinstance(v, Number):
            return self.const(v, like)
        raise UnsupportedOpError(f"{_describe(self.node)}: unsupported operand {v!r}")

    def emit(self, op: str, inputs: Sequence[Optional[Value]], **attrs: Any) -> Value:
        node = self.node
        src = _describe(node)
        try:
            t = infer(op, [None if x is None else x.type for x in inputs], attrs)
        except ShapeError as e:
            raise e.with_source(src) from None
        except UnsupportedOpError as e:
            raise UnsupportedOpError(f"{src}: {e}") from None
        out = Value(self.fresh(node.name), t, "op")
        self.nodes.append(Node(op, tuple(inputs), dict(attrs), out, src))
        return out

    # types ------------------------------------------------------------------

    def ir_dim(self, d: Any) -> Dim:
        if isinstance(d, int):
            return d
        if isinstance(d, torch.SymInt):
            expr = d.node.expr
            missing = [s for s in expr.free_symbols if s not in self.sym_map]
            if missing:
                raise UnsupportedOpError(f"{_describe(self.node)}: size depends on symbols {missing} that "
                                         "are not input dims (data dependent shapes are not supported)")
            return norm_dim(_plain_sympy(expr.subs(self.sym_map)))
        raise UnsupportedOpError(f"unexpected size {d!r}")

    def meta_type(self, fake) -> TensorType:
        return TensorType(fake.dtype, tuple(self.ir_dim(d) for d in fake.shape))


# ----------------------------------------------------------------------------- ATen handlers

Handler = Callable[[_Ctx, Dict[str, Any]], Any]
HANDLERS: Dict[str, Handler] = {}


def aten(*names: str):
    def deco(fn):
        for n in names:
            HANDLERS[n] = fn
        return fn
    return deco


def supported_aten_ops() -> List[str]:
    """ATen overloads the importer converts (Python size arithmetic and getitem are also accepted)."""
    return sorted(k for k in HANDLERS if k.startswith("aten."))


def _unsupported(c: _Ctx, why: str):
    raise UnsupportedOpError(f"{_describe(c.node)}: {why}")


@aten("aten.conv2d.default")
def _conv2d(c, a):
    if isinstance(a["padding"], str):
        _unsupported(c, f"padding={a['padding']!r} is not supported, give explicit padding")
    return c.emit("conv2d", [c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"])],
                  stride=list(a["stride"]), padding=list(a["padding"]), dilation=list(a["dilation"]),
                  groups=a["groups"])


@aten("aten.convolution.default")
def _convolution(c, a):
    if a["transposed"]:
        _unsupported(c, "transposed convolution is not supported yet")
    if len(a["stride"]) != 2:
        _unsupported(c, f"only 2D convolution is supported, got {len(a['stride'])}D")
    return c.emit("conv2d", [c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"])],
                  stride=list(a["stride"]), padding=list(a["padding"]), dilation=list(a["dilation"]),
                  groups=a["groups"])


@aten("aten.linear.default")
def _linear(c, a):
    return c.emit("linear", [c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"])])


@aten("aten.mm.default", "aten.bmm.default", "aten.matmul.default")
def _mm(c, a):
    x = c.tensor(a["self"])
    y = c.tensor(a["mat2"] if "mat2" in a else a["other"])
    return c.emit("matmul", [x, y])


@aten("aten.addmm.default")
def _addmm(c, a):
    """beta * input + alpha * (mat1 @ mat2)."""
    m = c.emit("matmul", [c.tensor(a["mat1"]), c.tensor(a["mat2"])])
    if a["alpha"] != 1:
        m = c.emit("mul", [m, c.const(a["alpha"], m)])
    bias = c.tensor(a["self"])
    if a["beta"] != 1:
        bias = c.emit("mul", [bias, c.const(a["beta"], bias)])
    return c.emit("add", [m, bias])


@aten("aten.t.default")
def _t(c, a):
    x = c.tensor(a["self"])
    return x if x.type.rank < 2 else c.emit("permute", [x], dims=[1, 0])


def _bn_train(c, a):
    """Training batch norm as primitive ops with the running statistics as explicit state."""
    from .train_ops import batch_norm_train

    if not c.training:
        _unsupported(c, "batch norm in training mode; import with training=True, or call model.eval() before "
                        "torch.export")
    x = c.tensor(a["input"])
    rm, rv = c.tensor(a["running_mean"]), c.tensor(a["running_var"])
    try:
        y, nm, nv = batch_norm_train(lambda op, ins, **kw: c.emit(op, ins, **kw), c.const, x, c.tensor(a["weight"]),
                                     c.tensor(a["bias"]), rm, rv, a["momentum"], float(a["eps"]))
    except UnsupportedOpError as e:
        _unsupported(c, str(e))
    for v, new in ((rm, nm), (rv, nv)):
        if v is not None and id(v) in c.state_of:
            c.state_out[c.state_of[id(v)]] = new
    return y, nm, nv


@aten("aten._native_batch_norm_legit_functional.default")
def _bn_functional(c, a):
    y, nm, nv = _bn_train(c, a)
    return (y, _Missing("the saved mean of batch norm"), _Missing("the saved rstd of batch norm"), nm, nv)


@aten("aten.batch_norm.default")
def _batch_norm(c, a):
    if a["training"]:
        return _bn_train(c, a)[0]
    return c.emit("batch_norm", [c.tensor(a[k]) for k in ("input", "weight", "bias", "running_mean", "running_var")],
                  eps=float(a["eps"]))


@aten("aten._native_batch_norm_legit_no_training.default")
def _bn_no_training(c, a):
    out = c.emit("batch_norm", [c.tensor(a[k]) for k in ("input", "weight", "bias", "running_mean", "running_var")],
                 eps=float(a["eps"]))
    return (out, _Missing("the saved mean of batch norm"), _Missing("the saved rstd of batch norm"))


@aten("aten.layer_norm.default", "aten.native_layer_norm.default")
def _layer_norm(c, a):
    out = c.emit("layer_norm", [c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"])],
                 normalized_shape=c.dims(a["normalized_shape"]), eps=float(a["eps"]))
    if str(c.node.target) == "aten.native_layer_norm.default":
        return (out, _Missing("the saved mean of layer norm"), _Missing("the saved rstd of layer norm"))
    return out


def _unary(op):
    def fn(c, a):
        attrs = {}
        if op == "gelu":
            attrs["approximate"] = a["approximate"]
        return c.emit(op, [c.tensor(a["self"])], **attrs)
    return fn


for _op in ("relu", "gelu", "hardswish", "hardsigmoid", "silu", "sigmoid", "tanh", "neg", "exp",
            "sqrt", "rsqrt", "logical_not"):
    HANDLERS[f"aten.{_op}.default"] = _unary(_op)
    if _op not in ("logical_not",):
        HANDLERS[f"aten.{_op}_.default"] = _unary(_op)


_CLAMP_STRICT: List[bool] = []


def clamp_bounds_are_strict() -> bool:
    """Whether this torch's clamp with scalar bounds sends gradient 0 at x == min or x == max.

    torch 2.14 changed it from 1 to 0 (the minimum norm subgradient, pytorch #191142); torch 2.11
    passes the bounds. Measured once from torch's own autograd, not read from the version, so the
    imported gradient follows the torch it is checked against."""
    if not _CLAMP_STRICT:
        with torch.inference_mode(False), torch.enable_grad():
            x = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
            torch.clamp(x, 0.0, 1.0).sum().backward()
        got = x.grad.tolist()
        if got not in ([0.0, 0.0], [1.0, 1.0]):
            raise IRError(f"internal error: torch.clamp's gradient at its bounds is {got}, neither 0 nor 1")
        _CLAMP_STRICT.append(got == [0.0, 0.0])
    return _CLAMP_STRICT[0]


def _clamp_strict() -> Dict[str, bool]:
    """The ``strict`` attr for an imported scalar bound clamp; absent when torch passes the bounds,
    so programs imported on torch 2.11 to 2.13 are unchanged."""
    return {"strict": True} if clamp_bounds_are_strict() else {}


@aten("aten.clamp.default", "aten.clamp_.default")
def _clamp(c, a):
    return c.emit("clamp", [c.tensor(a["self"])], min=a["min"], max=a["max"], **_clamp_strict())


@aten("aten.hardtanh.default", "aten.hardtanh_.default")
def _hardtanh(c, a):
    """clamp with ``strict``: torch's hardtanh backward is 0 at the bounds (torch.clamp before 2.14
    passes them).
    Exact zeros at a ReLU6 are common (mobilenet_v2 at its default init has 260 thousand), so the
    gradient differed there before 2026-11-24."""
    return c.emit("clamp", [c.tensor(a["self"])], min=a["min_val"], max=a["max_val"], strict=True)


@aten("aten.relu6.default")
def _relu6(c, a):
    return c.emit("clamp", [c.tensor(a["self"])], min=0.0, max=6.0, strict=True)


def _arith(op, reverse=False):
    def fn(c, a):
        xv, yv = c.val(a["self"]), c.val(a["other"])
        if isinstance(xv, Value):
            x = xv
            y = c.operand(a["other"], x)
        else:  # a Python scalar on the left, e.g. 1.0 - t
            y = c.tensor(a["other"])
            x = c.operand(a["self"], y)
        alpha = a.get("alpha", 1)
        if alpha != 1:
            if reverse:
                x = c.emit("mul", [x, c.const(alpha, x)])
            else:
                y = c.emit("mul", [y, c.const(alpha, y)])
        return c.emit(op, [y, x] if reverse else [x, y])
    return fn


for _ov in ("Tensor", "Scalar"):
    HANDLERS[f"aten.add.{_ov}"] = _arith("add")
    HANDLERS[f"aten.add_.{_ov}"] = _arith("add")
    HANDLERS[f"aten.sub.{_ov}"] = _arith("sub")
    HANDLERS[f"aten.sub_.{_ov}"] = _arith("sub")
    HANDLERS[f"aten.mul.{_ov}"] = _arith("mul")
    HANDLERS[f"aten.mul_.{_ov}"] = _arith("mul")
    HANDLERS[f"aten.div.{_ov}"] = _arith("div")
    HANDLERS[f"aten.div_.{_ov}"] = _arith("div")
    for _cmp in ("eq", "ne", "lt", "le", "gt", "ge"):
        HANDLERS[f"aten.{_cmp}.{_ov}"] = _arith(_cmp)
HANDLERS["aten.logical_and.default"] = _arith("logical_and")
HANDLERS["aten.logical_or.default"] = _arith("logical_or")
HANDLERS["aten.rsub.Scalar"] = _arith("sub", reverse=True)


@aten("aten.where.self")
def _where(c, a):
    return c.emit("where", [c.tensor(a["condition"]), c.tensor(a["self"]), c.tensor(a["other"])])


def _scalar_dtype(value, dtype, c=None):
    if dtype is not None:
        return dtype
    if isinstance(value, bool):
        return torch.bool
    if isinstance(value, int):
        return torch.int64
    return _default_float(c)


def _default_float(c=None):
    """The float dtype a creation op without ``dtype=`` produced when it was traced: the dtype
    torch.export recorded on the node (the default dtype at trace time, which may differ from the
    default dtype at import time), else the current default."""
    fake = c.node.meta.get("val") if c is not None and getattr(c, "node", None) is not None else None
    if isinstance(fake, torch.Tensor) and fake.dtype.is_floating_point:
        return fake.dtype
    return torch.get_default_dtype()


@aten("aten.full.default")
def _full(c, a):
    return c.emit("full", [], shape=c.dims(a["size"]), value=a["fill_value"],
                  dtype=_scalar_dtype(a["fill_value"], a["dtype"], c))


@aten("aten.scalar_tensor.default")
def _scalar_tensor(c, a):
    v = a["s"]
    dtype = _scalar_dtype(v, a["dtype"], c)
    return c.const(v, Value("_", TensorType(dtype, ()), "param"))


@aten("aten.arange.default", "aten.arange.start", "aten.arange.start_step")
def _arange(c, a):
    start, end, step = c.val(a.get("start", 0)), c.val(a["end"]), c.val(a.get("step", 1))
    if isinstance(end, sympy.Basic) and isinstance(start, int) and isinstance(step, int) and step > 0:
        # arange(T) with a symbolic T (positions in a transformer): the length stays symbolic
        dtype = a["dtype"] or torch.int64
        return c.emit("arange", [], start=start, end=norm_dim(end), step=step, dtype=dtype)
    if not all(isinstance(v, (int, float)) for v in (start, end, step)):
        _unsupported(c, "arange with symbolic bounds is not supported yet (only a symbolic end with an int "
                        "start and a positive int step)")
    dtype = a["dtype"]
    if dtype is None:
        dtype = torch.int64 if all(isinstance(v, int) for v in (start, end, step)) else _default_float(c)
    return c.emit("arange", [], start=start, end=end, step=step, dtype=dtype)


@aten("aten.full_like.default")
def _full_like(c, a):
    return c.emit("full_like", [c.tensor(a["self"])], value=a["fill_value"], dtype=a["dtype"])


def _reduce(op):
    def fn(c, a):
        x = c.tensor(a["self"])
        dtype = a.get("dtype")
        if dtype is not None:
            # torch casts the input to ``dtype`` first (coverage refresh: Gemma and SmolVLM pass one)
            if op == "any" or not (dtype.is_floating_point or dtype == torch.int64):
                _unsupported(c, f"{op} with dtype={dtype} is not supported (float dtypes and int64 are)")
            x = _cast_if(c, x, dtype)
        elif op == "sum" and not x.type.dtype.is_floating_point and x.type.dtype != torch.int64:
            x = _cast_if(c, x, torch.int64)      # torch sums bool and small integers as int64
        dim = a.get("dim")
        dims = None if dim is None else ([dim] if isinstance(dim, int) else list(dim))
        return c.emit(op, [x], dims=dims, keepdim=bool(a.get("keepdim", False)))
    return fn


for _name in ("aten.mean.dim", "aten.mean.default"):
    HANDLERS[_name] = _reduce("mean")
for _name in ("aten.sum.dim_IntList", "aten.sum.default"):
    HANDLERS[_name] = _reduce("sum")
for _name in ("aten.any.dim", "aten.any.dims", "aten.any.default"):
    HANDLERS[_name] = _reduce("any")


@aten("aten.softmax.int", "aten._softmax.default")
def _softmax(c, a):
    x = c.tensor(a["self"])
    if a.get("half_to_float"):
        _unsupported(c, "half_to_float softmax is not supported")
    if a.get("dtype") is not None and a["dtype"] != x.type.dtype:
        if not a["dtype"].is_floating_point:
            _unsupported(c, f"softmax with dtype={a['dtype']} is not supported")
        x = _cast_if(c, x, a["dtype"])    # torch casts first (Gemma 2's eager attention: float32)
    return c.emit("softmax", [x], dim=a["dim"])


# aten.scaled_dot_product_attention.default: see _sdpa_gqa in the decoder LLM section below.


@aten("aten.view.default", "aten.reshape.default", "aten._unsafe_view.default")
def _view(c, a):
    return c.emit("reshape", [c.tensor(a["self"])], shape=c.dims(a["size"] if "size" in a else a["shape"]))


@aten("aten.flatten.using_ints")
def _flatten(c, a):
    x = c.tensor(a["self"])
    r = max(x.type.rank, 1)
    s, e = a["start_dim"] % r, a["end_dim"] % r
    if x.type.rank == 0:
        return c.emit("reshape", [x], shape=[1])
    merged = sympy.Integer(1)
    for d in x.type.shape[s:e + 1]:
        merged = merged * sympy.sympify(d)
    shape = list(x.type.shape[:s]) + [norm_dim(merged)] + list(x.type.shape[e + 1:])
    return c.emit("reshape", [x], shape=shape)


@aten("aten.unflatten.int")
def _unflatten(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    shape = list(x.type.shape[:d]) + c.dims(a["sizes"]) + list(x.type.shape[d + 1:])
    return c.emit("reshape", [x], shape=shape)


@aten("aten.unsqueeze.default")
def _unsqueeze(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % (x.type.rank + 1)
    shape = list(x.type.shape)
    shape.insert(d, 1)
    return c.emit("reshape", [x], shape=shape)


@aten("aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default")
def _squeeze(c, a):
    x = c.tensor(a["self"])
    r = x.type.rank
    dim = a.get("dim")
    dims = range(r) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    drop = set()
    for d in dims:
        size = x.type.shape[d % max(r, 1)] if r else 1
        if not isinstance(size, int):
            _unsupported(c, f"squeeze of a symbolic dim ({size}) cannot be decided ahead of time")
        if size == 1:
            drop.add(d % max(r, 1))
    shape = [s for i, s in enumerate(x.type.shape) if i not in drop]
    return x if not drop else c.emit("reshape", [x], shape=shape)


@aten("aten.permute.default")
def _permute(c, a):
    return c.emit("permute", [c.tensor(a["self"])], dims=list(a["dims"]))


@aten("aten.transpose.int")
def _transpose(c, a):
    x = c.tensor(a["self"])
    r = x.type.rank
    dims = list(range(r))
    i, j = a["dim0"] % r, a["dim1"] % r
    dims[i], dims[j] = dims[j], dims[i]
    return c.emit("permute", [x], dims=dims)


@aten("aten.expand.default")
def _expand(c, a):
    return c.emit("expand", [c.tensor(a["self"])], shape=c.dims(a["size"]))


@aten("aten.select.int")
def _select(c, a):
    return c.emit("select", [c.tensor(a["self"])], dim=a["dim"], index=c.dim(a["index"]))


@aten("aten.slice.Tensor")
def _slice(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    start = None if a["start"] is None else c.dim(a["start"])
    end = None if a["end"] is None else c.dim(a["end"])
    if end is not None and dims_equal(end, x.type.shape[d]):
        end = None
    if start is not None and not isinstance(start, int) or end is not None and not isinstance(end, int):
        _unsupported(c, "slice with symbolic bounds is not supported yet")
    return c.emit("slice", [x], dim=d, start=start, end=end, step=a["step"])


@aten("aten.cat.default", "aten.concatenate.default")   # concatenate: Zamba 2 (P60)
def _cat(c, a):
    ts = [c.tensor(t) for t in a["tensors"]]
    # torch.cat skips a one dimensional tensor of size 0 next to tensors of another rank (its legacy rule); the
    # transformers KV cache starts every layer as torch.tensor([]) and cats the first keys onto it
    keep = [t for t in ts if not (t.type.rank == 1 and dims_equal(t.type.shape[0], 0))]
    if keep and len(keep) < len(ts) and any(t.type.rank != 1 for t in keep):
        dtype = ts[0].type.dtype
        for t in ts[1:]:
            dtype = torch.promote_types(dtype, t.type.dtype)
        if len(keep) == 1:
            x = keep[0]
            return x if x.type.dtype == dtype else c.emit("cast", [x], dtype=dtype)
        ts = keep
    # torch.cat promotes mixed dtypes (SAM joins its float points with a padding point of another
    # dtype); the casts make the IR cat homogeneous, so the compiler's layout step takes it
    fake = c.node.meta.get("val") if c.node is not None else None
    want = getattr(fake, "dtype", None)
    if want is not None and any(t.type.dtype != want for t in ts):
        ts = [t if t.type.dtype == want else c.emit("cast", [t], dtype=want) for t in ts]
    return c.emit("cat", ts, dim=a["dim"])


@aten("aten.contiguous.default", "aten.clone.default", "aten.alias.default", "aten.detach.default",
      "aten.lift_fresh_copy.default")
def _identity(c, a):
    return c.tensor(a["self"])


@aten("aten.dropout.default", "aten.dropout_.default")
def _dropout(c, a):
    x = c.tensor(a["input"] if "input" in a else a["self"])
    if a["train"]:
        return _dropout_train(c, x, a["p"])[0]
    return x


@aten("aten.native_dropout.default")
def _native_dropout(c, a):
    x = c.tensor(a["input"])
    if a["train"] is False:
        return (x, c.emit("full", [], shape=list(x.type.shape), value=True, dtype=torch.bool))
    return _dropout_train(c, x, a["p"])


def _dropout_train(c, x, p):
    """Keyed dropout: the mask is a ``random`` bernoulli draw at the site ``<module>/dropout#k``."""
    from .train_ops import dropout

    if not c.training:
        _unsupported(c, "dropout with train=True; import with training=True, or call model.eval() before "
                        "torch.export")
    try:
        return dropout(lambda op, ins, **kw: c.emit(op, ins, **kw), c.const, x, float(p), c.random_key(), [],
                       c.site_name("dropout"))
    except UnsupportedOpError as e:
        _unsupported(c, str(e))


@aten("aten.max_pool2d.default", "aten.max_pool2d_with_indices.default")
def _max_pool2d(c, a):
    out = c.emit("max_pool2d", [c.tensor(a["self"])], kernel_size=list(a["kernel_size"]),
                 stride=list(a["stride"]) or list(a["kernel_size"]), padding=list(a["padding"]),
                 dilation=list(a["dilation"]), ceil_mode=bool(a["ceil_mode"]))
    if str(c.node.target).endswith("with_indices.default"):
        return (out, _Missing("the indices output of max_pool2d"))
    return out


@aten("aten.adaptive_avg_pool2d.default", "aten._adaptive_avg_pool2d.default")
def _adaptive_avg_pool2d(c, a):
    return c.emit("adaptive_avg_pool2d", [c.tensor(a["self"])], output_size=c.dims(a["output_size"]))


@aten("aten.sym_size.int")
def _sym_size(c, a):
    x = c.tensor(a["self"])
    return x.type.shape[a["dim"] % x.type.rank]


_SCALAR_OPS = {
    operator.mul: lambda x, y: x * y,
    operator.add: lambda x, y: x + y,
    operator.sub: lambda x, y: x - y,
    operator.floordiv: lambda x, y: sympy.floor(sympy.sympify(x) / y),
}
for _fn in _SCALAR_OPS:
    HANDLERS[str(_fn)] = None  # marker, handled in _call
# comparisons of Python numbers torch.export keeps next to its assertions (importer_vision.py): only
# of known numbers, so they never decide anything about an input
_SCALAR_PREDICATES = {operator.ge: operator.ge, operator.gt: operator.gt, operator.le: operator.le,
                      operator.lt: operator.lt, operator.eq: operator.eq, operator.ne: operator.ne,
                      operator.and_: operator.and_, operator.or_: operator.or_}
for _fn in _SCALAR_PREDICATES:
    HANDLERS[str(_fn)] = None
HANDLERS[str(operator.getitem)] = None


# ----------------------------------------------------------------------------- ATen handlers added 2026-11-24
# Coverage for every torchvision classifier up to 100M params and Hugging Face style decoders. Many
# lower onto existing IR ops (chunk and split become slices, roll becomes slices plus cat, einsum
# becomes permute, reshape and matmul); the rest map to the new IR ops in shapes.py.


def _reshape_if(c, x: Value, shape: Sequence[Dim]) -> Value:
    shape = [norm_dim(d) for d in shape]
    if len(shape) == x.type.rank and all(dims_equal(a, b) for a, b in zip(shape, x.type.shape)):
        return x
    return c.emit("reshape", [x], shape=shape)


def _permute_if(c, x: Value, dims: Sequence[int]) -> Value:
    dims = list(dims)
    return x if dims == list(range(x.type.rank)) else c.emit("permute", [x], dims=dims)


def _cast_if(c, x: Value, dtype: torch.dtype) -> Value:
    return x if x.type.dtype == dtype else c.emit("cast", [x], dtype=dtype)


def _static_size(c, x: Value, d: int, what: str) -> int:
    size = x.type.shape[d]
    if not isinstance(size, int):
        _unsupported(c, f"{what} along the symbolic dim {size} is not supported (its pieces would have "
                        "data dependent sizes); split along a fixed size dim")
    return size


def _slice_args(c, x: Value, a: Dict[str, Any]) -> Dict[str, Any]:
    """Normalized slice attributes, exactly as the slice handler builds them."""
    d = a["dim"] % x.type.rank
    start = None if a["start"] is None else c.dim(a["start"])
    end = None if a["end"] is None else c.dim(a["end"])
    if end is not None and dims_equal(end, x.type.shape[d]):
        end = None
    if start is not None and not isinstance(start, int) or end is not None and not isinstance(end, int):
        _unsupported(c, "slice with symbolic bounds is not supported yet")
    return dict(dim=d, start=start, end=end, step=a["step"])


for _op in ("cos", "sin", "log", "abs", "reciprocal", "erf", "expm1"):
    HANDLERS[f"aten.{_op}.default"] = _unary(_op)
    HANDLERS[f"aten.{_op}_.default"] = _unary(_op)

for _ov in ("Tensor", "Scalar"):
    HANDLERS[f"aten.fmod.{_ov}"] = _arith("fmod")
    HANDLERS[f"aten.remainder.{_ov}"] = _arith("remainder")
HANDLERS["aten.maximum.default"] = _arith("maximum")
HANDLERS["aten.minimum.default"] = _arith("minimum")


@aten("aten.pow.Tensor_Scalar", "aten.pow.Tensor_Tensor", "aten.pow_.Scalar", "aten.pow_.Tensor")
def _pow(c, a):
    x = c.tensor(a["self"])
    return c.emit("pow", [x, c.operand(a["exponent"], x)])


@aten("aten.pow.Scalar")
def _pow_scalar_base(c, a):
    e = c.tensor(a["exponent"])
    return c.emit("pow", [c.operand(a["self"], e), e])


@aten("aten.clamp_min.default", "aten.clamp_min_.default")
def _clamp_min(c, a):
    return c.emit("clamp", [c.tensor(a["self"])], min=a["min"], max=None, **_clamp_strict())


@aten("aten.clamp_max.default", "aten.clamp_max_.default")
def _clamp_max(c, a):
    return c.emit("clamp", [c.tensor(a["self"])], min=None, max=a["max"], **_clamp_strict())


@aten("aten.clamp_min.Tensor")
def _clamp_min_t(c, a):
    return c.emit("maximum", [c.tensor(a["self"]), c.tensor(a["min"])])


@aten("aten.clamp_max.Tensor")
def _clamp_max_t(c, a):
    return c.emit("minimum", [c.tensor(a["self"]), c.tensor(a["max"])])


# dtype conversion ---------------------------------------------------------------


def _check_plain_copy(c, a):
    layout = a.get("layout")
    if layout is not None and layout != torch.strided:
        _unsupported(c, f"layout {layout} is not supported, only dense tensors")


@aten("aten._to_copy.default", "aten.to.dtype", "aten.to.dtype_layout", "aten.to.device")
def _to_copy(c, a):
    _check_plain_copy(c, a)
    x = c.tensor(a["self"])
    dtype = a.get("dtype")
    return x if dtype is None else _cast_if(c, x, dtype)


@aten("aten.type_as.default")
def _type_as(c, a):
    return _cast_if(c, c.tensor(a["self"]), c.tensor(a["other"]).type.dtype)


# creation ------------------------------------------------------------------------


def _full_of(c, size, value, dtype):
    return c.emit("full", [], shape=c.dims(size), value=value, dtype=_scalar_dtype(value, dtype, c))


@aten("aten.zeros.default")
def _zeros(c, a):
    return _full_of(c, a["size"], 0.0 if a["dtype"] is None else 0, a["dtype"] or _default_float(c))


@aten("aten.ones.default")
def _ones(c, a):
    return _full_of(c, a["size"], 1.0 if a["dtype"] is None else 1, a["dtype"] or _default_float(c))


@aten("aten.new_zeros.default", "aten.new_ones.default", "aten.new_full.default")
def _new_full(c, a):
    x = c.tensor(a["self"])
    value = {"aten.new_zeros.default": 0, "aten.new_ones.default": 1}.get(str(c.node.target), a.get("fill_value"))
    return c.emit("full", [], shape=c.dims(a["size"]), value=value, dtype=a["dtype"] or x.type.dtype)


@aten("aten.scatter.value", "aten.scatter_.value")
def _scatter_value(c, a):
    """x with ``value`` written at ``index`` along dim. Every write is the same value, so repeated
    indices are exact: the positions hit are those where a scatter added count is positive."""
    x, index = c.tensor(a["self"]), c.tensor(a["index"])
    d = a["dim"] % max(x.type.rank, 1)
    ones = c.emit("full", [], shape=list(index.type.shape), value=1.0, dtype=torch.float32)
    zeros = c.emit("full", [], shape=list(x.type.shape), value=0.0, dtype=torch.float32)
    hit = c.emit("gt", [c.emit("scatter_add", [zeros, index, ones], dim=d), c.const(0.0, zeros)])
    return c.emit("where", [hit, c.emit("full_like", [x], value=a["value"], dtype=None), x])


@aten("aten.zeros_like.default", "aten.ones_like.default")
def _zeros_like(c, a):
    value = 0 if str(c.node.target) == "aten.zeros_like.default" else 1
    return c.emit("full_like", [c.tensor(a["self"])], value=value, dtype=a["dtype"])


@aten("aten.fill.Scalar", "aten.fill_.Scalar")
def _fill_scalar(c, a):
    return c.emit("full_like", [c.tensor(a["self"])], value=a["value"], dtype=None)


@aten("aten.zero_.default", "aten.zero.default")
def _zero(c, a):
    return c.emit("full_like", [c.tensor(a["self"])], value=0, dtype=None)


@aten("aten.empty.memory_format", "aten.empty.default", "aten.new_empty.default", "aten.empty_like.default")
def _empty(c, a):
    """``torch.empty`` leaves its memory unwritten; the IR has no unwritten value, so the entries are
    NaN, which is what PyTorch itself writes there under ``torch.use_deterministic_algorithms(True)``
    (``torch.utils.deterministic.fill_uninitialized_memory``, on by default). Code that writes every
    entry before reading it (DeepSeek's MLA ``expand_kv``: ``new_empty``, then two slice copies, which
    import as ``slice_scatter``) imports exactly, and an entry read before it is written gives NaN in
    the result instead of a silent zero. Integer and bool buffers are refused (TS1203): no integer
    value marks an unwritten entry."""
    target = str(c.node.target)
    like = c.tensor(a["self"]) if target in ("aten.new_empty.default", "aten.empty_like.default") else None
    dtype = a.get("dtype") or (like.type.dtype if like is not None else _default_float(c))
    if not dtype.is_floating_point:
        err = UnsupportedOpError(
            f"[TS1203] {_describe(c.node)}: an uninitialized {dtype} buffer (torch.empty) is not imported; "
            "the IR marks unwritten float entries with NaN, and no integer or bool value can mark them. "
            "Create it with torch.zeros or torch.full instead.")
        err.code = "TS1203"
        raise err
    if target == "aten.empty_like.default":
        return c.emit("full_like", [like], value=math.nan, dtype=dtype)
    return c.emit("full", [], shape=c.dims(a["size"]), value=math.nan, dtype=dtype)


def _broadcast_into(c, x: Value, src: Value) -> Value:
    """src converted to x's dtype and broadcast to x's shape: what x holds after x.copy_(src)."""
    src = _cast_if(c, src, x.type.dtype)
    if src.type.rank == x.type.rank and all(dims_equal(p, q) for p, q in zip(src.type.shape, x.type.shape)):
        return src
    return c.emit("expand", [src], shape=list(x.type.shape))


@aten("aten.fill.Tensor", "aten.fill_.Tensor")
def _fill_tensor(c, a):
    x, v = c.tensor(a["self"]), c.tensor(a["value"])
    if v.type.rank != 0:
        _unsupported(c, f"fill with a value of rank {v.type.rank}; the value must be a 0 dim tensor")
    return _broadcast_into(c, x, v)


@aten("aten.copy.default", "aten.copy_.default")
def _copy(c, a):
    return _broadcast_into(c, c.tensor(a["self"]), c.tensor(a["src"]))


@aten("aten.masked_fill.Scalar", "aten.masked_fill_.Scalar")
def _masked_fill(c, a):
    x, mask, value = c.tensor(a["self"]), c.tensor(a["mask"]), a["value"]
    if isinstance(value, float) and not x.type.dtype.is_floating_point:
        _unsupported(c, f"masked_fill of a {x.type.dtype} tensor with the float {value}")
    return c.emit("where", [mask, c.const(value, x), x])


@aten("aten.masked_fill.Tensor", "aten.masked_fill_.Tensor")
def _masked_fill_t(c, a):
    x, mask, v = c.tensor(a["self"]), c.tensor(a["mask"]), c.tensor(a["value"])
    if v.type.rank != 0:
        _unsupported(c, "masked_fill with a value that is not a 0 dim tensor")
    return c.emit("where", [mask, _cast_if(c, v, x.type.dtype), x])


@aten("aten.where.ScalarSelf")
def _where_scalar_self(c, a):
    other = c.tensor(a["other"])
    return c.emit("where", [c.tensor(a["condition"]), c.const(a["self"], other), other])


@aten("aten.where.ScalarOther")
def _where_scalar_other(c, a):
    x = c.tensor(a["self"])
    return c.emit("where", [c.tensor(a["condition"]), x, c.const(a["other"], x)])


@aten("aten.where.Scalar")
def _where_scalars(c, a):
    s, o = a["self"], a["other"]
    dtype = torch.promote_types(_scalar_dtype(s, None), _scalar_dtype(o, None))
    like = Value("_", TensorType(dtype, ()), "param")
    return c.emit("where", [c.tensor(a["condition"]), c.const(s, like), c.const(o, like)])


# indexing ------------------------------------------------------------------------


@aten("aten.index.Tensor")
def _index(c, a):
    return c.emit("index", [c.tensor(a["self"])] + [c.tensor(i) for i in a["indices"]])


@aten("aten.index_select.default")
def _index_select(c, a):
    return c.emit("index_select", [c.tensor(a["self"]), c.tensor(a["index"])], dim=a["dim"])


@aten("aten.gather.default")
def _gather(c, a):
    return c.emit("gather", [c.tensor(a["self"]), c.tensor(a["index"])], dim=a["dim"])


@aten("aten.embedding.default")
def _embedding(c, a):
    """weight[indices]. padding_idx and scale_grad_by_freq change only gradients, and the program says
    so with ``frozen`` (the value, no derivative): rows at padding_idx are read from a frozen weight,
    so they send nothing back (as torch); with scale_grad_by_freq a row read k times in the batch
    gets 1/k of each of its cotangents, written frozen(r) + (r - frozen(r)) / k, which is r exactly."""
    if a.get("sparse"):
        _unsupported(c, "sparse embedding gradients are not supported")
    w, ids = c.tensor(a["weight"]), c.tensor(a["indices"])
    rows = c.emit("index", [w, ids])
    pad = a.get("padding_idx", -1)
    pad = None if pad is None or pad < 0 else pad
    if pad is None and not a.get("scale_grad_by_freq"):
        return rows
    held = c.emit("frozen", [rows])
    if a.get("scale_grad_by_freq"):
        n = 1
        for d in ids.type.shape:
            n = n * d
        flat = c.emit("reshape", [ids], shape=[norm_dim(n)])
        ones = c.emit("full", [], shape=[norm_dim(n)], value=1.0, dtype=w.type.dtype)
        zeros = c.emit("full", [], shape=[w.type.shape[0]], value=0.0, dtype=w.type.dtype)
        counts = c.emit("index", [c.emit("scatter_add", [zeros, flat, ones], dim=0), ids])
        view = list(ids.type.shape) + [1]
        delta = c.emit("div", [c.emit("sub", [rows, held]), c.emit("reshape", [counts], shape=view)])
        rows = c.emit("add", [held, delta])
        held = c.emit("frozen", [rows])
    if pad is None:
        return rows
    keep = c.emit("reshape", [c.emit("ne", [ids, c.const(pad, ids)])], shape=list(ids.type.shape) + [1])
    return c.emit("where", [keep, rows, held])


@aten("aten.slice_scatter.default")
def _slice_scatter(c, a):
    x = c.tensor(a["self"])
    return c.emit("slice_scatter", [x, c.tensor(a["src"])], **_slice_args(c, x, a))


@aten("aten.select_scatter.default")
def _select_scatter(c, a):
    x = c.tensor(a["self"])
    return c.emit("select_scatter", [x, c.tensor(a["src"])], dim=a["dim"] % x.type.rank, index=c.dim(a["index"]))


# shape ---------------------------------------------------------------------------


def _pieces(c, x: Value, d: int, sizes: Sequence[int]):
    out, start = [], 0
    for s in sizes:
        out.append(c.emit("slice", [x], dim=d, start=start, end=start + s, step=1))
        start += s
    return tuple(out)


@aten("aten.chunk.default")
def _chunk(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    if a["chunks"] == 1:
        return (x,)
    n = _static_size(c, x, d, "chunk")
    each = -(-n // a["chunks"])
    sizes = [min(each, n - i) for i in range(0, n, each)] if n else [0]
    return _pieces(c, x, d, sizes)


@aten("aten.split.Tensor")
def _split(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    n = _static_size(c, x, d, "split")
    s = a["split_size"]
    sizes = [min(s, n - i) for i in range(0, n, s)] if n else [0]
    return _pieces(c, x, d, sizes)


@aten("aten.split_with_sizes.default", "aten.split.sizes")
def _split_with_sizes(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    sizes = list(a["split_sizes"] if "split_sizes" in a else a["split_size"])
    n = _static_size(c, x, d, "split")
    if sum(sizes) != n:
        _unsupported(c, f"split sizes {sizes} do not add up to the dim size {n}")
    return _pieces(c, x, d, sizes)


@aten("aten.unbind.int")
def _unbind(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    n = _static_size(c, x, d, "unbind")
    return tuple(c.emit("select", [x], dim=d, index=i) for i in range(n))


@aten("aten.stack.default")
def _stack(c, a):
    ts = [c.tensor(t) for t in a["tensors"]]
    d = a["dim"] % (ts[0].type.rank + 1)
    ups = []
    for t in ts:
        shape = list(t.type.shape)
        shape.insert(d, 1)
        ups.append(c.emit("reshape", [t], shape=shape))
    return c.emit("cat", ups, dim=d)


@aten("aten.swapaxes.default", "aten.swapdims.default")
def _swapaxes(c, a):
    x = c.tensor(a["self"])
    r = x.type.rank
    i = (a["axis0"] if "axis0" in a else a["dim0"]) % r
    j = (a["axis1"] if "axis1" in a else a["dim1"]) % r
    dims = list(range(r))
    dims[i], dims[j] = dims[j], dims[i]
    return _permute_if(c, x, dims)


@aten("aten.movedim.int")
def _movedim(c, a):
    x = c.tensor(a["self"])
    r = x.type.rank
    s, t = a["source"] % r, a["destination"] % r
    dims = [i for i in range(r) if i != s]
    dims.insert(t, s)
    return _permute_if(c, x, dims)


@aten("aten.expand_as.default")
def _expand_as(c, a):
    return c.emit("expand", [c.tensor(a["self"])], shape=list(c.tensor(a["other"]).type.shape))


@aten("aten.view_as.default", "aten.reshape_as.default")
def _view_as(c, a):
    return _reshape_if(c, c.tensor(a["self"]), c.tensor(a["other"]).type.shape)


@aten("aten.as_strided.default")
def _as_strided(c, a):
    """Only the case that is a plain reshape: a contiguous input, offset 0, and the contiguous strides
    of the new size (strides of size 1 dims do not matter). Anything else reads memory the IR has no
    notion of and is refused."""
    x = c.tensor(a["self"])
    size, stride = c.dims(a["size"]), c.dims(a["stride"])
    offset = a.get("storage_offset")
    fake = a["self"].meta.get("val") if isinstance(a["self"], torch.fx.Node) else None
    try:
        contiguous = fake is not None and bool(fake.is_contiguous())
    except Exception:
        contiguous = False
    want, acc = [], sympy.Integer(1)
    for d in reversed(size):
        want.insert(0, norm_dim(acc))
        acc = acc * sympy.sympify(d)
    same = all(dims_equal(d, 1) or dims_equal(s, w) for d, s, w in zip(size, stride, want))
    if not (contiguous and (offset in (None, 0)) and same):
        _unsupported(c, f"as_strided(size={size}, stride={stride}, offset={offset}) is not a plain reshape of "
                        "a contiguous tensor; general strided views are not supported")
    return _reshape_if(c, x, size)


@aten("aten.pad.default")
def _pad(c, a):
    mode = a.get("mode", "constant")
    if mode != "constant":
        _unsupported(c, f"pad mode {mode!r} is not supported yet, only constant padding")
    v = a.get("value")
    return c.emit("pad", [c.tensor(a["self"])], pads=c.dims(a["pad"]), value=0.0 if v is None else v)


@aten("aten.constant_pad_nd.default")
def _constant_pad_nd(c, a):
    return c.emit("pad", [c.tensor(a["self"])], pads=c.dims(a["pad"]), value=a["value"])


@aten("aten.roll.default")
def _roll(c, a):
    """torch.roll: out[i] = x[(i - shift) mod n], i.e. cat(x[n - s:], x[:n - s]) along the dim."""
    x = c.tensor(a["self"])
    shifts, dims = list(a["shifts"]), list(a["dims"])
    if not dims:
        if len(shifts) != 1:
            _unsupported(c, "roll without dims takes exactly one shift")
        shape = x.type.shape
        flat = _reshape_if(c, x, [norm_dim(sympy.Mul(*[sympy.sympify(d) for d in shape]))])
        rolled = _roll_dim(c, flat, shifts[0], 0)
        return _reshape_if(c, rolled, shape)
    if len(shifts) != len(dims):
        _unsupported(c, f"roll with {len(shifts)} shifts for {len(dims)} dims")
    for s, d in zip(shifts, dims):
        x = _roll_dim(c, x, s, d % x.type.rank)
    return x


def _roll_dim(c, x: Value, shift: int, d: int) -> Value:
    n = _static_size(c, x, d, "roll")
    s = shift % n if n else 0
    if not s:
        return x
    tail = c.emit("slice", [x], dim=d, start=n - s, end=None, step=1)
    head = c.emit("slice", [x], dim=d, start=0, end=n - s, step=1)
    return c.emit("cat", [tail, head], dim=d)


@aten("aten.outer.default", "aten.ger.default")
def _outer(c, a):
    x, y = c.tensor(a["self"]), c.tensor(a["vec2"])
    if x.type.rank != 1 or y.type.rank != 1:
        _unsupported(c, "outer needs two vectors")
    return c.emit("mul", [_reshape_if(c, x, [x.type.shape[0], 1]), _reshape_if(c, y, [1, y.type.shape[0]])])


@aten("aten.einsum.default")
def _einsum(c, a):
    """Lowered to sum, permute, reshape and matmul, contracting the operands left to right."""
    eq = a["equation"].replace(" ", "")
    ops = [c.tensor(t) for t in a["tensors"]]
    if "..." in eq:
        _unsupported(c, f"einsum with an ellipsis ({a['equation']!r}) is not supported yet")
    if "->" in eq:
        lhs, out = eq.split("->")
    else:
        lhs = eq
        letters = lhs.replace(",", "")
        out = "".join(sorted(ch for ch in set(letters) if letters.count(ch) == 1))
    terms = lhs.split(",")
    if len(terms) != len(ops):
        _unsupported(c, f"einsum equation {a['equation']!r} names {len(terms)} operands, got {len(ops)}")
    sizes: Dict[str, Dim] = {}
    for term, t in zip(terms, ops):
        if len(set(term)) != len(term):
            _unsupported(c, f"einsum with a repeated index in one operand ({term!r}, a diagonal) is not supported")
        if len(term) != t.type.rank:
            _unsupported(c, f"einsum term {term!r} does not match an operand of rank {t.type.rank}")
        for ch, d in zip(term, t.type.shape):
            if ch in sizes and not dims_equal(sizes[ch], d):
                _unsupported(c, f"einsum index {ch} has size {sizes[ch]} in one operand and {d} in another "
                                "(broadcasting a size 1 index is not supported)")
            sizes[ch] = d
    if len(set(out)) != len(out) or any(ch not in sizes for ch in out):
        _unsupported(c, f"einsum output {out!r} is not valid for {a['equation']!r}")

    def presum(x, term, keep):
        drop = [i for i, ch in enumerate(term) if ch not in keep]
        if drop:
            x = c.emit("sum", [x], dims=drop, keepdim=False)
            term = "".join(ch for ch in term if ch in keep)
        return x, term

    def prod(chs):
        return norm_dim(sympy.Mul(*[sympy.sympify(sizes[ch]) for ch in chs]))

    later = lambda k: set(out).union(*[set(t) for t in terms[k + 1:]])  # noqa: E731
    cur, cur_t = presum(ops[0], terms[0], later(0) | set("".join(terms[1:])))
    for k in range(1, len(ops)):
        keep = later(k)
        cur, cur_t = presum(cur, cur_t, keep | set(terms[k]))
        nxt, nxt_t = presum(ops[k], terms[k], keep | set(cur_t))
        batch = [ch for ch in cur_t if ch in nxt_t and ch in keep]
        contr = [ch for ch in cur_t if ch in nxt_t and ch not in keep]
        af = [ch for ch in cur_t if ch not in nxt_t]
        bf = [ch for ch in nxt_t if ch not in cur_t]
        A = _permute_if(c, cur, [cur_t.index(ch) for ch in batch + af + contr])
        A = _reshape_if(c, A, [sizes[ch] for ch in batch] + [prod(af), prod(contr)])
        B = _permute_if(c, nxt, [nxt_t.index(ch) for ch in batch + contr + bf])
        B = _reshape_if(c, B, [sizes[ch] for ch in batch] + [prod(contr), prod(bf)])
        m = c.emit("matmul", [A, B])
        cur_t = "".join(batch + af + bf)
        cur = _reshape_if(c, m, [sizes[ch] for ch in cur_t])
    cur, cur_t = presum(cur, cur_t, set(out))
    return _permute_if(c, cur, [cur_t.index(ch) for ch in out])


# reductions, cumulative, masks ------------------------------------------------------------


@aten("aten.cumsum.default")
def _cumsum(c, a):
    x = c.tensor(a["self"])
    dtype = a.get("dtype")
    if dtype is not None:
        if not (dtype.is_floating_point or dtype == torch.int64):
            _unsupported(c, f"cumsum with dtype={dtype} is not supported (float dtypes and int64 are)")
        x = _cast_if(c, x, dtype)
    return c.emit("cumsum", [x], dim=a["dim"])


@aten("aten.triu.default")
def _triu(c, a):
    return c.emit("triu", [c.tensor(a["self"])], diagonal=a["diagonal"])


@aten("aten.tril.default")
def _tril(c, a):
    return c.emit("tril", [c.tensor(a["self"])], diagonal=a["diagonal"])


@aten("aten.linalg_vector_norm.default", "aten.norm.ScalarOpt_dim")
def _vector_norm(c, a):
    if a.get("dtype") is not None:
        _unsupported(c, "vector norm with an explicit dtype is not supported")
    ord_ = a["ord"] if "ord" in a else a["p"]
    ord_ = 2.0 if ord_ is None else float(ord_)
    dim = a.get("dim")
    dims = None if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    return c.emit("vector_norm", [c.tensor(a["self"])], ord=ord_, dims=dims, keepdim=bool(a.get("keepdim", False)))


# normalization ----------------------------------------------------------------------------


@aten("aten.group_norm.default")
def _group_norm(c, a):
    return c.emit("group_norm", [c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"])],
                  num_groups=a["num_groups"], eps=float(a["eps"]))


@aten("aten.native_group_norm.default")
def _native_group_norm(c, a):
    out = c.emit("group_norm", [c.tensor(a["input"]), c.tensor(a["weight"]), c.tensor(a["bias"])],
                 num_groups=a["group"], eps=float(a["eps"]))
    return (out, _Missing("the saved mean of group norm"), _Missing("the saved rstd of group norm"))


@aten("aten.instance_norm.default")
def _instance_norm(c, a):
    """With input statistics this is group norm with one group per channel; with running statistics
    it is the inference form of batch norm."""
    x = c.tensor(a["input"])
    if a["use_input_stats"]:
        ch = x.type.shape[1] if x.type.rank >= 2 else None
        if not isinstance(ch, int):
            _unsupported(c, "instance norm over a symbolic channel dim is not supported")
        return c.emit("group_norm", [x, c.tensor(a["weight"]), c.tensor(a["bias"])], num_groups=ch,
                      eps=float(a["eps"]))
    return c.emit("batch_norm", [x] + [c.tensor(a[k]) for k in ("weight", "bias", "running_mean", "running_var")],
                  eps=float(a["eps"]))


@aten("aten.rms_norm.default")
def _rms_norm(c, a):
    x = c.tensor(a["input"])
    eps = a.get("eps")
    if eps is None:  # PyTorch's default: the machine epsilon of the input dtype
        eps = torch.finfo(x.type.dtype if x.type.dtype.is_floating_point else torch.get_default_dtype()).eps
    return c.emit("rms_norm", [x, c.tensor(a["weight"])], normalized_shape=c.dims(a["normalized_shape"]),
                  eps=float(eps))


# pooling ----------------------------------------------------------------------------------


@aten("aten.avg_pool2d.default")
def _avg_pool2d(c, a):
    k = list(a["kernel_size"])
    return c.emit("avg_pool2d", [c.tensor(a["self"])], kernel_size=k, stride=list(a["stride"]) or k,
                  padding=list(a["padding"]), ceil_mode=bool(a["ceil_mode"]),
                  count_include_pad=bool(a["count_include_pad"]), divisor_override=a["divisor_override"])


@aten("aten.adaptive_max_pool2d.default")
def _adaptive_max_pool2d(c, a):
    out = c.emit("adaptive_max_pool2d", [c.tensor(a["self"])], output_size=c.dims(a["output_size"]))
    return (out, _Missing("the indices output of adaptive_max_pool2d"))


@aten("aten.adaptive_avg_pool1d.default")
def _adaptive_avg_pool1d(c, a):
    x = c.tensor(a["self"])
    (n,) = c.dims(a["output_size"])
    shape = list(x.type.shape)
    y = c.emit("reshape", [x], shape=shape[:-1] + [1, shape[-1]])
    y = c.emit("adaptive_avg_pool2d", [y], output_size=[1, n])
    return c.emit("reshape", [y], shape=shape[:-1] + [n])


@aten("aten._native_batch_norm_legit.no_stats")
def _bn_batch_stats(c, a):
    """Normalization with the statistics of the input itself and no running statistics, as
    instance norm exports (input reshaped to [1, N*C, ...]): per channel, over every other dim,
    (x - mean) / sqrt(biased variance + eps) * weight + bias. Nothing is mutated, so it is pure."""
    if not a["training"]:
        _unsupported(c, "batch norm without running statistics must use input statistics (training=True)")
    x = c.tensor(a["input"])
    if x.type.rank < 2:
        _unsupported(c, "batch norm input must be [N, C, ...]")
    dims = [0] + list(range(2, x.type.rank))
    mu = c.emit("mean", [x], dims=dims, keepdim=True)
    d = c.emit("sub", [x, mu])
    var = c.emit("mean", [c.emit("mul", [d, d])], dims=dims, keepdim=True)
    y = c.emit("mul", [d, c.emit("rsqrt", [c.emit("add", [var, c.const(float(a["eps"]), var)])])])
    view = [x.type.shape[1]] + [1] * (x.type.rank - 2)
    w, b = c.tensor(a["weight"]), c.tensor(a["bias"])
    if w is not None:
        y = c.emit("mul", [y, _reshape_if(c, w, view)])
    if b is not None:
        y = c.emit("add", [y, _reshape_if(c, b, view)])
    return (y, _Missing("the saved mean of batch norm"), _Missing("the saved rstd of batch norm"))


@aten("aten.repeat.default")
def _repeat(c, a):
    """x.repeat(r0, r1, ...): tile. Lowered to reshape, expand, reshape; repeats may be symbolic."""
    x = c.tensor(a["self"])
    reps = c.dims(a["repeats"])
    if len(reps) < x.type.rank:
        _unsupported(c, f"repeat with {len(reps)} repeats for a rank {x.type.rank} tensor")
    shape = [1] * (len(reps) - x.type.rank) + list(x.type.shape)
    inter, big, out = [], [], []
    for r, s in zip(reps, shape):
        inter += [1, s]
        big += [r, s]
        out.append(norm_dim(sympy.sympify(r) * sympy.sympify(s)))
    y = _reshape_if(c, x, inter)
    if not all(dims_equal(p, q) for p, q in zip(inter, big)):
        y = c.emit("expand", [y], shape=big)
    return _reshape_if(c, y, out)


@aten("aten._assert_tensor_metadata.default")
def _assert_tensor_metadata(c, a):
    """A run time assertion torch.export inserts after dtype conversions. Checked here, at import."""
    x = c.tensor(a["a"])
    if a.get("dtype") is not None and a["dtype"] != x.type.dtype:
        raise IRError(f"internal error: {_describe(c.node)} asserts dtype {a['dtype']} but the IR has {x.type}")
    size = a.get("size")
    if size is not None:
        size = c.dims(size)
        if len(size) != x.type.rank or not all(dims_equal(p, q) for p, q in zip(size, x.type.shape)):
            raise IRError(f"internal error: {_describe(c.node)} asserts size {size} but the IR has {x.type}")
    return None



# ----------------------------------------------------------------------------- driver

# in place: a name ending in "_" (add_), but of the dunder names only the __i...__ ones; aten.__and__
# and aten.__or__ (a & b, a | b) return a new tensor
_INPLACE = re.compile(r"^aten\.(?:(?!__)\w+_|__i(?:and|or|xor|lshift|rshift)__)\.\w+$")


def _target_name(node: torch.fx.Node) -> str:
    return str(node.target)


def _bind(node: torch.fx.Node) -> Dict[str, Any]:
    schema = node.target._schema
    out: Dict[str, Any] = {}
    pos = list(node.args)
    for i, arg in enumerate(schema.arguments):
        if not arg.kwarg_only and i < len(pos):
            out[arg.name] = pos[i]
        elif arg.name in node.kwargs:
            out[arg.name] = node.kwargs[arg.name]
        elif arg.has_default_value():
            out[arg.name] = arg.default_value
        else:
            raise UnsupportedOpError(f"{_describe(node)}: missing argument {arg.name}")
    return out


def _check_inplace(node: torch.fx.Node, order: Dict[str, int]):
    """An in place ATen op becomes a pure op. That is only valid if nobody reads the old value later."""
    target = node.args[0]
    if not isinstance(target, torch.fx.Node):
        return
    if target.op == "placeholder":
        raise UnsupportedOpError(f"{_describe(node)} mutates the program input or weight %{target.name}; "
                                 "the IR has value semantics and cannot express that (a model in training mode does this: "
                                 "import with training=True to make its buffers explicit state, or call "
                                 "model.eval() before torch.export)")
    me = order[node.name]
    for user in target.users:
        if user is not node and order[user.name] > me:
            raise UnsupportedOpError(f"{_describe(node)} mutates %{target.name}, which %{user.name} reads "
                                     "afterwards; the IR has value semantics and cannot express that")
    # Added 2026-11-24: the change is also visible through every other view of the same storage.
    root = _alias_root(target)
    if root.op == "placeholder" and root is not target:
        raise UnsupportedOpError(f"{_describe(node)} mutates a view of the program input or weight %{root.name}; "
                                 "the IR has value semantics and cannot express that")
    _check_alias_family(node, order, root, skip={target.name})


# ----------------------------------------------------------------------------- in place through views
#
# PyTorch code like ``mask = x.new_zeros(H, W); mask[a:b, c:d] = k`` exports as slice nodes followed
# by an in place ``fill_`` on the last slice. The IR has value semantics, so such a change is
# rewritten as a pure op on the slice followed by ``slice_scatter`` (or ``select_scatter``) back
# into every tensor of the chain, and later readers of the base tensor see the new value. Anything
# that cannot be rewritten exactly (a view read again after the change, a base that is itself a
# view or a program input) is refused.

_VIEW_CHAIN = {"aten.slice.Tensor", "aten.select.int"}
_ALIAS_OPS = {
    "aten.view.default", "aten._unsafe_view.default", "aten.reshape.default", "aten.permute.default",
    "aten.transpose.int", "aten.t.default", "aten.expand.default", "aten.unsqueeze.default",
    "aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default", "aten.flatten.using_ints",
    "aten.unflatten.int", "aten.alias.default", "aten.detach.default", "aten.contiguous.default",
    "aten.swapaxes.default", "aten.swapdims.default", "aten.movedim.int", "aten.expand_as.default",
    "aten.view_as.default", "aten.reshape_as.default", "aten.as_strided.default", "aten.narrow.default",
    "aten.diagonal.default", "aten.unfold.default", "aten.to.dtype", "aten.to.dtype_layout",
    "aten.to.device", "aten.type_as.default",
} | _VIEW_CHAIN
_MULTI_VIEW_OPS = {"aten.split.Tensor", "aten.split.sizes", "aten.split_with_sizes.default", "aten.chunk.default",
                   "aten.unbind.int"}


def _alias_source(n: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The tensor whose storage ``n`` shares, or None when ``n`` owns fresh storage."""
    if n.op != "call_function" or not n.args or not isinstance(n.args[0], torch.fx.Node):
        return None
    if _target_name(n) in _ALIAS_OPS or _INPLACE.match(_target_name(n)):
        return n.args[0]
    if n.target is operator.getitem:
        src = n.args[0]
        if _target_name(src) in _MULTI_VIEW_OPS and isinstance(src.args[0], torch.fx.Node):
            return src.args[0]
    return None


def _alias_root(n: torch.fx.Node) -> torch.fx.Node:
    while True:
        src = _alias_source(n)
        if src is None:
            return n
        n = src


def _alias_users(n: torch.fx.Node):
    for u in n.users:
        if _alias_source(u) is n:
            yield u
        elif _target_name(u) in _MULTI_VIEW_OPS and u.args and u.args[0] is n:
            for g in u.users:
                if _alias_source(g) is n:
                    yield g


def _check_alias_family(node: torch.fx.Node, order: Dict[str, int], root: torch.fx.Node, skip: set):
    """Refuse when a tensor sharing storage with the mutated one was created before the change and
    is read after it: in PyTorch that reader sees the change, in a pure program it would not."""
    me = order[node.name]
    seen, todo = set(), [root]
    while todo:
        m = todo.pop()
        if m.name in seen:
            continue
        seen.add(m.name)
        todo.extend(_alias_users(m))
        if m is node or m.name in skip or order[m.name] > me:
            continue
        for u in m.users:
            if u is not node and order[u.name] > me:
                raise UnsupportedOpError(
                    f"{_describe(node)} changes the storage that %{m.name} shares, and %{u.name} reads "
                    f"%{m.name} afterwards; the IR has value semantics and cannot express that")


def _plan_view_mutation(node: torch.fx.Node, order: Dict[str, int], last_mut: Dict[str, List[Tuple[int, str]]]):
    chain = []
    t = node.args[0]
    while isinstance(t, torch.fx.Node) and _target_name(t) in _VIEW_CHAIN:
        chain.append(t)
        t = t.args[0]
    base = t
    if not isinstance(base, torch.fx.Node):
        raise UnsupportedOpError(f"{_describe(node)}: cannot find the tensor this view belongs to")
    if base.op == "placeholder":
        raise UnsupportedOpError(f"{_describe(node)} mutates a slice of the program input or weight %{base.name}; "
                                 "the IR has value semantics and cannot express that (a model in training mode "
                                 "does this: call model.eval() before torch.export)")
    root = base
    if _alias_source(base) is not None:
        # a base that only renames a fresh tensor (Gemma 3n: ``x = mul(..); x += y;
        # x.contiguous().type_as(z)``, then ``x[1:] += w``): same values, shape and dtype all the
        # way up, so the base can stand for the fresh tensor; anything up the chain read after the
        # change is refused by the family check below, rooted at the fresh tensor
        while _alias_source(root) is not None and _is_renaming(root):
            root = _alias_source(root)
        if _alias_source(root) is not None:
            raise UnsupportedOpError(f"{_describe(node)} mutates a slice of %{base.name}, which is itself a view of "
                                     f"%{_alias_source(base).name}; only slices of a fresh tensor can be rewritten")
    # a view taken before an in place change of its base is stale, unless every such change was
    # written through this same view: _scatter_back then updated the view's value too (Qwen3.5's
    # rotary embedding writes two slices of ``freq[0]`` one after the other, P60)
    top = chain[-1].name
    stale = [via for pos, via in last_mut.get(base.name, []) if pos > order[top] and via != top]
    if stale:
        raise UnsupportedOpError(f"{_describe(node)} mutates the view %{top}, which was taken before an "
                                 f"earlier in place change of %{base.name} through %{stale[-1]}; take the view "
                                 "again after that change")
    _check_alias_family(node, order, root, skip={v.name for v in chain} | {base.name})
    return chain, base


_RENAMING_OPS = {"aten.contiguous.default", "aten.type_as.default", "aten.alias.default", "aten.detach.default",
                 "aten.to.dtype", "aten.to.dtype_layout", "aten.to.device"}


def _is_renaming(n: torch.fx.Node) -> bool:
    """``n`` holds exactly its source's values with the same shape and dtype: an identity alias
    (contiguous, a type or device change that changes nothing) or an in place op, whose result is
    its first argument."""
    name = _target_name(n)
    if _INPLACE.match(name):
        return True
    if name not in _RENAMING_OPS:
        return False
    out, src = n.meta.get("val"), n.args[0].meta.get("val") if isinstance(n.args[0], torch.fx.Node) else None
    return (isinstance(out, torch.Tensor) and isinstance(src, torch.Tensor) and out.dtype == src.dtype
            and tuple(out.shape) == tuple(src.shape))


def _scatter_back(c: "_Ctx", node: torch.fx.Node, chain, base: torch.fx.Node, result: Value) -> Value:
    """``result`` is what the view holds after the in place op. Write it back up the chain."""
    tv = c.val(chain[0])
    if not isinstance(result, Value) or result.type.rank != tv.type.rank or not all(
            dims_equal(a, b) for a, b in zip(result.type.shape, tv.type.shape)):
        raise UnsupportedOpError(f"{_describe(node)}: the in place result does not have the view's shape")
    result = _cast_if(c, result, tv.type.dtype)  # an in place op keeps the target's dtype
    cur = result
    for v in chain:
        parent = c.val(v.args[0])
        c.env[v.name] = cur
        a = _bind(v)
        if _target_name(v) == "aten.slice.Tensor":
            cur = c.emit("slice_scatter", [parent, cur], **_slice_args(c, parent, a))
        else:
            cur = c.emit("select_scatter", [parent, cur], dim=a["dim"] % parent.type.rank, index=c.dim(a["index"]))
    c.env[base.name] = cur
    return result


def _finite_int(x) -> Optional[int]:
    """An integer bound, or None for an infinite one. torch marks an unbounded Dim with its own
    ``int_oo``, whose ``is_finite`` is not False, so infinity is tested by value here."""
    if x is None:
        return None
    try:
        from torch.utils._sympy.numbers import int_oo
        if x == int_oo or x == -int_oo:
            return None
    except ImportError:
        pass
    if x in (sympy.oo, -sympy.oo) or getattr(x, "is_infinite", False):
        return None
    return int(x)


def _range_of(vr) -> Tuple[int, Optional[int]]:
    """(min, max) of a symbol from torch.export's range constraints; max None means unbounded."""
    if vr is None:
        return 1, None
    lo = _finite_int(vr.lower)
    return (1 if lo is None else max(lo, 1)), _finite_int(vr.upper)


def _convert_graph(c: "_Ctx", graph: torch.fx.Graph, owner: torch.nn.Module,
                   output_specs=None) -> List[Value]:
    """Convert every node of one FX graph (the program, or a control flow subgraph) into ``c``.
    ``owner`` is the module that holds the graph's ``get_attr`` targets (subgraphs). ``output_specs``
    (the program graph only): outputs that are BUFFER_MUTATION go to the training state
    (``c.state_out``), not to the returned user outputs."""
    order = {n.name: i for i, n in enumerate(graph.nodes)}
    # base tensor -> (position, top view) of each in place change through a view
    last_mut: Dict[str, List[Tuple[int, str]]] = {}
    outputs: List[Value] = []
    for node in graph.nodes:
        c.node = node
        if node.op == "output":
            outs = node.args[0]
            outs = outs if isinstance(outs, (list, tuple)) else [outs]
            # a None result (a dict entry the model leaves empty) is no value; the exporter keeps it in
            # the output structure
            specs = list(output_specs) if output_specs is not None else [None] * len(outs)
            outputs = []
            for o, spec in zip(outs, specs):
                if o is None:
                    continue
                if spec is not None and spec.kind.name == "BUFFER_MUTATION":
                    c.state_out[spec.target] = c.tensor(o)
                else:
                    outputs.append(c.tensor(o))
            continue
        if node.op == "get_attr":
            c.env[node.name] = _get_attr(owner, node.target)
            continue
        if node.op != "call_function":
            continue
        if node.target in HOP_HANDLERS:
            c.env[node.name] = HOP_HANDLERS[node.target](c, node)
            continue
        tname = _target_name(node)
        if node.target is operator.getitem:
            seq, idx = node.args
            c.env[node.name] = c.env[seq.name][idx]
            continue
        if node.target in (*_SCALAR_OPS, *_SCALAR_PREDICATES) and any(
                isinstance(v, torch.fx.Node) and c.env.get(v.name, 0) is None for v in node.args):
            c.env[node.name] = None     # arithmetic on a read only assertions use (importer_vision.py)
            continue
        if node.target in _SCALAR_OPS:
            x, y = (c.dim(v) for v in node.args)
            c.env[node.name] = norm_dim(_SCALAR_OPS[node.target](x, y))
            continue
        if node.target in _SCALAR_PREDICATES:
            vals = [c.val(v) for v in node.args]
            if not all(isinstance(v, (bool, int, float)) for v in vals):
                raise UnsupportedOpError(f"[TS861] {_describe(node)}: a comparison of data dependent sizes "
                                         f"({vals}); only known numbers are compared at import")
            c.env[node.name] = _SCALAR_PREDICATES[node.target](*vals)
            continue
        chain = None
        state_target = None
        if _INPLACE.match(tname) and isinstance(node.args[0], torch.fx.Node) and node.args[0].op == "placeholder"                 and id(c.env.get(node.args[0].name)) in c.state_of:
            state_target = c.state_of[id(c.env[node.args[0].name])]   # num_batches_tracked += 1
        elif _INPLACE.match(tname):
            tgt = node.args[0]
            if isinstance(tgt, torch.fx.Node) and _target_name(tgt) in _VIEW_CHAIN:
                chain, base = _plan_view_mutation(node, order, last_mut)
            else:
                _check_inplace(node, order)
        result = HANDLERS[tname](c, _bind(node))
        if state_target is not None:
            c.state_out[state_target] = result
        if chain is not None:
            result = _scatter_back(c, node, chain, base, result)
            last_mut.setdefault(base.name, []).append((order[node.name], chain[-1].name))
        c.env[node.name] = result
        if c.check and isinstance(result, Value):
            fake = node.meta.get("val")
            if isinstance(fake, torch.Tensor):
                expect = c.meta_type(fake)
                if not result.type.same_as(expect):
                    raise IRError(f"internal error: the IR shape rule gives {result.type} but torch.export "
                                  f"recorded {expect} for {_describe(node)}")
    return outputs


def import_exported(ep: "torch.export.ExportedProgram", name: str = "program",
                    dim_names: Optional[Dict[Tuple[str, int], str]] = None,
                    input_shapes: Optional[Dict[str, Sequence[Union[int, str]]]] = None,
                    training: bool = False) -> Program:
    """Convert an ExportedProgram to a shape checked IR program.

    dim_names: names for dynamic input dims, keyed by (input name, axis). Defaults to "{input}_{axis}".
    input_shapes: optional override of user input shapes (ints or symbol names). Every op is re shape
      checked against the new shapes, so a mismatch is reported here, before anything runs.
    training: import a training graph. Mutated buffers become explicit state (``Program.state``,
      initial values in ``Program.state_init``), dropout reads the key input ``Program.key``.
    """
    from .refusals import check_exported
    check_exported(ep, name)    # MoE, chunked linear attention, chunked SSM: refused by name (TS940 to TS942)
    from .importer_text import has_vmap
    if has_vmap(ep):            # torch.vmap inside the model (transformers' custom attention masks)
        ep = ep.run_decompositions({})
    graph = _inline_wrappers(ep.graph_module)
    sig = ep.graph_signature
    state_targets = _state_buffers(ep) if training else []

    # 1. every op must be known, and the program must be pure. Report all problems at once.
    problems = []
    from .importer_text import runtime_check_problem, runtime_checks
    checks = runtime_checks(graph, _target_name)     # torch._check of an .item(): refused with its line
    for item in dict.fromkeys(checks.values()):
        problems.append(runtime_check_problem(item))
    for node, in_region in _all_nodes(graph, ep.graph_module):
        if node.op == "call_function" and node.target in HOP_HANDLERS:
            continue
        if node in checks:
            continue
        if in_region and node.op == "call_function" and getattr(node.target, "__module__", "") == "_operator"                 and node.target not in _SCALAR_OPS and node.target is not operator.getitem:
            problems.append(f"  Python number arithmetic inside control flow ({_describe(node)}): a Python int or "
                            "float carried through torch.cond, while_loop or scan becomes a value torch.export "
                            "cannot type; carry it as a tensor instead, for example torch.tensor(0)")
            continue
        if node.op == "call_function" and _target_name(node) == "aten.index.Tensor" and any(
                isinstance(i, torch.fx.Node) and getattr(i.meta.get("val"), "dtype", None) in (torch.bool, torch.uint8)
                for i in node.args[1]):
            problems.append(f"  boolean mask indexing x[mask] at {_describe(node)} has a data dependent shape and is "
                            "not supported; keep the shape with torch.where(mask, x, fill) (the assignment "
                            "x[mask] = value is supported)")
        elif node.op == "call_function" and _target_name(node) not in HANDLERS:
            problems.append(f"  unsupported op {_describe(node)}")
        elif node.op not in ("placeholder", "call_function", "output", "get_attr"):
            problems.append(f"  unsupported graph node kind {node.op} at %{node.name}")
    for spec in sig.output_specs:
        if spec.kind.name == "BUFFER_MUTATION" and training:
            continue
        if spec.kind.name != "USER_OUTPUT":
            problems.append(f"  the program mutates {spec.target} ({spec.kind.name}); export the model in "
                            "eval mode or import with training=True, the IR has value semantics")
    if problems:
        raise UnsupportedOpError(f"truestar.ir cannot import {name} ({len(problems)} problem(s); nothing was "
                                 "skipped):\n" + "\n".join(problems)
                                 + "\nSupported ATen ops: truestar.ir.supported_aten_ops()")

    c = _Ctx(ep, name, check_against_export=input_shapes is None)
    c.training = training
    c.state_targets = list(state_targets)
    state_init: Dict[str, torch.Tensor] = {}
    dim_names = dict(dim_names or {})
    specs = {s.arg.name: s for s in sig.input_specs}
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    symbols: Dict[str, Tuple[int, Optional[int]]] = {}

    # 2. user inputs and their symbolic dims
    for ph in placeholders:
        spec = specs[ph.name]
        if spec.kind.name != "USER_INPUT":
            continue
        fake = ph.meta.get("val")
        if not isinstance(fake, torch.Tensor):
            raise UnsupportedOpError(f"input {ph.name} is a {type(fake).__name__}; only tensor inputs are supported")
        for axis, d in enumerate(fake.shape):
            if isinstance(d, torch.SymInt) and isinstance(d.node.expr, sympy.Symbol) and d.node.expr not in c.sym_map:
                nm = dim_names.get((ph.name, axis), f"{ph.name}_{axis}")
                c.sym_map[d.node.expr] = symbol(nm)
                vr = ep.range_constraints.get(d.node.expr)
                symbols[nm] = _range_of(vr)
    for ph in placeholders:  # derived dims (torch.export.Dim arithmetic such as 2*T): name the root symbol
        fake = ph.meta.get("val")
        if specs[ph.name].kind.name != "USER_INPUT" or not isinstance(fake, torch.Tensor):
            continue
        for axis, d in enumerate(fake.shape):
            if isinstance(d, torch.SymInt):
                free = [s for s in d.node.expr.free_symbols if s not in c.sym_map]
                if len(free) == 1:
                    nm = dim_names.get((ph.name, axis), f"{ph.name}_{axis}_root")
                    c.sym_map[free[0]] = symbol(nm)
                    vr = ep.range_constraints.get(free[0])
                    symbols[nm] = _range_of(vr)
    for ph in placeholders:
        spec = specs[ph.name]
        c.node = ph
        if spec.kind.name == "USER_INPUT":
            if input_shapes is not None and ph.name in input_shapes:
                dims = []
                for d in input_shapes[ph.name]:
                    if isinstance(d, str):
                        symbols.setdefault(d, (1, None))
                        d = symbol(d)
                    dims.append(d)
                t = TensorType(ph.meta["val"].dtype, tuple(dims))
            else:
                t = c.meta_type(ph.meta["val"])
            v = Value(c.fresh(ph.name), t, "input")
            c.inputs.append(v)
            c.env[ph.name] = v
            continue
        if not ph.users:
            continue
        kind = spec.kind.name
        if kind == "PARAMETER" or (kind == "BUFFER" and spec.target in ep.state_dict):
            data = ep.state_dict[spec.target]
        elif kind in ("BUFFER", "CONSTANT_TENSOR"):
            data = ep.constants[spec.target]
        else:
            raise UnsupportedOpError(f"input {ph.name} of kind {kind} is not supported")
        data = data.detach()
        if spec.target in state_targets:   # explicit state: an input now, an output after the ops
            v = Value(c.fresh(spec.target), TensorType(data.dtype, tuple(data.shape)), "input")
            c.state_in[spec.target] = v
            c.state_of[id(v)] = spec.target
            state_init[spec.target] = data.clone()
            c.env[ph.name] = v
            continue
        v = Value(c.fresh(spec.target), TensorType(data.dtype, tuple(data.shape)), "param")
        c.params.append(v)
        c.param_data[v.name] = data
        if kind == "PARAMETER":
            c.weights.add(v.name)
        c.env[ph.name] = v

    # 3. ops (control flow regions are converted recursively, see _convert_graph)
    outputs = _convert_graph(c, graph, ep.graph_module, sig.output_specs)
    inputs = list(c.inputs)
    state = []
    for t in state_targets:
        if t not in c.state_in:
            continue   # a mutated buffer nothing reads: its update is ignored as torch.export does
        state.append((t, len(inputs), len(outputs)))
        inputs.append(c.state_in[t])
        outputs.append(c.state_out.get(t, c.state_in[t]))
    if c.key is not None:
        inputs.append(c.key)
    prog = Program(name, inputs, c.params, c.param_data, c.nodes, outputs, symbols)
    prog.state = state
    prog.state_init = {t: state_init[t] for t, _, _ in state}
    prog.key = c.key.name if c.key is not None else None
    prog.verify()
    return prog


def _state_buffers(ep) -> List[str]:
    """Buffers a training graph changes: BUFFER_MUTATION outputs (core ATen graph), the running
    statistics a training ``aten.batch_norm`` reads and buffers changed in place (export graph)."""
    sig = ep.graph_signature
    by_name = {s.arg.name: s for s in sig.input_specs}
    out: List[str] = []

    def add(t):
        if t not in out:
            out.append(t)

    for spec in sig.output_specs:
        if spec.kind.name == "BUFFER_MUTATION":
            add(spec.target)
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        tname = _target_name(node)
        args = []
        if tname == "aten.batch_norm.default" and _bind(node).get("training"):
            b = _bind(node)
            args = [b["running_mean"], b["running_var"]]
        elif _INPLACE.match(tname):
            args = [node.args[0]]
        for x in args:
            if isinstance(x, torch.fx.Node) and x.op == "placeholder":
                spec = by_name.get(x.name)
                if spec is not None and spec.kind.name == "BUFFER":
                    add(spec.target)
    return out


def initial_state(prog: Program) -> List[torch.Tensor]:
    """The state inputs of an imported training program at their starting values, in input order."""
    return [prog.state_init[t].clone() for t, _, _ in getattr(prog, "state", [])]


# Higher order ops torch.export uses to scope a grad mode or an autocast region (Hugging Face rotary
# embeddings run under ``torch.no_grad()`` and ``torch.autocast(..., enabled=False)``). Neither
# changes a value in inference, so their body graph is inlined. An enabled autocast region would
# change dtypes and is refused. Name: (index of the body graph argument, check of the leading args).
_WRAPPER_HOPS = {
    "wrap_with_set_grad_enabled": 1,
    "wrap_with_autocast": 4,
}


def _inline_wrappers(gm: torch.fx.GraphModule) -> torch.fx.Graph:
    """``gm.graph`` with grad mode and disabled autocast wrappers replaced by their bodies (a new
    graph; the exported program is not changed). Placeholder names are kept, so the graph signature
    still applies."""
    graph = gm.graph
    if not any(n.op == "call_function" and getattr(n.target, "__name__", "") in _WRAPPER_HOPS
               for n in graph.nodes):
        return graph
    new = torch.fx.Graph()

    def copy_into(src_gm: torch.fx.GraphModule, g: torch.fx.Graph, env: Dict[torch.fx.Node, Any]) -> Any:
        hop_out: Dict[torch.fx.Node, Any] = {}
        result = None
        for n in g.nodes:
            if n.op == "placeholder" and g is not graph:
                continue  # bound by the caller
            if n.op == "output":
                if g is graph:
                    new.node_copy(n, lambda x: env[x])
                    return None
                result = torch.fx.node.map_arg(n.args[0], lambda x: env[x])
                return result
            if n.op == "call_function" and getattr(n.target, "__name__", "") in _WRAPPER_HOPS:
                name = n.target.__name__
                body_at = _WRAPPER_HOPS[name]
                if name == "wrap_with_autocast" and bool(n.args[2]):
                    raise UnsupportedOpError(f"an enabled torch.autocast region at %{n.name} changes dtypes inside "
                                             "the model; export it without autocast")
                body = getattr(src_gm, n.args[body_at].target)
                sub_env: Dict[torch.fx.Node, Any] = {}
                operands = [env[a] if isinstance(a, torch.fx.Node) else a for a in n.args[body_at + 1:]]
                phs = [p for p in body.graph.nodes if p.op == "placeholder"]
                if len(phs) != len(operands):
                    raise UnsupportedOpError(f"%{n.name}: {name} passes {len(operands)} operands to a body with "
                                             f"{len(phs)} inputs")
                for p, a in zip(phs, operands):
                    sub_env[p] = a
                hop_out[n] = copy_into(body, body.graph, sub_env)
                env[n] = hop_out[n]
                continue
            if n.op == "call_function" and n.target is operator.getitem and n.args[0] in hop_out:
                env[n] = hop_out[n.args[0]][n.args[1]]
                continue
            if n.op == "get_attr" and isinstance(getattr(src_gm, str(n.target), None), torch.fx.GraphModule):
                continue  # a wrapper body, inlined where it is called
            env[n] = new.node_copy(n, lambda x: env[x])
        return result

    copy_into(gm, graph, {})
    return new


def _refuse_aliased_args(args) -> None:
    """torch.export treats example inputs that share storage as one input for good: the graph reads
    only one of them, so the imported program would silently ignore the other (found 2026-11-24 with
    four KV caches made from one zeros tensor). Refuse instead."""
    leaves = [t for t in torch.utils._pytree.tree_leaves(tuple(args)) if isinstance(t, torch.Tensor)]
    seen: Dict[int, int] = {}
    for i, t in enumerate(leaves):
        if t.numel() == 0:
            continue
        key = t.untyped_storage().data_ptr()
        if key in seen:
            raise UnsupportedOpError(f"example inputs {seen[key]} and {i} (counting tensors in order) share storage; "
                                     "torch.export would keep only one of them and the imported program would ignore "
                                     "the other. Pass distinct tensors, for example t.clone()")
        seen[key] = i


# Small integer tensors a model builds from its own constants (SigLIP2's spatial_shapes, Pixtral's
# image_sizes, from a fixed image size) are read back with .item() or .tolist() as sizes. torch's
# fake tensors keep the values of constants up to CONSTANT_NUMEL_LIMIT elements (1 by default), so
# with the default those reads are data dependent: export fails (Pixtral) or keeps unknown sizes
# (SigLIP2). A model that hits that is exported again with the limit raised, which makes every such
# value a known number in the graph; a read of a value that depends on an input stays unknown and is
# refused at import (TS861). The first export is kept when it needs no constant.
_TRACKED_CONSTANT_NUMEL = 256


def _reads_unknown_scalar(ep) -> bool:
    for n in ep.graph.nodes:
        if n.op == "call_function" and _target_name(n) in ("aten.item.default", "aten._local_scalar_dense.default"):
            if isinstance(n.meta.get("val"), (torch.SymInt, torch.SymFloat, torch.SymBool)):
                return True
    return False


def _export_tracking_constants(model, args, dynamic_shapes, kwargs=None):
    import torch._subclasses.fake_tensor as fake_tensor

    def run():
        with plain_einsum():
            return torch.export.export(model, tuple(args), kwargs=kwargs or None, dynamic_shapes=dynamic_shapes)

    first_error = None
    try:
        ep = run()
        if not _reads_unknown_scalar(ep):
            return ep
    except Exception as exc:   # noqa: BLE001  (retried once below, re raised if that fails too)
        if "data-dependent" not in str(exc) and "DataDependent" not in type(exc).__name__                 and "GuardOnDataDependent" not in str(exc):
            raise
        first_error, ep = exc, None
    if not hasattr(fake_tensor, "CONSTANT_NUMEL_LIMIT"):
        if first_error is not None:
            raise first_error
        return ep
    old = fake_tensor.CONSTANT_NUMEL_LIMIT
    fake_tensor.CONSTANT_NUMEL_LIMIT = max(old, _TRACKED_CONSTANT_NUMEL)
    try:
        return run()
    except Exception:   # noqa: BLE001
        if first_error is not None:
            raise first_error from None
        return ep
    finally:
        fake_tensor.CONSTANT_NUMEL_LIMIT = old


def import_module(model: torch.nn.Module, args: Tuple[Any, ...], dynamic_shapes: Any = None,
                  name: Optional[str] = None, decompose: bool = False, **kw) -> Program:
    """Export ``model`` with torch.export and import it. Dim names come from torch.export.Dim objects.

    decompose=True imports the core ATen graph (``run_decompositions()``) instead. training=True
    imports the training graph with explicit state (see ``import_exported``); the model must then
    be in training mode, and nothing about it is changed (torch.export does not run it).
    """
    if kw.get("training") and not model.training:
        raise UnsupportedOpError("import_module(..., training=True) needs the model in training mode: call "
                                 "model.train() first")
    _refuse_aliased_args(args)
    from ..moe.coverage import lowered
    with lowered(model):        # covered MoE experts export in their static form (truestar.moe)
        return _import_module(model, args, dynamic_shapes, name, decompose, **kw)


def _import_module(model, args, dynamic_shapes, name, decompose, **kw) -> Program:
    from .refusals import check_module
    check_module(model, name)   # before export: a refused family never reaches torch.export
    from .hf_linattn import lowered
    with lowered(model):   # Mamba 2 SSD and Gated DeltaNet layers -> linear_attention (P60)
        try:
            ep = _export_tracking_constants(model, args, dynamic_shapes)
        except Exception as e:  # noqa: BLE001  (a data dependent size is named at the model's line)
            from .importer_text import explain_export_failure
            explain_export_failure(e, name or type(model).__name__)
            raise
    if decompose:
        ep = ep.run_decompositions()
    dim_names: Dict[Tuple[str, int], str] = {}
    if dynamic_shapes is not None:
        user = [s.arg.name for s in ep.graph_signature.input_specs if s.kind.name == "USER_INPUT"]
        specs = dynamic_shapes.values() if isinstance(dynamic_shapes, dict) else dynamic_shapes
        for inp, spec in zip(user, specs):
            for axis, d in (spec or {}).items():
                if d is not None and hasattr(d, "root") and hasattr(d.root, "__name__"):
                    dim_names[(inp, axis)] = d.root.__name__  # a derived dim names its root symbol
                elif d is not None and hasattr(d, "__name__"):
                    dim_names[(inp, axis)] = d.__name__
    return import_exported(ep, name=name or type(model).__name__, dim_names=dim_names, **kw)


# ----------------------------------------------------------------------------- decoder LLM ops (2026-11-24)
# See docs/LLM.md. Grouped query attention and repeat_interleave lower onto existing ops (a reshape,
# an expand and a reshape of the KV heads, then ``attention``), so every consumer of ``attention``
# already handles them. The rest map onto the new IR ops in shapes.py.

from .shapes import infer_outputs as _infer_outputs  # noqa: E402


def _emit_multi(c: _Ctx, op: str, inputs: Sequence[Optional[Value]], **attrs: Any) -> Tuple[Value, ...]:
    """Like ``_Ctx.emit`` for an op that defines several values (sort, topk)."""
    node = c.node
    src = _describe(node)
    try:
        ts = _infer_outputs(op, [None if x is None else x.type for x in inputs], attrs)
    except ShapeError as e:
        raise e.with_source(src) from None
    except UnsupportedOpError as e:
        raise UnsupportedOpError(f"{src}: {e}") from None
    outs = tuple(Value(c.fresh(node.name if i == 0 else f"{node.name}_{i}"), t, "op") for i, t in enumerate(ts))
    c.nodes.append(Node(op, tuple(inputs), dict(attrs), outs[0], src, outs[1:]))
    return outs


def repeat_heads(emit, x: Value, n: Dim, d: int) -> Value:
    """repeat_interleave(x, n, dim=d) as reshape, expand, reshape: entry i of the result is entry
    i // n of x. ``emit(op, inputs, **attrs)`` adds one node."""
    shape = list(x.type.shape)
    if dims_equal(n, 1):
        return x
    up = emit("reshape", [x], shape=shape[:d + 1] + [1] + shape[d + 1:])
    wide = emit("expand", [up], shape=shape[:d + 1] + [n] + shape[d + 1:])
    return emit("reshape", [wide], shape=shape[:d] + [norm_dim(sympy.sympify(shape[d]) * n)] + shape[d + 1:])


def gqa_factor(q: TensorType, k: TensorType) -> int:
    """How many query heads share one KV head; raises UnsupportedOpError when it is not a fixed int."""
    if q.rank < 3 or k.rank != q.rank:
        raise UnsupportedOpError("grouped query attention needs q and k of the same rank with a heads dim at -3")
    hq, hk = q.shape[-3], k.shape[-3]
    if not (isinstance(hq, int) and isinstance(hk, int)):
        raise UnsupportedOpError(f"grouped query attention needs fixed head counts, got q heads {hq} and kv heads {hk}")
    if hk == 0 or hq % hk:
        raise UnsupportedOpError(f"grouped query attention needs the q heads ({hq}) to be a multiple of the kv "
                                 f"heads ({hk})")
    return hq // hk


def _emitter(c):
    return lambda op, ins, **kw: c.emit(op, ins, **kw)


@aten("aten.scaled_dot_product_attention.default")
def _sdpa_gqa(c, a):
    """Attention; with enable_gqa=True the KV heads are repeated (repeat_interleave along dim -3)
    up to the query heads first, which is exactly PyTorch's definition."""
    if a["dropout_p"] != 0:
        _unsupported(c, f"attention dropout_p={a['dropout_p']} is not supported in inference")
    q, k, v = c.tensor(a["query"]), c.tensor(a["key"]), c.tensor(a["value"])
    if a.get("enable_gqa"):
        try:
            n = gqa_factor(q.type, k.type)
        except UnsupportedOpError as e:
            _unsupported(c, str(e))
        if v.type.rank != q.type.rank or not dims_equal(v.type.shape[-3], k.type.shape[-3]):
            _unsupported(c, "grouped query attention needs k and v with the same number of heads")
        if n > 1:
            return _gqa_grouped(c, q, k, v, c.tensor(a["attn_mask"]), n, a)
    return c.emit("attention", [q, k, v, c.tensor(a["attn_mask"])], scale=a["scale"], is_causal=bool(a["is_causal"]))


def _gqa_grouped(c, q: Value, k: Value, v: Value, mask: Optional[Value], n: int, a) -> Value:
    """Grouped query attention without copying the KV heads (2026-11-24, P38): the heads dim of q
    is split into (kv heads, n) and k, v get a size 1 dim there, so attention broadcasts each KV
    head over its n query heads. Every split is a view; repeating the KV heads materialized the
    cache n times per layer (a decode step's largest copies). Same math as the repeat: query head
    h reads KV head h // n."""
    r = q.type.rank
    d = r - 3
    hk = k.type.shape[d]

    def split(x: Value, at: int, parts) -> Value:
        s = list(x.type.shape)
        return c.emit("reshape", [x], shape=s[:at] + list(parts) + s[at + 1:])
    q5 = split(q, d, [hk, n])
    k5 = split(k, d, [hk, 1])
    v5 = split(v, d, [hk, 1])
    m5 = mask
    if mask is not None and mask.type.rank >= 3:
        md = mask.type.rank - 3
        hm = mask.type.shape[md]
        m5 = split(mask, md, [1, 1] if dims_equal(hm, 1) else [hk, n])
    out = c.emit("attention", [q5, k5, v5, m5], scale=a["scale"], is_causal=bool(a["is_causal"]))
    s = list(out.type.shape)
    return c.emit("reshape", [out], shape=s[:d] + [q.type.shape[d]] + s[d + 2:])


def _numel_of(x: Value):
    n = sympy.Integer(1)
    for d in x.type.shape:
        n = n * sympy.sympify(d)
    return sympy.expand(n)


@aten("aten.repeat_interleave.self_int")
def _repeat_interleave(c, a):
    x = c.tensor(a["self"])
    r = c.dim(a["repeats"])
    if a.get("dim") is None:
        x = _reshape_if(c, x, [norm_dim(_numel_of(x))])
        d = 0
    else:
        d = a["dim"] % x.type.rank
    return repeat_heads(_emitter(c), x, r, d)


@aten("aten.flip.default")
def _flip(c, a):
    x = c.tensor(a["self"])
    dims = [d % max(x.type.rank, 1) for d in a["dims"]]
    return x if not dims else c.emit("flip", [x], dims=dims)


@aten("aten.sort.default", "aten.sort.stable")
def _sort(c, a):
    x = c.tensor(a["self"])
    return _emit_multi(c, "sort", [x], dim=a["dim"] % max(x.type.rank, 1), descending=bool(a["descending"]),
                       stable=True)


@aten("aten.topk.default")
def _topk(c, a):
    x = c.tensor(a["self"])
    k = c.dim(a["k"])
    if not isinstance(k, int):
        _unsupported(c, f"topk with a symbolic k ({k}) is not supported")
    return _emit_multi(c, "topk", [x], k=k, dim=a["dim"] % max(x.type.rank, 1), largest=bool(a["largest"]),
                       sorted=True)


@aten("aten.argmax.default")
def _argmax(c, a):
    return c.emit("argmax", [c.tensor(a["self"])], dim=a.get("dim"), keepdim=bool(a.get("keepdim", False)))


@aten("aten.argmin.default")
def _argmin(c, a):
    return c.emit("argmin", [c.tensor(a["self"])], dim=a.get("dim"), keepdim=bool(a.get("keepdim", False)))


def _extreme_dim(op):
    def fn(c, a):
        """(values, indices) along dim: the arg reduction, then the values gathered at those indices."""
        x = c.tensor(a["self"])
        d = a["dim"] % max(x.type.rank, 1)
        keep = bool(a.get("keepdim", False))
        i = c.emit(op, [x], dim=d, keepdim=True)
        v = c.emit("gather", [x, i], dim=d)
        if not keep:
            shape = [s for j, s in enumerate(x.type.shape) if j != d]
            i, v = _reshape_if(c, i, shape), _reshape_if(c, v, shape)
        return (v, i)
    return fn


HANDLERS["aten.max.dim"] = _extreme_dim("argmax")
HANDLERS["aten.min.dim"] = _extreme_dim("argmin")


def _extreme_values(op):
    def fn(c, a):
        x = c.tensor(a["self"])
        dims = a.get("dim")
        dims = [] if dims is None else ([dims] if isinstance(dims, int) else list(dims))
        return c.emit(op, [x], dims=[d % max(x.type.rank, 1) for d in dims] or None,
                      keepdim=bool(a.get("keepdim", False)))
    return fn


# torch.export spells a max reduction several ways depending on the version (torch 2.14 emits amax
# where 2.11 kept max.dim): every spelling maps here.
HANDLERS["aten.amax.default"] = _extreme_values("amax")
HANDLERS["aten.amin.default"] = _extreme_values("amin")
HANDLERS["aten.max.default"] = _extreme_values("amax")  # full reduction, no dim
HANDLERS["aten.min.default"] = _extreme_values("amin")
def _bool_logic(op):
    def fn(c, a):
        x = c.tensor(a["self"])
        y = c.operand(a["other"], x)
        if x.type.dtype != torch.bool or y.type.dtype != torch.bool:
            _unsupported(c, f"bitwise {op} of integer tensors is not supported (only bool masks)")
        return c.emit(f"logical_{op}", [x, y])
    return fn


for _ov in ("Tensor", "Scalar"):  # mask & mask, mask | mask
    HANDLERS[f"aten.__and__.{_ov}"] = HANDLERS[f"aten.bitwise_and.{_ov}"] = _bool_logic("and")
    HANDLERS[f"aten.__or__.{_ov}"] = HANDLERS[f"aten.bitwise_or.{_ov}"] = _bool_logic("or")
    # in place forms (Mistral's sliding window mask: ``mask.bitwise_or_(window)``); the driver checks
    # nobody reads the old value, as for every in place op
    HANDLERS[f"aten.bitwise_and_.{_ov}"] = HANDLERS[f"aten.__iand__.{_ov}"] = _bool_logic("and")
    HANDLERS[f"aten.bitwise_or_.{_ov}"] = HANDLERS[f"aten.__ior__.{_ov}"] = _bool_logic("or")
HANDLERS["aten.max.other"] = _arith("maximum")  # elementwise torch.max(a, b)
HANDLERS["aten.min.other"] = _arith("minimum")


@aten("aten.index_copy.default", "aten.index_copy_.default")
def _index_copy(c, a):
    x = c.tensor(a["self"])
    return c.emit("index_copy", [x, c.tensor(a["index"]), _cast_if(c, c.tensor(a["source"]), x.type.dtype)],
                  dim=a["dim"] % x.type.rank)


@aten("aten.index_put.default", "aten.index_put_.default")
def _index_put(c, a):
    """``x[..., idx] = values`` with one integer index (becomes ``index_copy``), or ``x[mask] = value``
    with a boolean mask and a single value (becomes ``where``). Neither has a data dependent shape.
    Anything else (several index tensors, accumulate, a mask with many values) is refused."""
    x = c.tensor(a["self"])
    idx = c.val(a["indices"])
    vals = c.tensor(a["values"])
    if a.get("accumulate"):
        _unsupported(c, "index_put with accumulate=True is not supported")
    real = [(d, t) for d, t in enumerate(idx) if t is not None]
    if len(real) != 1:
        _unsupported(c, f"index_put with {len(real)} index tensors is not supported; use one index tensor")
    d, t = real[0]
    if t.type.dtype in (torch.bool, torch.uint8):
        m = t.type.rank
        if x.type.rank < d + m or not all(dims_equal(p, q) for p, q in zip(t.type.shape, x.type.shape[d:d + m])):
            _unsupported(c, f"boolean mask {t.type} does not match dims {d} to {d + m - 1} of {x.type}")
        if not all(dims_equal(s, 1) for s in vals.type.shape):
            _unsupported(c, "x[mask] = values with more than one value has a data dependent shape; "
                            "write torch.where(mask, values, x) instead")
        mask = _cast_if(c, t, torch.bool)
        mask = _reshape_if(c, mask, [1] * d + list(t.type.shape) + [1] * (x.type.rank - d - m))
        v = _cast_if(c, _reshape_if(c, vals, []), x.type.dtype)
        return c.emit("where", [mask, v, x])
    if t.type.rank != 1:
        _unsupported(c, f"index_put with a rank {t.type.rank} index is not supported; use a rank 1 index")
    want = list(x.type.shape[:d]) + [t.type.shape[0]] + list(x.type.shape[d + 1:])
    if vals.type.rank != len(want) or not all(dims_equal(p, q) for p, q in zip(vals.type.shape, want)):
        vals = c.emit("expand", [vals], shape=want)
    return c.emit("index_copy", [x, t, _cast_if(c, vals, x.type.dtype)], dim=d)


_plain_index = _index


@aten("aten.index.Tensor")
def _index_bool_checked(c, a):
    for t in c.val(a["indices"]):
        if isinstance(t, Value) and t.type.dtype in (torch.bool, torch.uint8):
            _unsupported(c, "boolean mask indexing x[mask] has a data dependent shape and is not supported; "
                            "keep the shape with torch.where(mask, x, fill) or x.masked_fill(~mask, fill) "
                            "(the assignment x[mask] = value is supported)")
    return _plain_index(c, a)


# chunk and split along a symbolic dim: supported when the piece count is static and every piece has
# the same provably integer size, as a reshape into (count, size) and one select per piece.


def _even_pieces(c, x: Value, d: int, count: int) -> Optional[Tuple[Value, ...]]:
    each = sympy.simplify(sympy.sympify(x.type.shape[d]) / count)
    if each.is_integer is not True:
        return None
    each = norm_dim(each)
    shape = list(x.type.shape)
    grid = c.emit("reshape", [x], shape=shape[:d] + [count, each] + shape[d + 1:])
    return tuple(c.emit("select", [grid], dim=d, index=i) for i in range(count))


_static_chunk, _static_split, _static_split_sizes = _chunk, _split, _split_with_sizes


def _refuse_uneven(c, x, d, what):
    _unsupported(c, f"{what} along the symbolic dim {x.type.shape[d]} is supported only into a fixed number "
                    "of equal pieces whose size is provably an integer (for example chunk(2) of a dim 2*T)")


@aten("aten.chunk.default")
def _chunk_sym(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    if isinstance(x.type.shape[d], int) or a["chunks"] == 1:
        return _static_chunk(c, a)
    out = _even_pieces(c, x, d, a["chunks"])
    if out is None:
        _refuse_uneven(c, x, d, "chunk")
    return out


@aten("aten.split.Tensor")
def _split_sym(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    s = c.dim(a["split_size"])
    if isinstance(x.type.shape[d], int) and isinstance(s, int):
        return _static_split(c, {**a, "split_size": s})
    count = sympy.simplify(sympy.sympify(x.type.shape[d]) / s)
    out = _even_pieces(c, x, d, int(count)) if count.is_Integer and int(count) > 0 else None
    if out is None:
        _refuse_uneven(c, x, d, "split")
    return out


@aten("aten.split_with_sizes.default", "aten.split.sizes")
def _split_sizes_sym(c, a):
    x = c.tensor(a["self"])
    d = a["dim"] % x.type.rank
    sizes = c.dims(a["split_sizes"] if "split_sizes" in a else a["split_size"])
    if isinstance(x.type.shape[d], int) and all(isinstance(s, int) for s in sizes):
        key = "split_sizes" if "split_sizes" in a else "split_size"
        return _static_split_sizes(c, {**a, key: sizes})
    out = None
    if sizes and all(dims_equal(s, sizes[0]) for s in sizes) and \
            dims_equal(sympy.sympify(sizes[0]) * len(sizes), x.type.shape[d]):
        out = _even_pieces(c, x, d, len(sizes))
    if out is None:
        _refuse_uneven(c, x, d, "split")
    return out


# ----------------------------------------------------------------------------- control flow (2026-11-24)
# torch.export captures torch.cond, while_loop and scan as higher order ops whose branches and bodies
# are subgraphs. Each subgraph becomes an IR region, converted by the same code as the program, so
# every op inside is checked and shape typed the same way. scan has no IR op of its own: it becomes
# a ``loop`` over the scanned dim that reads ``xs[k]`` with ``select_dynamic`` and writes ``ys[k]``
# with ``index_copy``, which also makes it differentiable by autodiff_cf.py with no new rule.

from .core import Builder as _Builder  # noqa: E402
from .regions import inline as _inline  # noqa: E402

_HOP = getattr(torch.ops, "higher_order", None)

# id(IR region) -> the torch subgraph it was imported from. Only read by the exporter, right after
# an import, to check branches its example inputs never take; entries are never relied on later.
TORCH_REGIONS: Dict[int, Any] = {}


def _get_attr(owner: torch.nn.Module, target: str):
    obj = owner
    for part in str(target).split("."):
        obj = getattr(obj, part)
    return obj


def _all_nodes(graph: torch.fx.Graph, owner: torch.nn.Module, in_region: bool = False):
    """(node, inside a control flow subgraph) for every node of the graph and, recursively, of the
    subgraphs its control flow ops use."""
    for node in graph.nodes:
        yield node, in_region
        if node.op == "get_attr":
            sub = _get_attr(owner, node.target)
            if isinstance(sub, torch.fx.GraphModule):
                yield from _all_nodes(sub.graph, sub, True)


def _hop_values(c: _Ctx, xs, what: str) -> List[Value]:
    out = []
    for i, x in enumerate(xs):
        v = c.val(x)
        if not isinstance(v, Value):
            _unsupported(c, f"{what} {i} is a {type(v).__name__} ({v!r}); only tensors can be passed into "
                            "control flow here, wrap a Python number in torch.tensor(...)")
        out.append(v)
    return out


def _import_region(c: _Ctx, gm: torch.fx.GraphModule, in_types: Sequence[TensorType], label: str,
                   prefix: Sequence[Tuple[str, TensorType]] = ()) -> Program:
    """One control flow subgraph as an IR region. Its placeholders get ``in_types`` (the outer
    operand types, so the outer symbols carry over); ``prefix`` adds leading inputs the subgraph
    does not have (the iteration index of a loop body)."""
    hop = c.node
    sub = _Ctx(c.ep, label, c.check)
    sub.sym_map = c.sym_map
    for nm, t in prefix:
        sub.inputs.append(Value(sub.fresh(nm), t, "input"))
    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(phs) != len(in_types):
        raise IRError(f"internal error: {label} of {_describe(hop)} takes {len(phs)} inputs, the op passes "
                      f"{len(in_types)}")
    for ph, t in zip(phs, in_types):
        fake = ph.meta.get("val")
        if c.check and isinstance(fake, torch.Tensor):
            want = c.meta_type(fake)
            if not want.same_as(t):
                raise IRError(f"internal error: {label} of {_describe(hop)} declares {ph.name}: {want} but "
                              f"receives {t}")
        v = Value(sub.fresh(ph.name), t, "input")
        sub.inputs.append(v)
        sub.env[ph.name] = v
    outs = _convert_graph(sub, gm.graph, gm)
    c.node = hop
    where = _describe(hop)
    for n in sub.nodes:  # nodes inside a subgraph point at torch's generated code: name the user's line
        if "<eval_with_key>" in n.source:
            n.source = f"{where} [{label}]"
    return Program(label, sub.inputs, sub.params, sub.param_data, sub.nodes, outs, {})


def _scalar_pred(c: _Ctx, p: Value) -> Value:
    if p.type.dtype != torch.bool:
        _unsupported(c, f"the torch.cond predicate must be a bool tensor, got {p.type}")
    if p.type.rank:
        if not all(dims_equal(d, 1) for d in p.type.shape):
            _unsupported(c, f"the torch.cond predicate must hold one value, got {p.type}")
        p = c.emit("reshape", [p], shape=[])
    return p


def _hop_cond(c: _Ctx, node: torch.fx.Node):
    """torch.cond(pred, true_fn, false_fn, operands) -> IR cond with both branches as regions."""
    pred, true_gm, false_gm, operands = node.args[:4]
    p = c.val(pred)
    if not isinstance(p, Value):
        _unsupported(c, f"the torch.cond predicate is the Python value {p!r}; torch.export already decided "
                        "it, so there is no branch to keep")
    p = _scalar_pred(c, p)
    ops = _hop_values(c, operands, "torch.cond operand")
    types = [v.type for v in ops]
    then = _import_region(c, c.val(true_gm), types, "then_branch")
    other = _import_region(c, c.val(false_gm), types, "else_branch")
    TORCH_REGIONS[id(then)] = c.val(true_gm)    # lets the exporter verify a branch against torch's own
    TORCH_REGIONS[id(other)] = c.val(false_gm)  # subgraph even when no check takes it
    return _emit_multi(c, "cond", [p] + ops, then_branch=then, else_branch=other)


def _hop_while(c: _Ctx, node: torch.fx.Node):
    """while_loop(cond_fn, body_fn, carried, additional) -> IR loop with a cond region."""
    cond_gm, body_gm, carried, extra = node.args[:4]
    carried = _hop_values(c, carried, "while_loop carried value")
    extra = _hop_values(c, extra, "while_loop additional input")
    types = [v.type for v in carried + extra]
    cond = _import_region(c, c.val(cond_gm), types, "cond")
    body = _import_region(c, c.val(body_gm), types, "body", prefix=[("iteration", TensorType(torch.int64, ()))])
    if len(cond.outputs) == 1 and cond.outputs[0].type.rank:
        _unsupported(c, f"the while_loop condition must be one bool value, got {cond.outputs[0].type}")
    return _emit_multi(c, "loop", carried + extra, n_carried=len(carried), body=body, cond=cond)


def _hop_scan(c: _Ctx, node: torch.fx.Node):
    """scan(combine_fn, init, xs, additional) -> IR loop over dim 0 of xs. Carried: the scan carry
    and one [n, ...] buffer per stacked output; xs[k] is read with select_dynamic and y_k written
    with index_copy. Returns the final carry, then the stacked outputs, as torch's scan does."""
    combine_gm, init, xs, extra = node.args[:4]
    init = _hop_values(c, init, "scan init")
    xs = _hop_values(c, xs, "scan xs")
    extra = _hop_values(c, extra, "scan additional input")
    if not xs:
        _unsupported(c, "scan with no xs has no length")
    n = xs[0].type.shape[0] if xs[0].type.rank else None
    for x in xs:
        if not x.type.rank or not dims_equal(x.type.shape[0], n):
            _unsupported(c, f"scan xs must share their first dim, got {[str(v.type) for v in xs]}")
    slices = [TensorType(x.type.dtype, x.type.shape[1:]) for x in xs]
    combine = _import_region(c, c.val(combine_gm), [v.type for v in init] + slices + [v.type for v in extra],
                             "combine")
    nc = len(init)
    if len(combine.outputs) < nc:
        _unsupported(c, f"scan combine returns {len(combine.outputs)} values for {nc} carried values")
    ys = combine.outputs[nc:]
    b = _Builder("body")
    k = b.input("iteration", torch.int64, ())
    carry = [b.input(f"carry{i}", v.type.dtype, v.type.shape) for i, v in enumerate(init)]
    bufs = [b.input(f"ys{i}", y.type.dtype, (n,) + tuple(y.type.shape)) for i, y in enumerate(ys)]
    xin = [b.input(f"xs{i}", v.type.dtype, v.type.shape) for i, v in enumerate(xs)]
    ein = [b.input(f"extra{i}", v.type.dtype, v.type.shape) for i, v in enumerate(extra)]
    x_k = [b.op("select_dynamic", x, k, dim=0) for x in xin]
    outs = _inline(b, combine, carry + x_k + ein)
    at = b.op("reshape", k, shape=[1])
    new_bufs = [b.op("index_copy", buf, at, b.op("reshape", y, shape=[1] + list(y.type.shape)), dim=0)
                for buf, y in zip(bufs, outs[nc:])]
    body = b.build(outs[:nc] + new_bufs)
    buf_init = [c.emit("full", [], shape=[n] + list(y.type.shape), value=0, dtype=y.type.dtype) for y in ys]
    return _emit_multi(c, "loop", init + buf_init + xs + extra, n_carried=nc + len(ys), body=body,
                       trip_count=n)


HOP_HANDLERS: Dict[Any, Callable[[_Ctx, torch.fx.Node], Any]] = {}
if _HOP is not None:
    for _name, _fn in (("cond", _hop_cond), ("while_loop", _hop_while), ("scan", _hop_scan)):
        if hasattr(_HOP, _name):
            HOP_HANDLERS[getattr(_HOP, _name)] = _fn


@aten("truestar.kvq_attention.default")
def _kvq_attention_op(c, a):
    """truestar.kvquant's torch op (``torch.ops.truestar.kvq_attention``): the quantized cache stays
    in its Hkv heads (no repeat_heads), its data in int8 or uint8 bits."""
    ins = [c.tensor(a[n]) for n in ("q", "k_data", "k_scale", "v_data", "v_scale", "k_tail", "v_tail",
                                     "tail_len")]
    ins.append(c.tensor(a["prefix_len"]) if a.get("prefix_len") is not None else None)
    extra = {}
    if a.get("k_bscale") is not None:       # 4 bit formats: E4M3 block scales, maybe rotated
        ins += [c.tensor(a["k_bscale"]), c.tensor(a["v_bscale"])]
        extra["rot"] = bool(a.get("rot", False))
    return c.emit("kvq_attention", ins, fmt=a["fmt"], group=int(a["group"]), vgroup=int(a["vgroup"]),
                  channel=bool(a["channel"]), scale=a["scale"], **extra)


@aten("truestar.mla_attention.default")
def _mla_attention_op(c, a):
    """truestar.mla's fused latent attention (``torch.ops.truestar.mla_attention``, mla/latent.py)."""
    ins = [c.tensor(a[n]) for n in ("q_nope", "q_rope", "cc", "rc", "lat", "kr", "pos", "w_uk", "w_uv")]
    return c.emit("mla_attention", ins, scale=float(a["scale"]))


from . import importer_extra  # noqa: E402,F401  (ATen ops added with the op coverage work)
from . import linattn_ops as _linattn_ops  # noqa: E402  (torch.ops.truestar.linear_attention, P60)
_linattn_ops.register_import(aten)
