"""``@truestar.function``: typed native functions compiled ahead of time from source.

    import truestar
    from truestar import Tensor, ops

    @truestar.function
    def layer(x: Tensor["B", 128, "f32"], w: Tensor[128, 64, "f32"]) -> Tensor["B", 64, "f32"]:
        h = x @ w
        if h.sum() > 0:          # tensor data: an IR cond, both branches compiled
            h = ops.relu(h)
        return h

The function is translated when it is decorated, before any data exists, so a shape error is raised
at the decorator with the line of the bad op. One compile serves every value of the symbolic dims
(``B`` above). Parameters that are not tensors are static: their value is part of the compile key,
and a function whose static parameters all have defaults is compiled at decoration time.

Execution goes through a pluggable executor. The default runs the float64 reference interpreter;
a code generator plugs in by implementing ``Executor.prepare``.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .. import pytree
from ..ir import reference
from ..ir.core import Builder, Program, ShapeError, TensorType
from .errors import FrontendError, FrontendShapeError, InputError, UnsupportedConstructError
from .translate import FunctionSource, RandomSites, Scope, T, Translator, is_number, is_tensor
from .types import TensorSpec

STATS = {"compiles": 0}


def compile_count() -> int:
    """Total compiles performed by every native function in this process."""
    return STATS["compiles"]


# ----------------------------------------------------------------------------- executors


class Executor:
    """The seam between compiled IR and whatever runs it.

    ``prepare(program)`` is called once per compiled program and returns a callable that takes the
    input tensors and returns the list of output tensors. A code generator compiles kernels in
    ``prepare``; the reference executor just closes over the program.
    """

    name = "executor"

    def prepare(self, program: Program) -> Callable[[Sequence[torch.Tensor]], List[torch.Tensor]]:
        raise NotImplementedError


class ReferenceExecutor(Executor):
    """Runs the float64 reference interpreter. Outputs are float64 for floating point results."""

    name = "reference"

    def prepare(self, program: Program):
        def run(inputs):
            return reference.run(program, *inputs)
        return run


_DEFAULT: List[Executor] = [ReferenceExecutor()]


def set_default_executor(executor: Executor) -> Executor:
    """Set the executor used by functions created without an explicit one. Returns the old one."""
    old = _DEFAULT[0]
    _DEFAULT[0] = executor
    return old


# ----------------------------------------------------------------------------- signature


@dataclass
class Signature:
    """The typed signature of one compiled program."""

    name: str
    params: List[Tuple[str, TensorType]]
    static: Dict[str, Any]
    outputs: List[TensorType]
    symbols: Dict[str, Tuple[int, Optional[int]]]
    returns_tuple: bool
    in_trees: Dict[str, Any] = dataclasses.field(default_factory=dict)  # structured parameters
    out_tree: Optional[Dict[str, Any]] = None  # the returned structure (pytree spec)

    def __str__(self) -> str:
        ps = ", ".join(f"{n}: {t}" for n, t in self.params)
        st = "".join(f", {k}={v!r}" for k, v in self.static.items())
        if self.out_tree is not None and self.out_tree["kind"] not in ("leaf", "tuple"):
            paths = pytree.leaf_paths(self.out_tree)
            outs = pytree.describe(self.out_tree) + " with " + ", ".join(
                f"{p}: {t}" for p, t in zip(paths, self.outputs))
            return f"{self.name}({ps}{st}) -> {outs}"
        outs = ", ".join(str(t) for t in self.outputs)
        outs = f"({outs})" if self.returns_tuple or len(self.outputs) != 1 else outs
        return f"{self.name}({ps}{st}) -> {outs}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": [{"name": n, "type": str(t)} for n, t in self.params],
            "static": {k: repr(v) for k, v in self.static.items()},
            "outputs": [str(t) for t in self.outputs],
            "symbols": {k: [lo, hi] for k, (lo, hi) in self.symbols.items()},
        }


@dataclass
class Compiled:
    program: Program
    signature: Signature
    run: Callable
    key_input: bool = False      # the program reads the derived random key as its last input
    sites: Tuple[str, ...] = ()  # names of its random sites, in program order (docs/RANDOM.md)


# ----------------------------------------------------------------------------- the decorator


def _annotation_tree(ann, resolve) -> Optional[Dict[str, Any]]:
    """The structure an annotation declares, as a pytree spec whose leaves hold their TensorSpec, or
    None when there is no Tensor[...] in it (a static parameter). Accepted: ``{"w": Tensor[..]}``,
    ``[Tensor[..], ...]``, ``(Tensor[..], ...)``, and a dataclass or NamedTuple class whose fields
    are annotated with Tensor[...] (no registration needed)."""
    ann = resolve(ann)
    if isinstance(ann, TensorSpec):
        return {"kind": "leaf", "spec": ann}
    kids = None
    if isinstance(ann, dict):
        kids = [_annotation_tree(v, resolve) for v in ann.values()]
        node = {"kind": "dict", "keys": list(ann.keys())}
    elif isinstance(ann, (list, tuple)):
        kids = [_annotation_tree(v, resolve) for v in ann]
        node = {"kind": "list" if isinstance(ann, list) else "tuple"}
    elif isinstance(ann, type) and (dataclasses.is_dataclass(ann) or (issubclass(ann, tuple) and hasattr(ann, "_fields"))):
        names = [f.name for f in dataclasses.fields(ann)] if dataclasses.is_dataclass(ann) else list(ann._fields)
        hints = {}
        for klass in reversed(ann.__mro__):
            hints.update(getattr(klass, "__annotations__", {}))
        kids = [_annotation_tree(_field_annotation(ann, hints.get(n)), resolve) for n in names]
        node = {"kind": "dataclass" if dataclasses.is_dataclass(ann) else "namedtuple", "name": ann.__name__,
                "module": ann.__module__, "fields": names, "cls": ann}
    if kids is None or all(k is None for k in kids):
        return None
    if any(k is None for k in kids):
        raise TypeError("every leaf of a structured parameter must be annotated Tensor[...]; mixing tensors "
                        "and static values in one structure is not supported")
    node["children"] = kids
    return node


def _field_annotation(cls, ann):
    if isinstance(ann, str):
        import sys
        env = dict(vars(sys.modules.get(cls.__module__, object())))
        try:
            return eval(ann, env)  # noqa: S307 (the user's own annotation)
        except Exception:
            return None
    return ann


def _ir_name(path: str) -> str:
    """An IR value name for a leaf path: params['w'][0] -> params_w_0."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", path)).strip("_") or "x"


def _tree_leaves(tree, path: str):
    """(path, TensorSpec) of every leaf of an annotation tree, in order."""
    return list(zip(pytree.leaf_paths(tree, path), _leaf_specs(tree)))


def _leaf_specs(tree):
    if tree["kind"] == "leaf":
        return [tree["spec"]]
    return [s for c in tree.get("children", []) for s in _leaf_specs(c)]


class NativeFunction:
    """A Python function compiled from source to Truestar IR."""

    def __init__(self, fn, dims: Optional[Dict[str, Tuple[int, Optional[int]]]] = None,
                 executor: Optional[Executor] = None, eager: bool = True, seed: int = 0):
        from .. import rng

        self.seed = rng.as_u64(seed)
        self.calls = 0     # call count: random ops draw fresh numbers on every call (settable)
        self.python = fn
        self.name = fn.__name__
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__wrapped__ = fn
        self.src = FunctionSource(fn)
        self.dims = dict(dims or {})
        self._executor = executor
        self.py_signature = inspect.signature(fn)
        self.specs: Dict[str, TensorSpec] = {}
        self.trees: Dict[str, Dict[str, Any]] = {}  # structured (pytree) parameters
        self.static_params: List[str] = []
        self._cache: Dict[Tuple, Compiled] = {}
        self.compile_count = 0
        self._custom_grad = None
        self._parse_signature()
        if eager and all(p.default is not inspect.Parameter.empty
                         for n, p in self.py_signature.parameters.items() if n in self.static_params):
            self._compiled(self._default_static())

    # -------------------------------------------------------------- signature parsing

    def _eval_annotation(self, ann):
        if isinstance(ann, str):
            env = dict(self.src.globals)
            for k, c in self.src.cells.items():
                try:
                    env[k] = c.cell_contents
                except ValueError:
                    pass
            try:
                return eval(ann, env)  # noqa: S307 (the user's own annotation)
            except Exception as e:
                raise self.src.error(self.src.tree, f"cannot evaluate annotation {ann!r}: {e}",
                                     code="bad_annotation") from None
        return ann

    def _parse_signature(self):
        args = self.src.tree.args
        for p in self.py_signature.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise self.src.error(self.src.tree, f"*{p.name} parameters are not supported in a native "
                                                    "function", UnsupportedConstructError)
            ann = self._eval_annotation(p.annotation) if p.annotation is not inspect.Parameter.empty else None
            if isinstance(ann, TensorSpec):
                self.specs[p.name] = ann
                continue
            try:
                tree = _annotation_tree(ann, lambda a: a)
            except TypeError as e:
                raise self.src.error(self.src.tree, f"parameter {p.name}: {e}", code="bad_annotation") from None
            if tree is not None:
                self.trees[p.name] = tree
            else:
                self.static_params.append(p.name)
        self.return_spec = None
        if self.py_signature.return_annotation is not inspect.Signature.empty:
            self.return_spec = self._eval_annotation(self.py_signature.return_annotation)
        del args

    def _default_static(self) -> Dict[str, Any]:
        return {n: self.py_signature.parameters[n].default for n in self.static_params}

    def _symbol_ranges(self) -> Dict[str, Tuple[int, Optional[int]]]:
        out = {}
        leaves = list(self.specs.values()) + [sp for t in self.trees.values() for sp in _leaf_specs(t)]
        for spec in leaves:
            for s in spec.symbols():
                out.setdefault(s, tuple(self.dims.get(s, (1, None))))
        return out

    # -------------------------------------------------------------- compile

    @property
    def executor(self) -> Executor:
        return self._executor or _DEFAULT[0]

    def _key(self, static: Dict[str, Any]) -> Tuple:
        key = []
        for n in self.static_params:
            v = static[n]
            try:
                hash(v)
            except TypeError:
                raise InputError(f"static argument {n}={v!r} must be hashable (it is part of the compile key)",
                                 function=self.name, file=self.src.file, line=self.src.line_of(self.src.tree)) from None
            key.append((n, type(v).__name__, v))
        return tuple(key)

    def _compiled(self, static: Dict[str, Any]) -> Compiled:
        key = self._key(static)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._compile(static)
            self._cache[key] = hit
        return hit

    def _compile(self, static: Dict[str, Any]) -> Compiled:
        b = Builder(self.name)
        ranges = self._symbol_ranges()
        for s, (lo, hi) in ranges.items():
            b.symbol(s, lo, hi)
        scope = Scope(b)
        params = []
        for n in self.py_signature.parameters:
            if n in self.specs:
                spec = self.specs[n]
                v = b.input(n, spec.dtype, spec.shape)
                scope.vars[n] = T(v, b)
                params.append((n, v.type))
            elif n in self.trees:
                leaves = []
                for path, spec in _tree_leaves(self.trees[n], n):
                    v = b.input(_ir_name(path), spec.dtype, spec.shape)
                    leaves.append(T(v, b))
                    params.append((path, v.type))
                scope.vars[n] = pytree.unflatten(self.trees[n], leaves)
            else:
                scope.vars[n] = static[n]
        sites = RandomSites(scope, self.name)
        tr = Translator(self.src, rng=sites)
        r = tr.block(self.src.tree.body, scope)
        if r is None:
            raise self.src.error(self.src.tree, f"{self.name} never returns a value", code="no_return")
        returns_tuple = isinstance(r.value, tuple) and not hasattr(r.value, "_fields")
        try:
            values, out_tree = pytree.flatten(r.value, lambda v: is_tensor(v) or is_number(v), keep_class=True)
        except pytree.StructureError as e:
            raise tr.err(r.node, f"the return value{' ' + e.path if e.path else ''} cannot be returned: "
                                 f"{str(e).split(': ', 1)[-1]}", code="type_error") from None
        outs = []
        for i, v in enumerate(values):
            if not is_tensor(v):
                if is_number(v):
                    v = tr.const(scope, v, None, r.node)
                else:
                    raise tr.err(r.node, f"return value {i} is {v!r}; a native function returns tensors",
                                 code="type_error")
            outs.append(v)
        self._check_return(tr, r.node, outs, returns_tuple, out_tree)
        prog = b.build([o.v for o in outs])
        if self._custom_grad is not None:
            prog = self._with_custom_grad(prog, params, static, ranges)
        self.compile_count += 1
        STATS["compiles"] += 1
        sig = Signature(self.name, params, {k: static[k] for k in self.static_params},
                        [o.type for o in outs], ranges, returns_tuple, dict(self.trees), out_tree)
        if sites.key is not None:
            prog.key = sites.key.v.name
        names = tuple(_random_sites(prog))
        return Compiled(prog, sig, self.executor.prepare(prog), sites.key is not None, names)

    def _check_return(self, tr, node, outs, returns_tuple, out_tree=None):
        spec = self.return_spec
        if spec is None:
            return
        specs = list(spec) if isinstance(spec, (tuple, list)) else [spec]
        if not all(isinstance(s, TensorSpec) for s in specs):
            try:
                want = _annotation_tree(spec, lambda a: a)
            except TypeError:
                return
            if want is None or out_tree is None:
                return
            if pytree.describe(want) != pytree.describe(out_tree):
                raise tr.err(node, f"the return annotation declares {pytree.describe(want)} but the function returns "
                                   f"{pytree.describe(out_tree)}", FrontendShapeError, code="annotation_mismatch")
            for (path, s), o in zip(_tree_leaves(want, "result"), outs):
                tr.check_spec(o, s, node, path)
            return
        if len(specs) != len(outs):
            raise tr.err(node, f"the return annotation declares {len(specs)} tensors but {len(outs)} are returned",
                         FrontendShapeError, code="annotation_mismatch")
        for i, (s, o) in enumerate(zip(specs, outs)):
            tr.check_spec(o, s, node, "the return value" if len(outs) == 1 else f"return value {i}")

    def _with_custom_grad(self, prog, params, static, ranges):
        from .custom_grad import wrap_program
        cg = self._custom_grad
        st = {k: static[k] for k in self.static_params}
        fwd, bwd = cg.regions(list(params), st, ranges)
        cg.verify(fwd, bwd, ranges, self._key(static))
        return wrap_program(prog, fwd, bwd, self.name, cg.src.where(cg.src.tree))

    # -------------------------------------------------------------- public API

    def grad(self, fn=None, *, unverified: Optional[str] = None, domain: Optional[Dict[str, Tuple[float, float]]] = None,
             sizes: Optional[Sequence[Dict[str, int]]] = None, seed: int = 0):
        """Attach a hand written gradient, checked now against the function (docs/AUTODIFF.md).

        fn(<primal args by name>, g, out=None) returns one gradient per floating point tensor
        parameter. domain={"x": (lo, hi)} sets where inputs are drawn (default (-2, 2)),
        sizes=[{"N": 5}] the symbol values (default 3 and 2). unverified="reason" skips the
        numeric check for a rule that differs on purpose. Raises CustomGradientError for a wrong rule.
        """
        if unverified is not None and not (isinstance(unverified, str) and unverified.strip()):
            raise InputError("unverified= must be a non empty reason string", function=self.name,
                             file=self.src.file, line=self.src.line_of(self.src.tree))

        def attach(f):
            from .custom_grad import CustomGrad
            if self._custom_grad is not None:
                raise InputError(f"{self.name} already has a custom gradient", function=self.name,
                                 file=self.src.file, line=self.src.line_of(self.src.tree))
            cg = CustomGrad(self, f, unverified, domain, sizes, seed)
            old = dict(self._cache)
            self._custom_grad = cg
            self._cache.clear()
            try:
                for key in old or ([self._key(self._default_static())] if self._eager_ok() else []):
                    self._compiled({n: v for n, _, v in key})
            except Exception:
                self._custom_grad = None
                self._cache = old
                raise
            return self
        return attach(fn) if fn is not None else attach

    def _eager_ok(self) -> bool:
        return all(self.py_signature.parameters[n].default is not inspect.Parameter.empty for n in self.static_params)

    def grad_report(self, **static):
        """What registration checked for the custom gradient (None without one)."""
        if self._custom_grad is None:
            return None
        full = self._default_static()
        full.update(static)
        self._compiled(full)
        return self._custom_grad.reports.get(self._key(full))


    def specialize(self, **static) -> Compiled:
        """Compile (or fetch from the cache) for these static argument values."""
        full = self._default_static()
        full.update(static)
        missing = [n for n in self.static_params if full.get(n, inspect.Parameter.empty) is inspect.Parameter.empty]
        if missing:
            raise InputError(f"static arguments without a value: {missing}", function=self.name,
                             file=self.src.file, line=self.src.line_of(self.src.tree))
        return self._compiled(full)

    @property
    def program(self) -> Program:
        return self.specialize().program

    @property
    def signature(self) -> Signature:
        return self.specialize().signature

    def ir_text(self, **static) -> str:
        return self.specialize(**static).program.to_text()

    def key_for(self, call: int) -> int:
        """The random key of call number ``call`` (0 based): ``rng.call_key(seed, call)``."""
        from .. import rng

        return rng.call_key(self.seed, call)

    @property
    def sites(self) -> Tuple[str, ...]:
        """The random sites of the default specialization (``fn/dropout#0`` and so on)."""
        return self.specialize().sites

    def __call__(self, *args, **kwargs):
        try:
            bound = self.py_signature.bind(*args, **kwargs)
        except TypeError as e:
            raise InputError(f"bad arguments for {self.name}: {e}", function=self.name, file=self.src.file,
                             line=self.src.line_of(self.src.tree)) from None
        bound.apply_defaults()
        static = {n: bound.arguments[n] for n in self.static_params}
        for n, v in static.items():
            if isinstance(v, torch.Tensor):
                raise InputError(f"{n} is a static (compile time) parameter but received a tensor; annotate it "
                                 f"as Tensor[...] to make it a tensor input", function=self.name,
                                 file=self.src.file, line=self.src.line_of(self.src.tree))
        c = self._compiled(static)
        tensors = self._flat_arguments(c, bound)
        try:
            if getattr(c.run, "binds_symbols", False):
                c.run.bind(tensors)   # a compiled runner checks shapes itself, once per distinct shape
            else:
                reference.bind_symbols(c.program, tensors)
        except ShapeError as e:
            names = {_ir_name(p): p for p, _ in c.signature.params}
            reason = re.sub(r"%(\w+)", lambda m: names.get(m.group(1), m.group(0)), e.reason)
            raise InputError(f"arguments do not match the signature {c.signature}: {reason}",
                             function=self.name, file=self.src.file, line=self.src.line_of(self.src.tree),
                             operands=e.operands) from None
        outs = c.run(tensors)
        if c.signature.out_tree is not None:
            return pytree.unflatten(c.signature.out_tree, outs)
        return tuple(outs) if c.signature.returns_tuple else outs[0]

    def _flat_arguments(self, c: Compiled, bound) -> List[torch.Tensor]:
        """The input tensors in program order. Structured arguments are matched against their
        declared structure; a mismatch names the exact leaf (``params['w']``)."""
        where = dict(function=self.name, file=self.src.file, line=self.src.line_of(self.src.tree))
        types = dict(c.signature.params)
        tensors: List[torch.Tensor] = []
        for n in self.py_signature.parameters:
            if n in self.trees:
                def check(path, x, i):
                    t = types[path]
                    if x.dtype != t.dtype:
                        raise InputError(f"{path} must have dtype {str(t).split('[')[0]} ({t}), got {x.dtype}",
                                         operands=[(f"{path} (declared)", t),
                                                   (f"{path} (given)", f"{x.dtype} {list(x.shape)}")], **where)
                try:
                    tensors += pytree.match(self.trees[n], bound.arguments[n], lambda x: isinstance(x, torch.Tensor),
                                            n, check)
                except pytree.StructureError as e:
                    raise InputError(f"{e} (declared {pytree.describe(self.trees[n])})", **where) from None
                continue
            if n not in self.specs:
                continue
            t = types[n]
            x = bound.arguments[n]
            if not isinstance(x, torch.Tensor) and hasattr(x, "__dlpack__"):
                from ..interop.dlpack import DLPackError, from_dlpack
                try:   # NumPy, CuPy, JAX arrays: zero copy, checked before PyTorch sees the memory
                    x = from_dlpack(x, name=n, expect=t, allow_readonly=True)  # functions never write inputs
                except DLPackError as e:
                    raise InputError(e.message, code=e.code, function=self.name, file=self.src.file,
                                     line=self.src.line_of(self.src.tree), suggestion=e.suggestion or None) from None
            if not isinstance(x, torch.Tensor):
                raise InputError(f"{n} must be a tensor {t}, got {type(x).__name__}", function=self.name,
                                 file=self.src.file, line=self.src.line_of(self.src.tree))
            if x.dtype != t.dtype:
                raise InputError(f"{n} must have dtype {str(t).split('[')[0]} ({t}), got {x.dtype}",
                                 function=self.name, file=self.src.file, line=self.src.line_of(self.src.tree),
                                 operands=[(f"{n} (declared)", t), (f"{n} (given)", f"{x.dtype} {list(x.shape)}")])
            tensors.append(x)
        if c.key_input:   # the derived key: same seed and call number, same numbers, on every backend
            from .. import rng

            tensors.append(torch.tensor(rng.as_i64(self.key_for(self.calls)), dtype=torch.int64))
            self.calls += 1
        return tensors

    def __repr__(self) -> str:
        try:
            return f"<truestar.function {self.signature}>"
        except FrontendError:
            return f"<truestar.function {self.name} (not compiled)>"


def function(fn=None, *, dims: Optional[Dict[str, Tuple[int, Optional[int]]]] = None,
             executor: Optional[Executor] = None, eager: bool = True, seed: int = 0):
    """Compile a typed Python function to Truestar IR from its source.

    ``dims`` gives symbol ranges, for example ``dims={"B": (1, 4096)}``. ``executor`` overrides the
    default (the float64 reference interpreter). ``eager=False`` defers the first compile to the
    first call (shape errors then surface at that call instead of at the decorator). ``seed`` is the
    program seed of its random ops: call n draws with key ``rng.call_key(seed, n)`` (docs/RANDOM.md).
    """
    def wrap(f):
        return NativeFunction(f, dims=dims, executor=executor, eager=eager, seed=seed)
    return wrap(fn) if fn is not None else wrap


native = function  # the Truestar name of the decorator (docs/NN.md); ``function`` stays as an alias


def _random_sites(prog: Program):
    for n in prog.nodes:
        if n.op == "random":
            yield n.attrs["site_name"]
        for v in n.attrs.values():
            if isinstance(v, Program):
                yield from _random_sites(v)


# ----------------------------------------------------------------------------- gradients


class GradFunction:
    """``truestar.grad(fn, wrt=...)``: calls ``fn`` and returns ``(value, grads)``.

    The first tensor ``fn`` returns must be a scalar (a loss); it is differentiated with respect to
    the parameters named in ``wrt``. A gradient has the structure of its parameter: a dict of
    tensors gives a dict of gradients. Loops and branches in ``fn`` are differentiated as written
    (one program for every length, see docs/AUTODIFF.md); ``loop_memory`` picks how loops keep what
    their backward needs ("auto", "store", "recompute", "sqrt").
    """

    def __init__(self, fn: "NativeFunction", wrt, loop_memory: str = "auto"):
        if not isinstance(fn, NativeFunction):
            raise TypeError(f"truestar.grad takes a @truestar.function, got {type(fn).__name__}")
        self.fn = fn
        self.single = isinstance(wrt, str)
        self.wrt = [wrt] if self.single else list(wrt)
        tensor_params = list(fn.specs) + list(fn.trees)
        bad = [w for w in self.wrt if w not in tensor_params]
        if bad or not self.wrt:
            raise InputError(f"truestar.grad: wrt names {bad or '[]'}, but the tensor parameters of {fn.name} are "
                             f"{tensor_params}", function=fn.name, file=fn.src.file, line=fn.src.line_of(fn.src.tree))
        self.loop_memory = loop_memory
        self._cache: Dict[Tuple, Tuple[Program, Callable, List[List[int]]]] = {}
        self.__name__ = f"grad_{fn.name}"

    def _leaves_of(self, c: Compiled, name: str) -> List[int]:
        return [i for i, (path, _) in enumerate(c.signature.params)
                if path == name or path.startswith(name + ".") or path.startswith(name + "[")]

    def _prepared(self, c: Compiled, key):
        hit = self._cache.get(key)
        if hit is None:
            from ..train.ops import grad as ir_grad  # the IR rules plus cast and log_softmax
            out0 = c.program.outputs[0]
            if out0.type.rank != 0 or not out0.type.dtype.is_floating_point:
                raise InputError(f"truestar.grad needs {self.fn.name} to return a float scalar first (a loss); its "
                                 f"first result is {out0.type}", function=self.fn.name, file=self.fn.src.file,
                                 line=self.fn.src.line_of(self.fn.src.tree))
            groups = [self._leaves_of(c, w) for w in self.wrt]
            names = [c.program.inputs[i].name for g in groups for i in g]
            gp = ir_grad(c.program, wrt=names, of=[0], seed="auto", loop_memory=self.loop_memory,
                         name=f"grad_{c.program.name}")
            hit = (gp, self.fn.executor.prepare(gp), groups)
            self._cache[key] = hit
        return hit

    def program(self, **static) -> Program:
        """The gradient program (forward and backward in one IR program)."""
        c = self.fn.specialize(**static)
        return self._prepared(c, self.fn._key({**self.fn._default_static(), **static}))[0]

    def __call__(self, *args, **kwargs):
        fn = self.fn
        bound = fn.py_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        static = {n: bound.arguments[n] for n in fn.static_params}
        c = fn._compiled(static)
        gp, run, groups = self._prepared(c, fn._key(static))
        tensors = fn._flat_arguments(c, bound)
        try:
            reference.bind_symbols(c.program, tensors)
        except ShapeError as e:
            raise InputError(f"arguments do not match the signature {c.signature}: {e.reason}", function=fn.name,
                             file=fn.src.file, line=fn.src.line_of(fn.src.tree), operands=e.operands) from None
        outs = run(tensors)
        n_out = len(c.program.outputs)
        value = (pytree.unflatten(c.signature.out_tree, outs[:n_out]) if c.signature.out_tree is not None
                 else outs[0])
        flat = outs[n_out:]
        grads, k = [], 0
        for w, g in zip(self.wrt, groups):
            part = flat[k:k + len(g)]
            k += len(g)
            grads.append(pytree.unflatten(fn.trees[w], part) if w in fn.trees else part[0])
        return value, (grads[0] if self.single else tuple(grads))


def grad(fn: "NativeFunction", wrt, loop_memory: str = "auto") -> GradFunction:
    """Gradients of a native function's scalar result: ``value, g = truestar.grad(loss, wrt="params")(...)``."""
    return GradFunction(fn, wrt, loop_memory)



class ValueAndGrad:
    """``truestar.value_and_grad(fn, wrt, has_aux=False)``, JAX's convention on top of
    ``truestar.grad``: without ``has_aux`` ``fn`` must return only its scalar loss and a call gives
    ``(loss, grads)``; with ``has_aux`` ``fn`` returns ``(loss, aux)`` (aux any structure of tensors,
    not differentiated) and a call gives ``((loss, aux), grads)``."""

    def __init__(self, fn: "NativeFunction", wrt, has_aux: bool = False, loop_memory: str = "auto"):
        self.grad_fn = GradFunction(fn, wrt, loop_memory)
        self.has_aux = bool(has_aux)
        self.__name__ = f"value_and_grad_{fn.name}"

    def _refuse(self, why: str):
        fn = self.grad_fn.fn
        raise InputError(f"value_and_grad({fn.name}, has_aux={self.has_aux}): {why}", function=fn.name,
                         file=fn.src.file, line=fn.src.line_of(fn.src.tree))

    def program(self, **static) -> Program:
        return self.grad_fn.program(**static)

    def __call__(self, *args, **kwargs):
        value, grads = self.grad_fn(*args, **kwargs)
        if self.has_aux:
            if not (isinstance(value, tuple) and len(value) == 2):
                self._refuse("with has_aux the function must return a pair (loss, aux)")
            return value, grads
        if not isinstance(value, torch.Tensor):
            self._refuse("the function returns more than its loss; pass has_aux=True and return (loss, aux)")
        return value, grads


def value_and_grad(fn: "NativeFunction", wrt, has_aux: bool = False, loop_memory: str = "auto") -> ValueAndGrad:
    """``loss, g = truestar.value_and_grad(f, wrt="p")(...)``; with ``has_aux=True``,
    ``(loss, aux), g = ...`` (see ``ValueAndGrad``)."""
    return ValueAndGrad(fn, wrt, has_aux, loop_memory)

# ----------------------------------------------------------------------------- emitting into a builder


def _map_structure(obj, fn, is_leaf):
    if is_leaf(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_structure(v, fn, is_leaf) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_structure(v, fn, is_leaf) for v in obj]
    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return tuple(_map_structure(v, fn, is_leaf) for v in obj)
    return obj


def emit_call(builder, fn, *args, **kwargs):
    """Translate a call of ``fn`` (a native function, a ``truestar.nn`` inline function or layer, or
    anything with ``__truestar_inline__``) into ``builder``, an ``ir.Builder`` or a builder like object
    (``truestar.fitting`` uses an editor of an existing program). Arguments may be IR ``Value``s,
    structures of them, or static Python values; the result has the same form, with ``Value`` leaves.
    The body is translated exactly as an inlined call inside a native function would be."""
    from ..ir.core import Value

    lead = ()
    if not isinstance(fn, NativeFunction):
        hook = getattr(fn, "__truestar_inline__", None)
        if hook is None:
            raise TypeError(f"emit_call needs a native function or an object with __truestar_inline__, got {fn!r}")
        fn, lead = hook()
    try:
        bound = fn.py_signature.bind(*lead, *args, **kwargs)
    except TypeError as e:
        raise InputError(f"bad arguments for {fn.name}: {e}", function=fn.name, file=fn.src.file,
                         line=fn.src.line_of(fn.src.tree)) from None
    bound.apply_defaults()
    root = Scope(builder)
    for pname, v in bound.arguments.items():
        root.vars[pname] = _map_structure(v, lambda x: T(x, builder), lambda x: isinstance(x, Value))
    tr = Translator(fn.src, 1, None)
    r = tr.block(fn.src.tree.body, root)
    if r is None:
        raise tr.err(fn.src.tree, f"{fn.name} returns no value", code="no_return")
    return _map_structure(r.value, lambda t: t.v, is_tensor)
