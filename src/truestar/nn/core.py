"""The pieces every layer is made of: immutable configs, explicit parameter trees, keyed initializers.

A layer is a frozen dataclass holding only its configuration (sizes, flags, dtype). It owns no
tensors. Its parameters are a plain nested dict of tensors that the caller holds and passes in:

    lin = nn.Linear(16, 4)
    p = lin.init(seed=0)            # {"w": f32[4, 16], "b": f32[4]}, the same bits on every backend
    y = lin(p, x)                   # eager: runs through PyTorch
    lin.spec                        # {"w": Tensor[4, 16], "b": Tensor[4]}, for a native signature

Inside a native function the same call ``lin(p, x)`` is translated from the source of
``Linear.forward`` (it is inlined, like a call to another native function), so one definition runs
eagerly, compiles, and is differentiated by the IR's autodiff.

Initialization is keyed: every parameter is drawn from ``truestar.rng`` at the key
``fold(seed, path)`` where ``path`` is its place in the tree (``"blocks.0.attn.q.w"``). A draw
depends on nothing else: not on how many layers were built before, not on the device, not on a
global generator. Adding a layer does not change the initial values of any other layer.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import functools
import math
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

from .. import rng
from ..frontend.types import DTYPES, TensorSpec, parse_dim

# ----------------------------------------------------------------------------- initializers


def fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """(fan_in, fan_out) of a weight as torch.nn.init computes them: dims 1 and 0 times the product
    of the remaining (receptive field) dims; a 1D tensor has fan_in = fan_out = its length."""
    if len(shape) == 0:
        raise ValueError("fan in and fan out need a tensor of at least one dim")
    if len(shape) == 1:
        return int(shape[0]), int(shape[0])
    rf = math.prod(shape[2:]) if len(shape) > 2 else 1
    return int(shape[1]) * rf, int(shape[0]) * rf


def gain(nonlinearity: str = "linear", param: Optional[float] = None) -> float:
    """The gain torch.nn.init.calculate_gain gives ``nonlinearity`` (linear, conv, sigmoid 1; tanh
    5/3; relu sqrt 2; leaky_relu sqrt(2 / (1 + a^2)) with a = ``param`` or 0.01; selu 3/4)."""
    if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
                        "conv_transpose3d", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        a = 0.01 if param is None else float(param)
        return math.sqrt(2.0 / (1 + a ** 2))
    if nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"no gain for nonlinearity {nonlinearity!r}")


@dataclasses.dataclass(frozen=True)
class Init:
    """How one parameter is drawn. ``kind`` is "normal" (``scale`` is the standard deviation),
    "uniform" (on [-scale, scale)), "constant" (every entry equals ``scale``), "trunc_normal"
    (standard deviation ``scale``, cut to [``lo``, ``hi``], torch's trunc_normal_ by inverse CDF), or a
    shape dependent scheme whose bound or deviation comes from the parameter's fans at draw time:
    "xavier_uniform", "xavier_normal" (``scale`` is the gain), "kaiming_uniform", "kaiming_normal"
    (``scale`` is the gain, ``mode`` "fan_in" or "fan_out")."""

    kind: str
    scale: float
    lo: float = -2.0
    hi: float = 2.0
    mode: str = "fan_in"

    def resolve(self, shape: Tuple[int, ...]) -> "Init":
        """The plain normal or uniform Init a shape dependent scheme is for this shape."""
        if self.kind in ("xavier_uniform", "xavier_normal"):
            fi, fo = fans(shape)
            std = self.scale * math.sqrt(2.0 / float(fi + fo))
            return Init("uniform", math.sqrt(3.0) * std) if self.kind == "xavier_uniform" else Init("normal", std)
        if self.kind in ("kaiming_uniform", "kaiming_normal"):
            fi, fo = fans(shape)
            std = self.scale / math.sqrt(fi if self.mode == "fan_in" else fo)
            return Init("uniform", math.sqrt(3.0) * std) if self.kind == "kaiming_uniform" else Init("normal", std)
        return self

    def draw(self, shape: Tuple[int, ...], dtype: torch.dtype, key: int, site: str) -> torch.Tensor:
        if self.kind == "constant":
            return torch.full(shape, float(self.scale), dtype=dtype)
        if self.kind.startswith(("xavier", "kaiming")):
            return self.resolve(shape).draw(shape, dtype, key, site)
        # draw in float64 when the parameter is float64, else in float32 (rng's float32 path is
        # computed in float32 arithmetic with fixed rounding), then convert: same bits everywhere
        work = torch.float64 if dtype == torch.float64 else torch.float32
        if self.kind == "normal":
            z = rng.generate("normal", shape, work, key, site, backend="torch")
            return (z * self.scale).to(dtype)
        if self.kind == "uniform":
            u = rng.generate("uniform", shape, work, key, site, backend="torch")
            return ((u * 2.0 - 1.0) * self.scale).to(dtype)
        if self.kind == "trunc_normal":
            # torch.nn.init.trunc_normal_'s method, in float64 from keyed float64 uniforms: map
            # U[0, 1) onto [2 cdf(lo) - 1, 2 cdf(hi) - 1), erfinv, scale, clamp into [lo, hi]
            u = rng.generate("uniform", shape, torch.float64, key, site, backend="torch")
            a = 2.0 * _normal_cdf(self.lo / self.scale) - 1.0
            b = 2.0 * _normal_cdf(self.hi / self.scale) - 1.0
            z = torch.erfinv(a + (b - a) * u) * (self.scale * math.sqrt(2.0))
            return z.clamp(self.lo, self.hi).to(dtype)
        raise ValueError(f"unknown initializer kind {self.kind!r}")


def _normal_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def normal(std: float) -> Init:
    return Init("normal", float(std))


def uniform(bound: float) -> Init:
    return Init("uniform", float(bound))


def constant(value: float) -> Init:
    return Init("constant", float(value))


zeros = constant(0.0)
ones = constant(1.0)


def trunc_normal(std: float = 1.0, lo: float = -2.0, hi: float = 2.0) -> Init:
    """Normal with standard deviation ``std`` (before the cut) truncated to [lo, hi] (absolute
    bounds, as torch.nn.init.trunc_normal_); ViT and timm models start from it."""
    if not lo < hi or not std > 0:
        raise ValueError(f"need lo < hi and std > 0, got {lo}, {hi}, {std}")
    return Init("trunc_normal", float(std), float(lo), float(hi))


def xavier_uniform(gain: float = 1.0) -> Init:
    """U[-b, b] with b = gain sqrt(6 / (fan_in + fan_out)) from the parameter's shape (Glorot)."""
    return Init("xavier_uniform", float(gain))


def xavier_normal(gain: float = 1.0) -> Init:
    """N(0, std^2) with std = gain sqrt(2 / (fan_in + fan_out))."""
    return Init("xavier_normal", float(gain))


def _kaiming(kind, a, mode, nonlinearity):
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    return Init(kind, gain(nonlinearity, a), mode=mode)


def kaiming_uniform(a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Init:
    """U[-b, b] with b = gain sqrt(3 / fan) (He), torch.nn.init.kaiming_uniform_'s arguments.
    ``kaiming_uniform(a=sqrt(5))`` is torch.nn's default Linear and Conv weight bound 1 / sqrt(fan_in)."""
    return _kaiming("kaiming_uniform", a, mode, nonlinearity)


def kaiming_normal(a: float = 0.0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> Init:
    """N(0, std^2) with std = gain / sqrt(fan)."""
    return _kaiming("kaiming_normal", a, mode, nonlinearity)


def fan_in_uniform(fan_in: int) -> Init:
    """Uniform on [-1/sqrt(fan_in), 1/sqrt(fan_in)): the scale torch.nn uses by default for linear and
    conv weights and biases, so trained models start from the same distribution."""
    return uniform(1.0 / math.sqrt(max(int(fan_in), 1)))


@dataclasses.dataclass(frozen=True)
class Param:
    """One parameter of a layer: its shape (ints) and how it is initialized."""

    shape: Tuple[int, ...]
    init: Init


def _dtype(name: str) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; use one of f32, f64, bf16, f16") from None


def param_key(seed: int, path: str) -> int:
    """The key a parameter at ``path`` is drawn with: ``rng.fold(seed, path)``."""
    return rng.fold(rng.as_u64(int(seed)), path)


# ----------------------------------------------------------------------------- inlined callables


class Inline:
    """A plain Python function written in the native subset, usable two ways: called eagerly it runs
    as Python through PyTorch; called inside a native function it is translated from its source
    and inlined. Losses and the layer forwards are Inline."""

    def __init__(self, fn, name: Optional[str] = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.__doc__ = fn.__doc__
        self.__name__ = self.name
        self.__wrapped__ = fn

    @functools.cached_property
    def native(self):
        from ..frontend.function import NativeFunction

        nf = NativeFunction(self.fn, eager=False)
        nf.name = self.name   # random sites inside are named after it (``loss/Linear#0/...``)
        return nf

    def __truestar_inline__(self):
        return self.native, ()

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"<truestar.nn inline {self.name}>"


def inline(fn) -> Inline:
    return Inline(fn)


# ----------------------------------------------------------------------------- the layer base


_NATIVE_METHODS: Dict[Tuple[type, str], Any] = {}


def native_method(cls, name: str = "forward"):
    """The native function translated from ``cls.<name>`` (cached per class). Its first parameter
    is ``self``, which a call binds to the (static) config object."""
    key = (cls, name)
    nf = _NATIVE_METHODS.get(key)
    if nf is None:
        from ..frontend.function import NativeFunction

        nf = NativeFunction(getattr(cls, name), eager=False)
        nf.name = cls.__name__ if name == "forward" else f"{cls.__name__}.{name}"
        _NATIVE_METHODS[key] = nf
    return nf


def _rule_fn(rules):
    if rules is None or callable(rules):
        return rules
    if not isinstance(rules, dict):
        raise TypeError(f"rules must be a dict of pattern -> Init or a function, got {type(rules).__name__}")
    for k, v in rules.items():
        if not isinstance(v, Init):
            raise TypeError(f"rules[{k!r}] must be an Init, got {type(v).__name__}")

    def rule(path, prm):
        for pat, init in rules.items():
            if fnmatch.fnmatchcase(path, pat):
                return init
        return None

    return rule


class Layer:
    """Base class of every layer. Subclasses are frozen dataclasses and define:

    * ``params()``: ``{name: Param}`` of their own tensors (default none);
    * ``children()``: ``{name: Layer}`` of the layers they are built from (default none);
    * ``state()``: ``{name: Param}`` of carried non trained values, such as batch norm running
      statistics (default none);
    * ``forward(self, ...)``: written in the native subset (``truestar.ops``, static Python on the
      config). It is run eagerly when the layer is called outside a native function and translated
      from its source inside one.

    The trees are derived from those three methods: ``spec`` and ``init`` for parameters,
    ``state_spec`` and ``init_state`` for state. A layer with neither params nor state is called
    without a tree (``nn.GELU()(x)``).
    """

    dtype: str = "f32"
    uses_training = False   # forward takes ``training=`` (dropout, batch norm, containers of them)

    # -- structure
    def params(self) -> Dict[str, Param]:
        return {}

    def children(self) -> Dict[str, "Layer"]:
        return {}

    def state(self) -> Dict[str, Param]:
        return {}

    @property
    def has_params(self) -> bool:
        return bool(self.params()) or any(c.has_params for c in self.parts.values())

    @property
    def has_state(self) -> bool:
        return bool(self.state()) or any(c.has_state for c in self.parts.values())

    # -- trees
    def _tree(self, which: str, fn, prefix: str = ""):
        """``{name: fn(path, Param, dtype)}`` over this layer's own entries and its children's, nested
        like the children. Each entry uses the dtype of the layer that owns it."""
        out = {}
        own = self.params() if which == "params" else self.state()
        dt = _dtype(self.dtype)
        for name, prm in own.items():
            out[name] = fn(prefix + name, prm, dt)
        for name, child in self.parts.items():
            if getattr(child, f"has_{which}"):
                out[name] = child._tree(which, fn, f"{prefix}{name}.")
        return out

    @property
    def spec(self) -> Dict[str, Any]:
        """The parameter tree as an annotation: a nested dict of ``Tensor[...]`` specs."""
        return self._tree("params", lambda path, prm, dt: TensorSpec(tuple(parse_dim(d) for d in prm.shape), dt))

    @property
    def state_spec(self) -> Dict[str, Any]:
        """The state tree (running statistics) as an annotation."""
        return self._tree("state", lambda path, prm, dt: TensorSpec(tuple(parse_dim(d) for d in prm.shape), dt))

    def init(self, seed: int = 0, rules=None) -> Dict[str, Any]:
        """Draw every parameter from its key ``fold(seed, path)``; same bits on every machine.

        ``rules`` overrides initializers: a dict ``{pattern: Init}`` whose first ``fnmatch`` pattern
        matching the path wins (``{"*.w": nn.xavier_uniform(), "*.b": nn.zeros}``), or a function
        ``(path, Param) -> Init or None``. A parameter keeps its key, so overriding one parameter
        never changes the bits of another."""
        rule = _rule_fn(rules)

        def draw(path, prm, dt):
            init = rule(path, prm) if rule is not None else None
            return (init or prm.init).draw(tuple(prm.shape), dt, param_key(seed, path), path)

        return self._tree("params", draw)

    def init_state(self) -> Dict[str, Any]:
        return self._tree("state", lambda path, prm, dt: prm.init.draw(tuple(prm.shape), dt, 0, path))

    @property
    def parts(self) -> Dict[str, "Layer"]:
        """The child layers by name (``children()``, computed once). Inside ``forward`` a child is
        reached as ``self.parts["name"]`` or as the attribute ``self.name``."""
        d = self.__dict__.get("_parts")
        if d is None:
            object.__setattr__(self, "_building", True)
            try:
                d = dict(self.children())
            finally:
                object.__delattr__(self, "_building")
            object.__setattr__(self, "_parts", d)
        return d

    def __getattr__(self, name):
        if name.startswith("_") or "_building" in self.__dict__:
            raise AttributeError(name)
        parts = self.parts
        if name in parts:
            return parts[name]
        raise AttributeError(f"{type(self).__name__} has no attribute or child layer {name!r}")

    def num_params(self) -> int:
        n = 0
        for _, prm in self.named_params():
            n += math.prod(prm.shape)
        return n

    def named_params(self, prefix: str = "") -> Iterator[Tuple[str, Param]]:
        for name, prm in self.params().items():
            yield prefix + name, prm
        for name, child in self.parts.items():
            yield from child.named_params(f"{prefix}{name}.")

    # -- calls
    def __truestar_inline__(self):
        return native_method(type(self)), (self,)

    def __call__(self, *args, **kwargs):
        return type(self).forward(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def layer(cls):
    """Class decorator for layers: a frozen dataclass compared and hashed by its config."""
    return dataclasses.dataclass(frozen=True, eq=True)(cls)
