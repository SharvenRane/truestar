"""Truestar tensor IR (Stage 2 foundation).

* ``core``: shape typed values, pure nodes, programs, the printer and a builder.
* ``shapes``: the shape and dtype rule of every op, with symbolic dims.
* ``importer``: torch.export (ATen) graphs to IR, loud on anything unsupported.
* ``reference``: the float64 reference interpreter, written from textbook formulas.
* ``autodiff``: reverse mode autodiff as a program transformation (``grad``), see docs/AUTODIFF.md.
* ``custom_vjp``: the op that carries a user written backward (docs/AUTODIFF.md, custom gradients).
* ``vmap``, ``forward``: batching and forward mode (``jvp``, ``jacfwd``, ``jacrev``, ``hessian``,
  ``linearize``) as program transformations, see docs/TRANSFORMS.md.

See docs/IR.md.
"""

from .core import (Builder, Dim, IRError, Node, Program, ShapeError, TensorType, UnsupportedOpError,
                   Value, symbol)
from .importer import import_exported, import_module, initial_state, supported_aten_ops
from .reference import run
from .shapes import RULES as _RULES
from . import custom_vjp as _custom_vjp  # noqa: F401  registers the custom_vjp op and its gradient rule
from . import random_batched as _random_batched  # noqa: F401  registers random_batched (vmap of random)


def ir_ops():
    """Names of every op the IR defines."""
    return sorted(_RULES)


__all__ = [
    "Builder", "Dim", "IRError", "Node", "Program", "ShapeError", "TensorType", "UnsupportedOpError",
    "Value", "symbol", "import_exported", "import_module", "initial_state", "supported_aten_ops", "run", "ir_ops",
    "grad", "vmap", "jvp", "jacfwd", "jacrev", "hessian", "linearize",
]


def __getattr__(name):
    # Loaded on first use: importing autodiff registers its extra ops (erf) and rules.
    if name == "grad":
        from .autodiff import grad

        return grad
    if name == "vmap":
        from .vmap import vmap

        return vmap
    if name in ("jvp", "jacfwd", "jacrev", "hessian", "linearize"):
        from . import forward

        return getattr(forward, name)
    raise AttributeError(f"module 'truestar.ir' has no attribute {name!r}")
