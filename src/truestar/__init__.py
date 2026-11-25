"""Truestar: the same answer on every chip."""

import sys as _sys
import types as _types

__version__ = "0.1.0"

__all__ = [
    "check",
    "grad_check",
    "CheckReport",
    "BackendResult",
    "LayerResult",
    "Backend",
    "available_backends",
    "resolve_backends",
    "__version__",
    "function",
    "native",
    "grad",
    "value_and_grad",
    "nn",
    "optim",
    "models",
    "fit",
    "train_step",
    "Tensor",
    "ops",
    "torch_op",
    "assert_close",
    "close_report",
    "verified_autotune",
]

# Imported on first use, so `truestar mcp` and `truestar codes` start without loading torch
# (a torch import with CUDA commits gigabytes of memory per process). The native front end
# (function, Tensor, ops) is lazy too, so `truestar check` does not pay for it.
_LAZY = {
    "check": "check", "grad_check": "gradcheck", "CheckReport": "check", "BackendResult": "check", "LayerResult": "check",
    "Backend": "backends", "available_backends": "backends", "resolve_backends": "backends",
    "function": "frontend", "native": "frontend", "grad": "frontend", "value_and_grad": "frontend",
    "fit": "fitting", "train_step": "fitting", "Tensor": "frontend", "ops": "frontend",
    "torch_op": "customop",
    "assert_close": "tolerance", "close_report": "tolerance",
    "verified_autotune": "autotune",
}


_SUBPACKAGES = ("nn", "optim", "models")


def __getattr__(name):
    if name in _SUBPACKAGES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'truestar' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


class _Package(_types.ModuleType):
    """When the import system binds the submodule truestar.check onto this package, keep the
    check() function there instead, as the eager `from .check import check` used to."""

    def __setattr__(self, name, value):
        if name in _LAZY and isinstance(value, _types.ModuleType) and hasattr(value, name):
            value = getattr(value, name)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Package
