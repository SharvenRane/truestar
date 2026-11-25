"""Truestar native front end: typed Python functions compiled ahead of time from source.

See docs/FRONTEND.md.
"""

from . import ops
from .errors import (FrontendError, FrontendNameError, FrontendShapeError, InputError, StaticValueError,
                     UnsupportedConstructError)
from .function import (Compiled, Executor, GradFunction, NativeFunction, ReferenceExecutor, Signature,
                       ValueAndGrad, compile_count, function, grad, native, set_default_executor,
                       value_and_grad)
from .types import Tensor, TensorSpec
from .custom_grad import CustomGradientError, GradReport

__all__ = [
    "ops", "function", "native", "grad", "value_and_grad", "ValueAndGrad", "GradFunction", "Tensor", "TensorSpec", "NativeFunction", "Signature", "Compiled", "Executor",
    "ReferenceExecutor", "set_default_executor", "compile_count", "FrontendError", "FrontendShapeError",
    "UnsupportedConstructError", "FrontendNameError", "StaticValueError", "InputError", "CustomGradientError",
    "GradReport",
]
