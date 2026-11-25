"""Native functions as PyTorch custom ops, with no C++ toolchain (pain row P17).

    @truestar.torch_op
    def f(x: Tensor["B", 64], w: Tensor[64, 64]) -> Tensor["B", 64]: ...

See docs/CUSTOM_OPS.md.
"""

from .executor import (Plan, ProgramMismatchError, SlowPathWarning, clear_plans, execute, plan,
                       register_executor, unregister_executor)
from .lower import NotLowerable, lower
from .op import CustomOpError, TorchOp, torch_op

__all__ = ["torch_op", "TorchOp", "CustomOpError", "execute", "plan", "Plan", "register_executor",
           "unregister_executor", "clear_plans", "SlowPathWarning", "ProgramMismatchError", "NotLowerable",
           "lower"]
