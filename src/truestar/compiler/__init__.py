"""Truestar compiler: the tensor IR lowered to verified kernels.

    from truestar.compiler import CompiledExecutor, compile_program

    cp = compile_program(program, "cuda", "float32")
    outs = cp([x])                 # first call is checked against the float64 reference interpreter
    print(cp.report)               # kernels, library kernels and every fallback with its reason

See docs/COMPILER.md.
"""

from .executor import CompiledExecutor, compile_program, export_executor, register_export_executors
from .lower import NotLowerable, Options, RULES
from .checks import SentinelError
from .report import CompileReport
from .runtime import STATS, CompiledProgram, FallbackNotAllowed, ProgramMismatchError

register_export_executors()

__all__ = ["SentinelError", "CompiledExecutor", "compile_program", "CompiledProgram", "CompileReport", "ProgramMismatchError",
           "FallbackNotAllowed", "NotLowerable", "Options", "RULES", "STATS", "export_executor",
           "register_export_executors"]
