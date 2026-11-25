"""Compile errors of the native front end.

Every error carries the location in the user's own function (file, line, column, source line),
the op, the operand types and, where the fix is obvious, a suggestion. ``to_dict()`` gives the same
facts as plain JSON compatible data, so a coding agent can repair the code without parsing prose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..ir.core import IRError, ShapeError, TensorType, UnsupportedOpError

SCHEMA = "truestar.frontend.error/1"


def _type_dict(t: Any) -> Dict[str, Any]:
    if isinstance(t, TensorType):
        return {"type": str(t), "dtype": str(t).split("[")[0], "shape": [str(d) for d in t.shape]}
    return {"type": str(t)}


class FrontendError(IRError):
    """Base class. ``code`` is a stable machine readable identifier."""

    code = "frontend_error"

    def __init__(self, message: str, *, code: Optional[str] = None, function: str = "",
                 file: str = "", line: Optional[int] = None, col: Optional[int] = None,
                 source_line: str = "", op: Optional[str] = None,
                 operands: Sequence[Tuple[str, Any]] = (), suggestion: Optional[str] = None):
        if code:
            self.code = code
        self.message = message
        self.function = function
        self.file = file
        self.line = line
        self.col = col
        self.source_line = source_line
        self.op = op
        self.operands = list(operands)
        self.suggestion = suggestion
        Exception.__init__(self, self.to_text())

    # ShapeError compatible attributes
    @property
    def reason(self) -> str:
        return self.message

    @property
    def source(self) -> str:
        return self.location()

    def location(self) -> str:
        where = f"{self.file}:{self.line}" if self.line is not None else self.file
        return f"{where} in {self.function}" if self.function else where

    def to_text(self) -> str:
        kind = self.code.replace("_", " ")
        head = f"{kind}"
        if self.op:
            head += f" in {self.op}"
        lines = [f"{head}: {self.message}", f"  at {self.location()}"]
        if self.source_line:
            lines.append(f"  {self.line} | {self.source_line.rstrip()}")
        for label, t in self.operands:
            lines.append(f"  {label}: {t}")
        if self.suggestion:
            lines.append(f"  fix: {self.suggestion}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "function": self.function,
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "source_line": self.source_line.strip(),
            "op": self.op,
            "operands": [dict(name=label, **_type_dict(t)) for label, t in self.operands],
            "suggestion": self.suggestion,
        }


class FrontendShapeError(FrontendError, ShapeError):
    """A shape or dtype mismatch found at compile time, before any data exists."""

    code = "shape_mismatch"


class UnsupportedConstructError(FrontendError, UnsupportedOpError):
    """Python the front end does not translate. Raised with the line, never skipped."""

    code = "unsupported_construct"


class FrontendNameError(FrontendError):
    code = "undefined_name"


class StaticValueError(FrontendError):
    """A value that must be known at compile time is not (for example a branch on a symbolic dim)."""

    code = "not_static"


class InputError(FrontendError, ShapeError):
    """Arguments passed to a compiled function do not match its typed signature."""

    code = "input_mismatch"


def to_dicts(errors: List[FrontendError]) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in errors]
