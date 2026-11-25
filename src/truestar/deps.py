"""Optional dependencies, checked before a path that needs them runs (code TS022).

Truestar's core needs only PyTorch. Some fast paths need more: the CPU kernels of the kernel layer
and the compiled executor are generated numba code (the `cpu-kernels` extra). When such a module is
missing, the user gets a ``MissingDependencyError`` that names the module, the extra that installs
it and the command to run, never a raw ``ModuleNotFoundError`` from inside a generated file.

``MissingDependencyError`` subclasses ``ModuleNotFoundError``, so code that already catches
``ImportError`` keeps working. ``to_dict()`` gives the same facts in the agent JSON style
(docs/AGENT_INTERFACE.md): a stable code, its slug and severity, and ``next_actions``.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional

from .codes import severity, slug

CODE = "TS022"

# module -> (pip extra, what it is used for)
OPTIONAL: Dict[str, tuple] = {
    "numba": ("cpu-kernels", "The CPU backend of the kernel layer and of the compiled executor"),
}


def install_command(extra: str) -> str:
    return f'pip install "truestar[{extra}]"'


class MissingDependencyError(ModuleNotFoundError):
    """An optional dependency a requested path needs is not installed (TS022)."""

    code = CODE

    def __init__(self, module: str, purpose: Optional[str] = None, extra: Optional[str] = None,
                 fallback: Optional[str] = None, package: Optional[str] = None):
        known_extra, known_purpose = OPTIONAL.get(module, (None, None))
        self.module = module
        self.extra = extra or known_extra
        self.purpose = purpose or known_purpose or "This path"
        self.fallback = fallback
        # package: the pip name when it differs from the import name (ai-edge-litert, ai_edge_litert)
        self.command = install_command(self.extra) if self.extra else f"pip install {package or module}"
        msg = (f"{CODE} {slug(CODE)}: {self.purpose} uses the optional module '{module}', which "
               f"is not installed. Install it with: {self.command}")
        if fallback:
            msg += f". Or {fallback}"
        super().__init__(msg, name=module)

    def to_dict(self) -> Dict[str, Any]:
        actions = [{"kind": "install_extra",
                    "description": f"Install the optional module '{self.module}'"
                                   + (f" (the '{self.extra}' extra)." if self.extra else "."),
                    "command": self.command}]
        if self.fallback:
            actions.append({"kind": "use_reference", "description": self.fallback[:1].upper()
                            + self.fallback[1:] + ".", "command": None})
        return {"code": CODE, "slug": slug(CODE), "severity": severity(CODE),
                "message": str(self), "module": self.module, "extra": self.extra,
                "next_actions": actions}


def available(module: str) -> bool:
    """True when ``module`` can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, purpose: Optional[str] = None, fallback: Optional[str] = None) -> None:
    """Raise ``MissingDependencyError`` (TS022) unless ``module`` can be imported."""
    if not available(module):
        raise MissingDependencyError(module, purpose, fallback=fallback)
