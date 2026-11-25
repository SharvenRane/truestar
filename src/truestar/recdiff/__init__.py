"""Record a model run and diff it across environments (pain P13, docs/RECORD_DIFF.md).

    rec = truestar.recdiff.record(model, x, device="cuda", ops="all")
    rec.save("before.tsrec")
    # ... upgrade torch, cuDNN, Triton or the driver ...
    report = truestar.recdiff.diff_replay(truestar.recdiff.load("before.tsrec"), model)
    print(report.to_text())

Names are imported on first use, so `truestar codes` and `truestar mcp` (which build the command
line parser, and so import truestar.recdiff.cli) never load torch.
"""

_LAZY = {
    "DIVERGED": "diff", "IDENTICAL": "diff", "NOISE": "diff", "PRECISION_CHANGE": "diff",
    "DiffReport": "diff", "ModuleChange": "diff", "OpChange": "diff", "ReplayError": "diff",
    "classify": "diff", "diff_recordings": "diff", "diff_replay": "diff",
    "differences": "fingerprint", "fingerprint": "fingerprint",
    "Recording": "record", "RecordTooLarge": "record", "load": "record", "record": "record",
    "RecordError": "store",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'truestar.recdiff' has no attribute {name!r}")


import sys as _sys
import types as _types


class _Package(_types.ModuleType):
    """`record` and `fingerprint` are both submodules and functions. When the import system
    binds the submodule onto this package, keep the function, as truestar/__init__.py does."""

    def __setattr__(self, name, value):
        if name in _LAZY and isinstance(value, _types.ModuleType) and hasattr(value, name):
            value = getattr(value, name)
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Package
