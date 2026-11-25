"""Verified remedies for what `truestar check` finds (docs/FIX.md).

    report = truestar.fix.fix(model, x)          # find, apply and verify remedies
    with truestar.fix.patched(report):           # scoped: only the blamed op, only on its device
        y = model(x)
    handle = truestar.fix.apply(report, model=model)   # permanent opt in until handle.remove()

    plan = truestar.fix.select_precision(model, x, calib)   # INT8: fewest layers kept in float

Names load on first use, so the command line parser can import truestar.fix.cli without torch.
"""

_LAZY = {
    "fix": "engine", "FixReport": "engine", "time_forward": "engine", "verify_op": "engine",
    "patched": "mode", "apply": "mode", "Remedy": "mode", "RemedyMode": "mode",
    "LibraryOverride": "mode",
    "select_precision": "precision", "PrecisionPlan": "precision", "Int8Simulator": "precision",
    "TABLE": "table", "Decomposition": "table", "candidates": "table", "table_rows": "table",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'truestar.fix' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
