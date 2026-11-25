"""Export is compile: the artifact that was verified is the artifact that ships.

* ``export(model, args, "model.tsx")``: torch.export import, float64 verification against PyTorch
  (example batch and a second batch size), then the artifact with its evidence. Refuses to write
  anything that was not verified.
* ``load("model.tsx")``: a ``Runner`` that executes the artifact with no model class and no
  torch.export.

See docs/EXPORT.md. The file format is documented in ``truestar.ir.serialize``.
"""

from .exporter import (DEFAULT_MAX_BATCH, DEFAULT_TOLERANCE, ExportError, ExportResult, VerificationError,
                       export, inputs_sha256)


def load(path, **kw):
    """Load an artifact for running (see truestar.export.runner.Runner)."""
    from .runner import load as _load
    return _load(path, **kw)


def __getattr__(name):
    if name in ("Runner", "register_executor", "EXECUTORS"):
        from . import runner
        return getattr(runner, name)
    raise AttributeError(name)


__all__ = ["export", "load", "ExportError", "VerificationError", "ExportResult", "inputs_sha256",
           "DEFAULT_TOLERANCE", "DEFAULT_MAX_BATCH", "Runner", "register_executor"]
