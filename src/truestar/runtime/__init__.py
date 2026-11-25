"""Truestar runtime: host overhead measurement and CUDA graph capture for any executor.

    from truestar.runtime import capture
    step = capture(compiled_fn, (x, w))     # records the launches once
    y = step(x2, w2)                        # one graph launch, bitwise equal to compiled_fn(x2, w2)

See docs/RUNTIME.md.
"""

from .graphs import (BucketError, CaptureError, CaptureMismatchError, CapturedFunction, Executor,
                     HostSyncError, InputMutationError, bitwise_equal, capture)
from . import timing

__all__ = ["capture", "CapturedFunction", "Executor", "CaptureError", "HostSyncError",
           "InputMutationError", "BucketError", "CaptureMismatchError", "bitwise_equal", "timing"]
