"""Failure attribution for training runs: CUDA out of memory and distributed hangs.

* :mod:`truestar.diag.memory` names the module instance, phase and allocation behind a CUDA OOM.
* :mod:`truestar.diag.collectives` records every collective per rank and, when ranks stall or
  crash, says which rank is behind and the first collective and source line where ranks disagree.
"""

from .memory import MemoryTracker, OOMReport, MemoryTimeline, track_memory, memory_timeline
from .collectives import CollectiveRecorder, Diagnosis, diagnose, record_collectives

__all__ = [
    "MemoryTracker",
    "OOMReport",
    "MemoryTimeline",
    "track_memory",
    "memory_timeline",
    "CollectiveRecorder",
    "Diagnosis",
    "diagnose",
    "record_collectives",
]
