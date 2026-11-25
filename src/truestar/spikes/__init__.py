"""Training spikes: a float64 shadow of sampled steps (SpikeWatch) and `truestar replay RUN --step N`,
which says whether a spike is numerics, data or the optimizer. See docs/SPIKES.md."""

from .replay import ReplayError, ReplayReport, replay
from .watch import Shadow, SpikeWatch, own_errors, shadow_check, train_step

__all__ = ["ReplayError", "ReplayReport", "replay", "Shadow", "SpikeWatch", "own_errors", "shadow_check",
           "train_step"]
