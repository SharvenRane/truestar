"""Per layer precision planner (`truestar compile --precision auto`, docs/PRECISION.md)."""

from .planner import Budget, PlannedModel, PrecisionPlan, apply, plan, score
from .units import PRECISIONS, PlanError

__all__ = ["Budget", "PRECISIONS", "PlanError", "PlannedModel", "PrecisionPlan", "apply", "plan",
           "score"]
