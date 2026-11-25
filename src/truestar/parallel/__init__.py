"""`truestar parallel`: one training step under a parallel layout against the single device step.

See docs/PARALLEL.md. Ranks are simulated on the CPU (threads, or gloo processes).
"""

from .check import TOL, Finding, ParallelReport, parallel_check
from .layout import Layout, LayoutError, Style, auto_plan, parse_layout, parse_style
from .runtime import FAULTS, MSE, CrossEntropy

__all__ = ["TOL", "Finding", "ParallelReport", "parallel_check", "Layout", "LayoutError", "Style", "auto_plan",
           "parse_layout", "parse_style", "FAULTS", "MSE", "CrossEntropy"]
