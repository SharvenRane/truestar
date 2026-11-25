"""`truestar range`: predict float16, bfloat16 and float8 range failures before deployment, on CPU,
and verify a remedy by emulation (docs/RANGE.md)."""

from .analysis import RangeReport, Site, range_check
from .formats import FORMATS, TARGETS, Format, Target, get_target

__all__ = ["range_check", "RangeReport", "Site", "FORMATS", "TARGETS", "Format", "Target", "get_target"]
