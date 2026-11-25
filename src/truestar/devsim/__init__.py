"""Simulated device numerics (docs/DEVICES.md).

* :mod:`.profiles`: one profile per device, every rule cited (``sim:ane``, ``sim:webgpu-f16``, ...)
* :mod:`.mode`: ``simulate(profile)``, the dispatch mode behind the ``sim:*`` check backends
* :mod:`.sweep`: ``truestar check --sweep``, bisect a size to the first divergent one
* :mod:`.specials`: ``truestar probe --specials``, NaN, inf, subnormal, minus zero and casts
"""

from .profiles import PROFILES, Profile, Rule, get_profile
from .mode import SimMode, simulate

__all__ = ["PROFILES", "Profile", "Rule", "get_profile", "SimMode", "simulate"]
