"""
Neuromodulation systems and management.

This module provides neuromodulator systems (VTA, locus coeruleus, nucleus basalis)
and management infrastructure for global neuromodulatory signals.
"""

from thalia.neuromodulation.systems import *
from thalia.neuromodulation.systems import __all__ as systems_all
from thalia.neuromodulation.manager import NeuromodulatorManager
from thalia.neuromodulation.homeostasis import (
    NeuromodulatorHomeostasis,
    NeuromodulatorHomeostasisConfig,
)
from thalia.neuromodulation.mixin import NeuromodulatorMixin

__all__ = systems_all + [
    "NeuromodulatorManager",
    "NeuromodulatorHomeostasis",
    "NeuromodulatorHomeostasisConfig",
    "NeuromodulatorMixin",
]
