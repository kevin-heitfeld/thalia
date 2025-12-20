"""
Neuromodulation systems and management.

This module provides neuromodulator systems (VTA, locus coeruleus, nucleus basalis)
and management infrastructure for global neuromodulatory signals.
"""

from thalia.neuromodulation.systems import (
    VTADopamineSystem,
    VTA,
    VTAConfig,
    LocusCoeruleusSystem,
    LocusCoeruleus,
    LocusCoeruleusConfig,
    NucleusBasalisSystem,
    NucleusBasalis,
    NucleusBasalisConfig,
)
from thalia.neuromodulation.manager import NeuromodulatorManager
from thalia.neuromodulation.homeostasis import (
    NeuromodulatorHomeostasis,
    NeuromodulatorHomeostasisConfig,
)
from thalia.neuromodulation.mixin import NeuromodulatorMixin

__all__ = [
    # VTA (Dopamine)
    "VTADopamineSystem",
    "VTA",
    "VTAConfig",
    # Locus Coeruleus (Norepinephrine)
    "LocusCoeruleusSystem",
    "LocusCoeruleus",
    "LocusCoeruleusConfig",
    # Nucleus Basalis (Acetylcholine)
    "NucleusBasalisSystem",
    "NucleusBasalis",
    "NucleusBasalisConfig",
    # Manager and utilities
    "NeuromodulatorManager",
    "NeuromodulatorHomeostasis",
    "NeuromodulatorHomeostasisConfig",
    "NeuromodulatorMixin",
]
