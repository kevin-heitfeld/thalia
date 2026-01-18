"""
Neuromodulation systems and management.

This module provides neuromodulator systems (VTA, locus coeruleus, nucleus basalis)
and management infrastructure for global neuromodulatory signals.
"""

from __future__ import annotations

from thalia.neuromodulation.homeostasis import (
    NeuromodulatorHomeostasis,
    NeuromodulatorHomeostasisConfig,
)
from thalia.neuromodulation.manager import NeuromodulatorManager
from thalia.neuromodulation.mixin import NeuromodulatorMixin
from thalia.neuromodulation.systems import (
    VTA,
    LocusCoeruleus,
    LocusCoeruleusConfig,
    LocusCoeruleusSystem,
    NucleusBasalis,
    NucleusBasalisConfig,
    NucleusBasalisSystem,
    VTAConfig,
    VTADopamineSystem,
)

__all__ = [
    # Locus Coeruleus (Norepinephrine)
    "LocusCoeruleus",
    "LocusCoeruleusConfig",
    "LocusCoeruleusSystem",
    # Nucleus Basalis (Acetylcholine)
    "NucleusBasalis",
    "NucleusBasalisConfig",
    "NucleusBasalisSystem",
    # VTA (Dopamine)
    "VTA",
    "VTAConfig",
    "VTADopamineSystem",
    # Manager and utilities
    "NeuromodulatorHomeostasis",
    "NeuromodulatorHomeostasisConfig",
    "NeuromodulatorManager",
    "NeuromodulatorMixin",
]
