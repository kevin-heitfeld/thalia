"""
Neuromodulation systems and management.

This module provides neuromodulator systems (VTA, locus coeruleus, nucleus basalis)
and management infrastructure for global neuromodulatory signals.
"""

from __future__ import annotations

from .homeostasis import NeuromodulatorHomeostasis, NeuromodulatorHomeostasisConfig
from .locus_coeruleus import LocusCoeruleusConfig, LocusCoeruleusSystem
from .nucleus_basalis import NucleusBasalisConfig, NucleusBasalisSystem
from .vta import VTAConfig, VTADopamineSystem
from .manager import NeuromodulatorManager

__all__ = [
    # Homeostasis
    "NeuromodulatorHomeostasis",
    "NeuromodulatorHomeostasisConfig",
    # Locus Coeruleus (Norepinephrine)
    "LocusCoeruleusConfig",
    "LocusCoeruleusSystem",
    # Nucleus Basalis (Acetylcholine)
    "NucleusBasalisConfig",
    "NucleusBasalisSystem",
    # VTA (Dopamine)
    "VTAConfig",
    "VTADopamineSystem",
    # Manager
    "NeuromodulatorManager",
]
