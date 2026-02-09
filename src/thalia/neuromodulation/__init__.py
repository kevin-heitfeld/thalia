"""Neuromodulation systems for Thalia.

Provides global neuromodulator management:
- Locus coeruleus norepinephrine system (arousal/uncertainty)
- Nucleus basalis acetylcholine system (attention/encoding)
- Neuromodulator coordination (DA-NE-ACh interactions)
"""

from __future__ import annotations

from .homeostasis import NeuromodulatorHomeostasis, NeuromodulatorHomeostasisConfig
from .locus_coeruleus import LocusCoeruleusConfig, LocusCoeruleusSystem
from .nucleus_basalis import NucleusBasalisConfig, NucleusBasalisSystem

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
]
