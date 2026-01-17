"""
Neuromodulator systems: VTA (dopamine), locus coeruleus (norepinephrine), nucleus basalis (acetylcholine).

These systems provide global neuromodulatory signals that influence learning,
attention, and arousal throughout the brain.
"""

from __future__ import annotations

from thalia.neuromodulation.systems.locus_coeruleus import (
    LocusCoeruleusConfig,
    LocusCoeruleusSystem,
)
from thalia.neuromodulation.systems.nucleus_basalis import (
    NucleusBasalisConfig,
    NucleusBasalisSystem,
)
from thalia.neuromodulation.systems.vta import VTAConfig, VTADopamineSystem

# Convenient aliases
VTA = VTADopamineSystem
LocusCoeruleus = LocusCoeruleusSystem
NucleusBasalis = NucleusBasalisSystem

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
]
