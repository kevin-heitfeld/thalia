"""
Neuromodulator systems: VTA (dopamine), locus coeruleus (norepinephrine), nucleus basalis (acetylcholine).

These systems provide global neuromodulatory signals that influence learning,
attention, and arousal throughout the brain.
"""

from thalia.neuromodulation.systems.vta import VTADopamineSystem, VTAConfig
from thalia.neuromodulation.systems.locus_coeruleus import (
    LocusCoeruleusSystem,
    LocusCoeruleusConfig,
)
from thalia.neuromodulation.systems.nucleus_basalis import (
    NucleusBasalisSystem,
    NucleusBasalisConfig,
)

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
