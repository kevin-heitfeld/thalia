"""
Neuromodulation systems and management.

This module provides neuromodulator systems (VTA, locus coeruleus, nucleus basalis)
and management infrastructure for global neuromodulatory signals.
"""

from thalia.constants.neuromodulation import (
    ACH_BASELINE,
    DA_BASELINE_STANDARD,
    DA_BASELINE_STRIATUM,
    NE_BASELINE,
    NE_GAIN_MIN,
    NE_GAIN_MAX,
    compute_ne_gain,
    decay_constant_to_tau,
    tau_to_decay_constant,
)
from thalia.neuromodulation.homeostasis import (
    NeuromodulatorHomeostasis,
    NeuromodulatorHomeostasisConfig,
)
from thalia.neuromodulation.manager import NeuromodulatorManager
from thalia.neuromodulation.mixin import NeuromodulatorMixin
from thalia.neuromodulation.systems import (
    LocusCoeruleus,
    LocusCoeruleusConfig,
    LocusCoeruleusSystem,
    NucleusBasalis,
    NucleusBasalisConfig,
    NucleusBasalisSystem,
    VTA,
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
    # Constants
    "ACH_BASELINE",
    "DA_BASELINE_STANDARD",
    "DA_BASELINE_STRIATUM",
    "NE_BASELINE",
    "NE_GAIN_MIN",
    "NE_GAIN_MAX",
    # Helper functions
    "compute_ne_gain",
    "decay_constant_to_tau",
    "tau_to_decay_constant",
]
