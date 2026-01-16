"""Homeostatic Mechanisms.

Stability mechanisms including unified homeostasis, synaptic scaling,
intrinsic plasticity, and metabolic constraints.
"""

from __future__ import annotations


from thalia.learning.homeostasis.synaptic_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
    StriatumHomeostasis,
)
from thalia.learning.homeostasis.intrinsic_plasticity import (
    IntrinsicPlasticityConfig,
    IntrinsicPlasticity,
    PopulationIntrinsicPlasticity,
)
from thalia.learning.homeostasis.metabolic import (
    MetabolicConfig,
    MetabolicConstraint,
    RegionalMetabolicBudget,
)

__all__ = [
    # Unified Homeostasis (constraint-based)
    "UnifiedHomeostasis",
    "UnifiedHomeostasisConfig",
    "StriatumHomeostasis",
    # Intrinsic Plasticity
    "IntrinsicPlasticityConfig",
    "IntrinsicPlasticity",
    "PopulationIntrinsicPlasticity",
    # Metabolic Constraints
    "MetabolicConfig",
    "MetabolicConstraint",
    "RegionalMetabolicBudget",
]
