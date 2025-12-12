"""Homeostatic Mechanisms.

Stability mechanisms including unified homeostasis, synaptic scaling,
intrinsic plasticity, and metabolic constraints.
"""

# Import from synaptic_homeostasis (main file)
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
from thalia.learning.homeostasis.homeostatic_regulation import (
    HomeostaticConfig,
    HomeostaticRegulator,
    NeuromodulatorCoordination,
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
    # Homeostatic Regulation
    "HomeostaticConfig",
    "HomeostaticRegulator",
    "NeuromodulatorCoordination",
]
