"""
Learning rules: STDP, homeostatic mechanisms, and reward-modulated learning.

This module provides:
1. UnifiedHomeostasis - constraint-based stability
2. Learning Strategies - pluggable, composable learning algorithms:
   - HebbianStrategy, STDPStrategy, BCMStrategy
   - ThreeFactorStrategy, ErrorCorrectiveStrategy
   - CompositeStrategy for combining strategies
3. Robustness Mechanisms:
   - IntrinsicPlasticity - threshold adaptation
   - MetabolicConstraint - energy-based regularization
"""

from __future__ import annotations

from .eligibility_trace_manager import (
    EligibilityTraceManager,
    EligibilitySTDPConfig,
)
from .homeostasis.intrinsic_plasticity import (
    IntrinsicPlasticity,
    IntrinsicPlasticityConfig,
    PopulationIntrinsicPlasticity,
)
from .homeostasis.metabolic import (
    MetabolicConfig,
    MetabolicConstraint,
    RegionalMetabolicBudget,
)
from .homeostasis.synaptic_homeostasis import (
    StriatumHomeostasis,
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from .strategies import (
    BCMConfig,
    BCMStrategy,
    CompositeStrategy,
    ErrorCorrectiveConfig,
    ErrorCorrectiveStrategy,
    HebbianConfig,
    HebbianStrategy,
    LearningConfig,
    STDPConfig,
    STDPStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
    LearningStrategyRegistry,
    LearningStrategy,
    create_strategy,
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
    # Learning Strategies
    "LearningConfig",
    "BCMConfig",
    "ErrorCorrectiveConfig",
    "HebbianConfig",
    "STDPConfig",
    "ThreeFactorConfig",
    "HebbianStrategy",
    "STDPStrategy",
    "BCMStrategy",
    "ThreeFactorStrategy",
    "ErrorCorrectiveStrategy",
    "CompositeStrategy",
    "LearningStrategyRegistry",
    "LearningStrategy",
    "create_strategy",
    # Eligibility Traces
    "EligibilityTraceManager",
    "EligibilitySTDPConfig",
]
