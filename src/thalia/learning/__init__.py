"""
Learning rules: STDP, homeostatic mechanisms, and reward-modulated learning.

This module provides:
1. Legacy BCMRule - existing BCM implementation
2. UnifiedHomeostasis - constraint-based stability
3. Learning Strategies - pluggable, composable learning algorithms:
   - HebbianStrategy, STDPStrategy, BCMStrategy
   - ThreeFactorStrategy, ErrorCorrectiveStrategy
   - CompositeStrategy for combining strategies
4. Robustness Mechanisms:
   - EIBalanceRegulator - E/I balance regulation
   - IntrinsicPlasticity - threshold adaptation
   - MetabolicConstraint - energy-based regularization
"""

from thalia.learning.bcm import (
    BCMRule,
    BCMConfig,
)
from thalia.learning.unified_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
    StriatumHomeostasis,
)
from thalia.learning.ei_balance import (
    EIBalanceConfig,
    EIBalanceRegulator,
    LayerEIBalance,
)
from thalia.learning.intrinsic_plasticity import (
    IntrinsicPlasticityConfig,
    IntrinsicPlasticity,
    PopulationIntrinsicPlasticity,
)
from thalia.learning.metabolic import (
    MetabolicConfig,
    MetabolicConstraint,
    RegionalMetabolicBudget,
)
from thalia.learning.strategies import (
    # Base classes
    LearningConfig,
    BaseStrategy,
    # Strategy configs
    HebbianConfig,
    STDPConfig,
    BCMConfig as BCMStrategyConfig,  # Renamed to avoid collision with bcm.BCMConfig
    ThreeFactorConfig,
    ErrorCorrectiveConfig,
    # Strategy implementations
    HebbianStrategy,
    STDPStrategy,
    BCMStrategy,
    ThreeFactorStrategy,
    ErrorCorrectiveStrategy,
    CompositeStrategy,
    # Factory
    create_strategy,
)
from thalia.learning.strategy_mixin import (
    LearningStrategyMixin,
)

__all__ = [
    # BCM (Bienenstock-Cooper-Munro) - legacy
    "BCMRule",
    "BCMConfig",
    # Unified Homeostasis (constraint-based)
    "UnifiedHomeostasis",
    "UnifiedHomeostasisConfig",
    "StriatumHomeostasis",
    # E/I Balance Regulation
    "EIBalanceConfig",
    "EIBalanceRegulator",
    "LayerEIBalance",
    # Intrinsic Plasticity
    "IntrinsicPlasticityConfig",
    "IntrinsicPlasticity",
    "PopulationIntrinsicPlasticity",
    # Metabolic Constraints
    "MetabolicConfig",
    "MetabolicConstraint",
    "RegionalMetabolicBudget",
    # Learning Strategies (new pluggable system)
    "LearningConfig",
    "BaseStrategy",
    "HebbianConfig",
    "STDPConfig",
    "BCMStrategyConfig",
    "ThreeFactorConfig",
    "ErrorCorrectiveConfig",
    "HebbianStrategy",
    "STDPStrategy",
    "BCMStrategy",
    "ThreeFactorStrategy",
    "ErrorCorrectiveStrategy",
    "CompositeStrategy",
    "create_strategy",
    # Strategy Mixin for Regions
    "LearningStrategyMixin",
]
