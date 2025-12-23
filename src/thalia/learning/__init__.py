"""
Learning rules: STDP, homeostatic mechanisms, and reward-modulated learning.

This module provides:
1. BCMRule - existing BCM implementation
2. UnifiedHomeostasis - constraint-based stability
3. Learning Strategies - pluggable, composable learning algorithms:
   - HebbianStrategy, STDPStrategy, BCMStrategy
   - ThreeFactorStrategy, ErrorCorrectiveStrategy
   - CompositeStrategy for combining strategies
4. Robustness Mechanisms:
   - EIBalanceRegulator - E/I balance regulation
   - IntrinsicPlasticity - threshold adaptation
   - MetabolicConstraint - energy-based regularization
5. Critical Period Gating - time-windowed plasticity modulation
"""

from thalia.learning.rules.bcm import (
    BCMRule,
    BCMConfig,
)
from thalia.learning.critical_periods import (
    CriticalPeriodGating,
    CriticalPeriodConfig,
    CriticalPeriodWindow,
)
from thalia.learning.social_learning import (
    SocialLearningModule,
    SocialLearningConfig,
    SocialContext,
    SocialCueType,
    compute_shared_attention,
)
from thalia.learning.homeostasis.synaptic_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
    StriatumHomeostasis,
)
from thalia.learning.ei_balance import (
    EIBalanceConfig,
    EIBalanceRegulator,
    LayerEIBalance,
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
from thalia.learning.rules.strategies import (
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
from thalia.learning.strategy_registry import (
    LearningStrategyRegistry,
    create_cortex_strategy,
    create_hippocampus_strategy,
    create_striatum_strategy,
    create_cerebellum_strategy,
)
from thalia.learning.strategy_mixin import (
    LearningStrategyMixin,
)
from thalia.learning.eligibility import (
    EligibilityTraceManager,
    STDPConfig as EligibilitySTDPConfig,
)

__all__ = [
    # BCM (Bienenstock-Cooper-Munro)
    "BCMRule",
    "BCMConfig",
    # Critical Period Gating
    "CriticalPeriodGating",
    "CriticalPeriodConfig",
    "CriticalPeriodWindow",
    # Social Learning
    "SocialLearningModule",
    "SocialLearningConfig",
    "SocialContext",
    "SocialCueType",
    "compute_shared_attention",
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
    # Learning Strategies
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
    # Strategy Registry
    "LearningStrategyRegistry",
    # Region-Specific Strategy Factories
    "create_cortex_strategy",
    "create_hippocampus_strategy",
    "create_striatum_strategy",
    "create_cerebellum_strategy",
    # Strategy Mixin for Regions
    "LearningStrategyMixin",
    # Eligibility Traces
    "EligibilityTraceManager",
    "EligibilitySTDPConfig",
]
