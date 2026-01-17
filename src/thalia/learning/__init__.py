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

from __future__ import annotations

from thalia.learning.critical_periods import (
    CriticalPeriodConfig,
    CriticalPeriodGating,
    CriticalPeriodWindow,
)
from thalia.learning.ei_balance import (
    EIBalanceConfig,
    EIBalanceRegulator,
    LayerEIBalance,
)
from thalia.learning.eligibility import (
    EligibilityTraceManager,
)
from thalia.learning.eligibility import STDPConfig as EligibilitySTDPConfig
from thalia.learning.homeostasis.intrinsic_plasticity import (
    IntrinsicPlasticity,
    IntrinsicPlasticityConfig,
    PopulationIntrinsicPlasticity,
)
from thalia.learning.homeostasis.metabolic import (
    MetabolicConfig,
    MetabolicConstraint,
    RegionalMetabolicBudget,
)
from thalia.learning.homeostasis.synaptic_homeostasis import (
    StriatumHomeostasis,
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.learning.rules.bcm import (
    BCMConfig,
    BCMRule,
)
from thalia.learning.rules.strategies import (
    BaseStrategy,
)
from thalia.learning.rules.strategies import (
    BCMConfig as BCMStrategyConfig,  # Base classes; Strategy configs; Strategy implementations; Factory; Renamed to avoid collision with bcm.BCMConfig
)
from thalia.learning.rules.strategies import (
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
    create_strategy,
)
from thalia.learning.social_learning import (
    SocialContext,
    SocialCueType,
    SocialLearningConfig,
    SocialLearningModule,
    compute_shared_attention,
)
from thalia.learning.strategy_mixin import (
    LearningStrategyMixin,
)
from thalia.learning.strategy_registry import (
    LearningStrategyRegistry,
    create_cerebellum_strategy,
    create_cortex_strategy,
    create_hippocampus_strategy,
    create_striatum_strategy,
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
