"""
Learning rules: STDP, homeostatic mechanisms, and reward-modulated learning.

This module provides:
1. Legacy BCMRule - existing BCM implementation
2. UnifiedHomeostasis - constraint-based stability
3. Learning Strategies - pluggable, composable learning algorithms:
   - HebbianStrategy, STDPStrategy, BCMStrategy
   - ThreeFactorStrategy, ErrorCorrectiveStrategy
   - CompositeStrategy for combining strategies
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

__all__ = [
    # BCM (Bienenstock-Cooper-Munro) - legacy
    "BCMRule",
    "BCMConfig",
    # Unified Homeostasis (constraint-based)
    "UnifiedHomeostasis",
    "UnifiedHomeostasisConfig",
    "StriatumHomeostasis",
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
]
