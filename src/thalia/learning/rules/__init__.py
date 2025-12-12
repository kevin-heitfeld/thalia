"""Learning Rules.

Core learning algorithms including BCM and pluggable learning strategies.
"""

from thalia.learning.rules.bcm import (
    BCMRule,
    BCMConfig,
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

__all__ = [
    # BCM (Bienenstock-Cooper-Munro)
    "BCMRule",
    "BCMConfig",
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
]
