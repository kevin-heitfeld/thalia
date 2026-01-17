"""Learning Rules.

Core learning algorithms including BCM and pluggable learning strategies.
"""

from __future__ import annotations

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
