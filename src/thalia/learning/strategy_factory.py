"""
Learning Strategy Factory.

Provides factory functions for creating learning strategies used by
regions with simple, standard learning rules.

Used By:
========
- Prefrontal: STDP with dopamine gating
- LayeredCortex: STDP for cortical learning

Not Used By (Custom Learning):
===============================
- Striatum: D1/D2 opponent pathways, goal conditioning
- Hippocampus: Theta-phase learning, trisynaptic circuit
- Cerebellum: Climbing fiber error signals
- SpikingPathway: Phase-dependent STDP, multi-neuromodulator

Author: Thalia Team
Date: December 11, 2025
"""

from typing import Dict, Any

from thalia.learning.strategies import (
    LearningStrategy,
    HebbianStrategy,
    STDPStrategy,
    BCMStrategy,
    ThreeFactorStrategy,
    ErrorCorrectiveStrategy,
    CompositeStrategy,
    # Configs
    HebbianConfig,
    STDPConfig,
    BCMConfig,
    ThreeFactorConfig,
    ErrorCorrectiveConfig,
)


# =============================================================================
# Strategy Registry
# =============================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
    "hebbian": HebbianStrategy,
    "stdp": STDPStrategy,
    "bcm": BCMStrategy,
    "three_factor": ThreeFactorStrategy,
    "error_corrective": ErrorCorrectiveStrategy,
    "composite": CompositeStrategy,
}

CONFIG_REGISTRY: Dict[str, type] = {
    "hebbian": HebbianConfig,
    "stdp": STDPConfig,
    "bcm": BCMConfig,
    "three_factor": ThreeFactorConfig,
    "error_corrective": ErrorCorrectiveConfig,
}


# =============================================================================
# Factory Functions (Used)
# =============================================================================

def create_learning_strategy(
    strategy_type: str,
    **config_kwargs: Any,
) -> LearningStrategy:
    """Create a learning strategy with configuration.

    Used by: Prefrontal (creates STDP strategy)

    Args:
        strategy_type: Type of strategy ("stdp", "three_factor", "bcm", etc.)
        **config_kwargs: Configuration parameters passed to strategy config

    Returns:
        Configured learning strategy instance

    Example:
        >>> # Prefrontal uses this for dopamine-gated STDP
        >>> stdp = create_learning_strategy(
        ...     "stdp",
        ...     learning_rate=0.02,
        ...     a_plus=0.01,
        ...     a_minus=0.012,
        ...     tau_plus=20.0,
        ...     tau_minus=20.0,
        ... )
    """
    if strategy_type not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {available}"
        )

    # Get config class
    config_class = CONFIG_REGISTRY.get(strategy_type)
    if config_class is None:
        raise ValueError(f"No config class for strategy: {strategy_type}")

    # Create config
    config = config_class(**config_kwargs)

    # Create strategy
    strategy_class = STRATEGY_REGISTRY[strategy_type]
    return strategy_class(config)


def create_cortex_strategy(
    learning_rate: float = 0.001,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    **kwargs: Any,
) -> STDPStrategy:
    """Create STDP strategy for cortical learning.

    Used by: LayeredCortex

    Args:
        learning_rate: Base learning rate
        a_plus: LTP amplitude
        a_minus: LTD amplitude
        tau_plus: LTP time constant (ms)
        tau_minus: LTD time constant (ms)
        **kwargs: Additional config parameters (dt_ms, w_min, w_max, soft_bounds)

    Returns:
        Configured STDP strategy
    """
    # Create STDP directly (more efficient than going through registry)
    config = STDPConfig(
        learning_rate=learning_rate,
        a_plus=a_plus,
        a_minus=a_minus,
        tau_plus=tau_plus,
        tau_minus=tau_minus,
        **kwargs,
    )
    return STDPStrategy(config)


__all__ = [
    "create_learning_strategy",
    "create_cortex_strategy",
]
