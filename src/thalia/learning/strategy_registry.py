"""
Learning Strategy Registry.

Provides a unified registration and factory system for learning strategies,
enabling dynamic strategy creation, plugin architectures, and consistent
strategy discovery.

Design Philosophy:
==================
- Follows the same pattern as ComponentRegistry for consistency
- Enables dynamic strategy creation from configuration
- Supports plugin system for external learning rules
- Makes learning strategies discoverable and pluggable
- Easier to experiment with new learning rules

Architecture:
=============
    LearningStrategyRegistry
        ├── "hebbian" → HebbianStrategy
        ├── "stdp" → STDPStrategy
        ├── "bcm" → BCMStrategy
        ├── "three_factor" → ThreeFactorStrategy
        ├── "error_corrective" → ErrorCorrectiveStrategy
        └── "composite" → CompositeStrategy

Usage Example:
==============
    # Register strategies (typically done in strategy module)
    @LearningStrategyRegistry.register("stdp")
    class STDPStrategy(LearningStrategy):
        ...

    @LearningStrategyRegistry.register("three_factor", aliases=["rl", "dopamine"])
    class ThreeFactorStrategy(LearningStrategy):
        ...

    # Create strategies dynamically in regions
    self.learning_strategy = LearningStrategyRegistry.create(
        "three_factor",
        ThreeFactorConfig(learning_rate=0.02, dopamine_sensitivity=0.5)
    )

    # Discover strategies
    available = LearningStrategyRegistry.list_strategies()
    # ['hebbian', 'stdp', 'bcm', 'three_factor', 'error_corrective', 'composite']

Benefits:
=========
1. **Pluggable Learning**: Easy to add new learning rules without modifying regions
2. **Discovery**: List all available strategies programmatically
3. **Plugin Support**: External packages can register custom strategies
4. **Consistency**: Same pattern as ComponentRegistry
5. **Validation**: Type checking and config validation
6. **Experimentation**: Quickly swap strategies for ablation studies

Author: Thalia Project
Date: December 11, 2025
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type

from thalia.core.errors import ConfigurationError
from thalia.learning.rules.strategies import (
    BCMConfig,
    BCMStrategy,
    CompositeStrategy,
    ErrorCorrectiveConfig,
    ErrorCorrectiveStrategy,
    LearningConfig,
    LearningStrategy,
    STDPConfig,
    STDPStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
)


class LearningStrategyRegistry:
    """Registry for all learning strategies.

    Maintains a registry of learning strategy classes with their configurations,
    enabling dynamic strategy creation and discovery.

    Registry Structure:
        _registry = {
            "hebbian": HebbianStrategy,
            "stdp": STDPStrategy,
            "bcm": BCMStrategy,
            ...
        }

    Attributes:
        _registry: Dict mapping strategy name to strategy class
        _configs: Dict mapping strategy name to config class
        _aliases: Dict mapping alias to canonical name
        _metadata: Strategy metadata (description, version, author, etc.)
    """

    _registry: Dict[str, Type[LearningStrategy]] = {}
    _configs: Dict[str, Type[LearningConfig]] = {}
    _aliases: Dict[str, str] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        config_class: Optional[Type[LearningConfig]] = None,
        aliases: Optional[List[str]] = None,
        description: str = "",
        version: str = "1.0",
        author: str = "",
    ) -> Callable[[Type[LearningStrategy]], Type[LearningStrategy]]:
        """Decorator to register a learning strategy.

        Args:
            name: Primary name for the strategy
            config_class: Configuration class for the strategy (optional)
            aliases: Optional list of alternative names
            description: Human-readable description
            version: Strategy version string
            author: Strategy author/maintainer

        Returns:
            Decorator function

        Raises:
            ValueError: If name already registered or strategy invalid

        Example:
            @LearningStrategyRegistry.register(
                "stdp",
                config_class=STDPConfig,
                aliases=["spike_timing"],
                description="Spike-timing dependent plasticity"
            )
            class STDPStrategy(LearningStrategy):
                '''STDP learning rule.'''
                ...

            @LearningStrategyRegistry.register("three_factor", aliases=["rl"])
            class ThreeFactorStrategy(LearningStrategy):
                '''Three-factor learning with neuromodulation.'''
                ...
        """

        def decorator(strategy_class: Type[LearningStrategy]) -> Type[LearningStrategy]:
            # Validate strategy class
            if not inspect.isclass(strategy_class):
                raise ConfigurationError(f"Strategy must be a class, got {type(strategy_class)}")

            # Check if name already registered
            if name in cls._registry:
                raise ConfigurationError(
                    f"Strategy '{name}' already registered as {cls._registry[name].__name__}"
                )

            # Register strategy
            cls._registry[name] = strategy_class

            # Register config if provided
            if config_class is not None:
                cls._configs[name] = config_class

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in cls._aliases:
                        raise ConfigurationError(
                            f"Alias '{alias}' already registered for '{cls._aliases[alias]}'"
                        )
                    cls._aliases[alias] = name

            # Store metadata
            cls._metadata[name] = {
                "class": strategy_class.__name__,
                "description": description or strategy_class.__doc__ or "",
                "version": version,
                "author": author,
                "aliases": aliases or [],
                "config_class": config_class.__name__ if config_class else None,
            }

            return strategy_class

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        config: LearningConfig,
        **kwargs: Any,
    ) -> LearningStrategy:
        """Create a learning strategy instance.

        Args:
            name: Strategy name (or alias)
            config: Strategy configuration object
            **kwargs: Additional arguments passed to strategy constructor

        Returns:
            Configured learning strategy instance

        Raises:
            ValueError: If strategy not found or creation fails

        Example:
            >>> # Create STDP strategy
            >>> stdp = LearningStrategyRegistry.create(
            ...     "stdp",
            ...     STDPConfig(learning_rate=0.02, a_plus=0.01)
            ... )

            >>> # Create using alias
            >>> rl_strategy = LearningStrategyRegistry.create(
            ...     "rl",  # Alias for "three_factor"
            ...     ThreeFactorConfig(learning_rate=0.02)
            ... )
        """
        # Resolve alias
        canonical_name = cls._aliases.get(name, name)

        # Check if strategy exists
        if canonical_name not in cls._registry:
            available = cls.list_strategies(include_aliases=True)
            raise ConfigurationError(
                f"Unknown learning strategy: '{name}'. "
                f"Available strategies: {', '.join(available)}"
            )

        # Get strategy class
        strategy_class = cls._registry[canonical_name]

        # Validate config type if registered
        if canonical_name in cls._configs:
            expected_config = cls._configs[canonical_name]
            if not isinstance(config, expected_config):
                raise ConfigurationError(
                    f"Strategy '{canonical_name}' expects config type {expected_config.__name__}, "
                    f"got {type(config).__name__}"
                )

        # Create strategy instance
        try:
            return strategy_class(config, **kwargs)  # type: ignore[call-arg]
        except Exception as e:
            raise ConfigurationError(f"Failed to create strategy '{canonical_name}': {e}") from e

    @classmethod
    def list_strategies(cls, include_aliases: bool = False) -> List[str]:
        """List all registered strategies.

        Args:
            include_aliases: Whether to include aliases in the list

        Returns:
            List of strategy names (and aliases if requested)

        Example:
            >>> LearningStrategyRegistry.list_strategies()
            ['hebbian', 'stdp', 'bcm', 'three_factor', 'error_corrective']

            >>> LearningStrategyRegistry.list_strategies(include_aliases=True)
            ['hebbian', 'stdp', 'spike_timing', 'bcm', 'three_factor', 'rl', ...]
        """
        strategies = list(cls._registry.keys())

        if include_aliases:
            strategies.extend(cls._aliases.keys())
            strategies = sorted(set(strategies))

        return sorted(strategies)

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a strategy.

        Args:
            name: Strategy name (or alias)

        Returns:
            Dictionary containing strategy metadata

        Raises:
            ValueError: If strategy not found

        Example:
            >>> meta = LearningStrategyRegistry.get_metadata("stdp")
            >>> print(meta["description"])
            Spike-timing dependent plasticity
            >>> print(meta["config_class"])
            STDPConfig
        """
        # Resolve alias
        canonical_name = cls._aliases.get(name, name)

        if canonical_name not in cls._metadata:
            raise ConfigurationError(f"Unknown strategy: '{name}'")

        return cls._metadata[canonical_name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Strategy name (or alias)

        Returns:
            True if strategy is registered, False otherwise

        Example:
            >>> LearningStrategyRegistry.is_registered("stdp")
            True
            >>> LearningStrategyRegistry.is_registered("unknown")
            False
        """
        canonical_name = cls._aliases.get(name, name)
        return canonical_name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a strategy (primarily for testing).

        Args:
            name: Strategy name to unregister

        Example:
            >>> LearningStrategyRegistry.unregister("custom_strategy")
        """
        if name in cls._registry:
            del cls._registry[name]

        if name in cls._configs:
            del cls._configs[name]

        if name in cls._metadata:
            del cls._metadata[name]

        # Remove aliases pointing to this strategy
        aliases_to_remove = [alias for alias, target in cls._aliases.items() if target == name]
        for alias in aliases_to_remove:
            del cls._aliases[alias]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies (primarily for testing).

        Example:
            >>> LearningStrategyRegistry.clear()
        """
        cls._registry.clear()
        cls._configs.clear()
        cls._aliases.clear()
        cls._metadata.clear()


# =============================================================================
# Region-Specific Strategy Factory Helpers
# =============================================================================


def create_cortex_strategy(
    learning_rate: float = 0.001,
    tau_theta: float = 5000.0,
    use_stdp: bool = True,
    use_bcm: bool = True,
    stdp_config: Optional[Any] = None,
    bcm_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
    """Create composite STDP+BCM strategy for cortical learning.

    Cortical learning typically combines:
    - STDP for spike-timing based plasticity
    - BCM for homeostatic sliding threshold

    Args:
        learning_rate: Base learning rate for STDP (ignored if stdp_config provided)
        tau_theta: BCM threshold adaptation time constant (ms, ignored if bcm_config provided)
        use_stdp: Whether to include STDP component
        use_bcm: Whether to include BCM component
        stdp_config: Optional custom STDPConfig instance
        bcm_config: Optional custom BCMConfig instance
        **kwargs: Additional parameters for strategies (if configs not provided)

    Returns:
        Configured learning strategy (composite if both enabled, single otherwise)

    Example:
        >>> # Standard cortex strategy (STDP + BCM composite)
        >>> strategy = create_cortex_strategy(learning_rate=0.001)

        >>> # Custom configs
        >>> stdp_cfg = STDPConfig(learning_rate=0.002, a_plus=0.02)
        >>> bcm_cfg = BCMConfig(tau_theta=10000.0)
        >>> strategy = create_cortex_strategy(stdp_config=stdp_cfg, bcm_config=bcm_cfg)

        >>> # BCM only (unsupervised feature learning)
        >>> strategy = create_cortex_strategy(use_stdp=False)
    """
    if use_stdp and use_bcm:
        # Create composite STDP+BCM strategy
        stdp = STDPStrategy(stdp_config or STDPConfig(learning_rate=learning_rate, **kwargs))
        bcm = BCMStrategy(
            bcm_config or BCMConfig(learning_rate=learning_rate, tau_theta=tau_theta, **kwargs)
        )
        return CompositeStrategy([stdp, bcm])
    elif use_stdp:
        return STDPStrategy(stdp_config or STDPConfig(learning_rate=learning_rate, **kwargs))
    elif use_bcm:
        return BCMStrategy(
            bcm_config or BCMConfig(learning_rate=learning_rate, tau_theta=tau_theta, **kwargs)
        )
    else:
        raise ConfigurationError("Must enable at least one learning rule (STDP or BCM)")


def create_hippocampus_strategy(
    learning_rate: float = 0.01,
    one_shot: bool = False,
    a_plus: Optional[float] = None,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    stdp_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
    """Create hippocampus-appropriate STDP with one-shot capability.

    Hippocampus supports both:
    - Standard STDP for gradual pattern completion (CA3 recurrence)
    - One-shot learning for episodic encoding (high learning rate)

    Args:
        learning_rate: Base learning rate (0.01 standard, 0.1 for one-shot)
        one_shot: Whether to use high learning rate for single-trial learning
        a_plus: LTP amplitude (if None, uses 0.01 standard or 0.1 one-shot)
        tau_plus: LTP time constant (ms)
        tau_minus: LTD time constant (ms)
        stdp_config: Optional custom STDPConfig instance (overrides all other params)
        **kwargs: Additional STDP parameters

    Returns:
        Configured STDP strategy

    Example:
        >>> # Standard hippocampal STDP
        >>> strategy = create_hippocampus_strategy()

        >>> # One-shot episodic encoding
        >>> strategy = create_hippocampus_strategy(one_shot=True)

        >>> # Custom config
        >>> cfg = STDPConfig(learning_rate=0.05, a_plus=0.05)
        >>> strategy = create_hippocampus_strategy(stdp_config=cfg)
    """
    # If custom config provided, use it directly
    if stdp_config is not None:
        return STDPStrategy(stdp_config)

    # Set defaults based on one-shot mode
    if one_shot:
        learning_rate = 0.1 if learning_rate == 0.01 else learning_rate
        a_plus = 0.1 if a_plus is None else a_plus
    else:
        a_plus = 0.01 if a_plus is None else a_plus

    return STDPStrategy(
        STDPConfig(
            learning_rate=learning_rate,
            a_plus=a_plus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            **kwargs,
        )
    )


def create_striatum_strategy(
    learning_rate: float = 0.001,
    eligibility_tau_ms: float = 1000.0,
    three_factor_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
    """Create three-factor learning for striatum (dopamine-modulated).

    Striatum uses three-factor rule:
    - Eligibility traces from pre/post activity
    - Dopamine signal gates weight changes
    - Δw = eligibility × dopamine

    Args:
        learning_rate: Base learning rate
        eligibility_tau_ms: Eligibility trace time constant (ms)
        three_factor_config: Optional custom ThreeFactorConfig instance
        **kwargs: Additional three-factor parameters (w_min, w_max, etc.)

    Returns:
        Configured three-factor strategy

    Example:
        >>> # Standard striatum learning
        >>> strategy = create_striatum_strategy()

        >>> # Longer eligibility traces (delayed rewards)
        >>> strategy = create_striatum_strategy(eligibility_tau_ms=2000.0)

        >>> # Custom config
        >>> cfg = ThreeFactorConfig(learning_rate=0.002, eligibility_tau=1500.0)
        >>> strategy = create_striatum_strategy(three_factor_config=cfg)
    """
    # If custom config provided, use it directly
    if three_factor_config is not None:
        return ThreeFactorStrategy(three_factor_config)

    return ThreeFactorStrategy(
        ThreeFactorConfig(learning_rate=learning_rate, eligibility_tau=eligibility_tau_ms, **kwargs)
    )


def create_cerebellum_strategy(
    learning_rate: float = 0.005,
    error_threshold: float = 0.01,
    error_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
    """Create error-corrective learning for cerebellum (supervised).

    Cerebellum uses delta rule for motor learning:
    - Δw ∝ error × input
    - Supervised by climbing fiber error signals

    Args:
        learning_rate: Base learning rate (typically slower than cortex)
        error_threshold: Minimum error to trigger learning
        error_config: Optional custom ErrorCorrectiveConfig instance
        **kwargs: Additional parameters (w_min, w_max, etc.)

    Returns:
        Configured error-corrective strategy

    Example:
        >>> # Standard cerebellar learning
        >>> strategy = create_cerebellum_strategy()

        >>> # Custom config
        >>> cfg = ErrorCorrectiveConfig(learning_rate=0.01, error_threshold=0.005)
        >>> strategy = create_cerebellum_strategy(error_config=cfg)
    """
    # If custom config provided, use it directly
    if error_config is not None:
        return ErrorCorrectiveStrategy(error_config)

    return ErrorCorrectiveStrategy(
        ErrorCorrectiveConfig(
            learning_rate=learning_rate, error_threshold=error_threshold, **kwargs
        )
    )
