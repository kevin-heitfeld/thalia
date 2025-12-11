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

from typing import Dict, Type, Optional, List, Callable, Any
import inspect

from thalia.learning.strategies import LearningStrategy, LearningConfig


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
                raise ValueError(f"Strategy must be a class, got {type(strategy_class)}")
            
            # Check if name already registered
            if name in cls._registry:
                raise ValueError(
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
                        raise ValueError(
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
            raise ValueError(
                f"Unknown learning strategy: '{name}'. "
                f"Available strategies: {', '.join(available)}"
            )
        
        # Get strategy class
        strategy_class = cls._registry[canonical_name]
        
        # Validate config type if registered
        if canonical_name in cls._configs:
            expected_config = cls._configs[canonical_name]
            if not isinstance(config, expected_config):
                raise ValueError(
                    f"Strategy '{canonical_name}' expects config type {expected_config.__name__}, "
                    f"got {type(config).__name__}"
                )
        
        # Create strategy instance
        try:
            return strategy_class(config, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to create strategy '{canonical_name}': {e}"
            ) from e
    
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
            raise ValueError(f"Unknown strategy: '{name}'")
        
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
        aliases_to_remove = [
            alias for alias, target in cls._aliases.items()
            if target == name
        ]
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
# Backward Compatibility Helpers
# =============================================================================

def create_learning_strategy(
    strategy_type: str,
    **config_kwargs: Any,
) -> LearningStrategy:
    """Create a learning strategy with configuration (backward compatible).
    
    This function provides backward compatibility with the old factory pattern
    while internally using the new registry system.
    
    Args:
        strategy_type: Type of strategy ("stdp", "three_factor", "bcm", etc.)
        **config_kwargs: Configuration parameters passed to strategy config
    
    Returns:
        Configured learning strategy instance
    
    Example:
        >>> # Old pattern (still works)
        >>> stdp = create_learning_strategy(
        ...     "stdp",
        ...     learning_rate=0.02,
        ...     a_plus=0.01,
        ...     a_minus=0.012,
        ... )
    """
    # Get config class from registry
    canonical_name = LearningStrategyRegistry._aliases.get(strategy_type, strategy_type)
    
    if canonical_name not in LearningStrategyRegistry._configs:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {', '.join(LearningStrategyRegistry.list_strategies())}"
        )
    
    config_class = LearningStrategyRegistry._configs[canonical_name]
    
    # Create config instance
    config = config_class(**config_kwargs)
    
    # Create strategy using registry
    return LearningStrategyRegistry.create(strategy_type, config)
