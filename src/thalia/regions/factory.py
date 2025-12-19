"""
Region Factory and Registry for Dynamic Brain Construction.

This module provides a factory pattern and decorator-based registry for
brain regions, enabling dynamic brain architectures and flexible region
selection.

Design Pattern:
===============
1. Regions register themselves using @register_region decorator
2. RegionFactory.create() instantiates regions by name
3. Brain construction becomes loop-driven instead of manual

Benefits:
- Easy to add/remove regions without modifying brain code
- Enables dynamic brain architectures via configuration
- Central registry makes region discovery straightforward
- Supports region type aliasing and alternatives

Usage Example:
==============
    # Register a region (done once in region module)
    @register_region("cortex")
    @register_region("layered_cortex")  # Alias
    class LayeredCortex(NeuralComponent):
        ...

    # Create region by name
    config = LayeredCortexConfig(
        n_input=256, n_output=128,
        l4_size=64, l23_size=96, l5_size=64, l6_size=32
    )
    cortex = RegionFactory.create("cortex", config)

    # Dynamic brain construction
    for region_name in ["cortex", "hippocampus", "striatum"]:
        config = get_config_for_region(region_name)
        regions[region_name] = RegionFactory.create(region_name, config)

Author: Thalia Project
Date: December 7, 2025
"""

from __future__ import annotations

from typing import Dict, Type, Optional, List, Callable, Any
import inspect

from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.errors import ConfigurationError, ComponentError
from thalia.regions.base import NeuralComponent


class RegionRegistry:
    """Central registry of available brain regions.

    Maintains a mapping from region names to region classes,
    supporting multiple names per region (aliases).
    """

    _registry: Dict[str, Type[NeuralComponent]] = {}
    _aliases: Dict[str, str] = {}  # alias -> canonical_name

    @classmethod
    def register(
        cls,
        name: str,
        region_class: Type[NeuralComponent],
        *,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a brain region class.

        Args:
            name: Primary name for the region
            region_class: Region class to register (NeuralRegion or NeuralComponent subclass)
            aliases: Optional list of alternative names

        Raises:
            ValueError: If name or alias already registered
        """
        if name in cls._registry:
            existing = cls._registry[name]
            if existing != region_class:
                raise ConfigurationError(
                    f"Region name '{name}' already registered to {existing.__name__}"
                )
            return  # Same class, already registered

        # Validate region_class is a proper region class (duck typing check)
        if not inspect.isclass(region_class):
            raise ConfigurationError(
                f"Region class must be a class, got {region_class}"
            )

        if not issubclass(region_class, NeuralComponent):
            raise ConfigurationError(
                f"Region class must be a NeuralRegion or NeuralComponent subclass, got {region_class}"
            )

        # Register primary name
        cls._registry[name] = region_class

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in cls._registry or alias in cls._aliases:
                    raise ConfigurationError(f"Alias '{alias}' already registered")
                cls._aliases[alias] = name

    @classmethod
    def get(cls, name: str) -> Optional[Type[NeuralComponent]]:
        """Get region class by name.

        Args:
            name: Region name or alias

        Returns:
            Region class if found, None otherwise
        """
        # Check if it's an alias
        if name in cls._aliases:
            name = cls._aliases[name]

        return cls._registry.get(name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a region name is registered."""
        return name in cls._registry or name in cls._aliases

    @classmethod
    def list_regions(cls) -> List[str]:
        """Get list of all registered region names."""
        return sorted(cls._registry.keys())

    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """Get mapping of aliases to canonical names."""
        return cls._aliases.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._registry.clear()
        cls._aliases.clear()


def register_region(
    name: str,
    *,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[NeuralComponent]], Type[NeuralComponent]]:
    """Decorator to register a brain region class.

    Args:
        name: Primary name for the region
        aliases: Optional list of alternative names

    Returns:
        Decorator function

    Example:
        @register_region("cortex", aliases=["layered_cortex"])
        class LayeredCortex(NeuralComponent):
            ...
    """
    def decorator(region_class: Type[NeuralComponent]) -> Type[NeuralComponent]:
        RegionRegistry.register(name, region_class, aliases=aliases)
        return region_class

    return decorator


class RegionFactory:
    """Factory for creating brain region instances.

    Provides a unified interface for instantiating regions by name,
    with automatic config type validation.
    """

    @staticmethod
    def create(
        name: str,
        config: NeuralComponentConfig,
        **kwargs: Any,
    ) -> NeuralComponent:
        """Create a brain region instance.

        Args:
            name: Region name or alias
            config: Configuration for the region
            **kwargs: Additional arguments passed to constructor

        Returns:
            Instantiated region

        Raises:
            ValueError: If region not registered
            TypeError: If config type doesn't match region requirements

        Example:
            cortex = RegionFactory.create(
                "cortex",
                LayeredCortexConfig(n_input=256, n_output=128)
            )
        """
        region_class = RegionRegistry.get(name)

        if region_class is None:
            available = RegionRegistry.list_regions()
            raise ConfigurationError(
                f"Unknown region '{name}'. Available regions: {available}"
            )

        # Instantiate region
        try:
            return region_class(config, **kwargs)
        except Exception as e:
            raise ComponentError(
                region_class.__name__,
                f"Failed to instantiate with config {type(config).__name__}: {e}"
            ) from e

    @staticmethod
    def create_batch(region_specs: Dict[str, NeuralComponentConfig]) -> Dict[str, NeuralComponent]:
        """Create multiple regions from a specification dict.

        Args:
            region_specs: Dict mapping region names to configs

        Returns:
            Dict mapping region names to instantiated regions

        Example:
            regions = RegionFactory.create_batch({
                "cortex": LayeredCortexConfig(n_input=256, n_output=128),
                "hippocampus": HippocampusConfig(n_input=128, n_output=64),
                "striatum": StriatumConfig(n_input=192, n_output=4),
            })
        """
        return {
            name: RegionFactory.create(name, config)
            for name, config in region_specs.items()
        }

    @staticmethod
    def is_available(name: str) -> bool:
        """Check if a region is available."""
        return RegionRegistry.is_registered(name)

    @staticmethod
    def list_available() -> List[str]:
        """Get list of all available regions."""
        return RegionRegistry.list_regions()


__all__ = [
    "RegionRegistry",
    "RegionFactory",
    "register_region",
]
