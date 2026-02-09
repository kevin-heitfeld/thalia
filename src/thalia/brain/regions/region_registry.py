"""
Unified Component Registry for Brain Regions.

This module provides a unified registration and factory system for all
brain regions, enabling dynamic regioncreation, plugin architectures,
and consistent region discovery.

Design Philosophy:
==================
- Enables dynamic region creation from configuration
- Supports plugin system for external regions
- Unified registration system
- Separates behavioral config from structural sizes

Architecture:
=============
    NeuralRegionRegistry
        ├── cortex → Cortex
        └── hippocampus → Hippocampus

Benefits:
=========
1. **Uniform Treatment**: All regions use same registration pattern
2. **Dynamic Creation**: Instantiate any region from config/name
3. **Plugin Support**: External packages can register regions
4. **Discovery**: List/inspect all available regions
5. **Validation**: Type checking and config validation
6. **Unified Registry**: Single source of truth for all region types
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type

from thalia.brain.configs import NeuralRegionConfig
from thalia.errors import ConfigurationError
from thalia.typing import RegionLayerSizes, RegionName

from .neural_region import NeuralRegion


_NeuralRegion = NeuralRegion[NeuralRegionConfig]


class NeuralRegionRegistry:
    """Unified registry for all brain regions.

    Registry Structure:
        _registry = {
            "cortex": Cortex,
            "hippocampus": Hippocampus,
        }

    Attributes:
        _registry: Dict of name -> class
        _aliases: Dict of alias -> canonical_name
        _metadata: Region metadata (description, version, author, etc.)
        _config_classes: Dict of name -> config class
    """

    _registry: Dict[RegionName, Type[_NeuralRegion]] = {}
    _aliases: Dict[RegionName, RegionName] = {}
    _metadata: Dict[RegionName, Dict[str, Any]] = {}
    _config_classes: Dict[RegionName, Optional[Type[NeuralRegionConfig]]] = {}

    @classmethod
    def register(
        cls,
        name: RegionName,
        *,
        aliases: Optional[List[RegionName]] = None,
        description: str = "",
        version: str = "1.0",
        author: str = "",
        config_class: Optional[Type[NeuralRegionConfig]] = None,
    ) -> Callable[[Type[_NeuralRegion]], Type[_NeuralRegion]]:
        """Decorator to register a region.

        Args:
            name: Primary name for the region
            aliases: Optional list of alternative names
            description: Human-readable description
            version: Component version string
            author: Component author/maintainer
            config_class: Optional config class for this region

        Returns:
            Decorator function

        Raises:
            ValueError: If name already registered
        """
        def decorator(component_class: Type[_NeuralRegion]) -> Type[_NeuralRegion]:
            # Check if already registered
            if name in cls._registry:
                existing = cls._registry[name]
                if existing != component_class:
                    raise ConfigurationError(
                        f"Region name '{name}' already registered to {existing.__name__}"
                    )
                return component_class  # Same class, already registered

            # Register primary name
            cls._registry[name] = component_class

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in cls._registry or alias in cls._aliases:
                        raise ConfigurationError(
                            f"Alias '{alias}' already registered for region"
                        )
                    cls._aliases[alias] = name

            # Store metadata
            cls._metadata[name] = {
                "description": description or component_class.__doc__ or "",
                "version": version,
                "author": author,
                "class": component_class.__name__,
                "module": component_class.__module__,
            }

            # Store config class if provided
            cls._config_classes[name] = config_class

            return component_class

        return decorator

    @classmethod
    def get(cls, name: RegionName) -> Optional[Type[_NeuralRegion]]:
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
    def create(
        cls,
        name: RegionName,
        config: NeuralRegionConfig,
        region_layer_sizes: RegionLayerSizes,
        **kwargs: Any,
    ) -> _NeuralRegion:
        """Create a region instance.

        Args:
            name: Region name or alias
            config: Configuration for the region
            region_layer_sizes: Multi-input sizes dict
            **kwargs: Additional arguments

        Returns:
            Instantiated region

        Raises:
            ValueError: If region not registered
            TypeError: If config type doesn't match region requirements
        """
        region_class = cls.get(name)

        if region_class is None:
            available = cls.list_regions()
            raise ValueError(f"Region '{name}' not registered. Available regions: {available}")

        try:
            return region_class(config=config, region_layer_sizes=region_layer_sizes, **kwargs)

        except TypeError as e:
            sig = inspect.signature(region_class.__init__)
            raise TypeError(
                f"Failed to create region '{name}': {e}\n"
                f"Region signature: {sig}\n"
                f"Expected config type for {region_class.__name__}"
            ) from e

    @classmethod
    def is_registered(cls, name: RegionName) -> bool:
        """Check if a region is registered.

        Args:
            name: Region name or alias

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry or name in cls._aliases

    @classmethod
    def list_regions(cls) -> List[RegionName]:
        """List all registered regions.

        Returns:
            List of region names
        """
        return sorted(cls._registry.keys())

    @classmethod
    def list_aliases(cls) -> Dict[RegionName, RegionName]:
        """Get mapping of aliases to canonical names.

        Returns:
            Dict mapping alias -> canonical_name
        """
        return cls._aliases.copy()

    @classmethod
    def get_config_class(cls, name: RegionName) -> Optional[Type[NeuralRegionConfig]]:
        """Get configuration class for a registered region.

        Args:
            name: Region name or alias

        Returns:
            Config class if registered, None otherwise
        """
        if name in cls._aliases:
            name = cls._aliases[name]

        return cls._config_classes.get(name)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._registry.clear()
        cls._aliases.clear()
        cls._metadata.clear()
        cls._config_classes.clear()


def register_region(
    name: RegionName,
    *,
    aliases: Optional[List[RegionName]] = None,
    description: str = "",
    version: str = "1.0",
    author: str = "",
    config_class: Optional[Type[NeuralRegionConfig]] = None,
) -> Callable[[Type[_NeuralRegion]], Type[_NeuralRegion]]:
    """Shorthand for @NeuralRegionRegistry.register(name).

    Convenience decorator for registering brain regions. This is the standard
    way to register regions in the unified NeuralRegionRegistry system.

    Args:
        name: Region name
        aliases: Optional list of alternative names
        description: Human-readable description
        version: Component version
        author: Component author
        config_class: Optional config class for this region

    Returns:
        Decorator function
    """
    return NeuralRegionRegistry.register(
        name,
        aliases=aliases,
        description=description,
        version=version,
        author=author,
        config_class=config_class,
    )
