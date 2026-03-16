"""Unified registry for brain regions."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import NeuralRegionConfig
from thalia.errors import ConfigurationError
from thalia.typing import PopulationSizes, RegionName

from .neural_region import NeuralRegion


_NeuralRegion = NeuralRegion[NeuralRegionConfig]


class NeuralRegionRegistry:
    """Unified registry for all brain regions."""

    _registry: Dict[RegionName, Type[_NeuralRegion]] = {}
    _aliases: Dict[RegionName, RegionName] = {}
    _metadata: Dict[RegionName, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: RegionName,
        *,
        aliases: Optional[List[RegionName]] = None,
        description: str = "",
    ) -> Callable[[Type[_NeuralRegion]], Type[_NeuralRegion]]:
        """Decorator to register a region.

        Args:
            name: Primary name for the region
            aliases: Optional list of alternative names
            description: Human-readable description

        Returns:
            Decorator function

        Raises:
            ValueError: If name already registered
        """
        def decorator(region_class: Type[_NeuralRegion]) -> Type[_NeuralRegion]:
            # Check if already registered
            if name in cls._registry:
                existing = cls._registry[name]
                if existing != region_class:
                    raise ConfigurationError(
                        f"Region name '{name}' already registered to {existing.__name__}"
                    )
                return region_class  # Same class, already registered

            # Register primary name
            cls._registry[name] = region_class

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
                "description": description or region_class.__doc__ or "",
                "class": region_class.__name__,
                "module": region_class.__module__,
            }

            return region_class

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
        population_sizes: PopulationSizes,
        region_name: RegionName,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> _NeuralRegion:
        """Create a region instance.

        Args:
            name: Region name or alias
            config: Configuration for the region
            population_sizes: Multi-input sizes dict
            region_name: Name to assign to the region instance

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
            return region_class(
                config=config,
                population_sizes=population_sizes,
                region_name=region_name,
                device=device,
            )

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
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._registry.clear()
        cls._aliases.clear()
        cls._metadata.clear()


def register_region(
    name: RegionName,
    *,
    aliases: Optional[List[RegionName]] = None,
    description: str = "",
) -> Callable[[Type[_NeuralRegion]], Type[_NeuralRegion]]:
    """Shorthand for @NeuralRegionRegistry.register(name).

    Convenience decorator for registering brain regions. This is the standard
    way to register regions in the unified NeuralRegionRegistry system.

    Args:
        name: Region name
        aliases: Optional list of alternative names
        description: Human-readable description

    Returns:
        Decorator function
    """
    return NeuralRegionRegistry.register(
        name,
        aliases=aliases,
        description=description,
    )
