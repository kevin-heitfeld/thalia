"""
Unified Component Registry for Brain Components.

This module provides a unified registration and factory system for all
brain components (regions, pathways, modules), enabling dynamic component
creation, plugin architectures, and consistent component discovery.

Design Philosophy:
==================
- Treats regions and pathways uniformly (component parity)
- Enables dynamic component creation from configuration
- Supports plugin system for external components
- Foundation for save/load of arbitrary component graphs
- Backward compatible with existing RegionFactory
- Separates behavioral config from structural sizes

Architecture:
=============
    ComponentRegistry
        ├── region:cortex → LayeredCortex
        ├── region:hippocampus → TrisynapticCircuit
        ├── pathway:spiking_stdp → SpikingPathway
        ├── pathway:attention → AttentionPathway
        └── module:oscillator → ThetaOscillator

Usage Example:
==============
    # Register components
    @ComponentRegistry.register("cortex", "region")
    class LayeredCortex(NeuralRegion):
        ...

    @ComponentRegistry.register("spiking_stdp", "pathway")
    class SpikingPathway(LearnableComponent):
        ...

    # Create components dynamically
    cortex = ComponentRegistry.create("region", "cortex", config)
    pathway = ComponentRegistry.create("pathway", "spiking_stdp", config)

    # Discover components
    regions = ComponentRegistry.list_components("region")
    pathways = ComponentRegistry.list_components("pathway")

Benefits:
=========
1. **Uniform Treatment**: Regions and pathways use same registration pattern
2. **Dynamic Creation**: Instantiate any component from config/name
3. **Plugin Support**: External packages can register components
4. **Discovery**: List/inspect all available components
5. **Validation**: Type checking and config validation
6. **Backward Compatible**: Works alongside existing RegionFactory

Author: Thalia Project
Date: December 11, 2025
"""

from __future__ import annotations

from typing import Dict, Type, Optional, List, Callable, Any, Tuple
import inspect

from thalia.core.protocols.component import BrainComponent
from thalia.core.errors import ConfigurationError


class ComponentRegistry:
    """Unified registry for all brain components (regions, pathways, modules).

    Maintains separate namespaces for different component types while
    providing a unified interface for registration and creation.

    Registry Structure:
        _registry = {
            "region": {"cortex": LayeredCortex, "hippocampus": Trisynaptic},
            "pathway": {"spiking_stdp": SpikingPathway, "attention": Attention},
            "module": {"oscillator": ThetaOscillator}
        }

    Attributes:
        _registry: Nested dict of component_type -> name -> class
        _aliases: Nested dict of component_type -> alias -> canonical_name
        _metadata: Component metadata (description, version, author, etc.)
    """

    _registry: Dict[str, Dict[str, Type[BrainComponent]]] = {
        "region": {},
        "pathway": {},
        "module": {},
    }

    _aliases: Dict[str, Dict[str, str]] = {
        "region": {},
        "pathway": {},
        "module": {},
    }

    _metadata: Dict[str, Dict[str, Dict[str, Any]]] = {
        "region": {},
        "pathway": {},
        "module": {},
    }

    # Map component names to their config classes
    _config_classes: Dict[str, Dict[str, Optional[Type]]] = {
        "region": {},
        "pathway": {},
        "module": {},
    }

    # Map component names to their event adapters
    _adapters: Dict[str, Dict[str, Optional[Type]]] = {
        "region": {},
        "pathway": {},
        "module": {},
    }

    @classmethod
    def register(
        cls,
        name: str,
        component_type: str = "region",
        *,
        aliases: Optional[List[str]] = None,
        description: str = "",
        version: str = "1.0",
        author: str = "",
        config_class: Optional[Type] = None,
    ) -> Callable[[Type[BrainComponent]], Type[BrainComponent]]:
        """Decorator to register a brain component.

        Args:
            name: Primary name for the component
            component_type: Type of component ("region", "pathway", "module")
            aliases: Optional list of alternative names
            description: Human-readable description
            version: Component version string
            author: Component author/maintainer
            config_class: Optional config class for this component

        Returns:
            Decorator function

        Raises:
            ValueError: If component_type invalid or name already registered

        Example:
            @ComponentRegistry.register(
                "cortex", "region",
                aliases=["layered_cortex"],
                config_class=LayeredCortexConfig
            )
            class LayeredCortex(NeuralRegion):
                '''Multi-layer cortical microcircuit.'''
                ...

            @ComponentRegistry.register(
                "spiking_stdp", "pathway",
                config_class=SpikingPathwayConfig
            )
            class SpikingPathway(LearnableComponent):
                '''STDP-learning spiking pathway.'''
                ...
        """
        # Validate component type
        if component_type not in cls._registry:
            raise ConfigurationError(
                f"Invalid component_type '{component_type}'. "
                f"Must be one of: {list(cls._registry.keys())}"
            )

        def decorator(component_class: Type[BrainComponent]) -> Type[BrainComponent]:
            # Validate component class
            if not inspect.isclass(component_class):
                raise ConfigurationError(
                    f"Component must be a class, got {component_class}"
                )

            # Check if already registered
            type_registry = cls._registry[component_type]
            if name in type_registry:
                existing = type_registry[name]
                if existing != component_class:
                    raise ConfigurationError(
                        f"{component_type.capitalize()} name '{name}' already "
                        f"registered to {existing.__name__}"
                    )
                return component_class  # Same class, already registered

            # Register primary name
            type_registry[name] = component_class

            # Register aliases
            if aliases:
                alias_registry = cls._aliases[component_type]
                for alias in aliases:
                    if alias in type_registry or alias in alias_registry:
                        raise ConfigurationError(
                            f"Alias '{alias}' already registered for {component_type}"
                        )
                    alias_registry[alias] = name

            # Store metadata
            cls._metadata[component_type][name] = {
                "description": description or component_class.__doc__ or "",
                "version": version,
                "author": author,
                "class": component_class.__name__,
                "module": component_class.__module__,
            }

            # Store config class if provided
            cls._config_classes[component_type][name] = config_class

            return component_class

        return decorator

    @classmethod
    def get(
        cls,
        component_type: str,
        name: str,
    ) -> Optional[Type[BrainComponent]]:
        """Get component class by type and name.

        Args:
            component_type: Type of component ("region", "pathway", "module")
            name: Component name or alias

        Returns:
            Component class if found, None otherwise

        Example:
            cortex_class = ComponentRegistry.get("region", "cortex")
            pathway_class = ComponentRegistry.get("pathway", "spiking_stdp")
        """
        if component_type not in cls._registry:
            return None

        type_registry = cls._registry[component_type]

        # Check if it's an alias
        alias_registry = cls._aliases[component_type]
        if name in alias_registry:
            name = alias_registry[name]

        return type_registry.get(name)

    @classmethod
    def create(
        cls,
        component_type: str,
        name: str,
        config: Any,
        **kwargs: Any,
    ) -> BrainComponent:
        """Create a component instance.

        Args:
            component_type: Type of component ("region", "pathway", "module")
            name: Component name or alias
            config: Configuration for the component (behavioral params only)
            **kwargs: Additional arguments (may include 'sizes' dict and 'device')

        Returns:
            Instantiated component

        Raises:
            ValueError: If component not registered
            TypeError: If config type doesn't match component requirements

        Example:
            # New pattern (sizes separate):
            cortex = ComponentRegistry.create(
                "region", "cortex",
                LayeredCortexConfig(stdp_lr=0.001),
                sizes={"l4_size": 128, "l23_size": 192, "l5_size": 128, ...},
                device="cpu"
            )

            # Pathways (no sizes needed):
            pathway = ComponentRegistry.create(
                "pathway", "spiking_stdp",
                PathwayConfig(learning_rate=0.001),
                n_input=256, n_output=128
            )
        """
        component_class = cls.get(component_type, name)

        if component_class is None:
            available = cls.list_components(component_type)
            raise ValueError(
                f"{component_type.capitalize()} '{name}' not registered. "
                f"Available {component_type}s: {available}"
            )

        # Create instance
        # Check if component expects (config, sizes, device) signature (new pattern)
        # by inspecting __init__ parameters
        import inspect
        sig = inspect.signature(component_class.__init__)
        params = list(sig.parameters.keys())

        try:
            # New pattern: (config, sizes, device) for regions like LayeredCortex
            if 'sizes' in params and 'device' in params:
                sizes = kwargs.pop('sizes', {})
                device = kwargs.pop('device', None)
                if device is None:
                    device = getattr(config, 'device', 'cpu')
                return component_class(config=config, sizes=sizes, device=device, **kwargs)

            # Old pattern: (config) or (config, **kwargs)
            else:
                return component_class(config, **kwargs)

        except TypeError as e:
            raise TypeError(
                f"Failed to create {component_type} '{name}': {e}\n"
                f"Component signature: {sig}\n"
                f"Expected config type for {component_class.__name__}"
            ) from e

    @classmethod
    def is_registered(
        cls,
        component_type: str,
        name: str,
    ) -> bool:
        """Check if a component is registered.

        Args:
            component_type: Type of component
            name: Component name or alias

        Returns:
            True if registered, False otherwise

        Example:
            if ComponentRegistry.is_registered("region", "cortex"):
                cortex = ComponentRegistry.create("region", "cortex", config)
        """
        if component_type not in cls._registry:
            return False

        type_registry = cls._registry[component_type]
        alias_registry = cls._aliases[component_type]

        return name in type_registry or name in alias_registry

    @classmethod
    def list_components(
        cls,
        component_type: Optional[str] = None,
    ) -> List[str] | Dict[str, List[str]]:
        """List all registered components.

        Args:
            component_type: If specified, list only this type.
                          If None, return dict of all types.

        Returns:
            If component_type specified: Sorted list of component names
            If component_type is None: Dict mapping type -> list of names

        Example:
            # List all regions
            regions = ComponentRegistry.list_components("region")
            # ['cerebellum', 'cortex', 'hippocampus', 'striatum']

            # List all pathways
            pathways = ComponentRegistry.list_components("pathway")
            # ['attention', 'replay', 'spiking_stdp']

            # List all components by type
            all_components = ComponentRegistry.list_components()
            # {'region': [...], 'pathway': [...], 'module': [...]}
        """
        if component_type is not None:
            if component_type not in cls._registry:
                return []
            return sorted(cls._registry[component_type].keys())

        # Return all types
        return {
            ctype: sorted(registry.keys())
            for ctype, registry in cls._registry.items()
            if registry  # Only include non-empty types
        }

    @classmethod
    def list_aliases(
        cls,
        component_type: str,
    ) -> Dict[str, str]:
        """Get mapping of aliases to canonical names.

        Args:
            component_type: Type of component

        Returns:
            Dict mapping alias -> canonical_name

        Example:
            aliases = ComponentRegistry.list_aliases("region")
            # {"layered_cortex": "cortex", "tri_circuit": "hippocampus"}
        """
        if component_type not in cls._aliases:
            return {}

        return cls._aliases[component_type].copy()

    @classmethod
    def get_component_info(
        cls,
        component_type: str,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered component.

        Args:
            component_type: Type of component
            name: Component name or alias

        Returns:
            Dict with description, version, author, class, module
            None if not registered

        Example:
            info = ComponentRegistry.get_component_info("region", "cortex")
            print(f"{info['description']} (v{info['version']})")
        """
        # Resolve alias to canonical name
        if component_type in cls._aliases:
            alias_registry = cls._aliases[component_type]
            if name in alias_registry:
                name = alias_registry[name]

        if component_type not in cls._metadata:
            return None

        return cls._metadata[component_type].get(name)

    @classmethod
    def get_config_class(
        cls,
        component_type: str,
        name: str,
    ) -> Optional[Type]:
        """Get configuration class for a registered component.

        Args:
            component_type: Type of component ("region", "pathway", "module")
            name: Component name or alias

        Returns:
            Config class if registered, None otherwise

        Example:
            config_class = ComponentRegistry.get_config_class("region", "cortex")
            config = config_class(n_neurons=500)
        """
        # Resolve alias to canonical name
        if component_type in cls._aliases:
            alias_registry = cls._aliases[component_type]
            if name in alias_registry:
                name = alias_registry[name]

        if component_type not in cls._config_classes:
            return None

        return cls._config_classes[component_type].get(name)

    @classmethod
    def register_adapter(
        cls,
        component_name: str,
        component_type: str,
        adapter_class: Type,
    ) -> None:
        """Register custom event adapter for a component.

        Event adapters wrap components for event-driven execution with
        the EventScheduler. Custom adapters enable:
        - Specialized input routing (e.g., cortex layers)
        - Layer-specific decay dynamics
        - Multi-source input buffering
        - Performance optimization

        If no adapter is registered, GenericEventAdapter is used automatically.

        Args:
            component_name: Registry name of component
            component_type: Type ('region', 'pathway', 'module')
            adapter_class: EventDrivenRegionBase subclass

        Raises:
            ValueError: If component_type invalid

        Example:
            from thalia.core.neural_region import NeuralRegion

            @register_region("my_region", config_class=MyConfig)
            class MyRegion(NeuralRegion):
                def forward(self, inputs: Dict[str, Tensor]):
                    return self.process(inputs)

            class MyRegionAdapter(EventDrivenRegionBase):
                def _process_spikes(self, spikes, source):
                    # Custom routing logic
                    if source == "sensory":
                        return self._region.layer1(spikes)
                    elif source == "feedback":
                        return self._region.layer2(spikes)
                    return None

            ComponentRegistry.register_adapter(
                "my_region",
                "region",
                MyRegionAdapter
            )
        """
        if component_type not in cls._adapters:
            raise ValueError(
                f"Invalid component_type '{component_type}'. "
                f"Must be one of: {list(cls._adapters.keys())}"
            )

        cls._adapters[component_type][component_name] = adapter_class

    @classmethod
    def get_adapter(
        cls,
        component_type: str,
        name: str,
    ) -> Optional[Type]:
        """Get custom event adapter for a component.

        Returns the registered adapter class if one exists, otherwise None.
        When None is returned, DynamicBrain will use GenericEventAdapter.

        Args:
            component_type: Type of component ("region", "pathway", "module")
            name: Component name or alias

        Returns:
            Adapter class if registered, None otherwise

        Example:
            adapter_class = ComponentRegistry.get_adapter("region", "cortex")
            if adapter_class:
                # Custom adapter registered (e.g., EventDrivenCortex)
                adapter = adapter_class(config, component)
            else:
                # Use GenericEventAdapter
                adapter = GenericEventAdapter(config, component)
        """
        # Resolve alias to canonical name
        if component_type in cls._aliases:
            alias_registry = cls._aliases[component_type]
            if name in alias_registry:
                name = alias_registry[name]

        if component_type not in cls._adapters:
            return None

        return cls._adapters[component_type].get(name)

    @classmethod
    def validate_component(
        cls,
        component_type: str,
        name: str,
        config: Any,
    ) -> Tuple[bool, Optional[str]]:
        """Validate that a component can be created with given config.

        Args:
            component_type: Type of component
            name: Component name
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
            error_message is None if valid

        Example:
            valid, error = ComponentRegistry.validate_component(
                "region", "cortex", config
            )
            if not valid:
                print(f"Invalid config: {error}")
        """
        component_class = cls.get(component_type, name)

        if component_class is None:
            return False, f"{component_type} '{name}' not registered"

        # Try to instantiate (dry run)
        try:
            # Check if __init__ signature accepts config
            sig = inspect.signature(component_class.__init__)
            params = list(sig.parameters.keys())

            if 'config' not in params and len(params) < 2:
                return False, f"{component_class.__name__} doesn't accept config"

            # Validation passed
            return True, None

        except Exception as e:
            return False, str(e)

    @classmethod
    def clear(
        cls,
        component_type: Optional[str] = None,
    ) -> None:
        """Clear the registry (mainly for testing).

        Args:
            component_type: If specified, clear only this type.
                          If None, clear all types.

        Example:
            # Clear all regions
            ComponentRegistry.clear("region")

            # Clear everything
            ComponentRegistry.clear()
        """
        if component_type is not None:
            if component_type in cls._registry:
                cls._registry[component_type].clear()
                cls._aliases[component_type].clear()
                cls._metadata[component_type].clear()
        else:
            for ctype in cls._registry:
                cls._registry[ctype].clear()
                cls._aliases[ctype].clear()
                cls._metadata[ctype].clear()


# Convenience decorator aliases for common component types
def register_region(
    name: str,
    *,
    aliases: Optional[List[str]] = None,
    description: str = "",
    version: str = "1.0",
    author: str = "",
    config_class: Optional[Type] = None,
) -> Callable[[Type[BrainComponent]], Type[BrainComponent]]:
    """Shorthand for @ComponentRegistry.register(name, "region").

    This provides backward compatibility with existing @register_region decorator
    while migrating to the unified registry.

    Args:
        name: Region name
        aliases: Optional list of alternative names
        description: Human-readable description
        version: Component version
        author: Component author
        config_class: Optional config class for this region

    Returns:
        Decorator function

    Example:
        @register_region("cortex", aliases=["layered_cortex"], config_class=LayeredCortexConfig)
        class LayeredCortex(NeuralRegion):
            ...
    """
    return ComponentRegistry.register(
        name, "region",
        aliases=aliases,
        description=description,
        version=version,
        author=author,
        config_class=config_class,
    )


def register_pathway(
    name: str,
    *,
    aliases: Optional[List[str]] = None,
    description: str = "",
    version: str = "1.0",
    author: str = "",
    config_class: Optional[Type] = None,
) -> Callable[[Type[BrainComponent]], Type[BrainComponent]]:
    """Shorthand for @ComponentRegistry.register(name, "pathway").

    Args:
        name: Pathway name
        aliases: Optional list of alternative names
        description: Human-readable description
        version: Component version
        author: Component author
        config_class: Optional config class for this pathway

    Returns:
        Decorator function

    Example:
        @register_pathway("spiking_stdp", aliases=["stdp_pathway"], config_class=SpikingPathwayConfig)
        class SpikingPathway(LearnableComponent):
            ...
    """
    return ComponentRegistry.register(
        name, "pathway",
        aliases=aliases,
        description=description,
        version=version,
        author=author,
        config_class=config_class,
    )


def register_module(
    name: str,
    *,
    aliases: Optional[List[str]] = None,
    description: str = "",
    version: str = "1.0",
    author: str = "",
    config_class: Optional[Type] = None,
) -> Callable[[Type[BrainComponent]], Type[BrainComponent]]:
    """Shorthand for @ComponentRegistry.register(name, "module").

    Args:
        name: Module name
        aliases: Optional list of alternative names
        description: Human-readable description
        version: Component version
        author: Component author
        config_class: Optional config class for this module

    Returns:
        Decorator function

    Example:
        @register_module("theta_oscillator", config_class=OscillatorConfig)
        class ThetaOscillator(BrainComponent):
            ...
    """
    return ComponentRegistry.register(
        name, "module",
        aliases=aliases,
        description=description,
        version=version,
        author=author,
        config_class=config_class,
    )
