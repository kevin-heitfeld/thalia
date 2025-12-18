"""
Brain Builder - Fluent API for Brain Construction

This module provides a fluent, progressive API for building brain architectures
using registered components.

Instead of verbose dataclass configuration:
    config = ThaliaConfig(
        global_config=GlobalConfig(...),
        brain=BrainConfig(
            regions=RegionSizes(...),
            ...
        ),
        ...
    )
    brain = DynamicBrain.from_thalia_config(config)

Use fluent builder pattern:
    brain = (
        BrainBuilder(config)
        .add_component("thalamus", "thalamic_relay", n_neurons=100)
        .add_component("cortex", "layered_cortex", n_neurons=500)
        .connect("thalamus", "cortex", "spiking")
        .build()
    )

Or use preset architectures:
    brain = BrainBuilder.preset("sensorimotor", config)

## Architecture v2.0: Axonal Projections vs Spiking Pathways

**TWO architectural patterns are available:**

### 1. AxonalProjection (Recommended for inter-region connections)
- **Biologically accurate**: Represents axon bundles (no synapses)
- **No weights**: Synapses belong to target region
- **No learning**: Plasticity occurs at synapses, not axons
- **Pure routing**: Concatenates sources, applies conduction delays
- **Usage**: `builder.connect(src, tgt, pathway_type="axonal")`

Example (cortex → striatum):
```python
# AxonalProjection: Just axons (2ms delay)
builder.connect("cortex", "striatum", pathway_type="axonal")

# Corticostriatal synapses are OWNED by striatum (at MSN dendrites)
# No double-synapse problem, clear ownership
```

### 2. SpikingPathway (Use when pathway has neurons)
- **For relay stations**: Thalamus, pathway interneurons
- **Has neurons**: Real neural populations in pathway
- **Has learning**: Pathway-specific plasticity
- **Computation**: Filters, gates, or transforms information
- **Usage**: `builder.connect(src, tgt, pathway_type="spiking")`

Example (sensory input → thalamus):
```python
# Thalamus IS a neural population (relay neurons)
builder.add_component("thalamus", "thalamus", n_input=784, n_output=256)

# Thalamus → cortex: Now just axons
builder.connect("thalamus", "cortex", pathway_type="axonal")
```

**Decision Rule:**
- Does the connection have neurons? → Use SpikingPathway
- Is it pure transmission? → Use AxonalProjection (recommended)

**Benefits of AxonalProjection:**
1. Matches biology (axons ≠ synapses)
2. Clear synaptic ownership (target owns synapses)
3. Simpler growth (no double-resize)
4. Better checkpointing (weights in regions, not pathways)

Author: Thalia Project
Date: December 15, 2025
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from pathlib import Path
import json

from thalia.core.dynamic_brain import DynamicBrain, ComponentSpec, ConnectionSpec
from thalia.managers.component_registry import ComponentRegistry
from thalia.regions.base import NeuralComponent
from thalia.regions.cortex import calculate_layer_sizes

if TYPE_CHECKING:
    from thalia.config import GlobalConfig


class BrainBuilder:
    """Fluent API for progressive brain construction.

    Supports:
        - Incremental component addition via method chaining
        - Connection definition with automatic pathway creation
        - Preset architectures for common use cases
        - Validation before building
        - Save/load component graphs to JSON

    Example - Custom Brain:
        builder = BrainBuilder(global_config)
        builder.add_component("input", "thalamic_relay", n_neurons=128)
        builder.add_component("process", "layered_cortex", n_neurons=512)
        builder.add_component("memory", "hippocampus", n_neurons=256)
        builder.connect("input", "process", "spiking")
        builder.connect("process", "memory", "spiking")
        brain = builder.build()

    Example - Preset Brain:
        brain = BrainBuilder.preset("sensorimotor", global_config)

    Example - Chained Construction:
        brain = (
            BrainBuilder(global_config)
            .add_component("thalamus", "thalamic_relay", n_neurons=100)
            .add_component("cortex", "layered_cortex", n_neurons=500)
            .connect("thalamus", "cortex", "spiking")
            .build()
        )
    """

    # Registry of preset architectures
    _presets: Dict[str, "PresetArchitecture"] = {}

    def __init__(
        self,
        global_config: "GlobalConfig",
        use_parallel: bool = False,
        n_workers: Optional[int] = None,
    ):
        """Initialize builder with global configuration.

        Args:
            global_config: Global configuration (device, dt_ms, etc.)
            use_parallel: Enable parallel execution (default: False)
            n_workers: Number of worker processes (default: auto)
        """
        self.global_config = global_config
        self.use_parallel = use_parallel
        self.n_workers = n_workers
        self._components: Dict[str, ComponentSpec] = {}
        self._connections: List[ConnectionSpec] = []
        self._registry = ComponentRegistry()

        # Register built-in event adapters (lazy registration)
        self._ensure_adapters_registered()

    @staticmethod
    def _ensure_adapters_registered() -> None:
        """Ensure built-in event adapters are registered (one-time).

        This registers EventDrivenRegionBase adapters for all built-in regions.
        Called once on first BrainBuilder instantiation.
        """
        # Check if already registered (avoid redundant registration)
        if not hasattr(BrainBuilder, "_adapters_registered"):
            from thalia.managers.component_registry import register_builtin_event_adapters
            register_builtin_event_adapters()
            BrainBuilder._adapters_registered = True

    def add_component(
        self,
        name: str,
        registry_name: str,
        **config_params: Any,
    ) -> "BrainBuilder":
        """Add a component (region, pathway, module) to the brain.

        Size Inference:
            - Only `n_output` (neuron count) is required for most components
            - `n_input` is automatically inferred from incoming connections
            - Components without incoming connections (input interfaces) must specify `n_input`
            - This prevents size mismatches between components and pathways

        Args:
            name: Instance name (e.g., "my_cortex", "visual_input")
            registry_name: Component type in registry (e.g., "layered_cortex")
            **config_params: Configuration parameters. Required:
                - n_output: Number of output neurons (always required)
                - n_input: Number of input neurons (only for input interfaces)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If component name already exists
            KeyError: If registry_name not found in ComponentRegistry

        Example:
            # Input interface (no incoming connections) - needs n_input
            builder.add_component(
                name="sensory_input",
                registry_name="thalamic_relay",
                n_input=128,
                n_output=128,
            )

            # Processing component - n_input inferred from connection
            builder.add_component(
                name="cortex",
                registry_name="layered_cortex",
                n_output=512,  # Only n_output needed!
            )

            builder.connect("sensory_input", "cortex")
        """
        # Validate name uniqueness
        if name in self._components:
            raise ValueError(f"Component '{name}' already exists")

        # Validate registry name exists
        # ComponentRegistry uses (component_type, name) not just (name)
        # We need to search all component types
        found = False
        component_type = None
        for ctype in ["region", "pathway", "module"]:
            if self._registry.is_registered(ctype, registry_name):
                found = True
                component_type = ctype
                break

        if not found:
            available = self._registry.list_components()
            raise KeyError(
                f"Registry name '{registry_name}' not found. "
                f"Available: {available}"
            )

        # Create component spec
        spec = ComponentSpec(
            name=name,
            component_type=component_type,
            registry_name=registry_name,
            config_params=config_params,
        )

        self._components[name] = spec
        return self

    def connect(
        self,
        source: str,
        target: str,
        pathway_type: str = "spiking",
        source_port: Optional[str] = None,
        target_port: Optional[str] = None,
        **config_params: Any,
    ) -> "BrainBuilder":
        """Connect two components with a pathway.

        Args:
            source: Source component name
            target: Target component name
            pathway_type: Pathway registry name:
                - "spiking": SpikingPathway (has neurons + weights + learning)
                - "axonal": AxonalProjection (pure spike routing, NO weights)
                - Other registered pathway types
            source_port: Output port on source (e.g., 'l23', 'l5')
            target_port: Input port on target (e.g., 'feedforward', 'top_down', 'ec_l3')
            **config_params: Pathway configuration parameters
                For axonal pathways, can specify 'axonal_delay_ms' (default: 2.0)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If source or target component doesn't exist

        Example:
            # Simple connection (backward compatible)
            builder.connect("thalamus", "cortex")

            # Axonal projection (v2.0 architecture - biologically accurate)
            builder.connect("cortex", "striatum", pathway_type="axonal", source_port="l5")

            # Layer-specific routing
            builder.connect("cortex", "hippocampus", source_port="l23")

            # Multiple input types
            builder.connect("thalamus", "cortex", target_port="feedforward")
            builder.connect("pfc", "cortex", target_port="top_down")

            # Axonal with custom delay
            builder.connect("hippocampus", "pfc",
                          pathway_type="axonal",
                          axonal_delay_ms=3.0)
        """
        # Validate components exist
        if source not in self._components:
            raise ValueError(f"Source component '{source}' not found")
        if target not in self._components:
            raise ValueError(f"Target component '{target}' not found")

        # Create connection spec with ports
        spec = ConnectionSpec(
            source=source,
            target=target,
            pathway_type=pathway_type,
            source_port=source_port,
            target_port=target_port,
            config_params=config_params,
        )

        self._connections.append(spec)
        return self

    def validate(self) -> List[str]:
        """Validate component graph before building.

        Checks:
            - All connection endpoints exist
            - No isolated components (warning)
            - No cycles (if strict=True)

        Returns:
            List of warning/error messages (empty if valid)
        """
        issues = []

        # Check for isolated components (no connections)
        connected_components = set()
        for conn in self._connections:
            connected_components.add(conn.source)
            connected_components.add(conn.target)

        isolated = set(self._components.keys()) - connected_components
        if isolated:
            issues.append(
                f"Warning: Isolated components (no connections): {isolated}"
            )

        return issues

    def _infer_component_sizes(self) -> None:
        """Infer n_input for components based on incoming connections.

        Supports port-based routing:
        - source_port: Routes specific layer outputs (e.g., 'l23', 'l5')
        - target_port: Differentiates input types (e.g., 'feedforward', 'top_down')

        For components without n_input specified:
        - Feedforward ports (None, 'feedforward', 'cortical', 'hippocampal'): Counted in n_input
        - Top-down/modulation ports ('top_down', 'pfc_modulation'): Set as separate config params
        - Sensory ports ('ec_l3'): Set as ec_l3_input_size
        """
        # Build incoming connection map with port information
        incoming: Dict[str, List[Tuple[str, "ConnectionSpec"]]] = {name: [] for name in self._components}
        for conn in self._connections:
            incoming[conn.target].append((conn.source, conn))

        # Infer n_input and port-specific sizes for each component
        for name, spec in self._components.items():
            sources = incoming[name]

            if not sources:
                # No incoming connections - this is an input interface
                if "n_input" not in spec.config_params:
                    raise ValueError(
                        f"Component '{name}' has no incoming connections and no n_input specified. "
                        f"Input interfaces must specify n_input explicitly."
                    )
                continue

            # Separate connections by target port
            feedforward_sources = []
            topdown_sources = []
            sensory_sources = []
            modulation_sources = []

            for source_name, conn in sources:
                target_port = conn.target_port

                # Determine port category
                if target_port in (None, "feedforward", "cortical", "hippocampal"):
                    # Standard feedforward inputs - count in n_input
                    feedforward_sources.append((source_name, conn))
                elif target_port == "top_down":
                    topdown_sources.append((source_name, conn))
                elif target_port == "ec_l3":
                    sensory_sources.append((source_name, conn))
                elif target_port == "pfc_modulation":
                    modulation_sources.append((source_name, conn))
                else:
                    # Unknown port - treat as feedforward with warning
                    print(f"Warning: Unknown target_port '{target_port}' on connection to '{name}', treating as feedforward")
                    feedforward_sources.append((source_name, conn))

            # Infer n_input from feedforward connections
            if "n_input" not in spec.config_params and feedforward_sources:
                feedforward_sizes = []
                for source_name, conn in feedforward_sources:
                    output_size = self._get_source_output_size(source_name, conn.source_port)
                    feedforward_sizes.append(output_size)

                spec.config_params["n_input"] = sum(feedforward_sizes)

            # Set port-specific sizes
            if sensory_sources:
                # ec_l3_input_size for hippocampus
                ec_l3_size = sum(self._get_source_output_size(src, conn.source_port)
                                for src, conn in sensory_sources)
                spec.config_params["ec_l3_input_size"] = ec_l3_size

            # Note: top_down and pfc_modulation are handled separately in forward pass
            # They don't contribute to n_input but are stored for reference
            if topdown_sources:
                spec.config_params["_has_topdown"] = True
            if modulation_sources:
                spec.config_params["_has_modulation"] = True

    def _get_source_output_size(self, source_name: str, source_port: Optional[str]) -> int:
        """Get output size from source component, optionally from specific port.

        Args:
            source_name: Name of source component
            source_port: Output port ('l23', 'l5', or None for full output)

        Returns:
            Output size (layer-specific if port specified)
        """
        source_spec = self._components[source_name]

        if "n_output" not in source_spec.config_params:
            raise ValueError(
                f"Component '{source_name}' must specify n_output before connecting"
            )

        # If no port specified, use full output
        if source_port is None:
            return source_spec.config_params["n_output"]

        # For layered cortex (including predictive_cortex), get layer sizes
        if source_spec.registry_name in ("cortex", "layered_cortex", "predictive_cortex"):
            config = source_spec.config_params

            # All layer sizes are now required
            if not all(k in config for k in ["l4_size", "l23_size", "l5_size", "l6_size"]):
                raise ValueError(
                    f"BrainBuilder: Cortex '{source_name}' must specify all layer sizes "
                    f"(l4_size, l23_size, l5_size, l6_size)"
                )

            l4_size = config["l4_size"]
            l23_size = config["l23_size"]
            l5_size = config["l5_size"]
            l6_size = config["l6_size"]

            # Return layer-specific size
            if source_port == "l23":
                return l23_size
            elif source_port == "l5":
                return l5_size
            elif source_port == "l4":
                return l4_size
            elif source_port == "l6":
                return l6_size
            else:
                raise ValueError(f"Unknown cortex port '{source_port}'")

        # For other components, ports not yet supported
        if source_port is not None:
            raise ValueError(
                f"Component '{source_spec.registry_name}' does not support port '{source_port}'"
            )

        return source_spec.config_params["n_output"]

    def _get_pathway_source_size(self, source_comp: NeuralComponent, source_port: Optional[str]) -> int:
        """Get output size for pathway from source component and port.

        Args:
            source_comp: Source component instance
            source_port: Output port specification

        Returns:
            Output size for pathway
        """
        if source_port is None:
            # Use full output
            return source_comp.n_output

        # Check for layer-specific outputs (cortex)
        if hasattr(source_comp, 'l23_size') and hasattr(source_comp, 'l5_size'):
            if source_port == "l23":
                return source_comp.l23_size
            elif source_port == "l5":
                return source_comp.l5_size
            elif source_port == "l4" and hasattr(source_comp, 'l4_size'):
                return source_comp.l4_size
            elif source_port == "l6" and hasattr(source_comp, 'l6_size'):
                return source_comp.l6_size
            else:
                raise ValueError(f"Unknown cortex port '{source_port}'")

        # For other components, ports not yet supported
        raise ValueError(
            f"Component {type(source_comp).__name__} does not support port '{source_port}'"
        )

    def _create_axonal_projection(
        self,
        target_specs: List[ConnectionSpec],
        components: Dict[str, NeuralComponent],
        target_name: str,
    ) -> "AxonalProjection":
        """Create AxonalProjection from connection specs.

        AxonalProjection has different initialization than standard pathways:
        - Takes list of (region_name, port, size, delay_ms) tuples
        - NO config class with n_input/n_output
        - Handles multi-source concatenation internally

        Args:
            target_specs: List of ConnectionSpec for this target
            components: Dict of instantiated components
            target_name: Name of target component

        Returns:
            AxonalProjection instance
        """
        from thalia.pathways.axonal_projection import AxonalProjection

        # Build sources list: [(region_name, port, size, delay_ms), ...]
        sources = []
        for spec in target_specs:
            source_comp = components[spec.source]
            source_size = self._get_pathway_source_size(source_comp, spec.source_port)

            # Get axonal delay (default: 2.0ms)
            delay_ms = spec.config_params.get("axonal_delay_ms", 2.0)

            sources.append((
                spec.source,      # region_name
                spec.source_port, # port (can be None)
                source_size,      # size
                delay_ms,         # delay_ms
            ))

        # Create AxonalProjection
        projection = AxonalProjection(
            sources=sources,
            device=self.global_config.device,
            dt_ms=self.global_config.dt_ms,
        )

        return projection

    def _get_pathway_target_size(self, target_comp: NeuralComponent, target_port: Optional[str]) -> int:
        """Get output size for pathway to target component.

        External pathways (between regions) act as axonal projections that
        concatenate/route spikes. The actual synaptic integration happens
        at the TARGET region's dendrites via internal weights.

        This is biologically accurate:
        - Axons carry spikes between regions (external pathway)
        - Synapses form ON target dendrites (internal region weights)
        - Each target neuron has many synaptic inputs on its dendrites

        For example (corticostriatal projection):
        - Cortex L5 (128), Hippocampus (64), PFC (32) → Striatum
        - External pathway: concatenates [128+64+32] → [224]
        - Striatum receives [224] and internal D1/D2 weights [70, 224]
          represent synapses ON MSN dendrites
        - Each of 70 MSNs integrates 224 synaptic inputs

        Args:
            target_comp: Target component instance
            target_port: Input port specification

        Returns:
            Output size for pathway (= concatenated input size)
        """
        # External pathway outputs concatenated inputs
        # Target's internal weights handle synaptic integration
        return target_comp.n_input

    def build(
        self,
        use_parallel: Optional[bool] = None,
        n_workers: Optional[int] = None,
    ) -> DynamicBrain:
        """Build DynamicBrain from specifications.

        Args:
            use_parallel: Override instance use_parallel setting
            n_workers: Override instance n_workers setting

        Steps:
            1. Infer component input sizes from connections
            2. Validate component graph
            3. Instantiate all components from registry
            4. Instantiate all pathways from registry
            5. Create DynamicBrain with component graph

        Returns:
            Constructed DynamicBrain instance

        Raises:
            ValueError: If validation fails or size inference fails
        """
        # Use provided values or fall back to instance values
        final_use_parallel = use_parallel if use_parallel is not None else self.use_parallel
        final_n_workers = n_workers if n_workers is not None else self.n_workers

        # Infer n_input for components based on connections
        self._infer_component_sizes()

        # Validate before building
        issues = self.validate()
        errors = [msg for msg in issues if msg.startswith("Error:")]
        if errors:
            raise ValueError(f"Validation failed:\n" + "\n".join(errors))

        # Instantiate components
        components: Dict[str, NeuralComponent] = {}
        for name, spec in self._components.items():
            # Get config class from registry
            config_class = self._registry.get_config_class(
                spec.component_type, spec.registry_name
            )

            if config_class is None:
                # Legacy component without config class metadata
                # Try to instantiate directly (will fail if config required)
                raise ValueError(
                    f"Component '{spec.registry_name}' has no config_class "
                    f"registered. Update registry with config_class metadata."
                )

            # Create config instance with device and dt_ms
            # Filter out internal flags (starting with _)
            config_params_with_globals = {
                k: v for k, v in spec.config_params.items()
                if not k.startswith("_")  # Remove internal flags like _has_topdown
            }
            config_params_with_globals.update({
                "device": self.global_config.device,
                "dt_ms": self.global_config.dt_ms,
            })
            config = config_class(**config_params_with_globals)

            # Create component from registry
            component = self._registry.create(
                spec.component_type,
                spec.registry_name,
                config=config,
            )

            # Move component to correct device (config device string might not be applied)
            if hasattr(component, 'to'):
                component.to(self.global_config.device)

            components[name] = component
            spec.instance = component

        # === MULTI-SOURCE PATHWAY CONSTRUCTION ===
        # Instantiate pathways - GROUP BY TARGET for multi-source pathways
        connections: Dict[Tuple[str, str], NeuralComponent] = {}

        # Group connections by target to create multi-source pathways
        connections_by_target: Dict[str, List[ConnectionSpec]] = {}
        for spec in self._connections:
            if spec.target not in connections_by_target:
                connections_by_target[spec.target] = []
            connections_by_target[spec.target].append(spec)

        # Create one pathway per target (multi-source if multiple inputs)
        for target_name, target_specs in connections_by_target.items():
            target_comp = components[target_name]

            if len(target_specs) == 1:
                # Single source - use standard pathway
                spec = target_specs[0]
                source_comp = components[spec.source]

                # Special handling for AxonalProjection (v2.0 architecture)
                if spec.pathway_type in ("axonal", "axonal_projection"):
                    pathway = self._create_axonal_projection(
                        target_specs, components, target_name
                    )
                    connections[(spec.source, spec.target)] = pathway
                    spec.instance = pathway
                    continue

                # Determine pathway component type (should be "pathway")
                pathway_component_type = None
                for ctype in ["pathway", "module"]:
                    if self._registry.is_registered(ctype, spec.pathway_type):
                        pathway_component_type = ctype
                        break

                if pathway_component_type is None:
                    raise ValueError(
                        f"Pathway '{spec.pathway_type}' not found in registry"
                    )

                # Get pathway config class
                config_class = self._registry.get_config_class(
                    pathway_component_type, spec.pathway_type
                )

                if config_class is None:
                    raise ValueError(
                        f"Pathway '{spec.pathway_type}' has no config_class "
                        f"registered. Update registry with config_class metadata."
                    )

                # Determine pathway sizes based on ports
                source_output_size = self._get_pathway_source_size(source_comp, spec.source_port)
                target_input_size = self._get_pathway_target_size(target_comp, spec.target_port)

                # Create pathway config with source/target sizes and global params
                # Filter out port specifications (they're not pathway config params)
                filtered_config_params = {
                    k: v for k, v in spec.config_params.items()
                    if not k.startswith("_")  # Remove internal flags
                }

                pathway_config_params = {
                    "n_input": source_output_size,
                    "n_output": target_input_size,
                    "device": self.global_config.device,
                    "dt_ms": self.global_config.dt_ms,
                    **filtered_config_params,  # User-specified params override defaults
                }
                pathway_config = config_class(**pathway_config_params)

                # Create pathway from registry
                pathway = self._registry.create(
                    pathway_component_type,
                    spec.pathway_type,
                    config=pathway_config,
                )

                # Move pathway to correct device
                if hasattr(pathway, 'to'):
                    pathway.to(self.global_config.device)

                connections[(spec.source, spec.target)] = pathway
                spec.instance = pathway

            else:
                # Multiple sources
                # Check if all specs request axonal pathway
                all_axonal = all(s.pathway_type in ("axonal", "axonal_projection")
                                for s in target_specs)

                if all_axonal:
                    # Create multi-source AxonalProjection
                    pathway = self._create_axonal_projection(
                        target_specs, components, target_name
                    )
                    for spec in target_specs:
                        connections[(spec.source, spec.target)] = pathway
                        spec.instance = pathway
                    continue

                # Multiple sources - create MultiSourcePathway
                from thalia.pathways.multi_source_pathway import MultiSourcePathway
                from thalia.core.base.component_config import PathwayConfig

                # Collect source information
                sources = []
                total_input_size = 0
                for spec in target_specs:
                    source_comp = components[spec.source]
                    source_output_size = self._get_pathway_source_size(source_comp, spec.source_port)
                    # Store port info for documentation (extraction happens in DynamicBrain before buffering)
                    sources.append((spec.source, spec.source_port))
                    total_input_size += source_output_size

                # Use first spec's config params as base (they should be compatible)
                base_spec = target_specs[0]
                filtered_config_params = {
                    k: v for k, v in base_spec.config_params.items()
                    if not k.startswith("_")
                }

                # Create multi-source pathway config
                pathway_config = PathwayConfig(
                    n_input=total_input_size,  # Will be recalculated by MultiSourcePathway
                    n_output=self._get_pathway_target_size(target_comp, base_spec.target_port),
                    device=self.global_config.device,
                    dt_ms=self.global_config.dt_ms,
                    **filtered_config_params,
                )

                # Create multi-source pathway
                pathway = MultiSourcePathway(
                    sources=sources,
                    target=target_name,
                    config=pathway_config,
                )

                # Set correct sizes for each source
                for spec in target_specs:
                    source_comp = components[spec.source]
                    source_output_size = self._get_pathway_source_size(source_comp, spec.source_port)
                    pathway.set_source_size(spec.source, source_output_size)

                # Move to correct device
                pathway.to(self.global_config.device)

                # Register pathway for all source->target pairs
                for spec in target_specs:
                    connections[(spec.source, spec.target)] = pathway
                    spec.instance = pathway

        # Create DynamicBrain
        brain = DynamicBrain(
            components=components,
            connections=connections,
            global_config=self.global_config,
            use_parallel=final_use_parallel,
            n_workers=final_n_workers,
            connection_specs={(spec.source, spec.target): spec for spec in self._connections},
        )

        # Store component specs for event adapter lookup
        brain._component_specs = {name: spec for name, spec in self._components.items()}

        # Store registry reference for adapter lookup
        brain._registry = self._registry

        return brain

    def save_spec(self, filepath: Path) -> None:
        """Save component graph specification to JSON.

        Allows sharing/versioning of brain architectures without code.

        Args:
            filepath: Path to JSON file

        Example:
            builder.save_spec(Path("architectures/my_brain.json"))
        """
        spec = {
            "components": [
                {
                    "name": spec.name,
                    "registry_name": spec.registry_name,
                    "config_params": spec.config_params,
                }
                for spec in self._components.values()
            ],
            "connections": [
                {
                    "source": spec.source,
                    "target": spec.target,
                    "pathway_type": spec.pathway_type,
                    "config_params": spec.config_params,
                }
                for spec in self._connections
            ],
        }

        with open(filepath, "w") as f:
            json.dump(spec, f, indent=2)

    @classmethod
    def load_spec(
        cls,
        filepath: Path,
        global_config: "GlobalConfig",
    ) -> "BrainBuilder":
        """Load component graph specification from JSON.

        Args:
            filepath: Path to JSON file
            global_config: Global configuration

        Returns:
            BrainBuilder populated with loaded specification

        Example:
            builder = BrainBuilder.load_spec(
                Path("architectures/my_brain.json"),
                global_config
            )
            brain = builder.build()
        """
        with open(filepath, "r") as f:
            spec = json.load(f)

        builder = cls(global_config)

        # Add components
        for comp in spec["components"]:
            builder.add_component(
                name=comp["name"],
                registry_name=comp["registry_name"],
                **comp["config_params"],
            )

        # Add connections
        for conn in spec["connections"]:
            builder.connect(
                source=conn["source"],
                target=conn["target"],
                pathway_type=conn["pathway_type"],
                **conn["config_params"],
            )

        return builder

    @classmethod
    def register_preset(
        cls,
        name: str,
        description: str,
        builder_fn: "PresetBuilderFn",
    ) -> None:
        """Register a preset architecture.

        Args:
            name: Preset name (e.g., "sensorimotor", "minimal")
            description: Human-readable description
            builder_fn: Function that configures a BrainBuilder

        Example:
            def build_minimal(builder: BrainBuilder):
                builder.add_component("input", "thalamic_relay", n_neurons=64)
                builder.add_component("output", "layered_cortex", n_neurons=128)
                builder.connect("input", "output")

            BrainBuilder.register_preset(
                name="minimal",
                description="Minimal 2-component brain for testing",
                builder_fn=build_minimal,
            )
        """
        cls._presets[name] = PresetArchitecture(
            name=name,
            description=description,
            builder_fn=builder_fn,
        )

    @classmethod
    def preset(
        cls,
        name: str,
        global_config: "GlobalConfig",
        use_parallel: bool = False,
        n_workers: Optional[int] = None,
        **overrides: Any,
    ) -> DynamicBrain:
        """Create brain from preset architecture.

        Args:
            name: Preset name (e.g., "sensorimotor")
            global_config: Global configuration
            use_parallel: Enable parallel execution (default: False)
            n_workers: Number of worker processes (default: auto)
            **overrides: Override default preset parameters

        Returns:
            Constructed DynamicBrain instance

        Raises:
            KeyError: If preset name not found

        Example:
            brain = BrainBuilder.preset("sensorimotor", global_config)

            # With parallel execution
            brain = BrainBuilder.preset(
                "sensorimotor",
                global_config,
                use_parallel=True,
                n_workers=4,
            )

            # With overrides
            brain = BrainBuilder.preset(
                "sensorimotor",
                global_config,
                cortex_neurons=1024,  # Override default
            )
        """
        if name not in cls._presets:
            available = list(cls._presets.keys())
            raise KeyError(
                f"Preset '{name}' not found. Available: {available}"
            )

        preset = cls._presets[name]
        builder = cls(global_config, use_parallel=use_parallel, n_workers=n_workers)

        # Apply preset builder function
        preset.builder_fn(builder, **overrides)

        return builder.build()

    @classmethod
    def list_presets(cls) -> List[Tuple[str, str]]:
        """List available preset architectures.

        Returns:
            List of (name, description) tuples
        """
        return [
            (name, preset.description)
            for name, preset in cls._presets.items()
        ]

    @classmethod
    def preset_builder(
        cls,
        name: str,
        global_config: "GlobalConfig",
        use_parallel: bool = False,
        n_workers: Optional[int] = None,
    ) -> "BrainBuilder":
        """Create builder initialized with preset architecture.

        Unlike preset(), this returns the builder so you can modify it
        before calling build().

        Args:
            name: Preset name (e.g., "sensorimotor")
            global_config: Global configuration
            use_parallel: Enable parallel execution (default: False)
            n_workers: Number of worker processes (default: auto)

        Returns:
            BrainBuilder instance with preset applied

        Raises:
            KeyError: If preset name not found

        Example:
            # Start with preset and add custom components
            builder = BrainBuilder.preset_builder("minimal", global_config)
            builder.add_component("pfc", "prefrontal", n_input=128, n_output=64)
            builder.connect("cortex", "pfc", pathway_type="spiking")
            brain = builder.build()
        """
        if name not in cls._presets:
            available = list(cls._presets.keys())
            raise KeyError(f"Preset '{name}' not found. Available: {available}")

        preset = cls._presets[name]
        builder = cls(global_config, use_parallel, n_workers)
        preset.builder_fn(builder)
        return builder


# Type alias for preset builder functions
from typing import Callable
PresetBuilderFn = Callable[["BrainBuilder", Any], None]


class PresetArchitecture:
    """Container for preset architecture definition."""

    def __init__(
        self,
        name: str,
        description: str,
        builder_fn: PresetBuilderFn,
    ):
        self.name = name
        self.description = description
        self.builder_fn = builder_fn


# ============================================================================
# Built-in Preset Architectures
# ============================================================================

def _build_minimal(builder: BrainBuilder, **overrides: Any) -> None:
    """Minimal 3-component brain for testing.

    Architecture:
        input (64) → process (128) → output (64)

    **Pathway Types**:
    - Input stage uses "thalamic_relay" region (has relay neurons)
    - Connections use AXONAL projections (v2.0 architecture)
    """
    n_input = overrides.get("n_input", 64)
    n_process = overrides.get("n_process", 128)
    n_output = overrides.get("n_output", 64)

    # Input interface - must specify both n_input and n_output
    # Uses thalamic relay (which has real neurons for sensory filtering)
    builder.add_component("input", "thalamic_relay", n_input=n_input, n_output=n_input)

    # Processing components - only n_output needed, n_input inferred from connections
    builder.add_component("process", "layered_cortex", **calculate_layer_sizes(n_process))
    builder.add_component("output", "layered_cortex", **calculate_layer_sizes(n_output))

    # Connections use axonal projections (pure spike routing)
    builder.connect("input", "process", pathway_type="axonal")
    builder.connect("process", "output", pathway_type="axonal")


def _build_sensorimotor(builder: BrainBuilder, **overrides: Any) -> None:
    """Sensorimotor architecture (6-region default).

    Architecture:
        Thalamus → Cortex ⇄ Hippocampus
                     ↓
                    PFC ⇄ Striatum
                     ↓
                Cerebellum

    This preset provides the standard 6-region sensorimotor architecture.

    **Pathway Types** (v2.0 Architecture):
    - Thalamus is a REGION (has relay neurons), not a pathway
    - All inter-region connections use AXONAL projections (pure spike routing)
    - Synapses are owned by target regions (biologically accurate)
    """
    # Default sizes (can be overridden)
    n_thalamus = overrides.get("n_thalamus", 128)
    n_cortex = overrides.get("n_cortex", 500)
    n_hippocampus = overrides.get("n_hippocampus", 200)
    n_pfc = overrides.get("n_pfc", 300)
    n_striatum = overrides.get("n_striatum", 150)
    n_cerebellum = overrides.get("n_cerebellum", 100)

    # Add regions (only thalamus needs n_input as it's the input interface)
    builder.add_component("thalamus", "thalamus", n_input=n_thalamus, n_output=n_thalamus)
    builder.add_component("cortex", "cortex", **calculate_layer_sizes(n_cortex))
    builder.add_component("hippocampus", "hippocampus", n_output=n_hippocampus)
    builder.add_component("pfc", "prefrontal", n_output=n_pfc)
    builder.add_component("striatum", "striatum", n_output=n_striatum)
    builder.add_component("cerebellum", "cerebellum", n_output=n_cerebellum)

    # Add connections using AXONAL projections (v2.0 architecture)
    # Why axonal? These are long-range projections with NO intermediate computation.
    # Synapses are located at TARGET dendrites, not in the pathway.

    # Thalamus → Cortex: Thalamocortical projection
    builder.connect("thalamus", "cortex", pathway_type="axonal")

    # Cortex L6 → Thalamus: Corticothalamic feedback for attentional modulation
    # L6 projects to TRN to implement selective attention (feedback loop)
    builder.connect("cortex", "thalamus", pathway_type="axonal", source_port="l6")

    # Cortex ⇄ Hippocampus: Bidirectional memory integration
    builder.connect("cortex", "hippocampus", pathway_type="axonal")
    builder.connect("hippocampus", "cortex", pathway_type="axonal")

    # Cortex → PFC: Executive control pathway
    builder.connect("cortex", "pfc", pathway_type="axonal")

    # Multi-source → Striatum: Corticostriatal + hippocampostriatal + PFC inputs
    # These will be automatically combined into single multi-source AxonalProjection
    builder.connect("cortex", "striatum", pathway_type="axonal")
    builder.connect("hippocampus", "striatum", pathway_type="axonal")
    builder.connect("pfc", "striatum", pathway_type="axonal")

    # Striatum → PFC: Basal ganglia gating of working memory
    builder.connect("striatum", "pfc", pathway_type="axonal")

    # Cerebellum: Motor/cognitive forward models
    # Receives multi-modal input (sensory + goals), outputs predictions
    builder.connect("cortex", "cerebellum", pathway_type="axonal")  # Sensorimotor input
    builder.connect("pfc", "cerebellum", pathway_type="axonal")     # Goal/context input
    builder.connect("cerebellum", "cortex", pathway_type="axonal")  # Forward model predictions


# Register built-in presets
BrainBuilder.register_preset(
    name="minimal",
    description="Minimal 3-component brain for testing (input→process→output)",
    builder_fn=_build_minimal,
)

BrainBuilder.register_preset(
    name="sensorimotor",
    description="6-region sensorimotor architecture (standard default)",
    builder_fn=_build_sensorimotor,
)


__all__ = [
    "BrainBuilder",
    "PresetArchitecture",
]
