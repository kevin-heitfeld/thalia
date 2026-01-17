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
    brain = BrainBuilder.preset("default", global_config)

Use fluent builder pattern:
    brain = (
        BrainBuilder(config)
        .add_component("thalamus", "thalamic_relay", n_neurons=100)
        .add_component("cortex", "layered_cortex", n_neurons=500)
        .connect("thalamus", "cortex", "spiking")
        .build()
    )

Or use preset architectures:
    brain = BrainBuilder.preset("default", config)

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

### 2. AxonalProjection Architecture
- **Pure axonal transmission**: Delays only, no weights
- **Synapses at target**: Weights stored in target region's dendrites
- **Multi-source support**: Multiple input streams to single target
- **Usage**: `builder.connect(src, tgt, pathway_type="axonal")`

**Benefits of AxonalProjection:**
1. Matches biology (axons ≠ synapses)
2. Clear synaptic ownership (target owns synapses)
3. Simpler growth (no double-resize)
4. Better checkpointing (weights in regions, not pathways)

Author: Thalia Project
Date: December 15, 2025
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from thalia.config import GlobalConfig
from thalia.config.size_calculator import LayerSizeCalculator
from thalia.core.component_spec import ComponentSpec, ConnectionSpec
from thalia.core.dynamic_brain import DynamicBrain
from thalia.core.protocols.component import BrainComponent, LearnableComponent
from thalia.managers.component_registry import ComponentRegistry
from thalia.pathways.axonal_projection import AxonalProjection

# Size parameter names that should be separated from behavioral config
SIZE_PARAMS = {
    "l4_size",
    "l23_size",
    "l5_size",
    "l6a_size",
    "l6b_size",
    "input_size",
    "output_size",
    "n_input",
    "n_output",
    "dg_size",
    "ca3_size",
    "ca2_size",
    "ca1_size",  # Hippocampus
    "relay_size",
    "trn_size",  # Thalamus
    "purkinje_size",
    "granule_size",
    "basket_size",
    "stellate_size",
    "dcn_size",  # Cerebellum
    "n_neurons",
    "n_actions",
    "neurons_per_action",  # Generic + Striatum
    "d1_size",
    "d2_size",  # Striatum pathways
    "input_sources",  # Striatum multi-source
    "total_neurons",  # Computed size metadata
}


def _separate_size_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate size parameters from behavioral parameters.

    Args:
        params: Combined parameters dict

    Returns:
        Tuple of (behavioral_params, size_params)
    """
    behavioral = {}
    sizes = {}

    for key, value in params.items():
        if key in SIZE_PARAMS:
            sizes[key] = value
        else:
            behavioral[key] = value

    return behavioral, sizes


def _compute_region_sizes(registry_name: str, size_params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute concrete sizes for a region from semantic size parameters.

    Some regions need size transformation:
    - Striatum: n_actions + neurons_per_action → d1_size, d2_size
    - Cerebellum: purkinje_size → granule_size
    - Hippocampus: input_size → dg_size, ca3_size, ca1_size, ca2_size
    - Cortex: l4_size, l23_size, l5_size → complete layer sizes

    Args:
        registry_name: Name of region type
        size_params: Raw size parameters from builder

    Returns:
        Computed size parameters suitable for region constructor
    """
    calc = LayerSizeCalculator()

    if registry_name == "striatum":
        # Striatum needs n_actions + neurons_per_action → d1_size, d2_size
        if "n_actions" in size_params:
            n_actions = size_params["n_actions"]
            neurons_per_action = size_params.get("neurons_per_action", 10)

            # Compute d1_size and d2_size
            computed = calc.striatum_from_actions(n_actions, neurons_per_action)

            # Merge computed sizes with original params
            result = {**size_params, **computed}
            return result

    elif registry_name == "cerebellum":
        # Cerebellum needs purkinje_size → granule_size
        if "purkinje_size" in size_params and "granule_size" not in size_params:
            purkinje_size = size_params["purkinje_size"]
            # expansion = size_params.get("granule_expansion_factor", 4.0)
            computed = calc.cerebellum_from_output(purkinje_size)
            return {**size_params, **computed}

    elif registry_name == "hippocampus":
        # Hippocampus needs input_size → all layer sizes
        if "input_size" in size_params and "dg_size" not in size_params:
            input_size = size_params["input_size"]
            computed = calc.hippocampus_from_input(input_size)
            return {**size_params, **computed}

    # No transformation needed for this region type
    return size_params


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
        brain = BrainBuilder.preset("default", global_config)

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
    _presets: Dict[str, PresetArchitecture] = {}

    def __init__(
        self,
        global_config: GlobalConfig,
    ):
        """Initialize builder with global configuration.

        Args:
            global_config: Global configuration (device, dt_ms, etc.)
        """
        self.global_config = global_config
        self._components: Dict[str, ComponentSpec] = {}
        self._connections: List[ConnectionSpec] = []
        self._registry = ComponentRegistry()

    def add_component(
        self,
        name: str,
        registry_name: str,
        **config_params: Any,
    ) -> BrainBuilder:
        """Add a component (region, pathway, module) to the brain.

        Size Inference:
            - Semantic size fields are required (e.g., relay_size, layer_sizes)
            - input_size is automatically inferred from incoming connections
            - Components without incoming connections (input interfaces) must specify input_size
            - This prevents size mismatches between components and pathways

        Args:
            name: Instance name (e.g., "my_cortex", "visual_input")
            registry_name: Component type in registry (e.g., "layered_cortex")
            **config_params: Configuration parameters. Required fields depend on region:
                - Thalamus: input_size, relay_size
                - Cortex: layer sizes (l4_size, l23_size, l5_size, l6a_size, l6b_size)
                - Hippocampus: ca1_size (or dg_size, ca3_size, ca2_size, ca1_size)
                - Prefrontal: wm_size
                - Striatum: n_actions, neurons_per_action
                - Cerebellum: purkinje_size

        Returns:
            Self for method chaining

        Raises:
            ValueError: If component name already exists
            KeyError: If registry_name not found in ComponentRegistry

        Example:
            # Input interface (no incoming connections) - needs input_size
            builder.add_component(
                name="sensory_input",
                registry_name="thalamic_relay",
                input_size=128,
                relay_size=128,
            )

            # Processing component - input_size inferred from connection
            builder.add_component(
                name="cortex",
                registry_name="layered_cortex",
                l4_size=64, l23_size=96, l5_size=32, l6a_size=16, l6b_size=16,
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
            raise KeyError(f"Registry name '{registry_name}' not found. " f"Available: {available}")

        # Ensure component_type is a string (should always be from found entry)
        if component_type is None:
            raise ValueError(f"Component type not found for '{registry_name}'")

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
        pathway_type: str = "axonal_projection",
        source_port: Optional[str] = None,
        target_port: Optional[str] = None,
        **config_params: Any,
    ) -> BrainBuilder:
        """Connect two components with a pathway.

        Args:
            source: Source component name
            target: Target component name
            pathway_type: Pathway registry name:
                - "axonal_projection": AxonalProjection (pure spike routing, NO weights)
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
            # Simple connection (default is axonal_projection)
            builder.connect("thalamus", "cortex")

            # Axonal projection with custom delay
            builder.connect("cortex", "striatum", pathway_type="axonal_projection", source_port="l5")

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
            issues.append(f"Warning: Isolated components (no connections): {isolated}")

        return issues

    def _infer_component_sizes(self) -> None:
        """Infer input_size for components based on incoming connections.

        Supports port-based routing:
        - source_port: Routes specific layer outputs (e.g., 'l23', 'l5')
        - target_port: Differentiates input types (e.g., 'feedforward', 'top_down')

        For components without input_size specified:
        - Feedforward ports (None, 'feedforward', 'cortical', 'hippocampal'): Counted in input_size
        - Top-down/modulation ports ('top_down', 'pfc_modulation'): Set as separate config params
        - Sensory ports ('ec_l3'): Set as ec_l3_input_size
        """
        # Build incoming connection map with port information
        incoming: Dict[str, List[Tuple[str, ConnectionSpec]]] = {
            name: [] for name in self._components
        }
        for conn in self._connections:
            incoming[conn.target].append((conn.source, conn))

        # Infer n_input and port-specific sizes for each component
        for name, spec in self._components.items():
            sources = incoming[name]

            if not sources:
                # No incoming connections - this is an input interface
                if "input_size" not in spec.config_params:
                    raise ValueError(
                        f"Component '{name}' has no incoming connections and no input_size specified. "
                        f"Input interfaces must specify 'input_size' explicitly."
                    )
                continue

            # Separate connections by target port
            feedforward_sources = []
            topdown_sources = []
            sensory_sources = []
            modulation_sources = []
            feedback_sources = []  # For L6 feedback to thalamus

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
                elif target_port in ("l6a_feedback", "l6b_feedback"):
                    # L6 corticothalamic feedback (separate from feedforward)
                    # l6a_feedback: type I corticothalamic (L6a → TRN)
                    # l6b_feedback: type II corticothalamic (L6b → relay)
                    feedback_sources.append((source_name, conn))
                else:
                    # Unknown port - treat as feedforward with warning
                    print(
                        f"Warning: Unknown target_port '{target_port}' on connection to '{name}', treating as feedforward"
                    )
                    feedforward_sources.append((source_name, conn))

            # Infer input_size from feedforward connections
            if "input_size" not in spec.config_params and feedforward_sources:
                feedforward_sizes = []
                for source_name, conn in feedforward_sources:
                    output_size = self._get_source_output_size(source_name, conn.source_port)
                    feedforward_sizes.append(output_size)

                spec.config_params["input_size"] = sum(feedforward_sizes)

            # Set port-specific sizes
            if sensory_sources:
                # ec_l3_input_size for hippocampus
                ec_l3_size = sum(
                    self._get_source_output_size(src, conn.source_port)
                    for src, conn in sensory_sources
                )
                spec.config_params["ec_l3_input_size"] = ec_l3_size

            # Note: top_down, pfc_modulation, and l6_feedback are handled separately in forward pass
            # They don't contribute to n_input but are stored for reference
            if topdown_sources:
                spec.config_params["_has_topdown"] = True
            if modulation_sources:
                spec.config_params["_has_modulation"] = True
            if feedback_sources:
                spec.config_params["_has_feedback"] = True

    def _get_output_size_from_params(self, spec: ComponentSpec) -> Optional[int]:
        """Get output size from config params using semantic field names.

        Args:
            spec: Component specification

        Returns:
            Output size if determinable, None otherwise
        """
        params = spec.config_params
        registry_name = spec.registry_name
        calc = LayerSizeCalculator()

        # Region-specific semantic output size computation
        if registry_name in ("cortex", "layered_cortex", "predictive_cortex"):
            # Cortex: output_size = l23_size + l5_size
            if "l23_size" in params and "l5_size" in params:
                return int(params["l23_size"] + params["l5_size"])

        elif registry_name in ("thalamus", "thalamic_relay"):
            # Thalamus: output_size = relay_size
            if "relay_size" in params:
                return int(params["relay_size"])

        elif registry_name == "hippocampus":
            # Hippocampus: output_size = ca1_size
            if "ca1_size" in params:
                return int(params["ca1_size"])
            # If ca1_size not specified but input_size is, compute it
            elif "input_size" in params:
                sizes = calc.hippocampus_from_input(params["input_size"])
                return sizes["ca1_size"]

        elif registry_name == "prefrontal":
            # Prefrontal: output_size = n_neurons (working memory neurons)
            if "n_neurons" in params:
                return int(params["n_neurons"])

        elif registry_name == "striatum":
            # Striatum: output_size = d1_size + d2_size = 2 * n_actions * neurons_per_action
            if "n_actions" in params and "neurons_per_action" in params:
                return int(2 * params["n_actions"] * params["neurons_per_action"])

        elif registry_name == "cerebellum":
            # Cerebellum: output_size = purkinje_size
            if "purkinje_size" in params:
                return int(params["purkinje_size"])

        # Could not determine output size
        return None

    def _get_source_output_size(self, source_name: str, source_port: Optional[str]) -> int:
        """Get output size from source component, optionally from specific port.

        Args:
            source_name: Name of source component
            source_port: Output port ('l23', 'l5', or None for full output)

        Returns:
            Output size (layer-specific if port specified)
        """
        source_spec = self._components[source_name]

        # Check for output size in either legacy or semantic form
        output_size = self._get_output_size_from_params(source_spec)
        if output_size is None:
            raise ValueError(
                f"Component '{source_name}' must specify output size before connecting. "
                f"For cortex: specify layer_sizes; for thalamus: relay_size; "
                f"for striatum: n_actions + neurons_per_action; etc."
            )

        # If no port specified, use full output
        if source_port is None:
            return output_size

        # For layered cortex (including predictive_cortex), get layer sizes
        if source_spec.registry_name in ("cortex", "layered_cortex", "predictive_cortex"):
            config = source_spec.config_params

            # All layer sizes are now required
            if not all(
                k in config for k in ["l4_size", "l23_size", "l5_size", "l6a_size", "l6b_size"]
            ):
                raise ValueError(
                    f"BrainBuilder: Cortex '{source_name}' must specify all layer sizes "
                    f"(l4_size, l23_size, l5_size, l6a_size, l6b_size)"
                )

            l4_size = config["l4_size"]
            l23_size = config["l23_size"]
            l5_size = config["l5_size"]
            l6a_size = config["l6a_size"]
            l6b_size = config["l6b_size"]

            # Return layer-specific size
            if source_port == "l23":
                return int(l23_size)
            elif source_port == "l5":
                return int(l5_size)
            elif source_port == "l4":
                return int(l4_size)
            elif source_port == "l6a":
                return int(l6a_size)
            elif source_port == "l6b":
                return int(l6b_size)
            else:
                raise ValueError(f"Unknown cortex port '{source_port}'")

        # For other components, ports not yet supported
        if source_port is not None:
            raise ValueError(
                f"Component '{source_spec.registry_name}' does not support port '{source_port}'"
            )

        # Should never reach here - output_size should have been determined above
        raise ValueError(
            f"Could not determine output size for '{source_name}'. "
            f"Registry: {source_spec.registry_name}, Params: {list(source_spec.config_params.keys())}"
        )

    def _get_pathway_source_size(
        self, source_comp: LearnableComponent, source_port: Optional[str]
    ) -> int:
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
        if hasattr(source_comp, "l23_size") and hasattr(source_comp, "l5_size"):
            if source_port == "l23":
                return int(source_comp.l23_size)  # type: ignore[arg-type]
            elif source_port == "l5":
                return int(source_comp.l5_size)  # type: ignore[arg-type]
            elif source_port == "l4" and hasattr(source_comp, "l4_size"):
                return int(source_comp.l4_size)  # type: ignore[arg-type]
            elif source_port == "l6a" and hasattr(source_comp, "l6a_size"):
                return int(source_comp.l6a_size)  # type: ignore[arg-type]
            elif source_port == "l6b" and hasattr(source_comp, "l6b_size"):
                return int(source_comp.l6b_size)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown cortex port '{source_port}'")

        # For other components, ports not yet supported
        raise ValueError(
            f"Component {type(source_comp).__name__} does not support port '{source_port}'"
        )

    def _initialize_target_weights(
        self,
        brain: DynamicBrain,
        components: Dict[str, LearnableComponent],
        connection_specs: Dict[Tuple[str, str], Any],
    ) -> None:
        """Initialize synaptic weights in target regions after pathways are created.

        For multi-source regions (striatum, NeuralRegion subclasses), notify them about
        their input sources so they can create appropriate synaptic weight matrices.

        Args:
            brain: The DynamicBrain instance
            components: Dict of component instances
            connection_specs: Connection specifications with port info
        """
        # Group connections by target to initialize weights once per target
        targets_to_sources: Dict[str, List[Tuple[str, Optional[str], int]]] = {}

        for (source_name, target_key), spec in connection_specs.items():
            # Handle compound keys like "striatum" or "striatum:feedforward"
            target_name = target_key.split(":")[0] if ":" in target_key else target_key
            # target_port = spec.target_port if hasattr(spec, "target_port") else None
            source_port = spec.source_port if hasattr(spec, "source_port") else None

            if target_name not in targets_to_sources:
                targets_to_sources[target_name] = []

            # Get source size
            source_comp = components[source_name]
            source_size = self._get_pathway_source_size(source_comp, source_port)

            targets_to_sources[target_name].append((source_name, source_port, source_size))

        # For each target, call add_input_source to initialize weights
        for target_name, sources in targets_to_sources.items():
            target_comp = components[target_name]

            # Special handling for Striatum (has add_input_source_striatum)
            # Check this FIRST before checking for standard add_input_source
            if hasattr(target_comp, "add_input_source_striatum"):
                for source_name, source_port, source_size in sources:
                    # Build source key (with port if specified)
                    source_key = f"{source_name}:{source_port}" if source_port else source_name

                    # Skip if input source already registered (avoid duplicate registration)
                    if (
                        hasattr(target_comp, "input_sources")
                        and source_key in target_comp.input_sources  # type: ignore[operator]
                    ):
                        continue

                    # Call striatum-specific method (creates D1/D2 weights)
                    target_comp.add_input_source_striatum(source_key, n_input=source_size)  # type: ignore[operator]

            # Check if target has standard add_input_source method (NeuralRegion, etc.)
            elif hasattr(target_comp, "add_input_source"):
                for source_name, source_port, source_size in sources:
                    # Build source key (with port if specified)
                    source_key = f"{source_name}:{source_port}" if source_port else source_name

                    # Skip if input source already registered (avoid duplicate registration)
                    if (
                        hasattr(target_comp, "input_sources")
                        and source_key in target_comp.input_sources  # type: ignore[operator]
                    ):
                        continue

                    # Call add_input_source (standard NeuralRegion method)
                    target_comp.add_input_source(source_key, n_input=source_size)  # type: ignore[operator]

    def _create_axonal_projection(
        self,
        target_specs: List[ConnectionSpec],
        components: Dict[str, LearnableComponent],
        target_name: str,
    ) -> AxonalProjection:
        """Create AxonalProjection from connection specs with per-target delay support.

        AxonalProjection has different initialization than standard pathways:
        - Takes list of (region_name, port, size, delay_ms[, target_delays]) tuples
        - NO config class with n_input/n_output
        - Handles multi-source concatenation internally
        - Supports per-target delay variation for realistic axonal branching

        Args:
            target_specs: List of ConnectionSpec for this target
            components: Dict of instantiated components
            target_name: Name of target component

        Returns:
            AxonalProjection instance with target-specific delays
        """
        # Build sources list: [(region_name, port, size, delay_ms[, target_delays]), ...]
        sources = []
        for spec in target_specs:
            source_comp = components[spec.source]
            source_size = self._get_pathway_source_size(source_comp, spec.source_port)

            # Get axonal delay (default: 2.0ms)
            delay_ms = spec.config_params.get("axonal_delay_ms", 2.0)

            # Check for per-target delays
            target_delays = spec.config_params.get("target_delays", None)

            if target_delays:
                # New format with per-target delays
                sources.append(
                    (
                        spec.source,  # region_name
                        spec.source_port,  # port (can be None)
                        source_size,  # size
                        delay_ms,  # default delay
                        target_delays,  # dict of target-specific delays
                    )
                )
            else:
                # Standard format (backward compatible)
                sources.append(
                    (
                        spec.source,  # region_name
                        spec.source_port,  # port (can be None)
                        source_size,  # size
                        delay_ms,  # delay_ms
                        {},  # empty target_delays dict for backward compatibility
                    )
                )

        # Create AxonalProjection with target_name for delay selection
        projection = AxonalProjection(
            sources=sources,  # type: ignore[arg-type]
            device=self.global_config.device,
            dt_ms=self.global_config.dt_ms,
            target_name=target_name,  # NEW: enables per-target delay selection
        )

        return projection

    def _get_pathway_target_size(
        self, target_comp: LearnableComponent, target_port: Optional[str]
    ) -> int:
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

    def build(self) -> DynamicBrain:
        """Build DynamicBrain from specifications.

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

        # Infer n_input for components based on connections
        self._infer_component_sizes()

        # Validate before building
        issues = self.validate()
        errors = [msg for msg in issues if msg.startswith("Error:")]
        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))

        # Instantiate components
        components: Dict[str, LearnableComponent] = {}
        for name, spec in self._components.items():
            # Get config class from registry
            config_class = self._registry.get_config_class(spec.component_type, spec.registry_name)

            if config_class is None:
                # Component missing config class metadata in registry
                raise ValueError(
                    f"Component '{spec.registry_name}' has no config_class "
                    f"registered. Update registry with config_class metadata."
                )

            # Create config instance with device and dt_ms
            # Filter out internal flags (starting with _)
            config_params_filtered = {
                k: v
                for k, v in spec.config_params.items()
                if not k.startswith("_")  # Remove internal flags like _has_topdown
            }

            # Separate size parameters from behavioral parameters
            behavioral_params, size_params = _separate_size_params(config_params_filtered)

            # Compute region-specific sizes (e.g., striatum n_actions → d1_size, d2_size)
            size_params = _compute_region_sizes(spec.registry_name, size_params)

            # Add global params to behavioral config
            behavioral_params.update(
                {
                    "device": self.global_config.device,
                    "dt_ms": self.global_config.dt_ms,
                }
            )
            config = config_class(**behavioral_params)

            # Create component from registry
            # Pass sizes separately if component needs them (e.g., LayeredCortex)
            component = self._registry.create(
                spec.component_type,
                spec.registry_name,
                config=config,
                sizes=size_params if size_params else {},
                device=self.global_config.device,
            )

            # Move component to correct device (config device string might not be applied)
            if hasattr(component, "to"):
                component.to(self.global_config.device)

            components[name] = cast(LearnableComponent, component)
            spec.instance = component

        # === MULTI-SOURCE PATHWAY CONSTRUCTION ===
        # Instantiate pathways - GROUP BY (TARGET, TARGET_PORT) for multi-source pathways
        # This allows multiple independent pathways to the same target (e.g., L6a and L6b to thalamus)
        connections: Dict[Tuple[str, str], Any] = {}

        # Group connections by (target, target_port) to create multi-source pathways
        # Key is (target_name, target_port) so L6a and L6b are separate groups
        connections_by_target_port: Dict[Tuple[str, Optional[str]], List[ConnectionSpec]] = {}
        conn_spec: ConnectionSpec
        for conn_spec in self._connections:
            group_key = (conn_spec.target, conn_spec.target_port)
            if group_key not in connections_by_target_port:
                connections_by_target_port[group_key] = []
            connections_by_target_port[group_key].append(conn_spec)

        # Create one pathway per (target, target_port) group (multi-source if multiple inputs)
        for (target_name, target_port), target_specs in connections_by_target_port.items():
            target_comp = components[target_name]

            pathway: AxonalProjection | BrainComponent

            if len(target_specs) == 1:
                # Single source - use standard pathway
                conn_spec = target_specs[0]
                source_comp = components[conn_spec.source]

                # Special handling for AxonalProjection (v2.0 architecture)
                if conn_spec.pathway_type in ("axonal", "axonal_projection"):
                    pathway = self._create_axonal_projection(target_specs, components, target_name)
                    # Use compound key if target_port specified (e.g., cortex→thalamus with l6a vs l6b)
                    # This allows multiple pathways with same (source, target) but different ports
                    conn_key = (
                        (conn_spec.source, f"{conn_spec.target}:{target_port}")
                        if target_port
                        else (conn_spec.source, conn_spec.target)
                    )
                    connections[conn_key] = pathway
                    conn_spec.instance = pathway
                    continue

                # Determine pathway component type (should be "pathway")
                pathway_component_type = None
                for ctype in ["pathway", "module"]:
                    if self._registry.is_registered(ctype, conn_spec.pathway_type):
                        pathway_component_type = ctype
                        break

                if pathway_component_type is None:
                    raise ValueError(f"Pathway '{conn_spec.pathway_type}' not found in registry")

                # Get pathway config class
                config_class = self._registry.get_config_class(
                    pathway_component_type, conn_spec.pathway_type
                )

                if config_class is None:
                    raise ValueError(
                        f"Pathway '{conn_spec.pathway_type}' has no config_class "
                        f"registered. Update registry with config_class metadata."
                    )

                # Determine pathway sizes based on ports
                source_output_size = self._get_pathway_source_size(
                    source_comp, conn_spec.source_port
                )
                target_input_size = self._get_pathway_target_size(
                    target_comp, conn_spec.target_port
                )

                # Create pathway config with source/target sizes and global params
                # Filter out port specifications (they're not pathway config params)
                filtered_config_params = {
                    k: v
                    for k, v in conn_spec.config_params.items()
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
                    conn_spec.pathway_type,
                    config=pathway_config,
                )

                # Move pathway to correct device
                if hasattr(pathway, "to"):
                    pathway.to(self.global_config.device)

                # Use compound key if target_port specified
                conn_key = (
                    (conn_spec.source, f"{conn_spec.target}:{target_port}")
                    if target_port
                    else (conn_spec.source, conn_spec.target)
                )
                connections[conn_key] = pathway
                conn_spec.instance = pathway

            else:
                # Multiple sources - use AxonalProjection
                pathway = self._create_axonal_projection(target_specs, components, target_name)
                # Use compound key if target_port specified
                first_spec = target_specs[0]
                conn_key = (
                    (first_spec.source, f"{first_spec.target}:{target_port}")
                    if target_port
                    else (first_spec.source, first_spec.target)
                )
                connections[conn_key] = pathway
                for conn_spec in target_specs:
                    conn_spec.instance = pathway

        # Create DynamicBrain
        # Build connection_specs dict with compound keys matching connections dict
        # For connections with target_port, use "target:port" key to avoid collisions
        connection_specs_dict = {}
        for conn_spec in self._connections:
            if conn_spec.target_port:
                key = (conn_spec.source, f"{conn_spec.target}:{conn_spec.target_port}")
            else:
                key = (conn_spec.source, conn_spec.target)
            connection_specs_dict[key] = conn_spec

        brain = DynamicBrain(
            components=components,
            connections=connections,
            global_config=self.global_config,
            connection_specs=connection_specs_dict,
        )

        # Store component specs for event adapter lookup
        brain._component_specs = {name: spec for name, spec in self._components.items()}

        # Store registry reference for adapter lookup
        brain._registry = self._registry

        # === INITIALIZE TARGET REGION WEIGHTS ===
        # After pathways are created, notify target regions about their input sources
        # so they can initialize synaptic weights (multi-source architecture)
        self._initialize_target_weights(brain, components, connection_specs_dict)

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
        global_config: GlobalConfig,
    ) -> BrainBuilder:
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
        builder_fn: PresetBuilderFn,
    ) -> None:
        """Register a preset architecture.

        Args:
            name: Preset name (e.g., "default", "minimal")
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
        global_config: GlobalConfig,
        **overrides: Any,
    ) -> DynamicBrain:
        """Create brain from preset architecture.

        Args:
            name: Preset name (e.g., "default")
            global_config: Global configuration
            **overrides: Override default preset parameters

        Returns:
            Constructed DynamicBrain instance

        Raises:
            KeyError: If preset name not found

        Example:
            brain = BrainBuilder.preset("default", global_config)

            # With overrides
            brain = BrainBuilder.preset(
                "default",
                global_config,
                cortex_neurons=1024,  # Override default
            )
        """
        if name not in cls._presets:
            available = list(cls._presets.keys())
            raise KeyError(f"Preset '{name}' not found. Available: {available}")

        preset = cls._presets[name]
        builder = cls(global_config)

        # Apply preset builder function
        preset.builder_fn(builder, **overrides)

        return builder.build()

    @classmethod
    def list_presets(cls) -> List[Tuple[str, str]]:
        """List available preset architectures.

        Returns:
            List of (name, description) tuples
        """
        return [(name, preset.description) for name, preset in cls._presets.items()]

    @classmethod
    def preset_builder(
        cls,
        name: str,
        global_config: GlobalConfig,
    ) -> BrainBuilder:
        """Create builder initialized with preset architecture.

        Unlike preset(), this returns the builder so you can modify it
        before calling build().

        Args:
            name: Preset name (e.g., "default")
            global_config: Global configuration

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
        builder = cls(global_config)
        preset.builder_fn(builder)
        return builder


# Type alias for preset builder functions
# Accepts BrainBuilder and optional keyword overrides
PresetBuilderFn = Callable[[BrainBuilder], None]


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

    **Semantic Configs**:
    - Thalamus: input_size, relay_size
    - Cortex: layer_sizes (computed from size parameter)
    """
    calc = LayerSizeCalculator()

    input_size = overrides.get("input_size", 64)
    process_size = overrides.get("process_size", 128)
    output_size = overrides.get("output_size", 64)

    # Input interface - uses thalamic relay (which has real neurons for sensory filtering)
    # Must specify input_size, relay_size, and trn_size explicitly (no incoming connections)
    builder.add_component(
        "input", "thalamic_relay", input_size=input_size, relay_size=input_size, trn_size=0
    )

    # Processing components - input_size inferred from connections
    builder.add_component("process", "layered_cortex", **calc.cortex_from_output(process_size))
    builder.add_component("output", "layered_cortex", **calc.cortex_from_output(output_size))

    # Connections use axonal projections (pure spike routing)
    builder.connect("input", "process", pathway_type="axonal")
    builder.connect("process", "output", pathway_type="axonal")


def _build_default(builder: BrainBuilder, **overrides: Any) -> None:
    """Default 6-region architecture for general-purpose learning.

    Architecture:
        Thalamus → Cortex ⇄ Hippocampus
                     ↓
                    PFC ⇄ Striatum
                     ↓
                Cerebellum

    This preset provides a balanced 6-region architecture suitable for:
    - Vision and audition (thalamocortical processing)
    - Sequential learning (hippocampal episodic memory)
    - Planning and working memory (prefrontal cortex)
    - Reinforcement learning (striatal reward processing)
    - Motor control and predictions (cerebellar forward models)

    **Pathway Types** (v2.0 Architecture):
    - Thalamus is a REGION (has relay neurons), not a pathway
    - All inter-region connections use AXONAL projections (pure spike routing)
    - Synapses are owned by target regions (biologically accurate)
    """
    calc = LayerSizeCalculator()

    # Default sizes (can be overridden)
    thalamus_relay_size = overrides.get("thalamus_relay_size", 128)
    cortex_size = overrides.get("cortex_size", 500)
    pfc_n_neurons = overrides.get("pfc_n_neurons", 300)
    striatum_actions = overrides.get("striatum_actions", 10)
    striatum_neurons_per_action = overrides.get("striatum_neurons_per_action", 15)
    cerebellum_purkinje_size = overrides.get("cerebellum_purkinje_size", 100)

    # Calculate cortex layer sizes
    # BIOLOGICAL CONSTRAINTS for corticothalamic feedback:
    # - L6b must match thalamus relay size (one-to-one direct modulation)
    # - L6a must match TRN size (which is trn_ratio * relay size, typically 20%)
    cortex_sizes = calc.cortex_from_output(cortex_size)
    cortex_sizes["l6b_size"] = thalamus_relay_size  # Override to match relay neurons
    cortex_sizes["l6a_size"] = int(thalamus_relay_size * 0.2)  # Match TRN size (20% of relay)

    # Cortex output size (L2/3 + L5) for computing downstream input sizes
    cortex_output_size = cortex_sizes["l23_size"] + cortex_sizes["l5_size"]

    # Add regions (only thalamus needs input_size as it's the input interface)
    thalamus_sizes = calc.thalamus_from_relay(thalamus_relay_size)
    builder.add_component("thalamus", "thalamus", **thalamus_sizes)
    builder.add_component("cortex", "cortex", **cortex_sizes)

    # Hippocampus: input from cortex (L2/3 + L5 = 1250)
    # Compute all layer sizes from this input using biological ratios
    # Result: DG=5000, CA3=1250, CA2=1000, CA1=2500 with default ratios
    hippocampus_input_size = cortex_output_size  # Will be verified by _infer_component_sizes
    hippocampus_sizes = calc.hippocampus_from_input(hippocampus_input_size)
    builder.add_component("hippocampus", "hippocampus", **hippocampus_sizes)

    # Prefrontal: specify n_neurons (working memory neurons)
    builder.add_component("pfc", "prefrontal", n_neurons=pfc_n_neurons)

    # Striatum: n_actions and neurons_per_action → d1_size, d2_size via _compute_region_sizes
    # input_size will be inferred from connections (cortex + hippocampus + pfc)
    builder.add_component(
        "striatum",
        "striatum",
        n_actions=striatum_actions,
        neurons_per_action=striatum_neurons_per_action,
    )

    # Cerebellum: specify purkinje_size
    builder.add_component("cerebellum", "cerebellum", purkinje_size=cerebellum_purkinje_size)

    # Add connections using AXONAL projections (v2.0 architecture)
    # Why axonal? These are long-range projections with NO intermediate computation.
    # Synapses are located at TARGET dendrites, not in the pathway.

    # Thalamus → Cortex: Thalamocortical projection
    # Fast, heavily myelinated pathway (Jones 2007, Sherman & Guillery 2006)
    # Distance: ~2-3cm, conduction velocity: ~10-20 m/s → 2-3ms delay
    builder.connect("thalamus", "cortex", pathway_type="axonal", axonal_delay_ms=2.5)

    # Cortex L6a/L6b → Thalamus: Dual corticothalamic feedback pathways
    # L6a (type I) → TRN: Inhibitory modulation for selective attention (slow pathway, low gamma 25-35 Hz)
    # L6b (type II) → Relay: Excitatory modulation for precision processing (fast pathway, high gamma 60-80 Hz)
    # Sherman & Guillery (2002): Dual pathways implement complementary feedback mechanisms
    #
    # Biologically realistic delays (Sherman & Guillery 2002):
    # - Corticothalamic axons: 8-12ms (documented in L6_TRN_FEEDBACK_LOOP.md)
    # - L6a→TRN: ~10ms (type I pathway, standard corticothalamic)
    # - L6b→Relay: ~5ms (type II pathway, slightly faster direct projection)
    #
    # Total feedback loop timing includes:
    # - Axonal delays (specified here): 10ms + 5ms
    # - Synaptic delays (~1-2ms per synapse): multiple synapses in loop
    # - Neural integration (tau_E=5ms, tau_I=10ms): conductance buildup/decay
    # - Membrane dynamics (tau_mem~20ms): integration to threshold
    # - Refractory periods (tau_ref): post-spike delays
    # These neural dynamics add ~10-20ms to total loop period, producing gamma oscillations
    builder.connect(
        "cortex",
        "thalamus",
        pathway_type="axonal",
        source_port="l6a",
        target_port="l6a_feedback",
        axonal_delay_ms=10.0,
    )
    builder.connect(
        "cortex",
        "thalamus",
        pathway_type="axonal",
        source_port="l6b",
        target_port="l6b_feedback",
        axonal_delay_ms=5.0,
    )

    # Cortex ⇄ Hippocampus: Bidirectional memory integration
    # Entorhinal cortex ↔ hippocampus: moderately myelinated (Witter et al. 2000)
    # Distance: ~3-5cm, conduction velocity: ~5-10 m/s → 5-8ms delay
    builder.connect("cortex", "hippocampus", pathway_type="axonal", axonal_delay_ms=6.5)
    builder.connect("hippocampus", "cortex", pathway_type="axonal", axonal_delay_ms=6.5)

    # Cortex → PFC: Executive control pathway
    # Corticocortical long-range connections (Miller & Cohen 2001)
    # Distance: ~5-10cm, conduction velocity: ~3-8 m/s → 10-15ms delay
    builder.connect("cortex", "pfc", pathway_type="axonal", axonal_delay_ms=12.5)

    # Multi-source → Striatum: Corticostriatal + hippocampostriatal + PFC inputs
    # These will be automatically combined into single multi-source AxonalProjection
    # Per-target delays model different myelination patterns (Gerfen & Surmeier 2011):
    # - Cortex → Striatum: Fast, heavily myelinated, short distance (~2-4cm) → 3-5ms
    # - Hippocampus → Striatum: Moderate, longer distance (~4-6cm) → 7-10ms
    # - PFC → Striatum: Variable, longest distance (~6-10cm) → 12-18ms
    builder.connect("cortex", "striatum", pathway_type="axonal", axonal_delay_ms=4.0)
    builder.connect("hippocampus", "striatum", pathway_type="axonal", axonal_delay_ms=8.5)
    builder.connect("pfc", "striatum", pathway_type="axonal", axonal_delay_ms=15.0)

    # Striatum → PFC: Basal ganglia gating of working memory
    # Via thalamus (MD/VA nuclei), total distance ~8-12cm (Haber 2003)
    # Includes striatum→thalamus→PFC relay → 15-20ms total delay
    builder.connect("striatum", "pfc", pathway_type="axonal", axonal_delay_ms=17.5)

    # Cerebellum: Motor/cognitive forward models
    # Receives multi-modal input (sensory + goals), outputs predictions
    # Corticopontocerebellar pathway: via pontine nuclei (Schmahmann 1996)
    # Distance: ~10-15cm total, includes relay → 20-30ms delay
    builder.connect(
        "cortex", "cerebellum", pathway_type="axonal", axonal_delay_ms=25.0
    )  # Sensorimotor input
    # PFC → Cerebellum: similar pathway length
    builder.connect(
        "pfc", "cerebellum", pathway_type="axonal", axonal_delay_ms=25.0
    )  # Goal/context input
    # Cerebellum → Cortex: via thalamus (VL/VA nuclei), moderately fast
    # Distance: ~8-12cm, includes thalamic relay → 15-20ms delay
    builder.connect(
        "cerebellum", "cortex", pathway_type="axonal", axonal_delay_ms=17.5
    )  # Forward model predictions


# Register built-in presets
BrainBuilder.register_preset(
    name="minimal",
    description="Minimal 3-component brain for testing (input→process→output)",
    builder_fn=_build_minimal,
)

BrainBuilder.register_preset(
    name="default",
    description="Default 6-region architecture for general-purpose learning",
    builder_fn=_build_default,
)

__all__ = [
    "BrainBuilder",
    "PresetArchitecture",
]
