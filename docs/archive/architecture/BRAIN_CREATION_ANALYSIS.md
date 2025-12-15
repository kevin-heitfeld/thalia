# Brain Creation Analysis & Improvement Recommendations (v2)

**Date:** December 15, 2025
**Analyst:** Software Architecture Expert
**Focus:** Current brain creation patterns and flexibility improvements
**Revision:** Updated for more ambitious refactoring (backwards compatibility not required)

---

## Executive Summary

The Thalia project currently uses **configuration-driven creation** via `ThaliaConfig` as the sole supported approach for brain creation. Legacy dictionary-based and factory methods have been removed.

**Key Finding:** The current system is rigid and tightly couples brain construction to a fixed architecture. A **component-based builder system** leveraging the existing `ComponentRegistry` infrastructure would enable:
- **User-defined custom components** (regions, pathways, modules)
- **Plugin architecture** for external extensions
- **Flexible brain topologies** beyond the hardcoded 6-region architecture
- **Dramatically simplified configuration** for common cases

**New Strategy:** Refactor brain creation to be fully dynamic and registry-based, treating the brain as a **graph of components** rather than a hardcoded structure.

---

## Current State Analysis

### Current Creation Method: `EventDrivenBrain.from_thalia_config()`

**Pattern:** Direct instantiation from hierarchical configuration object

```python
# Current approach (the only supported method)
config = ThaliaConfig(
    global_=GlobalConfig(device="cuda", dt_ms=1.0),
    brain=BrainConfig(
        sizes=RegionSizes(
            input_size=128,
            cortex_size=128,
            hippocampus_size=64,
            pfc_size=32,
            n_actions=7,
        ),
        encoding_timesteps=10,
        delay_timesteps=5,
        test_timesteps=10,
    ),
)

brain = EventDrivenBrain.from_thalia_config(config)
```

**Strengths:**
✅ Type-safe with dataclass validation
✅ Clear hierarchical structure (Global → Brain → Regions)
✅ Comprehensive with all necessary parameters
✅ Well-documented with examples
✅ Validated before creation via `validate_thalia_config()`

**Weaknesses:**
❌ Verbose for simple configurations
❌ Requires creating entire config hierarchy even for defaults
❌ No progressive/incremental assembly
❌ Hard to extend without modifying config dataclasses
❌ Difficult to share partial configurations
❌ No fluent/chainable interface

**Usage Pattern:**
- 100% of production code (training, notebooks, experiments)
- 100% of tests
- Clear standard across the codebase

---



## Problem Areas Requiring Improvement

### Problem 1: Configuration Verbosity

**Issue:** Even simple brain creation requires ~15-20 lines of nested configuration.

```python
# Current: Too much boilerplate for defaults
config = ThaliaConfig(
    global_=GlobalConfig(device="cuda"),  # Just want to change device
    brain=BrainConfig(                    # Everything else is default
        sizes=RegionSizes(),              # But must create full hierarchy
    ),
)
brain = EventDrivenBrain.from_thalia_config(config)
```

**Impact:**
- Discourages experimentation
- Harder to teach/onboard new users
- More code to maintain in tests

---

### Problem 2: No Progressive Assembly

**Issue:** Cannot build configuration incrementally or conditionally.

```python
# Current: Cannot do this
config = BrainConfig()
config.add_cortex(size=128)
config.add_hippocampus(size=64)
if use_planning:
    config.enable_planning()
brain = config.build()

# Must do this instead
cortex_size = 128
hippocampus_size = 64
use_model_based_planning = use_planning  # Must decide upfront

config = ThaliaConfig(
    global_=GlobalConfig(device="cuda"),
    brain=BrainConfig(
        sizes=RegionSizes(cortex_size=cortex_size, hippocampus_size=hippocampus_size),
        use_model_based_planning=use_model_based_planning,
    ),
)
```

**Impact:**
- Harder to build configurations programmatically
- Cannot easily compose configurations from multiple sources
- Difficult to create configuration templates/presets

---

### Problem 3: Difficult to Extend

**Issue:** Adding new region types or features requires modifying multiple config dataclasses.

**Example:** To add a new "Amygdala" region:
1. Add `amygdala_size` to `RegionSizes` ❌
2. Add `AmygdalaConfig` to `BrainConfig` ❌
3. Update `EventDrivenBrain.__init__()` ❌
4. Update size computation logic ❌
5. Update validation logic ❌

**Impact:**
- High coupling between regions and config system
- Changes ripple through multiple files
- Harder to experiment with new architectures

---

### Problem 4: No Configuration Sharing/Templates

**Issue:** Cannot easily share or reuse partial configurations.

```python
# Want to do this:
sensorimotor_base = BrainConfig.from_preset("sensorimotor")
my_config = sensorimotor_base.with_modifications(cortex_size=256)

# Current workaround: Copy-paste entire config
```

**Impact:**
- Configuration duplication across experiments
- Harder to maintain consistency
- Lost opportunity for best practices sharing

---

## Revised Recommendations (v2)

### Core Philosophy Change

**From:** Brain is a hardcoded structure with 6 fixed regions
**To:** Brain is a **directed graph of registered components**

This aligns with Thalia's existing infrastructure:
- ✅ `ComponentRegistry` already supports dynamic registration
- ✅ `NeuralComponent` protocol ensures uniform interfaces
- ✅ Component parity means regions and pathways are equals
- ✅ `@register_region`, `@register_pathway`, `@register_module` decorators exist

**Missing piece:** Brain construction still hardcoded in `EventDrivenBrain.__init__()` (lines 273-450)

---

## Recommendations

### Recommendation 1: Dynamic Component-Based Brain Builder

**Priority:** Critical
**Effort:** High (but architecturally cleaner long-term)
**Impact:** Transformative

**Proposal:** Replace hardcoded brain initialization with a component graph builder.

**New Architecture:**

```python
from thalia.core.brain_builder import BrainBuilder

# 1. SIMPLE CASE: Preset architectures
brain = BrainBuilder.from_preset("sensorimotor").with_device("cuda").build()

# 2. CUSTOM CASE: Declarative component graph
brain = (
    BrainBuilder()
    .with_device("cuda")
    .add_component("thalamus", "thalamic_relay", n_input=128, n_output=128)
    .add_component("cortex", "layered_cortex", n_input=128, n_output=256)
    .add_component("hippocampus", "trisynaptic", n_input=256, n_output=128)
    .add_component("striatum", "bg_circuit", n_input=256, n_output=10)
    .connect("thalamus", "cortex", pathway_type="spiking_stdp", delay_ms=5.0)
    .connect("cortex", "hippocampus", pathway_type="spiking_stdp", delay_ms=3.0)
    .connect("cortex", "striatum", pathway_type="spiking_stdp", delay_ms=5.0)
    .connect("hippocampus", "striatum", pathway_type="attention", delay_ms=6.0)
    .build()
)

# 3. PLUGIN CASE: User-defined custom components
from my_package import MyCustomRegion

@register_region("my_region", description="My custom region")
class MyCustomRegion(NeuralComponent):
    def __init__(self, config):
        super().__init__(config)
        # Custom implementation

    def forward(self, spikes):
        # Custom processing
        return output_spikes

# Now can use in brain builder
brain = (
    BrainBuilder()
    .add_component("custom", "my_region", n_input=128, n_output=64)
    .add_component("cortex", "layered_cortex", n_input=128, n_output=256)
    .connect("custom", "cortex", pathway_type="spiking_stdp")
    .build()
)

# 4. ADVANCED: Arbitrary topology
brain = (
    BrainBuilder()
    .add_component("v1", "visual_cortex", n_input=784, n_output=128)
    .add_component("v2", "visual_cortex", n_input=128, n_output=256)
    .add_component("v4", "visual_cortex", n_input=256, n_output=512)
    .add_component("it", "inferotemporal_cortex", n_input=512, n_output=256)
    .add_component("pfc", "prefrontal", n_input=256, n_output=128)
    # Hierarchical visual pathway
    .connect("v1", "v2", pathway_type="feedforward")
    .connect("v2", "v4", pathway_type="feedforward")
    .connect("v4", "it", pathway_type="feedforward")
    # Top-down feedback
    .connect("pfc", "it", pathway_type="attention")
    .connect("pfc", "v4", pathway_type="attention")
    .build()
)
```

**Key Benefits:**
- ✅ **Extensible**: Users can register custom components via decorator
- ✅ **Flexible topology**: Any directed graph structure
- ✅ **Type-safe**: Component registry validates types
- ✅ **Discoverable**: `ComponentRegistry.list_components("region")` shows all options
- ✅ **Version-able**: Component metadata includes version, author, description
- ✅ **Testable**: Each component tested independently
- ✅ **Checkpoint-compatible**: Dynamic graphs can be serialized
- ✅ **Clear migration**: Old configs translate to new builder syntax

**Implementation Sketch:**

```python
# thalia/core/brain_builder.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import torch.nn as nn

from thalia.managers.component_registry import ComponentRegistry
from thalia.regions.base import NeuralComponent
from thalia.core.base.component_config import NeuralComponentConfig


@dataclass
class ComponentSpec:
    """Specification for a brain component."""
    name: str              # Instance name (e.g., "cortex", "my_visual_cortex")
    component_type: str    # Registry type (e.g., "region", "pathway")
    registry_name: str     # Registry key (e.g., "layered_cortex", "spiking_stdp")
    config_params: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[NeuralComponent] = None


@dataclass
class ConnectionSpec:
    """Specification for a connection between components."""
    source: str            # Source component name
    target: str            # Target component name
    pathway_type: str      # Pathway registry name
    config_params: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[NeuralComponent] = None


class DynamicBrain(nn.Module):
    """Dynamic brain constructed from component graph.

    Unlike EventDrivenBrain (hardcoded 6 regions), DynamicBrain
    builds arbitrary topologies from registered components.

    Architecture:
        - components: Dict[name -> NeuralComponent]
        - connections: Dict[(source, target) -> Pathway]
        - topology: Directed graph of component dependencies

    Supports:
        - Custom user regions/pathways (via ComponentRegistry)
        - Arbitrary connectivity patterns
        - Dynamic component addition/removal
        - Checkpoint save/load of arbitrary graphs
    """

    def __init__(
        self,
        components: Dict[str, NeuralComponent],
        connections: Dict[Tuple[str, str], NeuralComponent],
        global_config: "GlobalConfig",
    ):
        super().__init__()

        self.global_config = global_config
        self.device = global_config.device

        # Store components as nn.ModuleDict for proper parameter tracking
        self.components = nn.ModuleDict(components)

        # Store connections (pathways)
        self.connections = nn.ModuleDict({
            f"{src}_to_{tgt}": pathway
            for (src, tgt), pathway in connections.items()
        })

        # Build topology graph for execution order
        self._topology = self._build_topology_graph()

        # Execution scheduler (for sequential or parallel execution)
        self._scheduler = None  # TODO: Integrate with EventScheduler

    def _build_topology_graph(self) -> Dict[str, List[str]]:
        """Build adjacency list of component dependencies."""
        graph = {name: [] for name in self.components.keys()}
        for (src, tgt) in self.connections.keys():
            if src in graph:
                graph[src].append(tgt)
        return graph

    def forward(self, input_data: Dict[str, torch.Tensor], n_timesteps: int) -> Dict[str, Any]:
        """Execute brain for n_timesteps.

        Args:
            input_data: Dict mapping component names to input tensors
            n_timesteps: Number of simulation timesteps

        Returns:
            Dict of outputs from all components
        """
        outputs = {}

        for t in range(n_timesteps):
            # Process components in topological order
            for component_name in self._topological_order():
                component = self.components[component_name]

                # Gather inputs for this component
                inputs = self._gather_component_inputs(
                    component_name,
                    input_data.get(component_name),
                    outputs
                )

                # Execute component
                component_output = component.forward(inputs)
                outputs[component_name] = component_output

        return outputs

    def _topological_order(self) -> List[str]:
        """Return components in topological execution order.

        Uses cached order or computes from scratch.
        """
        # TODO: Implement topological sort of self._topology
        # For now, return simple order
        return list(self.components.keys())

    def _gather_component_inputs(
        self,
        component_name: str,
        external_input: Optional[torch.Tensor],
        prior_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Collect inputs for a component from connections and external sources."""
        inputs = []

        # Check for external input (e.g., sensory input to thalamus)
        if external_input is not None:
            inputs.append(external_input)

        # Gather inputs from upstream connections
        for (src, tgt), pathway in self.connections.items():
            if tgt == component_name and src in prior_outputs:
                # Route through pathway
                pathway_output = pathway.forward(prior_outputs[src])
                inputs.append(pathway_output)

        # Combine inputs (concatenate or sum depending on component)
        if not inputs:
            # No inputs - component is source (e.g., oscillator)
            return None
        elif len(inputs) == 1:
            return inputs[0]
        else:
            # Multiple inputs - concatenate or sum
            # TODO: Make this configurable per component
            return torch.cat(inputs, dim=-1)

    def get_component(self, name: str) -> NeuralComponent:
        """Get component by name."""
        return self.components[name]

    def add_component(
        self,
        name: str,
        component: NeuralComponent
    ) -> None:
        """Dynamically add a component (e.g., during growth)."""
        self.components[name] = component
        self._topology[name] = []

    def add_connection(
        self,
        source: str,
        target: str,
        pathway: NeuralComponent
    ) -> None:
        """Dynamically add a connection."""
        self.connections[f"{source}_to_{target}"] = pathway
        if source in self._topology:
            self._topology[source].append(target)


class BrainBuilder:
    """Fluent builder for creating DynamicBrain instances.

    Provides both high-level presets and low-level component assembly.
    """

    def __init__(self):
        from thalia.config import GlobalConfig

        self._global_config = GlobalConfig()
        self._components: Dict[str, ComponentSpec] = {}
        self._connections: List[ConnectionSpec] = []
        self._preset_name: Optional[str] = None

    @classmethod
    def from_preset(cls, preset_name: str) -> "BrainBuilder":
        """Load preset architecture.

        Args:
            preset_name: One of "sensorimotor", "language", "multimodal", "minimal"
        """
        builder = cls()
        builder._preset_name = preset_name
        builder._apply_preset(preset_name)
        return builder

    @classmethod
    def minimal(cls, input_size: int = 128, n_actions: int = 2) -> "BrainBuilder":
        """Create minimal brain for testing."""
        return cls().from_preset("minimal").with_modifications(
            input_size=input_size,
            n_actions=n_actions
        )

    def with_device(self, device: str) -> "BrainBuilder":
        """Set compute device."""
        self._global_config.device = device
        return self

    def with_dt_ms(self, dt_ms: float) -> "BrainBuilder":
        """Set simulation timestep."""
        self._global_config.dt_ms = dt_ms
        return self

    def add_component(
        self,
        name: str,
        registry_name: str,
        component_type: str = "region",
        **config_params
    ) -> "BrainBuilder":
        """Add a component to the brain.

        Args:
            name: Instance name (e.g., "cortex", "my_visual_area")
            registry_name: Name in ComponentRegistry (e.g., "layered_cortex")
            component_type: Type in registry ("region", "pathway", "module")
            **config_params: Configuration parameters for the component

        Example:
            builder.add_component(
                "v1",
                "layered_cortex",
                component_type="region",
                n_input=784,
                n_output=128,
                l4_ratio=0.5,
            )
        """
        # Validate component exists in registry
        if not ComponentRegistry.is_registered(component_type, registry_name):
            available = ComponentRegistry.list_components(component_type)
            raise ValueError(
                f"{component_type} '{registry_name}' not registered. "
                f"Available: {available}"
            )

        # Inject global config
        config_params.setdefault("device", self._global_config.device)
        config_params.setdefault("dt_ms", self._global_config.dt_ms)

        self._components[name] = ComponentSpec(
            name=name,
            component_type=component_type,
            registry_name=registry_name,
            config_params=config_params,
        )
        return self

    def connect(
        self,
        source: str,
        target: str,
        pathway_type: str = "spiking_stdp",
        **config_params
    ) -> "BrainBuilder":
        """Create connection between components.

        Args:
            source: Source component name
            target: Target component name
            pathway_type: Pathway type in registry
            **config_params: Pathway configuration

        Example:
            builder.connect(
                "cortex",
                "hippocampus",
                pathway_type="spiking_stdp",
                stdp_lr=0.01,
                delay_ms=3.0,
            )
        """
        # Validate components exist
        if source not in self._components:
            raise ValueError(f"Source component '{source}' not added yet")
        if target not in self._components:
            raise ValueError(f"Target component '{target}' not added yet")

        # Inject global config
        config_params.setdefault("device", self._global_config.device)
        config_params.setdefault("dt_ms", self._global_config.dt_ms)

        # Infer sizes from components (if not specified)
        if "n_input" not in config_params:
            config_params["n_input"] = self._components[source].config_params.get("n_output", 128)
        if "n_output" not in config_params:
            config_params["n_output"] = self._components[target].config_params.get("n_input", 64)

        self._connections.append(ConnectionSpec(
            source=source,
            target=target,
            pathway_type=pathway_type,
            config_params=config_params,
        ))
        return self

    def with_modifications(self, **kwargs) -> "BrainBuilder":
        """Modify existing preset or component configurations."""
        for key, value in kwargs.items():
            if key in ["device", "dt_ms", "theta_frequency_hz"]:
                # Global config modification
                setattr(self._global_config, key, value)
            else:
                # Try to apply to components
                # (This would need smarter routing logic)
                pass
        return self

    def build(self) -> DynamicBrain:
        """Build the configured brain.

        Returns:
            DynamicBrain instance with all components and connections
        """
        # 1. Instantiate all components
        components: Dict[str, NeuralComponent] = {}
        for name, spec in self._components.items():
            # Create config object
            # TODO: Get proper config class from registry metadata
            config_class = self._get_config_class(spec.component_type, spec.registry_name)
            config = config_class(**spec.config_params)

            # Create component instance
            component = ComponentRegistry.create(
                spec.component_type,
                spec.registry_name,
                config
            )
            components[name] = component

        # 2. Instantiate all connections
        connections: Dict[Tuple[str, str], NeuralComponent] = {}
        for conn_spec in self._connections:
            # Create pathway config
            from thalia.core.base.component_config import PathwayConfig
            pathway_config = PathwayConfig(**conn_spec.config_params)

            # Create pathway instance
            pathway = ComponentRegistry.create(
                "pathway",
                conn_spec.pathway_type,
                pathway_config
            )
            connections[(conn_spec.source, conn_spec.target)] = pathway

        # 3. Construct brain
        brain = DynamicBrain(
            components=components,
            connections=connections,
            global_config=self._global_config,
        )

        return brain

    def _apply_preset(self, preset_name: str) -> None:
        """Apply a named preset configuration."""
        if preset_name == "minimal":
            self.add_component("thalamus", "thalamic_relay", n_input=64, n_output=64)
            self.add_component("cortex", "layered_cortex", n_input=64, n_output=32)
            self.add_component("striatum", "bg_circuit", n_input=32, n_output=2)
            self.connect("thalamus", "cortex")
            self.connect("cortex", "striatum")

        elif preset_name == "sensorimotor":
            self.add_component("thalamus", "thalamic_relay", n_input=128, n_output=128)
            self.add_component("cortex", "layered_cortex", n_input=128, n_output=128)
            self.add_component("hippocampus", "trisynaptic", n_input=128, n_output=64)
            self.add_component("pfc", "prefrontal", n_input=192, n_output=32)
            self.add_component("striatum", "bg_circuit", n_input=224, n_output=7)
            self.connect("thalamus", "cortex", delay_ms=5.0)
            self.connect("cortex", "hippocampus", delay_ms=3.0)
            self.connect("cortex", "pfc", delay_ms=4.0)
            self.connect("hippocampus", "pfc", delay_ms=5.0)
            self.connect("cortex", "striatum", delay_ms=5.0)
            self.connect("hippocampus", "striatum", delay_ms=6.0)
            self.connect("pfc", "striatum", delay_ms=4.0)

        elif preset_name == "language":
            # TODO: Implement language preset
            pass

        elif preset_name == "multimodal":
            # TODO: Implement multimodal preset
            pass

        else:
            raise ValueError(f"Unknown preset: {preset_name}")

    def _get_config_class(self, component_type: str, registry_name: str):
        """Get configuration class for a component.

        TODO: This should be in registry metadata
        """
        # Temporary mapping - should come from registry
        if component_type == "region":
            if registry_name == "layered_cortex":
                from thalia.regions.cortex import LayeredCortexConfig
                return LayeredCortexConfig
            elif registry_name == "trisynaptic":
                from thalia.regions.hippocampus import HippocampusConfig
                return HippocampusConfig
            # ... etc
        elif component_type == "pathway":
            from thalia.core.base.component_config import PathwayConfig
            return PathwayConfig

        # Fallback
        return NeuralComponentConfig
```

**Benefits Over Current System:**

1. **Truly Extensible**: Users can add components without modifying Thalia core
2. **Flexible Topology**: Not limited to fixed 6-region architecture
3. **Plugin System**: External packages can register components
4. **Better Testing**: Each component tested independently
5. **Checkpoint Serialization**: Save/load arbitrary graphs
6. **Clear Semantics**: "Add component" and "connect" vs. nested config hierarchies

---

### Recommendation 2: Enhanced ComponentRegistry Metadata

**Priority:** High
**Effort:** Low
**Impact:** High

**Current Gap:** `ComponentRegistry` stores component classes but not their config classes.

**Proposal:** Add config class metadata to registry.

**Proposal:** Add config class metadata to registry.

```python
# Enhanced registration
@ComponentRegistry.register(
    "layered_cortex",
    "region",
    config_class=LayeredCortexConfig,  # NEW
    description="Multi-layer cortical microcircuit",
    version="2.0",
    author="Thalia Team",
)
class LayeredCortex(NeuralComponent):
    ...

# Now builder can automatically infer config class
config_class = ComponentRegistry.get_config_class("region", "layered_cortex")
config = config_class(**user_params)
component = ComponentRegistry.create("region", "layered_cortex", config)
```

**Benefits:**
- ✅ Builder doesn't need hardcoded config class mapping
- ✅ Registry becomes source of truth
- ✅ Better error messages (show available params from config class)
- ✅ Enables config validation before instantiation

---

### Recommendation 3: Migration Strategy from EventDrivenBrain → DynamicBrain

**Priority:** Medium
**Effort:** Medium
**Impact:** Medium (short-term), High (long-term)

**Proposal:** Gradual migration with compatibility layer.

**Phase 1: Compatibility (Week 1-2)**
```python
# Old code continues to work
from thalia.core.brain import EventDrivenBrain
brain = EventDrivenBrain.from_thalia_config(config)  # Still works

# New code can use builder
from thalia.core.brain_builder import BrainBuilder
brain = BrainBuilder.from_preset("sensorimotor").build()  # Returns DynamicBrain
```

**Phase 2: Unified API (Week 3-4)**
```python
# EventDrivenBrain becomes a preset of DynamicBrain
class EventDrivenBrain(DynamicBrain):
    """Backwards-compatible brain with fixed 6-region architecture."""

    @classmethod
    def from_thalia_config(cls, config: ThaliaConfig) -> "EventDrivenBrain":
        # Translate old config to builder
        builder = BrainBuilder()
        builder._translate_thalia_config(config)
        brain = builder.build()  # Returns DynamicBrain
        return brain  # Wrap in EventDrivenBrain for compatibility
```

**Phase 3: Full Migration (Week 5+)**
- Update all tests to use `BrainBuilder`
- Update training scripts
- Update notebooks
- Deprecate `ThaliaConfig` (keep for reference)
- EventDrivenBrain becomes an alias

---

### Recommendation 4: User Plugin Guide & Examples

**Priority:** High
**Effort:** Low
**Impact:** High

**Proposal:** Create comprehensive plugin development guide.

```markdown
# docs/guides/CUSTOM_COMPONENTS.md

## Creating Custom Brain Regions

### 1. Define Your Region Class

```python
# my_project/regions/my_region.py

from thalia.regions.base import NeuralComponent
from thalia.core.base.component_config import NeuralComponentConfig
from thalia.managers.component_registry import register_region
import torch

@register_region(
    "my_custom_region",
    description="My amazing custom brain region",
    version="1.0",
    author="Your Name"
)
class MyCustomRegion(NeuralComponent):
    """Custom brain region with specialized processing."""

    def __init__(self, config: NeuralComponentConfig):
        super().__init__(config)

        # Initialize your custom components
        self.neurons = ...
        self.weights = ...
        self.learning_strategy = ...

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Process one timestep.

        Args:
            input_spikes: [n_input] binary spike tensor

        Returns:
            output_spikes: [n_output] binary spike tensor
        """
        # Your custom processing logic
        # MUST return binary spikes (not firing rates!)
        return output_spikes

    def reset_state(self) -> None:
        """Reset dynamic state between trials."""
        self.membrane.zero_()
        # Reset other state variables

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics."""
        return {
            "spikes": self.spikes.sum().item(),
            "firing_rate": self.spikes.float().mean().item(),
            # Add your custom metrics
        }
```

### 2. Use in Brain

```python
from thalia.core.brain_builder import BrainBuilder
from my_project.regions.my_region import MyCustomRegion  # Registers automatically

brain = (
    BrainBuilder()
    .add_component("input", "thalamic_relay", n_input=784, n_output=256)
    .add_component("custom", "my_custom_region", n_input=256, n_output=128)
    .add_component("output", "layered_cortex", n_input=128, n_output=10)
    .connect("input", "custom", pathway_type="spiking_stdp")
    .connect("custom", "output", pathway_type="spiking_stdp")
    .build()
)
```

### 3. Test Your Region

```python
# tests/test_my_region.py

def test_my_region_forward():
    config = NeuralComponentConfig(n_input=100, n_output=50)
    region = MyCustomRegion(config)

    # Test with sparse input
    input_spikes = torch.zeros(100, dtype=torch.bool)
    input_spikes[:10] = True

    output = region.forward(input_spikes)

    assert output.shape == (50,)
    assert output.dtype == torch.bool
    assert output.sum() > 0  # Should produce some spikes

def test_my_region_learning():
    region = MyCustomRegion(config)

    # Run multiple timesteps
    for t in range(100):
        output = region.forward(input_spikes)

    # Check that weights changed
    assert not torch.allclose(region.weights, initial_weights)
```

### 4. Share Your Component

```python
# setup.py (for your plugin package)

setup(
    name="thalia-my-regions",
    version="1.0.0",
    packages=["my_project"],
    install_requires=["thalia>=2.0.0"],
    entry_points={
        "thalia.regions": [
            "my_custom_region = my_project.regions.my_region:MyCustomRegion",
        ]
    }
)
```

Now others can use: `pip install thalia-my-regions`

## Creating Custom Pathways

Same process, just use `@register_pathway` instead of `@register_region`:

```python
@register_pathway(
    "my_custom_pathway",
    description="Custom connectivity pattern",
    version="1.0",
    author="Your Name"
)
class MyCustomPathway(NeuralComponent):
    # Same interface as regions!
    ...
```

## Creating Custom Learning Rules

```python
from thalia.learning.rules.strategies import LearningStrategy, register_strategy

@register_strategy(
    "my_learning_rule",
    description="My novel plasticity rule",
    version="1.0"
)
class MyLearningStrategy(LearningStrategy):
    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        # Your learning logic (must be LOCAL!)
        delta_w = ...
        return weights + delta_w
```

---

```python
# New: Configuration presets module
from thalia.config.presets import (
    sensorimotor_config,
    language_config,
    multimodal_config,
    minimal_test_config,
)

# Quick start with presets
config = sensorimotor_config(device="cuda")
brain = EventDrivenBrain.from_thalia_config(config)

# Or modify preset
config = language_config(device="cuda")
config.brain.sizes.cortex_size = 512  # Customize
brain = EventDrivenBrain.from_thalia_config(config)
```

**Implementation:**

```python
# thalia/config/presets.py
def sensorimotor_config(device: str = "cpu", **overrides) -> ThaliaConfig:
    """Preset for sensorimotor learning (curriculum stage 0)."""
    config = ThaliaConfig(
        global_=GlobalConfig(device=device, dt_ms=1.0),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=128,
                cortex_size=128,
                hippocampus_size=64,
                pfc_size=32,
                n_actions=7,
            ),
            encoding_timesteps=10,
            delay_timesteps=5,
            test_timesteps=10,
        ),
    )

    # Apply overrides
    for key, value in overrides.items():
        # Smart routing logic
        ...

    return config

def minimal_test_config(device: str = "cpu") -> ThaliaConfig:
    """Minimal brain for unit tests."""
    return ThaliaConfig(
        global_=GlobalConfig(device=device),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=10,
                cortex_size=20,
                hippocampus_size=15,
                pfc_size=10,
                n_actions=3,
            ),
        ),
    )
```

**Benefits:**
- ✅ Faster onboarding
- ✅ Consistency across experiments
- ✅ Best practices encoded
- ✅ Reduces test boilerplate

---

## Migration Path (v2 - More Ambitious)

### Phase 1: Core Infrastructure (Weeks 1-2)
1. ✅ Implement `DynamicBrain` class (component graph executor)
2. ✅ Implement `BrainBuilder` with `.add_component()` and `.connect()`
3. ✅ Enhance `ComponentRegistry` with config class metadata
4. ✅ Create preset architectures ("sensorimotor", "language", etc.)
5. ✅ Add comprehensive unit tests

**Deliverable:** `BrainBuilder.from_preset("sensorimotor").build()` works

---

### Phase 2: User Plugin Support (Weeks 3-4)
6. ✅ Create plugin development guide (`docs/guides/CUSTOM_COMPONENTS.md`)
7. ✅ Add example custom region/pathway implementations
8. ✅ Create plugin template repository
9. ✅ Add plugin discovery mechanism (entry points)
10. ✅ Test with external plugin package

**Deliverable:** Users can `pip install` custom components

---

### Phase 3: Migration & Compatibility (Weeks 5-6)
11. ✅ Create `ThaliaConfig` → `BrainBuilder` translator
12. ✅ Update `EventDrivenBrain.from_thalia_config()` to use builder internally
13. ✅ Add deprecation warnings to old patterns
14. ✅ Update training scripts to use builder
15. ✅ Update notebooks with builder examples

**Deliverable:** Old code still works, new code uses builder

---

### Phase 4: Test & Documentation (Weeks 7-8)
16. ✅ Migrate all tests to use builder or registry
17. ✅ Update architecture documentation
18. ✅ Create video tutorials
19. ✅ Write migration guide
20. ✅ Update API reference

**Deliverable:** Complete documentation and examples

---

### Phase 5: Cleanup (Week 9+)
21. ✅ Remove deprecated `ThaliaConfig` paths
22. ✅ Simplify `EventDrivenBrain` (becomes preset)
23. ✅ Archive old config system documentation
24. ✅ Performance optimization
25. ✅ Community feedback integration

**Deliverable:** Clean, modern codebase

---

## Breaking Changes & Migration

### What Breaks

1. **Direct `EventDrivenBrain(config)` construction**
   - Old: `brain = EventDrivenBrain(config)`
   - New: `brain = BrainBuilder.from_preset("sensorimotor").build()`

2. **Hardcoded region access patterns**
   - Old: `brain.cortex.impl.forward(x)`
   - New: `brain.get_component("cortex").forward(x)`

3. **ThaliaConfig structure**
   - Old: Nested dataclasses
   - New: Component graph specification

### Migration Tools

**Automatic Translator:**
```python
from thalia.migration import translate_config

# Translate old config to builder
old_config = ThaliaConfig(...)
builder = translate_config(old_config)
brain = builder.build()
```

**Compatibility Wrapper:**
```python
# Old code continues working via wrapper
class EventDrivenBrain(DynamicBrain):
    @classmethod
    def from_thalia_config(cls, config):
        builder = translate_config(config)
        return builder.build()
```

---

## Comparison: Current vs. Proposed (v2)

### 1. Creating a Simple Brain

**Current (17 lines):**
```python
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

config = ThaliaConfig(
    global_=GlobalConfig(device="cuda"),
    brain=BrainConfig(
        sizes=RegionSizes(
            input_size=128,
            cortex_size=128,
            hippocampus_size=64,
            pfc_size=32,
            n_actions=7,
        ),
    ),
)
brain = EventDrivenBrain.from_thalia_config(config)
```

**Proposed (3 lines):**
```python
from thalia.core.brain_builder import BrainBuilder

brain = BrainBuilder.from_preset("sensorimotor").with_device("cuda").build()
```

**Savings:** 82% fewer lines, same functionality.

---

### 2. Creating a Custom Brain (Standard Topology)

**Current (Impossible without modifying Thalia core):**
```python
# Cannot add custom regions without modifying EventDrivenBrain.__init__()
# Must fork Thalia and add to hardcoded region list
```

**Proposed (13 lines):**
```python
from thalia.core.brain_builder import BrainBuilder
from my_package import MyCustomRegion  # User's plugin

brain = (
    BrainBuilder()
    .with_device("cuda")
    .add_component("thalamus", "thalamic_relay", n_input=128, n_output=128)
    .add_component("my_region", "my_custom_region", n_input=128, n_output=256)
    .add_component("cortex", "layered_cortex", n_input=256, n_output=128)
    .add_component("striatum", "bg_circuit", n_input=128, n_output=7)
    .connect("thalamus", "my_region", pathway_type="spiking_stdp")
    .connect("my_region", "cortex", pathway_type="spiking_stdp")
    .connect("cortex", "striatum", pathway_type="spiking_stdp")
    .build()
)
```

**Impact:** **User plugins now possible** without modifying Thalia.

---

### 3. Creating Non-Standard Topology

**Current (Impossible):**
```python
# EventDrivenBrain hardcodes this topology:
#   Thalamus → Cortex → Hippocampus
#                    ↘ PFC
#                    ↘ Striatum → Cerebellum
#   Hippocampus → PFC → Striatum
#
# Cannot create:
#   - Hierarchical visual cortex (V1→V2→V4→IT)
#   - Multiple cortical columns
#   - Arbitrary feedback loops
#   - Modular sub-networks
```

**Proposed (Hierarchical visual system, 13 lines):**
```python
brain = (
    BrainBuilder()
    .add_component("v1", "visual_cortex", n_input=784, n_output=128)
    .add_component("v2", "visual_cortex", n_input=128, n_output=256)
    .add_component("v4", "visual_cortex", n_input=256, n_output=512)
    .add_component("it", "inferotemporal_cortex", n_input=512, n_output=256)
    .add_component("pfc", "prefrontal", n_input=256, n_output=128)
    # Feedforward
    .connect("v1", "v2").connect("v2", "v4").connect("v4", "it")
    # Feedback
    .connect("pfc", "it", pathway_type="attention")
    .connect("pfc", "v4", pathway_type="attention")
    .build()
)
```

**Impact:** **Arbitrary topologies** now possible.

---

### 4. Testing Individual Components

**Current:**
```python
# Must create entire EventDrivenBrain to test one region
# Heavy dependencies, slow tests
config = ThaliaConfig(...)
brain = EventDrivenBrain.from_thalia_config(config)
hippocampus = brain.hippocampus  # Access via brain

# Or directly instantiate (bypasses brain initialization)
hippocampus = Hippocampus(config)  # But config is complicated
```

**Proposed:**
```python
# Direct component creation via registry
from thalia.managers.component_registry import ComponentRegistry

hippocampus = ComponentRegistry.create(
    "region",
    "trisynaptic",
    HippocampusConfig(n_input=128, n_output=64)
)

# Lightweight, fast, isolated testing
```

**Impact:** **Faster, cleaner unit tests**.

---

## Risks & Mitigation (v2)

### Risk 1: Larger Refactoring Surface
**Concern:** More code changes = higher risk of breakage
**Mitigation:**
- Phased rollout with compatibility layer
- Extensive testing at each phase
- Community testing period before full release
- Clear migration documentation
- **Accept:** Some breakage is OK for better long-term design

### Risk 2: Component Registry Incomplete
**Concern:** Not all regions registered yet
**Mitigation:**
- Audit existing regions (already ~6 registered)
- Add registration to remaining regions
- Update contribution guide
- **Current status:** Most core regions already use `@register_region`

### Risk 3: Learning Curve
**Concern:** New API to learn
**Mitigation:**
- Presets handle 80% of use cases (no learning needed)
- Clear documentation with examples
- Video tutorials
- Plugin template repository
- **Benefit:** Simpler API overall (add_component vs. nested configs)

### Risk 4: Performance Impact
**Concern:** Dynamic dispatch might be slower
**Mitigation:**
- Profile and optimize hot paths
- JIT compilation where beneficial
- Parallel execution still supported
- **Likely:** Negligible impact (dispatching not the bottleneck)

### Risk 5: Checkpoint Compatibility
**Concern:** Old checkpoints might not load
**Mitigation:**
- Write checkpoint migration tool
- Keep old checkpoint loader available
- Document checkpoint version compatibility
- Add version metadata to checkpoints

---

## Conclusion (v2)

**Recommendation:** Implement **Component-Based Brain Builder** (Recommendation 1) + **Enhanced Registry** (Recommendation 2) + **Plugin Guide** (Recommendation 4).

**This is a more ambitious refactoring that will break some existing code, but provides:**

### Transformative Benefits

1. **✅ User Extensibility**: Custom regions/pathways without forking Thalia
2. **✅ Plugin Ecosystem**: `pip install thalia-visual-cortex`, `pip install thalia-language-networks`
3. **✅ Flexible Topologies**: Arbitrary brain architectures, not locked to 6 regions
4. **✅ Better Testing**: Independent component testing, faster CI
5. **✅ Cleaner Code**: 50-80% less boilerplate for common cases
6. **✅ Research-Friendly**: Easy experimentation with novel architectures
7. **✅ Production-Ready**: Clear upgrade path, comprehensive docs

### Trade-Offs

- ❌ **Breaking changes** to brain initialization (accept this)
- ❌ **2-3 month migration** for full ecosystem update
- ❌ **Learning curve** for new API (mitigated by presets)

### Success Metrics

- ✅ 3+ external plugins created by community within 6 months
- ✅ 80% of users use presets (don't need to know details)
- ✅ Test suite runs 30%+ faster (component isolation)
- ✅ Zero issues reported for custom component integration

**Recommendation:** Proceed with phased rollout. Accept short-term migration pain for long-term architectural flexibility.

**Next Steps:**
1. ✅ Get team consensus on breaking changes
2. ✅ Create detailed implementation plan
3. ✅ Start Phase 1 (core infrastructure)
4. ✅ Weekly progress updates
5. ✅ Community preview at Phase 3

---

## Risks & Mitigation

### Risk 1: Builder Complexity
**Concern:** Builder adds another API to learn.
**Mitigation:**
- Builder is optional (sugar over existing API)
- Clear documentation on when to use what
- Presets handle most common cases

### Risk 2: Maintenance Burden
**Concern:** Two APIs to maintain.
**Mitigation:**
- Builder is thin wrapper (10% new code)
- Builder calls existing validation logic
- Single source of truth (`ThaliaConfig`)

### Risk 3: Configuration Drift
**Concern:** Builder and `ThaliaConfig` get out of sync.
**Mitigation:**
- Builder exports to `ThaliaConfig` (single format)
- Validation happens at build time
- Tests ensure equivalence

---

## Conclusion

**Recommendation:** Implement **Builder Pattern** (Recommendation 1) + **Presets** (Recommendation 2).

**Benefits:**
- 50-80% reduction in boilerplate for common cases
- Progressive assembly for complex configurations
- Backwards compatible (zero breaking changes)
- Better onboarding experience
- Enables configuration sharing/templates

**Effort:** ~1-2 weeks implementation + testing + documentation

**Next Steps:**
1. Review this analysis with team
2. Get approval for builder pattern
3. Create implementation branch
4. Implement `BrainBuilder` class
5. Add tests and documentation
6. Gradual adoption in examples/notebooks
