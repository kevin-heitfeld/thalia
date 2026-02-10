# ADR-015: Population-Based Routing System

**Status**: Accepted
**Date**: 2026-01-25
**Updated**: 2026-02-10 (Terminology: port → population)
**Deciders**: Architecture Review (Tier 3.2)
**Related**: Architecture Review 2026-01-25 (Tier 3.2)

## Context

LayeredCortex has multiple output populations (L2/3, L5, L6a, L6b) with distinct biological functions and projection targets:
- **L2/3**: Cortico-cortical projections (inter-area communication)
- **L5**: Subcortical projections (motor output, brainstem)
- **L6a**: Projects to TRN (thalamic reticular nucleus, gating)
- **L6b**: Projects to thalamic relay neurons (feedback modulation)

Currently, `forward()` returns `torch.cat([l23_spikes, l5_spikes])` as a single output tensor, making it impossible to route L6a→TRN and L6b→relay separately. This concatenation + slicing approach is:
- Biologically inaccurate (different cell types don't merge outputs)
- Difficult to maintain (magic indices for slicing)
- Inflexible (can't add new output pathways without breaking existing code)

### Biological Motivation

In the brain, different neuron types within a region project to different targets:
- L6a corticothalamic (CT) neurons → TRN (lateral inhibition, attentional gating)
- L6b CT neurons → Thalamic relay cells (feedback gain modulation)
- L5 pyramidal neurons → Subcortical structures (motor output)
- L2/3 pyramidal neurons → Other cortical areas (information processing)

Thalia should reflect this biological organization by allowing explicit routing from specific layer outputs to specific target inputs.

## Decision

Implement a **population-based routing system** in the NeuralRegion base class that allows regions to:
1. **Register named output populations** (e.g., "l6a", "l6b", "default")
2. **Set population-specific outputs** during forward pass
3. **Route connections** from specific source populations to target regions/populations

**Note**: Terminology update (2026-02-10): "Port" was engineering terminology; "population" is biologically accurate, referring to distinct neuron populations within a region (cortical layers, hippocampal subregions, striatal pathways, etc.).

### API Design

#### 1. NeuralRegion Base Class Extensions

```python
class NeuralRegion(nn.Module):
    """Base class with population-based routing support."""

    def __init__(self, n_neurons: int, device: str):
        super().__init__()
        self.n_neurons = n_neurons
        self.device = device

        # Population infrastructure
        self._population_outputs: Dict[str, torch.Tensor] = {}
        self._population_sizes: Dict[str, int] = {}
        self._registered_populations: Set[str] = set()

    def register_output_population(self, population_name: str, size: int) -> None:
        """Register an output population for routing.

        Args:
            population_name: Name of the population (e.g., "l6a", "default")
            size: Number of neurons in this output

        Raises:
            ValueError: If population already registered
        """
        if population_name in self._registered_populations:
            raise ValueError(f"Population '{population_name}' already registered")

        self._population_sizes[population_name] = size
        self._registered_populations.add(population_name)

    def set_population_output(self, population_name: str, spikes: torch.Tensor) -> None:
        """Store output for a specific population.

        Args:
            population_name: Name of the population
            spikes: Spike tensor for this population

        Raises:
            ValueError: If population not registered
        """
        if population_name not in self._registered_populations:
            raise ValueError(f"Population '{population_name}' not registered. "
                           f"Available populations: {list(self._registered_populations)}")

        if spikes.shape[0] != self._population_sizes[population_name]:
            raise ValueError(f"Population '{population_name}' expects {self._population_sizes[population_name]} "
                           f"neurons, got {spikes.shape[0]}")

        self._population_outputs[population_name] = spikes

    def get_population_output(self, population_name: Optional[str] = None) -> torch.Tensor:
        """Get output from a specific population.

        Args:
            population_name: Name of the population. If None, returns "default" population.

        Returns:
            Spike tensor from the specified population

        Raises:
            ValueError: If population not found or no output set
        """
        if population_name is None:
            population_name = "default"

        if population_name not in self._population_outputs:
            raise ValueError(f"No output set for population '{population_name}'. "
                           f"Available outputs: {list(self._population_outputs.keys())}")

        return self._population_outputs[population_name]

    def clear_population_outputs(self) -> None:
        """Clear all population outputs (called at start of forward pass)."""
        self._population_outputs.clear()

    def get_registered_populations(self) -> List[str]:
        """Get list of all registered population names."""
        return sorted(self._registered_populations)
```

#### 2. LayeredCortex Implementation

```python
class LayeredCortex(NeuralRegion):
    def __init__(self, config: CortexConfig, sizes: Dict[str, int], device: str):
        # ... existing initialization ...

        # Register output populations (cortical layers)
        self.register_output_population("default", self.l23_size + self.l5_size)
        self.register_output_population("l23", self.l23_size)
        self.register_output_population("l5", self.l5_size)
        self.register_output_population("l6a", self.l6a_size)
        self.register_output_population("l6b", self.l6b_size)

    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process input and set population outputs."""
        self.clear_population_outputs()  # Clear previous outputs

        # ... existing layer processing ...
        # l4_spikes, l23_spikes, l5_spikes, l6a_spikes, l6b_spikes = ...

        # Set population outputs
        self.set_population_output("l23", l23_spikes)
        self.set_population_output("l5", l5_spikes)
        self.set_population_output("l6a", l6a_spikes)
        self.set_population_output("l6b", l6b_spikes)
        self.set_population_output("default", torch.cat([l23_spikes, l5_spikes]))

        # Return default output (backward compatibility)
        return self.get_population_output("default")
```

#### 3. BrainBuilder Integration

```python
class BrainBuilder:
    def connect(
        self,
        source: str,
        target: str,
        source_population: Optional[str] = None,
        target_population: Optional[str] = None,
        delay_ms: float = 0.0,
    ) -> "BrainBuilder":
        """Connect two regions with optional population specification.

        Args:
            source: Source region name
            target: Target region name
            source_population: Output population of source (None = "default")
            target_population: Input population of target (for routing to specific dendrites)
            delay_ms: Axonal delay in milliseconds

        Example:
            >>> builder.connect("cortex", "thalamus", source_population="l6a", target_population="trn")
            >>> builder.connect("cortex", "thalamus", source_population="l6b", target_population="relay")
        """
        # Validate source region and population
        source_region = self.components.get(source)
        if source_region is None:
            raise ValueError(f"Source region '{source}' not found")

        if source_population is not None:
            if source_population not in source_region.get_registered_populations():
                raise ValueError(
                    f"Population '{source_population}' not found in '{source}'. "
                    f"Available: {source_region.get_registered_populations()}"
                )

        # Store connection with population info
        connection_key = (source, target)
        self.topology[source].append(target)
        self._connection_specs[connection_key] = {
            "source_population": source_population,
            "target_population": target_population,
            "delay_ms": delay_ms,
        }

        return self
```

#### 4. AxonalProjection Updates

```python
class AxonalProjection(RoutingComponent):
    def __init__(
        self,
        sources: List[Tuple[str, Optional[str]]],  # (region_name, population_name)
        device: str = "cpu",
        dt_ms: float = 1.0,
    ):
        """Initialize with source regions and optional populations.

        Args:
            sources: List of (region_name, population_name) tuples
            device: Device for tensors
            dt_ms: Timestep in milliseconds
        """
        self.sources = sources  # [(region_name, population), ...]
        # ... rest of initialization ...

    def forward(self, source_outputs: Dict[str, NeuralRegion]) -> Dict[str, torch.Tensor]:
        """Route from source populations to target.

        Args:
            source_outputs: Dict mapping region names to region objects

        Returns:
            Dict mapping source names to routed spike tensors
        """
        routed = {}
        for source_name, population_name in self.sources:
            region = source_outputs[source_name]

            # Get population-specific output
            spikes = region.get_population_output(population_name)

            # Apply delay if configured
            if self.delays.get(source_name) is not None:
                spikes = self.delays[source_name].forward(spikes)

            routed[source_name] = spikes

        return routed
```

### Usage Example

```python
# Create brain with population-based routing
builder = BrainBuilder(config)

# Add regions
builder.add_region("cortex", cortex_config)
builder.add_region("thalamus", thalamus_config)

# Connect with specific populations
builder.connect(
    "cortex", "thalamus",
    source_population="l6a",      # L6a neurons → TRN
    target_population="trn",
    delay_ms=2.0
)
builder.connect(
    "cortex", "thalamus",
    source_population="l6b",      # L6b neurons → Relay
    target_population="relay",
    delay_ms=2.0
)
builder.connect(
    "cortex", "prefrontal",  # Default output → PFC
    delay_ms=5.0
)

brain = builder.build()
```

## Consequences

### Positive

1. **Biological Accuracy**: Matches reality where different neuron populations project to different targets
2. **Explicit Routing**: No magic indices or concatenation/slicing logic
3. **Extensible**: Easy to add new populations without breaking existing connections
4. **Type-Safe**: Population registration ensures outputs match expected dimensions
5. **Backward Compatible**: Default population preserves existing behavior
6. **Biologically Accurate Terminology**: "Population" correctly describes cortical layers, hippocampal subregions, striatal pathways, etc.

### Negative

1. **Complexity**: Adds new API surface to learn and maintain
2. **Breaking Changes**: Requires updates to connection logic throughout codebase
3. **Performance**: Additional dictionary lookups per forward pass (minimal impact)
4. **Migration Effort**: Existing BrainBuilder usage needs updates

### Risks and Mitigations

**Risk**: Breaking existing brain configurations
**Mitigation**: Default population behavior ensures backward compatibility for regions that don't use multiple populations

**Risk**: Confusion between population names and region names
**Mitigation**: Clear naming conventions and validation errors with helpful messages

**Risk**: Performance overhead
**Mitigation**: Benchmark shows <1% overhead vs. concatenation approach

## Implementation Plan

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Add population methods to NeuralRegion base class
- [x] Add unit tests for population registration and retrieval
- [x] Update type aliases (SourceSpec to include population)

**Commit**: c21ca3c (2026-01-25)

### Phase 2: LayeredCortex Integration ✅ COMPLETE
- [x] Register L6a, L6b, L23, L5, default populations
- [x] Update forward() to set population outputs
- [x] Add integration tests for cortex populations

**Commit**: c21ca3c (2026-01-25)

### Phase 3: Routing Updates ✅ COMPLETE
- [x] Update AxonalProjection for population-aware routing
- [x] Update BrainBuilder.connect() with population parameters (already supported)
- [x] Update DynamicBrain execution to use populations

**Commit**: 45f27dc (2026-01-25)

### Phase 4: Integration Testing & Region Updates ✅ COMPLETE
- [x] End-to-end tests with L6a→TRN, L6b→relay routing
- [x] Add populations to ALL NeuralRegion subclasses (7 total):
  - [x] ThalamicRelay (relay, TRN populations)
  - [x] TrisynapticHippocampus (CA1, CA3, DG, CA2 populations)
  - [x] Striatum (D1, D2 pathway populations)
  - [x] Prefrontal (executive control population)
  - [x] Cerebellum (Purkinje cell population)
  - [x] MultimodalIntegration (integrated population)
  - [x] PredictiveCortex (L2/3+L5 representation populations)

**Commit**: edcad07 (2026-01-25)

### Phase 5: Documentation ✅ COMPLETE
- [x] Update copilot-instructions.md with population usage patterns
- [x] Update API documentation
- [x] Migration guide (backward compatibility via "default" population - no migration needed!)
- [x] Terminology update (2026-02-10): Renamed "port" → "population" for biological accuracy

**Note**: Phase 5 (Other Regions) was completed as part of Phase 4 to ensure full system integration.

## Alternatives Considered

### Alternative 1: Slice-Based Routing (Current)
**Approach**: Concatenate all outputs, slice at connection sites
**Rejected**: Fragile (magic indices), difficult to maintain, doesn't scale

### Alternative 2: Multiple forward() Methods
**Approach**: `forward_default()`, `forward_l6a()`, `forward_l6b()`
**Rejected**: Violates single responsibility, complicates execution loop

### Alternative 3: Separate Regions for Each Layer
**Approach**: L6aCortex, L6bCortex as independent regions
**Rejected**: Breaks biological coherence, massive code duplication

## References

- Architecture Review 2026-01-25, Tier 3.2
- ADR-011: Cortex Layer Consolidation (biological coherence)
- Neuroscience: Corticothalamic cell types and projections
  - L6a CT neurons (burst firing) → TRN
  - L6b CT neurons (regular spiking) → Relay neurons

## Notes

This is a **major architectural enhancement** that enables biologically-accurate routing patterns. The population-based system is:
- Extensible to other regions with multiple output types (layers, subregions, pathways)
- Compatible with future target-population routing (multi-input populations)
- Foundation for more complex routing patterns (e.g., context-dependent routing)
- **Biologically accurate terminology**: "Population" correctly describes distinct neuron groups within a region

**Terminology Rationale** (2026-02-10):
- "Port" was engineering terminology borrowed from hardware/software interfaces
- "Population" is the correct neuroscience term for distinct groups of neurons:
  - Cortex: Different **layers** are different populations (L2/3, L4, L5, L6)
  - Hippocampus: Different **subregions** are different populations (DG, CA3, CA2, CA1)
  - Striatum: Different **pathways** are different populations (D1 direct, D2 indirect)
  - Thalamus: Different **nuclei** are different populations (relay, TRN)

Future work: Target populations for multi-modal input regions (e.g., PFC with separate "cortical" and "limbic" input populations).
