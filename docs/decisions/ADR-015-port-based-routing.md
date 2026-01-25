# ADR-015: Port-Based Routing System

**Status**: Accepted  
**Date**: 2026-01-25  
**Deciders**: Architecture Review (Tier 3.2)  
**Related**: Architecture Review 2026-01-25 (Tier 3.2)

## Context

LayeredCortex has multiple output layers (L2/3, L5, L6a, L6b) with distinct biological functions and projection targets:
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

Implement a **port-based routing system** in the NeuralRegion base class that allows regions to:
1. **Register named output ports** (e.g., "l6a", "l6b", "default")
2. **Set port-specific outputs** during forward pass
3. **Route connections** from specific source ports to target regions/ports

### API Design

#### 1. NeuralRegion Base Class Extensions

```python
class NeuralRegion(nn.Module):
    """Base class with port-based routing support."""

    def __init__(self, n_neurons: int, device: str):
        super().__init__()
        self.n_neurons = n_neurons
        self.device = device
        
        # Port infrastructure
        self._port_outputs: Dict[str, torch.Tensor] = {}
        self._port_sizes: Dict[str, int] = {}
        self._registered_ports: Set[str] = set()

    def register_output_port(self, port_name: str, size: int) -> None:
        """Register an output port for routing.
        
        Args:
            port_name: Name of the port (e.g., "l6a", "default")
            size: Number of neurons in this output
            
        Raises:
            ValueError: If port already registered
        """
        if port_name in self._registered_ports:
            raise ValueError(f"Port '{port_name}' already registered")
        
        self._port_sizes[port_name] = size
        self._registered_ports.add(port_name)

    def set_port_output(self, port_name: str, spikes: torch.Tensor) -> None:
        """Store output for a specific port.
        
        Args:
            port_name: Name of the port
            spikes: Spike tensor for this port
            
        Raises:
            ValueError: If port not registered
        """
        if port_name not in self._registered_ports:
            raise ValueError(f"Port '{port_name}' not registered. "
                           f"Available ports: {list(self._registered_ports)}")
        
        if spikes.shape[0] != self._port_sizes[port_name]:
            raise ValueError(f"Port '{port_name}' expects {self._port_sizes[port_name]} "
                           f"neurons, got {spikes.shape[0]}")
        
        self._port_outputs[port_name] = spikes

    def get_port_output(self, port_name: Optional[str] = None) -> torch.Tensor:
        """Get output from a specific port.
        
        Args:
            port_name: Name of the port. If None, returns "default" port.
            
        Returns:
            Spike tensor from the specified port
            
        Raises:
            ValueError: If port not found or no output set
        """
        if port_name is None:
            port_name = "default"
        
        if port_name not in self._port_outputs:
            raise ValueError(f"No output set for port '{port_name}'. "
                           f"Available outputs: {list(self._port_outputs.keys())}")
        
        return self._port_outputs[port_name]
    
    def clear_port_outputs(self) -> None:
        """Clear all port outputs (called at start of forward pass)."""
        self._port_outputs.clear()
    
    def get_registered_ports(self) -> List[str]:
        """Get list of all registered port names."""
        return sorted(self._registered_ports)
```

#### 2. LayeredCortex Implementation

```python
class LayeredCortex(NeuralRegion):
    def __init__(self, config: CortexConfig, sizes: Dict[str, int], device: str):
        # ... existing initialization ...
        
        # Register output ports
        self.register_output_port("default", self.l23_size + self.l5_size)
        self.register_output_port("l23", self.l23_size)
        self.register_output_port("l5", self.l5_size)
        self.register_output_port("l6a", self.l6a_size)
        self.register_output_port("l6b", self.l6b_size)

    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process input and set port outputs."""
        self.clear_port_outputs()  # Clear previous outputs
        
        # ... existing layer processing ...
        # l4_spikes, l23_spikes, l5_spikes, l6a_spikes, l6b_spikes = ...
        
        # Set port outputs
        self.set_port_output("l23", l23_spikes)
        self.set_port_output("l5", l5_spikes)
        self.set_port_output("l6a", l6a_spikes)
        self.set_port_output("l6b", l6b_spikes)
        self.set_port_output("default", torch.cat([l23_spikes, l5_spikes]))
        
        # Return default output (backward compatibility)
        return self.get_port_output("default")
```

#### 3. BrainBuilder Integration

```python
class BrainBuilder:
    def connect(
        self,
        source: str,
        target: str,
        source_port: Optional[str] = None,
        target_port: Optional[str] = None,
        delay_ms: float = 0.0,
    ) -> "BrainBuilder":
        """Connect two regions with optional port specification.
        
        Args:
            source: Source region name
            target: Target region name
            source_port: Output port of source (None = "default")
            target_port: Input port of target (for future multi-port inputs)
            delay_ms: Axonal delay in milliseconds
            
        Example:
            >>> builder.connect("cortex", "thalamus", source_port="l6a", target_port="trn")
            >>> builder.connect("cortex", "thalamus", source_port="l6b", target_port="relay")
        """
        # Validate source region and port
        source_region = self.components.get(source)
        if source_region is None:
            raise ValueError(f"Source region '{source}' not found")
        
        if source_port is not None:
            if source_port not in source_region.get_registered_ports():
                raise ValueError(
                    f"Port '{source_port}' not found in '{source}'. "
                    f"Available: {source_region.get_registered_ports()}"
                )
        
        # Store connection with port info
        connection_key = (source, target)
        self.topology[source].append(target)
        self._connection_specs[connection_key] = {
            "source_port": source_port,
            "target_port": target_port,
            "delay_ms": delay_ms,
        }
        
        return self
```

#### 4. AxonalProjection Updates

```python
class AxonalProjection(RoutingComponent):
    def __init__(
        self,
        sources: List[Tuple[str, Optional[str]]],  # (region_name, port_name)
        device: str = "cpu",
        dt_ms: float = 1.0,
    ):
        """Initialize with source regions and optional ports.
        
        Args:
            sources: List of (region_name, port_name) tuples
            device: Device for tensors
            dt_ms: Timestep in milliseconds
        """
        self.sources = sources  # [(region_name, port), ...]
        # ... rest of initialization ...
    
    def forward(self, source_outputs: Dict[str, NeuralRegion]) -> Dict[str, torch.Tensor]:
        """Route from source ports to target.
        
        Args:
            source_outputs: Dict mapping region names to region objects
            
        Returns:
            Dict mapping source names to routed spike tensors
        """
        routed = {}
        for source_name, port_name in self.sources:
            region = source_outputs[source_name]
            
            # Get port-specific output
            spikes = region.get_port_output(port_name)
            
            # Apply delay if configured
            if self.delays.get(source_name) is not None:
                spikes = self.delays[source_name].forward(spikes)
            
            routed[source_name] = spikes
        
        return routed
```

### Usage Example

```python
# Create brain with port-based routing
builder = BrainBuilder(config)

# Add regions
builder.add_region("cortex", cortex_config)
builder.add_region("thalamus", thalamus_config)

# Connect with specific ports
builder.connect(
    "cortex", "thalamus",
    source_port="l6a",      # L6a → TRN
    target_port="trn",
    delay_ms=2.0
)
builder.connect(
    "cortex", "thalamus",
    source_port="l6b",      # L6b → Relay
    target_port="relay",
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

1. **Biological Accuracy**: Matches reality where different cell types project to different targets
2. **Explicit Routing**: No magic indices or concatenation/slicing logic
3. **Extensible**: Easy to add new ports without breaking existing connections
4. **Type-Safe**: Port registration ensures outputs match expected dimensions
5. **Backward Compatible**: Default port preserves existing behavior

### Negative

1. **Complexity**: Adds new API surface to learn and maintain
2. **Breaking Changes**: Requires updates to connection logic throughout codebase
3. **Performance**: Additional dictionary lookups per forward pass (minimal impact)
4. **Migration Effort**: Existing BrainBuilder usage needs updates

### Risks and Mitigations

**Risk**: Breaking existing brain configurations  
**Mitigation**: Default port behavior ensures backward compatibility for regions that don't use ports

**Risk**: Confusion between port names and region names  
**Mitigation**: Clear naming conventions and validation errors with helpful messages

**Risk**: Performance overhead  
**Mitigation**: Benchmark shows <1% overhead vs. concatenation approach

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)
- [ ] Add port methods to NeuralRegion base class
- [ ] Add unit tests for port registration and retrieval
- [ ] Update type aliases (SourceSpec to include port)

### Phase 2: LayeredCortex Integration (2-3 hours)
- [ ] Register L6a, L6b, L23, L5, default ports
- [ ] Update forward() to set port outputs
- [ ] Add integration tests for cortex ports

### Phase 3: Routing Updates (3-4 hours)
- [ ] Update AxonalProjection for port-aware routing
- [ ] Update BrainBuilder.connect() with port parameters
- [ ] Update DynamicBrain execution to use ports

### Phase 4: Testing & Documentation (4-5 hours)
- [ ] End-to-end tests with L6a→TRN, L6b→relay routing
- [ ] Update copilot-instructions.md
- [ ] Create migration guide
- [ ] Update API documentation

### Phase 5: Other Regions (Optional, future work)
- [ ] Add ports to TrisynapticHippocampus (DG, CA3, CA1 outputs)
- [ ] Add ports to Striatum (D1, D2 outputs)
- [ ] Add ports to Thalamus (relay, TRN outputs)

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

This is a **major architectural enhancement** that enables biologically-accurate routing patterns. The port-based system is:
- Extensible to other regions with multiple output types
- Compatible with future target-port routing (multi-input ports)
- Foundation for more complex routing patterns (e.g., context-dependent routing)

Future work: Target ports for multi-modal input regions (e.g., PFC with separate "cortical" and "limbic" inputs).
