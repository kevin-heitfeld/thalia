# Port-Based Routing Pattern

## Overview

Port-based routing allows multi-layer regions to expose specific internal outputs for targeted connections. Instead of routing all outputs to all targets, ports enable biologically-accurate layer-specific connectivity patterns.

**Why Ports?**
- **Biological accuracy**: Cortex L2/3 → other cortex, L5 → striatum, L6a → TRN, L6b → relay
- **Explicit routing**: Clear intent in architecture definitions
- **Flexible connectivity**: Same region can have multiple projection types
- **No manual slicing**: Builder handles output selection automatically

## Basic Usage

### Source Ports (Output Selection)

Connect specific layer outputs from multi-layer regions:

```python
from thalia.core.brain_builder import BrainBuilder

builder = BrainBuilder(global_config)

# Add layered cortex (6 layers: L4, L2/3, L5, L6a, L6b)
builder.add_component(
    "cortex", "layered_cortex",
    n_input=784,
    n_output=256,
    l4_size=128,
    l23_size=128,   # Cortico-cortical output
    l5_size=96,     # Subcortical output
    l6a_size=16,    # CT type I → TRN (spatial attention)
    l6b_size=16,    # CT type II → relay (gain modulation)
)

# Layer-specific routing with source_port
builder.connect("cortex", "hippocampus", source_port="l23")  # Cortico-cortical
builder.connect("cortex", "striatum", source_port="l5")      # Cortico-subcortical
builder.connect("cortex", "thalamus", source_port="l6a", target_port="trn")  # CT type I → TRN
builder.connect("cortex", "thalamus", source_port="l6b", target_port="relay")  # CT type II → relay
```

**Without ports**, all connections would receive the concatenated output `[L2/3_spikes, L5_spikes]`, mixing projection types.

### Target Ports (Input Differentiation)

Differentiate input types on multi-input regions:

```python
# Cortex receives feedforward + top-down inputs
builder.connect("thalamus", "cortex", target_port="feedforward")
builder.connect("pfc", "cortex", target_port="top_down")

# Hippocampus CA1 receives cortical + entorhinal inputs
builder.connect("cortex", "hippocampus", target_port="cortical")
builder.connect("entorhinal", "hippocampus", target_port="ec_l3")
```

**Target ports** set config parameters like `pfc_modulation_size` or `ec_l3_input_size` automatically.

## Regions with Source Ports

### LayeredCortex

**Available Ports**:
- `"l23"` → Layer 2/3 (cortico-cortical connections)
- `"l5"` → Layer 5 (cortico-subcortical connections)
- `"l6a"` → Layer 6a (CT type I → TRN, spatial attention, low gamma)
- `"l6b"` → Layer 6b (CT type II → relay, gain modulation, high gamma)
- `"l4"` → Layer 4 (rarely used externally)

**Biological Projections**:
```python
# Cortico-cortical (L2/3 → other cortex)
builder.connect("v1", "v2", source_port="l23")
builder.connect("v1", "hippocampus", source_port="l23")

# Cortico-subcortical (L5 → striatum, thalamus, brainstem)
builder.connect("motor_cortex", "striatum", source_port="l5")
builder.connect("pfc", "thalamus", source_port="l5")

# Corticothalamic feedback (dual pathways)
builder.connect("cortex", "thalamus", source_port="l6a", target_port="trn")  # Spatial attention
builder.connect("cortex", "thalamus", source_port="l6b", target_port="relay")  # Gain modulation
```

**Default Output** (no port specified):
Concatenated `[L2/3_spikes, L5_spikes]` for backward compatibility.

### Future: TrisynapticHippocampus

**Planned Ports**:
- `"ca1"` → CA1 output (default)
- `"ca3"` → CA3 output (direct access for recurrent connections)

**Example Usage**:
```python
# Standard CA1 output
builder.connect("hippocampus", "pfc")

# Direct CA3 access for recurrent connections
builder.connect("hippocampus", "hippocampus", source_port="ca3")
```

*(Not yet implemented - CA1 is currently the only output)*

### Future: Cerebellum

**Planned Ports**:
- `"dcn"` → Deep cerebellar nuclei output
- `"purkinje"` → Purkinje cell output (for debugging/visualization)

*(Not yet implemented)*

## Target Ports

Target ports configure how regions handle multiple input types.

### Common Target Ports

**Feedforward Inputs** (contribute to `n_input`):
- `"feedforward"` → Primary input (default)
- `"cortical"` → Cortical input (hippocampus)
- `"hippocampal"` → Hippocampal input (prefrontal)

**Modulatory Inputs** (set separate config params):
- `"top_down"` → Top-down attention/modulation
- `"pfc_modulation"` → Prefrontal control signals
- `"ec_l3"` → Entorhinal cortex L3 input (hippocampus CA1)

### Example: Multi-Input Cortex

```python
# Cortex with feedforward + top-down inputs
builder.add_component(
    "cortex", "layered_cortex",
    n_input=256,  # Feedforward size (auto-inferred if omitted)
    # pfc_modulation_size auto-set by top_down connection
    n_output=512,
)

builder.connect("thalamus", "cortex")  # Default is feedforward
builder.connect("pfc", "cortex", target_port="top_down")
```

**Result**: Builder sets `pfc_modulation_size` automatically based on PFC output size.

### Example: Multi-Input Hippocampus

```python
# Hippocampus CA1 with cortical + entorhinal inputs
builder.add_component(
    "hippocampus", "trisynaptic_hippocampus",
    n_output=128,
    # n_input and ec_l3_input_size auto-inferred
)

builder.connect("cortex", "hippocampus", target_port="cortical")
builder.connect("entorhinal", "hippocampus", target_port="ec_l3")
```

**Result**: Builder calculates `n_input` (cortical) and `ec_l3_input_size` separately.

## Implementation Details

### How Port Routing Works

**1. Size Inference** (`brain_builder.py:_get_source_output_size()`):
```python
def _get_source_output_size(source_name: str, source_port: Optional[str]) -> int:
    if source_port == "l23":
        return config["l23_size"]
    elif source_port == "l5":
        return config["l5_size"]
    elif source_port == "l6a":
        return config["l6a_size"]
    elif source_port == "l6b":
        return config["l6b_size"]
    # ... etc
```

**2. Pathway Creation**:
```python
# ConnectionSpec stores port information
spec = ConnectionSpec(
    source="cortex",
    target="striatum",
    source_port="l5",  # Stored for pathway construction
    target_port=None,
)
```

**3. Runtime Routing** (handled by pathway):
- **AxonalProjection**: Concatenates multi-source inputs with port-specific slicing
- Port slicing happens during build (input size inference)
- Runtime: direct tensor concatenation, no dynamic routing overhead

### Adding Port Support to New Regions

**Step 1**: Define port-to-size mapping in `brain_builder.py`:
```python
def _get_source_output_size(self, source_name: str, source_port: Optional[str]) -> int:
    # ... existing code ...

    # Add new region
    if source_spec.registry_name == "my_region":
        if source_port == "port_a":
            return config["port_a_size"]
        elif source_port == "port_b":
            return config["port_b_size"]
```

**Step 2**: Document ports in region docstring:
```python
class MyRegion(NeuralComponent):
    """Multi-output region with port-based routing.

    **Port-Based Routing**:
    - source_port="port_a" → Primary output
    - source_port="port_b" → Secondary output

    Usage:
        >>> builder.connect("my_region", "target", source_port="port_a")
    """
```

**Step 3**: Return appropriate outputs in `forward()`:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Return concatenated outputs for backward compatibility
    return torch.cat([port_a_output, port_b_output], dim=0)
```

Pathways handle slicing based on `source_port` specification.

## Best Practices

### 1. Use Biologically-Motivated Ports

Match neuroscience projection patterns:
```python
# ✅ Correct: L5 → striatum (cortico-striatal)
builder.connect("motor_cortex", "striatum", source_port="l5")

# ❌ Wrong: L2/3 → striatum (not a biological projection)
builder.connect("motor_cortex", "striatum", source_port="l23")
```

### 2. Specify Layer Sizes Explicitly

Don't rely on defaults for layered regions:
```python
# ✅ Explicit layer sizes
builder.add_component(
    "cortex", "layered_cortex",
    n_input=784,
    l4_size=128,
    l23_size=128,
    l5_size=96,
    l6a_size=16,
    l6b_size=16,
)

# ❌ Missing layer sizes (will raise error during connection)
builder.add_component("cortex", "layered_cortex", n_input=784)
```

### 3. Document Multi-Input Regions

Make target port expectations clear:
```python
class MyRegion(NeuralComponent):
    """Region with feedforward + modulatory inputs.

    **Target Ports**:
    - target_port="feedforward" → Counted in n_input (default)
    - target_port="modulation" → Sets modulation_size config param
    """
```

### 4. Omit Ports for Single-Output Regions

Only use ports when region has multiple output types:
```python
# ✅ No port needed (single output)
builder.connect("thalamus", "cortex")

# ❌ Unnecessary port (thalamus has single output)
builder.connect("thalamus", "cortex", source_port="relay")
```

## Common Patterns

### Pattern 1: Cortical Hierarchy

```python
# V1 → V2 → PFC hierarchy
builder.add_component("v1", "layered_cortex", ...)
builder.add_component("v2", "layered_cortex", ...)
builder.add_component("pfc", "layered_cortex", ...)

# Feedforward: L2/3 → next cortex L4
builder.connect("v1", "v2", source_port="l23")
builder.connect("v2", "pfc", source_port="l23")

# Feedback: L6a → previous cortex (top-down modulation via TRN)
builder.connect("v2", "v1", source_port="l6a", target_port="top_down")
builder.connect("pfc", "v2", source_port="l6a", target_port="top_down")
```

### Pattern 2: Cortico-Basal Ganglia Loop

```python
# Cortex → Striatum → Thalamus → Cortex
builder.add_component("cortex", "layered_cortex", ...)
builder.add_component("striatum", "striatum_circuit", ...)
builder.add_component("thalamus", "thalamic_relay", ...)

# Cortex L5 → Striatum (action selection)
builder.connect("cortex", "striatum", source_port="l5")

# Striatum → Thalamus → Cortex L4 (closed loop)
builder.connect("striatum", "thalamus")
builder.connect("thalamus", "cortex")
```

### Pattern 3: Cortico-Hippocampal Loop

```python
# Cortex ↔ Hippocampus via entorhinal
builder.add_component("cortex", "layered_cortex", ...)
builder.add_component("hippocampus", "trisynaptic_hippocampus", ...)

# Cortex L2/3 → Hippocampus (via entorhinal)
builder.connect("cortex", "hippocampus", source_port="l23")

# Hippocampus → PFC (memory retrieval)
builder.connect("hippocampus", "pfc")
```

## Troubleshooting

### Error: "Component 'X' does not support port 'Y'"

**Cause**: Region doesn't have multi-output architecture.

**Solution**: Remove `source_port` or check region documentation:
```python
# Check what ports are available
print(region.__doc__)  # Look for "Port-Based Routing" section
```

### Error: "Must specify all layer sizes (l4_size, l23_size, l5_size, l6a_size, l6b_size)"

**Cause**: Using `source_port` with cortex without explicit layer sizes.

**Solution**: Specify all layer sizes in `add_component()`:
```python
builder.add_component(
    "cortex", "layered_cortex",
    n_input=784,
    l4_size=128,
    l23_size=128,
    l5_size=96,
    l6a_size=16,
    l6b_size=16,
)
```

### Unexpected Output Size

**Cause**: Pathway receiving wrong port output.

**Debug**:
```python
# Check actual sizes
print(f"L2/3 size: {cortex.l23_size}")
print(f"L5 size: {cortex.l5_size}")

# Verify connection spec
print(builder._connections)
```

## Related Documentation

- **Architecture**: `docs/architecture/ARCHITECTURE_OVERVIEW.md`
- **Brain Builder**: `src/thalia/core/brain_builder.py`
- **Layered Cortex**: `src/thalia/regions/cortex/layered_cortex.py`
- **Component Parity**: `docs/patterns/component-parity.md`

## References

**Neuroscience**:
- Harris & Shepherd (2015): Cortical layer projections
- Rockland & Pandya (1979): Cortico-cortical connections
- Morishima & Kawaguchi (2006): Layer-specific outputs

**Implementation**:
- BrainBuilder: Connection orchestration with port routing
- AxonalProjection: Pure spike routing (no weights)
- LayeredCortex: 6-layer architecture with biological projections (L4, L2/3, L5, L6a, L6b)
