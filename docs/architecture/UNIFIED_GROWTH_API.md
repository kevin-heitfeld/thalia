# Unified Growth API Implementation

**Date**: December 2025
**Status**: ✅ Fully Migrated - Old API Removed

## Overview

This document describes the unified growth API that provides a consistent interface for dynamic growth across all neural components (regions and pathways) in Thalia.

## Motivation

Previously, regions and pathways had inconsistent growth APIs:
- **Regions**: Only `add_neurons()` - couldn't handle upstream growth
- **Pathways**: `grow_source()` and `grow_target()` - unclear directionality

**Key Discovery**: Multi-source architecture requires per-source growth instead of global input growth.

## Unified API

All neural regions now support:

```python
# Grow input from specific source (multi-source architecture)
region.grow_source(source_name: str, new_size: int)

# Grow output dimension (internal neuron population)
region.grow_output(n_new: int)
```

### API Semantics

- **`grow_source(source_name, new_size)`**: Expand synaptic weight columns for specific input source
  - Weight shape per source: `[n_output, old_size]` → `[n_output, new_size]`
  - Preserves existing learned weights for that source
  - Other sources remain completely unchanged
  - Initializes new input weights small to avoid disruption

- **`grow_output(n)`**: Expand output neuron population
  - Weight shape for ALL sources: `[n_output, n_input]` → `[n_output + n, n_input]`
  - Creates new neurons with new output weights
  - For regions: distributes growth across layers based on ratios
  - Expands ALL synaptic weight matrices (adds rows to each source)

## Implementation Status

### ✅ AxonalProjection
- **Has `grow_source()`**: Updates routing metadata when a source grows
- **NO `grow_output()`**: Pure spike routing with NO weights or learnable parameters
- **Growth coordination**: GrowthManager calls grow_source() on pathways during cascading growth
- **Automatic adaptation**: Updates total output size and delay buffers when sources change

### ⚠️ Note on Growth
In the current architecture:
- **Regions**: Own synaptic weights per source, implement `grow_source()`/`grow_output()`
- **AxonalProjection**: Pure routing, has `grow_source()` but NOT `grow_output()`
- **Sensory pathways**: May have encoding parameters that can grow

### ✅ LayeredCortex
- `grow_source(source_name, new_size)`: Expands synaptic weights for specific source (e.g., "thalamus", "cortex:l6a")
- `grow_output()`: Distributes growth across L4/L2/3/L5/L6a/L6b maintaining layer ratios
- **Multi-source**: Separate weight matrices per source, grows each independently

### ✅ TrisynapticHippocampus
- `grow_source(source_name, new_size)`: Expands synaptic weights for specific source (e.g., "ec_l3", "cortex")
- `grow_output()`: Distributes growth across DG/CA3/CA2/CA1 layers

### ✅ Striatum
- `grow_source(source_name, new_size)`: Expands D1/D2 weights for specific cortical/thalamic source
- `grow_output()`: Grows action space (n_actions)

### ✅ GrowthManager
- Propagates growth through multi-source architecture
- When region grows: calls `projection.grow_source()` AND `target_region.grow_source()`
- Maintains connectivity consistency across all sources

## Important Notes

### Cortex Layer Ratios

LayeredCortex grows neurons **proportionally across layers** based on ratios:
- Default ratios: L4=1.0, L2/3=1.5, L5=1.0, L6a=0.25, L6b=0.25
- Calling `grow_output(30)` actually adds:
  - L4: 30 neurons
  - L2/3: 45 neurons
  - L5: 30 neurons
  - L6a: 7-8 neurons
  - L6b: 7-8 neurons
  - **Total output** (L2/3 + L5): **75** (not 30!)
  - **L6a/L6b**: Available via port routing

This is **correct behavior** - the cortex maintains architectural proportions.

### Weight Matrix Convention

All components use PyTorch standard: **`weights[output, input]`**

When growing:
- **Input dimension**: Add **columns** (dimension 1)
- **Output dimension**: Add **rows** (dimension 0)

## Testing

Comprehensive test suite: `tests/unit/test_unified_growth_api.py`

**Test Coverage** (v3.0+ tests):
1. ✅ Region `grow_source()` expands weights for specific source
2. ✅ Region `grow_output()` expands neuron population
3. ✅ AxonalProjection `grow_source()` updates routing
4. ✅ Multi-source growth (cortex + thalamus → striatum)
5. ✅ GrowthManager coordination (propagates through pathways + regions)
6. ✅ Forward pass after multi-source growth

**Note**: AxonalProjection has `grow_source()` but not `grow_output()` (raises NotImplementedError).

## Usage Examples

### Example: Growing a Region (Current Architecture)
```python
# Primary visual cortex with multi-source inputs
v1 = LayeredCortex(
    config=LayeredCortexConfig(...),
    sizes={"l4_size": 128, "l23_size": 192, "l5_size": 128, ...},
    device=device
)

# Add input source (creates synaptic weights)
v1.add_input_source("thalamus", n_input=128, learning_strategy="stdp")

# If thalamus grows from 128 → 138 neurons
# Region grows its synaptic weights for that source
v1.grow_source("thalamus", new_size=138)  # [n_output, 128] → [n_output, 138]

# Region itself grows output
v1.grow_output(30)  # Adds neurons across all layers proportionally
```

### Historical Example: Pathway Growth (Deprecated in v2.0+)
```python
# NOTE: This pattern is from v1.0 when pathways had weights
# In v2.0+, use AxonalProjection (weightless) + region synaptic weights

# V1 itself expands (will add ~95 to output due to layer ratios)
v1.grow_output(40)  # Adds L4:40, L2/3:60, L5:40, L6a:10, L6b:10 → output +100
```

## Future Work

- [ ] Add deprecation warnings to old methods
- [ ] Update specialized pathways (SpikingAttentionPathway, SpikingReplayPathway)
- [ ] Add growth visualization tools
- [ ] Document growth strategies for curriculum training

## References

- **Original Issue**: Growth mechanism review revealed missing `grow_input()` for regions
- **Key Insight**: Both regions and pathways need bidirectional growth (input + output)
- **Implementation**: `src/thalia/regions/*/`, `src/thalia/pathways/spiking_pathway.py`
- **Tests**: `tests/unit/test_unified_growth_api.py`
