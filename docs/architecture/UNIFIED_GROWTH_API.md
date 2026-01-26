# Unified Growth API Implementation

**Date**: December 2025
**Status**: ✅ Fully Migrated - Old API Removed

## Overview

This document describes the unified growth API that provides a consistent interface for dynamic growth across all neural components (regions and pathways) in Thalia.

## Motivation

Previously, regions and pathways had inconsistent growth APIs:
- **Regions**: Only `add_neurons()` - couldn't handle upstream growth
- **Pathways**: `grow_source()` and `grow_target()` - unclear directionality

**Key Discovery**: Both regions AND pathways have asymmetric dimensions (`n_input ≠ n_output`) and both need bidirectional growth support.

## Unified API

All neural components now support:

```python
# Grow input dimension (upstream growth)
component.grow_input(n_new: int)

# Grow output dimension (internal neuron population)
component.grow_output(n_new: int)
```

### API Semantics

- **`grow_input(n)`**: Expand input weight matrix **columns** to accept larger input
  - Weight shape: `[n_output, n_input]` → `[n_output, n_input + n]`
  - Preserves existing learned weights
  - Initializes new input weights small to avoid disruption

- **`grow_output(n)`**: Expand output neuron population
  - Weight shape: `[n_output, n_input]` → `[n_output + n, n_input]`
  - Creates new neurons with new output weights
  - For regions: distributes growth across layers based on ratios

## Implementation Status

### ✅ AxonalProjection
- **NO growth needed**: Pure spike routing, NO weights
- Growth happens at target regions which own synaptic weights
- AxonalProjection automatically adapts to source/target size changes

### ⚠️ Note on Growth
In the current architecture:
- **Regions grow**: They own neurons and synaptic weights
- **Pathways don't grow**: AxonalProjection is weightless routing
- **Sensory pathways**: May have encoding parameters that can grow

The examples below show the growth API design, but AxonalProjection itself doesn't need growth.

### ✅ LayeredCortex
- `grow_input()`: Expands `w_input_l4` columns for upstream growth
- `grow_output()`: Distributes growth across L4/L2/3/L5/L6a/L6b maintaining layer ratios
- **Includes**: `w_l23_inhib`, `l23_phase_prefs`, STP module expansion

### ✅ TrisynapticHippocampus
- `grow_input()`: Expands EC input weights (`w_ec_dg`, `w_ec_ca3`)
- `grow_output()`: Distributes growth across DG/CA3/CA2/CA1 layers

### ✅ Striatum
- `grow_input()`: Expands D1/D2 pathway weight columns
- `grow_output()`: Grows action space (n_actions)

### ✅ GrowthCoordinator
- Updated to propagate input growth to downstream regions
- After pathway grows input, calls `target_region.grow_input()`
- Maintains connectivity consistency across growth

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

### Backward Compatibility

⚠️ **Old API Removed** (December 2025):
- `add_neurons()` → **REMOVED** - Use `grow_output()`
- `grow_source()` → **REMOVED** - Use `grow_input()`
- `grow_target()` → **REMOVED** - Use `grow_output()`

All code must now use the unified `grow_input()` / `grow_output()` API.

## Testing

Comprehensive test suite: `tests/unit/test_unified_growth_api.py`

**Test Coverage** (10 tests, all passing):
1. ✅ Pathway `grow_input()` expands input dimension
2. ✅ Pathway `grow_output()` expands output dimension
3. ✅ Cortex `grow_input()` expands input weights
4. ✅ Cortex `grow_output()` expands neuron population (with layer ratios!)
5. ✅ Hippocampus `grow_input()` expands EC weights
6. ✅ Striatum `grow_input()` expands D1/D2 weights
7. ✅ Bidirectional pathway growth (input then output)
8. ✅ Bidirectional region growth (input then output)
9. ✅ Forward pass after unified growth
10. ✅ Backward compatibility with old methods

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

### End-to-End Growth with GrowthCoordinator
```python
coordinator = GrowthCoordinator(brain)

# Grow pathway - automatically propagates to target region
coordinator.coordinate_growth(
    pathway_name="thalamus_to_cortex",
    n_new_source=10,  # Thalamus grows by 10
    n_new_target=20   # Cortex grows by 20 (actual: ~50 with ratios)
)

# Coordinator automatically:
# 1. Grows pathway input dimension
# 2. Grows pathway output dimension
# 3. Grows target region input dimension (NEW!)
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
