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

### ✅ SpikingPathway
- `grow_input()`: Expands input (pre-synaptic) dimension
- `grow_output()`: Expands output (post-synaptic) dimension

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

### Growing a Pathway
```python
# Visual pathway from retina to thalamus
visual_pathway = SpikingPathway(
    config=PathwayConfig(n_input=784, n_output=128)
)

# Retina grows from 784 → 804
visual_pathway.grow_input(20)  # [128, 784] → [128, 804]

# Thalamus needs more neurons
visual_pathway.grow_output(10)  # [128, 804] → [138, 804]
```

### Growing a Region
```python
# Primary visual cortex
v1 = LayeredCortex(
    config=LayeredCortexConfig(n_input=128, n_output=256)
)

# Thalamus upstream grows from 128 → 138
v1.grow_input(10)  # Expands w_input_l4: [L4, 128] → [L4, 138]

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
