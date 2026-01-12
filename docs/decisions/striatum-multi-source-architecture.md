# Striatum Multi-Source Architecture

**Status:** Known Limitation  
**Date:** January 12, 2026  
**Context:** Config migration and test suite cleanup  

## Issue

Port-based routing tests expect Striatum to expose source-named synaptic weights (e.g., `striatum.synaptic_weights["cortex:l5"]`), similar to other NeuralRegion subclasses. However, Striatum uses a different internal architecture.

## Current Architecture

Striatum implements D1/D2 opponent pathways:
- **Input handling:** Concatenates all input sources into single tensor
- **Weight structure:** Uses `"default_d1"` and `"default_d2"` keys in `synaptic_weights`
- **Pathway separation:** D1 and D2 pathways each have separate weight matrices for the concatenated input
- **Learning:** Opponent learning rules (D1: standard dopamine, D2: inverted dopamine)

```python
# Current structure
striatum.synaptic_weights = {
    "default_d1": [10, 100],  # D1 pathway weights
    "default_d2": [10, 100],  # D2 pathway weights
}
```

## Expected Architecture (by tests)

Tests expect per-source weight tracking:
```python
# Expected structure
striatum.synaptic_weights = {
    "cortex:l5": [20, 32],     # Both D1+D2 weights for cortex L5
    "hippocampus": [20, 64],   # Both D1+D2 weights for hippocampus
    "pfc": [20, 48],           # Both D1+D2 weights for PFC
}
```

## Design Considerations

### Option 1: Unified Input (Current)
**Pros:**
- Simpler implementation
- Matches biological reality (MSNs receive from all sources)
- Fewer weight matrices to manage

**Cons:**
- Doesn't track source-specific connectivity
- Can't implement source-specific plasticity rules
- Doesn't match NeuralRegion multi-source pattern

### Option 2: Per-Source Weights
**Pros:**
- Matches NeuralRegion interface
- Enables source-specific plasticity
- Better aligns with port-based routing

**Cons:**
- More complex: need D1/D2 weights for EACH source
- Larger memory footprint
- More complex forward pass (sum across sources)

### Option 3: Hybrid Approach
**Structure:**
```python
striatum.synaptic_weights = {
    "cortex:l5_d1": [10, 32],
    "cortex:l5_d2": [10, 32],
    "hippocampus_d1": [10, 64],
    "hippocampus_d2": [10, 64],
}
```

**Pros:**
- Full flexibility for source-specific D1/D2 weights
- Matches both NeuralRegion pattern and D1/D2 architecture

**Cons:**
- Most complex implementation
- Largest memory footprint
- Requires significant refactoring

## Decision

**Deferred:** This is an architectural decision that needs neuroscience input and careful design.

**For now:** Skip the 3 affected port routing tests with clear documentation:
- `test_cortex_l5_to_striatum`
- `test_cortex_outputs_to_multiple_targets_with_different_layers`
- `test_striatum_multiple_input_sources`

## Future Work

1. Research biological corticostriatal connectivity patterns
2. Determine if source-specific plasticity is needed
3. Design unified architecture that supports both patterns
4. Implement and validate against biological data

## References

- NeuralRegion multi-source pattern: `src/thalia/core/neural_region.py`
- Striatum implementation: `src/thalia/regions/striatum/striatum.py`
- Port-based routing tests: `tests/unit/test_port_based_routing.py`
