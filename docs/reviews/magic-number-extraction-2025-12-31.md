# Magic Number Extraction - Tier 1.1 Implementation

**Date**: December 31, 2025  
**Related**: Architecture Review 2025-12-24, Tier 1.1  
**Status**: ✅ Complete

## Summary

Extracted magic numbers to named constants with biological context, improving code readability and biological interpretability. This was the first item from the Architecture Review Tier 1 recommendations.

## Changes Made

### 1. Extended `oscillator_constants.py` Documentation

**File**: `src/thalia/regulation/oscillator_constants.py` (line 211-217)

**Change**: Updated `GAMMA_LEARNING_MODULATION_SCALE` documentation to clarify dual usage:

```python
GAMMA_LEARNING_MODULATION_SCALE: float = 0.5
"""Scale factor for gamma-phase learning rate modulation.

effective_lr = base_lr * (SCALE + SCALE * gamma_mod)
Range: [50%, 100%] of base learning rate

Also used for gamma-phase input gain modulation:
effective_input = input * (SCALE + SCALE * gamma_amplitude)
Range: [50%, 100%] of base input
"""
```

**Biological Context**: This constant represents the fundamental modulation pattern used throughout the brain for gamma-phase gating. The 0.5 scale creates a [50%, 100%] range, ensuring baseline processing continues even during gamma troughs while allowing full strength during gamma peaks.

### 2. Updated `oscillator_utils.py` Function

**File**: `src/thalia/utils/oscillator_utils.py`

**Changes**:
- Added `GAMMA_LEARNING_MODULATION_SCALE` to imports (line 27)
- Changed function signature from `scale: float = 0.5` to `scale: float = GAMMA_LEARNING_MODULATION_SCALE` (line 218)
- Updated docstring to reference the constant name instead of hardcoded value (line 222)

**Before**:
```python
def compute_learning_rate_modulation(
    base_lr: float,
    gamma_modulation: float,
    scale: float = 0.5,  # Magic number!
) -> float:
    """...
    scale: Modulation scale (default: 0.5 for 50-100% range)
    """
```

**After**:
```python
def compute_learning_rate_modulation(
    base_lr: float,
    gamma_modulation: float,
    scale: float = GAMMA_LEARNING_MODULATION_SCALE,
) -> float:
    """...
    scale: Modulation scale (default: GAMMA_LEARNING_MODULATION_SCALE = 0.5
           for 50-100% range)
    """
```

### 3. Updated Hippocampus Region

**File**: `src/thalia/regions/hippocampus/trisynaptic.py`

**Changes**:
- Added `GAMMA_LEARNING_MODULATION_SCALE` to imports (line 117)
- Replaced inline magic numbers with named constant (lines 1240-1241)

**Before**:
```python
gamma_modulation = 0.5 + 0.5 * gamma_amplitude  # [0.5, 1.0]
```

**After**:
```python
scale = GAMMA_LEARNING_MODULATION_SCALE
gamma_modulation = scale + scale * gamma_amplitude  # [0.5, 1.0]
```

## Testing

### Validation Performed

1. **Syntax Check**: ✅ No syntax errors in modified files
2. **Import Test**: ✅ All modified modules import successfully
3. **Functional Test**: ✅ `compute_learning_rate_modulation()` produces identical results:
   - `gamma=1.0`: 0.01 (100% of base_lr=0.01)
   - `gamma=0.0`: 0.005 (50% of base_lr=0.01)
4. **Hippocampus Test**: ✅ `TrisynapticHippocampus` imports correctly and uses constant
5. **Range Verification**: ✅ Gamma modulation stays within [0.5, 1.0] range

### Test Code

```python
from thalia.utils.oscillator_utils import compute_learning_rate_modulation
from thalia.regulation.oscillator_constants import GAMMA_LEARNING_MODULATION_SCALE

lr_full = compute_learning_rate_modulation(0.01, gamma_modulation=1.0)
lr_half = compute_learning_rate_modulation(0.01, gamma_modulation=0.0)

assert abs(lr_full - 0.01) < 1e-9   # 100% of base
assert abs(lr_half - 0.005) < 1e-9  # 50% of base
```

## Assessment

### What Was Already Good

The architecture review found that **most files already use named constants correctly**:
- `oscillator_constants.py` already contained 20+ well-documented constants
- `visualization/constants.py` already excellent (NODE_ALPHA_DEFAULT, EDGE_ALPHA_DEFAULT, etc.)
- Neurons use `neuron_constants.py` (TAU_MEM_MS, V_THRESHOLD_MV, etc.)
- Most regions already reference oscillator constants

### Remaining Magic Numbers

After this extraction, remaining numeric literals fall into these categories:

1. **Configuration Defaults**: Values in `__init__` methods setting up region-specific parameters
   - Example: `sparsity=0.3, weight_scale=0.5` in hippocampus pathway initialization
   - **Assessment**: These are region-specific and better suited as config parameters

2. **Mathematical Constants**: Standard values like `threshold * 0.9` for "near threshold"
   - **Assessment**: Context-dependent, extracting would reduce clarity

3. **Clamp Bounds**: Values like `.clamp(-0.5, 0.5)` for safe numeric ranges
   - **Assessment**: Inline is clearer for safety bounds

4. **Test Tolerances**: Values like `< 0.01` in assertions
   - **Assessment**: Standard practice in test code

### Impact

- **Files Modified**: 3
- **Lines Changed**: ~10
- **Breaking Changes**: 0 (constants already existed, just adding usage)
- **Biological Clarity**: Improved (constant name conveys biological meaning)
- **Maintenance**: Improved (single source of truth for gamma modulation scale)

## Biological Significance

The extracted constant `GAMMA_LEARNING_MODULATION_SCALE = 0.5` represents a fundamental principle in neural processing:

**Gamma Oscillations as Attentional Gates**:
- **Gamma Peak**: Full processing (100% strength) - neurons maximally responsive
- **Gamma Trough**: Reduced processing (50% strength) - baseline maintenance
- **Reference**: Fries (2009) "A mechanism for cognitive dynamics: neuronal communication through neuronal coherence"

This 50-100% range ensures:
1. **Continuous Processing**: Even at gamma trough, 50% processing maintains baseline function
2. **Phase Selectivity**: 2x difference between peak and trough provides temporal selectivity
3. **Biological Plausibility**: Matches observed gamma modulation depth in cortex (~40-60%)

By extracting this to a named constant, we:
- Make the biological principle explicit
- Ensure consistency across regions (hippocampus, cortex, prefrontal)
- Enable easy adjustment if future research suggests different modulation depths

## Next Steps (Tier 1 Remaining)

As per the Architecture Review 2025-12-24:

- ✅ **1.1 Magic Number Extraction**: Complete (this document)
- ⏭️ **1.2 Component Terminology**: Document "component" vs "region" vs "pathway" usage
- ⏭️ **1.3 Weight Initialization Guidelines**: Add torch.randn/rand avoidance to docs
- ⏭️ **1.4 Mixin Documentation**: (already completed during Phase 3.2)
- ⏭️ **1.5 Growth API Documentation**: Expand examples in existing docs

**Estimated Remaining Effort**: 2-4 hours for Tier 1 completion

## References

- Architecture Review 2025-12-24: `docs/reviews/architecture-review-2025-12-24.md`
- Oscillator Constants: `src/thalia/regulation/oscillator_constants.py`
- Copilot Instructions: `.github/copilot-instructions.md` (Section: "Code Patterns")
