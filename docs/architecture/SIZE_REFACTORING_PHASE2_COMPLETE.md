# Size Specification Refactoring - Phase 2 Implementation

**Date**: January 10, 2026
**Status**: ✅ PHASE 2 COMPLETE
**Previous**: Phase 1 (LayerSizeCalculator)
**Next**: Phase 3 (Builder Input Size Inference)

---

## What Was Implemented

### 1. Updated `LayeredCortexConfig`

**File**: `src/thalia/regions/cortex/config.py`

#### Changed Docstring
- **Before**: Vague "Layer Sizes (REQUIRED)" with auto-compute mention
- **After**: Clear SIZE SPECIFICATION section with three patterns:
  1. FROM INPUT: `calc.cortex_from_input(input_size)`
  2. FROM OUTPUT: `calc.cortex_from_output(target_output_size)`
  3. FROM SCALE: `calc.cortex_from_scale(scale_factor)`
- Each pattern shows complete example code
- Makes `LayerSizeCalculator` the primary API

#### Removed Auto-Compute from `__post_init__()`
- **Before**: If all sizes were 0, auto-computed from `input_size`
- **After**: No auto-compute - requires explicit sizes
- Implements **Design Decision 1**: "Be explicit"
- Users must call `LayerSizeCalculator` explicitly

#### Updated Validation
- **Before**: Skipped validation if `input_size=0`, allowed ambiguous configs
- **After**: Always validates core layers (L4, L2/3, L5) are > 0
- Error message includes helpful `LayerSizeCalculator` usage example
- L6a/L6b still optional (can be 0 if not modeling corticothalamic feedback)

#### Updated `from_input_size()` Factory
- **Before**: Used deprecated `compute_cortex_layer_sizes()`
- **After**: Uses new `LayerSizeCalculator`
- Convenience method for common pattern
- Returns complete config with all layer sizes

### 2. Deprecated `calculate_layer_sizes()`

**File**: `src/thalia/regions/cortex/config.py`

- Added `DeprecationWarning` to function
- Kept function working for backward compatibility
- Clear migration message with example:
  ```python
  from thalia.config import LayerSizeCalculator
  calc = LayerSizeCalculator()
  sizes = calc.cortex_from_output(target_output_size=n_output)
  ```
- Removal planned for v0.4.0

### 3. Input Size is Optional

Implements **Design Decision 3**: `input_size` can be 0 in config.

- BrainBuilder will infer it from connections (Phase 3)
- Config allows `input_size=0` without validation error
- Supports two-pass building pattern

---

## Testing

Created `temp/test_phase2.py` with comprehensive tests:

### Test 1: Explicit Sizes Required ✅
- Configs with size=0 correctly raise `ValueError`
- Error message includes helpful migration guide
- Explicit sizes work correctly

### Test 2: LayerSizeCalculator Integration ✅
- All three patterns work with config:
  - `cortex_from_input(192)` → config
  - `cortex_from_output(300)` → config
  - `cortex_from_scale(128)` → config
- Sizes computed correctly
- Properties (`output_size`, `total_neurons`) work

### Test 3: `from_input_size()` Factory ✅
- Factory method uses new calculator
- Produces identical results to direct calculator call
- Convenience for common pattern

### Test 4: Deprecation Warning ✅
- Old `calculate_layer_sizes()` emits `DeprecationWarning`
- Function still works (backward compatibility)
- Clear migration message

### Test 5: Optional Input Size ✅
- Config accepts `input_size=0`
- Validation passes with explicit layer sizes
- Ready for Phase 3 builder inference

**Result**: All tests passing ✓

---

## Benefits Achieved

### 1. Clarity
- **Before**: "You can specify sizes OR let them auto-compute"
- **After**: "You MUST use LayerSizeCalculator to compute sizes"
- No ambiguity

### 2. Debuggability
- Size calculations are explicit and traceable
- No hidden auto-compute magic
- Error messages guide users to correct usage

### 3. Consistency
- One way to specify sizes: `LayerSizeCalculator`
- All configs will follow same pattern
- Easier to teach and learn

### 4. Flexibility
- Three specification patterns for different use cases
- Custom ratios supported via `BiologicalRatios`
- Optional `input_size` for builder inference

---

## Migration Examples

### Pattern 1: Update Existing Code

```python
# OLD (no longer works):
config = LayeredCortexConfig(
    input_size=192,
    l4_size=0,  # Auto-compute
    l23_size=0,
    l5_size=0,
)

# NEW (required):
from thalia.config import LayerSizeCalculator
calc = LayerSizeCalculator()
sizes = calc.cortex_from_input(input_size=192)
config = LayeredCortexConfig(
    input_size=sizes["input_size"],
    l4_size=sizes["l4_size"],
    l23_size=sizes["l23_size"],
    l5_size=sizes["l5_size"],
    l6a_size=sizes["l6a_size"],
    l6b_size=sizes["l6b_size"],
)
```

### Pattern 2: Use Factory Method

```python
# Convenience for common case:
config = LayeredCortexConfig.from_input_size(input_size=192)
```

### Pattern 3: Specify Sizes Explicitly

```python
# For fine-grained control:
config = LayeredCortexConfig(
    l4_size=288,
    l23_size=576,
    l5_size=288,
    l6a_size=115,
    l6b_size=74,
)
# input_size=0 will be inferred by builder (Phase 3)
```

---

## Files Changed

1. **Modified**: `src/thalia/regions/cortex/config.py`
   - Updated `LayeredCortexConfig` docstring
   - Removed auto-compute from `__post_init__()`
   - Updated validation with helpful error messages
   - Updated `from_input_size()` to use new calculator
   - Added deprecation warning to `calculate_layer_sizes()`

2. **Created**: `temp/test_phase2.py`
   - 5 comprehensive tests
   - All passing ✓

---

## Design Decisions Implemented

From the refactoring plan:

✅ **Decision 1**: No auto-compute in `__post_init__()` - Users must use `LayerSizeCalculator` explicitly
✅ **Decision 3**: `input_size` in config is optional - Builder will infer in Phase 3

---

## Breaking Changes

### For Users Who Relied on Auto-Compute

**Before**:
```python
config = LayeredCortexConfig(input_size=192)  # Auto-computed layers
```

**After**:
```python
config = LayeredCortexConfig.from_input_size(input_size=192)  # Explicit factory
# OR
from thalia.config import LayerSizeCalculator
calc = LayerSizeCalculator()
sizes = calc.cortex_from_input(192)
config = LayeredCortexConfig(**sizes)  # Explicit sizes
```

### Migration Strategy

1. **Search for**: `LayeredCortexConfig(input_size=...` without explicit layer sizes
2. **Replace with**: Either `LayeredCortexConfig.from_input_size(...)` or calculator usage
3. **Run tests**: Verify sizes are correct

---

## Next Steps (Phase 3)

According to the refactoring plan:

### Phase 3: Builder Input Size Inference

1. **Two-Pass Building**:
   - Pass 1: Create components with configs as-is
   - Pass 2: Infer `input_size` from connection graph
   - Pass 3: Finalize components with correct `input_size`
   - Pass 4: Create connections

2. **Add `finalize_input_size()` to Regions**:
   - Check if `config.input_size == 0`
   - If so, set it from inferred value
   - If not, validate it matches inferred value
   - Ensures consistency

3. **Update `from_thalia_config()`**:
   - Use `LayerSizeCalculator` for cortex
   - Remove hardcoded layer size calculations
   - Let builder infer `input_size` from connections

---

## Summary

Phase 2 successfully implements:
- ✅ Removed auto-compute from configs (explicit over implicit)
- ✅ Required `LayerSizeCalculator` usage with clear docs
- ✅ Updated error messages to guide users
- ✅ Deprecated duplicate `calculate_layer_sizes()` function
- ✅ Made `input_size` optional for builder inference
- ✅ Comprehensive testing (all passing)
- ✅ Clear migration path for existing code

The config API is now clean and explicit. Ready for Phase 3 to implement builder-side input size inference from the connection graph.
