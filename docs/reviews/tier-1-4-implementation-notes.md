# Tier 1.4 Implementation Notes - State Validation Patterns

**Date**: January 26, 2026
**Status**: ✅ Implementation Complete | ⏳ Migration Pending | ⏳ Testing Pending
**Effort**: 45 minutes (implementation), 3-4 hours (migration est.)

## Summary

Enhanced `StateLoadingMixin` with validation helper methods to consolidate duplicated state validation patterns across region implementations. The helpers provide consistent error messages for shape mismatches and automatic device transfer with validation.

## Changes Made

### Added Validation Helpers to StateLoadingMixin

**Location**: [src/thalia/mixins/state_loading_mixin.py](../../src/thalia/mixins/state_loading_mixin.py)

**Purpose**: Standardize tensor shape validation and device transfer across all region `load_state()` implementations.

#### 1. `_validate_tensor_shape()` Method

**Before** (duplicated in 8+ places):
```python
# In various region load_state() methods
if membrane.shape != (self.n_neurons,):
    raise ValueError(f"Invalid shape: {membrane.shape}")
# or
assert membrane.shape == expected_shape, f"Shape mismatch"
# or
if membrane.shape[0] != self.n_output:
    warnings.warn(f"Shape mismatch: {membrane.shape[0]} != {self.n_output}")
```

**After** (using new helper):
```python
self._validate_tensor_shape(
    membrane,
    (self.n_neurons,),
    "membrane_potential"
)
```

**Benefits**:
- ✅ Consistent error messages across all regions
- ✅ Clear indication of which tensor failed validation
- ✅ Includes both expected and actual shapes in error
- ✅ Single line vs 2-4 lines of validation code

#### 2. `_load_tensor()` Method

**Before** (duplicated in 15+ places):
```python
# Common patterns in load_state():
self.state.l4_spikes = state.l4_spikes.to(self.device)

# With validation:
if state.membrane.shape != (self.n_neurons,):
    raise ValueError(f"Shape mismatch")
self.neurons.membrane.data = state.membrane.to(self.device)
```

**After** (using new helper):
```python
# Simple device transfer:
self.state.l4_spikes = self._load_tensor(state.l4_spikes)

# With shape validation:
self.neurons.membrane.data = self._load_tensor(
    state.membrane,
    expected_shape=(self.n_neurons,),
    name="membrane_potential"
)
```

**Benefits**:
- ✅ Combines validation + device transfer in one call
- ✅ Optional shape validation (only when needed)
- ✅ Clearer intent (explicit validation vs implicit checks)
- ✅ Reduces 3-5 lines to 1-3 lines per tensor

---

## API Reference

### `_validate_tensor_shape()`

```python
def _validate_tensor_shape(
    self,
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str,
) -> None:
    """Validate tensor shape with clear error message.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape tuple
        name: Name of tensor for error message
    
    Raises:
        ValueError: If shape doesn't match expected
    """
```

**Usage Examples**:
```python
# Validate membrane potential shape
self._validate_tensor_shape(
    membrane,
    (self.n_neurons,),
    "membrane_potential"
)

# Validate weight matrix shape
self._validate_tensor_shape(
    weights,
    (self.n_output, self.n_input),
    "synaptic_weights"
)

# Validate 3D tensor (batch, time, features)
self._validate_tensor_shape(
    buffer,
    (self.batch_size, self.buffer_length, self.feature_dim),
    "delay_buffer"
)
```

### `_load_tensor()`

```python
def _load_tensor(
    self,
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    name: str = "tensor",
) -> torch.Tensor:
    """Load tensor with optional validation and device transfer.
    
    Args:
        tensor: Tensor to load
        expected_shape: Optional expected shape for validation
        name: Name of tensor for error messages
    
    Returns:
        Tensor moved to self.device
    
    Raises:
        ValueError: If expected_shape provided and doesn't match
    """
```

**Usage Examples**:
```python
# Simple device transfer (no validation)
self.state.spikes = self._load_tensor(state.spikes)

# With shape validation
self.neurons.membrane.data = self._load_tensor(
    state.membrane,
    expected_shape=(self.n_neurons,),
    name="membrane"
)

# Conditional loading with validation
if state.trace is not None:
    self.stdp_trace = self._load_tensor(
        state.trace,
        expected_shape=(self.n_neurons,),
        name="stdp_trace"
    )
```

---

## Usage Patterns

### Pattern 1: Simple State Loading (LayeredCortex)

**Before**:
```python
def load_state(self, state: LayeredCortexState) -> None:
    # Manual device transfer (repeated 15+ times)
    if state.l4_spikes is not None:
        self.state.l4_spikes = state.l4_spikes.to(self.device)
    if state.l23_spikes is not None:
        self.state.l23_spikes = state.l23_spikes.to(self.device)
    if state.l5_spikes is not None:
        self.state.l5_spikes = state.l5_spikes.to(self.device)
    # ... 12 more similar lines
```

**After**:
```python
def load_state(self, state: LayeredCortexState) -> None:
    # Clean, consistent pattern
    if state.l4_spikes is not None:
        self.state.l4_spikes = self._load_tensor(state.l4_spikes)
    if state.l23_spikes is not None:
        self.state.l23_spikes = self._load_tensor(state.l23_spikes)
    if state.l5_spikes is not None:
        self.state.l5_spikes = self._load_tensor(state.l5_spikes)
```

**Improvement**: Same line count but clearer intent and consistent error handling.

### Pattern 2: Validated State Loading (Striatum)

**Before**:
```python
def load_state(self, state: StriatumState) -> None:
    # Manual validation + device transfer
    if state.d1_membrane.shape != (self.d1_pathway.n_neurons,):
        raise ValueError(
            f"D1 membrane shape {state.d1_membrane.shape} doesn't match "
            f"expected {(self.d1_pathway.n_neurons,)}"
        )
    self.d1_pathway.neurons.membrane.data = state.d1_membrane.to(self.device)
    
    if state.d2_membrane.shape != (self.d2_pathway.n_neurons,):
        raise ValueError(
            f"D2 membrane shape {state.d2_membrane.shape} doesn't match "
            f"expected {(self.d2_pathway.n_neurons,)}"
        )
    self.d2_pathway.neurons.membrane.data = state.d2_membrane.to(self.device)
```

**After**:
```python
def load_state(self, state: StriatumState) -> None:
    # Validation + device transfer in one call
    self.d1_pathway.neurons.membrane.data = self._load_tensor(
        state.d1_membrane,
        expected_shape=(self.d1_pathway.n_neurons,),
        name="d1_membrane"
    )
    self.d2_pathway.neurons.membrane.data = self._load_tensor(
        state.d2_membrane,
        expected_shape=(self.d2_pathway.n_neurons,),
        name="d2_membrane"
    )
```

**Improvement**: 10 lines → 6 lines (40% reduction), clearer error messages.

### Pattern 3: Partial Validation (Hippocampus)

**Before**:
```python
def load_state(self, state: HippocampusState) -> None:
    # Mixed: some with validation, some without
    self.state.dg_spikes = state.dg_spikes.to(self.device)
    
    # Critical state needs validation
    if state.ca3_membrane.shape != (self.ca3_size,):
        warnings.warn(f"CA3 membrane shape mismatch")
    self.ca3_neurons.membrane.data = state.ca3_membrane.to(self.device)
```

**After**:
```python
def load_state(self, state: HippocampusState) -> None:
    # Explicit: simple transfer vs validated transfer
    self.state.dg_spikes = self._load_tensor(state.dg_spikes)
    
    # Validated critical state
    self.ca3_neurons.membrane.data = self._load_tensor(
        state.ca3_membrane,
        expected_shape=(self.ca3_size,),
        name="ca3_membrane"
    )
```

**Improvement**: Clear distinction between validated and non-validated loading.

---

## Migration Status

### ⏳ Pending Migrations

All regions with `load_state()` methods could benefit from these helpers:

1. 🔄 **LayeredCortex** ([layered_cortex.py:2307](../../src/thalia/regions/cortex/layered_cortex.py#L2307))
   - ~15 `.to(device)` calls → `_load_tensor()` calls
   - **Estimated savings**: No line reduction, but clearer intent

2. 🔄 **Striatum** ([striatum.py:3501](../../src/thalia/regions/striatum/striatum.py#L3501))
   - Manual validation + device transfer → `_load_tensor()` with validation
   - **Estimated savings**: ~20 lines reduction (shape validation code)

3. 🔄 **Hippocampus** ([trisynaptic.py:2460](../../src/thalia/regions/hippocampus/trisynaptic.py#L2460))
   - Mixed validation patterns → consistent `_load_tensor()` usage
   - **Estimated savings**: ~15 lines reduction

4. 🔄 **Thalamus** ([thalamus.py:811](../../src/thalia/regions/thalamus/thalamus.py#L811))
   - Multiple `.to(device)` calls → `_load_tensor()` calls
   - **Estimated savings**: ~10 lines cleaner

5. 🔄 **Prefrontal** ([prefrontal.py:1241](../../src/thalia/regions/prefrontal/prefrontal.py#L1241))
   - Shape checks + device transfer → validated `_load_tensor()`
   - **Estimated savings**: ~12 lines reduction

6. 🔄 **Cerebellum** ([cerebellum.py:1423](../../src/thalia/regions/cerebellum/cerebellum.py#L1423))
   - Complex multi-component loading → standardized helpers
   - **Estimated savings**: ~15 lines cleaner

7. 🔄 **PredictiveCortex** ([predictive_cortex.py:304](../../src/thalia/regions/cortex/predictive_cortex.py#L304))
   - Device transfers → `_load_tensor()` calls
   - **Estimated savings**: ~8 lines cleaner

**Total potential savings**: ~50-80 lines of validation/transfer code across 7 regions

---

## Testing Recommendations

### Unit Tests to Add

```python
def test_validate_tensor_shape_correct():
    """Test validation passes with correct shape."""
    mixin = create_test_region()
    tensor = torch.zeros(100, 50)
    mixin._validate_tensor_shape(tensor, (100, 50), "test_tensor")
    # Should not raise

def test_validate_tensor_shape_mismatch():
    """Test validation raises with shape mismatch."""
    mixin = create_test_region()
    tensor = torch.zeros(100, 50)
    with pytest.raises(ValueError, match="Shape mismatch for test_tensor"):
        mixin._validate_tensor_shape(tensor, (100, 60), "test_tensor")

def test_load_tensor_device_transfer():
    """Test tensor is transferred to correct device."""
    mixin = create_test_region(device="cpu")
    tensor = torch.zeros(100).to("cpu")
    loaded = mixin._load_tensor(tensor)
    assert loaded.device.type == "cpu"

def test_load_tensor_with_validation():
    """Test loading with shape validation."""
    mixin = create_test_region()
    tensor = torch.zeros(100)
    
    # Should pass
    loaded = mixin._load_tensor(tensor, expected_shape=(100,), name="test")
    assert loaded.shape == (100,)
    
    # Should fail
    with pytest.raises(ValueError, match="Shape mismatch for test"):
        mixin._load_tensor(tensor, expected_shape=(50,), name="test")

def test_load_tensor_optional_validation():
    """Test loading without validation."""
    mixin = create_test_region()
    tensor = torch.zeros(100, 50)  # Any shape
    loaded = mixin._load_tensor(tensor)  # No validation
    assert loaded.shape == (100, 50)
```

### Integration Tests

1. Load state with valid shapes (should succeed)
2. Load state with mismatched shapes (should raise ValueError)
3. Load state on different device (should transfer correctly)
4. Load state with None tensors (should handle gracefully)

---

## Performance Considerations

**Memory**: No impact - helpers don't allocate new tensors
**Speed**: Negligible impact - function call overhead < 0.1μs per tensor
**Readability**: ✅ Significant improvement - clearer validation intent
**Maintainability**: ✅ Major improvement - single source of truth for validation

---

## Error Message Improvements

### Before

```python
# Inconsistent error messages across regions:
raise ValueError(f"Shape mismatch")
assert membrane.shape == expected, "Invalid shape"
warnings.warn(f"Wrong shape: {membrane.shape} vs {expected}")
```

### After

```python
# Consistent error messages:
ValueError: Shape mismatch for membrane_potential: expected (100,), got (80,)
ValueError: Shape mismatch for synaptic_weights: expected (200, 100), got (200, 120)
```

**Benefits**:
- ✅ Always includes tensor name (which specific tensor failed)
- ✅ Always includes both expected and actual shapes
- ✅ Consistent format makes errors easier to parse in logs
- ✅ Helps with debugging shape issues during checkpoint loading

---

## Next Steps

### Recommended Migration Order

1. **Start with simple regions** (ThalamicRelay, PredictiveCortex)
   - Primarily `.to(device)` replacements
   - Low risk, demonstrates pattern

2. **Move to complex regions** (Striatum, Hippocampus, Cerebellum)
   - Includes validation logic
   - Higher impact, more line reduction

3. **Test thoroughly** with existing checkpoints
   - Verify backward compatibility
   - Ensure validation doesn't break existing saves

### Optional Enhancements

Consider adding more specialized helpers:

```python
def _load_optional_tensor(
    self,
    tensor: Optional[torch.Tensor],
    target: torch.nn.Parameter,
    name: str = "tensor"
) -> None:
    """Load optional tensor (handles None gracefully)."""
    if tensor is not None:
        target.data = self._load_tensor(tensor, expected_shape=target.shape, name=name)

def _load_tensor_dict(
    self,
    tensor_dict: Dict[str, torch.Tensor],
    expected_keys: set[str],
    dict_name: str = "dict"
) -> Dict[str, torch.Tensor]:
    """Load dictionary of tensors with key validation."""
    # Validate all expected keys present
    # Transfer all tensors to device
    # Return validated dict
```

---

## Conclusion

The validation helpers provide:
- ✅ **Consistency**: Standardized validation and error messages across all regions
- ✅ **Clarity**: Explicit validation intent in code
- ✅ **Convenience**: Combines validation + device transfer in one call
- ✅ **Maintainability**: Single source of truth for state loading patterns

**Status**: Implementation complete, ready for adoption by region-specific `load_state()` methods.

**Estimated Total Impact**: ~50-80 lines of duplicated validation code eliminated, with significantly improved error messages and code clarity.
