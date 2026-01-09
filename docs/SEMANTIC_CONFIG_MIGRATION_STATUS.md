# Semantic Config Migration Status

## Overview

Migration of test files from deprecated `n_input`/`n_output` parameters to semantic config patterns (e.g., `input_size`, `relay_size`, `purkinje_size`).

**Current Test Pass Rate: 65/295 region tests (22%)**
- Started: 74/75 tests passing (98.7%)
- Target: >95% pass rate with semantic configs

## Core Issue

The test base class `tests/utils/region_test_base.py` and its subclasses still reference `params["n_input"]` and `params["n_output"]` throughout all test methods. This causes cascading failures in all `*_base.py` test files that inherit from it.

## Semantic Config Patterns

### By Region

- **Thalamus**: `input_size`, `relay_size` (auto-computes `trn_size`)
- **Cortex**: `input_size`, `layer_sizes=[L4, L23, L5, L6a, L6b]`
- **Hippocampus**: `input_size`, `ca3_size`, `ca1_size`, `output_size`
- **Prefrontal**: `input_size`, `n_neurons`
- **Striatum**: `n_actions`, `neurons_per_action`, `input_sources={}`
- **Cerebellum**: `input_size`, `purkinje_size`

## Completed ✅

### Specialized Test Files (50+ tests passing)

- `tests/utils/test_helpers.py` - Added semantic config documentation
- `tests/unit/test_striatum_d1d2_delays.py` - 11/11 passing
- `tests/unit/test_purkinje_learning.py` - Configs migrated
- `tests/unit/regions/test_thalamus_stp.py` - 19/19 passing
- `tests/unit/regions/test_cerebellum_stp.py` - 13/13 passing
- `tests/unit/regions/test_cortex_gap_junctions.py` - 7/7 passing

### Partially Complete (base files need region_test_base.py fix)

- `tests/unit/regions/test_cerebellum_base.py` - 0/29 (base class issue)
- `tests/unit/regions/test_cortex_base.py` - 0/26 (base class issue + n_neurons)
- `tests/unit/regions/test_striatum_base.py` - Updated but not validated
- `tests/unit/regions/test_thalamus_base.py` - Updated but not validated
- `tests/unit/regions/test_prefrontal_base.py` - Batch updated
- `tests/unit/regions/test_hippocampus_base.py` - Batch updated
- `tests/unit/regions/test_hippocampus_state.py` - Batch updated

## Blocking Issues

### 1. region_test_base.py Incompatibility (CRITICAL)

**File**: `tests/utils/region_test_base.py`

**Problem**: All ~20 test methods expect old field names:
```python
# Current (BROKEN):
def test_initialization(self):
    params = self.get_default_params()
    region = self.create_region(**params)
    assert region.config.n_input == params["n_input"]  # ❌ Both fail
    assert region.config.n_output == params["n_output"]  # ❌ Both fail

# Needed (WORKING):
def test_initialization(self):
    params = self.get_default_params()
    region = self.create_region(**params)
    assert region.config.input_size == params["input_size"]  # ✅
    output_field = self._get_output_field_name()
    assert getattr(region.config, output_field) == params[output_field]  # ✅
```

**Solution**: Add helper methods for dynamic field name lookup:
```python
def _get_input_field_name(self):
    """Get semantic input field name (always 'input_size')"""
    return "input_size"

def _get_output_field_name(self):
    """Get semantic output field name based on region type"""
    region_type = type(self).__name__.replace("Test", "").lower()
    mapping = {
        "cerebellum": "purkinje_size",
        "striatum": "n_actions",
        "hippocampus": "output_size",
        "thalamus": "relay_size",
        "cortex": "output_size",
        "prefrontal": "n_neurons"
    }
    return mapping.get(region_type, "output_size")
```

**Impact**: Blocks ~150 tests in all `*_base.py` files

### 2. LayeredCortexConfig n_neurons Property

**File**: `tests/unit/regions/test_cortex_base.py`

**Problem**: `get_default_params()` tries to return `config.n_neurons` which doesn't exist

**Solution A**: Add `n_neurons` property to `LayeredCortexConfig`
**Solution B**: Remove `n_neurons` from test params dict (simpler)

**Impact**: Blocks 26 cortex tests

## Remaining Work

### Phase 1: Fix Infrastructure (CRITICAL)

1. **Update region_test_base.py** (~2 hours)
   - Add helper methods for semantic field names
   - Update all ~20 test methods to use helpers
   - Test with cerebellum_base.py, cortex_base.py, striatum_base.py
   - Expected: ~150 tests should pass after this fix

2. **Fix LayeredCortexConfig n_neurons** (~15 min)
   - Either add property or remove from test params
   - Expected: 26 cortex tests should pass

### Phase 2: Complete Base Files (~1 hour)

3. **Validate remaining *_base.py files**
   - test_striatum_base.py
   - test_thalamus_base.py
   - test_prefrontal_base.py
   - test_hippocampus_base.py
   - test_hippocampus_state.py
   - Expected: Should mostly work after Phase 1 fixes

### Phase 3: Specialized Test Files (~2 hours)

4. **Update remaining specialized tests** (~15 files)
   - test_cerebellum_enhanced.py
   - test_cerebellum_io_gap_junctions.py
   - test_striatum_fsi_gap_junctions.py
   - test_hippocampus_gap_junctions.py
   - test_phase_coding_emergence.py
   - test_predictive_cortex_base.py
   - Others in tests/unit/regions/
   - Method: PowerShell batch replacements with validation
   - Expected: ~50+ additional tests passing

### Phase 4: Unit & Integration Tests (~2 hours)

5. **Update unit test files** (~25 files)
   - tests/unit/*.py (non-region tests)
   - tests/integration/*.py
   - Special handling for builder.add_component() patterns
   - Expected: ~100+ additional tests passing

### Phase 5: Full Validation

6. **Run complete test suite**
   - Goal: >95% pass rate (~285/300 tests)
   - Validate no regressions from semantic migration
   - Update documentation with final results

## Estimated Timeline

- Phase 1 (Infrastructure): 2-3 hours - **BLOCKING**
- Phase 2 (Base files): 1 hour - **Depends on Phase 1**
- Phase 3 (Specialized): 2 hours
- Phase 4 (Unit/Integration): 2 hours
- Phase 5 (Validation): 30 min

**Total: ~7-8 hours of focused work**

## Success Criteria

- ✅ All test files use semantic config patterns
- ✅ No references to deprecated `n_input`/`n_output`
- ✅ >95% test pass rate (280+ tests passing)
- ✅ Test code quality improved (more readable, intent-revealing names)
- ✅ Documentation updated with semantic patterns

## Notes for Future Work

1. **PowerShell batch replacements are efficient** but can introduce syntax errors - always validate after
2. **Incremental testing is essential** - catch issues early before cascading failures
3. **Git commits provide rollback points** - commit after each validated batch
4. **region_test_base.py is shared infrastructure** - fixing it enables ~150 tests to pass
5. **Region-specific patterns vary** - cerebellum uses `purkinje_size`, striatum uses `n_actions`, etc.
