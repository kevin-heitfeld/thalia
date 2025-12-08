# Test Fix Plan - 107 Failures

**Status**: December 8, 2025 - Before Curriculum Learning Implementation

## Summary
- **Total Tests**: 598
- **Passing**: 453 (76%)
- **Failing**: 107 (18%)  
- **Skipped**: 38 (6%)

## Issue Categories

### 1. ADR-005 Batch Dimension Violations (~70 failures)
**Problem**: Tests use 2D inputs `torch.randn(1, n_neurons)` but code expects 1D `torch.randn(n_neurons)` per ADR-005.

**Affected Test Files**:
- `tests/ablation/` (9 failures)
- `tests/benchmark/` (6 failures)
- `tests/integration/test_cortex_with_robustness.py` (7 failures)
- `tests/unit/test_checkpoint_io.py` (2 failures)
- `tests/unit/test_checkpoint_state.py` (5 failures)
- `tests/unit/test_core.py` (8 failures - dendritic)
- `tests/unit/test_error_handling.py` (13 failures)
- `tests/unit/test_fixtures.py` (1 failure)
- `tests/unit/test_integration_pathways.py` (5 failures)
- `tests/unit/test_predictive_attention.py` (4 failures)
- `tests/unit/test_properties.py` (11 failures)
- `tests/unit/test_validation.py` (14 failures)

**Fix**: Update test fixtures to use 1D inputs

### 2. Component Protocol Compliance (3 failures)
**Problem**: `BrainRegion` missing methods from `BrainComponent` protocol:
- `get_diagnostics()` - exists in pathways but not regions
- `check_health()` - exists in regions but not pathways

**Files to Fix**:
- `src/thalia/regions/base.py` - Add `get_diagnostics()` method
- `src/thalia/core/pathway_protocol.py` - Add `check_health()` method

### 3. SpikeDecoder Abstract Class (10 failures)
**Problem**: Tests instantiate `SpikeDecoder` directly, but it's abstract with missing `decode()` method.

**Affected**: `tests/unit/test_language.py` (10 failures)

**Fix Options**:
1. Create concrete `RateSpikeDecoder` implementation
2. Make tests use Mock or create test implementation
3. Make `decode()` non-abstract with default implementation

### 4. GrowthEvent API Mismatch (2 failures)
**Problem**: `GrowthEvent` constructor changed, tests use old `region_name` kwarg.

**Files**: `tests/unit/test_growth.py`

**Fix**: Update test to match new GrowthEvent signature

### 5. Dendritic Neuron Issues (8 failures)
**Problem**: Multiple issues in dendritic computation:
- `.item()` called on multi-element tensors
- Index errors for 1D tensors
- Weight shape mismatches

**Files**: `tests/unit/test_core.py`

**Fix**: Review dendritic neuron implementation for 1D compatibility

### 6. Pathway Zero Output (5 failures)
**Problem**: Attention and replay pathways return all zeros.

**Files**: `tests/unit/test_integration_pathways.py`

**Possible Causes**:
- Insufficient weight initialization
- Missing activation
- Incorrect spike processing

### 7. Growth Manager Issues (2 failures)
**Problem**:
- `neuron_count` mismatch (320 vs 32)
- Possible issue with layer counting vs total neurons

**Files**: `tests/unit/test_growth.py`

### 8. Test Infrastructure Issues (1 failure)
**Problem**: Custom region registration test fails due to missing abstract methods.

**Files**: `tests/unit/test_region_factory.py`

## Priority Fix Order

### Phase 1: Critical Infrastructure (before curriculum)
1. ✅ Component protocol compliance (`get_diagnostics()`, `check_health()`)
2. ✅ GrowthEvent API fix
3. ✅ SpikeDecoder abstraction fix

### Phase 2: Test Fixtures (can parallelize)
4. Create ADR-005 compliant test utilities
5. Update all test files to use 1D inputs

### Phase 3: Implementation Bugs
6. Fix dendritic neuron shape handling
7. Debug pathway zero output issue
8. Fix growth manager neuron counting

## Immediate Action

**Before starting curriculum learning**, we should:
1. Fix Protocol compliance (10 min)
2. Fix GrowthEvent API (5 min)  
3. Fix SpikeDecoder (10 min)
4. Create test utility for 1D inputs (15 min)
5. Re-run tests and verify critical paths work

**Then** proceed with curriculum implementation while tests are being fixed in parallel.

## Notes

- Most failures are test infrastructure issues, not core logic bugs
- The 453 passing tests cover core functionality
- ADR-005 compliance is important but doesn't block curriculum work
- Can proceed with curriculum once critical protocol issues are fixed
