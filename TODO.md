# TODO

## 🔍 Obsolete Code Cleanup Report

### **Category 1: Backward Compatibility for Single Tensor Input** ✅ **COMPLETE**

**Status**: Removed from all 7 region files, input_routing.py, and tests updated

**Completed Actions:**
- ✅ Removed `Union[Dict[str, torch.Tensor], torch.Tensor]` → now `Dict[str, torch.Tensor]`
- ✅ Removed `isinstance(inputs, torch.Tensor)` conversion logic from InputRouter
- ✅ Removed unused `Union` imports from all region files
- ✅ Updated all docstrings to remove backward compatibility mentions
- ✅ Updated 11 tests in test_cortex_base.py to pass dict inputs
- ✅ Updated test documentation in region_test_base.py
- ✅ All tests passing (26/26)

**Files Modified:**
- src/thalia/utils/input_routing.py
- src/thalia/regions/thalamus/thalamus.py
- src/thalia/regions/striatum/striatum.py
- src/thalia/regions/prefrontal/prefrontal.py
- src/thalia/regions/hippocampus/trisynaptic.py
- src/thalia/regions/cortex/layered_cortex.py
- src/thalia/regions/cerebellum/cerebellum.py
- tests/unit/regions/test_cortex_base.py
- tests/utils/region_test_base.py

**Impact**: ~50 lines removed, API simplified and type-safe

---

### **Category 2: Curriculum Backward Compatibility Checks**

**Affected Files:**
1. stage_manager.py
   - Lines 2299-2309 (backward compatibility checking in evaluation)
   - Lines 2470-2590 (`_check_backward_compatibility()` method)
   - Lines 731 (stage task loader cache comment)
   - Line 33 (docstring mention)

2. stage_evaluation.py
   - Line 23 (docstring section header)
   - Lines 469, 499 (Stage -0.5 backward compat checks)
   - Lines 615, 632 (Stage 0 backward compat checks)
   - Lines 754, 777 (Stage 1 backward compat checks)
   - Lines 906, 929 (Stage 2 backward compat checks)
   - Lines 1105, 1124 (Stage 3 backward compat checks)

**Issue:** Extensive code for checking whether previous curriculum stages are maintained (catastrophic forgetting detection). Since there are no existing checkpoints, this entire regression testing system is premature.

**Recommended Action:**
- Remove `_check_backward_compatibility()` method entirely
- Remove all backward compatibility checks from `_evaluate_milestone()`
- Remove optional `*_datasets` parameters from stage evaluation functions (used for regression testing)
- Remove stage task loader cache
- Simplify evaluation functions to only check current stage performance

---

### **Category 3: Ablation Surgery Compatibility Shim**

**Affected File:**
ablation.py (lines 30-75)

**Issue:** The `ablate_pathway()` function contains extensive documentation and error handling for "v3.0 architecture" (AxonalProjection), but immediately raises `NotImplementedError` for the only pathway type currently in use.

**Current Code:**
```python
def ablate_pathway(...):
    """...extensive v3.0 documentation..."""
    # Check if pathway has learnable parameters
    if not has_learnable_params:
        raise NotImplementedError(
            f"Cannot ablate routing pathway '{pathway_name}'..."
        )
```

**Recommended Action:**
- Remove `ablate_pathway()` entirely or mark as deprecated
- Document recommended approach: use `lesion_region()` or manually zero weights
- If kept, drastically simplify documentation

---

### **Category 4: Placeholder Evaluation Functions**

**Affected File:**
stage_evaluation.py (lines 70-140)

**Issue:** Multiple health check functions that are all placeholders returning `True`:
- `check_firing_rates()` → "Placeholder for now, return True"
- `check_no_runaway_excitation()` → "Placeholder for now, return True"
- `check_bcm_convergence()` → "Placeholder for now, return True"
- `check_weight_saturation()` → "Placeholder for now, return True"
- `check_no_silent_regions()` → "Placeholder for now, return True"

**Recommended Action:**
- Either implement these properly using existing diagnostics (HealthMonitor, CriticalityMonitor)
- Or remove them and use the actual diagnostic systems directly
- Don't keep stub functions that always pass

---

### **Category 5: Commented-Out Legacy Code**

**Affected File:**
learning_config.py (lines 195-210)

**Issue:** Large commented-out example showing "Old style" vs "New style" config patterns.

**Recommended Action:**
- Remove commented examples (lines 195-210)
- The documentation explains the pattern well without commented code

---

### **Category 6: Documentation Mentions**

**Minor cleanup:** Several docstrings mention "backward compatibility" but this is just explaining the design, not actual legacy code:
- input_routing.py docstring
- dynamic_brain.py (line 585) - "backward compat" comment

**Recommended Action:**
- After removing actual backward compat code, update these docstrings to remove those mentions

---

## 📊 Summary Statistics

- ✅ **Category 1 COMPLETE**: 9 files modified, ~50 lines removed
- **6 region files** - Union types removed, docstrings updated
- **1 utility file** - backward compat conversion logic removed
- **2 test files** - updated to use dict inputs
- **2 curriculum files** with extensive (300+ lines) backward compatibility checking - TODO
- **1 surgery file** with unused ablation functionality - IN PROGRESS
- **5 placeholder functions** in evaluation module - TODO
- **1 config file** with commented legacy examples - TODO

**Estimated Remaining LOC Reduction:** ~450-650 lines

---

## 🎯 Cleanup Progress

1. ✅ **COMPLETE:** Remove single tensor input backward compatibility (Category 1)
   - Most pervasive across codebase
   - Simplifies API significantly
   - No functionality loss (all code already uses dict format)

2. 🔄 **IN PROGRESS:** Remove ablation surgery shim (Category 3)
   - Remove `ablate_pathway()` function
   - Update/remove tests

3. **TODO:** Remove curriculum backward compatibility checks (Category 2)
   - Large amount of code (~300 lines)
   - No checkpoints exist to test against
   - Can be reimplemented when actually needed

4. **TODO:** Replace placeholder functions (Category 4)
   - Either implement properly or remove

5. **TODO:** Clean up comments (Categories 5, 6)
   - Smaller impact but improves code clarity
