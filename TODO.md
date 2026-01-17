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

### **Category 3: Ablation Surgery Compatibility Shim** ✅ **COMPLETE**

**Status**: Removed ablate_pathway() and restore_pathway() functions, replaced with documentation

**Completed Actions:**
- ✅ Removed `ablate_pathway()` function (lines 26-84)
- ✅ Removed `restore_pathway()` function (lines 87-115)
- ✅ Removed helper functions: `_save_ablation_state()`
- ✅ Kept `_get_pathway()` helper (may be used elsewhere)
- ✅ Removed 3 tests from test_surgery.py
- ✅ Updated surgery/__init__.py to remove exports
- ✅ Replaced module with documentation pointing to alternatives
- ✅ All surgery tests passing (9/9)

**Files Modified:**
- src/thalia/surgery/ablation.py (now documentation-only)
- src/thalia/surgery/__init__.py (removed imports/exports)
- tests/unit/test_surgery.py (removed 3 tests)

**Impact**: ~150 lines removed, tests simplified

**Alternatives Documented:**
1. `lesion_region(brain, "source_name")` - silence source region
2. `brain.components["target"].synaptic_weights["source"].zero_()` - zero weights at target
3. `del brain.connections[("source", "target")]` - remove pathway from connections

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

### **Category 5: Commented-Out Legacy Code** ✅ **COMPLETE**

**Status**: Removed commented-out backward compatibility example

**Completed Actions:**
- ✅ Removed commented "Old style" vs "New style" config example (lines 195-210)
- ✅ Cleaned up learning_config.py example section

**Files Modified:**
- src/thalia/config/learning_config.py

**Impact**: ~15 lines removed

---

### **Category 6: Documentation Mentions** ✅ **COMPLETE**

**Status**: Removed "backward compatibility" mentions from documentation

**Completed Actions:**
- ✅ Updated dynamic_brain.py forward() docstring (removed "backward compat" label)
- ✅ Broadcast mode documentation now neutral (no longer labeled as compatibility feature)

**Files Modified:**
- src/thalia/core/dynamic_brain.py

**Impact**: Documentation clarity improved

---

## 📊 Summary Statistics

- ✅ **Category 1 COMPLETE**: 9 files modified, ~50 lines removed
- ✅ **Category 3 COMPLETE**: 3 files modified, ~150 lines removed
- ✅ **Category 5 COMPLETE**: 1 file modified, ~15 lines removed
- ✅ **Category 6 COMPLETE**: 1 file modified, documentation improved
- **Category 2 TODO**: 2 curriculum files with extensive (300+ lines) backward compatibility checking
- **Category 4 TODO**: 5 placeholder functions in evaluation module

**Total LOC Removed So Far:** ~215 lines
**Estimated Remaining LOC Reduction:** ~350-450 lines

---

## 🎯 Cleanup Progress

1. ✅ **COMPLETE:** Remove single tensor input backward compatibility (Category 1)
   - Most pervasive across codebase
   - Simplified API with type-safe Dict[str, torch.Tensor] signatures
   - 9 files modified, ~50 lines removed

2. ✅ **COMPLETE:** Remove ablation surgery shim (Category 3)
   - Removed ablate_pathway() and restore_pathway() functions
   - Replaced with documentation pointing to alternatives
   - 3 files modified, ~150 lines removed

3. ✅ **COMPLETE:** Clean up commented legacy code (Category 5)
   - Removed commented "Old style" vs "New style" config example
   - 1 file modified, ~15 lines removed

4. ✅ **COMPLETE:** Update documentation mentions (Category 6)
   - Removed "backward compat" labels from docstrings
   - 1 file modified, documentation improved

5. **TODO:** Remove curriculum backward compatibility checks (Category 2)
   - Large amount of code (~300 lines)
   - No checkpoints exist to test against
   - Can be reimplemented when actually needed

6. **TODO:** Replace placeholder functions (Category 4)
   - 5 placeholder health check functions
   - Either implement properly or remove
