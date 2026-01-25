# Documentation Consolidation Action Plan

**Date**: January 25, 2026
**Current File Count**: 113 files
**Target File Count**: 73-85 files (25-35% reduction)
**Files to Process**: 28-40 files

---

## Phase 1: Archive Completed Implementation Summaries

### Files to Archive (10 files)

**Phase/Progress Documents** (Move to `docs/archive/design/`):

1. ✅ `docs/design/phase0-cerebellum-stp-progress.md`
   - **Status**: Phase 0 Complete ✅
   - **Reason**: Implementation complete, tests passing
   - **Action**: ARCHIVE to `docs/archive/design/phase0-cerebellum-stp-progress.md`

2. ✅ `docs/design/phase1-region-state-foundation.md`
   - **Status**: Phase 1 Complete ✅ (December 21, 2025)
   - **Reason**: Implementation complete, 24/24 tests passing
   - **Action**: ARCHIVE to `docs/archive/design/phase1-region-state-foundation.md`

3. ✅ `docs/design/state-management-refactoring-plan.md`
   - **Status**: COMPLETED (Phase 5 deferred, all other phases done)
   - **Reason**: 190/190 tests passing, ~27 hours of work documented
   - **Action**: ARCHIVE to `docs/archive/design/state-management-refactoring-plan.md`
   - **Note**: Keep reference in `docs/patterns/state-management.md`

**Implementation Summaries** (Already in archive, verify completeness):

4. ✅ `docs/archive/NOISE_IMPLEMENTATION_SUMMARY.md` - Already archived
5. ✅ `docs/archive/SPILLOVER_IMPLEMENTATION.md` - Already archived
6. ✅ `docs/archive/oscillator_pathology_detection.md` - Already archived

**Active Design Documents to Keep**:
- ✅ `docs/design/gap_junction_implementation.md` - Complete but good reference
- ✅ `docs/design/stp-biological-requirements.md` - Complete but good reference
- ✅ `docs/design/multi_region_biological_accuracy.md` - Review document, keep
- ✅ `docs/design/striatum_biological_accuracy_improvements.md` - Investigation, keep

---

## Phase 2: Consolidate Architecture Documents

### Files to Consolidate (5 files → 2 files)

7. ✅ **MERGE**: `docs/architecture/SEMANTIC_CONFIG_MIGRATION_STATUS.md` + `docs/architecture/BRAIN_CREATION_ARCHITECTURE_REVIEW.md`
   - **Reason**: Both are configuration/creation-related progress tracking
   - **Action**: MERGE into `docs/architecture/CONFIGURATION_MIGRATION_STATUS.md`
   - **Content**: Combine semantic config progress with brain creation review
   - **Delete**: Both original files after merge

**Architecture Files to Keep**:
- ✅ `docs/architecture/ARCHITECTURE_OVERVIEW.md` - Referenced in copilot-instructions
- ✅ `docs/architecture/CENTRALIZED_SYSTEMS.md` - Referenced in code
- ✅ `docs/architecture/SUPPORTING_COMPONENTS.md` - Referenced in docs
- ✅ `docs/architecture/UNIFIED_GROWTH_API.md` - Referenced in code
- ✅ `docs/architecture/INDEX.md` - Navigation
- ✅ `docs/architecture/README.md` - Navigation

**Already Archived**:
- ✅ `docs/archive/architecture/L6_TRN_FEEDBACK_LOOP.md`
- ✅ `docs/archive/architecture/BIOLOGICAL_ARCHITECTURE_SPEC.md`
- ✅ `docs/archive/architecture/OSCILLATION_EMERGENCE_ANALYSIS.md`
- ⚠️ `docs/archive/architecture/REFACTOR_EXPLICIT_AXONS_SYNAPSES.md` - Check if should be active

---

## Phase 3: Clean Up Decisions Directory

### Files to Review (2 files)

8. ✅ **REVIEW**: `docs/decisions/striatum-multi-source-architecture.md`
   - **Status**: Implementation complete
   - **Referenced**: In test skips (test_port_based_routing.py)
   - **Action**: KEEP (referenced in code)

9. ✅ **REVIEW**: `docs/decisions/striatum-multi-source-implementation-plan.md`
   - **Status**: Likely superseded by architecture doc
   - **Action**: Check if redundant with architecture doc, if so ARCHIVE

**Decisions Files to Keep**:
- All ADRs (adr-001 through adr-014) - Architecture decisions are permanent records
- `docs/decisions/README.md` - Navigation

---

## Phase 4: Clean Up Patterns Directory

### No Changes Needed

All patterns files are:
- Referenced in copilot-instructions.md or code
- Active documentation of current patterns
- Non-redundant

**Keep all 11 files in patterns/**

---

## Phase 5: Verify API Directory

### No Changes Needed

All files in `docs/api/` are:
- Auto-generated (should never be manually edited)
- Referenced in DOCUMENTATION_INDEX.md
- Part of automated documentation system

**Keep all 20 files in api/** - DO NOT MODIFY

---

## Phase 6: Clean Up Papers Directory

### Files to Review (3 files)

Papers are research references, generally keep unless clearly unused.

10. ✅ **VERIFY**: `docs/papers/extraction/*.txt`
    - **Action**: Keep (research extraction notes)

11. ✅ **VERIFY**: `docs/papers/pdf/*.pdf`
    - **Action**: Keep (reference papers)

---

## Phase 7: Archive Root-Level Status Documents

### Files to Archive (2 files)

12. ⚠️ **REVIEW**: `docs/DOCUMENTATION_VALIDATION.md`
    - **Status**: Describes validation system
    - **Referenced**: In DOCUMENTATION_INDEX as "automated doc validation system"
    - **Action**: Keep if validation script still exists, otherwise ARCHIVE

13. ⚠️ **REVIEW**: `docs/REGRESSION_TESTING.md`
    - **Status**: Testing documentation
    - **Referenced**: Unknown
    - **Action**: Check if still relevant or superseded by tests/WRITING_TESTS.md

---

## Phase 8: Verify Archive is Complete

### Already Archived (Review for completeness)

**Archive Root**:
- ✅ `ablation_results.md`
- ✅ `CURRICULUM_VALIDATION_SUMMARY.md`
- ✅ `NOISE_IMPLEMENTATION_SUMMARY.md`
- ✅ `oscillator_pathology_detection.md`
- ✅ `PLANNING-v1.md`
- ✅ `SPILLOVER_IMPLEMENTATION.md`
- ✅ `README.md`

**Archive Subdirectories**:
- ✅ `architecture/` (4 files)
- ✅ `design/` (2 files)
- ✅ `reviews/` (1 file)

---

## Summary of Actions

### Archive (Move to docs/archive/)
1. phase0-cerebellum-stp-progress.md → archive/design/
2. phase1-region-state-foundation.md → archive/design/
3. state-management-refactoring-plan.md → archive/design/
4. striatum-multi-source-implementation-plan.md → archive/decisions/ (if redundant)

### Merge (Consolidate related docs)
5. SEMANTIC_CONFIG_MIGRATION_STATUS.md + BRAIN_CREATION_ARCHITECTURE_REVIEW.md
   → CONFIGURATION_MIGRATION_STATUS.md

### Delete (After merge)
6. SEMANTIC_CONFIG_MIGRATION_STATUS.md (after merge)
7. BRAIN_CREATION_ARCHITECTURE_REVIEW.md (after merge)

### Review/Verify (Check relevance)
8. DOCUMENTATION_VALIDATION.md (check if script exists)
9. REGRESSION_TESTING.md (check if superseded)
10. archive/architecture/REFACTOR_EXPLICIT_AXONS_SYNAPSES.md (should it be active?)

### Preserve (No changes)
- All ADRs (permanent records)
- All patterns/ files (active documentation)
- All api/ files (auto-generated)
- All quick references (actively used)
- papers/ directory (research references)

---

## Expected Impact

**Before**: 113 files
**After**: ~95-100 files (13-18 file reduction, ~12-16%)

**Note**: This is lower than the 25-35% target because:
1. Recent consolidation (Dec 21, 2025) already reduced files by 13%
2. Many docs are actively referenced in code
3. API docs are auto-generated and should not be removed
4. ADRs are permanent records

**Alternative to reach 25-35%**:
- Could consolidate design documents more aggressively
- Could merge some patterns documents (but risk losing granularity)
- Could remove papers/pdfs (but lose research context)

**Recommendation**: Proceed with conservative consolidation (12-16%) since project was recently cleaned up.

---

## Next Steps

1. Verify references to files marked for archive
2. Create merge document for configuration docs
3. Execute archive moves
4. Update DOCUMENTATION_INDEX.md
5. Update copilot-instructions.md if any referenced files change
6. Verify all links functional
7. Run validation: `python scripts/validate_docs.py`

