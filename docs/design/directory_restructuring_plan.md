# Directory Restructuring Migration Plan

**Date**: December 12, 2025  
**Status**: Planning  
**Impact**: High - affects all imports throughout codebase

## Overview

This plan restructures `src/thalia/` to improve organization, discoverability, and alignment with biological architecture principles. The current `core/` directory is bloated (37+ files), and pathways are scattered across multiple locations.

## Goals

1. **Reduce `core/` bloat** - Move to ~5-10 files (protocols, base classes, errors)
2. **Consolidate pathways** - Single `pathways/` directory
3. **Group neuromodulation** - All neuromodulator systems together
4. **Clarify learning** - Separate rules, homeostasis, modulation
5. **Improve discoverability** - Logical, biology-aligned structure

## Phase 1: Preparation (Low Risk)

### 1.1 Create New Directory Structure
Create empty directories for new organization:
```
src/thalia/
├── components/
│   ├── neurons/
│   ├── synapses/
│   └── coding/
├── neuromodulation/
│   └── systems/
├── pathways/
│   └── attention/
├── learning/
│   ├── rules/
│   ├── homeostasis/
│   └── modulation/
├── memory/
│   ├── consolidation/
│   └── replay/
├── coordination/
├── managers/
└── regulation/
```

### 1.2 Inventory All Import Dependencies
Run analysis to map all imports of files that will move:
```bash
# Files moving from core/
python scripts/analyze_imports.py core/neuron.py
python scripts/analyze_imports.py core/vta.py
python scripts/analyze_imports.py core/pathway_manager.py
# ... etc for all files being moved
```

### 1.3 Create Import Compatibility Layer
Add to `core/__init__.py` to maintain backward compatibility during migration:
```python
# Backward compatibility imports
from thalia.components.neurons.neuron import LIFNeuron, ConductanceLIFNeuron
from thalia.neuromodulation.systems.vta import VTA
# ... etc
```

## Phase 2: Move Component Files (Medium Risk)

### 2.1 Move Neuron-Related Files
**From**: `core/`  
**To**: `components/neurons/`  
**Files**:
- `neuron.py`
- `neuron_constants.py`
- `dendritic.py`

**Actions**:
1. Move files to `components/neurons/`
2. Update `__init__.py` in new location
3. Add backward-compatible imports to `core/__init__.py`
4. Run tests: `pytest tests/unit/test_neurons.py`

### 2.2 Move Synapse-Related Files
**From**: `core/`  
**To**: `components/synapses/`  
**Files**:
- `weight_init.py`
- `stp.py`
- `stp_presets.py`
- `traces.py`

**Actions**:
1. Move files to `components/synapses/`
2. Update `__init__.py` in new location
3. Add backward-compatible imports
4. Run tests: `pytest tests/unit/test_synapses.py tests/unit/test_weight_init.py`

### 2.3 Move Spike Coding Files
**From**: `core/`  
**To**: `components/coding/`  
**Files**:
- `spike_coding.py`
- `spike_utils.py`

**Actions**:
1. Move files to `components/coding/`
2. Update imports in files that use these
3. Run tests: `pytest tests/unit/test_spike_coding.py`

## Phase 3: Move Neuromodulation (Medium Risk)

### 3.1 Move Neuromodulator System Files
**From**: `core/`  
**To**: `neuromodulation/systems/`  
**Files**:
- `vta.py`
- `locus_coeruleus.py`
- `nucleus_basalis.py`

**Actions**:
1. Move files to `neuromodulation/systems/`
2. Update `brain.py` imports
3. Update any region files that directly import these
4. Run tests: `pytest tests/unit/test_neuromodulation.py`

### 3.2 Move Neuromodulator Management Files
**From**: `core/`  
**To**: `neuromodulation/`  
**Files**:
- `neuromodulator_manager.py` → `manager.py`
- `neuromodulator_homeostasis.py` → `homeostasis.py`
- `neuromodulator_mixin.py` → `mixin.py`

**Actions**:
1. Move and rename files
2. Update imports in `Brain` class
3. Update imports in all regions using neuromodulator mixin
4. Run integration tests: `pytest tests/integration/test_neuromodulation_integration.py`

## Phase 4: Consolidate Pathways (High Risk)

### 4.1 Move Base Pathway Files
**From**: `integration/`  
**To**: `pathways/`  
**Files**:
- `spiking_pathway.py`

**From**: `core/`  
**To**: `pathways/`  
**Files**:
- `pathway_protocol.py` → move to `core/protocols/`
- `pathway_manager.py` → `manager.py`

**Actions**:
1. Create `pathways/base.py` consolidating base pathway functionality
2. Move `spiking_pathway.py` to `pathways/`
3. Move `pathway_manager.py` to `pathways/manager.py`
4. Update all pathway imports throughout codebase
5. Run comprehensive pathway tests

### 4.2 Move Sensory Pathways and Remove sensory/ Directory
**From**: `sensory/`  
**To**: `pathways/`  
**Files**:
- `sensory/pathways.py` → `pathways/sensory_pathways.py`

**Actions**:
1. Move file to `pathways/sensory_pathways.py`
2. Update imports in sensory processing code
3. Remove empty `sensory/` directory
4. Run sensory tests

### 4.3 Move Attention Pathways
**From**: `integration/pathways/`  
**To**: `pathways/attention/`  
**Files**:
- `attention.py`
- `spiking_attention.py`
- `crossmodal_binding.py`

**Actions**:
1. Move files to `pathways/attention/`
2. Update imports in regions using attention
3. Run attention tests

### 4.4 Consolidate Pathway Manager Usage
**Critical**: Update all files that import `PathwayManager`:
- `core/brain.py`
- Region files that manage pathways
- Test files

## Phase 5: Reorganize Learning (Medium Risk)

### 5.1 Move Learning Rules
**From**: `learning/`  
**To**: `learning/rules/`  
**Files**:
- `bcm.py`
- `strategies.py`

**Actions**:
1. Move to `learning/rules/`
2. Update imports in regions using these rules
3. Run learning tests

### 5.2 Move Homeostasis Files
**From**: `learning/`  
**To**: `learning/homeostasis/`  
**Files**:
- `unified_homeostasis.py`
- `synaptic_homeostasis.py`
- `intrinsic_plasticity.py`
- `metabolic.py`

**From**: `core/`  
**To**: `learning/homeostasis/`  
**Files**:
- `homeostatic_regulation.py`

**Actions**:
1. Move files to `learning/homeostasis/`
2. Update imports in regions
3. Run homeostasis tests

### 5.3 Move Learning Modulation Files
**From**: `core/`  
**To**: `learning/modulation/`  
**Files**:
- `eligibility_utils.py`
- `predictive_coding.py`

**Actions**:
1. Move files
2. Update imports in striatum and other regions
3. Run learning modulation tests

## Phase 6: Reorganize Memory (Low Risk)

### 6.1 Move Consolidation Files
**From**: `memory/`  
**To**: `memory/consolidation/`  
**Files**:
- `consolidation.py`
- `advanced_consolidation.py`

**From**: `core/`  
**To**: `memory/consolidation/`  
**Files**:
- `consolidation_manager.py` → `manager.py`

**Actions**:
1. Move files
2. Update imports in hippocampus and brain
3. Run consolidation tests

### 6.2 Move Replay Files
**From**: `integration/pathways/`  
**To**: `memory/replay/`  
**Files**:
- `spiking_replay.py`

**Actions**:
1. Move file
2. Update imports in hippocampus
3. Run replay tests

## Phase 7: Extract Coordination & Management (Low Risk)

### 7.1 Move Coordination Files
**From**: `core/`  
**To**: `coordination/`  
**Files**:
- `oscillator.py`
- `trial_coordinator.py`
- `growth.py`

**Actions**:
1. Move files
2. Update imports in brain and regions
3. Run coordination tests

### 7.2 Move Manager Files
**From**: `core/`  
**To**: `managers/`  
**Files**:
- `component_registry.py`
- `base_manager.py`

**Actions**:
1. Move files
2. Update imports throughout codebase
3. Run registry tests

### 7.3 Move Regulation Files
**From**: `core/`  
**To**: `regulation/`  
**Files**:
- `homeostasis_constants.py`
- `learning_constants.py`
- `normalization.py`

**Actions**:
1. Move files
2. Update imports in learning and regions
3. Run regulation tests

## Phase 8: Reorganize Core Protocols (Medium Risk)

### 8.1 Move Protocol Files
**From**: `core/`  
**To**: `core/protocols/`  
**Files**:
- `component_protocol.py`
- `pathway_protocol.py`
- `protocols.py`

**Actions**:
1. Create `core/protocols/` directory
2. Move protocol files
3. Update imports throughout codebase (HIGH IMPACT)
4. Run all tests

### 8.2 Create Core Base Directory
Create `core/base/` to house all base classes and foundational configs.

### 8.3 Organize Core Base Classes
**From**: `core/`  
**To**: `core/base/`  
**Files**:
- `component_config.py` (foundational config base classes)

**From**: `regions/`  
**To**: `core/base/`  
**Files**:
- `regions/base.py` → `core/base/region.py` (copy, keep original for now)

**Actions**:
1. Move `component_config.py` to `core/base/`
2. Copy `regions/base.py` to `core/base/region.py`
3. Keep original for backward compatibility
4. Update imports throughout codebase

### 8.4 Move Utils to Dedicated Directory
**From**: `core/`  
**To**: `utils/`  
**Files**:
- `utils.py`

**Actions**:
1. Create `utils/` directory
2. Move `utils.py` 
3. Update imports in files that use utilities

### 8.5 Slim Down Core
**Remaining in `core/`**:
- `core/protocols/` (protocol definitions)
- `core/base/` (base classes and configs)
- `brain.py` (main entry point)
- `errors.py` (error definitions)
- `region_components.py` (keep for now, may split later)
- `__init__.py` (with backward-compatible imports)

## Phase 9: Update Mixins (Low Risk)

### 9.1 Consolidate Mixin Files
**Strategy**: Generic mixins in `mixins/`, domain mixins stay with domains but re-exported

**Move to `mixins/`** (generic, cross-cutting):
- `core/diagnostics_mixin.py` → `mixins/diagnostics_mixin.py`
- `core/mixins.py` → split into:
  - `mixins/device_mixin.py`
  - `mixins/configurable_mixin.py`
- `mixins/growth_mixin.py` (already there)

**Keep in domains** (domain-specific, re-export only):
- `neuromodulation/mixin.py` (NeuromodulatorMixin)
- `learning/strategy_mixin.py` (StrategyMixin)

**Actions**:
1. Split `core/mixins.py` into separate files in `mixins/`
2. Move `core/diagnostics_mixin.py` to `mixins/`
3. Create `mixins/__init__.py` with:
   - Direct exports of generic mixins
   - Re-exports of domain mixins for convenience
4. Update imports throughout codebase

## Phase 10: Update All Imports (High Risk)

### 10.1 Generate Import Mapping
Create `migration_import_map.json`:
```json
{
  "thalia.core.neuron": "thalia.components.neurons.neuron",
  "thalia.core.vta": "thalia.neuromodulation.systems.vta",
  "thalia.core.pathway_manager": "thalia.pathways.manager",
  // ... complete mapping
}
```

### 10.2 Automated Import Updates
Run script to update imports:
```bash
python scripts/migrate_imports.py --mapping migration_import_map.json --dry-run
# Review changes
python scripts/migrate_imports.py --mapping migration_import_map.json --apply
```

### 10.3 Manual Import Review
Review and fix complex imports:
- Circular import issues
- Conditional imports
- Dynamic imports
- `__init__.py` re-exports

## Phase 11: Update Tests (High Risk)

### 11.1 Update Unit Tests
For each moved module, update corresponding test imports:
```bash
# Example
tests/unit/test_neuron.py → update imports to thalia.components.neurons
tests/unit/test_vta.py → update imports to thalia.neuromodulation.systems
```

### 11.2 Update Integration Tests
Update integration tests that import from multiple moved modules:
- `tests/integration/test_brain_coordination.py`
- `tests/integration/test_pathway_integration.py`
- `tests/integration/test_curriculum_integration.py`

### 11.3 Update Test Fixtures
Update conftest.py files with new import paths.

## Phase 12: Update Documentation (Medium Risk)

### 12.1 Update Architecture Docs
Update documentation references:
- `docs/design/architecture.md`
- `docs/patterns/component-parity.md`
- `docs/patterns/state-management.md`
- `README.md`

### 12.2 Update Code Examples
Update all code examples in:
- `docs/GETTING_STARTED_CURRICULUM.md`
- Docstrings with example imports
- Jupyter notebooks

### 12.3 Update ADRs
Create new ADR documenting this restructuring:
- `docs/decisions/adr-012-directory-restructuring.md`

## Phase 13: Remove Backward Compatibility (Low Risk)

### 13.1 Remove Compatibility Imports
After 1-2 releases with deprecation warnings:
1. Remove backward-compatible imports from `core/__init__.py`
2. Remove old directory structures if empty
3. Add deprecation notices in release notes

### 13.2 Clean Up Empty Directories
Remove empty directories:
- `integration/pathways/` (if empty)
- `integration/` (if only had pathways/)
- `sensory/` (removed in Phase 4)

## Testing Strategy

### Per-Phase Testing
After each phase:
1. Run affected unit tests
2. Run full test suite
3. Test import statements manually
4. Check for circular imports

### Integration Testing
After phases 4, 8, and 10:
1. Full integration test suite
2. Manual testing of training scripts
3. Run curriculum training example
4. Test checkpoint loading/saving

### Regression Testing
Before Phase 13:
1. Full regression test suite
2. Compare outputs with pre-migration baseline
3. Performance benchmarks

## Rollback Strategy

### Per-Phase Rollback
1. Keep git commits atomic per phase
2. Tag each phase completion
3. Can rollback to any phase tag

### Emergency Rollback
If critical issues found:
1. Revert to pre-migration tag
2. Keep backward-compatible imports active
3. Fix issues in separate branch
4. Re-attempt migration

## Risk Mitigation

### High-Risk Items
1. **Pathway consolidation** (Phase 4) - many imports, critical functionality
2. **Protocol reorganization** (Phase 8) - affects entire codebase
3. **Import updates** (Phase 10) - can break everything if wrong

### Mitigation Strategies
1. **Feature branch**: Do all work in `feature/directory-restructuring`
2. **Backward compatibility**: Keep old imports working during transition
3. **Incremental testing**: Test after each phase
4. **Automated tools**: Use scripts for bulk import updates
5. **Code review**: Review each phase before merging

## Timeline Estimate

| Phase | Effort | Risk | Duration |
|-------|--------|------|----------|
| 1. Preparation | Low | Low | 2-4 hours |
| 2. Components | Medium | Medium | 4-6 hours |
| 3. Neuromodulation | Medium | Medium | 3-4 hours |
| 4. Pathways | High | High | 8-12 hours |
| 5. Learning | Medium | Medium | 4-6 hours |
| 6. Memory | Low | Low | 2-3 hours |
| 7. Coordination | Low | Low | 2-3 hours |
| 8. Core Protocols | High | Medium | 8-10 hours |
| 9. Mixins | Medium | Low | 3-4 hours |
| 10. Import Updates | Very High | High | 8-12 hours |
| 11. Test Updates | High | High | 6-8 hours |
| 12. Documentation | Medium | Medium | 4-6 hours |
| 13. Cleanup | Low | Low | 2-3 hours |
| **Total** | | | **55-80 hours** |

## Success Criteria

1. ✅ All tests pass with new structure
2. ✅ No circular import errors
3. ✅ All documentation updated
4. ✅ Backward compatibility maintained for 1 release
5. ✅ Training scripts work without modification
6. ✅ Checkpoint loading/saving works
7. ✅ Performance unchanged (benchmark comparison)
8. ✅ Code coverage maintained or improved

## Communication Plan

### Before Starting
- Create GitHub issue describing restructuring
- Post in team chat about upcoming changes
- Create migration branch

### During Migration
- Daily updates on progress
- Post completion of each phase
- Flag any blockers immediately

### After Completion
- Pull request with detailed description
- Migration guide for external contributors
- Release notes with import changes
- Deprecation timeline for old imports

## Automation Scripts Needed

1. **`scripts/analyze_imports.py`** - Map all imports of a file
2. **`scripts/migrate_imports.py`** - Bulk update import statements
3. **`scripts/check_circular_imports.py`** - Detect circular imports
4. **`scripts/validate_structure.py`** - Verify new structure is complete

## Notes

- Keep `core/brain.py` in core/ as it's the main entry point
- Some files may benefit from splitting during move (e.g., `protocols.py`, `core/mixins.py`)
- `component_config.py` moves to `core/base/` to clarify it's foundational infrastructure
- `region_components.py` kept as single file for now (can split later if needed)
- Watch for hidden dependencies in `__init__.py` files
- `sensory/` directory removed - consolidate into `pathways/`
- Generic mixins physically in `mixins/`, domain mixins re-exported from there

## Decisions on Open Questions

### 1. `utils.py` Location
**Decision**: Move to dedicated `utils/` directory  
**Rationale**: Better separation of concerns, utils are not core abstractions

### 2. `region_components.py` Splitting
**Decision**: Keep as single file for now  
**Rationale**: Can be split later if it becomes unwieldy, avoid over-engineering

### 3. `sensory/` Directory
**Decision**: Remove (consolidate into `pathways/`)  
**Rationale**: Only contains `pathways.py`, no other sensory-specific code to justify separate directory

### 4. Mixins Strategy
**Decision**: Re-export domain-specific mixins, keep generic mixins in `mixins/`  
**Rationale**:
- Generic cross-cutting mixins (`DeviceMixin`, `ConfigurableMixin`, `DiagnosticCollectorMixin`) physically live in `mixins/`
- Domain-specific mixins (`NeuromodulatorMixin` from `neuromodulation/`, `StrategyMixin` from `learning/`) stay with their domains but are re-exported from `mixins/__init__.py` for convenience
- Provides single import point while maintaining cohesion

**Implementation**:
```python
# mixins/__init__.py
from thalia.mixins.device_mixin import DeviceMixin  # Physical
from thalia.mixins.growth_mixin import GrowthMixin  # Physical
from thalia.neuromodulation.mixin import NeuromodulatorMixin  # Re-export
from thalia.learning.strategy_mixin import StrategyMixin  # Re-export
```

### 5. `component_config.py` Location
**Decision**: Move to `core/base/` (not back to `config/`)  
**Rationale**:
- Was deliberately moved from `config/` to break circular imports (CONFIG ↔ REGIONS)
- Contains base classes (`NeuralComponentConfig`, `PathwayConfig`) that are fundamental infrastructure, not runtime configuration
- Used by 20+ files as foundational abstractions
- `core/base/` clearly indicates these are base classes for component architecture
- Moving back to `config/` would recreate circular dependency

## Next Steps

1. Review and approve this plan
2. Create GitHub issue and project board
3. Set up feature branch
4. Create automation scripts
5. Begin Phase 1

---

**Plan Status**: ⏸️ Awaiting approval  
**Last Updated**: December 12, 2025  
**Estimated Start**: TBD  
**Estimated Completion**: TBD
