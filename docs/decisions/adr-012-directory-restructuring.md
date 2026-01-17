# ADR-012: Directory Restructuring for Domain-Based Organization

**Status**: Implemented
**Date**: 2025-12-12
**Author**: System Architect

## Context

The original `src/thalia/` directory structure had grown organically, resulting in:
- **Bloated `core/` directory** with 37+ disparate files
- **Unclear organization** mixing infrastructure, components, and domain logic
- **Difficult navigation** for new contributors
- **Scattered related functionality** across multiple directories
- **Maintenance challenges** due to unclear file locations

Example issues:
- Neuron models in `core/neuron.py` alongside brain coordinator (`core/brain.py`)
- Neuromodulation spread across `core/vta.py`, `core/lc.py`, etc.
- Learning rules split between `learning/` and scattered `core/` files
- Pathway files in `integration/pathways/` separate from pathway manager

## Decision

Reorganize the directory structure following **domain-driven design principles**:

### New Structure

```
src/thalia/
├── components/          # Neural components (primitives)
│   ├── neurons/        # LIF, ConductanceLIF, etc.
│   ├── synapses/       # Synapse models, STP
│   └── coding/         # Spike encoding/decoding
├── neuromodulation/     # Neuromodulator systems
│   ├── systems/        # VTA, LC, ACh sources
│   ├── manager.py      # NeuromodulatorManager
│   └── mixin.py        # NeuromodulatorMixin
├── pathways/            # Neural pathways & connections
│   ├── attention/      # Attention mechanisms
│   ├── spiking_pathway.py
│   └── manager.py
├── learning/            # Learning mechanisms
│   ├── rules/          # STDP, BCM, three-factor
│   ├── homeostasis/    # Homeostatic plasticity
│   └── eligibility/    # Eligibility traces
├── memory/              # Memory systems
│   └── consolidation/  # Memory consolidation
├── coordination/        # Brain coordination & timing
│   ├── oscillator.py
│   ├── growth.py
│   └── trial_coordinator.py
├── managers/            # Component managers
│   ├── component_registry.py
│   └── base_manager.py
├── regulation/          # Constants & normalization
│   ├── homeostasis_constants.py
│   ├── learning_constants.py
│   └── normalization.py
├── mixins/              # Reusable mixins
│   ├── device_mixin.py
│   ├── diagnostics_mixin.py
│   └── growth_mixin.py
├── utils/               # Utility functions
│   └── core_utils.py
└── core/                # Core infrastructure only
    ├── protocols/      # Protocol definitions
    ├── base/           # Base config classes
    ├── brain.py        # Main coordinator
    ├── diagnostics.py
    ├── errors.py
    └── region_components.py
```

### Key Principles

1. **Domain Cohesion**: Group related functionality together
2. **Clear Boundaries**: Each directory has single, clear responsibility
3. **Reduced Core**: Only true infrastructure remains in `core/`
4. **Discoverability**: Intuitive locations for all components
5. **Scalability**: Easy to add new components to appropriate domains

## Implementation

Executed in **10 phases** with **12 atomic commits**:

1. **Phase 1**: Create directory structure
2. **Phase 2**: Move components (neurons, synapses, coding) - 3 commits
3. **Phase 3**: Move neuromodulation - 2 commits
4. **Phase 4**: Consolidate pathways - 1 commit
5. **Phase 5**: Reorganize learning - 1 commit
6. **Phase 6**: Consolidate memory - 1 commit
7. **Phase 7**: Extract coordination & management - 1 commit
8. **Phase 8**: Reorganize protocols & base classes - 1 commit
9. **Phase 9**: Split and move mixins - 1 commit
10. **Phase 10**: Final core cleanup - 1 commit

Total changes:
- **37+ files moved** to new locations
- **145+ files updated** with new import paths
- **12 new `__init__.py`** files with verified exports
- **100% backward compatibility** maintained through re-exports

### Migration Strategy

- Used `git mv` to preserve file history
- Updated imports using automated regex replacements
- Created `__init__.py` with verified exports for each new directory
- Tested imports after each phase
- Maintained backward compatibility in `core/__init__.py`
- Each phase committed atomically for easy rollback

## Consequences

### Positive

1. **Improved Organization**
   - Clear domain boundaries
   - Intuitive file locations
   - Reduced `core/` from 37+ to 7 files

2. **Better Maintainability**
   - Related code grouped together
   - Easier to find and modify components
   - Clear dependency structure

3. **Enhanced Discoverability**
   - New contributors can navigate easily
   - Import paths reflect logical organization
   - Documentation aligns with code structure

4. **Scalability**
   - Easy to add new components
   - Clear patterns for organization
   - Room for future growth

5. **Backward Compatibility**
   - All existing code continues to work
   - Re-exports prevent breaking changes
   - Gradual migration path available

### Negative

1. **Import Path Changes**
   - All imports needed updating
   - Documentation needed updates
   - Some muscle memory adjustment required

2. **Initial Learning Curve**
   - Contributors need to learn new structure
   - Old references may be outdated
   - Migration guide needed

3. **Maintenance Overhead**
   - Backward compatibility layer needs eventual removal
   - Need to update new code to use new paths
   - Risk of inconsistent import styles during transition

### Migration Path

**Immediate (Current State)**:
- All imports work via new paths
- Backward compatibility maintained
- Documentation updated

**Short-term (1-2 releases)**:
- Add deprecation warnings for old import paths
- Update all internal code to new paths
- Communicate changes in release notes

**Long-term (Future release)**:
- Remove backward compatibility layer
- Clean up empty directories
- Finalize documentation

## Related

- **Component Parity** (`docs/patterns/component-parity.md`): Regions and pathways both benefit from clearer organization
- **State Management** (`docs/patterns/state-management.md`): Clearer organization aids understanding of state flow
- **ADR-008**: Neural component consolidation (preceded this restructuring)

## References

- Feature branch: `feature/directory-restructuring`
- Commits: 12 atomic commits (Phase 1-10)
- Testing: All imports verified after each phase
- Documentation: Updated in parallel with restructuring
