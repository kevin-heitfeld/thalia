# Thalia Documentation Index

**Last Updated**: December 21, 2025 (Post-Consolidation)

This is a comprehensive searchable index of all documentation in the Thalia project.

**Recent Changes** (Dec 21, 2025):
- ✅ Archived 9 completed implementation documents
- ✅ Fixed broken code references to non-existent docs
- ✅ Reduced total file count by 13% (67 → 58 files)

## Quick Navigation

- [Root Documentation](#root-documentation) — Getting started guides and quick references
- [API Reference](#api-reference-directory) — Auto-generated API documentation
- [Architecture](#architecture-directory) — System design and component overview
- [Design](#design-directory) — Technical specifications and implementation details
- [Patterns](#patterns-directory) — Implementation patterns and best practices
- [Decisions](#decisions-directory) — Architecture decision records (ADRs)
- [Archive](#archive-directory) — Historical documentation

---

## Root Documentation

Located in: `docs/`

### Getting Started & Quick References

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **README.md** | Documentation hub and navigation | 🟢 Current | Root README, copilot-instructions |
| **GETTING_STARTED_CURRICULUM.md** | Tutorial for curriculum training | 🟢 Current | Root README, CURRICULUM_QUICK_REFERENCE |
| **CURRICULUM_QUICK_REFERENCE.md** | API reference for training pipeline | 🟢 Current | Root README, copilot-instructions |
| **DATASETS_QUICK_REFERENCE.md** | Stage-specific datasets reference | 🟢 Current | Root README, copilot-instructions |
| **MONITORING_GUIDE.md** | Health checks and diagnostics | 🟢 Current | Root README, copilot-instructions |
| **AI_ASSISTANT_GUIDE.md** | Navigation guide for AI assistants | 🟢 Current | copilot-instructions |
| **MULTILINGUAL_DATASETS.md** | Multilingual support documentation | 🟢 Current | Root README |
| **DOCUMENTATION_VALIDATION.md** | Automated doc validation system | 🟢 Current | CI/CD processes |

### Safety & Quality Systems

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **CURRICULUM_SAFETY_SYSTEM.md** | Comprehensive safety system guide | 🟢 Current | Training code, safety docs |
| **SAFETY_QUICK_START.md** | Quick start guide for safety integration | 🟢 Current | Training tutorials |

### Implementation Details (Active)

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **NOISE_SYSTEM.md** | Noise scheduling system | 🟢 Current | Training code |

---

## API Reference Directory

Located in: `docs/api/`

> **Auto-generated documentation** - Always synchronized with code
> Run `python scripts/generate_api_docs.py` to update

| File | Purpose | Generated From | Last Updated |
|------|---------|----------------|--------------|
| **COMPONENT_CATALOG.md** | All registered regions and pathways | `@register_region`, `@register_pathway` decorators | Auto |
| **LEARNING_STRATEGIES_API.md** | Learning strategy factory functions | `create_*_strategy()` functions | Auto |
| **CONFIGURATION_REFERENCE.md** | Configuration dataclasses | `*Config` dataclass definitions | Auto |
| **DATASETS_REFERENCE.md** | Dataset classes and factory functions | `*Dataset` classes, `create_stage*()` functions | Auto |
| **DIAGNOSTICS_REFERENCE.md** | Diagnostic monitor classes | `*Monitor` classes in `diagnostics/` | Auto |
| **EXCEPTIONS_REFERENCE.md** | Custom exception hierarchy | Exception classes in `core/errors.py` | Auto |

**Benefits**:
- ✅ Always synchronized with codebase
- ✅ No manual maintenance required
- ✅ Catches undocumented components
- ✅ Consistent formatting

---

## Architecture Directory

Located in: `docs/architecture/`

High-level system architecture and component organization.

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **ARCHITECTURE_OVERVIEW.md** | Complete system overview | 🟢 Current | Root README, copilot-instructions |
| **CENTRALIZED_SYSTEMS.md** | Neuromodulators, oscillators, goals | 🟢 Current | TODO.md, training code |
| **SUPPORTING_COMPONENTS.md** | Infrastructure and utilities | 🟢 Current | Documentation links |
| **REFACTOR_EXPLICIT_AXONS_SYNAPSES.md** | Architecture design | 🟢 Current | Architecture decisions |
| **UNIFIED_GROWTH_API.md** | Growth method standardization | 🟢 Current | Growth implementations |
| **INDEX.md** | Searchable component reference | 🟢 Current | Navigation |
| **README.md** | Architecture directory overview | 🟢 Current | Root README |

### Archived Architecture Documents

| File | Purpose | Status | Archived Date |
|------|---------|--------|---------------|
| **L6_TRN_FEEDBACK_LOOP.md** | L6→TRN feedback implementation (✅ Complete) | 📦 Archived | Dec 21, 2025 |
| **BIOLOGICAL_ARCHITECTURE_SPEC.md** | Biological communication spec (✅ Complete) | 📦 Archived | Dec 21, 2025 |
| **OSCILLATION_EMERGENCE_ANALYSIS.md** | Oscillation emergence (✅ Implemented) | 📦 Archived | Dec 21, 2025 |

### Key Topics Covered
- **Architecture**: NeuralRegion base class (nn.Module + 4 mixins)
- **Learning Strategies**: Pluggable STDP, BCM, Hebbian, Three-factor, Error-corrective
- **Brain Regions**: Cortex, Hippocampus, Striatum, PFC, Cerebellum, Thalamus
- **Centralized Systems**: VTA, LC, NB, Oscillators, Goal Hierarchy
- **Pathways**: AxonalProjection (pure routing, no weights)
- **Data Flow**: Multi-source integration via Dict[str, Tensor]

---

## Design Directory

Located in: `docs/design/`

Detailed technical specifications for core systems.

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **curriculum_strategy.md** | Training stages and consolidation | 🟢 Current | copilot-instructions, stage_manager.py |
| **checkpoint_format.md** | Serialization and state persistence | 🟢 Current | GETTING_STARTED, stage_manager.py |
| **delayed_gratification.md** | TD(λ) and multi-step credit (Phases 1-3) | 🟢 Current | copilot-instructions |
| **circuit_modeling.md** | Circuit timing and D1/D2 delays | 🟢 Current | copilot-instructions |
| **neuron_models.md** | LIF and ConductanceLIF neurons | 🟢 Current | Pattern references |
| **parallel_execution.md** | Multi-core CPU performance | 🟢 Current | ADR-014 |
| **architecture.md** | System architecture details | 🟢 Current | state-management.md |
| **README.md** | Design directory overview | 🟢 Current | Documentation hub |

### Archived Design Documents

| File | Purpose | Status | Archived Date |
|------|---------|--------|---------------|
| **CLOCK_DRIVEN_OPTIMIZATIONS.md** | Clock-driven optimizations (Phase 1 complete) | 📦 Archived | Dec 21, 2025 |
| **trn_feedback_and_cerebellum_enhancement.md** | TRN enhancement plan (✅ Superseded) | 📦 Archived | Dec 21, 2025 |

### Key Topics Covered
- Curriculum stages (sensorimotor → grammar → reading)
- TD(λ) learning and Dyna planning
- Checkpoint format (PyTorch + optional binary)
- Circuit delays in D1/D2 pathways
- Parallel event-driven execution

---

## Patterns Directory

Located in: `docs/patterns/`

Common implementation patterns and best practices.

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **learning-strategies.md** | Comprehensive learning strategy guide | 🟢 Current | copilot-instructions, code comments |
| **component-parity.md** | Regions and pathways consistency | 🟢 Current | copilot-instructions, base.py |
| **state-management.md** | When to use RegionState vs attributes | 🟢 Current | copilot-instructions, neuron_models.md |
| **mixins.md** | Available mixins and usage | 🟢 Current | copilot-instructions, region files |
| **configuration.md** | Config hierarchy and parameters | 🟢 Current | mixins.md |
| **port-based-routing.md** | Multi-port pathway connections | 🟢 Current | Architecture docs |
| **README.md** | Patterns directory overview | 🟢 Current | Documentation hub |

### Key Topics Covered
- **Learning Strategy Pattern**: Pluggable strategies (Hebbian, STDP, BCM, Three-factor, Error-corrective)
- **Strategy Registry**: Dynamic discovery and creation
- **Region-Specific Factories**: `create_cortex_strategy()`, `create_striatum_strategy()`, etc.
- **Composite Strategies**: Combining multiple learning rules (e.g., STDP + BCM)
- **State Management**: Dataclass pattern for neural state
- **Mixin Composition**: 4 mixins (Neuromodulator, Growth, Resettable, Diagnostics)
- **Component Parity**: NeuralRegion architecture
| **component-parity.md** | Component design patterns | 🟢 Current | surgery/__init__.py |
| **component-interface-enforcement.md** | Protocol enforcement guide | 🟢 Current | base.py, component.py |
| **component-standardization.md** | Component standardization patterns | 🟢 Current | Pattern references |
| **README.md** | Patterns directory overview | 🟢 Current | Root README |

### Key Topics Covered
- Learning strategies: STDP, BCM, Hebbian, three-factor
- Component protocols and parity
- State management decision criteria
- Mixin patterns: Diagnostics, Growth, Neuromodulation
- Configuration best practices

---

## Decisions Directory

Located in: `docs/decisions/`

Architecture Decision Records (ADRs) documenting key technical choices.

| ADR | Title | Status | Referenced In |
|-----|-------|--------|---------------|
| **adr-001-simulation-backend.md** | Use PyTorch with GPU acceleration | Accepted | Root README, docs README |
| **adr-002-numeric-precision.md** | Mixed precision with float32 | Accepted | Root README |
| **adr-003-clock-driven.md** | Fixed timestep simulation | Accepted | Root README, neuron_models.md |
| **adr-004-bool-spikes.md** | Use bool tensors for spikes | Accepted | decisions/README |
| **adr-005-no-batch-dimension.md** | Single-brain architecture | Accepted | decisions/README |
| **adr-006-temporal-coding.md** | Temporal/latency coding for sensory | Accepted | component-parity.md |
| **adr-007-pytorch-consistency.md** | Standard forward() convention | Accepted | base.py, component-parity.md |
| **adr-008-neural-component-consolidation.md** | Unified component protocol | Accepted | component-interface-enforcement.md |
| **adr-009-pathway-neuron-consistency.md** | Pathways inherit NeuralComponent | Accepted | decisions/README |
| **adr-010-region-axonal-delays.md** | Regions handle delays | Accepted | decisions/README |
| **adr-011-large-file-justification.md** | Biological circuit integrity > file size | Accepted | Region files, review prompt |
| **adr-012-directory-restructuring.md** | Domain-based organization | Accepted | decisions/README |
| **adr-013-explicit-pathway-projections.md** | All dimensional transforms via pathways | Accepted | decisions/README |
| **adr-014-distributed-computation.md** | Multi-core CPU support | Superseded | decisions/README (Event-driven removed) |
| **README.md** | ADRs overview | 🟢 Current | Root README |

### Key Topics Covered
- Simulation backend and performance
- Biological plausibility constraints
- Component design principles
- Architecture patterns

---

## Documentation by Topic

### Training & Curriculum
- `GETTING_STARTED_CURRICULUM.md` — Tutorial
- `CURRICULUM_QUICK_REFERENCE.md` — API reference
- `design/curriculum_strategy.md` — Strategy and stages
- `design/checkpoint_format.md` — State persistence

### Datasets
- `DATASETS_QUICK_REFERENCE.md` — All datasets
- `MULTILINGUAL_DATASETS.md` — Multilingual support

### Learning Rules
- `patterns/learning-strategies.md` — All learning strategies
- `design/delayed_gratification.md` — TD(λ) and credit assignment

### System Architecture
- `architecture/ARCHITECTURE_OVERVIEW.md` — Complete overview
- `architecture/CENTRALIZED_SYSTEMS.md` — Global coordination
- `design/architecture.md` — Technical details

### Implementation Patterns
- `patterns/component-parity.md` — Regions and pathways
- `patterns/state-management.md` — State handling
- `patterns/mixins.md` — Mixin patterns
- `patterns/configuration.md` — Configuration hierarchy

### Monitoring & Diagnostics
- `MONITORING_GUIDE.md` — Health checks and visualization
- `architecture/SUPPORTING_COMPONENTS.md` — Diagnostics infrastructure

### Circuit Modeling
- `design/circuit_modeling.md` — D1/D2 delays
- `design/neuron_models.md` — Neuron implementations

---

## Archive Directory

Located in: `docs/archive/`

Historical documentation for completed implementations and superseded designs.

### Recent Additions (December 21, 2025)

**Implementation Summaries** (Completed work):
- `NOISE_IMPLEMENTATION_SUMMARY.md` — Noise system implementation (now in NOISE_SYSTEM.md)
- `SPILLOVER_IMPLEMENTATION.md` — Spillover transmission complete
- `oscillator_pathology_detection.md` — Oscillator health monitoring complete

**Architecture** (Completed implementations):
- `architecture/L6_TRN_FEEDBACK_LOOP.md` — L6→TRN feedback loop (✅ Complete Dec 20, 2025)
- `architecture/BIOLOGICAL_ARCHITECTURE_SPEC.md` — Biological communication spec (✅ Complete)
- `architecture/OSCILLATION_EMERGENCE_ANALYSIS.md` — Oscillation emergence (✅ Implemented Dec 20, 2025)

**Design** (Superseded or phase-complete):
- `design/CLOCK_DRIVEN_OPTIMIZATIONS.md` — Clock-driven optimizations (Phase 1 complete)
- `design/trn_feedback_and_cerebellum_enhancement.md` — TRN enhancement plan (superseded by L6_TRN_FEEDBACK_LOOP.md)

**Reviews** (Completed sessions):
- `reviews/architecture-review-2025-12-20.md` — Architecture review session

### Previous Archive Contents

- `ablation_results.md` — Ablation study results
- `CURRICULUM_VALIDATION_SUMMARY.md` — Curriculum validation findings
- `PLANNING-v1.md` — Original planning document
- Various subdirectories: `architecture/`, `design/`, `patterns/`, `reviews/`

---

## Search Tips

Use this index to find documentation by:
- **File name**: Search the tables above
- **Topic**: Use the "Documentation by Topic" section
- **Reference**: Check "Referenced In" column
- **Status**: Filter by status indicators (🟢 Current, 📦 Archived)

For full-text search across all documentation:
```powershell
# Windows PowerShell
Get-ChildItem -Path "docs" -Recurse -Include *.md | Select-String -Pattern "your_search_term"
```

---

**Maintained by**: Documentation consolidation process
**Next Review**: As needed when significant changes occur
