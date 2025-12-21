# Thalia Documentation Index

**Last Updated**: December 21, 2025
**Total Active Files**: 38 documentation files

This is a comprehensive searchable index of all documentation in the Thalia project.

## Quick Navigation

- [Root Documentation](#root-documentation) — Getting started guides and quick references
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
| **MULTILINGUAL_DATASETS.md** | Multilingual support documentation | 🟢 Current | Root README |

---

## Architecture Directory

Located in: `docs/architecture/`

High-level system architecture and component organization.

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **ARCHITECTURE_OVERVIEW.md** | Complete system overview (459 lines) | 🟢 Current | Root README, copilot-instructions |
| **CENTRALIZED_SYSTEMS.md** | Neuromodulators, oscillators, goals | 🟢 Current | TODO.md, training code |
| **SUPPORTING_COMPONENTS.md** | Infrastructure and utilities | 🟢 Current | Documentation links |
| **HIERARCHICAL_GOALS_COMPLETE.md** | Goal hierarchy implementation | 🟢 Current | design-docs-update |
| **GOAL_HIERARCHY_IMPLEMENTATION_SUMMARY.md** | Goal implementation summary | 🟢 Current | Internal reference |
| **INDEX.md** | Searchable component reference | 🟢 Current | Navigation |
| **README.md** | Architecture directory overview | 🟢 Current | Root README |

### Key Topics Covered
- 5-level complexity hierarchy (Primitives → Integration)
- Brain regions: Cortex, Hippocampus, Striatum, PFC, Cerebellum, Thalamus
- Centralized systems: VTA, LC, NB, Oscillators
- Pathways and data flow

---

## Design Directory

Located in: `docs/design/`

Detailed technical specifications for core systems.

| File | Purpose | Status | Referenced In |
|------|---------|--------|---------------|
| **curriculum_strategy.md** | Training stages and consolidation | 🟢 Current | copilot-instructions, stage_manager.py |
| **checkpoint_format.md** | Serialization and state persistence | 🟢 Current | GETTING_STARTED, stage_manager.py |
| **checkpoint_growth_compatibility.md** | Checkpoint growth strategy | 🟢 Current | CONTRIBUTING.md |
| **delayed_gratification.md** | TD(λ) and multi-step credit (Phases 1-3) | 🟢 Current | copilot-instructions |
| **circuit_modeling.md** | Circuit timing and D1/D2 delays | 🟢 Current | copilot-instructions |
| **neuron_models.md** | LIF and ConductanceLIF neurons | 🟢 Current | Pattern references |
| **parallel_execution.md** | Multi-core CPU performance | 🟢 Current | ADR-014 |
| **architecture.md** | System architecture details | 🟢 Current | state-management.md |
| **README.md** | Design directory overview | 🟢 Current | Documentation hub |

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
| **learning-strategies.md** | Comprehensive learning strategy guide (920 lines) | 🟢 Current | copilot-instructions, code comments |
| **component-parity.md** | Regions and pathways consistency | 🟢 Current | copilot-instructions, base.py |
| **state-management.md** | When to use RegionState vs attributes | 🟢 Current | copilot-instructions, neuron_models.md |
| **mixins.md** | Available mixins and usage | 🟢 Current | copilot-instructions, region files |
| **configuration.md** | Config hierarchy and parameters | 🟢 Current | mixins.md |
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
| **adr-014-distributed-computation.md** | Multi-core CPU support | Accepted | decisions/README |
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
