# Thalia Documentation

**Status**: 🚧 Active Development — Documentation evolving with the codebase
**Last Updated**: December 21, 2025 (Post-Consolidation)

**Recent Updates**:
- ✅ December 21, 2025: Documentation consolidation complete
  - Archived 9 completed implementation documents
  - Fixed broken code references
  - Reduced active documentation by 18% (67→55 active files)
  - All validation checks passing

## Overview

Thalia is a biologically-accurate spiking neural network framework for building multi-modal, biologically-plausible ML models. This documentation provides guidance for understanding, using, and extending the framework.

## Quick Start

**New to Thalia?**
1. Start with [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md) to understand the system
2. Review [Getting Started with Curriculum Training](GETTING_STARTED_CURRICULUM.md) for hands-on tutorial
3. Check [Quick References](#quick-references) for common APIs and patterns

## Documentation Structure

### 📚 [API Reference](./api/) (Auto-Generated)
Always-up-to-date API documentation generated from code.
- **[COMPONENT_CATALOG.md](./api/COMPONENT_CATALOG.md)** — All registered regions and pathways
- **[LEARNING_STRATEGIES_API.md](./api/LEARNING_STRATEGIES_API.md)** — Learning strategy factory functions
- **[CONFIGURATION_REFERENCE.md](./api/CONFIGURATION_REFERENCE.md)** — Configuration dataclasses
- **[DATASETS_REFERENCE.md](./api/DATASETS_REFERENCE.md)** — Dataset classes and factory functions
- **[DIAGNOSTICS_REFERENCE.md](./api/DIAGNOSTICS_REFERENCE.md)** — Diagnostic monitors
- **[EXCEPTIONS_REFERENCE.md](./api/EXCEPTIONS_REFERENCE.md)** — Exception hierarchy
- Run `python scripts/generate_api_docs.py` to regenerate

### 🏗️ [Architecture](./architecture/)
High-level system architecture and design philosophy.
- **[ARCHITECTURE_OVERVIEW.md](./architecture/ARCHITECTURE_OVERVIEW.md)** — START HERE for system overview
- **[CENTRALIZED_SYSTEMS.md](./architecture/CENTRALIZED_SYSTEMS.md)** — Neuromodulators, oscillators, goals
- **[REFACTOR_EXPLICIT_AXONS_SYNAPSES.md](./architecture/REFACTOR_EXPLICIT_AXONS_SYNAPSES.md)** — architecture rationale
- Complexity layers (5-level architecture)
- Component interactions and data flow

### 📐 [Design](./design/)
Detailed technical specifications for core systems.
- **[Checkpoint Format](./design/checkpoint_format.md)** — Serialization and state persistence
- **[Curriculum Strategy](./design/curriculum_strategy.md)** — Training stages and consolidation
- **[Neuron Models](./design/neuron_models.md)** — ConductanceLIF neurons (ONLY neuron model)
- **[Circuit Modeling](./design/circuit_modeling.md)** — D1/D2 delays and biological timing
- **[Delayed Gratification](./design/delayed_gratification.md)** — TD(λ) and multi-step credit assignment

### 🛡️ [Safety Systems](./CURRICULUM_SAFETY_SYSTEM.md)
**CRITICAL**: Comprehensive safety systems for curriculum training.
- **[Safety System Guide](./CURRICULUM_SAFETY_SYSTEM.md)** — Stage gates, monitoring, graceful degradation
- **[Safety Quick Start](./SAFETY_QUICK_START.md)** — Quick reference for safety features
- Stage gates and transition criteria
- Failure detection and recovery

### 🔧 [Patterns](./patterns/)
Common implementation patterns and best practices.
- **[Learning Strategies](./patterns/learning-strategies.md)** — Strategy pattern for learning rules (920 lines)
- **[Component Parity](./patterns/component-parity.md)** — Regions vs pathways
- **[State Management](./patterns/state-management.md)** — When to use RegionState vs attributes
- **[Mixins](./patterns/mixins.md)** — Available mixins and their usage
- **[Configuration](./patterns/configuration.md)** — Config hierarchy and parameters

### 🧠 [Decisions](./decisions/)
Architecture Decision Records (ADRs) documenting key technical choices.
- **[ADR-001: Simulation Backend](./decisions/adr-001-simulation-backend.md)** — PyTorch with GPU
- **[ADR-003: Clock-Driven Simulation](./decisions/adr-003-clock-driven.md)** — Fixed timestep approach
- **[ADR-004: Bool Spikes](./decisions/adr-004-bool-spikes.md)** — Binary spike representation
- **[ADR-005: No Batch Dimension](./decisions/adr-005-no-batch-dimension.md)** — Single-sample processing
- **[ADR-010: Region Axonal Delays](./decisions/adr-010-region-axonal-delays.md)** — Delays in pathways
- **[ADR-013: Explicit Pathway Projections](./decisions/adr-013-explicit-pathway-projections.md)** — v2.0 architecture

## Quick References

### Essential Guides
- **[Getting Started with Curriculum](./GETTING_STARTED_CURRICULUM.md)** — Step-by-step tutorial
- **[Curriculum Quick Reference](./CURRICULUM_QUICK_REFERENCE.md)** — API reference for training
- **[Datasets Quick Reference](./DATASETS_QUICK_REFERENCE.md)** — Stage-specific datasets
- **[Monitoring Guide](./MONITORING_GUIDE.md)** — Diagnostics and visualization
- **[AI Assistant Guide](./AI_ASSISTANT_GUIDE.md)** — Navigation for AI assistants

### For New Contributors
1. Read the [project README](../README.md) for vision and principles
2. Review [Architecture Overview](./architecture/ARCHITECTURE_OVERVIEW.md) to understand the system
3. Check [Patterns](./patterns/) for implementation guidelines
4. See [Decisions](./decisions/) for rationale behind key choices
5. Review [CONTRIBUTING.md](../CONTRIBUTING.md) for code standards

### For Development
- **Adding a Region**: See [patterns/learning-strategies.md](./patterns/learning-strategies.md) and [AI_ASSISTANT_GUIDE.md](./AI_ASSISTANT_GUIDE.md)
- **Learning Rules**: See [patterns/learning-strategies.md](./patterns/learning-strategies.md) for strategy pattern
- **Testing**: See [../tests/WRITING_TESTS.md](../tests/WRITING_TESTS.md)
- **Training**: See [design/curriculum_strategy.md](./design/curriculum_strategy.md)
- **Safety Systems**: See [CURRICULUM_SAFETY_SYSTEM.md](./CURRICULUM_SAFETY_SYSTEM.md)
- **Monitoring**: See [MONITORING_GUIDE.md](./MONITORING_GUIDE.md)

### For Understanding Design
- **v2.0 Architecture**: [architecture/REFACTOR_EXPLICIT_AXONS_SYNAPSES.md](./architecture/REFACTOR_EXPLICIT_AXONS_SYNAPSES.md)
- **Learning Strategies**: [patterns/learning-strategies.md](./patterns/learning-strategies.md)
- **Checkpoint Design**: [design/checkpoint_format.md](./design/checkpoint_format.md)
- **Training Stages**: [design/curriculum_strategy.md](./design/curriculum_strategy.md)
- **Circuit Timing**: [design/circuit_modeling.md](./design/circuit_modeling.md)

## Status Legend

- 🟢 **Current** — Up to date with codebase
- 🟡 **Partial** — Accurate but incomplete
- 🔴 **Outdated** — Needs revision
- 📦 **Archived** — Historical reference only
- 🚧 **Draft** — Work in progress

## Contributing to Documentation

Documentation lives in `docs/` and follows this structure:
- **Architecture** — System-level design
- **Design** — Detailed specifications
- **Patterns** — Implementation guidance
- **Decisions** — ADRs for key technical choices
- **Archive** — Historical/completed documents

When adding documentation:
1. Choose the appropriate directory
2. Use clear, concise language
3. Add status indicators
4. Update this README with links to new docs

### 📦 [Archive](./archive/)

Completed implementations, superseded designs, and historical documentation.

**Contents**:
- Completed implementation summaries (noise system, spillover, oscillator monitoring)
- Finished architecture specifications (L6_TRN feedback, biological spec, oscillation analysis)
- Superseded design documents (clock optimizations, TRN enhancement plans)
- Completed review sessions

**See**: [archive/README.md](./archive/README.md) for full inventory and consolidation history

**When to reference**:
- Understanding implementation history
- Reviewing design decisions and rationale
- Finding completed feature details

**When NOT to reference**:
- Active development (use current docs instead)
- New contributor onboarding (use getting started guides)
- API usage (use quick references)

## Notes

- **API Documentation**: Deferred until API stabilizes (post-Stage 0)
- **Tutorials**: Deferred until core patterns are established
- **Getting Started Guide**: Coming after initial experiments validate the approach

---

**Last Updated**: December 13, 2025
