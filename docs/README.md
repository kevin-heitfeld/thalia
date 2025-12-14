# Thalia Documentation

**Status**: 🚧 Active Development — Documentation evolving with the codebase

## Overview

Thalia is a biologically-accurate spiking neural network framework for building multi-modal, biologically-plausible ML models. This documentation provides guidance for understanding, using, and extending the framework.

## Documentation Structure

### 🏗️ [Architecture](./architecture/)
High-level system architecture and design philosophy.
- System overview
- Complexity layers (5-level architecture)
- Component interactions

### 📐 [Design](./design/)
Detailed technical specifications for core systems.
- **[Checkpoint Format](./design/checkpoint_format.md)** — Serialization and state persistence
- **[Curriculum Strategy](./design/curriculum_strategy.md)** — Training stages and consolidation
- **[Neuron Models](./design/neuron_models.md)** — LIF and conductance-based neurons
- **[Robustness Config](./design/robustness_config_guide.md)** — Stability mechanisms

### 🛡️ [Safety Systems](./CURRICULUM_SAFETY_SYSTEM.md)
**CRITICAL**: Comprehensive safety systems for curriculum training.
- **[Safety System Guide](./CURRICULUM_SAFETY_SYSTEM.md)** — Stage gates, monitoring, graceful degradation
- **[Validation Summary](./CURRICULUM_VALIDATION_SUMMARY.md)** — Expert review + engineering consensus
- **Stage 1 Survival Checklist** — Hard criteria for Stage 1→2 transition
- **Kill-Switch Map** — Module criticality and failure handling

### 🔧 [Patterns](./patterns/)
Common implementation patterns and best practices.
- **[Configuration](./patterns/configuration.md)** — Config hierarchy and parameters
- **[State Management](./patterns/state-management.md)** — When to use RegionState vs attributes
- **[Mixins](./patterns/mixins.md)** — Available mixins and their usage

### 🧠 [Decisions](./decisions/)
Architecture Decision Records (ADRs) documenting key technical choices.
- **[ADR-001: Simulation Backend](./decisions/adr-001-simulation-backend.md)** — PyTorch with GPU
- **[ADR-002: Numeric Precision](./decisions/adr-002-numeric-precision.md)** — Mixed precision strategy
- **[ADR-003: Clock-Driven Simulation](./decisions/adr-003-clock-driven.md)** — Fixed timestep approach

### 📚 [Research](./research/)
Research notes, paper summaries, and theoretical foundations.

### 📦 [Archive](./archive/)
Historical documentation and completed planning documents.
- Original planning documents
- Completed refactoring summaries
- Historical ablation results

## Quick Links

### For New Contributors
1. Read the [project README](../README.md) for vision and principles
2. Review [Architecture Overview](./architecture/README.md) to understand the system
3. Check [Patterns](./patterns/) for implementation guidelines
4. See [Decisions](./decisions/) for rationale behind key choices

### For Development
- **Adding a Region**: See [patterns/configuration.md](./patterns/configuration.md) and [patterns/mixins.md](./patterns/mixins.md)
- **Testing**: See [../tests/README.md](../tests/README.md) and [../tests/WRITING_TESTS.md](../tests/WRITING_TESTS.md)
- **Training**: See [design/curriculum_strategy.md](./design/curriculum_strategy.md)
- **Safety Systems**: See [CURRICULUM_SAFETY_SYSTEM.md](./CURRICULUM_SAFETY_SYSTEM.md) (🛡️ **REQUIRED for Stage 1+**)
- **Monitoring & Diagnostics**: See [MONITORING_GUIDE.md](./MONITORING_GUIDE.md) (⭐ start here)

### For Understanding Design
- **Why PyTorch?**: [decisions/adr-001-simulation-backend.md](./decisions/adr-001-simulation-backend.md)
- **Checkpoint Design**: [design/checkpoint_format.md](./design/checkpoint_format.md)
- **Training Stages**: [design/curriculum_strategy.md](./design/curriculum_strategy.md)

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

## Notes

- **API Documentation**: Deferred until API stabilizes (post-Stage 0)
- **Tutorials**: Deferred until core patterns are established
- **Getting Started Guide**: Coming after initial experiments validate the approach

---

**Last Updated**: December 13, 2025
