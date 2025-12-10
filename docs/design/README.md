# Design Documents

Technical design specifications for the Thalia framework.

## Current Documents

### Core Systems
- **[Checkpoint Format](checkpoint_format.md)** 🟢 — Binary checkpoint format and state serialization
- **[Curriculum Strategy](curriculum_strategy.md)** 🟢 — Training stages and consolidation approach
- **[Neuron Models](neuron_models.md)** 🟢 — LIF and conductance-based neuron implementations
- **[Robustness Configuration](robustness_config_guide.md)** 🟢 — Stability mechanisms and configuration

### Implementation Plans
- **[Delayed Gratification Plan](delayed_gratification_plan.md)** 🚧 — Overview and curriculum stage analysis
- **[Implementation Quick Start](IMPLEMENTATION_QUICK_START.md)** 🚧 — Quick reference for all three phases
- **[Phase 1: TD(λ)](PHASE1_TD_LAMBDA.md)** 🚧 — Multi-step credit assignment (Weeks 1-3)
- **[Phase 2: Model-Based Planning](PHASE2_MODEL_BASED.md)** 🚧 — Forward simulation and tree search (Weeks 4-7)
- **[Phase 3: Hierarchical Goals](PHASE3_HIERARCHICAL.md)** 🚧 — Goal hierarchies and hyperbolic discounting (Weeks 8-11)

### Additional Implementations
- **[Cognitive Load Implementation](cognitive_load_implementation.md)** 🟡 — Cognitive load monitoring
- **[Curriculum Implementation](curriculum_implementation.md)** 🟡 — Curriculum training details
- **[Metacognition Implementation](metacognition_implementation.md)** 🟡 — Metacognitive monitoring

### Architecture Reference
- **[Architecture Overview](architecture.md)** 🟡 — High-level system design (may be moved to `../architecture/`)

## Status Legend

- 🟢 **Current** — Up to date with codebase
- 🟡 **Partial** — Accurate but incomplete
- 🔴 **Outdated** — Needs revision
- 🚧 **Draft** — Work in progress

## Related Documentation

- **[Patterns](../patterns/)** — Implementation patterns and best practices
- **[Decisions](../decisions/)** — Architecture decision records (ADRs)
- **[Architecture](../architecture/)** — System-level architecture docs

## Implementation Plan Documents

The **Delayed Gratification** implementation plan spans three phases over 11 weeks:

1. **Phase 1 (Weeks 1-3)**: Multi-step credit assignment via TD(λ) — extends temporal credit from 1 second to 5-10 seconds
2. **Phase 2 (Weeks 4-7)**: Model-based planning — enables mental simulation of action sequences
3. **Phase 3 (Weeks 8-11)**: Hierarchical goals — adds goal decomposition and context-dependent discounting

**Start Here**: Read [IMPLEMENTATION_QUICK_START.md](IMPLEMENTATION_QUICK_START.md) for overview, then dive into individual phase documents.

**Why Important**: Enables true delayed gratification — the ability to pursue long-term goals despite short-term costs. Critical for:
- Sensorimotor learning (Stage -0.5): Multi-step action→feedback delays
- Grammar generation (Stage 2): Planning ahead multiple words
- Essay writing (Stage 3): Maintaining coherence across paragraphs
- Abstract reasoning (Stage 4+): Complex problem decomposition

---

**Last Updated**: December 10, 2025
