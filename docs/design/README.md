# Design Documents

Technical design specifications for the Thalia framework.

## Current Documents

### Core Systems
- **[Checkpoint Format](checkpoint_format.md)** 🟢 — Binary checkpoint format and state serialization
- **[Curriculum Strategy](curriculum_strategy.md)** 🟢 — Training stages and developmental progression (expert-reviewed)
- **[Neuron Models](neuron_models.md)** 🟢 — ConductanceLIF neuron implementation (ONLY neuron model)
- **[Parallel Execution](parallel_execution.md)** 🔴 — SUPERSEDED (Event-driven removed, now clock-driven)

### Architecture & Planning
- **[Architecture Reference](architecture.md)** 🟢 — Quick reference to comprehensive architecture docs
- **[Circuit Modeling](circuit_modeling.md)** 🟢 — Biological circuit timing and implementation priorities

### Implementation Plans

#### Delayed Gratification (Planning Systems)
- **[Delayed Gratification](delayed_gratification.md)** 🟢 — Implementation complete, integration testing in progress
- **Phase 2: Model-Based Planning** 🟢 — **IMPLEMENTED** (`src/thalia/planning/`)
  - `dyna.py` - Dyna-style background planning
  - `coordinator.py` - Mental simulation coordinator
- **Phase 3: Hierarchical Goals** 🟢 — **IMPLEMENTED** (`src/thalia/regions/prefrontal_hierarchy.py`)
  - Goal decomposition and stack management
  - Options learning and caching
  - Hyperbolic temporal discounting
  - See `../architecture/HIERARCHICAL_GOALS_COMPLETE.md` for details

**Status Update**: Phases 2 and 3 are now complete. Phase 1 (TD(λ)) may already be covered by existing eligibility traces.

### Completed Implementations (Archive for Reference)
These documents describe implementations that are now complete. Kept for historical context:
- **[Checkpoint Growth Compatibility](checkpoint_growth_compatibility.md)** 🟢 — Checkpoint format evolution (v1.0 complete)

## Status Legend

- 🟢 **Current** — Up to date with codebase
- 🟡 **Needs Review** — Accurate but may need updating or consolidation
- 🔴 **Outdated** — Needs significant revision
- 🚧 **Draft** — Work in progress / Planning phase

## Related Documentation

- **[Architecture](../architecture/)** — System-level architecture docs (primary reference for current architecture)
- **[Patterns](../patterns/)** — Implementation patterns and best practices
- **[Decisions](../decisions/)** — Architecture decision records (ADRs)

## Key Implementation Status

### ✅ Fully Implemented
1. **Neuromodulator Systems** - VTA, LC, NB with coordination (`src/thalia/neuromodulation/`)
2. **Oscillator System** - 5 rhythms + cross-frequency coupling (`src/thalia/coordination/oscillator.py`)
3. **Goal Hierarchy** - Hierarchical planning with options learning (`src/thalia/regions/prefrontal_hierarchy.py`)
4. **Memory Consolidation** - Replay and offline learning (`src/thalia/memory/consolidation/`)
5. **Attention Systems** - Thalamus + attention pathways (`src/thalia/regions/thalamus.py`, `src/thalia/pathways/attention/`)
6. **Language Processing** - Token encoding/decoding + brain integration (`src/thalia/language/`)
7. **Social Learning** - Imitation, pedagogy, joint attention (`src/thalia/learning/social_learning.py`)
8. **Metacognition** - Confidence calibration (`src/thalia/training/evaluation/metacognition.py`)
9. **Mental Simulation** - Dyna-style planning (`src/thalia/planning/`)

### 🔨 In Progress / Planning
1. **Amygdala Integration** - Emotional state processing (not yet implemented)
2. **Advanced Prioritization** - Cognitive load-aware replay scheduling (partial)

### 📋 Planned Future Work
1. **Theory of Mind** - Modeling others' mental states
2. **Advanced Executive Functions** - Complex reasoning integration
3. **Circadian Rhythms** - Sleep/wake cycle regulation

## Documentation Maintenance

### When to Update
- **Implementation complete**: Update status from 🚧 to 🟢
- **Code refactored**: Verify docs match new structure
- **New features added**: Add to implementation status list
- **Architecture changed**: Update architecture.md or point to `../architecture/`

### Consolidation Opportunities
- ✅ `architecture.md` → **DONE** - Now a quick reference to comprehensive architecture docs
- Phase documents → Consider archiving once integration testing is complete

---

**Last Updated**: December 21, 2025
