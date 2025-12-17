# Thalia Architecture Reference

**Last Updated**: December 13, 2025
**Status**: 🟢 Current - Consolidated reference document

---

## Overview

This document provides a quick reference to Thalia's architecture. For comprehensive documentation, see the **[Architecture Documentation](../architecture/)** directory.

---

## Quick Links

### 📚 Comprehensive Guides
- **[Architecture Overview](../architecture/ARCHITECTURE_OVERVIEW.md)** - Complete system architecture (START HERE)
- **[Centralized Systems](../architecture/CENTRALIZED_SYSTEMS.md)** - Neuromodulators, oscillators, goals, consolidation
- **[Supporting Components](../architecture/SUPPORTING_COMPONENTS.md)** - Managers, action selection, environments
- **[Architecture Index](../architecture/INDEX.md)** - Searchable component index

### 🏗️ Core Concepts
- **[Biological Plausibility](../architecture/ARCHITECTURE_OVERVIEW.md#core-architecture-principles)** - Spike-based processing, local learning, neuromodulation
- **[Regional Specialization](../architecture/ARCHITECTURE_OVERVIEW.md#brain-regions)** - Each region has its own learning rule
- **[Event-Driven Processing](../architecture/ARCHITECTURE_OVERVIEW.md#event-driven-computation)** - Spike propagation with axonal delays

---

## Component Hierarchy (5 Levels)

Thalia is organized into **5 complexity levels** from primitive components to full brain integration:

```
Level 0: PRIMITIVES
  ↓ (neurons, spikes, traces)
Level 1: LEARNING RULES
  ↓ (STDP, BCM, Hebbian, three-factor)
Level 2: STABILITY MECHANISMS
  ↓ (homeostasis, E/I balance, normalization)
Level 3: REGIONS (Isolated)
  ↓ (cortex, hippocampus, striatum, PFC, cerebellum)
Level 4: INTEGRATION
  ↓ (DynamicBrain, pathways, neuromodulators, oscillators)
```

### Level 0: Primitives
**Components**: LIF neurons, ConductanceLIF, spike traces, short-term plasticity
**Location**: `src/thalia/components/neurons/`, `src/thalia/components/synapses/`
**Details**: See [Neuron Models](neuron_models.md)

### Level 1: Learning Rules
**Components**: STDP, BCM, Hebbian, three-factor (striatum), error-corrective (cerebellum)
**Location**: `src/thalia/learning/rules/`, `src/thalia/learning/homeostasis/`
**Details**: See [Learning Strategies](../architecture/ARCHITECTURE_OVERVIEW.md#learning-rules)

### Level 2: Stability Mechanisms
**Components**: Homeostatic plasticity, E/I balance, weight normalization
**Location**: `src/thalia/learning/homeostasis/`
**Details**: See [Homeostasis](../architecture/SUPPORTING_COMPONENTS.md#homeostasis-management)

### Level 3: Regions
**Components**: Cortex, Hippocampus, Prefrontal, Striatum, Cerebellum, Thalamus
**Location**: `src/thalia/regions/`
**Details**: See [Brain Regions](../architecture/ARCHITECTURE_OVERVIEW.md#brain-regions)

### Level 4: Integration
**Components**: DynamicBrain, NeuromodulatorManager, OscillatorManager, PathwayManager
**Location**: `src/thalia/core/dynamic_brain.py`, `src/thalia/neuromodulation/`, `src/thalia/coordination/`
**Details**: See [Centralized Systems](../architecture/CENTRALIZED_SYSTEMS.md)
- **Implements**: Sliding threshold, rate-based learning
- **Parameters**: `learning_rate`, `tau_theta`, `p` (LTP/LTD power)
- **Use when**: Need competition between synapses

#### Three-Factor Learning
- **File**: `src/thalia/learning/strategies.py`
- **Implements**: STDP + neuromodulator (dopamine)
- **Parameters**: STDP params + `tau_eligibility`
- **Use when**: Reinforcement learning scenarios

#### Hebbian Learning
- **File**: `src/thalia/learning/strategies.py`
- **Implements**: Simple "fire together, wire together"
- **Use when**: Simple association learning

### Dependencies
- Level 0: Uses spike traces for temporal dynamics
- No dependencies on stability or regions

### Testing
- **Location**: `tests/unit/test_core.py`, `tests/unit/test_brain_regions.py`
- **Approach**: Apply to random weight matrices, verify convergence
- **Example**: STDP strengthens coincident pre-post activity

---

## Level 2: STABILITY MECHANISMS

### Purpose
Prevent pathological dynamics (runaway excitation, activity collapse, weight explosion).


---

## Brain Regions Quick Reference

### Cortex
**File**: `src/thalia/regions/layered_cortex.py`
**Learning**: Hebbian/BCM/STDP
**Function**: Sensory processing, feature extraction
**Layers**: L4 (input) → L2/3 (integration) → L5 (output)

### Hippocampus
**File**: `src/thalia/regions/hippocampus/`
**Learning**: One-shot Hebbian
**Function**: Episodic memory, sequence learning
**Circuit**: DG (separation) → CA3 (completion) → CA1 (comparator)

### Striatum
**File**: `src/thalia/regions/striatum/`
**Learning**: Three-factor (eligibility × dopamine)
**Function**: Action selection, reinforcement learning
**Pathways**: D1 (Go) and D2 (No-Go) opponent processing

### Prefrontal Cortex
**File**: `src/thalia/regions/prefrontal_hierarchy.py`
**Learning**: Gated Hebbian
**Function**: Working memory, goal hierarchy, planning
**Features**: Persistent activity, goal stack, options learning

### Cerebellum
**File**: `src/thalia/regions/cerebellum.py`
**Learning**: Error-corrective (supervised)
**Function**: Motor control, prediction
**Circuit**: Mossy fibers → Granule cells → Purkinje cells

### Thalamus
**File**: `src/thalia/regions/thalamus.py`
**Function**: Sensory relay, attention gating
**Features**: TRN (thalamic reticular nucleus) for selective attention

---

## Integration Systems

### Neuromodulation
**Location**: `src/thalia/neuromodulation/`
**Systems**: VTA (dopamine), LC (norepinephrine), NB (acetylcholine)
**Manager**: `NeuromodulatorManager` coordinates all three systems
**See**: [Centralized Systems](../architecture/CENTRALIZED_SYSTEMS.md)

### Oscillators
**Location**: `src/thalia/coordination/oscillator.py`
**Rhythms**: Delta, theta, alpha, beta, gamma
**Manager**: `OscillatorManager` with cross-frequency coupling
**See**: [Centralized Systems](../architecture/CENTRALIZED_SYSTEMS.md#oscillator-system)

### Planning & Goals
**Location**: `src/thalia/planning/`, `src/thalia/regions/prefrontal_hierarchy.py`
**Features**: Dyna-style planning, goal hierarchy, options learning
**See**: [Hierarchical Goals](../architecture/HIERARCHICAL_GOALS_COMPLETE.md)

### Memory Consolidation
**Location**: `src/thalia/memory/consolidation/`
**Features**: Replay engine, offline learning, priority-based scheduling
**See**: [Centralized Systems](../architecture/CENTRALIZED_SYSTEMS.md#consolidation-system)

---

## Testing & Patterns

### Testing by Level
- **Level 0-1**: Unit tests (`tests/unit/test_core.py`)
- **Level 2**: Robustness tests (`tests/unit/test_robustness.py`)
- **Level 3**: Region tests (`tests/unit/test_brain_regions.py`)
- **Level 4**: Integration tests (`tests/integration/`)

### Common Patterns
- **State Management**: See [State Management](../patterns/state-management.md)
- **Component Parity**: See [Component Parity](../patterns/component-parity.md)
- **Mixins**: See [Mixins](../patterns/mixins.md)

---

## Dependency Rules

```
✅ ALLOWED:
Level 4 → 3, 2, 1, 0
Level 3 → 2, 1, 0
Level 2 → 1, 0
Level 1 → 0
Level 0 → PyTorch only

❌ FORBIDDEN:
- Upward dependencies (lower levels cannot import higher levels)
- Cross-region dependencies (regions cannot import each other)
- Circular dependencies
```

---

## Related Documentation

### Design Documents
- **[Neuron Models](neuron_models.md)** - LIF and conductance-based neuron details
- **[Curriculum Strategy](curriculum_strategy.md)** - Training stages and progression
- **[Parallel Execution](parallel_execution.md)** - Event-driven processing
- **[Circuit Modeling](circuit_modeling.md)** - Biological circuit timing

### Architecture Documents
- **[Architecture Overview](../architecture/ARCHITECTURE_OVERVIEW.md)** - Comprehensive guide
- **[Centralized Systems](../architecture/CENTRALIZED_SYSTEMS.md)** - Managers and coordination
- **[Supporting Components](../architecture/SUPPORTING_COMPONENTS.md)** - Infrastructure

### Patterns & Decisions
- **[Patterns](../patterns/)** - Implementation patterns and best practices
- **[ADRs](../decisions/)** - Architecture decision records

---

**For detailed architecture information, see [Architecture Documentation](../architecture/)**.
