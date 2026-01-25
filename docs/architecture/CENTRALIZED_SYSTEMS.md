# Centralized Systems Architecture

**Status**: ✅ COMPLETE
**Last Updated**: December 2025

## Overview

Thalia implements several brain-wide systems as centralized managers that compute once and broadcast to all regions. This design mirrors biological architecture where certain nuclei project globally (VTA, LC, NB) or coordinate across regions (oscillators).

## Benefits of Centralization

1. **Biological Accuracy** - Matches neuroanatomy of global projection systems
2. **Consistency** - All regions receive synchronized signals
3. **Efficiency** - Compute once, broadcast to all (no redundant computation)
4. **Testability** - Each system can be tested independently
5. **Maintainability** - Clear separation of concerns

## Centralized Systems

### 1. Neuromodulator Systems (NeuromodulatorManager)

**Location**: `src/thalia/neuromodulation/`

**Components**:
- **VTADopamineSystem** (`systems/vta.py`) - Dopamine for reward/learning
- **LocusCoeruleusSystem** (`systems/locus_coeruleus.py`) - Norepinephrine for arousal/uncertainty
- **NucleusBasalisSystem** (`systems/nucleus_basalis.py`) - Acetylcholine for attention/encoding
- **NeuromodulatorCoordination** (`homeostasis.py`) - Biological interactions between systems

**Manager**: `NeuromodulatorManager` (`manager.py`)

**Integration**:
```python
# In DynamicBrain.__init__()
self.neuromodulator_manager = NeuromodulatorManager()

# In timestep updates
self.neuromodulator_manager.update(
    dt_ms=self.config.dt_ms,
    intrinsic_reward=intrinsic_reward,
    uncertainty=uncertainty,
    prediction_error=prediction_error
)

# Broadcast to all regions (using brain.components dict)
regions = {name: comp for name, comp in self.components.items()}
self.neuromodulator_manager.broadcast_to_regions(regions)
```
```

**Coordination**:
- **NE-ACh**: Moderate arousal optimal for encoding (inverted-U)
- **DA-ACh**: High reward + low novelty suppresses encoding
- **DA-NE**: High prediction error + high arousal boosts both

**Overhead**: <0.001% computational cost

---

### 2. Oscillator System (OscillatorManager)

**Location**: `src/thalia/coordination/oscillator.py`

**Oscillators**:
- **Delta** (2 Hz) - Sleep consolidation, NREM gating
- **Theta** (8 Hz) - Working memory slots, encoding/retrieval (acts as biological septum)
- **Alpha** (10 Hz) - Attention suppression, sensory gating
- **Beta** (20 Hz) - Motor control, action maintenance
- **Gamma** (40 Hz) - ⚠️ **DISABLED BY DEFAULT** - Should emerge from L6→TRN loop (~25ms)

**Gamma Emergence** (Dec 20, 2025):
- Explicit gamma oscillator disabled by default in `OscillatorManager.__init__()`
- Gamma should emerge from local circuit timing (L6→TRN→Thalamus feedback)
- Enable explicitly if needed: `brain.oscillators.enable_oscillator('gamma', True)`
- See: `OSCILLATION_EMERGENCE_ANALYSIS.md` for full rationale

**Cross-Frequency Coupling**:
1. **Theta-Gamma** - Working memory capacity (~7±2 items)
2. **Beta-Gamma** - Motor timing coordination
3. **Delta-Theta** - Sleep consolidation (replay gating)
4. **Alpha-Gamma** - Attention-dependent feature binding
5. **Theta-Beta** - Working memory → action coordination

**Integration**:
```python
# In DynamicBrain.__init__()
self.oscillator_manager = OscillatorManager()

# In timestep updates
self.oscillator_manager.advance(dt_ms=self.config.dt_ms)

# Broadcast to all regions
phases = self.oscillator_manager.get_phases()
signals = self.oscillator_manager.get_signals()
coupled_amps = self.oscillator_manager.get_coupled_amplitudes()

for region in all_regions:
    region.set_oscillator_phases(phases, signals, coupled_amps)
```

**Overhead**: <0.001% computational cost

---

### 3. Goal Hierarchy System (GoalHierarchyManager)

**Location**: `src/thalia/regions/prefrontal_hierarchy.py`

**Components**:
- **Goal** - Hierarchical goal data structure
- **GoalHierarchyManager** - Goal stack and decomposition
- **HyperbolicDiscounter** - Context-dependent temporal discounting

**Features**:
- Goal selection (value-based with deadline pressure)
- Goal decomposition (state-dependent subgoal generation)
- Goal stack management (working memory capacity limits)
- Options learning (caching successful policies)
- Progress tracking (completion criteria)

**Integration**:
```python
# In PrefrontalCortex
self.goal_manager = GoalHierarchyManager(config)

# In TrialCoordinator.forward()
if goal_manager is not None:
    current_goal = goal_mgr.get_current_goal()
    if current_goal is not None:
        goal_mgr.update_progress(current_goal, pfc_state)
        if current_goal.status == "completed":
            goal_mgr.pop_goal()
    goal_mgr.advance_time()

# In TrialCoordinator.select_action()
if current_goal has policy:
    action = goal.policy(pfc_state)  # Use learned option
    return action, 1.0

# In TrialCoordinator.deliver_reward()
goal_mgr.record_option_attempt(goal.name, success)
if success_rate > threshold:
    goal_mgr.cache_option(goal, policy, success_rate)
```

---

### 4. Memory Consolidation System (ConsolidationManager)

**Location**: `src/thalia/memory/consolidation/`

**Components**:
- **ConsolidationManager** (`manager.py`) - Consolidation coordinator
- **ReplayEngine** (`src/thalia/regions/hippocampus/replay_engine.py`) - Sequence replay

**Features**:
- Spontaneous hippocampal replay during rest/sleep
- Memory pressure detection (limited hippocampal capacity)
- Sleep stage simulation (NREM/REM alternation)
- Sharp-wave ripple triggered consolidation

**Integration**:
```python
# Consolidation is triggered directly via DynamicBrain.consolidate()
# Uses hippocampal spontaneous replay mechanism

# Lower acetylcholine → trigger consolidation
stats = brain.consolidate(duration_ms=10000, verbose=True)
# Output: "Consolidation: 23 ripples in 10000ms (2.3 Hz)"

# Available consolidation utilities
from thalia.memory.consolidation import (
    MemoryPressureDetector,     # Detect when consolidation is needed
    SleepStageController,        # Simulate NREM/REM cycles
    ConsolidationTrigger,        # Coordinate consolidation events
)

# Run consolidation (offline replay)
self.consolidation_manager.consolidate(n_cycles=10)
```

**Overhead**: Minimal (only during consolidation cycles)

---

## System Comparison

| System | Location | Lines | Overhead | Biological Analog |
|--------|----------|-------|----------|-------------------|
| Neuromodulators | `neuromodulation/` | ~1000 | <0.001% | VTA, LC, NB nuclei |
| Oscillators | `coordination/oscillator.py` | ~300 | <0.001% | Thalamic pacemakers |
| Goals | `regions/prefrontal_hierarchy.py` | ~800 | Minimal | PFC hierarchy |
| Consolidation | `memory/consolidation/` | ~600 | Minimal* | Sleep replay |

*Only during consolidation cycles

---

## Design Pattern

All centralized systems follow this pattern:

1. **Initialization**: Brain creates manager during `__init__()`
2. **Update**: Manager computes signals during timestep
3. **Broadcast**: Brain distributes signals to all regions
4. **Usage**: Regions use signals for local computation

**Key Principle**: Compute globally, apply locally

---

## Future Extensions

Potential additional centralized systems:

1. **EmotionalStateManager** - Amygdala-mediated emotional states
2. **CircadianManager** - Sleep/wake cycle regulation
3. **PriorityScheduler** - Priority-based consolidation scheduling
4. **MetaCognitionManager** - Self-monitoring and cognitive control

---

## Testing

Each system has comprehensive tests:
- Unit tests: `tests/unit/neuromodulation/`, `tests/unit/coordination/`
- Integration tests: `tests/integration/`

---

## References

- **Neuromodulation**: `NEUROMODULATOR_CENTRALIZATION_COMPLETE.md` (archived - redundant)
- **Brain Coordination**: `BRAIN_COORDINATION_INTEGRATION.md` (archived - redundant)
- **Oscillators**: `OSCILLATOR_INTEGRATION_COMPLETE.md` (archived - redundant)
- **Goals**: `HIERARCHICAL_GOALS_COMPLETE.md`, `GOAL_HIERARCHY_IMPLEMENTATION_SUMMARY.md`
