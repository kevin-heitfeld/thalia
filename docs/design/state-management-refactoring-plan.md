# State Management Refactoring - Implementation Plan

**Date**: December 21, 2025
**Status**: ⏳ IN PROGRESS (Phase 2.4 COMPLETED → Phase 3.1 NEXT)
**Estimated Effort**: 59-73 hours
**Breaking Changes**: Very High
**Risk Level**: High
**Completed**: 143/143 tests passing (Phase 0, 1, 2.1, 2.2, 2.3, 2.4)

---

## Executive Summary

This document outlines a comprehensive refactoring of state management to create unified `RegionState` and `PathwayState` abstract base classes. This refactoring is justified by the user's research requirements:
- **Long-term training** (weeks/months) → Need checkpoint migration
- **Reproducible research** → Need exact state snapshots
- **Reusable brain components** → Need transfer learning support
- **Interactive tools** → Need unified state inspection API

**Key Benefits**:
1. Checkpoint evolution without breaking old saves
2. Exact brain state serialization for reproducibility
3. Transfer learning and component reuse
4. Generic debugging and inspection tools
5. Automated testing infrastructure
6. **[NEW]** Pathway state preservation (in-flight spikes, delay buffers)

**Progress Summary**:
- ✅ Phase 0: Pathway State Foundation (13/13 tests) - Commit db0321f
- ✅ Phase 1: RegionState Foundation (24/24 tests) - Commit 79b8bba
- ✅ Phase 2.1: PrefrontalState (22/22 tests) - Commit 917f7ba
- ✅ Phase 2.2: ThalamicRelayState (22/22 tests) - Commit 815b9e4
- ✅ Phase 2.3: HippocampusState (16/16 tests) - Commit 8a68154
- ✅ Phase 2.4: LayeredCortexState (17/17 tests) - Commit [PENDING]
- **Total Tests Passing**: 143/143 (100%)

---

## Executive Summary

This document outlines a comprehensive refactoring of state management to create unified `RegionState` and `PathwayState` abstract base classes. This refactoring is justified by the user's research requirements:
- **Long-term training** (weeks/months) → Need checkpoint migration
- **Reproducible research** → Need exact state snapshots
- **Reusable brain components** → Need transfer learning support
- **Interactive tools** → Need unified state inspection API

**Key Benefits**:
1. Checkpoint evolution without breaking old saves
2. Exact brain state serialization for reproducibility
3. Transfer learning and component reuse
4. Generic debugging and inspection tools
5. Automated testing infrastructure
6. **[NEW]** Pathway state preservation (in-flight spikes, delay buffers)

**Investigation Results** (December 21, 2025):
- ✅ Striatum architecture fully mapped (no state dataclass, needs consolidation)
- ✅ Pathway state requirements identified (AxonalProjection critical)
- ✅ Oscillator state clarified (transient, not serialized)
- ✅ Biological justifications documented for all state components
- ⚠️ Increased scope: Pathway state infrastructure required (Phase 0)
- ⚠️ Revised timeline: 59-73 hours (was 47-59 hours)

---

## Current State Analysis

### Existing State Implementations

**1. NeuralComponentState (base.py)**
```python
@dataclass
class NeuralComponentState:
    """Base state for all neural components."""
    membrane: Optional[torch.Tensor] = None
    spikes: Optional[torch.Tensor] = None
    eligibility: Optional[torch.Tensor] = None
    dopamine: float = 0.0
    acetylcholine: float = 0.0
    norepinephrine: float = 0.0
    t: int = 0
    # ... more fields
```
- Used as base for all region states
- Has basic fields but **no serialization methods**
- No versioning support

**2. Region-Specific States (all inherit NeuralComponentState)**

| Region | State Class | Location | Fields | Notes |
|--------|-------------|----------|--------|-------|
| Hippocampus | `HippocampusState` | `hippocampus/config.py:234` | 10+ fields | DG/CA3/CA1 spikes, traces, persistent activity |
| Cortex | `LayeredCortexState` | `cortex/config.py:191` | 15+ fields | L4/L2/3/L5/L6a/L6b spikes, traces, modulation |
| Prefrontal | `PrefrontalState` | `prefrontal.py:185` | 4 fields | Working memory, gates, rules |
| Thalamus | `ThalamicRelayState` | `thalamus.py:201` | 8 fields | Relay/TRN spikes, mode, gating |
| Predictive Cortex | `PredictiveCortexState` | `cortex/predictive_cortex.py:114` | Delegates to LayeredCortex | Wrapper state |

**3. Components Without State Classes**

**Cerebellum** (`regions/cerebellum_region.py`):
- Uses raw attributes (no dataclass)
- State includes: error signals, parallel fiber traces, Purkinje cell membrane
- Needs: CerebellumState dataclass

**Striatum** (`regions/striatum/striatum.py`, **INVESTIGATED**):
- **NO dataclass state** - uses raw attributes
- **Has StriatumStateTracker** (`state_tracker.py`) - manages temporal state:
  - D1/D2 vote accumulators [n_actions]
  - Recent spikes for lateral inhibition [n_output]
  - Last action, exploration state, RPE tracking
  - Trial activity statistics
- **Has CheckpointManager** (`checkpoint_manager.py`) - custom serialization:
  - Serializes: neuron_state, pathway_state, learning_state, exploration_state, rpe_state, goal_state, action_state, delay_state
  - Format version: "1.0.0" (elastic tensor format)
  - ~200 lines of manual serialization code
- **D1/D2 Pathway State** (separate populations):
  - Weights stored in parent's `synaptic_weights["default_d1"]` and `synaptic_weights["default_d2"]`
  - Eligibility traces per pathway
  - TD(λ) traces (if enabled)
  - Delay buffers for temporal competition (D1: ~15ms, D2: ~25ms)
- **Needs**: Unified StriatumState dataclass consolidating state_tracker + pathway states

### Checkpoint Managers

**Existing checkpoint implementations**:
1. `regions/striatum/checkpoint_manager.py` - Custom serialization
2. `regions/hippocampus/checkpoint_manager.py` - Custom serialization
3. `regions/prefrontal_checkpoint_manager.py` - Custom serialization

**Problem**: Each uses different patterns, duplicated serialization code (~50 lines each).

---

## Critical Design Decisions

### 1. Multiple Inheritance vs Composition

**Current Plan**: `HippocampusState(NeuralComponentState, RegionState)`

**Risk**: Diamond inheritance if both parents have base class features

**Alternatives Considered**:

**A. Protocol-Based (Recommended)**:
```python
# RegionState as Protocol (no inheritance)
class RegionState(Protocol):
    STATE_VERSION: ClassVar[int]
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data, device): ...
    def reset(self) -> None: ...

# States implement protocol
@dataclass
class HippocampusState:
    # NeuralComponentState fields
    membrane: Optional[torch.Tensor] = None
    spikes: Optional[torch.Tensor] = None

    # Hippocampus-specific
    dg_spikes: Optional[torch.Tensor] = None
    ca3_spikes: Optional[torch.Tensor] = None

    # Implement RegionState protocol
    STATE_VERSION: ClassVar[int] = 1
    def to_dict(self): ...
    @classmethod
    def from_dict(cls, data, device): ...
    def reset(self): ...
```

**Pros**: No inheritance complexity, clear ownership, easier to test
**Cons**: No automatic interface enforcement (requires mypy)

**B. Composition**:
```python
@dataclass
class HippocampusState:
    base: NeuralComponentState
    dg_spikes: Optional[torch.Tensor] = None
    ca3_spikes: Optional[torch.Tensor] = None

    # Delegate to base
    @property
    def membrane(self): return self.base.membrane
```

**Pros**: No inheritance, explicit relationships
**Cons**: More verbose, access indirection

**Decision**: ✅ **CONFIRMED** - Use Protocol-Based approach (Alternative A) to avoid inheritance complexity while maintaining type safety.

**Implementation** (Phase 1):
```python
# src/thalia/core/region_state.py
from typing import Protocol, ClassVar, Dict, Any, Optional
import torch

class RegionState(Protocol):
    """Protocol for region state serialization (no inheritance required).

    Regions implement this protocol by providing these methods.
    Type checkers (mypy, pyright) will verify implementation.
    """
    STATE_VERSION: ClassVar[int]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: Optional[torch.device] = None) -> "RegionState":
        """Deserialize state from checkpoint with automatic migration."""
        ...

    def reset(self) -> None:
        """Reset state to initial values."""
        ...

# Regions implement protocol (no inheritance)
@dataclass
class HippocampusState:
    """Hippocampus state implementing RegionState protocol."""
    STATE_VERSION: ClassVar[int] = 1

    # Base neural state (inline from NeuralComponentState)
    membrane: Optional[torch.Tensor] = None
    spikes: Optional[torch.Tensor] = None
    dopamine: float = 0.0
    acetylcholine: float = 0.0

    # Hippocampus-specific state
    dg_spikes: Optional[torch.Tensor] = None
    ca3_spikes: Optional[torch.Tensor] = None
    ca1_spikes: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Implements RegionState protocol."""
        return {
            "version": self.STATE_VERSION,
            "membrane": self.membrane.cpu() if self.membrane is not None else None,
            "dg_spikes": self.dg_spikes.cpu() if self.dg_spikes is not None else None,
            # ... all fields
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: Optional[torch.device] = None) -> "HippocampusState":
        """Implements RegionState protocol with migration."""
        # Type checker verifies this matches protocol
        ...

    def reset(self) -> None:
        """Implements RegionState protocol."""
        if self.membrane is not None:
            self.membrane.zero_()
        # ... reset all state
```

**Why Protocol over ABC**:
- ✅ No diamond inheritance risk
- ✅ Flexible (can add to existing dataclasses)
- ✅ Type-safe (mypy/pyright verify implementation)
- ✅ No runtime overhead
- ✅ Clear ownership (state fields inline, not inherited)

---

### 2. Trace Serialization Strategy

**Question**: Should traces be serialized with their current decay state or reset to zero?

**Analysis**:
- **Eligibility traces** (tau=100-2000ms): MUST serialize current values
  - Represent molecular tags (CaMKII, PKA) that persist across trials
  - Learning depends on eligibility at reward time
  - Resetting would break credit assignment

- **STDP traces** (tau=10-50ms): SHOULD serialize current values
  - Represent calcium concentrations from recent spikes
  - Short time constant means they decay quickly anyway
  - Preserving avoids transient artifacts after reload

- **CA3 persistent activity**: MUST serialize
  - Active attractor state represents current memory content
  - Resetting would lose working memory state

**Decision**: Serialize all traces AS-IS. Document in RegionState docstring that traces preserve temporal dynamics across checkpoints.

---

### 3. Oscillator State Storage

**Question**: Should oscillator phases be part of RegionState?

**Analysis**:
- **Current approach** (investigated): Oscillators managed globally by Brain
  - Regions receive phases via `set_oscillator_phases()` each timestep
  - Phases stored as instance attributes (`_theta_phase`, etc.)
  - Oscillators restart from t=0 after checkpoint load

**Why NOT serialize**:
- Oscillator phase is **transient** (recomputed every timestep)
- Brain re-broadcasts phases immediately after checkpoint load
- Preserving phase would require storing Brain's oscillator state
- Phase mismatch between regions creates coupling artifacts (worse than reset)

**Decision**: Oscillator phases are NOT part of RegionState. Document that oscillators restart from t=0 after load. This is biologically plausible (network resynchronization after perturbation).

---

### 4. Delay Buffer Growth

**Question**: How do delay buffers handle source region growth?

**Current gap**: AxonalProjection has no `grow_source()` method documented

**Required behavior**:
```python
# Cortex L5 grows from 128 → 148 neurons
cortex.grow_output(20)

# Pathway must expand delay buffer
projection.grow_source("cortex", "l5", new_size=148)
# Internally: expand buffer from [delay, 128] → [delay, 148]
# Preserve existing spikes, initialize new axons with zeros
```

**Implementation** (Phase 0):
```python
class CircularDelayBuffer:
    def grow(self, new_size: int) -> None:
        \"\"\"Expand buffer to accommodate more neurons.\"\"\"
        old_size = self.size
        n_new = new_size - old_size

        # Expand buffer [max_delay+1, old_size] → [max_delay+1, new_size]
        new_buffer = torch.zeros(
            (self.max_delay + 1, new_size),
            dtype=self.dtype,
            device=self.device,
        )
        new_buffer[:, :old_size] = self.buffer
        self.buffer = new_buffer
        self.size = new_size
```

**Decision**: Add delay buffer growth support in Phase 0. Document growth protocol in PathwayState docstring.

---

## Reviewed Architecture

After investigation and design review, the architecture has been refined:

### Why These State Components Exist

**Membrane Potentials** (`membrane: torch.Tensor`):
- **Biological**: Voltage across neuronal membrane due to ion channel dynamics
- **Persistence**: Decays with time constant τ_mem (~10-30ms) but carries information across timesteps
- **Checkpoint Need**: Ongoing depolarization affects next spike timing

**Eligibility Traces** (`eligibility: torch.Tensor`):
- **Biological**: Synaptic tags marking recently-active synapses (molecular cascades: CaMKII, PKA)
- **Time constant**: 100-2000ms (Yagishita et al., 2014)
- **Purpose**: Bridge temporal gap between action and delayed reward
- **Checkpoint Need**: Eligibility persists across trials - must save for credit assignment

**STDP Traces** (`*_trace: torch.Tensor`):
- **Biological**: Pre/post-synaptic calcium concentrations (from NMDA receptors)
- **Time constant**: 10-50ms (Bi & Poo, 1998)
- **Purpose**: Detect spike timing coincidence for STDP learning
- **Checkpoint Need**: Partial traces carry learning momentum

**CA3 Persistent Activity** (Hippocampus):
- **Biological**: Recurrent excitation maintains activity after stimulus offset
- **Mechanism**: Strong CA3→CA3 collaterals create attractor dynamics
- **Purpose**: Pattern completion and short-term buffer
- **Checkpoint Need**: Ongoing attractor state represents current memory content

**Dopamine/ACh/NE Levels** (region-local effects):
- **Biological**: Neuromodulators bind to receptors, altering excitability and plasticity
- **Time constant**: 100-1000ms (slow diffusion, slow receptor unbinding)
- **Purpose**: Context-dependent learning (DA=reward, ACh=attention, NE=arousal)
- **Checkpoint Need**: Modulator effects persist, influencing next learning event

**Axonal Delay Buffers** (PathwayState):
- **Biological**: Action potentials propagate at finite velocity (1-100 m/s)
- **Delays**: Distance/velocity = delay (thalamus: 2-5ms, cortex: 1-10ms)
- **Purpose**: Temporal dynamics (e.g., D1 arrives before D2 in striatum)
- **Checkpoint Need**: In-flight spikes lost without serialization

**D1/D2 Pathway Delays** (Striatum, investigated):
- **Biological**: Direct pathway (D1): Striatum→GPi→Thalamus (~15ms)
- **Biological**: Indirect pathway (D2): Striatum→GPe→STN→GPi→Thalamus (~25ms)
- **Purpose**: Temporal competition creates action selection dynamics
- **Checkpoint Need**: Delay buffer state affects next action selection

**Oscillator Phases** (NOT checkpointed):
- **Biological**: LFP oscillations from network dynamics (theta: 4-8 Hz, gamma: 30-80 Hz)
- **Time constant**: Continuous oscillation (self-sustaining)
- **Purpose**: Coordinate timing across regions (e.g., theta-gamma coupling)
- **No checkpoint need**: Oscillators restart from t=0 (phase is transient)

---

## Proposed Architecture

### 1. RegionState Abstract Base Class

```python
# src/thalia/core/region_state.py (NEW FILE)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, ClassVar
import torch

@dataclass
class RegionState(ABC):
    """Abstract base class for all region state with standardized serialization.

    This provides:
    1. Versioned serialization (to_dict/from_dict)
    2. State reset capabilities
    3. Automatic checkpoint migration
    4. Type-safe state management

    Design Principles:
    - All tensor fields should be Optional (for partial state loading)
    - Version field enables automatic migration
    - Subclasses must implement to_dict/from_dict/reset

    Example:
        @dataclass
        class MyRegionState(RegionState):
            STATE_VERSION: ClassVar[int] = 1

            my_tensor: Optional[torch.Tensor] = None
            my_float: float = 0.0

            def to_dict(self) -> Dict[str, Any]:
                return {
                    "version": self.STATE_VERSION,
                    "my_tensor": self.my_tensor.cpu() if self.my_tensor is not None else None,
                    "my_float": self.my_float,
                }

            @classmethod
            def from_dict(cls, data: Dict[str, Any], device: torch.device) -> "MyRegionState":
                version = data.get("version", 1)

                # Migration example
                if version < 2:
                    data = cls._migrate_v1_to_v2(data)

                return cls(
                    my_tensor=data["my_tensor"].to(device) if data["my_tensor"] is not None else None,
                    my_float=data["my_float"],
                )

            def reset(self) -> None:
                if self.my_tensor is not None:
                    self.my_tensor.zero_()
                self.my_float = 0.0

    Author: Thalia Project
    Date: December 2025
    """

    # Subclasses should define this
    STATE_VERSION: ClassVar[int] = 1

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary.

        Must include:
        - "version": int - STATE_VERSION for migration support
        - All state fields in serializable format
        - Tensors should be moved to CPU

        Returns:
            Dictionary with all state data
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], device: Optional[torch.device] = None) -> "RegionState":
        """Deserialize state from dictionary with automatic migration.

        Should:
        1. Check version field
        2. Apply migrations if needed (call _migrate_vX_to_vY)
        3. Move tensors to specified device
        4. Construct and return instance

        Args:
            data: Serialized state dictionary
            device: Target device for tensors (default: CPU)

        Returns:
            New instance with loaded state
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset state to initial values.

        Should:
        - Zero tensors (or reinitialize to appropriate values)
        - Reset scalars to defaults
        - Preserve tensor shapes/devices
        """
        pass

    def diff(self, other: "RegionState") -> Dict[str, Any]:
        """Compute difference between states (for incremental checkpointing).

        Default implementation compares to_dict() outputs.
        Subclasses can override for custom diff logic.

        Args:
            other: State to compare against

        Returns:
            Dictionary with only changed fields
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        diff = {}
        for key, value in self_dict.items():
            other_value = other_dict.get(key)
            if not self._values_equal(value, other_value):
                diff[key] = value

        return diff

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """Compare values handling tensors specially."""
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.equal(a, b)
        elif a is None and b is None:
            return True
        elif type(a) != type(b):
            return False
        else:
            return a == b

    # Migration helpers (subclasses can add _migrate_vX_to_vY methods)
    @classmethod
    def _apply_migrations(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all necessary migrations to bring data to current version.

        Example:
            if version < 2:
                data = cls._migrate_v1_to_v2(data)
            if version < 3:
                data = cls._migrate_v2_to_v3(data)
        """
        version = data.get("version", 1)
        current_version = cls.STATE_VERSION

        if version >= current_version:
            return data

        # Subclasses should implement specific migration methods
        # e.g., _migrate_v1_to_v2, _migrate_v2_to_v3, etc.
        for v in range(version, current_version):
            migration_method = f"_migrate_v{v}_to_v{v+1}"
            if hasattr(cls, migration_method):
                data = getattr(cls, migration_method)(data)
                data["version"] = v + 1

        return data
```

### 2. Integration with NeuralComponentState

**Option A: RegionState inherits from NeuralComponentState**
```python
@dataclass
class RegionState(NeuralComponentState, ABC):
    """Combines base component state with serialization protocol."""
    # Inherits: membrane, spikes, dopamine, etc.
    # Adds: to_dict, from_dict, reset, versioning
```

**Option B: RegionState as parallel hierarchy (RECOMMENDED)**
```python
# RegionState is independent ABC
# Specific states inherit from BOTH:
@dataclass
class HippocampusState(NeuralComponentState, RegionState):
    """Hippocampus state with serialization."""
```

**Rationale for Option B**:
- More flexible (not all components need RegionState)
- Clearer separation of concerns
- Easier to adopt incrementally

---

## Implementation Phases

### Phase 0: Pathway State Foundation (6-8 hours) ✅ **COMPLETED**

**Status**: ✅ COMPLETED (December 21, 2025)

**Rationale**: Pathways (especially AxonalProjection) need state management parallel to regions.

**Completed Tasks**:
1. ✅ Created `src/thalia/core/pathway_state.py` with PathwayState protocol
2. ✅ Implemented AxonalProjectionState for delay buffer serialization
3. ✅ Updated AxonalProjection with `get_state()` and `load_state()` methods
4. ✅ Wrote comprehensive unit tests (13 tests, all passing)

**Deliverables**:
- ✅ `src/thalia/core/pathway_state.py` (~250 lines)
- ✅ Updated `src/thalia/pathways/axonal_projection.py` (+40 lines)
- ✅ `tests/unit/core/test_pathway_state.py` (~450 lines, 13 tests)

**Test Results**:
```
13 passed in 4.82s

Key test coverage:
- State creation and serialization roundtrip
- Multi-source pathway state preservation
- Device transfer (CPU ↔ GPU)
- Delay buffer preservation with in-flight spikes
- Pointer position accuracy
- Backward compatibility (get_full_state/load_full_state)
```

**Validation**:
- [x] Delay buffer roundtrip test passes (spikes preserved)
- [x] Multiple sources with different delays serialize correctly
- [x] In-flight spikes restored at correct buffer positions
- [x] Pointer state preserved
- [x] Device management works (CPU/CUDA)

**Why First**: Regions depend on pathways; pathway state must exist before region state can reference it.

**Next**: Phase 1 (RegionState Foundation) OR Cerebellum STP (parallel track)

---

### Phase 1: Foundation (4-6 hours) ✅ **COMPLETED**

**Status**: ✅ COMPLETED (December 21, 2025)

**Tasks**:
1. ✅ Create `src/thalia/core/region_state.py` with RegionState ABC
2. ✅ Add comprehensive docstrings and type hints
3. ✅ Write unit tests for base class (validate_state_protocol, utility functions)

**Deliverables**:
- ✅ `src/thalia/core/region_state.py` (~329 lines)
- ✅ `tests/unit/core/test_region_state.py` (~475 lines, 24 tests)

**Test Results**:
```
24 passed in 1.2s

Key test coverage:
- BaseRegionState initialization and protocol methods
- to_dict/from_dict serialization roundtrip
- Device transfer (CPU ↔ CUDA)
- State reset functionality
- File I/O with utility functions
- Protocol validation
- Edge cases (None fields, nested dicts, partial fields)
```

**Validation**:
- [x] Base class tests pass
- [x] Type hints validate with pyright
- [x] Docstrings follow project conventions

---

### Phase 2: Migrate Existing Dataclass States (12-16 hours)

**Order of Implementation** (easiest to hardest):

#### 2.1 PrefrontalState (2 hours) ✅ **COMPLETED**

**Status**: ✅ COMPLETED (December 21, 2025)

- **Why first**: Simplest state (4 fields + neuromodulators)
- **Files**: `regions/prefrontal.py`
- **Completed Changes**:
  - ✅ PrefrontalState inherits from BaseRegionState
  - ✅ Added missing neuromodulator fields (acetylcholine, norepinephrine)
  - ✅ Implemented to_dict() serializing all 9 state fields
  - ✅ Implemented from_dict() with device transfer
  - ✅ Implemented reset() resetting all fields
  - ✅ Updated get_state() and load_state() in Prefrontal region
  - ✅ Added STP state field for recurrent connections

**Test Results**:
```
22/22 tests passing (100%)

Key test coverage:
- Protocol compliance validation
- Integration with Prefrontal region
- Device transfer (CPU/CUDA)
- File I/O with utility functions
- STP state persistence
- Working memory state preservation
- Dopamine gating state (handles DA dynamics)
- Edge cases (None fields, nested STP dicts)
```

**Commit**: 917f7ba

#### 2.2 ThalamicRelayState (3 hours) ✅ **COMPLETED**

**Status**: ✅ COMPLETED (December 21, 2025)

- **Why second**: Moderate complexity (14 fields, dual STP pathways)
- **Files**: `regions/thalamus.py`
- **Completed Changes**:
  - ✅ ThalamicRelayState inherits from BaseRegionState
  - ✅ Added explicit neuromodulator fields (dopamine, acetylcholine, norepinephrine)
  - ✅ Implemented to_dict() serializing all 14 state fields
  - ✅ Implemented from_dict() with device transfer (signature fixed to match base)
  - ✅ Implemented reset() as in-place mutation (returns None)
  - ✅ Updated get_state() and load_state() in ThalamicRelay region
  - ✅ Added dual STP state fields (sensory→relay and L6→relay pathways)

**Special Considerations**:
- Dual STP modules: sensory→relay (U=0.4, moderate depression) and L6→relay (U=0.7, strong depression)
- L6 STP state may have None tensors if no L6 input provided (expected behavior)
- Burst/tonic mode state preserved across checkpoints
- Alpha oscillation gating state captured for attention modulation
- Method signatures must match BaseRegionState (from_dict device as str, reset() returns None)

**Test Results**:
```
22/22 tests passing (100%)
110/110 total state management tests passing (100%)

Key test coverage:
- Protocol compliance validation
- Integration with ThalamicRelay region
- Device transfer (CPU/CUDA)
- File I/O with utility functions
- Dual STP state persistence (sensory and L6 feedback)
- Relay and TRN neuron state preservation
- Burst/tonic mode state preservation
- Alpha gating state preservation
- Edge cases (None fields, nested STP dicts, partial fields)
```

**Commit**: 815b9e4

#### 2.3 HippocampusState (4 hours) ✅ **COMPLETED**

**Status**: ✅ COMPLETED (December 21, 2025)
**Duration**: ~4 hours
**Commit**: 8a68154- **Files**: `regions/hippocampus/config.py`, `regions/hippocampus/trisynaptic.py`
- **Completed Changes**:
  - ✅ HippocampusState inherits from BaseRegionState
  - ✅ Added explicit neuromodulator fields (dopamine=0.2, acetylcholine=0.0, norepinephrine=0.0)
  - ✅ Implemented to_dict() serializing all 19 state fields (15 base + 4 STP)
  - ✅ Implemented from_dict() with device transfer (device as str parameter)
  - ✅ Implemented reset() as in-place mutation (returns None)
  - ✅ Updated get_state() to capture STP state from all 4 pathways
  - ✅ Created load_state() to restore complete hippocampus state
  - ✅ Added 4 STP state fields for all pathways

**Special Considerations**:
- **Complex State**: 11 hippocampus-specific fields + 4 STP state dicts = 15 total fields
- **4 STP Pathways**: Each with nested dict (`u`, `x` tensors):
  * `stp_mossy_state` (DG→CA3, facilitation U=0.1→0.9)
  * `stp_schaffer_state` (CA3→CA1, depression U=0.5)
  * `stp_ec_ca1_state` (EC→CA1 direct, depression U=0.4)
  * `stp_ca3_recurrent_state` (CA3 recurrent, depression U=0.3)
- **CA3 Persistent Activity**: `ca3_persistent` tensor preserves attractor dynamics across checkpoints
- **Memory Traces**: Sample trace, DG trace, CA3 trace, NMDA trace all preserved
- **Trisynaptic Circuit**: DG→CA3→CA1 activity patterns fully captured

**Test Results**:
```
16/16 tests passing (100%)
126/126 total state management tests passing (100%)

Test Coverage:
- Protocol compliance (6 tests): to_dict, from_dict, reset, device transfer with nested STP dicts
- Integration (4 tests): get_state captures 4 STP states, load_state restores all, round-trip consistency
- File I/O (2 tests): save/load with utility functions, CUDA→CPU transfer
- Edge cases (4 tests): None tensors, missing STP modules, partial fields, all 4 pathways populated
```

**Complexity Notes**:
- Most complex migration so far: 4 STP pathways (vs 1-2 in previous phases)
- Nested dict serialization/deserialization for all STP states
- CA3 persistent activity trace (bistable neuron dynamics)
- Memory encoding traces (sample, DG, CA3, NMDA)
- Feedforward inhibition strength preservation

#### 2.4 LayeredCortexState (5 hours) ✅ **COMPLETED**

**Status**: ✅ COMPLETED (December 21, 2025)
**Duration**: ~5 hours
**Commit**: [PENDING]

- **Why last**: Most complex (15+ fields, multi-layer)
- **Files**: `regions/cortex/config.py`, `regions/cortex/layered_cortex.py`
- **Completed Changes**:
  - ✅ LayeredCortexState inherits from BaseRegionState
  - ✅ Added explicit neuromodulator fields (dopamine=0.0, acetylcholine=0.0, norepinephrine=0.0)
  - ✅ Implemented to_dict() serializing all 23 state fields (19 base + 1 STP + 3 scalars)
  - ✅ Implemented from_dict() with device transfer (device as str parameter)
  - ✅ Implemented reset() as in-place mutation (returns None)
  - ✅ Created get_state() to capture complete cortex state
  - ✅ Created load_state() to restore complete cortex state
  - ✅ Added STP state field for L2/3 recurrent pathway

**Special Considerations**:
- **Complex State**: 23 total fields across 6 layers (L4, L2/3, L5, L6a, L6b)
- **6 Layer Spike States**: Each with independent spike patterns
- **5 STDP Trace Fields**: One per layer for spike-timing dependent plasticity
- **L2/3 Recurrent Activity**: Accumulator for lateral connections
- **Top-Down Modulation**: From higher cortical areas
- **Feedforward Inhibition (FFI) Strength**: Dynamic inhibition control
- **Alpha Suppression**: Oscillatory attention modulation (1.0 = no suppression)
- **Gamma Attention**: Phase and per-neuron gating
- **1 STP Pathway**: L2/3 recurrent (depression, U=0.5) - CRITICAL for WM flexibility
- **Plasticity Monitoring**: last_plasticity_delta tracks continuous learning

**Test Results**:
```
17/17 tests passing (100%)
143/143 total state management tests passing (100%)

Test Coverage:
- Protocol compliance (7 tests): to_dict, from_dict, reset, device transfer with STP dict
- Integration (4 tests): get_state captures STP, load_state restores all, round-trip consistency
- File I/O (2 tests): save/load with utility functions, CUDA→CPU transfer
- Edge cases (4 tests): None tensors, missing STP, all 6 layers populated, 2D STP tensors
```

**Complexity Notes**:
- Most fields of any state so far: 23 total (vs 19 for hippocampus, 14 for thalamus)
- 6-layer architecture requires careful spike routing
- Gamma attention adds phase-based gating dynamics
- Alpha suppression modulates layer responsiveness
- STP on L2/3 recurrent enables flexible working memory
- Top-down modulation enables hierarchical predictive coding

**Validation for each**:
- [ ] State roundtrip: `s2 = StateClass.from_dict(s1.to_dict(), device)` → `s1 == s2`
- [ ] Reset works: `s.reset()` zeros appropriate fields
- [ ] Tensor devices preserved
- [ ] All tests pass

---

### Phase 3: Convert Dict-Based Regions (8-10 hours)

**Regions without dataclass states**:

#### 3.1 Cerebellum (3 hours)
- **Current**: Raw attributes, no dataclass
- **Create**: `CerebellumState(NeuralComponentState, RegionState)`
- **Files**: `regions/cerebellum_region.py`
- **New fields**: Error signals, parallel fiber traces, Purkinje cell state

#### 3.2 Striatum (8 hours) - **INVESTIGATED** ✅
- **Current**: NO state dataclass, has StriatumStateTracker + CheckpointManager
- **Files**: `regions/striatum/striatum.py` (1979 lines), `checkpoint_manager.py` (651 lines), `state_tracker.py` (297 lines)
- **Investigation results**:
  - ❌ No StriatumState dataclass exists
  - ✅ StriatumStateTracker manages temporal state (votes, action, exploration)
  - ✅ CheckpointManager has custom serialization (~200 lines)
  - ✅ D1/D2 have separate weight matrices (`synaptic_weights["default_d1/d2"]`)
  - ✅ D1/D2 delay buffers for temporal competition (15ms vs 25ms)
- **Create**: `StriatumState` consolidating:
  - D1/D2 pathway states (weights, eligibility, TD(λ) traces)
  - Vote accumulators and recent spikes (from state_tracker)
  - Exploration state (from exploration component)
  - RPE and value estimates
  - Goal modulation weights (PFC→Striatum)
  - D1/D2 delay buffers (CircularDelayBuffer state)
- **Complexity**: High due to D1/D2 separation and multiple component states

**Validation**:
- [ ] Regions initialize without errors
- [ ] State save/load works
- [ ] Existing functionality preserved

---

### Phase 4: Update Checkpoint Managers (8-10 hours)

**Files to update**:
1. `regions/striatum/checkpoint_manager.py`
2. `regions/hippocampus/checkpoint_manager.py`
3. `regions/prefrontal_checkpoint_manager.py`

**Changes**:

**Before**:
```python
def save_state(self, region):
    return {
        "membrane": region.state.membrane.cpu() if region.state.membrane is not None else None,
        "spikes": region.state.spikes.cpu() if region.state.spikes is not None else None,
        # ... 20+ more lines of manual serialization ...
    }

def load_state(self, region, data):
    if data["membrane"] is not None:
        region.state.membrane = data["membrane"].to(region.device)
    if data["spikes"] is not None:
        region.state.spikes = data["spikes"].to(region.device)
    # ... 20+ more lines of manual deserialization ...
```

**After**:
```python
def save_state(self, region):
    return region.state.to_dict()

def load_state(self, region, data):
    region.state = type(region.state).from_dict(data, region.device)
```

**Eliminated**: ~150 lines of duplicated serialization code

**Validation**:
- [ ] Old checkpoints can still be loaded (backward compatibility)
- [ ] New checkpoints save correctly
- [ ] Checkpoint migration test passes

---

### Phase 5: Version Migration Infrastructure (4-6 hours)

**Implement migration examples for:**

#### 5.1 HippocampusState v1 → v2 Example
```python
@dataclass
class HippocampusState(NeuralComponentState, RegionState):
    STATE_VERSION: ClassVar[int] = 2  # Bumped from 1

    # NEW in v2: Separate CA3 recurrent gain
    ca3_recurrent_gain: float = 1.0

    @classmethod
    def _migrate_v1_to_v2(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migration: Add ca3_recurrent_gain field."""
        data["ca3_recurrent_gain"] = 1.0  # Default value
        return data
```

#### 5.2 LayeredCortexState v1 → v2 Example
```python
@dataclass
class LayeredCortexState(NeuralComponentState, RegionState):
    STATE_VERSION: ClassVar[int] = 2

    # NEW in v2: L6b feedback pathway
    l6b_spikes: Optional[torch.Tensor] = None
    l6b_trace: Optional[torch.Tensor] = None

    @classmethod
    def _migrate_v1_to_v2(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migration: Add L6b fields."""
        data["l6b_spikes"] = None
        data["l6b_trace"] = None
        return data
```

**Documentation**:
- Create `docs/patterns/state-management.md`
- Document migration process
- Provide migration examples
- Explain versioning strategy

**Validation**:
- [ ] v1 checkpoints load with v2 code
- [ ] Migrations apply correctly
- [ ] Migration tests pass

---

### Phase 6: Comprehensive Testing (6-8 hours)

#### 6.1 Unit Tests (`tests/unit/core/test_region_state.py`)
```python
def test_state_roundtrip_serialization():
    """Test all states serialize and deserialize correctly."""

def test_state_reset():
    """Test reset() zeroes appropriate fields."""

def test_state_diff():
    """Test diff() identifies changes."""

def test_version_migration():
    """Test automatic version migration."""

def test_tensor_device_handling():
    """Test tensors move to correct device."""
```

#### 6.2 Property-Based Tests
```python
from hypothesis import given, strategies as st

@given(st.floats(), st.integers())
def test_state_preserves_values(float_val, int_val):
    """Property: Roundtrip preserves all values."""
```

#### 6.3 Integration Tests
```python
def test_checkpoint_save_load_with_new_api():
    """Test full checkpoint workflow."""

def test_transfer_learning():
    """Test transplanting state between brains."""

def test_pathway_delay_preservation():
    """Test in-flight spikes preserved across checkpoint."""
    # Create projection with delays
    projection = AxonalProjection(
        sources=[("cortex", "l5", 128, 5.0)],  # 5ms delay
        device="cpu", dt_ms=1.0
    )

    # Send spikes at t=0, t=1, t=2
    for t in range(3):
        spikes = torch.rand(128) > 0.9  # Sparse spikes
        projection.forward({"cortex:l5": spikes})

    # Save state (spikes in-flight)
    state = projection.get_state()

    # Load into new projection
    projection2 = AxonalProjection(
        sources=[("cortex", "l5", 128, 5.0)],
        device="cpu", dt_ms=1.0
    )
    projection2.load_state(state)

    # Continue for 5 more steps - should see delayed spikes emerge
    for t in range(5):
        output = projection2.forward({"cortex:l5": torch.zeros(128)})
        # Assert delayed spikes appear at correct times

def test_striatum_d1_d2_delay_buffers():
    """Test D1/D2 temporal competition preserved across checkpoint."""
    # D1 arrives at t=15ms, D2 at t=25ms
    # Save checkpoint during delay window, verify competition dynamics preserved
```

#### 6.4 Biological Validity Tests (**NEW**)
```python
def test_eligibility_trace_decay_after_load():
    """Verify eligibility traces decay correctly after checkpoint load."""
    striatum = create_striatum()

    # Build eligibility
    for _ in range(10):
        striatum.forward(test_input)

    # Check eligibility is high
    assert striatum.d1_pathway.eligibility.max() > 0.5

    # Save and load
    state = striatum.get_full_state()
    striatum.load_full_state(state)

    # Continue simulation - eligibility should decay with tau_ms
    initial_elig = striatum.d1_pathway.eligibility.clone()
    for _ in range(100):  # 100ms
        striatum.forward(torch.zeros(striatum.config.n_input))

    # After 100ms with tau=100ms, should decay to ~37% (e^-1)
    final_elig = striatum.d1_pathway.eligibility
    expected_ratio = np.exp(-1)
    actual_ratio = final_elig.mean() / initial_elig.mean()
    assert 0.3 < actual_ratio < 0.45  # Allow 20% tolerance

def test_membrane_potential_range():
    """Verify membrane potentials stay in biologically valid range."""
    region = create_test_region()

    # Checkpoint and restore multiple times
    for _ in range(5):
        region.forward(test_input)
        state = region.get_full_state()
        region.load_full_state(state)

    # Check membrane in valid range
    V = region.neurons.membrane
    assert (V >= -80).all(), "Hyperpolarization below K+ reversal"
    assert (V <= 50).all(), "Depolarization above Na+ reversal"

def test_neuromodulator_bounds():
    """Verify neuromodulator levels stay in [0, 1] range."""
    striatum = create_striatum()

    # Multiple learning cycles
    for _ in range(100):
        striatum.forward(test_input)
        striatum.deliver_reward(np.random.uniform(-1, 1))

    # Save/load
    state = striatum.get_full_state()
    striatum.load_full_state(state)

    # Check dopamine in valid range
    assert 0 <= striatum.tonic_dopamine <= 1.5, "Dopamine out of bio range"

def test_no_negative_firing_rates():
    """Verify no negative spike counts after state restoration."""
    region = create_test_region()

    # Run simulation
    for _ in range(50):
        spikes = region.forward(test_input)
        assert (spikes >= 0).all(), "Negative spikes detected"

    # Save and load
    state = region.get_full_state()
    region.load_full_state(state)

    # Continue - still no negative spikes
    for _ in range(50):
        spikes = region.forward(test_input)
        assert (spikes >= 0).all(), "Negative spikes after load"
```

**Validation**:
- [ ] All unit tests pass (target: 95%+ coverage)
- [ ] Property-based tests pass (1000+ examples)
- [ ] Integration tests pass
- [ ] **Biological validity tests pass** (new requirement)
- [ ] No regression in existing tests

---

### Phase 7: Documentation (3-4 hours)

#### 7.1 Pattern Documentation (`docs/patterns/state-management.md`)
```markdown
# State Management Pattern

## Overview
All region states inherit from `RegionState` ABC...

## Adding New State Fields
1. Add field to dataclass
2. Bump STATE_VERSION
3. Implement migration method
4. Update to_dict/from_dict
5. Add tests

## Migration Example
...

## Best Practices
- Always version your state
- Test migrations thoroughly
- Document breaking changes
- Preserve backward compatibility when possible
```

#### 7.2 API Documentation
- Update `docs/api/MODULE_EXPORTS.md`
- Add `RegionState` to component catalog
- Document state lifecycle

#### 7.3 Migration Guide for Users
```markdown
# Migrating to New State Management

## For Checkpoint Users
Your old checkpoints will load automatically...

## For Region Authors
If you've created custom regions...
```

---

## Risk Mitigation

### High-Risk Areas

**1. Checkpoint Backward Compatibility**
- **Risk**: Old checkpoints fail to load
- **Mitigation**:
  - Implement automatic v1 → v2 migration
  - Test with real saved checkpoints from current codebase
  - Provide manual migration script if needed
  - **[NEW]** Document checkpoint format changes in ADR

**2. Tensor Device Management**
- **Risk**: Tensors end up on wrong device
- **Mitigation**:
  - Always pass device to from_dict()
  - Test device handling explicitly (CPU↔GPU transfers)
  - Validate device in integration tests
  - **[NEW]** Add device assertion in from_dict() method

**3. Pathway State Serialization**
- **Risk**: In-flight spikes lost, breaking temporal dynamics
- **Mitigation**:
  - Comprehensive delay buffer serialization tests
  - Test checkpoint during active transmission
  - Verify spike timing preserved after load
  - **[NEW]** Add integration test for multi-source delays

**4. Striatum D1/D2 State Complexity**
- **Risk**: State consolidation breaks D1/D2 learning dynamics
- **Mitigation**:
  - Incremental migration (state_tracker → StriatumState)
  - Keep existing checkpoint_manager alongside new system
  - Test three-factor learning before and after migration
  - **[NEW]** Add D1/D2 opponent process validation test

**5. Eligibility Trace Dynamics**
- **Risk**: Trace serialization disrupts learning momentum
- **Mitigation**:
  - Test eligibility decay after checkpoint load
  - Verify credit assignment across checkpoint boundaries
  - Monitor learning curves for discontinuities
  - **[NEW]** Add biological validity test for trace decay

**6. Breaking Existing Code**
- **Risk**: Region code breaks with new API
- **Mitigation**:
  - Implement incrementally (region by region)
  - Run full test suite after each region
  - Keep old code commented during transition
  - **[NEW]** Add migration checklist per region

### Testing Strategy

**Progressive Validation**:
1. After Phase 1: Base class works
2. After Phase 2.1: One region works (PFC)
3. After Phase 2.2: Two regions work (PFC + Thalamus)
4. After Phase 2.4: All major regions work
5. After Phase 4: All checkpoints work

**Rollback Plan**:
- Git branching strategy:
  ```
  main
    └── feature/state-management-refactor
         ├── phase1-base-class
         ├── phase2-migrate-states
         └── phase3-dict-regions
  ```
- Each phase can be reverted independently

---

## Success Criteria

### Must Have (Phase 1-4)
- [ ] RegionState ABC implemented and tested
- [ ] All existing dataclass states migrated
- [ ] Dict-based regions converted to dataclasses
- [ ] Checkpoint managers use new API
- [ ] All 42+ existing tests pass
- [ ] Zero breaking changes for end users

### Should Have (Phase 5-6)
- [ ] Version migration infrastructure working
- [ ] Comprehensive test suite (95%+ coverage)
- [ ] Property-based tests implemented
- [ ] Integration tests for transfer learning

### Nice to Have (Phase 7)
- [ ] Complete pattern documentation
- [ ] Migration guide for users
- [ ] Examples of version migration
- [ ] Interactive debugging tools

---

## Timeline Estimate

| Phase | Description | Hours | Dependencies |
|-------|-------------|-------|--------------|
| 0 | **Pathway State Foundation** | 6-8 | None |
| 1 | Foundation (RegionState ABC) | 4-6 | Phase 0 |
| 2.1 | Migrate PrefrontalState | 2 | Phase 1 |
| 2.2 | Migrate ThalamicRelayState | 3 | Phase 1 |
| 2.3 | Migrate HippocampusState | 4 | Phase 1 |
| 2.4 | Migrate LayeredCortexState | 5 | Phase 1 |
| 3.1 | Convert Cerebellum | 3 | Phase 2 |
| 3.2 | Convert Striatum **(revised)** | 8 | Phase 2 |
| 4 | Update checkpoint managers | 8-10 | Phase 2-3 |
| 5 | Version migration | 4-6 | Phase 4 |
| 6 | Comprehensive testing | 8-10 | Phase 5 |
| 7 | Documentation | 4-5 | Phase 6 |
| **Total** | | **59-73 hours** | |

**Realistic timeline**: 3-4 weeks of focused work (revised from 2-3 weeks)

**Investigation Summary** (December 21, 2025):
- ✅ Striatum architecture fully documented (D1/D2 separation, state_tracker, checkpoint_manager)
- ✅ Pathway state requirements identified (AxonalProjection delay buffers critical)
- ✅ Oscillator state clarified (transient, not checkpointed)
- ✅ Neuromodulator scope defined (local effects vs global levels)
- ⚠️ Increased complexity: Pathway state infrastructure needed (Phase 0)
- ⚠️ Striatum more complex than expected (8 hours vs 5 hours)
- ⚠️ Testing needs more time for pathway + region integration (8-10 hours vs 6-8)

---

## Commit Strategy

**Commit after each phase** with descriptive messages:

```bash
# Phase 1
git commit -m "feat(state): Add RegionState abstract base class with versioning

- Created src/thalia/core/region_state.py
- Implements to_dict/from_dict/reset protocol
- Adds version migration infrastructure
- Includes comprehensive docstrings and type hints

Tests: tests/unit/core/test_region_state.py
Refs: #XXX"

# Phase 2.1
git commit -m "refactor(prefrontal): Migrate PrefrontalState to RegionState

- PrefrontalState now inherits from RegionState
- Implements to_dict/from_dict/reset methods
- Version 1 baseline established
- All prefrontal tests passing

Breaking changes: None (backward compatible)
Refs: #XXX"

# ... etc for each phase
```

---

## Review Checkpoints

### After Phase 1
**Reviewer should verify**:
- [ ] RegionState API is clean and usable
- [ ] Docstrings are comprehensive
- [ ] Type hints are correct
- [ ] Base tests cover edge cases

### After Phase 2
**Reviewer should verify**:
- [ ] All migrated states work correctly
- [ ] Serialization is correct
- [ ] No performance regression
- [ ] Tests are comprehensive

### After Phase 4
**Reviewer should verify**:
- [ ] Old checkpoints still load
- [ ] New checkpoints save correctly
- [ ] Code duplication eliminated
- [ ] API is consistent

### Final Review
**Reviewer should verify**:
- [ ] All success criteria met
- [ ] Documentation is complete
- [ ] No breaking changes
- [ ] Performance is acceptable

---

## Pathway State Management (CRITICAL ADDITION)

### AxonalProjection State

**Investigation Result**: AxonalProjection has NO state dataclass but MUST serialize delay buffers.

Location: `src/thalia/pathways/axonal_projection.py`

**Current State**:
- `_delay_buffers: Dict[str, CircularDelayBuffer]` - one per source
- CircularDelayBuffer (`utils/delay_buffer.py`):
  - `buffer: torch.Tensor` [max_delay+1, size] - stores in-flight spikes
  - `ptr: int` - current write position
  - `max_delay: int`, `size: int`, `device: str`, `dtype: torch.dtype`

**Why Critical**:
- Delay buffers contain **in-flight spikes** that haven't reached target yet
- Without serialization, spikes are lost across checkpoint boundaries
- Biological accuracy depends on proper axonal delays (thalamus: 2-5ms, cortex L6→TRN: 3ms, etc.)

**Solution**: Create `PathwayState` ABC parallel to `RegionState`

```python
@dataclass
class PathwayState(ABC):
    """Abstract base for pathway state (parallel to RegionState)."""
    STATE_VERSION: ClassVar[int] = 1

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device) -> "PathwayState": ...

    @abstractmethod
    def reset(self) -> None: ...


@dataclass
class AxonalProjectionState(PathwayState):
    """State for AxonalProjection with delay buffers."""
    STATE_VERSION: ClassVar[int] = 1

    # Dict mapping source_key -> (buffer_tensor, ptr, max_delay, size)
    delay_buffers: Dict[str, Tuple[torch.Tensor, int, int, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "delay_buffers": {
                key: {
                    "buffer": buf.cpu(),
                    "ptr": ptr,
                    "max_delay": max_delay,
                    "size": size,
                }
                for key, (buf, ptr, max_delay, size) in self.delay_buffers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device) -> "AxonalProjectionState":
        delay_buffers = {}
        for key, buf_data in data["delay_buffers"].items():
            delay_buffers[key] = (
                buf_data["buffer"].to(device),
                buf_data["ptr"],
                buf_data["max_delay"],
                buf_data["size"],
            )
        return cls(delay_buffers=delay_buffers)

    def reset(self) -> None:
        for buf, _, _, _ in self.delay_buffers.values():
            buf.zero_()
```

**Impact on Timeline**: +6-8 hours for PathwayState infrastructure

---

## Open Questions

1. **Striatum State Structure** (✅ RESOLVED):
   - ✅ No StriatumState dataclass exists (uses raw attributes)
   - ✅ Has StriatumStateTracker for temporal state management
   - ✅ Has CheckpointManager with custom serialization
   - ✅ D1/D2 pathways have separate weight matrices and eligibility traces
   - **Action**: Create StriatumState consolidating state_tracker + pathway states

2. **Oscillator State** (✅ RESOLVED via investigation):
   - **Stored globally**: Oscillator phases/amplitudes managed by Brain, not regions
   - **Region state**: Regions store `_theta_phase`, `_beta_phase`, etc. as instance attributes (set via `set_oscillator_phases()`)
   - **Not in dataclass**: Oscillator state is transient (recomputed each timestep)
   - **No serialization needed**: Oscillators restart from t=0 after checkpoint load
   - **Action**: Document that oscillator state is NOT part of RegionState

3. **Neuromodulator State** (✅ CLARIFIED via investigation):
   - **NeuralComponentState fields**: `dopamine`, `acetylcholine`, `norepinephrine` are region-LOCAL **effects**
   - **Global levels**: Managed by NeuromodulatorManager (centralized, ADR-015)
   - **Checkpoint strategy**:
     - Save global modulator levels in Brain checkpoint
     - Save region-local modulator effects in RegionState (for regions with modulator-dependent dynamics)
   - **Action**: Clarify in docstrings that NeuralComponentState stores local effects, not global levels

4. **Partial State Loading**: Should we support loading only some fields?
   - Use case: Transfer only weights, not traces
   - Implementation: Optional fields + partial=True flag

5. **State Compression**: Should we compress large tensor fields?
   - Trade-off: Disk space vs load time
   - Consider: Only for archival checkpoints
   - **Defer to Phase 7**: Not critical for initial implementation

6. **Distributed Training**: How does this interact with DDP?
   - Need: Broadcast state across processes
   - Test: Multi-GPU state synchronization
   - **Defer to future**: Not part of current roadmap

7. **STP (Short-Term Plasticity) State** (✅ RESOLVED):
   - **Found**: `src/thalia/components/synapses/stp.py` (455 lines)
   - **Class**: `ShortTermPlasticity(nn.Module)` with state variables:
     - `u: torch.Tensor` - Release probability (facilitation variable) [n_pre] or [n_pre, n_post]
     - `x: torch.Tensor` - Available resources (depression variable) [n_pre] or [n_pre, n_post]
   - **Dynamics**: Tsodyks-Markram model with time constants τ_f (facilitation) and τ_d (depression)
   - **Biological Justification**: STRONGLY supported by literature
     - Mossy Fibers (DG→CA3): U~0.03 (Salin et al. 1996) - massive facilitation (10x!)
     - CA3 Recurrent: CRITICAL for preventing frozen attractors (code comments say "CRITICAL")
     - Schaffer Collaterals (CA3→CA1): Depression enables novelty detection
   - **Implementation Status**: Fully implemented with biologically-validated presets
   - **Current Default**: `stp_enabled: bool = False` (likely disabled for initial testing)
   - **State Lifecycle**: Resets on `reset_state()`, evolves during `forward()`
   - **Checkpoint Need**: YES - u/x state affects transmission efficacy

   **Decision**:
   - **RECOMMENDATION**: Enable STP by default for hippocampus (biological evidence is overwhelming)
   - If region uses STP (`config.stp_enabled == True`), add STP state to RegionState
   - Hippocampus STP state (when enabled):
     ```python
     @dataclass
     class HippocampusState:
         # ... existing fields ...

         # STP state (only if config.stp_enabled)
         stp_mossy_u: Optional[torch.Tensor] = None  # DG→CA3 facilitation
         stp_mossy_x: Optional[torch.Tensor] = None  # DG→CA3 depression
         stp_schaffer_u: Optional[torch.Tensor] = None  # CA3→CA1 depression
         stp_schaffer_x: Optional[torch.Tensor] = None
         stp_ca3_recurrent_u: Optional[torch.Tensor] = None  # Prevents frozen attractors
         stp_ca3_recurrent_x: Optional[torch.Tensor] = None
         stp_ec_ca1_u: Optional[torch.Tensor] = None  # EC→CA1 direct
         stp_ec_ca1_x: Optional[torch.Tensor] = None
     ```
   - Serialize/deserialize u and x tensors like other state
   - **Phase 2.3**: Add STP state to HippocampusState migration
   - **Other Regions** (see `docs/design/stp-biological-requirements.md`):
     - **Hippocampus**: ✅ Enabled (Dec 2025) - 4 pathways (mossy fiber, schaffer, EC→CA1, CA3 recurrent)
     - **Cortex**: ✅ Always enabled - L2/3 recurrent depression (hardcoded)
     - **Prefrontal**: ✅ Enabled by default - recurrent depression for WM flexibility
     - **Cerebellum**: ⚠️ NOT implemented - **HIGH PRIORITY** (PF→Purkinje depression CRITICAL for timing)
     - **Thalamus**: ⚠️ NOT implemented - **HIGH PRIORITY** (sensory gating depression)
     - **Striatum**: ⚠️ NOT implemented - MODERATE PRIORITY (corticostriatal depression)

8. **Pathway Growth and State** (✅ RESOLVED):
   - **Status**: CircularDelayBuffer already has `grow()` method implemented
   - **Location**: `src/thalia/utils/delay_buffer.py:141`
   - **Semantics**:
     - ✅ Preserves in-flight spikes (existing data copied)
     - ✅ Initializes new neurons to zeros
     - ✅ Pointer and delay timing unchanged
   - **Action for Phase 0**: Implement `AxonalProjection.grow_source()` method
   - **Testing**: Add growth + delay preservation integration tests

   **AxonalProjection.grow_source() Design**:
   ```python
   def grow_source(self, source_name: str, source_port: Optional[str], new_size: int) -> None:
       """Expand projection when upstream region grows.

       Example:
           cortex.grow_output(20)  # L5: 128 → 148
           projection.grow_source("cortex", "l5", 148)
       """
       source_key = f"{source_name}:{source_port}" if source_port else source_name
       for spec in self.sources:
           if spec.compound_key() == source_key:
               spec.size = new_size
               self._delay_buffers[source_key].grow(new_size)  # Uses existing method
               self.n_output += (new_size - spec.size)
               return
       raise ValueError(f"Source '{source_key}' not found")
   ```

---

## Appendix: Example Migration

### Complete Example: Adding New Field to HippocampusState

**Step 1: Add field and bump version**
```python
@dataclass
class HippocampusState(NeuralComponentState, RegionState):
    STATE_VERSION: ClassVar[int] = 2  # Was 1

    # Existing fields...
    dg_spikes: Optional[torch.Tensor] = None
    ca3_spikes: Optional[torch.Tensor] = None

    # NEW in v2
    ca3_recurrent_gain: float = 1.0
```

**Step 2: Implement migration**
```python
    @classmethod
    def _migrate_v1_to_v2(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add ca3_recurrent_gain field (v1 → v2)."""
        data["ca3_recurrent_gain"] = 1.0
        data["version"] = 2
        return data
```

**Step 3: Update from_dict to use migration**
```python
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device) -> "HippocampusState":
        # Apply migrations
        data = cls._apply_migrations(data)

        # Construct with migrated data
        return cls(
            dg_spikes=_load_tensor(data, "dg_spikes", device),
            ca3_spikes=_load_tensor(data, "ca3_spikes", device),
            ca3_recurrent_gain=data.get("ca3_recurrent_gain", 1.0),
        )
```

**Step 4: Test migration**
```python
def test_hippocampus_v1_to_v2_migration():
    """Test v1 checkpoint loads with v2 code."""
    # Create v1 checkpoint data
    v1_data = {
        "version": 1,
        "dg_spikes": torch.zeros(100),
        "ca3_spikes": torch.zeros(200),
        # No ca3_recurrent_gain
    }

    # Load with v2 code
    state = HippocampusState.from_dict(v1_data, device=torch.device("cpu"))

    # Verify migration applied
    assert state.ca3_recurrent_gain == 1.0  # Default value
    assert state.dg_spikes.shape == (100,)
```

---

## References

- Original architecture review: `docs/reviews/architecture-review-2025-12-21.md` (Section 3.2)
- State management patterns: To be created in Phase 7
- Checkpoint format: `docs/api/CHECKPOINT_FORMAT.md`
- Component protocols: `docs/api/PROTOCOLS_REFERENCE.md`

---

---

## Investigation Summary (December 21, 2025)

### Critical Decisions Made

**1. Inheritance Approach** (✅ CONFIRMED):
- **Decision**: Use Protocol-based RegionState (no inheritance)
- **Rationale**: Avoids diamond inheritance, flexible, type-safe
- **Implementation**: Phase 1
- **Status**: Ready to implement

**2. Delay Buffer Growth Protocol** (✅ DESIGNED):
- **Finding**: CircularDelayBuffer.grow() already exists (`delay_buffer.py:141`)
- **Preserves**: In-flight spikes, pointer state, delay timing
- **New work**: Implement AxonalProjection.grow_source() in Phase 0
- **Testing**: Growth + delay preservation tests
- **Status**: Ready to implement

**3. STP State Management** (✅ RESOLVED):
- **Found**: ShortTermPlasticity class with u/x state variables
- **Currently**: Disabled by default (`stp_enabled: bool = False`)
- **If enabled**: Add STP state fields to RegionState
- **Blocking**: No (not used yet)
- **Status**: Document for future enhancement

---

### What Was Investigated

1. **Striatum Architecture** (`src/thalia/regions/striatum/`)
   - ✅ Main file: 1979 lines, coordinates D1/D2 opponent pathways
   - ✅ No StriatumState dataclass (uses raw attributes)
   - ✅ StriatumStateTracker (297 lines): Manages votes, action, exploration
   - ✅ CheckpointManager (651 lines): Custom serialization (~200 lines)
   - ✅ D1/D2 separate weight matrices in parent's `synaptic_weights` dict
   - ✅ D1/D2 delay buffers for temporal competition (15ms vs 25ms)

2. **Pathway State** (`src/thalia/pathways/axonal_projection.py`)
   - ✅ AxonalProjection: Pure spike routing with delays
   - ✅ Uses CircularDelayBuffer for each source
   - ✅ NO state serialization currently implemented
   - ⚠️ Critical gap: In-flight spikes lost across checkpoints

3. **Delay Buffers** (`src/thalia/utils/delay_buffer.py`)
   - ✅ CircularDelayBuffer: Ring buffer for axonal delays
   - ✅ State: buffer tensor [max_delay+1, size], ptr (int), metadata
   - ✅ O(1) read/write, GPU-compatible
   - ⚠️ No growth method (needed for curriculum learning)

4. **Oscillator Integration** (multiple files)
   - ✅ Oscillators managed globally by Brain
   - ✅ Regions receive phases via `set_oscillator_phases()` broadcast
   - ✅ Phases stored as instance attributes (transient)
   - ✅ NOT serialized (restart from t=0 after load)

5. **Neuromodulator Storage** (`src/thalia/regions/base.py`)
   - ✅ NeuralComponentState has dopamine/ACh/NE fields
   - ✅ These are region-LOCAL effects, not global levels
   - ✅ Global levels managed by NeuromodulatorManager (centralized)
   - ✅ Checkpoint strategy: Save both global and local effects

6. **Existing State Dataclasses**
   - ✅ HippocampusState: 10+ fields (DG/CA3/CA1 spikes, traces, persistent activity)
   - ✅ LayeredCortexState: 15+ fields (L4/L2/3/L5/L6a/L6b, traces, modulation)
   - ✅ PrefrontalState: 4 fields (working memory, gates, rules)
   - ✅ ThalamicRelayState: 8 fields (relay/TRN spikes, mode, gating)
   - ❌ CerebellumState: Does not exist (uses raw attributes)
   - ❌ StriatumState: Does not exist (uses state_tracker + checkpoint_manager)

### Key Findings

**Architecture Insights**:
- Striatum is more complex than expected (D1/D2 separation, state_tracker, multiple components)
- Pathways need state management parallel to regions (delay buffers critical)
- Oscillator state is transient and should NOT be checkpointed
- Neuromodulator storage requires careful distinction (local vs global)

**Implementation Impact**:
- Phase 0 required: Pathway state infrastructure (6-8 hours)
- Striatum conversion takes longer: 8 hours (was 5 hours)
- Testing needs more time: 8-10 hours (was 6-8 hours) for pathway integration
- Total timeline: 59-73 hours (was 47-59 hours)

**Design Refinements**:
- Use Protocol-based approach for RegionState (avoid multiple inheritance)
- Serialize traces AS-IS (preserve temporal dynamics)
- Add delay buffer growth support (curriculum learning requirement)
- Add biological validity tests (trace decay, membrane range, etc.)

### Files Analyzed

| File | Lines | Purpose | Findings |
|------|-------|---------|----------|
| `striatum/striatum.py` | 1979 | Main striatum region | No state dataclass, complex D1/D2 coordination |
| `striatum/state_tracker.py` | 297 | Temporal state management | Vote accumulators, action tracking |
| `striatum/checkpoint_manager.py` | 651 | Custom serialization | ~200 lines manual serialize/deserialize |
| `pathways/axonal_projection.py` | 413 | Pure axonal transmission | Delay buffers, NO state serialization |
| `utils/delay_buffer.py` | 230 | Circular buffer for delays | Ring buffer, needs growth method |
| `regions/base.py` | ~90 | Base state classes | NeuralComponentState defined |
| `core/protocols/component.py` | 1254 | Component protocols | Oscillator broadcast pattern |
| `hippocampus/config.py` | 270 | Hippocampus config/state | HippocampusState dataclass |
| `cortex/config.py` | ~300 | Cortex config/state | LayeredCortexState dataclass |

### Next Steps

**Before Phase 0**:
1. ✅ Investigation complete
2. ✅ Design decisions documented
3. ✅ Biological justifications added
4. ⬜ Review with user
5. ⬜ Create Phase 0 branch

**Phase 0 Tasks** (Pathway State Foundation):
1. Create `src/thalia/core/pathway_state.py` with PathwayState ABC
2. Implement AxonalProjectionState for delay buffer serialization
3. Add `grow()` method to CircularDelayBuffer
4. Write comprehensive tests (delay preservation, growth, multi-source)

---

**Document Status**: UPDATED POST-INVESTIGATION
**Last Updated**: December 21, 2025
**Investigation Completed**: December 21, 2025
**Investigator**: AI Assistant (comprehensive codebase analysis)
**Files Investigated**: 9 core files, ~4500 lines analyzed
**Next Review**: After Phase 0 completion (Pathway State Foundation)
**Approved By**: Pending
