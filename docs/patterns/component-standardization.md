# Component Standardization Pattern

## Overview

The **Component Standardization Pattern** provides a unified interface for region sub-components (learning, homeostasis, memory, exploration). This pattern replaced the inconsistent naming (Manager/Coordinator/Engine) with standardized Component base classes.

**Status**: ✅ Implemented (Tier 2.1 - December 2024)

## Problem

Before standardization, different regions used inconsistent naming for similar functionality:
- Striatum: `LearningManager`, `HomeostasisManager`, `ExplorationManager`
- Hippocampus: `PlasticityManager`, `EpisodeManager`
- Other coordinators: `ForwardPassCoordinator`, `StateTracker`, `ReplayEngine`

This inconsistency made it hard to:
1. Understand component responsibilities at a glance
2. Implement new regions with unclear naming conventions
3. Share interfaces across regions
4. Document patterns clearly

## Solution

### Base Component Classes

All region components now inherit from standardized base classes in `src/thalia/core/region_components.py`:

```python
from thalia.core.region_components import (
    LearningComponent,      # Learning/plasticity logic
    HomeostasisComponent,   # Stability/balance mechanisms
    MemoryComponent,        # Episode/buffer management
    ExplorationComponent,   # Exploration strategies
)
```

Each base class:
- Inherits from `BaseManager` for lifecycle consistency
- Defines abstract methods for component-specific operations
- Provides default diagnostic methods
- Uses flexible signatures (`*args, **kwargs`) for region-specific needs

### Standardized Naming Convention

**Pattern**: `{Region}{Component}`

Examples:
- `StriatumLearningComponent` (was `LearningManager`)
- `StriatumHomeostasisComponent` (was `HomeostasisManager`)
- `StriatumExplorationComponent` (was `ExplorationManager`)
- `HippocampusLearningComponent` (was `PlasticityManager`)
- `HippocampusMemoryComponent` (was `EpisodeManager`)

## Implementation

### 1. Create Region Component

```python
# src/thalia/regions/striatum/learning_component.py

from thalia.core.region_components import LearningComponent

class StriatumLearningComponent(LearningComponent):
    """Manages three-factor learning for striatum (eligibility × dopamine)."""
    
    def __init__(self, config, context):
        super().__init__(config, context)
        # Initialize component-specific state
        
    def apply_learning(self, *args, **kwargs) -> Dict[str, Any]:
        """Apply dopamine-modulated learning."""
        # Implementation here
        return {"learning_applied": True, "metrics": ...}
    
    def reset_state(self) -> None:
        """Reset component state."""
        # Clear eligibility traces, etc.
    
    def get_learning_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics."""
        diag = super().get_learning_diagnostics()
        diag.update({"eligibility_mean": ...})
        return diag

# Backwards compatibility alias
LearningManager = StriatumLearningComponent
```

### 2. Update Region to Use Component

```python
# src/thalia/regions/striatum/striatum.py

from .learning_component import StriatumLearningComponent
from .homeostasis_component import StriatumHomeostasisComponent
from .exploration_component import StriatumExplorationComponent

class Striatum(NeuralComponent):
    def __init__(self, config):
        # ...
        
        # Use new component names
        self.learning = StriatumLearningComponent(config, context)
        self.homeostasis = StriatumHomeostasisComponent(config, context)
        self.exploration = StriatumExplorationComponent(config, context)
        
    # Backwards compatibility properties
    @property
    def learning_manager(self):
        return self.learning
    
    @property
    def homeostasis_manager(self):
        return self.homeostasis
    
    @property
    def exploration_manager(self):
        return self.exploration
```

### 3. Backwards Compatibility

Two levels of compatibility:

**Module-level aliases** (in component files):
```python
# learning_component.py
LearningManager = StriatumLearningComponent
```

**Instance properties** (in region files):
```python
# striatum.py
@property
def learning_manager(self):
    return self.learning
```

This ensures:
- Old imports continue working: `from .learning_manager import LearningManager`
- Old attribute access works: `region.learning_manager.apply_learning()`
- New code uses standardized names: `region.learning.apply_learning()`

## Component Types

### LearningComponent

**Purpose**: Manages learning rules and weight updates.

**Abstract Methods**:
- `apply_learning(*args, **kwargs) -> Dict[str, Any]`

**Typical Responsibilities**:
- Apply learning rule (STDP, three-factor, supervised, etc.)
- Manage eligibility traces
- Update synaptic weights
- Track learning metrics

**Examples**:
- `StriatumLearningComponent`: Three-factor learning (eligibility × dopamine)
- `HippocampusLearningComponent`: STDP with synaptic scaling

### HomeostasisComponent

**Purpose**: Maintains stability and prevents runaway dynamics.

**Abstract Methods**:
- `apply_homeostasis(*args, **kwargs) -> Dict[str, Any]`

**Typical Responsibilities**:
- Synaptic scaling (weight normalization)
- Intrinsic plasticity (threshold adaptation)
- Activity-dependent modulation
- E/I balance maintenance

**Examples**:
- `StriatumHomeostasisComponent`: Budget-constrained D1/D2 balance
- `HippocampusHomeostasisComponent`: CA3 synaptic scaling

### MemoryComponent

**Purpose**: Manages episodic or working memory buffers.

**Abstract Methods**:
- `store_memory(*args, **kwargs) -> None`
- `retrieve_memories(*args, **kwargs) -> Any`

**Typical Responsibilities**:
- Store episodes/experiences
- Retrieve relevant memories
- Manage buffer capacity
- Prioritize memories for replay

**Examples**:
- `HippocampusMemoryComponent`: Episodic buffer with priority sampling
- `PrefrontalMemoryComponent`: Working memory with gating (future)

### ExplorationComponent

**Purpose**: Manages exploration vs exploitation strategies.

**Abstract Methods**:
- `compute_exploration_bonus(*args, **kwargs) -> torch.Tensor`

**Typical Responsibilities**:
- UCB (Upper Confidence Bound) tracking
- Adaptive exploration parameters
- Action count tracking
- Performance history

**Examples**:
- `StriatumExplorationComponent`: UCB + adaptive tonic dopamine

## When to Create a Component

### DO create a component when:

✅ Logic is complex (>100 lines)  
✅ Component has dedicated state (traces, buffers, history)  
✅ Functionality maps to a biological subsystem  
✅ Multiple regions might need similar functionality  
✅ Clear separation of concerns improves clarity  

### DON'T create a component for:

❌ Simple coordination logic (<50 lines)  
❌ One-time initialization code  
❌ Pure I/O operations (checkpointing)  
❌ Simple state tracking (use instance variables)  

### Examples of NON-components:

- **ForwardPassCoordinator**: Simple orchestration → absorbed into region's `forward()`
- **StateTracker**: Simple state storage → instance variables
- **CheckpointManager**: Pure I/O utility → keep as utility module
- **ReplayEngine**: Complex reusable algorithm → keep as utility (used by sleep system too)

## Benefits

### 1. Clear Responsibilities

```python
# Before: What does "Manager" do?
learning_manager.apply_dopamine_learning()

# After: Clearly a learning component
learning.apply_learning()
```

### 2. Consistent Interfaces

All learning components share the same interface:
```python
# Works for ANY region with learning
region.learning.apply_learning(...)
region.learning.reset_state()
region.learning.get_learning_diagnostics()
```

### 3. Discoverability

```python
# What components does striatum have?
striatum.learning      # LearningComponent
striatum.homeostasis   # HomeostasisComponent
striatum.exploration   # ExplorationComponent

# What about hippocampus?
hippocampus.learning   # LearningComponent
hippocampus.memory     # MemoryComponent
```

### 4. Documentation Clarity

Pattern documentation can reference "LearningComponent" as a concept rather than listing region-specific manager names.

## Migration Checklist

When refactoring a region to use components:

- [ ] Create component base classes in `region_components.py`
- [ ] Create region-specific components (e.g., `StriatumLearningComponent`)
- [ ] Add backwards compatibility aliases in component files
- [ ] Update region `__init__` to use new component names
- [ ] Add backwards compatibility properties in region
- [ ] Update all component references within region methods
- [ ] Verify tests still pass with old imports
- [ ] Update documentation to reference new names

## Related Patterns

- **Component Parity** (`component-parity.md`): Regions AND pathways as components
- **State Management** (`state-management.md`): When to use RegionState vs component attributes
- **Mixins** (`mixins.md`): Shared functionality via mixins vs components

## References

- **ADR-008**: Neural Component Consolidation (regions as components)
- **Tier 2.1**: Architecture Review - Unify Manager Pattern (Option A: Functional Decomposition)
- **Implementation**: Commit 7e34373 (December 2024)

## Future Extensions

### Planned Component Types

- **GatingComponent**: For PFC working memory gating
- **AttentionComponent**: For top-down attention mechanisms
- **ConsolidationComponent**: For sleep-based consolidation

### Cross-Region Components

Some components might be generic enough to share:

```python
# Generic UCB exploration (not region-specific)
from thalia.components.exploration import UCBExploration

# Striatum can use it directly
self.exploration = UCBExploration(config, context)
```

This requires careful design to avoid over-abstraction.
