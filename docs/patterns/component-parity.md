# Component Parity: Regions and Pathways are Equals

**Date**: December 7, 2025  
**Status**: Active Design Pattern  
**Related**: [State Management](./state-management.md), [Configuration](./configuration.md)

## Problem

Pathways are easy to forget when implementing new features. Historically, we've added features to `BrainRegion` but forgotten to add them to `BaseNeuralPathway`, causing:

1. **Feature asymmetry**: Regions can grow, pathways can't → curriculum learning breaks
2. **API inconsistency**: Code works with regions but not pathways → confusion
3. **Maintenance burden**: Discovering missing features late in development
4. **Conceptual error**: Treating pathways as "glue code" rather than first-class components

## Core Principle

**Pathways are just as important as regions.**

Both are active learning components that:
- Process and transform spike trains
- Learn continuously during forward passes
- Maintain temporal state
- Need growth for curriculum learning
- Require diagnostics and health monitoring
- Must support checkpointing

## Solution: BrainComponent Protocol

Created `src/thalia/core/component_protocol.py` defining the **unified interface** for all components:

```python
from thalia.core.component_protocol import BrainComponent

# Both regions and pathways implement this protocol
class BrainRegion(BrainComponent, ...):
    """Implements full BrainComponent interface."""
    pass

class BaseNeuralPathway(BrainComponent, ...):
    """Implements full BrainComponent interface."""
    pass
```

### Enforced Methods

All brain components MUST implement:

#### 1. Processing
```python
def forward(self, *args, **kwargs) -> Any:
    """Transform input to output with continuous learning."""
```

#### 2. State Management
```python
def reset_state(self) -> None:
    """Clear temporal dynamics (membrane, spikes, traces)."""
```

#### 3. Growth (Curriculum Learning)
```python
def add_neurons(self, n_new: int, initialization: str, sparsity: float) -> None:
    """Expand capacity without disrupting existing circuits."""

def get_capacity_metrics(self) -> CapacityMetrics:
    """Report utilization to guide growth decisions."""
```

#### 4. Diagnostics
```python
def get_diagnostics(self) -> Dict[str, Any]:
    """Report activity, learning, health metrics."""

def check_health(self) -> HealthReport:
    """Detect pathologies (silence, saturation, NaN)."""
```

#### 5. Checkpointing
```python
def get_full_state(self) -> Dict[str, Any]:
    """Serialize weights, config, growth history."""

def load_full_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint."""
```

## Development Workflow

### When Adding New Features

**OLD WAY** (easy to forget pathways):
1. Implement feature in `BrainRegion`
2. Write tests for regions
3. Maybe remember pathways later

**NEW WAY** (enforced parity):
1. Add method to `BrainComponent` protocol
2. Implement for `BrainRegion` base class
3. Implement for `BaseNeuralPathway` base class
4. Write tests for both regions AND pathways
5. Type checker fails if either is missing

### Example: Adding Growth Support (Phase 2)

```python
# Step 1: Add to protocol (already done)
class BrainComponent(Protocol):
    def add_neurons(self, n_new: int, ...) -> None: ...
    def get_capacity_metrics(self) -> CapacityMetrics: ...

# Step 2: Implement for regions
class BrainRegion(BrainComponent):
    def add_neurons(self, n_new, initialization, sparsity):
        # Expand weight matrices preserving existing weights
        ...
    
    def get_capacity_metrics(self):
        from thalia.core.growth import GrowthManager
        return GrowthManager(self.name).get_capacity_metrics(self)

# Step 3: Implement for pathways (REQUIRED by protocol)
class BaseNeuralPathway(BrainComponent):
    def add_neurons(self, n_new, initialization, sparsity):
        # Expand pathway matrices when connected regions grow
        ...
    
    def get_capacity_metrics(self):
        from thalia.core.growth import GrowthManager
        return GrowthManager(self.name).get_capacity_metrics(self)

# Step 4: Tests for both
def test_region_growth(striatum): ...
def test_pathway_growth(cortex_to_hippo): ...
```

## Why Pathways Matter

### Pathways are NOT just "glue"

**Pathways are active learning components:**

1. **Inter-region pathways** (SpikingPathway):
   - Learn via STDP during forward passes
   - Adapt connection strengths continuously
   - Can become saturated (too strong) or silent (too weak)
   - Need to grow when connected regions grow

2. **Sensory pathways** (SensoryPathway):
   - Transform raw inputs (images, audio, tokens) to spikes
   - Learn optimal encoding strategies
   - Can have runaway activity or silence
   - Need checkpointing like regions

3. **Specialized pathways** (Attention, Replay):
   - Implement complex gating and routing
   - Have their own state and learning rules
   - Require diagnostics to debug
   - Must grow with system

### What Breaks Without Pathway Parity

**Example: Curriculum Learning**

```python
# System starts small for simple tasks
brain = Brain(n_cortex_neurons=1000)

# Learns task 1 successfully
train(brain, easy_task)

# Now try harder task - cortex needs more capacity
brain.cortex.add_neurons(n_new=500)  # ✅ Cortex grows to 1500

# BUT pathway still expects 1000 inputs! 💥
brain.cortex_to_hippo.forward(cortex_spikes)  # Shape mismatch!
# Expected: [batch, 1000]
# Got: [batch, 1500]
```

**Fix**: Pathway must also grow:
```python
brain.cortex.add_neurons(n_new=500)
brain.cortex_to_hippo.add_neurons(n_new=500)  # Pathway tracks region size
```

## Code Review Checklist

When reviewing PRs, check:

- [ ] Does this add new functionality to BrainRegion?
- [ ] If yes, is it also added to BaseNeuralPathway?
- [ ] Are tests written for both regions AND pathways?
- [ ] Is BrainComponent protocol updated?
- [ ] Does documentation mention both regions and pathways?

## Migration Guide

### Updating Existing Code

If you find code that only works with regions:

**Before:**
```python
def analyze_learning(region: BrainRegion) -> Dict:
    """Analyze learning in brain region."""
    metrics = region.get_diagnostics()
    health = region.check_health()
    return {"metrics": metrics, "health": health}
```

**After:**
```python
def analyze_learning(component: BrainComponent) -> Dict:
    """Analyze learning in brain component (region or pathway)."""
    metrics = component.get_diagnostics()
    health = component.check_health()
    return {"metrics": metrics, "health": health}
```

## Related Patterns

- **State Management**: Both regions and pathways use RegionState / PathwayState dataclasses
- **Configuration**: Both use config dataclasses inheriting from RegionConfigBase
- **Mixins**: Both can use NeuromodulatorMixin, DiagnosticsMixin, etc.

## References

- `src/thalia/core/component_protocol.py` - Unified protocol definition
- `src/thalia/regions/base.py` - BrainRegion implements protocol
- `src/thalia/core/pathway_protocol.py` - BaseNeuralPathway implements protocol
- `src/thalia/core/growth.py` - GrowthManager works with both

---

**Key Takeaway**: When you implement something for regions, implement it for pathways too. The protocol enforces this.
