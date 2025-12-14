# Component Parity: Regions and Pathways are Equals

**Date**: December 7, 2025
**Status**: Active Design Pattern
**Related**: [State Management](./state-management.md), [Configuration](./configuration.md)

## Problem

Pathways are easy to forget when implementing new features. Historically, we've added features to regions but forgotten to add them to pathways, causing:

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

Both regions and pathways now inherit from `NeuralComponent`, a unified base class that implements the complete BrainComponent protocol (defined in `src/thalia/core/protocols/component.py`):

```python
from thalia.regions.base import NeuralComponent

# Both regions and pathways inherit from NeuralComponent
class LayeredCortex(NeuralComponent):
    """Region implementing full BrainComponent interface."""
    pass

class SpikingPathway(NeuralComponent):
    """Pathway implementing full BrainComponent interface."""
    pass
```

### Enforced Methods

All brain components MUST implement:

#### 1. Processing (Standard PyTorch Convention)
```python
def forward(self, *args, **kwargs) -> Any:
    """Transform input to output with continuous learning.

    Standard PyTorch method - enables callable syntax: component(input)
    All regions, pathways, and sensory encoders use this.
    """
```

#### 2. State Management
```python
def reset_state(self) -> None:
    """Clear temporal dynamics (membrane, spikes, traces)."""
```

#### 3. Growth (Curriculum Learning)
```python
def grow_input(self, n_new: int, initialization: str, sparsity: float) -> None:
    """Expand input dimension when upstream regions grow."""

def grow_output(self, n_new: int, initialization: str, sparsity: float) -> None:
    """Expand output dimension (neuron population) during curriculum."""

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
1. Implement feature in regions
2. Write tests for regions
3. Maybe remember pathways later

**NEW WAY** (enforced parity):
1. Add method to `BrainComponent` protocol
2. Implement in `NeuralComponent` base class (used by both regions and pathways)
3. Override in specialized subclasses as needed
4. Write tests for both regions AND pathways
5. Type checker fails if implementation is missing

### Example: Unified Growth API

```python
# Step 1: Protocol defines both methods (enforced)
class BrainComponent(Protocol):
    def grow_input(self, n_new: int, ...) -> None: ...
    def grow_output(self, n_new: int, ...) -> None: ...
    def get_capacity_metrics(self) -> CapacityMetrics: ...

# Step 2: NeuralComponent base implements for both regions and pathways
class NeuralComponent(BrainComponent):
    def grow_input(self, n_new, initialization, sparsity):
        # Expand input weight columns when upstream grows
        ...

    def grow_output(self, n_new, initialization, sparsity):
        # Expand neuron population (output dimension)
        ...

    def get_capacity_metrics(self):
        from thalia.coordination.growth import GrowthManager
        return GrowthManager(self.name).get_capacity_metrics(self)

# Step 3: Both regions and pathways inherit unified implementation
class Striatum(NeuralComponent):
    # Inherits growth methods from NeuralComponent
    pass

class SpikingPathway(NeuralComponent):
    # Inherits growth methods from NeuralComponent
    pass

# Step 4: Tests for both
def test_region_growth(striatum): ...
def test_pathway_growth(cortex_to_hippo): ...
```

## Why Pathways Matter

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

## Code Review Checklist

When reviewing PRs, check:

- [ ] Does this add new functionality to regions?
- [ ] If yes, does it also work for pathways?
- [ ] Are tests written for both regions AND pathways?
- [ ] Is BrainComponent protocol updated?
- [ ] Does documentation mention both regions and pathways?

## Migration Guide

### Updating Existing Code

If you find code that only works with regions:

**Before:**
```python
def analyze_learning(region: NeuralComponent) -> Dict:
    """Analyze learning in brain region only."""
    metrics = region.get_diagnostics()
    health = region.check_health()
    return {"metrics": metrics, "health": health}
```

**After:**
```python
from thalia.regions.base import NeuralComponent

def analyze_learning(component: NeuralComponent) -> Dict:
    """Analyze learning in any neural component (region or pathway)."""
    metrics = component.get_diagnostics()
    health = component.check_health()
    return {"metrics": metrics, "health": health}
```

## Recent Example: Temporal Coding (ADR-006) and PyTorch Consistency (ADR-007)

During the 1D bool tensor migration, we updated sensory pathways to use **temporal/latency coding** and standardized on PyTorch's `forward()` method:

### Before (Rate Coding + Non-standard API ❌)
```python
class VisualPathway:
    def encode(self, image: torch.Tensor):  # Non-standard method name
        """Output: [output_size] - single timestep rate coding"""
        activity = self.process(image)
        return activity.unsqueeze(0)  # Add fake time dimension
```

### After (Temporal Coding + Standard PyTorch ✅)
```python
class VisualPathway:
    def forward(self, image: torch.Tensor):  # Standard PyTorch convention
        """Output: [n_timesteps, output_size] - temporal spike train

        Information encoded in WHEN neurons spike:
        - High activity → early spike (t=0)
        - Low activity → late spike (t=19)
        """
        activity = self.retina.process(image)  # [output_size]
        spikes = self._generate_temporal_spikes(activity)  # [n_timesteps, output_size]
        return spikes
```

### Brain Processing
```python
# Brain consumes sequentially (no batch dimension)
# Now uses standard callable syntax:
spikes, metadata = visual_pathway(image)  # Calls forward() automatically
for t in range(n_timesteps):
    brain.forward(spikes[t])  # spikes[t] is 1D [output_size]
```

**Key Point**: Sensory pathways are active components that encode information, just like regions process it. Both deserve equal attention and implementation rigor.

## Related Patterns

- **State Management**: Both regions and pathways use RegionState / PathwayState dataclasses
- **Configuration**: Both use config dataclasses inheriting from RegionConfigBase
- **Mixins**: Both can use NeuromodulatorMixin, DiagnosticsMixin, etc.
- **Temporal Coding**: See [ADR-006](../decisions/adr-006-temporal-coding.md) for sensory pathway encoding

## References

- `src/thalia/core/protocols/component.py` - BrainComponent protocol definition
- `src/thalia/regions/base.py` - NeuralComponent base class (used by regions and pathways)
- `src/thalia/pathways/protocol.py` - Additional pathway-specific protocol
- `src/thalia/coordination/growth.py` - GrowthManager works with all components
- `docs/decisions/adr-006-temporal-coding.md` - Temporal/latency coding for sensory pathways
- `docs/decisions/adr-007-pytorch-consistency.md` - Standard forward() convention
- `docs/decisions/adr-008-neural-component-consolidation.md` - Unified NeuralComponent architecture

---

**Key Takeaway**: When you implement something for regions, implement it for pathways too. The protocol enforces this.
