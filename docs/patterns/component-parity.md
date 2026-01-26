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

**Synapses at Target Dendrites**

In the current architecture:
- **Regions** are active learning components with synaptic weights
- **AxonalProjections** are pure spike routing (NO weights, NO learning)
- Learning happens at TARGET synapses, not in axonal pathways

### Regions (NeuralRegion)
Active components that:
- Process and transform spike trains
- Learn continuously during forward passes
- Maintain temporal state
- Store synaptic weights at dendrites (`synaptic_weights` dict)
- Need growth for curriculum learning
- Require diagnostics and health monitoring
- Must support checkpointing

### AxonalProjections
Passive routing components that:
- Route spikes from sources to targets
- Implement axonal conduction delays (CircularDelayBuffer)
- Do NOT have synaptic weights
- Do NOT learn
- Transform spike timing (delays only)
- Support multi-source routing
- Are simpler than regions (no growth needed for weights)

## Solution: Simplified Architecture

### NeuralRegion Base Class

All brain regions inherit from `NeuralRegion` (defined in `src/thalia/core/neural_region.py`):

```python
from thalia.core.neural_region import NeuralRegion
from thalia.learning import create_cortex_strategy

# Regions implement full learning interface via Strategy pattern
class LayeredCortex(NeuralRegion):
    """Region with synaptic weights at dendrites."""

    def __init__(self, config):
        super().__init__(
            n_neurons=config.n_output,
            neuron_config=config.neuron_config,
            default_learning_strategy="stdp",
            device=config.device,
        )

        # Add input sources with per-source learning strategies
        self.add_input_source(
            "thalamus",
            n_input=config.n_input,
            learning_strategy="stdp",  # Uses default
        )
        self.add_input_source(
            "hippocampus",
            n_input=config.hipp_input_size,
            learning_strategy="bcm",  # Different strategy
        )

    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process multi-source input with automatic per-source learning."""
        # Synaptic integration (inherited from NeuralRegion)
        total_current = torch.zeros(self.n_neurons, device=self.device)
        for source_name, spikes in source_spikes.items():
            weights = self.synaptic_weights[source_name]
            total_current += weights @ spikes

        # Spike generation
        output_spikes, _ = self.neurons(total_current, g_inh)

        # Learning happens automatically in NeuralRegion.forward()
        # via self.strategies[source_name].compute_update()

        return output_spikes
```

**Key Features**:
- **Inheritance**: `nn.Module + 7 mixins` (BrainComponent, Neuromodulator, Growth, Resettable, Diagnostics, StateLoading, LearningStrategy)
- **Synaptic Weights**: `Dict[str, Tensor]` per source at dendrites
- **Learning Strategies**: `Dict[str, LearningStrategy]` per source
- **Forward**: `Dict[str, Tensor] → Tensor` with automatic learning
- **Mixins**: NeuromodulatorMixin, GrowthMixin, ResettableMixin, DiagnosticsMixin

### AxonalProjection (No Learning)

Pathways are now simple routers:

```python
from thalia.pathways.axonal_projection import AxonalProjection

# AxonalProjection: Pure routing, no weights
class AxonalProjection:
    """Pure spike routing with axonal delays."""

    def forward(self, source_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Route spikes with delays, NO learning."""
        delayed_spikes = {}
        for source_name, spikes in source_outputs.items():
            # Apply axonal delay via CircularDelayBuffer
            delayed = self.delay_buffers[source_name](spikes)
            delayed_spikes[source_name] = delayed
        return delayed_spikes
```

### Enforced Methods

All `NeuralRegion` subclasses MUST implement:

#### 1. Processing (Standard PyTorch Convention)
```python
def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform multi-source input to output with continuous learning.

    Args:
        source_spikes: {"source_name": spike_tensor} from AxonalProjections

    Returns:
        Output spikes from neurons

    Standard PyTorch method - enables callable syntax: region(input)
    Learning happens during forward pass (per-source strategies).
    """
```

#### 2. State Management
```python
def reset_state(self) -> None:
    """Clear temporal dynamics (membrane, spikes, traces)."""
```

#### 3. Growth (Curriculum Learning)
```python
def grow_output(self, n_new: int) -> None:
    """Expand output dimension by adding neurons.

    Effects: Expands all synaptic_weights (adds rows), adds neurons, updates n_neurons
    """

def grow_input(self, n_new: int) -> None:
    """Expand input dimension to accept more inputs.

    Effects: Expands all synaptic_weights (adds columns), NO new neurons
    """

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
    """Serialize synaptic_weights, config, growth history."""

def load_full_state(self, state: Dict[str, Any]) -> None:
    """Restore from checkpoint."""
```

**Note**: AxonalProjections are simpler - they only need routing logic and delay buffers,
no learning or growth methods required.

## Development Workflow

### When Adding New Features to Regions

**Simplified Process** (pathways are routers only):
1. Implement feature in region (e.g., new synaptic mechanism)
2. Add to `NeuralRegion` base class if common
3. Update region-specific implementations
4. Write tests for regions
5. Update documentation

**Note**: Most new features won't apply to AxonalProjections since they don't learn.
Only add to pathways if it's about spike routing or delay mechanics.

### Example: Adding Spike-Timing Precision

```python
# Step 1: Add to regions that need it
class LayeredCortex(NeuralRegion):
    def __init__(self, config):
        super().__init__(config)
        self.spike_precision_ms = config.spike_precision_ms

    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Use precise spike timing in learning
        ...

# Step 2: AxonalProjection doesn't need it (just routing)
# No changes to axonal_projection.py

# Step 3: Tests
def test_cortex_spike_precision():
    # Test region feature
    ...
```
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

# Step 2: NeuralRegion base implements growth for all regions
class NeuralRegion(nn.Module, BrainComponentMixin, ...):
    def grow_input(self, n_new, initialization, sparsity):
        # Expand input weight columns when upstream grows
        ...

    def grow_output(self, n_new, initialization, sparsity):
        # Expand neuron population (output dimension)
        ...

    def get_capacity_metrics(self):
        from thalia.coordination.growth import GrowthManager
        return GrowthManager(self.name).get_capacity_metrics(self)

# Step 3: All regions inherit unified implementation from NeuralRegion
class Striatum(NeuralRegion):
    # Inherits growth methods from NeuralRegion
    pass

# Note: Pathways (AxonalProjection) don't grow - they're pure spike routing
# Growth happens at regions, which own synaptic weights

# Step 4: Tests for both
def test_region_growth(striatum): ...
def test_pathway_growth(cortex_to_hippo): ...
```

## Why Pathways Matter

**Pathways:**

1. **Axonal projections** (AxonalProjection):
   - Pure spike routing with axonal delays (NO learning)
   - NO synaptic weights (weights belong to target regions)
   - Multi-source concatenation and delay buffering
   - Biologically accurate: axons transmit, dendrites integrate

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
def analyze_learning(region: NeuralRegion) -> Dict:
    """Analyze learning in brain region only."""
    metrics = region.get_diagnostics()
    health = region.check_health()
    return {"metrics": metrics, "health": health}
```

**After:**
```python
from thalia.core.protocols.component import BrainComponent

def analyze_learning(component: BrainComponent) -> Dict:
    """Analyze any component (region or pathway) via protocol."""
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
