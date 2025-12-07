# State Management Pattern Guide

**Date**: December 7, 2025  
**Purpose**: Clarify when and how to use state management in Thalia brain regions

---

## Overview

Brain regions in Thalia need to manage different types of data:
- **Mutable state** that changes every timestep (spikes, voltages, traces)
- **Learned parameters** that change during learning (weights)
- **Configuration** that never changes (hyperparameters)
- **Objects** that have their own state (neurons, STP modules)

This guide explains the **state management pattern** used consistently across all regions.

---

## The Pattern

### Use `RegionState` Dataclass For Mutable State

**When**: Data that changes every `forward()` call and represents the current "state" of the region.

**Examples**:
- Spike outputs from different layers
- Membrane voltages
- Prediction errors
- Attention weights
- Trace values

**Implementation**:
```python
from dataclasses import dataclass
from thalia.regions.base import RegionState

@dataclass
class MyRegionState(RegionState):
    """Mutable state updated every forward pass.
    
    All fields default to None and are initialized in reset_state().
    """
    # Spike outputs
    spikes: Optional[torch.Tensor] = None
    
    # Internal dynamics
    voltage: Optional[torch.Tensor] = None
    current: Optional[torch.Tensor] = None
    
    # Learning signals
    prediction: Optional[torch.Tensor] = None
    error: Optional[torch.Tensor] = None
    
    # Scalar metrics
    free_energy: float = 0.0
```

**Usage in Region**:
```python
class MyRegion(BrainRegion):
    def __init__(self, config):
        super().__init__(config)
        self.state = MyRegionState()  # Create state container
    
    def forward(self, input_spikes):
        # Update state every forward pass
        self.state.spikes = self.neurons(input_spikes)
        self.state.voltage = self.neurons.v
        return self.state.spikes
    
    def reset_state(self):
        """Initialize state with proper tensor shapes."""
        self.state = MyRegionState(
            spikes=torch.zeros(self.config.n_output, device=self.device),
            voltage=torch.zeros(self.config.n_output, device=self.device),
        )
```

---

### Use Direct Attributes For Everything Else

**When**: Data that doesn't change every timestep.

#### 1. Configuration (Immutable)
```python
class MyRegion(BrainRegion):
    def __init__(self, config: MyRegionConfig):
        super().__init__(config)
        self.config = config  # or self.my_config
        self.device = config.device
```

**Why direct attribute**: Never changes after initialization.

---

#### 2. Learned Parameters (Mutable but Slow)
```python
class MyRegion(BrainRegion):
    def __init__(self, config):
        super().__init__(config)
        # Weights change during learning, but not every forward pass
        self.weights = nn.Parameter(
            WeightInitializer.xavier(
                n_output=config.n_output,
                n_input=config.n_input,
                device=config.device
            )
        )
```

**Why direct attribute**: Changes during `learn()`, not `forward()`. Part of the model's learned state, not ephemeral computational state.

---

#### 3. Sub-objects With Their Own State
```python
class MyRegion(BrainRegion):
    def __init__(self, config):
        super().__init__(config)
        # Objects manage their own internal state
        self.neurons = LIFNeuron(
            n_neurons=config.n_output,
            config=LIFConfig()
        )
        self.stp = ShortTermPlasticity(
            n_synapses=config.n_output,
            config=STPConfig()
        )
```

**Why direct attribute**: These objects have their own `reset_state()` and manage their own state internally.

---

#### 4. Accumulators and Metrics
```python
class MyRegion(BrainRegion):
    def __init__(self, config):
        super().__init__(config)
        # Diagnostic counters (not part of computational state)
        self._total_spikes = 0
        self._timesteps = 0
        self._last_plasticity_delta = 0.0
```

**Why direct attribute**: Diagnostic/monitoring data, not part of the forward pass state.

---

## Decision Tree

```
Does this data change every forward() call?
│
├─ YES: Does it represent the "current state" of the region?
│   │
│   ├─ YES: Use RegionState dataclass ✓
│   │       Examples: spikes, voltage, prediction, error
│   │
│   └─ NO: Use direct attribute
│           Examples: accumulators, diagnostic counters
│
└─ NO: Use direct attribute ✓
        Examples: config, weights, neurons, modules
```

---

## Real-World Examples

### Example 1: Simple Region (Striatum)

```python
@dataclass
class StriatumState(RegionState):
    """State that changes every timestep."""
    d1_spikes: Optional[torch.Tensor] = None
    d2_spikes: Optional[torch.Tensor] = None
    chosen_action: Optional[int] = None
    dopamine: float = 0.0

class Striatum(BrainRegion):
    def __init__(self, config: StriatumConfig):
        super().__init__(config)
        
        # Mutable state (changes every forward)
        self.state = StriatumState()
        
        # Configuration (immutable)
        self.config = config
        self.striatum_config = config
        
        # Learned parameters (change during learn())
        self.d1_weights = nn.Parameter(self._initialize_pathway_weights())
        self.d2_weights = nn.Parameter(self._initialize_pathway_weights())
        
        # Objects with internal state
        self.d1_neurons = ConductanceLIF(...)
        self.d2_neurons = ConductanceLIF(...)
        
        # Eligibility traces (change every forward, but managed by object)
        self.eligibility = EligibilityTraces(...)
```

---

### Example 2: Complex Region (PredictiveCortex)

```python
@dataclass
class PredictiveCortexState(RegionState):
    """State updated every timestep."""
    # Layer-specific outputs
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None
    
    # Predictive coding signals
    prediction: Optional[torch.Tensor] = None
    error: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    
    # Attention
    attention_weights: Optional[torch.Tensor] = None
    
    # Scalar metrics
    free_energy: float = 0.0

class PredictiveCortex(BrainRegion):
    def __init__(self, config: PredictiveCortexConfig):
        super().__init__(config)
        
        # Mutable state
        self.state = PredictiveCortexState()
        
        # Configuration
        self.predictive_config = config
        
        # Composition (cortex manages its own state)
        self.cortex = LayeredCortex(config)
        self.prediction_layer = PredictiveCodingLayer(...)
        self.attention = ScalableSpikingAttention(...)
        
        # Diagnostic accumulators (not in state)
        self._total_free_energy = 0.0
        self._timesteps = 0
        self._cumulative_l4_spikes = 0
```

**Note**: Composition-based regions delegate state management to their components. The region's state aggregates outputs from components.

---

### Example 3: Delegation Pattern (PredictiveCortex)

```python
class PredictiveCortex(BrainRegion):
    def forward(self, input_spikes):
        # Get outputs from composed cortex
        cortex_output = self.cortex.forward(input_spikes)
        
        # Extract state from inner cortex
        self.state.l4_spikes = self.cortex.state.l4_spikes
        self.state.l23_spikes = self.cortex.state.l23_spikes
        self.state.l5_spikes = self.cortex.state.l5_spikes
        
        # Compute own state
        self.state.prediction, self.state.error = self.prediction_layer(...)
        
        return cortex_output
```

**Pattern**: Composed regions extract relevant state from sub-components and store in their own state container.

---

## Common Pitfalls

### ❌ Don't Put Configuration in State
```python
# WRONG
@dataclass
class MyState(RegionState):
    n_neurons: int = 64  # This doesn't change!
    learning_rate: float = 0.01  # This doesn't change!
```

**Why wrong**: State should only contain data that changes during `forward()`.

---

### ❌ Don't Put Objects in State
```python
# WRONG
@dataclass
class MyState(RegionState):
    neurons: LIFNeuron = None  # Objects manage their own state!
```

**Why wrong**: Objects have their own state management. Use direct attributes.

---

### ❌ Don't Forget to Initialize State
```python
# WRONG
def reset_state(self):
    self.state = MyRegionState()  # Fields are None!
```

```python
# CORRECT
def reset_state(self):
    self.state = MyRegionState(
        spikes=torch.zeros(self.config.n_output, device=self.device),
        voltage=torch.zeros(self.config.n_output, device=self.device),
    )
```

**Why wrong**: Uninitialized tensors (None) will cause errors in first forward pass.

---

### ❌ Don't Access State Before reset_state()
```python
# WRONG
def __init__(self, config):
    super().__init__(config)
    self.state = MyRegionState()
    
    # Trying to use state before it's initialized!
    self.something = self.state.spikes.shape[0]  # Error: spikes is None!
```

**Why wrong**: State tensors are None until `reset_state()` is called.

---

## Best Practices

### ✅ Always Initialize State in reset_state()
```python
def reset_state(self):
    """Initialize all state tensors with proper shapes."""
    self.state = MyRegionState(
        spikes=torch.zeros(self.n_output, device=self.device),
        voltage=torch.zeros(self.n_output, device=self.device),
        current=torch.zeros(self.n_output, device=self.device),
    )
    
    # Also reset sub-objects
    self.neurons.reset_state()
    self.stp.reset_state()
```

---

### ✅ Document What Goes in State
```python
@dataclass
class MyRegionState(RegionState):
    """Mutable state for MyRegion.
    
    Updated every forward pass. Call reset_state() before starting
    a new sequence.
    
    Fields:
        spikes: Binary spike output [n_output]
        voltage: Membrane potential [n_output]
        current: Synaptic current [n_output]
        prediction: Predicted input [n_input]
    """
    spikes: Optional[torch.Tensor] = None
    voltage: Optional[torch.Tensor] = None
    current: Optional[torch.Tensor] = None
    prediction: Optional[torch.Tensor] = None
```

---

### ✅ Use Type Hints
```python
from typing import Optional
import torch

@dataclass
class MyState(RegionState):
    spikes: Optional[torch.Tensor] = None  # ✓ Clear type
    voltage: Optional[torch.Tensor] = None
    free_energy: float = 0.0  # Scalars use concrete types
```

---

### ✅ Group Related Fields
```python
@dataclass
class LayeredCortexState(RegionState):
    """State for layered cortex."""
    # Layer outputs
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None
    
    # Layer voltages
    l4_voltage: Optional[torch.Tensor] = None
    l23_voltage: Optional[torch.Tensor] = None
    l5_voltage: Optional[torch.Tensor] = None
    
    # Plasticity signals
    eligibility: Optional[torch.Tensor] = None
```

**Pattern**: Group related fields with blank lines and comments.

---

## FAQ

**Q: Why use dataclasses instead of regular classes?**  
A: Dataclasses give us:
- Automatic `__init__` with all fields
- Nice `__repr__` for debugging
- Easy field defaults
- Type hints enforced

**Q: Can I add methods to RegionState?**  
A: Generally no. State should be pure data. Put methods in the region class.

**Q: What if my region doesn't need state?**  
A: You can skip the RegionState dataclass, but you still need `self.state = RegionState()` for the base class.

**Q: Should I put numpy arrays in state?**  
A: No, only torch tensors. Convert numpy → torch before storing in state.

**Q: How do I access state from outside?**  
A: `region.state.spikes`, `region.state.voltage`, etc. State is public API.

**Q: Can state be batched?**  
A: Currently Thalia enforces single-instance (no batching). State tensors are 1D.

---

## Summary

| Data Type | Storage | Example |
|-----------|---------|---------|
| Timestep state | `RegionState` dataclass | `self.state.spikes` |
| Configuration | Direct attribute | `self.config` |
| Learned parameters | Direct attribute (nn.Parameter) | `self.weights` |
| Sub-objects | Direct attribute | `self.neurons` |
| Metrics/counters | Direct attribute (private) | `self._total_spikes` |

**Golden Rule**: If it changes every `forward()` call and represents the current computational state → `RegionState`. Everything else → direct attribute.

---

**Last Updated**: December 7, 2025  
**See Also**: 
- `docs/patterns/configuration.md` - Configuration patterns
- `docs/design/architecture.md` - Overall architecture
- `thalia/regions/base.py` - BrainRegion base class
