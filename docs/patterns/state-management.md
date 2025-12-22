# State Management Pattern Guide

**Date**: December 22, 2025 (Updated for Phase 6 State Refactoring)
**Purpose**: Clarify when and how to use state management in Thalia brain regions

---

## Overview

Brain regions in Thalia need to manage different types of data:
- **Mutable state** that changes every timestep (spikes, voltages, traces)
- **Learned parameters** that change during learning (weights)
- **Configuration** that never changes (hyperparameters)
- **Objects** that have their own state (neurons, STP modules)

This guide explains the **state management pattern** used consistently across all regions.

**Key Updates (December 2025)**:
- All state classes now inherit from `RegionState` or `BaseRegionState`
- Unified serialization via `to_dict()` and `from_dict()`
- Version migration support via `STATE_VERSION`
- Pathway state support via `PathwayState` protocol
- Checkpoint preservation of synaptic weights and delay buffers

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
class MyRegion(NeuralComponent):
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
class MyRegion(NeuralComponent):
    def __init__(self, config: MyRegionConfig):
        super().__init__(config)
        self.config = config  # or self.my_config
        self.device = config.device
```

**Why direct attribute**: Never changes after initialization.

---

#### 2. Learned Parameters (Mutable but Slow)
```python
class MyRegion(NeuralComponent):
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
class MyRegion(NeuralComponent):
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
class MyRegion(NeuralComponent):
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

class Striatum(NeuralComponent):
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

class PredictiveCortex(NeuralComponent):
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
class PredictiveCortex(NeuralComponent):
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

**Last Updated**: December 22, 2025
**See Also**:
- `docs/patterns/configuration.md` - Configuration patterns
- `docs/design/architecture.md` - Overall architecture
- `thalia/regions/base.py` - NeuralComponent base class
- `docs/api/STATE_CLASSES_REFERENCE.md` - All state classes
- `docs/design/state-management-refactoring-plan.md` - State refactoring details

---

## Advanced Topics

### State Serialization and Checkpointing

**Two Levels of State APIs**:

1. **`get_state()` / `load_state()`** - State dataclass (clean API)
   ```python
   state = region.get_state()  # Returns RegionState dataclass
   region.load_state(state)     # Accepts RegionState dataclass
   ```

2. **`get_full_state()` / `load_full_state()`** - Dict with weights (checkpoints)
   ```python
   full_state = region.get_full_state()  # Returns Dict[str, Any]
   region.load_full_state(full_state)    # Accepts Dict[str, Any]
   ```

**Key Difference**: `get_full_state()` includes **synaptic weights** which are `nn.Parameter` objects and live outside the state dataclass:

```python
def get_full_state(self) -> Dict[str, Any]:
    """Get complete state for checkpointing."""
    state_obj = self.get_state()
    state = state_obj.to_dict()

    # Add synaptic weights (required for checkpointing)
    state['synaptic_weights'] = {
        name: weights.detach().clone()
        for name, weights in self.synaptic_weights.items()
    }

    return state

def load_full_state(self, state: Dict[str, Any]) -> None:
    """Load complete state from checkpoint."""
    # Restore state dataclass
    state_obj = MyRegionState.from_dict(state, device=str(self.device))
    self.load_state(state_obj)

    # Restore synaptic weights
    if 'synaptic_weights' in state:
        for name, weights in state['synaptic_weights'].items():
            if name in self.synaptic_weights:
                self.synaptic_weights[name].data = weights.to(self.device)
```

**Why Two APIs?**
- **`get_state()`**: Clean dataclass for accessing current state, testing, diagnostics
- **`get_full_state()`**: Complete checkpoint including learned parameters for saving/loading

---

### Version Migration with STATE_VERSION

All state classes include a `STATE_VERSION` field for handling schema changes:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, ClassVar
import torch
from thalia.core.region_state import BaseRegionState

@dataclass
class MyRegionState(BaseRegionState):
    """State for MyRegion."""
    STATE_VERSION: ClassVar[int] = 2  # Incremented when schema changes

    # Original fields (v1)
    spikes: Optional[torch.Tensor] = None
    membrane: Optional[torch.Tensor] = None

    # Added in v2
    eligibility: Optional[torch.Tensor] = None

    @classmethod
    def _migrate_from_v1(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate v1 checkpoint to v2 schema.

        Called automatically when loading old checkpoints.
        """
        # Add new field with default value
        data['eligibility'] = None
        return data
```

**When to Bump Version**:
1. Adding new fields → Bump version, add migration method
2. Removing fields → Bump version, migration removes old keys
3. Renaming fields → Bump version, migration maps old → new names
4. Changing field types → Bump version, migration converts types

**Migration Flow**:
```python
# Loading old checkpoint
state_dict = torch.load("checkpoint_v1.ckpt")
region_data = state_dict['regions']['cortex']

# BaseRegionState.from_dict() automatically detects version mismatch
loaded_state = MyRegionState.from_dict(region_data, device='cpu')
# Internally calls _migrate_from_v1() if needed
```

---

### Device Management

State serialization must handle device transfer correctly:

```python
@dataclass
class MyRegionState(BaseRegionState):
    """State with proper device handling."""

    spikes: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict (always CPU for portability)."""
        return {
            'STATE_VERSION': self.STATE_VERSION,
            'spikes': self.spikes.cpu() if self.spikes is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str) -> "MyRegionState":
        """Deserialize from dict, placing tensors on target device."""
        device_obj = torch.device(device)

        # Handle version migration
        version = data.get('STATE_VERSION', 1)
        if version < cls.STATE_VERSION:
            data = cls._migrate_from_v1(data)

        return cls(
            spikes=data['spikes'].to(device_obj) if data.get('spikes') is not None else None,
        )
```

**Best Practices**:
- Always serialize to CPU (`.cpu()`) for portability
- Accept `device` parameter in `from_dict()`
- Use `torch.device()` to handle both string and device objects
- Check `is not None` before calling `.to(device)`

---

### Pathway State Pattern

Pathways (like `AxonalProjection`) also need state management for delay buffers:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch

@dataclass
class AxonalProjectionState:
    """State for AxonalProjection with delay buffers.

    Delay buffers store in-flight spikes that haven't reached the target yet.
    This is critical for biological accuracy (axonal delays: 2-25ms).
    """
    STATE_VERSION: int = 1

    # Delay buffer per source: {source_name: (buffer, pointer, max_delay, size)}
    delay_buffers: Optional[Dict[str, Tuple[torch.Tensor, int, int, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize delay buffers to dict."""
        if self.delay_buffers is None:
            return {'STATE_VERSION': self.STATE_VERSION, 'delay_buffers': None}

        serialized_buffers = {}
        for source_name, (buffer, ptr, max_delay, size) in self.delay_buffers.items():
            serialized_buffers[source_name] = {
                'buffer': buffer.cpu(),
                'pointer': ptr,
                'max_delay': max_delay,
                'size': size,
            }

        return {
            'STATE_VERSION': self.STATE_VERSION,
            'delay_buffers': serialized_buffers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str) -> "AxonalProjectionState":
        """Deserialize delay buffers from dict."""
        device_obj = torch.device(device)

        if data.get('delay_buffers') is None:
            return cls(delay_buffers=None)

        reconstructed_buffers = {}
        for source_name, buffer_data in data['delay_buffers'].items():
            reconstructed_buffers[source_name] = (
                buffer_data['buffer'].to(device_obj),
                buffer_data['pointer'],
                buffer_data['max_delay'],
                buffer_data['size'],
            )

        return cls(delay_buffers=reconstructed_buffers)
```

**Why Pathway State Matters**:
- Preserves **in-flight spikes** during checkpoints
- Maintains **temporal dynamics** (D1 vs D2 delays in striatum)
- Ensures **biological accuracy** (no artificial spike loss)

---

### State Best Practices Summary

#### ✅ DO:
1. **Use `BaseRegionState` as base class** for all region states
2. **Include `STATE_VERSION`** as `ClassVar[int]`
3. **Implement `to_dict()` and `from_dict()`** for serialization
4. **Add migration methods** when bumping version
5. **Handle device properly** (serialize to CPU, load to target device)
6. **Document state fields** in docstring
7. **Keep state dataclass pure data** (no methods except serialization)
8. **Add synaptic weights in `get_full_state()`**, not in state dataclass

#### ❌ DON'T:
1. **Don't put configuration in state** (it's immutable)
2. **Don't put nn.Parameter in state** (weights go in get_full_state())
3. **Don't put objects in state** (neurons, STP modules manage their own state)
4. **Don't forget device transfer** in from_dict()
5. **Don't skip version when changing schema**
6. **Don't serialize to CUDA** (always use .cpu() for portability)
7. **Don't hardcode device strings** (use device parameter)

---

### State Testing Checklist

When adding or modifying state classes:

```python
def test_state_roundtrip():
    """Test state serialization preserves data."""
    region = MyRegion(config)

    # Run some timesteps
    for _ in range(10):
        region.forward(input_spikes)

    # Save and load state
    state = region.get_state()
    state_dict = state.to_dict()
    loaded_state = MyRegionState.from_dict(state_dict, device='cpu')

    # Verify preservation
    assert torch.allclose(state.spikes, loaded_state.spikes)
    assert torch.allclose(state.membrane, loaded_state.membrane)

def test_checkpoint_with_weights():
    """Test full checkpoint includes weights."""
    region = MyRegion(config)

    # Run learning
    for _ in range(10):
        region.forward(input_spikes)
        region.apply_learning()

    # Save full state
    full_state = region.get_full_state()

    # Verify weights included
    assert 'synaptic_weights' in full_state
    assert 'default' in full_state['synaptic_weights']

    # Load into new region
    region2 = MyRegion(config)
    region2.load_full_state(full_state)

    # Verify weights match
    assert torch.allclose(
        region.synaptic_weights['default'],
        region2.synaptic_weights['default']
    )

def test_device_transfer():
    """Test state transfers between devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    region_cpu = MyRegion(config.replace(device='cpu'))
    region_cpu.forward(input_spikes)

    # Save from CPU
    state_dict = region_cpu.get_full_state()

    # Load to CUDA
    region_cuda = MyRegion(config.replace(device='cuda'))
    region_cuda.load_full_state(state_dict)

    # Verify device
    assert region_cuda.state.spikes.device.type == 'cuda'
    assert region_cuda.synaptic_weights['default'].device.type == 'cuda'

def test_version_migration():
    """Test v1 checkpoint loads into v2 state."""
    # Create v1-style data
    old_data = {
        'STATE_VERSION': 1,
        'spikes': torch.rand(100),
        'membrane': torch.rand(100),
        # 'eligibility' missing (added in v2)
    }

    # Load with v2 state class (triggers migration)
    state_v2 = MyRegionState.from_dict(old_data, device='cpu')

    # Verify migration
    assert state_v2.STATE_VERSION == 2
    assert state_v2.eligibility is None  # Default from migration
    assert state_v2.spikes is not None  # Original data preserved
```

---

## Implementation Checklist

When creating a new region with state:

- [ ] Create `MyRegionState` dataclass inheriting from `BaseRegionState`
- [ ] Add `STATE_VERSION = 1` as `ClassVar[int]`
- [ ] Define all mutable state fields with type hints and `= None` defaults
- [ ] Document state class in docstring
- [ ] Implement `to_dict()` with device handling (`.cpu()`)
- [ ] Implement `from_dict()` accepting device parameter
- [ ] Implement `get_state()` returning state dataclass
- [ ] Implement `load_state()` accepting state dataclass
- [ ] Implement `get_full_state()` including synaptic weights
- [ ] Implement `load_full_state()` restoring weights
- [ ] Call `self.state = MyRegionState()` in `__init__`
- [ ] Initialize state tensors in `reset_state()`
- [ ] Update state fields in `forward()`
- [ ] Write roundtrip test
- [ ] Write checkpoint test with weights
- [ ] Write device transfer test (if CUDA available)
- [ ] Add state class to `docs/api/STATE_CLASSES_REFERENCE.md` (auto-generated)

---

**Last Updated**: December 22, 2025
**See Also**:
- `docs/patterns/configuration.md` - Configuration patterns
- `docs/design/architecture.md` - Overall architecture
- `thalia/regions/base.py` - NeuralComponent base class
- `docs/api/STATE_CLASSES_REFERENCE.md` - All state classes (auto-generated)
- `docs/api/CHECKPOINT_FORMAT.md` - Checkpoint structure
- `docs/design/state-management-refactoring-plan.md` - State refactoring details
- `tests/integration/test_biological_validity.py` - State validation tests
- `tests/unit/core/test_state_properties.py` - Property-based state tests
