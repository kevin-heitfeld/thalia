# Component Interface Enforcement

**Date**: December 11, 2025
**Status**: Implemented
**Related**: ADR-008 (Neural Component Consolidation), component-parity.md

## Overview

As of December 11, 2025, Thalia enforces the `BrainComponent` protocol through an abstract base class (`BrainComponentBase`). This ensures all neural components (regions and pathways) implement the complete interface at compile time.

## Motivation

Previously, the `BrainComponent` protocol was defined but not enforced:
- Missing methods only discovered at runtime
- Easy to forget implementing required methods
- No compile-time checking by IDEs or type checkers
- Component parity could be violated silently

**Solution**: Convert the protocol into an abstract base class that all components must inherit from.

## Architecture

### Class Hierarchy

```
BrainComponentBase (ABC)           # Enforces interface
    │
    ├── Provides: Abstract methods for all required interface methods
    │
    └── Forces: Subclasses to implement all methods or raise TypeError

nn.Module (PyTorch)                # PyTorch functionality
    │
    └── Provides: Parameter management, device handling, state dict

BrainComponentMixin                # Default implementations
    │
    ├── Provides: Default check_health() (returns healthy)
    │   Provides: Default get_capacity_metrics() (uses GrowthManager)
    │   Provides: Default set_oscillator_phases() (stores but doesn't use)
    │   Provides: Default grow_input() (raises helpful NotImplementedError)
    │   Provides: Default grow_output() (raises helpful NotImplementedError)
    │
    └── Can be overridden: Subclasses customize as needed

NeuralComponent                    # Unified base for ALL components
    │
    ├── Inherits: BrainComponentBase + nn.Module + BrainComponentMixin
    │   Inherits: NeuromodulatorMixin + LearningStrategyMixin + DiagnosticsMixin
    │
    ├── Implements: Abstract methods from BrainComponentBase
    │   Implements: Properties (device, dtype)
    │
    └── Provides: Common functionality for all neural components

Regions                            Pathways
    │                                  │
    ├── Striatum                       ├── SpikingPathway
    ├── Hippocampus                    ├── VisualPathway
    ├── LayeredCortex                  ├── LanguagePathway
    ├── PredictiveCortex               └── ...
    ├── Prefrontal
    ├── Cerebellum
    └── ...
```

### Key Design Decisions

1. **Multiple Inheritance Order Matters**:
   ```python
   class NeuralComponent(
       BrainComponentBase,      # 1. Enforces abstract interface (MUST be first)
       nn.Module,               # 2. PyTorch functionality
       BrainComponentMixin,     # 3. Default implementations
       NeuromodulatorMixin,     # 4. Neuromodulation
       LearningStrategyMixin,   # 5. Learning strategies
       DiagnosticsMixin         # 6. Diagnostics helpers
   ):
   ```

2. **Property Pattern for device/dtype**:
   - Abstract properties in `BrainComponentBase`
   - Implemented as properties with setters in `NeuralComponent`
   - Allows assignment in `__init__`: `self.device = torch.device(config.device)`
   - Provides getter for external access: `component.device`

3. **Default Implementations in Mixin**:
   - Components can use defaults or override
   - Reduces boilerplate for common cases
   - Helpful error messages for unimplemented features

## Required Methods

All components MUST implement:

### Core Processing
- `forward(*args, **kwargs) -> Any` - Process input, apply learning

### State Management
- `reset_state() -> None` - Clear temporal state (membrane potentials, traces)

### Neuromodulation & Oscillators
- `set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes) -> None`

### Growth (Curriculum Learning)
- `grow_input(n_new, initialization, sparsity) -> None`
- `grow_output(n_new, initialization, sparsity) -> None`
- `get_capacity_metrics() -> CapacityMetrics`

### Diagnostics
- `get_diagnostics() -> Dict[str, Any]`
- `check_health() -> HealthReport`

### Checkpointing
- `get_full_state() -> Dict[str, Any]`
- `load_full_state(state: Dict[str, Any]) -> None`

### Properties
- `device: torch.device` (property)
- `dtype: torch.dtype` (property)

## Implementation Guide

### For New Components

```python
from thalia.regions.base import NeuralComponent
from thalia.config.base import RegionConfig

class MyNewRegion(NeuralComponent):
    """My new brain region."""

    def __init__(self, config: RegionConfig):
        super().__init__(config)
        # NeuralComponent sets device and dtype automatically
        # Just initialize your weights and neurons
        self.weights = self._initialize_weights()
        self.neurons = self._create_neurons()

    def _initialize_weights(self) -> torch.Tensor:
        return WeightInitializer.sparse_random(...)

    def _create_neurons(self):
        return LIFNeuron(...)

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        # Process input and apply learning
        ...

    def reset_state(self) -> None:
        super().reset_state()  # Reset base state
        # Reset any additional state

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            'firing_rate': self.get_firing_rate(),
            'weight_mean': self.weights.mean().item(),
            ...
        }

    def get_full_state(self) -> Dict[str, Any]:
        return {
            'weights': self.weights.detach().cpu(),
            'config': self.config,
            'version': '1.0',
        }

    def load_full_state(self, state: Dict[str, Any]) -> None:
        self.weights.data.copy_(state['weights'].to(self.device))

    # Optional: Override defaults from BrainComponentMixin
    def grow_input(self, n_new: int, initialization: str, sparsity: float):
        # Custom input dimension growth
        ...

    def grow_output(self, n_new: int, initialization: str, sparsity: float):
        # Custom output dimension growth
        ...

    def check_health(self) -> HealthReport:
        # Custom health checks
        ...
```

### For Existing Components

If you already have a component that inherits from `NeuralComponent`, you're done! The enforcement happens automatically.

If Python raises `TypeError: Can't instantiate abstract class`, it means you're missing required methods:

```python
TypeError: Can't instantiate abstract class MyRegion with abstract methods 'forward', 'reset_state'
```

**Solution**: Implement the missing methods listed in the error.

## Migration Checklist

For each component:

- [ ] Inherits from `NeuralComponent` (or `BrainComponentBase` + `nn.Module` + `BrainComponentMixin`)
- [ ] Implements `forward()` - core processing
- [ ] Implements `reset_state()` - clear temporal state
- [ ] Implements `get_diagnostics()` - return activity metrics
- [ ] Implements `get_full_state()` - serialize for checkpointing
- [ ] Implements `load_full_state()` - restore from checkpoint
- [ ] Has `device` property (automatic if inherits from `NeuralComponent`)
- [ ] Has `dtype` property (automatic if inherits from `NeuralComponent`)
- [ ] Optional: Override `grow_input()` for input dimension growth
- [ ] Optional: Override `grow_output()` for output dimension growth
- [ ] Optional: Override `check_health()` for custom health checks
- [ ] Optional: Override `set_oscillator_phases()` if using oscillators

## Testing Interface Compliance

Use the provided test script to verify all components implement the interface:

```bash
python temp/test_component_enforcement.py
```

Output:
```
Testing Component Interface Enforcement

======================================================================

1. Testing Brain Regions:
----------------------------------------------------------------------
✓ Striatum                       - PASS
✓ TrisynapticHippocampus         - PASS
✓ LayeredCortex                  - PASS
✓ PredictiveCortex               - PASS
✓ Prefrontal                     - PASS
✓ Cerebellum                     - PASS

2. Testing Neural Pathways:
----------------------------------------------------------------------
✓ SpikingPathway                 - PASS
✓ VisualPathway                  - PASS
✓ LanguagePathway                - PASS

======================================================================
✓ All components implement the BrainComponent interface correctly!
```

## Benefits

### 1. Compile-Time Checking
```python
# Before: Runtime error (hard to debug)
>>> region = MyRegion(config)
>>> region.get_diagnostics()  # AttributeError: 'MyRegion' has no attribute 'get_diagnostics'

# After: Instantiation error (caught early)
>>> region = MyRegion(config)
TypeError: Can't instantiate abstract class MyRegion with abstract methods 'get_diagnostics'
```

### 2. IDE Support
- IDEs show which methods are required
- Autocomplete suggests abstract methods to implement
- Type checkers (mypy, pyright) catch missing implementations

### 3. Component Parity Guarantee
- Impossible to forget implementing required methods
- Regions and pathways guaranteed to have same interface
- Consistent API across all neural components

### 4. Self-Documenting Code
- Abstract methods clearly mark required interface
- Mixin provides default implementations where sensible
- Clear separation: required (abstract) vs optional (mixin)

## Default Implementations

Components can use default implementations from `BrainComponentMixin`:

### 1. `check_health()` - Returns healthy by default
```python
def check_health(self) -> HealthReport:
    return HealthReport(
        component_name=self.__class__.__name__,
        is_healthy=True,
        issues=[],
        warnings=[],
    )
```

**Override if**: You want custom health checks (silence detection, saturation, etc.)

### 2. `get_capacity_metrics()` - Uses GrowthManager
```python
def get_capacity_metrics(self) -> CapacityMetrics:
    from thalia.core.growth import GrowthManager
    manager = GrowthManager(component_name=self.name)
    return manager.get_capacity_metrics(self)
```

**Override if**: You want custom capacity metrics

### 3. `set_oscillator_phases()` - Stores but doesn't use
```python
def set_oscillator_phases(self, phases, signals, theta_slot, coupled_amplitudes):
    self._oscillator_phases = phases
    self._oscillator_signals = signals
    self._oscillator_theta_slot = theta_slot
    self._coupled_amplitudes = coupled_amplitudes
```

**Override if**: You want to use oscillator phases in your forward pass

### 4. `grow_input()` and `grow_output()` - Raise helpful errors
```python
def grow_input(self, n_new, initialization, sparsity):
    raise NotImplementedError(
        f"{self.__class__.__name__}.grow_input() not yet implemented. "
        f"This is required for handling upstream region growth. "
        f"See src/thalia/mixins/growth_mixin.py for implementation guide."
    )

def grow_output(self, n_new, initialization, sparsity):
    raise NotImplementedError(
        f"{self.__class__.__name__}.grow_output() not yet implemented. "
        f"Growth is essential for curriculum learning. "
        f"See src/thalia/mixins/growth_mixin.py for implementation guide."
    )
```

**Override**: Implement both methods for all components that support growth

## Common Patterns

### Pattern 1: Minimal Region (Using Defaults)
```python
class MinimalRegion(NeuralComponent):
    def forward(self, x): ...
    def reset_state(self): ...
    def get_diagnostics(self): ...
    def get_full_state(self): ...
    def load_full_state(self, state): ...
    # Uses default: check_health, get_capacity_metrics, set_oscillator_phases
    # Must implement: grow_input, grow_output (or use defaults with NotImplementedError)
```

### Pattern 2: Full-Featured Region (Custom Implementations)
```python
class FullRegion(NeuralComponent):
    def forward(self, x): ...
    def reset_state(self): ...
    def get_diagnostics(self): ...
    def get_full_state(self): ...
    def load_full_state(self, state): ...
    def check_health(self): ...          # Custom health checks
    def get_capacity_metrics(self): ...  # Custom capacity metrics
    def grow_input(self, n_new): ...     # Handles upstream growth
    def grow_output(self, n_new): ...    # Supports curriculum growth
    def set_oscillator_phases(self, phases): ...  # Uses oscillators
```

### Pattern 3: Pathway (Same Interface as Regions)
```python
class MyPathway(NeuralComponent):
    # Identical interface to regions!
    def forward(self, spikes): ...
    def reset_state(self): ...
    def get_diagnostics(self): ...
    def get_full_state(self): ...
    def load_full_state(self, state): ...
```

## Relationship to Other Patterns

### Component Parity Principle
Interface enforcement is the **implementation** of the component parity principle:
- Principle: Regions and pathways are equals
- Implementation: Both inherit from same abstract base
- Result: Guaranteed feature parity

See: `docs/patterns/component-parity.md`

### ADR-008: Neural Component Consolidation
Interface enforcement supports the ADR-008 decision:
- Decision: Unify regions and pathways under `NeuralComponent`
- Enforcement: `BrainComponentBase` ensures consistent implementation
- Benefit: Architectural equality backed by compile-time checking

See: `docs/decisions/adr-008-neural-component-consolidation.md`

### Learning Strategy Pattern
Interface enforcement complements the learning strategy pattern:
- Interface: Requires `forward()` where learning happens
- Strategy: Pluggable learning rules via `LearningStrategyMixin`
- Together: Consistent processing + flexible learning

See: `docs/patterns/learning-strategy-pattern.md`

## Summary

**Before**: Protocol defined interface, but not enforced
- Missing methods discovered at runtime
- Easy to violate component parity
- No IDE/type checker support

**After**: Abstract base class enforces interface
- Missing methods caught at instantiation
- Component parity guaranteed by type system
- Full IDE and type checker support

**Result**: More robust, maintainable, and self-documenting codebase.

---

**Implementation Date**: December 11, 2025
**Status**: ✅ Complete - All regions and pathways compliant
**Next Steps**: Monitor for any edge cases as new components are added
