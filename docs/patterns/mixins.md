# Mixin Patterns Guide

**Date**: December 7, 2025
**Purpose**: Understanding Thalia's mixin system

> **ðŸ“š For complete method signatures and parameters of core NeuralRegion mixins, see [MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md)**

This guide focuses on **usage patterns** and **when to use** different mixins (both core and specialized).

---

## Overview

Thalia uses **mixins** to share functionality across brain regions without deep inheritance hierarchies. This guide covers:
- What mixins are available (core and specialized)
- When to use each mixin
- How to use mixins effectively
- Method resolution order (MRO) considerations

### Core vs Specialized Mixins

**Core Mixins** (used by all NeuralRegion subclasses):
- `DiagnosticsMixin` - Health monitoring
- `NeuromodulatorMixin` - Neuromodulator handling
- `GrowthMixin` - Dynamic growth support
- `ResettableMixin` - State reset functionality

**Specialized Mixins** (region-specific):
- `ActionSelectionMixin` - Action selection for striatum
- Other specialized mixins for specific regions

> For complete API reference of core mixins, see [MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md)

---

## Core Mixins (All Regions)

### 1. DiagnosticsMixin
**File**: `src/thalia/mixins/diagnostics_mixin.py`
**Purpose**: Health monitoring and diagnostics

> For complete method signatures, see [DiagnosticsMixin in MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md#diagnosticsmixin)

**Key Methods**:
- `weight_diagnostics()` - Weight statistics (mean, std, sparsity)
- `spike_diagnostics()` - Firing rates and spike statistics
- `collect_standard_diagnostics()` - Full diagnostic report

**Usage Pattern**:
```python
class MyRegion(NeuralComponent, DiagnosticsMixin):
    def forward(self, x):
        output = self._compute(x)

        # Collect diagnostics for monitoring
        diagnostics = self.collect_standard_diagnostics(
            region_name="my_region",
            weight_matrices={"weights": self.weights},
            spike_tensors={"output": output}
        )

        return output
```

---

### 2. NeuromodulatorMixin
**File**: `src/thalia/neuromodulation/mixin.py`
**Purpose**: Standardized neuromodulator handling

> For complete method signatures, see [NeuromodulatorMixin in MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md#neuromodulatormixin)

**Key Methods**:
- `set_neuromodulators()` - Set dopamine, acetylcholine, norepinephrine
- `get_effective_learning_rate()` - Modulate learning by dopamine
- `get_neuromodulator_state()` - Get current neuromodulator levels

**Usage Pattern**:
```python
class MyRegion(NeuralComponent, NeuromodulatorMixin):
    def forward(self, x):
        # Get dopamine-modulated learning rate
        effective_lr = self.get_effective_learning_rate(
            base_lr=self.config.learning_rate,
            dopamine_sensitivity=2.0
        )

        # Apply learning with modulation
        self.apply_learning(effective_lr)
        return output
```

---

### 3. GrowthMixin
**File**: `src/thalia/mixins/growth_mixin.py`
**Purpose**: Support for dynamic neuron growth

> For complete method signatures, see [GrowthMixin in MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md#growthmixin)

**Key Features**:
- **Phase 1** (January 2026): Validation helpers for post-growth correctness
- **Phase 2** (January 2026): Opt-in registration API for automatic component growth

**Phase 1 Validation Helpers**:
```python
class MyRegion(NeuralComponent, GrowthMixin):
    def grow_output(self, n_new: int):
        old_n = self.config.n_output

        # ... perform growth ...

        # Validate growth completed correctly
        self._validate_output_growth(
            old_n,
            n_new,
            check_neurons=True,      # Verify neuron count
            check_config=True,       # Verify config updated
            check_state_buffers=True # Verify state tensors
        )
```

**Phase 2 Registration API**:
```python
class MyRegion(NeuralComponent, GrowthMixin):
    def __init__(self, config):
        super().__init__(config)

        # Create STP modules
        self.stp_feedforward = ShortTermPlasticity(...)
        self.stp_recurrent = ShortTermPlasticity(...)

        # Register for automatic growth
        # Non-recurrent: grows in both contexts
        self._register_stp('stp_feedforward', direction='both')

        # Recurrent: grows both pre+post during grow_output
        self._register_stp('stp_recurrent', direction='post', recurrent=True)

    def grow_output(self, n_new: int):
        # ... expand weights and neurons ...

        # Automatically grow all registered STP modules
        self._auto_grow_registered_components('output', n_new)

        # Validate
        self._validate_output_growth(old_n, n_new)
```

**STP Direction Semantics**:
- `direction='pre'`: Grow during `grow_input()` only (tracks external inputs)
- `direction='post'`: Grow during `grow_output()` only (tracks outputs)
- `direction='both'`: Non-recurrent STP that participates in both contexts
  - During `grow_input()`: grows 'pre' dimension
  - During `grow_output()`: grows 'post' dimension
- `recurrent=True`: For recurrent STP (same population)
  - During `grow_output()`: grows BOTH 'pre' and 'post' dimensions

---

### 4. ResettableMixin
**File**: `src/thalia/mixins/resettable_mixin.py`
**Purpose**: State reset functionality

> For complete method signatures, see [ResettableMixin in MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md#resettablemixin)

**Key Methods**:
- `reset_state()` - Reset region to initial state
- `reset_standard_state()` - Reset specific state attributes

**Usage Pattern**:
```python
class MyRegion(NeuralComponent, ResettableMixin):
    def reset_state(self):
        # Reset standard attributes
        self.reset_standard_state(['v_mem', 'trace', 'adaptation'])
```

---

## Specialized Mixins (Region-Specific)

### ActionSelectionMixin (Striatum)
**File**: `src/thalia/regions/striatum/action_selection.py`
**Purpose**: Action selection strategies

**Provides**:
```python
# Selection strategies
select_action_softmax(q_values: torch.Tensor, temperature: float) -> int
select_action_greedy(q_values: torch.Tensor, epsilon: float) -> int
select_action_thompson(q_values: torch.Tensor, uncertainty: torch.Tensor) -> int

# Utilities
compute_policy(q_values: torch.Tensor, temperature: float) -> torch.Tensor
add_exploration_noise(q_values: torch.Tensor, noise_std: float) -> torch.Tensor
```

**Usage**:
```python
class Striatum(NeuralComponent, ActionSelectionMixin):
    def select_action(self, state):
        q_values = self.compute_q_values(state)

        # Use mixin method
        action = self.select_action_softmax(
            q_values,
            temperature=self.config.softmax_temperature
        )
        return action
```

---

### 3. SpikingNeuronMixin
**File**: `src/thalia/core/spiking_mixin.py`
**Purpose**: Spiking neuron dynamics

**Provides**:
```python
# Neuron dynamics
lif_dynamics(v_mem: torch.Tensor, i_syn: torch.Tensor, dt_ms: float) -> Tuple[torch.Tensor, torch.Tensor]
adaptive_lif_dynamics(...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Spike generation
spike_from_voltage(v_mem: torch.Tensor, threshold: float) -> torch.Tensor
soft_spike(v_mem: torch.Tensor, threshold: float, sharpness: float) -> torch.Tensor

# Refractory period
apply_refractory(spikes: torch.Tensor, refractory_state: torch.Tensor) -> torch.Tensor
```

**Usage**:
```python
class MySpikingRegion(NeuralComponent, SpikingNeuronMixin):
    def forward(self, input_current):
        # Use mixin method for LIF dynamics
        self.v_mem, spikes = self.lif_dynamics(
            self.v_mem,
            input_current,
            dt_ms=self.config.dt_ms
        )
        return spikes
```

---

### 4. DendriticComputationMixin
**File**: `src/thalia/core/dendritic_mixin.py`
**Purpose**: Dendritic integration and NMDA spikes

**Provides**:
```python
# Dendritic integration
dendritic_integration(
    proximal_input: torch.Tensor,
    distal_input: torch.Tensor,
    modulatory_input: torch.Tensor
) -> torch.Tensor

# NMDA spikes
compute_nmda_spike(
    dendritic_voltage: torch.Tensor,
    threshold: float
) -> torch.Tensor

# Plateau potentials
plateau_potential(
    dendritic_activity: torch.Tensor,
    tau_ms: float
) -> torch.Tensor
```

**Usage**:
```python
class L5Pyramidal(NeuralComponent, DendriticComputationMixin):
    def forward(self, proximal, distal, modulatory):
        # Use dendritic integration from mixin
        integrated = self.dendritic_integration(
            proximal, distal, modulatory
        )

        # Check for NMDA spikes
        nmda_spikes = self.compute_nmda_spike(
            integrated,
            threshold=self.nmda_threshold
        )
        return nmda_spikes
```

---

## Mixin Method Reference

> **ðŸ“š For complete mixin API documentation with all method signatures, see [MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md)**

This section focuses on **usage patterns** and **best practices** for working with mixins.

### Common Diagnostic Patterns

**Health Monitoring Pattern**:
```python
class MyRegion(NeuralRegion):
    def forward(self, x):
        output = self._compute(x)

        # Use DiagnosticsMixin to check health
        health = self.check_health()
        if not health.is_healthy:
            logger.warning(f"Health issue: {health.issues}")

        return output
```

**Weight Health Pattern**:
```python
# Check weights for common issues
weight_health = self.check_weight_health(self.weights, name="feedforward")
if weight_health.has_dead_neurons:
    logger.warning(f"Dead neurons detected in {name}")
```

**Spike Monitoring Pattern**:
```python
# Detect pathological activity
if self.detect_runaway_excitation(spikes):
    logger.error("Runaway excitation - reducing learning rate")
    self.learning_rate *= 0.5

if self.detect_silence(spikes):
    logger.warning("Network silence - check inputs")
```

### Neuromodulator Usage Patterns

**Setting Neuromodulators**:
```python
# Set specific neuromodulators (NeuromodulatorMixin)
region.set_neuromodulator("dopamine", 0.8)  # Reward signal
region.set_neuromodulator("acetylcholine", 1.0)  # Attention

# Or set multiple at once
region.set_neuromodulators(dopamine=0.8, acetylcholine=1.0)
```

**Modulating Learning Rates**:
```python
# Get dopamine-modulated learning rate
effective_lr = self.get_effective_learning_rate(
    base_lr=self.config.learning_rate,
    dopamine_sensitivity=2.0
)
```

---

## Usage Patterns

### Diagnostic Monitoring Pattern

```python
class MyRegion(NeuralRegion):
    def forward(self, x):
        output = self._compute(x)

        # Collect comprehensive diagnostics
        diagnostics = self.collect_standard_diagnostics(
            region_name="my_region",
            weight_matrices={"weights": self.weights},
            spike_tensors={"output": output}
        )

        return output
```

### Neuromodulator-Gated Learning Pattern

```python
class MyRegion(NeuralRegion):
    def apply_learning(self, pre_spikes, post_spikes):
        # Get dopamine-modulated learning rate
        effective_lr = self.get_effective_learning_rate(
            base_lr=self.config.learning_rate,
            dopamine_sensitivity=2.0
        )

        # Only learn when dopamine is present
        if self.neuromodulators.dopamine > 0.3:
            weight_update = pre_spikes * post_spikes * effective_lr
            self.weights += weight_update
```

### Action Selection Pattern (Striatum)

```python
class Striatum(NeuralRegion, ActionSelectionMixin):
    def select_action(self, state, exploration=True):
        q_values = self.compute_q_values(state)

        if exploration:
            # Softmax exploration
            action = self.select_action_softmax(
                q_values,
                temperature=self.config.temperature
            )
        else:
            # Greedy exploitation
            action = self.select_action_greedy(
                q_values,
                epsilon=0.0
            )

        return action
```

---

## Method Resolution Order (MRO)

When using multiple mixins, Python's MRO determines method lookup order:

```python
class MyRegion(NeuralComponent, DiagnosticsMixin, NeuromodulatorMixin):
    pass

# MRO: MyRegion â†’ NeuralComponent â†’ DiagnosticsMixin â†’ NeuromodulatorMixin â†’ ...
```

**Best Practices**:
1. **Order matters**: Place most specific mixins first
2. **Avoid conflicts**: Don't use mixins with overlapping method names
3. **Use super()**: Always call `super()` in overridden methods
4. **Check MRO**: Use `MyRegion.__mro__` to inspect resolution order

### Example: Proper super() Usage

```python
class MyRegion(NeuralComponent, DiagnosticsMixin):
    def reset_state(self):
        # Call parent's reset first
        super().reset_state()

        # Then reset region-specific state
        self.custom_state = torch.zeros(...)
```

---

## When to Create New Mixins

**Create a new mixin when**:
- âœ… Functionality is shared across 3+ regions
- âœ… Methods are cohesive (related to single concern)
- âœ… No state dependencies (or minimal state)
- âœ… Can be tested independently

**Don't create a mixin when**:
- âŒ Only used by single region (just use methods)
- âŒ Requires complex region-specific state
- âŒ Logic is tightly coupled to specific region

### Example: Good Mixin Candidate

```python
# âœ… GOOD: Shared, cohesive, stateless
class CompetitiveInhibitionMixin:
    """WTA competition for sparse coding."""

    def apply_wta_inhibition(
        self,
        activity: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Keep top-k neurons, inhibit rest."""
        topk_vals, topk_idx = activity.topk(k)
        inhibited = torch.zeros_like(activity)
        inhibited.scatter_(0, topk_idx, topk_vals)
        return inhibited
```

### Example: Bad Mixin Candidate

```python
# âŒ BAD: Too specific, requires hippocampus state
class PatternCompletionMixin:
    """Only used by hippocampus CA3."""

    def complete_pattern(self, partial_input):
        # Requires CA3-specific recurrent weights
        return self.ca3_recurrent @ partial_input
```

---

## Testing Mixins

Mixins should be tested independently:

```python
# Test mixin in isolation
class MockRegion(DiagnosticsMixin):
    def __init__(self):
        self.weights = torch.randn(10, 10)

def test_weight_diagnostics():
    region = MockRegion()
    diag = region.weight_diagnostics(
        region.weights,
        prefix="test"
    )

    assert "test/mean" in diag
    assert "test/std" in diag
```

---

## See Also

- **[MIXINS_REFERENCE.md](../api/MIXINS_REFERENCE.md)** - Complete API reference for all core mixins
- **[component-parity.md](component-parity.md)** - Component design patterns
- **[state-management.md](state-management.md)** - When to use mixins vs state classes
