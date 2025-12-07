# Mixin Patterns Guide

**Date**: December 7, 2025  
**Purpose**: Understanding Thalia's mixin system

---

## Overview

Thalia uses **mixins** to share functionality across brain regions without deep inheritance hierarchies. This guide covers:
- What mixins are available
- What methods each mixin provides
- How to use mixins effectively
- Method resolution order (MRO) considerations

---

## Available Mixins

### 1. DiagnosticsMixin
**File**: `src/thalia/core/diagnostics_mixin.py`  
**Purpose**: Health monitoring and diagnostics

**Provides**:
```python
# Health checks
check_health() -> HealthMetrics
get_firing_rate(spikes: torch.Tensor) -> float
check_weight_health(weights: torch.Tensor, name: str) -> WeightHealth

# Spike monitoring
detect_runaway_excitation(spikes: torch.Tensor) -> bool
detect_silence(spikes: torch.Tensor) -> bool

# Weight monitoring  
check_gradient_health(param: torch.nn.Parameter) -> GradientHealth
```

**Usage**:
```python
class MyRegion(BrainRegion, DiagnosticsMixin):
    def forward(self, x):
        output = self._compute(x)
        
        # Check health after computation
        health = self.check_health()
        if not health.is_healthy:
            logger.warning(f"Health issue: {health.issues}")
        
        return output
```

---

### 2. ActionSelectionMixin
**File**: `src/thalia/regions/striatum/action_selection_mixin.py`  
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
class Striatum(BrainRegion, ActionSelectionMixin):
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
class MySpikingRegion(BrainRegion, SpikingNeuronMixin):
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
class L5Pyramidal(BrainRegion, DendriticComputationMixin):
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

### DiagnosticsMixin Methods

```python
def check_health(self) -> HealthMetrics:
    """Check overall region health.
    
    Returns:
        HealthMetrics with is_healthy, firing_rate, weight_stats, issues
    """

def get_firing_rate(self, spikes: torch.Tensor) -> float:
    """Compute firing rate from spikes.
    
    Args:
        spikes: Binary spike tensor [batch, neurons] or [batch, neurons, time]
    
    Returns:
        Firing rate in Hz
    """

def check_weight_health(
    self,
    weights: torch.Tensor,
    name: str = "weights"
) -> WeightHealth:
    """Check weight matrix for issues.
    
    Checks:
        - NaN/Inf values
        - Dead neurons (all weights zero)
        - Extreme values (>5 std from mean)
    
    Returns:
        WeightHealth with has_nan, has_inf, has_dead_neurons, extreme_count
    """

def detect_runaway_excitation(
    self,
    spikes: torch.Tensor,
    threshold: float = 0.9
) -> bool:
    """Detect if >90% of neurons are firing.
    
    Args:
        spikes: Binary spike tensor
        threshold: Fraction of neurons firing to trigger alert
    
    Returns:
        True if runaway excitation detected
    """

def detect_silence(
    self,
    spikes: torch.Tensor,
    threshold: float = 0.01
) -> bool:
    """Detect if <1% of neurons are firing.
    
    Args:
        spikes: Binary spike tensor
        threshold: Minimum fraction of neurons that should fire
    
    Returns:
        True if silence detected
    """

def check_gradient_health(
    self,
    param: torch.nn.Parameter
) -> GradientHealth:
    """Check parameter gradients for issues.
    
    Checks:
        - Gradient norm
        - NaN/Inf in gradients
        - Vanishing gradients (norm < 1e-7)
        - Exploding gradients (norm > 10)
    
    Returns:
        GradientHealth with gradient_norm, has_nan, is_vanishing, is_exploding
    """
```

---

### ActionSelectionMixin Methods

```python
def select_action_softmax(
    self,
    q_values: torch.Tensor,
    temperature: float = 1.0
) -> int:
    """Select action using softmax policy.
    
    Args:
        q_values: Action values [n_actions]
        temperature: Softmax temperature (higher = more random)
    
    Returns:
        Selected action index
    """

def select_action_greedy(
    self,
    q_values: torch.Tensor,
    epsilon: float = 0.1
) -> int:
    """Epsilon-greedy action selection.
    
    Args:
        q_values: Action values [n_actions]
        epsilon: Probability of random action
    
    Returns:
        Selected action index
    """

def select_action_thompson(
    self,
    q_values: torch.Tensor,
    uncertainty: torch.Tensor
) -> int:
    """Thompson sampling (sample from posterior).
    
    Args:
        q_values: Mean action values [n_actions]
        uncertainty: Uncertainty estimates [n_actions]
    
    Returns:
        Selected action index
    """

def compute_policy(
    self,
    q_values: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Compute softmax policy without sampling.
    
    Args:
        q_values: Action values [n_actions]
        temperature: Softmax temperature
    
    Returns:
        Action probabilities [n_actions]
    """

def add_exploration_noise(
    self,
    q_values: torch.Tensor,
    noise_std: float = 0.1
) -> torch.Tensor:
    """Add Gaussian exploration noise to Q-values.
    
    Args:
        q_values: Action values [n_actions]
        noise_std: Standard deviation of noise
    
    Returns:
        Noisy Q-values
    """
```

---

### SpikingNeuronMixin Methods

```python
def lif_dynamics(
    self,
    v_mem: torch.Tensor,
    i_syn: torch.Tensor,
    dt_ms: float = 1.0,
    tau_mem_ms: float = 20.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Leaky integrate-and-fire dynamics.
    
    dV/dt = (-V + I) / tau_mem
    
    Args:
        v_mem: Membrane voltage [batch, neurons]
        i_syn: Synaptic current [batch, neurons]
        dt_ms: Timestep in milliseconds
        tau_mem_ms: Membrane time constant
        v_threshold: Spike threshold
        v_reset: Reset voltage after spike
    
    Returns:
        (new_v_mem, spikes) both [batch, neurons]
    """

def adaptive_lif_dynamics(
    self,
    v_mem: torch.Tensor,
    adaptation: torch.Tensor,
    i_syn: torch.Tensor,
    dt_ms: float = 1.0,
    tau_mem_ms: float = 20.0,
    tau_adapt_ms: float = 100.0,
    adaptation_increment: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adaptive LIF with spike-frequency adaptation.
    
    Returns:
        (new_v_mem, new_adaptation, spikes)
    """

def spike_from_voltage(
    self,
    v_mem: torch.Tensor,
    threshold: float = 1.0
) -> torch.Tensor:
    """Generate binary spikes from voltage.
    
    Args:
        v_mem: Membrane voltage
        threshold: Spike threshold
    
    Returns:
        Binary spikes (0 or 1)
    """

def soft_spike(
    self,
    v_mem: torch.Tensor,
    threshold: float = 1.0,
    sharpness: float = 10.0
) -> torch.Tensor:
    """Differentiable soft spike function (sigmoid).
    
    Args:
        v_mem: Membrane voltage
        threshold: Center of sigmoid
        sharpness: Slope of sigmoid
    
    Returns:
        Soft spikes (0 to 1)
    """
```

---

### DendriticComputationMixin Methods

```python
def dendritic_integration(
    self,
    proximal_input: torch.Tensor,
    distal_input: torch.Tensor,
    modulatory_input: Optional[torch.Tensor] = None,
    proximal_weight: float = 0.7,
    distal_weight: float = 0.3
) -> torch.Tensor:
    """Integrate inputs across dendritic compartments.
    
    Args:
        proximal_input: Feed-forward input (near soma)
        distal_input: Feedback input (apical dendrite)
        modulatory_input: Optional neuromodulation
        proximal_weight: Weight for proximal input
        distal_weight: Weight for distal input
    
    Returns:
        Integrated dendritic signal
    """

def compute_nmda_spike(
    self,
    dendritic_voltage: torch.Tensor,
    threshold: float = 0.5,
    sharpness: float = 10.0
) -> torch.Tensor:
    """Compute NMDA-mediated dendritic spike.
    
    Args:
        dendritic_voltage: Voltage in apical dendrite
        threshold: NMDA spike threshold
        sharpness: Spike sharpness
    
    Returns:
        NMDA spike strength (0 to 1)
    """

def plateau_potential(
    self,
    dendritic_activity: torch.Tensor,
    tau_ms: float = 50.0,
    dt_ms: float = 1.0
) -> torch.Tensor:
    """Compute prolonged plateau potential after dendritic spike.
    
    Args:
        dendritic_activity: Dendritic spike activity
        tau_ms: Plateau decay time constant
        dt_ms: Timestep
    
    Returns:
        Plateau potential
    """
```

---

## Method Resolution Order (MRO)

When a class inherits from multiple mixins, **method resolution order** matters.

### Example MRO:
```python
class Striatum(BrainRegion, DiagnosticsMixin, ActionSelectionMixin):
    pass

# MRO: Striatum → BrainRegion → DiagnosticsMixin → ActionSelectionMixin → object
```

**Method lookup**:
1. Search `Striatum`
2. Search `BrainRegion`
3. Search `DiagnosticsMixin`
4. Search `ActionSelectionMixin`
5. Search `object`

---

### Name Collision Example

If two mixins have the same method:

```python
class MixinA:
    def process(self, x):
        return x * 2

class MixinB:
    def process(self, x):
        return x + 1

class MyClass(MixinA, MixinB):
    pass

obj = MyClass()
obj.process(5)  # Returns 10 (MixinA's version, first in MRO)
```

**Solution**: Use descriptive method names to avoid collisions.

---

## When to Create a New Mixin

### ✅ Create New Mixin When:

1. **Multiple regions need the same functionality**
   ```python
   # Good: Create mixin for shared oscillation code
   class OscillationMixin:
       def generate_theta(self, freq_hz, phase): ...
       def generate_gamma(self, freq_hz): ...
   
   class Hippocampus(BrainRegion, OscillationMixin):
       pass
   
   class PrefrontalCortex(BrainRegion, OscillationMixin):
       pass
   ```

2. **Cross-cutting concerns** (logging, diagnostics, monitoring)
   ```python
   class PerformanceMonitoringMixin:
       def start_timing(self): ...
       def stop_timing(self): ...
       def log_performance(self): ...
   ```

3. **Optional functionality** (can be mixed in or not)
   ```python
   class VisualizationMixin:
       def plot_activity(self): ...
       def save_animation(self): ...
   ```

---

### ❌ Don't Create Mixin When:

1. **Only one class needs it** → Just add methods to that class
2. **Functionality is tightly coupled to state** → Use inheritance instead
3. **Complex interactions between mixins** → Refactor to composition

---

## Mixin Best Practices

### ✅ Keep Mixins Focused
```python
# Good: Focused on one concern
class DiagnosticsMixin:
    def check_health(self): ...
    def get_firing_rate(self): ...

# Bad: Too many unrelated things
class UtilsMixin:
    def check_health(self): ...
    def select_action(self): ...
    def load_weights(self): ...
    def plot_results(self): ...
```

---

### ✅ Use Descriptive Method Names
```python
# Good: Clear what mixin provides
class SpikingNeuronMixin:
    def lif_dynamics(self, ...): ...
    def adaptive_lif_dynamics(self, ...): ...

# Bad: Generic names likely to collide
class NeuronMixin:
    def update(self, ...): ...  # What kind of update?
    def compute(self, ...): ...  # Compute what?
```

---

### ✅ Document Mixin Dependencies
```python
class MyMixin:
    """Provides utility methods for brain regions.
    
    This mixin assumes the class has:
        - self.state.spikes (torch.Tensor)
        - self.config.dt_ms (float)
    
    Example:
        class MyRegion(BrainRegion, MyMixin):
            pass
    """
```

---

### ✅ Avoid State in Mixins
```python
# Good: Stateless mixin (pure functions)
class ActionSelectionMixin:
    def select_action_softmax(self, q_values, temperature):
        # No self.state or self.attributes
        probs = torch.softmax(q_values / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()

# Bad: Mixin with state (couples to host class)
class ActionSelectionMixin:
    def __init__(self):
        self.last_action = None  # State in mixin
```

---

### ✅ Make Mixins Independent
```python
# Good: Can use DiagnosticsMixin without ActionSelectionMixin
class DiagnosticsMixin:
    def check_health(self):
        # Standalone functionality
        pass

# Bad: DiagnosticsMixin depends on ActionSelectionMixin
class DiagnosticsMixin:
    def check_health(self):
        # Assumes self.select_action() exists
        action = self.select_action(...)  # Coupling!
```

---

## Real-World Examples

### Example 1: Striatum
```python
class Striatum(BrainRegion, DiagnosticsMixin, ActionSelectionMixin):
    """Basal ganglia striatum with action selection.
    
    Mixins Used:
        - DiagnosticsMixin: check_health(), detect_runaway_excitation()
        - ActionSelectionMixin: select_action_softmax()
    """
    
    def forward(self, sensory_input, reward_signal):
        # Compute Q-values
        q_values = self._compute_q_values(sensory_input)
        
        # Use ActionSelectionMixin
        action = self.select_action_softmax(
            q_values,
            temperature=self.config.softmax_temperature
        )
        
        # Use DiagnosticsMixin
        health = self.check_health()
        if not health.is_healthy:
            logger.warning("Striatum health issue")
        
        return action
```

---

### Example 2: Layer 5 Pyramidal Neurons
```python
class L5Pyramidal(BrainRegion, SpikingNeuronMixin, DendriticComputationMixin):
    """Layer 5 pyramidal neurons with dendritic computation.
    
    Mixins Used:
        - SpikingNeuronMixin: lif_dynamics()
        - DendriticComputationMixin: dendritic_integration(), compute_nmda_spike()
    """
    
    def forward(self, proximal_input, distal_feedback):
        # Use DendriticComputationMixin
        integrated = self.dendritic_integration(
            proximal_input,
            distal_feedback
        )
        
        # Use SpikingNeuronMixin
        self.state.v_mem, spikes = self.lif_dynamics(
            self.state.v_mem,
            integrated,
            dt_ms=self.config.dt_ms
        )
        
        return spikes
```

---

### Example 3: Hippocampus (No Mixins)
```python
class TrisynapticHippocampus(BrainRegion):
    """Hippocampus with specialized circuit.
    
    Mixins Used: None
    
    Note: Hippocampus uses highly specialized computations
    that aren't shared with other regions, so mixins aren't needed.
    """
    
    def forward(self, input_pattern):
        # Specialized DG→CA3→CA1 circuit
        dg_out = self._dentate_gyrus(input_pattern)
        ca3_out = self._ca3_autoassociation(dg_out)
        ca1_out = self._ca1_comparison(ca3_out, input_pattern)
        return ca1_out
```

---

## Adding Mixin Documentation to Classes

### Recommended Pattern:
```python
class Striatum(BrainRegion, DiagnosticsMixin, ActionSelectionMixin):
    """Basal ganglia striatum for action selection.
    
    The striatum learns action values through dopamine-modulated
    plasticity and selects actions using softmax policy.
    
    Mixins Provide:
        From DiagnosticsMixin:
            - check_health() → HealthMetrics
            - get_firing_rate(spikes) → float
            - detect_runaway_excitation(spikes) → bool
        
        From ActionSelectionMixin:
            - select_action_softmax(q_values, temperature) → int
            - select_action_greedy(q_values, epsilon) → int
            - compute_policy(q_values, temperature) → torch.Tensor
    
    Attributes:
        state (StriatumState): Mutable state (spikes, dopamine, eligibility)
        config (StriatumConfig): Configuration parameters
        w_cortex_to_d1 (torch.Tensor): Cortex → D1 pathway weights
        w_cortex_to_d2 (torch.Tensor): Cortex → D2 pathway weights
    """
```

---

## FAQ

**Q: Can mixins have `__init__`?**  
A: Yes, but be careful with `super().__init__()` to avoid MRO issues. Prefer stateless mixins.

**Q: How do I know which methods come from mixins?**  
A: Check the class docstring (should list mixin methods) or use `dir(obj)` and `type(obj).__mro__`.

**Q: What if two mixins have the same method?**  
A: First one in the inheritance list wins (left-to-right). Avoid name collisions with descriptive names.

**Q: Should I create a mixin or use composition?**  
A: **Mixin** for stateless utilities shared across many classes. **Composition** for stateful objects.

**Q: Can a mixin inherit from another mixin?**  
A: Yes, but keep the hierarchy shallow. Deep mixin hierarchies are hard to reason about.

---

## Summary

**Available Mixins**:
- `DiagnosticsMixin`: Health monitoring
- `ActionSelectionMixin`: Action selection strategies
- `SpikingNeuronMixin`: Spiking neuron dynamics
- `DendriticComputationMixin`: Dendritic integration

**When to use**:
- Multiple regions need same functionality
- Cross-cutting concerns (diagnostics, monitoring)
- Optional functionality

**Best practices**:
- Keep mixins focused and stateless
- Use descriptive method names
- Document what the mixin assumes
- List mixin methods in class docstrings

---

**Last Updated**: December 7, 2025  
**See Also**:
- `docs/patterns/state-management.md` - State patterns
- `docs/patterns/configuration.md` - Configuration guide
- `src/thalia/core/` - Mixin implementations
