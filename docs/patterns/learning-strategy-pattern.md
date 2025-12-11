# Learning Strategy Pattern

**Status**: ✅ Implemented (December 11, 2025)  
**Version**: 1.0  
**Related ADRs**: ADR-008 (Neural Component Consolidation)

## Overview

The **Learning Strategy Pattern** consolidates learning rules across brain regions into pluggable, composable strategies. Instead of each region implementing custom `_apply_learning()` methods with duplicated STDP/BCM/three-factor logic, regions instantiate strategies from `thalia.learning.strategies` and use them via a unified interface.

This pattern **eliminates code duplication**, improves **testability**, enables **easy experimentation** with hybrid learning rules, and aligns with **neuroscience literature** (learning rules as concepts, not implementation details).

---

## Problem Statement

### Before: Duplicated Learning Logic

Each region implemented learning differently despite conceptual similarities:

```python
# Striatum: Three-factor rule
class Striatum:
    def _apply_learning(self, ...):
        weight_update = eligibility * dopamine * learning_rate
        self.weights += weight_update

# Hippocampus: One-shot Hebbian
class TrisynapticHippocampus:
    def _apply_learning(self, ...):
        weight_update = pre * post * learning_rate
        self.weights += weight_update

# Cortex: BCM with sliding threshold
class LayeredCortex:
    def _apply_learning(self, ...):
        phi = post * (post - bcm_threshold)
        weight_update = pre * phi * learning_rate
        self.weights += weight_update
```

**Issues**:
- ❌ Code duplication (eligibility traces, weight clamping, metrics)
- ❌ Hard to test learning rules independently
- ❌ Difficult to compose multiple learning rules
- ❌ No discovery mechanism for available rules

---

## Solution: Strategy Pattern

### Architecture

```
thalia.learning.strategies
├── BaseStrategy (abstract base)
│   ├── compute_update(weights, pre, post, **kwargs) → (new_weights, metrics)
│   ├── reset_state()
│   ├── _apply_bounds(weights, dw) → new_weights
│   └── _compute_metrics(old_w, new_w, dw) → Dict[str, float]
│
├── HebbianStrategy: Δw ∝ pre × post
├── STDPStrategy: Spike-timing dependent plasticity
├── BCMStrategy: Bienenstock-Cooper-Munro with sliding threshold
├── ThreeFactorStrategy: Δw = eligibility × modulator
├── ErrorCorrectiveStrategy: Δw = pre × (target - actual)
└── CompositeStrategy: Compose multiple strategies

LearningStrategyRegistry
├── register(name, config_class, aliases) → decorator
├── create(name, config) → strategy instance
├── list_strategies() → List[str]
└── get_metadata(name) → Dict[str, Any]
```

---

## Core Components

### 1. Base Strategy

All strategies inherit from `BaseStrategy`:

```python
from thalia.learning.strategies import BaseStrategy, LearningConfig

class MyStrategy(BaseStrategy):
    def __init__(self, config: LearningConfig):
        super().__init__(config)
        # Initialize strategy-specific state

    def compute_update(
        self,
        weights: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weight update."""
        # 1. Compute raw weight change (dw)
        dw = self._compute_dw(pre, post, **kwargs)
        
        # 2. Apply bounds (soft or hard)
        new_weights = self._apply_bounds(weights, dw)
        
        # 3. Compute metrics
        metrics = self._compute_metrics(weights, new_weights, dw)
        
        return new_weights, metrics

    def reset_state(self) -> None:
        """Reset traces, thresholds, etc."""
        # Override if strategy has state
        pass
```

### 2. Registry Pattern

Register strategies at module load time:

```python
from thalia.learning.strategy_registry import LearningStrategyRegistry

@LearningStrategyRegistry.register(
    "stdp",
    config_class=STDPConfig,
    aliases=["spike_timing"],
    description="Spike-timing dependent plasticity",
    version="1.0",
)
class STDPStrategy(BaseStrategy):
    """STDP learning with LTP/LTD windows."""
    ...
```

Create strategies dynamically:

```python
from thalia.learning import LearningStrategyRegistry, STDPConfig

# Create STDP strategy
stdp = LearningStrategyRegistry.create(
    "stdp",
    STDPConfig(learning_rate=0.02, a_plus=0.01, a_minus=0.012)
)

# List available strategies
available = LearningStrategyRegistry.list_strategies()
# ['hebbian', 'stdp', 'bcm', 'three_factor', 'error_corrective', 'composite']
```

### 3. Strategy Mixin (Optional)

For easier integration, regions can use `LearningStrategyMixin`:

```python
from thalia.learning import LearningStrategyMixin, LearningStrategyRegistry, STDPConfig

class MyRegion(LearningStrategyMixin, BrainRegion):
    def __init__(self, config):
        super().__init__(config)
        
        # Set strategy
        self.learning_strategy = LearningStrategyRegistry.create(
            "stdp",
            STDPConfig(learning_rate=config.stdp_lr)
        )
    
    def forward(self, input_spikes):
        output_spikes = self._compute_output(input_spikes)
        
        # Apply learning automatically
        metrics = self.apply_strategy_learning(
            pre_activity=input_spikes,
            post_activity=output_spikes,
            weights=self.weights,
        )
        
        return output_spikes
```

---

## Available Strategies

### HebbianStrategy

**Rule**: Δw[j,i] = lr × pre[i] × post[j]

**Use Case**: Basic correlation-based learning

```python
from thalia.learning import HebbianStrategy, HebbianConfig

strategy = HebbianStrategy(HebbianConfig(
    learning_rate=0.01,
    normalize=False,
    decay_rate=0.0,
))
```

---

### STDPStrategy

**Rule**: 
- LTP: A+ × pre_trace × post_spike (pre before post)
- LTD: A- × post_trace × pre_spike (post before pre)

**Use Case**: Spike-timing dependent plasticity for cortex, hippocampus

```python
from thalia.learning import STDPStrategy, STDPConfig

strategy = STDPStrategy(STDPConfig(
    learning_rate=0.001,
    a_plus=0.01,      # LTP amplitude
    a_minus=0.012,    # LTD amplitude
    tau_plus=20.0,    # LTP time constant (ms)
    tau_minus=20.0,   # LTD time constant (ms)
))
```

---

### BCMStrategy

**Rule**: Δw[j,i] = lr × pre[i] × φ(post[j], θ[j])

Where φ(c, θ) = c(c - θ) and θ → E[c²]

**Use Case**: Cortical learning with automatic metaplasticity

```python
from thalia.learning import BCMStrategy, BCMConfig

strategy = BCMStrategy(BCMConfig(
    learning_rate=0.01,
    tau_theta=5000.0,   # Threshold time constant (ms)
    theta_init=0.01,    # Initial threshold
    power=2.0,          # Power for threshold (c^p)
))
```

---

### ThreeFactorStrategy

**Rule**: Δw = lr × eligibility × modulator

**Use Case**: Dopamine-modulated learning in striatum, prefrontal

```python
from thalia.learning import ThreeFactorStrategy, ThreeFactorConfig

strategy = ThreeFactorStrategy(ThreeFactorConfig(
    learning_rate=0.005,
    eligibility_tau=100.0,   # Eligibility trace decay (ms)
    modulator_tau=50.0,      # Modulator decay (ms)
))

# During forward pass
strategy.update_eligibility(pre, post)  # Accumulate correlations

# When modulator arrives (e.g., dopamine)
new_weights, metrics = strategy.compute_update(
    weights=self.weights,
    pre=pre,
    post=post,
    modulator=dopamine_level,  # RPE from VTA
)
```

---

### ErrorCorrectiveStrategy

**Rule**: Δw[j,i] = lr × pre[i] × (target[j] - actual[j])

**Use Case**: Supervised learning in cerebellum

```python
from thalia.learning import ErrorCorrectiveStrategy, ErrorCorrectiveConfig

strategy = ErrorCorrectiveStrategy(ErrorCorrectiveConfig(
    learning_rate=0.02,
    error_threshold=0.01,  # Minimum error to trigger learning
))

# During learning
new_weights, metrics = strategy.compute_update(
    weights=self.weights,
    pre=input_spikes,
    post=output_spikes,
    target=target_output,  # Supervised target
)
```

---

### CompositeStrategy

**Compose multiple strategies** for hybrid learning:

```python
from thalia.learning import CompositeStrategy, STDPStrategy, BCMStrategy

# STDP modulated by BCM threshold
composite = CompositeStrategy([
    STDPStrategy(stdp_config),
    BCMStrategy(bcm_config),  # Modulates STDP output
])

# Use like any strategy
new_weights, metrics = composite.compute_update(weights, pre, post)
```

---

## Migration Guide

### Step 1: Identify Current Learning Logic

Find where your region implements learning:

```python
# OLD: Custom learning in forward() or _apply_learning()
class OldCerebellum(BrainRegion):
    def _apply_error_learning(self, output_spikes, target):
        error = target - output_spikes.float()
        dw = self.learning_rate * torch.outer(error, self.input_trace)
        self.weights += dw
        clamp_weights(self.weights, self.w_min, self.w_max)
```

### Step 2: Choose Appropriate Strategy

Match your learning rule to a strategy:

| Current Rule | Strategy | Config |
|--------------|----------|--------|
| Hebbian correlation | `HebbianStrategy` | `HebbianConfig` |
| Spike-timing plasticity | `STDPStrategy` | `STDPConfig` |
| BCM with threshold | `BCMStrategy` | `BCMConfig` |
| Eligibility × dopamine | `ThreeFactorStrategy` | `ThreeFactorConfig` |
| Supervised error | `ErrorCorrectiveStrategy` | `ErrorCorrectiveConfig` |

### Step 3: Instantiate Strategy in `__init__`

```python
from thalia.learning import LearningStrategyRegistry, ErrorCorrectiveConfig

class NewCerebellum(BrainRegion):
    def __init__(self, config):
        super().__init__(config)
        
        # Create strategy
        self.learning_strategy = LearningStrategyRegistry.create(
            "error_corrective",
            ErrorCorrectiveConfig(
                learning_rate=config.learning_rate,
                error_threshold=config.error_threshold,
                w_min=config.w_min,
                w_max=config.w_max,
            )
        )
```

### Step 4: Replace Custom Learning with Strategy Call

```python
# NEW: Use strategy
def _apply_error_learning(self, output_spikes, target):
    # Strategy handles everything: dw computation, bounds, metrics
    new_weights, metrics = self.learning_strategy.compute_update(
        weights=self.weights,
        pre=self.input_trace,  # Or input_spikes
        post=output_spikes,
        target=target,
    )
    
    # Update weights
    self.weights.data.copy_(new_weights)
    
    return metrics
```

### Step 5: Test

```python
# Verify learning still works
region = NewCerebellum(config)
output = region.forward(input_spikes)
metrics = region._apply_error_learning(output, target)

assert metrics["error"] > 0  # Learning occurred
assert region.weights.max() <= config.w_max  # Bounds respected
```

---

## Complete Example: Cerebellum Migration

### Before (Custom Logic)

```python
class Cerebellum(BrainRegion):
    def _apply_error_learning(self, output_spikes, target):
        cfg = self.cerebellum_config
        error = self.climbing_fiber.compute_error(output_spikes.float(), target.float())
        
        if error.abs().max() < cfg.error_threshold:
            return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}
        
        error_sign = torch.sign(error).unsqueeze(1)
        effective_lr = self.get_effective_learning_rate()
        dw = self.stdp_eligibility * error_sign * error.abs().unsqueeze(1) * effective_lr
        
        # Apply soft bounds manually
        if cfg.soft_bounds:
            w_normalized = (self.weights - cfg.w_min) / (cfg.w_max - cfg.w_min + 1e-6)
            ltp_factor = 1.0 - w_normalized
            ltd_factor = w_normalized
            dw = torch.where(dw > 0, dw * ltp_factor, dw * ltd_factor)
        
        old_weights = self.weights.clone()
        self.weights = clamp_weights(self.weights + dw, cfg.w_min, cfg.w_max, inplace=False)
        
        # Compute metrics manually
        actual_dw = self.weights - old_weights
        ltp = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
        ltd = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0
        
        return {"error": error.abs().mean().item(), "ltp": ltp, "ltd": ltd}
```

### After (Using Strategy)

```python
from thalia.learning import LearningStrategyRegistry, ErrorCorrectiveConfig

class Cerebellum(BrainRegion):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize error-corrective strategy
        self.learning_strategy = LearningStrategyRegistry.create(
            "error_corrective",
            ErrorCorrectiveConfig(
                learning_rate=config.learning_rate,
                error_threshold=config.error_threshold,
                w_min=config.w_min,
                w_max=config.w_max,
                soft_bounds=config.soft_bounds,
            )
        )
    
    def _apply_error_learning(self, output_spikes, target):
        cfg = self.cerebellum_config
        
        # Compute error (domain-specific logic kept)
        error = self.climbing_fiber.compute_error(output_spikes.float(), target.float())
        
        # SIMPLE: Strategy handles dw computation, bounds, metrics
        new_weights, metrics = self.learning_strategy.compute_update(
            weights=self.weights,
            pre=self.stdp_eligibility,  # Use accumulated eligibility
            post=output_spikes,
            target=target,
        )
        
        # Apply dopamine modulation (domain-specific)
        effective_lr = self.get_effective_learning_rate()
        new_weights = self.weights + (new_weights - self.weights) * effective_lr
        
        # Update weights
        self.weights.data.copy_(new_weights)
        
        return metrics
```

**Benefits**:
- ✅ 50% less code
- ✅ Bounds and metrics handled automatically
- ✅ Testable strategy independent of region
- ✅ Easy to swap strategies for experiments

---

## Testing Strategies

### Unit Test: Strategy in Isolation

```python
def test_error_corrective_strategy():
    """Test ErrorCorrectiveStrategy computes correct updates."""
    config = ErrorCorrectiveConfig(learning_rate=0.1)
    strategy = ErrorCorrectiveStrategy(config)
    
    # Setup
    weights = torch.ones(3, 4) * 0.5
    pre = torch.tensor([1.0, 0.0, 1.0, 0.0])
    post = torch.tensor([0.5, 0.2, 0.8])
    target = torch.tensor([1.0, 0.0, 1.0])
    
    # Compute update
    new_weights, metrics = strategy.compute_update(
        weights=weights,
        pre=pre,
        post=post,
        target=target,
    )
    
    # Verify
    error = target - post
    expected_dw = 0.1 * torch.outer(error, pre)
    assert torch.allclose(new_weights - weights, expected_dw, atol=0.01)
    assert metrics["error"] > 0
```

### Integration Test: Strategy with Region

```python
def test_cerebellum_uses_strategy():
    """Test Cerebellum uses ErrorCorrectiveStrategy correctly."""
    config = CerebellumConfig(n_input=10, n_output=5)
    cerebellum = Cerebellum(config)
    
    # Forward pass
    input_spikes = torch.rand(10) < 0.3
    output = cerebellum.forward(input_spikes)
    
    # Apply learning
    target = torch.rand(5)
    metrics = cerebellum.deliver_error(target, output)
    
    # Verify strategy was used
    assert hasattr(cerebellum, 'learning_strategy')
    assert isinstance(cerebellum.learning_strategy, ErrorCorrectiveStrategy)
    assert metrics["error"] >= 0
```

---

## Best Practices

### 1. Choose the Right Strategy

| Learning Scenario | Strategy |
|-------------------|----------|
| Unsupervised correlation detection | `HebbianStrategy` |
| Spike-timing dependent plasticity | `STDPStrategy` |
| Cortical feature learning | `BCMStrategy` |
| Reinforcement learning | `ThreeFactorStrategy` |
| Supervised motor learning | `ErrorCorrectiveStrategy` |
| Hybrid (e.g., STDP + BCM) | `CompositeStrategy` |

### 2. Configure Strategies via Region Config

```python
@dataclass
class MyRegionConfig(RegionConfig):
    # Strategy selection
    learning_strategy: str = "stdp"
    
    # Strategy-specific params
    stdp_lr: float = 0.001
    stdp_a_plus: float = 0.01
    stdp_a_minus: float = 0.012
```

Then in `__init__`:

```python
def __init__(self, config):
    super().__init__(config)
    
    # Create strategy from config
    if config.learning_strategy == "stdp":
        strategy_config = STDPConfig(
            learning_rate=config.stdp_lr,
            a_plus=config.stdp_a_plus,
            a_minus=config.stdp_a_minus,
        )
        self.learning_strategy = LearningStrategyRegistry.create("stdp", strategy_config)
```

### 3. Use Mixin for Common Patterns

If your region just needs standard strategy application:

```python
from thalia.learning import LearningStrategyMixin

class MyRegion(LearningStrategyMixin, BrainRegion):
    def forward(self, input_spikes):
        output_spikes = self._compute_output(input_spikes)
        
        # Mixin provides apply_strategy_learning()
        metrics = self.apply_strategy_learning(
            pre_activity=input_spikes,
            post_activity=output_spikes,
            weights=self.weights,
        )
        
        return output_spikes
```

### 4. Compose Strategies for Hybrid Rules

```python
# Combine STDP and BCM
composite = CompositeStrategy([
    STDPStrategy(stdp_config),     # Spike-timing correlations
    BCMStrategy(bcm_config),       # Threshold modulation
])

# Or: STDP + three-factor (dopamine)
composite = CompositeStrategy([
    STDPStrategy(stdp_config),
    ThreeFactorStrategy(tf_config),
])
```

### 5. Reset State Between Episodes

```python
def reset_state(self):
    super().reset_state()
    
    # Reset strategy state (traces, thresholds)
    if self.learning_strategy is not None:
        self.learning_strategy.reset_state()
```

---

## FAQ

### Q: Can I still use custom learning logic?

**A**: Yes! Strategies are optional. If your region has highly specialized learning (e.g., hippocampal consolidation with multiple stages), keep custom logic. Use strategies for common patterns.

### Q: Can pathways use learning strategies?

**A**: **Yes! Pathways inherit from `NeuralComponent` which includes `LearningStrategyMixin`.**

All pathways can use the same strategies as regions - no pathway-specific wrappers needed:

```python
# SpikingPathway uses STDP strategy (migrated from custom _apply_stdp)
from thalia.integration.spiking_pathway import SpikingPathway, SpikingPathwayConfig
from thalia.learning import LearningStrategyRegistry, STDPConfig

# Create pathway with strategy-based learning
config = SpikingPathwayConfig(
    source_size=64,
    target_size=128,
    stdp_lr=0.01,  # Learning rate for STDP
)
pathway = SpikingPathway(config)

# The pathway automatically initializes STDPStrategy in __init__:
#   self.learning_strategy = LearningStrategyRegistry.create('stdp', ...)

# Forward pass applies learning automatically
for timestep in range(n_timesteps):
    source_spikes = get_source_activity()
    target_spikes = pathway(source_spikes, dt=1.0, time_ms=timestep)
    # Learning happens inside forward() via self.apply_strategy_learning()
```

**Component parity in action**: Regions and pathways use identical learning infrastructure!

**Migration example**: SpikingPathway was migrated from custom `_apply_stdp()` to strategy pattern:

```python
# BEFORE (custom learning):
class SpikingPathway(NeuralComponent):
    def forward(self, source_spikes):
        target_spikes = self.neurons(...)
        self._apply_stdp(source_spikes, target_spikes)  # Custom method
        return target_spikes
    
    def _apply_stdp(self, pre, post):
        # 100+ lines of custom STDP logic
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(...)
        dw = self.config.stdp_lr * (ltp - ltd)
        self.weights.data += dw
        clamp_weights(...)

# AFTER (strategy pattern):
class SpikingPathway(NeuralComponent):
    def __init__(self, config):
        super().__init__(config)
        self.learning_strategy = LearningStrategyRegistry.create(
            'stdp',
            STDPConfig(learning_rate=config.stdp_lr, ...)
        )
    
    def forward(self, source_spikes):
        target_spikes = self.neurons(...)
        # Use strategy (inherited from LearningStrategyMixin)
        _ = self.apply_strategy_learning(
            pre_activity=source_spikes,
            post_activity=target_spikes,
            weights=self.weights,
        )
        return target_spikes
```

**Benefits of migration**:
- ✅ Removed 100+ lines of custom STDP code
- ✅ Now can swap strategies (STDP → BCM → Hebbian) by changing 1 line
- ✅ Shares trace management with regions (consistent behavior)
- ✅ Backward compatible (old `learn()` method still works)

**Special pathway cases**:
- **Sensory pathways** (Visual, Audio, Language): No learning needed - they encode, not learn
- **Attention pathways**: Use strategies + pass attention via kwargs
- **Replay pathways**: Don't set `learning_strategy` (no plasticity during replay)

### Q: How do strategies handle neuromodulation?

**A**: Strategies provide the **base learning rule**. Regions handle **neuromodulation** (dopamine, ACh, NE) by:
1. Calling `get_effective_learning_rate()` (includes dopamine modulation)
2. Passing `modulator` parameter to `compute_update()` (for three-factor rule)
3. Gating strategy application (e.g., ACh encoding/retrieval in hippocampus)

```python
# Example: Dopamine-modulated learning
effective_lr = self.get_effective_learning_rate()  # Includes DA modulation
new_weights, metrics = strategy.compute_update(
    weights=self.weights * effective_lr,  # Scale by DA
    pre=pre,
    post=post,
)
```

### Q: Can I create custom strategies?

**A**: Yes! Subclass `BaseStrategy` and register:

```python
@LearningStrategyRegistry.register("my_rule")
class MyCustomStrategy(BaseStrategy):
    def compute_update(self, weights, pre, post, **kwargs):
        # Your custom learning logic
        dw = self._my_custom_rule(pre, post)
        new_weights = self._apply_bounds(weights, dw)
        metrics = self._compute_metrics(weights, new_weights, dw)
        return new_weights, metrics
```

### Q: How do strategies interact with growth?

**A**: Strategies operate on weight tensors. When regions grow:
1. Expand weight tensor (region's responsibility)
2. Strategy automatically handles new dimensions
3. Reset strategy state if needed: `strategy.reset_state()`

---

## Related Documentation

- **Component Parity**: `docs/patterns/component-parity.md`
- **State Management**: `docs/patterns/state-management.md`
- **ADR-008**: Neural Component Consolidation
- **API Reference**: `src/thalia/learning/strategies.py`

---

## Changelog

- **2025-12-11**: Initial implementation (Tier 2.3 of architecture review)
- **2025-12-11**: Added comprehensive documentation and examples
