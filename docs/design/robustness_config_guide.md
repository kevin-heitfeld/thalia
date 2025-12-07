# Robustness Configuration Guide

**Last Updated**: December 6, 2025  
**Status**: Evidence-based recommendations from ablation study

## Quick Start

```python
from thalia.config import RobustnessConfig
from thalia.regions import LayeredCortex
from thalia.regions.cortex import LayeredCortexConfig

# Recommended: Use stable() preset for most work
config = LayeredCortexConfig(
    n_input=128,
    n_output=64,
    robustness=RobustnessConfig.stable()  # RECOMMENDED DEFAULT
)
cortex = LayeredCortex(config)
```

## Presets Overview

| Preset | Mechanisms | Overhead | Stability | Use Case |
|--------|-----------|----------|-----------|----------|
| `minimal()` | Divisive norm only | ~5-10% | Basic | Prototyping, debugging |
| `stable()` | + E/I balance | ~15-20% | Good | **Most experiments** ⭐ |
| `full()` | All mechanisms | ~30-40% | Maximum | Production, research |

## Evidence from Ablation Study

Our ablation testing (December 2025) quantified the impact of each mechanism:

### Divisive Normalization: CRITICAL ⚠️

**Impact**: +1080% variance increase without it

```python
# DON'T disable divisive norm unless you have a very good reason
config = RobustnessConfig(
    enable_divisive_norm=False  # ❌ BAD: System becomes unstable
)

# DO use at minimum:
config = RobustnessConfig.minimal()  # ✅ GOOD: Includes divisive norm
```

**What it does**:
- Provides automatic gain control
- Makes responses invariant to input contrast
- Prevents saturation with strong inputs

**When you need it**:
- Always. This is the foundation of stability.

---

### E/I Balance: VALUABLE 📊

**Impact**: +26% variance increase without it

```python
# For research/experiments, add E/I balance:
config = RobustnessConfig.stable()  # ✅ RECOMMENDED
# Includes: divisive norm + E/I balance
```

**What it does**:
- Maintains healthy excitation/inhibition ratio
- Prevents runaway excitation
- Reduces activity variance

**When you need it**:
- Most experiments and production code
- When input strengths vary
- When stability matters more than speed

---

### Intrinsic Plasticity: MINOR ℹ️

**Impact**: 15% adaptation improvement (in short contexts)

```python
# Only enable if you need adaptive thresholds:
config = RobustnessConfig(
    enable_ei_balance=True,
    enable_divisive_norm=True,
    enable_intrinsic_plasticity=True,  # Optional
)
```

**What it does**:
- Adapts neuron thresholds based on firing rates
- Helps maintain target firing rates over time
- May be more useful in longer training contexts

**When you need it**:
- Long training runs (>1000 steps)
- When firing rate homeostasis is critical
- Research into adaptation mechanisms

---

### Criticality Monitoring: DIAGNOSTIC 🔬

**Impact**: Provides insights, doesn't directly improve stability

```python
# Enable for research/diagnostics:
config = RobustnessConfig.full()  # Includes criticality monitoring

# Or enable selectively:
config = RobustnessConfig.stable()
config.enable_criticality = True
```

**What it does**:
- Tracks branching ratio (avalanche dynamics)
- Detects subcritical/supercritical states
- Can apply corrective weight scaling

**When you need it**:
- Research into criticality and SOC
- Diagnosing network dynamics
- When branching ratio matters

---

### Metabolic Constraints: SPARSE CODING 🎯

**Impact**: Encourages efficiency and sparsity

```python
# Enable when sparsity is a goal:
config = RobustnessConfig.full()  # Includes metabolic

# Or customize:
config = RobustnessConfig.stable()
config.enable_metabolic = True
config.metabolic.energy_budget = 100.0  # Adjust budget
```

**What it does**:
- Penalizes excessive activity
- Provides intrinsic reward for efficiency
- Encourages sparse representations

**When you need it**:
- Sparse coding objectives
- Energy-efficient representations
- Biological plausibility research

---

## Usage Patterns

### Pattern 1: Start Simple, Add Complexity

```python
# Start with minimal for fast iteration
config = RobustnessConfig.minimal()
# ... prototype your model ...

# Upgrade to stable for experiments
config = RobustnessConfig.stable()
# ... run experiments ...

# Use full for production
config = RobustnessConfig.full()
# ... deploy ...
```

### Pattern 2: Debug by Ablation

```python
# If something breaks, start minimal
config = RobustnessConfig.minimal()

# Add mechanisms one-by-one to find the issue
config.enable_ei_balance = True  # Does this help?
config.enable_intrinsic_plasticity = True  # Or this?
```

### Pattern 3: Custom Configuration

```python
# Fine-tune for your specific needs
config = RobustnessConfig(
    enable_ei_balance=True,
    enable_divisive_norm=True,
    enable_intrinsic_plasticity=False,
    enable_criticality=True,  # For diagnostics
    enable_metabolic=False,
)

# Customize sub-configs
config.ei_balance.target_ratio = 0.25  # Adjust E/I target
config.divisive_norm.gain = 2.0        # Adjust normalization strength
```

### Pattern 4: Profile-Based Selection

```python
# Quick prototype
if args.mode == "prototype":
    robustness = RobustnessConfig.minimal()

# Research experiments  
elif args.mode == "experiment":
    robustness = RobustnessConfig.stable()  # MOST COMMON

# Production deployment
elif args.mode == "production":
    robustness = RobustnessConfig.full()
```

---

## Decision Tree

```
Need robustness mechanisms?
│
├─ No (just exploring) → Use RobustnessConfig.minimal()
│                        (divisive norm only)
│
├─ Yes, for experiments → Use RobustnessConfig.stable() ⭐
│                         (divisive norm + E/I balance)
│                         RECOMMENDED DEFAULT
│
└─ Yes, maximum → Use RobustnessConfig.full()
                  (all mechanisms enabled)
```

---

## Performance vs Stability Trade-off

| Configuration | Overhead | Variance Reduction | When to Use |
|---------------|----------|-------------------|-------------|
| No robustness | 0% | 0% (baseline) | Never recommended |
| `minimal()` | 5-10% | ~50% | Prototyping |
| `stable()` | 15-20% | ~75% | Most work ⭐ |
| `full()` | 30-40% | ~85% | Production |

**Recommendation**: The overhead of `stable()` (15-20%) is worth it for the stability improvement (75% variance reduction) in almost all cases.

---

## Common Mistakes

### ❌ Mistake 1: Disabling Divisive Norm

```python
# DON'T DO THIS
config = RobustnessConfig(enable_divisive_norm=False)
```

**Problem**: System becomes unstable (+1080% variance)  
**Solution**: Always keep divisive norm enabled

### ❌ Mistake 2: Using No Robustness

```python
# DON'T DO THIS
cortex = LayeredCortex(LayeredCortexConfig(
    n_input=128, n_output=64
    # Missing: robustness config
))
```

**Problem**: Defaults may not be optimal  
**Solution**: Explicitly specify robustness preset

### ❌ Mistake 3: Copying Full Config Everywhere

```python
# DON'T DO THIS for prototyping
config = RobustnessConfig.full()  # Unnecessary overhead
```

**Problem**: Slows down iteration  
**Solution**: Start with `minimal()` or `stable()`

---

## Examples

### Example 1: Basic Experiment

```python
from thalia.config import RobustnessConfig
from thalia.regions import LayeredCortex
from thalia.regions.cortex import LayeredCortexConfig

# Recommended configuration
config = LayeredCortexConfig(
    n_input=128,
    n_output=64,
    robustness=RobustnessConfig.stable()  # ⭐ RECOMMENDED
)

cortex = LayeredCortex(config)

# Run experiment
for step in range(1000):
    output = cortex.forward(input_data[step])
```

### Example 2: Debugging Instability

```python
# Start minimal
config = RobustnessConfig.minimal()
cortex = LayeredCortex(LayeredCortexConfig(
    n_input=128, n_output=64,
    robustness=config
))

# If you see instability, enable E/I balance
config.enable_ei_balance = True

# Still unstable? Check diagnostics
diagnostics = cortex.get_diagnostics()
print(f"E/I ratio: {diagnostics['ei_ratio']}")
print(f"Mean activity: {diagnostics['mean_activity']}")
```

### Example 3: Sparse Coding

```python
# Enable metabolic constraints for sparsity
config = RobustnessConfig.stable()  # Start with stable base
config.enable_metabolic = True      # Add metabolic
config.metabolic.energy_budget = 50.0  # Tighten budget for more sparsity

cortex = LayeredCortex(LayeredCortexConfig(
    n_input=128, n_output=64,
    robustness=config
))
```

### Example 4: Research on Criticality

```python
# Enable criticality monitoring
config = RobustnessConfig.stable()
config.enable_criticality = True

cortex = LayeredCortex(LayeredCortexConfig(
    n_input=128, n_output=64,
    robustness=config
))

# Track branching ratio over time
for step in range(1000):
    output = cortex.forward(input_data[step])
    
    diagnostics = cortex.get_diagnostics()
    branching_ratio = diagnostics.get('branching_ratio', None)
    if branching_ratio:
        print(f"Step {step}: branching ratio = {branching_ratio:.3f}")
```

---

## Summary

**Default recommendation for 95% of use cases:**

```python
robustness = RobustnessConfig.stable()
```

This provides excellent stability with reasonable overhead, backed by empirical evidence from our ablation study.

**When to deviate:**
- Prototyping/debugging → `minimal()`
- Production/maximum stability → `full()`
- Specific needs → Custom configuration

**Key insight from ablation study:**
- Divisive normalization is **CRITICAL** (never disable)
- E/I balance is **VALUABLE** (use in most cases)
- Other mechanisms are useful but not essential for basic stability
