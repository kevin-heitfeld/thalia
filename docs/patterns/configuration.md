# Configuration Guide

**Date**: December 7, 2025  
**Purpose**: Understand Thalia's configuration system and hierarchy

---

## Overview

Thalia uses a hierarchical configuration system with specialized config classes for different components. This guide explains:
- Config class hierarchy
- When to create new configs
- How configs inherit and compose
- Best practices

---

## Config Hierarchy

```
ThaliaConfig (top-level)
├─ BrainConfig
│  ├─ RegionConfig (base for all regions)
│  │  ├─ StriatumConfig
│  │  ├─ PrefrontalConfig  
│  │  ├─ LayeredCortexConfig
│  │  │  └─ PredictiveCortexConfig
│  │  ├─ TrisynapticConfig
│  │  └─ CerebellumConfig
│  │
│  └─ PathwayConfig (base for pathways)
│     └─ SpikingPathwayConfig
│
├─ LanguageConfig
├─ TrainingConfig
└─ RobustnessConfig
```

---

## Base Config Classes

### ThaliaConfig - Top Level
**Purpose**: Global configuration for entire brain system  
**File**: `src/thalia/config/thalia_config.py`

**Key Parameters**:
```python
@dataclass
class ThaliaConfig:
    """Top-level configuration for Thalia brain."""
    device: str = "cpu"
    dt_ms: float = 1.0
    seed: Optional[int] = None
    
    # Sub-configs
    brain: BrainConfig = field(default_factory=BrainConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
```

**When to use**: Creating entire brain systems

**Example**:
```python
config = ThaliaConfig(
    device="cuda",
    dt_ms=1.0,
    brain=BrainConfig(
        n_sensory=256,
        enable_hippocampus=True,
    )
)
brain = Brain.from_thalia_config(config)
```

---

### BrainConfig - Brain System Level
**Purpose**: Configuration for multi-region brain architecture  
**File**: `src/thalia/config/brain_config.py`

**Key Parameters**:
```python
@dataclass
class BrainConfig:
    """Configuration for brain system with multiple regions."""
    n_sensory: int = 256
    n_motor: int = 64
    
    # Region toggles
    enable_hippocampus: bool = True
    enable_striatum: bool = True
    enable_cerebellum: bool = False
    
    # Region-specific configs
    striatum: StriatumConfig = field(default_factory=StriatumConfig)
    hippocampus: TrisynapticConfig = field(default_factory=TrisynapticConfig)
    prefrontal: PrefrontalConfig = field(default_factory=PrefrontalConfig)
```

---

### RegionConfig - Base for All Regions
**Purpose**: Minimal interface for all brain regions  
**File**: `src/thalia/regions/base.py`

**Key Parameters**:
```python
@dataclass
class RegionConfig:
    """Base configuration for all brain regions."""
    n_input: int = 128
    n_output: int = 64
    device: str = "cpu"
    dt_ms: float = 1.0
    
    # Learning parameters
    learning_rate: float = 0.01
    w_min: float = 0.0
    w_max: float = 1.0
```

**All region configs inherit from this.**

---

## Region-Specific Configs

### StriatumConfig
**Extends**: `RegionConfig`  
**File**: `src/thalia/regions/striatum/config.py`

**Adds**:
```python
@dataclass
class StriatumConfig(RegionConfig):
    """Striatum-specific parameters."""
    # D1/D2 pathway params
    n_d1: Optional[int] = None
    n_d2: Optional[int] = None
    
    # Dopamine modulation
    dopamine_tau_ms: float = 100.0
    dopamine_baseline: float = 0.2
    
    # Eligibility traces
    eligibility_tau_ms: float = 1000.0
    eligibility_decay: float = 0.95
    
    # Action selection
    softmax_temperature: float = 1.0
    exploration_noise: float = 0.1
```

**Usage**:
```python
striatum_config = StriatumConfig(
    n_input=256,
    n_output=64,
    dopamine_tau_ms=150.0,
    eligibility_tau_ms=800.0,
)
striatum = Striatum(striatum_config)
```

---

### LayeredCortexConfig
**Extends**: `RegionConfig`  
**File**: `src/thalia/regions/cortex/layered_cortex.py`

**Adds**:
```python
@dataclass
class LayeredCortexConfig(RegionConfig):
    """Layered cortex with L4→L2/3→L5 microcircuit."""
    # Layer sizes
    l4_size: Optional[int] = None  # Auto: n_input
    l23_size: Optional[int] = None  # Auto: n_output  
    l5_size: Optional[int] = None  # Auto: n_output // 2
    
    # Layer sparsity
    l4_sparsity: float = 0.1
    l23_sparsity: float = 0.15
    l5_sparsity: float = 0.1
    
    # Plasticity
    stdp_lr: float = 0.01
    stdp_a_plus: float = 0.01
    stdp_a_minus: float = 0.012
```

---

### PredictiveCortexConfig
**Extends**: `LayeredCortexConfig`  
**File**: `src/thalia/regions/cortex/predictive_cortex.py`

**Adds**:
```python
@dataclass
class PredictiveCortexConfig(LayeredCortexConfig):
    """Predictive cortex with error-based learning."""
    # Predictive coding
    prediction_enabled: bool = True
    prediction_tau_ms: float = 50.0
    error_tau_ms: float = 5.0
    prediction_learning_rate: float = 0.01
    
    # Precision (attention)
    use_precision_weighting: bool = True
    initial_precision: float = 1.0
    
    # Attention
    use_attention: bool = True
    attention_type: AttentionType = AttentionType.WINNER_TAKE_ALL
    n_attention_heads: int = 4
```

**Inherits all LayeredCortexConfig parameters plus adds predictive coding.**

---

### TrisynapticConfig
**Extends**: `RegionConfig`  
**File**: `src/thalia/regions/hippocampus/config.py`

**Adds**:
```python
@dataclass
class TrisynapticConfig(RegionConfig):
    """Hippocampus with DG→CA3→CA1 circuit."""
    # Circuit sizes
    dg_size: int = 200
    ca3_size: int = 150
    ca1_size: int = 128
    
    # Theta oscillation
    theta_freq_hz: float = 7.0
    theta_phase_offset: float = 0.0
    
    # Learning
    ca3_learning_rate: float = 0.05
    ca1_learning_rate: float = 0.02
    
    # Pattern separation
    dg_sparsity: float = 0.05
    ca3_sparsity: float = 0.10
```

---

## System-Level Configs

### LanguageConfig
**Purpose**: Language-specific parameters  
**File**: `src/thalia/config/language_config.py`

```python
@dataclass
class LanguageConfig:
    """Configuration for language learning."""
    vocab_size: int = 1000
    max_sequence_length: int = 20
    embedding_dim: int = 128
    
    # Learning
    word_learning_rate: float = 0.1
    syntax_learning_rate: float = 0.05
```

---

### TrainingConfig
**Purpose**: Training loop parameters  
**File**: `src/thalia/config/training_config.py`

```python
@dataclass
class TrainingConfig:
    """Configuration for training."""
    n_episodes: int = 100
    n_trials_per_episode: int = 10
    
    # Optimization
    batch_size: int = 1  # Currently must be 1
    learning_rate: float = 0.01
    
    # Logging
    log_interval: int = 10
    save_checkpoints: bool = True
```

---

### RobustnessConfig
**Purpose**: Noise and robustness parameters  
**File**: `src/thalia/config/robustness_config.py`

```python
@dataclass
class RobustnessConfig:
    """Configuration for robustness mechanisms."""
    # Homeostasis
    enable_homeostasis: bool = True
    homeostasis_tau_ms: float = 10000.0
    
    # Divisive normalization
    enable_divisive_norm: bool = True
    norm_epsilon: float = 1e-6
    
    # E/I balance
    enable_ei_balance: bool = True
    ei_target_ratio: float = 4.0
```

---

## When to Create New Config Classes

### ✅ Create New Config When:

1. **Adding a new brain region**
   ```python
   @dataclass
   class ThalamusConfig(RegionConfig):
       """Thalamus-specific parameters."""
       relay_mode: str = "burst"
       burst_threshold: float = 0.5
   ```

2. **Adding significant new functionality**
   ```python
   @dataclass
   class MemoryConfig:
       """Configuration for episodic memory."""
       buffer_size: int = 1000
       consolidation_threshold: float = 0.8
   ```

3. **When config grows beyond 10-15 parameters**
   - Split into logical sub-configs
   - Keep related parameters together

---

### ❌ Don't Create New Config When:

1. **Adding 1-2 parameters to existing config**
   - Just add fields to existing config

2. **Temporary experimental parameters**
   - Use function arguments instead

3. **Derived values**
   - Compute from other config values, don't store

---

## Config Patterns

### Pattern 1: Auto-Sizing
**Problem**: User doesn't know optimal layer sizes  
**Solution**: Auto-compute from other parameters

```python
@dataclass
class LayeredCortexConfig(RegionConfig):
    l4_size: Optional[int] = None
    l23_size: Optional[int] = None
    
    def __post_init__(self):
        """Auto-size layers if not specified."""
        if self.l4_size is None:
            self.l4_size = self.n_input
        if self.l23_size is None:
            self.l23_size = self.n_output
```

---

### Pattern 2: Config Composition
**Problem**: Complex systems need hierarchical configs  
**Solution**: Nest configs

```python
@dataclass
class BrainConfig:
    striatum: StriatumConfig = field(default_factory=StriatumConfig)
    hippocampus: TrisynapticConfig = field(default_factory=TrisynapticConfig)
    
    def __post_init__(self):
        """Propagate device to all sub-configs."""
        self.striatum.device = self.device
        self.hippocampus.device = self.device
```

---

### Pattern 3: Config Validation
**Problem**: Invalid parameter combinations  
**Solution**: Validate in `__post_init__`

```python
@dataclass
class MyConfig(RegionConfig):
    min_value: float = 0.0
    max_value: float = 1.0
    
    def __post_init__(self):
        """Validate parameters."""
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be < max_value")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
```

---

### Pattern 4: Factory Method
**Problem**: Creating regions from ThaliaConfig  
**Solution**: `from_thalia_config()` class method

```python
class Striatum(BrainRegion):
    @classmethod
    def from_thalia_config(cls, config: ThaliaConfig) -> "Striatum":
        """Create Striatum from ThaliaConfig."""
        return cls(config.brain.striatum)
```

---

## Config Best Practices

### ✅ Use Descriptive Names
```python
# Good
eligibility_tau_ms: float = 1000.0

# Bad
tau: float = 1000.0  # Which tau?
```

---

### ✅ Include Units in Names
```python
# Good
dopamine_tau_ms: float = 100.0
max_frequency_hz: float = 40.0

# Bad
dopamine_tau: float = 100.0  # Milliseconds? Seconds?
```

---

### ✅ Provide Sensible Defaults
```python
@dataclass
class MyConfig(RegionConfig):
    """Good defaults based on literature."""
    theta_frequency_hz: float = 7.0  # Rodent theta: 6-10 Hz
    gamma_frequency_hz: float = 40.0  # Gamma: 30-80 Hz
```

---

### ✅ Document Parameter Meanings
```python
@dataclass
class StriatumConfig(RegionConfig):
    """Striatum configuration.
    
    Parameters:
        eligibility_tau_ms: Time constant for eligibility trace decay.
            Biological range: 500-2000ms (Yagishita et al., 2014)
        
        dopamine_baseline: Tonic dopamine level (0-1 range).
            ~0.2 represents normal baseline firing (~4 Hz)
    """
    eligibility_tau_ms: float = 1000.0
    dopamine_baseline: float = 0.2
```

---

### ✅ Group Related Parameters
```python
@dataclass
class MyConfig(RegionConfig):
    # Network architecture
    n_hidden: int = 128
    n_layers: int = 3
    
    # Learning parameters
    learning_rate: float = 0.01
    momentum: float = 0.9
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 1e-4
```

---

## Common Config Patterns

### Accessing Nested Configs
```python
# Create top-level config
config = ThaliaConfig(
    device="cuda",
    brain=BrainConfig(
        striatum=StriatumConfig(
            eligibility_tau_ms=800.0
        )
    )
)

# Access nested values
print(config.brain.striatum.eligibility_tau_ms)  # 800.0
```

---

### Overriding Defaults
```python
# Override specific parameters
config = ThaliaConfig(
    device="cuda",
    brain=BrainConfig(
        striatum=StriatumConfig(
            n_input=256,
            n_output=64,
            eligibility_tau_ms=800.0,  # Custom value
        )
    )
)
```

---

### Partial Configuration
```python
# Use defaults for most, override a few
striatum_config = StriatumConfig(
    n_input=256,
    n_output=64,
    # All other params use defaults
)
```

---

## Config Reference Tables

### RegionConfig (Base)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_input | int | 128 | Input dimension |
| n_output | int | 64 | Output dimension |
| device | str | "cpu" | Compute device |
| dt_ms | float | 1.0 | Timestep (ms) |
| learning_rate | float | 0.01 | Base learning rate |
| w_min | float | 0.0 | Min weight value |
| w_max | float | 1.0 | Max weight value |

---

### StriatumConfig (extends RegionConfig)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_d1 | int | None | D1 pathway size (auto: n_output) |
| n_d2 | int | None | D2 pathway size (auto: n_output) |
| dopamine_tau_ms | float | 100.0 | Dopamine decay time constant |
| eligibility_tau_ms | float | 1000.0 | Eligibility trace time constant |
| softmax_temperature | float | 1.0 | Action selection temperature |

---

### LayeredCortexConfig (extends RegionConfig)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| l4_size | int | None | Layer 4 size (auto: n_input) |
| l23_size | int | None | Layer 2/3 size (auto: n_output) |
| l5_size | int | None | Layer 5 size (auto: n_output // 2) |
| l4_sparsity | float | 0.1 | L4 activation sparsity |
| stdp_lr | float | 0.01 | STDP learning rate |

---

## FAQ

**Q: Why so many config classes?**  
A: Each region has unique parameters. Separate configs prevent parameter pollution and enable type safety.

**Q: Can I add parameters to RegionConfig?**  
A: Only add truly universal parameters. Region-specific ones go in region configs.

**Q: How do I see all available parameters?**  
A: Check the config class definition or use `dataclasses.fields(ConfigClass)`.

**Q: Should configs be mutable?**  
A: Generally no. Treat as immutable after initialization.

**Q: Can I share configs between regions?**  
A: Yes, but be careful. They share the same object reference.

---

## Summary

**Config Hierarchy**: ThaliaConfig → BrainConfig → RegionConfig → Specific configs

**When to create**: New regions, significant new functionality, >10-15 parameters

**Best practices**:
- Descriptive names with units
- Sensible defaults
- Good documentation
- Group related parameters
- Validate in `__post_init__`

---

**Last Updated**: December 7, 2025  
**See Also**:
- `docs/patterns/state-management.md` - State patterns
- `src/thalia/config/` - Config implementations
- `docs/design/architecture.md` - System architecture
