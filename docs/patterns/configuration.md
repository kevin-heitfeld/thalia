# Configuration Guide

**Date**: December 10, 2025
**Purpose**: Understand Thalia's configuration system and hierarchy
**Status**: ✅ Clean - Duplicates removed, canonical locations established

---

## Overview

Thalia uses a hierarchical configuration system with specialized config classes for different components. This guide explains:
- Config class hierarchy
- When to create new configs
- How configs inherit and compose
- Best practices

**Key Principle**: Each region config is defined in its own module (single source of truth). The central config system imports and re-exports these canonical configs.

---

## Config Hierarchy

```
ThaliaConfig (top-level)
├─ GlobalConfig (device, vocab_size, dt_ms, theta_frequency_hz)
├─ BrainConfig (sizes, timing, oscillator couplings)
│  ├─ RegionSizes (input_size, cortex_size, hippocampus_size, pfc_size, n_actions)
│  └─ Region Configs (imported from canonical locations):
│     ├─ LayeredCortexConfig           → thalia.regions.cortex.config
│     ├─ TrisynapticConfig (Hippo)     → thalia.regions.hippocampus.config
│     ├─ StriatumConfig                → thalia.regions.striatum.config
│     ├─ PrefrontalConfig (PFC)        → thalia.regions.prefrontal
│     └─ CerebellumConfig              → thalia.regions.cerebellum
│
├─ LanguageConfig (encoding, decoding, position)
├─ TrainingConfig (learning rates, checkpointing, logging)
└─ RobustnessConfig (E/I balance, divisive norm, intrinsic plasticity)
```

**✅ No Duplication**: Region configs are defined once in their respective modules and imported by the central config system.

---

## Recent Cleanup (Dec 2025)

### What Was Fixed
1. **Removed Duplicate Configs**: `StriatumConfig`, `PFCConfig`, `CerebellumConfig`, `HippocampusConfig` were duplicated in `brain_config.py` with incomplete parameter sets
2. **Established Canonical Locations**: Each region config is now defined only in its own module
3. **Fixed Imports**: Central config system now imports from canonical locations
4. **Removed Unused Parameters**: VTA-related params (`rpe_avg_tau`, `rpe_clip`) removed from `StriatumConfig` (they belong in `VTAConfig`)

### Migration Guide
If you were importing region configs from `thalia.config`, nothing changes - they're still exported from the same place, just sourced from canonical locations:

```python
# Still works (recommended)
from thalia.config import StriatumConfig, PrefrontalConfig

# Also works (direct import from canonical location)
from thalia.regions.striatum.config import StriatumConfig
from thalia.regions.prefrontal import PrefrontalConfig
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
---

## Base Config Classes

### GlobalConfig - Universal Parameters
**Purpose**: Parameters that affect everything (device, timing, vocabulary)
**File**: `src/thalia/config/global_config.py`

**Key Parameters**:
```python
@dataclass
class GlobalConfig(BaseConfig):
    """Universal parameters shared across all modules."""
    # Timing
    dt_ms: float = 1.0                  # Simulation timestep
    theta_frequency_hz: float = 8.0     # Theta oscillation (4-12 Hz)
    gamma_frequency_hz: float = 40.0    # Gamma oscillation (30-100 Hz)

    # Vocabulary
    vocab_size: int = 50257             # Token vocabulary (GPT-2 default)

    # Sparsity
    default_sparsity: float = 0.05      # Default target sparsity

    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0
```

---

### ThaliaConfig - Top Level
**Purpose**: Global configuration for entire brain system
**File**: `src/thalia/config/thalia_config.py`

**Key Parameters**:
```python
@dataclass
class ThaliaConfig:
    """Top-level configuration for Thalia brain."""
    # Sub-configs
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
```

**When to use**: Creating entire brain systems

**Example**:
```python
config = ThaliaConfig(
    global_=GlobalConfig(device="cuda", vocab_size=10000),
    brain=BrainConfig(
        sizes=RegionSizes(input_size=256, cortex_size=128),
    )
)
brain = EventDrivenBrain.from_config(config)
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
    # Region sizes
    sizes: RegionSizes = field(default_factory=RegionSizes)

    # Region-specific configs (imported from canonical locations)
    cortex: LayeredCortexConfig = field(default_factory=LayeredCortexConfig)
    hippocampus: TrisynapticConfig = field(default_factory=TrisynapticConfig)
    striatum: StriatumConfig = field(default_factory=StriatumConfig)
    pfc: PrefrontalConfig = field(default_factory=PrefrontalConfig)
    cerebellum: CerebellumConfig = field(default_factory=CerebellumConfig)

    # Region type selection
    cortex_type: CortexType = CortexType.LAYERED

    # Timing (trial phases)
    encoding_timesteps: int = 15
    delay_timesteps: int = 10
    test_timesteps: int = 15
```

**Note**: All region configs are imported from their canonical locations, not defined here.

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
    learning_rule: LearningRule = LearningRule.STDP

    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0

    # Sparsity
    sparsity: float = 0.1
```

**All region configs inherit from this.**

---

## Region-Specific Configs (Canonical Locations)

### StriatumConfig
**Extends**: `RegionConfig`
**File**: `src/thalia/regions/striatum/config.py` ✅ (canonical)

**Key Features**:
- Three-factor learning (eligibility × dopamine)
- D1/D2 opponent pathways (Go/No-Go)
- Population coding (multiple neurons per action)
- Adaptive exploration (UCB, uncertainty-driven)

**Selected Parameters**:
```python
@dataclass
class StriatumConfig(RegionConfig):
    """Striatum-specific parameters."""
    # Eligibility traces
    eligibility_tau_ms: float = 1000.0

    # Learning rates
    learning_rate: float = 0.005
    stdp_lr: float = 0.005

    # Population coding
    population_coding: bool = True
    neurons_per_action: int = 10

    # D1/D2 pathways
    d1_lr_scale: float = 1.0
    d2_lr_scale: float = 1.0

    # Homeostasis
    homeostatic_enabled: bool = True
    homeostatic_rate: float = 0.1

    # Action selection
    softmax_temperature: float = 2.0
    lateral_inhibition: bool = True
```

**Usage**:
```python
striatum_config = StriatumConfig(
    n_input=256,
    n_output=64,
    eligibility_tau_ms=800.0,
    softmax_temperature=1.5,
)
striatum = Striatum(striatum_config)
```

---

### LayeredCortexConfig
**Extends**: `RegionConfig`
**File**: `src/thalia/regions/cortex/config.py` ✅ (canonical)

**Key Features**:
- L4→L2/3→L5 microcircuit
- Feedforward inhibition (FFI)
- STDP/BCM learning
- Layer-specific sparsity

**Selected Parameters**:
```python
@dataclass
class LayeredCortexConfig(RegionConfig):
    """Layered cortex with L4→L2/3→L5 microcircuit."""
    # Layer sparsity
    l4_sparsity: float = 0.1
    l23_sparsity: float = 0.15
    l5_sparsity: float = 0.1

    # Plasticity
    stdp_lr: float = 0.01
    bcm_lr: float = 0.001
    stdp_tau_ms: float = 20.0
    bcm_tau_theta_ms: float = 5000.0

    # Inhibition
    lateral_inhibition_enabled: bool = True
    inhibition_strength: float = 2.0
```

---

### PredictiveCortexConfig
**Extends**: `LayeredCortexConfig`
**File**: `src/thalia/regions/cortex/predictive_cortex.py` ✅ (canonical)

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
```

**Inherits all LayeredCortexConfig parameters plus adds predictive coding.**

---

### TrisynapticConfig (Hippocampus)
**Extends**: `RegionConfig`
**File**: `src/thalia/regions/hippocampus/config.py` ✅ (canonical)

**Key Features**:
- DG→CA3→CA1 trisynaptic circuit
- Pattern separation (DG expansion)
- Pattern completion (CA3 recurrence)
- NMDA-based coincidence detection

**Selected Parameters**:
```python
@dataclass
class TrisynapticConfig(RegionConfig):
    """Hippocampus with DG→CA3→CA1 circuit."""
    # DG sparsity (VERY sparse for pattern separation)
    dg_sparsity: float = 0.02
    dg_inhibition: float = 5.0

    # CA3 recurrent dynamics
    ca3_recurrent_strength: float = 0.4
    ca3_sparsity: float = 0.10
    ca3_learning_rate: float = 0.1

    # CA1 output
    ca1_sparsity: float = 0.15

    # NMDA coincidence detection
    nmda_tau: float = 50.0
    nmda_threshold: float = 0.4

    # Learning rates
    learning_rate: float = 0.2
    ec_ca1_learning_rate: float = 0.5
```

---

### PrefrontalConfig (PFC)
**Extends**: `RegionConfig`
**File**: `src/thalia/regions/prefrontal.py` ✅ (canonical)

**Key Features**:
- Dopamine-gated STDP
- Working memory maintenance
- Rule learning
- Top-down attention

**Selected Parameters**:
```python
@dataclass
class PrefrontalConfig(RegionConfig):
    """Prefrontal cortex configuration."""
    # Working memory
    wm_decay_tau_ms: float = 500.0  # Slow decay
    wm_noise_std: float = 0.01

    # Gating
    gate_threshold: float = 0.5  # DA level to open update gate
    gate_strength: float = 2.0

    # Dopamine
    dopamine_tau_ms: float = 100.0
    dopamine_baseline: float = 0.2

    # Learning rates
    wm_lr: float = 0.1
    rule_lr: float = 0.01
    stdp_lr: float = 0.02
```

---

### CerebellumConfig
**Extends**: `RegionConfig`
**File**: `src/thalia/regions/cerebellum.py` ✅ (canonical)

**Key Features**:
- Error-corrective learning
- Parallel fiber → Purkinje cell
- Climbing fiber error signals
- Eligibility traces for temporal credit

**Selected Parameters**:
```python
@dataclass
class CerebellumConfig(RegionConfig):
    """Cerebellum configuration."""
    # Learning rates
    learning_rate_ltp: float = 0.02
    learning_rate_ltd: float = 0.02
    stdp_lr: float = 0.02

    # Error signaling
    error_threshold: float = 0.01
    temporal_window_ms: float = 10.0

    # Eligibility traces
    eligibility_tau_ms: float = 500.0
    heterosynaptic_ratio: float = 0.2
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
