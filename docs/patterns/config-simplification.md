# Config Hierarchy Simplification

**Date**: December 11, 2025
**Status**: Implemented
**Related**: Architecture Review 2025-12-11, Section 2.4

## Overview

The config hierarchy has been simplified and consolidated to reduce complexity, eliminate duplication, and provide a clearer structure for configuration management.

## Changes Made

### 1. New Configuration Classes

#### TrainingConfig (`config/training_config.py`)

Consolidates all training-specific parameters that were previously scattered or missing:

```python
from thalia.config import TrainingConfig

config = TrainingConfig(
    n_epochs=100,
    use_stdp=True,
    use_bcm=True,
    use_hebbian=True,
    use_homeostasis=True,
    learning_rate_scale=1.0,
    checkpoint_every_n_epochs=10,
)
```

**Key fields**:
- Learning toggles: `use_stdp`, `use_bcm`, `use_hebbian`, `use_homeostasis`
- Training params: `n_epochs`, `batch_size`, `learning_rate_scale`
- Monitoring: `validate_every_n_epochs`, `log_diagnostics`, `checkpoint_every_n_epochs`
- Curriculum: `use_curriculum`, `curriculum_start_difficulty`, `curriculum_end_difficulty`

#### NeuromodulationConfig (`config/brain_config.py`)

Consolidates neuromodulator parameters that were previously in BrainConfig:

```python
from thalia.config import NeuromodulationConfig

config = NeuromodulationConfig(
    dopamine_baseline=0.0,
    dopamine_learning_threshold=0.01,
    dopamine_decay_tau_ms=100.0,
    use_norepinephrine=False,
    use_acetylcholine=True,
)
```

**Key fields**:
- Dopamine (VTA): `dopamine_baseline`, `dopamine_learning_threshold`, `dopamine_decay_tau_ms`
- Norepinephrine (LC): `use_norepinephrine`, `norepinephrine_baseline`, `norepinephrine_gain_scale`
- Acetylcholine (NB): `use_acetylcholine`, `acetylcholine_encoding_level`, `acetylcholine_retrieval_level`

#### PathwayConfig (`config/base.py`)

New base class for pathway configurations (completing component parity):

```python
from thalia.config import PathwayConfig

config = PathwayConfig(
    n_input=128,
    n_output=64,
    axonal_delay_ms=5.0,
    stdp_enabled=True,
    learning_rate=0.001,
    sparsity=0.1,
)
```

**Key fields**:
- Dimensions: `n_input`, `n_output`
- Weight bounds: `w_min`, `w_max`
- Learning: `stdp_enabled`, `learning_rate`
- Pathway params: `sparsity`, `bidirectional`

### 2. Enhanced Base Classes

#### RegionConfigBase

Added weight bounds to base class (previously duplicated across regions):

```python
@dataclass
class RegionConfigBase(NeuralComponentConfig):
    n_input: int = 128
    n_output: int = 64
    learn: bool = True
    w_min: float = 0.0  # NEW
    w_max: float = 1.0  # NEW
```

### 3. Updated ThaliaConfig

Now includes all sub-configs with clear hierarchy:

```python
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, TrainingConfig

config = ThaliaConfig(
    global_=GlobalConfig(device="cuda", vocab_size=10000),
    brain=BrainConfig(sizes=RegionSizes(cortex_size=256)),
    training=TrainingConfig(n_epochs=100, use_stdp=True),
    language=LanguageConfig(),
    robustness=RobustnessConfig(),
)
```

## Clear Hierarchy

The new hierarchy is:

```
ThaliaConfig (top-level)
├── global_: GlobalConfig
│   ├── device, dtype, seed
│   ├── dt_ms, theta_frequency_hz
│   ├── vocab_size, default_sparsity
│   └── w_min, w_max
├── brain: BrainConfig
│   ├── sizes: RegionSizes
│   ├── neuromodulation: NeuromodulationConfig  ← NEW
│   ├── cortex, hippocampus, striatum, pfc, cerebellum (region configs)
│   └── oscillator_couplings, timing, feature flags
├── training: TrainingConfig  ← NEW
│   ├── Learning toggles (use_stdp, use_bcm, etc)
│   ├── Training params (n_epochs, batch_size, etc)
│   └── Monitoring & curriculum
├── language: LanguageConfig
│   ├── encoding, decoding, position, sequence_memory
│   └── (unchanged)
└── robustness: RobustnessConfig
    └── (unchanged)
```

## Migration Guide

### For Existing Code

If you have code that accesses config fields, **no changes required** for most cases. The new fields are additive.

#### Using new TrainingConfig

**Before** (would have caused AttributeError):
```python
# This was broken - training field didn't exist
print(config.training.use_stdp)
```

**Now** (works correctly):
```python
config = ThaliaConfig(
    training=TrainingConfig(use_stdp=True, n_epochs=100)
)
print(config.training.use_stdp)  # ✅ Works
```

#### Using new NeuromodulationConfig

**Before**:
```python
# Neuromodulation params were scattered or hardcoded
brain = Brain(...)
brain.set_dopamine(0.5)  # Manual setting
```

**Now**:
```python
config = ThaliaConfig(
    brain=BrainConfig(
        neuromodulation=NeuromodulationConfig(
            dopamine_baseline=0.5,
            dopamine_learning_threshold=0.01,
        )
    )
)
brain = Brain.from_thalia_config(config)
# Neuromodulation initialized from config
```

#### Using new PathwayConfig

**Before** (pathways had inconsistent config):
```python
pathway = SpikingPathway(n_input=128, n_output=64, stdp_enabled=True)
```

**Now** (can use standardized config):
```python
from thalia.config import PathwayConfig

config = PathwayConfig(
    n_input=128,
    n_output=64,
    stdp_enabled=True,
    learning_rate=0.001,
)
pathway = SpikingPathway(config)
```

### For Region Implementations

Region configs can now use weight bounds from base class:

```python
@dataclass
class MyRegionConfig(RegionConfigBase):
    # No need to redefine w_min/w_max - inherited from base
    my_custom_param: float = 0.5
```

## Benefits

### 1. Clear Separation of Concerns

- **GlobalConfig**: Universal parameters (device, timing, vocab)
- **BrainConfig**: Brain architecture (regions, neuromodulation)
- **TrainingConfig**: Training procedure (learning toggles, epochs)
- **LanguageConfig**: Language processing (encoding, decoding)

### 2. Reduced Duplication

- Weight bounds (`w_min`, `w_max`) now in `RegionConfigBase` and `PathwayConfig`
- Neuromodulation params consolidated in `NeuromodulationConfig`
- Training params consolidated in `TrainingConfig`

### 3. Component Parity

- Regions have `RegionConfigBase`
- Pathways have `PathwayConfig`
- Both are equals in the architecture (per Component Parity Principle)

### 4. Better Discoverability

Users can now find all training-related params in one place:

```python
config.training.  # IDE autocomplete shows all training options
config.brain.neuromodulation.  # IDE autocomplete shows all neuromodulation options
```

## Testing

### Verify Imports

```python
# Test that all new classes are exported
from thalia.config import (
    TrainingConfig,
    NeuromodulationConfig,
    PathwayConfig,
    ThaliaConfig,
)

config = ThaliaConfig()
assert hasattr(config, 'training')
assert hasattr(config.brain, 'neuromodulation')
print("✅ All imports working")
```

### Verify Hierarchy

```python
config = ThaliaConfig()

# Check training config
assert config.training.use_stdp == True
assert config.training.n_epochs == 10

# Check neuromodulation config
assert config.brain.neuromodulation.dopamine_baseline == 0.0
assert config.brain.neuromodulation.use_acetylcholine == True

# Check pathway config
pathway_config = PathwayConfig(n_input=128, n_output=64)
assert pathway_config.w_min == 0.0
assert pathway_config.w_max == 1.0

print("✅ Hierarchy working correctly")
```

## Future Work

### Phase 2 (Optional, Future)

These are **not implemented yet** but could be considered:

1. **Further consolidate region configs**: Extract common neuron config, learning config, homeostasis config into separate classes that regions compose

2. **Simplify BrainConfig**: Currently has ~15 feature flags. Could group into sub-configs like:
   - `execution: ExecutionConfig` (parallel, timing)
   - `features: FeatureConfig` (goal_conditioning, model_based_planning)

## Config Validation Helpers

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

The config system now includes comprehensive validation helpers that check for common configuration mistakes automatically.

### Available Validation Methods

#### 1. `validate_striatum_pfc_sizes()`

Ensures PFC size matches when goal conditioning is enabled.

```python
config = ThaliaConfig(
    brain=BrainConfig(
        sizes=RegionSizes(pfc_size=32),
        use_goal_conditioning=True,
    )
)

issues = config.validate_striatum_pfc_sizes()
if issues:
    print("❌ PFC size mismatch:", issues)
else:
    print("✅ PFC sizes valid")
```

**Checks**:
- When `use_goal_conditioning=True`, verifies `striatum_pfc_size == pfc_size`
- Critical for PFC → Striatum modulation weights

#### 2. `validate_timing()`

Validates timing parameters for biological plausibility.

```python
config = ThaliaConfig(
    global_=GlobalConfig(
        dt_ms=0.5,
        theta_frequency_hz=8.0,
    ),
    brain=BrainConfig(
        encoding_timesteps=15,
    )
)

issues = config.validate_timing()
for issue in issues:
    print(f"⚠️  {issue}")
```

**Checks**:
- `dt_ms` is reasonable (0.1-10 ms)
- `theta_frequency_hz` is in biological range (4-12 Hz)
- Encoding/delay/test timesteps are positive
- Encoding duration allows at least one theta cycle

#### 3. `validate_sparsity()`

Validates sparsity parameters across regions.

```python
config = ThaliaConfig(
    global_=GlobalConfig(default_sparsity=0.05),
    brain=BrainConfig(
        sizes=RegionSizes(cortex_size=128),
    )
)

issues = config.validate_sparsity()
if not issues:
    print("✅ Sparsity valid across all regions")
```

**Checks**:
- Default sparsity is in reasonable range (0.01-0.3)
- Active neurons > 1 for each region (cortex, hippocampus, PFC)
- Warns if active neurons < 3 (may be too few for robust coding)

#### 4. `validate_neuromodulation()`

Validates neuromodulator parameter ranges.

```python
config = ThaliaConfig(
    brain=BrainConfig(
        neuromodulation=NeuromodulationConfig(
            dopamine_baseline=0.1,
            dopamine_learning_threshold=0.01,
            use_acetylcholine=True,
            acetylcholine_encoding_level=0.8,
            acetylcholine_retrieval_level=0.2,
        )
    )
)

issues = config.validate_neuromodulation()
if not issues:
    print("✅ Neuromodulation parameters valid")
```

**Checks**:
- Dopamine baseline in range (-1.0 to 1.0)
- Dopamine learning threshold is positive and small
- Norepinephrine baseline in range (0.0 to 1.0) if enabled
- Acetylcholine levels in range (0.0 to 1.0) if enabled
- Encoding ACh level > retrieval level (proper distinction)

#### 5. `validate_training_params()`

Validates training configuration.

```python
config = ThaliaConfig(
    training=TrainingConfig(
        n_epochs=100,
        batch_size=32,
        learning_rate_scale=1.0,
        checkpoint_every_n_epochs=10,
        use_curriculum=True,
        curriculum_start_difficulty=0.3,
        curriculum_end_difficulty=0.9,
    )
)

issues = config.validate_training_params()
if not issues:
    print("✅ Training parameters valid")
```

**Checks**:
- `n_epochs` is positive and reasonable
- `batch_size` is positive
- `learning_rate_scale` is positive and not too large
- `checkpoint_every_n_epochs` won't be skipped
- Curriculum difficulty range is valid (0.0 to 1.0)
- Start difficulty < end difficulty (progressive learning)

### Automatic Validation

All validation helpers are called automatically in `validate()`:

```python
config = ThaliaConfig(
    # ... your config ...
)

# Automatic validation on creation (called in __post_init__)
# OR explicit validation:
issues = config.validate()

if issues:
    print("⚠️  Configuration Issues:")
    for issue in issues:
        print(f"   - {issue}")
else:
    print("✅ Configuration valid!")
```

### Example: Catching Common Mistakes

```python
# Mistake 1: PFC size mismatch with goal conditioning
config = ThaliaConfig(
    brain=BrainConfig(
        sizes=RegionSizes(pfc_size=32),
        use_goal_conditioning=True,
        # striatum_pfc_size defaults to 32, but let's say we change it
    )
)
# ✅ Auto-detected: "Goal conditioning enabled but size mismatch: striatum_pfc_size=64 != pfc_size=32"

# Mistake 2: Too sparse for region size
config = ThaliaConfig(
    global_=GlobalConfig(default_sparsity=0.01),
    brain=BrainConfig(sizes=RegionSizes(cortex_size=50)),
)
# ✅ Auto-detected: "cortex: only 0.5 active neurons - may be too few for robust coding"

# Mistake 3: Curriculum backwards
config = ThaliaConfig(
    training=TrainingConfig(
        use_curriculum=True,
        curriculum_start_difficulty=0.9,  # Start hard!
        curriculum_end_difficulty=0.3,    # End easy!
    )
)
# ✅ Auto-detected: "curriculum_start_difficulty (0.9) should be < end_difficulty (0.3) for progressive learning"

# Mistake 4: Encoding too short for theta cycle
config = ThaliaConfig(
    global_=GlobalConfig(dt_ms=1.0, theta_frequency_hz=8.0),
    brain=BrainConfig(encoding_timesteps=5),  # Only 5ms
)
# ✅ Auto-detected: "Encoding duration (5.0 ms) < one theta cycle (125.0 ms). Phase coding may not work properly."
```

### Integration with Training Scripts

```python
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, TrainingConfig

def create_experiment_config(device: str = "cpu") -> ThaliaConfig:
    """Create validated config for experiment."""
    config = ThaliaConfig(
        global_=GlobalConfig(device=device),
        brain=BrainConfig(use_goal_conditioning=True),
        training=TrainingConfig(n_epochs=100),
    )
    
    # Validation happens automatically, but you can check explicitly
    issues = config.validate()
    if issues:
        print("⚠️  Configuration warnings:")
        for issue in issues:
            print(f"   {issue}")
        
        # Decide whether to proceed or fix
        if any("must be" in issue for issue in issues):
            raise ValueError("Configuration has errors that must be fixed")
    
    return config

# Use in training
config = create_experiment_config(device="cuda")
brain = Brain.from_thalia_config(config)
```

### Benefits

1. **Catch mistakes early**: Before training starts, not hours later
2. **Clear error messages**: Explains what's wrong and why
3. **Biological plausibility**: Enforces neuroscience constraints
4. **Automatic**: Runs on config creation, no manual checks needed
5. **Composable**: Each validator is independent and can be called separately

### Non-Breaking Deprecation Path

If we want to rename or remove fields in the future, use this pattern:

```python
@dataclass
class BrainConfig:
    new_field: int = 10

    @property
    def old_field(self) -> int:
        import warnings
        warnings.warn(
            "old_field is deprecated, use new_field instead. "
            "Will be removed in v0.4.0",
            DeprecationWarning,
            stacklevel=2
        )
        return self.new_field
```

## Summary

The config hierarchy is now:
- ✅ **Clearer**: Obvious where each parameter belongs
- ✅ **Less duplicated**: Weight bounds, neuromodulation, training params consolidated
- ✅ **More consistent**: Regions and pathways both have base config classes
- ✅ **Backward compatible**: Additive changes only, no breaking changes

This implementation completes **Tier 2.4** from the Architecture Review 2025-12-11.
