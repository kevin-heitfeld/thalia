# Configuration Patterns

## Overview

This document defines patterns for organizing and validating configuration classes in Thalia.

> **📚 For complete configuration class documentation with all fields and defaults, see [CONFIGURATION_REFERENCE.md](../api/CONFIGURATION_REFERENCE.md)**

This guide focuses on **patterns** for organizing and validating configuration.

**Last Updated**: December 12, 2025

---

## Table of Contents

1. [Configuration Organization](#configuration-organization)
2. [Declarative Validation Pattern](#declarative-validation-pattern)
3. [When to Extract Config to Separate File](#when-to-extract-config-to-separate-file)
4. [Validation Best Practices](#validation-best-practices)

---

## Configuration Organization

### Config Inheritance Pattern Standard

**Rule**: When using multiple inheritance with learning configs, ALWAYS use this order:

```python
class RegionConfig(NeuralComponentConfig, LearningConfig):
    """Standard pattern: structural config first, then behavioral config."""
    pass
```

**Rationale**:
- `NeuralComponentConfig` contains structural parameters (n_neurons, device, etc.)
- Learning configs contain behavioral parameters (learning_rate, tau, etc.)
- **Structural should take precedence in Python's Method Resolution Order (MRO)**
- Consistent order prevents subtle MRO bugs

**Examples**:

✅ **Correct** (NeuralComponentConfig first):
```python
class StriatumConfig(NeuralComponentConfig, ModulatedLearningConfig):
    pass

class HippocampusConfig(NeuralComponentConfig, STDPLearningConfig):
    pass

class CerebellumConfig(NeuralComponentConfig, ErrorCorrectiveLearningConfig):
    pass
```

❌ **Incorrect** (reversed order):
```python
class BadConfig(STDPLearningConfig, NeuralComponentConfig):  # WRONG ORDER
    pass
```

**Single Inheritance** (no learning config): Fine to use directly
```python
class LayeredCortexConfig(NeuralComponentConfig):  # No learning config mixin
    pass
```

---

### Extract to `region/config.py` when:

✅ **Config > 50 lines**
- Long configs make main region file hard to navigate
- Extract improves discoverability

✅ **Multiple related configs**
- Region config + component configs (learning, memory, etc.)
- Better to group in one file

✅ **Config reused in multiple files**
- Avoids circular imports
- Central source of truth

✅ **Region is "major"**
- Cortex, hippocampus, striatum, cerebellum, prefrontal
- Consistency: all major regions follow same pattern

### Keep inline when:

⚪ **Config < 50 lines**
- Small, simple configs don't need extraction
- Easier to see region and config together

⚪ **Single config for simple region**
- Example: Thalamus relay (single-purpose region)
- No related component configs

⚪ **Region is auxiliary/supporting**
- Feedforward inhibition, simple transformations
- Not a major computational region

### Always centralize in `thalia.config.*`:

🌍 **Brain settings**
- Device, dtype, seed (in `BrainConfig`)
- Cross-region parameters (theta frequency, dt_ms)

🌍 **Cross-region coordination**
- `BrainConfig`, `TrainingConfig`
- Region size specifications

🌍 **Shared base classes**
- `BaseConfig`, `NeuralComponentConfig`, `PathwayConfig`
- Common patterns for all configs

---

## Declarative Validation Pattern

### Basic Usage

Use `ValidatedConfig` mixin for declarative validation rules:

```python
from dataclasses import dataclass
from thalia.config import NeuralComponentConfig, ValidatedConfig

@dataclass
class MyRegionConfig(NeuralComponentConfig, ValidatedConfig):
    learning_rate: float = 0.001
    n_neurons: int = 100
    dopamine_threshold: float = 0.5
    sparsity: float = 0.05

    # Validation rules (applied in __post_init__)
    _validation_rules = {
        'learning_rate': ('positive', 'finite'),
        'n_neurons': ('positive_integer', 'range(1, 10000)'),
        'dopamine_threshold': ('range(0.0, 1.0)',),
        'sparsity': ('probability',),
    }

    def __post_init__(self):
        """Validate after initialization."""
        super().__post_init__()  # Call parent validation
        self.validate_config()  # Apply declarative rules
```

### Available Validators

| Validator | Description | Example |
|-----------|-------------|---------|
| `'positive'` | Value > 0 | `learning_rate: ('positive',)` |
| `'non_negative'` | Value >= 0 | `bias: ('non_negative',)` |
| `'finite'` | Not inf or nan | `weight: ('finite',)` |
| `'positive_integer'` | Integer > 0 | `n_neurons: ('positive_integer',)` |
| `'probability'` | Value in [0, 1] | `dropout: ('probability',)` |
| `'range(min, max)'` | Value in [min, max] | `'range(0.0, 1.0)'` |
| `'non_empty_string'` | Non-empty string | `name: ('non_empty_string',)` |

### Multiple Rules

Apply multiple validators to one field:

```python
_validation_rules = {
    'learning_rate': ('positive', 'finite', 'range(1e-6, 1.0)'),
    'n_output': ('positive_integer', 'range(1, 10000)'),
}
```

Rules are checked in order. All must pass.

### Custom Validation

For complex validation logic, override `__post_init__`:

```python
@dataclass
class StriatumConfig(NeuralComponentConfig, ValidatedConfig):
    n_d1: int = 50
    n_d2: int = 50
    n_output: int = 4

    _validation_rules = {
        'n_d1': ('positive_integer',),
        'n_d2': ('positive_integer',),
        'n_output': ('positive_integer',),
    }

    def __post_init__(self):
        super().__post_init__()
        self.validate_config()

        # Custom cross-field validation
        if self.n_d1 + self.n_d2 > 10000:
            raise ConfigValidationError(
                f"Total D1+D2 neurons ({self.n_d1 + self.n_d2}) exceeds "
                f"reasonable limit (10000)"
            )
```

---

## When to Extract Config to Separate File

### Decision Tree

```
Is config > 50 lines?
├─ Yes → Extract to config.py
└─ No
   └─ Are there multiple related configs?
      ├─ Yes → Extract to config.py
      └─ No
         └─ Is region a "major" region?
            ├─ Yes (cortex, hippocampus, striatum, etc.)
            │  └─ Extract for consistency
            └─ No
               └─ Keep inline (simple/auxiliary region)
```

### Examples

**Extract** (major region with multiple configs):
```
src/thalia/regions/striatum/
├── config.py              # ← Extract here
│   ├── StriatumConfig
│   ├── StriatumLearningConfig
│   └── StriatumHomeostasisConfig
├── striatum.py
├── d1_pathway.py
└── d2_pathway.py
```

**Keep Inline** (simple auxiliary region):
```python
# src/thalia/regions/thalamus.py

@dataclass
class ThalamicRelayConfig(NeuralComponentConfig):
    """Config for thalamic relay (simple, ~30 lines)."""
    n_input: int = 64
    n_output: int = 64
    relay_gain: float = 1.0

class ThalamicRelay(NeuralComponent):
    def __init__(self, config: ThalamicRelayConfig):
        ...
```

---

## Validation Best Practices

### 1. Validate Early

Catch errors at config creation, not 30 minutes into training:

```python
# ✅ Good: Fails immediately
config = MyConfig(learning_rate=-0.1)  # ConfigValidationError

# ❌ Bad: Fails during training
brain = Brain(config)
brain.train()  # Error: negative learning rate!
```

### 2. Provide Clear Error Messages

```python
# ✅ Good
raise ConfigValidationError(
    f"learning_rate={self.learning_rate} must be positive. "
    f"Typical values: 0.001-0.1"
)

# ❌ Bad
raise ValueError("Invalid learning rate")
```

### 3. Document Validation Rules

```python
@dataclass
class MyConfig(NeuralComponentConfig, ValidatedConfig):
    """Configuration for MyRegion.

    Attributes:
        learning_rate: Base learning rate (must be positive, typical: 0.001-0.1)
        n_neurons: Number of neurons (must be 1-10000)
        sparsity: Target sparsity (must be probability in [0, 1])
    """
    learning_rate: float = 0.001
    n_neurons: int = 100
    sparsity: float = 0.05

    _validation_rules = {
        'learning_rate': ('positive', 'finite'),
        'n_neurons': ('positive_integer', 'range(1, 10000)'),
        'sparsity': ('probability',),
    }
```

### 4. Test Config Validation

```python
def test_config_validation():
    """Test that invalid configs are rejected."""
    # Should reject negative learning rate
    with pytest.raises(ConfigValidationError):
        MyConfig(learning_rate=-0.1)

    # Should reject too many neurons
    with pytest.raises(ConfigValidationError):
        MyConfig(n_neurons=20000)

    # Should accept valid config
    config = MyConfig(learning_rate=0.01, n_neurons=100)
    assert config.learning_rate == 0.01
```

### 5. Biological Plausibility Checks

For biological parameters, validate against known ranges:

```python
_validation_rules = {
    'tau_mem': ('range(5.0, 50.0)',),  # Membrane time constant (ms)
    'v_threshold': ('range(0.5, 2.0)',),  # Threshold (normalized)
    'tau_syn': ('range(1.0, 20.0)',),  # Synaptic time constant (ms)
}

def __post_init__(self):
    super().__post_init__()
    self.validate_config()

    # Warn if outside typical biological range
    if self.tau_mem > 30:
        import warnings
        warnings.warn(
            f"tau_mem={self.tau_mem}ms is longer than typical cortical "
            f"neurons (10-30ms). Still valid but unusual."
        )
```

---

## Migration Guide

### Converting Existing Configs

**Before** (manual validation):
```python
@dataclass
class OldConfig(NeuralComponentConfig):
    learning_rate: float = 0.001
    n_output: int = 100

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.n_output, int):
            raise TypeError("n_output must be integer")
        if self.n_output <= 0 or self.n_output > 10000:
            raise ValueError("n_output must be in range [1, 10000]")
```

**After** (declarative):
```python
@dataclass
class NewConfig(NeuralComponentConfig, ValidatedConfig):
    learning_rate: float = 0.001
    n_output: int = 100

    _validation_rules = {
        'learning_rate': ('positive', 'finite'),
        'n_output': ('positive_integer', 'range(1, 10000)'),
    }

    def __post_init__(self):
        super().__post_init__()
        self.validate_config()
```

**Benefits**:
- **6 lines → 2 lines** of validation code
- Consistent error messages
- Easier to test
- Declarative and readable

---

## See Also

- `src/thalia/config/validation.py` - Validation implementation
- `src/thalia/config/base.py` - Base config classes
- `docs/patterns/component-parity.md` - Component design patterns
