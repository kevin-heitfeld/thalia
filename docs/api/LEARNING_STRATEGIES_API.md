# Learning Strategies API

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:47:07
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all learning strategy factory functions available in Thalia.

Total: 5 factory functions

## Factory Functions

### `create_cerebellum_strategy()`

**Returns**: `LearningStrategy`

**Source**: `thalia\learning\strategy_registry.py`

**Parameters**:

- `learning_rate`
- `error_threshold`
- `error_config`

**Description**: Create error-corrective learning for cerebellum (supervised).

---

### `create_cortex_strategy()`

**Returns**: `LearningStrategy`

**Source**: `thalia\learning\strategy_registry.py`

**Parameters**:

- `learning_rate`
- `tau_theta`
- `use_stdp`
- `use_bcm`
- `stdp_config`
- `bcm_config`

**Description**: Create composite STDP+BCM strategy for cortical learning.

---

### `create_hippocampus_strategy()`

**Returns**: `LearningStrategy`

**Source**: `thalia\learning\strategy_registry.py`

**Parameters**:

- `learning_rate`
- `one_shot`
- `a_plus`
- `tau_plus`
- `tau_minus`
- `stdp_config`

**Description**: Create hippocampus-appropriate STDP with one-shot capability.

---

### `create_strategy()`

**Returns**: `LearningStrategy`

**Source**: `thalia\learning\rules\strategies.py`

**Parameters**:

- `rule_name`

**Description**: Factory function to create learning strategies by name.

---

### `create_striatum_strategy()`

**Returns**: `LearningStrategy`

**Source**: `thalia\learning\strategy_registry.py`

**Parameters**:

- `learning_rate`
- `eligibility_tau_ms`
- `three_factor_config`

**Description**: Create three-factor learning for striatum (dopamine-modulated).

---

