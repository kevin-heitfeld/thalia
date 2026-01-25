# Usage Examples

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-25 23:23:15
> Generated from: `scripts/generate_api_docs.py`

This document catalogs usage examples extracted from docstrings and training scripts.

Total: 4 examples

## Examples by Category

### Component

#### Example:

**Source**: `thalia\core\neural_region.py`

```python
self.clear_port_outputs()
                # process layers ...
                self.set_port_output("l6a", l6a_spikes)
                return self.get_port_output("default")
```

---

### Learning

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
stdp = LearningStrategyRegistry.create(
                "stdp",
                STDPConfig(learning_rate=0.02, a_plus=0.01)
            )
```

---

#### Example from strategy_registry.py

**Source**: `thalia\learning\strategy_registry.py`

```python
rl_strategy = LearningStrategyRegistry.create(
                "rl",  # Alias for "three_factor"
                ThreeFactorConfig(learning_rate=0.02)
            )
```

---

#### Example from strategy_registry.py

**Source**: `thalia\learning\strategy_registry.py`

```python
stdp_cfg = STDPConfig(learning_rate=0.002, a_plus=0.02)
        bcm_cfg = BCMConfig(tau_theta=10000.0)
        strategy = create_cortex_strategy(stdp_config=stdp_cfg, bcm_config=bcm_cfg)
```

---

