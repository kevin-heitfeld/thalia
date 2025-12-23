# Usage Examples

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-23 15:50:40
> Generated from: `scripts/generate_api_docs.py`

This document catalogs usage examples extracted from docstrings and training scripts.

Total: 4 examples

## Examples by Category

### Component

#### Example:

**Source**: `thalia\core\neural_region.py`

```python
region = NeuralRegion(
            n_neurons=500,
            neuron_config=ConductanceLIFConfig(),
            default_learning_rule="stdp"
        )
        >>>
        # Add input sources with their synaptic weights
        region.add_input_source("thalamus", n_input=128)  # Uses default STDP
        region.add_input_source("hippocampus", n_input=200, learning_rule="bcm")  # Override
        >>>
        # Forward pass with multi-source input
        outputs = region.forward({
            "thalamus": thalamic_spikes,      # [128]
            "hippocampus": hippocampal_spikes # [200]
        })  # Returns [500]
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

