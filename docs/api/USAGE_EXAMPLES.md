# Usage Examples

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:35:38
> Generated from: `scripts/generate_api_docs.py`

This document catalogs usage examples extracted from docstrings and training scripts.

Total: 13 examples

## Examples by Category

### Component

#### Example:

**Source**: `thalia\core\neural_region.py`

```python
...     "thalamus": thalamic_spikes,      # [128]
        ...     "hippocampus": hippocampal_spikes # [200]
        ... })  # Returns [500]
    Subclassing:
        Regions with internal structure (like LayeredCortex) should:
        1. Call super().__init__() to get synaptic_weights dict
        2. Define internal neurons/weights for within-region processing
        3. Override forward() to apply synaptic weights then internal processing
            class LayeredCortex(NeuralRegion):
                def __init__(self, ...):
                    super().__init__(n_neurons=l23_size + l5_size, ...)
                    self.l4_neurons = ConductanceLIF(l4_size, ...)
                    self.l23_neurons = ConductanceLIF(l23_size, ...)
                    self.w_l4_l23 = nn.Parameter(...)
                def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
                    thalamic_current = self._apply_synapses("thalamus", inputs["thalamus"])
                    l4_spikes = self.l4_neurons(thalamic_current)
                    l23_spikes = self._internal_l4_to_l23(l4_spikes)
                    return l23_spikes
```

---

#### Example: {"thalamus": [128], "hippocampus": [200]}

**Source**: `thalia\core\neural_region.py`

```python
Sources not in dict are treated as silent (no spikes)
        Returns:
            Output spikes [n_neurons]
        Raises:
            ValueError: If input source not registered with add_input_source()
```

---

#### Example:

**Source**: `thalia\core\neural_region.py`

```python
"""
        for name in subsystem_names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, 'reset_state'):
                    subsystem.reset_state()
    def get_diagnostics(self) -> Dict[str, Any]:
```

---

### Diagnostic

#### Usage:

**Source**: `thalia\diagnostics\health_monitor.py`

```python
======
    from thalia.diagnostics.health_monitor import HealthMonitor, HealthConfig
    monitor = HealthMonitor(HealthConfig())
    report = monitor.check_health(brain.get_diagnostics())
    if not report.is_healthy:
        print(f"Warning: {report.summary}")
        for issue in report.issues:
            print(f"  - {issue.description}")
            print(f"    Recommendation: {issue.recommendation}")
Author: Thalia Project
Date: December 2025
```

---

### Learning

#### Usage Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
==============
    @LearningStrategyRegistry.register("stdp")
    class STDPStrategy(LearningStrategy):
        ...
    @LearningStrategyRegistry.register("three_factor", aliases=["rl", "dopamine"])
    class ThreeFactorStrategy(LearningStrategy):
        ...
    self.learning_strategy = LearningStrategyRegistry.create(
        "three_factor",
        ThreeFactorConfig(learning_rate=0.02, dopamine_sensitivity=0.5)
    )
    available = LearningStrategyRegistry.list_strategies()
Benefits:
=========
1. **Pluggable Learning**: Easy to add new learning rules without modifying regions
2. **Discovery**: List all available strategies programmatically
3. **Plugin Support**: External packages can register custom strategies
4. **Consistency**: Same pattern as ComponentRegistry
5. **Validation**: Type checking and config validation
6. **Experimentation**: Quickly swap strategies for ablation studies
Author: Thalia Project
Date: December 11, 2025
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
@LearningStrategyRegistry.register(
                "stdp",
                config_class=STDPConfig,
                aliases=["spike_timing"],
                description="Spike-timing dependent plasticity"
            )
            class STDPStrategy(LearningStrategy):
                '''STDP learning rule.'''
                ...
            @LearningStrategyRegistry.register("three_factor", aliases=["rl"])
            class ThreeFactorStrategy(LearningStrategy):
                '''Three-factor learning with neuromodulation.'''
                ...
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
...     "rl",  # Alias for "three_factor"
            ...     ThreeFactorConfig(learning_rate=0.02)
            ... )
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
['hebbian', 'stdp', 'spike_timing', 'bcm', 'three_factor', 'rl', ...]
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
"""
        if name in cls._registry:
            del cls._registry[name]
        if name in cls._configs:
            del cls._configs[name]
        if name in cls._metadata:
            del cls._metadata[name]
        aliases_to_remove = [
            alias for alias, target in cls._aliases.items()
            if target == name
        ]
        for alias in aliases_to_remove:
            del cls._aliases[alias]
    @classmethod
    def clear(cls) -> None:
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
"""
        cls._registry.clear()
        cls._configs.clear()
        cls._aliases.clear()
        cls._metadata.clear()
def create_cortex_strategy(
    learning_rate: float = 0.001,
    tau_theta: float = 5000.0,
    use_stdp: bool = True,
    use_bcm: bool = True,
    stdp_config: Optional[Any] = None,
    bcm_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
"""
    from thalia.learning.rules.strategies import (
        STDPConfig, STDPStrategy,
        BCMConfig, BCMStrategy,
        CompositeStrategy,
    )
    if use_stdp and use_bcm:
        stdp = STDPStrategy(
            stdp_config or STDPConfig(learning_rate=learning_rate, **kwargs)
        )
        bcm = BCMStrategy(
            bcm_config or BCMConfig(
                learning_rate=learning_rate,
                tau_theta=tau_theta,
                **kwargs
            )
        )
        return CompositeStrategy([stdp, bcm])
    elif use_stdp:
        return STDPStrategy(
            stdp_config or STDPConfig(learning_rate=learning_rate, **kwargs)
        )
    elif use_bcm:
        return BCMStrategy(
            bcm_config or BCMConfig(
                learning_rate=learning_rate,
                tau_theta=tau_theta,
                **kwargs
            )
        )
    else:
        raise ConfigurationError("Must enable at least one learning rule (STDP or BCM)")
def create_hippocampus_strategy(
    learning_rate: float = 0.01,
    one_shot: bool = False,
    a_plus: Optional[float] = None,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    stdp_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
"""
    from thalia.learning.rules.strategies import STDPConfig, STDPStrategy
    if stdp_config is not None:
        return STDPStrategy(stdp_config)
    if one_shot:
        learning_rate = 0.1 if learning_rate == 0.01 else learning_rate
        a_plus = 0.1 if a_plus is None else a_plus
    else:
        a_plus = 0.01 if a_plus is None else a_plus
    return STDPStrategy(
        STDPConfig(
            learning_rate=learning_rate,
            a_plus=a_plus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            **kwargs
        )
    )
def create_striatum_strategy(
    learning_rate: float = 0.001,
    eligibility_tau_ms: float = 1000.0,
    three_factor_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
```

---

#### Example:

**Source**: `thalia\learning\strategy_registry.py`

```python
"""
    from thalia.learning.rules.strategies import ThreeFactorConfig, ThreeFactorStrategy
    if three_factor_config is not None:
        return ThreeFactorStrategy(three_factor_config)
    return ThreeFactorStrategy(
        ThreeFactorConfig(
            learning_rate=learning_rate,
            eligibility_tau=eligibility_tau_ms,
            **kwargs
        )
    )
def create_cerebellum_strategy(
    learning_rate: float = 0.005,
    error_threshold: float = 0.01,
    error_config: Optional[Any] = None,
    **kwargs: Any,
) -> LearningStrategy:
```

---

