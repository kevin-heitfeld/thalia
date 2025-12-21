# Mixins Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 21:45:51
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all mixin classes used by `NeuralRegion`. These mixins provide standard functionality to all brain regions.

Total: 4 mixins

## NeuralRegion Composition

```python
class NeuralRegion(nn.Module,
                   NeuromodulatorMixin,
                   GrowthMixin,
                   ResettableMixin,
                   DiagnosticsMixin
    ):
    # ...
```

## Mixin Classes

### `DiagnosticsMixin`

**Source**: `thalia\mixins\diagnostics_mixin.py`

**Description**: Mixin providing common diagnostic computation patterns.

**Public Methods**:

- `weight_diagnostics(weights, prefix, include_histogram)`
- `spike_diagnostics(spikes, prefix, dt_ms)`
- `trace_diagnostics(trace, prefix)`
- `learning_diagnostics(ltp, ltd, prefix)`
- `membrane_diagnostics(membrane, threshold, prefix)`
- `similarity_diagnostics(pattern_a, pattern_b, prefix, eps)`
- `collect_all_diagnostics(weights, spikes, traces)`
- `collect_standard_diagnostics(region_name, weight_matrices, spike_tensors, trace_tensors, custom_metrics)`

---

### `GrowthMixin`

**Source**: `thalia\mixins\growth_mixin.py`

**Description**: Mixin providing utility methods for region neuron growth.

---

### `NeuromodulatorMixin`

**Source**: `thalia\neuromodulation\mixin.py`

**Description**: Mixin providing standardized neuromodulator handling for brain regions.

**Public Methods**:

- `set_neuromodulators(dopamine, norepinephrine, acetylcholine)`
- `set_neuromodulator(name, level)`
- `decay_neuromodulators(dt_ms, dopamine_tau_ms, acetylcholine_tau_ms, norepinephrine_tau_ms)`
- `get_effective_learning_rate(base_lr, dopamine_sensitivity)`
- `get_neuromodulator_state()`

---

### `ResettableMixin`

**Source**: `thalia\mixins\resettable_mixin.py`

**Description**: Mixin for components with resettable state.

**Public Methods**:

- `reset_state()`

---

