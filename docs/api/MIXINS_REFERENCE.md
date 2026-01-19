# Mixins Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-19 05:37:19
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

### [``DiagnosticsMixin``](../../src/thalia/mixins/diagnostics_mixin.py#L40)

**Source**: [`thalia/mixins/diagnostics_mixin.py`](../../src/thalia/mixins/diagnostics_mixin.py)

**Description**: Mixin providing common diagnostic computation patterns.

**Public Methods**:

- [`weight_diagnostics(weights, prefix, include_histogram)`](../../src/thalia/mixins/diagnostics_mixin.py#L51)
- [`spike_diagnostics(spikes, prefix, dt_ms)`](../../src/thalia/mixins/diagnostics_mixin.py#L102)
- [`trace_diagnostics(trace, prefix)`](../../src/thalia/mixins/diagnostics_mixin.py#L140)
- [`learning_diagnostics(ltp, ltd, prefix)`](../../src/thalia/mixins/diagnostics_mixin.py#L165)
- [`membrane_diagnostics(membrane, threshold, prefix)`](../../src/thalia/mixins/diagnostics_mixin.py#L192)
- [`similarity_diagnostics(pattern_a, pattern_b, prefix, eps)`](../../src/thalia/mixins/diagnostics_mixin.py#L220)
- [`collect_all_diagnostics(weights, spikes, traces)`](../../src/thalia/mixins/diagnostics_mixin.py#L262)
- [`collect_standard_diagnostics(region_name, weight_matrices, spike_tensors, trace_tensors, custom_metrics)`](../../src/thalia/mixins/diagnostics_mixin.py#L296)

---

### [``GrowthMixin``](../../src/thalia/mixins/growth_mixin.py#L51)

**Source**: [`thalia/mixins/growth_mixin.py`](../../src/thalia/mixins/growth_mixin.py)

**Description**: Mixin providing utility methods for region neuron growth.

**Public Methods**:

- [`named_modules()`](../../src/thalia/mixins/growth_mixin.py#L113)

---

### [``NeuromodulatorMixin``](../../src/thalia/neuromodulation/mixin.py#L212)

**Source**: [`thalia/neuromodulation/mixin.py`](../../src/thalia/neuromodulation/mixin.py)

**Description**: Mixin providing standardized neuromodulator handling for brain regions.

**Public Methods**:

- [`set_neuromodulators(dopamine, norepinephrine, acetylcholine)`](../../src/thalia/neuromodulation/mixin.py#L232)
- [`set_neuromodulator(name, level)`](../../src/thalia/neuromodulation/mixin.py#L281)
- [`decay_neuromodulators(dt_ms, dopamine_tau_ms, acetylcholine_tau_ms, norepinephrine_tau_ms)`](../../src/thalia/neuromodulation/mixin.py#L303)
- [`get_effective_learning_rate(base_lr, dopamine_sensitivity)`](../../src/thalia/neuromodulation/mixin.py#L335)
- [`get_neuromodulator_state()`](../../src/thalia/neuromodulation/mixin.py#L368)

---

### [``ResettableMixin``](../../src/thalia/mixins/resettable_mixin.py#L16)

**Source**: [`thalia/mixins/resettable_mixin.py`](../../src/thalia/mixins/resettable_mixin.py)

**Description**: Mixin for components with resettable state.

**Public Methods**:

- [`reset_state()`](../../src/thalia/mixins/resettable_mixin.py#L33)
- [`reset_standard_state(state_attrs)`](../../src/thalia/mixins/resettable_mixin.py#L46)

---

