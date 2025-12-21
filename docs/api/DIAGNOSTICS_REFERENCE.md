# Diagnostics Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:20:26
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all diagnostic monitor classes for system health and performance monitoring.

Total: 4 monitors

## Monitor Classes

### `CriticalityMonitor`

**Source**: `thalia\diagnostics\criticality.py`

**Description**: Monitor network criticality via branching ratio.

**Key Methods**:

- `reset_state()`
- `update()`
- `get_branching_ratio()`
- `get_weight_scaling()`
- `get_state()`

---

### `HealthMonitor`

**Source**: `thalia\diagnostics\health_monitor.py`

**Description**: Monitor network health and detect pathological states.

**Key Methods**:

- `check_health()`
- `get_trend_summary()`
- `reset_history()`

---

### `MetacognitiveMonitor`

**Source**: `thalia\diagnostics\metacognition.py`

**Description**: Stage-aware metacognitive monitoring system.

**Key Methods**:

- `estimate_confidence()`
- `should_abstain()`
- `calibrate()`
- `set_stage()`
- `get_stage()`

---

### `OscillatorHealthMonitor`

**Source**: `thalia\diagnostics\oscillator_health.py`

**Description**: Monitor oscillator health and detect pathological patterns.

**Key Methods**:

- `check_health()`
- `reset_history()`
- `get_oscillator_statistics()`
- `compute_phase_coherence()`
- `compute_region_pair_coherence()`

---

