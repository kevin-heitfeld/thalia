# Diagnostics Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-14 17:31:39
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all diagnostic monitor classes for system health and performance monitoring.

Total: **4** monitors

![Monitors](https://img.shields.io/badge/Monitors-4-blue) ![Diagnostics](https://img.shields.io/badge/Type-Diagnostics-yellow) ![Real--time](https://img.shields.io/badge/Mode-Real--time-green)

## 📊 Monitoring Workflow

```mermaid
graph LR
    A[Brain Training] --> B[HealthMonitor]
    A --> C[CriticalityMonitor]
    A --> D[MetacognitiveMonitor]
    A --> E[TrainingMonitor]
    B --> F[Health Reports]
    C --> F
    D --> F
    E --> G[Training Metrics]
```

## 🔍 Monitor Classes

### [``CriticalityMonitor``](../../src/thalia/diagnostics/criticality.py#L112)

**Source**: [`thalia/diagnostics/criticality.py`](../../src/thalia/diagnostics/criticality.py)

**Description**: Monitor network criticality via branching ratio.

**Key Methods**:

- [`reset_state()`](../../src/thalia/diagnostics/criticality.py#L142)
- [`update(spikes)`](../../src/thalia/diagnostics/criticality.py#L149)
- [`get_branching_ratio()`](../../src/thalia/diagnostics/criticality.py#L250)
- [`get_weight_scaling()`](../../src/thalia/diagnostics/criticality.py#L254)
- [`get_state()`](../../src/thalia/diagnostics/criticality.py#L264)

---

### [``HealthMonitor``](../../src/thalia/diagnostics/health_monitor.py#L152)

**Source**: [`thalia/diagnostics/health_monitor.py`](../../src/thalia/diagnostics/health_monitor.py)

**Description**: Monitor network health and detect pathological states.

**Key Methods**:

- [`check_health(diagnostics)`](../../src/thalia/diagnostics/health_monitor.py#L190)
- [`get_trend_summary()`](../../src/thalia/diagnostics/health_monitor.py#L406)
- [`reset_history()`](../../src/thalia/diagnostics/health_monitor.py#L433)

---

### [``MetacognitiveMonitor``](../../src/thalia/diagnostics/metacognition.py#L219)

**Source**: [`thalia/diagnostics/metacognition.py`](../../src/thalia/diagnostics/metacognition.py)

**Description**: Stage-aware metacognitive monitoring system.

**Key Methods**:

- [`estimate_confidence(population_activity)`](../../src/thalia/diagnostics/metacognition.py#L250)
- [`should_abstain(confidence)`](../../src/thalia/diagnostics/metacognition.py#L320)
- [`calibrate(population_activity, actual_correct, dopamine)`](../../src/thalia/diagnostics/metacognition.py#L348)
- [`set_stage(stage)`](../../src/thalia/diagnostics/metacognition.py#L385)
- [`get_stage()`](../../src/thalia/diagnostics/metacognition.py#L389)

---

### [``OscillatorHealthMonitor``](../../src/thalia/diagnostics/oscillator_health.py#L164)

**Source**: [`thalia/diagnostics/oscillator_health.py`](../../src/thalia/diagnostics/oscillator_health.py)

**Description**: Monitor oscillator health and detect pathological patterns.

**Key Methods**:

- [`check_health(phases, frequencies, amplitudes, signals, couplings)`](../../src/thalia/diagnostics/oscillator_health.py#L194)
- [`reset_history()`](../../src/thalia/diagnostics/oscillator_health.py#L441)
- [`get_oscillator_statistics(oscillator)`](../../src/thalia/diagnostics/oscillator_health.py#L448)
- [`compute_phase_coherence(region1_phases, region2_phases, oscillator)`](../../src/thalia/diagnostics/oscillator_health.py#L484)
- [`compute_region_pair_coherence(region_phases, region_pairs, oscillators)`](../../src/thalia/diagnostics/oscillator_health.py#L536)

---

## 💡 Monitoring Best Practices

### When to Use Each Monitor

- **HealthMonitor**: Every training run (catches pathological states)
- **CriticalityMonitor**: When tuning network connectivity
- **MetacognitiveMonitor**: For confidence estimation and active learning
- **TrainingMonitor**: For visualization and metric tracking

### Interpreting Results

✅ **Healthy network**: Firing rates 0.01-0.1, weights stable, no NaN

⚠️ **Warning signs**: Extreme firing rates, rapid weight changes

❌ **Critical issues**: NaN values, zero activity, runaway excitation

### Performance Tips

- Check health every 10-100 steps (not every step)
- Store history for trend analysis
- Use thresholds to trigger adaptive responses
- Log detailed diagnostics only when issues detected

