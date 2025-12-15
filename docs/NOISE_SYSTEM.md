# Noise System Documentation

## Overview

Thalia implements comprehensive noise scheduling across developmental stages to improve generalization, robustness, and biological realism. The noise system was added December 15, 2025.

## Architecture

### Noise Types

1. **Membrane Noise**: Stochastic fluctuations in neuron membrane potentials
   - Models ion channel noise and thermal fluctuations
   - Applied at every timestep during integration
   - Region-specific (e.g., hippocampus more variable than cerebellum)

2. **Weight Noise**: Exploration noise during synaptic plasticity
   - Enables exploration of weight space
   - Stage-dependent (off early, on later)
   - Helps escape local minima

3. **Oscillator Phase Noise**: Drift in neural oscillator phases
   - Models biological oscillator variability (~0.05 rad or ~3 degrees)
   - More impactful than spike timing jitter at 1ms resolution
   - Affects population synchrony and phase coding robustness
   - Increases across developmental stages (0.03→0.07 rad)

4. **Spike Timing Jitter**: NOT IMPLEMENTED
   - Sub-millisecond timing variability (0.1-1ms)
   - Cannot be represented at 1ms timestep resolution
   - Membrane noise already provides natural spike timing variability
   - Oscillator phase noise is more effective for robustness

5. **Working Memory Noise**: Stochastic drift in PFC maintenance
   - Models PFC variability during maintenance
   - Constant 0.02 std across stages

6. **Proprioceptive Noise**: Sensor noise in sensorimotor feedback
   - 10% noise (biologically realistic)
   - Applied during sensorimotor tasks

7. **REM Consolidation Noise**: High noise during schema extraction
   - 0.3-0.4 std during REM sleep replay
   - Creates variations for generalization

## Curriculum-Based Scheduling

### Stage Profiles

| Stage | Membrane | Weight | Osc Phase | Augmentation | Rationale |
|-------|----------|--------|-----------|--------------|-----------|
| -0.5: Sensorimotor | 0.01 | OFF | OFF | 5% | Stable foundation |
| 0: Phonology | 0.01 | 0.02 | 0.03 rad | 5% | Critical period foundations |
| 1: Toddler | 0.015 | 0.03 | 0.04 rad | 10% | Developing robustness |
| 2: Grammar | 0.02 | 0.04 | 0.05 rad | 15% | Generalization |
| 3: Reading | 0.025 | 0.05 | 0.06 rad | 15% | Moderate-high |
| 4+: Abstract | 0.03 | 0.05 | 0.07 rad | 20% | Robustness & exploration |### Adaptive Modulation

The noise scheduler adapts based on:

1. **Criticality State**:
   - Subcritical (< 0.95): Boost noise 1.5× for exploration
   - Supercritical (> 1.15): Reduce noise 0.5× for stability

2. **Performance**:
   - Low (< 60%): Reduce noise 0.7× to stabilize learning
   - High (> 85%): Increase noise 1.2× to push generalization

3. **Region Type**:
   - Hippocampus: 1.2× baseline (more variable)
   - Cerebellum: 0.7× baseline (precise timing)
   - PFC: 0.9× baseline (stability for WM)

## Implementation

### Key Files

- `src/thalia/training/curriculum/noise_scheduler.py` - Main scheduler
- `src/thalia/components/neurons/neuron.py` - Membrane noise application
- `src/thalia/regions/prefrontal.py` - WM noise
- `src/thalia/tasks/stimulus_utils.py` - Proprioceptive noise utilities
- `src/thalia/memory/consolidation/advanced_consolidation.py` - REM noise

### Usage

```python
from thalia.training.curriculum import NoiseScheduler, NoiseSchedulerConfig

# Initialize scheduler
config = NoiseSchedulerConfig(
    enabled=True,
    enable_criticality_adaptation=True,
    enable_performance_adaptation=True,
    verbose=True,
)
scheduler = NoiseScheduler(config)

# In training loop
scheduler.set_stage(CurriculumStage.TODDLER)
profile = scheduler.get_current_profile()

# Apply to brain
for region_name, region in brain.regions.items():
    noise_std = scheduler.get_membrane_noise_for_region(region_name)
    region.neurons.config.noise_std = noise_std

# Update based on performance
scheduler.update(
    current_stage=stage,
    performance=0.75,  # 75% accuracy
    criticality=1.05,   # Slightly supercritical
)
```

### Integration with CurriculumTrainer

The `CurriculumTrainer` automatically:
1. Initializes `NoiseScheduler` in `__init__`
2. Updates noise profile when stage changes
3. Adapts noise every 1000 steps based on metrics
4. Re-applies noise profile every 5000 steps

## Biological Motivation

### Why Noise?

1. **Real neurons are noisy**: Ion channel stochasticity, thermal fluctuations
2. **Aids exploration**: Prevents getting stuck in local minima
3. **Improves generalization**: Regularization effect similar to dropout
4. **Critical for learning**: Noise-driven exploration enables discovery
5. **Developmental trajectory**: Young brains are noisier, mature brains stabilize

### References

- Faisal et al. (2008): "Noise in the nervous system" - Nature Reviews Neuroscience
- Stein et al. (2005): "Neuronal variability: noise or part of the signal?"
- McDonnell & Ward (2011): "The benefits of noise in neural systems"
- Thalia curriculum strategy: `docs/design/curriculum_strategy.md`

## Default Configuration

By default:
- **Membrane noise**: Enabled (0.01 std) - Changed December 15, 2025
- **Weight noise**: Stage-dependent (OFF early, ON from Stage 0+)
- **Adaptation**: Enabled for both criticality and performance
- **Region variation**: Applied (hippocampus 1.2×, cerebellum 0.7×)

## Testing

To verify noise is working:

```python
# Check membrane noise
assert brain.cortex.impl.neurons.config.noise_std > 0

# Check scheduler
trainer = CurriculumTrainer(brain, ...)
profile = trainer.noise_scheduler.get_current_profile()
print(f"Membrane noise: {profile.membrane_noise_std}")
print(f"Weight noise: {profile.enable_weight_noise}")

# Check adaptation
trainer.noise_scheduler.update(
    CurriculumStage.TODDLER,
    performance=0.5,  # Low
    criticality=0.9,  # Subcritical
)
# Should boost noise due to low perf + subcritical state
```

## Future Enhancements

Potential improvements (not yet critical):

1. **Data augmentation pipeline**: Systematic vision/audio/text augmentation
2. **Annealing schedules**: Gradual noise reduction within stages
3. **Task-specific noise**: Different noise for different task types
4. **Noise-aware consolidation**: Adjust replay noise based on schema complexity
5. **Finer timestep resolution**: Use 0.1ms timesteps to enable sub-millisecond precision

## Changelog

**December 15, 2025**:
- ✅ Added `NoiseScheduler` with curriculum-based profiles
- ✅ Enabled membrane noise by default (0.01 std)
- ✅ Integrated with `CurriculumTrainer`
- ✅ Added adaptive modulation (criticality + performance)
- ✅ Implemented region-specific noise scaling
- ✅ Weight noise support (stage-dependent)
- ✅ **Oscillator phase noise** (0.03-0.07 rad across stages)
- ✅ Documentation and testing guidelines
- ℹ️  Spike timing jitter deemed not useful at 1ms resolution
