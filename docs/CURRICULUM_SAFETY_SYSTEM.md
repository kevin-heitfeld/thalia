# Curriculum Safety System

**Status**: Implementation Complete
**Priority**: CRITICAL - Use before any Stage 1+ training
**Consensus Design**: Expert Review + ChatGPT Engineering Analysis

## Overview

The Curriculum Safety System implements the critical design principle:

> **"You cannot have a stage where a single failure cascades and destroys the entire system."**

This system protects Thalia's training through three integrated components:

1. **Stage Gates** - Hard criteria that must be met before advancing stages
2. **Continuous Monitoring** - Real-time health checks and anomaly detection
3. **Graceful Degradation** - Module failure handling without system collapse

## Critical: Stage 1 is Highest Risk

Both expert review and engineering analysis agree: **Stage 1 (Working Memory & Object Permanence) is the critical risk node** in the curriculum. The safety system provides:

- **Stricter monitoring** (checks every 500 steps vs 1000 for other stages)
- **Tighter thresholds** (more conservative failure detection)
- **Mandatory gate** (cannot proceed to Stage 2 without passing ALL criteria)

## Quick Start

```python
from thalia.training.curriculum import CurriculumSafetySystem

# Initialize safety system
safety = CurriculumSafetySystem(
    brain=brain,
    current_stage=1,  # Stage 1 is highest risk
    enable_auto_intervention=True,
    checkpoint_callback=save_checkpoint
)

# Training loop
for step in range(training_steps):
    # Normal training step
    result = train_step(brain, batch)

    # Update safety monitoring
    intervention = safety.update(brain, step, result)

    # Handle intervention if triggered
    if intervention:
        actions = safety.handle_intervention(intervention, brain)
        execute_actions(actions)

    # Periodic status check
    if step % 5000 == 0:
        status = safety.get_status()
        logger.info(f"Health score: {status.health_score:.2f}")
        logger.info(f"Active alerts: {status.active_alerts}")

# Before advancing to next stage
can_advance, gate_result = safety.can_advance_stage()

if can_advance:
    safety.advance_to_next_stage()
    # Continue training in Stage 2
else:
    logger.warning(f"Gate failures: {gate_result.failures}")
    logger.warning(f"Recommendations: {gate_result.recommendations}")
    # Extend Stage 1 training
```

## Stage 1 Survival Checklist

**ALL criteria must be met before advancing to Stage 2. NO EXCEPTIONS.**

### Oscillator Stability
- ✅ Theta frequency: 6.5-8.5 Hz
- ✅ Theta variance: <15%
- ✅ Gamma-theta phase locking: >0.4
- ✅ Frequency drift: <5% over 10k steps

### Working Memory
- ✅ 2-back accuracy: ≥80% (rolling window)
- ✅ No decay across 3 consolidation cycles
- ✅ ≥3 stable attractors reliably retrievable

### Cross-Modal Interference
- ✅ Phonology ↔ Vision drift below threshold
- ✅ No simultaneous collapse of both modalities
- ✅ Temporal decoupling enabled if needed

### Global Health
- ✅ All regions firing in range (0.05-0.15 Hz)
- ✅ No chronic dopamine saturation
- ✅ Replay produces measurable improvement

**If ANY criterion fails:**
- Extend Stage 1 (add 1-2 weeks)
- Reduce task difficulty
- Force consolidation
- **NEVER proceed to Stage 2**

## Kill-Switch Map (Graceful Degradation)

Not all failures are equal. The system handles failures based on criticality:

| Module Failure | Severity | System Response |
|---|---|---|
| **Language** | ✅ DEGRADABLE | Continue thinking, planning, navigation |
| **Grammar** | ✅ DEGRADABLE | Communication degrades, semantics intact |
| **Reading** | ✅ DEGRADABLE | Semantic understanding maintained |
| **Vision** | ⚠️ LIMITED | Navigation restricted, compensate with other modalities |
| **Phonology** | ⚠️ LIMITED | Language slowed but not blocked |
| **Working Memory** | ❌ CRITICAL | **EMERGENCY STOP** - Rollback to checkpoint |
| **Oscillators** | ❌ CRITICAL | **EMERGENCY STOP** - System must halt |
| **Replay** | ❌ CRITICAL | **STOP LEARNING** - Debugging mode |

### Critical Systems (Cannot Fail)

If these drop >30%, system triggers **EMERGENCY STOP**:
- Working memory
- Oscillators (theta/gamma)
- Replay consolidation

**Response**: Freeze learning, rollback to last checkpoint, investigate

### Degradable Systems (Can Fail Gracefully)

If these drop >70%, system **continues without them**:
- Language generation
- Grammar production
- Reading comprehension

**Response**: Disable module temporarily, continue with remaining capabilities

### Limited Degradation Systems

If these drop >50%, system enters **partial shutdown**:
- Vision processing
- Phonology processing

**Response**: Reduce load on module, enable fallback mechanisms

## Intervention Types

The system can trigger five types of interventions:

### 1. REDUCE_LOAD
**When**: Cognitive overload detected (firing rates >0.20, dopamine saturated)

**Actions**:
- Reduce task complexity temporarily
- Lower learning rates
- Simplify stimuli

**Example**:
```python
if intervention == InterventionType.REDUCE_LOAD:
    brain.learning_rate *= 0.5  # Halve learning rate
    task_complexity = max(0.5, task_complexity - 0.1)  # Reduce difficulty
```

### 2. CONSOLIDATE
**When**: Performance degradation or region silence detected

**Actions**:
- Trigger emergency consolidation
- 10k replay steps
- Prioritize high-error experiences

**Example**:
```python
if intervention == InterventionType.CONSOLIDATE:
    offline_consolidation(brain, replay_buffer, n_steps=10000)
```

### 3. TEMPORAL_SEPARATION
**When**: Cross-modal interference detected (phonology + vision both failing)

**Actions**:
- Stop training both modalities simultaneously
- Alternate: phonology-only periods, then vision-only periods
- Prevents BCM threshold drift

**Example**:
```python
if intervention == InterventionType.TEMPORAL_SEPARATION:
    # Train phonology for 1000 steps
    train_modality(brain, 'phonology', steps=1000)
    # Then train vision for 1000 steps
    train_modality(brain, 'vision', steps=1000)
    # Repeat until interference resolves
```

### 4. EMERGENCY_STOP
**When**: Critical system failure (WM collapse, oscillator instability)

**Actions**:
- Freeze all learning
- Rollback to last checkpoint
- Enter debugging mode
- Requires manual investigation

**Example**:
```python
if intervention == InterventionType.EMERGENCY_STOP:
    brain.freeze_learning()
    load_checkpoint('last_stable.pt')
    logger.error("Manual intervention required")
    sys.exit(1)
```

### 5. ROLLBACK
**When**: Multiple criteria failing, system unstable

**Actions**:
- Load previous checkpoint
- Retry with adjusted hyperparameters
- More conservative training

## Monitoring Metrics

The system tracks comprehensive metrics for gate evaluation:

### Oscillator Metrics
- `theta_frequency`: Current theta frequency (Hz)
- `theta_variance`: Variance in theta frequency
- `gamma_theta_phase_locking`: Phase-locking value (0-1)

### Working Memory Metrics
- `n_back_accuracy`: Rolling window accuracy on 2-back task
- `wm_capacity`: Number of items maintainable
- `attractor_stability`: Stability of WM attractors

### Performance Metrics
- `task_accuracy`: Overall task performance
- `modality_performances`: Per-modality accuracy (dict)

### Health Metrics
- `mean_firing_rate`: Average across all regions
- `region_firing_rates`: Per-region firing rates (dict)
- `dopamine_level`: Current dopamine level (0-1)
- `no_region_silence`: All regions active (bool)

### Consolidation Metrics
- `replay_improvement`: Performance gain from consolidation
- `weight_saturation`: Fraction of saturated synapses

## Integration with Curriculum Trainer

```python
from thalia.training.curriculum import CurriculumTrainer, CurriculumSafetySystem

class SafeCurriculumTrainer(CurriculumTrainer):
    """Curriculum trainer with integrated safety system."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize safety system
        self.safety = CurriculumSafetySystem(
            brain=self.brain,
            current_stage=self.current_stage,
            enable_auto_intervention=True,
            checkpoint_callback=self.save_checkpoint
        )

    def train_step(self, batch):
        """Single training step with safety monitoring."""
        # Normal training
        result = super().train_step(batch)

        # Safety update
        intervention = self.safety.update(
            self.brain,
            self.global_step,
            result
        )

        # Handle intervention
        if intervention:
            self._handle_intervention(intervention)

        return result

    def _handle_intervention(self, intervention):
        """Execute intervention response."""
        actions = self.safety.handle_intervention(intervention, self.brain)

        if 'trigger_consolidation' in actions['actions']:
            self.consolidate(emergency=True)

        if 'reduce_task_complexity' in actions['actions']:
            self.reduce_difficulty()

        if 'rollback_to_checkpoint' in actions['actions']:
            self.load_last_checkpoint()
            raise TrainingInterrupted("Rollback triggered")

    def can_advance_to_next_stage(self):
        """Check if ready for next stage."""
        can_advance, gate_result = self.safety.can_advance_stage()

        if can_advance:
            return True
        else:
            logger.warning(
                f"Cannot advance: {gate_result.failures}"
            )
            return False

    def advance_stage(self):
        """Advance to next stage (after passing gate)."""
        if not self.can_advance_to_next_stage():
            raise ValueError("Stage gate criteria not met")

        self.safety.advance_to_next_stage()
        self.current_stage = self.safety.current_stage
        logger.info(f"Advanced to stage {self.current_stage}")
```

## Logging and Reporting

### Health Score
Continuous health score (0.0-1.0):
- **1.0**: Perfect health
- **0.8-1.0**: Healthy
- **0.6-0.8**: Minor issues
- **0.4-0.6**: Concerning
- **<0.4**: Critical

```python
status = safety.get_status()
print(f"Health Score: {status.health_score:.2f}")

if status.health_score < 0.6:
    print("WARNING: System health concerning")
    print(f"Active alerts: {status.active_alerts}")
    print(f"Degraded modules: {status.degraded_modules}")
```

### Summary Reports
```python
summary = safety.get_summary()

print(f"Stage: {summary['stage']}")
print(f"Health: {summary['health_score']:.2f}")
print(f"Can advance: {summary['can_advance']}")
print(f"Total interventions: {summary['total_interventions']}")
print(f"By type: {summary['interventions_by_type']}")
print(f"Steps in stage: {summary['steps_in_stage']}")
```

## Best Practices

### 1. Always Use Safety System for Stage 1+
```python
# ❌ DON'T: Train without safety
trainer = CurriculumTrainer(brain, stage=1)
trainer.train(steps=50000)

# ✅ DO: Use safety system
trainer = SafeCurriculumTrainer(brain, stage=1)
trainer.train(steps=50000)  # Safety monitoring active
```

### 2. Never Override Emergency Stops
```python
# ❌ DON'T: Ignore critical failures
if intervention == InterventionType.EMERGENCY_STOP:
    logger.warning("Emergency stop, but continuing anyway")
    continue  # DANGEROUS!

# ✅ DO: Respect critical failures
if intervention == InterventionType.EMERGENCY_STOP:
    logger.error("Critical failure - halting")
    sys.exit(1)
```

### 3. Check Gate Before Advancing
```python
# ❌ DON'T: Advance on schedule
if global_step >= 50000:
    advance_to_stage_2()

# ✅ DO: Advance on criteria
if safety.can_advance_stage():
    safety.advance_to_next_stage()
else:
    extend_current_stage(weeks=2)
```

### 4. Log All Interventions
```python
if intervention:
    logger.warning(f"Intervention at step {step}: {intervention.value}")
    log_to_wandb({
        'intervention': intervention.value,
        'health_score': safety.get_status().health_score
    })
```

### 5. Periodic Health Checks
```python
# Check health every 5k steps
if step % 5000 == 0:
    status = safety.get_status()

    if status.health_score < 0.7:
        logger.warning(f"Health declining: {status.health_score:.2f}")
        # Consider consolidation or load reduction
```

## Troubleshooting

### Problem: Stage 1 Gate Keeps Failing

**Symptoms**: Cannot pass Stage 1 gate after extended training

**Possible Causes**:
1. Theta oscillator instability
2. WM capacity insufficient
3. Cross-modal interference

**Solutions**:
```python
gate_result = safety.can_advance_stage()

if 'Theta frequency' in str(gate_result.failures):
    # Oscillator problem - check theta implementation
    check_oscillator_coupling()

if 'n-back accuracy' in str(gate_result.failures):
    # WM problem - extend training or reduce N
    train_n_back_longer(additional_steps=10000)

if 'interference' in str(gate_result.failures):
    # Interference - enable temporal separation
    enable_temporal_separation_mode()
```

### Problem: Frequent Emergency Stops

**Symptoms**: System repeatedly triggers EMERGENCY_STOP

**Possible Causes**:
1. Learning rates too high
2. Task difficulty too aggressive
3. Insufficient consolidation

**Solutions**:
```python
# Reduce learning rates
brain.learning_rate *= 0.5

# Reduce task difficulty
task_difficulty = 0.5  # Start easier

# More frequent consolidation
consolidation_interval = 15000  # Was 25000
```

### Problem: Degraded Modules Not Recovering

**Symptoms**: Modules enter degraded state and don't recover

**Solutions**:
```python
# Force consolidation on degraded module
degraded = safety.degradation_manager.degraded_modules

for module in degraded:
    targeted_consolidation(brain, module, steps=10000)

# If still degraded, may need to rollback
if module still in degraded_modules:
    load_checkpoint('before_degradation.pt')
```

## Architecture Decisions

### Why Stage 1 Gets Special Treatment

**Rationale**: Both expert review and ChatGPT engineering analysis identified Stage 1 as the highest-risk transition:
- Working memory emerging (fragile theta-gamma coupling)
- Multi-modal learning (phonology + vision + WM simultaneously)
- Bilingual foundations (increased interference potential)

**Impact**: If Stage 1 fails, all downstream stages inherit instability. Better to over-invest here than debug Stage 3 failures.

### Why Kill-Switch Map

**Rationale**: "You cannot have a stage where a single failure cascades and destroys the entire system." - Not all modules are equally critical.

**Impact**: Language/grammar failures don't collapse thinking/planning. Only WM/oscillators/replay are single points of failure.

### Why Continuous Monitoring

**Rationale**: ChatGPT: "Systems fail at rare coupling instabilities, not where you expect."

**Impact**: Real-time anomaly detection catches problems before they become catastrophic.

## Future Extensions

### Stage 2-6 Gates
Currently only Stage 1 has explicit gate. Future work:
- Stage 2 gate (grammar compositionality)
- Stage 3 gate (reading comprehension)
- Stage 4+ gates (abstract reasoning)

### Predictive Failure Detection
Use metric trends to predict failures before they occur:
- ML model trained on failure patterns
- Early warning 1000+ steps before failure
- Proactive interventions

### Adaptive Thresholds
Learn optimal thresholds from training data:
- Individual brain variations
- Task-specific adjustments
- Stage-specific calibration

## References

1. **Expert Review**: `docs/design/curriculum_strategy.md` (v0.6.0, A+ grade)
2. **ChatGPT Analysis**: `temp/chatgpt_curriculum_3.md` (Consensus design)
3. **Stage Gates**: `src/thalia/training/curriculum/stage_gates.py`
4. **Monitoring**: `src/thalia/training/curriculum/stage_monitoring.py`
5. **Integration**: `src/thalia/training/curriculum/safety_system.py`
