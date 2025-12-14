# Safety System Integration - Quick Start

## Overview

The safety system is now **automatically integrated** into `CurriculumTrainer`. All Stage 1+ training benefits from comprehensive monitoring, intervention handling, and stage gate enforcement.

## Default Behavior (Recommended)

```python
from thalia.training.curriculum import CurriculumTrainer
from thalia.config.curriculum_growth import CurriculumStage

# Safety system is ENABLED by default
trainer = CurriculumTrainer(
    brain=brain,
    checkpoint_dir='checkpoints',
    verbose=True,
    # enable_safety_system=True  # Default - no need to specify
)

# Safety monitoring is automatic
trainer.train_stage(
    stage=CurriculumStage.WORKING_MEMORY,
    config=stage1_config,
    task_loader=loader
)

# Safety gate is checked automatically
if trainer.evaluate_stage_readiness(CurriculumStage.WORKING_MEMORY):
    print("✅ Ready for Stage 2!")
else:
    print("❌ Not ready - see safety gate failures")
```

## What Happens Automatically

### During Training

**Every Step**:
- Monitor oscillator stability (theta 6.5-8.5 Hz, phase-locking >0.4)
- Track working memory performance (2-back accuracy, attractor stability)
- Detect cross-modal interference (phonology/vision simultaneous failure)
- Check firing rate health (all regions active 0.05-0.15 Hz)
- Watch for dopamine saturation

**Every 5k Steps** (Logged):
```
[Stage WORKING_MEMORY] Step 25,000 |
✅ 42.3 steps/s | Forward: 23.5ms | GPU: 1024MB | CPU: 512MB | FR: 0.082
🛡️ Safety: 🟢 Health=0.87 | Alerts=0 | Interventions=0
```

### Interventions (Automatic)

If problems detected, trainer **automatically** handles them:

**1. REDUCE_LOAD** (Cognitive overload):
```
⚠️ SAFETY INTERVENTION: REDUCE_LOAD
📉 Reducing cognitive load
   Learning rate: 0.0010 → 0.0005
```

**2. CONSOLIDATE** (Performance degradation):
```
⚠️ SAFETY INTERVENTION: CONSOLIDATE
⏸️ Triggering emergency consolidation
   10,000 replay steps
```

**3. TEMPORAL_SEPARATION** (Cross-modal interference):
```
⚠️ SAFETY INTERVENTION: TEMPORAL_SEPARATION
🔀 Enabling temporal separation of modalities
   Will alternate phonology/vision training
```

**4. EMERGENCY_STOP** (Critical failure):
```
⚠️ SAFETY INTERVENTION: EMERGENCY_STOP
❌ EMERGENCY STOP triggered
Actions: ['freeze_learning', 'rollback_to_checkpoint']
System frozen. Manual investigation required.

RuntimeError: Emergency stop triggered
```

### Stage Transitions

**Before Transition**:
```python
trainer.evaluate_stage_readiness(CurriculumStage.WORKING_MEMORY)
```

**Safety Gate Check**:
```
================================================================================
Evaluating readiness for Stage WORKING_MEMORY
================================================================================

✅ Safety gate PASSED for Stage WORKING_MEMORY
Gate metrics: {
  'mean_theta_frequency': 7.82,
  'theta_variance': 0.12,
  'gamma_theta_locking': 0.47,
  'n_back_2_accuracy': 0.83,
  'mean_firing_rate': 0.089,
  'replay_improvement': 0.034
}

Milestone Results:
  ✅ reaching_accuracy: True
  ✅ n_back_2_performance: True
  ✅ language_command_following: True
  ✅ firing_rate_stability: True

✅ Stage WORKING_MEMORY ready for transition!
```

**If Gate Fails**:
```
❌ SAFETY GATE FAILED for Stage WORKING_MEMORY
Failures: [
  'Theta variance 0.18 exceeds max 0.15',
  '2-back accuracy 0.76 below min 0.80',
  'Gamma-theta locking 0.35 below min 0.40'
]
Recommendations: [
  'Extend Stage 1 (add 1-2 weeks)',
  'Reduce WM load to stabilize oscillations',
  'More frequent consolidation'
]

Must address these issues before proceeding.
```

## Safety Status Monitoring

### Programmatic Access

```python
# Get current safety status
status = trainer.safety_system.get_status()

print(f"Stage: {status.stage}")
print(f"Health Score: {status.health_score:.2f}")  # 0.0-1.0
print(f"Can Advance: {status.can_advance}")
print(f"Active Alerts: {status.active_alerts}")
print(f"Degraded Modules: {status.degraded_modules}")
print(f"Total Interventions: {status.interventions_triggered}")

# Get detailed summary
summary = trainer.safety_system.get_summary()
print(summary)
```

### Health Score Interpretation

- **1.0**: Perfect health
- **0.8-1.0**: 🟢 Healthy
- **0.6-0.8**: 🟡 Minor issues
- **0.4-0.6**: 🟠 Concerning
- **<0.4**: 🔴 Critical

## Disabling Safety (Not Recommended)

```python
# Only for testing/debugging
trainer = CurriculumTrainer(
    brain=brain,
    enable_safety_system=False,  # ⚠️ NOT RECOMMENDED
    verbose=True
)
```

**Why not recommended**:
- No protection against oscillator instability
- No working memory collapse detection
- No cross-modal interference handling
- No stage gate enforcement
- Higher risk of catastrophic failure

**Use cases**:
- Unit testing specific components
- Debugging brain initialization
- Research experiments with known-safe configs

## Stage 1 (Highest Risk)

Stage 1 gets **extra monitoring**:
- Checks every 500 steps (vs 1000 for other stages)
- Stricter thresholds (theta variance <0.18 vs 0.20)
- Higher WM requirements (65% vs 60% critical)
- More frequent gate checks

This is **intentional** - Stage 1 determines success of all downstream stages.

## Emergency Procedures

### If EMERGENCY_STOP Triggered

1. **Don't panic** - system has saved checkpoint
2. **Check logs** for failure details
3. **Investigate** the specific failure (oscillators? WM? region silence?)
4. **Adjust hyperparameters** based on failure mode
5. **Load last checkpoint** and retry with adjustments

### If Repeated Interventions

```python
# Check intervention history
summary = trainer.safety_system.get_summary()
print(f"Total interventions: {summary['total_interventions']}")
print(f"By type: {summary['interventions_by_type']}")

# Example output:
# Total interventions: 15
# By type: {'REDUCE_LOAD': 8, 'CONSOLIDATE': 7}

# If too many interventions:
# - Learning rate may be too high
# - Task difficulty too aggressive
# - Insufficient consolidation frequency
```

### If Gate Keeps Failing

```python
# Check gate failures
can_advance, gate_result = trainer.safety_system.can_advance_stage()

if not can_advance:
    print("Failed criteria:")
    for failure in gate_result.failures:
        print(f"  - {failure}")

    print("\nRecommendations:")
    for rec in gate_result.recommendations:
        print(f"  - {rec}")

    print("\nMetrics:")
    for key, value in gate_result.metrics.items():
        print(f"  {key}: {value}")
```

## Advanced: Manual Gate Override (Use Carefully)

```python
# Check gate status
can_advance, gate_result = trainer.safety_system.can_advance_stage()

if not can_advance:
    # Review failures
    print(gate_result.failures)

    # If you understand the risk and want to proceed anyway:
    # (e.g., slight theta variance but everything else perfect)

    # Option 1: Extend training
    trainer.train_stage(stage, config, loader)  # More training

    # Option 2: Adjust thresholds (research use only)
    trainer.safety_system.stage_gates[1].THETA_MAX_VARIANCE = 0.20  # Was 0.15

    # Option 3: Force advance (NOT RECOMMENDED - no safety net)
    trainer.safety_system.current_stage += 1
    # ⚠️ Bypasses ALL safety checks - use at own risk
```

## Integration with Logging

### W&B Integration

```python
import wandb

def log_safety_callback(step: int, metrics: dict):
    """Log safety metrics to W&B."""
    if trainer.safety_system:
        status = trainer.safety_system.get_status()
        wandb.log({
            'safety/health_score': status.health_score,
            'safety/num_alerts': len(status.active_alerts),
            'safety/num_degraded': len(status.degraded_modules),
            'safety/total_interventions': status.interventions_triggered,
        }, step=step)

trainer.callbacks.append(log_safety_callback)
```

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/curriculum')

def tensorboard_safety_callback(step: int, metrics: dict):
    if trainer.safety_system:
        status = trainer.safety_system.get_status()
        writer.add_scalar('Safety/HealthScore', status.health_score, step)
        writer.add_scalar('Safety/Alerts', len(status.active_alerts), step)
        writer.add_scalar('Safety/Interventions', status.interventions_triggered, step)

trainer.callbacks.append(tensorboard_safety_callback)
```

## Summary

**Key Points**:
1. ✅ Safety system is **enabled by default** - no configuration needed
2. ✅ Monitoring is **automatic** - runs every step
3. ✅ Interventions are **automatic** - trainer handles them
4. ✅ Stage gates are **mandatory** - cannot bypass without explicit code
5. ✅ Stage 1 gets **extra scrutiny** - highest risk transition
6. ✅ All safety events are **logged** - visible in output

**When to Intervene Manually**:
- Emergency stop triggered (investigate, adjust, retry)
- Repeated interventions (adjust hyperparameters)
- Gate repeatedly fails (extend training or adjust thresholds)

**When NOT to Intervene**:
- Occasional interventions (system is working correctly)
- Minor health score fluctuations (0.7-0.9 is normal)
- Single gate failure (address issues, don't bypass)

**Remember**: The safety system exists because both expert review and engineering analysis agreed:

> "You cannot have a stage where a single failure cascades and destroys the entire system."

Let it protect your training runs. 🛡️
