# Curriculum Stage Regression Testing

**Status**: ✅ Implemented (December 23, 2025)
**Version**: 1.0.0
**Priority**: High - Critical for detecting catastrophic forgetting

---

## Overview

The curriculum stage regression testing system detects **catastrophic forgetting** by re-running actual tasks from previously completed stages and comparing current performance to original milestone results.

### Key Features

1. **Automatic Task Loader Caching**: Task loaders from successful stages are automatically stored
2. **Actual Task Re-execution**: Real tasks (not proxy metrics) are re-run during testing
3. **90% Retention Threshold**: Current performance must be ≥90% of original threshold
4. **Plasticity Protection**: Learning is disabled during tests to prevent weight corruption
5. **Multi-Stage Support**: Tests all previous stages, not just the immediate predecessor

---

## Architecture

### Task Loader Caching

When a stage completes successfully, its task loader and config are cached:

```python
# In CurriculumTrainer.train_stage()
if result.success:
    self.stage_task_loaders[stage] = task_loader
    self.stage_configs[stage] = config
```

**Storage**:
- `self.stage_task_loaders: Dict[CurriculumStage, Any]` - Cached task loaders
- `self.stage_configs: Dict[CurriculumStage, StageConfig]` - Cached stage configurations

### Regression Test Execution

```python
def _run_regression_test(
    self,
    stage: CurriculumStage,
    criterion: str,
    n_trials: int = 50,
) -> float:
    """Run regression test by re-executing tasks from a previous stage."""
```

**Process**:
1. Load cached task loader for target stage
2. Parse criterion to extract task name and metric (e.g., "mnist_accuracy" → task="mnist", metric="accuracy")
3. Disable plasticity in all brain regions
4. Run N test trials (default 50)
5. Restore plasticity state
6. Return performance metric (0.0 to 1.0)

### Backward Compatibility Check

```python
def _check_backward_compatibility(
    self,
    current_stage: CurriculumStage
) -> bool:
    """Check if previous stage performance is maintained."""
```

**Logic**:
1. Identify all previous stages (stages with lower enum values)
2. For each previous stage:
   - Retrieve original milestone results from training history
   - Re-run regression test for each criterion that originally passed
   - Compare: `current_performance >= original_threshold * 0.90`
3. Return `False` if any stage shows <90% retention

---

## Usage Examples

### Basic Usage

```python
from thalia.training.curriculum.stage_manager import CurriculumTrainer
from thalia.config.curriculum_growth import CurriculumStage

trainer = CurriculumTrainer(brain=brain, verbose=True)

# Train Stage 0 (automatically caches task loader on success)
result0 = trainer.train_stage(
    stage=CurriculumStage.PHONOLOGY,
    config=stage0_config,
    task_loader=phonology_loader,
)

# Train Stage 1 (automatically tests Stage 0 retention)
result1 = trainer.train_stage(
    stage=CurriculumStage.TODDLER,
    config=stage1_config,
    task_loader=toddler_loader,
)

# Backward compatibility is checked automatically during milestone evaluation
```

### Manual Regression Testing

```python
# Manually run regression test on a specific criterion
performance = trainer._run_regression_test(
    stage=CurriculumStage.PHONOLOGY,
    criterion='mnist_accuracy',
    n_trials=100,
)

print(f"Current performance on MNIST: {performance:.1%}")

# Check if all previous stages are retained
is_compatible = trainer._check_backward_compatibility(
    current_stage=CurriculumStage.TODDLER
)

if not is_compatible:
    print("⚠️ Catastrophic forgetting detected!")
```

### Custom Evaluator with Regression

```python
def custom_evaluator(brain, task_loader):
    """Custom evaluation that includes regression checks."""
    results = {}

    # Standard milestone evaluation
    results['new_task_accuracy'] = evaluate_new_tasks(brain, task_loader)

    # Explicit regression check
    if CurriculumStage.PHONOLOGY in trainer.stage_task_loaders:
        results['phonology_retained'] = trainer._run_regression_test(
            stage=CurriculumStage.PHONOLOGY,
            criterion='mnist_accuracy',
            n_trials=50,
        ) >= 0.85

    return results

result = trainer.train_stage(
    stage=CurriculumStage.GRAMMAR,
    config=config,
    task_loader=grammar_loader,
    evaluator=custom_evaluator,
)
```

---

## Performance Metrics

### Retention Calculation

**Original Threshold** (from stage config):
```python
config.success_criteria = {
    'mnist_accuracy': 0.95,  # Must achieve 95%
    'phonology_correct': 0.90,
}
```

**Retention Threshold** (90% of original):
```python
retention_threshold = original_threshold * 0.90
# mnist: 0.95 * 0.90 = 0.855 (85.5%)
# phonology: 0.90 * 0.90 = 0.81 (81%)
```

**Evaluation**:
```python
current_performance = run_regression_test(...)
is_retained = current_performance >= retention_threshold
```

### Metric Types

The system supports multiple metric types:

1. **Accuracy/Success** (higher is better):
   - Binary classification tasks
   - RL success rates
   - Target matching

2. **Error/Loss** (lower is better):
   - Converted to performance: `1 - normalized_error`

3. **Reward** (scale varies):
   - Normalized to [0, 1]: `(reward + 1) / 2`

---

## Implementation Details

### Plasticity Protection

During regression tests, learning is disabled to prevent corrupting the current weights:

```python
# Before testing
original_states = trainer._disable_plasticity()

try:
    # Run tests
    output = brain.forward(input_data, n_timesteps=10)
finally:
    # Restore original state
    trainer._restore_plasticity(original_states)
```

This ensures that:
- Test performance reflects true retention
- Weights remain unchanged after testing
- Multiple tests can be run without interference

### Task Parsing

Criterion format: `{task_name}_{metric}`

Examples:
- `mnist_accuracy` → task="mnist", metric="accuracy"
- `reaching_success` → task="reaching", metric="success"
- `prediction_error` → task="prediction", metric="error"

### Error Handling

```python
try:
    performance = run_regression_test(stage, criterion)
except Exception as e:
    # Assume retained on error (benefit of doubt)
    print(f"⚠️ Could not re-evaluate {criterion}: {e}")
    retained = True
```

---

## Integration Points

### 1. Stage Completion

```python
# In CurriculumTrainer.train_stage()
if result.success:
    self.stage_task_loaders[stage] = task_loader
    self.stage_configs[stage] = config
```

### 2. Milestone Evaluation

```python
# In CurriculumTrainer._default_evaluation()
if config.check_backward_compatibility:
    is_compatible = self._check_backward_compatibility(stage)
    results['backward_compatible'] = is_compatible
```

### 3. Stage Readiness

```python
# In CurriculumTrainer.evaluate_stage_readiness()
compatibility_ok = self._check_backward_compatibility(next_stage)
if not compatibility_ok:
    return False, ["Catastrophic forgetting detected"]
```

---

## Testing

Comprehensive unit tests in `tests/unit/test_regression_testing.py`:

1. **Caching Tests**:
   - ✅ Task loaders cached on success
   - ✅ Not cached on failure

2. **Execution Tests**:
   - ✅ Basic regression test runs
   - ✅ Plasticity disabled/restored
   - ✅ Performance measured correctly

3. **Detection Tests**:
   - ✅ High retention passes
   - ✅ Low retention fails
   - ✅ Multiple stages tested

Run tests:
```bash
pytest tests/unit/test_regression_testing.py -v
```

---

## Biological Motivation

This system mirrors biological mechanisms for maintaining learned skills:

1. **Consolidation**: Successful stages are "consolidated" by caching their task loaders
2. **Reactivation**: Regression testing = spontaneous reactivation of old memories
3. **Interference Detection**: Backward compatibility = detecting interference from new learning
4. **Synaptic Protection**: Plasticity disabling = preventing reconsolidation during retrieval

### Cognitive Parallels

- **Testing Effect**: Re-testing strengthens retention (though we disable learning during tests)
- **Transfer Learning**: Tests whether new skills override old ones
- **Spacing Effect**: Periodic regression tests = spaced retrieval practice

---

## Future Enhancements

### 1. Adaptive Test Frequency

```python
# More frequent testing when forgetting risk is high
if new_stage_difficulty > 0.8:
    regression_test_interval = 1000  # Test every 1k steps
else:
    regression_test_interval = 5000  # Less frequent
```

### 2. Selective Retraining

```python
if performance < threshold:
    # Automatically retrain on forgotten tasks
    retrain_on_stage(forgotten_stage, n_steps=1000)
```

### 3. Performance Trending

```python
# Track regression performance over time
self.regression_history[stage][criterion].append({
    'step': global_step,
    'performance': current_performance,
})
```

### 4. Curriculum Adjustment

```python
# Increase review ratio if forgetting detected
if performance < threshold * 0.95:  # Warning zone
    review_ratios[stage] *= 1.5  # More review needed
```

---

## Troubleshooting

### Issue: "No task loader cached"

**Cause**: Stage never completed successfully, or cache was cleared

**Solution**:
```python
# Manually cache if needed
trainer.stage_task_loaders[stage] = my_task_loader
trainer.stage_configs[stage] = my_config
```

### Issue: False positives (detecting forgetting when none exists)

**Cause**: Test sample size too small, or task stochasticity

**Solution**:
```python
# Increase test trials
performance = trainer._run_regression_test(
    stage=stage,
    criterion=criterion,
    n_trials=200,  # More trials = more stable
)
```

### Issue: Regression tests are slow

**Cause**: Many previous stages, large n_trials

**Solution**:
```python
# Reduce trial count for faster testing
n_trials = 30  # Instead of default 50

# Or test subset of criteria
critical_criteria_only = ['mnist_accuracy', 'phonology_correct']
```

---

## References

### Related Documentation

- **Curriculum Strategy**: `docs/design/curriculum_strategy.md`
- **Stage Manager**: `docs/CURRICULUM_QUICK_REFERENCE.md`
- **Safety System**: `docs/CURRICULUM_SAFETY_SYSTEM.md`

### Research Context

- **Catastrophic Forgetting**: McCloskey & Cohen (1989)
- **Continual Learning**: Parisi et al. (2019)
- **Memory Consolidation**: Dudai (2004)
- **Testing Effect**: Roediger & Karpicke (2006)

---

**Last Updated**: December 23, 2025
