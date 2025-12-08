# Curriculum Training Examples

This directory contains examples demonstrating how to use Thalia's curriculum training system.

## Examples

### 1. `curriculum_training_example.py`

**Purpose**: Demonstrates the complete curriculum training pipeline from Stage -0.5 (Sensorimotor) to Stage 0 (Phonology).

**What it shows**:
- Initializing `CurriculumTrainer`
- Configuring stages with `StageConfig`
- Training through multiple stages
- Evaluating milestones
- Stage transitions with gradual difficulty ramps
- Checkpoint management

**How to run**:
```bash
cd examples
python curriculum_training_example.py
```

**Note**: This is a minimal demonstration. For full curriculum training:
1. Real task datasets needed (MNIST, phonology, etc.)
2. Longer training durations (50k-60k steps per stage)
3. Custom evaluation functions for your specific tasks

## Key Components

### CurriculumTrainer

Main orchestration class that coordinates:
- Stage progression
- Interleaved task sampling
- Growth triggering
- Consolidation cycles
- Milestone evaluation
- Checkpoint management

```python
from thalia.training import CurriculumTrainer, CurriculumStage

trainer = CurriculumTrainer(
    brain=brain,
    growth_config=growth_config,
    checkpoint_dir="checkpoints/my_curriculum",
)

result = trainer.train_stage(
    stage=CurriculumStage.SENSORIMOTOR,
    config=stage_config,
    task_loader=task_loader,
    evaluator=evaluation_function,
)
```

### StageConfig

Configuration for a single curriculum stage:

```python
from thalia.training import StageConfig, TaskConfig

config = StageConfig(
    duration_steps=50000,
    task_configs={
        'task1': TaskConfig(weight=0.4, difficulty=0.5),
        'task2': TaskConfig(weight=0.6, difficulty=0.7),
    },
    success_criteria={
        'task1_accuracy': 0.90,
        'task2_success': 0.85,
    },
    consolidation_interval=10000,
    checkpoint_interval=5000,
)
```

### Stage Evaluation

Milestone checking functions:

```python
from thalia.training import (
    evaluate_stage_sensorimotor,
    evaluate_stage_phonology,
    check_system_health,
)

# Evaluate Stage -0.5
results = evaluate_stage_sensorimotor(brain, sensorimotor_wrapper)

if all(results.values()):
    print("✅ Stage complete!")
else:
    failed = [k for k, v in results.items() if not v]
    print(f"❌ Failed: {failed}")
```

## Curriculum Stages

The curriculum follows human cognitive development:

- **Stage -0.5** (Sensorimotor): Motor control, reaching, manipulation
- **Stage 0** (Phonology): MNIST, temporal sequences, phoneme discrimination
- **Stage 1** (Toddler): CIFAR-10, working memory, object permanence
- **Stage 2** (Grammar): Composition, multilingual, executive function
- **Stage 3** (Reading): Decoding, comprehension, planning
- **Stage 4** (Abstract): Reasoning, analogies, metacognition
- **Stage 5** (Expert): Domain expertise, multi-modal integration
- **Stage 6** (LLM-level): Full language understanding and generation

See `docs/design/curriculum_strategy.md` for complete strategy.

## Next Steps

1. **Implement real task loaders**: Replace mock loaders with actual datasets
2. **Add custom evaluators**: Implement domain-specific evaluation functions
3. **Tune hyperparameters**: Adjust growth thresholds, consolidation frequency, etc.
4. **Monitor training**: Add callbacks for logging, visualization, etc.
5. **Scale up**: Increase duration, brain size, task complexity

## Related Documentation

- [`docs/design/curriculum_strategy.md`](../docs/design/curriculum_strategy.md) - Complete training strategy
- [`docs/design/curriculum_implementation.md`](../docs/design/curriculum_implementation.md) - Implementation plan
- [`docs/design/checkpoint_format.md`](../docs/design/checkpoint_format.md) - Checkpoint system
- [`docs/patterns/component-parity.md`](../docs/patterns/component-parity.md) - Regions and pathways

## Getting Help

If you encounter issues:
1. Check the implementation plan: `docs/design/curriculum_implementation.md`
2. Review test examples: `tests/integration/test_curriculum*.py`
3. File an issue with details about your setup and error messages
