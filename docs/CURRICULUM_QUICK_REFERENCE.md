# Curriculum Training Quick Reference

## Quick API Reference

### Initialize Trainer

```python
from thalia.training.curriculum.stage_manager import CurriculumTrainer
from thalia.config.curriculum_growth import get_curriculum_growth_config

trainer = CurriculumTrainer(
    brain=brain,
    growth_config=get_curriculum_growth_config(),
    checkpoint_dir="checkpoints/my_curriculum",
)
```

### Train a Stage

```python
from thalia.config.curriculum_growth import CurriculumStage
from thalia.training.curriculum.stage_manager import StageConfig, TaskConfig

result = trainer.train_stage(
    stage=CurriculumStage.SENSORIMOTOR,
    config=StageConfig(
        duration_steps=50000,
        task_configs={
            'task1': TaskConfig(weight=0.4, difficulty=0.5),
            'task2': TaskConfig(weight=0.6, difficulty=0.7),
        },
        success_criteria={
            'task1_accuracy': 0.90,
        },
    ),
    task_loader=my_task_loader,
    evaluator=my_evaluator,
)
```

### Check Milestones

```python
if trainer.evaluate_stage_readiness(CurriculumStage.SENSORIMOTOR):
    print("✅ Ready for next stage")
```

### Transition Between Stages

```python
trainer.transition_to_stage(
    new_stage=CurriculumStage.PHONOLOGY,
    old_stage=CurriculumStage.SENSORIMOTOR,
    weeks=4,  # Gradual ramp
)
```

### Evaluation

```python
from thalia.training.curriculum.stage_evaluation import evaluate_stage_sensorimotor

results = evaluate_stage_sensorimotor(brain, sensorimotor_wrapper)
# Returns: {'criterion': True/False, ...}

# Note: Each stage has its own evaluation function:
# - evaluate_stage_sensorimotor()
# - evaluate_stage_phonology()
# - evaluate_stage_toddler()
# - etc.
```

## Curriculum Stages

```python
from thalia.config.curriculum_growth import CurriculumStage

CurriculumStage.SENSORIMOTOR  # -1: Motor control
CurriculumStage.PHONOLOGY     #  0: Sensory foundations
CurriculumStage.TODDLER       #  1: Object permanence
CurriculumStage.GRAMMAR       #  2: Composition
CurriculumStage.READING       #  3: Reading & writing
CurriculumStage.ABSTRACT      #  4: Abstract reasoning
```

## Task Loader Template

```python
class MyTaskLoader:
    def get_task(self, task_name: str) -> dict:
        return {
            'input': torch.Tensor,      # Spike pattern
            'n_timesteps': int,         # Simulation steps
            'reward': float,            # Optional (RL)
            'target': torch.Tensor,     # Optional (supervised)
        }
```

## Evaluator Template

```python
def my_evaluator(brain, task_loader) -> dict:
    """Return dict of {criterion: bool}"""
    results = {}

    # Test performance
    accuracy = test_task(brain, task_loader)
    results['task_accuracy'] = accuracy > 0.90

    # Check health
    from thalia.training import check_system_health
    results.update(check_system_health(brain))

    return results
```

## Common Configurations

### Stage -0.5 (Sensorimotor)
```python
StageConfig(
    duration_steps=50000,
    task_configs={
        'motor_control': TaskConfig(0.4, 0.5),
        'reaching': TaskConfig(0.35, 0.6),
        'manipulation': TaskConfig(0.25, 0.7),
    },
    success_criteria={
        'reaching_accuracy': 0.90,
        'manipulation_success': 0.85,
        'prediction_error': 0.05,
    },
)
```

### Stage 0 (Phonology)
```python
StageConfig(
    duration_steps=60000,
    task_configs={
        'mnist': TaskConfig(0.40, 0.5),
        'temporal': TaskConfig(0.25, 0.6),
        'phonology': TaskConfig(0.35, 0.7),
    },
    success_criteria={
        'mnist_accuracy': 0.95,
        'sequence_prediction': 0.90,
        'phoneme_discrimination': 0.90,
    },
    review_stages={-1: 0.10},  # 10% Stage -0.5 review
)
```

## Checkpointing

```python
# Auto-save during training
config = StageConfig(checkpoint_interval=5000)

# Manual save
trainer.save_checkpoint(name="my_checkpoint")

# Load and resume
trainer.load_checkpoint("checkpoints/my_checkpoint.thalia")
```

## Monitoring

```python
# Add callback
def my_callback(step: int, metrics: dict):
    print(f"Step {step}: {metrics}")

trainer.add_callback(my_callback)

# Check history
for result in trainer.training_history:
    print(f"{result.stage.name}: {result.success}")
```

## Health Checks

```python
from thalia.training import check_system_health

health = check_system_health(brain)
# Returns: {
#   'firing_stability': bool,
#   'no_runaway': bool,
#   'bcm_convergence': bool,
#   'weight_health': bool,
#   'no_silence': bool,
# }
```

## Growth Configuration

```python
from thalia.config.curriculum_growth import get_curriculum_growth_config

# Conservative (slower growth)
config = get_curriculum_growth_config(conservative=True)

# Aggressive (faster growth)
config = get_curriculum_growth_config(conservative=False)
```

## Common Issues

### Milestones not met
```python
trainer.extend_stage(
    stage=CurriculumStage.SENSORIMOTOR,
    additional_steps=20000,
    reason="Performance below threshold",
)
```

### Catastrophic forgetting
```python
# Increase review proportion
config = StageConfig(
    review_stages={0: 0.20},  # More review
)
```

### Growth not triggering
```python
# Use less conservative config
config = get_curriculum_growth_config(conservative=False)
```

## Documentation

- Full strategy: `docs/design/curriculum_strategy.md`
- Getting started: `docs/GETTING_STARTED_CURRICULUM.md`
- Example: `examples/curriculum_training_example.py`
- Checkpoints: `docs/design/checkpoint_format.md`

## Test Commands

```bash
# Run example
python examples/curriculum_training_example.py

# Run tests (when implemented)
pytest tests/integration/test_curriculum*.py

# Check imports
python -c "from thalia.training import CurriculumTrainer; print('✅ OK')"
```
