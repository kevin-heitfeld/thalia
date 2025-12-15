# Getting Started with Curriculum Training

This guide helps you get started with Thalia's curriculum training system for developing brain models from basic sensorimotor control to LLM-level capabilities.

## Quick Start

**Basic Usage Pattern**

```python
from thalia.core import Brain  # EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes
from thalia.config.curriculum_growth import CurriculumStage, get_curriculum_growth_config
from thalia.training.curriculum.stage_manager import CurriculumTrainer, StageConfig

# 1. Create brain with ThaliaConfig
config = ThaliaConfig(
    global_=GlobalConfig(device="cpu"),
    brain=BrainConfig(
        sizes=RegionSizes(
            input_size=400,
            cortex_size=10000,
            hippocampus_size=2000,
            pfc_size=5000,
            n_actions=4,
        ),
    ),
)
brain = EventDrivenBrain.from_thalia_config(config)

# 2. Initialize trainer
trainer = CurriculumTrainer(
    brain=brain,
    growth_config=get_curriculum_growth_config(),
    checkpoint_dir="checkpoints/my_training",
)

# 3. Train a stage
result = trainer.train_stage(
    stage=CurriculumStage.SENSORIMOTOR,
    config=StageConfig(
        duration_steps=50000,
        task_configs={...},
        success_criteria={...},
    ),
    task_loader=your_task_loader,
    evaluator=your_evaluation_function,
)

# 4. Check if ready for next stage
if trainer.evaluate_stage_readiness(CurriculumStage.SENSORIMOTOR):
    # Proceed to next stage
    trainer.transition_to_stage(
        new_stage=CurriculumStage.PHONOLOGY,
        old_stage=CurriculumStage.SENSORIMOTOR,
    )
```

## Architecture Overview

### CurriculumTrainer

The main orchestrator that coordinates all subsystems:

```
CurriculumTrainer
├── Brain (your neural model)
├── GrowthManager (automatic capacity expansion)
├── MemoryPressureDetector (consolidation triggering)
├── InterleavedCurriculumSampler (task mixing)
├── StageTransitionProtocol (smooth transitions)
└── CheckpointManager (state persistence)
```

### Training Flow

```
┌─────────────────┐
│   Initialize    │
│     Brain       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train Stage N  │◄─────┐
│  - Sample tasks │      │
│  - Forward pass │      │
│  - Local learn  │      │
│  - Check growth │      │
│  - Consolidate  │      │
└────────┬────────┘      │
         │               │
         ▼               │
┌─────────────────┐      │
│  Evaluate       │      │
│  Milestones     │      │
└────────┬────────┘      │
         │               │
    ┌────┴────┐          │
    │ Pass?   │          │
    └─┬────┬──┘          │
      │    │             │
     Yes   No            │
      │    │             │
      │    └─────────────┘
      │           (extend)
      ▼
┌─────────────────┐
│   Transition    │
│  to Stage N+1   │
└─────────────────┘
```

## Curriculum Stages

Training follows human cognitive development:

| Stage | Name | Duration | Key Tasks |
|-------|------|----------|-----------|
| -0.5 | Sensorimotor | 4 weeks | Motor control, reaching, manipulation |
| 0 | Phonology | 8 weeks | MNIST, sequences, phoneme discrimination |
| 1 | Toddler | 8 weeks | CIFAR-10, N-back, object permanence |
| 2 | Grammar | 10 weeks | Composition, multilingual, executive function |
| 3 | Reading | 12 weeks | Decoding, comprehension, planning |
| 4 | Abstract | 16 weeks | Reasoning, analogies, metacognition |
| 5 | Expert | 20 weeks | Domain expertise, multi-modal |
| 6 | LLM-level | 24+ weeks | Full language capabilities |

See [`curriculum_strategy.md`](design/curriculum_strategy.md) for complete details.

## Configuration

### StageConfig

Controls training for a single stage:

```python
from thalia.training import StageConfig, TaskConfig

config = StageConfig(
    # Duration
    duration_steps=50000,

    # Tasks with weights
    task_configs={
        'task1': TaskConfig(weight=0.4, difficulty=0.5),
        'task2': TaskConfig(weight=0.6, difficulty=0.7),
    },

    # Success criteria (go/no-go)
    success_criteria={
        'task1_accuracy': 0.90,
        'task2_success': 0.85,
    },

    # Review previous stages
    review_stages={
        0: 0.10,  # 10% review of Stage 0
    },

    # Consolidation
    consolidation_interval=10000,
    consolidation_cycles=5,

    # Checkpointing
    checkpoint_interval=5000,
)
```

### Growth Configuration

Controls when and how brain grows:

```python
from thalia.config.curriculum_growth import get_curriculum_growth_config

# Conservative growth (slower, safer)
growth_config = get_curriculum_growth_config(conservative=True)

# Aggressive growth (faster, riskier)
growth_config = get_curriculum_growth_config(conservative=False)

# Custom growth per component
from thalia.config.curriculum_growth import CurriculumGrowthConfig

custom_config = CurriculumGrowthConfig(
    enable_growth=True,
    performance_plateau_threshold=0.02,
    performance_window_steps=5000,
)
```

## Task Loaders

You need to implement task loaders for each stage. A task loader provides samples on demand:

```python
class MyTaskLoader:
    def get_task(self, task_name: str) -> dict:
        """Return a task sample.

        Returns:
            dict with keys:
            - 'input': torch.Tensor (spike pattern)
            - 'n_timesteps': int
            - 'reward': float (optional, for RL tasks)
            - 'target': torch.Tensor (optional, for supervised tasks)
        """
        # Your task generation logic here
        return {
            'input': input_spikes,
            'n_timesteps': 10,
            'reward': 1.0,  # if RL
            'target': target,  # if supervised
        }
```

### Existing Task Support

- **Stage -0.5**: `SensorimotorWrapper` (Gymnasium + MuJoCo)
- **Stage 0**: `PhonologicalDataset`, MNIST (torchvision)
- **Stage 1+**: `ExecutiveFunctionTasks`, `NBackTask`

## Evaluation Functions

Evaluation functions check if milestones are met:

```python
from thalia.training import evaluate_stage_sensorimotor

results = evaluate_stage_sensorimotor(brain, sensorimotor_wrapper)

# Results is dict: {criterion_name: True/False}
if all(results.values()):
    print("✅ Stage complete!")
else:
    failed = [k for k, v in results.items() if not v]
    print(f"❌ Failed: {failed}")
```

### Built-in Evaluators

- `evaluate_stage_sensorimotor(brain, wrapper)`
- `evaluate_stage_phonology(brain, datasets, wrapper)`
- `evaluate_stage_toddler(brain, datasets, stage0_datasets)`

### Custom Evaluators

```python
def my_custom_evaluator(brain, task_loader):
    """Custom evaluation function."""
    results = {}

    # Test task performance
    accuracy = test_my_task(brain, task_loader)
    results['my_task_accuracy'] = accuracy > 0.90

    # Check system health
    from thalia.training import check_system_health
    health = check_system_health(brain)
    results.update(health)

    return results

# Use in training
trainer.train_stage(
    stage=CurriculumStage.PHONOLOGY,
    config=config,
    task_loader=task_loader,
    evaluator=my_custom_evaluator,
)
```

## Checkpointing

Checkpoints are automatically saved during training:

```python
# Automatic saves at intervals
config = StageConfig(
    checkpoint_interval=5000,  # Every 5000 steps
)

# Manual save
trainer.save_checkpoint(name="my_checkpoint")

# Load and resume
trainer.load_checkpoint("checkpoints/my_checkpoint.thalia")
```

Checkpoint locations:
```
checkpoints/
├── stage_-1_step_5000.thalia
├── stage_-1_step_10000.thalia
├── stage_-1_final.thalia
├── stage_0_step_5000.thalia
└── ...
```

## Monitoring & Debugging

### Add Callbacks

```python
def my_callback(step: int, metrics: dict):
    """Called every 1000 steps."""
    print(f"Step {step}: {metrics}")

trainer.add_callback(my_callback)
```

### Check Training History

```python
# After training
for result in trainer.training_history:
    print(f"Stage {result.stage.name}:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.training_time_seconds:.1f}s")
    print(f"  Milestones: {result.milestone_results}")
```

### Health Checks

```python
from thalia.training import check_system_health

health = check_system_health(brain)

if not all(health.values()):
    failed = [k for k, v in health.items() if not v]
    print(f"⚠️  Health issues: {failed}")
```

## Common Issues & Solutions

### Issue: Stage milestones not met

**Solution**: Extend training duration
```python
trainer.extend_stage(
    stage=CurriculumStage.SENSORIMOTOR,
    additional_steps=20000,
    reason="Milestones not met",
)
```

### Issue: Catastrophic forgetting

**Solution**: Increase review proportion
```python
config = StageConfig(
    review_stages={
        0: 0.20,  # Increase to 20%
    },
)
```

### Issue: Growth not triggering

**Solution**: Check capacity thresholds
```python
from thalia.config.curriculum_growth import get_curriculum_growth_config

# Lower thresholds for earlier growth
config = get_curriculum_growth_config(conservative=False)
```

### Issue: Silent regions

**Solution**: Check initialization and input strength
```python
# Increase input strength
brain.set_input_gain(2.0)

# Lower neuron thresholds
brain.set_threshold_multiplier(0.8)
```

## Next Steps

1. **Read the full strategy**: [`curriculum_strategy.md`](design/curriculum_strategy.md)
2. **Review implementation plan**: [`curriculum_implementation.md`](design/curriculum_implementation.md)
3. **Run tests**: `pytest tests/integration/test_curriculum*.py`

## Getting Help

- **Implementation questions**: See [`curriculum_implementation.md`](design/curriculum_implementation.md)
- **Architecture questions**: See [`architecture.md`](design/architecture.md)
- **Checkpoint issues**: See [`checkpoint_format.md`](design/checkpoint_format.md)
- **Component patterns**: See [`patterns/component-parity.md`](patterns/component-parity.md)

## References

- Curriculum strategy: `docs/design/curriculum_strategy.md`
- Checkpoint format: `docs/design/checkpoint_format.md`
