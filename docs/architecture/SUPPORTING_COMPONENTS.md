# Supporting Components Architecture

**Last Updated**: December 13, 2025

## Overview

This document covers supporting components that enable Thalia's core functionality but don't fit into the main regions/pathways/centralized systems categories.

## Component Categories

1. **Managers** - Standardized component encapsulation
2. **Decision Making** - Action selection utilities
3. **Environments** - Task environment wrappers
4. **Diagnostics** - System monitoring and debugging
5. **I/O** - Checkpoint and data management
6. **Training** - Training loop infrastructure

---

## 1. Managers System

**Location**: `src/thalia/managers/`

### Purpose

Managers encapsulate specific responsibilities within brain regions (learning, homeostasis, exploration) using a standardized initialization pattern.

### BaseManager Pattern

```python
from thalia.managers.base_manager import BaseManager, ManagerContext

@dataclass
class ManagerContext:
    """Shared context for manager initialization."""
    device: torch.device
    n_input: int
    n_output: int
    dt_ms: float

class BaseManager(Generic[ConfigT], ABC):
    """Base class for all managers."""
    def __init__(self, config: ConfigT, context: ManagerContext):
        self.config = config
        self.context = context

    @abstractmethod
    def reset_state(self) -> None:
        """Reset manager state at trial boundaries."""
        pass
```

### Benefits

- **Consistency** - All managers follow same pattern
- **Testability** - Can be tested in isolation
- **Composability** - Easy to combine in regions
- **Extensibility** - Add new managers without breaking existing code

### Component Registry

**Location**: `src/thalia/managers/component_registry.py`

Dynamic component registration and discovery:
```python
# Register a component
@component_registry.register("my_component")
class MyComponent:
    pass

# Retrieve component
ComponentClass = component_registry.get("my_component")
```

### Usage in Regions

Regions compose managers for specific responsibilities:
```python
class Hippocampus(BaseRegion):
    def __init__(self, config, context):
        super().__init__(config, context)

        # Compose managers
        self.learning_manager = LearningManager(
            config.learning,
            ManagerContext(device=self.device, n_input=..., n_output=...)
        )
        self.exploration_manager = ExplorationManager(...)
```

---

## 2. Decision Making

**Location**: `src/thalia/decision_making/`

### ActionSelector

**File**: `action_selection.py`

Standalone utilities for converting neural activity to discrete actions.

#### Features

1. **Selection Strategies**:
   - Softmax (temperature-based probabilistic)
   - Greedy (highest value)
   - Epsilon-greedy (ε-exploration)
   - UCB (Upper Confidence Bound)

2. **Population Coding**:
   - Multiple neurons per action
   - Distributed representation
   - Noise robustness

3. **Vote Accumulation**:
   - Accumulate votes over timesteps
   - Optional decay
   - Temporal integration

4. **UCB Exploration**:
   - Track action counts
   - Compute exploration bonuses
   - Balance exploration/exploitation

#### Configuration

```python
@dataclass
class ActionSelectionConfig:
    mode: SelectionMode = SelectionMode.SOFTMAX
    temperature: float = 1.0
    epsilon: float = 0.1
    ucb_c: float = 2.0
    neurons_per_action: int = 1
    accumulate_votes: bool = False
    vote_decay: float = 0.0
```

#### Usage

```python
from thalia.decision_making.action_selection import ActionSelector, ActionSelectionConfig

selector = ActionSelector(ActionSelectionConfig(
    mode=SelectionMode.SOFTMAX,
    temperature=1.0,
    neurons_per_action=10
))

# Select action from spikes
action, confidence = selector.select_action(
    spikes=striatum_spikes,  # [n_neurons] binary
    n_actions=4
)

# Update exploration statistics (for UCB)
selector.update_action_count(action, success=True)
```

#### Integration Points

- **Striatum**: Primary action selection
- **PFC**: Goal-directed action selection
- **Motor Cortex**: Motor command generation

---

## 3. Environments

**Location**: `src/thalia/environments/`

### SensorimotorWrapper

**File**: `sensorimotor_wrapper.py`

Wraps Gymnasium/MuJoCo environments for spiking neural networks.

#### Purpose

Enables **Stage -0.5 (Sensorimotor Grounding)** training:
- Motor babbling (exploration)
- Reaching tasks (goal-directed)
- Forward/inverse models (cerebellum)

#### Key Features

**1. Proprioception Encoding**
- Joint angles/velocities → spike patterns
- Rate coding (higher velocity → higher rate)
- Population coding (distributed representation)
- Realistic sensory noise

**2. Motor Decoding**
- Spike patterns → motor commands (torques)
- Population vector decoding
- Smoothing for realistic control
- Action bounds enforcement

**3. Task Support**
- Motor babbling
- Reaching to targets
- Object manipulation
- Curriculum integration

#### Supported Environments

- **Reacher-v4** (PRIMARY) - 2-joint arm reaching
- **Pusher-v4** - 3-DOF manipulation
- **HalfCheetah-v4** - Locomotion (stretch goal)

#### Configuration

```python
@dataclass
class SensorimotorConfig:
    env_name: str = "Reacher-v4"

    # Encoding
    n_proprioceptive_neurons: int = 64
    encoding_type: str = "rate"  # "rate" or "population"
    max_firing_rate: float = 100.0  # Hz

    # Decoding
    motor_smoothing: float = 0.3
    action_bound_enforcement: bool = True

    # Task
    task_type: str = "babbling"  # "babbling", "reaching"
    difficulty: float = 1.0
```

#### Usage

```python
from thalia.environments.sensorimotor_wrapper import SensorimotorWrapper

env = SensorimotorWrapper(SensorimotorConfig(
    env_name="Reacher-v4",
    task_type="reaching"
))

# Reset
obs, info = env.reset()
proprioceptive_spikes = env.encode_observation(obs)

# Step
motor_spikes = cerebellum.forward(...)
action = env.decode_motor_spikes(motor_spikes)
next_obs, reward, done, truncated, info = env.step(action)
```

#### Integration

- **Cerebellum**: Forward/inverse models
- **Motor Cortex**: Motor command generation
- **Somatosensory Cortex**: Proprioceptive processing
- **Curriculum**: Progressive task difficulty

---

## 4. Diagnostics

**Location**: `src/thalia/core/diagnostics.py`

### BrainSystemDiagnostics

Real-time monitoring and debugging for the brain system.

#### Features

1. **Health Checks**:
   - NaN detection in weights/activations
   - Silent neurons (never spike)
   - Overactive neurons (always spike)
   - Weight explosion/vanishing

2. **Activity Monitoring**:
   - Firing rates per region
   - Spike counts over time
   - Population statistics

3. **Learning Progress**:
   - Weight change tracking
   - Synaptic efficacy
   - Learning rate adaptation

4. **Performance Metrics**:
   - Forward pass timing
   - Memory usage
   - Throughput (samples/sec)

#### Usage

```python
from thalia.core.diagnostics import BrainSystemDiagnostics
from thalia.core.dynamic_brain import DynamicBrain, BrainBuilder

diagnostics = BrainSystemDiagnostics()

# Attach to brain
brain = BrainBuilder.preset("sensorimotor", global_config)
diagnostics.attach(brain)

# Run diagnostics
health_report = diagnostics.check_health()
activity_stats = diagnostics.get_activity_stats()

# Detect issues
if health_report.has_silent_neurons:
    print(f"Silent neurons in: {health_report.silent_regions}")
```

---

## 5. I/O System

**Location**: `src/thalia/io/`

### Checkpoint Management

Save and restore brain state for:
- Training checkpoints
- Experiment versioning
- Ablation studies
- Deployment

#### Format

**See**: `../design/checkpoint_format.md`

Binary format with:
- Version info
- Config snapshot
- Region states (weights, traces, etc.)
- Optimizer states
- Metadata

#### Usage

```python
from thalia.io import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    brain=brain,
    optimizer=optimizer,
    epoch=100,
    path="checkpoints/brain_epoch_100.ckpt"
)

# Load
checkpoint = load_checkpoint("checkpoints/brain_epoch_100.ckpt")
brain.load_state_dict(checkpoint['brain_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
```

---

## 6. Training Infrastructure

**Location**: `src/thalia/training/`

### Training Loop

Handles:
- Curriculum stage progression
- Batch processing
- Logging and metrics
- Early stopping
- Learning rate scheduling

#### Curriculum Integration

**See**: `../design/curriculum_strategy.md`

Stages:
- **Stage -0.5**: Sensorimotor grounding
- **Stage 0**: Sensory processing
- **Stage 1**: Episodic memory
- **Stage 2**: Working memory
- **Stage 3**: Action selection
- **Stage 4**: Planning and language

#### Usage

```python
from thalia.training import TrainingLoop, TrainingConfig

loop = TrainingLoop(
    brain=brain,
    config=TrainingConfig(
        curriculum_stage="stage_0",
        max_epochs=1000,
        early_stopping_patience=50
    )
)

# Train
loop.run()
```

---

## Design Philosophy

### Separation of Concerns

Each component has a **single, clear responsibility**:
- Managers: Component encapsulation
- Decision Making: Action selection logic
- Environments: Task interface
- Diagnostics: Monitoring/debugging
- I/O: Persistence
- Training: Training loop coordination

### Composability

Components can be **mixed and matched**:
```python
# Different action selectors
striatum_selector = ActionSelector(softmax_config)
pfc_selector = ActionSelector(greedy_config)

# Different environments
motor_env = SensorimotorWrapper("Reacher-v4")
manipulation_env = SensorimotorWrapper("Pusher-v4")
```

### Extensibility

New components can be added **without modifying existing code**:
```python
# Register new component
@component_registry.register("my_new_component")
class MyNewComponent(BaseManager):
    pass
```

---

## Integration with Core Systems

### Brain Regions

Regions use these components:
```python
class Striatum(BaseRegion):
    def __init__(self, config, context):
        # Use action selector
        self.action_selector = ActionSelector(config.action_selection)

        # Use learning manager
        self.learning_manager = LearningManager(config.learning, context)

        # Use diagnostics
        self.diagnostics = RegionDiagnostics()
```

### Trial Coordinator

Coordinator uses:
- Action selectors for decision making
- Diagnostics for monitoring
- Checkpointing for state management

### Training Loop

Training uses:
- Environments for tasks
- Diagnostics for progress tracking
- Checkpointing for resumption

---

## Testing

Each component has unit tests:
```
tests/unit/
├── managers/ - Manager base class and registry
├── decision_making/ - Action selection
├── environments/ - Environment wrappers
├── diagnostics/ - Health checks and monitoring
└── io/ - Checkpoint save/load
```

Run tests:
```bash
pytest tests/unit/managers/
pytest tests/unit/decision_making/
pytest tests/unit/environments/
```

---

## Future Enhancements

### Short-Term

1. **Enhanced Diagnostics** - Real-time visualization
2. **More Environments** - Text, vision tasks
3. **Advanced Action Selection** - Hierarchical selection, options

### Long-Term

1. **Distributed Training** - Multi-GPU/node support
2. **Online Learning** - Continual learning infrastructure
3. **Meta-Learning** - Learning-to-learn components

---

## References

- **Managers**: `src/thalia/managers/base_manager.py` (docstrings)
- **Action Selection**: `src/thalia/decision_making/action_selection.py` (docstrings)
- **Environments**: `src/thalia/environments/sensorimotor_wrapper.py` (docstrings)
- **Diagnostics**: `src/thalia/core/diagnostics.py` (docstrings)
- **Checkpoint Format**: `../design/checkpoint_format.md`
- **Curriculum**: `../design/curriculum_strategy.md`
