# Copilot Instructions for Thalia

This file provides context for AI assistants working with the Thalia codebase.

## Project Overview

**Thalia** is a biologically-accurate spiking neural network framework for building multi-modal, biologically-plausible ML models with LLM-level (or better) capabilities.

**Architecture Philosophy**:
- **Not**: Traditional deep learning with backpropagation
- **Is**: Neuroscience-inspired spiking networks with local learning rules and neuromodulation
- **Goal**: Match or exceed LLM capabilities using biologically-plausible mechanisms

## Architecture Principles

### 1. Brain Regions AND Pathways are Specialized
Each region has its own learning rule:
- **Striatum**: Three-factor rule (eligibility × dopamine) for RL
- **Hippocampus**: One-shot Hebbian for episodic memory
- **Cortex**: Unsupervised Hebbian/BCM/STDP for features
- **Cerebellum**: Supervised error-corrective (delta rule)
- **Prefrontal**: Gated Hebbian for working memory

### 2. All Processing is Spike-Based
- Use binary spikes (0 or 1), not firing rates
- ConductanceLIF neurons (primary neuron model) with membrane dynamics
- Temporal dynamics matter (spike timing, delays, traces)
- Note: Simple LIF is not implemented - use ConductanceLIF for all regions

### 3. Neuromodulation is Key
- **Dopamine**: Gates learning in striatum and prefrontal
- **Acetylcholine**: Modulates encoding/retrieval in hippocampus
- **Norepinephrine**: Arousal and gain modulation
- Set via `region.set_dopamine(level)`, not passed every forward()

## Code Patterns

### Weight Initialization
```python
# Always use WeightInitializer registry
weights = WeightInitializer.gaussian(n_output, n_input, mean=0.3, std=0.1, device=device)
weights = WeightInitializer.xavier(n_output, n_input, device=device)
weights = WeightInitializer.sparse_random(n_output, n_input, sparsity=0.2, device=device)

# Never: torch.randn() or torch.rand() directly
```

### Device Management
```python
# Pattern 1 (preferred): Specify device at creation
tensor = torch.zeros(size, device=device)

# Pattern 2 (only for nn.Module): Move after creation
module.to(device)

# Pattern 2 (only for external data): Move to correct device
input_data = batch["input"].to(self.device)
```

### Learning Strategies
```python
# Use the learning strategy pattern (NOT manual learning code)
from thalia.learning import create_strategy

# Create strategy for region
strategy = create_strategy(
    "three_factor",  # or "hebbian", "stdp", "bcm"
    learning_rate=0.001,
    eligibility_tau_ms=100.0,
)

# Apply during forward pass
new_weights, metrics = strategy.compute_update(
    weights=self.weights,
    pre_spikes=pre_spikes,
    post_spikes=post_spikes,
    modulator=dopamine_level,  # For three-factor rule
)
self.weights.data = new_weights

# Note: Method is compute_update(), not apply()
# Note: Factory is create_strategy(), not create_learning_strategy()
```

## Common Imports

```python
# Brain and configuration
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

# Learning strategies
from thalia.learning import create_strategy

# Curriculum training
from thalia.config.curriculum_growth import CurriculumStage, get_curriculum_growth_config
from thalia.training.curriculum.stage_manager import CurriculumTrainer, StageConfig

# Datasets
from thalia.datasets import (
    create_stage0_temporal_dataset,
    create_stage1_cifar_datasets,
    create_stage2_grammar_dataset,
    create_stage3_reading_dataset,
    GrammarLanguage,
    ReadingLanguage,
)

# Diagnostics
from thalia.diagnostics import HealthMonitor, CriticalityMonitor, MetacognitiveMonitor
from thalia.training.visualization import TrainingMonitor
```

## Biological Accuracy Constraints

### DO:
- Use spike-based processing (binary spikes)
- Implement local learning rules (no backprop)
- Respect biological time constants (tau_mem ~10-30ms)
- Use neuromodulators for gating/modulation
- Maintain causality (no future information)

### DON'T:
- Use global error signals or backpropagation
- Accumulate firing rates instead of individual spikes
- Implement non-local learning rules
- Use negative firing rates
- Access future timesteps in current computation

## Implemented Features (December 2025)

### Core Systems ✅
- **Brain Regions**: Cortex (laminar L4→L2/3→L5), Hippocampus (DG→CA3→CA1), Striatum (D1/D2 pathways), PFC, Cerebellum, Thalamus
- **Neurons**: ConductanceLIF (conductance-based, voltage-dependent currents)
- **Learning Rules**: STDP, BCM, Hebbian, three-factor (dopamine-gated), error-corrective
- **Neuromodulators**: Dopamine, acetylcholine, norepinephrine (centralized management)
- **Oscillators**: Theta (8Hz), alpha (10Hz), gamma (40Hz) coordination

### Planning & Memory ✅
- **TD(λ)**: Multi-step credit assignment (`src/thalia/regions/striatum/td_lambda.py`)
- **Dyna Planning**: Model-based planning (`src/thalia/planning/dyna.py`)
- **Goal Hierarchy**: Hierarchical goal management (`src/thalia/regions/prefrontal_hierarchy.py`)
- **Working Memory**: PFC gating and maintenance
- **Episodic Memory**: Hippocampal one-shot learning

### Training & Infrastructure ✅
- **Curriculum Training**: Stage-based developmental training (`src/thalia/training/curriculum/`)
- **Checkpoints**: PyTorch format (primary) + binary format (optional)
- **Parallel Execution**: Multi-core CPU support (`src/thalia/events/parallel.py`)
- **Diagnostics**: Health monitor, training monitor, criticality monitor
- **Datasets**: Temporal sequences, CIFAR-10, Grammar (3 languages), Reading (3 languages)

## Key Documentation

### Primary Documentation
- **Architecture Overview**: `docs/architecture/ARCHITECTURE_OVERVIEW.md`
- **Learning Strategies**: `docs/patterns/learning-strategies.md`
- **Component Parity**: `docs/patterns/component-parity.md`
- **State Management**: `docs/patterns/state-management.md`
- **Mixins**: `docs/patterns/mixins.md`

### Implementation Status
- **Delayed Gratification**: `docs/design/delayed_gratification.md` (Phases 1-3 complete)
- **Circuit Modeling**: `docs/design/circuit_modeling.md` (D1/D2 delays implemented)
- **Curriculum Strategy**: `docs/design/curriculum_strategy.md` (Expert-reviewed)

### Quick References
- **Curriculum Training**: `docs/CURRICULUM_QUICK_REFERENCE.md`
- **Datasets**: `docs/DATASETS_QUICK_REFERENCE.md`
- **Monitoring**: `docs/MONITORING_GUIDE.md`
