# Architecture Guide

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-26 13:45:45
> Generated from: `scripts/generate_api_docs.py`

This guide provides architectural diagrams and design patterns for the Thalia framework.

## 🏗️ System Architecture Overview

```mermaid
graph TB
    subgraph User["User Layer"]
        Script["Training Script"]
    end

    subgraph API["High-Level API"]
        Builder["BrainBuilder"]
        Trainer["CurriculumTrainer"]
        Datasets["Dataset Factories"]
    end

    subgraph Brain["Brain Layer"]
        DB["DynamicBrain"]
        Registry["ComponentRegistry"]
        Regions["Neural Regions"]
        Pathways["Axonal Pathways"]
    end

    subgraph Components["Component Layer"]
        Neurons["Neuron Models"]
        Synapses["Synaptic Weights"]
        Learning["Learning Strategies"]
        Neuromod["Neuromodulators"]
    end

    subgraph Support["Support Systems"]
        Config["Configuration"]
        Diagnostics["Diagnostics"]
        Checkpoints["Checkpointing"]
    end

    Script --> Builder
    Script --> Trainer
    Script --> Datasets
    Builder --> DB
    Trainer --> DB
    DB --> Registry
    DB --> Regions
    DB --> Pathways
    Regions --> Neurons
    Regions --> Synapses
    Regions --> Learning
    Regions --> Neuromod
    Pathways --> Neurons
    Trainer --> Diagnostics
    DB --> Config
    DB --> Checkpoints
```

## 📊 Data Flow Architecture

```mermaid
graph LR
    Input["Input Data"]
    Encoding["Spike Encoding"]
    Thalamus["Thalamus<br/>(relay)"]  
    Cortex["Cortex<br/>(processing)"]
    Hippo["Hippocampus<br/>(memory)"]
    Striatum["Striatum<br/>(action)"]
    Output["Output Spikes"]
    Dopamine["Dopamine<br/>(reward)"]  

    Input --> Encoding
    Encoding --> Thalamus
    Thalamus --> Cortex
    Cortex --> Hippo
    Cortex --> Striatum
    Hippo --> Cortex
    Striatum --> Output
    Dopamine -.->|modulates| Striatum
    Dopamine -.->|modulates| Cortex
```

## 🧩 Component Composition Pattern

```mermaid
classDiagram
    class NeuralRegion {
        +forward()
        +reset_state()
        +get_state()
    }

    class NeuromodulatorMixin {
        +set_neuromodulators()
        +decay_neuromodulators()
    }

    class GrowthMixin {
        +grow_output()
        +grow_input()
    }

    class ResettableMixin {
        +reset_state()
    }

    class DiagnosticsMixin {
        +collect_diagnostics()
    }

    class ConductanceLIF {
        +forward()
        -update_voltage()
    }

    class LearningStrategy {
        +compute_update()
    }

    NeuralRegion <|-- LayeredCortex
    NeuralRegion <|-- Hippocampus
    NeuralRegion <|-- Striatum
    NeuromodulatorMixin <|-- LayeredCortex
    GrowthMixin <|-- LayeredCortex
    ResettableMixin <|-- LayeredCortex
    DiagnosticsMixin <|-- LayeredCortex
    LayeredCortex *-- ConductanceLIF
    LayeredCortex *-- LearningStrategy
```

## ⏰ Brain Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Configuration
    Configuration --> Construction: BrainBuilder.build()
    Construction --> Initialization: initialize components
    Initialization --> Training: start training loop
    Training --> Forward: process batch
    Forward --> Learning: update weights
    Learning --> Diagnostics: check health
    Diagnostics --> Training: next batch
    Training --> Checkpoint: save progress
    Checkpoint --> Training: resume
    Training --> Evaluation: epoch end
    Evaluation --> Training: continue
    Evaluation --> [*]: training complete
```

## ⚡ Spike Processing Pipeline

```mermaid
sequenceDiagram
    participant D as Dataset
    participant R as Region
    participant N as Neurons
    participant L as Learning
    participant NM as Neuromodulators

    D->>R: Input spikes (batch)
    R->>R: Synaptic integration
    R->>N: Synaptic currents
    N->>N: Update membrane voltage
    N->>N: Check threshold
    N-->>R: Output spikes
    R->>L: Pre & post spikes
    NM-->>L: Modulator levels
    L->>L: Compute weight update
    L-->>R: Updated weights
    R-->>D: Output for next region
```

## 🎓 Learning Strategy Selection

```mermaid
graph TD
    Start[Choose Learning Strategy]
    Cortical{Cortical<br/>Region?}
    Reward{Reward-based<br/>Learning?}
    Memory{One-shot<br/>Memory?}
    Motor{Motor<br/>Learning?}

    Start --> Cortical
    Cortical -->|Yes| STDP_BCM[create_cortex_strategy<br/>STDP + BCM]
    Cortical -->|No| Reward
    Reward -->|Yes| ThreeFactor[create_striatum_strategy<br/>Three-factor]
    Reward -->|No| Memory
    Memory -->|Yes| Hippocampal[create_hippocampus_strategy<br/>Fast STDP]
    Memory -->|No| Motor
    Motor -->|Yes| Error[create_cerebellum_strategy<br/>Error-corrective]
    Motor -->|No| Hebbian[Basic Hebbian]
```

## ⚙️ Configuration Hierarchy

```mermaid
graph TD
    Brain[BrainConfig<br/>Architecture]
    Regional[*RegionConfig<br/>Region-specific]
    Builder[BrainBuilder<br/>Size specification]

    Brain --> Regional
    Builder --> Regional
    Regional --> Cortex[LayeredCortexConfig]
    Regional --> Hippo[HippocampusConfig]
    Regional --> Stri[StriatumConfig]
```

**Note**: Region sizes are specified directly in BrainBuilder.add_component() calls.

## 💡 Architectural Best Practices

### Design Principles

1. **Biological Plausibility**: All designs follow neuroscience principles
2. **Local Learning**: No global error signals or backpropagation
3. **Spike-Based**: Binary spikes, not firing rates
4. **Modular Composition**: Regions are independent, composable units
5. **Mixins for Cross-Cutting**: Common functionality via mixins

### Component Guidelines

- **Regions**: Inherit from `NeuralRegion`, use standard mixins
- **Pathways**: Pure spike routing, no learning
- **Learning**: Implement `LearningStrategy` protocol
- **Configs**: Pure dataclasses, no logic
- **Neurons**: Only `ConductanceLIF` model used

### Growth Strategy

1. Start with small networks (64-256 neurons per region)
2. Train on Stage 0 (temporal sequences)
3. Grow network based on curriculum needs
4. Add new regions via `ComponentRegistry`
5. Use dynamic weight initialization

## 📚 Related Documentation

- [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md) - All available regions and pathways
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) - Module dependency structure
- [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md) - Learning rule selection
- [API_INDEX.md](API_INDEX.md) - Complete component index

