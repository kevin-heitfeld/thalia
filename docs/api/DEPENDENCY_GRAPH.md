# Dependency Graph

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-31 19:33:39
> Generated from: `scripts/generate_api_docs.py`

This document visualizes the dependency relationships between Thalia modules.

## 🔗 Component Dependencies

```mermaid
graph TD
    Core["Core (protocols, errors)"]
    Components["Components (neurons, synapses)"]
    Regions["Regions (cortex, hippocampus, etc.)"]
    Pathways["Pathways (axonal projection)"]
    Learning["Learning (strategies, registry)"]
    Brain["Brain (DynamicBrain)"]
    Training["Training (curriculum, monitors)"]
    Datasets["Datasets"]

    Core --> Components
    Core --> Learning
    Components --> Regions
    Components --> Pathways
    Learning --> Regions
    Regions --> Brain
    Pathways --> Brain
    Brain --> Training
    Datasets --> Training
    Learning --> Training
```

## 🧠 Region Dependencies

```mermaid
graph LR
    NeuralRegion["NeuralRegion (base)"]
    Mixins["Mixins"]
    Config["*Config"]
    Neurons["ConductanceLIF"]
    Strategy["LearningStrategy"]

    cerebellum["cerebellum"]
    NeuralRegion --> cerebellum
    Mixins --> cerebellum
    Config --> cerebellum
    Neurons --> cerebellum
    Strategy --> cerebellum
    multimodal_integration["multimodal_integration"]
    NeuralRegion --> multimodal_integration
    Mixins --> multimodal_integration
    Config --> multimodal_integration
    Neurons --> multimodal_integration
    Strategy --> multimodal_integration
    prefrontal["prefrontal"]
    NeuralRegion --> prefrontal
    Mixins --> prefrontal
    Config --> prefrontal
    Neurons --> prefrontal
    Strategy --> prefrontal
    thalamus["thalamus"]
    NeuralRegion --> thalamus
    Mixins --> thalamus
    Config --> thalamus
    Neurons --> thalamus
    Strategy --> thalamus
    cortex["cortex"]
    NeuralRegion --> cortex
    Mixins --> cortex
    Config --> cortex
    Neurons --> cortex
    Strategy --> cortex
    predictive_cortex["predictive_cortex"]
    NeuralRegion --> predictive_cortex
    Mixins --> predictive_cortex
    Config --> predictive_cortex
    Neurons --> predictive_cortex
    Strategy --> predictive_cortex
    More["... +2 more regions"]
    NeuralRegion --> More
```

## 📦 Module Import Layers

```mermaid
graph TB
    subgraph Layer1["Layer 1: Foundation"]
        L1A["core.protocols"]
        L1B["core.errors"]
        L1C["config"]
    end

    subgraph Layer2["Layer 2: Components"]
        L2A["components.neurons"]
        L2B["components.synapses"]
        L2C["neuromodulation"]
    end

    subgraph Layer3["Layer 3: Learning"]
        L3A["learning.rules"]
        L3B["learning.strategies"]
        L3C["learning.registry"]
    end

    subgraph Layer4["Layer 4: Regions & Pathways"]
        L4A["regions.*"]
        L4B["pathways.*"]
        L4C["mixins.*"]
    end

    subgraph Layer5["Layer 5: Brain"]
        L5A["core.dynamic_brain"]
        L5B["core.builder"]
    end

    subgraph Layer6["Layer 6: Training & Apps"]
        L6A["training.*"]
        L6B["datasets.*"]
        L6C["diagnostics.*"]
    end

    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer2 --> Layer4
    Layer3 --> Layer4
    Layer4 --> Layer5
    Layer5 --> Layer6
```

## 📋 Dependency Guidelines

### Import Rules

1. **Downward dependencies only**: Higher layers import from lower layers
2. **No circular imports**: Modules at the same layer should not import each other
3. **Core is foundation**: All modules can import from `core`
4. **Regions are independent**: Regions should not import from other regions

### Common Import Patterns

```python
# Layer 1 (Foundation)
from thalia.core.protocols import NeuralComponent
from thalia.core.errors import ConfigurationError
from thalia.config import ThaliaConfig

# Layer 2 (Components)
from thalia.components.neurons import ConductanceLIF
from thalia.components.synapses import WeightInitializer
from thalia.neuromodulation import NeuromodulatorManager

# Layer 3 (Learning)
from thalia.learning import create_strategy
from thalia.learning.rules import STDPRule

# Layer 4 (Regions)
from thalia.regions.cortex import LayeredCortex
from thalia.mixins import GrowthMixin

# Layer 5 (Brain)
from thalia.core.dynamic_brain import DynamicBrain
from thalia.core.builder import BrainBuilder

# Layer 6 (Training)
from thalia.training import CurriculumTrainer
from thalia.datasets import create_stage0_temporal_dataset
from thalia.diagnostics import HealthMonitor
```

## ⚠️ Avoiding Circular Dependencies

### Common Pitfalls

- **Region importing Brain**: Use dependency injection instead
- **Config importing Components**: Keep configs as pure data
- **Cross-region imports**: Use protocols/interfaces instead
- **Training importing specific regions**: Use registry pattern

### Solutions

1. **Protocols**: Define interfaces in `core.protocols`
2. **Registry**: Use `ComponentRegistry` for dynamic lookup
3. **Dependency Injection**: Pass dependencies through constructors
4. **Type hints**: Use string literals for forward references

