# Copilot Instructions for Thalia

This file provides context for AI assistants working with the Thalia codebase.

## Navigation & Documentation

**For detailed navigation guidance, search strategies, and common tasks, see:**
- `docs/AI_ASSISTANT_GUIDE.md` - Comprehensive navigation guide with search patterns
- `docs/COPILOT_AVAILABLE_TOOLS.md` - Complete list of available tools with descriptions
- `docs/architecture/ARCHITECTURE_OVERVIEW.md` - System architecture overview (start here for architecture questions)

**Important: Auto-Generated Documentation**
- All files in `docs/api/` are **auto-generated** - never edit manually!
- To update API docs, run the VSCode task: "Generate API Documentation"
- Or run directly: `python scripts/generate_api_docs.py`

## Python Environment Setup

**Configure the Python environment before running any Python code, tests, or commands.**

Use the `configure_python_environment` tool at the start of every session or before running:
- Python scripts (training, examples, etc.)
- Tests (`pytest`, `runTests`)
- Python terminal commands
- Package installations

This ensures the correct environment is activated with all dependencies (pytest, torch, etc.) available. If you see import errors or missing packages that should be installed, you likely forgot this step.

## Project Overview

**Thalia** is a biologically-accurate spiking neural network framework for building multi-modal, biologically-plausible ML models with LLM-level (or better) capabilities.

**Architecture Philosophy**:
- **Not**: Traditional deep learning with backpropagation
- **Is**: Neuroscience-inspired spiking networks with local learning rules and neuromodulation
- **Goal**: Match or exceed LLM capabilities using biologically-plausible mechanisms

**Key Components**:
- **NeuralRegion**: Base class (nn.Module + 4 mixins) for brain regions
- **Learning Strategies**: Pluggable learning rules (STDP, BCM, Hebbian, Three-factor, etc.)
- **AxonalProjection**: Pure spike routing with delays (NO weights)
- **Synaptic Weights**: Stored at target dendrites in `region.synaptic_weights` dict

## Architecture Principles

### 1. Brain Regions AND Pathways are Specialized
Each region uses pluggable learning strategies:
- **Striatum**: `create_striatum_strategy()` → Three-factor (eligibility × dopamine)
- **Hippocampus**: `create_hippocampus_strategy()` → STDP (one-shot capable)
- **Cortex**: `create_cortex_strategy()` → STDP + BCM (composite)
- **Cerebellum**: `create_cerebellum_strategy()` → Error-corrective (delta rule)
- **Prefrontal**: Gated Hebbian (custom implementation)

### 2. All Processing is Spike-Based
- Use binary spikes (0 or 1), not firing rates
- ConductanceLIF neurons (ONLY neuron model) with conductance-based dynamics
- Temporal dynamics matter (spike timing, delays, traces)

### 3. Synapses at Target Dendrites
- **Axons** (AxonalProjection): Pure spike routing with CircularDelayBuffer for axonal delays
- **Synapses**: Weights stored at target regions in `synaptic_weights` dict
- **Learning**: Region-specific, per-source customization
- Input to regions: `Dict[str, torch.Tensor]` (multi-source spike inputs)

### 4. Neuromodulation is Key
- **Dopamine**: Gates learning in striatum and prefrontal
- **Acetylcholine**: Modulates encoding/retrieval in hippocampus
- **Norepinephrine**: Arousal and gain modulation
- Set via `region.set_neuromodulators(dopamine=level)`, not passed every forward()

## Code Patterns

### Neuron Creation
```python
# Use NeuronFactory registry for dynamic creation
from thalia.components.neurons import NeuronFactory

neurons = NeuronFactory.create("pyramidal", n_neurons=100, device=device)
relay = NeuronFactory.create("relay", n_neurons=64, device=device)
l23 = NeuronFactory.create("cortical_layer", n_neurons=256, device=device, layer="L2/3")

# Available types: pyramidal, relay, trn, cortical_layer
types = NeuronFactory.list_types()

# Direct function calls also work (traditional approach)
from thalia.components.neurons import create_pyramidal_neurons
neurons = create_pyramidal_neurons(n_neurons=100, device=device)
```

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
```

### NeuralRegion Forward Pattern
```python
# Regions receive Dict[str, Tensor] from multiple axonal sources
class MyRegion(NeuralRegion):
    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process multi-source input.

        Args:
            source_spikes: {"thalamus": spikes, "cortex:l5": spikes}

        Returns:
            Output spikes from neurons
        """
        # Synaptic integration (weights stored in self.synaptic_weights)
        total_current = torch.zeros(self.n_neurons, device=self.device)
        for source_name, spikes in source_spikes.items():
            weights = self.synaptic_weights[source_name]  # [n_neurons, n_input]
            total_current += weights @ spikes

        # Neuron dynamics
        output_spikes = self.neurons(total_current)

        # Learning (per source)
        for source_name, spikes in source_spikes.items():
            new_weights, _ = self.strategies[source_name].compute_update(
                weights=self.synaptic_weights[source_name],
                pre_spikes=spikes,
                post_spikes=output_spikes,
            )
            self.synaptic_weights[source_name].data = new_weights

        # Port-based routing (ADR-015): Set port outputs before return
        self.clear_port_outputs()
        self.set_port_output("default", output_spikes)

        return output_spikes
```

### Port-Based Routing (ADR-015)

All NeuralRegion subclasses support port-based routing for biologically-accurate connections:

```python
# Register ports in __init__ (declarative)
class LayeredCortex(NeuralRegion):
    def __init__(self, config: CortexConfig, ...):
        # ... initialization ...

        # Register output ports
        self.register_output_port("default", self.l23_size + self.l5_size)
        self.register_output_port("l23", self.l23_size)
        self.register_output_port("l5", self.l5_size)
        self.register_output_port("l6a", self.l6a_size)  # → TRN
        self.register_output_port("l6b", self.l6b_size)  # → Relay

    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process input and set port outputs."""
        self.clear_port_outputs()  # Clear previous outputs

        # ... layer processing ...
        # l4_spikes, l23_spikes, l5_spikes, l6a_spikes, l6b_spikes = ...

        # Set port outputs (runtime)
        self.set_port_output("l23", l23_spikes)
        self.set_port_output("l5", l5_spikes)
        self.set_port_output("l6a", l6a_spikes)
        self.set_port_output("l6b", l6b_spikes)
        self.set_port_output("default", torch.cat([l23_spikes, l5_spikes]))

        return self.get_port_output("default")

# Connect with specific ports in BrainBuilder
builder.connect(
    "cortex", "thalamus",
    source_port="l6a",      # L6a CT neurons → TRN
    target_port="trn",
    delay_ms=2.0
)
builder.connect(
    "cortex", "thalamus",
    source_port="l6b",      # L6b CT neurons → Relay
    target_port="relay",
    delay_ms=2.0
)
```

**Port Registration Pattern (REQUIRED for all regions):**
1. `__init__`: Call `self.register_output_port("default", size)` before `self.to(device)`
2. `forward()`: Call `self.clear_port_outputs()` at start, then `self.set_port_output("default", output)` before return

**Current Port Support:**
- **LayeredCortex**: default, l23, l5, l6a, l6b (5 ports)
- **All Other Regions**: default port (backward compatible)

**Biological Rationale:**
- L6a CT neurons → TRN (burst firing, lateral inhibition)
- L6b CT neurons → Relay (regular spiking, feedback modulation)
- Different cell types project to different targets (no output concatenation)

**Implementation Details:**
- Ports stored in `_port_outputs: Dict[str, torch.Tensor]`
- AxonalProjection extracts port outputs via `region.get_port_output(port)`
- BrainBuilder passes `source_port` parameter to connections
- DynamicBrain passes region objects (not tensors) to pathways

## Common Imports

```python
# Brain and configuration
from thalia.core.dynamic_brain import DynamicBrain, BrainBuilder
from thalia.config import BrainConfig

# Learning strategies
from thalia.learning import create_strategy, create_cortex_strategy, create_striatum_strategy

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
    PhonologicalDataset,
    Language,  # For phonology
)

# Diagnostics
from thalia.diagnostics import HealthMonitor, CriticalityMonitor, MetacognitiveMonitor
from thalia.diagnostics.health_monitor import HealthReport  # Dataclass, not dict
from thalia.training.visualization import TrainingMonitor

# Neuron models
from thalia.components.neurons import (
    NeuronFactory,
    ConductanceLIF,
    ConductanceLIFConfig,
    create_pyramidal_neurons,
    create_relay_neurons,
)

# Synaptic components
from thalia.components.synapses import WeightInitializer, ShortTermPlasticity

# Axonal projections and delays
from thalia.pathways.axonal_projection import AxonalProjection, SourceSpec
from thalia.utils.delay_buffer import CircularDelayBuffer

# Creating a brain
brain = BrainBuilder.preset("default", brain_config)  # Use presets

# Accessing components
region = brain.components["cortex"]                     # Get region by name
pathway = brain.connections[("thalamus", "cortex")]    # Get pathway (AxonalProjection)
all_regions = brain.components                          # Dict of all components
```

## Finding Code (PowerShell)

```powershell
# Find component registrations
Select-String -Path src\* -Pattern "@register_region" -Recurse
Select-String -Path src\* -Pattern "@register_pathway" -Recurse

# Find growth implementations
Select-String -Path src\* -Pattern "def grow_output" -Recurse
Select-String -Path src\* -Pattern "def grow_input" -Recurse

# Find learning strategies
Select-String -Path src\* -Pattern "create_strategy" -Recurse

# Find axonal projection and delay logic
Select-String -Path src\* -Pattern "AxonalProjection" -Recurse
Select-String -Path src\* -Pattern "CircularDelayBuffer" -Recurse
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

## Brain Architecture

**DynamicBrain** is the sole brain implementation (as of December 2025):
- Flexible architecture built with `BrainBuilder`
- Component-based: regions and pathways are independent components
- Access pattern: `brain.components["region_name"]` for all regions
- No direct attributes: `brain.cortex` is NOT supported

## Implemented Features (December 2025)

### Core Systems ✅
- **Brain Regions**: Cortex (laminar L4→L2/3→L5), Hippocampus (DG→CA3→CA1), Striatum (D1/D2 pathways), PFC, Cerebellum, Thalamus
- **Neurons**: ConductanceLIF (conductance-based, voltage-dependent currents) - ONLY neuron model
- **Learning Rules**: STDP, BCM, Hebbian, three-factor (dopamine-gated), error-corrective
- **Neuromodulators**: Dopamine, acetylcholine, norepinephrine (centralized management via NeuromodulatorManager)
- **Oscillators**: Delta, theta, alpha, beta, gamma with cross-frequency coupling

### Training & Infrastructure ✅
- **Curriculum Training**: Stage-based developmental training (`src/thalia/training/curriculum/`)
- **Checkpoints**: PyTorch format (primary) + binary format (optional)
- **Clock-Driven Execution**: Fixed timestep simulation with axonal delays in pathways
- **Diagnostics**: Health monitor, training monitor, criticality monitor
- **Datasets**: Temporal sequences, Phonology (3 languages), CIFAR-10, Grammar (3 languages), Reading (3 languages)

## Type Alias Glossary

> **For complete type alias documentation, see `docs/api/TYPE_ALIASES.md`**

**Quick reference** (most commonly used types):

```python
# Component Organization
ComponentGraph = Dict[str, NeuralRegion]           # name -> component instance
ConnectionGraph = Dict[Tuple[str, str], NeuralRegion]  # (src, tgt) -> pathway
TopologyGraph = Dict[str, List[str]]                  # src -> [tgt1, tgt2, ...]

# Multi-Source Pathways
SourceSpec = Tuple[str, Optional[str]]                # (region_name, port)
SourceOutputs = Dict[str, torch.Tensor]               # {region_name: output_spikes}
SynapticWeights = Dict[str, torch.Tensor]             # {"source_name": weight_matrix}
LearningStrategies = Dict[str, LearningStrategy]      # {"source_name": strategy}

# Port-Based Routing
SourcePort = Optional[str]                            # 'l23', 'l5', 'l4', None
TargetPort = Optional[str]                            # 'feedforward', 'top_down', None

# State Management
StateDict = Dict[str, torch.Tensor]                   # Component state for checkpointing
CheckpointMetadata = Dict[str, Any]                   # Training progress, stage info
```

## Standard Growth API

All `NeuralRegion` subclasses implement these standardized signatures:

```python
def grow_output(self, n_new: int) -> None:
    """Grow output dimension by adding neurons.

    Effects: Expands weights (adds rows), adds neurons, updates config.n_output
    """

def grow_input(self, n_new: int) -> None:
    """Grow input dimension to accept more inputs.

    Effects: Expands weights (adds columns), NO new neurons, updates config.n_input
    """

def grow_source(self, source_name: str, new_size: int) -> None:
    """Grow input for specific source (MultiSourcePathway only).

    Effects: Updates input_sizes[source_name], resizes weights, preserves other sources
    """
```

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
