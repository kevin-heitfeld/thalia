# Thalia Architecture Overview

**Last Updated**: January 26, 2026

## Introduction

Thalia is a biologically-accurate spiking neural network framework implementing multiple brain regions with specialized learning rules. The architecture emphasizes biological plausibility through local learning rules, neuromodulation, spike-based processing, and clock-driven execution with axonal delays.

**Current Architecture**:
- **Regions**: Inherit from `NeuralRegion` (nn.Module + mixins)
- **Pathways**: Use `AxonalProjection` for pure spike routing
- **Learning**: Pluggable strategies via `LearningStrategy` pattern
- **Synapses**: Stored at target dendrites, NOT in axons

## Core Architecture Principles

### 1. Biological Plausibility
- **Spike-based processing**: Binary spikes (0 or 1), not firing rates
- **Local learning rules**: No backpropagation, pluggable strategies per region
- **Neuromodulation**: Dopamine, norepinephrine, acetylcholine gate learning
- **Temporal dynamics**: ConductanceLIF neurons (ONLY neuron model) with conductance-based dynamics
- **Causality**: No access to future information
- **Clock-driven execution**: Regular timesteps with axonal delays in pathways

### 2. Synapses at Target Dendrites
**Key Innovation**: Separation of axons (transmission) from synapses (integration)

- **Axons** (`AxonalProjection`): Pure spike routing with axonal delays
  - NO synaptic weights
  - NO learning rules
  - ONLY conduction delays via `CircularDelayBuffer`
  - Biologically accurate: 1-20ms delays

- **Synapses** (at target `NeuralRegion`):
  - Weights stored in `region.synaptic_weights` dict: `{source_name: weight_matrix}`
  - Per-source learning strategies
  - Region-specific plasticity rules
  - Inputs arrive as `Dict[str, torch.Tensor]` (multi-source)

**Benefits**:
- ✅ Matches neuroscience: synapses at postsynaptic dendrites
- ✅ Per-source customization: different learning rules for different inputs
- ✅ Natural multi-source integration: no complex pathway merging
- ✅ Clear separation of concerns: axons route, synapses learn

### 3. Regional Specialization
Each brain region uses specialized learning strategies from the pluggable LearningStrategy system:

| Region | Learning Strategy | Rule Type | Parameters |
|--------|------------------|-----------|------------|
| **Cortex** | `create_cortex_strategy()` | STDP + BCM | Composite: spike-timing + homeostatic |
| **Hippocampus** | `create_hippocampus_strategy()` | STDP | Asymmetric windows, one-shot capable |
| **Prefrontal** | Gated Hebbian | Custom | Working memory maintenance |
| **Striatum** | `create_striatum_strategy()` | Three-factor | Eligibility × dopamine |
| **Cerebellum** | `create_cerebellum_strategy()` | Error-corrective | Supervised delta rule |

### 4. Clock-Driven Computation
- Regions updated at regular timesteps (typically 1ms)
- Axonal projections implement delays via `CircularDelayBuffer`
- Natural temporal dynamics from spike propagation
- Biologically realistic timing (1ms timesteps, delays 1-20ms)

---

## System Architecture

### High-Level Structure

```
DynamicBrain (Component-Based Architecture)
├── Centralized Systems
│   ├── NeuromodulatorManager (DA, NE, ACh)
│   ├── OscillatorManager (5 rhythms + cross-frequency coupling)
│   ├── GoalHierarchyManager (hierarchical planning)
│   └── ConsolidationManager (memory replay & offline learning)
├── Brain Regions (brain.components["name"])
│   ├── NeuralRegion
│   │   ├── Inheritance: nn.Module + 7 mixins
│   │   │   ├── BrainComponentMixin (oscillator phase properties)
│   │   │   ├── NeuromodulatorMixin (DA/ACh/NE)
│   │   │   ├── GrowthMixin (dynamic expansion)
│   │   │   ├── ResettableMixin (state management)
│   │   │   ├── DiagnosticsMixin (health monitoring)
│   │   │   ├── StateLoadingMixin (checkpoint loading)
│   │   │   └── LearningStrategyMixin (pluggable learning rules)
│   │   ├── Synaptic Weights: Dict[source_name, Tensor]
│   │   ├── Learning Strategies: Dict[source_name, Strategy]
│   │   └── Forward: Dict[str, Tensor] → Tensor
│   ├── LayeredCortex (L4→L2/3→L5→L6a/L6b laminar microcircuit)
│   ├── PredictiveCortex (Predictive coding with error-driven learning)
│   ├── TrisynapticHippocampus (DG→CA3→CA2→CA1 with social memory)
│   ├── Striatum (D1/D2 pathways, three-factor learning)
│   ├── Prefrontal (working memory, gated Hebbian)
│   ├── Cerebellum (error-corrective, motor learning)
│   ├── Thalamus (sensory relay, attention gating)
│   └── MultimodalIntegration (cross-modal binding and integration)
├── Axonal Projections (brain.connections[(src, tgt)])
│   ├── AxonalProjection (pure spike routing, NO weights)
│   ├── CircularDelayBuffer (1-20ms axonal delays)
│   ├── Multi-source concatenation
│   └── Port-based routing ("cortex:l5", "hipp:ca1")
└── Coordination
    ├── TrialCoordinator (task execution)
    └── Clock-driven simulation (fixed timesteps)
```

---

## Core Components

### 1. DynamicBrain

**Location**: `src/thalia/core/dynamic_brain.py`

Central orchestrator that:
- Manages all brain regions and their interconnections
- Built using flexible `BrainBuilder` or from configuration
- Component-based: access regions via `brain.components["name"]` (no direct attributes like `brain.cortex`)
- Coordinates centralized systems (neuromodulators, oscillators, goals)
- Clock-driven execution with axonal delays in pathways
- Provides high-level APIs for task execution

**Key Methods**:
- `forward()` - Process input for N timesteps
- `select_action()` - Choose action via striatum
- `deliver_reward()` - Deliver reward and trigger learning
- `reset()` - Reset all regions and systems

**Component Access**:
- `brain.components["cortex"]` - Get region by name
- `brain.connections[("thalamus", "cortex")]` - Get AxonalProjection
- `brain.components` - Dict of all components

---

### 2. AxonalProjection

**Location**: `src/thalia/pathways/axonal_projection.py`

**Function**: Pure spike routing with realistic axonal delays

**Architecture Philosophy**:
- Axons transmit spikes WITHOUT synaptic weights
- Synapses belong to target regions (at dendrites)
- Implements biologically accurate conduction delays

**Key Features**:
- **Multi-source routing**: Accepts `Dict[str, Tensor]` from multiple regions
- **Delay buffers**: `CircularDelayBuffer` for 1-20ms axonal delays
- **Port-based routing**: Supports output ports (e.g., "cortex:l5", "hipp:ca1")
- **No learning**: Pure transmission (learning at target synapses)

**Usage Pattern**:
```python
# Create axonal projection
projection = AxonalProjection(
    sources=[
        SourceSpec(region_name="cortex", port="l5", size=256, delay_ms=2.0),
        SourceSpec(region_name="thalamus", port=None, size=128, delay_ms=5.0),
    ],
    config=config,
)

# Forward pass: route spikes with delays
source_outputs = {"cortex:l5": cortex_spikes, "thalamus": thal_spikes}
delayed_spikes = projection(source_outputs)  # Dict[str, Tensor]
```

**Biological Accuracy**:
- Matches neuroscience: axons = transmission only
- Realistic delays: 1-5ms local, 5-20ms long-range
- No "pathway learning" (that's at synapses)

---

### Port-Based Routing

**Purpose**: Layer-specific routing for biologically accurate connectivity patterns

Multi-layer regions (e.g., LayeredCortex) can expose specific internal outputs through named **ports**, enabling targeted connections that match biological projection patterns.

#### Available Ports

**LayeredCortex**:
- `source_port="l23"` → Cortico-cortical connections (L2/3 pyramidal neurons)
- `source_port="l5"` → Cortico-subcortical connections (L5 pyramidal neurons)
- `source_port="l6a"` → Corticothalamic type I → TRN (spatial attention)
- `source_port="l6b"` → Corticothalamic type II → Relay (gain modulation)

**Thalamus**:
- `source_port="relay"` → Thalamocortical relay neurons
- `source_port="trn"` → Thalamic reticular nucleus (inhibitory gating)
- `target_port="trn"` → Input to TRN (from cortex L6a)
- `target_port="relay"` → Input to relay (from cortex L6b, ascending sensory)

#### Usage in BrainBuilder

```python
from thalia.core.brain_builder import BrainBuilder

builder = BrainBuilder(global_config)

# Biologically accurate cortical projections
builder.connect("v1", "v2", source_port="l23")          # Cortico-cortical
builder.connect("motor", "striatum", source_port="l5")  # Cortico-subcortical
builder.connect("pfc", "thalamus", source_port="l6a", target_port="trn")   # Attention gating
builder.connect("pfc", "thalamus", source_port="l6b", target_port="relay") # Gain modulation

# Thalamo-cortical pathways
builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward")
```

**Benefits**:
- ✅ **Biological accuracy**: Matches anatomical projection patterns
- ✅ **Explicit routing**: Clear intent in architecture definitions
- ✅ **No manual slicing**: Builder handles output selection automatically
- ✅ **Flexible connectivity**: Same region, multiple projection types

**For Complete Documentation**: See [docs/patterns/port-based-routing.md](../patterns/port-based-routing.md)

---

### 3. NeuralRegion

**Location**: `src/thalia/core/neural_region.py`

**Function**: Base class for brain regions with synaptic weights at dendrites

**Architecture**:
```python
class NeuralRegion(nn.Module, NeuromodulatorMixin, GrowthMixin,
                   ResettableMixin, DiagnosticsMixin):
    """Simplified hierarchy independent of legacy BrainComponent."""
```

**Key Features**:
- **Mixin-based design**: Composed from 4 specialized mixins (no deep inheritance)
- **Multi-source synapses**: `synaptic_weights` dict stores weights per source
- **Per-source learning**: Different `LearningStrategy` instances for different inputs
- **Dict-based input**: `forward(source_spikes: Dict[str, Tensor]) → Tensor`
- **Integrated neurons**: ConductanceLIF population for spike generation
- **Flexible initialization**: Regions add sources via `add_input_source()`

**Forward Pass Pattern**:
```python
class MyRegion(NeuralRegion):
    def __init__(self, n_neurons, ...):
        super().__init__(n_neurons=n_neurons, ...)
        # Synaptic weights and strategies added via add_input_source()

    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Synaptic integration (weights at dendrites)
        total_current = torch.zeros(self.n_neurons, device=self.device)
        for source_name, spikes in source_spikes.items():
            weights = self.synaptic_weights[source_name]
            total_current += weights @ spikes

        # 2. Spike generation
        output_spikes, _ = self.neurons(total_current, g_inh)

        # 3. Per-source learning (automatic during forward)
        for source_name, spikes in source_spikes.items():
            new_weights, _ = self.strategies[source_name].compute_update(
                weights=self.synaptic_weights[source_name],
                pre_spikes=spikes,
                post_spikes=output_spikes,
            )
            self.synaptic_weights[source_name].data = new_weights

        return output_spikes
```

**Benefits**:
- Natural multi-source integration
- Region-specific plasticity control
- Matches biological dendrite organization

---

### 4. Centralized Systems

#### Neuromodulator Systems

**Location**: `src/thalia/neuromodulation/`

Three global projection systems:
- **VTA** (Dopamine) - Reward prediction error, learning gate
- **Locus Coeruleus** (Norepinephrine) - Arousal, uncertainty
- **Nucleus Basalis** (Acetylcholine) - Attention, encoding/retrieval

Managed by `NeuromodulatorManager` with biological coordination (DA-ACh, NE-ACh, DA-NE).

**See**: `CENTRALIZED_SYSTEMS.md`

#### Oscillator System

**Location**: `src/thalia/coordination/oscillator.py`

Five brain rhythms:
- **Delta** (2 Hz) - Sleep consolidation
- **Theta** (8 Hz) - Working memory, encoding/retrieval
- **Alpha** (10 Hz) - Attention gating
- **Beta** (20 Hz) - Motor control
- **Gamma** (40 Hz) - Feature binding

With 5 cross-frequency couplings (theta-gamma, beta-gamma, delta-theta, alpha-gamma, theta-beta).

**See**: `CENTRALIZED_SYSTEMS.md`

#### Goal Hierarchy System

**Location**: `src/thalia/regions/prefrontal_hierarchy.py`

Hierarchical planning and temporal abstraction:
- Goal decomposition and stack management
- Options learning (caching successful policies)
- Hyperbolic temporal discounting

**See**: `CENTRALIZED_SYSTEMS.md`, `HIERARCHICAL_GOALS_COMPLETE.md`

---

### 5. Brain Regions

#### Cortex

**Location**: `src/thalia/regions/cortex/`

**Function**: Sensory processing and feature extraction

**Implementations**:
- **LayeredCortex**: 6-layer canonical microcircuit (L4→L2/3→L5→L6a/L6b)
- **PredictiveCortex**: Predictive coding with local error-driven learning

**Learning Rules**:
- BCM (Bienenstock-Cooper-Munro) for feature selectivity
- STDP (Spike-Timing-Dependent Plasticity)
- Lateral inhibition for competition
- Predictive coding (error-based local learning)

**Key Features**:
- Multi-modal processing (vision, audio, text)
- Hierarchical feature extraction
- Sparse coding
- Corticothalamic feedback (L6a→TRN, L6b→Relay)
- Predictive error minimization (PredictiveCortex)

#### Hippocampus

**Location**: `src/thalia/regions/hippocampus/`

**Function**: Episodic memory encoding and retrieval

**Subregions**:
- **DG** (Dentate Gyrus) - Pattern separation
- **CA3** - Autoassociative memory, pattern completion
- **CA2** - Social memory, temporal context (stability hub)
- **CA1** - Output gating, context integration

**Learning Rules**:
- One-shot Hebbian (rapid encoding)
- ACh-gated encoding/retrieval switching

**Key Features**:
- Sequence encoding and replay
- Context-dependent retrieval
- Consolidation to cortex

#### Prefrontal Cortex (PFC)

**Location**: `src/thalia/regions/prefrontal/prefrontal.py`

**Function**: Working memory, planning, cognitive control

**Learning Rules**:
- Gated Hebbian (dopamine-gated maintenance)
- Hierarchical goal management

**Key Features**:
- Multi-item working memory (theta-phase coding)
- Goal decomposition and tracking
- Mental simulation (planning)

#### Striatum

**Location**: `src/thalia/regions/striatum/`

**Function**: Action selection and reinforcement learning

**Architecture**: Multi-source with D1/D2 opponent pathways
- Per-source synaptic weights (e.g., `"cortex:l5_d1"`, `"hippocampus_d2"`)
- Separate D1-MSNs (direct/GO) and D2-MSNs (indirect/NOGO)
- Source-specific eligibility traces with different tau values:
  - Cortex: 1000ms (standard corticostriatal)
  - Hippocampus: 300ms (fast episodic context)
  - Thalamus: 500ms (phasic signals)

**Learning Rules**:
- Three-factor Hebbian (eligibility × dopamine × activity)
- TD(λ) for multi-step credit assignment
- Direct (D1) / indirect (D2) pathway competition
- Per-source short-term plasticity (STP):
  - Cortical: depressing (U=0.4)
  - Thalamic: facilitating (U=0.25)
  - Hippocampal: depressing (U=0.35)

**Key Features**:
- Model-free RL with TD(λ)
- Dyna planning (model-based lookahead)
- Eligibility traces (300-1000ms per source)
- Action chunking (habits)
- Soft winner-take-all action selection

#### Cerebellum

**Location**: `src/thalia/regions/cerebellum/cerebellum.py`

**Function**: Motor learning and error correction with coordinated plasticity

**Learning Rules**:
- **Per-Purkinje dendritic learning**: Each Purkinje cell has independent parallel fiber synaptic weights
- **Supervised delta rule**: Δw = learning_rate × error × pre_activity (LTD/LTP)
- **Error-driven adjustment**: Climbing fiber signals from inferior olive (IO) neurons

**Gap Junction Synchronization** (December 23, 2025):
- **IO gap junctions**: Electrical synapses (<1ms coupling) synchronize complex spikes
- **Functional connectivity**: Based on shared parallel fiber inputs (spatial proximity proxy)
- **Coordinated learning**: Related Purkinje cells receive correlated error signals
- **Biological accuracy**: Matches observed IO neuron synchronization (<50μm proximity)

**Key Features**:
- **Precise timing**: Sub-millisecond temporal precision for motor control
- **Motor adaptation**: Online error correction via climbing fiber feedback
- **Forward models**: Predicts sensory consequences of actions
- **Distributed plasticity**: Each Purkinje cell learns independently with gap junction coordination
- **LTD/LTP mechanisms**: Depression for excess activity, potentiation for insufficient activity

**Implementation Details**:
- Enhanced microcircuit with granule cells, Golgi cells, basket/stellate interneurons
- Gap junctions use `GapJunctionCoupling` with coupling current I_gap = Σ g[i,j] * (V[j] - V[i])
- Tests: `test_purkinje_learning.py` (6 tests), `test_cerebellum_gap_junctions.py` (5 tests)

#### Thalamus

**Location**: `src/thalia/regions/thalamus/thalamus.py`

**Function**: Sensory relay, gating, and attentional modulation

**Key Features**:
- **Sensory relay** - All sensory modalities pass through thalamus
- **Attentional gating** - Alpha oscillations suppress irrelevant inputs
- **Mode switching** - Burst vs tonic modes for different processing needs
- **TRN inhibition** - Thalamic Reticular Nucleus implements "searchlight" attention

**Attention Mechanisms**:
- Bottom-up (stimulus-driven): Salience, motion, novelty detection
- Top-down (goal-directed): PFC → Cortex attention pathways
- Developmental progression: Shifts from reactive (bottom-up) to proactive (top-down)

**See**: `src/thalia/pathways/attention/` for attention pathway implementations

#### MultimodalIntegration

**Location**: `src/thalia/regions/multisensory.py`

**Function**: Cross-modal binding and integration of multiple sensory modalities

**Learning Rules**:
- Hebbian plasticity for strengthening co-active cross-modal connections
- BCM rule for competitive learning within modality pools
- STDP for spike-timing dependent temporal binding

**Key Features**:
- Separate processing pools for visual, auditory, and language inputs
- Gamma-band (40 Hz) synchronization for feature binding
- Cross-modal attention and enhancement
- Audiovisual speech integration (McGurk effect)
- Embodied language grounding (words → perceptual features)

---

### 4. Pathways

**Location**: `src/thalia/pathways/`

Inter-region connections with:
- Excitatory and inhibitory projections
- Axonal delays (biological realism)
- Plasticity (pathway-specific learning rules)

**Key Pathways**:
- Cortex → Hippocampus (encoding)
- Hippocampus → PFC (retrieval)
- PFC → Striatum (action selection)
- Striatum → Motor output

---

### 5. Coordination Systems

#### TrialCoordinator

**Location**: `src/thalia/coordination/trial_coordinator.py`

Manages task execution:
- Forward pass (sensory → memory → action)
- Action selection (goal-driven → mental simulation → model-free)
- Reward delivery and learning
- Trial lifecycle management

**Integration Points**:
- Goal hierarchy (PFC)
- Mental simulation (planning)
- Intrinsic rewards (curiosity, novelty)

---

## Data Flow

### Forward Pass (Encoding)

```
Input → Cortex → Hippocampus → PFC
         ↓           ↓           ↓
      Features   Encoding    Working Memory
```

### Action Selection

```
PFC (goals) → Striatum → Motor Output
    ↓            ↓
 Simulation   Model-Free RL
```

### Learning

```
Reward → VTA (dopamine) → All Regions
         ↓
    Dopamine gates learning in:
    - Striatum (RL)
    - PFC (working memory)
    - Hippocampus (encoding)
```

---

## Key Design Patterns

### 1. Centralized Computation, Distributed Application

Centralized systems (neuromodulators, oscillators) compute once and broadcast to all regions.

**See**: `CENTRALIZED_SYSTEMS.md`

### 2. Mixin-Based Functionality

Regions compose functionality via 7 mixins (from `NeuralRegion` base class):
- `BrainComponentMixin` - Oscillator phase properties and defaults
- `NeuromodulatorMixin` - Dopamine/ACh/NE handling
- `GrowthMixin` - Dynamic neuron expansion (grow_source/grow_output for multi-source architecture)
- `ResettableMixin` - State reset helpers
- `DiagnosticsMixin` - Health monitoring and metrics
- `StateLoadingMixin` - Common state restoration logic
- `LearningStrategyMixin` - Pluggable learning rules (STDP, BCM, three-factor)

**See**: `../patterns/mixins.md`

### 3. Component Parity

All functionality implemented for BOTH regions AND pathways.

**See**: `../patterns/component-parity.md`

### 4. State Management

Use region-specific state dataclasses (inheriting from `RegionState`) for mutable neural state, direct attributes for configuration and learning parameters.

**See**: `../patterns/state-management.md`

---

## Configuration

### BrainConfig

**Location**: `src/thalia/config/brain_config.py`

Brain-level configuration for device, timing, and oscillators:
```python
from thalia.config import BrainConfig

brain_config = BrainConfig(
    device="cpu",  # or "cuda"
    dt_ms=1.0,     # Simulation timestep
    encoding_timesteps=10,
    delay_timesteps=5,
    test_timesteps=10,
)
```

### BrainBuilder

**Location**: `src/thalia/core/brain_builder.py`

Fluent API for constructing brain architectures:
```python
from thalia.core import BrainBuilder
from thalia.config import LayerSizeCalculator

builder = BrainBuilder(brain_config)
calc = LayerSizeCalculator()

# Add components with semantic sizes
thalamus_sizes = calc.thalamus_from_relay(128)
builder.add_component("thalamus", "thalamus", **thalamus_sizes)
builder.add_component("cortex", "cortex", l4_size=128, l23_size=192, l5_size=128)

# Connect components with explicit ports
builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal")

brain = builder.build()
```

**See**: `../patterns/component-parity.md` for component registry details

---

## Testing

### Test Structure

```
tests/
├── unit/
│   ├── regions/ - Individual region tests
│   ├── pathways/ - Pathway tests
│   ├── neuromodulation/ - Neuromodulator tests
│   └── coordination/ - Oscillator, coordinator tests
└── integration/
    └── brain/ - Full system tests
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/unit/regions/test_cortex.py

# With coverage
pytest --cov=thalia
```

---

## Documentation Structure

```
docs/
├── architecture/ - System architecture (this document)
├── patterns/ - Common implementation patterns
├── design/ - Detailed design specifications
├── decisions/ - Architecture Decision Records (ADRs)
└── reviews/ - Architecture reviews and audits
```

---

## Key Achievements

### Completed Systems ✅ (December 2025)

1. **Neuromodulator Centralization** - All 3 systems (VTA, LC, NB) + biological coordination
2. **Oscillator Integration** - All 5 oscillators + 5 cross-frequency couplings
3. **Goal Hierarchy** - Full hierarchical planning with options learning
4. **Multi-Modal Sensory** - Vision, audio, text processing with spike encoding
5. **Memory Consolidation** - Coordinated replay and offline learning
6. **Attention Systems** - Thalamus gating + attention pathways (bottom-up & top-down)
7. **Language Processing** - Spiking language models with token encoding/decoding
8. **Social Learning** - Imitation learning, pedagogy detection, joint attention
9. **Metacognition** - Confidence calibration and uncertainty estimation
10. **TD(λ) Learning** - Multi-step credit assignment for delayed rewards
11. **Dyna Planning** - Model-based planning with mental simulation
12. **Curriculum Training** - Developmental stages from sensorimotor to reading
13. **Learning Strategies** - Unified learning rule system (Hebbian, STDP, BCM, three-factor)
14. **Axonal Delays** - CircularDelayBuffer in pathways for biological timing

### Biological Fidelity ✅

- ✅ Spike-based processing
- ✅ Local learning rules
- ✅ Neuromodulation
- ✅ Oscillatory coordination
- ✅ Axonal delays
- ✅ Regional specialization

---

## Future Directions

### Short-Term

1. **Emotional States** - Amygdala region integration
2. **Advanced Replay Scheduling** - Priority-based consolidation with cognitive load monitoring

### Long-Term

1. **Theory of Mind** - Modeling others' mental states
2. **Advanced Executive Functions** - Complex reasoning and problem-solving integration

---

## References

### Architecture Documents
- `CENTRALIZED_SYSTEMS.md` - Centralized system architecture
- `HIERARCHICAL_GOALS_COMPLETE.md` - Goal hierarchy implementation

### Design Documents

- `../design/neuron_models.md` - Neuron model specifications
- `../design/curriculum_strategy.md` - Training curriculum

### Patterns
- `../patterns/component-parity.md` - Region/pathway parity
- `../patterns/state-management.md` - State management patterns
- `../patterns/mixins.md` - Available mixins

### Decisions (ADRs)
- See `../decisions/` for all Architecture Decision Records
