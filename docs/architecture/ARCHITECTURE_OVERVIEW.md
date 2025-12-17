# Thalia Architecture Overview

**Last Updated**: December 13, 2025

## Introduction

Thalia is a biologically-accurate spiking neural network framework implementing multiple brain regions with specialized learning rules. The architecture emphasizes biological plausibility through local learning rules, neuromodulation, and spike-based processing.

## Core Architecture Principles

### 1. Biological Plausibility
- **Spike-based processing**: Binary spikes (0 or 1), not firing rates
- **Local learning rules**: No backpropagation, each region has specialized rules
- **Neuromodulation**: Dopamine, norepinephrine, acetylcholine gate learning
- **Temporal dynamics**: LIF/Conductance-LIF neurons with membrane dynamics
- **Causality**: No access to future information

### 2. Regional Specialization
Each brain region has its own learning rule:
- **Cortex**: Unsupervised Hebbian/BCM/STDP for feature extraction
- **Hippocampus**: One-shot Hebbian for episodic memory
- **Prefrontal Cortex**: Gated Hebbian for working memory
- **Striatum**: Three-factor rule (eligibility × dopamine) for reinforcement learning
- **Cerebellum**: Supervised error-corrective (delta rule) for motor learning

### 3. Event-Driven Computation
- Regions communicate via events with axonal delays
- Natural temporal dynamics from spike propagation
- Clock-driven updates with biologically realistic timesteps

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
│   ├── Thalamus (sensory relay, attention gating)
│   ├── Cortex (sensory processing)
│   ├── Hippocampus (episodic memory)
│   ├── Prefrontal Cortex (working memory, planning)
│   ├── Striatum (action selection, RL)
│   └── Cerebellum (motor learning)
├── Pathways (brain.connections[(src, tgt)])
│   ├── Excitatory projections
│   ├── Inhibitory projections
│   ├── Attention pathways (PFC → Cortex)
│   └── Axonal delays
└── Coordination
    ├── TrialCoordinator (task execution)
    └── Event system (async communication)
```

---

## Core Components

### 1. DynamicBrain

**Location**: `src/thalia/core/dynamic_brain.py`

Central orchestrator that:
- Manages all brain regions and their interconnections
- Built using flexible `BrainBuilder` or from configuration
- Component-based: access regions via `brain.components["name"]`
- Coordinates centralized systems (neuromodulators, oscillators, goals)
- Handles event-driven communication between regions
- Provides high-level APIs for task execution

**Key Methods**:
- `forward()` - Process input for N timesteps
- `select_action()` - Choose action via striatum
- `deliver_reward()` - Deliver reward and trigger learning
- `reset()` - Reset all regions and systems

**Component Access**:
- `brain.components["cortex"]` - Get region by name
- `brain.connections[("thalamus", "cortex")]` - Get pathway
- `brain.components` - Dict of all components

---

### 2. Centralized Systems

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

### 3. Brain Regions

#### Cortex

**Location**: `src/thalia/regions/cortex/`

**Function**: Sensory processing and feature extraction

**Learning Rules**:
- BCM (Bienenstock-Cooper-Munro) for feature selectivity
- STDP (Spike-Timing-Dependent Plasticity)
- Lateral inhibition for competition

**Key Features**:
- Multi-modal processing (vision, audio, text)
- Hierarchical feature extraction
- Sparse coding

#### Hippocampus

**Location**: `src/thalia/regions/hippocampus/`

**Function**: Episodic memory encoding and retrieval

**Subregions**:
- **DG** (Dentate Gyrus) - Pattern separation
- **CA3** - Autoassociative memory, pattern completion
- **CA1** - Output gating, context integration

**Learning Rules**:
- One-shot Hebbian (rapid encoding)
- ACh-gated encoding/retrieval switching

**Key Features**:
- Sequence encoding and replay
- Context-dependent retrieval
- Consolidation to cortex

#### Prefrontal Cortex (PFC)

**Location**: `src/thalia/regions/prefrontal.py`

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

**Learning Rules**:
- Three-factor Hebbian (eligibility × dopamine × activity)
- Direct/indirect pathway competition

**Key Features**:
- Model-free RL (Q-learning)
- Eligibility traces
- Action chunking (habits)

#### Cerebellum

**Location**: `src/thalia/regions/cerebellum.py`

**Function**: Motor learning and error correction

**Learning Rules**:
- Supervised delta rule (granule → Purkinje)
- Error-driven adjustment

**Key Features**:
- Precise timing
- Motor adaptation
- Forward models

#### Thalamus

**Location**: `src/thalia/regions/thalamus.py`

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

#### Event System

**Location**: `src/thalia/events/`

Event-driven communication:
- Async message passing between regions
- Axonal delays
- Event queuing and dispatch

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

Regions compose functionality via mixins:
- `NeuromodulatedMixin` - Dopamine/NE/ACh receptivity
- `OscillatorMixin` - Oscillator phase tracking
- `IntrinsicMotivationMixin` - Curiosity/novelty
- `CheckpointMixin` - State serialization

**See**: `../patterns/mixins.md`

### 3. Component Parity

All functionality implemented for BOTH regions AND pathways.

**See**: `../patterns/component-parity.md`

### 4. State Management

Use `RegionState` for neural state, direct attributes for learning parameters.

**See**: `../patterns/state-management.md`

---

## Configuration

### ThaliaConfig

**Location**: `src/thalia/config/`

Hierarchical configuration:
```python
ThaliaConfig
├── global_ (device, dtype, seed)
├── brain (region sizes, connections)
├── learning (rates, rules)
├── curriculum (training stages)
└── neuromodulation (DA, NE, ACh parameters)
```

**See**: `../design/curriculum_strategy.md`

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

### Completed Systems ✅

1. **Neuromodulator Centralization** - All 3 systems (VTA, LC, NB) + coordination
2. **Oscillator Integration** - All 5 oscillators + 5 cross-frequency couplings
3. **Goal Hierarchy** - Full hierarchical planning with options learning
4. **Event-Driven Communication** - Async messaging with axonal delays
5. **Multi-Modal Sensory** - Vision, audio, text processing
6. **Memory Consolidation** - Coordinated replay and offline learning
7. **Attention Systems** - Thalamus gating + attention pathways (bottom-up & top-down)
8. **Language Processing** - Spiking language models with token encoding/decoding
9. **Social Learning** - Imitation learning, pedagogy detection, joint attention
10. **Metacognition** - Confidence calibration and uncertainty estimation

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
- `../design/architecture.md` - Original architecture design
- `../design/neuron_models.md` - Neuron model specifications
- `../design/curriculum_strategy.md` - Training curriculum

### Patterns
- `../patterns/component-parity.md` - Region/pathway parity
- `../patterns/state-management.md` - State management patterns
- `../patterns/mixins.md` - Available mixins

### Decisions (ADRs)
- See `../decisions/` for all Architecture Decision Records
