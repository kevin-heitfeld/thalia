# THALIA Architecture: Complexity Layers

> A bottom-up guide to understanding THALIA's component hierarchy

**Last Updated**: December 6, 2025  
**Status**: Living document - updated as architecture evolves

---

## Overview

THALIA is organized into **5 complexity levels**, from primitive components
to full brain integration. Each level depends only on components from lower
levels, creating a clean dependency hierarchy.

This layering makes the system:
- **Understandable**: Trace any behavior to specific mechanisms
- **Testable**: Each level can be tested independently
- **Debuggable**: Issues can be isolated to specific layers
- **Maintainable**: Changes propagate predictably

---

## The 5 Levels

```
Level 0: PRIMITIVES
  ↓ (neurons, spikes, traces)
Level 1: LEARNING RULES
  ↓ (STDP, BCM, Hebbian)
Level 2: STABILITY MECHANISMS
  ↓ (homeostasis, E/I balance, normalization)
Level 3: REGIONS (Isolated)
  ↓ (cortex, hippocampus, striatum, PFC)
Level 4: INTEGRATION
  ↓ (EventDrivenBrain, pathways, communication)
```

---

## Level 0: PRIMITIVES

### Purpose
Basic building blocks that implement biological neuron dynamics.

### Components

#### Neurons
- **`LIFNeuron`**: Leaky Integrate-and-Fire neuron
  - File: `src/thalia/core/neuron.py`
  - Implements: Membrane dynamics, spike threshold, refractory period
  - Use when: Simple rate-based computations

- **`ConductanceLIF`**: Conductance-based LIF with E/I separation
  - File: `src/thalia/core/neuron.py`
  - Implements: Reversal potentials, shunting inhibition
  - Use when: Need realistic inhibition dynamics

- **`DendriticNeuron`**: Neuron with nonlinear dendritic branches
  - File: `src/thalia/core/dendritic.py`
  - Implements: Dendritic spikes, plateau potentials, branch independence
  - Use when: Need complex credit assignment

#### Traces
- **`SpikeTraces`**: Exponentially decaying spike history
  - File: `src/thalia/core/traces.py`
  - Implements: Configurable time constants
  - Use when: Need temporal integration for learning rules

- **`ShortTermPlasticity`**: Synaptic facilitation and depression
  - File: `src/thalia/core/stp.py`
  - Implements: U (release probability), tau_fac, tau_rec
  - Use when: Need dynamic synapses

### Dependencies
- PyTorch tensors only
- No dependencies on other THALIA components

### Testing
- **Location**: `tests/unit/test_core.py`
- **Approach**: Synthetic inputs, verify basic dynamics
- **Example**: Test that LIF neuron spikes when input exceeds threshold

---

## Level 1: LEARNING RULES

### Purpose
Local learning rules that modify synaptic weights based on activity.

### Components

#### Spike-Timing-Dependent Plasticity (STDP)
- **File**: `src/thalia/learning/strategies.py`
- **Implements**: Pre-before-post potentiation, post-before-pre depression
- **Parameters**: `a_plus`, `a_minus`, `tau_plus`, `tau_minus`
- **Use when**: Temporal credit assignment needed

#### BCM (Bienenstock-Cooper-Munro)
- **File**: `src/thalia/learning/bcm.py`
- **Implements**: Sliding threshold, rate-based learning
- **Parameters**: `learning_rate`, `tau_theta`, `p` (LTP/LTD power)
- **Use when**: Need competition between synapses

#### Three-Factor Learning
- **File**: `src/thalia/learning/strategies.py`
- **Implements**: STDP + neuromodulator (dopamine)
- **Parameters**: STDP params + `tau_eligibility`
- **Use when**: Reinforcement learning scenarios

#### Hebbian Learning
- **File**: `src/thalia/learning/strategies.py`
- **Implements**: Simple "fire together, wire together"
- **Use when**: Simple association learning

### Dependencies
- Level 0: Uses spike traces for temporal dynamics
- No dependencies on stability or regions

### Testing
- **Location**: `tests/unit/test_core.py`, `tests/unit/test_brain_regions.py`
- **Approach**: Apply to random weight matrices, verify convergence
- **Example**: STDP strengthens coincident pre-post activity

---

## Level 2: STABILITY MECHANISMS

### Purpose
Prevent pathological dynamics (runaway excitation, activity collapse, weight explosion).

### Components

#### Unified Homeostasis
- **File**: `src/thalia/learning/unified_homeostasis.py`
- **Implements**: Target firing rate, intrinsic plasticity, synaptic scaling
- **Parameters**: `target_rate`, `tau_homeostasis`, `adaptation_lr`
- **Use when**: Need stable firing rates across training
- **Prevents**: Activity collapse, seizures

#### E/I Balance Regulation
- **File**: `src/thalia/learning/ei_balance.py`
- **Implements**: Dynamic inhibition scaling to maintain E/I ratio
- **Parameters**: `target_ratio` (default 4.0), `adaptation_rate`
- **Use when**: Network has separate E/I populations
- **Prevents**: Over-inhibition, under-inhibition

#### Divisive Normalization
- **File**: `src/thalia/core/normalization.py`
- **Implements**: Gain control via pooled inhibition
- **Parameters**: `semi_saturation`, `epsilon`
- **Use when**: Need contrast invariance, gain control
- **Prevents**: Saturation, input magnitude dependency

#### Intrinsic Plasticity
- **File**: `src/thalia/learning/intrinsic_plasticity.py`
- **Implements**: Threshold adaptation to match target rate
- **Parameters**: `target_rate`, `learning_rate`, `tau_avg`
- **Use when**: Neurons need adaptive excitability
- **Prevents**: Silent neurons, over-active neurons

#### Metabolic Constraints
- **File**: `src/thalia/learning/metabolic.py`
- **Implements**: Energy-based regularization
- **Parameters**: `energy_per_spike`, `budget`
- **Use when**: Need sparse representations
- **Prevents**: Wasteful over-activity

#### Criticality Monitoring
- **File**: `src/thalia/diagnostics/criticality.py`
- **Implements**: Branching ratio tracking, avalanche detection
- **Parameters**: `target_branching` (1.0), `adaptation_rate`
- **Use when**: Need to stay at edge of chaos
- **Prevents**: Subcritical (dying activity), supercritical (explosions)

### Dependencies
- Level 0: Uses neuron state (voltage, spikes)
- Level 1: Modulates learning rules (e.g., homeostatic scaling of STDP)
- No dependencies on regions

### Testing
- **Location**: `tests/unit/test_robustness.py`
- **Approach**: Create pathological scenarios, verify correction
- **Example**: E/I balance corrects runaway excitation

---

## Level 3: REGIONS (Isolated)

### Purpose
Functional brain regions with specialized computations, tested without inter-region connections.

### Components

#### LayeredCortex
- **File**: `src/thalia/regions/cortex/layered_cortex.py`
- **Implements**: L2/3, L4, L5 with canonical microcircuit
- **Features**: 
  - L4 receives thalamic input
  - L2/3 does recurrent processing
  - L5 outputs to subcortex
  - Optional dendritic nonlinearity
  - Optional robustness mechanisms
- **Use when**: Need cortical computation (sensory, association)
- **Parameters**: `LayeredCortexConfig`

#### TrisynapticHippocampus
- **File**: `src/thalia/regions/hippocampus/trisynaptic.py`
- **Implements**: DG → CA3 → CA1 pathway
- **Features**:
  - DG: Sparse pattern separation
  - CA3: Recurrent auto-association
  - CA1: Pattern completion
  - Theta modulation
- **Use when**: Need memory formation, sequence learning
- **Parameters**: `TrisynapticConfig`

#### Striatum
- **File**: `src/thalia/regions/striatum/base.py`
- **Implements**: D1 (Go) and D2 (NoGo) pathways
- **Features**:
  - Eligibility traces for RL
  - Dopamine-modulated plasticity
  - Action selection
- **Use when**: Need reinforcement learning
- **Parameters**: `n_units`, `action_dim`

#### Prefrontal Cortex
- **File**: `src/thalia/regions/prefrontal.py`
- **Implements**: Working memory with gating
- **Features**:
  - Persistent activity
  - Distractor rejection
  - Context-dependent routing
- **Use when**: Need working memory, cognitive control
- **Parameters**: `n_units`, `working_memory_capacity`

#### Cerebellum
- **File**: `src/thalia/regions/cerebellum.py`
- **Implements**: Mossy fibers → Granule cells → Purkinje cells
- **Features**:
  - Error-corrective learning
  - Climbing fiber teaching signal
  - Motor prediction
- **Use when**: Need fine motor control, prediction
- **Parameters**: `n_mossy`, `n_granule`, `n_purkinje`

### Dependencies
- Level 0: Built from primitive neurons
- Level 1: Use learning rules internally
- Level 2: Can optionally use stability mechanisms
- **No inter-region dependencies**: Each region is testable in isolation

### Testing
- **Location**: `tests/unit/test_brain_regions.py`
- **Approach**: Dummy inputs, verify internal dynamics
- **Example**: LayeredCortex L4 responds to input, L2/3 integrates, L5 outputs

---

## Level 4: INTEGRATION

### Purpose
Connect regions into a functioning brain with event-driven communication.

### Components

#### EventDrivenBrain
- **File**: `src/thalia/core/brain.py`
- **Implements**: 
  - Event queue for spike propagation
  - Axonal delays
  - Region registration
  - Pathway connections
- **Features**:
  - Asynchronous spike communication
  - Configurable delays
  - Dopamine system
  - Oscillation generation (theta, gamma)
  - Sleep/wake cycles
- **Use when**: Need full brain simulation
- **Parameters**: `EventDrivenBrainConfig` or `ThaliaConfig`

#### Pathways
- **File**: `src/thalia/integration/pathways/`
- **Implements**: Connections between regions
- **Types**:
  - Sensory pathways (input → cortex)
  - Cortico-striatal (cortex → striatum)
  - Cortico-thalamic (cortex ↔ thalamus)
  - Hippocampal-cortical (memory consolidation)
- **Use when**: Connecting regions

#### Spiking Pathways
- **File**: `src/thalia/integration/spiking_pathway.py`
- **Implements**: Spike-based inter-region communication
- **Features**: Spike encoding/decoding, delays

### Dependencies
- All lower levels (0-3)
- Orchestrates regions into unified system

### Testing
- **Location**: `tests/unit/test_brain_regions.py` (integration tests)
- **Approach**: Full brain simulation with sensory input
- **Example**: Sensory input → cortex → hippocampus → striatum flow

---

## Cross-Cutting Concerns

### Neuromodulation (Dopamine)
- **Files**: `src/thalia/core/brain.py`, `src/thalia/regions/striatum/`
- **Implements**: Global dopamine signal (tonic + phasic)
- **Affects**: All levels
  - Level 1: Modulates learning rate (three-factor STDP)
  - Level 2: Can affect homeostasis targets
  - Level 3: Striatum relies on it for RL
  - Level 4: Brain-wide dopamine broadcast

### Oscillations (Theta, Gamma)
- **Files**: `src/thalia/regions/theta_dynamics.py`, `src/thalia/regions/gamma_dynamics.py`
- **Implements**: Phase-based modulation
- **Affects**: Levels 3-4
  - Level 3: Regions can be theta/gamma modulated
  - Level 4: Brain generates oscillations

### Predictive Coding
- **Files**: `src/thalia/core/predictive_coding.py`
- **Implements**: Top-down predictions, bottom-up errors
- **Affects**: Levels 3-4
  - Level 3: Regions can implement predictive layers
  - Level 4: Hierarchical prediction

---

## Dependency Rules

### Allowed Dependencies
```
Level 4 → can use → Levels 0, 1, 2, 3
Level 3 → can use → Levels 0, 1, 2
Level 2 → can use → Levels 0, 1
Level 1 → can use → Level 0
Level 0 → can use → PyTorch only
```

### Forbidden Dependencies
- **No upward dependencies**: Level N cannot depend on Level N+1
- **No cross-region dependencies** (within Level 3): Cortex cannot directly import from Hippocampus
- **No circular dependencies**: Ever

### How to Check Dependencies
```bash
# Visualize import graph (if you have pydeps installed)
pydeps src/thalia --max-bacon=2 --cluster

# Grep for suspicious imports
grep -r "from thalia.regions" src/thalia/learning/
# Should return nothing (Level 2 importing Level 3)
```

---

## Testing Strategy by Level

### Level 0: Unit Tests
- **Focus**: Basic neuron dynamics
- **Inputs**: Synthetic currents, spike trains
- **Assertions**: Voltage trajectories, spike timing, trace decay

### Level 1: Unit Tests
- **Focus**: Learning rule correctness
- **Inputs**: Random spike patterns, weight matrices
- **Assertions**: Weight changes match expected STDP/BCM/Hebbian

### Level 2: Unit + Integration Tests
- **Focus**: Stability mechanisms prevent pathologies
- **Inputs**: Pathological scenarios (runaway excitation, collapse)
- **Assertions**: Correction back to healthy state, health monitor passes

### Level 3: Unit + Integration Tests
- **Focus**: Region-specific computations
- **Inputs**: Dummy sensory inputs, task-specific patterns
- **Assertions**: Internal dynamics, output quality, health

### Level 4: Integration Tests
- **Focus**: Full brain behavior
- **Inputs**: Complete tasks (classification, RL, sequence learning)
- **Assertions**: Task performance, inter-region communication, health

---

## Adding New Components

When adding a new component, ask:

1. **What level does this belong to?**
   - Primitive neuron type? → Level 0
   - Learning rule? → Level 1
   - Stability mechanism? → Level 2
   - Brain region? → Level 3
   - Integration/communication? → Level 4

2. **What are its dependencies?**
   - List all THALIA imports
   - Verify they're from lower levels only
   - If not, redesign or split into multiple components

3. **How will it be tested?**
   - Unit tests for isolated functionality
   - Integration tests for interactions
   - Add fixtures to appropriate conftest.py

4. **Does it increase complexity?**
   - Document why it's necessary
   - Consider if existing components could be extended
   - Plan for ablation test to validate value

---

## Evolution of the Architecture

### Current State (December 2025)
- All 5 levels implemented
- 184 unit tests passing
- Robustness mechanisms integrated
- Health monitoring available

### Planned Improvements
- More integration tests (Level 1+2, Level 3+4 interactions)
- Ablation tests to validate each mechanism
- Configuration profiles (MINIMAL, STABLE, FULL)
- Architecture visualization tools

### Historical Changes
- **2025-12-06**: Added health monitoring (Level 2), reorganized tests
- **2025-12-06**: Integrated robustness mechanisms into LayeredCortex (Level 3)
- Earlier: Added EventDrivenBrain (Level 4), dopamine system

---

## Further Reading

- **Implementation Plans**: `docs/design/complexity_mitigation_plan.md`
- **Robustness Mechanisms**: `docs/design/hyperparameter_robustness_plan.md`
- **Integration Testing**: `tests/integration/README.md`
- **Ablation Testing**: `tests/ablation/README.md`
- **API Documentation**: `docs/api/` (to be added)

---

## Questions?

If you're confused about where something belongs:

1. Check its imports - what levels does it depend on?
2. Check the tests - where is it tested?
3. Ask: "Could this be simpler by using only lower-level components?"
4. Consult this document

**Principle**: When in doubt, prefer lower levels. The higher the level,
the more complex the interactions, and the harder to debug.
