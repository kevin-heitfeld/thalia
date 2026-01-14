# Multi-Region Biological Accuracy Review

**Document Purpose**: Comprehensive assessment of biological accuracy across all major brain regions in Thalia, identifying strengths and enhancement opportunities.

**Review Date**: December 2025
**Scope**: LayeredCortex, Hippocampus, Thalamus, Cerebellum, Prefrontal, Striatum
**Methodology**: Architecture analysis, learning rule assessment, temporal dynamics review

---

## Executive Summary

### Overall Assessment

Thalia's brain regions demonstrate **strong foundational biological accuracy** with clear opportunities for enhancement:

**Strengths**:
- ✅ Spike-based processing throughout (no rate coding shortcuts)
- ✅ Region-specific learning rules (STDP, BCM, error-corrective, three-factor)
- ✅ Neuromodulator integration (DA, ACh, NE)
- ✅ Oscillatory dynamics (theta, gamma, alpha)
- ✅ Local learning (no backpropagation)

**Common Enhancement Opportunities**:
- 🔄 Heterogeneous cellular properties (most regions use uniform parameters)
- 🔄 Multi-timescale dynamics (fast/slow traces, multiple time constants)
- 🔄 Axonal delays (limited implementation across regions)
- 🔄 Dendritic computation (simplified in most regions)
- 🔄 NMDA-dependent plasticity (not consistently implemented)

### Priority Ranking (Biological Gap × Functional Impact)

1. **HIGH PRIORITY** (Large gap, high impact):
   - **Hippocampus**: Multi-timescale consolidation, NMDA-dependent plasticity
   - **Prefrontal**: Heterogeneous maintenance dynamics, DA receptor subtypes

2. **MEDIUM PRIORITY** (Moderate gap or impact):
   - **Cortex**: Layer-specific heterogeneity, NMDA-gated plasticity
   - **Cerebellum**: Complex spike dynamics, dendritic calcium

3. **LOW PRIORITY** (Small gap or lower impact):
   - **Thalamus**: Burst dynamics already strong, minor enhancements possible
   - **Striatum**: Phase 1 complete, defer Phase 2/3 as planned

---

## Region 1: LayeredCortex (Neocortex)

### Current Implementation

**Architecture** (2182 lines, `cortex/layered_cortex.py`):
```python
# 6-layer canonical microcircuit
L4 → L2/3 → L5 → L6a/L6b

# Learning: BCM + STDP composite
# Modulation: Theta-gamma coupling, stimulus gating
# Connectivity: Feedforward, recurrent, feedback pathways
```

**Key Features**:
- ✅ Laminar structure with biologically-inspired flow
- ✅ BCM (metaplasticity) + STDP (timing) composite learning
- ✅ Gap junctions for fast synchronization
- ✅ Theta-gamma modulation for encoding/retrieval
- ✅ Stimulus-gated suppression (attention-like)

### Biological Accuracy Assessment

#### Strengths
1. **Laminar Organization**: Faithful 6-layer structure with canonical L4→L2/3→L5 flow
2. **Dual Learning Rules**: BCM for rate homeostasis + STDP for temporal precision
3. **Cross-Frequency Coupling**: Theta-gamma coherence for memory operations
4. **Gap Junction Dynamics**: Fast electrical coupling (τ ~1-2ms)

#### Enhancement Opportunities

**Priority 1: Layer-Specific Heterogeneity**
```python
# Current: Uniform neuron parameters across layers
# Biological reality: Each layer has distinct cell types with different properties

# Enhancement (similar to striatum Phase 1):
layer_configs = {
    "L4": {"tau_mem": 10.0, "v_threshold": -50.0},  # Fast, spiny stellate
    "L2/3": {"tau_mem": 20.0, "v_threshold": -55.0},  # Pyramidal
    "L5": {"tau_mem": 30.0, "v_threshold": -50.0},  # Thick-tuft pyramidal
    "L6": {"tau_mem": 15.0, "v_threshold": -55.0},  # Corticothalamic
}

# Expected biological impact:
# - L4: Fast sensory processing (τ ~10ms)
# - L2/3: Integration and association (τ ~20ms)
# - L5: Output and decision (τ ~30ms, higher threshold)
# - L6: Feedback control (τ ~15ms)
```

**Priority 2: NMDA-Gated Plasticity**
```python
# Current: STDP without voltage-gating
# Biological reality: NMDA receptors require postsynaptic depolarization

# Enhancement:
class NMDAGatedSTDP:
    def compute_update(self, pre_spikes, post_spikes, post_voltage):
        nmda_gate = (post_voltage > -40.0).float()  # Mg²⁺ unblock
        stdp_update = self.base_stdp.compute_update(pre_spikes, post_spikes)
        return stdp_update * nmda_gate  # Only learn when depolarized

# Expected impact: More selective learning, prevents spurious associations
```

**Priority 3: Apical vs Basal Dendrites**
```python
# Current: Single dendritic compartment
# Biological reality: Apical (feedback) vs basal (feedforward) integration

# Enhancement:
class TwoCompartmentPyramidal:
    def forward(self, feedforward_input, feedback_input):
        basal_current = self.basal_weights @ feedforward_input
        apical_current = self.apical_weights @ feedback_input

        # Apical modulates basal (gating, not additive)
        effective_current = basal_current * (1 + 0.5 * apical_current)
        return self.neurons(effective_current)

# Expected impact: Context-dependent processing, predictive coding
```

### Implementation Priority: MEDIUM
- **Rationale**: Cortex is functionally strong but could benefit from layer-specific tuning
- **Recommended Phase**: After hippocampus and prefrontal (higher-priority regions)

---

## Region 2: Hippocampus (Episodic Memory)

### Current Implementation

**Architecture** (2708 lines, `hippocampus/trisynaptic.py`):
```python
# Trisynaptic circuit: DG → CA3 → CA1
# DG: Pattern separation (sparse coding)
# CA3: Pattern completion (recurrent attractor)
# CA1: Pattern comparison (mismatch detection)

# Learning: STDP with theta-gated encoding/retrieval
# Modulation: Acetylcholine gates encoding vs retrieval
```

**Key Features**:
- ✅ Anatomically accurate trisynaptic pathway
- ✅ Pattern separation in DG (sparse activation ~2%)
- ✅ Pattern completion in CA3 (recurrent dynamics)
- ✅ Theta-gated encoding/retrieval (ACh-modulated)
- ✅ One-shot learning capability

### Biological Accuracy Assessment

#### Strengths
1. **Trisynaptic Architecture**: Faithful DG→CA3→CA1 implementation
2. **Pattern Separation**: DG achieves ~2% sparsity (biological: 1-5%)
3. **Theta Rhythm**: 6-10 Hz oscillations gate encoding/retrieval
4. **ACh Modulation**: High ACh = encoding, low ACh = retrieval

#### Enhancement Opportunities

**Priority 1: Multi-Timescale Consolidation** ⚠️ HIGH IMPACT
```python
# Current: Single STDP time constant (τ ~20ms)
# Biological reality: Fast encoding + slow consolidation over hours/days

# Enhancement (analogous to striatum multi-timescale eligibility):
class MultiTimescaleHippocampus:
    def __init__(self):
        # Fast trace: Synaptic tagging (minutes)
        self.fast_trace_tau_ms = 60_000  # 1 minute

        # Slow trace: Systems consolidation (hours)
        self.slow_trace_tau_ms = 3_600_000  # 1 hour

        # Consolidation: Gradual transfer to neocortex
        self.consolidation_rate = 0.001  # 0.1% per timestep

    def forward(self, inputs):
        # Immediate encoding (fast trace)
        fast_weights_delta = self.stdp(inputs, self.output)
        self.fast_trace = 0.95 * self.fast_trace + fast_weights_delta

        # Gradual consolidation (slow trace)
        slow_consolidation = self.consolidation_rate * self.fast_trace
        self.slow_trace = 0.999 * self.slow_trace + slow_consolidation

        # Combined update: Fast (episodic) + slow (semantic)
        self.weights += self.fast_trace + 0.1 * self.slow_trace

# Expected impact:
# - Fast trace: Episodic detail (what happened?)
# - Slow trace: Semantic gist (what does it mean?)
# - Biological fidelity: Matches systems consolidation theory
```

**Priority 2: NMDA-Dependent LTP** ⚠️ HIGH IMPACT
```python
# Current: STDP without NMDA voltage-gating
# Biological reality: Hippocampal LTP requires NMDA receptor activation

# Enhancement:
class NMDADependentHippocampalSTDP:
    def compute_update(self, pre_spikes, post_spikes, post_voltage, ca2_conc):
        # NMDA unblock requires depolarization + calcium influx
        nmda_gate = ((post_voltage > -40.0) & (ca2_conc > 0.5)).float()

        # Standard STDP window
        stdp_window = self.compute_stdp_window(pre_spikes, post_spikes)

        # Gate by NMDA (nonlinear threshold)
        return stdp_window * nmda_gate

# Expected impact:
# - Selective potentiation (only when strongly activated)
# - Cooperativity (requires multiple inputs)
# - Associativity (Hebbian + temporal)
```

**Priority 3: Sharp-Wave Ripples (Offline Replay)**
```python
# Current: Online learning during experience
# Biological reality: Offline consolidation during SWRs (~150 Hz)

# Enhancement:
class SharpWaveRippleReplay:
    def offline_consolidation(self):
        # Trigger: Low ACh, high DA, theta OFF
        if self.ach_level < 0.3 and self.da_level > 0.6:
            # Replay recent trajectories at 10-20× speed
            for trajectory in self.recent_trajectories:
                compressed_trajectory = self.compress_time(trajectory, factor=15)
                self.replay_trajectory(compressed_trajectory)

                # Strengthen replayed synapses
                self.weights += 0.1 * self.replay_trace

# Expected impact:
# - Rapid consolidation during sleep/rest
# - Selective strengthening of important memories
# - Transfer to neocortex (systems consolidation)
```

### Implementation Priority: HIGH
- **Rationale**: Hippocampus is central to episodic learning; multi-timescale consolidation is crucial
- **Recommended Phase**: Phase 1 (similar to striatum multi-timescale eligibility)

---

## Region 3: Thalamus (Sensory Gateway)

### Current Implementation

**Architecture** (1479 lines, `thalamus.py`):
```python
# Sensory relay + TRN gating
# Relay: Thalamic core nuclei (VPL, VPM, LGN, MGN)
# TRN: Thalamic reticular nucleus (inhibitory shell)

# Modes: Burst (attention capture) vs tonic (faithful relay)
# Modulation: NE-gated arousal, alpha oscillations for attention
```

**Key Features**:
- ✅ Dual-mode operation (burst vs tonic)
- ✅ TRN inhibitory gating
- ✅ Alpha oscillations (8-12 Hz) for attention
- ✅ T-type calcium channels for burst generation

### Biological Accuracy Assessment

#### Strengths
1. **Burst vs Tonic**: Voltage-dependent T-type Ca²⁺ channels
2. **TRN Gating**: Inhibitory control of information flow
3. **Alpha Oscillations**: Rhythmic gating for selective attention
4. **Arousal Modulation**: NE-dependent gain control

#### Enhancement Opportunities

**Priority 1: Corticothalamic Feedback** (Minor)
```python
# Current: Primarily feedforward (sensory → thalamus → cortex)
# Biological reality: Massive feedback from cortex L6 → thalamus

# Enhancement:
class ThalamocorticalLoop:
    def forward(self, sensory_input, cortical_feedback):
        # Feedforward: Sensory → thalamus
        relay_current = self.relay_weights @ sensory_input

        # Feedback: Cortex L6 → thalamus (modulatory)
        feedback_gain = torch.sigmoid(self.feedback_weights @ cortical_feedback)

        # Gain modulation (not additive)
        gated_relay = relay_current * feedback_gain

        # TRN inhibition (lateral)
        trn_inhibition = self.trn(gated_relay)
        return self.relay_neurons(gated_relay - trn_inhibition)

# Expected impact: Attention-based gain control, predictive suppression
```

**Priority 2: First-Order vs Higher-Order Nuclei**
```python
# Current: Single relay type
# Biological reality: First-order (sensory) vs higher-order (cortical relay)

# Enhancement:
thalamic_nuclei = {
    "first_order": ["VPL", "VPM", "LGN", "MGN"],  # Sensory relay
    "higher_order": ["pulvinar", "MD", "LP"],  # Cortico-cortical relay
}

# Expected impact: Distinguish sensory vs cognitive relay pathways
```

### Implementation Priority: LOW
- **Rationale**: Thalamus is already biologically strong; enhancements are incremental
- **Recommended Phase**: Phase 3 or later (polish, not foundational)

---

## Region 4: Cerebellum (Motor Learning)

### Current Implementation

**Architecture** (1665 lines, `cerebellum_region.py`):
```python
# Enhanced microcircuit: Granule → Purkinje → DCN
# Learning: Error-corrective (delta rule) via climbing fibers
# Granule layer: 4× sparse expansion (pattern separation)
# Purkinje cells: 100 dendritic compartments (simplified)
```

**Key Features**:
- ✅ Climbing fiber error signals
- ✅ LTD at active parallel fiber-Purkinje synapses
- ✅ Granule layer sparse expansion (3% activity)
- ✅ Supervised learning (not reinforcement)

### Biological Accuracy Assessment

#### Strengths
1. **Error-Corrective Learning**: Direct teaching signal (Δw ∝ error)
2. **Climbing Fiber LTD**: Active PF + CF → depression
3. **Sparse Granule Activity**: 3% matches biological 2-5%
4. **Fast Learning**: 1-10 trials (vs 100s for RL)

#### Enhancement Opportunities

**Priority 1: Complex Spike Dynamics**
```python
# Current: Binary climbing fiber signal (0 or 1)
# Biological reality: Complex spikes (bursts of 2-7 spikes)

# Enhancement:
class ComplexSpikeClimbingFiber:
    def generate_complex_spike(self, error_magnitude):
        # Error → burst length (2-7 spikes)
        n_spikes = int(2 + 5 * min(error_magnitude, 1.0))

        # Inter-spike interval: 1-2ms (very fast burst)
        spike_times = torch.arange(n_spikes) * 1.5  # 1.5ms ISI

        # Each spike triggers calcium influx
        ca2_influx_per_spike = 0.2
        total_ca2 = n_spikes * ca2_influx_per_spike

        return n_spikes, total_ca2

# Expected impact: Graded error signal (small vs large errors)
```

**Priority 2: Dendritic Calcium Compartments**
```python
# Current: Simplified 100 "compartments" (actually just dimensions)
# Biological reality: ~200,000 parallel fiber synapses, calcium domains

# Enhancement:
class DendriticCalciumCompartments:
    def __init__(self, n_compartments=1000):
        # Each compartment: ~200 synapses
        self.ca2_per_compartment = torch.zeros(n_compartments)

    def forward(self, parallel_fiber_input, complex_spike_ca2):
        # Parallel fiber activity raises local Ca²⁺
        pf_ca2 = 0.1 * parallel_fiber_input

        # Climbing fiber raises global Ca²⁺
        cf_ca2 = complex_spike_ca2  # Broadcast to all compartments

        # LTD when both high (Δw ∝ -Ca²⁺_PF × Ca²⁺_CF)
        ltd_magnitude = pf_ca2 * cf_ca2

        return -self.learning_rate * ltd_magnitude

# Expected impact: Spatially-specific learning (local depression)
```

**Priority 3: Purkinje Cell Pause**
```python
# Current: Continuous firing
# Biological reality: Purkinje cells pause during complex spikes

# Enhancement:
class PurkinjePauseResponse:
    def forward(self, parallel_fiber_input, complex_spike):
        # Normal: High tonic rate (~50 Hz)
        simple_spikes = self.neurons(parallel_fiber_input)

        # Complex spike → pause (10-50ms)
        if complex_spike > 0:
            pause_duration_ms = 20.0
            self.paused_until = self.current_time + pause_duration_ms

        # Suppress simple spikes during pause
        if self.current_time < self.paused_until:
            simple_spikes = torch.zeros_like(simple_spikes)

        return simple_spikes

# Expected impact: More realistic output dynamics, DCN disinhibition
```

### Implementation Priority: MEDIUM
- **Rationale**: Cerebellum is functional but could benefit from complex spike dynamics
- **Recommended Phase**: Phase 2 (after high-priority hippocampus/prefrontal)

---

## Region 5: Prefrontal Cortex (Working Memory)

### Current Implementation

**Architecture** (1347 lines, `prefrontal.py`):
```python
# Gated working memory: DA-modulated maintenance
# Rule learning: Context-dependent stimulus-response mappings
# Recurrent excitation: Maintain activity during delays
# Integration timescale: τ ~500ms (slower than sensory cortex)
```

**Key Features**:
- ✅ Dopamine-gated WM updates (burst = update, dip = clear)
- ✅ Recurrent maintenance (persistent activity)
- ✅ Rule neurons for context-dependent behavior
- ✅ Slow integration (500ms vs 10-30ms in sensory cortex)

### Biological Accuracy Assessment

#### Strengths
1. **DA Gating**: Burst → update, baseline → maintain, dip → clear
2. **Recurrent Maintenance**: Activity persists without external input
3. **Cognitive Control**: Rule-based flexible behavior
4. **Slow Time Constants**: τ ~500ms for temporal abstraction

#### Enhancement Opportunities

**Priority 1: Heterogeneous WM Maintenance** ⚠️ HIGH IMPACT
```python
# Current: Uniform recurrent strength across all neurons
# Biological reality: Heterogeneous maintenance (some neurons stable, others flexible)

# Enhancement (analogous to striatum heterogeneous STP):
def sample_heterogeneous_wm_neurons(n_neurons, stability_cv=0.3):
    """Sample heterogeneous WM maintenance properties.

    - Stable neurons: Strong recurrence, resist updates (τ ~1-2s)
    - Flexible neurons: Weak recurrence, rapid updates (τ ~100-200ms)
    """
    # Lognormal distribution for recurrent strength
    mean_recurrent = 0.5
    std_recurrent = mean_recurrent * stability_cv

    recurrent_strength = torch.distributions.LogNormal(
        torch.log(mean_recurrent),
        std_recurrent
    ).sample((n_neurons,))

    # Stable neurons (high recurrence) have longer time constants
    tau_mem = 100 + 400 * recurrent_strength  # 100-500ms range

    return recurrent_strength, tau_mem

# Expected impact:
# - Stable neurons: Maintain context/goals over long delays
# - Flexible neurons: Rapid updating for new information
# - Biological fidelity: Matches recording data (mixed selectivity)
```

**Priority 2: D1 vs D2 Receptor Subtypes** ⚠️ HIGH IMPACT
```python
# Current: Single dopamine signal
# Biological reality: D1 (excitatory, "Go") vs D2 (inhibitory, "NoGo")

# Enhancement:
class D1D2ModulatedPrefrontal:
    def __init__(self):
        # 60% D1-dominant, 40% D2-dominant (approximate ratio)
        self.d1_neurons = int(0.6 * self.n_neurons)
        self.d2_neurons = int(0.4 * self.n_neurons)

    def forward(self, inputs, dopamine_level):
        # D1: Enhance signals (Go pathway)
        d1_gain = 1.0 + 0.5 * dopamine_level
        d1_output = self.d1_pathway(inputs) * d1_gain

        # D2: Suppress noise (NoGo pathway)
        d2_gain = 1.0 - 0.3 * dopamine_level
        d2_output = self.d2_pathway(inputs) * d2_gain

        # Competition: D1 "releases" actions, D2 "withholds"
        return d1_output - 0.5 * d2_output

# Expected impact:
# - D1: Update WM when DA high (new info important)
# - D2: Maintain WM when DA low (protect current state)
# - Biological fidelity: Matches PFC D1/D2 receptor distributions
```

**Priority 3: Multi-Item WM (Item-Specific Gating)**
```python
# Current: Single WM "slot" (all-or-none gating)
# Biological reality: Multiple independent items (4 ± 1 capacity)

# Enhancement:
class MultiItemWorkingMemory:
    def __init__(self, n_items=4):
        # Each item: Independent neural population
        self.items = [WorkingMemorySlot() for _ in range(n_items)]

    def forward(self, inputs, gate_signals):
        # gate_signals: [n_items] binary (which items to update)
        for i, (item, gate) in enumerate(zip(self.items, gate_signals)):
            if gate > 0.5:
                # Update this item
                item.update(inputs[i])
            else:
                # Maintain this item
                item.maintain()

        return [item.read() for item in self.items]

# Expected impact:
# - Capacity limit: ~4 items (biological)
# - Independent gating: Update item 1 while maintaining item 2
# - Chunking: Combine items to expand effective capacity
```

### Implementation Priority: HIGH
- **Rationale**: PFC is central to cognitive control; heterogeneous dynamics are crucial
- **Recommended Phase**: Phase 1 (alongside hippocampus multi-timescale consolidation)

---

## Region 6: Striatum (Action Selection)

### Current Implementation

**Status**: ✅ **Phase 1 Complete** (December 2025)

**Recent Enhancements**:
1. ✅ Heterogeneous STP (per-synapse U, tau_d, tau_f variability)
2. ✅ Multi-timescale eligibility (fast ~500ms, slow ~60s)
3. ✅ Per-source D1/D2 weights (independent pathways)
4. ✅ All 29 tests passing

**Deferred Enhancements** (Phase 2/3):
- Per-synapse axonal delays
- NMDA-dependent synaptic integration
- Dendritic compartments (distal vs proximal)
- Calcium-dependent eligibility traces

### Implementation Priority: LOW (Deferred)
- **Rationale**: Phase 1 complete; defer Phase 2/3 per user request
- **Recommended Phase**: After hippocampus and prefrontal (higher-priority regions)

---

## Cross-Region Enhancement Themes

### Theme 1: Multi-Timescale Dynamics
**Regions Affected**: Hippocampus, Striatum, Prefrontal, Cortex

**Current Status**:
- ✅ Striatum: Fast (500ms) + slow (60s) eligibility (Phase 1 complete)
- ❌ Hippocampus: Single STDP timescale (~20ms)
- ❌ Prefrontal: Single maintenance timescale (~500ms)
- ❌ Cortex: Single BCM timescale (~1s)

**Biological Motivation**:
- Fast traces: Immediate encoding (synaptic tagging)
- Slow traces: Consolidation (systems-level transfer)
- Multiple timescales enable:
  * Short-term → long-term memory transfer
  * Rapid learning + stable retention
  * Flexible vs persistent representations

**Implementation Priority**: HIGH (Hippocampus first, then Prefrontal)

---

### Theme 2: Heterogeneous Cellular Properties
**Regions Affected**: All regions

**Current Status**:
- ✅ Striatum: Heterogeneous STP (Phase 1 complete)
- ❌ Cortex: Uniform neuron parameters within layers
- ❌ Hippocampus: Uniform CA1/CA3 pyramidal cells
- ❌ Prefrontal: Uniform WM maintenance neurons
- ❌ Cerebellum: Uniform Purkinje cells

**Biological Motivation**:
- Real neurons vary 2-10× in:
  * Time constants (τ_mem: 5-50ms)
  * Thresholds (V_th: -55 to -45 mV)
  * Calcium dynamics (τ_Ca: 20-200ms)
- Heterogeneity provides:
  * Mixed selectivity (flexible representations)
  * Distributed temporal encoding
  * Robustness to perturbations

**Implementation Priority**: HIGH (Prefrontal first, then Cortex)

---

### Theme 3: NMDA-Dependent Plasticity
**Regions Affected**: Hippocampus, Cortex, Striatum

**Current Status**:
- ❌ All regions: STDP without voltage-gating

**Biological Motivation**:
- NMDA receptors require:
  * Presynaptic glutamate release
  * Postsynaptic depolarization (Mg²⁺ unblock)
  * Both conditions → Ca²⁺ influx → LTP
- Benefits:
  * Cooperativity (multiple inputs needed)
  * Associativity (Hebbian + temporal)
  * Selectivity (only strong associations)

**Implementation Priority**: MEDIUM (Hippocampus first, highest impact)

---

### Theme 4: Dendritic Computation
**Regions Affected**: Cortex, Hippocampus, Cerebellum

**Current Status**:
- ❌ Cortex: Single compartment (no apical vs basal distinction)
- ❌ Hippocampus: Single compartment
- ⚠️ Cerebellum: 100 "compartments" (simplified)

**Biological Motivation**:
- Dendrites are not passive cables:
  * Apical vs basal compartments
  * NMDA spikes (local nonlinearities)
  * Calcium domains (spatially-specific learning)
- Enables:
  * Context-dependent processing
  * Predictive coding (feedback modulates feedforward)
  * Spatially-specific plasticity

**Implementation Priority**: MEDIUM (Cortex apical/basal first)

---

## Recommended Implementation Roadmap

### Phase 1A: Hippocampus Multi-Timescale (HIGH PRIORITY)
**Duration**: 1-2 days
**Effort**: Medium (similar to striatum multi-timescale eligibility)

**Tasks**:
1. Add `fast_trace_tau_ms`, `slow_trace_tau_ms`, `consolidation_rate` to HippocampusConfig
2. Implement dual eligibility traces in TrisynapticHippocampus
3. Add consolidation logic (fast → slow transfer)
4. Create test suite (8-10 tests, similar to striatum)
5. Verify 29+ total tests passing (10 hippocampus + 19 existing)

**Expected Benefits**:
- Fast encoding of episodes (minutes)
- Slow consolidation to semantics (hours)
- Systems-level transfer (hippocampus → cortex)

---

### Phase 1B: Prefrontal Heterogeneous WM (HIGH PRIORITY)
**Duration**: 1-2 days
**Effort**: Medium

**Tasks**:
1. Add `heterogeneous_wm`, `stability_cv` to PrefrontalConfig
2. Implement `sample_heterogeneous_wm_neurons()` (lognormal distribution)
3. Create stable vs flexible neuron populations
4. Add D1/D2 receptor subtype logic
5. Create test suite (10-12 tests)

**Expected Benefits**:
- Stable neurons: Long-term context/goals
- Flexible neurons: Rapid updating
- D1/D2: Biological gating mechanisms

---

### Phase 2A: Cortex Layer-Specific Heterogeneity (MEDIUM PRIORITY)
**Duration**: 2-3 days
**Effort**: High (6 layers, multiple cell types)

**Tasks**:
1. Define layer-specific configs (L2/3, L4, L5, L6a/b)
2. Implement heterogeneous neuron initialization
3. Add NMDA-gated plasticity (optional)
4. Add apical/basal compartments (L5 pyramidal first)
5. Create comprehensive test suite (15-20 tests)

**Expected Benefits**:
- L4: Fast sensory processing
- L2/3: Associative integration
- L5: Output and decision-making
- L6: Feedback control

---

### Phase 2B: Cerebellum Complex Spikes (MEDIUM PRIORITY)
**Duration**: 1-2 days
**Effort**: Medium

**Tasks**:
1. Implement complex spike burst generation (2-7 spikes)
2. Add dendritic calcium compartments (1000 domains)
3. Implement Purkinje cell pause response
4. Create test suite (8-10 tests)

**Expected Benefits**:
- Graded error signals (small vs large errors)
- Spatially-specific LTD
- Realistic output dynamics

---

### Phase 3: Lower-Priority Enhancements
**Duration**: 3-5 days
**Effort**: Variable

**Tasks**:
1. Thalamus corticothalamic feedback
2. Striatum Phase 2 (NMDA, dendrites) - **deferred per user**
3. Cross-region NMDA plasticity
4. Advanced dendritic computation

---

## Testing Strategy

### Per-Region Test Suites
Each enhancement should include comprehensive tests:

**Hippocampus Multi-Timescale** (8-10 tests):
```python
test_fast_trace_decay()          # τ ~60s
test_slow_trace_persistence()    # τ ~3600s
test_consolidation_transfer()    # Fast → slow
test_combined_learning()         # Fast + slow weights
test_episodic_vs_semantic()      # Qualitative difference
test_systems_consolidation()     # Hippocampus → cortex
test_trace_initialization()      # Correct shapes
test_config_validation()         # Parameter ranges
```

**Prefrontal Heterogeneous WM** (10-12 tests):
```python
test_heterogeneous_sampling()           # Lognormal distribution
test_stable_neuron_persistence()        # Long delays
test_flexible_neuron_updating()         # Rapid changes
test_d1_d2_receptor_subtypes()          # Differential modulation
test_multi_item_capacity()              # 4 ± 1 items
test_item_specific_gating()             # Independent updates
test_recurrent_strength_variability()   # 2-10× range
test_config_validation()                # Parameter checks
```

### Regression Testing
After each phase:
```bash
# Run ALL existing tests (ensure no regressions)
pytest tests/ -v

# Expected: All 29+ tests passing
```

---

## Biological Accuracy Metrics

### Quantitative Benchmarks

**Hippocampus**:
- DG sparsity: 1-5% (current: ~2%) ✅
- CA3 attractor dynamics: τ ~50-200ms ✅
- Theta frequency: 6-10 Hz ✅
- **Enhancement target**: Multi-timescale (fast: 1min, slow: 1hr)

**Prefrontal**:
- WM maintenance: τ ~500ms-2s ✅
- DA gating threshold: ~0.6 normalized ✅
- **Enhancement target**: Heterogeneity (CV ~0.3), D1/D2 ratio (60/40)

**Cortex**:
- L4 time constant: ~10ms (target)
- L2/3 time constant: ~20ms (target)
- L5 time constant: ~30ms (target)
- Gamma frequency: 30-80 Hz (current: ~40 Hz) ✅

**Cerebellum**:
- Granule sparsity: 2-5% (current: 3%) ✅
- **Enhancement target**: Complex spike bursts (2-7 spikes)

**Striatum**:
- D1/D2 balance: 50/50 ✅
- Eligibility tau: fast ~500ms, slow ~60s ✅ (Phase 1 complete)
- STP heterogeneity: CV ~0.2-0.5 ✅ (Phase 1 complete)

---

## Expected Biological Fidelity Impact

### Before Enhancements (Current State)
**Overall Fidelity Score**: 7.5/10

**Strengths**:
- Spike-based processing: ✅
- Region-specific learning: ✅
- Neuromodulation: ✅
- Basic circuit structure: ✅

**Gaps**:
- Single timescales (should be multi-scale)
- Uniform parameters (should be heterogeneous)
- Simplified STDP (should be NMDA-gated)
- Single-compartment neurons (should have dendrites)

### After Phase 1A+1B (Hippocampus + Prefrontal)
**Overall Fidelity Score**: 8.5/10

**Improvements**:
- Multi-timescale dynamics: ✅ (hippocampus, striatum)
- Heterogeneous populations: ✅ (prefrontal, striatum)
- Systems consolidation: ✅ (hippocampus → cortex)

**Remaining Gaps**:
- NMDA-gated plasticity
- Dendritic computation
- Layer-specific cortical tuning

### After Phase 2A+2B (Cortex + Cerebellum)
**Overall Fidelity Score**: 9.0/10

**Improvements**:
- Layer-specific heterogeneity: ✅ (cortex)
- Complex spike dynamics: ✅ (cerebellum)
- Apical/basal compartments: ✅ (cortex L5)

**Remaining Gaps**:
- Full NMDA implementation across all regions
- Advanced dendritic computation

### After Phase 3 (All Enhancements)
**Overall Fidelity Score**: 9.5/10

**Near-Complete Biological Accuracy**:
- Multi-timescale: ✅
- Heterogeneity: ✅
- NMDA-gated: ✅
- Dendritic: ✅
- Circuit-level: ✅

---

## Conclusion

### Summary of Recommendations

**HIGH PRIORITY** (Implement first):
1. **Hippocampus**: Multi-timescale consolidation (fast + slow traces)
2. **Prefrontal**: Heterogeneous WM (stable vs flexible neurons, D1/D2)

**MEDIUM PRIORITY** (Implement second):
3. **Cortex**: Layer-specific heterogeneity (L4/L2/3/L5/L6 tuning)
4. **Cerebellum**: Complex spike dynamics (burst generation, calcium compartments)

**LOW PRIORITY** (Defer or polish later):
5. **Thalamus**: Corticothalamic feedback (incremental improvement)
6. **Striatum**: Phase 2/3 (deferred per user request)

### Expected Timeline
- **Phase 1A+1B**: 2-4 days (hippocampus + prefrontal)
- **Phase 2A+2B**: 3-5 days (cortex + cerebellum)
- **Phase 3**: 3-5 days (lower-priority enhancements)
- **Total**: 8-14 days for comprehensive multi-region enhancement

### Next Steps
1. ✅ Review this document with user (confirm priorities)
2. → Begin Phase 1A: Hippocampus multi-timescale consolidation
3. → Phase 1B: Prefrontal heterogeneous WM
4. → Iterate based on results

---

**Document Status**: ✅ Complete, ready for review
**Last Updated**: December 2025
**Related Documents**:
- `docs/design/striatum_biological_accuracy_investigation.md` (Phase 1 complete)
- `docs/patterns/learning-strategies.md` (Learning rule patterns)
- `docs/architecture/ARCHITECTURE_OVERVIEW.md` (System architecture)
