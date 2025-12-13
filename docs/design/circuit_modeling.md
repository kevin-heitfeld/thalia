# Circuit Modeling Plan

**Status**: 🟢 Current (Reviewed December 13, 2025)
**Created**: December 12, 2025
**Purpose**: Identify which biological circuits should be explicitly modeled and their relevance to curriculum stages

---

## Overview

This document prioritizes biological circuit implementations based on:
1. **Biological accuracy**: How well-documented is the circuit timing?
2. **Functional importance**: Does explicit modeling improve learning/performance?
3. **Curriculum relevance**: Which training stages depend on this circuit?
4. **Implementation effort**: Complexity vs benefit tradeoff

---

## Currently Modeled Circuits

### ✅ Cortex: L4→L2/3→L5 Laminar Microcircuit
**Status**: Fully implemented with inter-layer delays (`src/thalia/regions/layered_cortex.py`)

**Biological Timing**:
- L4→L2/3: ~2ms (feedforward sensory)
- L2/3→L5: ~2ms (associative to output)
- L2/3 recurrent: ~3-5ms (lateral integration)

**Functional Role**:
- L4: Thalamic input processing (sensory features)
- L2/3: Associative integration + lateral connections
- L5: Output to subcortical structures (hippocampus, striatum)

**Curriculum Relevance**:
- **Stage 0**: Critical for sensory processing (MNIST, phonemes)
- **Stage 1-2**: Feature integration for objects and sequences
- **Stage 3-4**: Hierarchical feature extraction for language/reasoning

---

### ✅ Hippocampus: DG→CA3→CA1 Trisynaptic Circuit
**Status**: Fully implemented with inter-layer delays

**Biological Timing**:
- DG→CA3: ~3ms (mossy fiber, sparse pattern separation)
- CA3→CA1: ~3ms (Schaffer collateral, pattern completion)
- Total: ~6ms within hippocampus (completes in one theta cycle ~100-150ms with cortex)

**Functional Role**:
- DG: Pattern separation (orthogonalize similar inputs)
- CA3: Auto-associative memory (pattern completion)
- CA1: Comparator (match retrieval vs input)

**Curriculum Relevance**:
- **Stage 0**: Simple associations (phoneme-to-sound mappings)
- **Stage 1**: Object permanence, working memory support
- **Stage 2**: Episode encoding, sequence learning
- **Stage 3-4**: Episodic retrieval for language understanding, reasoning

---

### ✅ Striatum: D1/D2 Opponent Pathways with Temporal Competition
**Status**: Fully implemented with pathway-specific delays (December 12, 2025)

**Biological Timing**:
- **D1 "Go" pathway**: Striatum → GPi/SNr → Thalamus (~15-20ms total)
  - Direct inhibition of GPi/SNr → disinhibits thalamus → facilitates action
- **D2 "No-Go" pathway**: Striatum → GPe → STN → GPi/SNr (~23-28ms total)
  - Indirect route via GPe and STN → inhibits thalamus → suppresses action
- **Key insight**: D1 pathway is ~8ms FASTER than D2 pathway
  - Creates temporal competition window
  - D1 "vote" arrives first, D2 "veto" arrives later
  - Explains action selection timing and impulsivity

**Functional Role**:
- **D1 neurons**: "Go" signal (dopamine reinforces actions taken)
- **D2 neurons**: "No-Go" signal (dopamine reinforces actions avoided)
- **Temporal competition**: Fast D1 vs slower D2 determines action commitment

**Why Explicit Delays Matter**:
- **Action selection timing**: Explains reaction time variability
- **Impulsivity**: If D2 delayed too much, premature actions
- **Parkinson's**: Loss of D1 → slower D2 dominates → bradykinesia
- **Huntington's**: Loss of D2 → unchecked D1 → hyperkinesia

**Curriculum Relevance**:
- **Stage -0.5**: Motor control (action selection for reaching)
- **Stage 1**: Policy learning (object manipulation decisions)
- **Stage 2**: Sequence learning (action timing in sequences)
- **Stage 3-4**: Cognitive control (inhibit prepotent responses)

**Implementation Details**:
See implementation in:
- `src/thalia/regions/striatum/config.py`: Config parameters (d1_to_output_delay_ms, d2_to_output_delay_ms)
- `src/thalia/regions/striatum/striatum.py`: Circular delay buffers and forward pass logic
- `src/thalia/regions/striatum/checkpoint_manager.py`: Checkpoint support for delay buffers
- `tests/unit/test_striatum_d1d2_delays.py`: Comprehensive test suite (9 tests, all passing)

Key features:
- D1 delay: 15ms (direct pathway)
- D2 delay: 25ms (indirect pathway, arrives 10ms later)
- Circular buffers for efficient memory usage
- Checkpoint save/restore support
- Backward compatible (zero delays disable buffering)

---

## Circuits to Implement

### 🟡 MEDIUM PRIORITY: Thalamus-Cortex-TRN Loop

**Status**: TRN exists but feedback loop not explicitly modeled

**Biological Timing**:
- Thalamus → Cortex L4: ~5-8ms (sensory relay)
- Cortex L6 → TRN: ~8-12ms (corticothalamic feedback)
- TRN → Thalamus: ~3-5ms (inhibitory gating)
- **Full loop**: ~16-25ms (one gamma cycle!)

**Functional Role**:
- **Thalamus**: Sensory relay and gating
- **TRN**: Inhibitory shell around thalamus ("searchlight" attention)
- **Cortex L6**: Top-down attentional control
- **Loop function**: Implements selective attention via inhibitory gating

**Why Explicit Loop Matters**:
- **Attentional selection**: TRN inhibits unattended thalamic neurons
- **Searchlight effect**: TRN coordinates which sensory channels get through
- **Gamma oscillations**: Loop timing generates 40Hz gamma rhythm
- **Sleep spindles**: TRN bursting creates spindle oscillations during sleep

**Curriculum Relevance**:
- **Stage 0**: Basic sensory gating (filter noise during learning)
- **Stage 1**: Object-based attention (attend to relevant objects)
- **Stage 2-3**: Selective attention for language (focus on relevant words)
- **Stage 4**: Top-down attention for reasoning (attend to task-relevant info)

**Implementation Approach**:
```python
# In LayeredCortexConfig
l6_size: int = 64  # Add L6 for corticothalamic feedback
l6_to_trn_delay_ms: float = 10.0

# In ThalamicRelayConfig
trn_to_relay_delay_ms: float = 4.0

# In brain forward loop
# 1. Thalamus → Cortex L4 (already exists)
cortex_input = thalamus.forward(sensory_input)

# 2. Cortex L6 → TRN (top-down attention)
l6_feedback = cortex.get_l6_output()  # New method
trn_input = apply_delay(l6_feedback, l6_to_trn_delay_ms)

# 3. TRN → Thalamus (inhibitory gating)
trn_inhibition = trn.forward(trn_input)
thalamus.apply_trn_inhibition(trn_inhibition)
```

**Estimated Effort**: 4-6 hours (requires adding L6 layer to cortex)

---

### 🟡 MEDIUM PRIORITY: PFC-Striatum-Thalamus Loop

**Status**: Connections exist but loop not explicitly modeled

**Biological Timing**:
- PFC → Striatum: ~5-8ms (goal/rule context)
- Striatum → GPi/SNr: ~15-20ms (action selection)
- GPi/SNr → Thalamus: ~3-5ms (disinhibition)
- Thalamus → PFC: ~5-8ms (feedback)
- **Full loop**: ~30-40ms (beta oscillation period!)

**Functional Role**:
- **PFC**: Maintains goals and rules
- **Striatum**: Selects actions based on PFC context
- **Thalamus**: Relays selected action back to PFC
- **Loop function**: Goal-directed action selection and monitoring

**Why Explicit Loop Matters**:
- **Beta oscillations**: Loop timing generates 15-30Hz beta rhythm
- **Motor preparation**: Beta desynchronization before movement
- **Cognitive control**: PFC gates which actions are allowed
- **Parkinson's treatment**: DBS at beta frequency disrupts pathological loop

**Curriculum Relevance**:
- **Stage 1**: Working memory maintenance (PFC-thalamus loop)
- **Stage 2**: Rule learning (PFC provides context to striatum)
- **Stage 3-4**: Goal-directed behavior (hierarchical control)

**Implementation Approach**:
```python
# Add to PathwayManager
pfc_to_striatum_delay_ms: float = 6.0
striatum_to_thalamus_delay_ms: float = 20.0  # Via GPi/SNr
thalamus_to_pfc_delay_ms: float = 6.0

# This is mostly pathway delays (already supported)
# Just need to track loop explicitly for beta coupling
```

**Estimated Effort**: 2-3 hours (mostly configuration, pathways exist)

---

### 🟢 LOW PRIORITY: Cerebellum Microcircuit

**Status**: Simplified to parallel fiber→Purkinje→output

**Biological Detail**:
- Mossy fibers → Granule cells: ~1-2ms
- Granule cells → Purkinje cells (parallel fibers): ~2-4ms
- Climbing fibers → Purkinje cells: ~1-2ms (error signal)
- Purkinje cells → Deep cerebellar nuclei: ~1-2ms
- **Total**: ~5-10ms (extremely fast, sub-millisecond precision possible)

**Why Not Prioritized**:
- Current simplified model sufficient for error-corrective learning
- Extreme precision matters more for fine motor control
- Our curriculum focuses on cognition/language over motor refinement
- Could add later if motor precision becomes bottleneck

**Curriculum Relevance**:
- **Stage -0.5**: Sensorimotor coordination (current model adequate)
- **Later stages**: Minimal direct impact on cognitive tasks

---

### 🟢 LOW PRIORITY: Hippocampus-PFC Binding

**Status**: Pathways exist but could be more structured

**Biological Detail**:
- CA1 → PFC: ~5-8ms (episodic to working memory)
- PFC → CA1: ~8-12ms (goal-directed retrieval)
- **Binding mechanism**: Theta synchrony coordinates information transfer

**Why Not Prioritized**:
- Current pathways adequate for information flow
- Theta synchrony already implemented via oscillators
- More about coordination than circuit structure

**Curriculum Relevance**:
- **Stage 2-4**: Working memory + episodic interaction
- **Current implementation sufficient**

---

## Summary & Recommendations

### Priority Order

1. **✅ D1/D2 pathway delays** (COMPLETED - December 12, 2025)
   - ✅ Implemented with configurable delays (D1: 15ms, D2: 25ms)
   - ✅ Circular delay buffers with checkpoint support
   - ✅ Comprehensive test suite (9 tests, all passing)
   - Impact: Enables realistic action selection timing and temporal competition
   - Critical for Stages -0.5, 1, 2, 3-4

2. **🟡 Thalamus-Cortex-TRN loop** (4-6 hours)
   - Important for attention mechanisms
   - Gamma oscillation generation
   - Relevant for all stages (sensory gating)

3. **🟡 PFC-Striatum-Thalamus loop** (2-3 hours)
   - Explains beta oscillations functionally
   - Goal-directed control
   - Most relevant for Stages 2-4

4. **🟢 Cerebellum/Hippocampus-PFC** (defer)
   - Current implementations sufficient
   - Lower impact on curriculum goals
   - Revisit if specific issues arise

### Implementation Strategy

**Phase 1** (✅ COMPLETED - December 12, 2025):
- ✅ Implemented D1/D2 pathway delays
- ✅ Added checkpoint support for delay buffers
- ✅ Validated with comprehensive test suite
- Next: Monitor impact on action selection during curriculum training

**Phase 2** (After curriculum Stage 1 validation):
- Add TRN loop if attention issues arise
- Validate gamma oscillation generation

**Phase 3** (After curriculum Stage 3 validation):
- Add PFC-Striatum-Thalamus loop if goal-directed control weak
- Measure beta oscillation emergence

---

## Circuit-to-Curriculum Mapping

### Stage -0.5: Sensorimotor Grounding
**Critical Circuits**:
- ✅ **D1/D2 pathways**: Action selection for reaching/manipulation (IMPLEMENTED)
- ✅ **Cerebellum**: Forward/inverse models (current implementation OK)

**Why**: Motor control requires fast, accurate action selection. D1/D2 timing affects reaction times and movement smoothness.

---

### Stage 0: Sensory Foundations
**Critical Circuits**:
- ✅ **Cortex L4→L2/3→L5**: Sensory feature extraction (MNIST, phonemes)
- 🟡 **Thalamus-TRN loop**: Sensory gating and noise filtering

**Why**: High-quality sensory representations require proper laminar timing and attentional filtering.

---

### Stage 1: Object Permanence & Working Memory
**Critical Circuits**:
- ✅ **Cortex laminar**: Object recognition
- ✅ **Hippocampus trisynaptic**: Episodic associations
- ✅ **D1/D2 pathways**: Policy learning (which object to attend) (IMPLEMENTED)
- 🟡 **PFC-Thalamus loop**: Working memory maintenance

**Why**: Working memory requires stable PFC activity coordinated with thalamic gating. Policy learning needs proper action timing.

---

### Stage 2: Sequence Learning & Simple Language
**Critical Circuits**:
- ✅ **Hippocampus trisynaptic**: Episode sequences
- ✅ **Cortex laminar**: Temporal patterns
- ✅ **D1/D2 pathways**: Action sequences (verb learning) (IMPLEMENTED)
- 🟡 **PFC-Striatum loop**: Rule representation

**Why**: Sequence learning depends on proper temporal credit assignment. D1/D2 delays affect which actions in sequence get reinforced.

---

### Stage 3: Language & Multi-Modal Integration
**Critical Circuits**:
- ✅ **All previously implemented circuits**
- 🟡 **TRN loop**: Selective attention to relevant modalities
- 🟡 **PFC-Striatum-Thalamus loop**: Beta coherence for cognitive control

**Why**: Language requires coordinated attention across modalities and top-down control of information flow.

---

### Stage 4: Abstract Reasoning & Meta-Learning
**Critical Circuits**:
- ✅ **All circuits** (integrated system)
- 🟡 **PFC-Striatum loop**: Goal hierarchy and cognitive control

**Why**: Abstract reasoning requires full system integration with strong top-down control.

---

## Biological Accuracy vs Functional Necessity

**Key Insight**: Not all biological details need explicit modeling for functional intelligence.

**Explicitly Model When**:
- ✅ Timing affects learning outcomes (D1/D2 credit assignment)
- ✅ Circuit dynamics generate emergent properties (gamma/beta oscillations)
- ✅ Delays explain behavioral phenomena (reaction times, impulsivity)

**Can Abstract When**:
- ❌ Sub-millisecond precision not critical for cognition
- ❌ Circuit structure matters more than exact timing
- ❌ High implementation cost, low functional impact

**Our Approach**: Model circuits where timing is computationally relevant, abstract where structure suffices.

---

## Next Steps

1. ✅ **Complete hippocampus delay checkpoint implementation** (COMPLETED)
2. ✅ **Implement D1/D2 pathway delays** (COMPLETED - December 12, 2025)
3. 🔄 **Monitor curriculum training for D1/D2 delay impact** (evaluate action selection timing)
4. ⏸️ **Monitor curriculum training for attention/control issues** (inform TRN/PFC-Striatum loop priority)
5. 📊 **Measure oscillation emergence** (validate that explicit delays generate expected rhythms)

---

## References

- **Striatum**: Mink (1996) - Basal ganglia motor circuits
- **Thalamus-TRN**: Halassa & Kastner (2017) - Thalamic functions in cognitive control
- **PFC-Striatum**: Engel & Fries (2010) - Beta oscillations and cognitive control
- **Cortex laminar**: Douglas & Martin (2004) - Canonical microcircuit
- **Hippocampus**: Lisman & Jensen (2013) - Theta-gamma code for memory

---

**Document Status**: Updated December 12, 2025 - D1/D2 pathway delays completed
