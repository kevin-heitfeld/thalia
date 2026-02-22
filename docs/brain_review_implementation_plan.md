# Brain Review & Implementation Plan

**Date:** February 22, 2026
**Scope:** Full review of `src/thalia/brain/` — biological accuracy, architecture quality, and refactoring opportunities.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Biological Issues — Critical](#2-biological-issues--critical)
3. [Biological Issues — High Priority](#3-biological-issues--high-priority)
4. [Biological Issues — Medium Priority](#4-biological-issues--medium-priority)
5. [Architecture Issues — Critical](#5-architecture-issues--critical)
6. [Architecture Issues — High Priority](#6-architecture-issues--high-priority)
7. [Architecture Issues — Medium Priority](#7-architecture-issues--medium-priority)
8. [Missing Brain Regions](#8-missing-brain-regions)
9. [What Is Working Well](#9-what-is-working-well)
10. [Implementation Plan](#10-implementation-plan)

---

## 1. Executive Summary

The Thalia brain has strong foundations: conductance-based LIF neurons, a clean synaptic weight architecture, biologically grounded learning rules (STDP, BCM, three-factor, D1/D2 asymmetry), and proper spike routing with axonal delays. Several regions (Cortex, Hippocampus, Striatum, VTA, Thalamus, PFC, Cerebellum, LC, NB, Medial Septum, SNr) are well-specified and capture key biological mechanisms.

However, there are **three classes of problems** that need to be addressed:

1. **Biological inaccuracies** — mechanisms that are either wrong or so simplified as to break emergent properties that the rest of the system depends on (NMDA voltage-gating, missing LHb negative RPE pathway, basal ganglia indirect/hyperdirect paths, T-channels disabled).
2. **Architectural debt** — the neuromodulator dual-routing design, `DynamicBrain` containing region-specific business logic, and duplicated infrastructure code all create ambiguity and will compound as the model grows.
3. **Missing regions** — the amygdala, lateral habenula, STN/GPe, inferior olive, and subiculum are all explicitly referenced by existing regions but not implemented, leaving broken circuit loops.

The plan below is prioritized so that each phase yields a self-consistent, runnable model.

---

## 2. Biological Issues — Critical

### 2.1 NMDA Receptor: Missing Voltage-Dependent Mg²⁺ Unblock

**Location:** `ConductanceLIF`, `NeuralRegion._split_excitatory_conductance()`

**Problem:** NMDA receptors are implemented as a slow conductance (`tau_nmda ≈ 100ms`) but without the essential Mg²⁺ voltage-gate. Biologically, NMDA channels are blocked by Mg²⁺ at resting potential and only unblock when the postsynaptic membrane is sufficiently depolarized. This voltage-dependence is the **molecular basis of coincidence detection** — the mechanism that makes STDP and Hebbian learning biologically possible.

Without the voltage-gate, NMDA channels open regardless of postsynaptic state, which:
- Destroys the coincidence detection property
- Removes the synergism between AMPA (fast drive to threshold) and NMDA (coincidence amplifier)
- Causes NMDA to accumulate pathologically at rest (the reason `nmda_ratio` was forced to 0% in production)

**Fix:** Implement the Jahr–Stevens Mg²⁺ unblock factor inside `ConductanceLIF.step()`:
```
f_nmda(V) = 1 / (1 + [Mg²⁺]·exp(-γ·V) / K_d)
```
With `[Mg²⁺] = 1 mM`, `γ = 0.062 mV⁻¹`, `K_d = 3.57 mM` (Jahr & Stevens 1990). In normalized units (threshold = 1.0), this becomes a sigmoid over the normalized voltage `V_norm`. The NMDA conductance contribution becomes `g_nmda_eff = g_nmda × f_nmda(V)`. This enables safe use of realistic NMDA ratios (~20–30%) and restores proper leaky coincidence detection.

---

### 2.2 Neuromodulator Dual-Routing — Parallel and Inconsistent Pathways

**Location:** `DynamicBrain.__init__` (neuromodulator_tracts), `DynamicBrain.forward()` (STEP 3), `NeuromodulatorReceptor` in target regions

**Problem:** Neuromodulators (DA, NE, ACh) currently travel via **two independent pathways simultaneously**:

1. **Pathway A (AxonalTract):** Spikes from VTA/LC/NB are routed like any other spike, through `AxonalTract` with delay buffers, arriving as raw `g_exc` conductance at target neurons. If the connection exists, the downstream neuron's membrane potential is affected directly.
2. **Pathway B (NeuromodulatorTract):** `DynamicBrain` reads the same spikes from `_last_brain_output`, passes them through a `NeuromodulatorTract` IIR low-pass filter modeling volume diffusion, and broadcasts the filtered concentration as a `NeuromodulatorInput` dict to all regions simultaneously.

Target regions further process the Pathway B signal through `NeuromodulatorReceptor`, which applies another IIR filter for receptor kinetics. So the same spike is filtered twice (once in `NeuromodulatorTract`, once in `NeuromodulatorReceptor`) before being used in the actual computation.

Pathway A directly drives postsynaptic currents AND Pathway B modulates gain, learning rates, and gating. There is currently no guarantee that the two are consistent or non-redundant.

**Fix:** Unify to a single pipeline. The `NeuromodulatorReceptor` inside target regions is the right place for the IIR kinetics. Remove `NeuromodulatorTract` and the `neuromodulator_tracts` ModuleDict from `DynamicBrain`. Remove the Pathway B broadcast loop (STEP 3 in `DynamicBrain.forward`). Route neuromodulator spikes only via `AxonalTract`, and make each target region's `NeuromodulatorReceptor.step()` consume those spike inputs directly: `receptor.update(spikes_from_axonal_tract)`. The `NeuromodulatorInput` dict that regions receive can be built from the already-arrived `SynapticInput` for that region by extracting the neuromodulator-specific `SynapseId` entries.

---

### 2.3 Thalamic T-Type Ca²⁺ Channels: Disabled

**Location:** `Thalamus.__init__`, both `NeuronFactory.create_relay_neurons()` and `create_trn_neurons()` with `enable_t_channels=False`

**Problem:** The comment says "TEMPORARY: Disable T-channels to test TRN-relay correlation." T-type Ca²⁺ channels are the primary mechanism for thalamic burst mode firing and are fundamental to:
- Burst/tonic mode switching (the region's stated core functionality)
- Alpha/sleep spindle oscillation generation (TRN↔relay feedback loop)
- Attentional gating via synchronized oscillations
- The thalamus→cortex "up-state" triggering during consolidation

Without T-channels, the burst mode is purely heuristic, TRN-relay correlation tests are biologically invalid, and the attend/suppress functionality is broken. This is not a minor optimization; it breaks the region's primary biological mechanism.

**Fix:** Implement T-channel dynamics properly. In LIF terms: when `V < V_hyperpolarize_threshold` for duration `T_deinactivation`, a transient inward current `I_T = g_T × m_inf(V) × h(V) × (V - E_Ca)` is added. `h` de-inactivates with a slow time constant (~100ms) when hyperpolarized. The burst consists of 3–5 spikes at 200–500 Hz riding on the Ca²⁺ "hump," followed by absolute refractoriness. Enable T-channels, fix whatever bug was observed, and properly validate burst-mode transitions.

---

### 2.4 VTA: No Lateral Habenula → Negative RPE Pathway

**Location:** `VTA`, `DynamicBrain.deliver_reward()`

**Problem:** The VTA computes RPE as approximately `r - V(s)` with `V(s)` estimated from the SNr firing rate. This captures positive prediction errors (DA bursts for better-than-expected reward) but the **negative prediction error (DA pause for omitted reward)** requires a different biological pathway.

In the real brain, negative PEs follow: **Lateral Habenula (LHb) → Rostromedial Tegmental Nucleus (RMTg) → VTA**. The LHb receives the expectation-vs-reality comparison from basal ganglia output (GPi/SNr). RMTg is a GABAergic relay that inhibits VTA DA neurons, causing the pause.

Without LHb/RMTg, the system cannot properly learn from punishment or reward omission. The current `r - V(s)` heuristic in `DynamicBrain._compute_intrinsic_reward()` is not a substitute.

**Fix:** Implement LHb and RMTg as proper brain regions. See Section 8.

---

### 2.5 Intrinsic Reward Computation Embedded in DynamicBrain

**Location:** `DynamicBrain._compute_intrinsic_reward()`, `deliver_reward()`

**Problem:** `DynamicBrain` computes intrinsic reward by reading cortex L4 activity as "prediction error" and CA1 activity as "memory retrieval quality," then combining them into a scalar reward delivered to `RewardEncoder`. This is biologically wrong at multiple levels:

- L4 activity is NOT prediction error — it's the summed feedforward input. The prediction error signal would live in the apical dendrites of L5 neurons or in specific laminar differences.
- CA1 firing rate is NOT a memory retrieval quality signal — it is a spatial/temporal index.
- Intrinsic motivation circuits (curiosity, free energy minimization) live in the ACC, anterior insula, and VTA, not in a method on the brain orchestrator class.
- This couples `DynamicBrain` to specific region types by name, violating the generic orchestrator contract.

**Fix:** Remove `_compute_intrinsic_reward()` from `DynamicBrain`. Create an `AnteriorCingulateCortex` region (or extend the `RewardEncoder`) that properly computes prediction error from the difference between L5 predictions and L4 sensory responses. Wire it into the reward circuit via explicit connections.

---

## 3. Biological Issues — High Priority

### 3.1 Basal Ganglia Indirect and Hyperdirect Pathways Missing

**Location:** `Striatum`, `SubstantiaNigra`, `BrainBuilder` presets

**Problem:** The basal ganglia is currently a two-nucleus simplification (Striatum → SNr → VTA/Thalamus). The critical indirect and hyperdirect pathways are explicitly acknowledged as "future work" in `SubstantiaNigra` but absent:

- **Indirect pathway:** D2-MSN → GPe (inhibitory) → STN (disinhibited excitatory) → SNr/GPi. This delays actions, implements the "no-go" signal with a slight lag versus the direct pathway.
- **Hyperdirect pathway:** Cortex → STN (direct, ~5–7ms) → SNr/GPi. This is the **fastest** basal ganglia output, providing a rapid braking signal *before* striatal processing completes. It implements the "global suppression" that precedes action selection.
- **GPi:** The second output nucleus, projecting to thalamus VA/VL (motor pathway), parallel to SNr.

Without these, there is no proper speed-accuracy tradeoff, no meaningful action suppression, and only a caricature of reinforcement learning. The D2 pathway in `Striatum` currently projects "directly" to SNr (via a simplified bypass) producing a biologically implausible shortcut.

**Fix:** Add GPe, STN, and GPi as proper regions. See Section 10, Phase 3.

---

### 3.2 Cortex Missing SOM+/VIP Disinhibitory Circuit

**Location:** `CorticalInhibitoryNetwork`, `Cortex`

**Problem:** The `CorticalInhibitoryNetwork` models only PV+ (parvalbumin) fast-spiking interneurons (FSIs). Two additional interneuron classes are critical:

- **SOM+ (Somatostatin) Martinotti cells:** Target the apical dendrites of pyramidal cells (not soma like PV+). They implement feedback inhibition with ~100ms delay (activated by pyramidal firing). Because they target dendrites, they gate calcium plateau potentials and top-down input, not just somatic integration.
- **VIP (Vasoactive Intestinal Peptide) cells:** The most important for attention. VIP cells are activated by cholinergic input (ACh) and top-down signals from PFC. They *inhibit* SOM and PV interneurons — creating **disinhibition** of pyramidal cells. This is the primary biological substrate for ACh-mediated attention gating.

The circuit: `PFC/ACh → VIP → inhibit SOM/PV → disinhibit pyramidal` creates the "attentional spotlight" that amplifies attended streams. Without VIP+SOM, ACh modulation of cortical computation has no proper circuit substrate.

**Fix:** Add `SOMInterneuronPopulation` (targeting dendrites) and `VIPInterneuronPopulation` (inhibiting SOM/PV) to `CorticalInhibitoryNetwork`. Connect ACh receptor output and PFC top-down to VIP population drive.

---

### 3.3 Hippocampal Subiculum Missing

**Location:** `Hippocampus`, connections to neocortex

**Problem:** In all current models, CA1 projects directly to cortex. Biologically, the subiculum is the large output structure between CA1 and the entorhinal cortex:
- CA1 → Subiculum → Entorhinal Cortex (EC) → Neocortex
- Subiculum also projects to: lateral septal nucleus, hypothalamus, prefrontal cortex
- Subiculum neurons have "burst" and "regular-spiking" subtypes with different output targets
- The subiculum implements temporal compression and context-dependent firing rate coding

Skipping the subiculum treats CA1 as a simple pass-through to cortex, losing the output-gating and rate transformation that the subiculum performs.

**Fix:** Add a `Subiculum` population within `Hippocampus` (or as a separate region). Route `CA1 → Subiculum → EC` in the circuit connectivity.

---

### 3.4 MedialSeptum and Hippocampus Theta: Redundant Oscillators

**Location:** `Hippocampus` (internal theta counter), `MedialSeptum`

**Problem:** The hippocampus maintains its own internal theta oscillation (phase variable advancing with `dt_ms`). The medial septum also generates theta rhythm via intrinsic bursting. These are currently **independent oscillators** — the MS theta drives hippocampal neurons via ACh/GABA projections, but the hippocampus also has its own internal theta counter that controls encoding/retrieval phase. They can easily drift out of synchrony, producing biologically implausible behavior.

**Fix:** Remove the internal phase counter from `Hippocampus`. The hippocampal theta phase should be driven **exclusively** by MS input. The hippocampus detects theta phase from the incoming MS spike rate pattern (ACh neurons fire at peaks, GABA neurons fire at troughs), not from an internal clock. This is how the actual septohippocampal circuit works.

---

### 3.5 Cerebellar Inferior Olive Not a Proper Region

**Location:** `Cerebellum`, `VectorizedPurkinjeLayer`

**Problem:** The inferior olive (IO) sends climbing fiber inputs to Purkinje cells, generating complex spikes at ~1 Hz. The IO is currently treated as an abstract external input signal (`climbing_fiber_input` parameter), with no region, no circuit dynamics, and no feedback from DCN. This means:
- There is no nucleo-olivary feedback (DCN → IO inhibitory loop) which controls the timing and probability of complex spikes.
- The IO's rhythmic pacemaking (~1 Hz from intrinsic gap junctions) is not modeled.
- Motor error computation — the actual function of the IO — happens outside the brain circuit.
- The Cerebellum cannot participate in the recurrent learning loops (cerebro-cerebellar) that are needed for internal model forward simulation.

**Fix:** Add an `InferiorOlive` region. See Section 8.

---

### 3.6 Entorhinal Cortex Input to Hippocampus Not Differentiated

**Location:** `Hippocampus.__init__`, connection wiring in `BrainBuilder`

**Problem:** The hippocampus receives a single "cortex" or "entorhinal" input that goes to all sub-regions. Biologically the perforant path from EC to hippocampus has **two functionally distinct streams**:
- **MEC (Medial EC) → DG and CA3:** Grid cells, spatial/temporal coding, episodic sequence encoding
- **LEC (Lateral EC) → CA1 direct:** Object/item coding, context representation

The DG pattern separation function is specifically driven by MEC sparse input. The direct CA1 input from LEC provides the "what" (object identity) to be combined with the CA3 "where" (spatial context from Schaffer collaterals). Bundling these as a single unnamed cortical input loses this critical functional separation.

**Fix:** Separate EC inputs into `ec_mec` (targets DG and CA3 direct) and `ec_lec` (targets CA1 direct) in the hippocampus connection API. Update BrainBuilder to wire cortical outputs to the appropriate EC subdivision.

---

## 4. Biological Issues — Medium Priority

### 4.1 STDP Window Shape Not Per-Region

**Location:** `NeuralRegionConfig` (tau_plus_ms = tau_minus_ms = 20ms), `STDPConfig`

**Problem:** All regions share identical default STDP window parameters (`tau_plus = tau_minus = 20ms`, `A+/A- = 1.0`). Biologically, STDP window shape varies significantly by region and synapse type:
- **Hippocampal CA3→CA1:** Asymmetric narrow window (~20ms LTP, ~20ms LTD) — classical Hebbian
- **Cortical L2/3:** Wider LTD window (~40ms) versus LTP (~20ms)
- **Striatal corticostriatal:** Much slower eligibility kinetics (500ms–60s) — already handled by D1/D2STDP
- **Cerebellum (parallel fiber → Purkinje):** Anti-Hebbian (LTD when PF and CF coincide, not LTP)

Per-region STDP parameters are already partially configurable via `STDPConfig` but the defaults in `NeuralRegionConfig` are never overridden, causing all STDP-using regions to share identical parameters.

**Fix:** Remove STDP parameters from `NeuralRegionConfig` base class (they create false redundancy). Require each region's config to explicitly specify its `STDPConfig` when STDP is used. Add sensible biological defaults per region type.

---

### 4.2 Missing Serotonin System (Dorsal Raphe Nucleus)

**Location:** No implementation

**Problem:** Serotonin (5-HT) from the dorsal raphe nucleus (DRN) modulates temporal discounting (patience), fear extinction, social behavior, and mood. It acts as an opponent to dopamine in several circuits:
- High 5-HT → prefer delayed rewards (Daw et al. 2002)
- 5-HT → LHb suppression (reduces negative PE → less fear response)
- 5-HT2A receptors → cortical desynchronization (attention, arousal)
- 5-HT1A receptors → autoreceptors (feedback regulation)

Without 5-HT, the temporal discounting aspect of RL is hardcoded via the discount factor γ rather than being an adaptive biological parameter.

**Fix:** Add DRN region (Section 8, Tier 2). Add `5ht` as a neuromodulator type in `NeuromodulatorReceptor`. Connect DRN to target regions.

---

### 4.3 Missing Theta-Gamma Phase Coupling Infrastructure

**Location:** `Cortex` (gamma via FSI), `Hippocampus` (theta), `MedialSeptum`

**Problem:** Theta-gamma coupling (nested gamma cycles within theta waves) is the proposed mechanism for "chunking" information in hippocampus-cortex interactions. Items in working memory are represented in different gamma cycles of a single theta wave. The current implementation has theta oscillation in hippocampus and gamma oscillation in cortex as completely independent systems with no phase relationship.

**Fix:** Add a cross-region `OscillationCoordinator` that tracks and enforces phase relationships. Gamma cycles in cortex and hippocampus should reset/align to the MS-driven theta phase. This does not need to be a neural region — it can be a lightweight synchronization object in `DynamicBrain`.

---

### 4.4 L-LTP Synaptic Tagging Only in Hippocampus

**Location:** `hippocampus/synaptic_tagging.py`, `Hippocampus`

**Problem:** Late-phase LTP (protein synthesis-dependent, requires synaptic tagging) is implemented only in `Hippocampus` via `SynapticTagging`. Long-term memories engage neocortex too (especially in system consolidation), and cortical synapses exhibit their own late-phase LTP (with partially different molecular substrates). The current design implies only hippocampal synapses have long-term consolidation.

**Fix:** Generalize `SynapticTagging` as a reusable component (it already exists independently). Integrate it into `NeuralRegion` as an optional `synaptic_tagging: Optional[SynapticTagging]` that any region can use. Enable it for `Cortex` L2/3 synapses receiving hippocampal replay.

---

## 5. Architecture Issues — Critical

### 5.1 `SynapseIdXxxDict` — Triplicated Infrastructure Code

**Location:** `NeuralRegion` (lines ~68–280): `SynapseIdParameterDict`, `SynapseIdModuleDict`, `SynapseIdBufferDict`

**Problem:** Three separate container classes each duplicate the exact same `_encode` / `_decode` static methods:
```python
@staticmethod
def _encode(s: SynapseId) -> str:
    inh = "1" if s.is_inhibitory else "0"
    return f"{s.source_region}|{s.source_population}|{s.target_region}|{s.target_population}|{inh}"
```

This is copied verbatim across three classes. Any change to the encoding schema (e.g., adding a `synapse_type` field) must be made in three places. All three also duplicate `__contains__`, `__len__`, `keys()`, `values()`, `items()`, and iterator logic.

**Fix:** Move `to_key() -> str` and `from_key(s: str) -> SynapseId` directly onto `SynapseId`. The encoding is a property of the type being serialized — it has no business living in the containers. All three container classes then become trivial wrappers calling `key.to_key()` / `SynapseId.from_key(k)`, and any future wire-format change (e.g., adding `synapse_type` in Phase 10) is a single edit in one place.

---

### 5.2 `NeuralRegion` Has Two Parallel Learning Strategy Variables

**Location:** `NeuralRegion.__init__`: `_learning_strategies` (SynapseIdModuleDict) AND `learning_strategy` (Optional[LearningStrategy])

**Problem:** The base class exposes both a per-synapse dict (`_learning_strategies`) and a single-strategy reference (`learning_strategy`). Subclasses set one or the other or both. There is no authoritative contract for which one to use. `Hippocampus` and `Striatum` use `_learning_strategies`; some other regions use the singletone `learning_strategy`. Any code iterating over a region's learning must know which style the region uses.

**Fix:** Remove `learning_strategy: Optional[LearningStrategy]` entirely. All learning is per-synapse and stored in `_learning_strategies`. Add a convenience method `get_learning_strategy(synapse_id)` and a `register_learning_strategy(synapse_id, strategy)` method. Migrate all regions from the single-strategy pattern.

---

### 5.3 `DynamicBrain` Contains Region-Specific Business Logic

**Location:** `DynamicBrain.consolidate()`, `deliver_reward()`, `_compute_intrinsic_reward()`, `_get_cortex_l4_activity()`, `_get_hippocampus_ca1_activity()`

**Problem:** `DynamicBrain` is supposed to be a generic simulation orchestrator (iterate all regions, route spikes, advance time). Instead it imports and type-checks for `Hippocampus`, `Striatum`, `RewardEncoder`, and `Cortex` by name, tightly coupling the orchestrator to specific region implementations. This means:
- You cannot run any brain without those regions
- Adding new region types is invisible to DynamicBrain (it won't coordinate them)
- Testing the orchestrator requires the full brain to be assembled

**Fix:** Move `consolidate()` into a `MemoryConsolidationController` class (or into `Hippocampus` itself with a `self.set_sleep_mode(True)` API). Move `deliver_reward()` to a `RewardSystem` helper that wraps `RewardEncoder` + `Striatum`. Remove `_compute_intrinsic_reward()` entirely (see 2.5). `DynamicBrain.forward()` should contain exactly: read delays → inject external inputs → read neuromodulator state → execute regions → write outputs → advance time.

---

### 5.4 `AxonalTractDict` Keyed by **Target** Instead of **Source**

**Location:** `DynamicBrain.__init__` and `AxonalTractDict` usage

**Problem:** `AxonalTractDict` is keyed by `(target_region, target_population)`. Biologically and architecturally, axonal tracts emanate from **source** populations — a given axon arises from one neuron population and may project to multiple targets. Keying by target makes it impossible for one source population to project to two different targets (which is the common case: e.g., L5 → Striatum *and* L5 → Thalamus simultaneously). Currently each such projection is a separate `AxonalTract` instance keyed to a different target, but the key direction is confusing and makes the architecture imply it's a receiver-side resource rather than a sender-side one.

**Fix:** Rekey `AxonalTractDict` by `SynapseId` (which contains full routing info: source, target, polarity). This is already the natural key used by `AxonalTractSourceSpec`. One `AxonalTract` per connection (source_pop → target_pop), keyed by the `SynapseId` that fully identifies the projection. `DynamicBrain.forward()` iterates all tracts, reads their delayed outputs (which are already tagged with `SynapseId`), and routes to the appropriate target region.

---

## 6. Architecture Issues — High Priority

### 6.1 `NeuralRegion.update_all_synaptic_weights()` Missing

**Location:** Call sites distributed across each region's `forward()` method

**Problem:** Each region independently calls its learning strategy's `update_weights()` method at the end of its `forward()`. There is no consistent protocol for when and how weights are updated. Some regions update within `forward()`, others don't update at all. Moving plasticity outside `forward()` is biologically important: synaptic weight changes should not affect the current timestep's dynamics (synaptic weights are considered fixed within a timestep — they only change because of the *previous* activity).

**Fix:** Add `NeuralRegion.update_all_synaptic_weights(neuromodulator_input)` called at the end of each region's `forward()` or as a separate `DynamicBrain.update_plasticity()` pass after all regions have run. Standardize the `LearningStrategy.update_weights(pre_spikes, post_spikes, neuromodulator_input, weights)` signature uniformly across all strategy types.

---

### 6.2 `ConnectionSpec` Does Not Carry STP Config

**Location:** `BrainBuilder.connect()`, `_create_axonal_tract()`

**Problem:** `BrainBuilder.connect()` always passes `stp_config=None` when calling `target_region.add_input_source()`. Short-term plasticity (STP) is biologically synapse-type and pathway-specific (mossy fibers are strongly facilitating; corticostriatal synapses are depressing). By never passing STP config from the builder, all externally wired connections have no STP. STP presets exist (`MOSSY_FIBER_PRESET`, `CORTICOSTRIATAL_PRESET`, `THALAMO_STRIATAL_PRESET`) but are only used for *internal* connections defined in region `__init__` methods.

**Fix:** Add `stp_config: Optional[STPConfig] = None` to `ConnectionSpec`. Add `stp_config` parameter to `BrainBuilder.connect()`. Pass it through to `target_region.add_input_source()`. Standardize biological STP presets per projection type in `BrainBuilder` preset methods (or in the connection spec as a named preset string).

---

### 6.3 Config Duplication: STDP Parameters in `NeuralRegionConfig`

**Location:** `NeuralRegionConfig` (lines ~120–145) vs `STDPConfig`

**Problem:** `NeuralRegionConfig` contains `tau_plus_ms`, `tau_minus_ms`, `a_plus`, `a_minus`, `eligibility_tau_ms`, `heterosynaptic_ratio` — all of which are also in `STDPConfig` and `LearningConfig`. The base config should not know about STDP; learning algorithms are pluggable and the region config should not bake in assumptions about which learning rule is used.

**Fix:** Remove STDP parameters from `NeuralRegionConfig`. Region-specific learning parameters should live exclusively in the learning strategy config objects (passed to the strategy constructor). The region config should only contain region-level properties: neuron parameters, homeostasis thresholds, gap junction settings.

---

### 6.4 Intra-Region Delays Bypass Axonal Infrastructure

**Location:** `Cortex`, `Hippocampus`, `Striatum`, `PFC`, `Thalamus` — all use `CircularDelayBuffer` directly for modeling axonal delays between their internal sub-populations

**Problem:** Many regions instantiate raw `CircularDelayBuffer` instances for modeling delays between their sub-populations (e.g., L4→L2/3 delay in Cortex, DG→CA3 in Hippocampus). This is the same mechanism as `AxonalTract` but without the unified routing infrastructure, delay-buffer registration, or temporal parameter update propagation. When `DynamicBrain.set_timestep()` is called, `AxonalTract.update_temporal_parameters()` updates tract buffers, but region-internal `CircularDelayBuffer` instances are not updated unless each region's `update_temporal_parameters()` explicitly handles them.

**Fix:** Either (a) expose a `register_internal_delay_buffer(name, buffer)` on `NeuralRegion` so that `update_temporal_parameters()` can iterate and update all registered buffers, or (b) replace region-internal buffers with lightweight `InternalAxonalTract` objects that participate in the same update protocol. Option (a) is lower-risk.

---

### 6.5 Population Name Type Safety

**Location:** All region constructors, `BrainBuilder.connect()`

**Problem:** Population names in `population_sizes` dict use string keys, and the `BrainBuilder.connect()` API takes `source_population` and `target_population` as plain strings. Nothing prevents typos (`"l23"` vs `"L23"` vs `"l2_3"`). The `PopulationName` type alias is `str`, providing no static enforcement. The `population_names.py` enums exist but are not used for validation.

**Fix:** Change `BrainBuilder.connect()` to accept the typed enum values (or validate against the region's registered populations at the time of connection). `get_population_size()` should validate the population exists at call time — it already does this, providing runtime safety, but connecting with a nonexistent population name fails only at `build()` time, not at connection-definition time.

---

## 7. Architecture Issues — Medium Priority

### 7.1 `neuromodulator_outputs` ClassVar Has No Interface Contract

**Location:** `VTA`, `LC`, `NucleusBasalis`, `MedialSeptum`

**Problem:** The `neuromodulator_outputs` ClassVar (dict mapping modulator key to output population name) is declared informally via `hasattr` checks in `DynamicBrain`. There is no abstract base class or Protocol that enforces which regions must declare it, what the dict structure must look like, or how it interacts with the `NeuromodulatorReceptor` at the receiving end.

**Fix:** Create a `NeuromodulatorSource` mixin class (or a Protocol) with the `neuromodulator_outputs: ClassVar[Dict[str, str]]` declaration. Neuromodulator-producing regions inherit from this mixin. `DynamicBrain` can then use `isinstance(region, NeuromodulatorSource)` instead of `hasattr`. This also enables static type checking.

---

### 7.2 BrainBuilder Validation Is Insufficient

**Location:** `BrainBuilder.validate()`

**Problem:** The current `validate()` only checks for isolated regions (no connections). It should also verify:
- All named source populations exist in the source region
- All named target populations exist in the target region
- `n_input` for external inputs is > 0
- Axonal delay > 0 (zero delay implies instantaneous signaling, which violates causality)
- No self-loops that could cause immediate recurrence within a single timestep without delay

**Fix:** Add these checks to `validate()`. Call `validate()` automatically at the start of `build()` and raise on errors (currently it returns warnings).

---

### 7.3 `AxonalTractSourceSpec` Encodes Delay Twice

**Location:** `AxonalTractSourceSpec`, `AxonalTract.__init__`

**Problem:** `AxonalTractSourceSpec` has `delay_ms` and `delay_std_ms`. The `AxonalTract` constructor divides by `dt_ms` to get integer delay steps. However, if `dt_ms` changes via `set_timestep()`, the delay *in steps* changes (because `delay_ms` stays fixed but `dt_ms` changes), requiring the buffer to be reallocated. Currently `update_temporal_parameters()` on `AxonalTract` does not reallocate delay buffers; it only updates STP modules. This means changing `dt_ms` while a model is running could silently produce the wrong delays.

**Fix:** Store `delay_ms` (not `delay_steps`) as the authoritative value. On each `update_temporal_parameters()`, recompute and reallocate the `CircularDelayBuffer` with the new step count. For `HeterogeneousDelayBuffer`, re-sample delays from the same distribution (or detect the change and warn).

---

## 8. Missing Brain Regions

### Tier 1 — Critical Circuit Gaps

| Region | Missing Mechanism | Blocked Circuit |
|--------|------------------|-----------------|
| **Amygdala** (BLA + CeA) | Emotional learning, CS-US association, valence | LC/NB "amygdala inputs" are unimplemented; fear/reward learning broken |
| **Lateral Habenula (LHb)** | Negative RPE → DA pause | VTA negative PE pathway (RPE is one-sided positive only) |
| **RMTg** (Rostromedial Tegmental Nucleus) | Relay LHb→VTA inhibition | LHb output has nowhere to go |
| **Subthalamic Nucleus (STN)** | Indirect + hyperdirect BG pathway | D2 pathway terminally simplified in SNr |
| **Globus Pallidus External (GPe)** | D2→GPe→STN indirect pathway | Full BG loop impossible without it |
| **Inferior Olive (IO)** | Climbing fiber error signals, complex spikes | Cerebellar supervised learning has no proper error source or nucleo-olivary feedback |

### Tier 2 — Architecturally Necessary

| Region | Missing Mechanism | Impact |
|--------|------------------|--------|
| **Dorsal Raphe Nucleus (DRN)** | Serotonin (5-HT): patience, mood | Temporal discounting is hardcoded γ, not adaptive |
| **Pontine Nuclei** | Cortex→Cerebellum relay | Properly modeled cortico-cerebellar loops |
| **Subiculum** | CA1 output gateway, rate coding | CA1→cortex projection bypasses subiculum gating |
| **Globus Pallidus Internal (GPi)** | Second BG output (motor pathway) | Motor channel BG loop through VA/VL thalamus |

### Tier 3 — Future Completeness

| Region | Notes |
|--------|-------|
| **Anterior Cingulate Cortex (ACC)** | Conflict detection, intrinsic reward; alternative to `_compute_intrinsic_reward()` |
| **Superior Colliculus** | SNr already projects here; orienting/saccade control |
| **Anterior Insula** | Interoception, body state, surprise |
| **Septal-Hippocampal Nucleus** | Lateral septum for spatial memory output routing |
| **Multiple cortical areas** | V1, V2, IT, motor cortex hierarchy |

---

## 9. What Is Working Well

These aspects are solid and should be preserved:

- **Conductance-based LIF neurons** with proper reversal potentials, shunting inhibition, and refractory periods — biologically well-grounded.
- **D1/D2 asymmetric learning** (D1STDPStrategy / D2STDPStrategy) with multi-timescale eligibility traces (fast ~500ms, slow ~60s) — matches Yagishita et al. 2014 data well.
- **SynapseId routing key** — encoding full routing metadata (source region, source pop, target region, target pop, polarity) in one object that propagates through the entire spike pipeline is an elegant design that enables O(1) routing without reconstruction.
- **AxonalTract heterogeneous delays** with per-neuron sampling from Gaussian distribution — biologically appropriate wiring variability.
- **STP presets** (mossy fiber, Schaffer collateral, corticostriatal, thalamocortical) with biologically calibrated facilitation/depression parameters.
- **NeuromodulatorReceptor** with per-neuron concentration dynamics, receptor saturation, and desensitization — more realistic than simple scalar modulation.
- **Hippocampal SynapticTagging** and `SpontaneousReplayGenerator` — the tag-based consolidation mechanism and sharp-wave ripple replay are well-specified.
- **BrainBuilder fluent API** — clean progressive construction with validation before building.
- **CorticalInhibitoryNetwork** PV+ FSI implementation with Hebbian lateral inhibition.
- **VectorizedPurkinjeLayer** for cerebellar efficiency — good performance design.
- **Region registry with `@register_region` decorator** — proper plugin architecture for extensibility.

---

## 10. Implementation Plan

Each phase is **self-contained**: the brain runs correctly after each phase completes.

---

### Phase 1 — Architecture Foundations ✅ COMPLETE

*Goal: Fix architectural debt that makes every subsequent phase harder.*

**1.1 — Unify Neuromodulator Routing** ✅

Remove `NeuromodulatorTract` and the `neuromodulator_tracts` `ModuleDict` from `DynamicBrain`. Remove STEP 3 of `DynamicBrain.forward()` (the broadcast loop). Modify `DynamicBrain.forward()` so that for each region, the `SynapticInput` dict it receives already contains neuromodulator spike entries (with their `SynapseId`). Each target region's `NeuromodulatorReceptor.step()` is called via the existing `forward()` path using the spike data from the axonal tract. The `NeuromodulatorReceptor` already implements IIR filtering for receptor kinetics — this is sufficient; the extra `NeuromodulatorTract` layer should be removed.

Add a `NeuromodulatorSource` mixin Protocol:
```python
class NeuromodulatorSource(Protocol):
    neuromodulator_outputs: ClassVar[Dict[str, str]]
```
Replace all `hasattr(region, 'neuromodulator_outputs')` checks with `isinstance(region, NeuromodulatorSource)`.

**1.2 — Move Encoding/Decoding onto `SynapseId`** ✅

Add `to_key() -> str` and `from_key(s: str) -> SynapseId` directly to `SynapseId`. The encoding is a property of the type being serialized — it has no business living in the containers that use it. Every `_encode`/`_decode` static method across `SynapseIdParameterDict`, `SynapseIdModuleDict`, and `SynapseIdBufferDict` is replaced by a single call to the method on the instance or class:

```python
# Before (duplicated in 3 container classes)
@staticmethod
def _encode(s: SynapseId) -> str:
    inh = "1" if s.is_inhibitory else "0"
    return f"{s.source_region}|{s.source_population}|{s.target_region}|{s.target_population}|{inh}"

# After (on SynapseId itself)
@dataclass(frozen=True)
class SynapseId:
    ...
    def to_key(self) -> str:
        inh = "1" if self.is_inhibitory else "0"
        return f"{self.source_region}|{self.source_population}|{self.target_region}|{self.target_population}|{inh}"

    @classmethod
    def from_key(cls, key: str) -> "SynapseId":
        src_r, src_p, tgt_r, tgt_p, inh = key.split("|")
        return cls(source_region=src_r, source_population=src_p,
                   target_region=tgt_r, target_population=tgt_p,
                   is_inhibitory=(inh == "1"))
```

All three container classes then become trivial wrappers — their `__setitem__`/`__getitem__` call `key.to_key()` and their iterators call `SynapseId.from_key(k)`. Any future change to the wire format (e.g., adding a `synapse_type` field in Phase 10) is a single edit in `SynapseId`, and the type checker will catch any callers that construct keys manually.

**1.3 — Remove Single-Strategy Variable from `NeuralRegion`** ✅

Remove `self.learning_strategy: Optional[LearningStrategy]`. Migrate all regions that currently use it (check each `forward()` for `self.learning_strategy.update_weights()`): add the strategy to `_learning_strategies` with the appropriate synapse id. Add convenience methods to `NeuralRegion`:
```python
def register_learning_strategy(self, synapse_id: SynapseId, strategy: LearningStrategy) -> None
def get_learning_strategy(self, synapse_id: SynapseId) -> Optional[LearningStrategy]
```

**1.4 — Rekey `AxonalTractDict` to `SynapseId`** ✅

Change `AxonalTractDict` to key by `SynapseId` instead of `(RegionName, PopulationName)`. Update `DynamicBrain.__init__`, `DynamicBrain.forward()` write step, and `BrainBuilder._create_axonal_tract()`. This is a refactor with no behaviour change: one tract per projection, keyed by the SynapseId that already fully identifies it.

**1.5 — Extract Brain-Level Logic from `DynamicBrain`** ✅

Move `consolidate()` to `Hippocampus` as `hippocampus.consolidate(duration_ms)`. Keep a thin shim `brain.consolidate()` that delegates to the hippocampus region if present. Move `deliver_reward()` to a `RewardSystem` helper class. Remove `_compute_intrinsic_reward()`, `_get_cortex_l4_activity()`, and `_get_hippocampus_ca1_activity()` — these will be replaced by proper circuits in later phases.

**1.6 — Register Internal Delay Buffers** *(deferred to Phase 10)*

Add to `NeuralRegion`:
```python
def _register_delay_buffer(self, name: str, buffer: CircularDelayBuffer | HeterogeneousDelayBuffer) -> None
def update_temporal_parameters(self, dt_ms: float) -> None  # override to call super() + update registered buffers
```
All regions' internal `CircularDelayBuffer` instances (L4→L2/3 in Cortex, DG→CA3 in Hippocampus, etc.) should be registered via `_register_delay_buffer()`. The base `update_temporal_parameters()` iterates and updates them.

---

### Phase 2 — NMDA Voltage-Gating (Critical Biological Fix) ✅ COMPLETE

*Goal: Restore proper coincidence detection.*

**2.1 — Implement Mg²⁺ Unblock in `ConductanceLIF`** ✅

Add normalized voltage-dependent NMDA gate to `ConductanceLIF.step()`. The effective NMDA conductance: `g_nmda_eff = g_nmda * sigmoid(k * (V_norm - V_half))` where `k ≈ 5.0` and `V_half ≈ 0.3` in normalized units (calibrated to match biological 50% unblock at -30mV relative to resting). This is a smooth approximation that avoids the sharp threshold but captures the voltage-dependence.

Raise the default `nmda_ratio` in `_split_excitatory_conductance()` from `0.0` to `0.20` and do a quick diagnostic run to confirm stable network dynamics.

**2.2 — Validate STDP Coincidence Detection is Restored** ✅

Write a unit test: two populations with a single synapse, fire pre before post (LTP scenario). Confirm weight increases only when NMDA unblock occurs (post is depolarized), not when post is silent.

---

### Phase 3 — Basal Ganglia Completion

*Goal: Complete the indirect and hyperdirect pathways for proper action selection.*

**3.1 — Add `GlobuspallidusExternal (GPe)`**

New region `src/thalia/brain/regions/globus_pallidus/gpe.py`. GABAergic neurons (high tonic ~50Hz baseline). Receives: D2-MSN inhibition. Projects: STN inhibition, GPi feedback inhibition (arkypallidal cells → Striatum, prototypic cells → STN). Two neuron subtypes: `prototypic` (→STN, →GPi) and `arkypallidal` (→Striatum feedback inhibition).

**3.2 — Add `SubthalamicNucleus (STN)`**

New region. Glutamatergic neurons (~12,000 in humans). Inputs: GPe inhibition (pausing activity), cortex direct (hyperdirect path, fast). Outputs: SNr excitation (strong), GPe feedback (recurrent). This creates oscillatory dynamics in the BG when properly wired. The hyperdirect cortex→STN→SNr projection (~5ms) is faster than cortex→Striatum→SNr (~50ms+), providing the "global suppression" brake before action selection.

**3.3 — Add `GlobuspallidusInternal (GPi)`**

New region. Second BG output nucleus (motor loop, parallel to SNr for limbic/cognitive loop). GABAergic. Receives: D1-MSN (direct), STN (indirect), GPe. Projects: Thalamus VA/VL (motor gating). Required for full motor action selection.

**3.4 — Update `SubstantiaNigra` (SNr) Connections**

Remove the `D2_direct_bypass` that currently connects D2-MSNs directly to SNr. Route D2→GPe→STN→SNr properly. Test indirect pathway: D2 activation should increase SNr firing with a delay (not immediately). Test hyperdirect pathway: cortical stimulation should cause rapid SNr burst before striatal integration.

**3.5 — Update `BrainBuilder` presets to wire full BG loop**

Update the complete brain preset to include GPe, STN, GPi with correct delays and connectivity strengths (using biological references for each projection).

---

### Phase 4 — Lateral Habenula + Negative RPE

*Goal: Complete the dopamine system's prediction error signal.*

**4.1 — Add `LateralHabenula (LHb)`**

New region. Glutamatergic neurons with high baseline (~5Hz). Inputs: GPi/SNr (encodes negative PE: "worse than expected"), contextual from PFC. Output: RMTg excitation. The LHb fires strongly when expected reward is not delivered or punishment is received. Crucially, its activity is antiphasic to VTA DA neurons.

**4.2 — Add `RMTg` (Rostromedial Tegmental Nucleus)**

New region. GABAergic relay. Receives LHb excitation. Projects to VTA as strong inhibition — this causes the DA pause on negative PE. The RMTg provides the pause asymmetry: positive PEs are fast (direct reward → VTA), negative PEs go through LHb→RMTg→VTA (with a short delay ~5ms).

**4.3 — Update `VTA` to Receive RMTg Inhibition**

Add `rmt_inhibitory` population as an input to VTA. Remove the simplified scalar RPE computation. The RPE now emerges entirely from spiking circuit dynamics:
- Positive RPE: Reward encoder → VTA→ DA burst
- Negative RPE: LHb → RMTg → VTA inhibition → DA pause
- Baseline: Tonic DA pacemaker activity

---

### Phase 5 — Cortical Disinhibition Circuit

*Goal: Implement the VIP→SOM→Pyramidal disinhibition substrate for attention.*

**5.1 — Add SOM+ Interneurons to `CorticalInhibitoryNetwork`**

Add `SOMPopulation` alongside `FSIPopulation`. SOM+ neurons target apical dendrites (not soma) — model this by having SOM+ output go to a separate `g_inh_apical` conductance channel that is processed differently by the neuron model (it reduces the contribution of NMDA-mediated plateau potentials rather than just hyperpolarizing soma).

**5.2 — Add VIP Interneurons to `CorticalInhibitoryNetwork`**

Add `VIPPopulation`. VIP neurons receive: ACh drive (from NB), top-down input from PFC, and feedback from L2/3 pyramidal cells. VIP neurons project onto SOM+ and PV/FSI neurons (disinhibitory), NOT onto pyramidal cells directly.

**5.3 — Wire ACh → VIP in Cortex**

When `NeuromodulatorReceptor` for ACh activates in cortex, its output should drive VIP population activity. This creates the chain: NB ACh burst → VIP activation → SOM/PV inhibition → pyramidal disinhibition.

---

### Phase 6 — Amygdala

*Goal: Emotional learning, valence, and anxiety modulation.*

**6.1 — Add `BasolateralAmygdala (BLA)`**

New region. Excitatory principal cells (~70%) + PV interneurons (~30%). Inputs: cortex (CS sensory associations), hippocampus CA1 (context), VTA (US reward/punishment). Output: diverse — see 6.3. Learning: three-factor STDP with surprise signal (CS arrives before US, US delivery creates DA/NE surprise, potentiates CS→BLA weights).

**6.2 — Add `CentralAmygdala (CeA)`**

New region. GABAergic. Receives BLA input. Projects to brainstem (autonomic fear output). The BLA→CeA pathway is the standard fear expression circuit.

**6.3 — Wire Amygdala into Existing Circuits**

- BLA → LC: Emotional salience → NE burst (arousal)
- BLA → NB: Emotional salience → ACh burst (attentional capture)
- BLA → Striatum D1/D2: Emotional bias of action value
- BLA → Hippocampus CA1: Stress hormones → memory encoding enhancement during emotional events
- BLA → VTA: Learned reward signal (conditioned reinforcement)
- BLA → PFC executive: Emotional context for cognitive control

---

### Phase 7 — Cerebellar Inferior Olive

*Goal: Proper supervised learning signal for cerebellum.*

**7.1 — Add `InferiorOlive (IO)` Region**

New region. Inferior olive neurons fire at ~1Hz (complex spikes) synchronized by gap junctions (~40% coupling density). Inputs: brainstem motor error comparison, nucleo-olivary inhibitory feedback from DCN (-inhibitory, reduces IO output after learning occurs). Output: climbing fiber input to Purkinje cells (complex spike trigger).

**7.2 — Add `PontineNuclei` Region**

Extract the current inline `n_mossy` mossy fiber layer from `Cerebellum.__init__` into a proper `PontineNuclei` region. Pontine nuclei receive cortical input and project to granule cells as mossy fibers.

**7.3 — Add Nucleo-Olivary Feedback**

Wire DCN → IO inhibitory projection. This creates the adaptive timing circuit: the more the DCN fires, the more IO is suppressed, reducing complex spike rate, reducing Purkinje LTD, allowing granule→Purkinje weights to recover.

---

### Phase 8 — Hippocampal Circuit Completion

*Goal: Add subiculum and fix the theta oscillator hierarchy.*

**8.1 — Add Subiculum Population to `Hippocampus`**

Add `subiculum` population. Route `CA1 → Subiculum → EC/Cortex` instead of `CA1 → Cortex` directly. Subiculum neurons can be modeled as a simple rate-transforming layer with burst and regular-firing subtypes.

**8.2 — Remove Internal Theta Oscillator from `Hippocampus`**

Delete the internal phase counter from `Hippocampus`. Instead, the hippocampus detects theta phase from the incoming MS spike pattern: MS ACh neurons fire during encoding phase, MS GABA neurons fire during retrieval phase. The hippocampal theta phase becomes emergent from MS→CA3/CA1 drive.

**8.3 — Separate MEC and LEC Inputs**

Split the single `ec` input population name into `ec_mec` (targets DG, CA3 direct path) and `ec_lec` (targets CA1 direct path) with separate synaptic weight matrices. Update BrainBuilder cortex→hippocampus wiring to specify which input carries spatial (MEC) vs. object (LEC) information.

---

### Phase 9 — Serotonin System

*Goal: Add adaptive temporal discounting and mood modulation.*

**9.1 — Add `DorsalRapheNucleus (DRN)` Region**

New region. Serotonergic neurons (~200,000 in humans). Inputs: LHb (punishment drives DRN quiescence), PFC, hypothalamus. Outputs: cortex, hippocampus, striatum, amygdala. Add `'5ht'` to the neuromodulator type registry. Implement `5-HT1A` (autoreceptor, inhibitory) and `5-HT2A` (excitatory, cortical) receptor types in `NeuromodulatorReceptor`.

**9.2 — Implement Temporal Discounting Modulation**

5-HT level modulates the effective discount factor γ in striatal three-factor STDP. High 5-HT → γ closer to 1.0 (patient, long-horizon), Low 5-HT → γ closer to 0.9 (impulsive, short-horizon). This is implemented by having the DRN 5-HT concentration scale the eligibility trace decay rate in `D1STDPStrategy` / `D2STDPStrategy`.

---

### Phase 10 — Architecture Polish

*Goal: Clean up remaining tech debt introduced by phases above.*

**10.1 — Add `SynapseType` to `SynapseId`**

Extend `SynapseId` with `synapse_type: SynapseType = SynapseType.GLUTAMATE_AMPA`. Update `_encode`/`_decode` across all containers. This enables per-synapse type-specific kinetics routing without needing to split conductances post-hoc.

**10.2 — `OscillationCoordinator`**

Add a lightweight `OscillationCoordinator` to `DynamicBrain` that tracks oscillatory phase for each registered oscillator (theta from MS, gamma from cortex FSI, beta from BG). Exposes `get_phase(oscillator_name) -> float`. Enables theta-gamma nesting: cortical gamma resets at each MS theta peak.

**10.3 — Full BrainBuilder Validation**

Implement the complete validation checklist in `BrainBuilder.validate()`: population existence check, delay > 0 check, weight scale range, no zero-delay self-loops. Call `validate()` automatically at `build()` start with `raise` on errors (not just warnings).

**10.4 — Generalize `SynapticTagging` to `NeuralRegion`**

Move `SynapticTagging` from the hippocampus-only implementation into `NeuralRegion` as optional component (`self.synaptic_tagging: Optional[SynapticTagging] = None`). Enable for Cortex L2/3 synapses. This supports cortical late-phase LTP during system consolidation (hippocampus→cortex replay potentiating cortical synapses).

---

## Summary Table

| Phase | Description | Biological Priority | Effort |
|-------|-------------|--------------------|-|
| 1 | Architecture foundations | ★★★★★ | ✅ Done |
| 2 | NMDA voltage-gating | ★★★★★ | ✅ Done |
| 3 | Basal ganglia completion (GPe, STN, GPi) | ★★★★★ | Large |
| 4 | LHb + RMTg + negative RPE | ★★★★★ | Medium |
| 5 | Cortical disinhibition (SOM+/VIP) | ★★★★☆ | Medium |
| 6 | Amygdala (BLA + CeA) | ★★★★★ | Large |
| 7 | Inferior Olive + Pontine Nuclei | ★★★★☆ | Medium |
| 8 | Subiculum + theta hierarchy | ★★★☆☆ | Small |
| 9 | Serotonin (DRN) | ★★★☆☆ | Medium |
| 10 | Architecture polish | ★★★☆☆ | Small |

---

*References: Jahr & Stevens (1990) NMDA Mg²⁺ block; Turrigiano & Nelson (2004) homeostatic plasticity; Yagishita et al. (2014) striatal eligibility traces; Hasselmo & McGaughy (2004) ACh and cortical function; Daw et al. (2002) serotonin and patience; Magill et al. (2004) STN hyperdirect pathway; Hikosaka et al. (2010) LHb and negative RPE; Cardin et al. (2009) PV/SOM/VIP interneuron circuit roles.*
