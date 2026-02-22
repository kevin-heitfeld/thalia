# Brain Architecture Review: Implementation Plan

**Date**: February 22, 2026
**Scope**: Full review of `src/thalia/brain/` — neuroscience accuracy, software architecture, and refactoring opportunities.

---

## Executive Summary

Thalia has a remarkably strong biological foundation: conductance-based LIF neurons, STDP + BCM + three-factor learning, multi-type inhibitory networks (PV/SST/VIP), theta-modulated hippocampal encoding/retrieval, structured neuromodulator volume transmission, and explicit axonal delays. The architecture already surpasses most published SNN frameworks in biological plausibility.

However, there are **three critical bugs** that corrupt computation silently, **several missing brain structures** that are essential for the stated goals, and **significant code duplication** that makes maintenance error-prone. This plan is organized by priority.

---

## Part 1 — Critical Bugs (Break Biology, Fix Immediately)

### BUG-1: Inhibitory Conductances Silently Dropped in Base Class

**File**: `src/thalia/brain/regions/neural_region.py`, `_integrate_synaptic_inputs_at_dendrites()`

**Problem**: The `is_inhibitory` branch in `_integrate_synaptic_inputs_at_dendrites` is `pass` — doing nothing. Every call to this method from a region's `forward()` **silently discards all inhibitory conductances**. Regions that rely on the base integration method (Striatum, Thalamus, Hippocampus for some paths) do not actually apply the inhibitory portion of their registered internal connections (`FSI→D1`, `TRN→relay`, etc.) through this path.

```python
if synapse_id.is_inhibitory:
    # TODO: Also integrate inhibitory conductance here?
    pass   # ← ALL INHIBITORY CONDUCTANCE DISCARDED
```

The Cortex forward pass works around this by computing inhibitory conductances entirely outside the base method (via `_compute_layer_inhibition`). But any region using `_integrate_synaptic_inputs_at_dendrites` for inhibitory synapses is silently broken.

**Fix**:

The method signature and return type must change. Return a named tuple `(g_exc, g_inh)` and accumulate both separately. Remove all `filter_by_*` parameters in favour of explicit calls per population.

```python
@dataclass
class DendriteOutput:
    g_exc: torch.Tensor
    g_inh: torch.Tensor

def _integrate_synaptic_inputs_at_dendrites(
    self,
    synaptic_inputs: SynapticInput,
    n_neurons: int,
    *,
    filter_by_target_population: Optional[PopulationName] = None,
) -> DendriteOutput:
    g_exc = torch.zeros(n_neurons, device=self.device)
    g_inh = torch.zeros(n_neurons, device=self.device)
    for synapse_id, source_spikes in synaptic_inputs.items():
        ...
        weights = self._synaptic_weights[synapse_id]
        conductance = weights @ source_spikes.float()
        if synapse_id.is_inhibitory:
            g_inh += conductance
        else:
            # apply STP, accumulate
            g_exc += conductance
    g_exc = g_exc.clamp(min=0.0)
    g_inh = g_inh.clamp(min=0.0)
    return DendriteOutput(g_exc=g_exc, g_inh=g_inh)
```

Update all call sites to unpack `(g_exc, g_inh)`. The Cortex code does its own inhibitory computation — leave that as-is but also profit from the unified path for internal inhibitory synapses.

### BUG-2: Critical State Tensors Not Registered as Buffers

**Files**: Multiple regions

The following tensors are plain Python attributes, **not** registered via `register_buffer`. This means they are invisible to `.to(device)`, `state_dict()`, and `load_state_dict()`. On GPU transfer or model save/load these are silently lost.

| Region | Tensor | Problem |
|---|---|---|
| `hippocampus.py` | `ca3_persistent` | Critical attractor state |
| `hippocampus.py` | `nmda_trace` | NMDA temporal integration |
| `hippocampus.py` | `stored_dg_pattern` | Pattern comparison state |
| `hippocampus.py` | `_homeostasis_firing_history` | Python list, not tensor |
| `hippocampus.py` | `_dg_ca3_fast`, `_dg_ca3_slow` | Consolidation weights |
| `striatum.py` | `_da_concentration_d1/d2` | DA gating state |
| `striatum.py` | `_ne_concentration_d1/d2` | NE modulation state |
| `thalamus.py` | `input_to_trn` | Weight tensor (not a parameter!) |
| `thalamus.py` | `_last_trn_spikes` | Not registered |
| `cortex.py` | `_da_concentration_l*` | Per-layer concentration |
| `cortex.py` | `_ne_concentration_l*` | Per-layer concentration |
| `cortex.py` | `_ach_concentration_l*` | Per-layer concentration |
| `prefrontal.py` | Various WM state tensors | Check needed |

**Fix**: Call `self.register_buffer('name', tensor)` for every persistent state tensor that should survive `.to(device)` or `state_dict()`. For tensors that should not be saved (e.g., transient spike buffers), use `register_buffer('name', tensor, persistent=False)`. Convert `_homeostasis_firing_history` from a Python list to a pre-allocated `torch.Tensor` ring buffer.

Also: `thalamus.input_to_trn` is a weight matrix (not a delay buffer) that should be an `nn.Parameter` or at minimum a registered buffer. Currently it is neither.

### BUG-3: Cerebellum Uses Biological mV Units, Not Normalized Units

**File**: `src/thalia/brain/regions/cerebellum/cerebellum.py`

```python
granule_config = ConductanceLIFConfig(
    v_threshold=-50.0,   # ← biological mV unit
    v_reset=-65.0,       # ← biological mV unit
    tau_mem=5.0,
    ...
)
```

Every other region uses normalized units where `v_rest=0.0`, `v_threshold=1.0` (as documented in `ConductanceLIFConfig`). The granule cell config passes `-50.0` as threshold in a field expecting a normalized scalar, making these neurons functionally broken (they will never spike since `v_threshold` is far below `v_rest=0.0`).

**Fix**: Normalize cerebellum to the same unit convention. Granule cell properties in normalized units:
- `v_threshold = 0.85` (excitable, lower than pyramidal 1.0)
- `v_reset = 0.0`
- `tau_mem = 5.0` ms (correct, fast)
- `tau_E = 2.5` ms (correct)
- `tau_I = 6.0` ms (correct)

### BUG-4: `is_inhibitory=False` Hardcoded in Axonal Tract Routing

**File**: `src/thalia/brain/brain.py`, `forward()`

```python
synapse_id = SynapseId(
    ...
    is_inhibitory=False,  # TODO: Support inhibitory tracts in the future
)
```

This makes it impossible to define long-range inhibitory projections (e.g., SNr→thalamus, which is the primary thalamic gating mechanism of the basal ganglia). The entire SNr output is functionally silent because its inhibitory axonal projection to Thalamus cannot be represented.

**Fix**: Add `is_inhibitory: bool = False` to `AxonalTractSourceSpec` and propagate it to the `SynapseId` created in the brain's `forward()`. Update `BrainBuilder.connect()` to accept an `is_inhibitory` argument. Document that inhibitory long-range projections are now supported.

---

## Part 2 — High-Priority Neuroscience Gaps

### NEU-1: Missing Entorhinal Cortex (EC)

**Severity**: High — EC is the primary interface between neocortex and hippocampus.

**Current state**: Hippocampus receives inputs labeled as coming directly from `cortex`. In biology, cortical output goes to **Entorhinal Cortex layer II/III** first, which then projects to hippocampus via the perforant path:

```
Neocortex → Entorhinal Cortex (EC) → Hippocampus (DG, CA3, CA1)
```

EC performs:
1. **Spatial grid cells** (medial EC): Spatial context encoding
2. **Object/scene cells** (lateral EC): Semantic content encoding
3. **Temporal context**: Timestamps for episodic sequences
4. The **CA1 direct perforant path** from EC bypasses DG/CA3 entirely and provides the "true memory" comparison signal for match/mismatch detection

**Fix**:
- Create `EntorhinalCortex` region with two populations: `mec` (medial) and `lec` (lateral)
- `mec` neurons implement grid-cell-like spatial coding (hexagonal periodic firing using Fourier basis or competitive learning)
- `lec` neurons implement object/scene binding (simpler sparse coding)
- Connect BrainBuilder presets: `Cortex.L23 → EC.mec+lec → Hippocampus.DG+CA3+CA1`
- The direct EC→CA1 perforant path is critical for Novelty/Familiarity comparison in CA1

### NEU-2: Missing Lateral Habenula (LHb) → RMTg → VTA Aversion Pathway

**Severity**: High — Negative reward prediction errors require this circuit. Currently VTA receives only positive reward signals.

**Current state**: VTA computes RPE as `reward - SNr_value`. The "reward" signal comes entirely from `RewardEncoder`, which encodes positive rewards. The LHb→RMTg pathway is what actually computes negative PE.

**Biology**:
- LHb neurons burst for **worse-than-expected** outcomes (negative RPE)
- LHb → RMTg (rostromedial tegmental nucleus) → VTA GABAergic neurons
- VTA GABA neurons pause VTA DA neurons → DA pause = negative RPE signal

**Fix**:
- Create `LateralHabenula` region with inputs from PFC (predicted value) and reward system (actual reward)
- Connect LHb → `VTA.gaba_neurons` (which inhibit `VTA.da_neurons`)
- Update RewardEncoder to model both positive valence neurons (connected to VTA directly) and negative valence neurons (connected to LHb)
- Alternatively, integrate LHb into the existing VTA region as a subpopulation model

### NEU-3: Missing Serotonin (5-HT) / Raphe Nuclei

**Severity**: High — 5-HT modulates time discounting, punishment aversion, and patience. DA and 5-HT are opponent systems in RL.

**Current state**: `NeuromodulatorType` includes `'5ht'` as a type alias but no region produces it, no receptor processes it.

**Biology**:
- Dorsal Raphe Nucleus (DRN) produces serotonin
- 5-HT encodes **patience** and **future value discounting** (high 5-HT = more patient)
- Opponent to DA: DA signals immediate reward, 5-HT signals delayed punishment avoidance
- Modulates: Cortex (impulsivity), Striatum (D1/D2 balance), Hippocampus (context)

**Fix**:
- Create `DorsalRaphe` region with spiking 5-HT neurons
- Add `NeuromodulatorReceptor` for `'5ht'` in Cortex, Striatum, Hippocampus
- 5-HT modulates STDP time windows (higher 5-HT → longer eligibility traces → longer-horizon credit assignment)
- Add `5ht_receptor` to `NeuralRegionConfig` analog

### NEU-4: Missing Amygdala

**Severity**: Medium-High — Fear learning, emotional memory tagging, prediction-error-gated hippocampal encoding.

**Biology**:
- Basolateral Amygdala (BLA): Fear conditioning, reward association learning
- BLA projects to: Hippocampus (strengthens emotional memories), PFC (emotion-cognition interaction), Striatum (emotional action selection)
- Inputs from: Thalamus (fast subcortical fear path), Cortex (slower cortical path)

**Fix**:
- Create `Amygdala` region with `bla` (basolateral) and `ceal` (central lateral) populations
- BLA learns via STDP the association between sensory stimuli and aversive/rewarding outcomes
- BLA modulates hippocampal encoding strength (emotional memory enhancement)
- BLA → NB (nucleus basalis) pathway: Emotional salience → cholinergic attention

### NEU-5: Incomplete Basal Ganglia — Missing GPe and STN

**Severity**: Medium — The indirect pathway (D2→GPe→STN→SNr) is modeled as a direct D2→SNr "excitation" in the current code, which is a significant simplification.

**Biology**:
- **Direct pathway**: Striatum D1 → SNr/GPi (inhibitory) → release thalamus (Go)
- **Indirect pathway**: Striatum D2 → GPe (inhibitory) → STN (release) → SNr/GPi (excitatory) → suppress thalamus (NoGo)
- **Hyperdirect pathway**: Cortex L5 → STN → SNr (fast NoGo, stops initiated actions)

The current implementation models D2 MSNs as directly exciting SNr, which gets the direction right but misses the important temporal dynamics of the D2→GPe→STN pathway (it adds an extra ~10ms delay).

**Fix**:
- Create `GlobusPallidus` region with `gpe` (external) population
- Create `SubthalamicNucleus` (STN) region
- Connect: `Striatum.D2 → GPe → STN → SNr` (replacing current direct D2→SNr excitation)
- Also connect `Cortex.L5 → STN` (hyperdirect pathway for action cancellation)

### NEU-6: Intrinsic Reward Computation is Biologically Wrong

**Severity**: Medium — The current `_compute_intrinsic_reward` in `DynamicBrain` misinterprets what L4 activity means.

**Current flaws**:

1. **L4 activity as prediction error**: `cortex_reward = 1 - 2 * l4_activity`. When `l4_activity = 0` → reward = +1.0. But zero L4 activity could mean either **perfect prediction** (good) OR **complete cortical silence/death** (catastrophically bad). These must be distinguished.

2. **CA1 firing rate as recall quality**: High CA1 rate does NOT necessarily mean good recall — it could mean runaway excitation. True recall quality should be measured as **pattern overlap** between current CA1 output and a stored target pattern, not as absolute firing rate.

3. **VTA should compute RPE, not the brain-level method**: The intrinsic reward computation is in `DynamicBrain._compute_intrinsic_reward()` as a Python float computation, disconnected from the biological reward circuit. VTA DA neurons should be the source of intrinsic reward signals.

**Fix**:
- Remove `_compute_intrinsic_reward()` from `DynamicBrain` entirely
- Move intrinsic reward computation into VTA's forward pass as a proper spiking computation
- VTA receives: prediction error from EC/Cortex (novel/surprising stimuli → DA burst), pattern completion success from CA1 (via an EC→VTA loop)
- Add a proper novelty signal: compute KL-divergence or cosine distance between predictions and observations inside VTA or a dedicated novelty detector

### NEU-7: Theta Phase in Hippocampus Uses Internal Sinusoid, Not Septal Drive

**Severity**: Medium — The medial septum region exists but it's unclear whether it actually drives hippocampal theta via axonal connections, or whether Hippocampus uses an internal theta phase counter.

Looking at the code, hippocampus has its own `theta_phase` state and advances it internally. If it does, this is disconnected from the `MedialSeptum` region, making the septal region vestigial.

**Fix**:
- Verify that `MedialSeptum.forward()` outputs go through an AxonalTract to `Hippocampus`
- Hippocampus should derive its theta phase from the actual septal spike pattern (CA3 receives the septal ACh bursts, OLM interneurons respond to septal GABA bursts)
- Remove any internal theta sinusoid from Hippocampus — theta should emerge from septal drive

### NEU-8: STDP in Cortex Uses SAME Config for ALL Layers

**Severity**: Low-Medium — Different cortical layers have profoundly different plasticity rules biologically.

**Current state**: All five cortical layers (L2/3, L4, L5, L6a, L6b) use identical `stdp_cfg` and `bcm_cfg` instances. In biology:

- **L4**: Has little plasticity (mainly feedforward relay); STDP is symmetrical
- **L2/3**: Strong STDP with LTP bias (Hebbian association); BCM threshold slides fast
- **L5**: STDP gated by DA (for motor/prediction error learning)
- **L6**: Anti-Hebbian bias (predictions should SUPPRESS, not strengthen responses)

**Fix**: Create per-layer learning configurations using a `Dict[CortexLayer, STDPConfig]` in `CortexConfig`.

---

## Part 3 — Software Architecture Issues

### ARCH-1: Duplicated `_encode`/`_decode` Logic Across Four Containers

**Affected files**: `neural_region.py` (×3 classes) and `brain.py`

`SynapseIdParameterDict`, `SynapseIdModuleDict`, `SynapseIdBufferDict`, and `AxonalTractDict` all implement the same `|`-delimited string encoding. Any change to the encoding format (e.g., adding a field to `SynapseId`) requires updating four places.

**Fix**: Extract a shared mixin:

```python
class SynapseIdKeyMixin:
    """Shared encoding/decoding for SynapseId ↔ str."""

    _SEP = "|"

    @staticmethod
    def _encode(s: SynapseId) -> str:
        inh = "1" if s.is_inhibitory else "0"
        return f"{s.source_region}|{s.source_population}|{s.target_region}|{s.target_population}|{inh}"

    @staticmethod
    def _decode(key: str) -> SynapseId:
        src_r, src_p, tgt_r, tgt_p, inh = key.split("|")
        return SynapseId(src_r, src_p, tgt_r, tgt_p, is_inhibitory=(inh == "1"))
```

All four container classes inherit from this mixin.

### ARCH-2: `__call__` Override Breaking PyTorch Hook System

**Files**: `neural_region.py`, `brain.py`

```python
def __call__(self, *args, **kwds):
    assert False, "Use forward() instead."
```

This violates `nn.Module` conventions. PyTorch's hook system (`register_forward_hook`, `register_forward_pre_hook`) relies on `__call__` dispatching to `forward`. Debugging tools, profilers, and Captum-style attribution methods all assume standard `nn.Module` call semantics.

The intent (prevent accidental calling from training loops) is valid, but the implementation is too aggressive.

**Fix**: Remove the `__call__` override entirely. Add a docstring note to `forward()` stating that it must be called directly. If training-loop protection is needed, add a runtime assertion *inside* `forward()` that checks for gradient context (`torch.is_grad_enabled()`).

### ARCH-3: `configs.py` is a Monolithic God File (~1200 lines)

**File**: `src/thalia/brain/configs.py`

All region configs live in one 1200-line file. Adding or modifying configs for one region requires navigating a huge file, causing merge conflicts in multi-developer workflows.

**Fix**: Split into per-region config files co-located with each region:

```
brain/
  configs.py              ← BrainConfig + NeuralRegionConfig base only
  regions/
    cortex/
      cortex_config.py    ← CortexConfig, CortexLayer
    hippocampus/
      hippocampus_config.py
    striatum/
      striatum_config.py
    ...
```

All existing imports can be maintained via re-exports in `brain/configs.py` for backward compatibility during transition.

### ARCH-4: `_firing_rate_alpha` Uses Linear Approximation Instead of Correct EMA

**File**: `neural_region.py`

```python
self._firing_rate_alpha = self.dt_ms / self.config.gain_tau_ms
```

The correct EMA decay coefficient for a first-order system with time constant `τ` and timestep `dt` is:

```
α = 1 - exp(-dt / τ)
```

The linear approximation `dt/τ` is only accurate when `dt << τ`. At `dt=1ms` and `tau=2000ms`, the error is negligible, but at larger timesteps (dynamic timestep changes via `set_timestep()`) or shorter time constants, this becomes incorrect. This is inconsistent with how all other exponential decay constants are computed (the rest correctly use `math.exp(-dt/tau)`).

**Fix**:
```python
import math
self._firing_rate_alpha = 1.0 - math.exp(-self.dt_ms / self.config.gain_tau_ms)
```
Update `update_temporal_parameters` similarly.

### ARCH-5: `enable_neuromodulation` Defaults to `False` — A Footgun

**File**: `configs.py`

```python
enable_neuromodulation: bool = False
```

Neuromodulation is a core biological mechanism, not an optional feature. Having it default to `False` means that building a brain without explicitly setting `enable_neuromodulation=True` on every region config silently produces a brain with no neuromodulatory effects — which is almost certainly not the user's intent.

**Fix**: Change the default to `True`. If disabling neuromodulation is needed for ablation studies, it should be an explicit opt-out.

### ARCH-6: Population Name Enums are Centralized, Creating Coupling

**File**: `src/thalia/brain/regions/population_names.py`

Adding a new region requires editing this centralized file. If a region wants to add a new population name, it must do so in a separate file from the region itself, making the coupling non-obvious.

**Fix**: Move population name enums inside each region's `__init__.py` or module file. The `brain_builder.py` imports from `population_names` — these can be re-exported from each region's `__init__.py`. Consider deprecating the centralized `population_names.py` in favor of per-region enums.

### ARCH-7: `NeuralRegionConfig` Base Is Bloated

**File**: `configs.py`

`NeuralRegionConfig` contains many fields that aren't universally applicable:
- `gap_junction_strength/threshold/max_neighbors` — only used by regions WITH gap junctions
- `heterosynaptic_ratio` — only relevant to STDP
- `eligibility_tau_ms` — only relevant to three-factor rules
- `a_plus`, `a_minus`, `tau_plus_ms`, `tau_minus_ms` — only relevant to STDP

These belong in their respective learning strategy configs, not in the base region config.

**Fix**: Remove STDP-specific and gap-junction-specific fields from `NeuralRegionConfig`. Regions that need them should use typed config subclasses (`CortexConfig`, etc.) that include those fields explicitly. The base config should contain only universal parameters: `device`, `dt_ms`, `seed`, `w_min`, `w_max`, `learning_rate`, `target_firing_rate`, `gain_learning_rate`, `gain_tau_ms`.

### ARCH-8: `brain.deliver_reward()` and `brain.consolidate()` are Type-Specific

**File**: `src/thalia/brain/brain.py`

Both methods hardcode specific region types:
```python
striatum = self.get_first_region_of_type(Striatum)
if striatum is None:
    raise ValueError("Striatum not found...")
```

This tightly couples the top-level `DynamicBrain` to specific region implementations. If a user builds a brain without a `Striatum` (e.g., a sensory processing brain), `deliver_reward()` fails.

**Fix**: Replace with a protocol-based approach using Python `Protocol` typing:

```python
class SupportsRewardDelivery(Protocol):
    def receive_reward(self, reward: float) -> None: ...

class SupportsConsolidation(Protocol):
    def consolidate(self) -> ConsolidationResult: ...
```

Rename `DynamicBrain.deliver_reward()` to `deliver_reward_to_regions()` and iterate over all regions that implement the protocol. Same for `consolidate()`.

### ARCH-9: `_integrate_synaptic_inputs_at_dendrites` Filter Parameters

The method has three filter parameters (`filter_by_source_region`, `filter_by_source_population`, `filter_by_target_population`) which are essentially a workaround for the fact that a single `SynapticInput` dict contains inputs for all populations. This is awkward — callers must filter by target population to get the right neurons.

**Fix**: The `forward()` method should pre-split `synaptic_inputs` by `target_population` before calling `_integrate_synaptic_inputs_at_dendrites`. This can be done once in `_pre_forward()` or in a new `_split_inputs_by_population()` helper. Then `_integrate_synaptic_inputs_at_dendrites` doesn't need filter parameters at all.

```python
def _split_inputs_by_population(
    self,
    synaptic_inputs: SynapticInput,
) -> Dict[PopulationName, SynapticInput]:
    """Group synaptic inputs by target population."""
    split: Dict[PopulationName, SynapticInput] = defaultdict(dict)
    for synapse_id, spikes in synaptic_inputs.items():
        split[synapse_id.target_population][synapse_id] = spikes
    return split
```

---

## Part 4 — Consolidation / Refactoring Opportunities

### REFACTOR-1: Hippocampus Internal Subregion Communication

The Hippocampus `forward()` method is ~400+ lines with manual management of 10+ `CircularDelayBuffer` objects for internal spike propagation (DG→CA3, CA3→CA3, CA3→CA2, CA3→CA1, CA2→CA1). This is the same job the `AxonalTract` system does for inter-region communication.

**Proposal**: Create a `HippocampalSubcircuit` class that encapsulates the DG→CA3→CA2→CA1 circuit with its delay buffers, inhibitory networks, and learning strategies. `Hippocampus` becomes a thin wrapper that routes external inputs to the subcircuit and collects its output. This reduces `hippocampus.py` from ~1700 lines to ~800 lines.

Alternatively (more radical): Model DG, CA3, CA2, CA1 as separate `NeuralRegion` instances connected by the brain's `AxonalTract` system. This would be the most architecturally consistent approach and would allow each subregion to have its own learning strategy, neuromodulator receptors, and population sizes configured independently. The trade-off: more `AxonalTract` objects and slightly more routing overhead.

### REFACTOR-2: Cortex Layer Processing Consolidation

The cortex `forward()` method processes L4, L2/3, L5, L6a, L6b with very similar boilerplate per layer:
1. Integrate excitatory inputs via `_integrate_synaptic_inputs_at_dendrites()`
2. Compute inhibition via `_compute_layer_inhibition()`
3. Apply NE gain modulation
4. Split AMPA/NMDA
5. Run neuron forward
6. Update homeostasis
7. Apply learning
8. Write to delay buffer

This pattern repeats 5 times with minor variations. It should be expressed as a `_process_layer()` method or a `CorticalLayerProcessor` dataclass that encapsulates layer-specific state.

**Proposal**:
```python
@dataclass
class CorticalLayer:
    name: str
    neurons: ConductanceLIF
    inhibitory_network: CorticalInhibitoryNetwork
    spike_buffer: CircularDelayBuffer
    membrane_buffer: CircularDelayBuffer
    da_concentration: torch.Tensor
    ne_concentration: torch.Tensor
    ach_concentration: torch.Tensor
    learning_strategy: CompositeStrategy
    baseline_inhibition_ratio: float
```

The `forward()` method iterates over `self.layers: List[CorticalLayer]` with a unified `_process_layer()` call. Layer-specific logic (e.g., L4 predictive coding inhibition, L5 output projection) is expressed as optional callbacks or subclass overrides.

### REFACTOR-3: Deduplicate Neuromodulator Receptor Patterns

Every region (Cortex, Hippocampus, Striatum, Thalamus, PFC) follows the same pattern:
1. Create one `NeuromodulatorReceptor` per modulator
2. Store per-population concentration tensors
3. In `forward()`: `receptor.update(spikes)` → slice per population → apply

This ~30-line pattern repeats across 5 regions with slight variations. Extract to a `NeuromodulatorBundle` class:

```python
class NeuromodulatorBundle(nn.Module):
    """Manages all neuromodulator receptors and concentrations for a region."""

    def __init__(
        self,
        populations: Dict[PopulationName, int],  # name → size
        modulators: List[NeuromodulatorType],      # which modulators to receive
        config: NeuromodulatorBundleConfig,
        device: torch.device,
    ):
        ...

    def update(
        self,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> Dict[NeuromodulatorType, Dict[PopulationName, torch.Tensor]]:
        """Returns {modulator_type: {population_name: concentration_tensor}}."""
        ...
```

Regions then call `concentrations = self.neuromod_bundle.update(neuromodulator_inputs)` and index as `concentrations['da']['l5']`.

### REFACTOR-4: Interneuron Network Consolidation

`CorticalInhibitoryNetwork` (in `cortex/`) and `HippocampalInhibitoryNetwork` (in `hippocampus/`) are structurally similar: both contain PV, SST, and OLM/VIP populations with E→I, I→E, I→I weights. The hippocampal version adds OLM cells for theta modulation.

**Proposal**: Extract a base `InhibitoryNetwork` class to `neural_region.py` or a shared module with:
- Configurable cell type populations
- Configurable E→I, I→E, I→I weight initialization
- Optional gap junction coupling (PV networks)
- Optional OLM cells (hippocampal theta)

`CorticalInhibitoryNetwork` and `HippocampalInhibitoryNetwork` extend this base with their specific connectivity patterns.

### REFACTOR-5: SynapseId Encoding Should Not Be In Container Classes

The `_encode`/`_decode` methods properly belong to `SynapseId` itself as `__str__` (already exists but different format) and a dedicated `to_key()` / `from_key()` classmethod pair:

```python
@dataclass(frozen=True)
class SynapseId:
    ...
    def to_storage_key(self) -> str:
        """Stable ASCII key for use as dict/ParameterDict key."""
        inh = "1" if self.is_inhibitory else "0"
        return f"{self.source_region}|{self.source_population}|{self.target_region}|{self.target_population}|{inh}"

    @classmethod
    def from_storage_key(cls, key: str) -> SynapseId:
        src_r, src_p, tgt_r, tgt_p, inh = key.split("|")
        return cls(src_r, src_p, tgt_r, tgt_p, is_inhibitory=(inh == "1"))
```

All containers then call `synapse_id.to_storage_key()` and `SynapseId.from_storage_key(key)`.

---

## Part 5 — Missing Features for LLM-Level Capabilities

These are features not currently present, needed for the long-term goal of matching LLM capabilities:

### FEAT-1: Predictive Coding Across Cortical Hierarchy

**Current**: Single cortex with L5/L6→L4 prediction. **Missing**: Multi-level hierarchy.

Real predictive coding requires `Higher_Cortex.L1 → Lower_Cortex.L2/3` feedback for top-down predictions, and `Lower_Cortex.L2/3 → Higher_Cortex.L4` bottom-up error signals. The current model only has this within a single cortical region.

**Proposal**: Add an inter-cortex `L1` population (apical dendritic tuft targets) to `CortexConfig` and support `top_down` connections that target L1 specifically for hierarchical predictive coding.

### FEAT-2: Dendritic Computation (Compartmental Neurons)

**Current**: Single-compartment ConductanceLIF for all neurons.

L5 pyramidal neurons have three functional compartments: basal dendrites (local input), apical dendrites (top-down input), and soma (integration). The "dendritic spike" triggered by coincident apical + basal input is a key mechanism for predictive learning.

**Proposal**: Implement a `CompartmentalLIF` neuron class with separate basal/apical/somatic compartments. This is a significant extension but essential for biologically accurate predictive coding.

### FEAT-3: Oscillatory Binding and Polychronization

**Current**: Single global `dt_ms` step. No inter-regional spike-timing coordination.

Gamma-theta hierarchical coding (fast gamma nested in slow theta) is thought to carry information capacity comparable to rate coding, with better temporal resolution. Cross-frequency coupling between cortex (gamma), hippocampus (theta), and thalamus (alpha/spindles) is critical for memory encoding.

**Proposal**: Track oscillatory phase for each region pair and use it to weight learning (STDP windows gated by oscillatory phase).

### FEAT-4: Structural Plasticity / Synaptogenesis

**Current**: Weights change, but zero-weight connections are never removed and no new connections can form.

**Proposal**: Add a `StructuralPlasticity` module that periodically:
1. Prunes synapses where `weight < prune_threshold` (remove from sparse weight matrix)
2. Sprouts new synapses where pre/post activity is high but no connection exists (grow new connection)

This is essential for long-term learning efficiency.

### FEAT-5: Context-Dependent Working Memory in PFC

**Current**: PFC has recurrent connectivity and DA gating.

**Missing**: Explicit "slot-based" working memory where different items can be maintained in separate neural subpopulations with item-specific maintenance via structured recurrent attractors.

---

## Implementation Order

### Phase 1 — Critical Bugs (Immediate, ~1 week)

1. **BUG-1**: Fix `_integrate_synaptic_inputs_at_dendrites` to return `(g_exc, g_inh)` and update all call sites
2. **BUG-2**: Register all persistent state tensors as buffers across all regions
3. **BUG-3**: Normalize Cerebellum to use consistent units
4. **BUG-4**: Add `is_inhibitory` to `AxonalTractSourceSpec` and propagate through routing

### Phase 2 — Architecture Cleanup (~2 weeks)

5. **ARCH-1**: Extract `SynapseIdKeyMixin` and remove duplication
6. **REFACTOR-5**: Move `to_storage_key`/`from_storage_key` to `SynapseId`
7. **ARCH-4**: Fix `_firing_rate_alpha` EMA computation
8. **ARCH-5**: Change `enable_neuromodulation` default to `True`
9. **ARCH-2**: Remove `__call__` assertion overrides
10. **ARCH-9**: Pre-split `SynapticInput` by population in `_pre_forward()`

### Phase 3 — Neuroscience Additions (~4 weeks)

11. **NEU-7**: Verify/fix Medial Septum → Hippocampus theta driving (remove internal sinusoid)
12. **NEU-8**: Create per-layer learning configs for Cortex
13. **NEU-6**: Move intrinsic reward into VTA spiking computation
14. **NEU-2**: Add LateralHabenula → RMTg → VTA aversion pathway
15. **NEU-3**: Implement DorsalRaphe + '5ht' neuromodulator system
16. **NEU-1**: Create EntorhinalCortex with EC→CA1 direct perforant path

### Phase 4 — Refactoring (~3 weeks)

17. **ARCH-3**: Split `configs.py` into per-region files
18. **ARCH-7**: Slim down `NeuralRegionConfig` base
19. **REFACTOR-3**: Extract `NeuromodulatorBundle`
20. **REFACTOR-4**: Unified `InhibitoryNetwork` base class
21. **REFACTOR-2**: `CorticalLayerProcessor` dataclass to reduce cortex forward boilerplate
22. **ARCH-8**: Protocol-based `deliver_reward()` and `consolidate()`

### Phase 5 — Missing Structures & Advanced Features (~6+ weeks)

23. **NEU-4**: Amygdala (BLA + CeAL)
24. **NEU-5**: GPe + STN for complete basal ganglia indirect pathway
25. **FEAT-1**: Inter-cortex hierarchical predictive coding
26. **FEAT-4**: Structural plasticity / synaptogenesis
27. **FEAT-2**: Compartmental L5 neurons for dendritic computation

---

## Appendix: Biological Accuracy Scorecard

| Feature | Current Status | Score |
|---|---|---|
| Conductance-based LIF with shunting inhibition | ✅ Implemented | 5/5 |
| Spike-based binary spikes (no rate codes) | ✅ Implemented | 5/5 |
| STDP with retrograde signaling | ✅ Implemented | 4/5 |
| BCM sliding threshold | ✅ Implemented | 4/5 |
| Three-factor DA-gated RL (D1/D2 pathways) | ✅ Implemented | 4/5 |
| NMDA slow excitation + AMPA fast excitation | ✅ Implemented | 4/5 |
| Short-term plasticity (STP/STD) | ✅ Implemented | 4/5 |
| PV/SST/VIP inhibitory cell types | ✅ Implemented | 5/5 |
| Gap junctions (PV, FSI) | ✅ Implemented | 4/5 |
| Theta-modulated hippocampal encoding/retrieval | ✅ Implemented | 4/5 |
| Sharp-wave ripple replay | ✅ Implemented | 3/5 |
| T-type Ca²⁺ channels (thalamic burst) | ✅ Implemented | 4/5 |
| OLM interneurons for theta | ✅ Implemented | 4/5 |
| Spike-frequency adaptation | ✅ Implemented | 4/5 |
| Axonal conduction delays (heterogeneous) | ✅ Implemented | 5/5 |
| Homeostatic intrinsic plasticity | ✅ Implemented | 4/5 |
| Synaptic scaling | ✅ Implemented | 3/5 |
| Neuromodulator volume transmission (DA/NE/ACh) | ✅ Implemented | 4/5 |
| DA reward prediction error (VTA) | ✅ Implemented | 3/5 |
| Entorhinal Cortex | ❌ Missing | 0/5 |
| Lateral Habenula / aversive RPE | ❌ Missing | 0/5 |
| Serotonin system | ❌ Missing | 0/5 |
| Amygdala | ❌ Missing | 0/5 |
| Complete basal ganglia (GPe/STN) | ⚠️ Partial | 2/5 |
| Inhibitory conductance in base method | ❌ Bug (BUG-1) | 0/5 |
| Registered state buffers | ❌ Bug (BUG-2) | 1/5 |
| Long-range inhibitory projections | ❌ Bug (BUG-4) | 0/5 |
| **Overall** | | **~72/100** |

After Phase 1-3 fixes: estimated ~88/100.
After Phase 4-5: estimated ~95/100.
