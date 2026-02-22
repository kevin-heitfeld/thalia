# Thalia Implementation Plan

Full implementation plan derived from the expert review (February 2026).
Issues are grouped by theme and ordered within each group by priority.
The `__call__` override in `DynamicBrain` and regions is **intentional** and excluded from this plan.

---

## Part 1 — Critical Software Bugs

These must be fixed first. They silently corrupt every training run by
breaking PyTorch's device/parameter/state-dict infrastructure.

---

### ✅ BUG-01: `_synaptic_weights`, `stp_modules`, `neuron_populations` are plain Python dicts

**File:** `src/thalia/brain/regions/neural_region.py`

**Problem:**
```python
self._synaptic_weights: Dict[SynapseId, nn.Parameter] = {}   # NOT registered
self.stp_modules: Dict[SynapseId, ShortTermPlasticity] = {}   # NOT registered
self.neuron_populations: Dict[PopulationName, ConductanceLIF] = {}  # NOT registered
```

Plain dicts are invisible to PyTorch's module system:
- `region.to(device)` does **not** move weights, STP modules, or neurons.
- `region.parameters()` does **not** yield the synaptic `nn.Parameter` objects.
- `torch.save(region.state_dict(), ...)` silently omits all learned weights.
- `forward()` runs on a different device than construction when `.to()` is called.

**Root cause:** `SynapseId` is a dataclass (not a string), so `nn.ParameterDict` /
`nn.ModuleDict` cannot be used directly with it as a key (both require string keys).

**Fix:**

1. Add a `_key(synapse_id)` helper that serializes `SynapseId` to a stable string.
2. Store in `nn.ParameterDict` / `nn.ModuleDict` using the serialized string key.
3. Keep a parallel `Dict[SynapseId, str]` for reverse lookup (or just re-serialize on access).
4. Expose `get_synaptic_weights(synapse_id)` unchanged (translates internally).

```python
# neural_region.py

@staticmethod
def _synapse_key(synapse_id: SynapseId) -> str:
    """Stable string key for nn.ParameterDict / nn.ModuleDict."""
    return str(synapse_id)  # or a canonical repr

class NeuralRegion(nn.Module, ABC, Generic[ConfigT]):
    def __init__(self, ...):
        super().__init__()
        # Use registered containers so .to(), .parameters(), .state_dict() all work
        self._synaptic_weights = nn.ParameterDict()    # SynapseId serialized to str key
        self.stp_modules = nn.ModuleDict()              # same key scheme
        self.neuron_populations = nn.ModuleDict()       # same key scheme
```

Internal helpers `add_synaptic_weights`, `get_synaptic_weights`, `has_synaptic_weights`,
`add_stp_module`, `_register_neuron_population` all stay public and unchanged — they
just translate through `_synapse_key()`.

**Impact:** All regions, all weight matrices, STP, and neuron populations become
fully tracked by PyTorch. No API changes outside `neural_region.py`.

**Status: DONE.** `SynapseIdParameterDict` and `SynapseIdModuleDict` wrapper classes added to
`neural_region.py`. Both use pipe-separated string keys internally so `nn.ParameterDict` /
`nn.ModuleDict` never see the `SynapseId` object directly. `neuron_populations` moved to
`nn.ModuleDict` (string keys already). Validated: `parameters()` yields all weight tensors,
`state_dict()` contains 703 synaptic keys, all `items()` return typed `SynapseId` objects.

---

### ✅ BUG-02: `axonal_tracts` in `DynamicBrain` is a plain Python dict

**File:** `src/thalia/brain/brain.py`

**Problem:**
```python
self.axonal_tracts: Dict[Tuple[RegionName, PopulationName], AxonalTract] = axonal_tracts
```

`AxonalTract` is an `nn.Module` with `CircularDelayBuffer` / `HeterogeneousDelayBuffer`
state. Storing it in a plain dict means:
- `brain.to(device)` leaves all delay buffers on their construction device.
- `brain.state_dict()` omits all delay state (replay is broken across save/load).

**Fix:**

Flatten the `(RegionName, PopulationName)` tuple keys to strings and store in
`nn.ModuleDict`:

```python
@staticmethod
def _tract_key(target_region: RegionName, target_population: PopulationName) -> str:
    return f"{target_region}__{target_population}"

# In __init__:
self._axonal_tracts_dict = nn.ModuleDict({
    DynamicBrain._tract_key(r, p): tract
    for (r, p), tract in axonal_tracts.items()
})
# Keep typed accessor:
@property
def axonal_tracts(self) -> Dict[Tuple[RegionName, PopulationName], AxonalTract]:
    ...  # reconstruct view by splitting key on "__"
```

**Status: DONE.** `AxonalTractDict(nn.Module)` added to `brain.py`; `self.axonal_tracts` is now
a typed `nn.ModuleDict` wrapper with `(RegionName, PopulationName)` tuple keys encoded as
`"region|population"` strings. Additionally, `CircularDelayBuffer` and
`HeterogeneousDelayBuffer` converted from plain classes to `nn.Module` subclasses with
`register_buffer` for `buffer` and `delays` tensors — delay state now appears in
`state_dict`. Validated: 60 axonal tract buffer keys in `state_dict`, 62 registered
buffer modules.

---

### ✅ BUG-03: `torch.set_grad_enabled(False)` is a global, permanent side effect

**Files:** `src/thalia/brain/brain.py` (line 96), `src/thalia/brain/brain_builder.py` (line 136)

**Problem:**
`set_grad_enabled(False)` is process-wide and permanent. It does not scope to
the brain or simulation loop. Any code sharing the process (e.g., an outer loop
computing a loss, a test harness, a profiler) silently has autograd disabled.

**Fix:**
1. Remove both `torch.set_grad_enabled(False)` calls.
2. Decorate every `forward()` method (on `DynamicBrain` and all regions) with
   `@torch.no_grad()`. This is scoped, composable, and overridable.

```python
@torch.no_grad()
def forward(self, synaptic_inputs: Optional[SynapticInput] = None) -> BrainOutput:
    ...
```

For `BrainBuilder`, remove the call entirely — construction does not execute
forward passes.

**Status: DONE.** Both `torch.set_grad_enabled(False)` calls removed (from `brain.py`
`__init__` and `brain_builder.py` `__init__`). `@torch.no_grad()` added to all 23
`forward()` methods: `DynamicBrain`, `NeuralRegion` base (abstract method), all 11
region subclasses, both inhibitory network submodules, `PurkinjeCellLayer`,
`ShortTermPlasticity`, and all 4 neuron classes (`ConductanceLIF`, `IzhikevichNeuron`,
`AcetylcholineNeuron`, `NorepinephrineNeuron`).

---

## Part 2 — Software Architecture Improvements

---

### ✅ ARCH-01: Hardcoded neuromodulator broadcast

**File:** `src/thalia/brain/brain.py`

**Problem:**
```python
if 'vta' in self._last_brain_output and 'da' in self._last_brain_output['vta']:
    neuromodulator_signals['da'] = ...
if 'locus_coeruleus' in ...:
    neuromodulator_signals['ne'] = ...
if 'nucleus_basalis' in ...:
    neuromodulator_signals['ach'] = ...
```

Adding serotonin (raphe), histamine (tuberomammillary), or lateral habenula
requires editing `brain.py`. This is an Open/Closed violation.

**Fix:**
Add a class-level registry on neuromodulator region classes:

```python
class VTA(NeuralRegion[VTAConfig]):
    neuromodulator_outputs: ClassVar[Dict[str, PopulationName]] = {'da': 'da'}

class LocusCoeruleus(NeuralRegion[...]):
    neuromodulator_outputs: ClassVar[Dict[str, PopulationName]] = {'ne': 'ne'}
```

In `DynamicBrain.forward()`, replace the hardcoded block with:

```python
neuromodulator_signals: NeuromodulatorInput = {}
if self._last_brain_output is not None:
    for region_name, region in self.regions.items():
        if hasattr(region, 'neuromodulator_outputs'):
            for mod_key, pop_name in region.neuromodulator_outputs.items():
                val = self._last_brain_output.get(region_name, {}).get(pop_name)
                neuromodulator_signals[mod_key] = val
```

New neuromodulator sources (lateral habenula → anti-reward signal, raphe → 5-HT)
require zero changes to `brain.py`.

**Status: DONE.** `VTA`, `LocusCoeruleus`, and `NucleusBasalis` each declare:
```python
neuromodulator_outputs: ClassVar[Dict[str, str]] = {'da': 'da'}  # etc.
```
`DynamicBrain.forward()` STEP 3 replaced with a generic loop over all regions
that have `neuromodulator_outputs`. Adding a new neuromodulator source (raphe → 5-HT,
habenula → anti-reward) now requires zero changes to `brain.py`.

---

### ✅ ARCH-02: Lazy initialization in learning strategies is fragile

**Files:** `src/thalia/learning/strategies.py`

**Problem:**
`_ensure_trace_manager()` and `_init_theta()` initialize lazily on first
`compute_update()` call. Dimension mismatches surface at simulation step 0,
not during construction. A wrong `n_pre` silently reallocates mid-simulation
when dimensions change.

**Fix:**
Add an explicit `setup(n_pre: int, n_post: int, device: torch.device)` protocol
to `LearningStrategy`:

```python
class LearningStrategy(nn.Module, ABC):
    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Called once during BrainBuilder.build(). Must initialize all state."""
        pass  # default: no-op for strategies without per-synapse state
```

`STDPStrategy.setup()` creates the `EligibilityTraceManager` and firing rate
buffers eagerly. `BCMStrategy.setup()` creates theta. `BrainBuilder.build()`
calls `strategy.setup()` for every synapse after dimensions are known.

**Status: DONE.** `setup()` / `ensure_setup()` protocol added to `LearningStrategy` base. `STDPStrategy`, `BCMStrategy`, and `ThreeFactorStrategy` override `setup()` to eagerly allocate all state via `register_buffer`. Lazy-init code removed entirely.

---

### ✅ ARCH-03: `learning_strategy` field on `NeuralRegion` is misleading

**File:** `src/thalia/brain/regions/neural_region.py`

**Problem:**
`self.learning_strategy: Optional[LearningStrategy] = None` implies one
learner per region. In practice, learning is per-synapse (each `SynapseId`
has its own strategy). The field is either unused or a `CompositeStrategy`
that re-dispatches. The API misleads subclass authors.

**Fix:**
Replace the single `learning_strategy` field with a per-synapse dict:

```python
self._learning_strategies: nn.ModuleDict()   # str(SynapseId) → LearningStrategy
```

Add helpers:
```python
def add_learning_strategy(self, synapse_id: SynapseId, strategy: LearningStrategy) -> None: ...
def get_learning_strategy(self, synapse_id: SynapseId) -> Optional[LearningStrategy]: ...
def apply_learning(self, synapse_id: SynapseId, pre_spikes, post_spikes, **kwargs) -> None: ...
```

`apply_learning()` centralizes the weight-update + clamp pattern that is
currently copy-pasted across Cortex, Hippocampus, Striatum, etc.

**Status: DONE.** `self._learning_strategies: SynapseIdModuleDict` added to `NeuralRegion.__init__`. `SynapseIdModuleDict.get()` added. `add_learning_strategy()`, `get_learning_strategy()`, and `apply_learning()` helpers added to `NeuralRegion`. `update_temporal_parameters()` iterates `_learning_strategies` with `id()`-based deduplication (shared `BCMStrategy` instances updated once). Cortex migrated from `strategies_lXX: nn.ModuleList` to `composite_lXX: CompositeStrategy`; all 9 `CompositeStrategy.compute_update(strategies=…)` static calls replaced with instance method calls. Backward-compat `self.learning_strategy` field preserved for Hippocampus, Thalamus, and PFC.

---

### ✅ ARCH-04: Neuromodulator signals use a 1-step delay rather than physical diffusion

**File:** `src/thalia/brain/brain.py`

**Problem:**
All neuromodulators are delayed by exactly one timestep (`_last_brain_output`).
Biological DA diffusion takes ~100–300 ms (Garris et al.); one dt=1ms step is
not meaningful. More importantly, neuromodulators should influence learning with
their biologically correct temporal profile, not a discrete 1-step lag.

**Fix:**
Route neuromodulator outputs through a dedicated `NeuromodulatorTract` — a
lightweight variant of `AxonalTract` with:
- A first-order low-pass filter (τ_diff = 100–300 ms) instead of a pure delay.
- Optional configurable decay per modulator type.

This can reuse `AxonalTract` infrastructure with a different buffer type, or be
a thin wrapper. The broadcast logic in `brain.py` remains but reads from these
tracts instead of `_last_brain_output` directly.

**Status: DONE.** `NeuromodulatorTract(nn.Module)` added to `brain.py` implementing a
first-order IIR low-pass filter:
$$\text{filtered}(t) = \alpha \cdot \text{filtered}(t-1) + (1-\alpha) \cdot \text{raw}(t), \quad \alpha = e^{-\Delta t / \tau}$$
Default τ: DA = 150 ms, NE = 80 ms, ACh = 60 ms (from Garris et al. / muscarinic onset
literature). `DynamicBrain.__init__` auto-creates one tract per modulator key discovered
via the ARCH-01 `neuromodulator_outputs` ClassVar. `filtered` buffer is lazily registered
(appears in `state_dict` on first spike). `set_timestep()` propagates `dt` changes. The
broadcast now reads from `NeuromodulatorTract.update(raw)` instead of
`_last_brain_output` directly.

---

## Part 3 — Biological Accuracy Improvements

---

### BIO-01: No dendritic compartmentalization (highest priority biological issue)

**Affected files:** `src/thalia/components/neurons/conductance_lif_neuron.py`,
`src/thalia/brain/regions/cortex/cortex.py`,
`src/thalia/brain/regions/neural_region.py`

**Problem:**
All neurons are point neurons. Pyramidal cells have functionally distinct:
- **Basal/proximal dendrites** — integrate feedforward/bottom-up input near soma.
- **Apical/distal dendrites** (in L1) — integrate feedback/top-down input distal from soma.

When both fire coincidently, a **dendritic calcium spike** triggers burst output
(Larkum 2013 — the "critical neuron" hypothesis). This is the proposed neural
substrate for attention, predictive coding (Rao & Ballard 1999), and
consciousness (Friston Free Energy). Without it:
- Top-down and bottom-up inputs are functionally equivalent.
- Attention cannot be mechanistically modeled as feedback modulation.
- The L6 corticothalamic projections have no coherent computational role.

**Fix — two-compartment LIF:**

Extend `ConductanceLIFConfig` with compartment parameters:

```python
# Two-compartment pyramidal model (Larkum et al. 1999)
enable_compartments: bool = False          # Disabled by default for non-pyramidal cells
g_couple: float = 0.1                     # Apical-basal coupling conductance
tau_apical: float = 50.0                  # Apical dendrite time constant (ms)
ca_spike_threshold: float = 0.5           # Apical depolarization for Ca2+ spike
ca_spike_boost: float = 2.0              # Burst probability multiplier on Ca spike
```

`ConductanceLIF.forward()` gains a second membrane equation for the apical
compartment. The coupling current is $I_{couple} = g_{couple}(V_{apical} - V_{soma})$.
When $V_{apical} \geq \theta_{Ca}$, a calcium spike increments burst probability
(or directly doubles the soma spike count for the current timestep via a
burst output flag).

**In `SynapseId`:** add an optional `target_compartment: Literal['soma', 'apical'] = 'soma'`
field. The cortex wires feedforward inputs (thalamus L4, L4→L2/3) to `'soma'`
and feedback inputs (top-down, L6) to `'apical'`.

**Priority subclasses:** pyramidal cells in Cortex L2/3, L5, and PFC Executive
population. Non-pyramidal cells (FSI, granule cells, DCN, MSNs) keep
`enable_compartments=False`.

---

### BIO-02: Interneuron diversity — VIP and SST subtypes missing

**Affected files:** `src/thalia/brain/regions/cortex/inhibitory_network.py`,
`src/thalia/brain/regions/cortex/cortex.py`

**Problem:**
`CorticalInhibitoryNetwork` models only PV+ fast-spiking cells (perisomatic
inhibition). Two critical subtypes are absent:

**PV+ (present):** Target soma/AIS. Fast GABA. Gamma oscillation generation.
✓ Already implemented.

**SST+ Martinotti (missing):** Target apical dendrites. Suppresses top-down
feedback selectively. Recruited by pyramidal bursts (mutual excitation loop).
Slow, persistent firing. τ_mem ~30ms.

**VIP+ (missing):** Inhibit SST cells ("disinhibitory microcircuit").
Primary target of cortical ACh and NE. Mechanism by which neuromodulators gate
learning and attention:
```
ACh/NE → VIP activation → SST suppression → apical dendrite disinhibition
→ top-down feedback unblocked → LTP permitted at apical synapses
```
Without VIP, ACh learning gating is phenomenological (scalar multiply on STDP)
rather than mechanistic.

**Fix:**
Add `sst` and `vip` populations to `CorticalInhibitoryNetwork`:

```python
@dataclass
class CorticalInhibitoryNetworkConfig:
    pv_fraction: float = 0.15    # PV fraction of total inhibitory (existing)
    sst_fraction: float = 0.05   # SST fraction
    vip_fraction: float = 0.03   # VIP fraction
    # Fractions sum to ~0.23 inhibitory total (correct 80/20 E/I ratio)
```

Wiring:
- Pyramidal bursts → SST (recruited by strong excitation feedback).
- SST → apical compartment of pyramidals (when BIO-01 is done).
- VIP → SST (disinhibition).
- ACh/NE receptor on VIP → activation under neuromodulator input.

**This requires BIO-01 (compartments) to be maximally effective**, but SST/VIP
can be added before BIO-01 with soma-targeting as a placeholder.

---

### BIO-03: VTA computes simplified RPE — missing V(s') temporal difference

**File:** `src/thalia/brain/regions/vta/vta_region.py`

**Problem:**
Current RPE: $\delta \approx r_t - V(s_t)$ (no next-state value).
Correct RPE: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

Consequences:
1. Only immediate rewards can be credited — delayed reward sequences are not
   properly attributed.
2. The DA pause at expected-but-omitted reward requires prospective expectation,
   which is impossible without a V(s') prediction.
3. The striatum already implements eligibility traces (τ=500–2000ms, Yagishita 2014).
   These are wasted without a proper TD error driving them.

**Fix — TD(λ) in VTA:**

The striatum's eligibility traces are already the λ-return mechanism.
VTA needs to maintain a prediction of future value:

```python
# VTA state additions
self.value_prediction: torch.Tensor   # V(s), updated each step
self.gamma: float = 0.99              # Temporal discount
self.value_decay: float               # EMA decay for value estimate
```

RPE computation:
```python
# On each forward pass:
# 1. Receive current reward r_t from RewardEncoder
# 2. Receive current striatal/SNr activity as V(s_t) proxy
# 3. Store last V(s_t) as V(s_{t-1})
# 4. δ = r_t + γ * V(s_t) - V(s_{t-1})
```

The "next-state value" $V(s_{t+1})$ is approximated by the current SNr activity
(which encodes current value), and $V(s_t)$ is the stored previous value.
This gives the correct temporal structure without needing explicit lookahead.

---

### BIO-04: No amygdala

**Problem:**
The amygdala (BLA — basolateral amygdala) is absent entirely. It mediates:
- **Fear/aversive conditioning** — the most well-characterized form of Hebbian
  synaptic plasticity in mammals (LeDoux 2000).
- **Emotional salience gating** — BLA → VTA feedback amplifies DA response to
  salient stimuli, not just rewarding ones (importance ≠ reward).
- **Memory consolidation gating** — BLA → hippocampus pathways determine which
  episodic memories are prioritized for offline consolidation.
- **Stress modulation** — norepinephrine acting through BLA gates consolidation
  during arousal (flashbulb memory mechanism).

**Fix — Phase 1 (minimal):**

Create `src/thalia/brain/regions/amygdala/amygdala.py`. Minimum viable circuit:

Populations:
- `bla_exc` — BLA principal excitatory neurons (~80%)
- `bla_inh` — local inhibitory (ITC cells — intercalated cell masses)
- `cea` — Central nucleus of amygdala (output to brainstem, autonomic)

Inputs: Cortex L5, thalamus relay (both carry sensory info), VTA (DA feedback).
Outputs:
- BLA → VTA (salience amplification — increases DA burst to salient events).
- BLA → Hippocampus CA1 (modulates encoding gate during emotional events).
- BLA → Striatum D1 (motivational gating of action selection).
- CEA → brainstem (autonomic arousal, not critical for Phase 1).

Learning rule: STDP with NE/stress modulation (NE from locus coeruleus enhances
BLA LTP — the flashbulb memory mechanism).

---

### BIO-05: GABA_B (metabotropic) not modeled

**Affected files:** `src/thalia/components/neurons/conductance_lif_neuron.py`

**Problem:**
Only GABA_A (fast ionotropic, τ ≈ 5–10 ms) is implemented. GABA_B
(slow G-protein coupled, τ ≈ 100–400 ms) mediates:
- Slow after-hyperpolarization following inhibitory bursts — required for
  UP/DOWN state transitions (slow-wave NREM sleep dynamics).
- Presynaptic inhibition (autoreceptors on GABAergic and glutamatergic terminals).
- HCN channel modulation in thalamus (contributes to spindle oscillations).
- Late-phase inhibition in cortex and hippocampus after strong stimulation.

Without GABA_B, memory consolidation during simulated sleep/rest is inaccurate
because UP/DOWN state alternation depends on it.

**Fix:**
Add a third (slow inhibitory) conductance channel to `ConductanceLIFConfig`:

```python
# GABA_B channel (slow metabotropic inhibition)
enable_gaba_b: bool = False            # Enable for cortex, hippocampus, thalamus
tau_I_slow: float = 200.0             # GABA_B time constant (100–400ms)
E_I_slow: float = -0.7               # GABA_B reversal (more hyperpolarizing, like K+)
gaba_b_ratio: float = 0.2            # Fraction of inh. input routed to slow channel
```

The membrane equation gains a third term:
$$C_m \frac{dV}{dt} = g_L(E_L - V) + g_E(E_E - V) + g_{I,fast}(E_{I,fast} - V) + g_{I,slow}(E_{I,slow} - V)$$

The slow conductance `g_I_slow` decays with `tau_I_slow` and is driven by a
fraction of the same inhibitory synaptic input as `g_I`.

---

### BIO-06: NMDA conductance is synapse-global, not pathway-specific

**Files:** `src/thalia/components/neurons/conductance_lif_neuron.py`,
`src/thalia/brain/regions/neural_region.py`

**Problem:**
```python
nmda_ratio: float = 0.0  # TODO: Consider making nmda_ratio a per-source parameter
```

The NMDA/AMPA ratio is a single scalar on the neuron. In biology:
- Distal (apical) dendritic synapses: high NMDA/AMPA ratio (≥0.5).
- Proximal (perisomatic) synapses: low NMDA/AMPA ratio (~0.1–0.2).
- NMDA at apical synapses implements coincidence detection between feedforward
  and feedback signals (the mechanism of top-down attention).
- Mossy fiber → CA3 synapses: near-zero NMDA (fast, reliable). CA3 recurrent
  synapses: high NMDA (temporal integration, pattern completion).

**Fix:**
Move `nmda_ratio` from `ConductanceLIFConfig` (neuron-level) to a per-synapse
parameter passed during conductance integration:

```python
# In NeuralRegion: store per-synapse NMDA ratio
self._synapse_nmda_ratio: Dict[str, float] = {}

def add_input_source(self, synapse_id, n_input, connectivity, weight_scale,
                     *, nmda_ratio: float = 0.0, stp_config=None):
    ...
    self._synapse_nmda_ratio[self._synapse_key(synapse_id)] = nmda_ratio
```

During `forward()`, each synaptic conductance computation receives its own
`nmda_ratio`. The NMDA conductance state (`g_nmda`) per input source is then
tracked separately in the region's forward pass and summed into the neuron.

---

### BIO-07: Striatum missing cholinergic interneurons (TANs)

**File:** `src/thalia/brain/regions/striatum/striatum.py`

**Problem:**
Tonically Active Neurons (TANs) — cholinergic interneurons (~1% of striatal
cells) — are absent. TANs:
- Fire tonically at 2–10 Hz baseline.
- **Pause** at reward delivery (conditioned pause response, Graybiel 1994).
  This pause releases ACh suppression at MSNs precisely when DA arrives.
- Gate the timing of MSN plasticity: the TAN pause + DA burst conjunction is
  the precise window during which corticostriatal STDP is permitted.
- Project to D1 and D2 MSNs via M1/M4 muscarinic receptors (opposing effects).
- Receive input from thalamus (CM/Pf) — the thalamic drive to TANs is the
  conditioned cue signal.

Without TANs, striatal ACh modulation comes only from nucleus basalis (extrinsic)
and misses the local, precisely-timed gate that controls MSN plasticity windows.

**Fix:**
Add `tan` population to `Striatum`:
- ~1% of total striatal neurons (or fixed count of 50–200).
- Tonic LIF neuron with high rhythmic drive.
- Receive thalamic (CM/Pf) input — add thalamus → striatum:TAN connection in `BrainBuilder`.
- TAN pause = DA input suppresses tonic firing (D2 autoreceptor-like logic).
- TAN output modulates MSN conductance via M4 (D1-MSNs → ACh suppresses) and
  M1 (D2-MSNs → ACh activates). Apply as a local ACh scaling on MSN g_I.
- Integrate into the three-factor learning window: plasticity gate = `(da_burst AND NOT tan_firing)`.

---

### BIO-08: Burst vs. single-spike discrimination

**Problem:**
Spikes are `bool` (ADR-004). At dt=1ms a burst (2–5 spikes within 5ms) is
indistinguishable from a single spike — only one bit per timestep per neuron.
Thalamic burst mode (T-channel implemented), hippocampal CA3, and PFC L5 all
use burst firing to signal novelty, certainty, or trigger downstream
LTP-enabling cascades (Lisman 1997: "bursts as the quantum of neural information").

**Fix options (pick one):**

**Option A — Lower dt:** Run at dt=0.1ms or dt=0.2ms. Bursts are then
temporally resolved. High computational cost.

**Option B — Burst count output (preferred):** Add an optional `burst_count`
output field (uint8, 0–4) alongside the bool spike. Regions that support burst
firing (thalamus relay, CA3, PFC L5, Purkinje) output this. Downstream regions
optionally consume it as a multiplier on the incoming conductance.

```python
@dataclass
class RegionOutputEntry:
    spikes: torch.BoolTensor   # Existing
    bursts: Optional[torch.Tensor] = None  # uint8, burst count [0–5]
```

A Purkinje cell's climbing-fiber input already triggers burst gating in the
cerebellum; this formalizes what is already partially implicit.

---

## Part 4 — Incremental Backlog (Lower Priority)

These are valid improvements but do not block the above phases.

| ID | Issue | Notes |
|----|-------|-------|
| BACK-01 | Synaptic scaling only implemented in Cortex | Extend to Hippocampus, Striatum, PFC |
| BACK-02 | Spillover transmission unused outside Cortex | Enable selectively in hippocampus, thalamus |
| BACK-03 | `baseline_noise_conductance_enabled` disabled globally | Re-enable, tune per-region |
| BACK-04 | SNr baseline drive disabled | Re-enable with calibrated rate |
| BACK-05 | CA3 persistent gain review (`ca3_persistent_gain`) | Profile against UP-state literature |
| BACK-06 | No EEG/LFP proxy diagnostics | Aggregate population firing → LFP band power |
| BACK-07 | No spike raster output | Add to diagnostics pipeline |
| BACK-08 | Lateral habenula (anti-reward) absent | BLA → LHb → RMTg → VTA pathway for punishment |
| BACK-09 | Raphe nucleus (serotonin) absent | 5-HT modulates mood, impulsivity, temporal discounting |
| BACK-10 | No spatial topology within populations | Tonotopy, retinotopy for sensory regions |

---

## Suggested Execution Order

```
Phase 1 — Foundation ✅ COMPLETE
  BUG-01  _synaptic_weights / stp_modules / neuron_populations as nn.ParameterDict/ModuleDict
  BUG-02  axonal_tracts as nn.ModuleDict; delay buffers as nn.Module with register_buffer
  BUG-03  Remove torch.set_grad_enabled(False), add @torch.no_grad()

Phase 2 — Architecture (partial)
  ARCH-01  ✅ Dynamic neuromodulator broadcast via neuromodulator_outputs ClassVar
  ARCH-02  Eager setup() in learning strategies
  ARCH-03  Per-synapse learning strategy dict
  ARCH-04  ✅ Neuromodulator diffusion tract (IIR low-pass, τ per modulator)

Phase 3 — Core Biology (2–3 sprints)
  BIO-06  Per-synapse NMDA ratio (enables Phase 4)
  BIO-01  Two-compartment pyramidal neuron
  BIO-02  VIP + SST interneuron subtypes
  BIO-05  GABA_B slow inhibitory conductance

Phase 4 — Learning Accuracy (1–2 sprints)
  BIO-03  TD(λ) RPE in VTA
  BIO-07  Striatal TAN cholinergic interneurons
  BIO-08  Burst count output alongside bool spikes

Phase 5 — New Regions (2+ sprints)
  BIO-04  Amygdala (BLA + CEA)
  BACK-08 Lateral habenula
  BACK-09 Raphe nucleus

Phase 6 — Calibration & Diagnostics (ongoing)
  BACK-01 through BACK-07
```
