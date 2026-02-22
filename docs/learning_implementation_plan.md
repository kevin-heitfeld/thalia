# Thalia Learning System — Comprehensive Implementation Plan

Expert review of the learning infrastructure (February 2026).
Grounded in direct code inspection of `strategies.py`, `eligibility_trace_manager.py`,
`neural_region.py`, `cortex.py`, `striatum.py`, `hippocampus.py`, `thalamus.py`, `prefrontal.py`.

The existing ARCH-02 and ARCH-03 proposals are correct but **significantly underscoped**.
They miss several critical infrastructure bugs and a number of biological accuracy gaps.
This plan supersedes and expands both.

---

## Assessment of ARCH-02 and ARCH-03

**ARCH-02 (eager `setup()`):** Correct diagnosis, incomplete fix.
The lazy init is a symptom — the root cause is that strategy state tensors are not
registered with PyTorch at all. Adding `setup()` alone will not fix `.to(device)` or `state_dict()`.

**ARCH-03 (per-synapse dict):** Correct diagnosis, correct fix.
The single `learning_strategy` field is misleading. But the plan understates the scope:
Cortex already has per-layer lists (`strategies_l23`, …) that are **plain Python lists, not `nn.ModuleList`**.
Every strategy instance and all its accumulated state (eligibility, theta, traces) is **invisible to PyTorch right now**.
This is a critical bug on par with BUG-01.

---

## Part A — Critical Infrastructure Bugs in Learning

These are silent corruptions: device moves, `state_dict()`, and checkpoint/restore all fail.

---

### ✅ LEARN-BUG-01: `EligibilityTraceManager` is not an `nn.Module`

**File:** `src/thalia/learning/eligibility_trace_manager.py`

**Problem:**
`EligibilityTraceManager` stores `input_trace`, `output_trace`, `eligibility` as plain
Python attributes (`self.input_trace = torch.zeros(...)`). It has a hand-written `.to()`
method, but because it is not an `nn.Module`:
- `STDPStrategy.to(device)` does NOT call the manager's `.to()`.
- `state_dict()` does NOT include trace tensors.
- After checkpoint restore, all STDP eligibility state is lost (traces reset to zero silently).

**Fix:**
Convert `EligibilityTraceManager` to `nn.Module`. Register all tensors with `register_buffer`:

```python
class EligibilityTraceManager(nn.Module):
    def __init__(self, n_input: int, n_output: int, config: STDPConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.register_buffer("input_trace",  torch.zeros(n_input,  device=device))
        self.register_buffer("output_trace", torch.zeros(n_output, device=device))
        self.register_buffer("eligibility",  torch.zeros(n_output, n_input, device=device))
```

Remove the hand-written `.to()` override — PyTorch handles it automatically.

**Impact:** All STDP trace state now appears in `state_dict()` and moves correctly on `.to(device)`.

**Status: DONE.** `EligibilityTraceManager` converted to `nn.Module`; `input_trace`, `output_trace`, and `eligibility` registered via `register_buffer`. Hand-written `.to()` override removed. All STDP trace state now appears in `state_dict()` and moves correctly on `.to(device)`.

---

### ✅ LEARN-BUG-02: `STDPStrategy` state tensors are not registered buffers

**File:** `src/thalia/learning/strategies.py` — `STDPStrategy`

**Problem:**
`self.firing_rates` and `self.retrograde_signal` are assigned as plain Python attributes,
not via `register_buffer()`. They will be lost on `.to(device)` and won't appear in
`state_dict()`. The `_trace_manager` is not a submodule.

```python
# Current (broken):
self.firing_rates = torch.zeros(n_post, ...)    # plain attribute
self.retrograde_signal = torch.zeros(n_post, ...)  # plain attribute
self._trace_manager = EligibilityTraceManager(...)  # not a submodule
```

**Fix:**
1. After LEARN-BUG-01, store the `EligibilityTraceManager` via `self.trace_manager = ...`
   (assigned as an `nn.Module` child, PyTorch auto-registers it as a submodule after `setup()`).
2. Use `register_buffer` for `firing_rates` and `retrograde_signal` — created eagerly in `setup()`,
   not lazily.

```python
class STDPStrategy(LearningStrategy):
    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        self.trace_manager = EligibilityTraceManager(n_pre, n_post, self.config, device)
        self.register_buffer("firing_rates",     torch.zeros(n_post, device=device))
        self.register_buffer("retrograde_signal", torch.zeros(n_post, device=device))
        cfg = self.config
        self._firing_rate_decay  = exp(-self._dt_ms / self._firing_rate_tau_ms)
        self._retrograde_decay   = exp(-self._dt_ms / cfg.retrograde_tau_ms)
```

**Status: DONE.** `STDPStrategy.setup()` now eagerly initializes `trace_manager` as an `nn.Module` child, and registers `firing_rates` and `retrograde_signal` via `register_buffer`. Lazy-init code removed entirely.

---

### ✅ LEARN-BUG-03: `BCMStrategy` and `ThreeFactorStrategy` use fragile lazy buffer re-registration

**File:** `src/thalia/learning/strategies.py`

**Problem:**
Current pattern: initialize `self.theta = None`, then on first call do
`delattr(self, "theta"); self.register_buffer("theta", new_tensor)`.
This is unnecessarily complex, brittle (fails if called after pickling or after `.to()` with
a stale shape), and means that `state_dict()` contains no theta/eligibility until step 0.

**Fix:**
Use `setup(n_pre, n_post, device)` (ARCH-02 protocol):

```python
class BCMStrategy(LearningStrategy):
    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        self.register_buffer("theta", torch.full((n_post,), self.config.theta_init, device=device))
        self.register_buffer("firing_rates", torch.zeros(n_post, device=device))

class ThreeFactorStrategy(LearningStrategy):
    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        self.register_buffer("eligibility", torch.zeros(n_post, n_pre, device=device))
```

All lazy-registration code (`if "theta" not in self._buffers: ...`) is deleted.

**Status: DONE.** `BCMStrategy` and `ThreeFactorStrategy` use `setup()` / `ensure_setup()`; `theta`, `eligibility`, and `firing_rates` initialised via `register_buffer` in `setup()`. Fragile `del`/re-register pattern removed.

---

### ✅ LEARN-BUG-04: Cortex strategy lists are plain Python lists — invisible to PyTorch

**File:** `src/thalia/brain/regions/cortex/cortex.py`

**Problem:**
```python
self.strategies_l23: List[LearningStrategy] = [STDPStrategy(stdp_cfg), BCMStrategy(bcm_cfg)]
self.strategies_l4:  List[LearningStrategy] = [...]
# ... etc.
```
These are NOT `nn.ModuleList`. The `STDPStrategy` and `BCMStrategy` instances inside are
completely invisible to PyTorch:
- `cortex.to(device)` does NOT move strategies or their traces, theta, firing_rates.
- `cortex.state_dict()` does NOT include any strategy state.
- `cortex.parameters()` does NOT yield strategy parameters.

This is exactly the same class of bug as BUG-01 but for learning state.

**Fix:**
After implementing ARCH-03 (per-synapse `_learning_strategies: SynapseIdModuleDict`),
these lists are replaced entirely. See LEARN-ARCH-03 below.

**Status: DONE.** Cortex `strategies_lXX` converted to `nn.ModuleList` in Part A; subsequently superseded by `CompositeStrategy` instances (`composite_l23`, `composite_l4`, etc.) as part of LEARN-ARCH-02 in Part B. All 5 layer strategies are now proper `nn.Module` submodules of `Cortex`.

---

### ✅ LEARN-BUG-05: Striatum eligibility traces are plain Python dicts

**File:** `src/thalia/brain/regions/striatum/striatum.py`

**Problem:**
The Striatum bypasses the `LearningStrategy` framework entirely with:
```python
self._eligibility_d1_fast: Dict[SynapseId, torch.Tensor] = {}
self._eligibility_d1_slow: Dict[SynapseId, torch.Tensor] = {}
self._eligibility_d2_fast: Dict[SynapseId, torch.Tensor] = {}
self._eligibility_d2_slow: Dict[SynapseId, torch.Tensor] = {}
```
Plain dicts of tensors. Not registered. All eligibility state is lost on `.to(device)` and
omitted from `state_dict()`. The Striatum also hard-codes three-factor learning logic that
already exists (partially) in `ThreeFactorStrategy`.

**Fix:**
Migrate Striatum to use `ThreeFactorStrategy` (with the unified STDP traces from LEARN-ARCH-UNIFY
below) via the per-synapse `_learning_strategies` dict. Striatum adds its D1/D2 inversion logic
via a `D1ThreeFactorStrategy` / `D2ThreeFactorStrategy` pair (see LEARN-ARCH-04).

**Status: DONE.** `SynapseIdBufferDict(nn.Module)` added to `neural_region.py`; Striatum's 4 plain `Dict[SynapseId, Tensor]` eligibility dicts migrated to `SynapseIdBufferDict`. Full migration to `D1STDPStrategy` / `D2STDPStrategy` completed in LEARN-ARCH-04.

---

### ✅ LEARN-BUG-06: Linear decay approximation in `EligibilityTraceManager.update_traces`

**File:** `src/thalia/learning/eligibility_trace_manager.py`

**Problem:**
```python
trace_decay = 1.0 - dt_ms / self.config.tau_plus
```
This is a first-order Euler approximation. For dt_ms=1ms, tau_plus=20ms, this gives
`decay=0.95`, which is close to `exp(-1/20)=0.951`. But if tau is small (e.g., tau_plus=5ms,
dt=2ms), the formula gives 0.6 while the correct value is `exp(-0.4)=0.67`. Worse, for
dt > tau the decay goes negative, producing oscillating traces — a simulation artifact with
no biological counterpart.

**Fix:**
```python
import math
trace_decay = math.exp(-dt_ms / self.config.tau_plus)
eligibility_decay = math.exp(-dt_ms / self.config.eligibility_tau_ms)
```
Pre-compute these in `setup()` or `update_temporal_parameters()` and cache them.
Same fix applies to eligibility decay in `accumulate_eligibility()`.

**Status: DONE.** All trace and eligibility decay updated from linear `1.0 - dt/tau` to `exp(-dt/tau)` in `EligibilityTraceManager`, `ThreeFactorStrategy`, and `BCMStrategy`. Decay factors pre-computed in `update_temporal_parameters()` and cached.

---

## Part B — Architecture Consolidation

---

### ✅ LEARN-ARCH-01: Formalize `setup()` protocol on `LearningStrategy`

**File:** `src/thalia/learning/strategies.py`

This is the ARCH-02 fix, now correctly scoped:

```python
class LearningStrategy(nn.Module, ABC):

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        """Called once by BrainBuilder.build() after synapse dimensions are known.

        Must initialize all registered buffers and submodules (no lazy init).
        The default implementation is a no-op (for strategies with no per-synapse state).
        """
        pass

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Called when brain timestep changes. Pre-compute decay factors here."""
        self._dt_ms = dt_ms
```

`BrainBuilder.build()` calls `strategy.setup(n_pre, n_post, device)` for every synapse
after dimensions are resolved. No region subclass ever calls lazy-init code again.

**Status: DONE.** `setup()` and `ensure_setup()` are concrete (non-abstract) methods on `LearningStrategy` base with default no-op implementations. `STDPStrategy`, `BCMStrategy`, and `ThreeFactorStrategy` each override `setup()` to eagerly allocate all per-synapse state via `register_buffer`.

---

### ✅ LEARN-ARCH-02: Replace `CompositeStrategy` static class with `nn.Module`

**File:** `src/thalia/learning/strategies.py`

**Problem:**
`CompositeStrategy` is a plain class with static methods. It owns nothing.
The strategies it operates on are in caller-owned lists.

**Fix:**
```python
class CompositeStrategy(LearningStrategy):
    """Ordered composite: applies sub-strategies sequentially, passing updated weights
    between them. Each sub-strategy must call setup() together."""

    def __init__(self, strategies: List[LearningStrategy]):
        super().__init__(config=strategies[0].config)
        # nn.ModuleList → all sub-strategies tracked by PyTorch
        self.sub_strategies = nn.ModuleList(strategies)

    def setup(self, n_pre: int, n_post: int, device: torch.device) -> None:
        for s in self.sub_strategies:
            s.setup(n_pre, n_post, device)

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        for s in self.sub_strategies:
            weights = s.compute_update(weights, pre_spikes, post_spikes, **kwargs)
        return weights

    def update_temporal_parameters(self, dt_ms: float) -> None:
        super().update_temporal_parameters(dt_ms)
        for s in self.sub_strategies:
            s.update_temporal_parameters(dt_ms)
```

This means callers never pass a list externally — they create a single `CompositeStrategy`
and register it as one unit.

**Status: DONE.** `CompositeStrategy` is now a proper `LearningStrategy(nn.Module)` subclass. Sub-strategies stored in `self.sub_strategies: nn.ModuleList`. `setup()`, `ensure_setup()`, `compute_update()`, and `update_temporal_parameters()` all propagate to sub-strategies. Cortex refactored to use `CompositeStrategy` instances (`composite_l23`, etc.) instead of bare `nn.ModuleList`.

---

### ✅ LEARN-ARCH-03: Per-synapse `_learning_strategies` dict on `NeuralRegion` (ARCH-03)

**File:** `src/thalia/brain/regions/neural_region.py`

**Full replacement of `self.learning_strategy: Optional[LearningStrategy] = None`:**

```python
# In __init__:
self._learning_strategies: SynapseIdModuleDict = SynapseIdModuleDict()

# New helpers:
def add_learning_strategy(
    self,
    synapse_id: SynapseId,
    strategy: LearningStrategy,
    n_pre: Optional[int] = None,
    n_post: Optional[int] = None,
) -> None:
    """Register a learning strategy for a synapse.

    If n_pre / n_post are provided, calls strategy.setup() immediately.
    Otherwise, BrainBuilder.build() must call setup() later.
    """
    self._learning_strategies[synapse_id] = strategy
    if n_pre is not None and n_post is not None:
        strategy.setup(n_pre, n_post, self.device)

def get_learning_strategy(self, synapse_id: SynapseId) -> Optional[LearningStrategy]:
    return self._learning_strategies.get(synapse_id)

def apply_learning(
    self,
    synapse_id: SynapseId,
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    **kwargs,
) -> None:
    """Apply plasticity to one synapse: update weights + clamp in one call.

    Centralizes the `weights.data = strategy.compute_update(...)` +
    `clamp_weights(weights.data, ...)` pattern currently copy-pasted in
    Cortex, Hippocampus, Thalamus, PFC, Striatum.
    """
    strategy = self._learning_strategies.get(synapse_id)
    if strategy is None or not WeightInitializer.GLOBAL_LEARNING_ENABLED:
        return
    weights = self.get_synaptic_weights(synapse_id)
    weights.data = strategy.compute_update(
        weights.data, pre_spikes, post_spikes, **kwargs
    )
    clamp_weights(weights.data, self.config.w_min, self.config.w_max)

def update_temporal_parameters(self, dt_ms: float) -> None:
    # ... existing STP update ...
    for strategy in self._learning_strategies.values():
        strategy.update_temporal_parameters(dt_ms)
```

**Migration:**
- Cortex: delete `strategies_l23/l4/l5/l6a/l6b`. Create one `CompositeStrategy(STDP + BCM)`
  per synapse id, register via `add_learning_strategy()`. Replace the `_apply_plasticity()`
  body with `self.apply_learning(synapse_id, ...)` calls.
- Hippocampus: same — replace `self.learning_strategy` single field.
- Thalamus, PFC: same.
- Striatum: migrate to D1/D2-specific strategies (see LEARN-ARCH-04).
**Status: DONE.** `self._learning_strategies: SynapseIdModuleDict` added to `NeuralRegion.__init__`. `SynapseIdModuleDict.get()` added. `add_learning_strategy()`, `get_learning_strategy()`, and `apply_learning()` helpers implemented on `NeuralRegion`. `update_temporal_parameters()` iterates `_learning_strategies` with `id()`-based deduplication. Cortex `_apply_plasticity` migrated to use `composite_lXX.compute_update()` instance calls; all 9 old `CompositeStrategy.compute_update(strategies=…)` static calls removed. Backward-compat `self.learning_strategy` field retained for Hippocampus, Thalamus, and PFC.
---

### ✅ LEARN-ARCH-04: D1Strategy / D2Strategy for Striatum

**File:** `src/thalia/learning/strategies.py` (new) + `striatum.py`

The Striatum's hand-rolled three-factor logic with inverted D2 learning
should become first-class strategies:

```python
@dataclass
class D1STDPConfig(STDPConfig):
    """D1 MSN: DA+ → LTP, DA- → LTD (canonical corticostriatal)."""
    fast_trace_tau_ms: float = 200.0   # Fast eligibility (immediate)
    slow_trace_tau_ms: float = 1500.0  # Slow eligibility (Yagishita 2014: ~1s window)
    slow_trace_weight: float = 0.3     # Combined: fast + 0.3*slow

@dataclass
class D2STDPConfig(D1STDPConfig):
    """D2 MSN: DA+ → LTD, DA- → LTP (inverted by Gi-coupling)."""
    pass

class D1STDPStrategy(LearningStrategy):
    """Three-factor STDP for D1 MSNs: Δw = (fast_elig + α*slow_elig) × DA × lr."""

    def setup(self, n_pre, n_post, device):
        self.fast_trace = EligibilityTraceManager(n_pre, n_post, self.config, device)
        self.slow_trace = EligibilityTraceManager(n_pre, n_post, self.config, device)
        self.register_buffer("firing_rates", torch.zeros(n_post, device=device))

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        da = kwargs.get("dopamine", 0.0)  # Per-neuron tensor [n_post] or scalar
        if isinstance(da, torch.Tensor):
            da_modulator = da.unsqueeze(1)  # [n_post, 1] for broadcasting
        else:
            da_modulator = da

        self.fast_trace.update_traces(pre_spikes, post_spikes, self._dt_ms)
        self.slow_trace.update_traces(pre_spikes, post_spikes, self._dt_ms)
        ltp_fast, ltd_fast = self.fast_trace.compute_ltp_ltd_separate(pre_spikes, post_spikes)
        ltp_slow, ltd_slow = self.slow_trace.compute_ltp_ltd_separate(pre_spikes, post_spikes)

        cfg = self.config
        fast_elig = ltp_fast - ltd_fast if isinstance(ltp_fast, torch.Tensor) else 0
        slow_elig = ltp_slow - ltd_slow if isinstance(ltp_slow, torch.Tensor) else 0
        combined = (fast_elig + cfg.slow_trace_weight * slow_elig
                    if isinstance(fast_elig, torch.Tensor) else slow_elig)

        if not isinstance(combined, torch.Tensor):
            return weights

        dw = cfg.learning_rate * combined * da_modulator
        return clamp_weights(weights + dw, cfg.w_min, cfg.w_max, inplace=False)

class D2STDPStrategy(D1STDPStrategy):
    """Identical to D1 except DA signal is inverted (Gi-coupled)."""
    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        da = kwargs.get("dopamine", 0.0)
        if isinstance(da, torch.Tensor):
            kwargs = {**kwargs, "dopamine": -da}
        else:
            kwargs = {**kwargs, "dopamine": -da}
        return super().compute_update(weights, pre_spikes, post_spikes, **kwargs)
```

---

### ✅ LEARN-ARCH-05: `PredictiveCodingStrategy` for Cortex anti-Hebbian weights

**File:** `src/thalia/learning/strategies.py` (new)

**Problem:**
The L5→L4 and L6→L4 anti-Hebbian updates in Cortex are raw outer products applied
directly to weight tensors outside the strategy framework:
```python
dW_l5 = torch.outer(l4_spikes_float, prev_l5_spikes.float())
l5_l4_weights.data.add_(pred_lr * dW_l5)
```
This bypasses the strategy system, doesn't benefit from `apply_learning()`, and can't be
replaced or reconfigured without editing Cortex.

**Fix:**
```python
@dataclass
class PredictiveCodingConfig(LearningConfig):
    """Anti-Hebbian learning for top-down prediction synapses.

    Δw[j,i] = +lr × post[j] × pre[i]   (anti-Hebbian: co-activation → suppress)
    Biology: L5/L6 prediction signals are strengthened whenever L4
    (the prediction target) fires unexpectedly after them.
    Weight sign: inhibitory (w represents suppression strength).
    """
    prediction_delay_steps: int = 1   # Temporal offset (predict Δt=1ms ahead)

class PredictiveCodingStrategy(LearningStrategy):
    """Anti-Hebbian rule: Δw = +lr × post_current × pre_delayed."""

    def setup(self, n_pre, n_post, device):
        self.register_buffer("pre_spike_buffer",
                             torch.zeros(self.config.prediction_delay_steps, n_pre, device=device))

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        # Read delayed pre-spikes (causal prediction: pre was active, post reacted)
        delayed_pre = self.pre_spike_buffer[0]  # oldest
        # Anti-Hebbian: post fires (error signal) × what-predicted-it (delayed pre)
        dw = self.config.learning_rate * torch.outer(post_spikes.float(), delayed_pre.float())
        # Shift buffer
        self.pre_spike_buffer = torch.roll(self.pre_spike_buffer, 1, 0)
        self.pre_spike_buffer[0] = pre_spikes.float()
        return clamp_weights(weights + dw, self.config.w_min, self.config.w_max, inplace=False)
```

---

### ✦ LEARN-ARCH-06: Unify duplicate weight-clamping (`compute_update` vs. caller)

**Files:** `strategies.py`, all region files

**Problem:**
`STDPStrategy.compute_update()` calls `clamp_weights()` internally before returning.
The calling region (Cortex) then calls `clamp_weights()` again on the returned tensor.
Double-clamping is harmless but redundant. More importantly, `BCMStrategy` and
`ThreeFactorStrategy` do NOT clamp internally — the caller must always clamp.
This inconsistency is a maintenance hazard.

**Fix:**
- `compute_update()` must NEVER clamp weights internally. It returns a raw `dw`-applied tensor.
- `apply_learning()` on `NeuralRegion` owns the clamp — exactly one place.
- All internal `clamp_weights()` calls inside `compute_update()` are removed.

---

## Part C — Biological Accuracy Improvements in Learning

---

### ✦ LEARN-BIO-01: Separate `tau_plus` and `tau_minus` for LTP/LTD traces

**File:** `src/thalia/learning/eligibility_trace_manager.py`

**Problem:**
`update_traces` uses `tau_plus` for all trace decay. The `tau_minus` field in
`STDPConfig` is defined but ignored:
```python
trace_decay = 1.0 - dt_ms / self.config.tau_plus   # tau_minus never used
```
Classical STDP has separate time constants: tau_plus (~20ms, LTP) ≠ tau_minus (~20ms, LTD)
but they are independently tunable. Some synapses (e.g., thalamocortical) have asymmetric
windows (tau_plus=20ms, tau_minus=40ms).

**Fix:**
```python
def update_traces(self, input_spikes, output_spikes, dt_ms):
    decay_plus  = exp(-dt_ms / self.config.tau_plus)
    decay_minus = exp(-dt_ms / self.config.tau_minus)
    self.input_trace  = self.input_trace  * decay_plus  + input_float
    self.output_trace = self.output_trace * decay_minus + output_float
```
LTP uses `input_trace` (decayed with `tau_plus`), LTD uses `output_trace` (decayed with `tau_minus`).
Pre-compute the two decay factors in `setup()` / `update_temporal_parameters()`.

---

### ✦ LEARN-BIO-02: Implement heterosynaptic plasticity (currently dead config field)

**File:** `src/thalia/learning/strategies.py` — `STDPStrategy`

**Problem:**
`STDPConfig.heterosynaptic_ratio: float = 0.3` is defined and documented as
"Fraction of LTD applied to non-active synapses" but is **never used** in `compute_update()`.
Heterosynaptic LTD (Bhatt et al. 2009) is a key stabilizer: synapses that did NOT contribute
to the postsynaptic response are weakened, freeing resources and preventing saturation.

**Fix:**
After computing the weight update `dw`, apply a weak global depression to all synapses
scaled by how much LTD was applied homosynaptically:
```python
# Heterosynaptic LTD: non-active synapses receive a small depression
if isinstance(ltd, torch.Tensor) and cfg.heterosynaptic_ratio > 0:
    # Active synapses: where pre_spikes > 0 (already got homosynaptic LTD)
    active_mask = (pre_spikes.float() > 0).unsqueeze(0)  # [1, n_pre]
    inactive_mask = ~active_mask
    # Mean homosynaptic LTD as scaling reference
    homo_ltd_mean = ltd.mean()
    # Apply a fraction of that as uniform heterosynaptic depression to inactive synapses
    hetero_depression = cfg.heterosynaptic_ratio * homo_ltd_mean * inactive_mask
    dw = dw - hetero_depression
```

---

### ✦ LEARN-BIO-03: Fix retrograde signaling model (biologically reversed)

**File:** `src/thalia/learning/strategies.py` — `STDPStrategy`

**Problem:**
The retrograde signal is used to **gate postsynaptic LTP**. In reality, biological retrograde
messengers (endocannabinoids: 2-AG, AEA) are released from **postsynaptic** dendrites and
diffuse backward to **presynaptic** terminals where they suppress glutamate/GABA release
(depolarization-induced suppression of excitation/inhibition, DSE/DSI). They act on a
timescale of seconds. They do NOT gate Hebbian LTP.

The current use case (track strong postsynaptic activity → gate LTP) is effectively a
postsynaptic activity threshold, which is better modeled as part of the BCM sliding threshold
or a minimum-activity gate. The retrograde mechanism is there to model DSE/DSI for
inhibitory gating of plasticity windows — specifically relevant to the VIP→SST→apical
pathway (BIO-02).

**Fix option A (remove):** Remove the retrograde signaling from `STDPStrategy` and
implement the minimum-activity gating explicitly as `activity_threshold` (already present).

**Fix option B (repurpose correctly):** Implement `retrograde_signal` as a presynaptic
**release modulation** factor. When strong postsynaptic activity is detected, the retrograde
signal suppresses the effective weight of subsequent pre→post conductance (reduces `source_conductance`
proportional to retrograde strength). This is handled at the region level in
`_integrate_synaptic_inputs_at_dendrites()`, not inside `compute_update()`.

**Recommended:** Option A for `STDPStrategy`. Option B as a separate `DSEModulator` module
added to region synaptic integration when BIO-01 (compartments) is implemented and VIP
disinhibition (BIO-02) becomes relevant.

---

### ✦ LEARN-BIO-04: BCM theta tracks instantaneous spikes, not firing rate

**File:** `src/thalia/learning/strategies.py` — `BCMStrategy._update_theta`

**Problem:**
```python
c = post_spikes.float()   # Binary {0, 1} per timestep
c_p = c.pow(cfg.p)        # Still binary at p=2
```
At dt=1ms, post_spikes is overwhelmingly 0 most timesteps (typical firing rate 5–20 Hz
≈ 0.5–2% duty cycle). The EMA of c² spends ~98% of time at 0, causing theta to collapse
toward zero except during the rare spike. This makes the BCM threshold ineffective as a
homeostatic signal.

**Fix:**
Track a **firing rate EMA** (already computed as `self.firing_rates`) and feed that into
theta, not the instantaneous binary spike:

```python
def _update_theta(self, post_spikes: torch.Tensor) -> None:
    # Use firing rate (EMA) not instantaneous spikes for theta tracking
    # This is biologically correct: theta tracks integrated calcium, not individual spikes
    c = self.firing_rates  # EMA firing rate [n_post], range [0, 1]
    c_p = c.pow(cfg.p)
    self.theta = self.decay_theta * self.theta + (1 - self.decay_theta) * c_p
    self.theta = self.theta.clamp(cfg.theta_min, cfg.theta_max)
```

The `compute_phi` function can keep using instantaneous `post_spikes.float()` (the BCM
potentiation/depression decision is instantaneous), but `theta` now tracks a meaningful
average.

---

### ✦ LEARN-BIO-05: Per-synapse BCM theta (currently per-strategy-instance, shared)

**File:** `src/thalia/brain/regions/cortex/cortex.py`

**Problem:**
Cortex creates ONE `BCMStrategy` per layer (`strategies_l23[1]`) but applies it to
MULTIPLE synapses targeting that layer (thalamic input, L4→L23, L23 recurrent, etc.).
They all share the same `theta` tensor. BCM homeostasis is biologically **per-neuron
postsynaptic** (tracks that neuron's history), not per-connection. With the current sharing,
the theta of L2/3 neurons is influenced by ALL their presynaptic sources jointly.

This is correct if the theta is understood as a postsynaptic property (one per postsynaptic
population). The key requirement is: one `BCMStrategy` per **postsynaptic population**, shared
by all connections targeting that population. This is what the current Cortex code achieves
for a layer sharing a strategy — but it only works for populations with the same n_post.

With ARCH-03 (per-synapse dict), the natural way to preserve this is:
- Create one `BCMStrategy` instance per postsynaptic population.
- Register the SAME BCM instance as the learning strategy for all synapses targeting that population.
- `SynapseIdModuleDict` would hold references to the same BCM instance multiple times.

This requires `SynapseIdModuleDict` to support shared module references (multiple keys →
same object). Currently `nn.ModuleDict` deduplicates on object identity — this needs checking.
Alternatively, use a **population-keyed BCM registry** separate from the per-synapse strategy:

```python
# In NeuralRegion:
self._population_bcm: nn.ModuleDict = nn.ModuleDict()  # PopulationName → BCMStrategy

def get_or_create_bcm(self, pop_name: PopulationName, n_post: int) -> BCMStrategy:
    if pop_name not in self._population_bcm:
        self._population_bcm[pop_name] = BCMStrategy(self.bcm_config)
        self._population_bcm[pop_name].setup(0, n_post, self.device)  # n_pre=0, BCM only uses n_post
    return self._population_bcm[pop_name]
```

STDP strategies remain per-synapse (they track pre-specific trace state).
BCM strategies are per-postsynaptic-population.

In `CompositeStrategy`, the BCM component is retrieved from the population registry; the
STDP component is unique per synapse.

---

### ✦ LEARN-BIO-06: Eligibility trace decay should be exponential, not linear

Already covered under LEARN-BUG-06 (EligibilityTraceManager traces) — same fix applies
to the eligibility accumulation decay in `accumulate_eligibility()` and to `ThreeFactorStrategy`'s
`decay_elig`. All linear `1.0 - dt/tau` decays → `exp(-dt/tau)`.

---

### ✦ LEARN-BIO-07: Spike-timing asymmetry window for different synapse types

**Files:** `src/thalia/learning/strategies.py`, region files

**Problem:**
All synapses use the same STDP window (tau_plus=20ms, tau_minus=20ms, symmetric).
Biology is pathway-specific:
- Thalamocortical: tau_plus≈20ms, tau_minus≈40ms (wider LTD window, matches sensory learning)
- CA3→CA1 (Schaffer): tau_plus≈20ms, tau_minus≈20ms (symmetric, standard)
- CA3 recurrent: tau_plus≈20ms, tau_minus≈10ms (asymmetric, favors LTP for sequence encoding)
- Corticostriatal: tau_plus≈500ms (eligibility-dominated, reward delayed)
- Cere. parallel fiber→Purkinje: ANTI-Hebbian (LTD when pre+post coincide after CF error)

**Fix:**
The `STDPConfig` already has both `tau_plus` and `tau_minus`. The issue is that:
1. Both are set to 20ms everywhere (copy-paste default).
2. They are not passed per-synapse by BrainBuilder — a single config is used.

After ARCH-03, `BrainBuilder.build()` can pass per-connection `STDPConfig` instances
with pathway-appropriate tau values. This requires `add_learning_strategy()` to accept a
config argument or the strategy to be pre-constructed by the builder.

**Immediate action:**
Document and use biologically correct defaults per region:
```python
# thalamus.py
stdp_cfg = STDPConfig(tau_plus=20.0, tau_minus=40.0, ...)   # Thalamocortical

# hippocampus.py (CA3 recurrent)
stdp_cfg = STDPConfig(tau_plus=20.0, tau_minus=10.0, ...)   # Sequence encoding asymmetry

# striatum.py (corticostriatal via D1/D2)
# tau is eligibility_tau, not trace tau — already 1000ms in StriatumConfig
```

---

### ✦ LEARN-BIO-08: Homeostatic synaptic scaling as a formal `LearningStrategy`

**File:** `src/thalia/brain/regions/neural_region.py` — `_apply_synaptic_scaling`

**Problem:**
`_apply_synaptic_scaling()` is a method on `NeuralRegion` that iterates
ALL synaptic weights targeting a given population and scales them. It's applied
inconsistently (only Cortex; Hippocampus, Striatum don't get it).

**Fix:**
```python
@dataclass
class SynapticScalingConfig(LearningConfig):
    min_activity: float = 0.005    # Target minimum population rate
    max_scale_factor: float = 3.0  # Maximum multiplicative scaling
    tau_scaling: float = 600000.0  # Very slow homeostatic timescale (10 min in ms)

class SynapticScalingStrategy(LearningStrategy):
    """Multiplicative homeostatic scaling (Turrigiano 2008).

    Applied at population level: if mean firing rate falls below target,
    scale all incoming weights up synchronously. Slow timescale (minutes).
    """
    def setup(self, n_pre, n_post, device):
        self.register_buffer("scale_factor", torch.ones((), device=device))
        self.register_buffer("mean_firing_rate", torch.zeros((), device=device))

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        cfg = self.config
        # Update mean firing rate
        alpha = self._dt_ms / cfg.tau_scaling  # Very slow EMA
        self.mean_firing_rate = (1 - alpha) * self.mean_firing_rate + alpha * post_spikes.float().mean()

        if self.mean_firing_rate < cfg.min_activity:
            deficit = cfg.min_activity - self.mean_firing_rate
            scale_update = 1.0 + alpha * deficit / cfg.min_activity
            self.scale_factor = (self.scale_factor * scale_update).clamp(max=cfg.max_scale_factor)
            return weights * self.scale_factor
        return weights
```

Add to region via `add_learning_strategy()` per synapse. Remove `_apply_synaptic_scaling()`
from `NeuralRegion`. Extend to Hippocampus and Striatum (covers BACK-01).

---

## Part D — `BrainBuilder` Integration

---

### ✦ LEARN-BUILD-01: `BrainBuilder.build()` calls `setup()` on all strategies

**File:** `src/thalia/brain/brain_builder.py`

After all connections are registered, `BrainBuilder.build()` iterates every region,
every registered synapse, and calls `strategy.setup(n_pre, n_post, device)`:

```python
def build(self) -> DynamicBrain:
    # ... existing build logic ...

    # Initialize all learning strategies with final dimensions
    for region_name, region in brain.regions.items():
        for synapse_id, strategy in region._learning_strategies.items():
            weights = region.get_synaptic_weights(synapse_id)
            n_post, n_pre = weights.shape
            strategy.setup(n_pre, n_post, region.device)

    return brain
```

This guarantees all strategy state is registered before step 0. No lazy init anywhere.

---

### ✦ LEARN-BUILD-02: `BrainBuilder` exposes per-connection learning configuration

**File:** `src/thalia/brain/brain_builder.py`

Currently callers wire connections without specifying learning strategies. After ARCH-03,
`add_connection()` (or equivalent) accepts an optional `strategy` parameter:

```python
def add_connection(
    self,
    source, target,
    *,
    connectivity: float,
    weight_scale: float,
    learning_strategy: Optional[LearningStrategy] = None,
    stp_config: Optional[STPConfig] = None,
) -> SynapseId:
    synapse_id = ...
    target_region.add_input_source(synapse_id, n_pre, connectivity, weight_scale,
                                   stp_config=stp_config)
    if learning_strategy is not None:
        target_region.add_learning_strategy(synapse_id, learning_strategy)
    return synapse_id
```

Region constructors no longer hard-code strategy instantiation — they're supplied by
the builder using per-pathway configs.

---

## Suggested Execution Order

```
Phase A — Critical infrastructure bugs (do first, blocks everything else)
  ✅ LEARN-BUG-01  EligibilityTraceManager → nn.Module with register_buffer
  ✅ LEARN-BUG-02  STDPStrategy firing_rates / retrograde_signal → register_buffer
  ✅ LEARN-BUG-03  BCMStrategy / ThreeFactorStrategy → remove fragile lazy register pattern
  ✅ LEARN-BUG-04  Cortex strategies_l2/3/4/5 → move into per-synapse dict (ties to Phase B)
  ✅ LEARN-BUG-05  Striatum eligibility dicts → SynapseIdBufferDict (complete; ARCH-04 migration also done)
  ✅ LEARN-BUG-06  Linear decay → exp(-dt/tau) in all trace/eligibility decay

Phase B — Architecture consolidation (ARCH-02 + ARCH-03 expanded)
  ✅ LEARN-ARCH-01  setup() protocol on LearningStrategy base
  ✅ LEARN-ARCH-02  CompositeStrategy → proper nn.Module with nn.ModuleList
  ✅ LEARN-ARCH-03  NeuralRegion: _learning_strategies dict + add/get/apply helpers
  ✅ LEARN-ARCH-04  D1STDPStrategy / D2STDPStrategy for Striatum
  ✅ LEARN-ARCH-05  PredictiveCodingStrategy for Cortex anti-Hebbian synapses
  ✅ LEARN-ARCH-06  Consolidate weight clamping: only in apply_learning(), never in compute_update()
  ♆ LEARN-BUILD-01 BrainBuilder.build() calls strategy.setup() for all synapses
  ♆ LEARN-BUILD-02 BrainBuilder.add_connection() accepts per-connection learning strategies

Phase C — Biological accuracy (learning-specific)
  LEARN-BIO-01  Separate tau_plus / tau_minus decay for LTP vs LTD traces
  LEARN-BIO-02  Implement heterosynaptic plasticity (STDPConfig field already exists, never used)
  LEARN-BIO-03  Fix retrograde signaling: remove from compute_update(); repurpose as DSE/DSI
  LEARN-BIO-04  BCM theta tracks firing rate EMA, not instantaneous binary spikes
  LEARN-BIO-05  Per-population BCM registry (shared theta across all synapses targeting same pop)
  LEARN-BIO-07  Pathway-specific STDP windows (thalamocortical, CA3, corticostriatal)
  LEARN-BIO-08  Homeostatic synaptic scaling → SynapticScalingStrategy; extend to Hippo, Striatum
```

---

## Summary of What the Existing Plan Misses

| Gap | Severity | Status | Notes |
|-----|----------|--------|-------|
| ✅ `EligibilityTraceManager` not an `nn.Module` | **Critical** | Done | All STDP trace state lost on `.to()` / checkpoint |
| ✅ `STDPStrategy.firing_rates` not a buffer | **Critical** | Done | Lost on `.to()` / checkpoint |
| ✅ Cortex strategy lists are plain Python lists | **Critical** | Done | All cortex learning state invisible to PyTorch |
| ✅ Striatum eligibility dicts are plain dicts | **Critical** | Done | Migrated to `SynapseIdBufferDict`; D1/D2 strategy migration completed (LEARN-ARCH-04) |
| Double weight-clamping (in strategy AND caller) | Moderate | Pending | Inconsistency, potential confusion |
| ✅ Linear decay vs `exp(-dt/tau)` | Moderate | Done | Wrong for small tau or large dt; oscillates |
| BCM theta tracking binary spikes | Moderate | Pending | Ineffective homeostasis at dt=1ms |
| Retrograde signaling model biologically reversed | Moderate | Pending | Not standard eCB-DSE/DSI |
| Heterosynaptic ratio config field never implemented | Low | Pending | Dead config parameter |
| Per-population BCM theta not explicit contract | Low | Pending | Currently works by accident |
| Pathway-specific STDP window asymmetry | Low | Pending | All synapses use same tau_plus=tau_minus=20ms |
| ✅ `CompositeStrategy` is a static class, not `nn.Module` | **Critical** | Done | Replaced with proper `LearningStrategy` subclass |
