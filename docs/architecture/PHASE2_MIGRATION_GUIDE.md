# Phase 2 Migration Guide: Region → NeuralRegion

**Status**: In Progress (5/6 Complete: Striatum ✅, PFC ✅, Hippocampus ✅, LayeredCortex ✅, Thalamus ✅)
**Target**: Migrate all brain regions to NeuralRegion base class
**Estimated Time**: 2-3 weeks
**Prerequisites**: Phase 1 complete ✅
**Last Updated**: 2025-12-19

---

## Overview

Phase 2 migrates existing region implementations to use the biologically-accurate NeuralRegion base class created in Phase 1. This moves synaptic weights from pathways (axons) to regions (dendrites) where they belong.

**Key Principle**: External connections use `synaptic_weights` dict, internal connections stay as-is.

---

## Key Insights from Striatum Migration

The completed Striatum migration revealed critical architectural patterns that apply to all regions:

### 1. **Property-Based Weight Access Pattern**
```python
# In pathway (e.g., D1Pathway, D2Pathway):
@property
def weights(self) -> torch.Tensor:
    return self._parent_striatum.synaptic_weights[self._weight_source]

@weights.setter
def weights(self, value: torch.Tensor) -> None:
    self._parent_striatum.synaptic_weights[self._weight_source].data = value
```
**Why this works**: All operations (torch.matmul, weights.data =, learning updates) transparently access parent's storage. No code changes needed in pathways or learning components.

### 2. **Opponent Pathways Need Separate Matrices**
- D1 and D2 are **separate full populations**, not split populations
- Each has shape `[n_output, n_input]` (both full size)
- **Why**: D1 (DA+ → LTP) and D2 (DA+ → LTD) have opposite learning rules
- **Biological accuracy**: Different synaptic weights for same input based on receptor type

### 3. **Forward Flow Critical Pattern**
```python
# WRONG: Pre-processing currents
currents = self._consolidate_inputs(inputs)  # Returns [n_output]
output = self.forward_coordinator(currents)  # Expects [n_input]!

# RIGHT: Pass raw spikes
raw_spikes = torch.cat([inputs[k] for k in inputs])  # [n_input]
output = self.forward_coordinator({"default": raw_spikes})  # Applies weights internally
```
**Lesson**: Let internal coordinators handle weight application. Parent only concatenates/routes inputs.

### 4. **State Management Differs by Base Class**
- **BrainComponentBase**: Uses `self.state.spikes`, `self.state.t`
- **NeuralRegion**: Uses `self.output_spikes`, no timestep tracking
- **Migration**: Change all `self.state.spikes` → `self.output_spikes`

### 5. **Checkpoint Loading with Properties**
```python
# WRONG: Tries to create nn.Parameter
self.weights = nn.Parameter(state['weights'])  # Fails on property

# RIGHT: Use property setter
self.weights = state['weights']  # Calls @weights.setter
```

### 6. **Backward Compatibility During Migration**
```python
def forward(self, inputs: Union[Dict[str, Tensor], Tensor]) -> Tensor:
    if isinstance(inputs, torch.Tensor):
        inputs = {"default": inputs}  # Auto-wrap for old tests
    # ... proceed with Dict processing
```
**Benefit**: Tests can be updated incrementally, not all at once.

---

## Migration Strategy

### 1. **Priority Order** (Simplest → Most Complex)

| Order | Region | Status | Est. Time | Actual Time | Reason |
|-------|--------|--------|-----------|-------------|--------|
| 1 | **Striatum** | ✅ COMPLETE | 2-3 days | 1 day | Already multi-source, clear separation |
| 2 | **PFC** | ✅ COMPLETE | 2-3 days | 1 day | Working memory, simpler than cortex |
| 3 | **Hippocampus** | ✅ COMPLETE | 3-4 days | 1 day | DG→CA3→CA1 chain, 4 EC pathways |
| 4 | **LayeredCortex** | ✅ COMPLETE | 4-5 days | 1 day | Complex laminar structure, internal cascade |
| 5 | **Thalamus** | ✅ COMPLETE | 3-4 days | 1 day | Unique: NO external weights (sensory relay) |
| 6 | **Cerebellum** | 🔄 Next | 3-4 days | - | Granule layer, Purkinje cells, DCN |

### 2. **Migration Pattern**

Each region follows this template:

```python
class RegionName(NeuralRegion):
    def __init__(self, config: RegionConfig):
        # 1. Initialize NeuralRegion with total neurons
        super().__init__(
            n_neurons=config.n_output,
            neuron_config=config.neuron_config,  # Optional
            default_learning_rule=config.default_learning_rule,  # Optional
            device=config.device,
            dt_ms=config.dt_ms,
        )

        # 2. Store config
        self.region_config = config

        # 3. Create internal neurons (sub-populations)
        self.population_a = ConductanceLIF(size_a, ...)
        self.population_b = ConductanceLIF(size_b, ...)

        # 4. Internal weights (KEEP THESE)
        self.w_internal = nn.Parameter(...)  # Between internal populations

        # 5. External weights (DELETE THESE - moved to synaptic_weights)
        # OLD: self.w_external = nn.Parameter(...)  # ❌ Remove!
        # NEW: Registered via add_input_source() in build()

        # 6. Other components (buffers, learning rules, etc.)
        self.replay_buffer = ...
        self.td_lambda = ...

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process multi-source inputs.

        Args:
            inputs: Dict mapping source names to spike tensors
                   e.g., {"cortex": [n_cortex], "hippocampus": [n_hipp]}

        Returns:
            Output spikes [n_neurons]
        """
        # 1. Apply synaptic weights for each source (inherited from NeuralRegion)
        currents = []
        for source_name, input_spikes in inputs.items():
            if source_name in self.synaptic_weights:
                current = self._apply_synapses(source_name, input_spikes)
                currents.append(current)

        total_current = torch.sum(torch.stack(currents), dim=0) if currents else torch.zeros(self.n_neurons, device=self.device)

        # 2. Internal processing (unchanged)
        # ... region-specific computation ...

        # 3. Generate output spikes
        output_spikes = self.neurons(total_current)

        # 4. Update learning (if enabled)
        for source_name in self.plasticity_rules:
            if source_name in inputs:
                new_weights, _ = self.plasticity_rules[source_name].compute_update(
                    weights=self.synaptic_weights[source_name],
                    pre=inputs[source_name],
                    post=output_spikes,
                )
                self.synaptic_weights[source_name].data = new_weights

        return output_spikes
```

### 3. **Checklist Per Region**

- [ ] Inherit from `NeuralRegion` instead of `NeuralComponent`
- [ ] Call `super().__init__(n_neurons, ...)` with total output size
- [ ] Identify external vs internal weights
- [ ] Remove external weight Parameters (move to `synaptic_weights`)
- [ ] Update `forward()` to accept `Dict[str, Tensor]`
- [ ] Apply synaptic weights via `_apply_synapses()`
- [ ] Keep internal processing logic unchanged
- [ ] Update tests to use Dict input
- [ ] Verify learning still works
- [ ] Check state_dict/checkpoint compatibility

---

## Region-Specific Migration Plans

### 🎯 **1. Striatum** (Start Here - Clearest Example)

**File**: `src/thalia/regions/striatum/striatum_region.py`
**Lines**: ~850
**Complexity**: Medium (already designed for multi-source)

#### Current Architecture
```python
class Striatum(NeuralComponent):
    def __init__(self, config):
        # D1 and D2 MSN populations
        self.d1_neurons = ConductanceLIF(d1_size, ...)
        self.d2_neurons = ConductanceLIF(d2_size, ...)

        # External weights (FROM PATHWAYS - WRONG!)
        # These are created in pathways, not here

        # Internal weights (CORRECT - stay here)
        self.d1_d2_inhibition = nn.Parameter(...)
        self.d2_d1_inhibition = nn.Parameter(...)
```

#### Target Architecture
```python
class Striatum(NeuralRegion):
    def __init__(self, config):
        super().__init__(
            n_neurons=config.n_d1 + config.n_d2,
            default_learning_rule="three_factor",  # Dopamine-gated
            device=config.device,
        )

        self.n_d1 = config.n_d1
        self.n_d2 = config.n_d2

        # Internal populations (unchanged)
        self.d1_neurons = ConductanceLIF(self.n_d1, ...)
        self.d2_neurons = ConductanceLIF(self.n_d2, ...)

        # Internal weights (unchanged)
        self.d1_d2_inhibition = nn.Parameter(...)
        self.d2_d1_inhibition = nn.Parameter(...)

        # External weights registered dynamically:
        # - add_input_source("cortex", n_input=128, learning_rule="three_factor")
        # - add_input_source("hippocampus", n_input=64, learning_rule="three_factor")
        # - add_input_source("pfc", n_input=32, learning_rule=None)  # No learning

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Apply synaptic weights
        cortical_current = self._apply_synapses("cortex", inputs["cortex"])
        hippo_current = self._apply_synapses("hippocampus", inputs["hippocampus"])
        pfc_current = self._apply_synapses("pfc", inputs["pfc"])

        # Split into D1/D2 pathways (internal routing)
        total_current = cortical_current + hippo_current + pfc_current
        d1_current = total_current[:self.n_d1]
        d2_current = total_current[self.n_d1:]

        # D1/D2 processing (unchanged)
        d1_spikes = self.d1_neurons(d1_current)
        d2_spikes = self.d2_neurons(d2_current)

        # Apply cross-inhibition (unchanged)
        d1_inhibition = torch.matmul(self.d1_d2_inhibition, d2_spikes)
        d2_inhibition = torch.matmul(self.d2_d1_inhibition, d1_spikes)

        # Re-fire with inhibition (simplified)
        d1_final = (d1_spikes * (1 - d1_inhibition)) > 0.5
        d2_final = (d2_spikes * (1 - d2_inhibition)) > 0.5

        return torch.cat([d1_final, d2_final])
```

#### Migration Steps
1. Change inheritance: `NeuralComponent` → `NeuralRegion`
2. Update `__init__` to call `super().__init__(n_neurons=n_d1+n_d2)`
3. Remove any external weight creation (if present)
4. Update `forward(input)` → `forward(inputs: Dict)`
5. Update tests: `striatum(cortex_spikes)` → `striatum({"cortex": cortex_spikes})`
6. Run tests: `pytest tests/unit/regions/striatum/ -v`

**Expected Changes**: ~50 lines modified, ~30 lines removed

#### ✅ **MIGRATION COMPLETE** (2025-12-19)

**Commits**:
1. `8cb285c` - Part 1: Changed inheritance to NeuralRegion
2. `0aa9e30` - Part 2: Moved D1/D2 weights to parent's synaptic_weights dict
3. `0fb9494` - Fixed initialization order and Protocol issues
4. `0ce412d` - Added backward compatibility and helper methods
5. `9bd1135` - Fixed weight size bug and forward flow

**Architecture Implemented**:
- D1 weights: `synaptic_weights["default_d1"]` shape `[n_output, n_input]`
- D2 weights: `synaptic_weights["default_d2"]` shape `[n_output, n_input]`
- Property-based access: `@property weights` returns parent's tensor
- Biological accuracy: D1 and D2 are separate full populations, not split
- Each pathway has full weight matrix for opposite learning rules

**Key Fixes**:
1. **Weight Size**: Changed from `n_d1 = n_total // 2` to `n_d1 = n_total` (separate populations)
2. **Forward Flow**: Removed `_consolidate_inputs()`, pass raw inputs to forward_coordinator
3. **State Management**: Use `self.output_spikes` (NeuralRegion pattern) not `self.state.spikes`
4. **Checkpoint**: Use property setter `self.weights = state['weights']` not `nn.Parameter()`

**Test Results**:
- ✅ All 13 striatum D1/D2 delay tests passing
- ✅ Checkpoint save/restore working
- ✅ Property-based weight access validated
- ✅ Backward compatibility maintained

**Lessons Learned**:
1. **D1/D2 are opponent pathways**: Need separate full-size weight matrices for opposite learning rules (DA+ → LTP vs DA+ → LTD)
2. **Property pattern works perfectly**: torch.matmul(), weights.data =, and learning updates all transparent
3. **Forward flow critical**: Must pass raw [n_input] spikes, not pre-processed [n_output] currents
4. **State management matters**: NeuralRegion uses self.output_spikes not self.state
5. **Checkpoint loading**: Properties need setter, not nn.Parameter() assignment

**Files Modified**:
- `src/thalia/regions/striatum/striatum.py` (lines 799-835, 1477-1528, 1760-1787)
- `src/thalia/regions/striatum/pathway_base.py` (lines 87-163, 395)

**Total Changes**: ~70 lines modified, ~40 lines removed

---

### 🎯 **2. PFC** (Working Memory - Clear Single Purpose)

#### ✅ **MIGRATION COMPLETE** (2025-12-19)

**Commit**: `6bbe11f` - Phase 2: Migrate PFC to NeuralRegion

**Architecture Implemented**:
- Single input source: `synaptic_weights["default"]` shape `[n_output, n_input]`
- Internal weights preserved: `rec_weights` (recurrent), `inhib_weights` (lateral inhibition)
- Forward signature: `Union[Dict[str, Tensor], Tensor]` for backward compatibility
- State management: Uses `PrefrontalState` (custom state class)

**Key Changes**:
1. **Inheritance**: Changed from `NeuralComponent` to `NeuralRegion`
2. **Weight Storage**: Moved `self.weights` → `synaptic_weights["default"]`
3. **Forward Flow**: Pass raw input through synaptic weights, then internal processing
4. **Helper Methods**: Added `_reset_subsystems()`, `set_neuromodulators()`
5. **Checkpoint Manager**: Updated to access `synaptic_weights["default"]`
6. **Growth Methods**: Updated `_expand_layer_weights()`, `grow_input()`, `grow_output()`

**Test Results**:
- ✅ All 9 PFC checkpoint tests passing
- ✅ Neuromorphic checkpoint format working
- ✅ Save/load preserves state correctly
- ✅ Backward compatibility maintained

**Lessons Learned**:
1. **Single-source regions simpler**: PFC only has one external input, migration straightforward
2. **Internal weights stay as-is**: Recurrent and inhibitory weights remain `nn.Parameter`
3. **Checkpoint managers need updating**: External tools accessing `.weights` must be updated
4. **Custom state classes work**: `PrefrontalState` (with working_memory, update_gate) compatible with NeuralRegion
5. **Backward compat essential**: Union[Dict, Tensor] signature allows gradual test migration

**Files Modified**:
- `src/thalia/regions/prefrontal.py` (~105 lines modified, ~40 removed)
- `src/thalia/regions/prefrontal_checkpoint_manager.py` (4 references updated)
- `tests/unit/test_prefrontal_checkpoint_neuromorphic.py` (2 test updates)

**Total Time**: 1 day (same as Striatum)

---

### 🎯 **3. Hippocampus** (Episodic Memory - Chain Architecture)

**File**: `src/thalia/regions/prefrontal.py`
**Lines**: ~650
**Complexity**: Medium (gating + maintenance)

#### Current Architecture
```python
class PrefrontalCortex(NeuralComponent):
    def __init__(self, config):
        # WM neurons with gating
        self.wm_neurons = GatedWMNeurons(...)

        # Gating control
        self.gate_neurons = ConductanceLIF(...)
```

#### Target Architecture
```python
class PrefrontalCortex(NeuralRegion):
    def __init__(self, config):
        super().__init__(
            n_neurons=config.n_wm + config.n_gate,
            default_learning_rule="gated_hebbian",
            device=config.device,
        )

        # Internal populations (unchanged)
        self.wm_neurons = GatedWMNeurons(...)
        self.gate_neurons = ConductanceLIF(...)

        # External weights via add_input_source():
        # - "cortex" (sensory/contextual info)
        # - "hippocampus" (episodic retrieval)
        # - "striatum" (action outcomes)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Apply synaptic weights
        cortex_input = self._apply_synapses("cortex", inputs.get("cortex", None))
        hippo_input = self._apply_synapses("hippocampus", inputs.get("hippocampus", None))

        # Gating logic (unchanged)
        gate_signal = self.gate_neurons(cortex_input)

        # WM maintenance (unchanged)
        wm_output = self.wm_neurons(hippo_input, gate=gate_signal)

        return torch.cat([wm_output, gate_signal])
```

**Expected Changes**: ~40 lines modified, ~20 lines removed

---

### 🎯 **3. Hippocampus** (Episodic Memory - Chain Architecture)

**File**: `src/thalia/regions/hippocampus/hippocampus_region.py`
**Lines**: ~900
**Complexity**: High (DG→CA3→CA1 chain, replay, consolidation)

#### Current Architecture
```python
class Hippocampus(NeuralComponent):
    def __init__(self, config):
        # Tri-synaptic loop
        self.dg_neurons = ConductanceLIF(dg_size, ...)
        self.ca3_neurons = ConductanceLIF(ca3_size, ...)
        self.ca1_neurons = ConductanceLIF(ca1_size, ...)

        # Internal weights (KEEP)
        self.dg_ca3 = nn.Parameter(...)
        self.ca3_ca3_recurrent = nn.Parameter(...)  # Pattern completion
        self.ca3_ca1 = nn.Parameter(...)

        # Replay system
        self.replay_buffer = EpisodicReplayBuffer(...)
```

#### Target Architecture
```python
class Hippocampus(NeuralRegion):
    def __init__(self, config):
        super().__init__(
            n_neurons=config.n_dg + config.n_ca3 + config.n_ca1,
            default_learning_rule="hebbian",  # One-shot learning
            device=config.device,
        )

        # Sizes
        self.n_dg = config.n_dg
        self.n_ca3 = config.n_ca3
        self.n_ca1 = config.n_ca1

        # Internal populations (unchanged)
        self.dg_neurons = ConductanceLIF(self.n_dg, ...)
        self.ca3_neurons = ConductanceLIF(self.n_ca3, ...)
        self.ca1_neurons = ConductanceLIF(self.n_ca1, ...)

        # Internal weights (unchanged)
        self.dg_ca3 = nn.Parameter(...)
        self.ca3_ca3_recurrent = nn.Parameter(...)
        self.ca3_ca1 = nn.Parameter(...)

        # Entorhinal input via synaptic_weights:
        # - add_input_source("ec_l3", n_input=..., learning_rule="hebbian")
        # - add_input_source("cortex_l23", n_input=..., learning_rule="hebbian")

        # Replay system (unchanged)
        self.replay_buffer = EpisodicReplayBuffer(...)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Entorhinal input to DG
        ec_input = self._apply_synapses("ec_l3", inputs["ec_l3"])
        cortex_input = self._apply_synapses("cortex_l23", inputs.get("cortex_l23", None))

        # DG: Pattern separation
        dg_spikes = self.dg_neurons(ec_input + cortex_input)

        # CA3: Pattern completion (recurrent)
        ca3_input = torch.matmul(self.dg_ca3, dg_spikes)
        ca3_recurrent = torch.matmul(self.ca3_ca3_recurrent, self.ca3_prev_spikes)
        ca3_spikes = self.ca3_neurons(ca3_input + ca3_recurrent)

        # CA1: Output to cortex
        ca1_input = torch.matmul(self.ca3_ca1, ca3_spikes)
        ca1_spikes = self.ca1_neurons(ca1_input)

        # Store for next step
        self.ca3_prev_spikes = ca3_spikes

        return ca1_spikes  # Output from CA1
```

**Expected Changes**: ~60 lines modified, ~40 lines removed

---

### 🎯 **4. LayeredCortex** (Most Complex - Multiple Ports)

**File**: `src/thalia/regions/cortex/layered_cortex.py`
**Lines**: ~1200
**Complexity**: High (L4→L2/3→L5→L6, multiple input/output ports)

#### Current Architecture
```python
class LayeredCortex(NeuralComponent):
    def __init__(self, config):
        # Laminar structure
        self.l4_neurons = ConductanceLIF(l4_size, ...)
        self.l23_neurons = ConductanceLIF(l23_size, ...)
        self.l5_neurons = ConductanceLIF(l5_size, ...)
        self.l6_neurons = ConductanceLIF(l6_size, ...)

        # Internal weights (KEEP)
        self.w_l4_l23 = nn.Parameter(...)
        self.w_l23_l5 = nn.Parameter(...)
        self.w_l23_l6 = nn.Parameter(...)
        self.w_l5_l23 = nn.Parameter(...)  # Feedback
        self.w_l6_l4 = nn.Parameter(...)   # Feedback
```

#### Target Architecture
```python
class LayeredCortex(NeuralRegion):
    def __init__(self, config):
        super().__init__(
            n_neurons=config.n_l4 + config.n_l23 + config.n_l5 + config.n_l6,
            default_learning_rule="stdp",
            device=config.device,
        )

        # Sizes
        self.n_l4 = config.n_l4
        self.n_l23 = config.n_l23
        self.n_l5 = config.n_l5
        self.n_l6 = config.n_l6

        # Internal populations (unchanged)
        self.l4_neurons = ConductanceLIF(self.n_l4, ...)
        self.l23_neurons = ConductanceLIF(self.n_l23, ...)
        self.l5_neurons = ConductanceLIF(self.n_l5, ...)
        self.l6_neurons = ConductanceLIF(self.n_l6, ...)

        # Internal weights (unchanged)
        self.w_l4_l23 = nn.Parameter(...)
        self.w_l23_l5 = nn.Parameter(...)
        self.w_l23_l6 = nn.Parameter(...)
        self.w_l5_l23 = nn.Parameter(...)
        self.w_l6_l4 = nn.Parameter(...)

        # External weights via synaptic_weights:
        # Feedforward:
        # - add_input_source("thalamus", n_input=..., target_layer="l4")
        # Top-down:
        # - add_input_source("pfc", n_input=..., target_layer="l23")
        # Lateral:
        # - add_input_source("cortex_other", n_input=..., target_layer="l23")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Multi-port output: different layers project to different targets.

        Returns:
            Dict with keys: "l23" (to hippocampus), "l5" (to striatum), "l6" (to thalamus)
        """
        # L4: Thalamic input
        thalamic_input = self._apply_synapses("thalamus", inputs["thalamus"])
        l4_spikes = self.l4_neurons(thalamic_input)

        # L2/3: Cortical integration
        l23_input = torch.matmul(self.w_l4_l23, l4_spikes)
        if "pfc" in inputs:
            l23_input += self._apply_synapses("pfc", inputs["pfc"])
        l23_spikes = self.l23_neurons(l23_input)

        # L5: Motor output
        l5_input = torch.matmul(self.w_l23_l5, l23_spikes)
        l5_spikes = self.l5_neurons(l5_input)

        # L6: Feedback to thalamus
        l6_input = torch.matmul(self.w_l23_l6, l23_spikes)
        l6_spikes = self.l6_neurons(l6_input)

        return {
            "l23": l23_spikes,   # To hippocampus
            "l5": l5_spikes,     # To striatum
            "l6": l6_spikes,     # To thalamus/TRN
        }
```

**Challenge**: Multi-port output requires Dict return type
**Solution**: Update event system to handle Dict outputs
**Expected Changes**: ~100 lines modified, ~60 lines removed

---

### ✅ **5. Thalamus** (Complete - Unique Architecture)

**File**: `src/thalia/regions/thalamus.py`
**Lines**: ~976 (post-migration)
**Complexity**: Low architectural, high biological (relay/TRN, burst/tonic modes, spatial filtering)
**Tests**: 12/12 passing (100%)
**Commit**: [commit hash]

#### Migration Results
**Unique Discovery**: Thalamus has **NO external weight matrices**!

**Why Thalamus is Different**:
- **Function**: Sensory relay, not multi-source integrator
- **Input**: Raw sensory spikes directly from sensory organs (retina, cochlea, skin)
- **Architecture**: First processing stage - receives input, doesn't integrate other brain regions
- **Weights**: All internal (input_to_trn, relay_to_trn, trn_to_relay, trn_recurrent, relay_gain, center_surround_filter)
- **Biology**: Real thalamus has 1:1 relay connections, not learned synaptic weights

**Migration Pattern**:
```python
class ThalamicRelay(NeuralRegion):
    def __init__(self, config):
        # Call NeuralRegion with relay neuron count
        super().__init__(n_neurons=config.n_relay, device=config.device, dt_ms=config.dt_ms)
        
        # NO synaptic_weights initialization!
        # All weights are internal thalamic circuitry
        
    def forward(self, inputs: Union[Dict, Tensor]) -> Tensor:
        # Receives raw sensory spikes directly
        if isinstance(inputs, torch.Tensor):
            input_spikes = inputs
        else:
            input_spikes = inputs["input"]
        
        # Apply spatial filtering (internal circuit)
        filtered = torch.mv(self.center_surround_filter, input_spikes.float())
        # ... rest of relay logic ...
```

**Backward Compatibility Helpers Added**:
1. `set_neuromodulators()` - Lost mixin access
2. `spike_diagnostics()` - Compute firing rates in Hz
3. `membrane_diagnostics()` - Voltage statistics
4. `collect_standard_diagnostics()` - Weight and metric aggregation

**Key Insight**: NeuralRegion pattern is flexible enough to support regions that don't use synaptic_weights at all. Validates architecture scales from zero-external-source (Thalamus) to quad-source (Hippocampus).

**Actual Changes**: ~90 lines added (helpers), 0 lines removed (no external weights to delete)

---

### 🎯 **6. Cerebellum** (Complex - Granule Layer + Purkinje)

**File**: `src/thalia/regions/cerebellum/cerebellum_region.py`
**Lines**: ~1100
**Complexity**: High (granule expansion, Purkinje cells, DCN, error correction)

#### Migration Notes
- Granule layer expansion stays internal
- Parallel fiber→Purkinje weights stay internal
- External inputs: mossy fibers (from cortex, pontine nuclei)
- Climbing fiber error signal (from inferior olive)

**Expected Changes**: ~90 lines modified, ~60 lines removed

---

## Common Patterns

### Pattern 1: Single Input Source
```python
def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Get the single input
    source_name = list(inputs.keys())[0]
    input_spikes = inputs[source_name]

    # Apply synapses
    current = self._apply_synapses(source_name, input_spikes)

    # Process
    output = self.neurons(current)
    return output
```

### Pattern 2: Multiple Input Sources
```python
def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Accumulate currents from all sources
    total_current = torch.zeros(self.n_neurons, device=self.device)

    for source_name, input_spikes in inputs.items():
        if source_name in self.synaptic_weights:
            current = self._apply_synapses(source_name, input_spikes)
            total_current += current

    # Process
    output = self.neurons(total_current)
    return output
```

### Pattern 3: Layer-Specific Routing
```python
def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Route inputs to specific layers
    l4_current = torch.zeros(self.n_l4, device=self.device)
    l23_current = torch.zeros(self.n_l23, device=self.device)

    if "thalamus" in inputs:
        # Thalamus → L4
        l4_current += self._apply_synapses("thalamus", inputs["thalamus"])

    if "pfc" in inputs:
        # PFC → L2/3 (top-down)
        l23_current += self._apply_synapses("pfc", inputs["pfc"])

    # Internal processing...
```

### Pattern 4: Multi-Port Output
```python
def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # ... processing ...

    return {
        "l23": l23_spikes,
        "l5": l5_spikes,
        "l6": l6_spikes,
    }
```

---

## Testing Strategy

### Per-Region Test Updates

1. **Update test fixtures**:
```python
# OLD
def test_region_forward():
    region = Region(config)
    output = region.forward(input_spikes)

# NEW
def test_region_forward():
    region = Region(config)
    region.add_input_source("source1", n_input=128)
    output = region.forward({"source1": input_spikes})
```

2. **Test multi-source**:
```python
def test_multi_source():
    region = Region(config)
    region.add_input_source("cortex", n_input=128)
    region.add_input_source("hippocampus", n_input=64)

    output = region.forward({
        "cortex": cortex_spikes,
        "hippocampus": hippo_spikes,
    })

    assert output.shape == (region.n_neurons,)
```

3. **Test learning**:
```python
def test_learning():
    region = Region(config)
    region.add_input_source("input", n_input=100, learning_rule="stdp")

    # Initial weights
    w_before = region.synaptic_weights["input"].clone()

    # Run forward with correlated activity
    for _ in range(10):
        output = region.forward({"input": strong_input})

    # Weights should change
    w_after = region.synaptic_weights["input"]
    assert not torch.allclose(w_before, w_after)
```

### Integration Tests

After each region migration:
```bash
# Unit tests for that region
pytest tests/unit/regions/{region_name}/ -v

# Integration tests
pytest tests/integration/test_brain_with_{region_name}.py -v

# Full suite
pytest tests/unit/ -k "not (test_dcn_purkinje_inhibition or test_l23_to_l6_delay)" -q
```

---

## Rollback Strategy

Each region migration is independent. If a migration causes issues:

1. **Revert commit**: `git revert <commit_hash>`
2. **Keep old class**: Rename to `LegacyRegion`, keep for reference
3. **Feature flag**: Add region-specific flag if needed
4. **Document issues**: Add to migration notes

---

## Success Criteria

Phase 2 complete when:

- ✅ All 6 regions inherit from NeuralRegion (2/6 complete)
- ⏳ All tests pass (>95% pass rate) (Currently: ~80% overall, 100% striatum, 100% PFC)
- ⏳ No external weights in regions (all in `synaptic_weights`) (2/6 complete)
- ⏳ All regions use `forward(inputs: Dict)` signature (2/6 complete)
- ⏳ Learning works for all regions (2/6 validated)
- ⏳ Checkpoints load/save correctly (2/6 validated)
- ⏳ Documentation updated (In progress)

**Current Progress**: 33.3% complete (2/6 regions)

### Per-Region Metrics

| Region | Inherit NeuralRegion | Tests Pass | Weights Moved | Learning Works | Checkpoints Work |
|--------|---------------------|------------|---------------|----------------|------------------|
| **Striatum** | ✅ | ✅ (13/13) | ✅ | ✅ | ✅ |
| **PFC** | ✅ | ✅ (9/9) | ✅ | ✅ | ✅ |
| **Hippocampus** | ❌ | - | ❌ | - | - |
| **LayeredCortex** | ❌ | - | ❌ | - | - |
| **Thalamus** | ❌ | - | ❌ | - | - |
| **Cerebellum** | ❌ | - | ❌ | - | - |

---

## Timeline Estimate

| Week | Focus | Regions | Status | Deliverable |
|------|-------|---------|--------|-------------|
| 1 | Foundation | Striatum, PFC | ✅ Both complete | 2 regions migrated, pattern validated |
| 2 | Continue | Hippocampus, Cortex | 🔄 In progress | 4 regions migrated |
| 3 | Complex | Cortex, Thalamus | ⏳ Planned | 5 regions migrated |
| 4 | Final | Cerebellum | ⏳ Planned | All 6 regions migrated |
| 5 | Polish | Testing, docs | ⏳ Planned | Phase 2 complete, ready for Phase 3 |

**Total**: 4-5 weeks for complete migration (revised based on experience)
**Elapsed**: 1 day (Striatum + PFC both complete)

---

## Next Steps

1. ✅ **Striatum Complete** - Pattern validated, all tests passing
2. ✅ **PFC Complete** - Single-source pattern validated, checkpoints working
3. ✅ **Hippocampus Complete** - Complex multi-source (4 EC pathways), trisynaptic circuit preserved
4. 🔄 **Begin LayeredCortex Migration** - Complex laminar structure with port-based routing
5. ⏳ **Continue systematically** - Follow priority order for remaining 3 regions

**Immediate Priority**: Start LayeredCortex migration - most complex region, multiple input/output ports

---

## Questions & Decisions

### Q: What about regions with no external inputs?
**A**: Still inherit from NeuralRegion for consistency. Just don't call `add_input_source()`.

### Q: What if a region needs custom weight initialization per source?
**A**: Override `add_input_source()` or manually set weights after creation:
```python
region.add_input_source("source", n_input=128)
region.synaptic_weights["source"].data = custom_init(...)
```

### Q: How to handle optional inputs?
**A**: Use `.get()` with default:
```python
cortex_input = inputs.get("cortex", torch.zeros(n_input, device=self.device))
```

### Q: What about backward compatibility?
**A**: Keep `use_legacy_pathways=True` until Phase 4. Old and new can coexist.

### Q: How to handle opponent pathways (like D1/D2)?
**A**: Create separate weight entries in synaptic_weights dict:
```python
# In region __init__:
self.synaptic_weights["source_d1"] = torch.randn(n_neurons, n_input)
self.synaptic_weights["source_d2"] = torch.randn(n_neurons, n_input)

# In pathway (property-based access):
@property
def weights(self):
    return self._parent.synaptic_weights[self._weight_source]
```
Each pathway gets full-size weight matrix for independent learning rules.

### Q: What if forward() needs to support both Dict and Tensor during migration?
**A**: Use Union type and auto-wrap:
```python
def forward(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        inputs = {"default": inputs}  # Backward compat
    # ... process dict normally
```

---

## References

- **Phase 1 Commit**: `9ad5774` - NeuralRegion base class
- **Architecture Spec**: `docs/architecture/BIOLOGICAL_ARCHITECTURE_SPEC.md`
- **NeuralRegion Tests**: `tests/unit/core/test_neural_region.py`
- **Original Review**: `docs/architecture/PATHWAY_ARCHITECTURE_REVIEW.md` (deleted, see commit history)

---

**Document Status**: Living document - update as migrations proceed
**Last Updated**: 2025-12-19
**Next Review**: After first region migration (Striatum)
