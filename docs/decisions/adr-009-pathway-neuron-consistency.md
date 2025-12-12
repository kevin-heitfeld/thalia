# ADR-009: Pathway Neuron Consistency

**Date**: December 11, 2025  
**Status**: Implemented  
**Context**: Component parity between regions and pathways

## Decision

SpikingPathway now uses `ConductanceLIF` neurons instead of manual LIF implementation, achieving consistency with brain regions.

## Rationale

### Before (Manual Implementation)
```python
# SpikingPathway manually implemented LIF dynamics:
- Manual membrane potential updates
- Manual refractory period tracking  
- Manual spike generation
- Code duplication from neuron.py (~50 lines)
- _create_neurons() returned None
```

### After (ConductanceLIF Integration)
```python
# SpikingPathway delegates to ConductanceLIF:
- Uses same neuron model as regions (Striatum, Prefrontal, etc.)
- Conductance-based dynamics (more biologically accurate)
- Proper shunting inhibition
- _create_neurons() returns ConductanceLIF object
- ~50 lines of manual LIF code removed
```

## Component Parity Principle

From `copilot-instructions.md`:
> **Pathways are just as important as regions!**
> - Pathways are active learning components, not just "glue code"
> - They learn via STDP/BCM during forward passes
> - They need growth when connected regions grow

**Regions and pathways should use the same high-quality neuron models.**

## Benefits

### 1. Biological Accuracy
- **Conductance-based dynamics**: Current flow depends on voltage difference from reversal potentials
- **Natural saturation**: Membrane potential can't exceed reversal potentials
- **Shunting inhibition**: Divisive (multiplicative) rather than subtractive

### 2. Code Quality
- **DRY**: No duplicate LIF implementation
- **Maintainability**: Bug fixes to neuron models auto-apply to pathways
- **Consistency**: Same API for regions and pathways

### 3. Architectural Consistency
- **Unified interface**: Both regions and pathways use `_create_neurons()`
- **State management**: Both use neuron object for membrane/refractory state
- **Growth**: Pathways expand neurons same way as regions

## Implementation Details

### Key Changes

1. **Imports**:
   ```python
   from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
   ```

2. **Neuron Creation**:
   ```python
   def _create_neurons(self) -> ConductanceLIF:
       neuron_config = ConductanceLIFConfig(
           tau_mem=cfg.tau_mem_ms,
           v_rest=cfg.v_rest,
           v_reset=cfg.v_reset,
           v_threshold=cfg.v_thresh,
           tau_ref=cfg.refractory_ms,
           dt_ms=cfg.dt_ms,
           device=cfg.device,
       )
       return ConductanceLIF(n_neurons=cfg.target_size, config=neuron_config)
   ```

3. **Forward Pass Simplification**:
   ```python
   # Before: Manual membrane dynamics (~40 lines)
   mem_decay = np.exp(-dt / cfg.tau_mem_ms)
   self.membrane = cfg.v_rest + (self.membrane - cfg.v_rest) * mem_decay + ...
   target_spikes = ((self.membrane >= effective_thresh) & can_spike).float()
   # ... manual reset, refractory, etc.
   
   # After: Delegate to neuron object
   target_spikes, _ = self.neurons(self.synaptic_current)
   ```

4. **State Management**:
   ```python
   def reset_state(self) -> None:
       self.neurons.reset_state()  # Reset neuron state
       self.synaptic_current.zero_()
       self.pre_trace.zero_()
       # ...
   ```

5. **Growth Support**:
   ```python
   def add_neurons(self, n_new: int, ...):
       # Preserve old neuron state
       old_state = self.neurons.get_state()
       
       # Create larger neuron object
       self.neurons = ConductanceLIF(n_neurons=new_size, config=...)
       
       # Restore old state + initialize new neurons
       self.neurons.reset_state()
       self.neurons.membrane[:old_size] = old_state["membrane"]
       # ...
   ```

### Preserved Pathway-Specific Features

The refactoring maintains all pathway-specific functionality:

- ✅ **Axonal delays**: Delay buffer and delayed spike delivery
- ✅ **Short-term plasticity (STP)**: Modulates effective weights
- ✅ **STDP learning**: Pre/post traces, dopamine modulation
- ✅ **BCM metaplasticity**: Sliding threshold for homeostasis
- ✅ **Temporal coding**: Phase coding, latency coding, etc.
- ✅ **Synaptic scaling**: Homeostatic weight normalization

### Test Coverage

New test suite (`test_spiking_pathway_neurons.py`) with 8 tests:

1. ✅ `test_spiking_pathway_uses_conductance_lif` - Verify neuron type
2. ✅ `test_spiking_pathway_forward_with_neurons` - Forward pass works
3. ✅ `test_spiking_pathway_phase_coding` - Phase modulation preserved
4. ✅ `test_spiking_pathway_reset_resets_neurons` - State reset works
5. ✅ `test_spiking_pathway_add_neurons_expands_neuron_object` - Growth works
6. ✅ `test_spiking_pathway_state_includes_neuron_state` - Checkpointing works
7. ✅ `test_spiking_pathway_learning_with_neurons` - STDP still works
8. ✅ `test_spiking_pathway_diagnostics_use_neuron_state` - Diagnostics work

All existing tests continue to pass:
- `test_brain_from_config.py`: 9/10 (1 CUDA skip)
- `test_pathway_registration.py`: 8/8
- `test_component_registry.py`: 31/31

**Total: 56 passing tests** across component infrastructure.

## Consequences

### Positive
- **Biological realism**: Conductance-based dynamics
- **Code simplification**: ~50 lines removed
- **Maintainability**: Single source of truth for neuron dynamics
- **Architectural consistency**: Regions and pathways use same components

### Neutral
- **Performance**: ConductanceLIF has similar computational cost to manual LIF
- **API**: External API unchanged (forward signature, learning interface)

### Considerations
- **Migration**: Old checkpoints with manual membrane state won't load directly
  - Solution: Conversion utility if needed for legacy checkpoints
- **Customization**: Neuron parameters now set via ConductanceLIFConfig
  - Solution: SpikingPathwayConfig maps to ConductanceLIFConfig in `_create_neurons()`

## Related Decisions

- **ADR-001**: PyTorch tensors for simulation backend
- **ADR-004**: Boolean spikes for biological accuracy
- **ADR-007**: PyTorch consistency across all components
- **ADR-008**: Neural component consolidation (regions inherit from NeuralComponent)

## References

- **Component Parity Pattern**: `docs/patterns/component-parity.md`
- **ConductanceLIF Implementation**: `src/thalia/core/neuron.py`
- **SpikingPathway**: `src/thalia/integration/spiking_pathway.py`
- **Test Suite**: `tests/unit/test_spiking_pathway_neurons.py`
