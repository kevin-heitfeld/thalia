# ADR-010: Axonal Delays for All Neural Components

**Status**: Accepted
**Date**: 2025-12-11
**Authors**: Thalia Team
**Related ADRs**: ADR-008 (Component Consolidation)

## Context

Previously, only `SpikingPathway` (inter-region connections) implemented axonal conduction delays, while brain regions produced immediate output. This created an architectural asymmetry:

- **Pathways**: Had delay buffers with 1-10ms delays
- **Regions**: Produced immediate output (0ms delay)

This violated our Component Parity Principle (see `docs/patterns/component-parity.md`): "Pathways are just as important as regions - they should share all fundamental mechanisms."

### Biological Reality

ALL neural connections have conduction delays:
- **Within-region (local axons)**: 0.5-2ms
- **Inter-region (long-range)**: 1-10ms
- **Thalamo-cortical**: 8-15ms
- **Striato-cortical**: 10-20ms

Action potentials conduct at 0.5-120 m/s depending on myelination and axon diameter. Even the shortest intra-cortical connections have measurable delays.

### Architectural Issue

The regions-vs-pathways asymmetry meant:
1. Regions and pathways used **different code paths** for output
2. Pathways were treated as "second-class" components
3. **Violated biological realism**: Real neurons don't produce instantaneous output
4. Made curriculum learning fragile (pathways could become silent while regions stayed active)

## Decision

**Add axonal delay buffers to ALL neural components (regions and pathways).**

### Implementation

1. **Added to `NeuralComponentConfig`**:
   ```python
   axonal_delay_ms: float = 1.0
   """Axonal conduction delay in milliseconds."""
   ```

2. **Added to `NeuralComponent` base class**:
   ```python
   self.axonal_delay_ms = config.axonal_delay_ms
   self.avg_delay_steps = int(self.axonal_delay_ms / config.dt_ms)
   self.max_delay_steps = max(1, int(self.axonal_delay_ms * 2 / config.dt_ms) + 1)
   # delay_buffer initialized lazily on first forward()
   ```

3. **Added helper method**:
   ```python
   def _apply_axonal_delay(
       self,
       output_spikes: torch.Tensor,
       dt: float,
   ) -> torch.Tensor:
       """Apply axonal delay to output spikes using circular buffer."""
       if not hasattr(self, 'delay_buffer') or self.delay_buffer is None:
           self._initialize_delay_buffer(output_spikes.shape[0])

       self.delay_buffer[self.delay_buffer_idx] = output_spikes
       delayed_idx = (self.delay_buffer_idx - self.avg_delay_steps) % self.delay_buffer.shape[0]
       delayed_spikes = self.delay_buffer[delayed_idx]
       self.delay_buffer_idx = (self.delay_buffer_idx + 1) % self.delay_buffer.shape[0]
       return delayed_spikes
   ```

4. **Updated all region forward() methods**:
   ```python
   # Before:
   return output_spikes

   # After:
   delayed_spikes = self._apply_axonal_delay(output_spikes, dt)
   return delayed_spikes
   ```

### Lazy Initialization

The delay buffer is **not** initialized in `__init__` to avoid conflicts with subclasses that use `register_buffer()` (like `SpikingPathway`). Instead, it's initialized on first call to `_apply_axonal_delay()`.

This allows:
- Regions to lazily create buffers
- Pathways to use `register_buffer()` without conflicts
- Unified code path for both types

## Consequences

### Positive

1. ✅ **Complete Component Parity**: Regions and pathways are now architecturally identical
2. ✅ **Biological Realism**: All neural connections have delays (not optional)
3. ✅ **Unified Codebase**: Same `_apply_axonal_delay()` method for all components
4. ✅ **Configuration-Only Differences**: Regions use 1-2ms, pathways use 5-10ms (same code)
5. ✅ **Curriculum Learning**: Regions and pathways can grow/forget together without conflicts

### Breaking Changes

⚠️ **Output Timing**: All region outputs are now delayed by `axonal_delay_ms` (default 1ms)

This means:
- Spikes generated at t=10ms appear in output at t=11ms
- Tests expecting immediate output will see zeros initially
- Multi-region networks have realistic propagation delays

**Migration**: Tests and experiments may need to run longer warmup periods to fill delay buffers.

### Performance

- **Memory**: +O(delay_steps × n_output) per component (~3-20 timesteps typical)
- **Compute**: Negligible (circular buffer index arithmetic)
- **Example**: 100-neuron region with 2ms delay @ 1ms timestep = 200 bools = 200 bytes

### Default Values

```python
# Within-region connections (local)
axonal_delay_ms: float = 1.0  # Regions default

# Inter-region connections (long-range)
axonal_delay_ms: float = 5.0  # Pathways typical

# Can be customized per region/pathway:
StriatumConfig(axonal_delay_ms=2.0)  # Striato-cortical delay
```

## Alternatives Considered

### 1. Keep regions immediate, only pathways have delays

**Rejected**: Violates biological realism and component parity principle.

### 2. Add delays but make them optional (delay_ms=0 means skip)

**Rejected**: Optional mechanisms lead to code paths that aren't tested. ALL neural connections have delays - this is not optional in biology.

### 3. Separate `DelayedRegion` and `ImmediateRegion` classes

**Rejected**: Creates artificial class hierarchy explosion. Better to have unified base with configuration differences.

## Verification

See `tests/unit/test_region_axonal_delays.py`:
- ✅ All regions (Striatum, Prefrontal, Hippocampus, Cerebellum, Cortex) have delays
- ✅ Delay buffer initialized lazily on first forward()
- ✅ Buffer resets correctly with `reset_state()`
- ✅ Regions and pathways share same `_apply_axonal_delay()` method
- ✅ Component parity verified: both use `NeuralComponent` base

## Related Documentation

- **Component Parity Principle**: `docs/patterns/component-parity.md`
- **ADR-008**: Neural Component Consolidation
- **ADR-004**: Bool Spikes (delay buffer uses `torch.bool`)
- **Copilot Instructions**: `.github/copilot-instructions.md` (updated)

## References

- Swadlow, H. A. (2000). "Information flow along neocortical axons." *Neuron*, 28(3), 647-650.
- Izhikevich, E. M., & Edelman, G. M. (2008). "Large-scale model of mammalian thalamocortical systems." *PNAS*, 105(9), 3593-3598.
- London, M., & Segev, I. (2001). "Synaptic scaling in vitro and in vivo." *Nature Neuroscience*, 4(9), 853-854.

---

**Implementation Date**: December 11, 2025
**Tests Added**: 10 tests in `test_region_axonal_delays.py`
**Lines Changed**: ~200 (base class + 5 regions)
**Breaking**: Yes (output timing changed)
**Backward Compatible**: No (all outputs now delayed)
