# ADR-003: Clock-Driven Simulation

**Date**: 2025-11-28
**Updated**: 2025-12-21
**Status**: Accepted (Supersedes all event-driven approaches)

### Context
We must choose the simulation scheduling model: event-driven vs clock-driven (fixed timestep). Biological plausibility and implementation complexity trade off here.

Initially, we implemented both event-driven and clock-driven modes. However, analysis revealed that event-driven execution is fundamentally incompatible with biological neural dynamics.

### Decision
Use **pure clock-driven fixed-timestep simulation** as the ONLY execution model. Event-driven architecture has been completely removed (December 2025).

### Rationale

**Clock-Driven is Biologically Accurate:**
- Real neurons are continuous analog systems (RC circuits), not discrete event processors
- Membrane potential decays continuously (tau_mem ≈ 20ms)
- Conductances decay continuously (tau_E ≈ 5ms, tau_I ≈ 10ms)
- Recurrent connections require every-timestep execution (CA3, TRN, cortical columns)
- Oscillators must tick every timestep (theta, gamma, alpha)
- Short-term plasticity evolves continuously

**Implementation Benefits:**
- Simpler architecture (no event scheduling overhead)
- Matches timestep-based design in all major neuroscience simulators (Brian, NEST, NEURON)
- Deterministic execution aids reproducibility and testing
- O(1) delay buffers instead of O(log n) priority queues
- GPU-friendly (pure tensor operations)

**Why Event-Driven Was Wrong:**
- Event-driven is only efficient for extremely sparse networks (<0.1% activity)
- Our networks are dense (1-30% spike rates in hippocampus, thalamus, striatum)
- Recurrent dynamics prevent "sleeping" between events
- Continuous processes cannot be "computed analytically" between events
- Violates biological reality (neurons don't stop processing when input stops)

### Implementation Details

**Execution Model:**
- Fixed 1ms timestep (dt_ms = 1.0)
- All regions execute every timestep
- Alphabetical execution order (handles circular dependencies gracefully)
- No event queue, no adapters, no priority scheduling

**Axonal Delays:**
- Implemented via `CircularDelayBuffer` in pathways
- O(1) read/write operations (ring buffer)
- Each pathway maintains delay buffers for all sources
- Deterministic delays (biologically accurate)
- GPU-compatible (pure tensor operations)

**Code Location:**
- Delay buffer: `src/thalia/utils/delay_buffer.py`
- Pathway integration: `src/thalia/pathways/axonal_projection.py`
- Brain execution: `src/thalia/core/dynamic_brain.py` (forward method)

### Consequences

**Positive:**
- ✅ Biologically accurate (continuous neural dynamics)
- ✅ Simpler codebase (~3200 lines removed)
- ✅ Better performance (O(1) delays vs O(log n) event queue)
- ✅ GPU-friendly (no CPU-side priority queue)
- ✅ Easier to checkpoint/restore (just tensor buffers)
- ✅ Compatible with parallel execution (each worker runs clock-driven)

**Trade-offs:**
- Executes all regions every timestep (not "sparse" like event-driven)
- However: Our networks are dense, so this is actually more efficient
- Memory: O(max_delay × spike_vector_size) per pathway (negligible at 1-20ms delays)

### References
- Event scheduler analysis: `temp/EVENT_SCHEDULER_ANALYSIS.md`
- Delay buffer implementation: `src/thalia/utils/delay_buffer.py`
- Delay buffer tests: `tests/unit/test_delay_buffer.py`
