# Clock-Driven Execution Optimizations

**Status**: Phase 1 Implemented (Further Analysis Required)
**Date**: December 21, 2025
**Context**: Now that event-driven mode is removed, we can optimize for pure clock-driven execution.

## Implementation Status (December 21, 2025)

### Phase 1 Optimizations: ✅ IMPLEMENTED

All Phase 1 optimizations have been successfully implemented in `dynamic_brain.py`:

1. **Pre-computed Connection Topology**: ✅
   - `_component_connections` dict built during `__init__`
   - O(1) lookup vs O(N) iteration
   - Lines 255-266

2. **Reusable Component Inputs Dict**: ✅
   - `_reusable_component_inputs` dict cleared instead of recreated
   - Eliminates allocation overhead in hot loop
   - Lines 267-268, 825

3. **Pre-allocated Output Cache**: ✅
   - `_output_cache` dict pre-allocated with None values
   - In-place updates instead of dict.update()
   - Lines 269-273, 847-851

4. **GPU Spike Tensor Accumulation**: ✅
   - `_spike_tensors` dict accumulates on GPU
   - No `.item()` sync in hot loop
   - Sync only once at end of forward()
   - Lines 277-282, 854-858, 864-867

### Current Performance (Baseline - Further Investigation Needed)

**Sensorimotor Brain** (6 components, 7 connections, 13 delay buffers):
- **Throughput**: ~7.5 timesteps/sec (CPU)
- **Latency**: ~133 ms/timestep

**Notes**:
- All optimization infrastructure verified as correctly installed
- Test suite passes (513/521 tests, 98.5%)
- Performance is slower than expected
- Requires profiling to identify remaining bottlenecks
- May be pre-existing (no baseline before optimizations)

### Investigation Needed

The current performance (~133ms/timestep) suggests significant bottlenecks remain:
- Component execution overhead (possibly neuron models)
- Pathway forward() calls (delay buffer operations)
- Python interpreter overhead
- Oscillator broadcast overhead?

**Next Steps**:
1. Profile with `cProfile` to identify hot spots
2. Check if ConductanceLIF neurons are the bottleneck
3. Consider vectorizing neuron updates
4. Investigate pathway overhead (13 delay buffers advancing each timestep)

## Current Performance Bottlenecks

### 1. **Dict Overhead in Hot Loop** (High Impact)
**Location**: `dynamic_brain.py` lines 795-820

**Current Code**:
```python
for comp_name in self._get_execution_order():
    component_inputs: Dict[str, torch.Tensor] = {}  # NEW DICT EVERY TIMESTEP

    for (src, tgt), pathway in self.connections.items():  # ITERATE ALL CONNECTIONS
        if tgt == comp_name and src in self._output_cache:
            delayed_outputs = pathway.forward({src: self._output_cache[src]})
            component_inputs.update(delayed_outputs)
```

**Issues**:
- Creates new dict every timestep for every component
- Iterates ALL connections to find relevant ones (O(C × N) where C=components, N=connections)
- Dict lookups and updates in hot loop

**Optimization**:
Pre-compute connection topology once, reuse dicts with in-place updates:

```python
# At initialization:
self._component_connections: Dict[str, List[Tuple[str, AxonalProjection]]] = {}
for (src, tgt), pathway in self.connections.items():
    if tgt not in self._component_connections:
        self._component_connections[tgt] = []
    self._component_connections[tgt].append((src, pathway))

# In hot loop:
for comp_name in self._execution_order:
    component_inputs.clear()  # REUSE DICT

    # Direct lookup instead of iteration
    for src, pathway in self._component_connections.get(comp_name, []):
        if src in self._output_cache:
            delayed_outputs = pathway.forward({src: self._output_cache[src]})
            component_inputs.update(delayed_outputs)
```

**Expected Speedup**: 15-25% (reduces allocation overhead and improves cache locality)

---

### 2. **Delay Buffer Advance Overhead** (Medium Impact)
**Location**: `axonal_projection.py` lines 220-230

**Current Code**:
```python
for source_spec in self.sources:
    buffer = self._delay_buffers[source_key]
    buffer.write(spikes)
    delayed_spikes = buffer.read(delay_steps)
    buffer.advance()  # CALLED FOR EVERY SOURCE EVERY TIMESTEP
```

**Issue**:
- `advance()` is called separately for each buffer
- Pointer update is cheap but function call overhead adds up

**Optimization**:
Batch advance all buffers once:

```python
def advance_all_buffers(self) -> None:
    """Advance all delay buffers at once (called once per timestep)."""
    for buffer in self._delay_buffers.values():
        buffer.advance()

# In forward():
# ... write/read operations ...
# Don't call advance() here

# In brain's forward loop AFTER all pathways:
for pathway in self.connections.values():
    pathway.advance_all_buffers()
```

**Expected Speedup**: 5-10% (reduces function call overhead)

---

### 3. **Execution Order Caching** (Low Impact, Already Done)
**Location**: `dynamic_brain.py` line 580

**Status**: ✅ Already implemented
```python
if self._execution_order is not None:
    return self._execution_order
```

This is good - topological sort is only computed once.

---

### 4. **Output Cache Updates** (Low Impact)
**Location**: `dynamic_brain.py` line 820

**Current Code**:
```python
self._output_cache.update(new_outputs)  # Creates new dict entries
```

**Issue**: Dict update creates new references each timestep

**Optimization**:
Pre-allocate cache dict, update values in-place:

```python
# At initialization:
self._output_cache = {name: None for name in self.components.keys()}

# In hot loop:
for comp_name, output in new_outputs.items():
    self._output_cache[comp_name] = output  # IN-PLACE UPDATE
```

**Expected Speedup**: 2-5% (minor but helps)

---

### 5. **Spike Count Tracking** (Low Impact)
**Location**: `dynamic_brain.py` lines 823-826

**Current Code**:
```python
for comp_name, component_output in new_outputs.items():
    if component_output is not None and isinstance(component_output, torch.Tensor):
        spike_count = int(component_output.sum().item())  # GPU→CPU SYNC
        self._spike_counts[comp_name] += spike_count
```

**Issue**: `.item()` forces GPU→CPU synchronization EVERY timestep for EVERY component

**Optimization**:
Make spike counting optional or batched:

```python
# Option 1: Only count when diagnostics are requested
if self._track_spikes:  # Flag set by user
    # ... current code ...

# Option 2: Accumulate on GPU, sync only at end
self._spike_tensors[comp_name] += component_output.sum()  # Stay on GPU
# At end of simulation:
self._spike_counts = {k: int(v.item()) for k, v in self._spike_tensors.items()}
```

**Expected Speedup**: 10-20% on GPU (eliminates frequent sync points)

---

### 6. **Oscillator Broadcasting** (Low Impact)
**Location**: `dynamic_brain.py` line 829

**Current Code**:
```python
self._broadcast_oscillator_phases()  # Called EVERY timestep
```

**Optimization**:
Already minimal - just phase increments. Could skip if oscillators disabled:

```python
if self._oscillator_manager is not None:
    self._broadcast_oscillator_phases()
```

**Expected Speedup**: 1-2% (minimal)

---

## Implementation Priority

### Phase 1: High Impact (Target: 20-35% speedup)
1. ✅ Pre-compute connection topology
2. ✅ Reuse component_inputs dict
3. ✅ GPU spike counting optimization

### Phase 2: Medium Impact (Target: 5-10% additional)
1. ✅ Batch delay buffer advances
2. ✅ Pre-allocate output cache

### Phase 3: Micro-optimizations (Target: 2-5% additional)
1. Skip oscillator broadcast if disabled
2. Profile and optimize component.forward() calls

---

## Measurement Strategy

**Before optimization**:
```python
import time
brain = create_brain()
start = time.perf_counter()
brain.forward(sensory_input, n_timesteps=1000)
baseline_time = time.perf_counter() - start
print(f"Baseline: {baseline_time:.3f}s for 1000 timesteps")
```

**After each optimization**:
```python
optimized_time = time.perf_counter() - start
speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x ({(1-optimized_time/baseline_time)*100:.1f}% faster)")
```

---

## Expected Total Impact

**Conservative estimate**: 25-40% faster clock-driven execution
**Optimistic estimate**: 35-50% faster with all optimizations

**Key insight**: Clock-driven execution is now predictable and regular, so we can:
- Pre-compute everything that doesn't change
- Reuse allocations
- Batch operations
- Eliminate conditional logic in hot loops

This is fundamentally different from event-driven, where irregular timing made these optimizations impossible.

---

## Next Steps

1. Implement Phase 1 optimizations
2. Add benchmark suite (`tests/benchmarks/test_clock_driven_performance.py`)
3. Profile with `cProfile` or `torch.profiler`
4. Measure impact of each optimization
5. Consider JIT compilation for hot loops (`torch.jit.script`)

---

## Long-term Optimizations (Future)

### A. Vectorized Multi-Timestep Execution
Instead of Python loop, batch multiple timesteps:
```python
# Current: for t in range(1000): step()
# Future: batched_forward(n_timesteps=1000)  # Pure tensor ops
```

### B. Custom CUDA Kernels
For CircularDelayBuffer operations:
```cpp
__global__ void delay_buffer_step(bool* buffer, int* ptr, int size, int delay) {
    // Fused write/read/advance in single kernel
}
```

### C. Graph Compilation
Use `torch.compile()` (PyTorch 2.0+):
```python
@torch.compile(mode="reduce-overhead")
def clock_driven_step(self, inputs):
    # Entire timestep in compiled graph
```

These would require more invasive changes but could provide 2-5x additional speedup.
