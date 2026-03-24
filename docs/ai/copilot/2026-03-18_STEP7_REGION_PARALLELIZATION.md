# Step 7: Region Parallelization — Detailed Implementation Plan

## Status: Implemented but OFF by default (GIL makes threading counterproductive)

## Implementation Result

**Benchmark** (30 steps, default brain, after Steps 1-6):
```
Sequential:      3.156s  (9.5 steps/s)
Parallel (w=4):  4.117s  (7.3 steps/s)
Speedup: 0.77x  ← SLOWER
```

**Root cause**: Phase 1 work is ~60-70% GIL-held Python (tiny tensor dispatch,
dict lookups, attribute access, conditionals). Only ~30-40% is GIL-free C++
kernel calls + BLAS matmuls. 4 threads fighting over the GIL plus task
submission overhead (660 submits/30 steps) outweighs the GIL-free overlap.

**What was done** (kept for future use):
- ✅ GIL release in all 4 C++ kernels (`py::call_guard<py::gil_scoped_release>()`)
- ✅ `ThreadPoolExecutor` infrastructure in `Brain.forward()` behind `GlobalConfig.PARALLEL_REGIONS`
- ✅ Precomputed field setup decoupled from region loop
- ✅ `PARALLEL_REGIONS = False` by default

**When to re-enable**: After fusing cortical inhibitory networks into C++
(~0.8s of `torch.mv` + ~0.24s of `_integrate_synaptic_inputs_at_dendrites`
moved from Python dispatch to GIL-free C++), the GIL-free fraction rises
to ~60-70% and threading becomes beneficial.

## 1. Blocker Analysis

### 1.1 GIL is Held by All C++ Kernels — **Primary Blocker (Solvable)**

All four C++ kernels use standard pybind11 bindings **without** `py::call_guard<py::gil_scoped_release>()`:

| Kernel | File | GIL Released? |
|---|---|---|
| `conductance_lif_step_cpp` | `src/thalia/utils/conductance_lif_kernel.cpp:385` | **No** |
| `two_compartment_lif_step_cpp` | `src/thalia/utils/two_compartment_lif_kernel.cpp:411` | **No** |
| `stp_step_cpp` | `src/thalia/utils/stp_kernel.cpp:69` | **No** |
| `philox_*_cpp` | `src/thalia/utils/philox_cpu_kernel.cpp:130` | **No** |

Since all C++ kernels hold the GIL, Python-level `ThreadPoolExecutor` parallelism would serialize completely — threads would queue on the GIL during kernel calls. **This must be fixed before any threading approach can work.**

**Solution**: Add `py::call_guard<py::gil_scoped_release>()` to all four PYBIND11_MODULE bindings. This is safe because:
- All kernels operate only on `at::Tensor` data pointers (no Python object access)
- `at::parallel_for` is GIL-independent (it uses native OS threads via the ATen thread pool)
- No kernel calls Python callbacks

### 1.2 `at::parallel_for` Thread Pool Contention — **Minor Risk**

All kernels share the global ATen thread pool (default = physical CPU cores). If multiple regions invoke TwoCompartmentLIF C++ kernels concurrently from different Python threads, their `at::parallel_for` calls would compete for the same pool. However:
- TwoCompartmentLIF populations are small (200-750 neurons each), grain size 128 → 2-6 work chunks per call
- The pool handles contention via work-stealing; small work items complete quickly
- The ConductanceLIFBatch kernel (Phase 2) is the heavy one, and it runs **after** Phase 1 — no contention

**Verdict**: Acceptable. Not a blocker.

### 1.3 Shared Mutable State During Phase 1 — **No Blocker**

Each region writes only to:
1. **Its own neuron state** (V, g_E, g_I, etc.) — region-local
2. **Disjoint slices of `ConductanceLIFBatch` input buffers** — `batch.g_ampa_input[start:end]` where `[start:end]` is unique per population. No overlap.
3. **Its own `RegionOutput` dict** — stored under unique `brain_output[region_name]` keys
4. **Region-local neuromodulator concentrations** — not shared

Reads are from:
- `region_inputs[region_name]` — pre-materialized, per-region dict
- `neuromodulator_signals` — shared read-only
- `_last_brain_output[region_name]` — previous timestep, immutable during this step

**No data races between regions.** All writes target disjoint memory ranges.

### 1.4 Pre-Phase 1 Dependencies — **Already Satisfied**

The following must complete before Phase 1:
- `STPBatch.step()` → single call, already centralized
- `GlobalSparseMatrix.integrate()` + `scatter_to_neuron_batch()` → centralized
- `NeuromodulatorHub.build()` → read-only from T-1

These all run sequentially before the region loop. No change needed.

### 1.5 Post-Phase 1 Dependencies — **Already Satisfied**

- `_neuron_batch.step()` (Phase 2) needs all input buffers filled → just barrier after Phase 1 threads join
- Learning (Phase 3) needs correct spikes → runs after Phase 2
- Axonal tract writes (Phase 4) need `brain_output` populated → runs after Phase 3

### 1.6 Subclass Neurons (VTA, LC, NB, DRN) — **No Blocker**

Subclass neurons (`SerotoninNeuron`, `NorepinephrineNeuron`, `AcetylcholineNeuron`) fire in-place during Phase 1 (not batched). Each is region-local, so parallel execution is safe. Their `forward()` calls the C++ ConductanceLIF kernel on small tensors — with GIL release, these overlap.

### 1.7 `_precomputed_stp_efficacy` / `_precomputed_sparse_conductances` Pattern — **Minor Refactor**

Currently set/cleared per-region in the sequential loop:
```python
region._precomputed_stp_efficacy = stp_efficacy  # shared read-only dict
region._precomputed_sparse_conductances = self._sparse_matrix.get_region_conductances(region_name)
```

For parallel execution, these must be set **before** launching threads (during Phase 0), not inside the thread. Simple refactor: set all regions' precomputed fields in a loop, then launch parallel execution.

## 2. Current Performance Baseline (After Steps 1-6)

From `data/profiling/2026-03-18T200557_after_step_6.txt`:

```
Total wall time: 4.937s (10.13 steps/s)   [was 7.455s / 6.71 steps/s at baseline]
Time accounted in regions: 2.949s

Top regions by time:
  cortex_sensory       613.7ms  (12.4%)
  cortex_association   498.8ms  (10.1%)
  prefrontal_cortex    420.7ms   (8.5%)
  hippocampus          309.7ms   (6.3%)
  locus_coeruleus      252.4ms   (5.1%)
  cerebellum           244.7ms   (5.0%)
  BLA                  179.9ms   (3.6%)
  striatum             108.8ms   (2.2%)
  -- rest < 70ms each --
```

Top remaining hotspots by self-time:
```
torch.mv              0.809s   (intra-region dense matmuls, mostly cortical inhibitory networks)
ConductanceLIF C++    0.433s   (Phase 2 batched kernel — NOT in Phase 1)
mul_                  0.272s   (scalar/small tensor ops across all regions)
_integrate_synaptic.. 0.244s   (remaining small dense matmuls)
_integrate_single..   0.194s   (per-connection matmul dispatch)
__getattr__           0.167s   (nn.Module attribute lookups)
STPBatch.step         0.167s   (Phase 0 — not parallelizable)
.float()              0.161s   (type conversions)
scatter_to_neuron..   0.159s   (Phase 0 — not parallelizable)
_apply_synaptic_sc..  0.129s   (homeostatic weight scaling)
```

### What's Parallelizable in Phase 1?

The **2.949s accounted in regions** is the Phase 1 + Phase 3 work. Phase 2 (ConductanceLIF batch, 0.433s) runs separately. Phase 0 (STP + sparse, ~0.33s) runs separately.

Within Phase 1, the region loop itself is ~2.6s of sequential Python execution. With perfect parallel speedup across ~8 cores, the theoretical limit is ~0.33s — saving ~2.3s. Realistically, Amdahl's law + threading overhead + thread pool contention limits this to **30-50% reduction of Phase 1 time** ≈ saving **0.8-1.3s**.

### Critical Path: 3 Cortical Columns

The three cortical columns (sensory, association, prefrontal) dominate Phase 1 at 1.53s combined. The single longest region (cortex_sensory at 613ms) sets the **minimum possible Phase 1 time** under perfect parallelization. This means:
- Sequential Phase 1: ~2.6s
- Parallel Phase 1 (limited by longest region): ~0.65s + threading overhead
- **Maximum Phase 1 speedup: ~75%** (saving ~1.95s)
- **Total walltime impact: ~35-40%** (from ~4.9s to ~3.1-3.3s → **~15-16 steps/s**)

## 3. Recommended Architecture

### Option A: Python ThreadPoolExecutor with GIL Release ✅ Recommended

**Approach**: Release the GIL in all C++ kernels, then use `concurrent.futures.ThreadPoolExecutor` to run regions in parallel during Phase 1.

**Pros**:
- Minimal refactoring — the region loop structure stays the same
- Each thread calls `region.forward()` independently
- Thread pool handles load balancing automatically
- GIL release is a one-line change per kernel
- Standard Python pattern — debuggable, well-understood

**Cons**:
- Python overhead per thread (~10-50μs) for GIL acquire/release at pure-Python sections
- Pure-Python work within `_step()` (neuromod updates, dict lookups, conditional logic) serializes on the GIL
- `at::parallel_for` contention between concurrent kernel calls

### Option B: Eliminate Python Region Loop Entirely ❌ Not Ready

The optimization document suggests restructuring so Brain.forward() becomes `gather inputs → one C++ call → scatter outputs`. This effectively moves the entire region loop into C++.

**Why not now**: The remaining Phase 1 work is highly heterogeneous:
- 22 different `_step()` implementations with region-specific logic (SWR state machines, RPE computation, gap junctions, ACh gating, etc.)
- TwoCompartmentLIF neurons fire during Phase 1 (9 populations, ~6000 neurons)
- Subclass neurons fire during Phase 1 (~6 populations)
- Intra-region matmuls with complex conditional logic
- This would require porting thousands of lines of region-specific Python to C++

**Verdict**: Far too much work for the ~3% additional benefit over Option A. Save this for a future GPU migration.

### Option C: Multiprocessing ❌ Rejected

Inter-process tensor serialization overhead exceeds the parallelization benefit for regions this small. Each region would need to serialize/deserialize its state tensors (~50-500 tensors) every timestep.

## 4. Detailed Implementation Plan — Option A

### Phase A: GIL Release in C++ Kernels (Prerequisite)

**Files to modify** (4 files, 1 line each):

1. **`src/thalia/utils/conductance_lif_kernel.cpp`** (line ~385):
   ```cpp
   // Before:
   m.def("conductance_lif_step_cpp", &conductance_lif_step_cpp, ...);

   // After:
   m.def("conductance_lif_step_cpp", &conductance_lif_step_cpp, ...,
         py::call_guard<py::gil_scoped_release>());
   ```

2. **`src/thalia/utils/two_compartment_lif_kernel.cpp`** (line ~411): Same pattern

3. **`src/thalia/utils/stp_kernel.cpp`** (line ~69): Same pattern

4. **`src/thalia/utils/philox_cpu_kernel.cpp`** (line ~130): Same pattern for all exposed functions

**Safety check**: Verify no kernel accesses Python objects (PyObject*, Python dicts, etc.). All four kernels work exclusively with `at::Tensor` data pointers → safe.

**Validation**: Run existing test suite (`test_conductance_lif_kernel.py`, `test_two_compartment_lif_kernel.py`, `test_stp_kernel.py`) — behavior must be identical.

**Build cache**: After modifying `.cpp` files, the JIT build cache (`~/.cache/torch_extensions/` or equivalent) must be invalidated. Delete the cache or touch the source files.

### Phase B: Prepare Region Pre-computation (Decouple Phase 0 → Phase 1 Setup)

Currently, `_precomputed_stp_efficacy` and `_precomputed_sparse_conductances` are set per-region inside the loop. For thread safety, set them **before** launching threads:

**File**: `src/thalia/brain/brain.py`, Phase 1 section (~line 382)

```python
# Before (sequential):
for region_name, region in self.regions.items():
    region._precomputed_stp_efficacy = stp_efficacy
    region._precomputed_sparse_conductances = self._sparse_matrix.get_region_conductances(region_name)
    region_output = region.forward(...)
    region._precomputed_stp_efficacy = None
    region._precomputed_sparse_conductances = None
    brain_output[region_name] = region_output

# After (parallel-ready):
# Phase 0.5: Set precomputed fields for ALL regions before launching threads
for region_name, region in self.regions.items():
    region._precomputed_stp_efficacy = stp_efficacy
    region._precomputed_sparse_conductances = self._sparse_matrix.get_region_conductances(region_name)

# Phase 1: Run all regions in parallel
# ... (see Phase C below)

# Phase 1.5: Clear precomputed fields
for region_name, region in self.regions.items():
    region._precomputed_stp_efficacy = None
    region._precomputed_sparse_conductances = None
```

### Phase C: Parallel Region Execution

**File**: `src/thalia/brain/brain.py`

```python
import concurrent.futures
from typing import Dict, Tuple

# At class level or module level:
_REGION_THREAD_POOL: Optional[concurrent.futures.ThreadPoolExecutor] = None

def _get_region_thread_pool(max_workers: int) -> concurrent.futures.ThreadPoolExecutor:
    """Lazy-init a reusable thread pool for region parallelization."""
    global _REGION_THREAD_POOL
    if _REGION_THREAD_POOL is None or _REGION_THREAD_POOL._max_workers != max_workers:
        if _REGION_THREAD_POOL is not None:
            _REGION_THREAD_POOL.shutdown(wait=False)
        _REGION_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    return _REGION_THREAD_POOL
```

**Phase 1 replacement**:

```python
def _execute_region(
    region_name: RegionName,
    region: NeuralRegion,
    synaptic_inputs_for_region: SynapticInput,
    neuromodulator_signals: NeuromodulatorInput,
    prev_region_output: Optional[RegionOutput],
) -> Tuple[RegionName, RegionOutput]:
    """Execute a single region's forward pass (thread-safe)."""
    region_output = region.forward(
        synaptic_inputs_for_region,
        neuromodulator_signals,
        prev_region_output,
    )
    return region_name, region_output

# In Brain.forward():
brain_output: BrainOutput = {}
region_synaptic_cache: Dict[RegionName, SynapticInput] = {}
pool = _get_region_thread_pool(max_workers=min(8, len(self.regions)))

futures = []
for region_name, region in self.regions.items():
    synaptic_inputs_for_region = region_inputs.get(region_name, {})
    region_synaptic_cache[region_name] = synaptic_inputs_for_region
    prev_region_output = (
        self._last_brain_output[region_name]
        if self._last_brain_output is not None else None
    )
    futures.append(
        pool.submit(
            _execute_region,
            region_name,
            region,
            synaptic_inputs_for_region,
            neuromodulator_signals,
            prev_region_output,
        )
    )

for future in concurrent.futures.as_completed(futures):
    region_name, region_output = future.result()
    brain_output[region_name] = region_output
```

### Phase D: Configurable Parallelization Toggle

Add a toggle to `GlobalConfig` for easy A/B testing and fallback:

```python
# In src/thalia/global_config.py:
PARALLEL_REGIONS: bool = True
PARALLEL_REGIONS_MAX_WORKERS: int = 8
```

Wrap Phase C in a conditional:
```python
if GlobalConfig.PARALLEL_REGIONS and len(self.regions) > 1:
    # Parallel path (Phase C)
    ...
else:
    # Sequential path (original loop)
    ...
```

### Phase E: ATen Thread Pool Tuning

When Phase 1 runs multiple regions concurrently via Python threads, each region's TwoCompartmentLIF or subclass neuron `forward()` call invokes `at::parallel_for` on the shared ATen pool. To prevent over-subscription:

```python
# In Brain.__init__ or at build time:
import torch
# Reserve some cores for Python-level parallelism
n_physical_cores = torch.get_num_threads()
# With N Python threads, each at::parallel_for should use fewer ATen threads
# to avoid N×M total threads competing for N cores
at_threads = max(1, n_physical_cores // min(8, len(self.regions)))
torch.set_num_threads(at_threads)
```

**Alternative**: Leave ATen threads at default and let the OS scheduler handle it. `at::parallel_for` tasks are short enough that contention is minimal. **Benchmark both approaches.**

### Phase F: Validation & Testing

1. **Numerical equivalence test**: Run 100 timesteps with `PARALLEL_REGIONS=False` and `PARALLEL_REGIONS=True`, compare all `brain_output` tensors. They must be **bitwise identical** (same computation, same RNG state, just different execution order).

   ⚠️ **RNG concern**: If Philox RNG uses a global counter, parallel execution could produce different random sequences depending on region execution order. Verify that RNG state is per-neuron (keyed by neuron index + timestep), not per-call-order. The existing Philox kernel uses `(neuron_seed, timestep)` → **order-independent**. ✅

2. **Thread-safety stress test**: Run 1000 timesteps with thread-sanitizer or manual validation, checking for:
   - Corrupted tensor values (NaN, unexpected values)
   - Inconsistent spike outputs
   - Memory errors

3. **Performance benchmark**: Run profiler with both `PARALLEL_REGIONS=True` and `False`:
   - Measure Phase 1 wall time
   - Measure total `forward()` wall time
   - Check for `at::parallel_for` contention (via ATen profiler or system thread stats)

## 5. Expected Results

| Metric | Before (Step 6) | After (Step 7) | Change |
|---|---|---|---|
| Total wall time (50 steps) | 4.94s | ~3.2-3.6s | -27-35% |
| Steps per second | 10.1 | ~14-16 | +39-58% |
| Phase 1 time | ~2.6s | ~0.7-1.0s | -62-73% |
| Phase 0 time (STP + sparse) | ~0.5s | ~0.5s (unchanged) | 0% |
| Phase 2 time (batched kernel) | ~0.45s | ~0.45s (unchanged) | 0% |

**Limiting factor**: cortex_sensory at 614ms is the single longest region and sets the floor for Phase 1 under parallel execution.

## 6. Implementation Order & Effort

| Sub-step | Description | Effort | Risk |
|---|---|---|---|
| **A** | GIL release in 4 C++ kernels | 30 min | Low — mechanical change, validated by existing tests |
| **B** | Decouple precomputed field setup | 15 min | Trivial refactor |
| **C** | ThreadPoolExecutor in Brain.forward | 45 min | Medium — thread-safety reasoning required |
| **D** | GlobalConfig toggle | 10 min | Trivial |
| **E** | ATen thread tuning | 30 min | Low — benchmark-driven |
| **F** | Validation & benchmarking | 60 min | Required for correctness confidence |

**Total**: One focused session.

## 7. Future Opportunities Beyond Step 7

After Step 7, the remaining single-threaded bottlenecks are:
1. **Phase 0** (STP + sparse integration + scatter): ~0.5s — could be overlapped with Phase 1 for regions that don't depend on sparse results
2. **Phase 2** (ConductanceLIF batch kernel): ~0.45s — already one C++ call, minimal Python overhead
3. **Homeostasis** (`_apply_synaptic_scaling`): ~0.13s — could be batched or deferred to every Nth step
4. **cortex_sensory single-region time** (614ms): Dominated by 5 inhibitory network computations + TwoCompartmentLIF L23/L5 — could be optimized with a fused cortical inhibitory kernel

The next highest-ROI optimization after Step 7 would be **fusing the cortical inhibitory network computation** into C++ (addresses the 0.81s in `torch.mv` + 0.24s in `_integrate_synaptic_inputs_at_dendrites`), which would reduce the critical-path region time and further improve parallel scaling.
