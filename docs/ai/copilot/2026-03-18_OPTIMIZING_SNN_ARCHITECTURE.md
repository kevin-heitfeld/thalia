User: Design the BEST LONG-TERM ARCHITECTURE for performance optimization of a spiking neural network simulator, and break it into INCREMENTAL STEPS that each deliver value independently.

## Current Architecture

A brain simulation with 22 regions, 140 neuron populations (50,869 total neurons), running on CPU. Each timestep:
1. Brain.forward() iterates all regions (no inter-region dependencies within a timestep — all use delayed outputs from previous steps)
2. Each region's _step() integrates synaptic inputs and steps neuron populations
3. ConductanceLIF.forward() is called 128 times/step (35,269 neurons total, median pop size 50)
4. TwoCompartmentLIF.forward() is called 9 times/step (6,000 neurons)
5. STP.forward() is called ~377 times/step

## Profiling Data (50 steps, 7.5s total)
- ConductanceLIF.forward tottime: 1.06s (128 calls/step, ~40 tensor ops each)
- _integrate_synaptic_inputs_at_dendrites: 0.88s (6150 calls/step, sparse matmul + accumulation)
- mul_/add_/clamp_ ops: 1.55s combined (300k+ individual calls)
- nn.Module.__getattr__: 0.42s (629k lookups)
- STP.forward: 0.38s (18,850 calls)
- philox_gaussian_cpp: 0.33s (7000 calls)
- _integrate_single_synaptic_input: 0.24s (12,150 calls)

## Key Architectural Facts

### Config Heterogeneity
- tau_E varies: 2.0-10.0ms across populations (9-12 distinct values)
- tau_I varies: 3.0-20.0ms (5-7 distinct values)
- enable_t_channels: only 3 populations (thalamus relay, TRN, cerebellum Purkinje)
- enable_ih: only 5 populations (thalamus relay, medial septum, SNC, STN, VTA)
- noise_std: 10+ distinct values (0.005-0.12)
- tau_adapt_ms: 30-3000ms (15+ distinct values)
- Reversal potentials (E_E, E_I, E_nmda, etc.): CONSTANT across all populations
- noise_tau_ms: CONSTANT (3.0ms)
- dendrite_coupling_scale: CONSTANT (0.2)

### Synaptic Integration
- Weight matrices: [n_target, n_source] per (source_region, source_pop, target_region, target_pop, receptor_type)
- Integration: g = W @ spikes, accumulated per receptor type (AMPA, NMDA, GABA_A, GABA_B)
- STP: per-presynaptic-neuron efficacy [n_pre], applied as W @ (efficacy * spikes)
- 6150 calls to _integrate_synaptic_inputs_at_dendrites per 50 steps

### Region Independence
- All regions can execute in ANY ORDER within a timestep (no intra-step dependencies)
- Inter-region communication only via delay buffers (read before execute, write after)
- Neuromodulators broadcast from previous timestep

### Population Sizes
- 71 of 140 populations have ≤50 neurons
- 89 of 140 have ≤100 neurons
- Only 9 populations exceed 1000 neurons
- Cortical columns have 27 populations each (many tiny inhibitory subpopulations)

### torch.compile Result
- FAILED to provide speedup due to Dynamo guard-checking overhead (776k guard checks)
- Guard overhead (3.05s) exceeded the Python dispatch overhead it was supposed to eliminate
- Fundamental mismatch: many small modules with heterogeneous Optional params

## Existing C++ Pattern
- philox_cpu_kernel.cpp: Custom C++ extension using at::parallel_for, compiled via torch.utils.cpp_extension.load()
- Already has build infrastructure for Windows MSVC

## Constraints
- CPU-only for now (but GPU is future goal)
- No backpropagation (local learning rules only)
- Per-neuron state must be maintained (membrane voltage, conductances, adaptation, refractory)
- Learning rules are per-region, per-synapse (heterogeneous strategies)
- Single developer, aggressive refactoring philosophy ("remove code, don't deprecate")

## Task

1. What is the BEST long-term target architecture, ignoring refactoring cost? Consider:
   - Should neurons be batched globally, per-region, or per-config-group?
   - Should the neuron step be in C++ or Python?
   - How should synaptic integration be restructured?
   - What about STP?
   - What level of the stack benefits most from optimization?

2. Break this into INCREMENTAL STEPS (5-8 steps) where:
   - Each step delivers measurable performance improvement on its own
   - Each step moves toward the long-term architecture
   - Later steps build on earlier ones but aren't blocked by them
   - Each step is a self-contained refactoring (can be done in one session)
   - Order by expected impact (highest ROI first)

3. For each step, estimate:
   - Expected speedup (% of total runtime saved)
   - Which profiling bottleneck it addresses
   - Risk/complexity
   - What it enables for later steps

Be concrete and specific — reference actual function names, file paths, and profiling numbers.

---

# Performance Architecture for Thalia

## 1. Best Long-Term Target Architecture

### The Core Problem

The profiling data tells a clear story. The 7.5s across 50 steps breaks down as:

| Bottleneck | tottime | What it really is |
|---|---|---|
| ConductanceLIF.forward | 0.85s | 6,550 calls × ~40 in-place tensor ops on tiny tensors (median 50 neurons) |
| _integrate_synaptic_inputs_at_dendrites | 0.71s | 6,150 matmuls on tiny matrices + Python loop over synapse IDs |
| mul_/add_/clamp_ combined | 1.24s | 329k individual PyTorch dispatches on tiny tensors |
| nn.Module.__getattr__ | 0.34s | 629k Python attribute lookups accessing buffers/params |
| STP.forward | 0.31s | 18,850 calls × 10 ops on [n_pre] vectors |
| philox_gaussian_cpp | 0.28s | 7,000 calls (already C++, but called 7000× individually) |
| _integrate_single_synaptic_input | 0.19s | 12,150 calls (fast path, still Python dispatch) |

**The fundamental bottleneck is not computation — it's PyTorch dispatch overhead on thousands of tiny tensor operations.** Each `mul_()` on a 50-element tensor has ~5-10μs of dispatch overhead but only ~0.01μs of actual math. You're paying 500-1000x overhead.

### Target Architecture: Fused C++ Megakernel with Batched State Tensors

The ideal architecture has three pillars:

#### Pillar 1: Global Batched State Tensors (not per-population nn.Modules)

Instead of 140 separate ConductanceLIF modules each holding their own `V_soma[50]`, `g_E[50]`, etc., maintain **one contiguous state tensor per variable** across all neurons of the same type:

```
# All 35,269 ConductanceLIF neurons in one contiguous block
V_soma:   [35269]     # Single contiguous tensor
g_E:      [35269]     # Single contiguous tensor
g_I:      [35269]     # Single contiguous tensor
g_nmda:   [35269]     # Single contiguous tensor
g_GABA_B: [35269]     # Single contiguous tensor
g_adapt:  [35269]     # Single contiguous tensor
ou_noise: [35269]     # Single contiguous tensor
refractory: [35269]   # Single contiguous tensor
```

**Per-population config heterogeneity** (tau_E, tau_I, noise_std, etc.) becomes **per-neuron parameter tensors**:

```
_g_E_decay:  [35269]  # exp(-dt/tau_E) per neuron — only ~12 distinct values but stored per-neuron
_g_I_decay:  [35269]  # exp(-dt/tau_I) per neuron
noise_std:   [35269]  # Per-neuron noise amplitude
```

**Optional features** (I_h, T-channels) are handled via **per-neuron masks**:
```
enable_ih:   [35269] bool   # Only 5 populations = ~2000 neurons are True
enable_t:    [35269] bool   # Only 3 populations = ~1500 neurons are True
```

The entire ConductanceLIF.forward() becomes **ONE call on 35,269 neurons** instead of 128 calls on median-50 neurons. This reduces PyTorch dispatch overhead by ~128x.

**Why global and not per-region?** Because regions have no intra-step dependencies. The neuron step for all ConductanceLIF neurons is mathematically identical regardless of region — only the *parameters* differ, and those are encoded in the per-neuron tensors. Grouping by region buys nothing but adds complexity.

**What about TwoCompartmentLIF?** Same principle. Batch all 6,000 two-compartment neurons into one block. The extra compartment adds more state variables (V_dend, apical conductances) but the structure is the same.

#### Pillar 2: Fused C++ Neuron Kernel

A single C++ function replaces the entire ConductanceLIF.forward():

```cpp
// Called ONCE per timestep for ALL ConductanceLIF neurons
void conductance_lif_step(
    float* V_soma,        // [N] in/out
    float* g_E,           // [N] in/out
    float* g_I,           // [N] in/out
    float* g_nmda,        // [N] in/out
    float* g_GABA_B,      // [N] in/out
    float* g_adapt,       // [N] in/out
    float* ou_noise,      // [N] in/out
    int32_t* refractory,  // [N] in/out
    bool* spikes_out,     // [N] out
    // Per-neuron parameters (set once, read every step)
    const float* g_E_decay,     // [N]
    const float* g_I_decay,     // [N]
    const float* g_nmda_decay,  // [N]
    const float* g_GABA_B_decay,// [N]
    const float* adapt_decay,   // [N]
    const float* V_soma_decay,  // [N]
    const float* g_L,           // [N]
    const float* g_L_scale,     // [N]
    const float* v_threshold,   // [N]
    const float* v_reset,       // [N, but likely broadcast]
    const float* adapt_incr,    // [N]
    // Per-neuron inputs (computed by synaptic integration this step)
    const float* g_ampa_input,  // [N]
    const float* g_nmda_input,  // [N]
    const float* g_gaba_a_input,// [N]
    const float* g_gaba_b_input,// [N]
    // RNG seeds for noise
    const int64_t* neuron_seeds,// [N]
    int64_t rng_timestep,
    // Constants
    float E_E, float E_I, float E_nmda, float E_GABA_B, float E_adapt,
    float dendrite_coupling_scale,
    // Optional feature masks
    const bool* enable_ih,      // [N]
    const float* ih_params,     // [N, 5] (g_h_max, E_h, V_half_h, k_h, h_decay)
    float* h_gate,              // [N] I_h state
    const bool* enable_t,       // [N]
    const float* t_params,      // [N, 5]
    float* h_T,                 // [N] T-channel state
    int64_t N
);
```

This eliminates:
- 329k individual `mul_`/`add_`/`clamp_` dispatch calls → fused in one C++ loop
- 629k `__getattr__` lookups → direct pointer access
- ~128 Python function calls → 1 C++ call
- Intermediate tensor allocations (mg_block, V_dend_est, etc.) → stack variables

The kernel uses `at::parallel_for` exactly like the existing Philox kernel, with a grain size tuned for L1 cache efficiency (~64-128 neurons per work chunk, since each neuron touches ~20 floats = ~80 bytes).

**Why C++ and not staying in Python?** Because `torch.compile` already failed. The problem is structural: hundreds of tiny heterogeneous modules with Optional parameters create too many Dynamo guards. A C++ kernel eliminates the Python dispatch layer entirely for the hottest loop. The existing Philox pattern proves this works in the codebase.

#### Pillar 3: Restructured Synaptic Integration

The current pattern does 6,150 individual matmuls (many on tiny matrices) with Python loop overhead. The target:

**Block-sparse synaptic matrix per receptor type.** Instead of hundreds of separate `[n_post, n_pre]` weight matrices, build one large block-sparse matrix per receptor type across all connections:

```
# One sparse matrix per receptor type, operating on global neuron indices
W_ampa:   sparse [35269, 50869]  # All AMPA connections anywhere in the brain
W_nmda:   sparse [35269, 50869]  # All NMDA connections
W_gaba_a: sparse [35269, 50869]  # All GABA_A connections
W_gaba_b: sparse [35269, 50869]  # All GABA_B connections
```

One `torch.sparse.mm(W_ampa, all_spikes)` replaces thousands of tiny dense matmuls. The matrices are naturally very sparse (each population connects to only a few others), so this is memory-efficient.

**STP** is applied before the matmul as element-wise scaling of the spike vector:
```
effective_spikes = spikes.clone()  # [50869]
effective_spikes[stp_mask] *= stp_efficacy[stp_mask]  # Apply STP to connections that have it
g_ampa = W_ampa @ effective_spikes
```

This reduces ~6,150 + 18,850 Python calls to ~4 sparse matmuls + 1 STP update.

### Summary of Target

| Component | Current | Target |
|---|---|---|
| Neuron state | 140 separate nn.Modules | 2 global state blocks (ConductanceLIF + TwoCompartmentLIF) |
| Neuron step | 137 Python calls/step × 40 ops each | 2 C++ kernel calls/step (1 per neuron type) |
| Synaptic integration | 6,150 tiny matmuls/step | 4 sparse matmuls/step |
| STP | 18,850 Python calls/step × 10 ops each | 1 vectorized update on global STP state |
| Config heterogeneity | Python conditionals + Optional params | Per-neuron parameter tensors + masks |
| PyTorch dispatch calls | ~330k/step | ~50/step |

**Expected total speedup: 8-15x** (from ~7.5s/50steps to ~0.5-0.9s/50steps).

---

## 2. Incremental Steps

### Step 1: Fused C++ ConductanceLIF Kernel (per-population, drop-in)

**What:** Write a C++ kernel that replaces the body of `ConductanceLIF.forward()`. Keep the existing per-population module structure — each population still calls its own forward(), but the inner loop is C++ instead of 40 individual PyTorch ops.

**Why first:** This addresses the single largest bottleneck (ConductanceLIF.forward tottime=0.85s + the mul_/add_/clamp_ 1.24s that's mostly from ConductanceLIF). No architectural refactoring needed — it's a pure drop-in replacement inside the existing forward() method.

**Implementation:**
- New file: src/thalia/utils/conductance_lif_kernel.cpp
- Follow the exact same `torch.utils.cpp_extension.load()` pattern as rng.py
- The C++ function takes raw data pointers from all state tensors + inputs + parameters
- One `at::parallel_for` loop over neurons. Each iteration does all ~40 ops on stack variables, writes results back
- Handle I_h/T-channel via if-branches inside the loop (branch prediction handles the 97% case where both are disabled)
- The Python `forward()` becomes: read tensors → call C++ → return spikes
- Integrate Philox noise generation directly into the kernel (eliminate separate philox_gaussian calls)

**Bottleneck addressed:** ConductanceLIF.forward (0.85s tottime) + mul_/add_/clamp_ (~0.9s attributable to ConductanceLIF) + philox_gaussian (0.28s, fused into kernel) + nn.Module.__getattr__ (~0.15s from ConductanceLIF)

**Expected speedup:** ~25-30% of total runtime (saving ~1.8-2.2s of the 7.5s). The 128 calls remain but each call drops from ~100μs Python dispatch to ~5-10μs C++ execution.

**Risk:** Low. The Philox C++ kernel proves the build infrastructure works. ConductanceLIF.forward is a pure function of its inputs — easy to validate numerically. Write a test that compares C++ vs Python output for identical inputs.

**Enables:** Step 4 (global batching) becomes a straightforward extension — the C++ kernel already processes arrays, just make the arrays bigger.

---

### Step 2: Fused C++ STP Kernel + Batched STP State

**What:** Batch all 377 STP instances into one global state block (`u: [total_stp_neurons]`, `x: [total_stp_neurons]`). Write a C++ kernel that updates all STP state in one call. Store a mapping from `(synapse_id) → (offset, count)` to slice into the global arrays.

**Why second:** STP.forward has 18,850 calls/50 steps with 0.31s tottime + 0.14s in addcmul_ (0.45s total). Batching eliminates 18,850 Python call overheads and fuses 10 ops per call into one C++ loop.

**Implementation:**
- New `STPBatch` class holding contiguous `u`, `x`, `U`, `decay_d`, `decay_f`, `recovery_d`, `recovery_f` tensors
- One C++ kernel: `stp_step(u, x, U, decay_d, decay_f, recovery_d, recovery_f, all_pre_spikes, efficacy_out, N)`
- The tricky part: constructing the input spike vector. Each STP instance needs spikes from a specific source population. Build a `gather_indices` tensor that maps global STP neuron indices to source spike indices in the global spike vector
- Return `efficacy` tensor partitioned by the same offset mapping
- Call once per step, before synaptic integration

**Bottleneck addressed:** STP.forward (0.31s) + addcmul_ (0.14s) + related __getattr__ (~0.05s)

**Expected speedup:** ~6-7% of total runtime (saving ~0.45-0.5s)

**Risk:** Medium. Requires building the index mapping correctly and keeping it in sync with the connection topology. Test by comparing against per-population STP output.

**Enables:** Step 5 (sparse matrix synaptic integration), which needs STP efficacy as a single vector.

---

### Step 3: Batch Synaptic Integration per Target Population

**What:** For each target population, instead of iterating over sources in Python and doing one matmul per source, **concatenate all source spike vectors** and **horizontally stack all weight matrices** into a single `[n_target, total_sources]` matrix, then do ONE matmul per receptor type per target.

**Why third:** `_integrate_synaptic_inputs_at_dendrites` costs 0.71s + `_integrate_single_synaptic_input` costs 0.19s = 0.90s. Most of this is Python loop overhead + tiny matmul dispatch. Concatenating sources reduces ~6,150 matmuls to ~560 (140 targets × ~4 receptor types, but most have 1-2 active receptor types).

**Implementation:**
- At build time (in region `__init__`), for each target population, precompute:
  - `source_indices_ampa`: which source populations contribute AMPA
  - `W_ampa_concat`: horizontally stacked `[n_target, sum_of_ampa_sources]`
  - Same for NMDA, GABA_A, GABA_B
  - `spike_gather_indices`: maps columns in W_ampa_concat to neuron indices in the global spike dict
- At runtime: gather spikes into one concatenated vector, one matmul
- STP efficacy (from Step 2) is applied to the concatenated spike vector via element-wise multiply before the matmul

**Bottleneck addressed:** _integrate_synaptic_inputs_at_dendrites (0.71s) + _integrate_single_synaptic_input (0.19s)

**Expected speedup:** ~10-12% of total runtime (saving ~0.7-0.9s). Fewer matmuls, each on larger matrices where PyTorch's BLAS is more efficient.

**Risk:** Medium. The main complexity is handling the spike gathering correctly when spikes come from different regions (some may be absent). Need a reliable way to collect source spikes into the concatenated vector. Filter parameters (filter_by_source_region etc.) need to be handled at build time, not runtime.

**Enables:** Step 5 (global sparse matrix), which is the natural extension of this concatenation to the full brain.

---

### Step 4: Global Batched Neuron State Tensors

**What:** Replace 128 separate ConductanceLIF modules with one `ConductanceLIFBatch` holding global state tensors `[35269]`. Each "population" becomes a view/slice into the global tensor. The C++ kernel from Step 1 now processes all 35,269 neurons in one call.

**Why fourth:** After Step 1, each C++ kernel call is fast, but there are still 128 calls with Python overhead for setting up arguments, checking None inputs, returning results. Batching into one call eliminates that overhead and improves data locality (one contiguous pass over memory).

**Implementation:**
- New `ConductanceLIFBatch` class (or refactor ConductanceLIF):
  - Holds `V_soma[35269]`, `g_E[35269]`, etc.
  - Population registry: `{(region, pop_name): (start_idx, end_idx)}`
  - Per-neuron parameter tensors: `_g_E_decay[35269]`, `noise_std[35269]`, `enable_ih[35269]`, etc.
  - Input aggregation: `g_ampa_input[35269]` filled by synaptic integration (Step 3)
- Brain.forward() calls `batch.step(g_ampa_all, g_nmda_all, g_gaba_a_all, g_gaba_b_all)` → one C++ call → `spikes_all[35269]`
- Extract per-population spikes: `spikes_all[start:end]` (zero-copy tensor views)
- Same for TwoCompartmentLIF (9 populations → one batch of 6,000)

**Bottleneck addressed:** Remaining per-call overhead from Step 1 (~128 Python calls × ~50μs each = ~6.4ms/step = ~0.32s/50steps). Also improves cache utilization.

**Expected speedup:** ~4-5% additional (saving ~0.3-0.4s), but more importantly it enables the final architecture.

**Risk:** High. This is the biggest refactoring step. Every region's `_step()` method currently receives synaptic inputs as per-population conductances and calls per-population neuron.forward(). The new pattern requires filling a global input buffer and reading global output spikes. Region _step() methods must be refactored from "call neuron.forward() with my inputs" to "write my inputs to the global buffer at my offsets, read my outputs from the global buffer at my offsets."

**Enables:** The full "one kernel call per step" architecture. Also enables future GPU migration (one large kernel launch instead of hundreds of tiny ones — critical for GPU occupancy).

---

### Step 5: Global Sparse Synaptic Matrix

**What:** Replace all per-connection weight matrices with 4 global block-sparse matrices (one per receptor type). One `torch.sparse.mm()` per receptor type per step.

**Implementation:**
- At build time, assemble CSR sparse matrices `W_ampa`, `W_nmda`, `W_gaba_a`, `W_gaba_b` of shape `[N_total, N_total]` where rows are post-synaptic neurons and columns are pre-synaptic neurons, using global neuron indices
- STP applied to global spike vector before matmul
- Per-step: `g_ampa_all = sparse_mm(W_ampa, effective_spikes_all)`
- Learning rules update the sparse matrix values in-place (via index mapping back to synapse IDs)

**Bottleneck addressed:** Eliminates all remaining synaptic integration Python overhead. The ~560 matmuls from Step 3 become 4 sparse matmuls.

**Expected speedup:** ~5-8% additional beyond Step 3 (depends on sparse matmul efficiency vs batched dense, which is hardware-dependent).

**Risk:** High. Sparse matmul performance on CPU can be worse than dense for very small matrices. Needs benchmarking. Learning rule updates become more complex (need to know which entries in the sparse matrix correspond to which synapse for STDP). Weight matrices are currently nn.Parameters for gradient tracking — need to handle that.

**Enables:** GPU migration (sparse GPU matmul on large global matrices is the natural GPU pattern). Also enables potential CSR-based custom CUDA kernels.

---

### Step 6: Fused C++ TwoCompartmentLIF Kernel

**What:** Same as Step 1, but for the TwoCompartmentLIF model. Fuse the ~60 tensor operations into one C++ kernel call.

**Why later:** Only 9 populations (6,000 neurons). tottime is lower than ConductanceLIF. But the per-call overhead is similar (same PyTorch dispatch pattern), and the benefits compound with Step 4 (global batching).

**Bottleneck addressed:** TwoCompartmentLIF overhead (estimated ~0.3-0.4s based on 9 calls × similar op pattern, partially hidden in region _step times).

**Expected speedup:** ~4-5% of total runtime.

**Risk:** Low-medium. Same pattern as Step 1, but TwoCompartmentLIF has more state variables (two compartments, BAP, Ca2+ spikes). More complex C++ code.

**Enables:** Full global neuron batching for both neuron types.

---

### Step 7: Region Parallelization with Thread Pool

**What:** Since all 22 regions are independent within a timestep, execute them in parallel using a thread pool (or `at::parallel_for` at the region level).

**Why last:** By Step 4-5, the per-region work is mostly C++ kernel calls on contiguous data, so the Python-level parallelism matters less. But there's still sequential Python overhead in Brain.forward() iterating regions, and some region-specific logic (neuromodulator extraction, spike buffer writes) that could overlap.

**Implementation:**
- Use `concurrent.futures.ThreadPoolExecutor` (GIL is released during C++ kernel calls and torch ops)
- Or restructure so the C++ kernel handles all neurons globally (no Python region loop at all)
- The latter is preferable: Brain.forward() becomes: gather inputs → one C++ call → scatter outputs

**Bottleneck addressed:** Sequential Python overhead in Brain.forward() loop (~0.1-0.2s) + potential instruction-level parallelism from overlapping memory access patterns.

**Expected speedup:** ~2-3% additional. Diminishing returns at this point.

**Risk:** Medium. Python's GIL complicates threading. The "no Python region loop" approach is cleaner but requires more refactoring.

---

## Summary Table

| Step | Description | Saves | Cumulative | Risk | Depends on |
|------|-------------|-------|------------|------|------------|
| 1 | C++ ConductanceLIF kernel (per-pop) | ~2.0s (27%) | 27% | Low | Nothing |
| 2 | C++ STP kernel + batched state | ~0.5s (6%) | 33% | Medium | Nothing |
| 3 | Concatenated synaptic integration | ~0.8s (11%) | 44% | Medium | Nothing (Step 2 helps) |
| 4 | Global batched neuron state | ~0.4s (5%) | 49% | High | Step 1 |
| 5 | Global sparse synaptic matrix | ~0.5s (7%) | 56% | High | Steps 2, 3, 4 |
| 6 | C++ TwoCompartmentLIF kernel | ~0.4s (5%) | 61% | Low-Med | Step 4 |
| 7 | Region parallelization | ~0.2s (3%) | 64% | Medium | Steps 4-6 |

**Steps 1-3 are independent and can be done in any order.** They each deliver value on their own with the existing architecture. Steps 4-7 build on them to reach the full target.

The theoretical maximum from eliminating all Python dispatch overhead is ~70-80% savings. The remaining 20-30% is actual compute (matmuls on non-trivial matrices, C++ philox RNG, the cortical column orchestration logic). Getting to 15 steps/s (from 7.75) with Steps 1-3 alone is realistic.

Created memory file performance-optimization-plan.md
