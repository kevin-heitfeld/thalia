# Step 5: Global Sparse Synaptic Matrix — Revised Implementation Plan

## Context

Steps 1–4 are implemented: fused C++ ConductanceLIF kernel, batched STP, batched per-target-pop synaptic integration (Step 3's `ConcatWeightBlock`), and global ConductanceLIFBatch. Step 3 reduced ~6,150 matmuls to ~560 (one per target-pop × receptor type). Step 5 reduces those ~560 to 4 (one per receptor type) via global sparse CSR matrices, eliminating the remaining Python dispatch overhead (~0.25–0.5s over 50 steps).

### Key Design Decisions

1. **Column-replication architecture** (not [N_total, N_total]). Each connection (SynapseId) gets its own column range in the sparse matrix. This naturally handles per-connection axonal delays (different delayed spikes) and per-connection STP efficacy without any special casing.

2. **Scope**: ConductanceLIF + subclass neuron targets only (~35,769 neurons). TwoCompartmentLIF targets (~6,000 neurons) stay on the existing per-region integration path — Step 6 will batch those.

3. **Sparse-native learning (option b)**: New `compute_update_sparse()` on all learning strategies. CSR values become the authoritative weight storage. Dense `nn.Parameter` removed for eligible connections.

4. **NeuronIndexRegistry**: Unified brain-wide index using the same `(region_name, pop_name)` hierarchy as Philox RNG seeding.

---

## Architecture Overview

### Sparse Matrix Shape

Per receptor type (AMPA, NMDA, GABA_A, GABA_B):
- **Rows** = target neurons in global target index (~35,769: batched ConductanceLIF + subclass neurons)
- **Columns** = per-connection source slots. Each connection has column range `[col_offset, col_offset + n_source)`. Total cols = Σ n_source across all eligible connections for that receptor type.

### Per-Timestep Flow

```
1. Axonal tracts produce delayed spikes (unchanged)
2. STPBatch computes efficacy per connection (unchanged)
3. NEW: GlobalSparseMatrix.integrate():
   a. Zero spike buffers
   b. For each connection: copy delayed spikes into column range, multiply by STP efficacy
   c. g_R = W_R_csr @ spike_buf_R  (4 sparse matmuls)
   d. Scatter g_R into neuron input buffers (ConductanceLIFBatch + subclass buffers)
4. Regions _step(): ConductanceLIF populations skip _integrate_synaptic_inputs_at_dendrites
5. _neuron_batch.step(): fused C++ kernel (unchanged)
6. NEW: Learning via compute_update_sparse() on CSR values directly
```

---

## Sub-Steps

### 5.1: NeuronIndexRegistry

**New file**: `src/thalia/brain/neuron_index_registry.py`

A brain-level registry mapping every `PopulationKey = (region_name, pop_name)` to a contiguous global index range, for ALL neuron types. Two index spaces:

- **`target_registry`**: ConductanceLIF + subclass neurons only. Used as sparse matrix row indices and conductance output buffer indices. Ordering: reuse `ConductanceLIFBatch.registry` for batched pops (same order), then append subclass populations sorted by (region, pop).
- **`source_registry`**: All 50,869 neurons. Used for future steps; not strictly needed for Step 5's column-replication approach but establishes the unified index.

```python
class NeuronIndexRegistry:
    target_registry: dict[PopulationKey, tuple[int, int]]   # eligible targets
    source_registry: dict[PopulationKey, tuple[int, int]]   # all neurons
    total_target_neurons: int
    total_source_neurons: int

    def is_eligible_target(self, pop_key: PopulationKey) -> bool: ...
    def get_neuron_seeds(self, pop_key: PopulationKey) -> torch.Tensor: ...
```

`get_neuron_seeds()` generates deterministic per-neuron seeds using `f"{region_name}_{population_name}_{neuron_id}"` — same scheme as existing `_create_neuron_seeds()` in `conductance_lif_neuron.py`, now unified in one place.

**Modify**: `brain.py` — create registry in `__init__` after `_create_neuron_batch()`.

**Test**: Verify index ranges are contiguous, non-overlapping, and match `ConductanceLIFBatch.registry` for batched pops.

---

### 5.2: GlobalSparseMatrix — Construction

**New file**: `src/thalia/brain/sparse_synaptic_matrix.py`

Build 4 CSR sparse matrices from all per-connection dense weight matrices targeting eligible populations.

**Construction algorithm** (called from `Brain.__init__`):

1. Iterate all regions' `synaptic_weights` entries
2. Filter: include only connections where target population is an eligible target (ConductanceLIF or subclass, NOT TwoCompartmentLIF)
3. Group by receptor type (from `SynapseId.receptor_type`)
4. Per receptor type, sort connections deterministically, assign column offsets
5. For each connection's dense `[n_post, n_pre]` weight matrix:
   - Extract non-zero positions: `row_local, col_local = (W != 0).nonzero(as_tuple=True)`
   - Offset rows by target pop's global row start
   - Offset cols by this connection's column offset
   - Collect (row, col, value) COO triples
6. Concatenate all COO triples, convert to CSR: `torch.sparse_csr_tensor(crow, col, values, size)`
7. Build `ConnectionSparseMeta` per connection: `(nnz, local_row_indices, local_col_indices, value_slice)` where `value_slice` is a 1D view into the CSR `.values()` tensor

**Key data structures**:

```python
@dataclass
class ConnectionSparseMeta:
    synapse_id: SynapseId
    receptor_type: ReceptorType
    nnz: int
    local_row_indices: torch.Tensor   # [nnz], 0..n_post-1
    local_col_indices: torch.Tensor   # [nnz], 0..n_pre-1
    n_post: int
    n_pre: int
    value_slice: slice                # into CSR .values()
    col_offset: int                   # in spike buffer
    n_source: int                     # = n_pre, width of column range

class GlobalSparseMatrix:
    W: dict[ReceptorType, torch.Tensor]          # 4 CSR matrices
    spike_buf: dict[ReceptorType, torch.Tensor]  # 4 pre-allocated buffers
    g_out: dict[ReceptorType, torch.Tensor]      # 4 output buffers [N_target]
    connections: dict[SynapseId, ConnectionSparseMeta]
    target_registry: dict[PopulationKey, tuple[int, int]]  # from NeuronIndexRegistry
```

**Remove**: `nn.Parameter` entries from `region.synaptic_weights` for connections now in the sparse matrix. Learning reads/writes via `ConnectionSparseMeta.value_slice` into CSR `.values()`.

**Keep**: Dense `nn.Parameter` for connections targeting TwoCompartmentLIF (not in sparse matrix).

**Test**: For a small brain, verify `W_ampa @ spike_buf` output matches the reference per-source matmul loop element-by-element.

---

### 5.3: Spike Buffer Fill + Global Matmul + Output Scatter

**Add to `GlobalSparseMatrix`**:

```python
def integrate(
    self,
    region_inputs: dict[RegionName, SynapticInput],
    last_brain_output: Optional[BrainOutput],
    stp_efficacy: dict[SynapseId, torch.Tensor],
) -> None:
```

**Spike source resolution per connection**:
- Inter-region connections: `region_inputs[target_region][synapse_id]` (delayed, from axonal tracts)
- Intra-region connections: `last_brain_output[source_region][source_population]` (1-step delay)

**Per-step algorithm**:
```python
for receptor_type in (AMPA, NMDA, GABA_A, GABA_B):
    buf = self.spike_buf[receptor_type]
    buf.zero_()
    for meta in self._connections_by_receptor[receptor_type]:
        spikes = _resolve_spikes(meta.synapse_id, region_inputs, last_brain_output)
        if spikes is None:
            continue
        spikes_f = spikes.float()
        if meta.synapse_id in stp_efficacy:
            spikes_f = spikes_f * stp_efficacy[meta.synapse_id]
        buf[meta.col_offset : meta.col_offset + meta.n_source] = spikes_f

    self.g_out[receptor_type] = torch.sparse.mm(
        self.W[receptor_type], buf.unsqueeze(1)
    ).squeeze(1)
    self.g_out[receptor_type].clamp_(min=0.0)
```

**Output scatter**: After matmul, write results into neuron input buffers:
- Batched ConductanceLIF: `neuron_batch.g_ampa_input[batch_start:batch_end] = g_out[AMPA][row_start:row_end]`
  - Optimization: if target_registry row ordering matches ConductanceLIFBatch.registry, this is a single contiguous copy.
- Subclass neurons: store in per-population buffer dict for regions to read during `_step()`

**Modify `brain.py`**: Call `self._sparse_matrix.integrate(...)` after STP step, before Phase 1. Clear batch inputs is now handled by the sparse matrix scatter (replaces `_neuron_batch.clear_inputs()`).

**Test**: Full forward pass, compare all `g_ampa_input` values against old path.

---

### 5.4: Region Integration Bypass

**Goal**: Regions skip `_integrate_synaptic_inputs_at_dendrites` for populations whose conductances are already computed by the global sparse matrix.

**Modify `neural_region.py`**:

Add field set by `Brain.forward()` before each region's `_step()`:
```python
self._precomputed_sparse_conductances: Optional[dict[PopulationName, DendriteOutput]] = None
```

Modify `_integrate_synaptic_inputs_at_dendrites`:
```python
# At the top, before any matmul:
if (
    self._precomputed_sparse_conductances is not None
    and filter_by_target_population in self._precomputed_sparse_conductances
):
    return self._precomputed_sparse_conductances[filter_by_target_population]
```

Also modify `_integrate_single_synaptic_input` to check the same cache for the target population.

**Modify `brain.py`**: Before each region's `forward()`, build the `_precomputed_sparse_conductances` dict from `GlobalSparseMatrix.g_out` for that region's eligible populations.

**Cleanup**: `build_batched_dendrite_weights()` should skip connections already in the sparse matrix. `ConcatWeightBlock` / `BatchedDendriteWeights` remain only for TwoCompartmentLIF targets.

**Test**: Run full brain, verify all outputs match pre-refactor baseline.

---

### 5.5: Sparse-Native Learning

**Goal**: Refactor learning strategies to work directly on 1D sparse value arrays, eliminating dense [n_post, n_pre] weight matrices and connectivity masks.

#### 5.5a: New LearningStrategy Interface

**Modify `src/thalia/learning/strategies.py`**:

Add abstract method to `LearningStrategy` base class:
```python
def compute_update_sparse(
    self,
    values: torch.Tensor,         # [nnz] current weights
    row_indices: torch.Tensor,     # [nnz] local post indices (0..n_post-1)
    col_indices: torch.Tensor,     # [nnz] local pre indices (0..n_pre-1)
    n_post: int,
    n_pre: int,
    pre_spikes: torch.Tensor,      # [n_pre]
    post_spikes: torch.Tensor,     # [n_post]
    **kwargs,
) -> torch.Tensor:                 # [nnz] updated values
```

Default implementation (on base class) for backward compatibility:
```python
def compute_update_sparse(self, values, row_indices, col_indices, n_post, n_pre, pre_spikes, post_spikes, **kwargs):
    # Reconstruct dense, delegate, extract — fallback for un-migrated strategies
    dense = torch.zeros(n_post, n_pre, device=values.device)
    dense[row_indices, col_indices] = values
    updated = self.compute_update(dense, pre_spikes, post_spikes, **kwargs)
    return updated[row_indices, col_indices]
```

#### 5.5b: Sparse Trace Infrastructure

**Modify `src/thalia/learning/eligibility_trace_manager.py`**:

Add sparse mode support. When initialized with `nnz` + row/col indices instead of (n_input, n_output):
- `eligibility`: shape `[nnz]` instead of `[n_output, n_input]`
- `input_trace`: `[n_input]` (unchanged — per-pre-neuron)
- `output_trace`: `[n_output]` (unchanged — per-post-neuron)

New method:
```python
def compute_ltp_ltd_separate_sparse(self, input_spikes, output_spikes, row_idx, col_idx):
    # LTP: A_plus * input_trace[col_idx] * output_spikes[row_idx]  → [nnz]
    # LTD: A_minus * output_trace[row_idx] * input_spikes[col_idx] → [nnz]
```

#### 5.5c: Migrate Each Strategy

Key transform patterns used across strategies:

| Dense Pattern | Sparse Equivalent |
|---|---|
| `torch.outer(post, pre)` → `[n_post, n_pre]` | `post[row_idx] * pre[col_idx]` → `[nnz]` |
| `eligibility[n_post, n_pre]` | `eligibility[nnz]` |
| `weights + dw` | `values + dw` |
| `ltp * mask.unsqueeze(1)` (mask `[n_post]`) | `ltp * mask[row_idx]` |
| No connectivity mask needed | Sparse entries ARE the valid connections |

**Per-strategy changes**:

- **STDPStrategy**: Override `compute_update_sparse`. Use `compute_ltp_ltd_separate_sparse`. Retrograde signal and firing_rates stay `[n_post]`, indexed by `row_idx`. Add `setup_sparse(nnz, row_idx, col_idx, n_pre, n_post, device)` to allocate sparse eligibility.
- **BCMStrategy**: Override `compute_update_sparse`. theta and firing_rates stay `[n_post]`. `phi[row_idx] * pre[col_idx]` replaces `torch.outer(phi, pre)`.
- **ThreeFactorStrategy**: Override. Sparse eligibility via trace manager. `eligibility[nnz] * modulator`.
- **D1STDPStrategy**: Override. `fast_trace[nnz]`, `slow_trace[nnz]` instead of `[n_post, n_pre]`.
- **D2STDPStrategy**: Inherits D1's sparse method with inverted dopamine.
- **PredictiveCodingStrategy**: Override. `pre_spike_buffer` stays `[delay_steps, n_pre]`. The outer product `outer(post, delayed_pre)` → `post[row_idx] * delayed_pre[col_idx]`.
- **MaIStrategy**: Override. Stateless — just `outer(cf, pre)` → `cf[row_idx] * pre[col_idx]`.
- **CompositeStrategy**: Override. Chain sub-strategies' `compute_update_sparse`.
- **TagAndCaptureStrategy**: Override. `tags[nnz]` instead of `[n_post, n_pre]`.

#### 5.5d: Learning Dispatch

**Modify `neural_region.py` `_apply_learning()`**:

For connections in the global sparse matrix:
```python
meta = self._sparse_connection_meta[synapse_id]
updated_values = strategy.compute_update_sparse(
    values=meta.weight_view,      # 1D view into CSR .values()
    row_indices=meta.local_row_indices,
    col_indices=meta.local_col_indices,
    n_post=meta.n_post, n_pre=meta.n_pre,
    pre_spikes=pre_spikes,
    post_spikes=post_spikes,
    **learning_kwargs,
)
updated_values.clamp_(self.config.w_min, self.config.w_max)
meta.weight_view.copy_(updated_values)
# No connectivity mask needed — sparse entries ARE valid connections
```

For connections NOT in sparse matrix (TwoCompartmentLIF targets): existing dense path unchanged.

**Test**: Verify weight updates match pre-refactor for STDP, BCM, ThreeFactor, D1/D2STDP on a small brain configuration.

---

### 5.6: Cleanup

1. **Remove `ConcatWeightBlock` / `BatchedDendriteWeights`** for eligible targets. Keep only for TwoCompartmentLIF targets.
2. **Remove dense `nn.Parameter`** from `synaptic_weights` for connections in the sparse matrix. Add `get_dense_weights(synapse_id)` helper for diagnostics that reconstructs from CSR.
3. **Update `build_batched_dendrite_weights()`** to skip eligible targets.
4. **Update diagnostics** (`analysis_learning.py`, `diagnostics_plots.py`) to use `get_dense_weights()` where they currently read `synaptic_weights` directly.

---

## Files Summary

### New Files
| File | Purpose |
|---|---|
| `src/thalia/brain/neuron_index_registry.py` | Brain-wide global neuron index |
| `src/thalia/brain/sparse_synaptic_matrix.py` | GlobalSparseMatrix: CSR construction, spike fill, matmul, output scatter |
| `tests/test_sparse_synaptic_matrix.py` | Equivalence tests |

### Modified Files
| File | Changes |
|---|---|
| `src/thalia/brain/brain.py` | Create registry + sparse matrix in `__init__`; call `integrate()` + set precomputed conductances in `forward()` |
| `src/thalia/brain/regions/neural_region.py` | Add `_precomputed_sparse_conductances` bypass; adapt `_apply_learning()` for sparse path; limit `build_batched_dendrite_weights` to TwoCompartmentLIF; add `get_dense_weights()` |
| `src/thalia/learning/strategies.py` | Add `compute_update_sparse()` to all 10 strategies; add `setup_sparse()` for strategies with [n_post, n_pre] state |
| `src/thalia/learning/eligibility_trace_manager.py` | Add sparse mode: `[nnz]` eligibility, `compute_ltp_ltd_separate_sparse()` |
| `src/thalia/diagnostics/analysis_learning.py` | Use `get_dense_weights()` for weight inspection |
| `src/thalia/diagnostics/diagnostics_plots.py` | Use `get_dense_weights()` for weight visualization |

### Unchanged
- Neuron models (`conductance_lif_neuron.py`, `two_compartment_lif_neuron.py`, subclass neurons)
- `conductance_lif_batch.py` (input buffer API unchanged, sparse matrix writes into it)
- `stp_batch.py` (runs before sparse matmul, returns efficacy dict as before)
- `brain_builder.py` (construction unchanged)
- `axonal_tract.py` (delay buffers unchanged)
- All region subclasses (base class bypass handles everything)

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| CPU sparse CSR matmul slower than batched dense for small blocks | No speedup | Benchmark in 5.3 before proceeding. Fallback: custom C++ sparse kernel using `at::parallel_for` |
| Learning strategy refactor breaks subtle behavior | Incorrect learning | Per-strategy equivalence tests comparing sparse vs dense compute_update on identical inputs |
| Intra-region connections missed by spike resolution | Silent integration errors | Comprehensive test: dump all per-population conductances before/after refactor |
| Column-replication memory overhead | Larger spike buffers | ~600 connections × ~100 source avg = 60K floats × 4 receptors = 1MB total. Negligible. |

## Verification

1. **Unit test** (5.2): CSR construction produces correct values for known weight matrices
2. **Forward equivalence** (5.3–5.4): Run full brain 100 steps. Compare per-population conductance inputs (g_ampa etc.) between old and new path. Max absolute error < 1e-6.
3. **End-to-end regression** (5.6): Run 200-step simulation, compare output spike counts and firing rates per population.
4. **Performance** (5.6): Profile `Brain.forward()`. Verify matmul call count drops from ~560 to ~4.

Note: Learning equivalence tests are not needed — learning capabilities have not been validated/measured yet, so sparse learning just needs to be structurally correct, not regression-tested against the dense path.
