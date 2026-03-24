"""Global sparse synaptic matrices for the entire brain.

Replaces the per-region, per-connection dense matmul system (~560 matmuls
per timestep) with four global sparse CSR matrices — one per receptor type
(AMPA, NMDA, GABA_A, GABA_B) — each requiring a single ``sparse @ dense``
operation per timestep.

**Column-replication architecture**: Each connection (SynapseId) gets its own
column range in the sparse matrix.  This naturally handles per-connection
axonal delays and per-connection STP efficacy: the spike buffer is filled
with correctly-delayed, STP-scaled spikes for each connection independently.

**Scope**: Only connections targeting ConductanceLIF or ConductanceLIF-subclass
populations (i.e. populations in the :class:`NeuronIndexRegistry` target
registry).  TwoCompartmentLIF targets remain on the per-region integration
path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from thalia.typing import (
    BrainOutput,
    PopulationKey,
    PopulationName,
    ReceptorType,
    RegionName,
    SynapseId,
    SynapticInput,
)

if TYPE_CHECKING:
    from thalia.brain.neuron_index_registry import NeuronIndexRegistry
    from thalia.brain.neurons.conductance_lif_batch import ConductanceLIFBatch
    from thalia.brain.regions.neural_region import NeuralRegion

logger = logging.getLogger(__name__)

# All receptor types in a fixed order for iteration
_RECEPTOR_TYPES = (ReceptorType.AMPA, ReceptorType.NMDA, ReceptorType.GABA_A, ReceptorType.GABA_B)


@dataclass
class ConnectionSparseMeta:
    """Sparse connectivity metadata for one connection in the global matrix.

    Provides the mapping between the CSR value array and the local
    (n_post, n_pre) space that learning rules operate in.
    """

    synapse_id: SynapseId
    receptor_type: ReceptorType
    nnz: int
    local_row_indices: torch.Tensor  # [nnz], values in 0..n_post-1
    local_col_indices: torch.Tensor  # [nnz], values in 0..n_pre-1
    n_post: int
    n_pre: int
    value_offset: int  # start index into CSR .values()
    value_end: int     # end index (exclusive) into CSR .values()
    col_offset: int    # start column in the spike buffer
    n_source: int      # width of column range (= n_pre)
    region_name: RegionName  # owning region (for learning dispatch)


class GlobalSparseMatrix:
    """Four CSR sparse matrices for global synaptic integration.

    Constructed at brain init from all per-connection dense weight matrices
    targeting eligible populations.  Per timestep, :meth:`integrate` fills
    spike buffers from axonal tract outputs, applies STP, and runs 4 sparse
    matmuls to produce global conductance vectors.
    """

    def __init__(
        self,
        regions: dict[RegionName, NeuralRegion],
        neuron_index: NeuronIndexRegistry,
        neuron_batch: ConductanceLIFBatch,
        device: torch.device,
    ) -> None:
        self._device = device
        self._neuron_index = neuron_index
        self._neuron_batch = neuron_batch
        self._total_target = neuron_index.total_target_neurons

        # Per-receptor: list of ConnectionSparseMeta, sorted deterministically
        self._connections_by_receptor: dict[ReceptorType, list[ConnectionSparseMeta]] = {
            r: [] for r in _RECEPTOR_TYPES
        }
        # Quick lookup: SynapseId → ConnectionSparseMeta
        self.connections: dict[SynapseId, ConnectionSparseMeta] = {}

        # Collect all eligible connections grouped by receptor type
        eligible: dict[ReceptorType, list[tuple[SynapseId, str, torch.Tensor]]] = {
            r: [] for r in _RECEPTOR_TYPES
        }

        for region_name, region in regions.items():
            for synapse_id, weight_param in region.synaptic_weights.items():
                target_key: PopulationKey = (synapse_id.target_region, synapse_id.target_population)
                if not neuron_index.is_eligible_target(target_key):
                    continue
                # Only include inter-region connections.  Intra-region connections
                # (source_region == target_region) go through prev_region_output
                # and _integrate_single_synaptic_input in each region's _step(),
                # which is separate from the synaptic_inputs dict.  Including
                # them here would cause double-counting.
                if synapse_id.source_region == synapse_id.target_region:
                    continue
                eligible[synapse_id.receptor_type].append(
                    (synapse_id, region_name, weight_param.data)
                )

        # Build CSR matrix per receptor type
        self.W: dict[ReceptorType, Optional[torch.Tensor]] = {}
        self.spike_buf: dict[ReceptorType, torch.Tensor] = {}
        self.g_out: dict[ReceptorType, torch.Tensor] = {}

        total_connections = 0

        for receptor_type in _RECEPTOR_TYPES:
            entries = eligible[receptor_type]
            if not entries:
                self.W[receptor_type] = None
                self.spike_buf[receptor_type] = torch.zeros(0, device=device)
                self.g_out[receptor_type] = torch.zeros(self._total_target, device=device)
                continue

            # Sort deterministically for reproducible column ordering
            entries.sort(key=lambda e: e[0].to_key())

            # Assign column offsets and extract COO triples
            col_offset = 0
            all_rows: list[torch.Tensor] = []
            all_cols: list[torch.Tensor] = []
            all_vals: list[torch.Tensor] = []
            value_offset = 0
            metas: list[ConnectionSparseMeta] = []

            for synapse_id, region_name, weight_data in entries:
                target_key = (synapse_id.target_region, synapse_id.target_population)
                row_start, _row_end = neuron_index.get_target_slice(target_key)
                n_post, n_pre = weight_data.shape

                # Extract non-zero entries
                local_rows, local_cols = weight_data.nonzero(as_tuple=True)
                values = weight_data[local_rows, local_cols]
                nnz = values.numel()

                if nnz > 0:
                    # Offset to global coordinates
                    global_rows = local_rows + row_start
                    global_cols = local_cols + col_offset
                    all_rows.append(global_rows)
                    all_cols.append(global_cols)
                    all_vals.append(values)

                meta = ConnectionSparseMeta(
                    synapse_id=synapse_id,
                    receptor_type=receptor_type,
                    nnz=nnz,
                    local_row_indices=local_rows.to(device),
                    local_col_indices=local_cols.to(device),
                    n_post=n_post,
                    n_pre=n_pre,
                    value_offset=value_offset,
                    value_end=value_offset + nnz,
                    col_offset=col_offset,
                    n_source=n_pre,
                    region_name=region_name,
                )
                metas.append(meta)
                self.connections[synapse_id] = meta

                value_offset += nnz
                col_offset += n_pre

            total_cols = col_offset
            total_nnz = value_offset
            total_connections += len(metas)

            if total_nnz == 0:
                self.W[receptor_type] = None
                self.spike_buf[receptor_type] = torch.zeros(total_cols, device=device)
                self.g_out[receptor_type] = torch.zeros(self._total_target, device=device)
                self._connections_by_receptor[receptor_type] = metas
                continue

            # Build COO and convert to CSR
            row_tensor = torch.cat(all_rows).to(device)
            col_tensor = torch.cat(all_cols).to(device)
            val_tensor = torch.cat(all_vals).to(device)

            # Create COO first, then convert to CSR for efficient matmul
            coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_tensor, col_tensor]),
                values=val_tensor,
                size=(self._total_target, total_cols),
                device=device,
            ).coalesce()

            csr = coo.to_sparse_csr()
            self.W[receptor_type] = csr

            # Now build value views: map each connection's meta to a slice
            # of the CSR values array.  The CSR conversion may reorder values
            # (row-major order), so we need to rebuild the mapping.
            csr_crow = csr.crow_indices()
            csr_col = csr.col_indices()
            csr_val = csr.values()

            # Rebuild per-connection value slices from the CSR structure.
            # For each connection, find its entries in the CSR by matching
            # (global_row, global_col) positions.
            self._rebuild_value_slices(metas, csr_crow, csr_col, csr_val, neuron_index)

            self.spike_buf[receptor_type] = torch.zeros(total_cols, device=device)
            self.g_out[receptor_type] = torch.zeros(self._total_target, device=device)
            self._connections_by_receptor[receptor_type] = metas

        logger.info(
            "GlobalSparseMatrix: %d connections, %d total nnz across 4 receptor types",
            total_connections,
            sum(m.nnz for m in self.connections.values()),
        )

        # Reverse index: (target_region, target_population) → [SynapseId]
        # Used by _apply_synaptic_scaling to avoid scanning all connections.
        self._synapse_ids_by_target: dict[tuple[RegionName, PopulationName], list[SynapseId]] = {}
        for synapse_id in self.connections:
            key = (synapse_id.target_region, synapse_id.target_population)
            if key not in self._synapse_ids_by_target:
                self._synapse_ids_by_target[key] = []
            self._synapse_ids_by_target[key].append(synapse_id)

    @staticmethod
    def _rebuild_value_slices(
        metas: list[ConnectionSparseMeta],
        csr_crow: torch.Tensor,
        csr_col: torch.Tensor,
        csr_val: torch.Tensor,
        neuron_index: NeuronIndexRegistry,
    ) -> None:
        """Rebuild value_offset/value_end after COO→CSR reordering.

        CSR stores entries in row-major order, which may differ from the
        insertion order used during construction.  We scan each connection's
        target rows in the CSR to locate its entries using vectorized ops.
        """
        device = csr_val.device

        for meta in metas:
            target_key = (meta.synapse_id.target_region, meta.synapse_id.target_population)
            row_start, row_end = neuron_index.get_target_slice(target_key)
            col_lo = meta.col_offset
            col_hi = meta.col_offset + meta.n_source

            # Get CSR row pointers for all target rows: [n_rows + 1]
            crow_slice = csr_crow[row_start: row_end + 1]
            base = int(crow_slice[0].item())
            end = int(crow_slice[-1].item())

            if base == end:
                # No entries at all in these rows
                meta.nnz = 0
                meta.value_offset = 0
                meta.value_end = 0
                meta.local_row_indices = torch.zeros(0, dtype=torch.long, device=device)
                meta.local_col_indices = torch.zeros(0, dtype=torch.long, device=device)
                meta._csr_indices = torch.zeros(0, dtype=torch.long, device=device)  # type: ignore[attr-defined]
                continue

            # Flat index of all CSR entries in target rows
            all_idx = torch.arange(base, end, dtype=torch.long, device=device)
            cols = csr_col[all_idx]

            # Filter by column range [col_lo, col_hi)
            mask = (cols >= col_lo) & (cols < col_hi)
            valid_idx = all_idx[mask]

            nnz = valid_idx.numel()
            meta.nnz = nnz

            if nnz == 0:
                meta.value_offset = 0
                meta.value_end = 0
                meta.local_row_indices = torch.zeros(0, dtype=torch.long, device=device)
                meta.local_col_indices = torch.zeros(0, dtype=torch.long, device=device)
                meta._csr_indices = torch.zeros(0, dtype=torch.long, device=device)
                continue

            # Compute local row indices via searchsorted on row pointers
            # crow_slice[1:] gives cumulative end pointers for each row
            # right=True because row i owns indices [crow[i], crow[i+1])
            local_rows = torch.searchsorted(crow_slice[1:], valid_idx, right=True)
            local_cols = cols[mask] - col_lo

            meta._csr_indices = valid_idx
            meta.local_row_indices = local_rows
            meta.local_col_indices = local_cols
            meta.value_offset = int(valid_idx[0].item())
            meta.value_end = int(valid_idx[-1].item()) + 1

    # =====================================================================
    # PER-STEP INTEGRATION
    # =====================================================================

    def integrate(
        self,
        region_inputs: dict[RegionName, SynapticInput],
        last_brain_output: Optional[BrainOutput],
        stp_efficacy: dict[SynapseId, torch.Tensor],
    ) -> None:
        """Fill spike buffers, apply STP, run sparse matmul, store results.

        After this call, ``g_out[receptor_type]`` contains the global
        conductance vector ``[N_target]`` for each receptor type.

        Args:
            region_inputs: Per-region synaptic inputs from axonal tracts.
            last_brain_output: Previous step's BrainOutput (for intra-region
                connections with 1-step delay).
            stp_efficacy: Per-connection STP efficacy from STPBatch.
        """
        for receptor_type in _RECEPTOR_TYPES:
            W = self.W[receptor_type]
            if W is None:
                self.g_out[receptor_type].zero_()
                continue

            buf = self.spike_buf[receptor_type]
            buf.zero_()

            for meta in self._connections_by_receptor[receptor_type]:
                spikes = self._resolve_spikes(
                    meta.synapse_id, region_inputs, last_brain_output,
                )
                if spikes is None:
                    continue

                # Apply STP efficacy (per-connection, not per-neuron)
                sid = meta.synapse_id
                if sid in stp_efficacy:
                    spikes = stp_efficacy[sid] * spikes

                buf[meta.col_offset: meta.col_offset + meta.n_source] = spikes

            # Sparse matmul: [N_target, N_cols] @ [N_cols, 1] → [N_target, 1]
            result = torch.mv(W, buf)
            result.clamp_(min=0.0)
            self.g_out[receptor_type] = result

    @staticmethod
    def _resolve_spikes(
        synapse_id: SynapseId,
        region_inputs: dict[RegionName, SynapticInput],
        last_brain_output: Optional[BrainOutput],
    ) -> Optional[torch.Tensor]:
        """Resolve the spike tensor for a connection.

        Inter-region connections: look up in region_inputs (axonal tract output).
        Intra-region connections: look up in last_brain_output (1-step delay).
        """
        target_region = synapse_id.target_region
        source_region = synapse_id.source_region
        source_pop = synapse_id.source_population

        # Try inter-region (from axonal tracts)
        region_dict = region_inputs.get(target_region)
        if region_dict is not None and synapse_id in region_dict:
            return region_dict[synapse_id]

        # Try intra-region (from previous step's output)
        if (
            last_brain_output is not None
            and source_region in last_brain_output
            and source_pop in last_brain_output[source_region]
        ):
            return last_brain_output[source_region][source_pop]

        return None

    # =====================================================================
    # OUTPUT SCATTER
    # =====================================================================

    def scatter_to_neuron_batch(self) -> None:
        """Copy sparse matmul results into ConductanceLIFBatch input buffers.

        Must be called after :meth:`integrate`.
        """
        batch = self._neuron_batch
        batch.clear_inputs()

        for pop_key, (row_start, row_end) in self._neuron_index.target_registry.items():
            if not batch.is_batched(pop_key):
                # Subclass neuron — handled via get_subclass_conductances()
                continue

            batch_start, batch_end = batch.registry[pop_key]
            n = row_end - row_start
            assert batch_end - batch_start == n

            batch.g_ampa_input[batch_start:batch_end] = self.g_out[ReceptorType.AMPA][row_start:row_end]
            batch.g_nmda_input[batch_start:batch_end] = self.g_out[ReceptorType.NMDA][row_start:row_end]
            batch.g_gaba_a_input[batch_start:batch_end] = self.g_out[ReceptorType.GABA_A][row_start:row_end]
            batch.g_gaba_b_input[batch_start:batch_end] = self.g_out[ReceptorType.GABA_B][row_start:row_end]

    def get_subclass_conductances(
        self, region_name: RegionName,
    ) -> dict[PopulationName, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get conductances for subclass neuron populations in a region.

        Returns dict mapping pop_name → (g_ampa, g_nmda, g_gaba_a, g_gaba_b)
        for populations that are eligible targets but NOT in ConductanceLIFBatch
        (i.e. SerotoninNeuron, NorepinephrineNeuron, AcetylcholineNeuron).
        """
        result: dict[PopulationName, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        batch = self._neuron_batch

        for pop_key, (row_start, row_end) in self._neuron_index.target_registry.items():
            r_name, pop_name = pop_key
            if r_name != region_name:
                continue
            if batch.is_batched(pop_key):
                continue  # Handled by scatter_to_neuron_batch

            result[pop_name] = (
                self.g_out[ReceptorType.AMPA][row_start:row_end],
                self.g_out[ReceptorType.NMDA][row_start:row_end],
                self.g_out[ReceptorType.GABA_A][row_start:row_end],
                self.g_out[ReceptorType.GABA_B][row_start:row_end],
            )

        return result

    def get_region_conductances(
        self, region_name: RegionName,
    ) -> dict[PopulationName, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get conductances for ALL eligible target populations in a region.

        Returns dict mapping pop_name → (g_ampa, g_nmda, g_gaba_a, g_gaba_b)
        for all populations in the target registry belonging to *region_name*
        (both batched ConductanceLIF and subclass neurons).

        Used by brain.py to set ``_precomputed_sparse_conductances`` on each
        region so that ``_integrate_synaptic_inputs_at_dendrites`` is bypassed
        for all sparse-eligible targets.
        """
        result: dict[PopulationName, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for pop_key, (row_start, row_end) in self._neuron_index.target_registry.items():
            r_name, pop_name = pop_key
            if r_name != region_name:
                continue

            result[pop_name] = (
                self.g_out[ReceptorType.AMPA][row_start:row_end],
                self.g_out[ReceptorType.NMDA][row_start:row_end],
                self.g_out[ReceptorType.GABA_A][row_start:row_end],
                self.g_out[ReceptorType.GABA_B][row_start:row_end],
            )

        return result

    # =====================================================================
    # WEIGHT ACCESS (for learning and diagnostics)
    # =====================================================================

    def get_weight_values(self, synapse_id: SynapseId) -> torch.Tensor:
        """Get the 1D weight values for a connection (view into CSR values)."""
        meta = self.connections[synapse_id]
        csr_val = self.W[meta.receptor_type].values()  # type: ignore[union-attr]
        return csr_val[meta._csr_indices]  # type: ignore[attr-defined]

    def set_weight_values(self, synapse_id: SynapseId, new_values: torch.Tensor) -> None:
        """Write updated weight values back into the CSR."""
        meta = self.connections[synapse_id]
        csr_val = self.W[meta.receptor_type].values()  # type: ignore[union-attr]
        csr_val[meta._csr_indices] = new_values  # type: ignore[attr-defined]

    def get_dense_weights(self, synapse_id: SynapseId) -> torch.Tensor:
        """Reconstruct a dense [n_post, n_pre] weight matrix from CSR entries.

        Useful for diagnostics and weight visualization.
        """
        meta = self.connections[synapse_id]
        values = self.get_weight_values(synapse_id)
        dense = torch.zeros(meta.n_post, meta.n_pre, device=self._device)
        dense[meta.local_row_indices, meta.local_col_indices] = values
        return dense

    def get_connection_meta(self, synapse_id: SynapseId) -> ConnectionSparseMeta:
        """Get sparse metadata for a connection."""
        return self.connections[synapse_id]

    def has_connection(self, synapse_id: SynapseId) -> bool:
        """Check if a connection is managed by this sparse matrix."""
        return synapse_id in self.connections

    def get_synapse_ids_for_target(
        self, target_region: RegionName, target_population: PopulationName,
    ) -> list[SynapseId]:
        """Return SynapseIds targeting a specific (region, population) pair."""
        return self._synapse_ids_by_target.get((target_region, target_population), [])
