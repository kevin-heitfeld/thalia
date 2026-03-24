"""Tests for the GlobalSparseMatrix and NeuronIndexRegistry.

Verifies:
1. CSR construction produces correct sparse structure
2. Sparse matmul produces same conductances as dense per-connection matmul
3. Weight read/write roundtrips correctly
4. get_dense_weights reconstructs the original matrix
5. _rebuild_value_slices correctly maps CSR entries back to connections
6. Only inter-region connections are included (intra-region filtered out)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from thalia.brain.neuron_index_registry import NeuronIndexRegistry
from thalia.brain.sparse_synaptic_matrix import GlobalSparseMatrix
from thalia.typing import ReceptorType, SynapseId


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

def _sid(src_reg: str, src_pop: str, tgt_reg: str, tgt_pop: str,
         receptor: ReceptorType = ReceptorType.AMPA) -> SynapseId:
    return SynapseId(
        source_region=src_reg,
        source_population=src_pop,
        target_region=tgt_reg,
        target_population=tgt_pop,
        receptor_type=receptor,
    )


def _make_weights(n_post: int, n_pre: int, *, seed: int, sparsity: float = 0.5) -> nn.Parameter:
    """Create a sparse weight matrix as nn.Parameter."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.rand(n_post, n_pre, generator=gen) * 0.1
    mask = torch.rand(n_post, n_pre, generator=gen) > sparsity
    w = w * mask.float()
    return nn.Parameter(w, requires_grad=False)


class FakeNeuron:
    """Minimal neuron mock for NeuronIndexRegistry."""
    def __init__(self, n: int, is_conductance_lif: bool = True, is_two_compartment: bool = False):
        self.n_neurons = n
        self._is_clif = is_conductance_lif
        self._is_tc = is_two_compartment


class FakeRegion:
    """Minimal region mock for GlobalSparseMatrix and NeuronIndexRegistry."""
    def __init__(
        self,
        name: str,
        populations: dict[str, int],
        weights: dict[SynapseId, nn.Parameter],
    ):
        self.name = name
        self.neuron_populations = {
            pop: FakeNeuron(n) for pop, n in populations.items()
        }
        self._weights = weights

    @property
    def synaptic_weights(self):
        return self._weights

    @property
    def device(self):
        return torch.device("cpu")


class FakeBatch:
    """Minimal ConductanceLIFBatch mock."""
    def __init__(self, registry: dict):
        self.registry = registry
        self.total_neurons = max((end for _, end in registry.values()), default=0)
        total = self.total_neurons
        self.g_ampa_input = torch.zeros(total)
        self.g_nmda_input = torch.zeros(total)
        self.g_gaba_a_input = torch.zeros(total)
        self.g_gaba_b_input = torch.zeros(total)

    def is_batched(self, key):
        return key in self.registry

    def clear_inputs(self):
        self.g_ampa_input.zero_()
        self.g_nmda_input.zero_()
        self.g_gaba_a_input.zero_()
        self.g_gaba_b_input.zero_()


def _build_simple_test_setup():
    """Build a minimal 2-region setup with known weights.

    Region A (source): pop_x (10 neurons)
    Region B (target): pop_y (8 neurons)
    Connection: A.pop_x → B.pop_y (AMPA, inter-region)

    Also adds an intra-region connection B.pop_y → B.pop_y that should be
    filtered out by the sparse matrix.
    """
    inter_sid = _sid("region_a", "pop_x", "region_b", "pop_y", ReceptorType.AMPA)
    intra_sid = _sid("region_b", "pop_y", "region_b", "pop_y", ReceptorType.AMPA)

    inter_weights = _make_weights(8, 10, seed=42, sparsity=0.3)
    intra_weights = _make_weights(8, 8, seed=99, sparsity=0.5)

    region_a = FakeRegion("region_a", {"pop_x": 10}, {})
    region_b = FakeRegion("region_b", {"pop_y": 8}, {
        inter_sid: inter_weights,
        intra_sid: intra_weights,
    })

    regions = {"region_a": region_a, "region_b": region_b}

    # Build batch registry (all target pops)
    batch_registry = {("region_b", "pop_y"): (0, 8)}
    batch = FakeBatch(batch_registry)

    # Build neuron index (reuse batch registry for target)
    neuron_index = MagicMock(spec=NeuronIndexRegistry)
    neuron_index.target_registry = {("region_b", "pop_y"): (0, 8)}
    neuron_index.total_target_neurons = 8

    def is_eligible(key):
        return key in neuron_index.target_registry

    def get_target_slice(key):
        return neuron_index.target_registry[key]

    neuron_index.is_eligible_target = is_eligible
    neuron_index.get_target_slice = get_target_slice

    return regions, batch, neuron_index, inter_sid, intra_sid, inter_weights, intra_weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCSRConstruction:
    """Test sparse matrix construction correctness."""

    def test_inter_region_included_intra_filtered(self):
        """Only inter-region connections should be in the sparse matrix."""
        regions, batch, neuron_index, inter_sid, intra_sid, _, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        assert sparse.has_connection(inter_sid), "Inter-region connection should be in sparse matrix"
        assert not sparse.has_connection(intra_sid), "Intra-region connection should NOT be in sparse matrix"

    def test_nnz_matches_original(self):
        """The number of nonzero entries should match the original weight matrix."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        expected_nnz = (inter_weights.data != 0).sum().item()
        meta = sparse.get_connection_meta(inter_sid)
        assert meta.nnz == expected_nnz

    def test_csr_matrix_shape(self):
        """CSR matrix should have correct dimensions."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        W = sparse.W[ReceptorType.AMPA]
        assert W is not None
        assert W.shape[0] == 8   # n_target
        assert W.shape[1] == 10  # n_source (one connection, 10 pre-neurons)


class TestWeightRoundtrip:
    """Test reading and writing weights through the sparse matrix."""

    def test_get_dense_weights_matches_original(self):
        """get_dense_weights should reconstruct the original weight matrix."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        reconstructed = sparse.get_dense_weights(inter_sid)
        torch.testing.assert_close(reconstructed, inter_weights.data)

    def test_weight_values_roundtrip(self):
        """get/set weight values should roundtrip correctly."""
        regions, batch, neuron_index, inter_sid, _, _, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        original_values = sparse.get_weight_values(inter_sid).clone()
        # Modify weights
        new_values = original_values * 2.0
        sparse.set_weight_values(inter_sid, new_values)

        # Read back
        read_back = sparse.get_weight_values(inter_sid)
        torch.testing.assert_close(read_back, new_values)

    def test_set_values_reflected_in_dense(self):
        """Weight changes via set_weight_values should appear in get_dense_weights."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        # Double all weights
        values = sparse.get_weight_values(inter_sid)
        sparse.set_weight_values(inter_sid, values * 2.0)

        # Reconstruct dense and verify
        dense = sparse.get_dense_weights(inter_sid)
        expected = inter_weights.data * 2.0
        torch.testing.assert_close(dense, expected)


class TestSparseMatmul:
    """Test that sparse matmul produces correct conductances."""

    def test_matmul_matches_dense(self):
        """Sparse matmul should produce the same result as dense matmul."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        # Create test spikes
        gen = torch.Generator().manual_seed(123)
        spikes = (torch.rand(10, generator=gen) > 0.5).float()

        # Dense matmul reference
        expected = (inter_weights.data @ spikes).clamp(min=0.0)

        # Sparse matmul via integrate
        region_inputs = {
            "region_b": {inter_sid: spikes.bool()},
        }
        sparse.integrate(region_inputs, None, {})

        result = sparse.g_out[ReceptorType.AMPA][:8]
        torch.testing.assert_close(result, expected)

    def test_matmul_with_stp_efficacy(self):
        """STP efficacy should scale spikes before matmul."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        # Spikes and STP efficacy
        spikes = torch.ones(10, dtype=torch.bool)
        efficacy = torch.full((10,), 0.5)
        stp = {inter_sid: efficacy}

        # Dense reference: W @ (efficacy * spikes)
        expected = (inter_weights.data @ (efficacy * spikes.float())).clamp(min=0.0)

        # Sparse
        region_inputs = {"region_b": {inter_sid: spikes}}
        sparse.integrate(region_inputs, None, stp)

        result = sparse.g_out[ReceptorType.AMPA][:8]
        torch.testing.assert_close(result, expected)

    def test_zero_spikes_gives_zero_conductance(self):
        """No spikes should produce zero conductance."""
        regions, batch, neuron_index, inter_sid, _, _, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        spikes = torch.zeros(10, dtype=torch.bool)
        region_inputs = {"region_b": {inter_sid: spikes}}
        sparse.integrate(region_inputs, None, {})

        result = sparse.g_out[ReceptorType.AMPA][:8]
        assert result.sum().item() == 0.0


class TestRebuildValueSlices:
    """Test that _rebuild_value_slices correctly maps CSR entries."""

    def test_local_indices_in_range(self):
        """Row and column indices should be within valid ranges."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))
        meta = sparse.get_connection_meta(inter_sid)

        n_post, n_pre = inter_weights.shape
        assert meta.local_row_indices.max().item() < n_post
        assert meta.local_col_indices.max().item() < n_pre
        assert meta.local_row_indices.min().item() >= 0
        assert meta.local_col_indices.min().item() >= 0

    def test_csr_indices_match_values(self):
        """_csr_indices should correctly address the CSR values array."""
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))
        meta = sparse.get_connection_meta(inter_sid)

        # The values at _csr_indices should match the original nonzero values
        W_csr = sparse.W[ReceptorType.AMPA]
        assert W_csr is not None

        csr_values = W_csr.values()
        indexed_values = csr_values[meta._csr_indices]

        # Compare against original nonzero values
        original_nz = inter_weights.data[meta.local_row_indices, meta.local_col_indices]
        torch.testing.assert_close(indexed_values, original_nz)


class TestMultipleConnections:
    """Test with multiple connections to the same target."""

    def _build_multi_connection_setup(self):
        """Two source regions → one target region."""
        sid_a = _sid("region_a", "pop_x", "region_c", "pop_z", ReceptorType.AMPA)
        sid_b = _sid("region_b", "pop_y", "region_c", "pop_z", ReceptorType.AMPA)

        w_a = _make_weights(6, 10, seed=1, sparsity=0.4)
        w_b = _make_weights(6, 8, seed=2, sparsity=0.4)

        region_a = FakeRegion("region_a", {"pop_x": 10}, {})
        region_b = FakeRegion("region_b", {"pop_y": 8}, {})
        region_c = FakeRegion("region_c", {"pop_z": 6}, {
            sid_a: w_a,
            sid_b: w_b,
        })

        regions = {"region_a": region_a, "region_b": region_b, "region_c": region_c}

        batch_registry = {("region_c", "pop_z"): (0, 6)}
        batch = FakeBatch(batch_registry)

        neuron_index = MagicMock(spec=NeuronIndexRegistry)
        neuron_index.target_registry = {("region_c", "pop_z"): (0, 6)}
        neuron_index.total_target_neurons = 6
        neuron_index.is_eligible_target = lambda k: k in neuron_index.target_registry
        neuron_index.get_target_slice = lambda k: neuron_index.target_registry[k]

        return regions, batch, neuron_index, sid_a, sid_b, w_a, w_b

    def test_both_connections_present(self):
        regions, batch, ni, sid_a, sid_b, _, _ = self._build_multi_connection_setup()
        sparse = GlobalSparseMatrix(regions, ni, batch, torch.device("cpu"))
        assert sparse.has_connection(sid_a)
        assert sparse.has_connection(sid_b)

    def test_dense_reconstruction_both(self):
        regions, batch, ni, sid_a, sid_b, w_a, w_b = self._build_multi_connection_setup()
        sparse = GlobalSparseMatrix(regions, ni, batch, torch.device("cpu"))
        torch.testing.assert_close(sparse.get_dense_weights(sid_a), w_a.data)
        torch.testing.assert_close(sparse.get_dense_weights(sid_b), w_b.data)

    def test_matmul_combines_both_sources(self):
        """Conductances from multiple sources should sum correctly."""
        regions, batch, ni, sid_a, sid_b, w_a, w_b = self._build_multi_connection_setup()
        sparse = GlobalSparseMatrix(regions, ni, batch, torch.device("cpu"))

        spikes_a = torch.ones(10, dtype=torch.bool)
        spikes_b = torch.ones(8, dtype=torch.bool)

        expected = (
            (w_a.data @ spikes_a.float()) + (w_b.data @ spikes_b.float())
        ).clamp(min=0.0)

        region_inputs = {
            "region_c": {
                sid_a: spikes_a,
                sid_b: spikes_b,
            },
        }
        sparse.integrate(region_inputs, None, {})

        result = sparse.g_out[ReceptorType.AMPA][:6]
        torch.testing.assert_close(result, expected)


class TestMultipleReceptorTypes:
    """Test connections across different receptor types."""

    def test_different_receptors_independent(self):
        """AMPA and GABA_A connections should produce independent conductances."""
        sid_ampa = _sid("region_a", "pop_x", "region_b", "pop_y", ReceptorType.AMPA)
        sid_gaba = _sid("region_a", "pop_x", "region_b", "pop_y", ReceptorType.GABA_A)

        w_ampa = _make_weights(5, 7, seed=10, sparsity=0.3)
        w_gaba = _make_weights(5, 7, seed=20, sparsity=0.3)

        region_a = FakeRegion("region_a", {"pop_x": 7}, {})
        region_b = FakeRegion("region_b", {"pop_y": 5}, {
            sid_ampa: w_ampa,
            sid_gaba: w_gaba,
        })

        regions = {"region_a": region_a, "region_b": region_b}
        batch_registry = {("region_b", "pop_y"): (0, 5)}
        batch = FakeBatch(batch_registry)

        neuron_index = MagicMock(spec=NeuronIndexRegistry)
        neuron_index.target_registry = {("region_b", "pop_y"): (0, 5)}
        neuron_index.total_target_neurons = 5
        neuron_index.is_eligible_target = lambda k: k in neuron_index.target_registry
        neuron_index.get_target_slice = lambda k: neuron_index.target_registry[k]

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        spikes = torch.ones(7, dtype=torch.bool)
        region_inputs = {"region_b": {sid_ampa: spikes, sid_gaba: spikes}}
        sparse.integrate(region_inputs, None, {})

        expected_ampa = (w_ampa.data @ spikes.float()).clamp(min=0.0)
        expected_gaba = (w_gaba.data @ spikes.float()).clamp(min=0.0)

        torch.testing.assert_close(sparse.g_out[ReceptorType.AMPA][:5], expected_ampa)
        torch.testing.assert_close(sparse.g_out[ReceptorType.GABA_A][:5], expected_gaba)
        # Other receptor types should be zero
        assert sparse.g_out[ReceptorType.NMDA][:5].sum().item() == 0.0
        assert sparse.g_out[ReceptorType.GABA_B][:5].sum().item() == 0.0


class TestScatterToBatch:
    """Test scatter_to_neuron_batch writes to the correct batch positions."""

    def test_scatter_writes_correct_positions(self):
        regions, batch, neuron_index, inter_sid, _, inter_weights, _ = _build_simple_test_setup()

        sparse = GlobalSparseMatrix(regions, neuron_index, batch, torch.device("cpu"))

        spikes = torch.ones(10, dtype=torch.bool)
        region_inputs = {"region_b": {inter_sid: spikes}}
        sparse.integrate(region_inputs, None, {})
        sparse.scatter_to_neuron_batch()

        expected_ampa = (inter_weights.data @ spikes.float()).clamp(min=0.0)
        torch.testing.assert_close(batch.g_ampa_input[:8], expected_ampa)
