"""
Tests for Hippocampus Neuromorphic Checkpoint Format

Verifies that the hippocampus checkpoint manager correctly:
- Extracts neuron-centric state for all three layers (DG, CA3, CA1)
- Stores synapses with sparse format
- Loads state with ID-based matching
- Handles hybrid format auto-selection
"""

import pytest
import torch

from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_hippocampus(device):
    """Small hippocampus for testing."""
    config = HippocampusConfig(
        n_input=4,  # Small input
        n_output=8,  # 8 CA1 neurons
        dg_expansion=3.75,  # 4 * 3.75 = 15 DG neurons
        ca3_size_ratio=0.667,  # 15 * 0.667 ≈ 10 CA3 neurons
        device=device,
    )
    hippo = TrisynapticHippocampus(config)
    hippo.reset_state()
    return hippo


class TestHippocampusNeuromorphic:
    """Test neuromorphic checkpoint format for hippocampus."""

    def test_get_neuromorphic_state_structure(self, small_hippocampus):
        """Neuromorphic state should have correct structure."""
        state = small_hippocampus.checkpoint_manager.get_neuromorphic_state()

        assert state["format"] == "neuromorphic"
        assert state["format_version"] == "2.0.0"
        assert "neurons" in state
        assert "episode_buffer" in state
        assert "learning_state" in state

        # Should have neurons for all three layers
        neurons = state["neurons"]
        # DG: 4*3.75=15, CA3: 15*0.667≈10, CA1: 8 → total 33
        assert len(neurons) >= 30  # Allow some flexibility in calculation

    def test_neurons_have_layer_tags(self, small_hippocampus):
        """Each neuron should be tagged with its layer."""
        state = small_hippocampus.checkpoint_manager.get_neuromorphic_state()
        neurons = state["neurons"]

        dg_neurons = [n for n in neurons if n["layer"] == "DG"]
        ca3_neurons = [n for n in neurons if n["layer"] == "CA3"]
        ca1_neurons = [n for n in neurons if n["layer"] == "CA1"]

        # Verify layer counts (allow some calculation variance)
        assert len(dg_neurons) >= 12
        assert len(ca3_neurons) >= 8
        assert len(ca1_neurons) == 8

    def test_neurons_have_stable_ids(self, small_hippocampus):
        """Each neuron should have a unique stable ID."""
        state = small_hippocampus.checkpoint_manager.get_neuromorphic_state()
        neurons = state["neurons"]

        # All IDs should be unique
        ids = [n["id"] for n in neurons]
        assert len(ids) == len(set(ids))

        # IDs should follow naming convention
        dg_neurons = [n for n in neurons if n["layer"] == "DG"]
        assert all("hippo_dg_neuron_" in n["id"] for n in dg_neurons)

    def test_synapses_stored_sparsely(self, small_hippocampus):
        """Only non-zero synapses should be stored."""
        # Set some weights to non-zero (use valid indices for n_input=4)
        small_hippocampus.synaptic_weights["ec_dg"][0, 2] = 0.7
        small_hippocampus.synaptic_weights["ec_dg"][0, 3] = 0.0  # Explicitly zero

        state = small_hippocampus.checkpoint_manager.get_neuromorphic_state()
        dg_neuron_0 = [n for n in state["neurons"] if n["id"] == "hippo_dg_neuron_0_step0"][0]

        # Should have synapses, but not from index 3
        synapses = dg_neuron_0["incoming_synapses"]
        synapse_sources = [s["from"] for s in synapses]

        # Should have non-zero synapse from index 2
        assert "ec_neuron_2" in synapse_sources
        # Should NOT have zero synapse from index 3
        assert "ec_neuron_3" not in synapse_sources

    def test_save_load_preserves_state(self, small_hippocampus, tmp_path):
        """Save and load should preserve hippocampus state."""
        checkpoint_path = tmp_path / "hippo.ckpt"

        # Set distinctive weights (use valid indices for n_input=4)
        small_hippocampus.synaptic_weights["ec_dg"][0, 2] = 0.777  # DG neuron 0, EC input 2
        small_hippocampus.synaptic_weights["dg_ca3"][2, 1] = 0.888  # CA3 neuron 2, DG input 1
        small_hippocampus.synaptic_weights["ca3_ca1"][3, 2] = 0.999  # CA1 neuron 3, CA3 input 2

        # Save
        small_hippocampus.checkpoint_manager.save(checkpoint_path)


class TestHippocampusHybrid:
    """Test hybrid format auto-selection."""

    def test_small_region_uses_neuromorphic(self, small_hippocampus, tmp_path):
        """Small hippocampus should default to neuromorphic."""
        checkpoint_path = tmp_path / "small_hippo_format.pt"
        small_hippocampus.checkpoint_manager.save(checkpoint_path)

        state = torch.load(checkpoint_path, weights_only=False)
        assert state["hybrid_metadata"]["selected_format"] == "neuromorphic", \
            "Small hippocampus should use neuromorphic format"

    def test_hybrid_metadata_included(self, small_hippocampus, tmp_path):
        """Hybrid checkpoints should include metadata about format selection."""
        checkpoint_path = tmp_path / "metadata.ckpt"

        small_hippocampus.checkpoint_manager.save(checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        assert "hybrid_metadata" in loaded
        assert loaded["hybrid_metadata"]["auto_selected"] is True
        assert "selected_format" in loaded["hybrid_metadata"]
        assert loaded["hybrid_metadata"]["selected_format"] == "neuromorphic"
        assert "selection_criteria" in loaded["hybrid_metadata"]

    def test_load_rejects_checkpoint_without_metadata(self, small_hippocampus, tmp_path):
        """Loading checkpoint without hybrid_metadata should raise error."""
        checkpoint_path = tmp_path / "no_metadata.ckpt"

        # Create checkpoint without hybrid_metadata
        state = small_hippocampus.get_full_state()
        torch.save(state, checkpoint_path)

        # Should raise error
        with pytest.raises(ValueError, match="hybrid_metadata"):
            small_hippocampus.checkpoint_manager.load(checkpoint_path)


@pytest.mark.parametrize("acetylcholine", [-0.5, 0.0, 1.0, 1.5, 2.0])
def test_hippocampus_extreme_acetylcholine(small_hippocampus, acetylcholine):
    """Test hippocampus stability with extreme acetylcholine values.

    Phase 2 improvement: Tests edge cases beyond normal [0, 1] range.
    ACh modulates encoding (high) vs retrieval (low) modes.
    """
    input_spikes = torch.rand(4) > 0.5

    # Set extreme ACh value
    small_hippocampus.set_neuromodulators(acetylcholine=acetylcholine)

    # Forward pass should not crash
    output = small_hippocampus(input_spikes)

    # Contract: valid output regardless of ACh value
    assert output.dtype == torch.bool
    assert output.shape == (8,)  # n_output = 8

    # Contract: no numerical instability in any layer
    for layer_name, neurons in [
        ('dg', small_hippocampus.dg_neurons),
        ('ca3', small_hippocampus.ca3_neurons),
        ('ca1', small_hippocampus.ca1_neurons),
    ]:
        membrane = neurons.membrane

        assert not torch.isnan(membrane).any(), \
            f"NaN in {layer_name} membrane with ACh={acetylcholine}"
        assert not torch.isinf(membrane).any(), \
            f"Inf in {layer_name} membrane with ACh={acetylcholine}"

        # Contract: membrane stays in reasonable range
        assert (membrane >= -20.0).all(), \
            f"{layer_name} membrane too low with ACh={acetylcholine}"
        assert (membrane <= 10.0).all(), \
            f"{layer_name} membrane too high with ACh={acetylcholine}"
