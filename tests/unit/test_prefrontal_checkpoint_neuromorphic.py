"""Test neuromorphic checkpoint format for prefrontal cortex."""

import pytest
import torch

from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.prefrontal_checkpoint_manager import PrefrontalCheckpointManager


@pytest.fixture
def device():
    """Device for testing."""
    return "cpu"


@pytest.fixture
def small_prefrontal(device):
    """Create small prefrontal for testing neuromorphic format."""
    config = PrefrontalConfig(
        n_input=8,
        n_output=50,  # Small enough to trigger neuromorphic (<200 threshold)
        device=device,
        dt_ms=1.0,
    )
    return Prefrontal(config)


class TestPrefrontalNeuromorphic:
    """Test neuromorphic checkpoint format extraction and restoration."""

    def test_get_neuromorphic_state_structure(self, small_prefrontal):
        """Verify neuromorphic state has correct structure."""
        manager = PrefrontalCheckpointManager(small_prefrontal)
        state = manager.get_neuromorphic_state()

        # Check top-level keys
        assert "neurons" in state
        assert "learning_state" in state
        assert "neuromodulator_state" in state
        assert "region_state" in state
        assert state["format"] == "neuromorphic"

        # Check neurons structure
        neurons = state["neurons"]
        assert isinstance(neurons, list)
        assert len(neurons) == 50  # n_output

        # Each neuron should have required fields
        for neuron in neurons:
            assert "id" in neuron
            assert "region" in neuron
            assert neuron["region"] == "prefrontal"
            assert "type" in neuron  # rule_neuron or wm_neuron
            assert neuron["type"] in ["rule_neuron", "wm_neuron"]
            assert "incoming_synapses" in neuron

            # State fields at top level
            assert "membrane" in neuron
            assert "working_memory" in neuron
            assert "update_gate" in neuron
            assert "created_step" in neuron

    def test_neurons_have_stable_ids(self, small_prefrontal):
        """Neuron IDs should be stable across multiple extractions."""
        manager = PrefrontalCheckpointManager(small_prefrontal)

        state1 = manager.get_neuromorphic_state()
        state2 = manager.get_neuromorphic_state()

        ids1 = [n["id"] for n in state1["neurons"]]
        ids2 = [n["id"] for n in state2["neurons"]]

        assert ids1 == ids2, "Neuron IDs should be stable"
        assert len(set(ids1)) == len(ids1), "IDs should be unique"

    def test_synapses_stored_by_type(self, small_prefrontal):
        """Synapses should be organized by type (feedforward/recurrent/inhibitory)."""
        manager = PrefrontalCheckpointManager(small_prefrontal)
        state = manager.get_neuromorphic_state()

        # Check each neuron has incoming_synapses with type tags
        for neuron in state["neurons"]:
            synapses = neuron["incoming_synapses"]
            assert isinstance(synapses, list)
            assert len(synapses) > 0  # Should have at least feedforward

            # Check synapse structure
            synapse_types = set()
            for syn in synapses:
                assert "from" in syn
                assert "type" in syn
                assert "weight" in syn
                assert isinstance(syn["weight"], float)
                synapse_types.add(syn["type"])

            # Should have at least feedforward synapses
            assert "feedforward" in synapse_types

    def test_working_memory_preserved(self, small_prefrontal):
        """Working memory state should be preserved per-neuron."""
        # Set some working memory values
        small_prefrontal.state.working_memory = torch.randn(50)
        small_prefrontal.state.update_gate = torch.rand(50)

        manager = PrefrontalCheckpointManager(small_prefrontal)
        state = manager.get_neuromorphic_state()

        # Check each neuron has WM and gate values
        for i, neuron in enumerate(state["neurons"]):
            wm_val = neuron["working_memory"]
            gate_val = neuron["update_gate"]

            # Should match the region's state for this neuron
            expected_wm = small_prefrontal.state.working_memory[i].item()
            expected_gate = small_prefrontal.state.update_gate[i].item()

            assert abs(wm_val - expected_wm) < 1e-6
            assert abs(gate_val - expected_gate) < 1e-6

    def test_save_load_preserves_state(self, small_prefrontal, tmp_path):
        """Save and load should preserve neuromorphic state."""
        # Set distinctive state
        small_prefrontal.weights.data.fill_(0.5)
        small_prefrontal.state.working_memory = torch.arange(50, dtype=torch.float32) * 0.1

        manager = PrefrontalCheckpointManager(small_prefrontal)
        checkpoint_path = tmp_path / "prefrontal_checkpoint.pt"

        # Save
        manager.save(str(checkpoint_path))

        # Modify state
        small_prefrontal.weights.data.fill_(0.9)
        small_prefrontal.state.working_memory.fill_(999.0)

        # Load
        manager.load(str(checkpoint_path))

        # Verify restoration
        assert torch.allclose(small_prefrontal.weights.data, torch.full_like(small_prefrontal.weights.data, 0.5))
        expected_wm = torch.arange(50, dtype=torch.float32) * 0.1
        assert torch.allclose(small_prefrontal.state.working_memory, expected_wm, atol=1e-5)


class TestPrefrontalHybrid:
    """Test hybrid format selection and metadata."""

    def test_small_region_uses_neuromorphic(self, small_prefrontal):
        """Small prefrontal (<200 neurons) should auto-select neuromorphic."""
        manager = PrefrontalCheckpointManager(small_prefrontal)
        assert manager._should_use_neuromorphic() is True

    def test_growth_enabled_uses_neuromorphic(self, device):
        """Prefrontal with growth (has add_neurons) should use neuromorphic."""
        config = PrefrontalConfig(
            n_input=8,
            n_output=300,  # Large, but has growth capability
            device=device,
            dt_ms=1.0,
        )
        prefrontal = Prefrontal(config)
        manager = PrefrontalCheckpointManager(prefrontal)
        # Prefrontal inherits add_neurons from GrowthMixin, so always neuromorphic
        assert manager._should_use_neuromorphic() is True

    def test_hybrid_metadata_included(self, small_prefrontal, tmp_path):
        """Saved checkpoint should include hybrid metadata."""
        manager = PrefrontalCheckpointManager(small_prefrontal)
        checkpoint_path = tmp_path / "prefrontal_hybrid.pt"

        manager.save(str(checkpoint_path))

        # Load raw checkpoint
        checkpoint = torch.load(checkpoint_path)

        assert "hybrid_metadata" in checkpoint
        metadata = checkpoint["hybrid_metadata"]
        assert metadata["selected_format"] == "neuromorphic"  # Small region
        assert metadata["selection_criteria"]["n_neurons"] == 50
        assert metadata["auto_selected"] is True

    def test_load_rejects_checkpoint_without_metadata(self, small_prefrontal, tmp_path):
        """Load should reject checkpoint without hybrid_metadata."""
        checkpoint_path = tmp_path / "invalid_checkpoint.pt"

        # Save checkpoint without metadata
        invalid_checkpoint = {
            "weights": small_prefrontal.weights.data,
            "state": {"working_memory": small_prefrontal.state.working_memory},
        }
        torch.save(invalid_checkpoint, checkpoint_path)

        manager = PrefrontalCheckpointManager(small_prefrontal)

        with pytest.raises(ValueError, match="hybrid_metadata"):
            manager.load(str(checkpoint_path))
