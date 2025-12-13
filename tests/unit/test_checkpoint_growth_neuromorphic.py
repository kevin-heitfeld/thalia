"""
Unit tests for neuromorphic checkpoint format (Phase 2).

Tests the neuron-centric approach where neurons have persistent IDs
and synapses are stored explicitly rather than as weight matrices.

Test Coverage:
- Neuron ID persistence across growth
- Loading with missing neurons (brain shrunk)
- Loading with extra neurons (brain grew)
- Synapse restoration by ID matching
- Partial checkpoint loading
- Neuron metadata tracking
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import torch

from thalia.regions.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig


@pytest.fixture
def device():
    """Return device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def base_config(device):
    """Create base striatum config with neuromorphic format."""
    return StriatumConfig(
        n_actions=5,
        n_input=100,
        growth_enabled=True,
        checkpoint_format="neuromorphic",  # Use neuron-centric format
        device=device,
    )


@pytest.fixture
def striatum_neuromorphic(base_config):
    """Create striatum with neuromorphic checkpoint format."""
    region = Striatum(base_config)
    region.reset()
    return region


class TestNeuronIDPersistence:
    """Test that neurons have stable IDs across growth."""

    def test_neurons_have_unique_ids(self, striatum_neuromorphic):
        """Each neuron should have a unique persistent ID."""
        state = striatum_neuromorphic.get_full_state()

        assert state["format"] == "neuromorphic"
        neurons = state["neurons"]

        # Should have one entry per neuron
        assert len(neurons) == 5

        # Each should have unique ID
        ids = [n["id"] for n in neurons]
        assert len(ids) == len(set(ids)), "Neuron IDs not unique!"

        # IDs should follow naming convention
        for neuron_id in ids:
            assert neuron_id.startswith("striatum_d1_neuron_") or \
                   neuron_id.startswith("striatum_d2_neuron_")

    def test_ids_persist_across_resets(self, striatum_neuromorphic):
        """Neuron IDs should persist across reset() calls."""
        # Get initial IDs
        state1 = striatum_neuromorphic.get_full_state()
        ids1 = [n["id"] for n in state1["neurons"]]

        # Reset
        striatum_neuromorphic.reset()

        # Get IDs again
        state2 = striatum_neuromorphic.get_full_state()
        ids2 = [n["id"] for n in state2["neurons"]]

        # IDs should match
        assert ids1 == ids2

    def test_new_neurons_get_new_ids(self, striatum_neuromorphic):
        """Growing brain should assign new IDs to new neurons."""
        # Get initial IDs
        state1 = striatum_neuromorphic.get_full_state()
        ids1 = set(n["id"] for n in state1["neurons"])

        # Grow
        striatum_neuromorphic.add_neurons(n_new=3)

        # Get new IDs
        state2 = striatum_neuromorphic.get_full_state()
        ids2 = set(n["id"] for n in state2["neurons"])

        # Original IDs should still exist
        assert ids1.issubset(ids2)

        # Should have 3 new IDs
        new_ids = ids2 - ids1
        assert len(new_ids) == 3

    def test_id_format_includes_creation_step(self, striatum_neuromorphic):
        """Neuron IDs should encode when they were created."""
        # Initial neurons created at step 0
        state = striatum_neuromorphic.get_full_state()

        for neuron in state["neurons"]:
            assert "created_step" in neuron
            assert neuron["created_step"] == 0

        # Advance to step 1000
        striatum_neuromorphic._current_step = 1000

        # Add neurons
        striatum_neuromorphic.add_neurons(n_new=2)

        state2 = striatum_neuromorphic.get_full_state()

        # New neurons should have created_step=1000
        neurons_by_step = {}
        for neuron in state2["neurons"]:
            step = neuron["created_step"]
            if step not in neurons_by_step:
                neurons_by_step[step] = []
            neurons_by_step[step].append(neuron)

        assert 0 in neurons_by_step
        assert 1000 in neurons_by_step
        assert len(neurons_by_step[0]) == 5  # Original
        assert len(neurons_by_step[1000]) == 2  # New


class TestLoadingWithMissingNeurons:
    """Test loading checkpoint into brain that has fewer neurons."""

    def test_load_with_missing_neurons_warns(self, striatum_neuromorphic, tmp_path):
        """Loading checkpoint with extra neurons should warn but not fail."""
        checkpoint_path = tmp_path / "extra_neurons.ckpt"

        # Create checkpoint with 8 neurons
        striatum_neuromorphic.add_neurons(n_new=3)
        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Create new brain with only 5 neurons
        small_brain = Striatum(striatum_neuromorphic.config)
        small_brain.reset()

        # Load should warn about missing neurons
        loaded = torch.load(checkpoint_path)

        with pytest.warns(UserWarning, match="Checkpoint has 8 neurons but brain has 5"):
            small_brain.load_full_state(loaded)

    def test_partial_neuron_restore(self, striatum_neuromorphic, tmp_path):
        """Should restore only neurons that exist in both checkpoint and brain."""
        checkpoint_path = tmp_path / "partial.ckpt"

        # Set state in 8-neuron brain
        striatum_neuromorphic.add_neurons(n_new=3)
        for i, neuron in enumerate(striatum_neuromorphic.neurons):
            neuron.membrane = float(i) * 0.1

        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Load into 5-neuron brain
        small_brain = Striatum(striatum_neuromorphic.config)
        small_brain.reset()

        loaded = torch.load(checkpoint_path)

        with pytest.warns(UserWarning):
            small_brain.load_full_state(loaded)

        # First 5 neurons should match checkpoint
        for i in range(5):
            expected = float(i) * 0.1
            actual = small_brain.neurons[i].membrane
            assert abs(actual - expected) < 1e-6

    def test_missing_neurons_tracked_in_log(self, striatum_neuromorphic, tmp_path, caplog):
        """Missing neurons should be logged for debugging."""
        checkpoint_path = tmp_path / "logged.ckpt"

        # Create and save 8-neuron brain
        striatum_neuromorphic.add_neurons(n_new=3)
        state = striatum_neuromorphic.get_full_state()
        checkpoint_ids = [n["id"] for n in state["neurons"]]
        torch.save(state, checkpoint_path)

        # Load into 5-neuron brain
        small_brain = Striatum(striatum_neuromorphic.config)
        small_brain.reset()

        loaded = torch.load(checkpoint_path)

        import logging
        with caplog.at_level(logging.DEBUG):
            with pytest.warns(UserWarning):
                small_brain.load_full_state(loaded)

        # Should log which neurons were skipped
        for neuron_id in checkpoint_ids[5:]:
            assert neuron_id in caplog.text


class TestLoadingWithExtraNeurons:
    """Test loading checkpoint into brain that has more neurons."""

    def test_load_preserves_new_neurons(self, striatum_neuromorphic, tmp_path):
        """Loading smaller checkpoint should preserve new neurons' state."""
        checkpoint_path = tmp_path / "small_checkpoint.ckpt"

        # Save 5-neuron state
        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Create 8-neuron brain
        large_brain = Striatum(striatum_neuromorphic.config)
        large_brain.reset()
        large_brain.add_neurons(n_new=3)

        # Set distinctive state in new neurons
        for i in range(5, 8):
            large_brain.neurons[i].membrane = 0.999

        # Load checkpoint
        loaded = torch.load(checkpoint_path)
        large_brain.load_full_state(loaded)

        # New neurons should keep their state (not overwritten)
        for i in range(5, 8):
            assert abs(large_brain.neurons[i].membrane - 0.999) < 1e-6

    def test_new_neurons_not_in_checkpoint_logged(self, striatum_neuromorphic, tmp_path, caplog):
        """Should log when brain has neurons not in checkpoint."""
        checkpoint_path = tmp_path / "log_extra.ckpt"

        # Save small
        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Load into large
        large_brain = Striatum(striatum_neuromorphic.config)
        large_brain.reset()
        large_brain.add_neurons(n_new=3)

        loaded = torch.load(checkpoint_path)

        import logging
        with caplog.at_level(logging.DEBUG):
            large_brain.load_full_state(loaded)

        # Should log that some neurons not in checkpoint
        assert "neurons not in checkpoint" in caplog.text.lower()


class TestSynapseRestoration:
    """Test synapse weight restoration by ID matching."""

    def test_synapses_stored_with_source_target_ids(self, striatum_neuromorphic):
        """Synapses should be stored with source and target neuron IDs."""
        state = striatum_neuromorphic.get_full_state()

        neurons = state["neurons"]

        for neuron in neurons:
            assert "incoming_synapses" in neuron

            for synapse in neuron["incoming_synapses"]:
                assert "from" in synapse
                assert "weight" in synapse
                assert "eligibility" in synapse

                # Source should be a valid ID string
                assert isinstance(synapse["from"], str)
                assert len(synapse["from"]) > 0

    def test_synapse_restoration_by_id(self, striatum_neuromorphic, tmp_path):
        """Synapses should be restored by matching source/target IDs."""
        checkpoint_path = tmp_path / "synapses.ckpt"

        # Set some weights
        striatum_neuromorphic.d1_pathway.weights[0, 10] = 0.8
        striatum_neuromorphic.d1_pathway.weights[0, 11] = 0.9

        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Reset and load
        striatum_neuromorphic.reset()

        loaded = torch.load(checkpoint_path)
        striatum_neuromorphic.load_full_state(loaded)

        # Weights should be restored
        assert abs(striatum_neuromorphic.d1_pathway.weights[0, 10].item() - 0.8) < 1e-6
        assert abs(striatum_neuromorphic.d1_pathway.weights[0, 11].item() - 0.9) < 1e-6

    def test_orphaned_synapses_handled(self, striatum_neuromorphic, tmp_path):
        """Synapses pointing to deleted neurons should be skipped."""
        checkpoint_path = tmp_path / "orphaned.ckpt"

        # Create checkpoint with 8 neurons
        striatum_neuromorphic.add_neurons(n_new=3)
        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Load into brain with only 5 neurons
        small_brain = Striatum(striatum_neuromorphic.config)
        small_brain.reset()

        loaded = torch.load(checkpoint_path)

        # Should handle orphaned synapses gracefully
        with pytest.warns(UserWarning):
            small_brain.load_full_state(loaded)

        # Should not crash, just skip orphaned connections


class TestPartialCheckpointLoading:
    """Test selectively loading subsets of neurons."""

    def test_load_only_d1_neurons(self, striatum_neuromorphic, tmp_path):
        """Should be able to load only D1 pathway neurons."""
        checkpoint_path = tmp_path / "partial_d1.ckpt"

        # Set state
        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Reset
        striatum_neuromorphic.reset()

        # Load only D1 neurons
        loaded = torch.load(checkpoint_path)
        striatum_neuromorphic.load_full_state(loaded, neuron_filter=lambda n: "d1" in n["id"])

        # D1 neurons should be restored, D2 should be reset

    def test_load_neurons_created_after_step(self, striatum_neuromorphic, tmp_path):
        """Should be able to load only neurons created after certain step."""
        checkpoint_path = tmp_path / "filtered_by_step.ckpt"

        # Create neurons at different steps
        striatum_neuromorphic._current_step = 0
        striatum_neuromorphic.reset()

        striatum_neuromorphic._current_step = 1000
        striatum_neuromorphic.add_neurons(n_new=3)

        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        # Reset and load only new neurons
        striatum_neuromorphic.reset()

        loaded = torch.load(checkpoint_path)
        striatum_neuromorphic.load_full_state(
            loaded,
            neuron_filter=lambda n: n["created_step"] >= 1000
        )


class TestNeuronMetadata:
    """Test neuron metadata tracking."""

    def test_neuron_type_stored(self, striatum_neuromorphic):
        """Each neuron should store its type (D1-MSN, D2-MSN, etc)."""
        state = striatum_neuromorphic.get_full_state()

        for neuron in state["neurons"]:
            assert "type" in neuron
            assert neuron["type"] in ["D1-MSN", "D2-MSN"]

    def test_neuron_location_metadata(self, striatum_neuromorphic):
        """Neurons should store their anatomical location."""
        state = striatum_neuromorphic.get_full_state()

        for neuron in state["neurons"]:
            assert "region" in neuron
            assert neuron["region"] == "striatum"

    def test_neuron_growth_history(self, striatum_neuromorphic):
        """Neurons should track their growth history."""
        # Initial neurons
        state1 = striatum_neuromorphic.get_full_state()

        for neuron in state1["neurons"]:
            assert neuron["created_step"] == 0
            assert "parent_id" not in neuron  # No parent (initial neurons)

        # Grow via splitting (if supported)
        if hasattr(striatum_neuromorphic, "split_neuron"):
            striatum_neuromorphic.split_neuron(neuron_id=0)

            state2 = striatum_neuromorphic.get_full_state()

            # New neuron should reference parent
            new_neurons = [n for n in state2["neurons"] if n["created_step"] > 0]
            assert len(new_neurons) > 0

            for new_neuron in new_neurons:
                if "parent_id" in new_neuron:
                    # Should reference valid parent
                    parent_id = new_neuron["parent_id"]
                    parent_exists = any(n["id"] == parent_id for n in state2["neurons"])
                    assert parent_exists


class TestNeuromorphicPerformance:
    """Test performance characteristics of neuromorphic format."""

    def test_load_time_scales_with_neurons_not_synapses(self, base_config, tmp_path):
        """Load time should depend on neuron count, not synapse count."""
        checkpoint_path = tmp_path / "perf_neuromorphic.ckpt"

        # Create region with many neurons but sparse connectivity
        config = base_config
        config.n_actions = 100
        config.sparsity = 0.01  # Only 1% connectivity

        region = Striatum(config)
        region.reset()

        state = region.get_full_state()
        torch.save(state, checkpoint_path)

        # Load should be fast despite many potential synapses
        import time
        start = time.perf_counter()
        loaded = torch.load(checkpoint_path)
        region.load_full_state(loaded)
        elapsed = time.perf_counter() - start

        # Should be <500ms for 100 neurons
        assert elapsed < 0.5, f"Load took {elapsed:.3f}s, too slow!"

    def test_checkpoint_size_scales_with_connectivity(self, base_config, tmp_path):
        """Checkpoint size should scale with actual connections, not capacity."""
        # Dense network
        config_dense = base_config
        config_dense.n_actions = 50
        config_dense.sparsity = 0.5  # 50% connectivity

        region_dense = Striatum(config_dense)
        region_dense.reset()

        state_dense = region_dense.get_full_state()

        # Sparse network (same size)
        config_sparse = base_config
        config_sparse.n_actions = 50
        config_sparse.sparsity = 0.05  # 5% connectivity

        region_sparse = Striatum(config_sparse)
        region_sparse.reset()

        state_sparse = region_sparse.get_full_state()

        # Sparse should have ~10x fewer synapses
        dense_synapses = sum(len(n["incoming_synapses"]) for n in state_dense["neurons"])
        sparse_synapses = sum(len(n["incoming_synapses"]) for n in state_sparse["neurons"])

        assert sparse_synapses < dense_synapses * 0.2  # At least 5x fewer


class TestNeuromorphicInspection:
    """Test debugging/inspection capabilities of neuromorphic format."""

    def test_can_inspect_individual_neurons(self, striatum_neuromorphic, tmp_path):
        """Should be able to examine individual neurons in checkpoint."""
        checkpoint_path = tmp_path / "inspect.ckpt"

        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path)

        # Can access neurons by ID
        neurons_by_id = {n["id"]: n for n in loaded["neurons"]}

        # Can inspect specific neuron
        first_neuron_id = loaded["neurons"][0]["id"]
        first_neuron = neurons_by_id[first_neuron_id]

        assert "membrane" in first_neuron
        assert "incoming_synapses" in first_neuron
        assert "type" in first_neuron

    def test_can_analyze_connectivity_patterns(self, striatum_neuromorphic, tmp_path):
        """Should be able to analyze connectivity from checkpoint."""
        checkpoint_path = tmp_path / "connectivity.ckpt"

        state = striatum_neuromorphic.get_full_state()
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path)

        # Build connectivity graph
        connectivity = {}
        for neuron in loaded["neurons"]:
            neuron_id = neuron["id"]
            sources = [s["from"] for s in neuron["incoming_synapses"]]
            connectivity[neuron_id] = sources

        # Can analyze patterns
        total_connections = sum(len(sources) for sources in connectivity.values())
        avg_fanin = total_connections / len(connectivity)

        assert avg_fanin > 0  # Should have some connections
