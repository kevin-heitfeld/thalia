"""Tests for growth mechanisms.

Tests neuron addition, capacity metrics, and checkpoint compatibility
with growth operations.
"""

import pytest
import torch

from thalia.core.growth import GrowthManager, CapacityMetrics, GrowthEvent
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.io import BrainCheckpoint


class TestGrowthManager:
    """Test GrowthManager functionality."""

    def test_growth_manager_initialization(self):
        """Test basic initialization."""
        manager = GrowthManager(region_name="test_region")
        assert manager.region_name == "test_region"
        assert len(manager.history) == 0

    def test_capacity_metrics_computation(self):
        """Test capacity metrics calculation."""
        # Create simple striatum
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region = Striatum(config)

        manager = GrowthManager(region_name="striatum")
        metrics = manager.get_capacity_metrics(region)

        assert isinstance(metrics, CapacityMetrics)
        assert 0 <= metrics.firing_rate <= 1
        assert 0 <= metrics.weight_saturation <= 1
        assert 0 <= metrics.synapse_usage <= 1
        assert metrics.neuron_count == 32
        assert metrics.synapse_count > 0

    def test_growth_history_tracking(self):
        """Test that growth events are recorded."""
        manager = GrowthManager(region_name="test")

        # Create mock event
        event = GrowthEvent(
            timestamp="2025-12-07T10:00:00",
            component_name="test",
            component_type="region",
            event_type="add_neurons",
            n_neurons_added=10,
            reason="test growth"
        )
        manager.history.append(event)

        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["n_neurons_added"] == 10

    def test_state_serialization(self):
        """Test growth manager state save/load."""
        manager = GrowthManager(region_name="test")

        # Add some history
        event = GrowthEvent(
            timestamp="2025-12-07T10:00:00",
            component_name="test",
            component_type="region",
            event_type="add_neurons",
            n_neurons_added=10
        )
        manager.history.append(event)

        # Get state
        state = manager.get_state()
        assert state["region_name"] == "test"
        assert len(state["history"]) == 1

        # Load into new manager
        manager2 = GrowthManager(region_name="other")
        manager2.load_state(state)
        assert manager2.region_name == "test"
        assert len(manager2.history) == 1


class TestStriatumGrowth:
    """Test growth operations on Striatum region."""

    @pytest.mark.skip(reason="Striatum.add_neurons() not yet implemented")
    def test_add_neurons_preserves_weights(self):
        """Test that adding neurons doesn't change existing weights."""
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region = Striatum(config)

        # Save original weights
        original_weights = region.weights.clone()

        # Add neurons
        region.add_neurons(n_new=16, initialization='sparse_random', sparsity=0.1)

        # Check existing weights unchanged
        new_weights = region.weights
        assert new_weights.shape == (48, 64)  # 32 + 16 = 48
        torch.testing.assert_close(
            new_weights[:32, :],  # Original neurons
            original_weights,
            rtol=0, atol=0  # Exact match
        )

        # Check new neurons have reasonable weights
        new_neuron_weights = new_weights[32:, :]
        assert new_neuron_weights.abs().max() > 0  # Not all zeros
        assert (new_neuron_weights == 0).float().mean() > 0.8  # Mostly sparse

    @pytest.mark.skip(reason="Striatum.add_neurons() not yet implemented")
    def test_capacity_metrics_api(self):
        """Test that regions expose capacity metrics."""
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region = Striatum(config)

        metrics = region.get_capacity_metrics()

        assert 'firing_rate' in metrics
        assert 'weight_saturation' in metrics
        assert 'synapse_usage' in metrics
        assert 'neuron_count' in metrics
        assert 'growth_recommended' in metrics

    @pytest.mark.skip(reason="Striatum.add_neurons() not yet implemented")
    def test_growth_with_checkpoint_roundtrip(self, tmp_path):
        """Test save/load checkpoint after growth."""
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region1 = Striatum(config)

        # Process some input
        input_spikes = torch.rand(64) > 0.8
        _ = region1.forward(input_spikes.float())

        # Add neurons
        region1.add_neurons(n_new=16)

        # Process more input
        _ = region1.forward(input_spikes.float())

        # Save state
        state1 = region1.get_full_state()

        # Create new region with ORIGINAL config
        region2 = Striatum(config)

        # Load state (should handle size mismatch)
        region2.load_full_state(state1)

        # Verify weights match
        torch.testing.assert_close(region1.weights, region2.weights)

        # Verify can process input
        output2 = region2.forward(input_spikes.float())
        assert output2.shape[0] == 48  # 32 + 16


class TestGrowthIntegration:
    """Integration tests for growth with full brain."""

    @pytest.mark.skip(reason="Brain.check_growth_needs() not yet implemented")
    def test_brain_check_growth_needs(self):
        """Test brain-level growth need detection."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig

        config = EventDrivenBrainConfig(
            input_size=64,
            n_actions=4,
            device="cpu"
        )
        brain = EventDrivenBrain(config)

        # Check growth needs
        growth_report = brain.check_growth_needs()

        assert isinstance(growth_report, dict)
        assert 'striatum' in growth_report or 'regions' in growth_report

    @pytest.mark.skip(reason="Growth checkpoint integration not yet complete")
    def test_checkpoint_preserves_growth_history(self, tmp_path):
        """Test that growth history survives checkpoint roundtrip."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig

        config = EventDrivenBrainConfig(
            input_size=64,
            n_actions=4,
            device="cpu"
        )
        brain = EventDrivenBrain(config)

        # Perform growth
        # brain.auto_grow(threshold=0.5)  # Would trigger if capacity high

        # Save checkpoint
        checkpoint_path = tmp_path / "growth_test.thalia"
        BrainCheckpoint.save(brain, checkpoint_path)

        # Load checkpoint
        state = BrainCheckpoint.load(checkpoint_path, device="cpu")

        # Check growth history in metadata
        assert 'growth_history' in state['metadata']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
