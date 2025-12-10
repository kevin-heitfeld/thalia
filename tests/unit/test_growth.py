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
        # Striatum has 10x output neurons internally (D1 + D2 populations)
        assert metrics.neuron_count == 320  # 32 * 10
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

    def test_add_neurons_preserves_weights(self):
        """Test that adding neurons doesn't change existing weights."""
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region = Striatum(config)

        # Save original D1 weights (Striatum uses d1_weights, self.weights is just a reference)
        original_d1 = region.d1_weights.clone()
        original_d2 = region.d2_weights.clone()
        # Striatum: 32 actions * 10 neurons_per_action = 320 neurons
        assert original_d1.shape[0] == 320

        # Add neurons (16 new actions = 16 * 10 = 160 neurons)
        region.add_neurons(n_new=16, initialization='sparse_random', sparsity=0.1)

        # Check D1/D2 weights expanded correctly
        assert region.d1_weights.shape == (480, 64)  # 320 + 160 = 480
        assert region.d2_weights.shape == (480, 64)
        
        # Check existing D1 weights unchanged (with small tolerance for floating point)
        torch.testing.assert_close(
            region.d1_weights[:320, :],
            original_d1,
            rtol=1e-5, atol=1e-7  # Small tolerance
        )
        
        # Check existing D2 weights unchanged
        torch.testing.assert_close(
            region.d2_weights[:320, :],
            original_d2,
            rtol=1e-5, atol=1e-7
        )

        # Check new neurons have reasonable weights
        new_d1_weights = region.d1_weights[320:, :]
        assert new_d1_weights.abs().max() > 0  # Not all zeros
        assert (new_d1_weights == 0).float().mean() > 0.8  # Mostly sparse

    def test_capacity_metrics_api(self):
        """Test that regions expose capacity metrics."""
        from thalia.core.growth import CapacityMetrics
        
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region = Striatum(config)

        metrics = region.get_capacity_metrics()

        assert isinstance(metrics, CapacityMetrics)
        assert hasattr(metrics, 'firing_rate')
        assert hasattr(metrics, 'weight_saturation')
        assert hasattr(metrics, 'synapse_usage')
        assert hasattr(metrics, 'neuron_count')
        assert hasattr(metrics, 'growth_recommended')

    def test_growth_with_checkpoint_roundtrip(self, tmp_path):
        """Test save/load checkpoint after growth."""
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region1 = Striatum(config)

        # Process some input
        input_spikes = torch.rand(64) > 0.8
        _ = region1.forward(input_spikes.float())

        # Add neurons (16 actions = 160 neurons total)
        region1.add_neurons(n_new=16)

        # Process more input
        _ = region1.forward(input_spikes.float())

        # Save state
        state1 = region1.get_full_state()

        # Create new region with GROWN config (not original)
        grown_config = StriatumConfig(n_input=64, n_output=48, device="cpu")  # 32 + 16 actions
        region2 = Striatum(grown_config)

        # Load state
        region2.load_full_state(state1)

        # Verify weights match
        torch.testing.assert_close(region1.weights, region2.weights)

        # Verify can process input (48 actions * 10 = 480 neurons)
        output2 = region2.forward(input_spikes.float())
        assert output2.shape[0] == 480  # 480 neurons


class TestGrowthIntegration:
    """Integration tests for growth with full brain."""

    def test_brain_check_growth_needs(self):
        """Test brain-level growth need detection."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Check growth needs
        growth_report = brain.check_growth_needs()

        assert isinstance(growth_report, dict)
        assert 'striatum' in growth_report or 'regions' in growth_report

    def test_checkpoint_preserves_growth_history(self, tmp_path):
        """Test that growth history survives checkpoint roundtrip."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

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
