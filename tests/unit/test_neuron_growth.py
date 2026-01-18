"""Tests for ConductanceLIF neuron growth functionality.

Tests that neuron populations can be grown dynamically while preserving
existing neuron state - critical for developmental/curriculum learning.
"""

import pytest
import torch

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


class TestNeuronGrowth:
    """Test suite for ConductanceLIF.grow_neurons() method."""

    @pytest.mark.parametrize(
        "initial_n,growth_amount",
        [
            (50, 10),
            (100, 20),
            (200, 50),
            (500, 100),
        ],
    )
    def test_grow_neurons_various_sizes(self, initial_n, growth_amount):
        """Test growth works with various population sizes."""
        config = ConductanceLIFConfig()
        neurons = ConductanceLIF(n_neurons=initial_n, config=config)
        neurons.reset_state()

        neurons.grow_neurons(growth_amount)

        expected_n = initial_n + growth_amount
        assert neurons.n_neurons == expected_n
        assert neurons.v_threshold.shape[0] == expected_n
        assert neurons.membrane.shape[0] == expected_n

    def test_grow_neurons_preserves_state(self):
        """Test that growth preserves existing neuron state."""
        config = ConductanceLIFConfig()
        initial_n = 50
        growth_amount = 30
        neurons = ConductanceLIF(n_neurons=initial_n, config=config)
        neurons.reset_state()

        # Simulate some activity
        g_exc = torch.rand(initial_n) * 0.5
        _ = neurons(g_exc)  # Simulate forward pass

        # Save old state
        old_membrane = neurons.membrane[:initial_n].clone()
        old_g_E = neurons.g_E[:initial_n].clone()
        old_g_I = neurons.g_I[:initial_n].clone()
        old_refractory = neurons.refractory[:initial_n].clone()

        # Grow population
        neurons.grow_neurons(growth_amount)

        # Check that old neuron state is preserved
        assert neurons.n_neurons == initial_n + growth_amount
        torch.testing.assert_close(neurons.membrane[:initial_n], old_membrane)
        torch.testing.assert_close(neurons.g_E[:initial_n], old_g_E)
        torch.testing.assert_close(neurons.g_I[:initial_n], old_g_I)
        torch.testing.assert_close(neurons.refractory[:initial_n], old_refractory)

    def test_grow_neurons_initializes_new_neurons(self):
        """Test that new neurons start at resting potential."""
        config = ConductanceLIFConfig(E_L=0.0)
        initial_n = 50
        growth_amount = 30
        neurons = ConductanceLIF(n_neurons=initial_n, config=config)
        neurons.reset_state()

        # Grow population
        neurons.grow_neurons(growth_amount)

        # New neurons should start at resting potential (E_L)
        assert neurons.membrane[initial_n:].allclose(torch.full((growth_amount,), config.E_L))
        assert neurons.g_E[initial_n:].allclose(torch.zeros(growth_amount))
        assert neurons.g_I[initial_n:].allclose(torch.zeros(growth_amount))
        assert neurons.g_adapt[initial_n:].allclose(torch.zeros(growth_amount))
        assert neurons.refractory[initial_n:].allclose(
            torch.zeros(growth_amount, dtype=torch.int32)
        )

    def test_grow_neurons_functional_after_growth(self):
        """Test that neurons remain functional after growth."""
        config = ConductanceLIFConfig()
        initial_n = 100
        growth_amount = 50
        neurons = ConductanceLIF(n_neurons=initial_n, config=config)
        neurons.reset_state()

        # Grow population
        neurons.grow_neurons(growth_amount)

        # Should be able to process input for all neurons
        total_n = initial_n + growth_amount
        g_exc = torch.rand(total_n) * 0.3
        spikes, voltage = neurons(g_exc)

        assert spikes.shape[0] == total_n
        assert voltage.shape[0] == total_n
        assert spikes.dtype == torch.bool

    def test_grow_neurons_zero_growth(self):
        """Test that zero growth is a no-op."""
        config = ConductanceLIFConfig()
        neurons = ConductanceLIF(n_neurons=100, config=config)
        neurons.reset_state()

        old_n = neurons.n_neurons
        neurons.grow_neurons(0)

        assert neurons.n_neurons == old_n

    def test_grow_neurons_multiple_times(self):
        """Test that neurons can be grown multiple times."""
        config = ConductanceLIFConfig()
        initial_n = 50
        first_growth = 25
        second_growth = 25
        neurons = ConductanceLIF(n_neurons=initial_n, config=config)
        neurons.reset_state()

        # First growth
        neurons.grow_neurons(first_growth)
        after_first = initial_n + first_growth
        assert neurons.n_neurons == after_first

        # Simulate activity
        g_exc = torch.rand(after_first) * 0.3
        _ = neurons(g_exc)  # Forward pass

        # Second growth
        neurons.grow_neurons(second_growth)
        total_n = after_first + second_growth
        assert neurons.n_neurons == total_n

        # Should still work
        g_exc = torch.rand(total_n) * 0.3
        spikes, voltage = neurons(g_exc)
        assert spikes.shape[0] == total_n

    def test_grow_neurons_preserves_thresholds(self):
        """Test that per-neuron thresholds are properly expanded."""
        config = ConductanceLIFConfig(v_threshold=1.0)
        initial_n = 50
        growth_amount = 30
        neurons = ConductanceLIF(n_neurons=initial_n, config=config)

        # Modify some thresholds for testing
        neurons.v_threshold[:10] = 0.8
        old_thresholds = neurons.v_threshold.clone()

        # Grow
        neurons.grow_neurons(growth_amount)

        # Old thresholds preserved
        torch.testing.assert_close(neurons.v_threshold[:initial_n], old_thresholds)

        # New thresholds use default
        assert neurons.v_threshold[initial_n:].allclose(
            torch.full((growth_amount,), config.v_threshold)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_grow_neurons_device_consistency(self):
        """Test that growth respects device placement."""
        config = ConductanceLIFConfig()
        initial_n = 50
        growth_amount = 30
        device = torch.device("cuda")
        neurons = ConductanceLIF(n_neurons=initial_n, config=config, device=device)
        neurons.reset_state()

        # Grow
        neurons.grow_neurons(growth_amount)

        # All tensors should be on CUDA
        assert neurons.membrane.device.type == "cuda"
        assert neurons.g_E.device.type == "cuda"
        assert neurons.v_threshold.device.type == "cuda"

    def test_grow_neurons_before_reset(self):
        """Test growing neurons before reset_state() is called."""
        config = ConductanceLIFConfig()
        initial_n = 50
        growth_amount = 30
        neurons = ConductanceLIF(n_neurons=initial_n, config=config)

        # Grow before reset (state is None)
        neurons.grow_neurons(growth_amount)

        total_n = initial_n + growth_amount
        assert neurons.n_neurons == total_n
        assert neurons.membrane is None  # State not initialized yet

        # Reset should create state for all neurons
        neurons.reset_state()
        assert neurons.membrane.shape[0] == total_n


class TestGrowthIntegrationWithRegions:
    """Test that growth works properly in region context."""

    def test_prefrontal_uses_grow_neurons(self):
        """Test that Prefrontal region uses neuron growth properly."""
        from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig

        initial_n = 64
        growth_amount = 32
        device = "cpu"

        # Create sizes dict (Phase 2 pattern)
        sizes = {
            "input_size": 128,
            "n_neurons": initial_n,
        }

        # Create config with behavioral parameters only
        pfc_config = PrefrontalConfig()
        pfc = Prefrontal(config=pfc_config, sizes=sizes, device=device)
        assert pfc.neurons.n_neurons == initial_n

        # Grow region
        pfc.grow_neurons(n_new=growth_amount)

        # Neurons should be grown, not recreated
        total_n = initial_n + growth_amount
        assert pfc.neurons.n_neurons == total_n
        assert pfc.n_neurons == total_n  # Use instance attribute, not config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
