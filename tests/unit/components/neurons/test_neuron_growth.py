"""Tests for ConductanceLIF neuron growth functionality.

Tests that neuron populations can be grown dynamically while preserving
existing neuron state - critical for developmental/curriculum learning.
"""

import torch
import pytest

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


class TestNeuronGrowth:
    """Test suite for ConductanceLIF.grow_neurons() method."""

    def test_grow_neurons_basic(self):
        """Test basic neuron growth increases population size."""
        config = ConductanceLIFConfig()
        neurons = ConductanceLIF(n_neurons=100, config=config)
        neurons.reset_state()

        # Grow by 20 neurons
        neurons.grow_neurons(20)

        assert neurons.n_neurons == 120
        assert neurons.v_threshold.shape[0] == 120

    def test_grow_neurons_preserves_state(self):
        """Test that growth preserves existing neuron state."""
        config = ConductanceLIFConfig()
        neurons = ConductanceLIF(n_neurons=50, config=config)
        neurons.reset_state()

        # Simulate some activity
        g_exc = torch.rand(50) * 0.5
        _ = neurons(g_exc)  # Simulate forward pass

        # Save old state
        old_membrane = neurons.membrane[:50].clone()
        old_g_E = neurons.g_E[:50].clone()
        old_g_I = neurons.g_I[:50].clone()
        old_refractory = neurons.refractory[:50].clone()

        # Grow population
        neurons.grow_neurons(30)

        # Check that old neuron state is preserved
        assert neurons.n_neurons == 80
        torch.testing.assert_close(neurons.membrane[:50], old_membrane)
        torch.testing.assert_close(neurons.g_E[:50], old_g_E)
        torch.testing.assert_close(neurons.g_I[:50], old_g_I)
        torch.testing.assert_close(neurons.refractory[:50], old_refractory)

    def test_grow_neurons_initializes_new_neurons(self):
        """Test that new neurons start at resting potential."""
        config = ConductanceLIFConfig(E_L=0.0)
        neurons = ConductanceLIF(n_neurons=50, config=config)
        neurons.reset_state()

        # Grow population
        neurons.grow_neurons(30)

        # New neurons should start at resting potential (E_L)
        assert neurons.membrane[50:].allclose(torch.full((30,), config.E_L))
        assert neurons.g_E[50:].allclose(torch.zeros(30))
        assert neurons.g_I[50:].allclose(torch.zeros(30))
        assert neurons.g_adapt[50:].allclose(torch.zeros(30))
        assert neurons.refractory[50:].allclose(torch.zeros(30, dtype=torch.int32))

    def test_grow_neurons_functional_after_growth(self):
        """Test that neurons remain functional after growth."""
        config = ConductanceLIFConfig()
        neurons = ConductanceLIF(n_neurons=100, config=config)
        neurons.reset_state()

        # Grow population
        neurons.grow_neurons(50)

        # Should be able to process input for all 150 neurons
        g_exc = torch.rand(150) * 0.3
        spikes, voltage = neurons(g_exc)

        assert spikes.shape[0] == 150
        assert voltage.shape[0] == 150
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
        neurons = ConductanceLIF(n_neurons=50, config=config)
        neurons.reset_state()

        # First growth
        neurons.grow_neurons(25)
        assert neurons.n_neurons == 75

        # Simulate activity
        g_exc = torch.rand(75) * 0.3
        _ = neurons(g_exc)  # Forward pass

        # Second growth
        neurons.grow_neurons(25)
        assert neurons.n_neurons == 100

        # Should still work
        g_exc = torch.rand(100) * 0.3
        spikes, voltage = neurons(g_exc)
        assert spikes.shape[0] == 100

    def test_grow_neurons_preserves_thresholds(self):
        """Test that per-neuron thresholds are properly expanded."""
        config = ConductanceLIFConfig(v_threshold=1.0)
        neurons = ConductanceLIF(n_neurons=50, config=config)

        # Modify some thresholds for testing
        neurons.v_threshold[:10] = 0.8
        old_thresholds = neurons.v_threshold.clone()

        # Grow
        neurons.grow_neurons(30)

        # Old thresholds preserved
        torch.testing.assert_close(neurons.v_threshold[:50], old_thresholds)

        # New thresholds use default
        assert neurons.v_threshold[50:].allclose(torch.full((30,), config.v_threshold))

    def test_grow_neurons_device_consistency(self):
        """Test that growth respects device placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = ConductanceLIFConfig()
        device = torch.device("cuda")
        neurons = ConductanceLIF(n_neurons=50, config=config)
        neurons.to(device)
        neurons.reset_state()

        # Grow
        neurons.grow_neurons(30)

        # All tensors should be on CUDA
        assert neurons.membrane.device.type == "cuda"
        assert neurons.g_E.device.type == "cuda"
        assert neurons.v_threshold.device.type == "cuda"

    def test_grow_neurons_before_reset(self):
        """Test growing neurons before reset_state() is called."""
        config = ConductanceLIFConfig()
        neurons = ConductanceLIF(n_neurons=50, config=config)

        # Grow before reset (state is None)
        neurons.grow_neurons(30)

        assert neurons.n_neurons == 80
        assert neurons.membrane is None  # State not initialized yet

        # Reset should create state for all neurons
        neurons.reset_state()
        assert neurons.membrane.shape[0] == 80


class TestGrowthIntegrationWithRegions:
    """Test that growth works properly in region context."""

    def test_prefrontal_uses_grow_neurons(self):
        """Test that Prefrontal region uses neuron growth properly."""
        from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig

        pfc_config = PrefrontalConfig(n_output=64, n_input=128)
        pfc = Prefrontal(pfc_config)
        assert pfc.neurons.n_neurons == 64

        # Grow region
        pfc.grow_output(n_new=32)

        # Neurons should be grown, not recreated
        assert pfc.neurons.n_neurons == 96
        assert pfc.config.n_output == 96


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
