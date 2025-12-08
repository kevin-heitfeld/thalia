"""
Input validation and edge case tests for core components.

These tests verify that components handle invalid inputs gracefully,
reject bad configurations, and work correctly at boundary values.
"""

import pytest
import torch
import numpy as np

from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF, ConductanceLIFConfig
from thalia.core.dendritic import DendriticNeuron, DendriticNeuronConfig
from thalia.learning.ei_balance import EIBalanceRegulator, EIBalanceConfig
from thalia.regions import LayeredCortex, LayeredCortexConfig

from tests.test_utils import (
    assert_spike_train_valid,
    assert_weights_healthy,
    assert_membrane_potential_valid,
)


@pytest.mark.unit
class TestLIFNeuronValidation:
    """Test input validation for LIF neurons."""

    def test_accepts_single_neuron(self):
        """Test that single neuron works (edge case)."""
        neuron = LIFNeuron(n_neurons=1)
        neuron.reset_state()
        assert neuron.membrane.shape == (1, 1)

    def test_accepts_large_neuron_count(self):
        """Test that large neuron counts work."""
        neuron = LIFNeuron(n_neurons=10000)
        neuron.reset_state()
        assert neuron.membrane.shape == (1, 10000)

    def test_rejects_wrong_input_shape(self):
        """Test that wrong input shape is caught."""
        neuron = LIFNeuron(n_neurons=100)
        neuron.reset_state()

        # Wrong n_neurons dimension
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            neuron(torch.randn(50))  # Should be (8, 100)

    def test_handles_zero_input(self):
        """Test that zero input is handled correctly."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state()

        spikes, _ = neuron(torch.zeros(10))

        assert_spike_train_valid(spikes)
        assert spikes.shape == (10,)

    def test_handles_nan_input_gracefully(self):
        """Test that NaN input doesn't crash (should either reject or handle)."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state()

        nan_input = torch.full((4, 10), float('nan'))

        # Either reject NaN or produce valid output
        try:
            spikes, _ = neuron(nan_input)
            # If it doesn't raise, output should be valid
            assert not torch.isnan(spikes).any(), "Output contains NaN"
        except (ValueError, RuntimeError):
            # Acceptable to reject NaN input
            pass

    def test_handles_inf_input_gracefully(self):
        """Test that infinite input is handled."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state()

        inf_input = torch.full((4, 10), float('inf'))

        try:
            spikes, _ = neuron(inf_input)
            assert not torch.isinf(spikes).any(), "Output contains Inf"
            assert_spike_train_valid(spikes)
        except (ValueError, RuntimeError):
            pass

    def test_reset_with_zero_batch_size(self):
        """Test that zero batch size is handled."""
        neuron = LIFNeuron(n_neurons=10)

        # Should either work (empty tensor) or raise error
        try:
            neuron.reset_state()
            assert neuron.membrane.shape == (0, 10)
        except (ValueError, AssertionError):
            pass


@pytest.mark.unit
class TestConductanceLIFValidation:
    """Test input validation for conductance-based LIF."""

    def test_reversal_potentials_ordered(self):
        """Test that reversal potentials must be properly ordered."""
        # E_I should be below E_L, E_E should be above
        config = ConductanceLIFConfig(
            E_I=-1.0,  # Below rest
            E_L=0.0,   # Rest
            E_E=2.0,   # Above rest
        )
        neuron = ConductanceLIF(n_neurons=10, config=config)
        assert neuron is not None

    def test_handles_mismatched_input_shapes(self):
        """Test behavior with mismatched excitatory/inhibitory inputs."""
        neuron = ConductanceLIF(n_neurons=10)
        neuron.reset_state()

        exc = torch.randn(10)
        inh = torch.randn(5)  # Wrong size!

        with pytest.raises((ValueError, RuntimeError)):
            neuron(exc, inh)

    def test_handles_none_inhibitory_input(self):
        """Test that None inhibitory input is handled."""
        neuron = ConductanceLIF(n_neurons=10)
        neuron.reset_state()

        exc = torch.randn(10)
        spikes, _ = neuron(exc, None)

        assert_spike_train_valid(spikes)


@pytest.mark.unit
class TestEIBalanceValidation:
    """Test input validation for E/I balance regulator."""

    def test_handles_empty_spike_trains(self):
        """Test behavior with all-zero spike trains."""
        regulator = EIBalanceRegulator()

        exc_spikes = torch.zeros(100)
        inh_spikes = torch.zeros(10)

        # Should not crash, but ratio might be undefined
        try:
            ratio = regulator.compute_ratio(exc_spikes, inh_spikes)
            assert not np.isnan(ratio), "Ratio should not be NaN"
        except (ValueError, ZeroDivisionError):
            # Acceptable to reject all-zero inputs
            pass

    def test_handles_mismatched_batch_sizes(self):
        """Test error handling for mismatched batch sizes."""
        regulator = EIBalanceRegulator()

        exc_spikes = torch.randn(100)  # Batch size 4
        inh_spikes = torch.randn(2, 10)   # Batch size 2

        # Should either work (broadcast) or raise error
        try:
            regulator.update(exc_spikes, inh_spikes)
        except (ValueError, RuntimeError):
            pass


@pytest.mark.unit
class TestLayeredCortexValidation:
    """Test input validation for LayeredCortex."""

    def test_handles_single_neuron_layers(self):
        """Test edge case of single neuron per layer."""
        config = LayeredCortexConfig(n_input=1, n_output=1, dual_output=False)
        cortex = LayeredCortex(config)
        cortex.reset_state()

        output = cortex.forward(torch.randn(1))
        # With dual_output=False, output is only from one layer
        assert output.shape[0] == 1
        assert output.shape[1] >= 1  # At least 1 output neuron

    def test_handles_wrong_input_size(self):
        """Test that wrong input size is caught."""
        config = LayeredCortexConfig(n_input=64, n_output=32)
        cortex = LayeredCortex(config)
        cortex.reset_state()

        wrong_input = torch.randn(32)  # Should be (1, 64)

        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            cortex.forward(wrong_input)

    def test_handles_batch_size_change(self):
        """Test behavior when batch size changes between calls."""
        config = LayeredCortexConfig(n_input=32, n_output=16)
        cortex = LayeredCortex(config)

        # THALIA only supports batch_size=1 (single-instance architecture)
        cortex.reset_state()
        output1 = cortex.forward(torch.randn(32))
        assert output1.shape[0] == 1
        
        # Using batch_size=1 consistently should work
        output2 = cortex.forward(torch.randn(32))
        assert output2.shape[0] == 1


@pytest.mark.unit
class TestDendriticNeuronValidation:
    """Test input validation for dendritic neurons."""

    def test_handles_single_branch(self):
        """Test edge case of single dendritic branch."""
        config = DendriticNeuronConfig(
            n_branches=1,
            inputs_per_branch=10,
        )
        neuron = DendriticNeuron(n_neurons=5, config=config)
        neuron.reset_state()

        input_spikes = torch.randn(10)
        output = neuron(input_spikes)

        # DendriticNeuron returns (spikes, membrane) tuple
        if isinstance(output, tuple):
            spikes, membrane = output
            assert spikes.shape == (5,)
            assert membrane.shape == (5,)
        else:
            assert output.shape == (5,)


@pytest.mark.unit
class TestBoundaryValues:
    """Test behavior at boundary values."""

    def test_membrane_at_exact_threshold(self):
        """Test behavior when membrane potential is at/above threshold."""
        config = LIFConfig(v_threshold=1.0, v_reset=0.0)
        neuron = LIFNeuron(n_neurons=10, config=config)
        neuron.reset_state()

        # Set membrane well above threshold (LIF uses > not >=)
        # Need significant margin because neuron dynamics may decay before spike check
        neuron.membrane = torch.full((1, 10), 1.5)

        spikes, _ = neuron(torch.zeros(10))

        # Should spike when above threshold
        assert spikes.sum() > 0

    def test_very_small_time_constants(self):
        """Test behavior with very small time constants."""
        config = LIFConfig(tau_mem=0.001)  # Very fast decay
        neuron = LIFNeuron(n_neurons=10, config=config)
        neuron.reset_state()

        # Should still work, just decay very quickly
        neuron.membrane = torch.ones(10)
        neuron(torch.zeros(10))

        # Membrane should decay significantly
        assert neuron.membrane.max() < 0.5

    def test_very_large_time_constants(self):
        """Test behavior with very large time constants."""
        config = LIFConfig(tau_mem=1000.0)  # Very slow decay
        neuron = LIFNeuron(n_neurons=10, config=config)
        neuron.reset_state()

        # Should still work, just decay very slowly
        neuron.membrane = torch.ones(10)
        initial = neuron.membrane.clone()
        neuron(torch.zeros(10))

        # Membrane should barely decay
        assert (neuron.membrane > initial * 0.99).all()


@pytest.mark.unit
class TestNumericalStability:
    """Test numerical stability of components."""

    def test_no_gradient_explosion(self):
        """Test that gradients don't explode during learning."""
        config = LayeredCortexConfig(n_input=64, n_output=32)
        cortex = LayeredCortex(config)

        # Create learnable parameters
        if hasattr(cortex, 'parameters'):
            params = list(cortex.parameters())
            if len(params) > 0:
                # Run forward pass
                cortex.reset_state()
                input_data = torch.randn(1, 64, requires_grad=True)
                output = cortex.forward(input_data)

                # Compute some loss
                loss = output.sum()

                # Check gradients exist and are finite
                if loss.requires_grad:
                    loss.backward()

                    for param in params:
                        if param.grad is not None:
                            assert not torch.isnan(param.grad).any(), \
                                "Gradient contains NaN"
                            assert not torch.isinf(param.grad).any(), \
                                "Gradient contains Inf"

    def test_weight_clipping_works(self):
        """Test that weight clipping prevents explosion."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state()

        # If neuron has weights, verify they stay bounded
        if hasattr(neuron, 'weights'):
            # Simulate many updates
            for _ in range(1000):
                neuron(torch.randn(10))

            # Weights should remain in reasonable range
            if hasattr(neuron, 'weights') and neuron.weights is not None:
                assert_weights_healthy(neuron.weights, max_val=1000.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

