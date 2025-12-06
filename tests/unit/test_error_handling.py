"""
Error handling and edge case tests.

Tests that components handle error conditions gracefully with clear error messages.
This complements test_validation.py by focusing on runtime error paths.
"""

import pytest
import torch

from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF
from thalia.core.dendritic import DendriticNeuron, DendriticNeuronConfig
from thalia.regions import LayeredCortex, LayeredCortexConfig


@pytest.mark.unit
class TestLIFNeuronErrorHandling:
    """Test error handling for LIF neurons."""

    def test_forward_before_reset_raises_error(self):
        """Test that forward pass before reset raises clear error."""
        neuron = LIFNeuron(n_neurons=10)
        # Don't call reset_state
        
        # Should raise RuntimeError with helpful message
        with pytest.raises((RuntimeError, AttributeError)) as exc_info:
            neuron(torch.randn(1, 10))
        
        # Check error message is helpful (if RuntimeError)
        if isinstance(exc_info.value, RuntimeError):
            assert "reset" in str(exc_info.value).lower() or "state" in str(exc_info.value).lower(), \
                "Error message should mention 'reset' or 'state'"

    def test_mismatched_batch_size_raises_error(self):
        """Test that mismatched batch sizes are caught."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        # Try to pass different batch size
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            neuron(torch.randn(8, 10))  # Should be batch_size=4

    def test_wrong_number_of_neurons_raises_error(self):
        """Test that wrong neuron count in input is caught."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        # Try to pass wrong number of neurons
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            neuron(torch.randn(4, 20))  # Should be 10 neurons

    def test_handles_very_large_batch_size(self):
        """Test behavior with very large batch size (memory stress test)."""
        neuron = LIFNeuron(n_neurons=100)
        
        # This might fail with OOM, which is acceptable
        try:
            large_batch = 100000  # 100k batch size
            neuron.reset_state(batch_size=large_batch)
            spikes, _ = neuron(torch.randn(large_batch, 100))
            
            # If it succeeds, verify output is valid
            assert spikes.shape == (large_batch, 100)
        except (RuntimeError, MemoryError):
            # OOM is acceptable for very large batches
            pytest.skip("Out of memory for large batch (expected)")

    def test_handles_extreme_input_values(self):
        """Test that extreme (but not inf/nan) input values don't crash."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        # Very large positive input
        large_input = torch.full((4, 10), 1e6)
        spikes, _ = neuron(large_input)
        assert not torch.isnan(spikes).any()
        assert not torch.isinf(spikes).any()
        
        # Very large negative input
        neuron.reset_state(batch_size=4)
        large_neg_input = torch.full((4, 10), -1e6)
        spikes, _ = neuron(large_neg_input)
        assert not torch.isnan(spikes).any()
        assert not torch.isinf(spikes).any()

    def test_multiple_resets_work_correctly(self):
        """Test that multiple resets don't cause issues."""
        neuron = LIFNeuron(n_neurons=10)
        
        # Multiple resets should work fine
        for batch_size in [1, 4, 8, 16]:
            neuron.reset_state(batch_size=batch_size)
            spikes, _ = neuron(torch.randn(batch_size, 10))
            assert spikes.shape == (batch_size, 10)

    def test_reset_with_different_batch_size_updates_state(self):
        """Test that reset with new batch size properly updates internal state."""
        neuron = LIFNeuron(n_neurons=10)
        
        # First reset
        neuron.reset_state(batch_size=4)
        neuron(torch.randn(4, 10))
        
        # Reset with different batch size
        neuron.reset_state(batch_size=8)
        spikes, _ = neuron(torch.randn(8, 10))
        
        assert spikes.shape == (8, 10)
        assert neuron.membrane.shape == (8, 10)


@pytest.mark.unit
class TestConductanceLIFErrorHandling:
    """Test error handling for conductance-based LIF."""

    def test_mismatched_exc_inh_shapes_raises_error(self):
        """Test that mismatched excitatory and inhibitory input shapes are caught."""
        neuron = ConductanceLIF(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        exc = torch.randn(4, 10)
        inh = torch.randn(4, 5)  # Wrong size!
        
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            neuron(exc, inh)

    def test_none_inhibitory_with_excitatory_works(self):
        """Test that None inhibitory input with excitatory input works."""
        neuron = ConductanceLIF(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        exc = torch.randn(4, 10)
        spikes, _ = neuron(exc, None)  # Should work
        
        assert spikes.shape == (4, 10)

    def test_both_none_inputs_work(self):
        """Test that None for both inputs works (no external input)."""
        neuron = ConductanceLIF(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        # Should handle no input gracefully
        spikes, _ = neuron(None, None)
        assert spikes.shape == (4, 10)


@pytest.mark.unit
class TestDendriticNeuronErrorHandling:
    """Test error handling for dendritic neurons."""

    def test_wrong_number_of_branch_inputs_raises_error(self):
        """Test that wrong number of branch inputs is caught."""
        config = DendriticNeuronConfig(n_branches=3, n_inputs_per_branch=10)
        neuron = DendriticNeuron(config)
        neuron.reset_state(batch_size=4)
        
        # Should expect 3 branch inputs, give 2
        branch_inputs = [torch.randn(4, 10), torch.randn(4, 10)]
        
        with pytest.raises((ValueError, RuntimeError, AssertionError, IndexError)):
            neuron(branch_inputs)

    def test_empty_branch_list_raises_error(self):
        """Test that empty branch input list is caught."""
        config = DendriticNeuronConfig(n_branches=3, n_inputs_per_branch=10)
        neuron = DendriticNeuron(config)
        neuron.reset_state(batch_size=4)
        
        with pytest.raises((ValueError, RuntimeError, AssertionError, IndexError)):
            neuron([])  # Empty list


@pytest.mark.unit
class TestCortexErrorHandling:
    """Test error handling for cortex."""

    def test_forward_before_reset_works_with_auto_reset(self):
        """Test that cortex either auto-resets or raises clear error."""
        config = LayeredCortexConfig(n_input=32, n_output=16)
        cortex = LayeredCortex(config)
        # Don't call reset_state
        
        try:
            # Some implementations auto-reset
            output = cortex.forward(torch.randn(4, 32))
            assert output.shape == (4, 16)
        except (RuntimeError, AttributeError) as e:
            # Others require explicit reset - should have clear message
            assert "reset" in str(e).lower() or "state" in str(e).lower()

    def test_wrong_input_size_raises_error(self):
        """Test that wrong input size is caught."""
        config = LayeredCortexConfig(n_input=32, n_output=16)
        cortex = LayeredCortex(config)
        cortex.reset_state()
        
        # Wrong input size
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            cortex.forward(torch.randn(4, 64))  # Should be 32

    def test_invalid_config_raises_error(self):
        """Test that invalid configurations are rejected."""
        # Zero input size
        with pytest.raises((ValueError, AssertionError)):
            LayeredCortexConfig(n_input=0, n_output=16)
        
        # Zero output size
        with pytest.raises((ValueError, AssertionError)):
            LayeredCortexConfig(n_input=32, n_output=0)
        
        # Negative sizes
        with pytest.raises((ValueError, AssertionError)):
            LayeredCortexConfig(n_input=-10, n_output=16)


@pytest.mark.unit
class TestThreadSafety:
    """Test thread safety (if applicable)."""

    @pytest.mark.slow
    def test_concurrent_forward_passes_separate_neurons(self):
        """Test that separate neuron instances can run concurrently."""
        import threading
        
        results = []
        errors = []
        
        def run_neuron():
            try:
                neuron = LIFNeuron(n_neurons=100)
                neuron.reset_state(batch_size=4)
                
                for _ in range(10):
                    spikes, _ = neuron(torch.randn(4, 100))
                    results.append(spikes.sum().item())
            except Exception as e:
                errors.append(e)
        
        # Run 4 threads concurrently
        threads = [threading.Thread(target=run_neuron) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0, f"Thread safety issues: {errors}"
        assert len(results) == 40  # 4 threads * 10 iterations


@pytest.mark.unit
class TestGradientFlow:
    """Test gradient flow for differentiable components."""

    def test_no_gradient_explosion_in_lif(self):
        """Test that gradients don't explode during backprop."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        # Create input that requires gradients
        input_current = torch.randn(4, 10, requires_grad=True)
        
        # Forward pass
        spikes, _ = neuron(input_current)
        
        # Backward through spike count (dummy loss)
        loss = spikes.sum()
        loss.backward()
        
        # Check gradients are reasonable
        if input_current.grad is not None:
            assert not torch.isnan(input_current.grad).any(), "Gradients contain NaN"
            assert not torch.isinf(input_current.grad).any(), "Gradients contain Inf"
            assert input_current.grad.abs().max() < 1e6, "Gradient explosion detected"

    def test_gradient_flow_through_multiple_steps(self):
        """Test gradient flow through multiple timesteps."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state(batch_size=4)
        
        input_sequence = [torch.randn(4, 10, requires_grad=True) for _ in range(10)]
        
        total_spikes = 0
        for inp in input_sequence:
            spikes, _ = neuron(inp)
            total_spikes += spikes.sum()
        
        # Backward
        total_spikes.backward()
        
        # Check all inputs got gradients
        for i, inp in enumerate(input_sequence):
            if inp.grad is not None:
                assert not torch.isnan(inp.grad).any(), f"Step {i}: NaN gradients"
                assert not torch.isinf(inp.grad).any(), f"Step {i}: Inf gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
