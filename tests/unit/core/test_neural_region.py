"""Tests for NeuralRegion base class (v2.0 architecture).

Tests the new biologically accurate architecture where:
- Weights live in regions (at dendrites), not pathways (in axons)
- Regions accept Dict[str, Tensor] inputs for multi-source integration
- Each source has its own synaptic weights and learning rule
"""

import pytest
import torch
import torch.nn as nn

from thalia.core.neural_region import NeuralRegion


class TestNeuralRegionBasics:
    """Test basic NeuralRegion functionality."""

    def test_initialization(self):
        """Test region creation with default parameters."""
        n_neurons = 100
        learning_rule = "stdp"
        region = NeuralRegion(
            n_neurons=n_neurons,
            default_learning_rule=learning_rule,
            device="cpu",
        )

        assert region.n_neurons == n_neurons  # Matches specified value
        assert region.n_neurons > 0  # Positive count invariant
        assert region.default_learning_rule == learning_rule
        assert len(region.synaptic_weights) == 0  # No sources added yet
        assert len(region.plasticity_rules) == 0
        assert region.n_input == 0  # Updated when sources added

    def test_add_single_input_source(self):
        """Test adding one input source."""
        n_neurons = 100
        n_input = 128
        region = NeuralRegion(n_neurons=n_neurons, device="cpu")

        # Add thalamic input
        region.add_input_source("thalamus", n_input=n_input)

        assert "thalamus" in region.synaptic_weights
        assert region.synaptic_weights["thalamus"].shape == (n_neurons, n_input), \
            f"Weight shape should be ({n_neurons}, {n_input}), got {region.synaptic_weights['thalamus'].shape}"
        assert not torch.isnan(region.synaptic_weights["thalamus"]).any(), \
            "Weights contain NaN values"
        assert not torch.isinf(region.synaptic_weights["thalamus"]).any(), \
            "Weights contain Inf values"
        assert region.n_input == n_input
        assert "thalamus" in region.input_sources

    def test_add_multiple_input_sources(self):
        """Test multi-source integration setup."""
        n_neurons = 100
        n_thalamus = 128
        n_hippocampus = 200
        n_pfc = 64
        region = NeuralRegion(
            n_neurons=n_neurons,
            default_learning_rule="stdp",
            device="cpu",
        )

        # Add three input sources
        region.add_input_source("thalamus", n_input=n_thalamus)
        region.add_input_source("hippocampus", n_input=n_hippocampus)
        region.add_input_source("pfc", n_input=n_pfc)

        # Check all sources registered
        assert len(region.synaptic_weights) == 3
        assert region.synaptic_weights["thalamus"].shape == (n_neurons, n_thalamus), \
            f"Thalamus weight shape should be ({n_neurons}, {n_thalamus}), got {region.synaptic_weights['thalamus'].shape}"
        assert region.synaptic_weights["hippocampus"].shape == (n_neurons, n_hippocampus), \
            f"Hippocampus weight shape should be ({n_neurons}, {n_hippocampus}), got {region.synaptic_weights['hippocampus'].shape}"
        assert region.synaptic_weights["pfc"].shape == (n_neurons, n_pfc), \
            f"PFC weight shape should be ({n_neurons}, {n_pfc}), got {region.synaptic_weights['pfc'].shape}"

        # Validate all weights
        for source in ["thalamus", "hippocampus", "pfc"]:
            assert not torch.isnan(region.synaptic_weights[source]).any(), \
                f"{source} weights contain NaN values"
            assert not torch.isinf(region.synaptic_weights[source]).any(), \
                f"{source} weights contain Inf values"

        # Check total input size
        assert region.n_input == n_thalamus + n_hippocampus + n_pfc

        # Check learning rules created (using default)
        assert len(region.plasticity_rules) == 3
        assert "thalamus" in region.plasticity_rules

    def test_duplicate_source_raises_error(self):
        """Test that adding same source twice raises error."""
        region = NeuralRegion(n_neurons=100, device="cpu")
        region.add_input_source("thalamus", n_input=128)

        with pytest.raises(ValueError, match="already exists"):
            region.add_input_source("thalamus", n_input=128)

    def test_per_source_learning_rules(self):
        """Test custom learning rule per input source."""
        region = NeuralRegion(
            n_neurons=100,
            default_learning_rule="stdp",  # Default
            device="cpu",
        )

        # Thalamus uses default STDP
        region.add_input_source("thalamus", n_input=128)

        # Hippocampus overrides with BCM
        region.add_input_source("hippocampus", n_input=200, learning_rule="bcm")

        # PFC has no learning
        region.add_input_source("pfc", n_input=64, learning_rule=None)

        # Check learning rules
        assert "thalamus" in region.plasticity_rules
        assert "hippocampus" in region.plasticity_rules
        assert "pfc" not in region.plasticity_rules  # No learning


class TestNeuralRegionForward:
    """Test forward pass with multi-source inputs."""

    def test_single_source_forward(self):
        """Test forward pass with one input source."""
        region = NeuralRegion(n_neurons=50, device="cpu")
        region.add_input_source("thalamus", n_input=128, learning_rule=None)  # No learning for simplicity

        # Create input spikes
        input_spikes = torch.rand(128) > 0.9  # 10% sparsity

        # Forward pass
        output = region.forward({"thalamus": input_spikes})

        # Check output
        assert output.shape == (50,), \
            f"Output shape should be (50,), got {output.shape}"
        assert output.dtype == torch.bool or output.dtype == torch.uint8, \
            f"Output should be bool or uint8, got {output.dtype}"
        assert not torch.isnan(output.float()).any(), \
            "Output contains NaN values"
        assert 0 <= output.sum() <= 50  # Some neurons spike

    def test_multi_source_forward(self):
        """Test forward pass with multiple input sources."""
        region = NeuralRegion(n_neurons=100, device="cpu")
        region.add_input_source("thalamus", n_input=128, learning_rule=None)
        region.add_input_source("hippocampus", n_input=200, learning_rule=None)
        region.add_input_source("pfc", n_input=64, learning_rule=None)

        # Create inputs (different sparsities)
        inputs = {
            "thalamus": torch.rand(128) > 0.9,
            "hippocampus": torch.rand(200) > 0.95,
            "pfc": torch.rand(64) > 0.85,
        }

        # Forward pass
        output = region.forward(inputs)

        # Check output
        assert output.shape == (100,), \
            f"Output shape should be (100,), got {output.shape}"
        assert output.dtype in [torch.bool, torch.uint8], \
            f"Output should be bool or uint8, got {output.dtype}"
        assert not torch.isnan(output.float()).any(), \
            "Output contains NaN values"

    def test_missing_source_raises_error(self):
        """Test that providing unregistered source raises error."""
        region = NeuralRegion(n_neurons=50, device="cpu")
        region.add_input_source("thalamus", n_input=128)

        # Try to provide input for unregistered source
        with pytest.raises(ValueError, match="No synaptic weights"):
            region.forward({"unknown_source": torch.rand(128) > 0.9})

    def test_partial_inputs_allowed(self):
        """Test that not all sources need to provide input."""
        region = NeuralRegion(n_neurons=100, device="cpu")
        region.add_input_source("thalamus", n_input=128, learning_rule=None)
        region.add_input_source("hippocampus", n_input=200, learning_rule=None)

        # Only provide thalamic input (hippocampus silent)
        output = region.forward({"thalamus": torch.rand(128) > 0.9})

        # Should work (hippocampus not in dict means no input from it)
        assert output.shape == (100,), \
            f"Output shape should be (100,), got {output.shape}"
        assert output.dtype in [torch.bool, torch.uint8], \
            f"Output should be bool or uint8, got {output.dtype}"
        assert not torch.isnan(output.float()).any(), \
            "Output contains NaN values"


class TestNeuralRegionLearning:
    """Test synaptic plasticity in NeuralRegion."""

    def test_learning_modifies_weights(self):
        """Test that plasticity actually updates synaptic weights."""
        region = NeuralRegion(
            n_neurons=50,
            default_learning_rule="hebbian",  # Simple Hebbian for testing
            device="cpu",
        )
        region.add_input_source("input", n_input=100, weight_scale=0.1)

        # Get initial weights
        initial_weights = region.synaptic_weights["input"].data.clone()

        # Create correlated input/output pattern (should strengthen weights)
        for _ in range(10):
            # Strong input (many spikes)
            input_spikes = torch.rand(100) > 0.3

            # Forward (will cause some output spikes and weight updates)
            output = region.forward({"input": input_spikes})

        # Check weights changed
        final_weights = region.synaptic_weights["input"].data
        weight_change = (final_weights - initial_weights).abs().sum()

        assert weight_change > 0, "Weights should change with learning"

    def test_no_learning_when_rule_none(self):
        """Test that weights don't change when learning_rule=None."""
        region = NeuralRegion(n_neurons=50, device="cpu")
        region.add_input_source("input", n_input=100, learning_rule=None)

        # Get initial weights
        initial_weights = region.synaptic_weights["input"].data.clone()

        # Run forward passes
        for _ in range(10):
            input_spikes = torch.rand(100) > 0.5
            region.forward({"input": input_spikes})

        # Weights should be unchanged
        final_weights = region.synaptic_weights["input"].data
        assert torch.allclose(initial_weights, final_weights), \
            "Weights should not change when learning_rule=None"


class TestNeuralRegionStateManagement:
    """Test state reset and device movement."""

    def test_reset_state(self):
        """Test that reset_state clears neuron state."""
        region = NeuralRegion(n_neurons=50, device="cpu")
        region.add_input_source("input", n_input=100, learning_rule=None)

        # Run forward to create state
        input_spikes = torch.rand(100) > 0.5
        region.forward({"input": input_spikes})

        # Output should be generated with correct shape and type
        assert region.output_spikes is not None
        assert region.output_spikes.shape == (50,), \
            f"Output spikes shape should be (50,), got {region.output_spikes.shape}"
        assert region.output_spikes.dtype == torch.bool, \
            f"Output spikes should be bool, got {region.output_spikes.dtype}"
        assert not torch.isnan(region.output_spikes.float()).any(), \
            "Output spikes contain NaN values"

        # Reset
        region.reset_state()

        # State should be cleared
        assert region.output_spikes is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_movement(self):
        """Test moving region to different device."""
        region = NeuralRegion(n_neurons=50, device="cpu")
        region.add_input_source("input", n_input=100)

        # Move to CUDA
        region.to(torch.device("cuda"))

        # Check weights moved
        assert region.synaptic_weights["input"].device.type == "cuda"

        # Test forward pass on CUDA
        input_spikes = torch.rand(100, device="cuda") > 0.5
        output = region.forward({"input": input_spikes})
        assert output.device.type == "cuda"


class TestNeuralRegionCompatibility:
    """Test compatibility interface (n_input, n_output, reset_state)."""

    def test_has_n_input_n_output(self):
        """Test that region exposes n_input and n_output."""
        n_neurons = 100
        n_source1 = 50
        n_source2 = 75
        region = NeuralRegion(n_neurons=n_neurons, device="cpu")

        assert hasattr(region, "n_input")
        assert hasattr(region, "n_output")
        assert region.n_output == n_neurons

        # n_input updates as sources added
        assert region.n_input == 0
        region.add_input_source("source1", n_input=n_source1)
        assert region.n_input == n_source1
        region.add_input_source("source2", n_input=n_source2)
        assert region.n_input == n_source1 + n_source2

    def test_is_nn_module(self):
        """Test that NeuralRegion is an nn.Module."""
        region = NeuralRegion(n_neurons=100, device="cpu")
        assert isinstance(region, nn.Module)

        # Not tied to old LearnableComponent hierarchy
        # This is intentional - v2.0 is independent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
