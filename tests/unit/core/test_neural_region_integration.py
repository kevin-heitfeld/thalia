"""Integration tests for NeuralRegion with BrainBuilder.

Tests the full workflow: BrainBuilder → connections → NeuralRegion.add_input_source()
"""

import pytest
import torch

from thalia.core.neural_region import NeuralRegion


class SimpleNeuralRegion(NeuralRegion):
    """Minimal NeuralRegion for testing integration."""

    def __init__(self, n_neurons: int, device: str = "cpu"):
        super().__init__(
            n_neurons=n_neurons,
            default_learning_rule="stdp",
            device=device,
        )


def test_manual_input_registration():
    """Test manually registering inputs after region creation."""
    region = SimpleNeuralRegion(n_neurons=50, device="cpu")

    # Manually register input sources
    region.add_input_source("cortex", n_input=128, learning_rule="stdp")
    region.add_input_source("hippocampus", n_input=64, learning_rule="bcm")

    # Verify registration
    assert "cortex" in region.synaptic_weights
    assert "hippocampus" in region.synaptic_weights
    assert region.synaptic_weights["cortex"].shape == (50, 128)
    assert region.synaptic_weights["hippocampus"].shape == (50, 64)

    # Test forward pass
    inputs = {
        "cortex": torch.rand(128) > 0.8,
        "hippocampus": torch.rand(64) > 0.8,
    }

    output = region.forward(inputs)
    assert output.shape == (50,)
    assert output.dtype == torch.bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
