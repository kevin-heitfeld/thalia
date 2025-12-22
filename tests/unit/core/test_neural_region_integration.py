"""Integration tests for NeuralRegion with BrainBuilder.

Tests the full workflow: BrainBuilder → connections → NeuralRegion.add_input_source()
"""

import pytest
import torch

from thalia.core.neural_region import NeuralRegion
from thalia.config import GlobalConfig


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


def test_brain_builder_workflow():
    """Test the intended workflow: BrainBuilder creates regions, then we register inputs.

    NOTE: This test documents the MANUAL approach until BrainBuilder is updated.
    Automatic input registration would require modifying BrainBuilder to detect
    NeuralRegion subclasses and call add_input_source() based on connections.
    """
    _ = GlobalConfig(device="cpu", dt_ms=1.0)  # For future use

    # For now, we can't use BrainBuilder directly with NeuralRegion
    # because it doesn't know about the new architecture yet.
    # This test documents what we WANT to work in the future.

    # What we want (future):
    # from thalia.core.brain_builder import BrainBuilder
    # builder = BrainBuilder(global_config)
    # builder.add_component("region_a", "simple_neural_region", n_neurons=50)
    # builder.add_component("region_b", "simple_neural_region", n_neurons=40)
    # builder.connect("region_a", "region_b", pathway_type="axonal")
    # brain = builder.build()
    # # BrainBuilder should automatically call:
    # # brain.components["region_b"].add_input_source("region_a", n_input=50)

    # What we have now (manual):
    region_a = SimpleNeuralRegion(n_neurons=50, device="cpu")
    region_b = SimpleNeuralRegion(n_neurons=40, device="cpu")

    # Manually register that region_b receives from region_a
    region_b.add_input_source("region_a", n_input=50, learning_rule="stdp")

    # Simulate forward pass
    # region_a needs an input source too
    region_a.add_input_source("external", n_input=100, learning_rule=None)
    a_output = region_a.forward({"external": torch.rand(100) > 0.8})
    b_output = region_b.forward({"region_a": a_output})

    assert b_output.shape == (40,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
