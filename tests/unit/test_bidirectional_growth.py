"""Test bidirectional growth support for pathways.

Tests that pathways can grow on both source (pre-synaptic) and target (post-synaptic)
sides without breaking connectivity or learning.
"""

import pytest
import torch
from thalia.pathways.spiking_pathway import SpikingPathway
from thalia.core.base.component_config import PathwayConfig


@pytest.fixture
def simple_pathway():
    """Create a simple spiking pathway for testing."""
    config = PathwayConfig(
        n_input=100,
        n_output=80,
        dt_ms=1.0,
        device="cpu",
    )
    return SpikingPathway(config)


def test_grow_output_expands_output_dimension(simple_pathway):
    """Test that grow_output() expands the pathway's output dimension."""
    initial_output_size = simple_pathway.config.n_output
    initial_input_size = simple_pathway.config.n_input
    initial_weights = simple_pathway.weights.data.clone()

    # Grow output by 20 neurons
    simple_pathway.grow_output(n_new=20)

    # Verify output dimension expanded
    assert simple_pathway.config.n_output == initial_output_size + 20, \
        "Output dimension should increase by 20"

    # Verify input dimension unchanged
    assert simple_pathway.config.n_input == initial_input_size, \
        "Input dimension should not change"

    # Verify weight matrix shape: [target, source]
    assert simple_pathway.weights.shape[0] == initial_output_size + 20, \
        "Weight matrix output dimension should increase"
    assert simple_pathway.weights.shape[1] == initial_input_size, \
        "Weight matrix input dimension should stay same"

    # Verify old weights preserved
    # Old weights should be in top rows
    assert torch.allclose(
        simple_pathway.weights[:initial_output_size, :],
        initial_weights,
        atol=0.01
    ), "Old weights should be preserved in top rows"


def test_grow_input_expands_input_dimension(simple_pathway):
    """Test that grow_input() expands the pathway's input dimension."""
    initial_output_size = simple_pathway.config.n_output
    initial_input_size = simple_pathway.config.n_input
    initial_weights = simple_pathway.weights.data.clone()

    # Grow input by 30 neurons
    simple_pathway.grow_input(n_new=30)

    # Verify input dimension expanded
    assert simple_pathway.config.n_input == initial_input_size + 30, \
        "Input dimension should increase by 30"

    # Verify output dimension unchanged
    assert simple_pathway.config.n_output == initial_output_size, \
        "Output dimension should not change"

    # Verify weight matrix shape: [target, source]
    assert simple_pathway.weights.shape[0] == initial_output_size, \
        "Weight matrix output dimension should stay same"
    assert simple_pathway.weights.shape[1] == initial_input_size + 30, \
        "Weight matrix input dimension should increase"

    # Verify old weights preserved (left columns)
    assert torch.allclose(
        simple_pathway.weights[:, :initial_input_size],
        initial_weights,
        atol=0.01
    ), "Old weights should be preserved in left columns"


def test_bidirectional_growth_sequence(simple_pathway):
    """Test growing both input and output in sequence."""
    initial_input = simple_pathway.config.n_input
    initial_output = simple_pathway.config.n_output

    # Grow output first
    simple_pathway.grow_output(n_new=10)
    assert simple_pathway.config.n_output == initial_output + 10
    assert simple_pathway.config.n_input == initial_input

    # Then grow input
    simple_pathway.grow_input(n_new=15)
    assert simple_pathway.config.n_output == initial_output + 10
    assert simple_pathway.config.n_input == initial_input + 15

    # Verify final weight matrix shape
    assert simple_pathway.weights.shape == (initial_output + 10, initial_input + 15), \
        "Weight matrix should match final dimensions"


def test_grow_output_preserves_traces(simple_pathway):
    """Test that grow_output() expands post-synaptic traces."""
    initial_output_size = simple_pathway.config.n_output
    initial_post_trace = simple_pathway.post_trace.clone()

    simple_pathway.grow_output(n_new=10)

    # Verify post_trace expanded
    assert simple_pathway.post_trace.shape[0] == initial_output_size + 10, \
        "Post-synaptic trace should expand with target"

    # Old traces should be preserved
    assert torch.allclose(
        simple_pathway.post_trace[:initial_output_size],
        initial_post_trace,
        atol=1e-6
    ), "Old post-synaptic traces should be preserved"


def test_grow_input_preserves_traces(simple_pathway):
    """Test that grow_input() expands pre-synaptic traces."""
    initial_input_size = simple_pathway.config.n_input
    initial_pre_trace = simple_pathway.pre_trace.clone()

    simple_pathway.grow_input(n_new=15)

    # Verify pre_trace expanded
    assert simple_pathway.pre_trace.shape[0] == initial_input_size + 15, \
        "Pre-synaptic trace should expand with source"

    # Old traces should be preserved
    assert torch.allclose(
        simple_pathway.pre_trace[:initial_input_size],
        initial_pre_trace,
        atol=1e-6
    ), "Old pre-synaptic traces should be preserved"


def test_forward_after_bidirectional_growth(simple_pathway):
    """Test that pathway forward pass still works after bidirectional growth."""
    # Grow both dimensions
    simple_pathway.grow_input(n_new=10)
    simple_pathway.grow_output(n_new=5)

    # Create input spikes matching new source size
    input_spikes = torch.zeros(simple_pathway.config.n_input)
    input_spikes[:20] = 1.0  # Activate some neurons

    # Forward pass should not crash
    output = simple_pathway.forward(input_spikes)

    # Verify output shape matches new target size
    assert output.shape[0] == simple_pathway.config.n_output, \
        "Output shape should match grown target dimension"


def test_growth_with_different_initializations():
    """Test that different initialization strategies work for growth."""
    config = PathwayConfig(n_input=50, n_output=40, dt_ms=1.0, device="cpu")

    # Test xavier initialization
    pathway1 = SpikingPathway(config)
    initial_size1 = pathway1.config.n_output
    pathway1.grow_output(n_new=10, initialization='xavier')
    assert pathway1.config.n_output == initial_size1 + 10

    # Test uniform initialization
    pathway2 = SpikingPathway(config)
    initial_size2 = pathway2.config.n_output
    pathway2.grow_output(n_new=10, initialization='uniform')
    assert pathway2.config.n_output == initial_size2 + 10

    # Test sparse_random initialization (default)
    pathway3 = SpikingPathway(config)
    initial_size3 = pathway3.config.n_output
    pathway3.grow_output(n_new=10, initialization='sparse_random', sparsity=0.2)
    assert pathway3.config.n_output == initial_size3 + 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
