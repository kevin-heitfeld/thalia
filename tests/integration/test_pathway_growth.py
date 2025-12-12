"""
Integration test for pathway growth.

Tests that all pathway types (base, attention, replay) correctly expand
their dimensions while preserving learned state.
"""

import pytest
import torch

from thalia.pathways.spiking_pathway import SpikingPathway
from thalia.pathways.attention.spiking_attention import SpikingAttentionPathway, SpikingAttentionPathwayConfig
from thalia.pathways.spiking_replay import SpikingReplayPathway, SpikingReplayPathwayConfig
from thalia.core.base.component_config import PathwayConfig


@pytest.fixture
def base_pathway():
    """Create basic spiking pathway."""
    config = PathwayConfig(
        n_input=32,
        n_output=64,
        device='cpu',
        dt_ms=1.0,
    )
    return SpikingPathway(config)


@pytest.fixture
def attention_pathway():
    """Create attention pathway."""
    config = SpikingAttentionPathwayConfig(
        n_input=32,  # PFC size
        n_output=96,  # Cortex L2/3 size
        device='cpu',
        dt_ms=1.0,
    )
    return SpikingAttentionPathway(config)


@pytest.fixture
def replay_pathway():
    """Create replay pathway."""
    config = SpikingReplayPathwayConfig(
        n_input=48,  # Hippocampus size
        n_output=96,  # Cortex size
        device='cpu',
        dt_ms=1.0,
    )
    return SpikingReplayPathway(config)


def test_base_pathway_growth(base_pathway):
    """Test that base pathway correctly expands dimensions."""
    initial_output_size = base_pathway.config.n_output
    initial_weights = base_pathway.weights.clone()

    # Grow pathway
    n_new = 16
    base_pathway.add_neurons(n_new)

    # Verify dimensions updated
    assert base_pathway.config.n_output == initial_output_size + n_new
    assert base_pathway.weights.shape[0] == initial_output_size + n_new
    assert base_pathway.axonal_delays.shape[0] == initial_output_size + n_new
    assert base_pathway.neurons.n_neurons == initial_output_size + n_new

    # Verify old weights preserved
    torch.testing.assert_close(
        base_pathway.weights[:initial_output_size, :],
        initial_weights,
        msg="Old weights should be preserved after growth"
    )


def test_attention_pathway_growth(attention_pathway):
    """Test that attention pathway correctly expands all layers."""
    initial_output_size = attention_pathway.config.n_output
    initial_input_size = attention_pathway.config.n_input

    # Run forward pass to initialize state
    pfc_input = torch.rand(initial_input_size) < 0.2
    _ = attention_pathway.forward(pfc_input)

    # Store initial state
    initial_encoder_weight = attention_pathway.attention_encoder[0].weight.clone()

    # Grow pathway
    n_new = 24
    attention_pathway.add_neurons(n_new)

    new_output_size = initial_output_size + n_new

    # Verify base pathway dimensions
    assert attention_pathway.config.n_output == new_output_size
    assert attention_pathway.weights.shape[0] == new_output_size

    # Verify attention_encoder expanded
    assert attention_pathway.attention_encoder[0].weight.shape == (new_output_size, new_output_size)
    assert attention_pathway.attention_encoder[1].normalized_shape == (new_output_size,)

    # Verify old encoder weights preserved in top-left block
    torch.testing.assert_close(
        attention_pathway.attention_encoder[0].weight[:initial_output_size, :initial_output_size],
        initial_encoder_weight,
        msg="Old attention encoder weights should be preserved"
    )

    # Verify gain_output expanded output dimension (input dimension of layer)
    # Linear layer stores weights as [out_features, in_features]
    # gain_output: n_output -> n_input, so weight shape is [n_input, n_output]
    assert attention_pathway.gain_output.weight.shape == (initial_input_size, new_output_size)


def test_replay_pathway_growth(replay_pathway):
    """Test that replay pathway correctly expands projection layers."""
    initial_output_size = replay_pathway.config.n_output
    initial_input_size = replay_pathway.config.n_input

    # Store some patterns in replay buffer
    for _ in range(5):
        pattern = torch.rand(initial_input_size) < 0.15
        replay_pathway.store_pattern(pattern)

    initial_buffer_size = len(replay_pathway.replay_buffer)
    initial_proj_weight = replay_pathway.replay_projection[0].weight.clone()

    # Grow pathway
    n_new = 32
    replay_pathway.add_neurons(n_new)

    new_output_size = initial_output_size + n_new

    # Verify base pathway dimensions
    assert replay_pathway.config.n_output == new_output_size
    assert replay_pathway.weights.shape[0] == new_output_size

    # Verify replay_projection expanded
    assert replay_pathway.replay_projection[0].weight.shape == (new_output_size, initial_input_size)
    assert replay_pathway.replay_projection[1].normalized_shape == (new_output_size,)

    # Verify old projection weights preserved
    torch.testing.assert_close(
        replay_pathway.replay_projection[0].weight[:initial_output_size, :],
        initial_proj_weight,
        msg="Old replay projection weights should be preserved"
    )

    # Verify replay buffer unchanged (input side)
    assert len(replay_pathway.replay_buffer) == initial_buffer_size
    for entry in replay_pathway.replay_buffer:
        # Buffer stores pattern dicts with 'pattern', 'priority', 'age' keys
        pattern = entry['pattern']
        assert pattern.shape[0] == initial_input_size, "Buffer patterns should remain input-sized"

    # Verify priority network unchanged (operates on input only)
    assert replay_pathway.priority_network[0].in_features == initial_input_size


def test_growth_preserves_forward_pass(base_pathway):
    """Test that pathway still functions after growth."""
    input_size = base_pathway.config.n_input

    # Run forward passes before growth
    for _ in range(5):
        input_spikes = torch.rand(input_size) < 0.2
        _ = base_pathway.forward(input_spikes)

    # Grow pathway
    base_pathway.add_neurons(16)

    # Verify forward pass still works
    input_spikes = torch.rand(input_size) < 0.2
    output = base_pathway.forward(input_spikes)

    assert output is not None
    assert output.shape[0] == base_pathway.config.n_output
    assert output.dtype == torch.bool


def test_coordinated_growth_scenario(attention_pathway):
    """Test realistic scenario: cortex grows, attention pathway must expand."""
    # Initial sizes
    pfc_size = attention_pathway.config.n_input    # 32 (unchanged)
    cortex_l23_size = attention_pathway.config.n_output  # 96 (will grow)

    # Simulate cortex growth: L2/3 adds neurons
    cortex_growth = 24  # L2/3 grows to 120

    # Attention pathway must grow to match new cortex size
    attention_pathway.add_neurons(cortex_growth)

    # Verify pathway now matches new cortex size
    assert attention_pathway.config.n_output == cortex_l23_size + cortex_growth

    # Verify PFC input size unchanged
    assert attention_pathway.config.n_input == pfc_size

    # Test forward pass with original PFC size
    pfc_input = torch.rand(pfc_size) < 0.2
    output = attention_pathway.forward(pfc_input)

    assert output.shape[0] == cortex_l23_size + cortex_growth


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
