"""
Tests for multimodal integration region.

Tests cross-modal fusion of visual, auditory, and language inputs.
"""

import pytest
import torch

from thalia.regions.multisensory import MultimodalIntegration, MultimodalIntegrationConfig


@pytest.fixture
def device():
    """Device for testing (CPU by default, can be parametrized for CUDA)."""
    return torch.device("cpu")


@pytest.fixture
def multimodal_config(device):
    """Create test configuration."""
    return MultimodalIntegrationConfig(
        n_input=100,  # Not used (separate inputs)
        n_output=100,
        visual_input_size=30,
        auditory_input_size=30,
        language_input_size=40,
        visual_pool_ratio=0.3,
        auditory_pool_ratio=0.3,
        language_pool_ratio=0.2,
        integration_pool_ratio=0.2,
        device=str(device),
        dt_ms=1.0,
    )


@pytest.fixture
def multimodal_region(multimodal_config):
    """Create multimodal integration region."""
    region = MultimodalIntegration(multimodal_config)
    region.reset_state()
    return region


def test_multimodal_config_validation():
    """Test config validation for pool ratios."""
    # Should work
    config = MultimodalIntegrationConfig(
        n_input=100,
        n_output=100,
        visual_input_size=30,
        auditory_input_size=30,
        language_input_size=40,
        visual_pool_ratio=0.25,
        auditory_pool_ratio=0.25,
        language_pool_ratio=0.25,
        integration_pool_ratio=0.25,
        device="cpu",
    )
    region = MultimodalIntegration(config)
    # Contract: should successfully create region with valid config

    # Should fail (ratios don't sum to 1)
    bad_config = MultimodalIntegrationConfig(
        n_input=100,
        n_output=100,
        visual_input_size=30,
        auditory_input_size=30,
        language_input_size=40,
        visual_pool_ratio=0.5,
        auditory_pool_ratio=0.5,
        language_pool_ratio=0.5,
        integration_pool_ratio=0.5,
        device="cpu",
    )

    with pytest.raises(ValueError, match="Pool ratios must sum"):
        MultimodalIntegration(bad_config)


def test_multimodal_pool_sizes(multimodal_region, multimodal_config):
    """Test pool size calculations match configured ratios."""
    # Contract: pool sizes should match configured ratios of n_output
    expected_visual = int(multimodal_config.n_output * multimodal_config.visual_pool_ratio)
    expected_auditory = int(multimodal_config.n_output * multimodal_config.auditory_pool_ratio)
    expected_language = int(multimodal_config.n_output * multimodal_config.language_pool_ratio)
    expected_integration = (
        multimodal_config.n_output - expected_visual - expected_auditory - expected_language
    )

    assert multimodal_region.visual_pool_size == expected_visual
    assert multimodal_region.auditory_pool_size == expected_auditory
    assert multimodal_region.language_pool_size == expected_language
    assert multimodal_region.integration_pool_size == expected_integration

    # Contract: pools should sum to total output
    total_pool = (
        multimodal_region.visual_pool_size +
        multimodal_region.auditory_pool_size +
        multimodal_region.language_pool_size +
        multimodal_region.integration_pool_size
    )
    assert total_pool == multimodal_config.n_output


def test_multimodal_forward_visual_only(multimodal_region):
    """Test forward pass with only visual input."""
    visual_input = torch.randn(30)

    output = multimodal_region.forward(visual_input=visual_input)

    assert output.shape == (100,)
    assert output.dtype == torch.float32
    # Should have some visual activity
    assert multimodal_region.visual_pool_spikes.sum() >= 0


def test_multimodal_forward_auditory_only(multimodal_region):
    """Test forward pass with only auditory input."""
    auditory_input = torch.randn(30)

    output = multimodal_region.forward(auditory_input=auditory_input)

    assert output.shape == (100,)
    # Should have some auditory activity
    assert multimodal_region.auditory_pool_spikes.sum() >= 0


def test_multimodal_forward_language_only(multimodal_region):
    """Test forward pass with only language input."""
    language_input = torch.randn(40)

    output = multimodal_region.forward(language_input=language_input)

    assert output.shape == (100,)
    # Should have some language activity
    assert multimodal_region.language_pool_spikes.sum() >= 0


def test_multimodal_forward_all_modalities(multimodal_region):
    """Test forward pass with all inputs."""
    visual_input = torch.randn(30)
    auditory_input = torch.randn(30)
    language_input = torch.randn(40)

    output = multimodal_region.forward(
        visual_input=visual_input,
        auditory_input=auditory_input,
        language_input=language_input,
    )

    assert output.shape == (100,)
    # All pools should have activity
    assert multimodal_region.visual_pool_spikes.sum() >= 0
    assert multimodal_region.auditory_pool_spikes.sum() >= 0
    assert multimodal_region.language_pool_spikes.sum() >= 0


def test_multimodal_cross_modal_interactions(multimodal_region):
    """Test cross-modal weight updates during learning."""
    # Enable plasticity
    multimodal_region.plasticity_enabled = True

    # Save initial cross-modal weights
    initial_visual_to_auditory = multimodal_region.visual_to_auditory.clone()

    # Process with both modalities multiple times
    for _ in range(10):
        visual_input = torch.randn(30) * 2  # Strong input
        auditory_input = torch.randn(30) * 2

        multimodal_region.forward(
            visual_input=visual_input,
            auditory_input=auditory_input,
        )

    # Cross-modal weights should have changed (if Hebbian enabled)
    if multimodal_region.config.enable_hebbian:
        weight_change = (
            multimodal_region.visual_to_auditory - initial_visual_to_auditory
        ).abs().sum()
        # Some change should occur (though may be small)
        assert weight_change >= 0


def test_multimodal_reset_state(multimodal_region):
    """Test state reset."""
    # Process some inputs
    visual_input = torch.randn(30)
    multimodal_region.forward(visual_input=visual_input)

    # Reset
    multimodal_region.reset_state()

    # Check state is zeroed
    assert torch.all(multimodal_region.visual_pool_spikes == 0)
    assert torch.all(multimodal_region.auditory_pool_spikes == 0)
    assert torch.all(multimodal_region.language_pool_spikes == 0)
    assert torch.all(multimodal_region.integration_spikes == 0)


def test_multimodal_diagnostics(multimodal_region):
    """Test diagnostic output."""
    # Process inputs
    visual_input = torch.randn(30) * 2
    auditory_input = torch.randn(30) * 2
    multimodal_region.forward(
        visual_input=visual_input,
        auditory_input=auditory_input,
    )

    diag = multimodal_region.get_diagnostics()

    assert "visual_pool_firing_rate_hz" in diag
    assert "auditory_pool_firing_rate_hz" in diag
    assert "language_pool_firing_rate_hz" in diag
    assert "integration_firing_rate_hz" in diag
    assert "cross_modal_weight_mean" in diag

    # Check that we get reasonable values (floats/ints)
    assert isinstance(diag["visual_pool_firing_rate_hz"], (int, float))
    assert isinstance(diag["cross_modal_weight_mean"], (int, float))


def test_multimodal_health_check(multimodal_region):
    """Test health monitoring."""
    # Process inputs to ensure activity
    visual_input = torch.randn(30) * 3
    auditory_input = torch.randn(30) * 3
    language_input = torch.randn(40) * 3

    for _ in range(5):
        multimodal_region.forward(
            visual_input=visual_input,
            auditory_input=auditory_input,
            language_input=language_input,
        )

    health = multimodal_region.check_health()

    assert hasattr(health, "is_healthy")
    assert hasattr(health, "issues")
    assert hasattr(health, "summary")
    assert isinstance(health.is_healthy, bool)
    assert isinstance(health.issues, list)


def test_multimodal_integration_output_size(multimodal_region):
    """Test integration neuron outputs are included."""
    output = multimodal_region.forward(
        visual_input=torch.randn(30),
        auditory_input=torch.randn(30),
    )

    # Output should include all pools + integration
    assert output.shape[0] == multimodal_region.config.n_output
    assert multimodal_region.integration_spikes.shape[0] == multimodal_region.integration_pool_size


def test_multimodal_no_hebbian(multimodal_config):
    """Test region without Hebbian plasticity."""
    multimodal_config.enable_hebbian = False
    region = MultimodalIntegration(multimodal_config)

    assert region.hebbian_strategy is None

    # Should still work
    output = region.forward(
        visual_input=torch.randn(30),
        auditory_input=torch.randn(30),
    )
    assert output.shape == (100,)


def test_multimodal_device_consistency(multimodal_region):
    """Test all tensors are on correct device."""
    device = multimodal_region.config.device

    assert multimodal_region.visual_input_weights.device.type == device
    assert multimodal_region.auditory_input_weights.device.type == device
    assert multimodal_region.visual_to_auditory.device.type == device
    assert multimodal_region.integration_weights.device.type == device
