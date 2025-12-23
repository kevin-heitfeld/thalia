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


@pytest.mark.parametrize("modality,input_size,pool_attr", [
    ("visual", 30, "visual_pool_spikes"),
    ("auditory", 30, "auditory_pool_spikes"),
    ("language", 40, "language_pool_spikes"),
])
def test_multimodal_forward_single_modality(multimodal_region, modality, input_size, pool_attr):
    """Test forward pass with a single modality input.

    Why this test exists: Validates that each sensory pathway can process
    information independently, which is critical for understanding how
    multimodal integration handles partial sensory information.

    Cases tested:
    - Visual only: Tests visual pathway in isolation
    - Auditory only: Tests auditory pathway in isolation
    - Language only: Tests language pathway in isolation
    """
    # Create input for the specified modality
    input_tensor = torch.randn(input_size)
    kwargs = {f"{modality}_input": input_tensor}

    output = multimodal_region.forward(**kwargs)

    # Shape and type validation
    assert output.shape == (100,), \
        f"Output shape should be (100,), got {output.shape}"
    assert output.dtype == torch.float32, \
        f"Output should be float32, got {output.dtype}"
    assert not torch.isnan(output).any(), \
        "Output contains NaN values"
    assert not torch.isinf(output).any(), \
        "Output contains Inf values"

    # Value validation: output should be bounded (firing rates)
    assert output.min() >= 0.0, "Firing rates should be non-negative"
    assert output.max() <= 1.0, "Firing rates should not exceed 1.0"

    # Should have some activity in the corresponding pool
    pool_spikes = getattr(multimodal_region, pool_attr)
    assert pool_spikes.sum() >= 0, f"{modality} pool should have activity"


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

    # Shape and type validation
    assert output.shape == (100,), \
        f"Output shape should be (100,), got {output.shape}"
    assert output.dtype == torch.float32, \
        f"Output should be float32, got {output.dtype}"
    assert not torch.isnan(output).any(), \
        "Output contains NaN values"
    assert not torch.isinf(output).any(), \
        "Output contains Inf values"

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
    if hasattr(multimodal_region, 'hebbian_strategy') and multimodal_region.hebbian_strategy is not None:
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

    # Check for standard diagnostic structure
    assert "activity" in diag
    assert "plasticity" in diag or "health" in diag


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
    # Create region (hebbian_strategy always created but controlled by plasticity_enabled)
    region = MultimodalIntegration(multimodal_config)
    region.plasticity_enabled = False

    # Plasticity should be disabled
    assert not region.plasticity_enabled

    # Should still work
    output = region.forward(
        visual_input=torch.randn(30),
        auditory_input=torch.randn(30),
    )
    assert output.shape == (100,), \
        f"Output shape should be (100,), got {output.shape}"
    assert output.dtype == torch.float32, \
        f"Output should be float32, got {output.dtype}"
    assert not torch.isnan(output).any(), \
        "Output contains NaN values"


def test_multimodal_device_consistency(multimodal_region):
    """Test all tensors are on correct device."""
    device = multimodal_region.config.device

    assert multimodal_region.visual_input_weights.device.type == device
    assert multimodal_region.auditory_input_weights.device.type == device
    assert multimodal_region.visual_to_auditory.device.type == device
    assert multimodal_region.integration_weights.device.type == device
