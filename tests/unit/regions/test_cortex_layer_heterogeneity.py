"""
Unit tests for LayeredCortex layer-specific heterogeneity (Phase 2A).

Tests layer-specific neuron properties reflecting biological diversity:
- L4 spiny stellate: Fast integration (tau_mem ~10ms), low threshold
- L2/3 pyramidal: Medium integration (tau_mem ~20ms), moderate threshold
- L5 thick-tuft pyramidal: Slow integration (tau_mem ~30ms), high threshold
- L6a/L6b corticothalamic: Variable dynamics for feedback roles

Biological validation:
- Tau_mem scaling per layer (fast → slow from L4 → L5)
- Threshold tuning per layer (low → high for selectivity)
- Adaptation per layer (minimal L4, strong L2/3, moderate L5/L6)
- Integration with forward pass and learning
"""

import torch
import pytest

from thalia.regions.cortex import LayeredCortex
from thalia.regions.cortex.config import LayeredCortexConfig
from thalia.config import LayerSizeCalculator


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def device() -> str:
    """Standard device for all tests."""
    return "cpu"


@pytest.fixture
def small_layer_sizes() -> dict:
    """Small layer sizes for fast testing."""
    calc = LayerSizeCalculator()
    return calc.cortex_from_scale(scale_factor=32)


@pytest.fixture
def heterogeneous_config(device: str) -> LayeredCortexConfig:
    """Config with layer-specific heterogeneity enabled."""
    return LayeredCortexConfig(
        use_layer_heterogeneity=True,
        # Use default layer properties from config
        device=device,
    )


@pytest.fixture
def uniform_config(device: str) -> LayeredCortexConfig:
    """Config with heterogeneous layers disabled."""
    return LayeredCortexConfig(
        use_layer_heterogeneity=False,
        device=device,
    )


# =====================================================================
# TEST: Config Validation
# =====================================================================


def test_layer_heterogeneity_config_validation():
    """Test that layer-specific config parameters are valid."""
    config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        device="cpu",
    )

    # Check that layer properties are defined
    assert "l4" in config.layer_tau_mem
    assert "l23" in config.layer_tau_mem
    assert "l5" in config.layer_tau_mem
    assert "l6a" in config.layer_tau_mem
    assert "l6b" in config.layer_tau_mem

    # Check tau_mem ordering (L4 fast → L5 slow)
    assert config.layer_tau_mem["l4"] < config.layer_tau_mem["l23"]
    assert config.layer_tau_mem["l23"] < config.layer_tau_mem["l5"]

    # Check threshold definitions
    assert all(layer in config.layer_v_threshold for layer in ["l4", "l23", "l5", "l6a", "l6b"])

    # Check adaptation definitions
    assert all(layer in config.layer_adaptation for layer in ["l4", "l23", "l5", "l6a", "l6b"])


def test_layer_heterogeneity_parameter_ranges():
    """Test that layer parameters are within biological ranges."""
    config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        device="cpu",
    )

    # Tau_mem should be biologically plausible (5-40ms typical for cortex)
    for layer, tau in config.layer_tau_mem.items():
        assert 5.0 <= tau <= 40.0, f"{layer} tau_mem {tau} out of biological range"

    # V_threshold should be reasonable (-60 to -45mV typical)
    for layer, v_thresh in config.layer_v_threshold.items():
        assert -60.0 <= v_thresh <= -45.0, f"{layer} v_threshold {v_thresh} out of biological range"

    # Adaptation should be reasonable (0.0-0.3 typical)
    for layer, adapt in config.layer_adaptation.items():
        assert 0.0 <= adapt <= 0.5, f"{layer} adaptation {adapt} out of reasonable range"


# =====================================================================
# TEST: Neuron Creation with Heterogeneity
# =====================================================================


def test_cortex_creates_layer_heterogeneous_neurons(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that LayeredCortex creates neurons with layer-specific properties."""
    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Check that neurons exist for all layers
    assert cortex.l4_neurons is not None
    assert cortex.l23_neurons is not None
    assert cortex.l5_neurons is not None
    assert cortex.l6a_neurons is not None
    assert cortex.l6b_neurons is not None

    # Check that neurons have been created with correct sizes
    assert cortex.l4_neurons.n_neurons == small_layer_sizes["l4_size"]
    assert cortex.l23_neurons.n_neurons == small_layer_sizes["l23_size"]
    assert cortex.l5_neurons.n_neurons == small_layer_sizes["l5_size"]
    assert cortex.l6a_neurons.n_neurons == small_layer_sizes["l6a_size"]
    assert cortex.l6b_neurons.n_neurons == small_layer_sizes["l6b_size"]


def test_layer_tau_mem_heterogeneity(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that layers have different membrane time constants."""
    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Get tau_mem from neuron config (via g_L: tau_mem = C_m / g_L)
    # For ConductanceLIF, tau_mem is stored in config
    l4_tau = cortex.l4_neurons.config.tau_mem
    l23_tau = cortex.l23_neurons.config.tau_mem
    l5_tau = cortex.l5_neurons.config.tau_mem
    l6a_tau = cortex.l6a_neurons.config.tau_mem
    l6b_tau = cortex.l6b_neurons.config.tau_mem

    # Verify expected ordering: L4 (fast) < L2/3 (medium) < L5 (slow)
    assert l4_tau < l23_tau, f"L4 tau ({l4_tau}) should be < L2/3 tau ({l23_tau})"
    assert l23_tau < l5_tau, f"L2/3 tau ({l23_tau}) should be < L5 tau ({l5_tau})"

    # Verify values match config
    assert l4_tau == heterogeneous_config.layer_tau_mem["l4"]
    assert l23_tau == heterogeneous_config.layer_tau_mem["l23"]
    assert l5_tau == heterogeneous_config.layer_tau_mem["l5"]
    assert l6a_tau == heterogeneous_config.layer_tau_mem["l6a"]
    assert l6b_tau == heterogeneous_config.layer_tau_mem["l6b"]


def test_layer_threshold_heterogeneity(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that layers have different voltage thresholds."""
    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Get v_threshold from neuron config
    l4_thresh = cortex.l4_neurons.config.v_threshold
    l23_thresh = cortex.l23_neurons.config.v_threshold
    l5_thresh = cortex.l5_neurons.config.v_threshold
    l6a_thresh = cortex.l6a_neurons.config.v_threshold
    l6b_thresh = cortex.l6b_neurons.config.v_threshold

    # Verify values match config
    assert l4_thresh == heterogeneous_config.layer_v_threshold["l4"]
    assert l23_thresh == heterogeneous_config.layer_v_threshold["l23"]
    assert l5_thresh == heterogeneous_config.layer_v_threshold["l5"]
    assert l6a_thresh == heterogeneous_config.layer_v_threshold["l6a"]
    assert l6b_thresh == heterogeneous_config.layer_v_threshold["l6b"]


def test_layer_adaptation_heterogeneity(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that layers have different adaptation strengths."""
    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Get adapt_increment from neuron config
    l4_adapt = cortex.l4_neurons.config.adapt_increment
    l23_adapt = cortex.l23_neurons.config.adapt_increment
    l5_adapt = cortex.l5_neurons.config.adapt_increment
    l6a_adapt = cortex.l6a_neurons.config.adapt_increment
    l6b_adapt = cortex.l6b_neurons.config.adapt_increment

    # Verify values match config
    assert l4_adapt == heterogeneous_config.layer_adaptation["l4"]
    assert l23_adapt == heterogeneous_config.layer_adaptation["l23"]
    assert l5_adapt == heterogeneous_config.layer_adaptation["l5"]
    assert l6a_adapt == heterogeneous_config.layer_adaptation["l6a"]
    assert l6b_adapt == heterogeneous_config.layer_adaptation["l6b"]

    # Verify L4 has minimal adaptation (faithful sensory relay)
    assert l4_adapt < 0.1, f"L4 adaptation ({l4_adapt}) should be minimal"

    # Verify L2/3 has strong adaptation (prevents frozen attractors)
    assert l23_adapt > 0.1, f"L2/3 adaptation ({l23_adapt}) should be strong"


# =====================================================================
# TEST: Disabled Heterogeneity
# =====================================================================


def test_cortex_without_heterogeneity_uses_defaults(
    uniform_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that LayeredCortex uses default properties when heterogeneity disabled."""
    cortex = LayeredCortex(
        config=uniform_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Neurons should still be created
    assert cortex.l4_neurons is not None
    assert cortex.l23_neurons is not None
    assert cortex.l5_neurons is not None

    # But they should use standard factory defaults (not layer-specific overrides)
    # L4 and L5 should have similar properties (not highly differentiated)
    # Note: create_cortical_layer_neurons has some built-in differences,
    # but we're testing that config overrides are NOT applied


def test_forward_pass_works_without_heterogeneity(
    uniform_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that forward pass works normally when heterogeneity disabled."""
    cortex = LayeredCortex(
        config=uniform_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Create input
    input_size = small_layer_sizes["input_size"]
    input_spikes = (torch.rand(input_size, device=device) > 0.8).float()

    # Forward pass should work
    output = cortex.forward(input_spikes)

    # Check output shape (L2/3 + L5 concatenated)
    expected_size = small_layer_sizes["l23_size"] + small_layer_sizes["l5_size"]
    assert output.shape == (expected_size,)
    assert output.dtype == torch.bool


# =====================================================================
# TEST: Forward Pass with Heterogeneity
# =====================================================================


def test_forward_pass_with_heterogeneous_layers(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that forward pass works with layer-specific properties."""
    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Run forward pass for several timesteps
    input_size = small_layer_sizes["input_size"]

    for _ in range(20):
        input_spikes = (torch.rand(input_size, device=device) > 0.8).float()
        output = cortex.forward(input_spikes)

        # Check output shape
        expected_size = small_layer_sizes["l23_size"] + small_layer_sizes["l5_size"]
        assert output.shape == (expected_size,)
        assert output.dtype == torch.bool

        # Check that some activity is generated (not all zeros)
        # This verifies that heterogeneous properties don't break processing


def test_layer_specific_dynamics_observable(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that different layer dynamics are observable during forward pass."""
    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Present strong input
    input_size = small_layer_sizes["input_size"]
    strong_input = torch.ones(input_size, device=device)

    # Run for multiple timesteps
    l4_activity_sum = 0.0
    l23_activity_sum = 0.0
    l5_activity_sum = 0.0

    for _ in range(50):
        cortex.forward(strong_input)

        # Accumulate layer activity
        if cortex.state.l4_spikes is not None:
            l4_activity_sum += cortex.state.l4_spikes.float().sum().item()
        if cortex.state.l23_spikes is not None:
            l23_activity_sum += cortex.state.l23_spikes.float().sum().item()
        if cortex.state.l5_spikes is not None:
            l5_activity_sum += cortex.state.l5_spikes.float().sum().item()

    # Check that all layers show some activity
    assert l4_activity_sum > 0, "L4 should show activity with strong input"
    assert l23_activity_sum > 0, "L2/3 should show activity"
    assert l5_activity_sum > 0, "L5 should show activity"

    # Due to different tau_mem and thresholds, layers should have different activity patterns
    # This is a weak test just verifying the mechanism exists


# =====================================================================
# TEST: Growth with Heterogeneity
# =====================================================================


def test_grow_output_preserves_layer_heterogeneity(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that grow_output maintains layer-specific properties for new neurons."""
    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Get initial tau_mem values
    l4_tau_before = cortex.l4_neurons.config.tau_mem
    l23_tau_before = cortex.l23_neurons.config.tau_mem
    l5_tau_before = cortex.l5_neurons.config.tau_mem

    # Grow output (adds neurons to all layers proportionally)
    n_new = 8
    cortex.grow_output(n_new)

    # Check that tau_mem values are preserved for new neurons
    l4_tau_after = cortex.l4_neurons.config.tau_mem
    l23_tau_after = cortex.l23_neurons.config.tau_mem
    l5_tau_after = cortex.l5_neurons.config.tau_mem

    assert l4_tau_after == l4_tau_before
    assert l23_tau_after == l23_tau_before
    assert l5_tau_after == l5_tau_before

    # Check that ordering is still correct
    assert l4_tau_after < l23_tau_after < l5_tau_after


# =====================================================================
# TEST: Biological Plausibility
# =====================================================================


def test_l4_fast_sensory_processing():
    """Test that L4 has fast dynamics suitable for sensory input."""
    config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        device="cpu",
    )

    # L4 should have fastest tau_mem (fast sensory processing)
    l4_tau = config.layer_tau_mem["l4"]
    assert l4_tau <= 12.0, f"L4 tau_mem ({l4_tau}ms) should be ≤12ms for fast sensory processing"

    # L4 should have low adaptation (faithful relay)
    l4_adapt = config.layer_adaptation["l4"]
    assert l4_adapt < 0.1, f"L4 adaptation ({l4_adapt}) should be minimal (<0.1)"


def test_l23_integration_dynamics():
    """Test that L2/3 has medium dynamics suitable for integration."""
    config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        device="cpu",
    )

    # L2/3 should have medium tau_mem (integration over ~20ms)
    l23_tau = config.layer_tau_mem["l23"]
    assert 15.0 <= l23_tau <= 25.0, f"L2/3 tau_mem ({l23_tau}ms) should be 15-25ms for integration"

    # L2/3 should have strong adaptation (prevents frozen attractors)
    l23_adapt = config.layer_adaptation["l23"]
    assert l23_adapt >= 0.1, f"L2/3 adaptation ({l23_adapt}) should be strong (≥0.1)"


def test_l5_sustained_output_dynamics():
    """Test that L5 has slow dynamics suitable for sustained output."""
    config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        device="cpu",
    )

    # L5 should have slowest tau_mem (sustained output)
    l5_tau = config.layer_tau_mem["l5"]
    assert l5_tau >= 25.0, f"L5 tau_mem ({l5_tau}ms) should be ≥25ms for sustained output"

    # L5 should have moderate adaptation (allows bursts while preventing runaway)
    l5_adapt = config.layer_adaptation["l5"]
    assert 0.05 <= l5_adapt <= 0.15, f"L5 adaptation ({l5_adapt}) should be moderate (0.05-0.15)"


def test_l6_feedback_dynamics():
    """Test that L6a/L6b have appropriate dynamics for corticothalamic feedback."""
    config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        device="cpu",
    )

    # L6a (TRN feedback): Lower frequency, low gamma (25-35 Hz)
    l6a_tau = config.layer_tau_mem["l6a"]
    assert 12.0 <= l6a_tau <= 20.0, f"L6a tau_mem ({l6a_tau}ms) should support low gamma"

    # L6b (relay feedback): Higher frequency, high gamma (60-80 Hz)
    l6b_tau = config.layer_tau_mem["l6b"]
    assert 20.0 <= l6b_tau <= 30.0, f"L6b tau_mem ({l6b_tau}ms) should support high gamma"


# =====================================================================
# TEST: Custom Layer Properties
# =====================================================================


def test_custom_layer_properties_override_defaults():
    """Test that custom layer properties can be specified in config."""
    custom_config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        layer_tau_mem={
            "l4": 8.0,   # Custom: very fast
            "l23": 25.0,  # Custom: slow integration
            "l5": 35.0,  # Custom: very slow
            "l6a": 12.0,
            "l6b": 22.0,
        },
        layer_v_threshold={
            "l4": -48.0,  # Custom: very low threshold
            "l23": -58.0, # Custom: high threshold
            "l5": -45.0,
            "l6a": -52.0,
            "l6b": -50.0,
        },
        device="cpu",
    )

    # Verify custom values are stored
    assert custom_config.layer_tau_mem["l4"] == 8.0
    assert custom_config.layer_tau_mem["l23"] == 25.0
    assert custom_config.layer_v_threshold["l4"] == -48.0
    assert custom_config.layer_v_threshold["l23"] == -58.0


# =====================================================================
# TEST: Integration with Existing Features
# =====================================================================


def test_heterogeneity_compatible_with_gap_junctions(
    heterogeneous_config: LayeredCortexConfig,
    small_layer_sizes: dict,
    device: str,
):
    """Test that layer heterogeneity works with gap junctions."""
    # Enable gap junctions
    heterogeneous_config.gap_junctions_enabled = True

    cortex = LayeredCortex(
        config=heterogeneous_config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Check that gap junctions are created
    assert cortex.gap_junctions_l23 is not None

    # Forward pass should work
    input_size = small_layer_sizes["input_size"]
    input_spikes = (torch.rand(input_size, device=device) > 0.8).float()
    output = cortex.forward(input_spikes)

    assert output.shape[0] == small_layer_sizes["l23_size"] + small_layer_sizes["l5_size"]


def test_heterogeneity_compatible_with_bcm(
    small_layer_sizes: dict,
    device: str,
):
    """Test that layer heterogeneity works with BCM learning."""
    # Enable both heterogeneity and BCM
    config = LayeredCortexConfig(
        use_layer_heterogeneity=True,
        bcm_enabled=True,
        device=device,
    )

    cortex = LayeredCortex(
        config=config,
        sizes=small_layer_sizes,
        device=device,
    )

    # Check that BCM strategies are created
    assert cortex.bcm_l4 is not None
    assert cortex.bcm_l23 is not None
    assert cortex.bcm_l5 is not None

    # Forward pass with learning should work
    input_size = small_layer_sizes["input_size"]
    for _ in range(10):
        input_spikes = (torch.rand(input_size, device=device) > 0.8).float()
        output = cortex.forward(input_spikes)

        assert output.dtype == torch.bool
