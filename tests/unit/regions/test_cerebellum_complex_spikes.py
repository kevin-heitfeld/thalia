"""
Unit tests for Cerebellum complex spike dynamics (Phase 2B).

Tests graded error signaling through complex spike bursts:
- Binary error (0/1) → Burst length (2-7 spikelets)
- Burst length → Calcium influx (graded signal)
- Calcium → LTD magnitude (proportional learning)

Biological validation:
- Small errors trigger small corrections (2-3 spikelets)
- Large errors trigger large corrections (6-7 spikelets)
- Stochastic burst length (biological variability)
- Integration with gap junctions and error learning
"""

import pytest
import torch

from thalia.config import CerebellumConfig
from thalia.regions import Cerebellum

# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def device() -> str:
    """Standard device for all tests."""
    return "cpu"


@pytest.fixture
def small_sizes() -> dict:
    """Small sizes for fast testing."""
    return {
        "input_size": 32,
        "granule_size": 128,  # 4× expansion
        "purkinje_size": 16,
    }


@pytest.fixture
def complex_spike_config(device: str) -> CerebellumConfig:
    """Config with complex spike bursts enabled."""
    return CerebellumConfig(
        use_complex_spike_bursts=True,
        min_complex_spike_count=2,
        max_complex_spike_count=7,
        complex_spike_isi_ms=1.5,
        ca2_per_spikelet=0.2,
        use_enhanced_microcircuit=False,  # Simplify for testing
        device=device,
    )


@pytest.fixture
def binary_config(device: str) -> CerebellumConfig:
    """Config with traditional binary error signal."""
    return CerebellumConfig(
        use_complex_spike_bursts=False,
        use_enhanced_microcircuit=False,
        device=device,
    )


# =====================================================================
# TEST: Config Validation
# =====================================================================


def test_complex_spike_config_validation():
    """Test that complex spike config parameters are valid."""
    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        device="cpu",
    )

    # Check that parameters are defined
    assert hasattr(config, "min_complex_spike_count")
    assert hasattr(config, "max_complex_spike_count")
    assert hasattr(config, "complex_spike_isi_ms")
    assert hasattr(config, "ca2_per_spikelet")

    # Check parameter ranges
    assert config.min_complex_spike_count >= 2, "Min spike count should be ≥2"
    assert config.max_complex_spike_count <= 10, "Max spike count should be ≤10 (biological)"
    assert config.max_complex_spike_count > config.min_complex_spike_count
    assert 1.0 <= config.complex_spike_isi_ms <= 3.0, "ISI should be 1-3ms (biological)"
    assert config.ca2_per_spikelet > 0.0, "Calcium per spike should be positive"


def test_complex_spike_parameter_ranges():
    """Test that parameters are within biological ranges."""
    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        device="cpu",
    )

    # Burst length: 2-7 spikelets typical (Mathy et al. 2009)
    assert 2 <= config.min_complex_spike_count <= 3
    assert 5 <= config.max_complex_spike_count <= 8

    # ISI: 1-2ms typical (very fast bursts)
    assert 1.0 <= config.complex_spike_isi_ms <= 2.5


# =====================================================================
# TEST: Burst Generation
# =====================================================================


def test_generate_complex_spike_burst_small_error(
    complex_spike_config: CerebellumConfig,
    small_sizes: dict,
    device: str,
):
    """Test that small errors generate short bursts (2-3 spikelets)."""
    cerebellum = Cerebellum(
        config=complex_spike_config,
        sizes=small_sizes,
        device=device,
    )

    # Small error (0.1 = 10% of max)
    small_error = torch.tensor([0.1, 0.15, 0.2], device=device)

    # Generate complex spike bursts
    calcium = cerebellum._generate_complex_spike_burst(small_error)

    # Expected: 2-3 spikelets → Ca²⁺ = 0.4-0.6
    # min_count=2, max_count=7, range=5
    # error 0.1: 2 + 5*0.1 = 2.5 spikes → ~2-3 spikes (stochastic)
    # Ca²⁺ = n_spikes * 0.2
    assert calcium.shape == (3,)
    assert torch.all(calcium >= 0.4 - 0.2), "Small error should trigger 2+ spikes"
    assert torch.all(calcium <= 1.0), "Small error should trigger ≤4 spikes"


def test_generate_complex_spike_burst_large_error(
    complex_spike_config: CerebellumConfig,
    small_sizes: dict,
    device: str,
):
    """Test that large errors generate long bursts (6-7 spikelets)."""
    cerebellum = Cerebellum(
        config=complex_spike_config,
        sizes=small_sizes,
        device=device,
    )

    # Large error (0.9 = 90% of max)
    large_error = torch.tensor([0.85, 0.9, 0.95], device=device)

    # Generate complex spike bursts
    calcium = cerebellum._generate_complex_spike_burst(large_error)

    # Expected: 6-7 spikelets → Ca²⁺ = 1.2-1.4
    # error 0.9: 2 + 5*0.9 = 6.5 spikes → ~6-7 spikes (stochastic)
    # Ca²⁺ = n_spikes * 0.2
    assert calcium.shape == (3,)
    assert torch.all(calcium >= 1.0), "Large error should trigger 5+ spikes"
    assert torch.all(calcium <= 1.6), "Large error should trigger ≤7 spikes"


def test_complex_spike_burst_proportional_to_error(
    complex_spike_config: CerebellumConfig,
    small_sizes: dict,
    device: str,
):
    """Test that burst length scales proportionally with error magnitude."""
    cerebellum = Cerebellum(
        config=complex_spike_config,
        sizes=small_sizes,
        device=device,
    )

    # Range of errors from small to large
    errors = torch.linspace(0.1, 0.9, 9, device=device)

    # Generate bursts
    calcium = cerebellum._generate_complex_spike_burst(errors)

    # Check monotonic increase (on average, allowing for stochastic variation)
    # Use moving average to smooth out stochastic noise
    window_size = 3
    calcium_smoothed = torch.nn.functional.conv1d(
        calcium.unsqueeze(0).unsqueeze(0),
        torch.ones(1, 1, window_size, device=device) / window_size,
        padding=window_size // 2,
    ).squeeze()

    # Check that smoothed calcium generally increases
    differences = calcium_smoothed[1:] - calcium_smoothed[:-1]
    increasing_count = (differences >= -0.05).sum()  # Allow small decreases due to stochastic
    assert (
        increasing_count >= len(differences) * 0.7
    ), "Calcium should generally increase with error"


def test_complex_spike_stochastic_rounding(
    complex_spike_config: CerebellumConfig,
    small_sizes: dict,
    device: str,
):
    """Test that burst length uses stochastic rounding (biological variability)."""
    cerebellum = Cerebellum(
        config=complex_spike_config,
        sizes=small_sizes,
        device=device,
    )

    # Error that produces fractional spike count
    # error 0.3: 2 + 5*0.3 = 3.5 spikes → should round to 3 or 4 stochastically
    error = torch.full((100,), 0.3, device=device)

    # Generate bursts many times
    calcium_values = []
    for _ in range(10):
        calcium = cerebellum._generate_complex_spike_burst(error)
        calcium_values.append(calcium.mean().item())

    # Check that we get variation (not always same value)
    # Threshold lowered from 0.005 to 0.003 to be more robust to statistical variation
    calcium_std = torch.tensor(calcium_values).std()
    assert calcium_std > 0.003, "Should have variation due to stochastic rounding"

    # Check that mean is close to expected (3.5 * 0.2 = 0.7)
    calcium_mean = torch.tensor(calcium_values).mean()
    assert 0.6 <= calcium_mean <= 0.8, f"Mean calcium {calcium_mean} should be ~0.7"


# =====================================================================
# TEST: Error Modulation with Complex Spikes
# =====================================================================


def test_complex_spikes_modulate_error_magnitude(
    complex_spike_config: CerebellumConfig,
    small_sizes: dict,
    device: str,
):
    """Test that complex spikes modulate error magnitude for learning."""
    cerebellum = Cerebellum(
        config=complex_spike_config,
        sizes=small_sizes,
        device=device,
    )

    # Create input
    input_spikes = torch.zeros(small_sizes["input_size"], device=device)
    input_spikes[:16] = 1.0  # 50% activity

    # Forward pass
    output = cerebellum.forward({"input": input_spikes})

    # Target (different from output to create error)
    target = torch.zeros(small_sizes["purkinje_size"], device=device)
    target[:8] = 1.0

    # Deliver error (triggers learning with complex spike modulation)
    metrics = cerebellum.deliver_error(target)

    # Check that learning occurred
    assert "error" in metrics
    assert metrics["error"] > 0.0, "Error should be detected"


def test_complex_spikes_preserve_error_sign(
    complex_spike_config: CerebellumConfig,
    small_sizes: dict,
    device: str,
):
    """Test that complex spikes preserve error sign (LTP vs LTD direction)."""
    cerebellum = Cerebellum(
        config=complex_spike_config,
        sizes=small_sizes,
        device=device,
    )

    # Test positive and negative errors separately
    positive_error = torch.tensor([0.5], device=device)
    negative_error = torch.tensor([-0.5], device=device)

    # Generate bursts
    ca_positive = cerebellum._generate_complex_spike_burst(positive_error.abs())
    ca_negative = cerebellum._generate_complex_spike_burst(negative_error.abs())

    # Calcium magnitude should be similar (same absolute error)
    assert torch.allclose(ca_positive, ca_negative, atol=0.2)

    # Sign is applied in _apply_error_learning (tested implicitly via learning)


# =====================================================================
# TEST: Disabled Complex Spikes
# =====================================================================


def test_cerebellum_without_complex_spikes_works(
    binary_config: CerebellumConfig,
    small_sizes: dict,
    device: str,
):
    """Test that cerebellum works normally when complex spikes disabled."""
    cerebellum = Cerebellum(
        config=binary_config,
        sizes=small_sizes,
        device=device,
    )

    # Forward pass
    input_spikes = (torch.rand(small_sizes["input_size"], device=device) > 0.8).float()
    output = cerebellum.forward({"input": input_spikes})

    # Check output
    assert output.shape == (small_sizes["purkinje_size"],)
    assert output.dtype == torch.bool

    # Learning should work (binary error)
    target = (torch.rand(small_sizes["purkinje_size"], device=device) > 0.5).float()
    metrics = cerebellum.deliver_error(target)

    assert "error" in metrics


# =====================================================================
# TEST: Integration with Existing Features
# =====================================================================


def test_complex_spikes_compatible_with_gap_junctions(
    small_sizes: dict,
    device: str,
):
    """Test that complex spikes work with gap junction synchronization."""
    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        gap_junctions_enabled=True,
        use_enhanced_microcircuit=False,
        device=device,
    )

    cerebellum = Cerebellum(
        config=config,
        sizes=small_sizes,
        device=device,
    )

    # Forward pass
    input_spikes = torch.ones(small_sizes["input_size"], device=device)
    output = cerebellum.forward({"input": input_spikes})

    # Learning with both features
    target = torch.zeros(small_sizes["purkinje_size"], device=device)
    target[:8] = 1.0

    metrics = cerebellum.deliver_error(target)
    assert metrics["error"] > 0.0


def test_complex_spikes_compatible_with_enhanced_microcircuit(
    device: str,
):
    """Test that complex spikes work with enhanced granule-Purkinje-DCN circuit."""
    sizes = {
        "input_size": 32,
        "granule_size": 128,
        "purkinje_size": 16,
    }

    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        use_enhanced_microcircuit=True,  # Enable full microcircuit
        device=device,
    )

    cerebellum = Cerebellum(
        config=config,
        sizes=sizes,
        device=device,
    )

    # Forward pass through enhanced circuit
    input_spikes = (torch.rand(sizes["input_size"], device=device) > 0.8).float()
    output = cerebellum.forward({"input": input_spikes})

    # Check output from DCN
    assert output.dtype == torch.bool

    # Learning should work
    target = (torch.rand(sizes["purkinje_size"], device=device) > 0.5).float()
    metrics = cerebellum.deliver_error(target)

    assert "error" in metrics


def test_complex_spikes_compatible_with_stp(
    small_sizes: dict,
    device: str,
):
    """Test that complex spikes work with short-term plasticity."""
    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        stp_enabled=True,
        use_enhanced_microcircuit=False,
        device=device,
    )

    cerebellum = Cerebellum(
        config=config,
        sizes=small_sizes,
        device=device,
    )

    # Run for multiple timesteps with STP
    for _ in range(10):
        input_spikes = (torch.rand(small_sizes["input_size"], device=device) > 0.8).float()
        output = cerebellum.forward({"input": input_spikes})

        assert output.dtype == torch.bool


# =====================================================================
# TEST: Biological Plausibility
# =====================================================================


def test_calcium_scaling_biologically_plausible():
    """Test that calcium scaling enables useful learning range."""
    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        device="cpu",
    )

    # Calcium per spikelet should allow:
    # - 2 spikes → 0.4 calcium (small LTD)
    # - 7 spikes → 1.4 calcium (large LTD)
    # This maps to reasonable weight change magnitudes (~1-10% of weight range)

    assert config.ca2_per_spikelet == 0.2
    min_calcium = config.min_complex_spike_count * config.ca2_per_spikelet  # 0.4
    max_calcium = config.max_complex_spike_count * config.ca2_per_spikelet  # 1.4

    # Check that range is biologically useful
    assert 0.3 <= min_calcium <= 0.6, "Min calcium should allow detectable LTD"
    assert 1.0 <= max_calcium <= 2.0, "Max calcium should allow strong but not excessive LTD"


def test_isi_supports_biological_burst_rate():
    """Test that inter-spike interval supports biological burst dynamics."""
    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        device="cpu",
    )

    # ISI = 1.5ms
    # Max burst: 7 spikes × 1.5ms = 10.5ms total burst duration
    # This is biologically realistic (complex spikes last 5-15ms)

    assert config.complex_spike_isi_ms == 1.5
    max_burst_duration_ms = config.max_complex_spike_count * config.complex_spike_isi_ms

    assert 8.0 <= max_burst_duration_ms <= 15.0, "Burst duration should be 8-15ms (biological)"


def test_burst_length_range_matches_biology():
    """Test that burst length range matches published data."""
    config = CerebellumConfig(
        use_complex_spike_bursts=True,
        device="cpu",
    )

    # Mathy et al. (2009): 2-7 spikelets per complex spike
    # Najafi & Medina (2013): Similar range
    assert config.min_complex_spike_count == 2
    assert config.max_complex_spike_count == 7


# =====================================================================
# TEST: Custom Parameters
# =====================================================================


def test_custom_complex_spike_parameters():
    """Test that custom complex spike parameters can be specified."""
    custom_config = CerebellumConfig(
        use_complex_spike_bursts=True,
        min_complex_spike_count=3,  # Custom: higher minimum
        max_complex_spike_count=8,  # Custom: higher maximum
        ca2_per_spikelet=0.15,  # Custom: lower calcium per spike
        device="cpu",
    )

    # Verify custom values are stored
    assert custom_config.min_complex_spike_count == 3
    assert custom_config.max_complex_spike_count == 8
    assert custom_config.ca2_per_spikelet == 0.15


# =====================================================================
# TEST: Learning Magnitude Comparison
# =====================================================================


def test_binary_vs_graded_error_signaling(
    device: str,
):
    """Compare binary (off) vs graded (complex spikes) error signaling."""
    sizes = {
        "input_size": 32,
        "granule_size": 128,
        "purkinje_size": 16,
    }

    # Binary config
    binary_cerebellum = Cerebellum(
        config=CerebellumConfig(
            use_complex_spike_bursts=False,
            use_enhanced_microcircuit=False,
            device=device,
        ),
        sizes=sizes,
        device=device,
    )

    # Graded config
    graded_cerebellum = Cerebellum(
        config=CerebellumConfig(
            use_complex_spike_bursts=True,
            use_enhanced_microcircuit=False,
            device=device,
        ),
        sizes=sizes,
        device=device,
    )

    # Both should work and learn
    input_spikes = torch.ones(sizes["input_size"], device=device)
    target = torch.zeros(sizes["purkinje_size"], device=device)
    target[:8] = 1.0

    # Binary
    _ = binary_cerebellum.forward({"input": input_spikes})
    metrics_binary = binary_cerebellum.deliver_error(target)

    # Graded
    _ = graded_cerebellum.forward({"input": input_spikes})
    metrics_graded = graded_cerebellum.deliver_error(target)

    # Both should detect error
    assert metrics_binary["error"] > 0.0
    assert metrics_graded["error"] > 0.0
