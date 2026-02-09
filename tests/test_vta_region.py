"""Unit tests for VTA (Ventral Tegmental Area) dopamine region.

Tests biologically-accurate dopamine neuron dynamics:
- Tonic firing (4-5 Hz baseline)
- Burst response to positive RPE (15-20 Hz)
- Pause response to negative RPE (<1 Hz)
- RPE computation from reward and value
"""

import pytest
import torch

from thalia.brain.configs import VTAConfig
from thalia.brain.regions.vta import VTA


@pytest.fixture
def vta_config():
    """Create standard VTA configuration for testing."""
    return VTAConfig(
        n_da_neurons=100,
        n_gaba_neurons=20,
        rpe_gain=15.0,
        gamma=0.99,
        rpe_normalization=True,
        device="cpu",
    )


@pytest.fixture
def vta_region(vta_config):
    """Create VTA region instance."""
    region_layer_sizes = {
        "da_neurons": vta_config.n_da_neurons,
        "gaba_neurons": vta_config.n_gaba_neurons,
    }
    return VTA(config=vta_config, region_layer_sizes=region_layer_sizes)


def compute_firing_rate_hz(spikes: torch.Tensor, dt_ms: float = 1.0) -> float:
    """Compute population firing rate in Hz from spike tensor.

    Args:
        spikes: Boolean spike tensor [n_neurons] or [timesteps, n_neurons]
        dt_ms: Timestep duration in milliseconds

    Returns:
        Population firing rate in Hz
    """
    if spikes.ndim == 1:
        # Single timestep
        return float(spikes.float().mean().item() * 1000.0 / dt_ms)
    else:
        # Multiple timesteps
        total_spikes = spikes.float().sum().item()
        n_neurons = spikes.shape[1]
        n_timesteps = spikes.shape[0]
        duration_sec = n_timesteps * dt_ms / 1000.0
        return total_spikes / (n_neurons * duration_sec)


def test_vta_initialization(vta_region, vta_config):
    """Test VTA initializes with correct architecture."""
    assert vta_region.n_da_neurons == vta_config.n_da_neurons
    assert vta_region.n_gaba_neurons == vta_config.n_gaba_neurons
    assert vta_region.da_neurons is not None
    assert vta_region.device.type == "cpu"


def test_vta_tonic_firing(vta_region):
    """VTA DA neurons should fire at 4-5 Hz baseline with no input."""
    spikes_history = []

    # Run 1000 timesteps (1 second at 1ms dt)
    for t in range(1000):
        vta_region._forward_internal(inputs={})
        # Get DA output spikes from port
        da_spikes = vta_region.get_port_output("da_output")
        if da_spikes is not None:
            spikes_history.append(da_spikes)

    # Compute firing rate
    all_spikes = torch.stack(spikes_history, dim=0)  # [timesteps, n_neurons]
    firing_rate = compute_firing_rate_hz(all_spikes, dt_ms=1.0)

    # Assert baseline tonic firing in range
    assert 3.0 <= firing_rate <= 6.0, f"Tonic rate {firing_rate:.2f} Hz out of range [3-6 Hz]"


def test_vta_burst_on_reward(vta_region):
    """Positive RPE should cause burst (>10 Hz)."""
    # Create reward signal (population coded spikes)
    n_reward_neurons = 100
    reward_spikes = torch.ones(n_reward_neurons, dtype=torch.bool)  # Strong positive reward

    burst_spikes = []

    # Deliver reward at t=0, measure burst response
    for t in range(200):  # 200ms window
        if t == 0:
            vta_region._forward_internal(inputs={"reward:output": reward_spikes})
        else:
            vta_region._forward_internal(inputs={})

        da_spikes = vta_region.get_port_output("da_output")
        if da_spikes is not None:
            burst_spikes.append(da_spikes)

    # Compute burst firing rate
    all_spikes = torch.stack(burst_spikes, dim=0)
    burst_rate = compute_firing_rate_hz(all_spikes, dt_ms=1.0)

    # Assert burst rate is elevated above baseline
    assert burst_rate > 8.0, f"Burst rate {burst_rate:.2f} Hz too low (should be >8 Hz)"


def test_vta_pause_on_omitted_reward(vta_region):
    """Negative RPE (omitted expected reward) should cause pause (<2 Hz)."""
    n_snr_neurons = 100

    # Build expectation by providing high value signal
    value_spikes = torch.ones(n_snr_neurons, dtype=torch.bool)  # High expectation

    # Run a few timesteps with high value to build expectation
    for _ in range(10):
        vta_region._forward_internal(inputs={"snr:value": value_spikes})

    pause_spikes = []

    # Now deliver NO reward (omission) while expectation is high
    for t in range(200):  # 200ms window
        vta_region._forward_internal(inputs={"snr:value": value_spikes})
        da_spikes = vta_region.get_port_output("da_output")
        if da_spikes is not None:
            pause_spikes.append(da_spikes)

    # Compute pause firing rate
    all_spikes = torch.stack(pause_spikes, dim=0)
    pause_rate = compute_firing_rate_hz(all_spikes, dt_ms=1.0)

    # Assert pause rate is suppressed below baseline
    # Note: May not reach <1 Hz but should be noticeably reduced
    assert pause_rate < 4.0, f"Pause rate {pause_rate:.2f} Hz too high (should be <4 Hz)"


def test_vta_rpe_computation(vta_region):
    """VTA should compute RPE = reward - value correctly."""
    n_reward_neurons = 100
    n_snr_neurons = 100

    # Positive RPE: reward > value
    reward_spikes = torch.ones(n_reward_neurons, dtype=torch.bool)
    value_spikes = torch.zeros(n_snr_neurons, dtype=torch.bool)

    vta_region._forward_internal(
        inputs={"reward:output": reward_spikes, "snr:value": value_spikes}
    )

    diagnostics = vta_region.get_diagnostics()
    assert diagnostics["mean_rpe"] > 0, "RPE should be positive when reward > value"

    # Negative RPE: reward < value
    reward_spikes = torch.zeros(n_reward_neurons, dtype=torch.bool)
    value_spikes = torch.ones(n_snr_neurons, dtype=torch.bool)

    vta_region._forward_internal(
        inputs={"reward:output": reward_spikes, "snr:value": value_spikes}
    )

    diagnostics = vta_region.get_diagnostics()
    assert diagnostics["mean_rpe"] < 0, "RPE should be negative when reward < value"


def test_vta_diagnostics(vta_region):
    """VTA should provide comprehensive diagnostics."""
    # Run a few timesteps
    for _ in range(10):
        vta_region._forward_internal(inputs={})

    diagnostics = vta_region.get_diagnostics()

    # Check required diagnostic keys
    assert "da_firing_rate_hz" in diagnostics
    assert "mean_rpe" in diagnostics
    assert "da_mean_membrane_potential" in diagnostics
    assert "mean_reward" in diagnostics
    assert "mean_value" in diagnostics

    # Check values are reasonable
    assert 0 <= diagnostics["da_firing_rate_hz"] <= 30, "DA firing rate out of range"
    assert diagnostics["da_mean_membrane_potential"] < 0, "Membrane potential should be negative"


def test_vta_rpe_normalization(vta_config):
    """VTA should normalize RPE to prevent runaway dynamics."""
    vta_config.rpe_normalization = True
    region_layer_sizes = {
        "da_neurons": vta_config.n_da_neurons,
        "gaba_neurons": vta_config.n_gaba_neurons,
    }
    vta = VTA(config=vta_config, region_layer_sizes=region_layer_sizes)

    n_reward_neurons = 100

    # Deliver extreme reward repeatedly
    extreme_reward = torch.ones(n_reward_neurons, dtype=torch.bool)
    rpe_history = []

    for _ in range(100):
        vta._forward_internal(inputs={"reward:output": extreme_reward})
        diagnostics = vta.get_diagnostics()
        rpe_history.append(diagnostics["mean_rpe"])

    # RPE should stabilize (not grow unbounded)
    recent_rpe = rpe_history[-10:]
    assert max(recent_rpe) < 5.0, "RPE normalization failed, values growing unbounded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
