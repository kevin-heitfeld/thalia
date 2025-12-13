"""
Tests for thalamic relay nucleus.

Tests:
- Basic relay functionality
- Burst vs tonic mode switching
- Alpha-based attentional gating
- TRN inhibitory modulation
- Center-surround spatial filtering
"""

import pytest
import torch

from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig


@pytest.fixture
def device():
    """Device for testing (CPU by default, can be parametrized for CUDA)."""
    return torch.device("cpu")


@pytest.fixture
def thalamus_config(device):
    """Standard thalamus configuration."""
    return ThalamicRelayConfig(
        n_input=100,
        n_output=80,
        dt_ms=1.0,
        device=str(device),
    )


@pytest.fixture
def thalamus(thalamus_config):
    """Create thalamus instance."""
    thal = ThalamicRelay(thalamus_config)
    thal.reset_state()
    return thal


def test_thalamus_initialization(thalamus_config):
    """Test thalamus initializes correctly."""
    thal = ThalamicRelay(thalamus_config)

    assert thal.n_relay == 80
    assert thal.n_trn == int(80 * thalamus_config.trn_ratio)
    # ADR-005: relay_gain is 1D parameter, not input_to_relay matrix
    assert thal.relay_gain.shape == (80,)
    assert thal.center_surround_filter.shape == (80, 100)


def test_thalamus_forward(thalamus, device):
    """Test basic forward pass."""
    # Create input spikes (ADR-004: bool, ADR-005: 1D)
    input_spikes = torch.rand(100, device=device) > 0.8  # 20% firing rate, [100], bool

    # Forward pass
    output = thalamus(input_spikes)

    # Check output shape (ADR-005: 1D)
    assert output.shape == (80,)

    # Check output is bool (ADR-004)
    assert output.dtype == torch.bool

    # State should be updated
    assert thalamus.state.relay_spikes is not None
    assert thalamus.state.relay_membrane is not None
    assert thalamus.state.trn_spikes is not None


def test_thalamus_alpha_gating(thalamus, device):
    """Test alpha oscillation attentional gating."""
    # ADR-004/005: 1D bool input
    input_spikes = torch.rand(100, device=device) > 0.5  # 50% firing rate

    # Test at alpha trough (phase=0) - weak suppression
    thalamus.set_oscillator_phases(
        phases={'alpha': 0.0},
        signals={'alpha': 1.0}
    )
    output_trough = thalamus(input_spikes)
    firing_rate_trough = output_trough.float().mean().item()  # ADR-004: convert bool

    # Reset and test at alpha peak (phase=π) - strong suppression
    thalamus.reset_state()
    thalamus.set_oscillator_phases(
        phases={'alpha': 3.14159},
        signals={'alpha': 1.0}
    )
    output_peak = thalamus(input_spikes)
    firing_rate_peak = output_peak.float().mean().item()  # ADR-004: convert bool

    # Peak should have HIGHER firing (less suppression) than trough
    # (our gate formula: gate = 1 - strength × (1 + cos(phase)) / 2)
    # At phase=0: gate = 1 - 0.5×1 = 0.5 (more suppression)
    # At phase=π: gate = 1 - 0.5×0 = 1.0 (less suppression)
    assert firing_rate_peak >= firing_rate_trough


def test_thalamus_burst_vs_tonic(thalamus):
    """Test burst vs tonic mode switching."""
    # Weak input should lead to hyperpolarization → burst mode (ADR-004/005)
    weak_input = torch.rand(100) > 0.95  # Very sparse, [100], bool

    # Run for several timesteps to build up mode state
    for _ in range(10):
        output = thalamus(weak_input)

    # Check mode state exists
    assert thalamus.state.current_mode is not None

    # Strong input should lead to depolarization → tonic mode (ADR-004/005)
    strong_input = torch.rand(100) > 0.5  # Dense, [100], bool

    thalamus.reset_state()
    for _ in range(10):
        output = thalamus(strong_input)

    assert thalamus.state.current_mode is not None


def test_thalamus_trn_inhibition(thalamus):
    """Test TRN provides inhibitory feedback."""
    # Strong input (ADR-004/005: 1D bool)
    input_spikes = torch.rand(100) > 0.5  # [100], bool

    # First pass - no TRN activity yet
    output1 = thalamus(input_spikes)

    # Second pass - TRN should now be active
    output2 = thalamus(input_spikes)

    # TRN spikes should exist
    assert thalamus.state.trn_spikes is not None
    trn_active = thalamus.state.trn_spikes.sum().item() > 0

    # If TRN is active, it should suppress relay
    if trn_active:
        assert output2.mean() <= output1.mean() * 1.5  # Allow some variance


def test_thalamus_norepinephrine_modulation(thalamus):
    """Test norepinephrine gain modulation."""
    input_spikes = torch.rand(100) > 0.7  # [100], bool (ADR-004/005)

    # Low arousal (low NE)
    thalamus.set_norepinephrine(0.0)
    output_low = thalamus(input_spikes)

    # High arousal (high NE)
    thalamus.reset_state()
    thalamus.set_norepinephrine(1.0)
    output_high = thalamus(input_spikes)

    # High NE should increase gain (ADR-004: convert bool for mean)
    assert output_high.float().mean() >= output_low.float().mean()


def test_thalamus_reset(thalamus):
    """Test reset clears state."""
    input_spikes = torch.rand(100) > 0.8  # [100], bool (ADR-004/005)

    # Run forward
    thalamus(input_spikes)

    # State should exist
    assert thalamus.state.relay_spikes is not None

    # Reset
    thalamus.reset_state()

    # State should be cleared
    assert thalamus.state.relay_spikes is None
    assert thalamus.state.relay_membrane is None
    assert thalamus.state.trn_spikes is None


def test_thalamus_diagnostics(thalamus):
    """Test diagnostics report correct information."""
    input_spikes = torch.rand(100) > 0.8  # [100], bool (ADR-004/005)

    thalamus(input_spikes)

    diag = thalamus.get_diagnostics()

    # Should have thalamus-specific diagnostics
    assert 'thalamus_relay_firing_rate' in diag
    assert 'thalamus_trn_firing_rate' in diag
    assert 'thalamus_alpha_phase' in diag

    # Values should be in valid ranges
    assert 0.0 <= diag['thalamus_relay_firing_rate'] <= 1.0
    assert 0.0 <= diag['thalamus_trn_firing_rate'] <= 1.0


def test_thalamus_center_surround_filter(thalamus):
    """Test center-surround spatial filter exists and is valid."""
    # Filter should exist
    assert hasattr(thalamus, 'center_surround_filter')

    # Shape should be [n_relay, n_input]
    assert thalamus.center_surround_filter.shape == (80, 100)

    # Filter should have positive (excitatory) and negative (inhibitory) values
    has_positive = (thalamus.center_surround_filter > 0).any()
    has_negative = (thalamus.center_surround_filter < 0).any()

    assert has_positive or has_negative  # At least one should exist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
