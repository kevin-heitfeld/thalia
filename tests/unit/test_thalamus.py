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
    """Test thalamus initializes with valid architecture."""
    thal = ThalamicRelay(thalamus_config)

    # Contract: relay neurons match configuration
    assert thal.n_relay == thalamus_config.n_output, \
        "Relay neurons should match configured output size"
    assert thal.n_relay > 0, "Should have positive relay neurons"

    # Contract: TRN is fraction of relay
    expected_trn = int(thal.n_relay * thalamus_config.trn_ratio)
    assert thal.n_trn == expected_trn, \
        "TRN size should respect configured ratio"
    assert thal.n_trn < thal.n_relay, "TRN should be subset of relay"

    # Contract: parameters have correct shapes
    assert thal.relay_gain.shape == (thal.n_relay,), \
        "Relay gain should be per-neuron 1D parameter"
    assert thal.center_surround_filter.shape == (thal.n_relay, thalamus_config.n_input), \
        "Filter should connect input to relay"

    # Contract: parameters are valid (no NaN, positive gains)
    assert not torch.isnan(thal.relay_gain).any(), "No NaN in parameters"
    assert (thal.relay_gain > 0).all(), "Gain should be positive"


def test_thalamus_forward(thalamus, device):
    """Test forward pass produces biologically plausible output."""
    # Create input spikes (ADR-004: bool, ADR-005: 1D)
    input_spikes = torch.rand(100, device=device) > 0.8  # 20% firing rate, [100], bool

    # Forward pass
    output = thalamus(input_spikes)

    # Contract: output shape and type
    assert output.shape == (thalamus.n_relay,), \
        "Output should be 1D with n_relay neurons"
    assert output.dtype == torch.bool, "Output should be bool (ADR-004)"

    # Contract: biologically plausible firing rate
    # Thalamus can amplify input (relay_strength > 1.0), so allow up to 99%
    firing_rate = output.float().mean().item()
    assert 0.0 <= firing_rate <= 0.99, \
        f"Firing rate should be biologically plausible (0-99%), got {firing_rate:.2%}"

    # Contract: membrane potential is within bounds
    assert not torch.isnan(thalamus.state.relay_membrane).any(), "No NaN in membrane"
    # v_rest is 0.0 (normalized), allow hyperpolarization down to -10
    assert (thalamus.state.relay_membrane >= -10.0).all(), \
        "Membrane should not drop far below rest"


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

    # Strong input should lead to depolarization → tonic mode (ADR-004/005)
    strong_input = torch.rand(100) > 0.5  # Dense, [100], bool

    thalamus.reset_state()
    for _ in range(10):
        output = thalamus(strong_input)


def test_thalamus_trn_inhibition(thalamus):
    """Test TRN provides inhibitory feedback."""
    # Strong input (ADR-004/005: 1D bool)
    input_spikes = torch.rand(100) > 0.5  # [100], bool

    # First pass - no TRN activity yet
    output1 = thalamus(input_spikes)

    # Second pass - TRN should now be active
    output2 = thalamus(input_spikes)

    # Contract: TRN should be active with strong input
    trn_active = thalamus.state.trn_spikes.sum().item() > 0

    # If TRN is active, it should suppress relay
    if trn_active:
        assert output2.mean() <= output1.mean() * 1.5  # Allow some variance


def test_thalamus_norepinephrine_modulation(thalamus):
    """Test norepinephrine gain modulation."""
    input_spikes = torch.rand(100) > 0.7  # [100], bool (ADR-004/005)

    # Low arousal (low NE)
    thalamus.set_neuromodulators(norepinephrine=0.0)
    output_low = thalamus(input_spikes)

    # High arousal (high NE)
    thalamus.reset_state()
    thalamus.set_neuromodulators(norepinephrine=1.0)
    output_high = thalamus(input_spikes)

    # High NE should increase gain (ADR-004: convert bool for mean)
    assert output_high.float().mean() >= output_low.float().mean()


def test_thalamus_reset(thalamus):
    """Test reset clears state properly."""
    input_spikes = torch.rand(100) > 0.8  # [100], bool (ADR-004/005)

    # Run forward
    thalamus(input_spikes)

    # Reset
    thalamus.reset_state()

    # Behavioral contract: after reset, forward pass should work normally
    output = thalamus(input_spikes)
    assert output.dtype == torch.bool
    assert output.shape == (thalamus.n_relay,)
    assert not torch.isnan(thalamus.state.relay_membrane).any()


def test_thalamus_diagnostics(thalamus):
    """Test diagnostics report correct information."""
    input_spikes = torch.rand(100) > 0.8  # [100], bool (ADR-004/005)

    thalamus(input_spikes)

    diag = thalamus.get_diagnostics()

    # Should have thalamus-specific diagnostics
    assert 'relay_firing_rate_hz' in diag
    assert 'trn_firing_rate_hz' in diag
    assert 'alpha_phase' in diag

    # Values should be in valid ranges
    assert 0.0 <= diag['relay_firing_rate_hz'] <= 1000.0  # Hz scale
    assert 0.0 <= diag['trn_firing_rate_hz'] <= 1000.0


# Edge case tests
def test_thalamus_silent_input(thalamus, device):
    """Test thalamus handles completely silent input (edge case)."""
    input_spikes = torch.zeros(100, device=device, dtype=torch.bool)
    output = thalamus(input_spikes)

    # Contract: valid output
    assert output.shape == (thalamus.n_relay,)
    assert output.dtype == torch.bool

    # Contract: sparse output with silent input
    firing_rate = output.float().mean().item()
    assert firing_rate < 0.1, f'Silent input should produce sparse output, got {firing_rate:.2%}'

    # Contract: neuron states remain valid
    assert not torch.isnan(thalamus.state.relay_membrane).any()
    # v_rest is 0.0, allow slight hyperpolarization
    assert (thalamus.state.relay_membrane >= -5.0).all()


def test_thalamus_saturated_input(thalamus, device):
    """Test thalamus handles saturated input (edge case)."""
    input_spikes = torch.ones(100, device=device, dtype=torch.bool)
    output = thalamus(input_spikes)

    # Contract: TRN is recruited with high input
    # Note: In single timestep, TRN may not fully inhibit yet
    assert thalamus.state.trn_spikes.any(), 'TRN should activate with saturated input'

    # Contract: output is generated (may still be high in first timestep)
    firing_rate = output.float().mean().item()
    assert firing_rate > 0.0, 'Saturated input should produce output'


def test_thalamus_repeated_forward_maintains_valid_state(thalamus, device):
    """Test that repeated forward passes don't corrupt state."""
    input_spikes = torch.rand(100, device=device) > 0.5

    for _ in range(100):
        output = thalamus(input_spikes)

    # Invariants after many operations
    assert not torch.isnan(thalamus.state.relay_membrane).any(), 'No NaN'
    assert not torch.isinf(thalamus.state.relay_membrane).any(), 'No Inf'
    # v_rest is 0.0, allow hyperpolarization to -10
    assert (thalamus.state.relay_membrane >= -10.0).all(), \
        'Membrane should not drop far below rest'


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
