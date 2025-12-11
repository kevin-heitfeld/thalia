"""
Tests for EligibilityTraceManager.compute_ltp_ltd_separate() method.

This method returns raw LTP and LTD components without combining them,
allowing custom modulation by neuromodulators before combining.
"""

import torch
import pytest

from thalia.core.eligibility_utils import EligibilityTraceManager, STDPConfig


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def config():
    return STDPConfig(
        stdp_tau_ms=20.0,
        eligibility_tau_ms=1000.0,
        stdp_lr=0.01,
        a_plus=1.0,
        a_minus=0.5,
        w_min=0.0,
        w_max=1.0,
        heterosynaptic_ratio=0.3,
    )


@pytest.fixture
def manager(config, device):
    return EligibilityTraceManager(
        n_input=10,
        n_output=5,
        config=config,
        device=device,
    )


def test_compute_ltp_ltd_separate_basic(manager, device):
    """Test that LTP/LTD are computed correctly as separate components."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.5
    manager.output_trace = torch.ones(5, device=device) * 0.3

    # Create spikes
    input_spikes = torch.zeros(10, device=device)
    input_spikes[0] = 1.0  # One input spike

    output_spikes = torch.zeros(5, device=device)
    output_spikes[2] = 1.0  # One output spike

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # LTP should exist (output spike with input trace)
    assert isinstance(ltp, torch.Tensor)
    assert ltp.shape == (5, 10)

    # LTD should exist (input spike with output trace)
    assert isinstance(ltd, torch.Tensor)
    assert ltd.shape == (5, 10)

    # Check LTP: should be non-zero where output spiked (row 2)
    assert ltp[2, :].sum() > 0  # Row 2 should have non-zero values
    assert ltp[0, :].sum() == 0  # Row 0 should be zero (no output spike)

    # Check LTD: should be non-zero where input spiked (column 0)
    assert ltd[:, 0].sum() > 0  # Column 0 should have non-zero values
    assert ltd[:, 1].sum() == 0  # Column 1 should be zero (no input spike)


def test_compute_ltp_ltd_separate_no_spikes(manager, device):
    """Test that 0 is returned when no spikes occur."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.5
    manager.output_trace = torch.ones(5, device=device) * 0.3

    # No spikes
    input_spikes = torch.zeros(10, device=device)
    output_spikes = torch.zeros(5, device=device)

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # Both should be 0
    assert ltp == 0
    assert ltd == 0


def test_compute_ltp_ltd_separate_only_output_spikes(manager, device):
    """Test LTP computation when only output spikes."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.5
    manager.output_trace = torch.zeros(5, device=device)

    # Only output spikes
    input_spikes = torch.zeros(10, device=device)
    output_spikes = torch.ones(5, device=device)

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # LTP should exist
    assert isinstance(ltp, torch.Tensor)
    assert ltp.shape == (5, 10)

    # LTD should be 0 (no input spikes)
    assert ltd == 0


def test_compute_ltp_ltd_separate_only_input_spikes(manager, device):
    """Test LTD computation when only input spikes."""
    # Set up traces
    manager.input_trace = torch.zeros(10, device=device)
    manager.output_trace = torch.ones(5, device=device) * 0.3

    # Only input spikes
    input_spikes = torch.ones(10, device=device)
    output_spikes = torch.zeros(5, device=device)

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # LTP should be 0 (no output spikes)
    assert ltp == 0

    # LTD should exist
    assert isinstance(ltd, torch.Tensor)
    assert ltd.shape == (5, 10)


def test_compute_ltp_ltd_separate_bool_spikes(manager, device):
    """Test that bool spikes are handled correctly."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.5
    manager.output_trace = torch.ones(5, device=device) * 0.3

    # Create bool spikes
    input_spikes = torch.zeros(10, dtype=torch.bool, device=device)
    input_spikes[0] = True

    output_spikes = torch.zeros(5, dtype=torch.bool, device=device)
    output_spikes[2] = True

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # Should work with bool spikes
    assert isinstance(ltp, torch.Tensor)
    assert isinstance(ltd, torch.Tensor)
    assert ltp.dtype == torch.float32
    assert ltd.dtype == torch.float32


def test_compute_ltp_ltd_separate_amplitude_scaling(manager, device):
    """Test that a_plus and a_minus scale LTP/LTD correctly."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device)
    manager.output_trace = torch.ones(5, device=device)

    # All spikes
    input_spikes = torch.ones(10, device=device)
    output_spikes = torch.ones(5, device=device)

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # Check that amplitudes are applied
    # LTP should be scaled by a_plus (1.0)
    # LTD should be scaled by a_minus (0.5)
    assert isinstance(ltp, torch.Tensor)
    assert isinstance(ltd, torch.Tensor)

    # LTP magnitude should be approximately a_plus * (trace outer product)
    expected_ltp_val = manager.config.a_plus * 1.0  # trace * spike
    assert torch.allclose(ltp[0, 0], torch.tensor(expected_ltp_val), atol=1e-5)

    # LTD magnitude should be approximately a_minus * (trace outer product)
    expected_ltd_val = manager.config.a_minus * 1.0  # trace * spike
    assert torch.allclose(ltd[0, 0], torch.tensor(expected_ltd_val), atol=1e-5)


def test_compute_ltp_ltd_separate_no_soft_bounds_applied(manager, device):
    """Test that soft bounds are NOT applied (left for caller to apply)."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.5
    manager.output_trace = torch.ones(5, device=device) * 0.3

    # Create spikes
    input_spikes = torch.ones(10, device=device)
    output_spikes = torch.ones(5, device=device)

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # Values should be raw (not modulated by weight proximity)
    # All values should be equal since traces are uniform
    assert isinstance(ltp, torch.Tensor)
    assert isinstance(ltd, torch.Tensor)

    # Check uniformity (soft bounds would make values non-uniform based on weights)
    ltp_unique_values = torch.unique(ltp)
    assert len(ltp_unique_values) == 1  # All values should be the same

    ltd_unique_values = torch.unique(ltd)
    assert len(ltd_unique_values) == 1  # All values should be the same


def test_compute_ltp_ltd_separate_allows_custom_modulation(manager, device):
    """Test that separate LTP/LTD can be modulated independently."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.5
    manager.output_trace = torch.ones(5, device=device) * 0.3

    # Create spikes
    input_spikes = torch.ones(10, device=device)
    output_spikes = torch.ones(5, device=device)

    # Compute LTP/LTD
    ltp, ltd = manager.compute_ltp_ltd_separate(input_spikes, output_spikes)

    # Apply custom modulation (e.g., dopamine)
    dopamine = 0.5
    ltp_modulated = ltp * (1 + dopamine)  # Dopamine enhances LTP
    ltd_modulated = ltd * (1 - 0.5 * dopamine)  # Dopamine reduces LTD

    # Verify modulation was applied
    assert torch.all(ltp_modulated > ltp)
    assert torch.all(ltd_modulated < ltd)

    # Combine after modulation
    learning_rate = 0.01
    hetero_ratio = 0.3
    dw = learning_rate * (ltp_modulated - hetero_ratio * ltd_modulated)

    # Should have valid weight update
    assert dw.shape == (5, 10)
    assert torch.all(torch.isfinite(dw))
