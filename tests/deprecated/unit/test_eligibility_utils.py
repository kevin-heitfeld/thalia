"""
Unit tests for EligibilityTraceManager.

Tests the consolidated eligibility trace computation utilities.
"""

import pytest
import torch

from thalia.core.eligibility_utils import EligibilityTraceManager, STDPConfig


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cpu')


@pytest.fixture
def config():
    """Get default STDP config."""
    return STDPConfig(
        stdp_tau_ms=20.0,
        eligibility_tau_ms=1000.0,
        stdp_lr=0.01,
        a_plus=1.0,
        a_minus=0.012,
        w_min=0.0,
        w_max=1.0,
        heterosynaptic_ratio=0.3,
    )


@pytest.fixture
def manager(config, device):
    """Create eligibility trace manager."""
    return EligibilityTraceManager(
        n_input=10,
        n_output=5,
        config=config,
        device=device,
    )


def test_initialization(manager):
    """Test manager initialization."""
    assert manager.n_input == 10
    assert manager.n_output == 5
    assert manager.input_trace.shape == (10,)
    assert manager.output_trace.shape == (5,)
    assert manager.eligibility.shape == (5, 10)
    assert torch.allclose(manager.input_trace, torch.zeros(10))
    assert torch.allclose(manager.output_trace, torch.zeros(5))
    assert torch.allclose(manager.eligibility, torch.zeros(5, 10))


def test_update_traces(manager, device):
    """Test trace updates."""
    input_spikes = torch.zeros(10, device=device)
    input_spikes[3] = 1.0
    input_spikes[7] = 1.0

    output_spikes = torch.zeros(5, device=device)
    output_spikes[1] = 1.0

    # First update
    manager.update_traces(input_spikes, output_spikes, dt_ms=1.0)

    # Check traces were updated
    assert manager.input_trace[3] > 0
    assert manager.input_trace[7] > 0
    assert manager.output_trace[1] > 0
    assert manager.input_trace[0] == 0  # No spike

    # Second update with decay
    input_spikes_2 = torch.zeros(10, device=device)
    output_spikes_2 = torch.zeros(5, device=device)

    manager.update_traces(input_spikes_2, output_spikes_2, dt_ms=1.0)

    # Traces should have decayed
    assert manager.input_trace[3] < 1.0
    assert manager.output_trace[1] < 1.0


def test_compute_stdp_eligibility(manager, device):
    """Test STDP eligibility computation."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.5
    manager.output_trace = torch.ones(5, device=device) * 0.8

    # Create weights
    weights = torch.ones(5, 10, device=device) * 0.5

    # Compute eligibility
    eligibility_update = manager.compute_stdp_eligibility(weights, lr_scale=1.0)

    # Check shape
    assert eligibility_update.shape == (5, 10)

    # With weights at midpoint (0.5), LTP and LTD should be balanced
    # Update should be positive (LTP dominates with a_plus=1.0 vs a_minus=0.012)
    assert torch.mean(eligibility_update) > 0


def test_compute_stdp_separate_ltd(manager, device):
    """Test STDP with separate LTP/LTD computation."""
    # Set up traces
    manager.input_trace = torch.ones(10, device=device) * 0.3
    manager.output_trace = torch.ones(5, device=device) * 0.7

    # Current spikes
    input_spikes = torch.zeros(10, device=device)
    input_spikes[2] = 1.0

    output_spikes = torch.zeros(5, device=device)
    output_spikes[1] = 1.0

    # Create weights
    weights = torch.ones(5, 10, device=device) * 0.3

    # Compute eligibility
    eligibility_update = manager.compute_stdp_eligibility_separate_ltd(
        input_spikes, output_spikes, weights, lr_scale=1.0
    )

    # Check shape
    assert eligibility_update.shape == (5, 10)

    # Row 1 (where output spiked) should have strongest update
    assert torch.abs(eligibility_update[1]).sum() > torch.abs(eligibility_update[0]).sum()


def test_accumulate_eligibility(manager, device):
    """Test eligibility accumulation."""
    # Create eligibility update
    eligibility_update = torch.ones(5, 10, device=device) * 0.1

    # Accumulate
    manager.accumulate_eligibility(eligibility_update, dt_ms=1.0)

    # Check eligibility increased
    assert torch.allclose(manager.eligibility, eligibility_update, atol=1e-5)

    # Accumulate again
    manager.accumulate_eligibility(eligibility_update, dt_ms=1.0)

    # Should have decayed previous + added new
    # New value should be > eligibility_update but < 2*eligibility_update (due to decay)
    assert torch.all(manager.eligibility > eligibility_update)
    assert torch.all(manager.eligibility < 2 * eligibility_update)


def test_decay_eligibility(manager, device):
    """Test eligibility decay."""
    # Set initial eligibility
    manager.eligibility = torch.ones(5, 10, device=device)

    # Decay
    manager.decay_eligibility(dt_ms=10.0)

    # Should have decayed
    assert torch.all(manager.eligibility < 1.0)
    assert torch.all(manager.eligibility >= 0.99)  # Small decay with tau=1000ms


def test_reset_traces(manager):
    """Test trace reset."""
    # Set traces to non-zero
    manager.input_trace = torch.ones(10)
    manager.output_trace = torch.ones(5)
    manager.eligibility = torch.ones(5, 10)

    # Reset
    manager.reset_traces()

    # All should be zero
    assert torch.allclose(manager.input_trace, torch.zeros(10))
    assert torch.allclose(manager.output_trace, torch.zeros(5))
    assert torch.allclose(manager.eligibility, torch.zeros(5, 10))


def test_reset_eligibility_only(manager):
    """Test eligibility-only reset."""
    # Set traces to non-zero
    manager.input_trace = torch.ones(10)
    manager.output_trace = torch.ones(5)
    manager.eligibility = torch.ones(5, 10)

    # Reset eligibility only
    manager.reset_eligibility()

    # Traces should be preserved
    assert torch.allclose(manager.input_trace, torch.ones(10))
    assert torch.allclose(manager.output_trace, torch.ones(5))
    assert torch.allclose(manager.eligibility, torch.zeros(5, 10))


def test_soft_bounds_prevent_saturation(manager, device):
    """Test that soft bounds prevent weight saturation."""
    # Set traces
    manager.input_trace = torch.ones(10, device=device)
    manager.output_trace = torch.ones(5, device=device)

    # Test at different weight levels
    weights_low = torch.ones(5, 10, device=device) * 0.1
    weights_mid = torch.ones(5, 10, device=device) * 0.5
    weights_high = torch.ones(5, 10, device=device) * 0.9

    elig_low = manager.compute_stdp_eligibility(weights_low, lr_scale=1.0)
    elig_mid = manager.compute_stdp_eligibility(weights_mid, lr_scale=1.0)
    elig_high = manager.compute_stdp_eligibility(weights_high, lr_scale=1.0)

    # LTP should be stronger when weights are low (far from w_max)
    assert torch.mean(elig_low) > torch.mean(elig_mid)
    assert torch.mean(elig_mid) > torch.mean(elig_high)


def test_device_transfer(manager):
    """Test moving manager to different device."""
    # Note: Only test if CUDA available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device('cuda:0')
    manager.to(device)

    assert manager.input_trace.device.type == 'cuda'
    assert manager.output_trace.device.type == 'cuda'
    assert manager.eligibility.device.type == 'cuda'


def test_bool_spike_handling(manager, device):
    """Test that manager handles bool spikes correctly."""
    # Create bool spikes
    input_spikes = torch.tensor([True, False, True, False, False, False, False, True, False, False], device=device)
    output_spikes = torch.tensor([False, True, False, False, True], device=device)

    # Update traces
    manager.update_traces(input_spikes, output_spikes, dt_ms=1.0)

    # Should have converted to float and updated traces
    assert manager.input_trace[0] > 0
    assert manager.input_trace[2] > 0
    assert manager.input_trace[7] > 0
    assert manager.output_trace[1] > 0
    assert manager.output_trace[4] > 0
