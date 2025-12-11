"""
Integration test for refactored eligibility trace utilities.

Tests that the refactored striatum and pathways still work correctly
with the consolidated EligibilityTraceManager.
"""

import pytest
import torch

from thalia.regions.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig
from thalia.regions.base import RegionConfig


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cpu')


@pytest.fixture
def striatum(device):
    """Create a small striatum for testing."""
    config = StriatumConfig(
        n_input=8,
        n_output=4,  # 4 actions, 1 neuron per action
        dt_ms=1.0,
        device=str(device),
        eligibility_tau_ms=100.0,
        stdp_tau_ms=20.0,
        stdp_lr=0.01,
        d1_lr_scale=1.0,
        d2_lr_scale=1.0,
        population_coding=False,  # Simple 1 neuron per action
        homeostatic_enabled=False,  # Disable for simple test
        ucb_exploration=False,  # Disable exploration
    )

    return Striatum(config=config)


def test_refactored_trace_manager_properties(striatum):
    """Test that trace manager properties work correctly."""
    # Access traces through backward compatibility properties
    d1_input_trace = striatum.d1_input_trace
    d1_output_trace = striatum.d1_output_trace
    d1_eligibility = striatum.d1_eligibility

    d2_input_trace = striatum.d2_input_trace
    d2_output_trace = striatum.d2_output_trace
    d2_eligibility = striatum.d2_eligibility

    # Check shapes
    assert d1_input_trace.shape == (8,)
    assert d1_output_trace.shape == (4,)
    assert d1_eligibility.shape == (4, 8)

    assert d2_input_trace.shape == (8,)
    assert d2_output_trace.shape == (4,)
    assert d2_eligibility.shape == (4, 8)

    # All should be zero initially
    assert torch.allclose(d1_input_trace, torch.zeros(8))
    assert torch.allclose(d1_eligibility, torch.zeros(4, 8))


def test_refactored_forward_updates_traces(striatum, device):
    """Test that forward pass updates traces correctly."""
    # Create input spikes
    input_spikes = torch.zeros(8, device=device)
    input_spikes[2] = 1.0
    input_spikes[5] = 1.0

    # Forward pass
    output_spikes = striatum(input_spikes)

    # Traces should be updated (non-zero after forward)
    # Note: We don't test exact values since that's implementation-dependent
    # Just verify the mechanism works
    d1_input_trace = striatum.d1_input_trace

    # At least the input trace should have been updated
    assert d1_input_trace.sum() > 0


def test_refactored_eligibility_updates(striatum, device):
    """Test that eligibility traces accumulate correctly."""
    # Run several timesteps to accumulate eligibility
    input_spikes = torch.ones(8, device=device)

    for _ in range(10):
        output_spikes = striatum(input_spikes)

    # Eligibility should have accumulated
    d1_elig = striatum.d1_eligibility
    d2_elig = striatum.d2_eligibility

    # At least one pathway should have non-zero eligibility
    total_elig = d1_elig.abs().sum() + d2_elig.abs().sum()
    assert total_elig > 0


def test_refactored_reward_delivery(striatum, device):
    """Test that reward delivery modulates weights using eligibility."""
    # Run forward passes to build eligibility
    input_spikes = torch.ones(8, device=device)

    for _ in range(5):
        output_spikes = striatum(input_spikes)

    # Select an action
    action = striatum.finalize_action()

    # Get initial weights
    d1_weights_before = striatum.d1_weights.clone()
    d2_weights_before = striatum.d2_weights.clone()

    # Deliver positive reward
    striatum.deliver_reward(reward=1.0)

    # Weights should have changed
    d1_weights_after = striatum.d1_weights
    d2_weights_after = striatum.d2_weights

    # At least one pathway should show weight changes
    d1_changed = not torch.allclose(d1_weights_before, d1_weights_after, atol=1e-6)
    d2_changed = not torch.allclose(d2_weights_before, d2_weights_after, atol=1e-6)

    assert d1_changed or d2_changed, "Weights should change after reward delivery"


def test_refactored_trace_reset(striatum):
    """Test that trace manager's reset method clears traces."""
    # Set traces to non-zero
    striatum.d1_pathway._trace_manager.input_trace[:] = 1.0
    striatum.d1_pathway._trace_manager.output_trace[:] = 1.0
    striatum.d1_pathway._trace_manager.eligibility[:] = 1.0

    # Reset using trace manager directly
    striatum.d1_pathway._trace_manager.reset_traces()

    # All traces should be zero
    assert torch.allclose(striatum.d1_input_trace, torch.zeros(8))
    assert torch.allclose(striatum.d1_output_trace, torch.zeros(4))
    assert torch.allclose(striatum.d1_eligibility, torch.zeros(4, 8))


def test_refactored_manager_device_consistency(striatum, device):
    """Test that trace manager is on correct device."""
    assert striatum.d1_pathway._trace_manager.device == device
    assert striatum.d2_pathway._trace_manager.device == device

    assert striatum.d1_pathway._trace_manager.input_trace.device == device
    assert striatum.d1_pathway._trace_manager.output_trace.device == device
    assert striatum.d1_pathway._trace_manager.eligibility.device == device


def test_refactored_pathway_update_eligibility(striatum, device):
    """Test pathway's update_eligibility method uses trace manager."""
    input_spikes = torch.zeros(8, device=device)
    input_spikes[3] = 1.0

    output_spikes = torch.zeros(4, device=device)
    output_spikes[1] = 1.0

    # Call pathway update directly
    striatum.d1_pathway.update_eligibility(input_spikes, output_spikes, dt_ms=1.0)

    # Traces should be updated
    assert striatum.d1_input_trace[3] > 0
    assert striatum.d1_output_trace[1] > 0


def test_refactored_soft_bounds_still_work(striatum, device):
    """Test that soft bounds are still applied via trace manager."""
    # Set weights near max
    with torch.no_grad():
        striatum.d1_weights[:] = 0.95

    # Build eligibility with high activity
    input_spikes = torch.ones(8, device=device)

    for _ in range(10):
        output_spikes = striatum(input_spikes)

    # Select action and reward
    action = striatum.finalize_action()
    striatum.deliver_reward(reward=10.0)  # Large reward

    # Weights should be clamped at w_max (1.0)
    assert torch.all(striatum.d1_weights <= 1.0)
    assert torch.all(striatum.d1_weights >= 0.0)
