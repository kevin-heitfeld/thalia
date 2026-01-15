"""
Tests for Striatum D1/D2 Pathway Delays (Temporal Competition)

Tests verify that:
1. D1 and D2 pathways have independent configurable delays
2. D1 "Go" signal arrives before D2 "No-Go" signal (temporal competition)
3. Delay buffers work correctly (circular buffer mechanics)
4. Action selection reflects proper temporal dynamics
5. Checkpoint save/restore works with delay buffers

Biological Rationale:
- D1 direct pathway: Striatum → GPi/SNr → Thalamus (~15ms)
- D2 indirect pathway: Striatum → GPe → STN → GPi/SNr → Thalamus (~25ms)
- D1 arrives ~10ms before D2, creating temporal competition window
- Explains impulsivity, action timing, and reaction time variability
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path for test imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.config.size_calculator import LayerSizeCalculator
from tests.utils.test_helpers import generate_sparse_spikes


@pytest.fixture
def device():
    """Device for testing (CPU by default, can be parametrized for CUDA)."""
    return torch.device("cpu")


@pytest.fixture
def striatum_sizes_with_delays():
    """Size configuration for striatum (4 actions, 1 neuron per action)."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=1)
    sizes['input_size'] = 50
    return sizes


@pytest.fixture
def striatum_config_with_delays(device):
    """Striatum configuration with explicit D1/D2 pathway delays."""
    return StriatumConfig(
        dt_ms=1.0,  # 1ms timesteps
        # Configure different delays for D1 and D2
        d1_to_output_delay_ms=15.0,  # D1 direct pathway: 15ms
        d2_to_output_delay_ms=25.0,  # D2 indirect pathway: 25ms (slower!)
        homeostasis_enabled=False,  # Disable for isolated delay testing
    )


@pytest.fixture
def striatum_config_no_delays(device):
    """Striatum configuration with zero delays."""
    return StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=0.0,  # No delay
        d2_to_output_delay_ms=0.0,  # No delay
        homeostasis_enabled=False,
    )


@pytest.fixture
def striatum_with_delays(striatum_config_with_delays, striatum_sizes_with_delays, device):
    """Create striatum instance with delays."""
    striatum = Striatum(config=striatum_config_with_delays, sizes=striatum_sizes_with_delays, device=device)
    striatum.add_input_source_striatum("default", striatum_sizes_with_delays['input_size'])
    striatum.reset_state()
    return striatum


@pytest.fixture
def striatum_no_delays(striatum_config_no_delays, striatum_sizes_with_delays, device):
    """Create striatum instance without delays."""
    striatum = Striatum(config=striatum_config_no_delays, sizes=striatum_sizes_with_delays, device=device)
    striatum.add_input_source_striatum("default", striatum_sizes_with_delays['input_size'])
    striatum.reset_state()
    return striatum


def test_delay_configuration_affects_temporal_competition(striatum_config_with_delays, striatum_sizes_with_delays, device):
    """Test that configured delays create temporal competition between D1 and D2.

    Why this test exists: Validates the core biological mechanism where D1 "Go"
    signals arrive before D2 "No-Go" signals, creating a temporal window for
    action initiation. This temporal competition is critical for reaction time,
    impulsivity, and decision-making dynamics.

    Expected behavior:
    - D1 vote arrives at t=15ms (direct pathway)
    - D2 vote arrives at t=25ms (indirect pathway)
    - 10ms window where D1 can initiate action before D2 inhibition
    """
    striatum = Striatum(config=striatum_config_with_delays, sizes=striatum_sizes_with_delays, device=device)
    striatum.add_input_source_striatum("default", striatum_sizes_with_delays['input_size'])
    striatum.reset_state()

    # Calculate expected delay difference from config
    dt_ms = striatum_config_with_delays.dt_ms
    d1_delay_steps = int(striatum_config_with_delays.d1_to_output_delay_ms / dt_ms)
    d2_delay_steps = int(striatum_config_with_delays.d2_to_output_delay_ms / dt_ms)
    expected_delay_diff = d2_delay_steps - d1_delay_steps  # Should be ~10 steps

    # Provide strong consistent input to activate both pathways
    input_spikes = torch.ones(50, dtype=torch.bool)

    # Track when each pathway first contributes votes
    d1_first_vote_time = None
    d2_first_vote_time = None

    for t in range(50):
        _ = striatum({"default": input_spikes})
        d1_votes, d2_votes = striatum.state_tracker.get_accumulated_votes()

        if d1_first_vote_time is None and d1_votes.sum() > 0:
            d1_first_vote_time = t
        if d2_first_vote_time is None and d2_votes.sum() > 0:
            d2_first_vote_time = t

        if d1_first_vote_time and d2_first_vote_time:
            break

    # Behavioral contract: D1 arrives before D2
    assert d1_first_vote_time is not None, "D1 should contribute votes"
    assert d2_first_vote_time is not None, "D2 should contribute votes"
    assert d1_first_vote_time < d2_first_vote_time, \
        "D1 (Go) should arrive before D2 (No-Go)"

    # Behavioral contract: delay difference matches configuration
    actual_delay_diff = d2_first_vote_time - d1_first_vote_time
    assert abs(actual_delay_diff - expected_delay_diff) <= 2, \
        f"Expected ~{expected_delay_diff}ms delay, got {actual_delay_diff}ms"


def test_delays_work_from_first_forward(striatum_with_delays, striatum_sizes_with_delays):
    """Test that delays take effect immediately from first forward pass."""
    # First forward pass with input
    input_spikes = generate_sparse_spikes(50, firing_rate=0.2)
    output = striatum_with_delays({"default": input_spikes})

    # Behavioral contract: output should be valid even on first pass
    assert output is not None
    # Striatum outputs D1+D2 spikes concatenated (not just n_actions)
    expected_shape = striatum_sizes_with_delays['d1_size'] + striatum_sizes_with_delays['d2_size']
    assert output.shape == (expected_shape,), f"Expected shape ({expected_shape},), got {output.shape}"
    assert output.dtype == torch.bool
    assert not torch.isnan(output).any(), "No NaN in output"

    # Behavioral contract: votes don't appear until delay passes
    d1_votes, d2_votes = striatum_with_delays.state_tracker.get_accumulated_votes()

    # On first pass, both should be zero (delays haven't passed yet)
    assert d1_votes.sum() == 0, "D1 votes should be zero before delay passes"
    assert d2_votes.sum() == 0, "D2 votes should be zero before delay passes"

    # Run enough steps for D1 delay to pass
    config = striatum_with_delays.config
    d1_delay_steps = int(config.d1_to_output_delay_ms / config.dt_ms)

    for _ in range(d1_delay_steps):
        _ = striatum_with_delays({"default": input_spikes})

    # Now D1 should have some votes
    d1_votes, d2_votes = striatum_with_delays.state_tracker.get_accumulated_votes()
    assert d1_votes.sum() > 0, "D1 votes should appear after delay"


def test_d1_arrives_before_d2(striatum_config_with_delays, striatum_sizes_with_delays, device):
    """Test that D1 votes arrive before D2 votes (temporal competition)."""
    striatum = Striatum(config=striatum_config_with_delays, sizes=striatum_sizes_with_delays, device=device)

    striatum.add_input_source_striatum("default", striatum_sizes_with_delays['input_size'])
    striatum.reset_state()

    # Create strong input that will activate D1 and D2
    input_spikes = torch.ones(50, dtype=torch.bool)  # Strong input

    # Calculate delay steps from PUBLIC config (not internal state)
    # This is the behavioral contract: delays configured via public API
    config = striatum.config
    d1_delay_steps = int(config.d1_to_output_delay_ms / config.dt_ms)  # 15
    d2_delay_steps = int(config.d2_to_output_delay_ms / config.dt_ms)  # 25

    # Run until D1 delay has passed (D1 should have votes, D2 should not)
    for _ in range(d1_delay_steps + 1):
        _ = striatum({"default": input_spikes})

    # Get accumulated votes via PUBLIC API (behavioral contract)
    d1_accumulated_after_d1_delay, d2_accumulated_after_d1_delay = striatum.state_tracker.get_accumulated_votes()

    # D1 should have accumulated votes by now (arrived)
    # D2 should still be zero or very small (not arrived yet)
    d1_total = d1_accumulated_after_d1_delay.sum().item()
    d2_total = d2_accumulated_after_d1_delay.sum().item()

    # D1 should have MORE votes than D2 at this intermediate point
    # because D1 arrives first (this is the key biological insight!)
    assert d1_total > d2_total, (
        f"D1 should have more votes than D2 after D1 delay passes. "
        f"D1 total: {d1_total}, D2 total: {d2_total}"
    )

    # Now run until D2 delay has also passed
    for _ in range(d2_delay_steps - d1_delay_steps):
        _ = striatum({"default": input_spikes})

    # Now both D1 and D2 should have accumulated votes (via PUBLIC API)
    d1_accumulated_final, d2_accumulated_final = striatum.state_tracker.get_accumulated_votes()

    assert d1_accumulated_final.sum().item() > 0
    assert d2_accumulated_final.sum().item() > 0


def test_zero_delays_produce_immediate_votes(striatum_no_delays):
    """Test that zero delays allow immediate vote accumulation."""
    # With zero delays, votes should appear immediately
    input_spikes = torch.ones(50, dtype=torch.bool)

    # Run single forward pass
    _ = striatum_no_delays({"default": input_spikes})

    # Behavioral contract: both D1 and D2 votes appear immediately
    d1_votes, d2_votes = striatum_no_delays.state_tracker.get_accumulated_votes()

    # With zero delay, both should have votes after first pass
    assert d1_votes.sum() > 0, "D1 votes should appear immediately with zero delay"
    assert d2_votes.sum() > 0, "D2 votes should appear immediately with zero delay"


def test_circular_buffer_wrapping(striatum_with_delays):
    """Test that circular buffer correctly wraps around.

    BEHAVIORAL CONTRACT: After running for many timesteps (more than buffer size),
    the striatum should still produce valid outputs with no crashes or NaN values.
    This validates buffer wrapping without accessing internal buffer state.
    """
    input_spikes = generate_sparse_spikes(50, firing_rate=0.2)

    # Calculate buffer size from PUBLIC config
    config = striatum_with_delays.config
    d1_delay_steps = int(config.d1_to_output_delay_ms / config.dt_ms)
    d2_delay_steps = int(config.d2_to_output_delay_ms / config.dt_ms)
    buffer_size = max(d1_delay_steps * 2 + 1, d2_delay_steps * 2 + 1)

    # Run for more than buffer size to force wrapping
    # BEHAVIORAL CONTRACT: No crashes, no NaN in outputs
    for _ in range(buffer_size + 10):
        _ = striatum_with_delays({"default": input_spikes})

    # Validate behavior: votes should still be valid (no NaN/inf)
    # This tests buffer health WITHOUT accessing internal buffer state
    d1_votes, d2_votes = striatum_with_delays.state_tracker.get_accumulated_votes()
    assert not torch.isnan(d1_votes).any(), "D1 votes contain NaN after buffer wrapping"
    assert not torch.isnan(d2_votes).any(), "D2 votes contain NaN after buffer wrapping"
    assert not torch.isinf(d1_votes).any(), "D1 votes contain Inf after buffer wrapping"
    assert not torch.isinf(d2_votes).any(), "D2 votes contain Inf after buffer wrapping"

    # Behavioral contract: striatum should still respond to new input
    # Run multiple timesteps with strong input to allow delays to deliver spikes
    strong_input = torch.ones(50, dtype=torch.bool)
    for _ in range(max(d1_delay_steps, d2_delay_steps) + 5):
        _ = striatum_with_delays({"default": strong_input})
    d1_after, d2_after = striatum_with_delays.state_tracker.get_accumulated_votes()
    # After strong input with sufficient delay time, at least one pathway should show increased votes
    assert d1_after.sum() > d1_votes.sum() or d2_after.sum() > d2_votes.sum(), \
        f"Striatum should still respond to input after buffer wrapping (d1: {d1_votes.sum():.1f} -> {d1_after.sum():.1f}, d2: {d2_votes.sum():.1f} -> {d2_after.sum():.1f})"


def test_checkpoint_preserves_delay_behavior(striatum_with_delays, striatum_config_with_delays, striatum_sizes_with_delays, device):
    """Test that checkpointing preserves D1/D2 temporal competition behavior."""
    # Run forward passes to build up delayed state
    input_spikes = torch.ones(50, dtype=torch.bool)
    for _ in range(30):  # Run past both delays
        _ = striatum_with_delays({"default": input_spikes})

    # Save checkpoint BEFORE seeing future behavior
    checkpoint = striatum_with_delays.checkpoint_manager.get_full_state()

    # Continue running original striatum to see future behavior
    for _ in range(10):
        _ = striatum_with_delays({"default": input_spikes})
    d1_votes_future, d2_votes_future = striatum_with_delays.state_tracker.get_accumulated_votes()

    # Create new striatum and restore checkpoint (back to 30-step state)
    striatum_restored = Striatum(config=striatum_config_with_delays, sizes=striatum_sizes_with_delays, device=device)
    striatum_restored.add_input_source("default", striatum_sizes_with_delays['input_size'])
    striatum_restored.reset_state()
    striatum_restored.checkpoint_manager.load_full_state(checkpoint)

    # Run restored striatum for same 10 steps
    for _ in range(10):
        _ = striatum_restored({"default": input_spikes})
    d1_votes_restored, d2_votes_restored = striatum_restored.state_tracker.get_accumulated_votes()

    # Behavioral contract: restored striatum should produce same future behavior
    # Votes should be similar (accounting for stochastic learning)
    # We check that the ratio of D1 to D2 is preserved within reasonable bounds
    d1_d2_ratio_expected = d1_votes_future.sum() / (d2_votes_future.sum() + 1e-6)
    d1_d2_ratio_restored = d1_votes_restored.sum() / (d2_votes_restored.sum() + 1e-6)

    # Allow tolerance for stochastic learning behavior
    assert abs(d1_d2_ratio_expected - d1_d2_ratio_restored) < 0.3, \
        f"Checkpoint should preserve D1/D2 vote ratio: {d1_d2_ratio_expected:.2f} vs {d1_d2_ratio_restored:.2f}"


def test_different_delay_values(device):
    """Test that different delay values produce different temporal dynamics."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=1)
    sizes['input_size'] = 50

    # Test with small delay difference
    config_small_diff = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=10.0,  # 10ms
        d2_to_output_delay_ms=12.0,  # 12ms (only 2ms difference)
        homeostasis_enabled=False,
    )
    striatum_small = Striatum(config=config_small_diff, sizes=sizes, device=device)

    # Test with large delay difference
    config_large_diff = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=10.0,  # 10ms
        d2_to_output_delay_ms=30.0,  # 30ms (20ms difference!)
        homeostasis_enabled=False,
    )
    striatum_small = Striatum(config=config_small_diff, sizes=sizes, device=device)
    striatum_small.add_input_source("default", sizes['input_size'])

    # Test with large delay difference
    config_large_diff = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=10.0,  # 10ms
        d2_to_output_delay_ms=30.0,  # 30ms (20ms difference!)
        homeostasis_enabled=False,
    )
    striatum_large = Striatum(config=config_large_diff, sizes=sizes, device=device)
    striatum_large.add_input_source("default", sizes['input_size'])

    # Test behavioral difference: larger delay difference creates longer competition window
    input_spikes = torch.ones(50, dtype=torch.bool)

    # For small difference: measure D1→D2 arrival gap
    striatum_small.reset_state()
    d1_time_small, d2_time_small = None, None
    for t in range(50):
        _ = striatum_small({"default": input_spikes})
        d1_votes, d2_votes = striatum_small.state_tracker.get_accumulated_votes()
        if d1_time_small is None and d1_votes.sum() > 0:
            d1_time_small = t
        if d2_time_small is None and d2_votes.sum() > 0:
            d2_time_small = t
        if d1_time_small and d2_time_small:
            break

    # For large difference: measure D1→D2 arrival gap
    striatum_large.reset_state()
    d1_time_large, d2_time_large = None, None
    for t in range(50):
        _ = striatum_large({"default": input_spikes})
        d1_votes, d2_votes = striatum_large.state_tracker.get_accumulated_votes()
        if d1_time_large is None and d1_votes.sum() > 0:
            d1_time_large = t
        if d2_time_large is None and d2_votes.sum() > 0:
            d2_time_large = t
        if d1_time_large and d2_time_large:
            break

    # Behavioral contract: larger configured delay produces larger temporal gap
    small_gap = d2_time_small - d1_time_small
    large_gap = d2_time_large - d1_time_large
    assert large_gap > small_gap, \
        f"Larger delay config should create larger temporal gap: {large_gap} vs {small_gap}"
    # Expected: small_gap ≈ 2ms, large_gap ≈ 20ms
    assert abs(small_gap - 2) <= 1, f"Small gap should be ~2ms, got {small_gap}"
    assert abs(large_gap - 20) <= 2, f"Large gap should be ~20ms, got {large_gap}"


def test_population_coding_with_delays(device):
    """Test that delays work correctly with population coding enabled."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=10)  # Population coding
    sizes['input_size'] = 50

    config_pop = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=15.0,
        d2_to_output_delay_ms=25.0,
        homeostasis_enabled=False,
    )
    striatum_pop = Striatum(config=config_pop, sizes=sizes, device=device)
    striatum_pop.add_input_source("default", sizes['input_size'])
    striatum_pop.reset_state()

    # Run forward pass
    input_spikes = generate_sparse_spikes(50, firing_rate=0.2)
    _ = striatum_pop({"default": input_spikes})

    # Behavioral contract: population coding aggregates to action-level votes
    # Verify votes are tracked per action (4 actions), not per neuron (40 neurons)
    d1_votes, d2_votes = striatum_pop.state_tracker.get_accumulated_votes()

    # Contract: vote tensors should be action-sized
    assert d1_votes.shape == (4,), f"D1 votes should be per-action, got shape {d1_votes.shape}"
    assert d2_votes.shape == (4,), f"D2 votes should be per-action, got shape {d2_votes.shape}"


def test_reset_clears_delay_state(device):
    """Test that state reset clears accumulated delay effects."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=1)
    sizes['input_size'] = 50

    config = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=15.0,
        d2_to_output_delay_ms=25.0,
        homeostasis_enabled=False,
    )
    striatum = Striatum(config=config, sizes=sizes, device=device)
    striatum.add_input_source_striatum("default", sizes['input_size'])

    # Run forward passes to accumulate votes
    input_spikes = torch.ones(50, dtype=torch.bool)
    for _ in range(30):
        _ = striatum({"default": input_spikes})

    # Votes should be accumulated
    d1_votes_before, d2_votes_before = striatum.state_tracker.get_accumulated_votes()
    assert d1_votes_before.sum() > 0 or d2_votes_before.sum() > 0, \
        "Should have accumulated votes before reset"

    # Reset state
    striatum.reset_state()

    # Behavioral contract: reset should clear accumulated votes
    d1_votes_after, d2_votes_after = striatum.state_tracker.get_accumulated_votes()
    assert d1_votes_after.sum() == 0, "D1 votes should be cleared after reset"
    assert d2_votes_after.sum() == 0, "D2 votes should be cleared after reset"


def test_striatum_silent_input(device):
    """Test striatum handles completely silent input (edge case)."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=1)
    sizes['input_size'] = 50

    config = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=15.0,
        d2_to_output_delay_ms=25.0,
    )
    striatum = Striatum(config=config, sizes=sizes, device=device)
    striatum.add_input_source_striatum("default", sizes['input_size'])

    # All-zero input
    input_spikes = torch.zeros(50, dtype=torch.bool)

    # Run multiple steps
    for _ in range(50):
        output = striatum({"default": input_spikes})

    # Contract: should not crash, produce valid output (D1+D2 neurons = 8)
    assert output.shape == (8,), f"Expected (8,), got {output.shape}"
    assert output.dtype == torch.bool

    # Contract: votes remain valid (no NaN/Inf)
    d1_votes, d2_votes = striatum.state_tracker.get_accumulated_votes()
    assert not torch.isnan(d1_votes).any(), "D1 votes should not have NaN"
    assert not torch.isnan(d2_votes).any(), "D2 votes should not have NaN"


def test_striatum_saturated_input(device):
    """Test striatum handles saturated input without corruption (edge case)."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=1)
    sizes['input_size'] = 50

    config = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=15.0,
        d2_to_output_delay_ms=25.0,
    )
    striatum = Striatum(config=config, sizes=sizes, device=device)
    striatum.add_input_source_striatum("default", sizes['input_size'])

    # All neurons firing
    input_spikes = torch.ones(50, dtype=torch.bool)

    # Run multiple steps with maximum input
    for _ in range(50):
        output = striatum({"default": input_spikes})

    # Contract: should produce valid output without saturation (D1+D2 neurons = 8)
    assert output.shape == (8,), f"Expected (8,), got {output.shape}"
    assert output.dtype == torch.bool

    # Contract: votes remain valid
    d1_votes, d2_votes = striatum.state_tracker.get_accumulated_votes()
    assert not torch.isnan(d1_votes).any(), "D1 votes should not have NaN"
    assert not torch.isnan(d2_votes).any(), "D2 votes should not have NaN"


def test_striatum_extreme_dopamine(device):
    """Test striatum handles extreme dopamine without NaN/Inf (edge case)."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=1)
    sizes['input_size'] = 50

    config = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=15.0,
        d2_to_output_delay_ms=25.0,
    )
    striatum = Striatum(config=config, sizes=sizes, device=device)

    input_spikes = generate_sparse_spikes(50, firing_rate=0.5)

    # Set extreme dopamine (at valid upper boundary)
    striatum.set_neuromodulators(dopamine=2.0)

    # Run forward pass
    output = striatum({"default": input_spikes})

    # Contract: extreme neuromodulation shouldn't cause numerical issues
    assert not torch.isnan(output.float()).any(), "Output should not have NaN"
    d1_votes, d2_votes = striatum.state_tracker.get_accumulated_votes()
    assert not torch.isnan(d1_votes).any(), "D1 votes should not have NaN"
    assert not torch.isnan(d2_votes).any(), "D2 votes should not have NaN"


def test_striatum_repeated_forward_numerical_stability(device):
    """Test that repeated forward passes maintain numerical stability."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=1)
    sizes['input_size'] = 50

    config = StriatumConfig(
        dt_ms=1.0,
        d1_to_output_delay_ms=15.0,
        d2_to_output_delay_ms=25.0,
    )
    striatum = Striatum(config=config, sizes=sizes, device=device)
    striatum.add_input_source_striatum("default", sizes['input_size'])

    input_spikes = generate_sparse_spikes(50, firing_rate=0.5)

    # Run many forward passes
    for _ in range(200):
        _output = striatum({"default": input_spikes})

    # Contract: long-term operation shouldn't cause corruption
    d1_votes, d2_votes = striatum.state_tracker.get_accumulated_votes()
    assert not torch.isnan(d1_votes).any(), "D1 votes should not have NaN"
    assert not torch.isnan(d2_votes).any(), "D2 votes should not have NaN"
    assert not torch.isinf(d1_votes).any(), "D1 votes should not have Inf"
    assert not torch.isinf(d2_votes).any(), "D2 votes should not have Inf"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
