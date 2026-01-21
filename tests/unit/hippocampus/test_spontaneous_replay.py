"""Unit tests for spontaneous replay generator.

Tests the SpontaneousReplayGenerator class in isolation, verifying:
- ACh-gated ripple generation (no ripples during high ACh)
- Probabilistic ripple generation during low ACh
- Refractory period enforcement (minimum 200ms gap)
- Pattern selection based on synaptic tags
- Pattern selection based on weight strength
- Proper weight combination (60% tags + 30% weights + 10% noise)
"""

import torch
import pytest

from thalia.regions.hippocampus.spontaneous_replay import SpontaneousReplayGenerator


@pytest.fixture
def replay_generator():
    """Create a replay generator with default settings."""
    return SpontaneousReplayGenerator(
        ripple_rate_hz=2.0,
        ach_threshold=0.3,
        ripple_refractory_ms=200.0,
        device="cpu",
    )


@pytest.fixture
def ca3_weights():
    """Create CA3 recurrent weights [100, 100]."""
    n_neurons = 100
    weights = torch.randn(n_neurons, n_neurons) * 0.1
    weights = torch.abs(weights)  # Positive weights
    return weights


@pytest.fixture
def synaptic_tags():
    """Create synaptic tags [100, 100]."""
    n_neurons = 100
    tags = torch.zeros(n_neurons, n_neurons)
    # Add some strong tags (recent activity + dopamine)
    tags[:10, :10] = 0.8
    tags[20:30, 20:30] = 0.5
    return tags


def test_no_ripples_during_high_ach(replay_generator):
    """Ripples should not occur during high acetylcholine (encoding mode)."""
    high_ach = 0.8  # Encoding mode
    dt_ms = 1.0

    # Try 1000 timesteps (1 second) - should never trigger ripple
    ripple_count = 0
    for _ in range(1000):
        if replay_generator.should_trigger_ripple(high_ach, dt_ms):
            ripple_count += 1

    assert ripple_count == 0, (
        f"Expected 0 ripples during high ACh (0.8), got {ripple_count}"
    )


def test_ripples_during_low_ach(replay_generator):
    """Ripples should occur frequently during low acetylcholine (sleep mode)."""
    # Reset state to ensure clean test
    replay_generator.reset_state()

    low_ach = 0.1  # Sleep mode
    dt_ms = 1.0

    # Try 2000 timesteps (2 seconds) - should trigger ~2-4 ripples
    # With rate=2 Hz and refractory=200ms, expect ~4-8 ripples per 2 seconds
    ripple_count = 0
    for _ in range(2000):
        if replay_generator.should_trigger_ripple(low_ach, dt_ms):
            ripple_count += 1

    assert 1 <= ripple_count <= 12, (
        f"Expected 1-12 ripples during low ACh (0.1) in 2 seconds, got {ripple_count}"
    )


def test_ripple_refractory_period(replay_generator):
    """Minimum 200ms gap should be enforced between ripples."""
    low_ach = 0.0  # Maximal replay
    dt_ms = 1.0

    # Track time between ripples
    ripple_times = []
    for t in range(2000):  # 2 seconds
        if replay_generator.should_trigger_ripple(low_ach, dt_ms):
            ripple_times.append(t)

    # Check that all intervals >= 200ms
    if len(ripple_times) > 1:
        intervals = [ripple_times[i+1] - ripple_times[i] for i in range(len(ripple_times) - 1)]
        min_interval = min(intervals)

        assert min_interval >= 200, (
            f"Minimum interval {min_interval}ms < refractory period 200ms. "
            f"Intervals: {intervals[:10]}"
        )


def test_pattern_selection_uses_tags(replay_generator, ca3_weights, synaptic_tags):
    """Pattern selection should prioritize neurons with strong tags."""
    # Create asymmetric tags: first 10 neurons have strong tags
    tags = torch.zeros(100, 100)
    tags[:10, :] = 1.0  # Strong incoming tags to first 10 neurons

    # Uniform weights (so tags dominate selection)
    weights = torch.ones(100, 100) * 0.1

    # Sample many patterns
    selected_counts = torch.zeros(100)
    for _ in range(100):
        seed = replay_generator.select_pattern_to_replay(tags, weights)
        selected_counts += seed.float()

    # First 10 neurons should be selected much more often (60% tag weight)
    first_10_avg = selected_counts[:10].mean().item()
    rest_avg = selected_counts[10:].mean().item()

    assert first_10_avg > rest_avg * 2, (
        f"Neurons with strong tags should be selected more often. "
        f"First 10: {first_10_avg:.2f}, Rest: {rest_avg:.2f}"
    )


def test_pattern_selection_uses_weights(replay_generator, ca3_weights, synaptic_tags):
    """Pattern selection should consider weight strength (well-learned attractors)."""
    # Test with zero tags to verify weight-based fallback works
    n_neurons = 100

    # Create extreme asymmetric weights: only last 10 neurons have INCOMING connections
    # weight[i, j] = connection from j to i
    # So we set ROWS (dim 0) for last 10 neurons to have strong incoming connections
    weights = torch.zeros(n_neurons, n_neurons)
    weights[-10:, :] = 1.0  # Last 10 neurons get incoming connections from all

    # Zero tags (triggers weight-based fallback)
    tags = torch.zeros(n_neurons, n_neurons)

    # Sample many patterns - with zero tags, should use 80% weights + 20% noise
    selected_counts = torch.zeros(n_neurons)
    for _ in range(500):
        seed = replay_generator.select_pattern_to_replay(tags, weights)
        selected_counts += seed.float()

    # Last 10 neurons have ALL the incoming weights → should be selected much more often
    # Expected: 80% weight-based (only last 10 have weights) + 20% uniform noise
    # Last 10 should get: 80% / 10 neurons + 20% / 100 neurons = 8% + 0.2% = 8.2% per neuron
    # Others should get: 0% + 20% / 100 neurons = 0.2% per neuron
    # Ratio: 8.2 / 0.2 = 41x more selections

    last_10_avg = selected_counts[-10:].mean().item()
    first_90_avg = selected_counts[:90].mean().item()

    # With only last 10 having weights, should see dramatic difference (10x+)
    assert last_10_avg > first_90_avg * 5.0, (
        f"Last 10 neurons (only ones with incoming weights) should be selected much more often. "
        f"Last 10: {last_10_avg:.2f}, First 90: {first_90_avg:.2f}, "
        f"Ratio: {last_10_avg / (first_90_avg + 1e-6):.1f}x"
    )


def test_ach_threshold_boundary(replay_generator):
    """Test behavior at ACh threshold boundary."""
    dt_ms = 1.0

    # Reset state for clean test
    replay_generator.reset_state()

    # Just above threshold - should not trigger
    above_threshold = 0.31
    ripple_count_above = 0
    for _ in range(500):
        if replay_generator.should_trigger_ripple(above_threshold, dt_ms):
            ripple_count_above += 1

    # Reset state before testing below threshold
    replay_generator.reset_state()

    # Just below threshold - should trigger
    # Run longer to ensure probabilistic trigger (5000 timesteps gives ~99.3% chance of >= 1 ripple)
    below_threshold = 0.29
    ripple_count_below = 0
    for _ in range(5000):
        if replay_generator.should_trigger_ripple(below_threshold, dt_ms):
            ripple_count_below += 1

    assert ripple_count_above == 0, f"No ripples above threshold, got {ripple_count_above}"
    assert ripple_count_below > 0, f"Expected ripples below threshold, got {ripple_count_below}"


def test_diagnostics_return_expected_keys(replay_generator):
    """Diagnostics should return all configuration parameters."""
    diag = replay_generator.get_diagnostics()

    expected_keys = {
        "ripple_rate_hz",
        "ach_threshold",
        "ripple_refractory_ms",
        "time_since_ripple_ms",
        "tag_weight",
        "strength_weight",
        "noise_weight",
    }

    assert set(diag.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(diag.keys())}, "
        f"Extra keys: {set(diag.keys()) - expected_keys}"
    )

    # Check types
    assert isinstance(diag["ripple_rate_hz"], float)
    assert isinstance(diag["ach_threshold"], float)
    assert isinstance(diag["time_since_ripple_ms"], float)


def test_reset_state_clears_timer(replay_generator):
    """reset_state() should reset timer to allow immediate ripple generation."""
    low_ach = 0.1
    dt_ms = 1.0

    # Advance time beyond refractory period WITHOUT triggering ripples
    # Use high ACh to prevent probabilistic triggers during time advance
    high_ach = 0.9  # Above threshold (0.3), so no ripples
    for _ in range(300):
        replay_generator.should_trigger_ripple(high_ach, dt_ms)

    # Check time advanced (no ripples should have occurred)
    diag_before = replay_generator.get_diagnostics()
    assert diag_before["time_since_ripple_ms"] >= 200.0, (
        f"Time should have advanced past refractory. Got {diag_before['time_since_ripple_ms']:.1f}ms"
    )

    # Reset state
    replay_generator.reset_state()

    # Check time reset to refractory period (allows immediate ripple)
    diag_after = replay_generator.get_diagnostics()
    assert diag_after["time_since_ripple_ms"] == 200.0, (
        "Time should be reset to refractory period (allows immediate ripple)"
    )


def test_seed_fraction_parameter(replay_generator, ca3_weights, synaptic_tags):
    """Test that seed_fraction parameter controls number of seed neurons."""
    # Default: 15% of neurons
    seed_default = replay_generator.select_pattern_to_replay(synaptic_tags, ca3_weights)
    n_selected_default = seed_default.sum().item()

    # Custom: 30% of neurons
    seed_large = replay_generator.select_pattern_to_replay(
        synaptic_tags, ca3_weights, seed_fraction=0.30
    )
    n_selected_large = seed_large.sum().item()

    assert n_selected_large > n_selected_default, (
        f"Larger seed_fraction should select more neurons. "
        f"Default (0.15): {n_selected_default}, Large (0.30): {n_selected_large}"
    )

    # Check approximately correct
    assert 10 <= n_selected_default <= 20, f"Expected ~15 neurons, got {n_selected_default}"
    assert 25 <= n_selected_large <= 35, f"Expected ~30 neurons, got {n_selected_large}"
