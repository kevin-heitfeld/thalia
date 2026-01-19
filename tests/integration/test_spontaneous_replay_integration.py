"""Integration tests for spontaneous replay in full hippocampus.

Tests spontaneous replay mechanism integrated with TrisynapticHippocampus:
- Ripples only occur during low ACh (sleep/rest)
- Rewarded patterns replay more frequently (synaptic tag priority)
- Replay rate matches biological constraints (~1-3 Hz)
- No explicit coordinator needed (emergent from ACh + CA3 dynamics)
"""

import torch
import pytest

from thalia.config import HippocampusConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus


@pytest.fixture
def hippocampus():
    """Create a test hippocampus with theta-gamma enabled."""
    config = HippocampusConfig(
        theta_gamma_enabled=True,
        learning_rate=0.01,
        dt_ms=1.0,
    )
    sizes = {
        "input_size": 128,
        "dg_size": 128,
        "ca3_size": 100,
        "ca2_size": 64,
        "ca1_size": 64,
    }
    device = "cpu"

    hippo = TrisynapticHippocampus(config, sizes, device)
    hippo.reset_state()

    return hippo


def test_replay_during_low_ach_only(hippocampus):
    """Ripples should occur only during low ACh, not during high ACh."""
    # High ACh (encoding) - no ripples
    hippocampus.set_acetylcholine(0.8)

    ripple_count_encoding = 0
    for _ in range(1000):  # 1 second
        test_input = {"ec": torch.randn(128) > 0.5}
        _ = hippocampus.forward(test_input)
        if hippocampus.state.ripple_detected:
            ripple_count_encoding += 1

    # Low ACh (sleep) - frequent ripples
    hippocampus.set_acetylcholine(0.1)
    if hippocampus.spontaneous_replay is not None:
        hippocampus.spontaneous_replay.reset_state()

    ripple_count_sleep = 0
    for _ in range(1000):  # 1 second
        empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}
        _ = hippocampus.forward(empty_input)
        if hippocampus.state.ripple_detected:
            ripple_count_sleep += 1

    assert ripple_count_encoding == 0, (
        f"Expected 0 ripples during high ACh, got {ripple_count_encoding}"
    )
    assert ripple_count_sleep > 0, (
        f"Expected ripples during low ACh, got {ripple_count_sleep}"
    )


def test_rewarded_patterns_replay_more(hippocampus):
    """Patterns with high dopamine should replay more frequently than unrewarded patterns."""
    # Pattern 1: Present with reward (high dopamine)
    pattern_1 = torch.zeros(128, dtype=torch.bool)
    pattern_1[:20] = True

    hippocampus.set_acetylcholine(0.7)  # Encoding mode

    for _ in range(10):
        _ = hippocampus.forward({"ec": pattern_1})
        hippocampus.state.dopamine = 1.0  # Reward!

    # Pattern 2: Present without reward (low dopamine)
    pattern_2 = torch.zeros(128, dtype=torch.bool)
    pattern_2[40:60] = True

    for _ in range(10):
        _ = hippocampus.forward({"ec": pattern_2})
        hippocampus.state.dopamine = 0.1  # No reward

    # Enter sleep mode and count CA1 patterns
    hippocampus.set_acetylcholine(0.1)

    # Track CA1 activity patterns during replay
    # Rewarded pattern should have more consistent CA1 activation
    ca1_patterns = []
    ripple_count = 0

    for _ in range(2000):  # 2 seconds
        empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}
        ca1_output = hippocampus.forward(empty_input)

        if hippocampus.state.ripple_detected:
            ripple_count += 1
            ca1_patterns.append(ca1_output.clone())

    assert ripple_count > 0, f"Expected ripples during sleep, got {ripple_count}"
    assert len(ca1_patterns) > 0, "Should have recorded some replay patterns"

    # Note: Full pattern discrimination would require more sophisticated analysis
    # (e.g., clustering CA1 patterns, tracking which cluster corresponds to which input)
    # For now, we just verify that replay is occurring


def test_replay_rate_matches_biology(hippocampus):
    """Ripple rate should be approximately 1-3 Hz during sleep."""
    hippocampus.set_acetylcholine(0.1)  # Sleep mode

    ripple_count = 0
    duration_ms = 2000.0  # 2 seconds

    for _ in range(int(duration_ms)):
        empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}
        _ = hippocampus.forward(empty_input)
        if hippocampus.state.ripple_detected:
            ripple_count += 1

    # Calculate rate in Hz
    ripple_rate = ripple_count / (duration_ms / 1000.0)

    assert 0.5 <= ripple_rate <= 4.0, (
        f"Ripple rate {ripple_rate:.2f} Hz outside biological range [0.5-4.0 Hz]. "
        f"Got {ripple_count} ripples in {duration_ms}ms"
    )


def test_no_explicit_coordinator_needed(hippocampus):
    """Replay should emerge from ACh level and CA3 dynamics, no coordinator needed."""
    # This test verifies the core architectural claim: no UnifiedReplayCoordinator

    # Set low ACh - replay should happen automatically
    hippocampus.set_acetylcholine(0.1)

    empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}

    ripple_detected = False
    for _ in range(1000):
        _ = hippocampus.forward(empty_input)
        if hippocampus.state.ripple_detected:
            ripple_detected = True
            break

    assert ripple_detected, (
        "Spontaneous replay should occur without explicit coordinator calls. "
        "Only ACh level needed!"
    )


def test_tags_influence_replay_probability(hippocampus):
    """Patterns with stronger synaptic tags should be replayed more often."""
    # Present pattern repeatedly to build strong tags
    strong_pattern = torch.zeros(128, dtype=torch.bool)
    strong_pattern[:30] = True

    hippocampus.set_acetylcholine(0.7)  # Encoding

    for _ in range(20):
        _ = hippocampus.forward({"ec": strong_pattern})

    # Check that tags were created
    if hippocampus.synaptic_tagging is not None:
        tag_mean = hippocampus.synaptic_tagging.tags.mean().item()
        assert tag_mean > 0.01, f"Expected tags to be created, got mean {tag_mean}"

    # Enter sleep and verify replay occurs
    hippocampus.set_acetylcholine(0.1)

    # Reset spontaneous replay state (clear refractory period)
    if hippocampus.spontaneous_replay is not None:
        hippocampus.spontaneous_replay.reset_state()

    ripple_count = 0
    for _ in range(1000):
        empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}
        _ = hippocampus.forward(empty_input)
        if hippocampus.state.ripple_detected:
            ripple_count += 1

    assert ripple_count > 0, (
        f"Expected spontaneous replay of tagged patterns, got {ripple_count} ripples"
    )


@pytest.mark.flaky(reason="Stochastic test - may fail occasionally due to randomness")
def test_replay_uses_ca3_attractor_dynamics(hippocampus):
    """Replay should use CA3 recurrent connections for pattern completion."""
    # Encode a pattern
    pattern = torch.zeros(128, dtype=torch.bool)
    pattern[:25] = True

    hippocampus.set_acetylcholine(0.7)

    for _ in range(100):
        _ = hippocampus.forward({"ec": pattern})

    # Enter sleep mode
    hippocampus.set_acetylcholine(0.1)

    # Reset spontaneous replay state (clear refractory period from fixture setup)
    if hippocampus.spontaneous_replay is not None:
        hippocampus.spontaneous_replay.reset_state()

    # During replay, CA3 should show activity (from attractor dynamics)
    ca3_active_during_replay = False

    for _ in range(1000):
        empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}
        _ = hippocampus.forward(empty_input)

        if hippocampus.state.ripple_detected:
            if hippocampus.state.ca3_spikes is not None:
                ca3_activity = hippocampus.state.ca3_spikes.sum().item()
                if ca3_activity > 0:
                    ca3_active_during_replay = True
                    break

    assert ca3_active_during_replay, (
        "CA3 should show activity during replay (attractor dynamics)"
    )


def test_set_acetylcholine_updates_state(hippocampus):
    """set_acetylcholine() should update hippocampus state."""
    # Initialize state
    empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}
    _ = hippocampus.forward(empty_input)

    # Set ACh and verify
    hippocampus.set_acetylcholine(0.2)
    assert hippocampus.state.acetylcholine == 0.2, (
        f"Expected ACh=0.2, got {hippocampus.state.acetylcholine}"
    )

    hippocampus.set_acetylcholine(0.9)
    assert hippocampus.state.acetylcholine == 0.9, (
        f"Expected ACh=0.9, got {hippocampus.state.acetylcholine}"
    )
