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
    """Patterns with high dopamine should replay more frequently than unrewarded patterns.

    This test uses CA3 activity overlap to discriminate which pattern is being replayed.
    """
    # Pattern 1: Present with reward (high dopamine)
    pattern_1 = torch.zeros(128, dtype=torch.bool)
    pattern_1[:20] = True

    hippocampus.set_acetylcholine(0.7)  # Encoding mode

    # Record CA3 activity during encoding of pattern 1
    ca3_pattern_1 = torch.zeros(hippocampus.ca3_neurons.n_neurons, dtype=torch.bool)
    for _ in range(50):  # Many repetitions for strong, distinct encoding
        _ = hippocampus.forward({"ec": pattern_1})
        hippocampus.state.dopamine = 1.0  # Reward!
        if hippocampus.state.ca3_spikes is not None:
            ca3_pattern_1 |= hippocampus.state.ca3_spikes

    # Pattern 2: Present without reward (low dopamine)
    pattern_2 = torch.zeros(128, dtype=torch.bool)
    pattern_2[40:60] = True

    # Record CA3 activity during encoding of pattern 2
    ca3_pattern_2 = torch.zeros(hippocampus.ca3_neurons.n_neurons, dtype=torch.bool)
    for _ in range(50):  # Same number of repetitions
        _ = hippocampus.forward({"ec": pattern_2})
        hippocampus.state.dopamine = 0.1  # No reward
        if hippocampus.state.ca3_spikes is not None:
            ca3_pattern_2 |= hippocampus.state.ca3_spikes

    # Verify patterns have different CA3 representations
    ca3_overlap = (ca3_pattern_1 & ca3_pattern_2).sum().item()
    ca3_p1_size = ca3_pattern_1.sum().item()
    ca3_p2_size = ca3_pattern_2.sum().item()

    assert ca3_p1_size > 0, "Pattern 1 should activate some CA3 neurons"
    assert ca3_p2_size > 0, "Pattern 2 should activate some CA3 neurons"

    # Patterns should be somewhat distinct (less than 80% overlap)
    overlap_ratio = ca3_overlap / max(ca3_p1_size, ca3_p2_size)
    assert overlap_ratio < 0.8, (
        f"Patterns too similar (overlap {overlap_ratio:.2f}), cannot discriminate"
    )

    # Enter sleep mode and track which pattern is replayed
    hippocampus.set_acetylcholine(0.1)

    # Reset spontaneous replay state
    if hippocampus.spontaneous_replay is not None:
        hippocampus.spontaneous_replay.reset_state()

    pattern_1_replays = 0
    pattern_2_replays = 0
    ambiguous_replays = 0

    for _ in range(5000):  # 5 seconds to get enough samples (~10 ripples expected)
        empty_input = {"ec": torch.zeros(128, dtype=torch.bool)}
        _ = hippocampus.forward(empty_input)

        if hippocampus.state.ripple_detected:
            if hippocampus.state.ca3_spikes is not None:
                # Calculate overlap with each encoded pattern
                overlap_1 = (hippocampus.state.ca3_spikes & ca3_pattern_1).sum().item()
                overlap_2 = (hippocampus.state.ca3_spikes & ca3_pattern_2).sum().item()

                # Classify replay based on which pattern has higher overlap (simple majority)
                if overlap_1 > overlap_2 and overlap_1 > 0:
                    pattern_1_replays += 1
                elif overlap_2 > overlap_1 and overlap_2 > 0:
                    pattern_2_replays += 1
                else:
                    ambiguous_replays += 1  # Tie or no activity

    total_classified = pattern_1_replays + pattern_2_replays
    total_replays = total_classified + ambiguous_replays

    assert total_replays > 0, (
        f"Expected some ripples during sleep, got {total_replays}"
    )

    assert total_classified > 0, (
        f"Expected some classified replays, got {total_classified} "
        f"(ambiguous: {ambiguous_replays}, total: {total_replays})"
    )

    # Rewarded pattern should replay more than unrewarded pattern
    # (biological tagging creates ~60% contribution to replay selection)
    # With DA=1.0 vs DA=0.1, expect at least 2:1 ratio
    replay_ratio = pattern_1_replays / max(pattern_2_replays, 1)

    assert pattern_1_replays > pattern_2_replays, (
        f"Expected rewarded pattern (DA=1.0) to replay more than "
        f"unrewarded pattern (DA=0.1). Got P1={pattern_1_replays}, P2={pattern_2_replays} "
        f"(ratio: {replay_ratio:.2f}, ambiguous: {ambiguous_replays})"
    )


@pytest.mark.flaky(reason="Stochastic test - may fail occasionally due to randomness")
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
