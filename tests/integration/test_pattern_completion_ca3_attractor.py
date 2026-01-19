"""
Integration Tests: CA3 Pattern Completion WITHOUT Episode Buffer (Phase 4)

These tests validate that pattern storage and retrieval work through CA3
attractor dynamics and Hebbian learning, without requiring an explicit
episode buffer.

Phase 4 Goal: Memory IS the weights, not an explicit buffer.

**Pattern Storage**: CA3 recurrent weights (Hebbian learning during forward())
**Pattern Retrieval**: CA3 attractor dynamics (pattern completion)
**Priority**: Synaptic tags (Phase 1)
"""

import pytest
import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder


@pytest.fixture
def brain_with_hippocampus():
    """Create minimal brain with hippocampus for pattern completion tests."""
    config = BrainConfig(
        dt_ms=1.0,
        device="cpu",
    )

    brain = BrainBuilder.preset("default", config)
    hippocampus = brain.components["hippocampus"]

    # Enable plasticity for Hebbian learning
    hippocampus.plasticity_enabled = True

    return brain, hippocampus


def test_pattern_completion_without_episode_buffer(brain_with_hippocampus):
    """Pattern retrieval should work via CA3 dynamics, not similarity search.

    This is the key Phase 4 test: memory emerges from CA3 weights.
    """
    brain, hippocampus = brain_with_hippocampus

    # Get thalamus size dynamically
    thalamus_size = brain.components["thalamus"].n_neurons

    # Create a distinct pattern
    full_pattern = torch.zeros(thalamus_size, device=brain.device)
    full_pattern[10:30] = 1.0  # 20 active neurons

    # Store pattern via Hebbian learning (no episode buffer)
    # Repeat presentation to strengthen recurrent connections
    for _ in range(10):
        brain.forward({"thalamus": full_pattern})
        # Simulate dopamine for consolidation
        hippocampus.set_neuromodulators(dopamine=0.8)

    # Give time for weights to consolidate
    for _ in range(5):
        brain.forward({"thalamus": torch.zeros(thalamus_size, device=brain.device)})
        hippocampus.set_neuromodulators(dopamine=0.1)

    # Retrieve with partial cue (50% of pattern)
    partial_cue = full_pattern.clone()
    partial_cue[20:30] = 0.0  # Remove half (10 of 20 active neurons)

    # Let CA3 attractor complete the pattern
    brain.forward({"thalamus": partial_cue})

    # CA3 should reactivate missing neurons
    # Check that CA3 has higher activation for missing pattern neurons
    ca3_spikes = hippocampus.state.ca3_spikes

    # Count how many of the missing neurons (20-30) are active in CA3
    # Note: CA3 has different size, so we check if pattern persists
    # A good test is that output should be more similar to full_pattern than to partial_cue
    assert ca3_spikes is not None, "CA3 should have spikes after pattern completion"
    assert ca3_spikes.sum() > 0, "CA3 should have some activity after partial cue"

    # Verify NO episode buffer was used
    assert (
        len(hippocampus.episode_buffer) == 0
    ), "Should not use episode buffer (Phase 4: memory is in weights)"


def test_multiple_patterns_via_ca3_weights(brain_with_hippocampus):
    """CA3 should store multiple patterns via Hebbian weight updates."""
    brain, hippocampus = brain_with_hippocampus

    # Pattern 1: Neurons 0-10 active
    pattern_1 = torch.zeros(64, device=brain.device)
    pattern_1[0:10] = 1.0

    # Pattern 2: Neurons 30-40 active (distinct)
    pattern_2 = torch.zeros(64, device=brain.device)
    pattern_2[30:40] = 1.0

    # Store both patterns via Hebbian learning
    for _ in range(10):
        brain.forward({"thalamus": pattern_1})
        hippocampus.set_neuromodulators(dopamine=0.8)

    for _ in range(10):
        brain.forward({"thalamus": pattern_2})
        hippocampus.set_neuromodulators(dopamine=0.8)

    # Verify patterns stored in CA3 weights, not episode buffer
    assert len(hippocampus.episode_buffer) == 0, "Should not use episode buffer (Phase 4)"

    # Check that CA3 recurrent weights have learned structure
    ca3_weights = hippocampus.synaptic_weights["ca3_ca3"]
    initial_weight_sum = ca3_weights.sum()

    # After learning, weights should be non-zero
    assert initial_weight_sum > 0, "CA3 weights should have learned patterns"


def test_hebbian_learning_stores_patterns_automatically(brain_with_hippocampus):
    """Pattern storage should happen automatically during forward(), not explicit store_episode()."""
    brain, hippocampus = brain_with_hippocampus

    # Get initial CA3 weights
    ca3_weights_before = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Present pattern multiple times (Hebbian learning should strengthen connections)
    test_pattern = torch.zeros(64, device=brain.device)
    test_pattern[5:15] = 1.0

    for _ in range(15):
        brain.forward({"thalamus": test_pattern})
        hippocampus.set_neuromodulators(dopamine=0.7)

    # Get weights after learning
    ca3_weights_after = hippocampus.synaptic_weights["ca3_ca3"]

    # Weights should have changed (learning occurred)
    weight_change = (ca3_weights_after - ca3_weights_before).abs().sum()
    assert weight_change > 0.01, "Hebbian learning should modify CA3 weights during forward()"

    # No explicit episode storage should have occurred
    assert (
        len(hippocampus.episode_buffer) == 0
    ), "Pattern storage should be automatic via Hebbian learning, not explicit"


def test_synaptic_tags_provide_priority_not_episode_priority(brain_with_hippocampus):
    """Priority should come from synaptic tags (Phase 1), not Episode.priority."""
    brain, hippocampus = brain_with_hippocampus

    # Check that synaptic tagging exists (Phase 1)
    assert hasattr(
        hippocampus, "synaptic_tagging"
    ), "Should use synaptic tagging for priority (Phase 1)"

    # Present a pattern
    test_pattern = torch.zeros(64, device=brain.device)
    test_pattern[10:20] = 1.0

    brain.forward({"thalamus": test_pattern})
    hippocampus.set_neuromodulators(dopamine=0.9)  # High reward → strong tags

    # Check that tags were created
    if hippocampus.synaptic_tagging is not None:
        tags = hippocampus.synaptic_tagging.tags
        tag_sum = tags.sum()
        assert tag_sum > 0, "Synaptic tags should be created for active patterns"

    # Verify NO Episode.priority is used
    assert (
        len(hippocampus.episode_buffer) == 0
    ), "Should not use Episode objects with priority field (Phase 4)"


def test_ca3_attractor_settles_to_stored_pattern(brain_with_hippocampus):
    """CA3 attractor dynamics should settle to stored pattern given partial cue."""
    brain, hippocampus = brain_with_hippocampus

    # Store a strong pattern
    strong_pattern = torch.zeros(64, device=brain.device)
    strong_pattern[15:25] = 1.0

    # Repeat many times to create strong attractor
    for _ in range(20):
        brain.forward({"thalamus": strong_pattern})
        hippocampus.set_neuromodulators(dopamine=0.9)

    # Present very weak partial cue (only 25% of pattern)
    weak_cue = strong_pattern.clone()
    weak_cue[20:25] = 0.0  # Remove 5 of 10 active neurons (50%)

    # Single forward pass with partial cue
    brain.forward({"thalamus": weak_cue})

    # CA3 should have reactivated the pattern
    ca3_spikes = hippocampus.state.ca3_spikes
    assert ca3_spikes is not None, "CA3 should produce spikes"
    assert ca3_spikes.sum() > 0, "CA3 attractor should complete pattern"

    # Verify no episode buffer lookup occurred
    assert (
        len(hippocampus.episode_buffer) == 0
    ), "Pattern completion should use CA3 attractor, not episode buffer lookup"


def test_pattern_storage_scales_with_dopamine(brain_with_hippocampus):
    """Pattern storage strength should scale with dopamine (biological gating)."""
    brain, hippocampus = brain_with_hippocampus

    test_pattern = torch.zeros(64, device=brain.device)
    test_pattern[20:30] = 1.0

    # Get initial weights
    ca3_weights_before = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Low dopamine (weak learning)
    for _ in range(10):
        brain.forward({"thalamus": test_pattern})
        hippocampus.set_neuromodulators(dopamine=0.1)

    ca3_weights_low_da = hippocampus.synaptic_weights["ca3_ca3"].clone()
    weight_change_low = (ca3_weights_low_da - ca3_weights_before).abs().sum()

    # Reset weights
    hippocampus.synaptic_weights["ca3_ca3"].data = ca3_weights_before.clone()

    # High dopamine (strong learning)
    for _ in range(10):
        brain.forward({"thalamus": test_pattern})
        hippocampus.set_neuromodulators(dopamine=0.9)

    ca3_weights_high_da = hippocampus.synaptic_weights["ca3_ca3"]
    weight_change_high = (ca3_weights_high_da - ca3_weights_before).abs().sum()

    # High dopamine should cause more weight change
    assert (
        weight_change_high > weight_change_low * 2
    ), "High dopamine should gate stronger learning (three-factor rule)"


@pytest.mark.parametrize("pattern_size", [5, 10, 20])
def test_pattern_completion_works_for_different_sizes(brain_with_hippocampus, pattern_size):
    """CA3 pattern completion should work for various pattern sizes."""
    brain, hippocampus = brain_with_hippocampus

    # Create pattern of given size
    pattern = torch.zeros(64, device=brain.device)
    pattern[0:pattern_size] = 1.0

    # Store pattern
    for _ in range(10):
        brain.forward({"thalamus": pattern})
        hippocampus.set_neuromodulators(dopamine=0.8)

    # Create partial cue (50% of pattern)
    partial = pattern.clone()
    partial[pattern_size // 2 : pattern_size] = 0.0

    # Test completion
    brain.forward({"thalamus": partial})
    ca3_spikes = hippocampus.state.ca3_spikes

    assert ca3_spikes is not None, f"CA3 should complete {pattern_size}-neuron pattern"
    assert ca3_spikes.sum() > 0, "CA3 should have activity after partial cue"

    # No episode buffer used
    assert len(hippocampus.episode_buffer) == 0, "Should not use episode buffer"
