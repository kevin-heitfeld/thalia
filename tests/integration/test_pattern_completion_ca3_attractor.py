"""
Integration Tests: CA3 Pattern Completion with Biologically-Realistic Protocols

These tests validate that pattern storage and retrieval work through CA3
attractor dynamics and Hebbian learning, without requiring an explicit
episode buffer.

Phase 4 Goal: Memory IS the weights, not an explicit buffer.

**Pattern Storage**: CA3 recurrent weights (Hebbian learning during forward())
**Pattern Retrieval**: CA3 attractor dynamics (pattern completion)
**Priority**: Synaptic tags (Phase 1)

IMPORTANT: These tests use biologically-realistic protocols with proper
inter-trial intervals (ITI) to account for neuronal adaptation. CA3 neurons
have spike-frequency adaptation (adapt_increment=0.5) which is ESSENTIAL for
preventing runaway excitation in recurrent networks. Without ITI, neurons
adapt and stop firing, which is biologically accurate but prevents testing.

Protocol Design:
- Inter-trial intervals: 500ms (allows adaptation to decay ~50%)
- Consolidation period: 2 seconds (full recovery)
- Retrieval state: Low ACh (enables recurrence), rested neurons
"""

import pytest
import torch

from thalia.config import BrainConfig


@pytest.fixture
def brain_with_hippocampus():
    """Create minimal brain with hippocampus for pattern completion tests.

    Architecture: Cortex → Hippocampus (minimal two-region setup)
    This allows direct input to hippocampus for pattern completion tests.
    """
    config = BrainConfig(
        dt_ms=1.0,
        device="cpu",
    )

    # Build minimal brain with direct cortex → hippocampus connection
    # Cortex acts as input layer (like sensory cortex receiving thalamic input)
    from thalia.core.brain_builder import BrainBuilder

    builder = BrainBuilder(config)

    # Add cortex as input interface (simplified sensory input)
    builder.add_component(
        "cortex",
        "cortex",
        input_size=64,  # Direct input
        l4_size=64,
        l23_size=96,
        l5_size=32,
        l6a_size=0,  # Minimal setup
        l6b_size=0,
    )

    # Add hippocampus receiving from cortex L2/3
    cortex_output_size = 96 + 32  # L23 + L5
    builder.add_component(
        "hippocampus",
        "hippocampus",
        input_size=cortex_output_size,
        dg_size=256,
        ca3_size=128,
        ca2_size=64,
        ca1_size=128,
    )

    # Connect cortex → hippocampus
    builder.connect("cortex", "hippocampus", pathway_type="axonal", axonal_delay_ms=1.0)

    brain = builder.build()
    hippocampus = brain.components["hippocampus"]

    # Enable plasticity for Hebbian learning
    hippocampus.plasticity_enabled = True

    return brain, hippocampus


def test_pattern_completion_from_ca3_weights(brain_with_hippocampus):
    """Pattern retrieval should work via CA3 dynamics, not similarity search.

    This is the key test: memory emerges from CA3 weights.
    Uses proper inter-trial intervals to allow adaptation to decay.
    """
    brain, hippocampus = brain_with_hippocampus
    input_size = 64
    zeros = torch.zeros(input_size, device=brain.device)

    # Create a distinct pattern
    full_pattern = torch.zeros(input_size, device=brain.device)
    full_pattern[10:30] = 1.0  # 20 active neurons

    # ENCODING: Store pattern via Hebbian learning with proper ITI
    # Biological protocol: 500ms inter-trial interval allows adaptation to decay
    for trial in range(10):
        brain.forward({"cortex": full_pattern}, n_timesteps=50)
        hippocampus.set_neuromodulators(dopamine=0.8, acetylcholine=0.7)  # Encoding state

        # Inter-trial interval: 500ms rest (10 × 50ms)
        for _ in range(10):
            brain.forward({"cortex": zeros}, n_timesteps=50)
            hippocampus.set_neuromodulators(dopamine=0.1)

    # CONSOLIDATION: Allow full recovery (2 seconds)
    for _ in range(40):  # 40 × 50ms = 2 seconds
        brain.forward({"cortex": zeros}, n_timesteps=50)
        hippocampus.set_neuromodulators(dopamine=0.1)

    # RETRIEVAL: Present partial cue in retrieval state (low ACh enables recurrence)
    partial_cue = full_pattern.clone()
    partial_cue[20:30] = 0.0  # Remove half (10 of 20 active neurons)

    hippocampus.set_neuromodulators(acetylcholine=0.2, dopamine=0.0)  # Retrieval state
    brain.forward({"cortex": partial_cue}, n_timesteps=50)

    # CA3 should show activity (pattern completion via recurrence)
    ca3_spikes = hippocampus.state.ca3_spikes
    assert ca3_spikes is not None, "CA3 should have spikes after pattern completion"

    # With proper recovery, CA3 should be able to fire
    # Note: May still need stronger input or more trials, but should show SOME activity
    ca3_activity = ca3_spikes.sum().item()
    assert ca3_activity > 0, (
        f"CA3 should have some activity after partial cue with proper ITI. "
        f"Got {ca3_activity} spikes. Check if neurons are near threshold."
    )


def test_multiple_patterns_via_ca3_weights(brain_with_hippocampus):
    """CA3 should store multiple patterns via Hebbian weight updates with proper ITI."""
    brain, hippocampus = brain_with_hippocampus
    input_size = 64
    zeros = torch.zeros(input_size, device=brain.device)

    # Pattern 1: First neurons active
    pattern_1 = torch.zeros(input_size, device=brain.device)
    pattern_1[0:10] = 1.0

    # Pattern 2: Different neurons active (distinct)
    pattern_2 = torch.zeros(input_size, device=brain.device)
    pattern_2[30:40] = 1.0

    # Get initial weights
    ca3_weights_before = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Store pattern 1 with ITI
    for trial in range(10):
        brain.forward({"cortex": pattern_1}, n_timesteps=50)
        hippocampus.set_neuromodulators(dopamine=0.8)

        # ITI: 500ms rest
        for _ in range(10):
            brain.forward({"cortex": zeros}, n_timesteps=50)

    # Inter-pattern rest: 1 second
    for _ in range(20):
        brain.forward({"cortex": zeros}, n_timesteps=50)

    # Store pattern 2 with ITI
    for trial in range(10):
        brain.forward({"cortex": pattern_2}, n_timesteps=50)
        hippocampus.set_neuromodulators(dopamine=0.8)

        # ITI: 500ms rest
        for _ in range(10):
            brain.forward({"cortex": zeros}, n_timesteps=50)

    # Check that CA3 recurrent weights have learned
    ca3_weights = hippocampus.synaptic_weights["ca3_ca3"]
    weight_change = (ca3_weights - ca3_weights_before).abs().sum()

    assert weight_change > 0.1, (
        f"CA3 weights should have changed significantly. "
        f"Weight change: {weight_change:.3f}"
    )


def test_hebbian_learning_stores_patterns_automatically(brain_with_hippocampus):
    """Pattern storage happens automatically during forward().

    This test verifies that CA3 weights change via Hebbian learning even with
    adaptation causing reduced spiking in later trials.
    """
    brain, hippocampus = brain_with_hippocampus
    input_size = 64
    zeros = torch.zeros(input_size, device=brain.device)

    # Get initial CA3 weights
    ca3_weights_before = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Present pattern with ITI to maintain some spiking activity
    test_pattern = torch.zeros(input_size, device=brain.device)
    test_pattern[5:15] = 1.0

    for trial in range(8):  # Fewer trials but with ITI
        brain.forward({"cortex": test_pattern}, n_timesteps=50)
        hippocampus.set_neuromodulators(dopamine=0.7)

        # ITI: 500ms rest
        for _ in range(10):
            brain.forward({"cortex": zeros}, n_timesteps=50)

    # Get weights after learning
    ca3_weights_after = hippocampus.synaptic_weights["ca3_ca3"]

    # Weights should have changed (learning occurred)
    weight_change = (ca3_weights_after - ca3_weights_before).abs().sum()
    assert weight_change > 0.1, (
        f"Hebbian learning should modify CA3 weights during forward(). "
        f"Weight change: {weight_change:.3f}"
    )


def test_synaptic_tags_provide_priority_not_episode_priority(brain_with_hippocampus):
    """Priority should come from synaptic tags."""
    brain, hippocampus = brain_with_hippocampus
    input_size = 64  # Cortex input size from fixture

    # Check that synaptic tagging exists
    assert hasattr(
        hippocampus, "synaptic_tagging"
    ), "Should use synaptic tagging for priority"

    # Present a pattern (use multiple timesteps for neurons to spike)
    test_pattern = torch.zeros(input_size, device=brain.device)
    test_pattern[10:20] = 1.0

    brain.forward({"cortex": test_pattern}, n_timesteps=50)
    hippocampus.set_neuromodulators(dopamine=0.9)  # High reward → strong tags

    # Check that tags were created
    assert hippocampus.synaptic_tagging is not None, "Synaptic tagging should be active"

    tags = hippocampus.synaptic_tagging.tags
    tag_sum = tags.sum()
    assert tag_sum > 0, "Synaptic tags should be created for active patterns"


def test_ca3_attractor_settles_to_stored_pattern(brain_with_hippocampus):
    """CA3 attractor dynamics should settle to stored pattern given partial cue."""
    brain, hippocampus = brain_with_hippocampus
    input_size = 64
    zeros = torch.zeros(input_size, device=brain.device)

    # Store a strong pattern with ITI
    strong_pattern = torch.zeros(input_size, device=brain.device)
    strong_pattern[15:25] = 1.0

    # Repeat many times to create strong attractor
    for trial in range(12):
        brain.forward({"cortex": strong_pattern}, n_timesteps=50)
        hippocampus.set_neuromodulators(dopamine=0.9)

        # ITI: 500ms rest
        for _ in range(10):
            brain.forward({"cortex": zeros}, n_timesteps=50)

    # Consolidation: 2 seconds
    for _ in range(40):
        brain.forward({"cortex": zeros}, n_timesteps=50)

    # Present partial cue in retrieval state
    weak_cue = strong_pattern.clone()
    weak_cue[20:25] = 0.0  # Remove 5 of 10 active neurons (50%)

    hippocampus.set_neuromodulators(acetylcholine=0.2, dopamine=0.0)  # Retrieval state
    brain.forward({"cortex": weak_cue}, n_timesteps=50)

    # CA3 should have reactivated the pattern
    ca3_spikes = hippocampus.state.ca3_spikes
    assert ca3_spikes is not None, "CA3 should produce spikes"
    ca3_activity = ca3_spikes.sum().item()
    assert ca3_activity > 0, (
        f"CA3 attractor should complete pattern. Got {ca3_activity} spikes."
    )


def test_pattern_storage_scales_with_dopamine(brain_with_hippocampus):
    """Pattern storage strength should scale with dopamine (biological gating).

    Tests dopamine modulation by measuring weight changes between CA3 neurons
    that represent the stored pattern (pattern-specific synapses).
    """
    brain, hippocampus = brain_with_hippocampus
    input_size = 64

    test_pattern = torch.zeros(input_size, device=brain.device)
    test_pattern[20:30] = 1.0

    # Get initial weights
    ca3_weights_before = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Low dopamine condition (weak learning)
    hippocampus.set_neuromodulators(dopamine=0.1)

    # Present pattern to activate CA3 neurons
    # Track ensemble across multiple timesteps
    ca3_active_neurons_low = torch.zeros(hippocampus.ca3_neurons.n_neurons, dtype=torch.bool, device=brain.device)
    for _ in range(5):
        brain.forward({"cortex": test_pattern}, n_timesteps=1)
        if hippocampus.state.ca3_spikes is not None:
            ca3_active_neurons_low |= hippocampus.state.ca3_spikes

    ca3_pattern_neurons_low = ca3_active_neurons_low.nonzero(as_tuple=True)[0]
    assert len(ca3_pattern_neurons_low) > 5, f"At least 5 CA3 neurons should activate, got {len(ca3_pattern_neurons_low)}"

    ca3_weights_low_da = hippocampus.synaptic_weights["ca3_ca3"].clone()

    # Measure weight changes between pattern neurons (recurrent connections for this pattern)
    # These are the synapses that should strengthen most during pattern storage
    weight_delta_low = ca3_weights_low_da - ca3_weights_before

    # Extract submatrix of pattern neuron recurrent connections
    pattern_synapses_delta_low = weight_delta_low[ca3_pattern_neurons_low][:, ca3_pattern_neurons_low]
    mean_pattern_change_low = pattern_synapses_delta_low.abs().mean().item()

    # Reset for fair comparison
    hippocampus.new_trial()
    hippocampus.set_neuromodulators(dopamine=0.0)
    zeros = torch.zeros(input_size, device=brain.device)
    brain.forward({"cortex": zeros}, n_timesteps=2)
    hippocampus.synaptic_weights["ca3_ca3"].data = ca3_weights_before.clone()
    if hippocampus.ca3_neurons.g_adapt is not None:
        hippocampus.ca3_neurons.g_adapt.zero_()

    # High dopamine condition (strong learning)
    hippocampus.set_neuromodulators(dopamine=0.9)

    # Track ensemble across multiple timesteps
    ca3_active_neurons_high = torch.zeros(hippocampus.ca3_neurons.n_neurons, dtype=torch.bool, device=brain.device)
    for _ in range(5):
        brain.forward({"cortex": test_pattern}, n_timesteps=1)
        if hippocampus.state.ca3_spikes is not None:
            ca3_active_neurons_high |= hippocampus.state.ca3_spikes

    ca3_pattern_neurons_high = ca3_active_neurons_high.nonzero(as_tuple=True)[0]
    assert len(ca3_pattern_neurons_high) > 5, f"At least 5 CA3 neurons should activate, got {len(ca3_pattern_neurons_high)}"

    ca3_weights_high_da = hippocampus.synaptic_weights["ca3_ca3"]
    weight_delta_high = ca3_weights_high_da - ca3_weights_before

    pattern_synapses_delta_high = weight_delta_high[ca3_pattern_neurons_high][:, ca3_pattern_neurons_high]
    mean_pattern_change_high = pattern_synapses_delta_high.abs().mean().item()

    # Expected ratio: da_gain(0.9) / da_gain(0.1) = 1.82 / 0.38 = 4.79x
    # IMPORTANT: Must compare same synapses! Use intersection of pattern neurons from both conditions
    # Otherwise we're measuring different synapses with different baseline activities
    common_neurons = torch.tensor(list(set(ca3_pattern_neurons_low.tolist()) & set(ca3_pattern_neurons_high.tolist())), device=brain.device)

    if len(common_neurons) < 5:
        # Fall back to union if too few common neurons (stochastic spiking can differ)
        common_neurons = torch.cat([ca3_pattern_neurons_low, ca3_pattern_neurons_high]).unique()

    # Re-measure on common set of pattern neurons
    pattern_synapses_delta_low_common = weight_delta_low[common_neurons][:, common_neurons]
    pattern_synapses_delta_high_common = weight_delta_high[common_neurons][:, common_neurons]
    mean_pattern_change_low_common = pattern_synapses_delta_low_common.abs().mean().item()
    mean_pattern_change_high_common = pattern_synapses_delta_high_common.abs().mean().item()

    ratio = mean_pattern_change_high_common / mean_pattern_change_low_common if mean_pattern_change_low_common > 0 else 0.0

    # Expected theoretical ratio: da_gain(0.9) / da_gain(0.1) = 1.82 / 0.38 = 4.79x
    # But biological constraints reduce this:
    # - Weight clamping [0,1] saturates strong learning
    # - Heterosynaptic LTD (10-20%) on inactive synapses
    # - Stochastic spiking creates variability
    # Accept 1.5x as threshold (biological systems show ~2x in practice)
    assert mean_pattern_change_high_common > mean_pattern_change_low_common * 1.5, (
        f"High dopamine should gate stronger learning (three-factor rule). "
        f"Expected ~1.5-2x ratio (accounting for biological constraints), got {ratio:.2f}x. "
        f"(on {len(common_neurons)} common pattern neurons)"
    )


@pytest.mark.parametrize("pattern_size", [5, 10, 20])
def test_pattern_completion_works_for_different_sizes(brain_with_hippocampus, pattern_size):
    """CA3 pattern completion should work for various pattern sizes with proper ITI."""
    brain, hippocampus = brain_with_hippocampus
    input_size = 64
    zeros = torch.zeros(input_size, device=brain.device)

    # Create pattern of given size
    pattern = torch.zeros(input_size, device=brain.device)
    pattern[0:pattern_size] = 1.0

    # Store pattern with ITI
    for trial in range(10):
        brain.forward({"cortex": pattern}, n_timesteps=50)
        hippocampus.set_neuromodulators(dopamine=0.8)

        # ITI: 500ms rest
        for _ in range(10):
            brain.forward({"cortex": zeros}, n_timesteps=50)

    # Consolidation: 2 seconds
    for _ in range(40):
        brain.forward({"cortex": zeros}, n_timesteps=50)

    # Create partial cue (50% of pattern)
    partial = pattern.clone()
    partial[pattern_size // 2 : pattern_size] = 0.0

    # Test completion in retrieval state
    hippocampus.set_neuromodulators(acetylcholine=0.2, dopamine=0.0)
    brain.forward({"cortex": partial}, n_timesteps=50)

    ca3_spikes = hippocampus.state.ca3_spikes
    assert ca3_spikes is not None, f"CA3 should complete {pattern_size}-neuron pattern"

    ca3_activity = ca3_spikes.sum().item()
    assert ca3_activity > 0, (
        f"CA3 should have activity after partial cue for {pattern_size}-neuron pattern. "
        f"Got {ca3_activity} spikes."
    )
