"""
Tests for synaptic tagging and capture mechanism.

Tests biological synaptic tagging (Frey-Morris 1997) implementation for
hippocampal memory consolidation.
"""

import torch

from thalia.regions.hippocampus.synaptic_tagging import SynapticTagging


class TestSynapticTagging:
    """Unit tests for synaptic tagging mechanism."""

    def test_tags_created_by_spike_coincidence(self):
        """Tags should appear at synapses where pre and post fire together."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")

        # Fire neurons 10-20 together (spike coincidence)
        pre_spikes = torch.zeros(100)
        pre_spikes[10:20] = 1.0
        post_spikes = torch.zeros(100)
        post_spikes[10:20] = 1.0

        tagging.update_tags(pre_spikes, post_spikes)

        # Tags should be strong at [10:20, 10:20] block (coincident firing)
        tag_sum = tagging.tags[10:20, 10:20].sum()
        assert tag_sum > 50, f"Should have strong tags where spikes coincide, got {tag_sum:.2f}"

        # Tags should be weak elsewhere (no coincidence)
        other_sum = tagging.tags[0:10, 0:10].sum()
        assert other_sum < 1.0, f"Should have weak tags where no coincidence, got {other_sum:.2f}"

    def test_tags_decay_over_time(self):
        """Tags should decay exponentially without reinforcement."""
        tagging = SynapticTagging(n_neurons=100, device="cpu", tag_decay=0.95)

        # Create strong tags
        spikes = torch.zeros(100)
        spikes[10:20] = 1.0
        tagging.update_tags(spikes, spikes)
        initial_strength = tagging.tags[10:20, 10:20].sum()

        assert initial_strength > 50, "Should create initial tags"

        # Let decay for 50 timesteps (no new activity)
        for _ in range(50):
            tagging.update_tags(torch.zeros(100), torch.zeros(100))

        final_strength = tagging.tags[10:20, 10:20].sum()

        # Should decay significantly: 0.95^50 ≈ 0.08
        decay_ratio = final_strength / initial_strength
        assert decay_ratio < 0.15, f"Tags should decay to ~8%, got {decay_ratio:.2%}"
        assert final_strength > 0, "Tags should decay gradually, not instantly"

    def test_dopamine_gates_consolidation(self):
        """Weight changes should only occur with dopamine present."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")
        weights = torch.ones(100, 100) * 0.5

        # Create tags via spike coincidence
        spikes = torch.zeros(100)
        spikes[10:20] = 1.0
        tagging.update_tags(spikes, spikes)

        # No dopamine → no consolidation
        weights_no_da = tagging.consolidate_tagged_synapses(weights, dopamine=0.0)
        assert torch.allclose(weights_no_da, weights), "No change without dopamine"

        # High dopamine → strong consolidation
        weights_high_da = tagging.consolidate_tagged_synapses(
            weights, dopamine=1.0, learning_rate=0.01
        )
        center_neuron = 15
        weight_change = weights_high_da[center_neuron, center_neuron] - weights[center_neuron, center_neuron]

        assert weight_change > 0.005, \
            f"Tagged synapses should strengthen with dopamine, got change={weight_change:.4f}"

    def test_replay_probability_proportional_to_tags(self):
        """Patterns with stronger tags should have higher replay probability."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")

        # Create weak tags for pattern 1
        spikes_1 = torch.zeros(100)
        spikes_1[10:15] = 1.0
        tagging.update_tags(spikes_1, spikes_1)

        # Create strong tags for pattern 2 (present 5 times)
        spikes_2 = torch.zeros(100)
        spikes_2[50:60] = 1.0
        for _ in range(5):
            tagging.update_tags(spikes_2, spikes_2)

        probs = tagging.get_replay_probabilities()

        # Pattern 2 should have much higher replay probability
        prob_pattern_1 = probs[10:15].sum().item()
        prob_pattern_2 = probs[50:60].sum().item()

        assert prob_pattern_2 > prob_pattern_1 * 2, \
            f"Strongly-tagged pattern should be more likely to replay: " \
            f"pattern_1={prob_pattern_1:.3f}, pattern_2={prob_pattern_2:.3f}"

    def test_tags_use_maximum_not_sum(self):
        """Repeated activity should saturate tags, not accumulate indefinitely."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")

        spikes = torch.zeros(100)
        spikes[10:20] = 1.0

        # First update
        tagging.update_tags(spikes, spikes)
        first_strength = tagging.tags[15, 15].item()

        # Second update (immediate, no decay)
        tagging.update_tags(spikes, spikes)
        second_strength = tagging.tags[15, 15].item()

        # Tags should saturate (maximum, not sum)
        assert abs(second_strength - first_strength) < 0.01, \
            f"Tags should saturate, not accumulate: first={first_strength:.3f}, second={second_strength:.3f}"

    def test_reset_tags_clears_all(self):
        """Reset should clear all tags."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")

        # Create tags
        spikes = torch.ones(100)
        tagging.update_tags(spikes, spikes)

        assert tagging.tags.max() > 0.5, "Should create tags"

        # Reset
        tagging.reset_tags()

        assert tagging.tags.max() < 0.01, "Reset should clear all tags"
        assert tagging.tags.sum() < 0.1, "Reset should clear all tags"

    def test_diagnostics_return_expected_keys(self):
        """Diagnostics should return expected statistics."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")

        # Create some tags
        spikes = torch.zeros(100)
        spikes[10:30] = 1.0
        tagging.update_tags(spikes, spikes)

        diag = tagging.get_diagnostics()

        # Check expected keys
        assert "tag_mean" in diag
        assert "tag_max" in diag
        assert "tag_nonzero" in diag
        assert "tag_total" in diag
        assert "tag_coverage" in diag

        # Check reasonable values
        assert diag["tag_mean"] > 0, "Should have positive mean tag strength"
        assert diag["tag_max"] <= 1.0, "Max tag should be <= 1.0"
        assert diag["tag_nonzero"] > 0, "Should have some nonzero tags"
        assert diag["tag_coverage"] > 0, "Should have positive coverage"

    def test_binary_and_float_spikes_both_work(self):
        """Should handle both binary and float spike inputs."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")

        # Binary spikes
        binary_spikes = torch.zeros(100, dtype=torch.bool)
        binary_spikes[10:20] = True
        tagging.update_tags(binary_spikes, binary_spikes)
        binary_strength = tagging.tags[15, 15].item()

        tagging.reset_tags()

        # Float spikes
        float_spikes = torch.zeros(100)
        float_spikes[10:20] = 1.0
        tagging.update_tags(float_spikes, float_spikes)
        float_strength = tagging.tags[15, 15].item()

        # Should produce similar tag strength
        assert abs(binary_strength - float_strength) < 0.1, \
            "Binary and float spikes should produce similar tags"


class TestSynapticTaggingIntegration:
    """Integration tests for synaptic tagging in learning scenarios."""

    def test_reward_prediction_via_tags(self):
        """Tags should predict which patterns will be rewarded."""
        tagging = SynapticTagging(n_neurons=100, device="cpu")
        weights = torch.ones(100, 100) * 0.3

        # Pattern 1: presented with reward 10 times
        pattern_rewarded = torch.zeros(100)
        pattern_rewarded[20:30] = 1.0

        for _ in range(10):
            tagging.update_tags(pattern_rewarded, pattern_rewarded)
            weights = tagging.consolidate_tagged_synapses(weights, dopamine=0.8)
            # Let some decay happen
            for _ in range(2):
                tagging.update_tags(torch.zeros(100), torch.zeros(100))

        # Pattern 2: presented without reward 10 times
        pattern_unrewarded = torch.zeros(100)
        pattern_unrewarded[60:70] = 1.0

        for _ in range(10):
            tagging.update_tags(pattern_unrewarded, pattern_unrewarded)
            weights = tagging.consolidate_tagged_synapses(weights, dopamine=0.1)
            # Let some decay happen
            for _ in range(2):
                tagging.update_tags(torch.zeros(100), torch.zeros(100))

        # Rewarded pattern should have stronger weights
        rewarded_weights = weights[20:30, 20:30].mean().item()
        unrewarded_weights = weights[60:70, 60:70].mean().item()

        assert rewarded_weights > unrewarded_weights * 1.2, \
            f"Rewarded pattern should have stronger weights: " \
            f"rewarded={rewarded_weights:.3f}, unrewarded={unrewarded_weights:.3f}"
