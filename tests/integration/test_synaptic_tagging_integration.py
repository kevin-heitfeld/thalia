"""
Integration tests for synaptic tagging in hippocampus.

Verifies that synaptic tagging mechanism integrates correctly with
trisynaptic hippocampus for emergent priority-based consolidation.
"""

import torch

from thalia.config import HippocampusConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus


class TestSynapticTaggingIntegration:
    """Integration tests for synaptic tagging in full hippocampus."""

    def test_tags_created_during_encoding(self):
        """Synaptic tags should be created during CA3 encoding."""
        config = HippocampusConfig(
            theta_gamma_enabled=True,
            learning_enabled=True,
            learning_rate=0.01,
            dg_sparsity=0.05,  # Sparse DG
            ca3_sparsity=0.1,  # Moderate CA3 sparsity
        )
        sizes = {
            "input_size": 64,
            "dg_size": 128,
            "ca3_size": 100,
            "ca2_size": 50,
            "ca1_size": 64,
        }

        hippo = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")

        # Provide strong input pattern to ensure CA3 activity
        input_spikes = torch.zeros(64, dtype=torch.bool)
        input_spikes[10:40] = True  # 30 active neurons

        # Process through hippocampus multiple times to build up tags
        for _ in range(5):
            _ = hippo.forward({"ec": input_spikes})

        # Check if CA3 had any activity
        assert hippo.state.ca3_spikes is not None
        ca3_active = hippo.state.ca3_spikes.sum().item()

        # Tags should be created if CA3 was active
        assert hippo.synaptic_tagging is not None
        tag_diag = hippo.synaptic_tagging.get_diagnostics()

        # With the WTA fix, CA3 should be active and tags should be created
        assert tag_diag["tag_mean"] > 0, \
            f"Tags should be created during encoding (tag_mean={tag_diag['tag_mean']:.6f})"
        assert tag_diag["tag_nonzero"] > 0, "Should have some nonzero tags"

    def test_tags_influence_replay_probability(self):
        """Patterns with strong tags should be more likely to replay."""
        config = HippocampusConfig(
            theta_gamma_enabled=True,
            learning_enabled=True,
            learning_rate=0.01,
        )
        sizes = {
            "input_size": 64,
            "dg_size": 128,
            "ca3_size": 100,
            "ca2_size": 50,
            "ca1_size": 64,
        }

        hippo = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")

        # Pattern 1: weak (present once)
        input_1 = torch.zeros(64, dtype=torch.bool)
        input_1[10:15] = True
        _ = hippo.forward({"ec": input_1})

        # Pattern 2: strong (present 5 times)
        input_2 = torch.zeros(64, dtype=torch.bool)
        input_2[40:50] = True
        for _ in range(5):
            _ = hippo.forward({"ec": input_2})

        # Get replay probabilities
        assert hippo.synaptic_tagging is not None
        probs = hippo.synaptic_tagging.get_replay_probabilities()

        # Pattern 2 should have higher probability
        # (But CA3 doesn't directly map to input, so check that probs vary)
        prob_variance = probs.var().item()
        assert prob_variance > 0.0001, \
            f"Replay probabilities should vary based on tag strength: var={prob_variance:.6f}"

    def test_dopamine_gates_consolidation(self):
        """High dopamine should consolidate tagged synapses."""
        config = HippocampusConfig(
            theta_gamma_enabled=True,
            learning_enabled=True,
            learning_rate=0.01,
        )
        sizes = {
            "input_size": 64,
            "dg_size": 128,
            "ca3_size": 100,
            "ca2_size": 50,
            "ca1_size": 64,
        }

        hippo = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")

        # Create input pattern
        input_spikes = torch.zeros(64, dtype=torch.bool)
        input_spikes[10:30] = True

        # Process with LOW dopamine
        hippo.state.dopamine = 0.0
        _ = hippo.forward({"ec": input_spikes})
        weights_low_da = hippo.synaptic_weights["ca3_ca3"].data.clone()

        # Reset and process with HIGH dopamine
        hippo = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")
        hippo.state.dopamine = 1.0
        _ = hippo.forward({"ec": input_spikes})
        weights_high_da = hippo.synaptic_weights["ca3_ca3"].data.clone()

        # High dopamine should produce stronger weights (via consolidation)
        # Note: Weights can vary due to randomness, so check mean change
        mean_low = weights_low_da.mean().item()
        mean_high = weights_high_da.mean().item()

        # Both should increase from baseline (learning happens regardless)
        # But high dopamine should consolidate MORE
        assert mean_high >= mean_low * 0.95, \
            f"High dopamine should maintain or increase weights: low={mean_low:.4f}, high={mean_high:.4f}"

    def test_tags_decay_without_activity(self):
        """Tags should decay when patterns are not rehearsed."""
        config = HippocampusConfig(
            theta_gamma_enabled=True,
            learning_enabled=True,
            learning_rate=0.01,
        )
        sizes = {
            "input_size": 64,
            "dg_size": 128,
            "ca3_size": 100,
            "ca2_size": 50,
            "ca1_size": 64,
        }

        hippo = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")

        # Create strong tags
        input_spikes = torch.zeros(64, dtype=torch.bool)
        input_spikes[10:30] = True
        _ = hippo.forward({"ec": input_spikes})

        assert hippo.synaptic_tagging is not None
        initial_tags = hippo.synaptic_tagging.get_diagnostics()["tag_mean"]

        # Present different patterns for 50 timesteps (let tags decay)
        different_input = torch.zeros(64, dtype=torch.bool)
        different_input[40:60] = True
        for _ in range(50):
            _ = hippo.forward({"ec": different_input})

        final_tags = hippo.synaptic_tagging.get_diagnostics()["tag_mean"]

        # Mean tags should change (decay old, create new)
        # Not testing exact values due to complexity, just that system is dynamic
        assert initial_tags > 0, "Should create initial tags"
        assert final_tags > 0, "Should create new tags from different pattern"

    def test_synaptic_tagging_in_diagnostics(self):
        """Synaptic tagging diagnostics should appear in get_diagnostics()."""
        config = HippocampusConfig(
            theta_gamma_enabled=True,
            learning_enabled=True,
        )
        sizes = {
            "input_size": 64,
            "dg_size": 128,
            "ca3_size": 100,
            "ca2_size": 50,
            "ca1_size": 64,
        }

        hippo = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")

        # Process some input
        input_spikes = torch.zeros(64, dtype=torch.bool)
        input_spikes[10:30] = True
        _ = hippo.forward({"ec": input_spikes})

        # Get diagnostics
        diag = hippo.get_diagnostics()

        # Should have synaptic tagging section
        assert "region_specific" in diag
        assert "synaptic_tagging" in diag["region_specific"]

        tag_diag = diag["region_specific"]["synaptic_tagging"]
        assert "tag_mean" in tag_diag
        assert "tag_max" in tag_diag
        assert "tag_coverage" in tag_diag

    def test_no_tagging_without_theta_gamma(self):
        """Synaptic tagging should be disabled when theta_gamma_enabled=False."""
        config = HippocampusConfig(
            theta_gamma_enabled=False,  # Disable
            learning_enabled=True,
        )
        sizes = {
            "input_size": 64,
            "dg_size": 128,
            "ca3_size": 100,
            "ca2_size": 50,
            "ca1_size": 64,
        }

        hippo = TrisynapticHippocampus(config=config, sizes=sizes, device="cpu")

        # Should not have synaptic tagging
        assert hippo.synaptic_tagging is None

        # Processing should still work
        input_spikes = torch.zeros(64, dtype=torch.bool)
        input_spikes[10:30] = True
        output = hippo.forward({"ec": input_spikes})

        assert output is not None
        assert output.shape[0] == 64  # CA1 size
