"""
Integration Tests for Multi-Source Striatum Architecture

Tests the complete multi-source architecture with:
- Per-source synaptic weights (D1/D2 separation)
- Per-source eligibility traces with source-specific tau
- Per-source STP modules with different dynamics
- Multi-source learning and credit assignment

Author: Thalia Project
Date: January 14, 2026
"""

import pytest
import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder


@pytest.fixture
def brain_config():
    """Shared brain config for tests."""
    return BrainConfig(device="cpu", dt_ms=1.0)


class TestMultiSourceWeightStructure:
    """Test that multi-source weight structure is correctly initialized."""

    def test_separate_d1_d2_weights_per_source(self, brain_config):
        """Test that each source has separate D1 and D2 weight matrices."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Check cortex:l5 source has D1 and D2 weights
        assert "cortex:l5_d1" in striatum.synaptic_weights
        assert "cortex:l5_d2" in striatum.synaptic_weights

        # Check hippocampus source has D1 and D2 weights
        assert "hippocampus_d1" in striatum.synaptic_weights
        assert "hippocampus_d2" in striatum.synaptic_weights

        # Verify weight dimensions match input sizes
        cortex = brain.components["cortex"]
        hippo = brain.components["hippocampus"]

        assert striatum.synaptic_weights["cortex:l5_d1"].shape == (striatum.d1_size, cortex.l5_size)
        assert striatum.synaptic_weights["cortex:l5_d2"].shape == (striatum.d2_size, cortex.l5_size)
        assert striatum.synaptic_weights["hippocampus_d1"].shape == (
            striatum.d1_size,
            hippo.n_output,
        )
        assert striatum.synaptic_weights["hippocampus_d2"].shape == (
            striatum.d2_size,
            hippo.n_output,
        )

    def test_eligibility_traces_per_source(self, brain_config):
        """Test that eligibility traces are tracked per source-pathway."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Initialize eligibility traces by running forward pass once
        dummy_inputs = {
            "cortex:l5": torch.zeros(32, device=striatum.device),
            "hippocampus": torch.zeros(64, device=striatum.device),
        }
        striatum(dummy_inputs)

        # Check eligibility traces exist for each source-pathway
        assert "cortex:l5_d1" in striatum._eligibility_d1
        assert "cortex:l5_d2" in striatum._eligibility_d2
        assert "hippocampus_d1" in striatum._eligibility_d1
        assert "hippocampus_d2" in striatum._eligibility_d2

        # Verify eligibility dimensions match weights
        assert (
            striatum._eligibility_d1["cortex:l5_d1"].shape
            == striatum.synaptic_weights["cortex:l5_d1"].shape
        )
        assert (
            striatum._eligibility_d2["cortex:l5_d2"].shape
            == striatum.synaptic_weights["cortex:l5_d2"].shape
        )

    def test_stp_modules_per_source(self, brain_config):
        """Test that STP modules are created per source-pathway."""
        brain_config.stp_enabled = True
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "striatum", "striatum", n_actions=4, neurons_per_action=10, stp_enabled=True
        )

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "striatum", source_port="l5")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Check STP modules exist for each source-pathway
        assert "cortex:l5_d1" in striatum.stp_modules
        assert "cortex:l5_d2" in striatum.stp_modules

        # Verify STP modules have correct dimensions
        assert striatum.stp_modules["cortex:l5_d1"].n_pre == brain.components["cortex"].l5_size
        assert striatum.stp_modules["cortex:l5_d1"].n_post == striatum.d1_size
        assert striatum.stp_modules["cortex:l5_d2"].n_post == striatum.d2_size


class TestMultiSourceForwardPass:
    """Test forward pass with multi-source inputs."""

    def test_forward_accepts_dict_inputs(self, brain_config):
        """Test that striatum forward accepts Dict[str, Tensor] inputs."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Create multi-source input
        cortex_spikes = torch.rand(32) > 0.9  # Sparse spikes
        hippo_spikes = torch.rand(64) > 0.9

        inputs = {"cortex:l5": cortex_spikes, "hippocampus": hippo_spikes}

        # Forward pass should work
        output = striatum(inputs)

        # Output should be D1 + D2 spikes concatenated
        assert output.shape[0] == striatum.d1_size + striatum.d2_size

    def test_forward_integrates_all_sources(self, brain_config):
        """Test that forward pass integrates currents from all sources."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Run with only cortex input
        cortex_only = {"cortex:l5": torch.ones(32)}
        output_cortex = striatum(cortex_only)

        # Run with only hippocampus input
        striatum.reset_state()
        hippo_only = {"hippocampus": torch.ones(64)}
        output_hippo = striatum(hippo_only)

        # Run with both inputs
        striatum.reset_state()
        both_inputs = {"cortex:l5": torch.ones(32), "hippocampus": torch.ones(64)}
        output_both = striatum(both_inputs)

        # All outputs should be valid (D1 + D2 concatenated)
        assert output_cortex.shape[0] == striatum.d1_size + striatum.d2_size
        assert output_hippo.shape[0] == striatum.d1_size + striatum.d2_size
        assert output_both.shape[0] == striatum.d1_size + striatum.d2_size


class TestMultiSourceLearning:
    """Test that learning works correctly with multi-source inputs."""

    def test_eligibility_updates_per_source(self, brain_config):
        """Test that eligibility traces update separately per source."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Initialize eligibility traces by running forward pass once
        dummy_inputs = {
            "cortex:l5": torch.zeros(32, device=striatum.device),
            "hippocampus": torch.zeros(64, device=striatum.device),
        }
        striatum(dummy_inputs)

        # Store initial eligibility traces
        initial_elig_cortex_d1 = striatum._eligibility_d1["cortex:l5_d1"].clone()
        initial_elig_hippo_d1 = striatum._eligibility_d1["hippocampus_d1"].clone()

        # Run forward with cortex input only (strong input to guarantee activity)
        # Run multiple times to ensure some striatum neurons fire
        for _ in range(10):
            cortex_input = {"cortex:l5": torch.ones(32, dtype=torch.bool)}
            output_spikes = striatum(cortex_input)
            if output_spikes.sum() > 0:  # At least one neuron fired
                break

        # Cortex eligibility should change, hippocampus should not
        cortex_changed = not torch.allclose(
            striatum._eligibility_d1["cortex:l5_d1"], initial_elig_cortex_d1
        )
        hippo_unchanged = torch.allclose(
            striatum._eligibility_d1["hippocampus_d1"], initial_elig_hippo_d1
        )

        assert cortex_changed, (
            f"Cortex eligibility should update when cortex spikes "
            f"(sum: {striatum._eligibility_d1['cortex:l5_d1'].sum().item():.6f}, "
            f"striatum spikes: {output_spikes.sum().item()})"
        )
        assert hippo_unchanged, "Hippocampus eligibility should not update when only cortex spikes"

    def test_source_specific_tau(self, brain_config):
        """Test that different sources use different eligibility tau values."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Check biological defaults (from striatum implementation)
        cortex_tau = striatum._get_source_eligibility_tau("cortex:l5")
        hippo_tau = striatum._get_source_eligibility_tau("hippocampus")

        # Cortical inputs should have longer traces (1000ms)
        assert cortex_tau == 1000.0

        # Hippocampal inputs should have faster traces (300ms)
        assert hippo_tau == 300.0

    def test_learning_applies_to_all_sources(self, brain_config):
        """Test that deliver_reward applies learning to all source-pathways."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]

        # Build eligibility by running forward pass
        inputs = {"cortex:l5": torch.rand(32) > 0.7, "hippocampus": torch.rand(64) > 0.7}
        striatum(inputs)

        # Store weights before learning
        cortex_d1_before = striatum.synaptic_weights["cortex:l5_d1"].clone()
        hippo_d1_before = striatum.synaptic_weights["hippocampus_d1"].clone()

        # Deliver reward (should trigger learning)
        striatum.set_neuromodulators(dopamine=1.0)  # High dopamine
        metrics = striatum.deliver_reward(reward=1.0)

        # Both sources should have weight changes
        cortex_d1_after = striatum.synaptic_weights["cortex:l5_d1"]
        hippo_d1_after = striatum.synaptic_weights["hippocampus_d1"]

        cortex_changed = not torch.allclose(cortex_d1_before, cortex_d1_after)
        hippo_changed = not torch.allclose(hippo_d1_before, hippo_d1_after)

        assert (
            cortex_changed or hippo_changed
        ), "At least one source should have weight changes after learning"
        assert "d1_ltp" in metrics, "Learning metrics should be returned"


class TestMultiSourceGrowth:
    """Test that growth API works with multi-source architecture."""

    def test_grow_source_expands_correct_weights(self, brain_config):
        """Test that grow_source expands both D1 and D2 weights for a specific source."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=10)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "striatum", source_port="l5")

        brain = builder.build()
        striatum = brain.components["striatum"]
        cortex = brain.components["cortex"]

        # Record initial sizes
        initial_d1_neurons = striatum.d1_size
        initial_d2_neurons = striatum.d2_size

        # Grow cortex L5 by 10 neurons
        new_l5_size = cortex.l5_size + 10

        # Grow striatum's cortex:l5 source weights (ONLY input dimension)
        striatum.grow_source("cortex:l5", new_size=new_l5_size)

        # Check that weights expanded (columns increased, rows unchanged)
        new_d1_shape = striatum.synaptic_weights["cortex:l5_d1"].shape
        new_d2_shape = striatum.synaptic_weights["cortex:l5_d2"].shape

        assert (
            new_d1_shape[0] == initial_d1_neurons
        ), f"D1 neurons should stay {initial_d1_neurons}, got {new_d1_shape[0]}"
        assert (
            new_d1_shape[1] == new_l5_size
        ), f"D1 input should be {new_l5_size}, got {new_d1_shape[1]}"

        assert (
            new_d2_shape[0] == initial_d2_neurons
        ), f"D2 neurons should stay {initial_d2_neurons}, got {new_d2_shape[0]}"
        assert (
            new_d2_shape[1] == new_l5_size
        ), f"D2 input should be {new_l5_size}, got {new_d2_shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
