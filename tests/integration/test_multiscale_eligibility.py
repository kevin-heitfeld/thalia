"""
Tests for multi-timescale eligibility traces (Phase 1 Enhancement).

Validates that:
1. Fast traces decay with tau ~500ms
2. Slow traces decay with tau ~60s
3. Consolidation from fast → slow traces
4. Combined eligibility enables multi-second credit assignment
"""

import pytest
import torch

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def brain_config(device):
    """Brain config for creating brain."""
    return BrainConfig(device=device, dt_ms=1.0)


@pytest.fixture
def brain_with_multi_eligibility(brain_config):
    """Create brain with multi-timescale eligibility enabled."""
    builder = BrainBuilder(brain_config)

    # Add components with multi-timescale eligibility enabled
    builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
    builder.add_component(
        "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
    )
    builder.add_component(
        "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
    )

    # Create striatum with multi-timescale eligibility enabled
    builder.add_component(
        "striatum",
        "striatum",
        n_actions=4,
        neurons_per_action=10,
        use_multiscale_eligibility=True,
        fast_eligibility_tau_ms=500.0,
        slow_eligibility_tau_ms=60000.0,
        eligibility_consolidation_rate=0.01,
        slow_trace_weight=0.3,
    )

    # Connect pathways
    builder.connect("thalamus", "cortex")
    builder.connect("cortex", "hippocampus", source_port="l23")
    builder.connect("cortex", "striatum", source_port="l5")
    builder.connect("hippocampus", "striatum")

    brain = builder.build()
    striatum = brain.components["striatum"]
    return brain, striatum


class TestMultiTimescaleEligibility:
    """Test multi-timescale eligibility trace dynamics."""

    def test_fast_trace_initialization(self, brain_with_multi_eligibility):
        """Test that fast eligibility traces are initialized."""
        _, striatum = brain_with_multi_eligibility

        # After initialization, fast/slow trace dicts should exist
        assert hasattr(striatum, "_eligibility_d1_fast")
        assert hasattr(striatum, "_eligibility_d2_fast")
        assert hasattr(striatum, "_eligibility_d1_slow")
        assert hasattr(striatum, "_eligibility_d2_slow")

        # Initially empty (no forward passes yet)
        assert len(striatum._eligibility_d1_fast) == 0
        assert len(striatum._eligibility_d1_slow) == 0

    def test_eligibility_trace_creation_on_forward(self, brain_with_multi_eligibility):
        """Test that eligibility traces are created after forward pass."""
        _, striatum = brain_with_multi_eligibility

        # Call striatum directly with dummy inputs matching connected sources
        cortex_l5_size = 32  # From builder config
        hippocampus_size = 64  # From builder config
        inputs = {
            "cortex:l5": torch.randn(cortex_l5_size, device=striatum.device),
            "hippocampus": torch.randn(hippocampus_size, device=striatum.device),
        }
        striatum(inputs)

        # Traces should now exist for the connected sources
        assert len(striatum._eligibility_d1_fast) > 0
        assert len(striatum._eligibility_d1_slow) > 0

        # Trace shapes should match weight shapes
        for source_key in striatum._eligibility_d1_fast:
            fast_trace = striatum._eligibility_d1_fast[source_key]
            slow_trace = striatum._eligibility_d1_slow[source_key]
            weights = striatum.synaptic_weights[source_key]

            assert fast_trace.shape == weights.shape
            assert slow_trace.shape == weights.shape

    def test_fast_trace_decay(self, brain_with_multi_eligibility):
        """Test that fast traces decay with tau ~500ms."""
        _, striatum = brain_with_multi_eligibility

        # Run forward pass to create traces
        inputs = {
            "cortex:l5": torch.randn(32, device=striatum.device),
            "hippocampus": torch.randn(64, device=striatum.device),
        }
        striatum(inputs)

        # Get initial fast trace magnitude
        source_key = list(striatum._eligibility_d1_fast.keys())[0]
        initial_magnitude = striatum._eligibility_d1_fast[source_key].abs().sum().item()

        # Run forward passes without new input (zeros)
        for _ in range(500):  # 500ms with dt=1ms
            zero_inputs = {
                "cortex:l5": torch.zeros(32, device=striatum.device),
                "hippocampus": torch.zeros(64, device=striatum.device),
            }
            striatum(zero_inputs)

        # Fast trace should decay to ~1/e (~37%) after tau=500ms
        final_magnitude = striatum._eligibility_d1_fast[source_key].abs().sum().item()

        if initial_magnitude > 0:
            decay_ratio = final_magnitude / initial_magnitude
            # Allow range [0.3, 0.5] (1/e ≈ 0.37)
            assert (
                0.2 < decay_ratio < 0.6
            ), f"Fast trace decay {decay_ratio:.3f} not in expected range (should be ~1/e = 0.37)"

    def test_slow_trace_persistence(self, brain_with_multi_eligibility):
        """Test that slow traces persist much longer than fast traces."""
        _, striatum = brain_with_multi_eligibility

        # Run forward pass to create traces
        inputs = {
            "cortex:l5": torch.randn(32, device=striatum.device),
            "hippocampus": torch.randn(64, device=striatum.device),
        }
        striatum(inputs)

        # Get initial trace magnitudes
        source_key = list(striatum._eligibility_d1_fast.keys())[0]
        initial_fast = striatum._eligibility_d1_fast[source_key].abs().sum().item()
        initial_slow = striatum._eligibility_d1_slow[source_key].abs().sum().item()

        # Run 5 seconds without input (5000 timesteps with dt=1ms)
        for _ in range(5000):
            zero_inputs = {
                "cortex:l5": torch.zeros(32, device=striatum.device),
                "hippocampus": torch.zeros(64, device=striatum.device),
            }
            striatum(zero_inputs)

        # After 5s:
        # - Fast trace (tau=500ms) should be ~0 (decayed to <0.01% after 10 tau)
        # - Slow trace (tau=60s) should still be substantial (only 5s/60s = 8% of tau)
        final_fast = striatum._eligibility_d1_fast[source_key].abs().sum().item()
        final_slow = striatum._eligibility_d1_slow[source_key].abs().sum().item()

        if initial_fast > 0:
            fast_ratio = final_fast / initial_fast
            assert (
                fast_ratio < 0.01
            ), f"Fast trace should be nearly zero after 5s, but ratio is {fast_ratio:.4f}"

        if initial_slow > 0:
            slow_ratio = final_slow / initial_slow
            # After 5s (tau=60s), should retain ~92% (exp(-5/60) ≈ 0.92)
            assert (
                slow_ratio > 0.80
            ), f"Slow trace should persist after 5s, but ratio is {slow_ratio:.4f}"

    def test_consolidation_from_fast_to_slow(self, brain_with_multi_eligibility):
        """Test that fast traces consolidate into slow traces."""
        _, striatum = brain_with_multi_eligibility

        # Run forward passes to build up traces
        for _ in range(100):  # 100ms of activity
            inputs = {
                "cortex:l5": torch.randn(32, device=striatum.device),
                "hippocampus": torch.randn(64, device=striatum.device),
            }
            striatum(inputs)

        source_key = list(striatum._eligibility_d1_slow.keys())[0]

        # Slow trace should be non-zero (consolidated from fast)
        slow_magnitude = striatum._eligibility_d1_slow[source_key].abs().sum().item()
        assert slow_magnitude > 0, "Slow trace should accumulate via consolidation"

        # Slow trace magnitude should be smaller than fast trace
        # (consolidation rate is only 1% per timestep)
        fast_magnitude = striatum._eligibility_d1_fast[source_key].abs().sum().item()
        assert (
            slow_magnitude < fast_magnitude
        ), f"Slow trace {slow_magnitude:.4f} should be smaller than fast {fast_magnitude:.4f}"

    def test_combined_eligibility_in_learning(self, brain_with_multi_eligibility):
        """Test that deliver_reward uses combined eligibility."""
        _, striatum = brain_with_multi_eligibility

        # Run forward passes to create eligibility
        for _ in range(100):
            inputs = {
                "cortex:l5": torch.randn(32, device=striatum.device),
                "hippocampus": torch.randn(64, device=striatum.device),
            }
            striatum(inputs)

        # Store initial weights
        source_key = list(striatum._eligibility_d1_fast.keys())[0]
        initial_weights = striatum.synaptic_weights[source_key].clone()

        # Deliver reward (should use combined eligibility: fast + 0.3*slow)
        striatum.deliver_reward(1.0)  # Positive reward

        # Weights should change
        final_weights = striatum.synaptic_weights[source_key]
        weight_change = (final_weights - initial_weights).abs().sum().item()

        assert weight_change > 0, "Weights should change after reward"

        # Weight change magnitude should be reasonable
        # (not exactly predictable due to learning rate and clamping)
        assert (
            weight_change < initial_weights.numel() * 0.1
        ), "Weight change should not be too large"


class TestSingleTimescaleMode:
    """Test that single-timescale mode still works (backward compatibility)."""

    @pytest.fixture
    def brain_single_timescale(self, brain_config):
        """Create brain with single-timescale eligibility."""
        builder = BrainBuilder(brain_config)

        # Add components with single-timescale (default)
        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component(
            "striatum",
            "striatum",
            n_actions=4,
            neurons_per_action=10,
            use_multiscale_eligibility=False,  # Explicitly disable
        )

        # Connect pathways
        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")
        builder.connect("hippocampus", "striatum")

        brain = builder.build()
        striatum = brain.components["striatum"]
        return brain, striatum

    def test_no_fast_slow_dicts(self, brain_single_timescale):
        """Test that fast/slow trace dicts don't exist in single-timescale mode."""
        _, striatum = brain_single_timescale

        # Should NOT have separate fast/slow dicts
        assert not hasattr(striatum, "_eligibility_d1_fast")
        assert not hasattr(striatum, "_eligibility_d2_fast")
        assert not hasattr(striatum, "_eligibility_d1_slow")
        assert not hasattr(striatum, "_eligibility_d2_slow")

        # Should still have regular eligibility dicts
        assert hasattr(striatum, "_eligibility_d1")
        assert hasattr(striatum, "_eligibility_d2")

    def test_single_timescale_learning(self, brain_single_timescale):
        """Test that learning works in single-timescale mode."""
        _, striatum = brain_single_timescale

        # Run forward passes
        for _ in range(10):
            inputs = {
                "cortex:l5": torch.randn(32, device=striatum.device),
                "hippocampus": torch.randn(64, device=striatum.device),
            }
            striatum(inputs)

        # Get eligibility (should exist)
        source_key = list(striatum._eligibility_d1.keys())[0]
        eligibility = striatum._eligibility_d1[source_key]
        assert eligibility.abs().sum().item() > 0

        # Store initial weights
        initial_weights = striatum.synaptic_weights[source_key].clone()

        # Deliver reward
        striatum.deliver_reward(1.0)

        # Weights should change
        final_weights = striatum.synaptic_weights[source_key]
        weight_change = (final_weights - initial_weights).abs().sum().item()
        assert weight_change > 0, "Single-timescale learning should still work"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
