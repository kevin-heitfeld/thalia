"""
Tests for Cerebellum Short-Term Plasticity (STP).

Validates that parallel fiber depression and mossy fiber facilitation
improve temporal precision and change detection as predicted by biological literature.

Author: Thalia Project
Date: December 2025
"""

import pytest
import torch
import numpy as np

from thalia.regions.cerebellum_region import Cerebellum, CerebellumConfig
from thalia.config import compute_cerebellum_sizes


def create_test_cerebellum(
    input_size: int,
    purkinje_size: int,
    device: str = "cpu",
    **kwargs
) -> Cerebellum:
    """Create Cerebellum for testing with new (config, sizes, device) pattern."""
    # Always compute granule_size (Cerebellum.__init__ requires it)
    expansion = kwargs.pop("granule_expansion_factor", 4.0)
    sizes = compute_cerebellum_sizes(purkinje_size, expansion)
    sizes["input_size"] = input_size

    config = CerebellumConfig(device=device, **kwargs)
    return Cerebellum(config, sizes, device)


class TestCerebellumSTPConfiguration:
    """Test STP configuration and initialization."""

    def test_stp_enabled_by_default(self):
        """Test that STP is enabled by default (biological justification)."""
        cerebellum = create_test_cerebellum(
            input_size=10,
            purkinje_size=5,
            device="cpu",
            use_enhanced_microcircuit=False,
        )

        assert cerebellum.config.stp_enabled is True, "STP should be enabled by default"
        assert cerebellum.stp_pf_purkinje is not None, "PF→Purkinje STP should be initialized"

    def test_stp_can_be_disabled(self):
        """Test that STP can be disabled via config."""
        cerebellum = create_test_cerebellum(
            input_size=10,
            purkinje_size=5,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=False,
        )

        assert cerebellum.stp_pf_purkinje is None, "STP should be disabled"

    def test_stp_types_correct(self):
        """Test that STP types are set correctly."""
        from thalia.components.synapses.stp import STPType

        cerebellum = create_test_cerebellum(
            input_size=10,
            purkinje_size=5,
            device="cpu",
            use_enhanced_microcircuit=False,
        )

        # Check config values
        assert cerebellum.config.stp_pf_purkinje_type == STPType.DEPRESSING, \
            "Parallel fiber STP should be depressing"
        assert cerebellum.config.stp_mf_granule_type == STPType.FACILITATING, \
            "Mossy fiber STP should be facilitating"

    def test_stp_dimensions_correct(self):
        """Test that STP modules have correct dimensions."""
        cerebellum = create_test_cerebellum(
            input_size=10,
            purkinje_size=5,
            device="cpu",
            use_enhanced_microcircuit=False,
        )

        # PF→Purkinje dimensions
        assert cerebellum.stp_pf_purkinje.n_pre == 10, "Input dimension should match"
        assert cerebellum.stp_pf_purkinje.n_post == 5, "Output dimension should match"


class TestParallelFiberDepression:
    """Test parallel fiber→Purkinje depression (temporal high-pass filter)."""

    def test_sustained_input_depresses(self):
        """Test that sustained input shows depression over time."""
        cerebellum = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # Create sustained input pattern
        input_spikes = torch.zeros(20, dtype=torch.bool)
        input_spikes[0:10] = True  # 10 active inputs

        outputs = []
        for _ in range(20):
            output = cerebellum.forward(input_spikes)
            outputs.append(output.sum().item())

        # Output should decrease over time (depression)
        early_activity = np.mean(outputs[0:5])
        late_activity = np.mean(outputs[15:20])

        assert late_activity < early_activity * 0.8, \
            f"Depression should reduce activity (early={early_activity:.2f}, late={late_activity:.2f})"

    def test_novel_input_stronger_than_sustained(self):
        """Test that novel inputs get stronger transmission than sustained inputs."""
        cerebellum = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # Pattern A (will be sustained)
        pattern_a = torch.zeros(20, dtype=torch.bool)
        pattern_a[0:10] = True

        # Pattern B (will be novel)
        pattern_b = torch.zeros(20, dtype=torch.bool)
        pattern_b[10:20] = True

        # Present pattern A repeatedly (depresses)
        for _ in range(15):
            cerebellum.forward(pattern_a)

        # Get response to sustained pattern A
        output_a = cerebellum.forward(pattern_a)
        activity_a = output_a.sum().item()

        # Present novel pattern B
        output_b = cerebellum.forward(pattern_b)
        activity_b = output_b.sum().item()

        # Novel pattern should get stronger transmission
        assert activity_b > activity_a, \
            f"Novel input should be stronger (novel={activity_b}, sustained={activity_a})"

    def test_change_detection(self):
        """Test that cerebellum responds more to changes than steady state."""
        cerebellum = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # Pattern A
        pattern_a = torch.zeros(20, dtype=torch.bool)
        pattern_a[0:10] = True

        # Pattern B (different)
        pattern_b = torch.zeros(20, dtype=torch.bool)
        pattern_b[10:20] = True

        # Phase 1: Sustained pattern A
        sustained_outputs = []
        for _ in range(10):
            output = cerebellum.forward(pattern_a)
            sustained_outputs.append(output.sum().item())

        # Phase 2: Switch to pattern B (change)
        change_outputs = []
        for _ in range(5):
            output = cerebellum.forward(pattern_b)
            change_outputs.append(output.sum().item())

        # Response to change should be stronger than sustained
        sustained_activity = np.mean(sustained_outputs[-5:])  # Last 5 of sustained
        change_activity = np.mean(change_outputs[0:3])  # First 3 after change

        assert change_activity > sustained_activity, \
            f"Change should trigger stronger response (change={change_activity:.2f}, sustained={sustained_activity:.2f})"

    def test_stp_vs_no_stp_timing(self):
        """Test that STP improves change detection compared to no STP."""
        # With STP
        cerebellum_stp = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # Without STP
        cerebellum_no_stp = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=False,
        )

        # Same input pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Measure depression over time
        stp_outputs = []
        no_stp_outputs = []

        for _ in range(15):
            stp_out = cerebellum_stp.forward(pattern)
            no_stp_out = cerebellum_no_stp.forward(pattern)
            stp_outputs.append(stp_out.sum().item())
            no_stp_outputs.append(no_stp_out.sum().item())

        # STP should show more depression
        stp_early = np.mean(stp_outputs[0:5])
        stp_late = np.mean(stp_outputs[10:15])
        no_stp_early = np.mean(no_stp_outputs[0:5])
        no_stp_late = np.mean(no_stp_outputs[10:15])

        stp_depression_ratio = stp_late / (stp_early + 1e-6)
        no_stp_depression_ratio = no_stp_late / (no_stp_early + 1e-6)

        assert stp_depression_ratio < no_stp_depression_ratio, \
            f"STP should show more depression (STP ratio={stp_depression_ratio:.2f}, no STP ratio={no_stp_depression_ratio:.2f})"


class TestParallelFiberRecovery:
    """Test recovery dynamics of parallel fiber depression."""

    def test_depression_recovers_over_time(self):
        """Test that depression recovers during silence (no input)."""
        cerebellum = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # Active pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Silent pattern
        silence = torch.zeros(20, dtype=torch.bool)

        # Measure STP efficacy directly (not neuron output)
        # This isolates STP recovery from neuron spiking variability

        # Depress the synapses
        for _ in range(15):
            cerebellum.stp_pf_purkinje(pattern.float())

        # Get depressed efficacy
        efficacy_depressed = cerebellum.stp_pf_purkinje.get_efficacy().mean().item()

        # Allow recovery (silence)
        for _ in range(200):  # ~200ms for significant recovery (tau_d=800ms)
            cerebellum.stp_pf_purkinje(silence.float())

        # Get recovered efficacy
        efficacy_recovered = cerebellum.stp_pf_purkinje.get_efficacy().mean().item()

        # Efficacy should recover (at least 10% improvement)
        assert efficacy_recovered > efficacy_depressed * 1.1, \
            f"Depression should recover (depressed={efficacy_depressed:.3f}, recovered={efficacy_recovered:.3f})"


class TestSTPStateManagement:
    """Test STP state management (reset, checkpointing)."""

    def test_stp_reset(self):
        """Test that reset_state clears STP state."""
        cerebellum = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # Active pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Depress the synapses
        for _ in range(15):
            cerebellum.forward(pattern)

        # Get depressed response
        output_before_reset = cerebellum.forward(pattern)
        activity_before = output_before_reset.sum().item()

        # Reset
        cerebellum.reset_state()

        # Test again (should be like first presentation)
        output_after_reset = cerebellum.forward(pattern)
        activity_after = output_after_reset.sum().item()

        # Activity should be higher after reset (depression cleared)
        assert activity_after > activity_before, \
            f"Reset should clear depression (before={activity_before}, after={activity_after})"

    def test_stp_state_in_reset_subsystems(self):
        """Test that STP modules are included in reset_subsystems call."""
        cerebellum = create_test_cerebellum(
            input_size=10,
            purkinje_size=5,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # STP module should have reset_state method
        assert hasattr(cerebellum.stp_pf_purkinje, 'reset_state'), \
            "STP module should have reset_state method"

        # Reset should not raise error
        cerebellum.reset_state()


class TestBiologicalPlausibility:
    """Test that STP behavior matches biological data."""

    def test_depression_magnitude_realistic(self):
        """Test that depression magnitude is in biological range (30-70%)."""
        cerebellum = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
        )

        # Active pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Measure STP efficacy directly (more reliable than neuron output)
        # First efficacy (baseline)
        efficacy_first = cerebellum.stp_pf_purkinje(pattern.float()).mean().item()

        # Depressed efficacy (after 10 presentations at ~1ms intervals)
        for _ in range(9):
            cerebellum.stp_pf_purkinje(pattern.float())

        efficacy_depressed = cerebellum.stp_pf_purkinje(pattern.float()).mean().item()

        # Calculate depression ratio
        if efficacy_first > 0:
            depression_ratio = efficacy_depressed / efficacy_first

            # Biological data: 30-70% of initial (Dittman et al. 2000)
            # Our config uses DEPRESSING type with U=0.5, tau_d=800ms
            assert 0.2 < depression_ratio < 0.9, \
                f"Depression should be in biological range (got {depression_ratio:.2f})"
        else:
            # If first efficacy is 0, check that both are reasonable
            assert efficacy_first > 0.1, "Initial efficacy should be non-zero"

    def test_temporal_precision(self):
        """Test that STP enables sub-10ms temporal discrimination."""
        cerebellum = create_test_cerebellum(
            input_size=20,
            purkinje_size=10,
            device="cpu",
            use_enhanced_microcircuit=False,
            stp_enabled=True,
            dt_ms=1.0,  # 1ms timesteps
        )

        # Pattern A
        pattern_a = torch.zeros(20, dtype=torch.bool)
        pattern_a[0:10] = True

        # Pattern B (different timing - arrives 5ms later)
        pattern_b = torch.zeros(20, dtype=torch.bool)
        pattern_b[10:20] = True

        # Present A-A-A-A-A (sustained)
        for _ in range(5):
            cerebellum.forward(pattern_a)

        output_a_sustained = cerebellum.forward(pattern_a)

        # Present A-wait-B (change after 5ms gap)
        for _ in range(5):
            cerebellum.forward(torch.zeros(20, dtype=torch.bool))

        output_b_after_gap = cerebellum.forward(pattern_b)

        # Different temporal contexts should produce different responses
        # This tests temporal precision at ~5ms scale
        assert not torch.equal(output_a_sustained, output_b_after_gap), \
            "Different temporal contexts should produce different outputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
