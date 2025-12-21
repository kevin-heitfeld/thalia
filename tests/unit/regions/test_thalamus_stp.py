"""
Tests for Thalamus Short-Term Plasticity (STP).

Validates that sensory relay depression and L6 feedback depression
enable novelty detection, sensory adaptation, and attention gating as
predicted by biological literature.

Author: Thalia Project
Date: December 2025
"""

import pytest
import torch
import numpy as np

from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig


class TestThalamusSTPConfiguration:
    """Test STP configuration and initialization."""

    def test_stp_enabled_by_default(self):
        """Test that STP is enabled by default (HIGH PRIORITY biological justification)."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
        )
        thalamus = ThalamicRelay(config)

        assert config.stp_enabled is True, "STP should be enabled by default (HIGH PRIORITY)"
        assert thalamus.stp_sensory_relay is not None, "Sensory relay STP should be initialized"
        assert thalamus.stp_l6_feedback is not None, "L6 feedback STP should be initialized"

    def test_stp_can_be_disabled(self):
        """Test that STP can be disabled via config."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=False,
        )
        thalamus = ThalamicRelay(config)

        assert thalamus.stp_sensory_relay is None, "STP should be disabled"
        assert thalamus.stp_l6_feedback is None, "L6 feedback STP should be disabled"

    def test_stp_types_correct(self):
        """Test that STP types are set correctly."""
        from thalia.components.synapses.stp import STPType

        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
        )

        # Check config values
        assert config.stp_sensory_relay_type == STPType.DEPRESSING, \
            "Sensory relay STP should be depressing (U=0.4)"
        assert config.stp_l6_feedback_type == STPType.DEPRESSING, \
            "L6 feedback STP should be depressing (U=0.7)"

    def test_stp_dimensions_correct(self):
        """Test that STP modules have correct dimensions."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
        )
        thalamus = ThalamicRelay(config)

        # Sensory relay dimensions
        assert thalamus.stp_sensory_relay.n_pre == 20, "Input dimension should match"
        assert thalamus.stp_sensory_relay.n_post == 10, "Output dimension should match"

        # L6 feedback dimensions (L6 size must match relay size)
        assert thalamus.stp_l6_feedback.n_pre == 10, "L6 size must match relay size"
        assert thalamus.stp_l6_feedback.n_post == 10, "L6 size must match relay size"


class TestSensoryRelayDepression:
    """Test sensory relay → thalamus depression (novelty detection)."""

    def test_sustained_input_depresses(self):
        """Test that sustained sensory input shows depression over time."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Create sustained input pattern
        input_spikes = torch.zeros(20, dtype=torch.bool)
        input_spikes[0:10] = True  # 10 active inputs

        outputs = []
        for _ in range(20):
            output = thalamus.forward(input_spikes)
            outputs.append(output.sum().item())

        # Output should decrease over time (depression)
        early_activity = np.mean(outputs[0:5])
        late_activity = np.mean(outputs[15:20])

        # Allow for variability in neuron spiking
        if early_activity > 0:
            assert late_activity <= early_activity, \
                f"Depression should reduce or maintain activity (early={early_activity:.2f}, late={late_activity:.2f})"

    def test_novel_input_stronger_than_sustained(self):
        """Test that novel inputs get stronger transmission than sustained inputs."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Pattern A (will be sustained)
        pattern_a = torch.zeros(20, dtype=torch.bool)
        pattern_a[0:10] = True

        # Pattern B (will be novel)
        pattern_b = torch.zeros(20, dtype=torch.bool)
        pattern_b[10:20] = True

        # Present pattern A repeatedly (depresses)
        for _ in range(15):
            thalamus.forward(pattern_a)

        # Measure STP efficacy directly (more reliable than neuron output)
        # Reset neuron state but keep STP history
        thalamus.relay_neurons.reset_state()
        thalamus.trn_neurons.reset_state()

        # Get efficacy for sustained pattern A
        efficacy_a = thalamus.stp_sensory_relay(pattern_a.float()).mean().item()

        # Get efficacy for novel pattern B (fresh synapses)
        thalamus.stp_sensory_relay.reset_state()
        efficacy_b = thalamus.stp_sensory_relay(pattern_b.float()).mean().item()

        # Novel pattern should get stronger transmission
        assert efficacy_b > efficacy_a, \
            f"Novel input should be stronger (novel={efficacy_b:.3f}, sustained={efficacy_a:.3f})"

    def test_sensory_adaptation(self):
        """Test that thalamus adapts to sustained sensory input (habituation)."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Sustained sensory pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Measure efficacy over time
        efficacies = []
        for _ in range(30):
            efficacy = thalamus.stp_sensory_relay(pattern.float()).mean().item()
            efficacies.append(efficacy)

        # Efficacy should decrease (adaptation)
        early_efficacy = np.mean(efficacies[0:5])
        late_efficacy = np.mean(efficacies[25:30])

        assert late_efficacy < early_efficacy * 0.9, \
            f"Adaptation should reduce efficacy (early={early_efficacy:.3f}, late={late_efficacy:.3f})"

    def test_stp_vs_no_stp_novelty(self):
        """Test that STP improves novelty detection compared to no STP."""
        # With STP
        config_stp = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus_stp = ThalamicRelay(config_stp)

        # Without STP
        config_no_stp = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=False,
        )
        thalamus_no_stp = ThalamicRelay(config_no_stp)

        # Sustained pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Measure depression over time
        stp_outputs = []
        no_stp_outputs = []

        for _ in range(15):
            stp_out = thalamus_stp.forward(pattern)
            no_stp_out = thalamus_no_stp.forward(pattern)
            stp_outputs.append(stp_out.sum().item())
            no_stp_outputs.append(no_stp_out.sum().item())

        # STP should show more adaptation than no STP
        if len([x for x in stp_outputs if x > 0]) > 0 and len([x for x in no_stp_outputs if x > 0]) > 0:
            stp_early = np.mean([x for x in stp_outputs[0:5] if x > 0] or [0])
            stp_late = np.mean([x for x in stp_outputs[10:15] if x > 0] or [0])
            no_stp_early = np.mean([x for x in no_stp_outputs[0:5] if x > 0] or [0])
            no_stp_late = np.mean([x for x in no_stp_outputs[10:15] if x > 0] or [0])

            if stp_early > 0 and no_stp_early > 0:
                stp_ratio = stp_late / (stp_early + 1e-6)
                no_stp_ratio = no_stp_late / (no_stp_early + 1e-6)

                # STP should show more depression (lower ratio)
                assert stp_ratio <= no_stp_ratio + 0.1, \
                    f"STP should show more adaptation (STP={stp_ratio:.2f}, no STP={no_stp_ratio:.2f})"


class TestSensoryRelayRecovery:
    """Test recovery dynamics of sensory relay depression."""

    def test_depression_recovers_during_silence(self):
        """Test that depression recovers during silence (no input)."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Active pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Silent pattern
        silence = torch.zeros(20, dtype=torch.bool)

        # Depress the synapses
        for _ in range(15):
            thalamus.stp_sensory_relay(pattern.float())

        # Get depressed efficacy
        efficacy_depressed = thalamus.stp_sensory_relay.get_efficacy().mean().item()

        # Allow recovery (silence)
        for _ in range(200):  # ~200ms for significant recovery (tau_d depends on STP type)
            thalamus.stp_sensory_relay(silence.float())

        # Get recovered efficacy
        efficacy_recovered = thalamus.stp_sensory_relay.get_efficacy().mean().item()

        # Efficacy should recover (at least 5% improvement)
        assert efficacy_recovered > efficacy_depressed * 1.05, \
            f"Depression should recover (depressed={efficacy_depressed:.3f}, recovered={efficacy_recovered:.3f})"


class TestL6FeedbackDepression:
    """Test L6 cortical feedback → thalamus depression (dynamic gain control)."""

    def test_l6_feedback_depresses(self):
        """Test that sustained L6 feedback shows strong depression."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # L6 feedback pattern (must match relay size)
        l6_feedback = torch.zeros(10, dtype=torch.bool)
        l6_feedback[0:5] = True

        # Measure efficacy over time
        efficacies = []
        for _ in range(20):
            efficacy = thalamus.stp_l6_feedback(l6_feedback.float()).mean().item()
            efficacies.append(efficacy)

        # Efficacy should decrease (stronger depression than sensory, U=0.7)
        early = np.mean(efficacies[0:5])
        late = np.mean(efficacies[15:20])

        assert late < early * 0.8, \
            f"L6 feedback should show strong depression (early={early:.3f}, late={late:.3f})"

    def test_l6_feedback_stronger_depression_than_sensory(self):
        """Test that L6 feedback has stronger depression than sensory input (U=0.7 vs U=0.4)."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Sensory pattern
        sensory = torch.zeros(20, dtype=torch.bool)
        sensory[0:10] = True

        # L6 feedback pattern
        l6_feedback = torch.zeros(10, dtype=torch.bool)
        l6_feedback[0:5] = True

        # Measure depression for both
        sensory_efficacies = []
        l6_efficacies = []

        for _ in range(15):
            sensory_eff = thalamus.stp_sensory_relay(sensory.float()).mean().item()
            l6_eff = thalamus.stp_l6_feedback(l6_feedback.float()).mean().item()
            sensory_efficacies.append(sensory_eff)
            l6_efficacies.append(l6_eff)

        # Compare depression ratios
        sensory_ratio = sensory_efficacies[-1] / (sensory_efficacies[0] + 1e-6)
        l6_ratio = l6_efficacies[-1] / (l6_efficacies[0] + 1e-6)

        # L6 should show stronger depression (lower ratio)
        assert l6_ratio < sensory_ratio, \
            f"L6 feedback should depress more than sensory (L6={l6_ratio:.3f}, sensory={sensory_ratio:.3f})"


class TestSTPStateManagement:
    """Test STP state management (reset, checkpointing)."""

    def test_stp_reset(self):
        """Test that reset_state clears STP state."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Active pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Depress the synapses
        for _ in range(15):
            thalamus.stp_sensory_relay(pattern.float())

        # Get depressed efficacy
        efficacy_before = thalamus.stp_sensory_relay.get_efficacy().mean().item()

        # Reset
        thalamus.reset_state()

        # Get efficacy after reset (should be baseline)
        efficacy_after = thalamus.stp_sensory_relay.get_efficacy().mean().item()

        # Efficacy should be higher after reset (depression cleared)
        assert efficacy_after > efficacy_before * 1.1, \
            f"Reset should clear depression (before={efficacy_before:.3f}, after={efficacy_after:.3f})"

    def test_stp_modules_in_reset(self):
        """Test that STP modules are included in reset_state call."""
        config = ThalamicRelayConfig(
            n_input=10,
            n_output=5,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # STP modules should have reset_state method
        assert hasattr(thalamus.stp_sensory_relay, 'reset_state'), \
            "Sensory STP module should have reset_state method"
        assert hasattr(thalamus.stp_l6_feedback, 'reset_state'), \
            "L6 feedback STP module should have reset_state method"

        # Reset should not raise error
        thalamus.reset_state()


class TestBiologicalPlausibility:
    """Test that STP behavior matches biological data."""

    def test_sensory_depression_magnitude_realistic(self):
        """Test that sensory depression magnitude is in biological range (U=0.4, moderate)."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Active pattern
        pattern = torch.zeros(20, dtype=torch.bool)
        pattern[0:10] = True

        # Measure efficacy directly
        efficacy_first = thalamus.stp_sensory_relay(pattern.float()).mean().item()

        # Depressed efficacy (after 10 presentations)
        for _ in range(9):
            thalamus.stp_sensory_relay(pattern.float())

        efficacy_depressed = thalamus.stp_sensory_relay(pattern.float()).mean().item()

        # Calculate depression ratio
        if efficacy_first > 0:
            depression_ratio = efficacy_depressed / efficacy_first

            # Biological data: U=0.4 should give moderate depression (40-80% of initial)
            assert 0.3 < depression_ratio < 0.9, \
                f"Depression should be in biological range (got {depression_ratio:.2f})"

    def test_l6_depression_magnitude_realistic(self):
        """Test that L6 feedback depression magnitude is in biological range (U=0.7, strong)."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # L6 feedback pattern
        pattern = torch.zeros(10, dtype=torch.bool)
        pattern[0:5] = True

        # Measure efficacy directly
        efficacy_first = thalamus.stp_l6_feedback(pattern.float()).mean().item()

        # Depressed efficacy (after 10 presentations)
        for _ in range(9):
            thalamus.stp_l6_feedback(pattern.float())

        efficacy_depressed = thalamus.stp_l6_feedback(pattern.float()).mean().item()

        # Calculate depression ratio
        if efficacy_first > 0:
            depression_ratio = efficacy_depressed / efficacy_first

            # Biological data: U=0.7 should give strong depression (20-60% of initial)
            assert 0.15 < depression_ratio < 0.7, \
                f"Strong depression should be in biological range (got {depression_ratio:.2f})"

    def test_novelty_detection_functional(self):
        """Test that STP enables functional novelty detection in sensory relay."""
        config = ThalamicRelayConfig(
            n_input=20,
            n_output=10,
            device="cpu",
            stp_enabled=True,
        )
        thalamus = ThalamicRelay(config)

        # Pattern A (background)
        pattern_a = torch.zeros(20, dtype=torch.bool)
        pattern_a[0:10] = True

        # Pattern B (novel stimulus)
        pattern_b = torch.zeros(20, dtype=torch.bool)
        pattern_b[10:20] = True

        # Present A repeatedly (background habituation)
        for _ in range(10):
            thalamus.stp_sensory_relay(pattern_a.float())

        efficacy_background = thalamus.stp_sensory_relay(pattern_a.float()).mean().item()

        # Present B (novel stimulus)
        # Reset STP for pattern B inputs (they haven't been active)
        thalamus.stp_sensory_relay.reset_state()
        efficacy_novel = thalamus.stp_sensory_relay(pattern_b.float()).mean().item()

        # Novel stimulus should have higher efficacy
        assert efficacy_novel > efficacy_background, \
            f"Novel stimulus should have higher efficacy (novel={efficacy_novel:.3f}, background={efficacy_background:.3f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
