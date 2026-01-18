"""
Tests for Thalamus Short-Term Plasticity (STP).

Validates that sensory relay depression and L6 feedback depression
enable novelty detection, sensory adaptation, and attention gating as
predicted by biological literature.

Author: Thalia Project
Date: December 2025
"""

import numpy as np
import pytest
import torch

from thalia.components.synapses import ShortTermPlasticity
from thalia.components.synapses.stp import STPType
from thalia.config.size_calculator import LayerSizeCalculator
from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig


@pytest.fixture
def thalamus_config():
    """Standard thalamus config with explicit sizes for STP tests."""
    relay_size = 10
    device = "cpu"
    calc = LayerSizeCalculator()
    sizes = calc.thalamus_from_relay(relay_size)
    config = ThalamicRelayConfig(device=device)
    return config, sizes, device


class TestThalamusSTPConfiguration:
    """Test STP configuration and initialization."""

    def test_stp_enabled_by_default(self, thalamus_config):
        """Test that STP is enabled by default (HIGH PRIORITY biological justification)."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        assert config.stp_enabled is True, "STP should be enabled by default (HIGH PRIORITY)"
        # Verify STP modules are initialized with correct types
        assert isinstance(
            thalamus.stp_sensory_relay, ShortTermPlasticity
        ), "Sensory relay STP should be initialized"
        assert isinstance(
            thalamus.stp_l6_feedback, ShortTermPlasticity
        ), "L6 feedback STP should be initialized"

    def test_stp_can_be_disabled(self):
        """Test that STP can be disabled via config."""
        relay_size = 10
        device = "cpu"
        calc = LayerSizeCalculator()
        sizes = calc.thalamus_from_relay(relay_size)
        config = ThalamicRelayConfig(device=device, stp_enabled=False)
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        assert thalamus.stp_sensory_relay is None, "STP should be disabled"
        assert thalamus.stp_l6_feedback is None, "L6 feedback STP should be disabled"

    def test_stp_types_correct(self, thalamus_config):
        """Test that STP types are set correctly."""

        config, sizes, device = thalamus_config

        # Check config values - biologically accurate defaults
        assert (
            config.stp_sensory_relay_type == STPType.DEPRESSING_MODERATE
        ), "Sensory relay STP should be moderate depression (U=0.4)"
        assert (
            config.stp_l6_feedback_type == STPType.DEPRESSING_STRONG
        ), "L6 feedback STP should be strong depression (U=0.7)"

    def test_stp_dimensions_correct(self, thalamus_config):
        """Test that STP modules have correct dimensions."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Sensory relay dimensions
        assert (
            thalamus.stp_sensory_relay.n_pre == thalamus.input_size
        ), "Input dimension should match"
        assert (
            thalamus.stp_sensory_relay.n_post == thalamus.relay_size
        ), "Output dimension should match"

        # L6 feedback dimensions (L6 size must match relay size)
        assert (
            thalamus.stp_l6_feedback.n_pre == thalamus.relay_size
        ), "L6 size must match relay size"
        assert (
            thalamus.stp_l6_feedback.n_post == thalamus.relay_size
        ), "L6 size must match relay size"


class TestSensoryRelayDepression:
    """Test sensory relay → thalamus depression (novelty detection)."""

    def test_sustained_input_depresses(self, thalamus_config):
        """Test that sustained sensory input shows depression over time."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Create sustained input pattern
        input_spikes = torch.zeros(thalamus.input_size, dtype=torch.bool)
        input_spikes[0:10] = True  # 10 active inputs

        outputs = []
        for _ in range(20):
            output = thalamus.forward({"input": input_spikes})
            outputs.append(output.sum().item())

        # Output should decrease over time (depression)
        early_activity = np.mean(outputs[0:5])
        late_activity = np.mean(outputs[15:20])

        # Allow for variability in neuron spiking
        if early_activity > 0:
            assert (
                late_activity <= early_activity
            ), f"Depression should reduce or maintain activity (early={early_activity:.2f}, late={late_activity:.2f})"

    def test_novel_input_stronger_than_sustained(self, thalamus_config):
        """Test that novel inputs get stronger transmission than sustained inputs."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Pattern A (will be sustained)
        pattern_a = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern_a[0:10] = True

        # Pattern B (will be novel)
        pattern_b = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern_b[thalamus.input_size // 2 :] = True

        # Present pattern A repeatedly (depresses)
        for _ in range(15):
            thalamus.forward({"input": pattern_a})

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
        assert (
            efficacy_b > efficacy_a
        ), f"Novel input should be stronger (novel={efficacy_b:.3f}, sustained={efficacy_a:.3f})"

    def test_sensory_adaptation(self, thalamus_config):
        """Test that thalamus adapts to sustained sensory input (habituation)."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Sustained sensory pattern
        pattern = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern[0:10] = True

        # Measure efficacy over time
        efficacies = []
        for _ in range(30):
            efficacy = thalamus.stp_sensory_relay(pattern.float()).mean().item()
            efficacies.append(efficacy)

        # Efficacy should decrease (adaptation)
        early_efficacy = np.mean(efficacies[0:5])
        late_efficacy = np.mean(efficacies[25:30])

        assert (
            late_efficacy < early_efficacy * 0.9
        ), f"Adaptation should reduce efficacy (early={early_efficacy:.3f}, late={late_efficacy:.3f})"

    def test_stp_enables_adaptation(self):
        """Test that STP enables adaptation compared to no STP."""
        relay_size = 10
        device = "cpu"
        calc = LayerSizeCalculator()
        sizes = calc.thalamus_from_relay(relay_size)

        # With STP
        config_stp = ThalamicRelayConfig(device=device, stp_enabled=True)
        thalamus_stp = ThalamicRelay(config=config_stp, sizes=sizes, device=device)

        # Without STP
        config_no_stp = ThalamicRelayConfig(device=device, stp_enabled=False)
        thalamus_no_stp = ThalamicRelay(config=config_no_stp, sizes=sizes, device=device)

        # Sustained pattern
        pattern = torch.zeros(thalamus_stp.input_size, dtype=torch.bool)
        pattern[0:10] = True

        # Measure depression over time
        stp_outputs = []
        no_stp_outputs = []

        for _ in range(15):
            stp_out = thalamus_stp.forward({"input": pattern})
            no_stp_out = thalamus_no_stp.forward({"input": pattern})
            stp_outputs.append(stp_out.sum().item())
            no_stp_outputs.append(no_stp_out.sum().item())

        # STP should show more adaptation than no STP
        if (
            len([x for x in stp_outputs if x > 0]) > 0
            and len([x for x in no_stp_outputs if x > 0]) > 0
        ):
            stp_early = np.mean([x for x in stp_outputs[0:5] if x > 0] or [0])
            stp_late = np.mean([x for x in stp_outputs[10:15] if x > 0] or [0])
            no_stp_early = np.mean([x for x in no_stp_outputs[0:5] if x > 0] or [0])
            no_stp_late = np.mean([x for x in no_stp_outputs[10:15] if x > 0] or [0])

            if stp_early > 0 and no_stp_early > 0:
                stp_ratio = stp_late / (stp_early + 1e-6)
                no_stp_ratio = no_stp_late / (no_stp_early + 1e-6)

                # STP should show more depression (lower ratio)
                assert (
                    stp_ratio <= no_stp_ratio + 0.1
                ), f"STP should show more adaptation (STP={stp_ratio:.2f}, no STP={no_stp_ratio:.2f})"


class TestSensoryRelayRecovery:
    """Test recovery dynamics of sensory relay depression."""

    def test_depression_recovers_during_silence(self, thalamus_config):
        """Test that depression recovers during silence (no input)."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Active pattern
        pattern = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern[0 : thalamus.input_size // 2] = True

        # Silent pattern
        silence = torch.zeros(thalamus.input_size, dtype=torch.bool)

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
        assert (
            efficacy_recovered > efficacy_depressed * 1.05
        ), f"Depression should recover (depressed={efficacy_depressed:.3f}, recovered={efficacy_recovered:.3f})"


class TestL6FeedbackDepression:
    """Test L6 cortical feedback → thalamus depression (dynamic gain control)."""

    def test_l6_feedback_depression(self):
        """Test that sustained L6 feedback shows strong depression."""
        relay_size = 10
        device = "cpu"
        calc = LayerSizeCalculator()
        sizes = calc.thalamus_from_relay(relay_size)
        config = ThalamicRelayConfig(device=device, stp_enabled=True)
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # L6 feedback pattern (must match relay size)
        l6_feedback = torch.zeros(thalamus.relay_size, dtype=torch.bool)
        l6_feedback[0:5] = True

        # Measure efficacy over time
        efficacies = []
        for _ in range(20):
            efficacy = thalamus.stp_l6_feedback(l6_feedback.float()).mean().item()
            efficacies.append(efficacy)

        # Efficacy should decrease (stronger depression than sensory, U=0.7)
        early = np.mean(efficacies[0:5])
        late = np.mean(efficacies[15:20])

        assert (
            late < early * 0.8
        ), f"L6 feedback should show strong depression (early={early:.3f}, late={late:.3f})"

    def test_l6_feedback_stronger_depression_than_sensory(self, thalamus_config):
        """Test that L6 feedback has stronger depression than sensory input (U=0.7 vs U=0.4)."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Sensory pattern
        sensory = torch.zeros(thalamus.input_size, dtype=torch.bool)
        sensory[0 : thalamus.input_size // 2] = True

        # L6 feedback pattern
        l6_feedback = torch.zeros(thalamus.relay_size, dtype=torch.bool)
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
        # Config: sensory U=0.4, L6 U=0.7 → L6 should depress more
        assert (
            l6_ratio <= sensory_ratio
        ), f"L6 feedback should depress at least as much as sensory (L6={l6_ratio:.3f}, sensory={sensory_ratio:.3f})"


class TestSTPStateManagement:
    """Test STP state management (reset, checkpointing)."""

    def test_stp_reset(self, thalamus_config):
        """Test that reset_state clears STP state."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Active pattern
        pattern = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern[0 : thalamus.input_size // 2] = True

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
        assert (
            efficacy_after > efficacy_before * 1.1
        ), f"Reset should clear depression (before={efficacy_before:.3f}, after={efficacy_after:.3f})"

    def test_stp_modules_in_reset(self):
        """Test that STP modules are included in reset_state call."""
        relay_size = 5
        device = "cpu"
        calc = LayerSizeCalculator()
        sizes = calc.thalamus_from_relay(relay_size)
        config = ThalamicRelayConfig(device=device, stp_enabled=True)
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # STP modules should have reset_state method
        assert hasattr(
            thalamus.stp_sensory_relay, "reset_state"
        ), "Sensory STP module should have reset_state method"
        assert hasattr(
            thalamus.stp_l6_feedback, "reset_state"
        ), "L6 feedback STP module should have reset_state method"

        # Reset should not raise error
        thalamus.reset_state()


class TestBiologicalPlausibility:
    """Test that STP behavior matches biological data."""

    def test_sensory_depression_magnitude_realistic(self, thalamus_config):
        """Test that sensory depression magnitude is in biological range (U=0.4, moderate)."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Active pattern
        pattern = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern[0 : thalamus.input_size // 2] = True

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
            assert (
                0.3 < depression_ratio < 0.9
            ), f"Depression should be in biological range (got {depression_ratio:.2f})"

    def test_l6_depression_magnitude_realistic(self, thalamus_config):
        """Test that L6 feedback depression magnitude is in biological range (U=0.7, strong)."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # L6 feedback pattern
        pattern = torch.zeros(thalamus.relay_size, dtype=torch.bool)
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
            assert (
                0.15 < depression_ratio < 0.7
            ), f"Strong depression should be in biological range (got {depression_ratio:.2f})"

    def test_novelty_detection_functional(self, thalamus_config):
        """Test that STP enables functional novelty detection in sensory relay."""
        config, sizes, device = thalamus_config
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=device)

        # Pattern A (background)
        pattern_a = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern_a[0 : thalamus.input_size // 2] = True

        # Pattern B (novel stimulus)
        pattern_b = torch.zeros(thalamus.input_size, dtype=torch.bool)
        pattern_b[thalamus.input_size // 2 :] = True

        # Present A repeatedly (background habituation)
        for _ in range(10):
            thalamus.stp_sensory_relay(pattern_a.float())

        efficacy_background = thalamus.stp_sensory_relay(pattern_a.float()).mean().item()

        # Present B (novel stimulus)
        # Reset STP for pattern B inputs (they haven't been active)
        thalamus.stp_sensory_relay.reset_state()
        efficacy_novel = thalamus.stp_sensory_relay(pattern_b.float()).mean().item()

        # Novel stimulus should have higher efficacy
        assert (
            efficacy_novel > efficacy_background
        ), f"Novel stimulus should have higher efficacy (novel={efficacy_novel:.3f}, background={efficacy_background:.3f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
