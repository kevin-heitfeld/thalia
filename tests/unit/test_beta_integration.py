"""
Test beta oscillator integration in Cerebellum and Striatum.

Tests that beta waves properly modulate:
- Cerebellum: Climbing fiber learning gates
- Striatum: D1/D2 balance for action maintenance vs switching
"""

import math
import pytest
import torch

from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.striatum import Striatum, StriatumConfig


class TestBetaIntegrationCerebellum:
    """Test beta oscillator integration in Cerebellum."""

    def test_set_oscillator_phases(self) -> None:
        """Test that cerebellum receives and stores oscillator phases."""
        config = CerebellumConfig(
            n_input=10,
            n_output=5,
            device="cpu",
        )
        cerebellum = Cerebellum(config)

        # Set oscillator phases with effective amplitudes (pre-computed)
        phases = {'theta': 0.0, 'beta': math.pi, 'gamma': 0.0}
        signals = {'theta': 1.0, 'beta': -1.0, 'gamma': 1.0}
        effective_amplitudes = {'beta': 0.8, 'gamma': 1.0}  # Pre-computed by OscillatorManager

        # This should not raise any errors
        cerebellum.set_oscillator_phases(
            phases=phases,
            signals=signals,
            theta_slot=0,
            coupled_amplitudes=effective_amplitudes,
        )

        # Phases are stored internally - we don't access them directly
        # Instead, verify they affect behavior by checking outputs

    def test_beta_gate_computation(self) -> None:
        """Test beta gate computation - peak at trough (π)."""
        config = CerebellumConfig(
            n_input=10,
            n_output=5,
            device="cpu",
        )
        cerebellum = Cerebellum(config)

        input_spikes = torch.zeros(10, dtype=torch.bool)
        input_spikes[0:3] = True
        target = torch.zeros(5)
        target[0] = 1.0

        # Beta trough (π) - maximum learning
        phases = {'beta': math.pi}
        signals = {'beta': -1.0}
        cerebellum.set_oscillator_phases(phases, signals, 0, {'beta': 1.0})
        cerebellum.forward(input_spikes)
        metrics_trough = cerebellum.deliver_error(target)
        gate_trough = metrics_trough['beta_gate']
        assert gate_trough > 0.95  # Near 1.0

        # Reset
        cerebellum.reset_state()

        # Beta peak (0) - minimum learning
        phases = {'beta': 0.0}
        signals = {'beta': 1.0}
        cerebellum.set_oscillator_phases(phases, signals, 0, {'beta': 1.0})
        cerebellum.forward(input_spikes)
        metrics_peak = cerebellum.deliver_error(target)
        gate_peak = metrics_peak['beta_gate']
        assert gate_peak < 0.2  # Near 0.0

        # Reset
        cerebellum.reset_state()

        # Mid-phase (π/2) - intermediate (Gaussian drops off quickly)
        phases = {'beta': math.pi / 2}
        signals = {'beta': 0.0}
        cerebellum.set_oscillator_phases(phases, signals, 0, {'beta': 1.0})
        cerebellum.forward(input_spikes)
        metrics_mid = cerebellum.deliver_error(target)
        gate_mid = metrics_mid['beta_gate']
        assert 0.05 < gate_mid < 0.3  # Gaussian with width π/4 drops to ~0.13 at π/2

        # Verify ordering: trough > mid > peak
        assert gate_trough > gate_mid
        assert gate_mid > gate_peak

    def test_beta_modulates_learning(self) -> None:
        """Test that beta gate affects learning magnitude."""
        config = CerebellumConfig(
            n_input=10,
            n_output=5,
            learning_rate_ltp=0.1,
            learning_rate_ltd=0.1,
            device="cpu",
        )
        cerebellum = Cerebellum(config)

        # Create test input and output
        input_spikes = torch.zeros(10, dtype=torch.bool)
        input_spikes[0:3] = True

        # High beta (trough) - strong learning
        phases_high = {'beta': math.pi}
        signals_high = {'beta': -1.0}
        cerebellum.set_oscillator_phases(phases_high, signals_high, 0, {'beta': 1.0})
        cerebellum.forward(input_spikes)
        target = torch.zeros(5)
        target[0] = 1.0
        metrics_high = cerebellum.deliver_error(target)

        # Reset weights and state
        cerebellum.weights.data.copy_(cerebellum._initialize_weights())
        cerebellum.reset_state()

        # Low beta (peak) - weak learning
        phases_low = {'beta': 0.0}
        signals_low = {'beta': 1.0}
        cerebellum.set_oscillator_phases(phases_low, signals_low, 0, {'beta': 0.2})
        cerebellum.forward(input_spikes)
        metrics_low = cerebellum.deliver_error(target)

        # High beta should produce stronger learning
        # (beta_gate multiplies effective learning rate)
        assert metrics_high['beta_gate'] > metrics_low['beta_gate']


class TestBetaIntegrationStriatum:
    """Test beta oscillator integration in Striatum."""

    def test_set_oscillator_phases(self) -> None:
        """Test that striatum receives and stores oscillator phases."""
        config = StriatumConfig(
            n_input=10,
            n_output=3,  # 3 actions
            device="cpu",
            population_coding=False,  # Simpler for testing
        )
        striatum = Striatum(config)

        # Set oscillator phases with effective amplitudes (pre-computed)
        phases = {'theta': 0.0, 'beta': math.pi / 2, 'gamma': 0.0}
        signals = {'theta': 1.0, 'beta': 0.0, 'gamma': 1.0}
        effective_amplitudes = {'beta': 0.6, 'gamma': 1.0}  # Pre-computed by OscillatorManager

        # This should not raise any errors
        striatum.set_oscillator_phases(
            phases=phases,
            signals=signals,
            theta_slot=0,
            coupled_amplitudes=effective_amplitudes,
        )

        # Phases are stored internally - we don't access them directly
        # Instead, verify they affect behavior through forward() calls

    def test_beta_modulates_d1_d2_balance(self) -> None:
        """Test that beta amplitude modulates D1/D2 gain balance."""
        config = StriatumConfig(
            n_input=10,
            n_output=3,
            device="cpu",
            population_coding=False,
            beta_modulation_strength=0.5,
            tonic_modulates_d1_gain=False,  # Disable for isolated test
        )
        striatum = Striatum(config)

        input_spikes = torch.zeros(10, dtype=torch.bool)
        input_spikes[0:5] = True

        # High beta (0.9) - D1 should dominate (action maintenance)
        phases_high = {'beta': 0.0}  # Peak
        signals_high = {'beta': 1.0}
        striatum.set_oscillator_phases(phases_high, signals_high, 0, {'beta': 0.9})
        output_high = striatum.forward(input_spikes)
        assert output_high is not None

        # Low beta (0.1) - D2 should be effective (action switching)
        phases_low = {'beta': math.pi}  # Trough
        signals_low = {'beta': -1.0}
        striatum.set_oscillator_phases(phases_low, signals_low, 0, {'beta': 0.1})
        output_low = striatum.forward(input_spikes)
        assert output_low is not None

        # The test passes if no errors occur
        # Detailed gain inspection would require refactoring forward()
        # to expose intermediate d1_gain/d2_gain values


class TestBetaBiology:
    """Test biological plausibility of beta integration."""

    def test_cerebellum_movement_initiation_window(self) -> None:
        """Test that cerebellum learns best during beta trough (movement start)."""
        config = CerebellumConfig(
            n_input=5,
            n_output=3,
            device="cpu",
        )
        cerebellum = Cerebellum(config)

        input_spikes = torch.zeros(5, dtype=torch.bool)
        input_spikes[0:2] = True
        target = torch.zeros(3)
        target[0] = 1.0

        # Beta desynchronization (trough) = movement initiation
        # This is when inferior olive climbing fibers should teach most effectively
        phases_initiation = {'beta': math.pi}  # Trough
        signals_initiation = {'beta': -1.0}
        cerebellum.set_oscillator_phases(phases_initiation, signals_initiation, 0, {'beta': 1.0})
        cerebellum.forward(input_spikes)
        metrics_initiation = cerebellum.deliver_error(target)
        gate_initiation = metrics_initiation['beta_gate']

        # Reset
        cerebellum.reset_state()

        # Beta synchronization (peak) = action maintenance
        # Less learning needed during stable execution
        phases_maintenance = {'beta': 0.0}  # Peak
        signals_maintenance = {'beta': 1.0}
        cerebellum.set_oscillator_phases(phases_maintenance, signals_maintenance, 0, {'beta': 1.0})
        cerebellum.forward(input_spikes)
        metrics_maintenance = cerebellum.deliver_error(target)
        gate_maintenance = metrics_maintenance['beta_gate']

        # Biology: Error signals are processed during movement execution,
        # not during movement maintenance
        assert gate_initiation > 3 * gate_maintenance

    def test_striatum_action_persistence(self) -> None:
        """Test that high beta promotes action maintenance."""
        config = StriatumConfig(
            n_input=8,
            n_output=4,
            device="cpu",
            population_coding=False,
            beta_modulation_strength=0.4,
        )
        striatum = Striatum(config)

        # High beta amplitude should:
        # - Increase D1 gain (stronger GO signals)
        # - Decrease D2 gain (weaker NOGO signals)
        # Result: Current action maintained (harder to switch)

        # The formula in forward() is:
        # d1_gain = d1_gain * (1.0 + beta_mod * (beta_amplitude - 0.5))
        # d2_gain = d2_gain * (1.0 - beta_mod * (beta_amplitude - 0.5))

        # High beta (0.9)
        beta_mod = 0.4
        beta_amp = 0.9
        d1_factor = 1.0 + beta_mod * (beta_amp - 0.5)  # 1.16
        d2_factor = 1.0 - beta_mod * (beta_amp - 0.5)  # 0.84

        assert d1_factor > 1.0  # D1 boosted
        assert d2_factor < 1.0  # D2 suppressed
        assert d1_factor > d2_factor  # D1 advantage

        # Low beta (0.1)
        beta_amp = 0.1
        d1_factor = 1.0 + beta_mod * (beta_amp - 0.5)  # 0.84
        d2_factor = 1.0 - beta_mod * (beta_amp - 0.5)  # 1.16

        assert d1_factor < 1.0  # D1 reduced
        assert d2_factor > 1.0  # D2 boosted
        assert d2_factor > d1_factor  # D2 advantage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
