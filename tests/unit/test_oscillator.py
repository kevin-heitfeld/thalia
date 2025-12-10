"""
Tests for Brain Oscillator Base Classes

Tests the core oscillator framework for phase tracking, frequency modulation,
synchronization, and state management.
"""

import math

from thalia.core.oscillator import (
    SinusoidalOscillator,
    OscillatorConfig,
    OSCILLATOR_DEFAULTS,
)


# ============================================================================
# Basic Oscillator Tests
# ============================================================================

class TestSinusoidalOscillator:
    """Tests for sinusoidal oscillator (all frequencies)."""

    def test_creation(self):
        """Test theta oscillator creation."""
        theta = SinusoidalOscillator(frequency_hz=8.0)
        assert theta.frequency_hz == 8.0
        assert theta.phase == 0.0
        assert theta.time_ms == 0.0

    def test_phase_advancement(self):
        """Test phase advances correctly over time."""
        theta = SinusoidalOscillator(frequency_hz=8.0, dt_ms=1.0)

        # Initial phase
        assert theta.phase == 0.0

        # Advance one full period (125ms for 8Hz)
        period_ms = 1000.0 / 8.0  # 125ms
        for _ in range(int(period_ms)):
            theta.advance(dt_ms=1.0)

        # Should be back near 0 (within floating point error)
        assert abs(theta.phase) < 0.01 or abs(theta.phase - 2*math.pi) < 0.01
        assert theta.time_ms == period_ms

    def test_oscillation_period(self):
        """Test oscillation period is correct."""
        theta = SinusoidalOscillator(frequency_hz=8.0)
        assert abs(theta.oscillation_period_ms - 125.0) < 0.1

        theta = SinusoidalOscillator(frequency_hz=10.0)
        assert abs(theta.oscillation_period_ms - 100.0) < 0.1

    def test_signal_generation(self):
        """Test sinusoidal signal generation."""
        theta = SinusoidalOscillator(frequency_hz=8.0, amplitude=1.0)

        # At phase 0
        assert abs(theta.signal - 0.0) < 0.01

        # At phase π/2 (quarter cycle)
        theta.sync_to_phase(math.pi / 2)
        assert abs(theta.signal - 1.0) < 0.01

        # At phase π (half cycle)
        theta.sync_to_phase(math.pi)
        assert abs(theta.signal - 0.0) < 0.01

        # At phase 3π/2 (three-quarter cycle)
        theta.sync_to_phase(3 * math.pi / 2)
        assert abs(theta.signal + 1.0) < 0.01

    def test_frequency_modulation(self):
        """Test changing oscillation frequency."""
        theta = SinusoidalOscillator(frequency_hz=8.0)
        initial_period = theta.oscillation_period_ms

        # Change to 4 Hz (slower, longer period)
        theta.set_frequency(4.0)
        assert theta.frequency_hz == 4.0
        assert theta.oscillation_period_ms == 250.0
        assert theta.oscillation_period_ms > initial_period

        # Change to 16 Hz (faster, shorter period)
        theta.set_frequency(16.0)
        assert theta.frequency_hz == 16.0
        assert abs(theta.oscillation_period_ms - 62.5) < 0.1

    def test_phase_synchronization(self):
        """Test synchronizing to specific phase."""
        theta = SinusoidalOscillator(frequency_hz=8.0)

        # Sync to quarter cycle
        theta.sync_to_phase(math.pi / 2)
        assert abs(theta.phase - math.pi/2) < 0.01

        # Sync to half cycle
        theta.sync_to_phase(math.pi)
        assert abs(theta.phase - math.pi) < 0.01

        # Sync wraps to [0, 2π)
        theta.sync_to_phase(3 * math.pi)
        assert theta.phase < 2 * math.pi

    def test_state_management(self):
        """Test getting and setting oscillator state."""
        theta = SinusoidalOscillator(frequency_hz=8.0)

        # Advance to some state
        theta.advance(dt_ms=50.0)
        theta.set_frequency(10.0)

        # Get state
        state = theta.get_state()
        assert "phase" in state
        assert "frequency_hz" in state
        assert "time_ms" in state

        # Create new oscillator and restore state
        theta2 = SinusoidalOscillator(frequency_hz=5.0)
        theta2.set_state(state)

        assert abs(theta2.phase - theta.phase) < 0.001
        assert theta2.frequency_hz == theta.frequency_hz
        assert theta2.time_ms == theta.time_ms

    def test_reset_state(self):
        """Test resetting oscillator state."""
        theta = SinusoidalOscillator(frequency_hz=8.0, initial_phase=0.0)

        # Advance and modify
        theta.advance(dt_ms=100.0)
        theta.set_frequency(12.0)

        # Reset
        theta.reset_state()

        assert theta.phase == 0.0
        assert theta.frequency_hz == 8.0
        assert theta.time_ms == 0.0


    def test_gamma_frequency(self):
        """Test gamma oscillator creation."""
        gamma = SinusoidalOscillator(frequency_hz=40.0)
        assert gamma.frequency_hz == 40.0
        assert gamma.phase == 0.0

    def test_gamma_period(self):
        """Test gamma oscillation period."""
        gamma = SinusoidalOscillator(frequency_hz=40.0)
        assert abs(gamma.oscillation_period_ms - 25.0) < 0.1

    def test_gamma_signal(self):
        """Test gamma signal generation."""
        gamma = SinusoidalOscillator(frequency_hz=40.0, amplitude=1.0)

        # Test at various phases
        gamma.sync_to_phase(0.0)
        assert abs(gamma.signal) < 0.01

        gamma.sync_to_phase(math.pi / 2)
        assert abs(gamma.signal - 1.0) < 0.01

    def test_alpha_frequency(self):
        """Test alpha oscillator creation."""
        alpha = SinusoidalOscillator(frequency_hz=10.0)
        assert alpha.frequency_hz == 10.0
        assert abs(alpha.oscillation_period_ms - 100.0) < 0.1

    def test_beta_frequency(self):
        """Test beta oscillator creation."""
        beta = SinusoidalOscillator(frequency_hz=20.0)
        assert beta.frequency_hz == 20.0
        assert abs(beta.oscillation_period_ms - 50.0) < 0.1

    def test_defaults_dict(self):
        """Test using OSCILLATOR_DEFAULTS dict."""
        # Create oscillators using defaults
        delta = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['delta'])
        theta = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['theta'])
        alpha = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['alpha'])
        beta = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['beta'])
        gamma = SinusoidalOscillator(frequency_hz=OSCILLATOR_DEFAULTS['gamma'])

        assert delta.frequency_hz == 2.0
        assert theta.frequency_hz == 8.0
        assert alpha.frequency_hz == 10.0
        assert beta.frequency_hz == 20.0
        assert gamma.frequency_hz == 40.0


class TestOscillatorInteractions:
    """Tests for interactions between oscillators."""

    def test_frequency_ratio(self):
        """Test frequency ratios between oscillators."""
        theta = SinusoidalOscillator(frequency_hz=8.0)
        gamma = SinusoidalOscillator(frequency_hz=40.0)

        # Gamma should be 5x faster than theta
        ratio = gamma.frequency_hz / theta.frequency_hz
        assert abs(ratio - 5.0) < 0.01

        # Periods should have inverse ratio
        period_ratio = theta.oscillation_period_ms / gamma.oscillation_period_ms
        assert abs(period_ratio - 5.0) < 0.01

    def test_phase_coupling(self):
        """Test synchronizing multiple oscillators."""
        theta = SinusoidalOscillator(frequency_hz=8.0)
        gamma = SinusoidalOscillator(frequency_hz=40.0)

        # Sync both to same phase
        target_phase = math.pi / 4
        theta.sync_to_phase(target_phase)
        gamma.sync_to_phase(target_phase)

        assert abs(theta.phase - gamma.phase) < 0.01

        # Advance both by same time
        for _ in range(10):
            theta.advance(dt_ms=1.0)
            gamma.advance(dt_ms=1.0)

        # Phases will differ due to frequency difference
        assert theta.phase != gamma.phase

    def test_nested_oscillations(self):
        """Test gamma nested within theta (5 cycles per theta)."""
        theta = SinusoidalOscillator(frequency_hz=8.0)
        gamma = SinusoidalOscillator(frequency_hz=40.0)

        # Count gamma cycles in one theta period
        theta_period_ms = theta.oscillation_period_ms
        gamma_cycles = 0
        prev_phase = 0.0

        for _ in range(int(theta_period_ms)):
            gamma.advance(dt_ms=1.0)
            # Count when phase wraps
            if gamma.phase < prev_phase:
                gamma_cycles += 1
            prev_phase = gamma.phase

        # Should be approximately 5 gamma cycles per theta
        assert abs(gamma_cycles - 5) <= 1


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================

class TestOscillatorEdgeCases:
    """Tests for edge cases and robustness."""

    def test_very_high_frequency(self):
        """Test oscillator with very high frequency."""
        high_gamma = SinusoidalOscillator(frequency_hz=100.0)
        assert high_gamma.oscillation_period_ms == 10.0

        # Should handle rapid advancement
        for _ in range(1000):
            high_gamma.advance(dt_ms=0.1)

        assert 0 <= high_gamma.phase < 2 * math.pi

    def test_very_low_frequency(self):
        """Test oscillator with very low frequency."""
        slow = SinusoidalOscillator(frequency_hz=1.0)
        assert slow.oscillation_period_ms == 1000.0

        # Should handle slow advancement
        slow.advance(dt_ms=100.0)
        assert slow.phase < math.pi

    def test_large_timestep(self):
        """Test with large timesteps (multiple cycles)."""
        theta = SinusoidalOscillator(frequency_hz=8.0)

        # Advance by 10 seconds (80 cycles)
        theta.advance(dt_ms=10000.0)

        # Phase should still be in [0, 2π)
        assert 0 <= theta.phase < 2 * math.pi
        assert theta.time_ms == 10000.0

    def test_amplitude_scaling(self):
        """Test signal amplitude scaling."""
        theta = SinusoidalOscillator(frequency_hz=8.0, amplitude=2.0)

        # Signal should be scaled
        theta.sync_to_phase(math.pi / 2)
        assert abs(theta.signal - 2.0) < 0.01

        theta.sync_to_phase(3 * math.pi / 2)
        assert abs(theta.signal + 2.0) < 0.01

    def test_phase_wrapping(self):
        """Test phase wrapping at boundaries."""
        theta = SinusoidalOscillator(frequency_hz=8.0)

        # Sync to phase > 2π
        theta.sync_to_phase(3 * math.pi)
        assert 0 <= theta.phase < 2 * math.pi

        # Sync to negative phase
        theta.sync_to_phase(-math.pi / 2)
        assert 0 <= theta.phase < 2 * math.pi

        # Advance should always keep phase wrapped
        for _ in range(1000):
            theta.advance(dt_ms=1.0)
            assert 0 <= theta.phase < 2 * math.pi

    def test_zero_timestep(self):
        """Test advancement with zero timestep.

        Note: Passing dt_ms=0.0 will fallback to config.dt_ms
        because of the 'or' operator. This is intentional behavior
        to avoid explicit None checks.
        """
        theta = SinusoidalOscillator(frequency_hz=8.0, dt_ms=1.0)
        initial_phase = theta.phase

        # Passing 0.0 falls back to config.dt_ms (1.0)
        theta.advance(dt_ms=0.0)

        # Should advance by 1.0ms (the config default)
        assert theta.phase > initial_phase
        assert theta.time_ms == 1.0


# ============================================================================
# Configuration Tests
# ============================================================================

class TestOscillatorConfig:
    """Tests for oscillator configuration."""

    def test_config_creation(self):
        """Test creating oscillator config."""
        config = OscillatorConfig(
            frequency_hz=8.0,
            dt_ms=1.0,
            initial_phase=math.pi / 4,
            amplitude=1.5,
        )

        assert config.frequency_hz == 8.0
        assert config.dt_ms == 1.0
        assert config.initial_phase == math.pi / 4
        assert config.amplitude == 1.5

    def test_initial_phase(self):
        """Test oscillator starts at configured initial phase."""
        initial = math.pi / 3
        theta = SinusoidalOscillator(frequency_hz=8.0, initial_phase=initial)

        assert abs(theta.phase - initial) < 0.01

    def test_default_timestep(self):
        """Test oscillator uses configured default timestep."""
        theta = SinusoidalOscillator(frequency_hz=8.0, dt_ms=2.0)

        # Advance without specifying dt
        theta.advance()

        # Should have advanced by 2ms
        expected_phase = 2.0 * 2.0 * math.pi * 8.0 / 1000.0
        assert abs(theta.phase - expected_phase) < 0.01
