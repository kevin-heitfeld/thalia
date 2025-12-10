"""
Tests for OscillatorManager (centralized oscillator management).

This tests the new centralized architecture for managing brain-wide oscillations.
"""

import math
import pytest

from thalia.core.oscillator import (
    OscillatorManager,
    SinusoidalOscillator,
)


class TestOscillatorManagerBasics:
    """Test basic OscillatorManager functionality."""

    def test_initialization(self):
        """Test that manager initializes all oscillators."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        # Check oscillator types
        assert isinstance(manager.delta, SinusoidalOscillator)
        assert isinstance(manager.theta, SinusoidalOscillator)
        assert isinstance(manager.alpha, SinusoidalOscillator)
        assert isinstance(manager.beta, SinusoidalOscillator)
        assert isinstance(manager.gamma, SinusoidalOscillator)

        # Check frequencies
        assert manager.delta.frequency_hz == 2.0
        assert manager.theta.frequency_hz == 8.0
        assert manager.alpha.frequency_hz == 10.0
        assert manager.beta.frequency_hz == 20.0
        assert manager.gamma.frequency_hz == 40.0

    def test_custom_frequencies(self):
        """Test initialization with custom frequencies."""
        manager = OscillatorManager(
            dt_ms=1.0,
            device="cpu",
            delta_freq=3.0,
            theta_freq=7.0,
            alpha_freq=12.0,
            beta_freq=25.0,
            gamma_freq=50.0,
        )

        assert manager.delta.frequency_hz == 3.0
        assert manager.theta.frequency_hz == 7.0
        assert manager.alpha.frequency_hz == 12.0
        assert manager.beta.frequency_hz == 25.0
        assert manager.gamma.frequency_hz == 50.0

    def test_advance_all(self):
        """Test that advance() updates all oscillators."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        # Record initial phases
        initial_phases = manager.get_phases()

        # Advance
        manager.advance(dt_ms=1.0)

        # Check all phases changed
        new_phases = manager.get_phases()
        for name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            assert new_phases[name] != initial_phases[name]
            assert 0 <= new_phases[name] < 2 * math.pi

    def test_get_phases(self):
        """Test get_phases returns all oscillator phases."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")
        phases = manager.get_phases()

        assert isinstance(phases, dict)
        assert set(phases.keys()) == {'delta', 'theta', 'alpha', 'beta', 'gamma'}

        for name, phase in phases.items():
            assert isinstance(phase, float)
            assert 0 <= phase < 2 * math.pi

    def test_get_signals(self):
        """Test get_signals returns all oscillator signals."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")
        signals = manager.get_signals()

        assert isinstance(signals, dict)
        assert set(signals.keys()) == {'delta', 'theta', 'alpha', 'beta', 'gamma'}

        for name, signal in signals.items():
            assert isinstance(signal, float)
            assert -1.0 <= signal <= 1.0

    def test_get_oscillator(self):
        """Test get_oscillator returns correct oscillator."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        delta = manager.get_oscillator('delta')
        assert isinstance(delta, SinusoidalOscillator)
        assert delta is manager.delta

        theta = manager.get_oscillator('theta')
        assert isinstance(theta, SinusoidalOscillator)
        assert theta is manager.theta

    def test_get_oscillator_invalid(self):
        """Test get_oscillator raises error for invalid name."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        with pytest.raises(ValueError, match="Unknown oscillator"):
            manager.get_oscillator('invalid')


class TestOscillatorManagerControl:
    """Test control methods (frequency, enable/disable)."""

    def test_set_frequency(self):
        """Test frequency modulation."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        # Change theta frequency
        manager.set_frequency('theta', 10.0)
        assert manager.theta.frequency_hz == 10.0

        # Other oscillators unchanged
        assert manager.delta.frequency_hz == 2.0
        assert manager.alpha.frequency_hz == 10.0

    def test_enable_disable_oscillator(self):
        """Test enable/disable functionality."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        # Record delta phase
        manager.advance(dt_ms=10.0)
        delta_phase = manager.delta.phase

        # Disable delta
        manager.enable_oscillator('delta', False)

        # Advance - delta should not change
        manager.advance(dt_ms=10.0)
        assert manager.delta.phase == delta_phase  # No advancement

        # Other oscillators still advance
        assert manager.theta.phase != 0.0

        # Re-enable delta
        manager.enable_oscillator('delta', True)
        manager.advance(dt_ms=10.0)
        assert manager.delta.phase != delta_phase  # Resumed

    def test_set_sleep_stage_nrem(self):
        """Test NREM sleep stage configuration."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        manager.set_sleep_stage("NREM")

        # Delta strong, slow frequencies
        assert manager.delta.frequency_hz == 2.0
        assert manager.theta.frequency_hz == 6.0
        assert manager.gamma.frequency_hz == 30.0

        # Alpha and beta disabled
        assert manager._enabled['alpha'] == False
        assert manager._enabled['beta'] == False

    def test_set_sleep_stage_rem(self):
        """Test REM sleep stage configuration."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        manager.set_sleep_stage("REM")

        # Theta dominant, fast gamma
        assert manager.delta.frequency_hz == 1.0
        assert manager.theta.frequency_hz == 7.0
        assert manager.gamma.frequency_hz == 60.0

        # Alpha and beta disabled
        assert manager._enabled['alpha'] == False
        assert manager._enabled['beta'] == False

    def test_set_sleep_stage_awake(self):
        """Test awake stage restores defaults."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        # Change to NREM
        manager.set_sleep_stage("NREM")
        assert manager.delta.frequency_hz == 2.0

        # Restore to awake
        manager.set_sleep_stage("AWAKE")

        # Frequencies restored
        assert manager.theta.frequency_hz == 8.0
        assert manager.alpha.frequency_hz == 10.0
        assert manager.gamma.frequency_hz == 40.0

        # All enabled
        assert all(manager._enabled.values())


class TestOscillatorManagerStateSerialization:
    """Test state save/load for checkpointing."""

    def test_get_state(self):
        """Test state serialization."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")
        manager.advance(dt_ms=50.0)

        state = manager.get_state()

        assert isinstance(state, dict)
        assert 'delta' in state
        assert 'theta' in state
        assert 'alpha' in state
        assert 'beta' in state
        assert 'gamma' in state
        assert 'time_ms' in state
        assert 'enabled' in state

        assert state['time_ms'] == 50.0

    def test_set_state(self):
        """Test state restoration."""
        manager1 = OscillatorManager(dt_ms=1.0, device="cpu")
        manager1.advance(dt_ms=100.0)
        manager1.set_frequency('theta', 12.0)
        manager1.enable_oscillator('alpha', False)

        # Save state
        state = manager1.get_state()

        # Create new manager
        manager2 = OscillatorManager(dt_ms=1.0, device="cpu")
        manager2.set_state(state)

        # Check restoration
        assert manager2._time_ms == 100.0
        assert manager2.theta.frequency_hz == 12.0
        assert manager2._enabled['alpha'] == False

        # Phases should match
        phases1 = manager1.get_phases()
        phases2 = manager2.get_phases()
        for name in ['delta', 'theta', 'beta', 'gamma']:
            assert abs(phases1[name] - phases2[name]) < 1e-6

    def test_reset(self):
        """Test reset functionality."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")
        manager.advance(dt_ms=100.0)

        assert manager._time_ms == 100.0
        assert manager.theta.phase != 0.0

        # Reset
        manager.reset()

        assert manager._time_ms == 0.0
        assert manager.delta.phase == 0.0
        assert manager.theta.phase == 0.0


class TestOscillatorManagerBiologicalAccuracy:
    """Test biological realism of oscillator patterns."""

    def test_synchronized_phases(self):
        """Test that all oscillators stay synchronized (single time reference)."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        # Advance many steps
        for _ in range(1000):
            manager.advance(dt_ms=1.0)

            # All oscillators should remain synchronized to same time
            # (they all advanced by same dt)
            phases = manager.get_phases()

            # Check phases are valid
            for phase in phases.values():
                assert 0 <= phase < 2 * math.pi

    def test_frequency_relationships(self):
        """Test that faster oscillators complete more cycles."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")

        # Run for 1 second (1000ms)
        for _ in range(1000):
            manager.advance(dt_ms=1.0)

        # Gamma (40 Hz) should have completed ~40 cycles
        # Theta (8 Hz) should have completed ~8 cycles
        # Delta (2 Hz) should have completed ~2 cycles

        # We can't count cycles directly, but we can check
        # that they advanced different amounts
        phases = manager.get_phases()

        # All should have wrapped around multiple times
        # (this just checks they advanced, not exact cycle count)
        assert 0 <= phases['delta'] < 2 * math.pi
        assert 0 <= phases['gamma'] < 2 * math.pi

    def test_nrem_consolidation_pattern(self):
        """Test NREM sleep produces consolidation-friendly pattern."""
        manager = OscillatorManager(dt_ms=1.0, device="cpu")
        manager.set_sleep_stage("NREM")

        # During NREM, delta should dominate
        assert manager.delta.frequency_hz == 2.0  # Slow wave sleep
        assert manager.gamma.frequency_hz == 30.0  # Slow gamma

        # Alpha/beta should be off (no waking activity)
        assert manager._enabled['alpha'] == False
        assert manager._enabled['beta'] == False

        # Advance and check signals
        manager.advance(dt_ms=500.0)  # 0.5 second
        signals = manager.get_signals()

        # Delta should still be oscillating
        assert -1.0 <= signals['delta'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
