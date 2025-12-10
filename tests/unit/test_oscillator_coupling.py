"""
Tests for cross-frequency coupling in OscillatorManager.

Verifies phase-amplitude coupling between oscillators (e.g., theta-gamma).
"""

import math
import pytest
from thalia.core.oscillator import OscillatorManager, OscillatorCoupling


class TestOscillatorCoupling:
    """Test OscillatorCoupling configuration."""

    def test_coupling_creation(self):
        """Test creating a coupling configuration."""
        coupling = OscillatorCoupling(
            oscillator='gamma',
            coupling_strength=0.8,
            min_amplitude=0.2,
            modulation_type='cosine'
        )

        assert coupling.oscillator == 'gamma'
        assert coupling.coupling_strength == 0.8
        assert coupling.min_amplitude == 0.2
        assert coupling.modulation_type == 'cosine'

    def test_coupling_validation_strength(self):
        """Test coupling_strength validation."""
        with pytest.raises(ValueError, match="coupling_strength must be in"):
            OscillatorCoupling(
                oscillator='gamma',
                coupling_strength=1.5  # Invalid: > 1.0
            )

    def test_coupling_validation_min_amplitude(self):
        """Test min_amplitude validation."""
        with pytest.raises(ValueError, match="min_amplitude must be in"):
            OscillatorCoupling(
                oscillator='gamma',
                min_amplitude=-0.1  # Invalid: < 0.0
            )

    def test_coupling_validation_modulation_type(self):
        """Test modulation_type validation."""
        with pytest.raises(ValueError, match="modulation_type must be"):
            OscillatorCoupling(
                oscillator='gamma',
                modulation_type='invalid'  # Invalid type
            )


class TestCrossFrequencyCoupling:
    """Test phase-amplitude coupling in OscillatorManager."""

    def test_default_couplings_all_five(self):
        """Test that default manager includes 4 biologically-motivated couplings."""
        manager = OscillatorManager()

        # Should have 4 default couplings (theta, alpha, beta, gamma - not delta)
        assert len(manager.couplings) == 4

        # Extract coupled oscillators (the fast ones being modulated)
        coupled_oscillators = [c.oscillator for c in manager.couplings]

        # Verify all expected coupled oscillators are present
        assert 'theta' in coupled_oscillators  # Modulated by delta (sleep consolidation)
        assert 'alpha' in coupled_oscillators  # Modulated by delta, theta (attention)
        assert 'beta' in coupled_oscillators   # Modulated by delta, theta, alpha (motor/WM)
        assert 'gamma' in coupled_oscillators  # Modulated by all (binding, timing, attention)

    def test_default_theta_gamma_coupling(self):
        """Test that theta-gamma coupling has correct parameters."""
        manager = OscillatorManager()

        # Find gamma coupling (the fast oscillator being modulated)
        gamma_coupling = None
        for c in manager.couplings:
            if c.oscillator == 'gamma':
                gamma_coupling = c
                break

        assert gamma_coupling is not None
        # Gamma coupling exists with reasonable parameters
        assert gamma_coupling.coupling_strength > 0.0
        assert gamma_coupling.min_amplitude >= 0.0
        assert gamma_coupling.modulation_type in ('cosine', 'sine')

    def test_custom_couplings(self):
        """Test creating manager with custom couplings."""
        couplings = [
            OscillatorCoupling(oscillator='gamma', coupling_strength=0.9, min_amplitude=0.1),
            OscillatorCoupling(oscillator='theta', coupling_strength=0.7, min_amplitude=0.3),
        ]
        manager = OscillatorManager(couplings=couplings)

        assert len(manager.couplings) == 2
        assert manager.couplings[0].oscillator == 'gamma'
        assert manager.couplings[1].oscillator == 'theta'

    def test_no_couplings(self):
        """Test manager with no couplings."""
        manager = OscillatorManager(couplings=[])

        assert len(manager.couplings) == 0
        # All amplitudes should be 1.0 without coupling
        assert manager.get_coupled_amplitude('gamma', 'theta') == 1.0


class TestGetCoupledAmplitude:
    """Test get_coupled_amplitude() method."""

    def test_theta_gamma_amplitude_at_trough(self):
        """Test gamma amplitude at theta trough (max gamma)."""
        manager = OscillatorManager()

        # Set theta to trough (phase = 0 or 2π)
        manager.theta.sync_to_phase(0.0)

        # Gamma should be at maximum amplitude
        amplitude = manager.get_coupled_amplitude('gamma', 'theta')

        # With cosine coupling, max = 1.0 at phase 0
        # amplitude = 0.2 + (1.0 - 0.2) * 1.0 * 0.8 = 0.2 + 0.64 = 0.84
        assert amplitude == pytest.approx(0.84, abs=0.01)

    def test_theta_gamma_amplitude_at_peak(self):
        """Test gamma amplitude at theta peak (min gamma)."""
        manager = OscillatorManager()

        # Set theta to peak (phase = π)
        manager.theta.sync_to_phase(math.pi)

        # Gamma should be at minimum amplitude
        amplitude = manager.get_coupled_amplitude('gamma', 'theta')

        # With cosine coupling, min = 0.0 at phase π
        # amplitude = 0.2 + (1.0 - 0.2) * 0.0 * 0.8 = 0.2
        assert amplitude == pytest.approx(0.2, abs=0.01)

    def test_amplitude_modulation_cycle(self):
        """Test amplitude modulates smoothly through theta cycle."""
        manager = OscillatorManager(theta_freq=8.0)

        amplitudes = []
        for _ in range(125):  # One theta cycle (~125ms at 8Hz)
            amp = manager.get_coupled_amplitude('gamma', 'theta')
            amplitudes.append(amp)
            manager.advance(dt_ms=1.0)

        # Check amplitude varies between min and max
        assert min(amplitudes) == pytest.approx(0.2, abs=0.05)
        assert max(amplitudes) == pytest.approx(0.84, abs=0.05)

        # Check smooth modulation (no discontinuities)
        diffs = [abs(amplitudes[i+1] - amplitudes[i]) for i in range(len(amplitudes)-1)]
        assert max(diffs) < 0.1  # No large jumps

    def test_no_coupling_returns_one(self):
        """Test uncoupled oscillators return amplitude 1.0."""
        manager = OscillatorManager(couplings=[])

        # Without coupling, amplitude should always be 1.0
        assert manager.get_coupled_amplitude('gamma', 'theta') == 1.0
        assert manager.get_coupled_amplitude('alpha', 'beta') == 1.0

    def test_sine_modulation(self):
        """Test sine modulation type (max at π/2)."""
        coupling = OscillatorCoupling(
            oscillator='gamma',
            modulation_type='sine'
        )
        manager = OscillatorManager(couplings=[coupling])

        # At phase = π/2, sine is maximum
        manager.theta.sync_to_phase(math.pi / 2)
        amp_max = manager.get_coupled_amplitude('gamma', 'theta')

        # At phase = 3π/2, sine is minimum
        manager.theta.sync_to_phase(3 * math.pi / 2)
        amp_min = manager.get_coupled_amplitude('gamma', 'theta')

        assert amp_max > amp_min


class TestCoupledSignals:
    """Test that get_signals() applies coupling."""

    def test_gamma_signal_modulated_by_theta(self):
        """Test gamma signal varies with theta phase."""
        manager = OscillatorManager()

        # At theta trough: gamma signal should be stronger
        manager.theta.sync_to_phase(0.0)
        manager.gamma.sync_to_phase(math.pi / 2)  # Gamma at peak
        signals_trough = manager.get_signals()

        # At theta peak: gamma signal should be weaker
        manager.theta.sync_to_phase(math.pi)
        manager.gamma.sync_to_phase(math.pi / 2)  # Gamma at peak
        signals_peak = manager.get_signals()

        # Gamma signal amplitude should differ
        assert abs(signals_trough['gamma']) > abs(signals_peak['gamma'])

        # Theta signal should be similar (not modulated)
        assert signals_trough['theta'] == pytest.approx(-signals_peak['theta'], abs=0.01)

    def test_uncoupled_oscillators_unchanged(self):
        """Test oscillators without coupling are unaffected."""
        manager = OscillatorManager()

        signals = manager.get_signals()

        # Delta, theta, alpha, beta should equal their base signals
        # (only gamma is coupled to theta)
        assert signals['delta'] == manager.delta.signal
        assert signals['theta'] == manager.theta.signal
        assert signals['alpha'] == manager.alpha.signal
        assert signals['beta'] == manager.beta.signal

        # Gamma is coupled to theta by default, so it should be modulated
        # unless we're at a phase where amplitude happens to be 1.0
        coupling_amp = manager.get_coupled_amplitude('gamma', 'theta')
        expected_gamma = manager.gamma.signal * coupling_amp
        assert signals['gamma'] == pytest.approx(expected_gamma, abs=0.01)


class TestThetaSlots:
    """Test theta slot calculation for sequence encoding."""

    def test_get_theta_slot_returns_valid_range(self):
        """Test slot index is always in valid range."""
        manager = OscillatorManager()

        for _ in range(200):  # More than one cycle
            slot = manager.get_theta_slot(n_slots=7)
            assert 0 <= slot < 7
            manager.advance(dt_ms=1.0)

    def test_slot_progression_through_cycle(self):
        """Test slots progress 0 → 1 → 2 → ... → 6 → 0."""
        manager = OscillatorManager(theta_freq=8.0)

        slots = []
        for _ in range(125):  # One theta cycle
            slots.append(manager.get_theta_slot(n_slots=7))
            manager.advance(dt_ms=1.0)

        # Should see all slots
        unique_slots = set(slots)
        assert unique_slots == {0, 1, 2, 3, 4, 5, 6}

        # Slots should progress sequentially (with wrapping)
        for i in range(len(slots) - 1):
            if slots[i+1] != slots[i]:  # Slot changed
                expected_next = (slots[i] + 1) % 7
                assert slots[i+1] == expected_next

    def test_slot_division_matches_gamma_cycles(self):
        """Test that 7 slots roughly match 7 gamma cycles in theta."""
        manager = OscillatorManager(theta_freq=8.0, gamma_freq=40.0)
        # 40 Hz / 8 Hz = 5 gamma cycles per theta (approximately)

        # For exact match, use gamma_freq = 7 * theta_freq
        manager2 = OscillatorManager(theta_freq=8.0, gamma_freq=56.0)

        slots_per_gamma = []
        prev_slot = 0
        gamma_cycles = 0
        prev_phase = manager2.gamma.phase

        for _ in range(125):  # One theta cycle
            # Count gamma cycles (zero crossings)
            if manager2.gamma.phase < prev_phase:
                gamma_cycles += 1
            prev_phase = manager2.gamma.phase

            slot = manager2.get_theta_slot(n_slots=7)
            if slot != prev_slot:
                slots_per_gamma.append(gamma_cycles)
                gamma_cycles = 0
                prev_slot = slot

            manager2.advance(dt_ms=1.0)

        # With 56 Hz gamma and 8 Hz theta: 7 gamma per theta
        # Each slot should span ~1 gamma cycle
        # Allow some tolerance for discretization
        if slots_per_gamma:
            avg_gamma_per_slot = sum(slots_per_gamma) / len(slots_per_gamma)
            assert 0.5 < avg_gamma_per_slot < 2.0

    def test_custom_slot_count(self):
        """Test get_theta_slot with different slot counts."""
        manager = OscillatorManager()

        # Test 5 slots (working memory capacity ~5±2)
        for _ in range(125):
            slot = manager.get_theta_slot(n_slots=5)
            assert 0 <= slot < 5
            manager.advance(dt_ms=1.0)

        # Reset and test 9 slots (upper bound ~7+2)
        manager.reset()
        for _ in range(125):
            slot = manager.get_theta_slot(n_slots=9)
            assert 0 <= slot < 9
            manager.advance(dt_ms=1.0)

    def test_slot_at_specific_phases(self):
        """Test slot calculation at known theta phases."""
        manager = OscillatorManager()

        # Phase 0 (trough) should be slot 0
        manager.theta.sync_to_phase(0.0)
        assert manager.get_theta_slot(n_slots=7) == 0

        # Phase π should be mid-cycle (slot ~3-4)
        manager.theta.sync_to_phase(math.pi)
        slot_mid = manager.get_theta_slot(n_slots=7)
        assert 3 <= slot_mid <= 4

        # Phase just before 2π should be last slot
        manager.theta.sync_to_phase(2 * math.pi - 0.01)
        slot_end = manager.get_theta_slot(n_slots=7)
        assert slot_end == 6


class TestMultipleCouplings:
    """Test manager with multiple simultaneous couplings."""

    def test_multiple_couplings_applied(self):
        """Test multiple couplings can coexist."""
        couplings = [
            OscillatorCoupling(oscillator='gamma', coupling_strength=0.8, min_amplitude=0.2),
            OscillatorCoupling(oscillator='theta', coupling_strength=0.6, min_amplitude=0.4),
            OscillatorCoupling(oscillator='beta', coupling_strength=0.5, min_amplitude=0.5),
        ]
        manager = OscillatorManager(couplings=couplings)

        # All couplings should be active
        gamma_amp = manager.get_coupled_amplitude('gamma', 'theta')
        theta_amp = manager.get_coupled_amplitude('theta', 'delta')
        beta_amp = manager.get_coupled_amplitude('beta', 'alpha')

        # All should return modulated amplitudes (not 1.0)
        assert gamma_amp != 1.0 or manager.theta.phase in [0, math.pi]
        assert theta_amp != 1.0 or manager.delta.phase in [0, math.pi]
        assert beta_amp != 1.0 or manager.alpha.phase in [0, math.pi]

    def test_no_interference_between_couplings(self):
        """Test couplings don't interfere with each other."""
        couplings = [
            OscillatorCoupling(oscillator='gamma', coupling_strength=0.8, min_amplitude=0.2),
            OscillatorCoupling(oscillator='alpha', coupling_strength=0.6, min_amplitude=0.3),
        ]
        manager = OscillatorManager(couplings=couplings)

        # Set phases
        manager.theta.sync_to_phase(0.0)
        manager.delta.sync_to_phase(math.pi)

        # Gamma coupled to theta (at trough)
        gamma_amp = manager.get_coupled_amplitude('gamma', 'theta')
        assert gamma_amp > 0.5  # High amplitude at theta trough

        # Alpha coupled to delta (at peak)
        alpha_amp = manager.get_coupled_amplitude('alpha', 'delta')
        assert alpha_amp < 0.5  # Low amplitude at delta peak

        # Beta not coupled to anything
        beta_amp = manager.get_coupled_amplitude('beta', 'theta')
        assert beta_amp == 1.0


class TestBiologicalAccuracy:
    """Test biological accuracy of coupling patterns."""

    def test_theta_gamma_working_memory_pattern(self):
        """Test theta-gamma coupling matches working memory literature."""
        # Lisman & Jensen (2013): ~7 gamma cycles per theta
        manager = OscillatorManager(theta_freq=8.0, gamma_freq=40.0)

        theta_period_ms = 1000.0 / 8.0  # 125 ms
        gamma_period_ms = 1000.0 / 40.0  # 25 ms
        gamma_per_theta = theta_period_ms / gamma_period_ms  # 5.0

        # Should be close to 7±2 (working memory capacity)
        assert 3 < gamma_per_theta < 9

    def test_delta_theta_nrem_pattern(self):
        """Test delta-theta coupling for NREM sleep."""
        coupling = OscillatorCoupling(oscillator='theta', coupling_strength=0.7, min_amplitude=0.1)
        manager = OscillatorManager(
            delta_freq=2.0,
            theta_freq=6.0,  # Slower during sleep
            couplings=[coupling]
        )

        # Theta should be nested in delta
        manager.delta.sync_to_phase(0.0)  # Delta trough
        theta_amp_high = manager.get_coupled_amplitude('theta', 'delta')

        manager.delta.sync_to_phase(math.pi)  # Delta peak
        theta_amp_low = manager.get_coupled_amplitude('theta', 'delta')

        assert theta_amp_high > theta_amp_low

    def test_coupling_preserves_frequency(self):
        """Test coupling modulates amplitude, not frequency."""
        manager = OscillatorManager()

        initial_freq = manager.gamma.frequency_hz

        # Advance through various theta phases
        for _ in range(200):
            manager.advance(dt_ms=1.0)

        # Frequency should remain constant
        assert manager.gamma.frequency_hz == initial_freq


class TestNewCouplings:
    """Test the 4 new coupling types added in Phase 5."""

    def test_beta_gamma_motor_timing(self):
        """Test beta-gamma coupling for motor timing."""
        manager = OscillatorManager()

        # Find gamma coupling (modulated by beta and others)
        gamma_coupling = None
        for c in manager.couplings:
            if c.oscillator == 'gamma':
                gamma_coupling = c
                break

        assert gamma_coupling is not None, "Gamma coupling should exist"

        # Test coupling behavior: max gamma at beta phase=0 (cosine max)
        manager.beta.sync_to_phase(0.0)  # Cosine max
        gamma_amp_high = manager.get_coupled_amplitude('gamma', 'beta')

        manager.beta.sync_to_phase(math.pi)  # Cosine min
        gamma_amp_low = manager.get_coupled_amplitude('gamma', 'beta')

        assert gamma_amp_high > gamma_amp_low
        # Gamma coupling has per_oscillator_strength={'beta': 0.6}, min_amplitude=0.2
        # At phase=0 (cos=1): 0.2 + (1-0.2)*1.0*0.6 = 0.68
        # At phase=π (cos=-1): 0.2 + (1-0.2)*0.0*0.6 = 0.2
        assert 0.67 <= gamma_amp_high <= 0.69
        assert 0.19 <= gamma_amp_low <= 0.21

    def test_delta_theta_sleep_consolidation(self):
        """Test delta-theta coupling for sleep consolidation."""
        manager = OscillatorManager()

        # Find theta coupling (modulated by delta)
        theta_coupling = None
        for c in manager.couplings:
            if c.oscillator == 'theta':
                theta_coupling = c
                break

        assert theta_coupling is not None, "Theta coupling should exist"

        # Test coupling behavior: max theta at delta phase=0 (cosine max)
        manager.delta.sync_to_phase(0.0)  # Cosine max (delta up-state)
        theta_amp_high = manager.get_coupled_amplitude('theta', 'delta')

        manager.delta.sync_to_phase(math.pi)  # Cosine min (delta down-state)
        theta_amp_low = manager.get_coupled_amplitude('theta', 'delta')

        assert theta_amp_high > theta_amp_low
        # With coupling_strength=0.7, min_amplitude=0.1:
        # At phase=0 (cos=1): 0.1 + (1-0.1)*1.0*0.7 = 0.73
        # At phase=π (cos=-1): 0.1 + (1-0.1)*0.0*0.7 = 0.1
        assert 0.72 <= theta_amp_high <= 0.74
        assert 0.09 <= theta_amp_low <= 0.11

    def test_alpha_gamma_attention_gating(self):
        """Test alpha-gamma coupling for attention gating."""
        manager = OscillatorManager()

        # Find gamma coupling (modulated by alpha and others)
        gamma_coupling = None
        for c in manager.couplings:
            if c.oscillator == 'gamma':
                gamma_coupling = c
                break

        assert gamma_coupling is not None, "Gamma coupling should exist"

        # Test coupling behavior: max gamma at alpha trough (inverse)
        manager.alpha.sync_to_phase(math.pi / 2)  # Alpha peak (sine)
        gamma_amp_sine_peak = manager.get_coupled_amplitude('gamma', 'alpha')

        manager.alpha.sync_to_phase(3 * math.pi / 2)  # Alpha trough (sine)
        gamma_amp_sine_trough = manager.get_coupled_amplitude('gamma', 'alpha')

        # Sine modulation: max at π/2, min at 3π/2
        assert gamma_amp_sine_peak > gamma_amp_sine_trough

    def test_theta_beta_working_memory_action(self):
        """Test theta-beta coupling for working memory-action coordination."""
        manager = OscillatorManager()

        # Find beta coupling (modulated by theta and others)
        beta_coupling = None
        for c in manager.couplings:
            if c.oscillator == 'beta':
                beta_coupling = c
                break

        assert beta_coupling is not None, "Beta coupling should exist"

        # Test coupling behavior: max beta at theta phase=0 (cosine max)
        manager.theta.sync_to_phase(0.0)  # Cosine max
        beta_amp_high = manager.get_coupled_amplitude('beta', 'theta')

        manager.theta.sync_to_phase(math.pi)  # Cosine min
        beta_amp_low = manager.get_coupled_amplitude('beta', 'theta')

        assert beta_amp_high > beta_amp_low
        # Beta coupling has per_oscillator_strength={'theta': 0.6}, min_amplitude=0.3
        # At phase=0 (cos=1): 0.3 + (1-0.3)*1.0*0.6 = 0.72
        # At phase=π (cos=-1): 0.3 + (1-0.3)*0.0*0.6 = 0.3
        assert 0.71 <= beta_amp_high <= 0.73
        assert 0.29 <= beta_amp_low <= 0.31

    def test_all_couplings_simultaneous(self):
        """Test that all 5 couplings work correctly together."""
        manager = OscillatorManager()

        # Advance through time
        for _ in range(100):
            manager.advance(dt_ms=1.0)

        # Get coupled signals (applies all couplings)
        signals = manager.get_signals()

        # All signals should be in valid range
        for name, signal in signals.items():
            assert -1.0 <= signal <= 1.0, f"{name} signal out of range: {signal}"

        # Fast oscillators should show coupling effects
        # (not always 1.0 amplitude)
        gamma_amp = manager.get_coupled_amplitude('gamma', 'theta')
        assert 0.2 <= gamma_amp <= 1.0  # Modulated by theta

        theta_amp = manager.get_coupled_amplitude('theta', 'delta')
        assert 0.1 <= theta_amp <= 1.0  # Modulated by delta

        beta_amp = manager.get_coupled_amplitude('beta', 'theta')
        assert 0.5 <= beta_amp <= 1.0  # Modulated by theta

    def test_coupled_amplitudes_dictionary(self):
        """Test that get_coupled_amplitudes returns all modulation factors."""
        manager = OscillatorManager()
        manager.advance(dt_ms=10.0)

        # Get all coupled amplitudes
        coupled_amps = manager.get_coupled_amplitudes()

        # Should have entries for all 5 couplings
        assert 'gamma_by_theta' in coupled_amps
        assert 'gamma_by_beta' in coupled_amps
        assert 'theta_by_delta' in coupled_amps
        assert 'gamma_by_alpha' in coupled_amps
        assert 'beta_by_theta' in coupled_amps

        # All should be in valid range
        for key, amp in coupled_amps.items():
            assert 0.0 <= amp <= 1.0, f"{key} amplitude out of range: {amp}"
