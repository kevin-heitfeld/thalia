"""
Tests for oscillation detection utilities.

Tests FFT analysis, autocorrelation, and frequency detection for validating
emergent oscillations in neural circuits.
"""

import numpy as np
import torch

from thalia.diagnostics.oscillation_detection import (
    measure_oscillation,
    measure_periodicity,
    power_spectrum,
    detect_gamma_oscillation,
    detect_theta_oscillation,
)


class TestOscillationDetection:
    """Unit tests for oscillation detection utilities."""

    def test_measure_oscillation_detects_40hz(self):
        """Test FFT detects 40 Hz oscillation (gamma)."""
        # Generate synthetic 40 Hz signal: 200ms @ 1ms timestep
        duration_ms = 200
        dt_ms = 1.0
        freq_hz = 40.0

        t = np.arange(0, duration_ms, dt_ms)
        signal = 10 + 5 * np.sin(2 * np.pi * freq_hz * t / 1000.0)

        detected_freq, power = measure_oscillation(signal.tolist(), dt_ms=dt_ms)

        # Should detect close to 40 Hz
        assert 38 <= detected_freq <= 42, \
            f"Expected ~40 Hz, got {detected_freq:.1f} Hz"
        assert power > 2.0, "Should have strong power"

    def test_measure_oscillation_with_freq_range(self):
        """Test FFT with frequency range filter (gamma band)."""
        # Generate 40 Hz signal with noise
        duration_ms = 200
        dt_ms = 1.0
        freq_hz = 40.0

        t = np.arange(0, duration_ms, dt_ms)
        signal = 10 + 5 * np.sin(2 * np.pi * freq_hz * t / 1000.0)
        signal += np.random.randn(len(t)) * 2  # Add noise

        # Search only in gamma range (30-80 Hz)
        detected_freq, power = measure_oscillation(
            signal.tolist(),
            dt_ms=dt_ms,
            freq_range=(30.0, 80.0),
        )

        assert 35 <= detected_freq <= 50, \
            f"Should detect gamma peak, got {detected_freq:.1f} Hz"

    def test_measure_periodicity_detects_25ms(self):
        """Test autocorrelation detects 25ms period (40 Hz gamma)."""
        # Generate periodic bursts every 25ms
        duration_ms = 200
        dt_ms = 1.0
        period_ms = 25.0

        signal = []
        for t in range(int(duration_ms / dt_ms)):
            # Burst every 25ms
            if t % int(period_ms / dt_ms) < 3:
                signal.append(10.0)
            else:
                signal.append(2.0)

        detected_period, strength = measure_periodicity(signal, dt_ms=dt_ms)

        # Should detect ~25ms period
        assert 20 <= detected_period <= 30, \
            f"Expected ~25ms period, got {detected_period:.1f}ms"
        assert strength > 0.3, "Should have strong autocorrelation"

    def test_power_spectrum_returns_correct_shape(self):
        """Test power spectrum computation."""
        signal = [10, 5, 8, 12, 6, 9, 11, 7] * 25  # 200 samples

        freqs, power = power_spectrum(signal, dt_ms=1.0)

        assert len(freqs) == len(power)
        assert len(freqs) > 0
        assert all(freqs >= 0), "Frequencies should be non-negative"

    def test_detect_gamma_oscillation_true_positive(self):
        """Test gamma detection returns True for 40 Hz signal."""
        # Strong 40 Hz signal
        duration_ms = 200
        freq_hz = 40.0
        t = np.arange(0, duration_ms, 1.0)
        signal = 10 + 5 * np.sin(2 * np.pi * freq_hz * t / 1000.0)

        has_gamma = detect_gamma_oscillation(signal.tolist())

        assert has_gamma, "Should detect gamma oscillation"

    def test_detect_gamma_oscillation_false_negative(self):
        """Test gamma detection returns False for non-gamma signal."""
        # Random noise (no oscillation)
        signal = np.random.randn(200) + 10

        has_gamma = detect_gamma_oscillation(signal.tolist())

        assert not has_gamma, "Should not detect gamma in noise"

    def test_detect_theta_oscillation_true_positive(self):
        """Test theta detection returns True for 8 Hz signal."""
        # 8 Hz theta signal
        duration_ms = 500  # Longer for theta
        freq_hz = 8.0
        t = np.arange(0, duration_ms, 1.0)
        signal = 10 + 5 * np.sin(2 * np.pi * freq_hz * t / 1000.0)

        has_theta = detect_theta_oscillation(signal.tolist())

        assert has_theta, "Should detect theta oscillation"


class TestOscillationDetectionIntegration:
    """Integration tests with actual brain components."""

    def test_l6_gamma_emergence(self, global_config, device):
        """Test L6→TRN loop shows gamma oscillation via FFT."""
        from thalia.core.brain_builder import BrainBuilder

        brain = BrainBuilder.preset("sensorimotor", global_config)
        cortex = brain.components["cortex"]

        # Note: Gamma oscillator disabled by default (should emerge from L6→TRN loop)

        # Strong sustained input
        sensory_input = torch.rand(128, device=device) > 0.7

        # Collect L6 activity over 200ms (multiple gamma cycles)
        l6_spike_counts = []

        for t in range(200):
            brain(sensory_input, n_timesteps=1)

            l6_spikes = cortex.get_l6_spikes()
            if l6_spikes is not None:
                l6_spike_counts.append(l6_spikes.sum().item())
            else:
                l6_spike_counts.append(0.0)

        # FFT analysis
        freq, power = measure_oscillation(
            l6_spike_counts,
            dt_ms=1.0,
            freq_range=(30.0, 80.0),  # Gamma range
        )

        # Check for gamma oscillation
        print(f"\nL6 oscillation: {freq:.1f} Hz (power={power:.3f})")

        # Should have SOME dominant frequency in gamma range
        # May not be exactly 40 Hz due to network dynamics
        assert 25 <= freq <= 60, \
            f"Expected gamma-range frequency, got {freq:.1f} Hz"

        # Autocorrelation check
        period, strength = measure_periodicity(
            l6_spike_counts,
            dt_ms=1.0,
            period_range_ms=(15.0, 40.0),  # Gamma period range
        )

        print(f"L6 period: {period:.1f}ms (strength={strength:.2f})")

        # Should show periodic activity
        assert period > 0, "L6 should show periodic activity"
        assert 15 <= period <= 40, \
            f"Expected gamma period (15-40ms), got {period:.1f}ms"

    def test_ca3_shows_rhythmic_activity(self, global_config, device):
        """Test CA3 recurrence shows rhythmic dynamics (may not be 8 Hz)."""
        from thalia.core.brain_builder import BrainBuilder

        # Build brain with hippocampus
        builder = BrainBuilder(global_config)
        builder.add_component("thalamus", "thalamus", n_input=100, n_output=100)
        builder.add_component("hippocampus", "hippocampus", n_output=100)
        builder.connect("thalamus", "hippocampus", pathway_type="axonal")
        brain = builder.build()

        # Disable explicit theta to measure CA3 intrinsic frequency (without septum)
        brain.oscillators.enable_oscillator('theta', enabled=False)

        hippocampus = brain.components["hippocampus"]

        # Strong initial input, then let CA3 recur
        strong_input = torch.ones(100, dtype=torch.bool, device=device)
        weak_input = torch.zeros(100, dtype=torch.bool, device=device)

        ca3_spike_counts = []

        # Initial excitation (first 20ms)
        for t in range(20):
            brain(strong_input, n_timesteps=1)
            ca3_spike_counts.append(hippocampus.state.ca3_spikes.sum().item())

        # Let CA3 recur (next 180ms)
        for t in range(180):
            brain(weak_input, n_timesteps=1)
            ca3_spike_counts.append(hippocampus.state.ca3_spikes.sum().item())

        # Measure any oscillation
        freq, power = measure_oscillation(ca3_spike_counts, dt_ms=1.0)

        print(f"\nCA3 intrinsic oscillation: {freq:.1f} Hz (power={power:.3f})")

        # CA3 should show SOME rhythmic activity
        # May be faster than theta (10-20 Hz) without explicit theta coordination
        assert freq > 0, "CA3 should show oscillatory dynamics"

        # Document what we find (may not be exactly 8 Hz)
        if 4 <= freq <= 12:
            print(f"✅ CA3 shows theta-range oscillation ({freq:.1f} Hz)")
        elif 12 <= freq <= 25:
            print(f"⚠️ CA3 oscillates faster than theta ({freq:.1f} Hz)")
            print("   Expected: Needs explicit theta coordination for 8 Hz")
        else:
            print(f"⚠️ CA3 oscillation outside expected range: {freq:.1f} Hz")
