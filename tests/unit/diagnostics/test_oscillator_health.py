"""
Tests for oscillator health monitoring.

Tests the OscillatorHealthMonitor class and its integration with HealthMonitor.
"""

import math
import pytest

from thalia.diagnostics.oscillator_health import (
    OscillatorHealthMonitor,
    OscillatorHealthConfig,
    OscillatorIssue,
)


class TestOscillatorHealthMonitor:
    """Test OscillatorHealthMonitor functionality."""

    def test_healthy_oscillators(self):
        """Test that healthy oscillators pass all checks."""
        monitor = OscillatorHealthMonitor()

        # Simulate healthy oscillator state
        phases = {
            'delta': 0.5,
            'theta': 1.2,
            'alpha': 2.1,
            'beta': 0.9,
            'gamma': 3.0,
        }
        frequencies = {
            'delta': 2.0,
            'theta': 8.0,
            'alpha': 10.0,
            'beta': 20.0,
            'gamma': 40.0,
        }
        amplitudes = {
            'delta': 1.0,
            'theta': 0.8,
            'alpha': 0.7,
            'beta': 0.6,
            'gamma': 0.5,
        }

        report = monitor.check_health(phases, frequencies, amplitudes)

        assert report.is_healthy
        assert len(report.issues) == 0
        assert report.overall_severity == 0.0

    def test_frequency_drift_detection(self):
        """Test detection of abnormal frequencies."""
        monitor = OscillatorHealthMonitor()

        # Theta frequency out of range (should be 4-10 Hz)
        phases = {'theta': 1.2}
        frequencies = {'theta': 15.0}  # Too high
        amplitudes = {'theta': 0.8}

        report = monitor.check_health(phases, frequencies, amplitudes)

        assert not report.is_healthy
        assert len(report.issues) > 0
        assert any(issue.issue_type == OscillatorIssue.FREQUENCY_DRIFT for issue in report.issues)
        assert any(issue.oscillator_name == 'theta' for issue in report.issues)

    def test_phase_locking_detection(self):
        """Test detection of stuck oscillator phases."""
        config = OscillatorHealthConfig(phase_lock_window=10)
        monitor = OscillatorHealthMonitor(config)

        # Simulate stuck phase (no advancement)
        phases = {'theta': 1.0}
        frequencies = {'theta': 8.0}
        amplitudes = {'theta': 0.8}

        # Feed same phase multiple times
        for _ in range(15):
            report = monitor.check_health(phases, frequencies, amplitudes)

        # Should detect phase locking
        assert not report.is_healthy
        assert any(issue.issue_type == OscillatorIssue.PHASE_LOCKING for issue in report.issues)

    def test_abnormal_amplitude_detection(self):
        """Test detection of pathological amplitudes."""
        monitor = OscillatorHealthMonitor()

        # Amplitude too low (dead oscillator)
        phases = {'gamma': 2.0}
        frequencies = {'gamma': 40.0}
        amplitudes = {'gamma': 0.01}  # Below minimum

        report = monitor.check_health(phases, frequencies, amplitudes)

        assert not report.is_healthy
        assert any(issue.issue_type == OscillatorIssue.ABNORMAL_AMPLITUDE for issue in report.issues)

        # Amplitude too high (pathological)
        amplitudes = {'gamma': 2.0}  # Above maximum
        report = monitor.check_health(phases, frequencies, amplitudes)

        assert not report.is_healthy
        assert any(issue.issue_type == OscillatorIssue.ABNORMAL_AMPLITUDE for issue in report.issues)

    def test_advancing_phases(self):
        """Test that advancing phases are considered healthy."""
        monitor = OscillatorHealthMonitor()

        # Simulate properly advancing phases
        for step in range(50):
            phase = (step * 0.1) % (2 * math.pi)
            phases = {'theta': phase}
            frequencies = {'theta': 8.0}
            amplitudes = {'theta': 0.8}

            report = monitor.check_health(phases, frequencies, amplitudes)

        # Should be healthy (phases are advancing)
        assert report.is_healthy

    def test_oscillator_statistics(self):
        """Test oscillator statistics computation."""
        monitor = OscillatorHealthMonitor()

        # Feed some data
        for step in range(20):
            phases = {'theta': (step * 0.1) % (2 * math.pi)}
            frequencies = {'theta': 8.0 + step * 0.01}  # Slight drift
            amplitudes = {'theta': 0.8}

            monitor.check_health(phases, frequencies, amplitudes)

        # Get statistics
        stats = monitor.get_oscillator_statistics('theta')

        assert 'frequency' in stats
        assert 'amplitude' in stats
        assert stats['frequency']['mean'] > 8.0  # Should reflect the drift
        assert stats['frequency']['std'] > 0  # Should have some variance

    def test_history_reset(self):
        """Test that history can be reset."""
        monitor = OscillatorHealthMonitor()

        # Build some history
        for _ in range(10):
            phases = {'theta': 1.0}
            frequencies = {'theta': 8.0}
            amplitudes = {'theta': 0.8}
            monitor.check_health(phases, frequencies, amplitudes)

        # Reset
        monitor.reset_history()

        # Statistics should be empty
        stats = monitor.get_oscillator_statistics('theta')
        assert stats == {}


class TestHealthMonitorIntegration:
    """Test integration with main HealthMonitor."""

    def test_health_monitor_with_oscillators(self):
        """Test that HealthMonitor includes oscillator checks."""
        from thalia.diagnostics import HealthMonitor

        monitor = HealthMonitor(enable_oscillator_monitoring=True)

        # Should have oscillator monitor
        assert monitor.oscillator_monitor is not None

        # Create diagnostic data with oscillator info
        diagnostics = {
            "spike_counts": {"cortex": 100, "hippocampus": 50},
            "oscillators": {
                "phases": {'theta': 1.2, 'gamma': 2.5},
                "frequencies": {'theta': 8.0, 'gamma': 40.0},
                "amplitudes": {'theta': 0.8, 'gamma': 0.6},
            },
        }

        report = monitor.check_health(diagnostics)

        # Should include oscillator checks (healthy in this case)
        assert report is not None

    def test_health_monitor_without_oscillators(self):
        """Test that HealthMonitor works without oscillator data."""
        from thalia.diagnostics import HealthMonitor

        monitor = HealthMonitor(enable_oscillator_monitoring=True)

        # Diagnostics without oscillator data
        diagnostics = {
            "spike_counts": {"cortex": 100, "hippocampus": 50},
        }

        report = monitor.check_health(diagnostics)

        # Should work gracefully without oscillator data
        assert report is not None

    def test_health_monitor_disabled_oscillators(self):
        """Test HealthMonitor with oscillator monitoring disabled."""
        from thalia.diagnostics import HealthMonitor

        monitor = HealthMonitor(enable_oscillator_monitoring=False)

        # Should not have oscillator monitor
        assert monitor.oscillator_monitor is None

        diagnostics = {
            "spike_counts": {"cortex": 100, "hippocampus": 50},
        }

        report = monitor.check_health(diagnostics)

        # Should work without oscillator monitoring
        assert report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
