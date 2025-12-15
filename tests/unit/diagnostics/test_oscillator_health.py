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


class TestCrossRegionSynchrony:
    """Test cross-region phase synchrony metrics."""

    def test_perfect_synchrony(self):
        """Test detection of perfect phase synchrony."""
        monitor = OscillatorHealthMonitor()

        # Two regions with identical phases (perfect synchrony)
        region1_phases = {'theta': 1.0, 'gamma': 2.0}
        region2_phases = {'theta': 1.0, 'gamma': 2.0}

        # Should get coherence = 1.0
        coherence = monitor.compute_phase_coherence(
            region1_phases, region2_phases, 'theta'
        )

        assert coherence == pytest.approx(1.0, abs=0.01)

    def test_zero_synchrony(self):
        """Test detection of opposite phases (no synchrony)."""
        monitor = OscillatorHealthMonitor()

        # Two regions with opposite phases (π radians apart)
        region1_phases = {'theta': 0.0, 'gamma': 0.0}
        region2_phases = {'theta': math.pi, 'gamma': math.pi}

        # Should get coherence ≈ 0.0
        coherence = monitor.compute_phase_coherence(
            region1_phases, region2_phases, 'theta'
        )

        assert coherence == pytest.approx(0.0, abs=0.01)

    def test_partial_synchrony(self):
        """Test detection of partial synchrony."""
        monitor = OscillatorHealthMonitor()

        # Two regions with π/2 phase difference
        region1_phases = {'theta': 0.0}
        region2_phases = {'theta': math.pi / 2}

        coherence = monitor.compute_phase_coherence(
            region1_phases, region2_phases, 'theta'
        )

        # cos(π/2) = 0, so (1 + 0)/2 = 0.5
        assert coherence == pytest.approx(0.5, abs=0.01)

    def test_region_pair_coherence(self):
        """Test computing coherence for multiple region pairs."""
        monitor = OscillatorHealthMonitor()

        # Three regions with varying synchrony
        region_phases = {
            'hippocampus': {'theta': 1.0, 'gamma': 2.0},
            'prefrontal': {'theta': 1.1, 'gamma': 2.5},  # Slight theta offset
            'cortex': {'theta': 1.0, 'gamma': 3.0},      # Large gamma offset
        }

        # Compute all pairs for theta and gamma
        coherence_map = monitor.compute_region_pair_coherence(
            region_phases,
            oscillators=['theta', 'gamma']
        )

        # Should have 3 pairs: hippo-pfc, hippo-cortex, pfc-cortex
        assert len(coherence_map) == 3

        # Hippocampus-prefrontal theta should be high (small phase diff)
        hippo_pfc = coherence_map.get('hippocampus-prefrontal')
        assert hippo_pfc is not None
        assert hippo_pfc['theta'] > 0.9  # Very close phases

    def test_working_memory_synchrony_detection(self):
        """Test detection of working memory synchrony (hippocampus-PFC theta)."""
        monitor = OscillatorHealthMonitor()

        # Good working memory: high hippocampus-PFC theta coherence
        region_phases_good = {
            'hippocampus': {'theta': 1.0, 'gamma': 2.0},
            'prefrontal': {'theta': 1.05, 'gamma': 2.2},  # Close theta phases
        }

        issues = monitor.check_cross_region_synchrony(region_phases_good)
        assert len(issues) == 0  # No issues with high coherence

        # Poor working memory: low hippocampus-PFC theta coherence
        region_phases_poor = {
            'hippocampus': {'theta': 0.0, 'gamma': 2.0},
            'prefrontal': {'theta': math.pi, 'gamma': 2.2},  # Opposite theta phases
        }

        issues = monitor.check_cross_region_synchrony(region_phases_poor)
        # Should detect desynchrony
        assert len(issues) > 0
        assert any(
            issue.issue_type == OscillatorIssue.CROSS_REGION_DESYNCHRONY
            for issue in issues
        )

    def test_cross_modal_binding_synchrony(self):
        """Test detection of cross-modal binding (visual-auditory gamma)."""
        monitor = OscillatorHealthMonitor()

        # Good binding: high visual-auditory gamma coherence
        region_phases_good = {
            'visual_cortex': {'gamma': 2.0, 'theta': 1.0},
            'auditory_cortex': {'gamma': 2.1, 'theta': 1.5},  # Close gamma phases
        }

        issues = monitor.check_cross_region_synchrony(region_phases_good)
        assert len(issues) == 0  # No issues

        # Poor binding: low gamma coherence
        region_phases_poor = {
            'visual_cortex': {'gamma': 0.0, 'theta': 1.0},
            'auditory_cortex': {'gamma': math.pi, 'theta': 1.5},  # Opposite gamma
        }

        issues = monitor.check_cross_region_synchrony(region_phases_poor)
        # Should detect desynchrony
        assert len(issues) > 0
        desynchrony_issues = [
            i for i in issues
            if i.issue_type == OscillatorIssue.CROSS_REGION_DESYNCHRONY
        ]
        assert len(desynchrony_issues) > 0

    def test_custom_synchrony_expectations(self):
        """Test checking synchrony with custom expectations."""
        monitor = OscillatorHealthMonitor()

        region_phases = {
            'region1': {'theta': 1.0, 'gamma': 2.0},
            'region2': {'theta': math.pi, 'gamma': 2.1},  # Opposite theta phases
        }

        # Custom expectation: require high theta coherence
        custom_expectations = {
            ('region1', 'region2'): {'theta': 0.7}  # High threshold
        }

        issues = monitor.check_cross_region_synchrony(
            region_phases,
            expected_synchrony=custom_expectations
        )

        # Should detect poor theta synchrony (opposite phases)
        assert len(issues) > 0
        assert any('theta' in issue.oscillator_name for issue in issues)

    def test_phase_wrapping(self):
        """Test that phase differences handle 2π wrapping correctly."""
        monitor = OscillatorHealthMonitor()

        # Phases near 0 and 2π should be considered close
        region1_phases = {'theta': 0.1}
        region2_phases = {'theta': 2 * math.pi - 0.1}  # Almost 2π

        coherence = monitor.compute_phase_coherence(
            region1_phases, region2_phases, 'theta'
        )

        # These are very close (0.2 radians apart after wrapping)
        assert coherence > 0.95

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
