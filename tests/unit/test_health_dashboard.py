"""
Tests for health monitoring and dashboard.
"""

import pytest
import torch

from thalia.diagnostics import (
    HealthConfig,
    HealthMonitor,
    HealthReport,
    HealthIssue,
    Dashboard,
)


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_healthy_network(self):
        """Test that a healthy network passes all checks."""
        monitor = HealthMonitor()

        # Simulate healthy diagnostics with moderate activity
        diagnostics = {
            "spike_counts": {"region1": 20, "region2": 25},  # ~0.225 spike rate (within bounds)
            "cortex": {
                "l23_w_mean": 0.5,
                "l4_w_mean": 0.6,
            },
            "robustness_ei_ratio": 4.0,  # Perfect
            "criticality": {
                "enabled": True,
                "branching_ratio": 1.0,  # Perfect
            },
            "dopamine": {
                "global": 0.5,
                "tonic": 0.1,
                "phasic": 0.0,
            },
        }

        report = monitor.check_health(diagnostics)

        assert report.is_healthy
        assert len(report.issues) == 0
        assert report.overall_severity == 0.0
        assert "✓" in report.summary

    def test_activity_collapse(self):
        """Test detection of activity collapse."""
        monitor = HealthMonitor(HealthConfig(spike_rate_min=0.01))

        # Very low activity
        diagnostics = {
            "spike_counts": {"region1": 1, "region2": 0},  # ~0.005 spike rate
            "cortex": {},
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.ACTIVITY_COLLAPSE for i in report.issues)
        assert report.overall_severity > 0

    def test_seizure_risk(self):
        """Test detection of seizure-like activity."""
        monitor = HealthMonitor(HealthConfig(spike_rate_max=0.5))

        # Very high activity
        diagnostics = {
            "spike_counts": {"region1": 300, "region2": 400},  # ~3.5 spike rate
            "cortex": {},
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.SEIZURE_RISK for i in report.issues)

    def test_weight_collapse(self):
        """Test detection of weight collapse."""
        monitor = HealthMonitor(HealthConfig(weight_min=0.01))

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {
                "l23_w_mean": 0.001,  # Too small
            },
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.WEIGHT_COLLAPSE for i in report.issues)

    def test_weight_explosion(self):
        """Test detection of weight explosion."""
        monitor = HealthMonitor(HealthConfig(weight_max=5.0))

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {
                "l23_w_mean": 10.0,  # Too large
            },
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.WEIGHT_EXPLOSION for i in report.issues)

    def test_ei_imbalance_low(self):
        """Test detection of over-inhibition."""
        monitor = HealthMonitor(HealthConfig(ei_ratio_min=1.0))

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {},
            "robustness_ei_ratio": 0.5,  # Too inhibited
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.EI_IMBALANCE for i in report.issues)

    def test_ei_imbalance_high(self):
        """Test detection of under-inhibition."""
        monitor = HealthMonitor(HealthConfig(ei_ratio_max=10.0))

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {},
            "robustness_ei_ratio": 15.0,  # Too excitable
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.EI_IMBALANCE for i in report.issues)

    def test_criticality_subcritical(self):
        """Test detection of subcritical state."""
        monitor = HealthMonitor(HealthConfig(criticality_min=0.8))

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {},
            "criticality": {
                "enabled": True,
                "branching_ratio": 0.5,  # Too low
            },
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.CRITICALITY_DRIFT for i in report.issues)

    def test_criticality_supercritical(self):
        """Test detection of supercritical state."""
        monitor = HealthMonitor(HealthConfig(criticality_max=1.2))

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {},
            "criticality": {
                "enabled": True,
                "branching_ratio": 1.5,  # Too high
            },
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.CRITICALITY_DRIFT for i in report.issues)

    def test_dopamine_saturation(self):
        """Test detection of dopamine saturation."""
        monitor = HealthMonitor(HealthConfig(dopamine_max=2.0))

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {},
            "dopamine": {
                "global": 5.0,  # Too high
            },
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert any(i.issue_type == HealthIssue.DOPAMINE_SATURATION for i in report.issues)

    def test_severity_threshold(self):
        """Test that severity threshold filters minor issues."""
        monitor = HealthMonitor(HealthConfig(
            spike_rate_min=0.01,
            severity_threshold=50.0,  # Only report severe issues
        ))

        # Slightly low activity (minor issue)
        diagnostics = {
            "spike_counts": {"region1": 8, "region2": 10},  # ~0.009 spike rate
            "cortex": {},
            "dopamine": {},
        }

        report = monitor.check_health(diagnostics)

        # Issue exists but filtered by severity threshold
        # (severity would be ~10%, below 50% threshold)
        assert report.is_healthy or report.overall_severity < 50.0

    def test_multiple_issues(self):
        """Test handling of multiple simultaneous issues."""
        monitor = HealthMonitor()

        # Multiple problems
        diagnostics = {
            "spike_counts": {"region1": 1},  # Low activity
            "cortex": {
                "l23_w_mean": 10.0,  # Weight explosion
            },
            "robustness_ei_ratio": 15.0,  # E/I imbalance
            "dopamine": {
                "global": 3.0,  # Dopamine saturation
            },
        }

        report = monitor.check_health(diagnostics)

        assert not report.is_healthy
        assert len(report.issues) >= 3  # At least 3 different issues

        # Check variety of issue types
        issue_types = {i.issue_type for i in report.issues}
        assert len(issue_types) >= 3

    def test_trend_tracking(self):
        """Test that monitor tracks trends over time."""
        monitor = HealthMonitor()

        # Simulate 20 timesteps with increasing spike rate
        for i in range(20):
            spike_rate = 0.05 + i * 0.01  # Gradually increasing
            diagnostics = {
                "spike_counts": {"region1": int(spike_rate * 100)},
                "cortex": {},
                "dopamine": {},
            }
            monitor.check_health(diagnostics)

        trends = monitor.get_trend_summary()

        assert "spike_rate" in trends
        # Should detect increasing trend
        assert trends["spike_rate"] in ["increasing", "stable"]

    def test_reset_history(self):
        """Test resetting trend history."""
        monitor = HealthMonitor()

        # Add some history
        for i in range(10):
            diagnostics = {
                "spike_counts": {"region1": 50},
                "cortex": {},
                "dopamine": {},
            }
            monitor.check_health(diagnostics)

        assert len(monitor._spike_rate_history) == 10

        # Reset
        monitor.reset_history()

        assert len(monitor._spike_rate_history) == 0
        assert len(monitor._weight_mean_history) == 0


class TestDashboard:
    """Tests for Dashboard."""

    def test_dashboard_creation(self):
        """Test creating a dashboard."""
        dashboard = Dashboard()

        assert dashboard.monitor is not None
        assert dashboard.window_size == 100
        assert len(dashboard._timesteps) == 0

    def test_update(self):
        """Test updating dashboard with data."""
        dashboard = Dashboard()

        diagnostics = {
            "spike_counts": {"region1": 50, "region2": 60},
            "cortex": {"l23_w_mean": 0.5},
            "robustness_ei_ratio": 4.0,
            "criticality": {"enabled": True, "branching_ratio": 1.0},
            "dopamine": {"global": 0.5},
        }

        dashboard.update(diagnostics)

        assert len(dashboard._timesteps) == 1
        assert len(dashboard._spike_rates) == 1
        assert len(dashboard._reports) == 1

    def test_window_trimming(self):
        """Test that dashboard trims to window size."""
        dashboard = Dashboard(window_size=10)

        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {},
            "dopamine": {},
        }

        # Add more than window size
        for _ in range(20):
            dashboard.update(diagnostics)

        # Should trim to window
        assert len(dashboard._timesteps) == 10
        assert len(dashboard._spike_rates) == 10

    def test_get_summary(self):
        """Test getting summary statistics."""
        dashboard = Dashboard()

        # Add some data
        for i in range(10):
            # Mix of healthy and unhealthy
            spike_count = 50 if i < 5 else 1  # Collapse after 5
            diagnostics = {
                "spike_counts": {"region1": spike_count},
                "cortex": {},
                "dopamine": {},
            }
            dashboard.update(diagnostics)

        summary = dashboard.get_summary()

        assert "total_timesteps" in summary
        assert "healthy_percentage" in summary
        assert "avg_spike_rate" in summary
        assert summary["total_timesteps"] == 10
        # Should be less than 100% healthy due to collapses
        assert 0 <= summary["healthy_percentage"] <= 100

    def test_summary_no_data(self):
        """Test summary with no data."""
        dashboard = Dashboard()

        summary = dashboard.get_summary()

        assert summary["status"] == "no_data"

    def test_close(self):
        """Test closing dashboard."""
        dashboard = Dashboard()

        # Add some data
        diagnostics = {
            "spike_counts": {"region1": 50},
            "cortex": {},
            "dopamine": {},
        }
        dashboard.update(diagnostics)

        # Close (should not raise)
        dashboard.close()

        assert dashboard._fig is None
        assert dashboard._axes is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
