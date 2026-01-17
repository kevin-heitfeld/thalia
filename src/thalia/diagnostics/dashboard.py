"""
Diagnostic Dashboard: Visualize network health metrics.

This module provides simple matplotlib-based visualizations for
monitoring network health during training and inference.

Features:
- Real-time health status display
- Metric time series plots
- Issue severity heatmaps
- Trend analysis

Usage:
======
    from thalia.diagnostics.dashboard import Dashboard

    dashboard = Dashboard()

    # During training loop
    for epoch in range(num_epochs):
        diagnostics = brain.get_diagnostics()
        dashboard.update(diagnostics)

        # Show live dashboard
        dashboard.show()

    # Save final report
    dashboard.save_report("training_health.png")

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .health_monitor import HealthConfig, HealthMonitor, HealthReport


class Dashboard:
    """Interactive dashboard for network health monitoring.

    This provides a simple matplotlib-based interface for visualizing
    network health metrics over time. Designed to be called in a loop
    during training/testing.
    """

    def __init__(
        self,
        health_config: Optional[HealthConfig] = None,
        window_size: int = 100,
        figsize: tuple[int, int] = (15, 10),
    ):
        """Initialize dashboard.

        Args:
            health_config: Configuration for health monitoring
            window_size: Number of timesteps to show in plots
            figsize: Figure size (width, height)
        """
        self.monitor = HealthMonitor(health_config)
        self.window_size = window_size
        self.figsize = figsize

        # History tracking
        self._timesteps: List[int] = []
        self._spike_rates: List[float] = []
        self._ei_ratios: List[float] = []
        self._branching_ratios: List[float] = []
        self._dopamine_levels: List[float] = []
        self._health_scores: List[float] = []  # 100 - severity
        self._reports: List[HealthReport] = []

        self._current_timestep = 0

        # Figure and axes (created on first show)
        self._fig: Optional[Figure] = None
        self._axes: Optional[List[Axes]] = None

    def update(self, diagnostics: Dict[str, Any]):
        """Update dashboard with new diagnostic data.

        Args:
            diagnostics: Dictionary from brain.get_diagnostics()
        """
        # Check health
        report = self.monitor.check_health(diagnostics)

        # Record metrics
        self._timesteps.append(self._current_timestep)
        self._spike_rates.append(report.metrics.get("avg_spike_rate", 0.0))
        self._ei_ratios.append(report.metrics.get("ei_ratio") or 0.0)

        # Branching ratio
        criticality = diagnostics.get("criticality", {})
        self._branching_ratios.append(criticality.get("branching_ratio") or 0.0)

        self._dopamine_levels.append(report.metrics.get("dopamine", 0.0))
        self._health_scores.append(100.0 - report.overall_severity)
        self._reports.append(report)

        # Trim to window
        if len(self._timesteps) > self.window_size:
            self._timesteps = self._timesteps[-self.window_size :]
            self._spike_rates = self._spike_rates[-self.window_size :]
            self._ei_ratios = self._ei_ratios[-self.window_size :]
            self._branching_ratios = self._branching_ratios[-self.window_size :]
            self._dopamine_levels = self._dopamine_levels[-self.window_size :]
            self._health_scores = self._health_scores[-self.window_size :]
            self._reports = self._reports[-self.window_size :]

        self._current_timestep += 1

    def _setup_time_series_plot(
        self,
        ax: Axes,
        data: List[float],
        title: str,
        ylabel: str,
        color: str = "blue",
        thresholds: Optional[Dict[str, float]] = None,
        target_value: Optional[float] = None,
    ) -> None:
        """Helper method to setup a time series plot with common formatting.

        Args:
            ax: Matplotlib axes object
            data: Data to plot
            title: Plot title
            ylabel: Y-axis label
            color: Line color
            thresholds: Dict with 'min' and 'max' threshold values
            target_value: Optional target line to draw
        """
        ax.plot(self._timesteps, data, color=color, linewidth=2)

        if thresholds:
            if "min" in thresholds:
                ax.axhline(
                    y=thresholds["min"], color="r", linestyle="--", alpha=0.5, label="Min threshold"
                )
            if "max" in thresholds:
                ax.axhline(
                    y=thresholds["max"], color="r", linestyle="--", alpha=0.5, label="Max threshold"
                )

        if target_value is not None:
            ax.axhline(y=target_value, color="g", linestyle=":", alpha=0.5, label="Target")

        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _setup_health_score_plot(self, ax: Axes) -> None:
        """Setup the overall health score plot."""
        ax.plot(self._timesteps, self._health_scores, "b-", linewidth=2)
        ax.axhline(y=90, color="g", linestyle="--", alpha=0.5, label="Good")
        ax.axhline(y=70, color="orange", linestyle="--", alpha=0.5, label="Warning")
        ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Critical")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Health Score")
        ax.set_title("Overall Health (100 - max severity)")
        ax.set_ylim([0, 105])
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    def _setup_issues_text_plot(self, ax: Axes, latest_report: HealthReport) -> None:
        """Setup the issues text display plot.

        Args:
            ax: Matplotlib axes object
            latest_report: Latest health report
        """
        ax.axis("off")

        # Status box
        status_color = "green" if latest_report.is_healthy else "red"
        status_text = "✓ HEALTHY" if latest_report.is_healthy else "⚠ ISSUES DETECTED"

        ax.text(
            0.5,
            0.95,
            status_text,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=status_color,
            transform=ax.transAxes,
        )

        # List issues
        y_pos = 0.85
        if latest_report.issues:
            ax.text(
                0.05,
                y_pos,
                "Current Issues:",
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                transform=ax.transAxes,
            )
            y_pos -= 0.1

            for issue in latest_report.issues[:5]:  # Show top 5
                severity_color = "orange" if issue.severity < 50 else "red"
                issue_text = f"• {issue.description}\n  → {issue.recommendation}"

                ax.text(
                    0.05,
                    y_pos,
                    issue_text,
                    ha="left",
                    va="top",
                    fontsize=9,
                    color=severity_color,
                    transform=ax.transAxes,
                    wrap=True,
                )
                y_pos -= 0.15
        else:
            ax.text(
                0.5,
                0.5,
                "All Systems Nominal",
                ha="center",
                va="center",
                fontsize=12,
                color="green",
                transform=ax.transAxes,
            )

    def show(self, block: bool = False):
        """Display the dashboard.

        Args:
            block: Whether to block execution (for interactive mode)
        """
        if not self._timesteps:
            print("No data to display yet")
            return

        # Create figure if needed
        if self._fig is None:
            self._fig, self._axes = plt.subplots(3, 2, figsize=self.figsize)
            self._fig.suptitle("Thalia Network Health Dashboard", fontsize=16)
            plt.ion()  # Interactive mode

        # Clear all axes
        for ax_row in self._axes:
            for ax in ax_row:
                ax.clear()

        # Get latest report
        latest_report = self._reports[-1]
        cfg = self.monitor.config

        # =====================================================================
        # Plot 1: Overall Health Score
        # =====================================================================
        self._setup_health_score_plot(self._axes[0][0])

        # =====================================================================
        # Plot 2: Spike Rate
        # =====================================================================
        self._setup_time_series_plot(
            self._axes[0][1],
            data=self._spike_rates,
            title="Average Spike Rate",
            ylabel="Spike Rate",
            color="purple",
            thresholds={"min": cfg.spike_rate_min, "max": cfg.spike_rate_max},
        )

        # =====================================================================
        # Plot 3: E/I Ratio
        # =====================================================================
        if any(r != 0.0 for r in self._ei_ratios):
            self._setup_time_series_plot(
                self._axes[1][0],
                data=self._ei_ratios,
                title="Excitation/Inhibition Balance",
                ylabel="E/I Ratio",
                color="green",
                thresholds={"min": cfg.ei_ratio_min, "max": cfg.ei_ratio_max},
                target_value=4.0,
            )
        else:
            ax = self._axes[1][0]
            ax.text(
                0.5,
                0.5,
                "E/I Balance Not Enabled",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Timestep")
            ax.set_title("Excitation/Inhibition Balance")
            ax.grid(True, alpha=0.3)

        # =====================================================================
        # Plot 4: Criticality (Branching Ratio)
        # =====================================================================
        if any(b != 0.0 for b in self._branching_ratios):
            self._setup_time_series_plot(
                self._axes[1][1],
                data=self._branching_ratios,
                title="Criticality (Branching Ratio)",
                ylabel="Branching Ratio",
                color="orange",
                thresholds={"min": cfg.criticality_min, "max": cfg.criticality_max},
                target_value=1.0,
            )
        else:
            ax = self._axes[1][1]
            ax.text(
                0.5,
                0.5,
                "Criticality Monitor Not Enabled",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Timestep")
            ax.set_title("Criticality (Branching Ratio)")
            ax.grid(True, alpha=0.3)

        # =====================================================================
        # Plot 5: Dopamine Level
        # =====================================================================
        self._setup_time_series_plot(
            self._axes[2][0],
            data=self._dopamine_levels,
            title="Global Dopamine",
            ylabel="Dopamine Level",
            color="red",
            thresholds={"max": cfg.dopamine_max},
        )

        # =====================================================================
        # Plot 6: Current Issues (Text)
        # =====================================================================
        self._setup_issues_text_plot(self._axes[2][1], latest_report)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Allow GUI to update

        if block:
            plt.show()

    def save_report(self, path: str | Path):
        """Save current dashboard to file.

        Args:
            path: Output file path (PNG, PDF, etc.)
        """
        if self._fig is None:
            self.show()  # Generate figure

        if self._fig is not None:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Dashboard saved to {path}")

    def close(self):
        """Close the dashboard."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary stats
        """
        if not self._reports:
            return {"status": "no_data"}

        # Compute statistics
        healthy_count = sum(1 for r in self._reports if r.is_healthy)
        total = len(self._reports)

        # Average metrics
        avg_spike_rate = (
            sum(self._spike_rates) / len(self._spike_rates) if self._spike_rates else 0.0
        )
        avg_health_score = (
            sum(self._health_scores) / len(self._health_scores) if self._health_scores else 0.0
        )

        # Issue frequency
        issue_counts: Dict[str, int] = {}
        for report in self._reports:
            for issue in report.issues:
                issue_type = issue.issue_type.value
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        return {
            "total_timesteps": total,
            "healthy_percentage": 100.0 * healthy_count / total,
            "avg_spike_rate": avg_spike_rate,
            "avg_health_score": avg_health_score,
            "issue_counts": issue_counts,
            "current_status": self._reports[-1].summary,
        }

    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("HEALTH DASHBOARD SUMMARY")
        print("=" * 60)
        print(f"Total timesteps: {summary['total_timesteps']}")
        print(f"Healthy: {summary['healthy_percentage']:.1f}%")
        print(f"Avg spike rate: {summary['avg_spike_rate']:.4f}")
        print(f"Avg health score: {summary['avg_health_score']:.1f}/100")
        print(f"\nCurrent status: {summary['current_status']}")

        if summary["issue_counts"]:
            print("\nIssue frequency:")
            for issue_type, count in sorted(summary["issue_counts"].items(), key=lambda x: -x[1]):
                print(f"  {issue_type}: {count} times")

        print("=" * 60 + "\n")
