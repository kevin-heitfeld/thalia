"""
Interactive Training Monitor with Matplotlib.

Provides real-time visualization of training progress using matplotlib.
Works in Jupyter notebooks, Colab, and local Python scripts.

Usage:
    from thalia.training import TrainingMonitor

    monitor = TrainingMonitor("training_runs/00_sensorimotor")
    monitor.show_progress()
    monitor.show_metrics()
    monitor.show_growth()

    # Auto-refresh in notebook
    monitor.start_auto_refresh(interval=5)  # Update every 5 seconds
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from thalia.constants.visualization import (
    ALPHA_SEMI_TRANSPARENT,
    AXIS_MARGIN_NEGATIVE,
    AXIS_MARGIN_POSITIVE,
    PROGRESS_BAR_HEIGHT,
    TEXT_POSITION_CENTER,
)


class TrainingMonitor:
    """
    Interactive training monitor with matplotlib visualizations.

    Features:
    - Real-time progress tracking
    - Metric plots (loss, accuracy, etc.)
    - Growth visualization (neuron counts)
    - Auto-refresh capability
    - Works in notebooks and scripts
    """

    def __init__(self, checkpoint_dir: str, figsize: tuple = (12, 8)):
        """
        Initialize monitor.

        Args:
            checkpoint_dir: Path to checkpoint directory
            figsize: Default figure size for plots
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.figsize = figsize
        self._stop_refresh = False
        self._refresh_thread = None
        self.data = self._load_data()

        # Set matplotlib style
        plt.style.use(
            "seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default"
        )

    def _load_data(self) -> Dict[str, Any]:
        """Load training data from checkpoints."""
        from thalia.io.checkpoint import BrainCheckpoint

        if not self.checkpoint_dir.exists():
            return {
                "error": f"Checkpoint directory not found: {self.checkpoint_dir}",
                "checkpoints": [],
            }

        # Find all checkpoints
        checkpoint_files = list(self.checkpoint_dir.glob("**/*.thalia"))

        if not checkpoint_files:
            return {
                "error": "No checkpoints found",
                "checkpoints": [],
            }

        # Load metadata from each checkpoint
        checkpoints = []
        for cp_file in sorted(checkpoint_files):
            try:
                info = BrainCheckpoint.info(cp_file)
                info["path"] = cp_file
                checkpoints.append(info)
            except Exception as e:
                print(f"Warning: Failed to load {cp_file}: {e}")

        return {
            "checkpoints": checkpoints,
            "latest": checkpoints[-1] if checkpoints else None,
        }

    def refresh(self, sections: Optional[List[str]] = None) -> None:
        """
        Refresh data and display selected sections.

        Args:
            sections: List of sections to show. Options: 'progress', 'metrics', 'growth'
                     If None, shows all sections.
        """
        self.data = self._load_data()

        if sections is None:
            self.show_all()
        else:
            for section in sections:
                if section == "progress":
                    self.show_progress()
                elif section == "metrics":
                    self.show_metrics()
                elif section == "growth":
                    self.show_growth()

    def show_progress(self) -> None:
        """Display training progress with matplotlib."""
        if "error" in self.data:
            print(f"[ERROR] {self.data['error']}")
            return

        latest = self.data.get("latest")
        if not latest:
            print("No checkpoint data available")
            return

        metadata = latest.get("metadata", {})
        stage_name = metadata.get("stage_name", "Unknown")
        step = metadata.get("step", 0)
        total_steps = metadata.get("total_steps", 0)

        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
        remaining = 100 - progress_pct

        # Create figure with 2 subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], 4))

        # Left: Progress pie chart
        colors = ["#4CAF50", "#E0E0E0"]
        explode = (0.05, 0)
        ax1.pie(
            [progress_pct, remaining],
            labels=["Complete", "Remaining"],
            autopct="%1.1f%%",
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={"fontsize": 12, "weight": "bold"},
        )
        ax1.set_title(f"🧠 {stage_name} Progress", fontsize=14, fontweight="bold")

        # Right: Status text
        ax2.axis("off")
        status_text = f"""
        Stage: {stage_name}
        Step: {step:,} / {total_steps:,}
        Progress: {progress_pct:.1f}%

        Checkpoints: {len(self.data['checkpoints'])}

        Status: {'[COMPLETE]' if progress_pct >= 100 else '[TRAINING]'}
        """
        ax2.text(
            0.1,
            TEXT_POSITION_CENTER,
            status_text.strip(),
            fontsize=11,
            family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        plt.show()

    def show_metrics(self) -> None:
        """Display training metrics over time."""
        if "error" in self.data:
            print(f"[ERROR] {self.data['error']}")
            return

        if not self.data["checkpoints"]:
            print("No checkpoint data available")
            return

        # Extract metrics from all checkpoints
        steps = []
        metrics_data = {}

        for cp in self.data["checkpoints"]:
            metadata = cp.get("metadata", {})
            step = metadata.get("step", 0)
            metrics = metadata.get("metrics", {})

            if not metrics:
                continue

            steps.append(step)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in metrics_data:
                        metrics_data[key] = []
                    metrics_data[key].append(value)

        if not metrics_data:
            print("No metrics available")
            return

        # Create subplots for different metrics
        n_metrics = len(metrics_data)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        _, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col]

            ax.plot(steps, values, linewidth=2, marker="o", markersize=4, color="#2196F3")
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(f"📈 {metric_name}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].axis("off")

        plt.tight_layout()
        plt.show()

    def show_growth(self) -> None:
        """Display neuron growth over time."""
        if "error" in self.data:
            print(f"[ERROR] {self.data['error']}")
            return

        if not self.data["checkpoints"]:
            print("No checkpoint data available")
            return

        # Extract growth data
        steps = []
        region_counts = {}

        for cp in self.data["checkpoints"]:
            metadata = cp.get("metadata", {})
            step = metadata.get("step", 0)
            regions = metadata.get("regions", {})

            if not regions:
                continue

            steps.append(step)
            for region_name, region_info in regions.items():
                n_neurons = region_info.get("n_neurons", 0)
                if region_name not in region_counts:
                    region_counts[region_name] = []
                region_counts[region_name].append(n_neurons)

        if not region_counts:
            print("No growth data available")
            return

        # Create stacked area chart
        _, ax = plt.subplots(figsize=(self.figsize[0], 6))

        # Stack the regions
        bottom = [0] * len(steps)
        colors = plt.cm.Set3(range(len(region_counts)))

        for idx, (region_name, counts) in enumerate(region_counts.items()):
            ax.fill_between(
                steps,
                bottom,
                [b + c for b, c in zip(bottom, counts)],
                label=region_name,
                alpha=0.7,
                color=colors[idx],
            )
            bottom = [b + c for b, c in zip(bottom, counts)]

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Number of Neurons", fontsize=12)
        ax.set_title("🌱 Neurogenesis: Brain Growth Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add total count annotation
        total = sum(counts[-1] for counts in region_counts.values())
        ax.text(
            0.98,
            0.98,
            f"Total: {total:,} neurons",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=ALPHA_SEMI_TRANSPARENT),
        )

        plt.tight_layout()
        plt.show()

    def show_all(self) -> None:
        """Display all monitoring sections."""
        self.show_progress()
        self.show_metrics()
        self.show_growth()

    def start_auto_refresh(self, interval: int = 5, sections: Optional[List[str]] = None) -> None:
        """
        Start auto-refreshing in background thread.

        Args:
            interval: Refresh interval in seconds
            sections: Sections to display (None = all)
        """
        if self._refresh_thread and self._refresh_thread.is_alive():
            print("Auto-refresh already running")
            return

        self._stop_refresh = False

        def refresh_loop():
            while not self._stop_refresh:
                plt.close("all")  # Close previous figures
                self.refresh(sections)
                time.sleep(interval)

        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self._refresh_thread.start()
        print(f"[OK] Auto-refresh started (interval: {interval}s)")
        print("   Call monitor.stop_auto_refresh() to stop")

    def stop_auto_refresh(self) -> None:
        """Stop auto-refresh."""
        self._stop_refresh = True
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2)
        print("🛑 Auto-refresh stopped")

    def save_report(self, output_path: str) -> None:
        """
        Save comprehensive training report to file.

        Args:
            output_path: Path to save report (PNG or PDF)
        """
        if "error" in self.data:
            print(f"[ERROR] Cannot save report: {self.data['error']}")
            return

        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Progress
        ax_progress = fig.add_subplot(gs[0, :])
        self._plot_progress_to_ax(ax_progress)

        # Metrics
        ax_metrics = fig.add_subplot(gs[1, :])
        self._plot_metrics_to_ax(ax_metrics)

        # Growth
        ax_growth = fig.add_subplot(gs[2, :])
        self._plot_growth_to_ax(ax_growth)

        # Add title
        fig.suptitle("🧠 Thalia Training Report", fontsize=18, fontweight="bold")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Report saved to: {output_path}")
        plt.close(fig)

    def _plot_progress_to_ax(self, ax):
        """Helper to plot progress to specific axes."""
        latest = self.data.get("latest")
        if not latest:
            ax.text(TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, "No data", ha="center", va="center")
            ax.axis("off")
            return

        metadata = latest.get("metadata", {})
        step = metadata.get("step", 0)
        total_steps = metadata.get("total_steps", 0)
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0

        ax.barh([0], [progress_pct], color="#4CAF50", height=PROGRESS_BAR_HEIGHT)
        ax.barh(
            [0],
            [100 - progress_pct],
            left=[progress_pct],
            color="#E0E0E0",
            height=PROGRESS_BAR_HEIGHT,
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(AXIS_MARGIN_NEGATIVE, AXIS_MARGIN_POSITIVE)
        ax.set_xlabel("Progress (%)")
        ax.set_title("Training Progress")
        ax.set_yticks([])
        ax.text(
            progress_pct / 2, 0, f"{progress_pct:.1f}%", ha="center", va="center", fontweight="bold"
        )

    def _plot_metrics_to_ax(self, ax):
        """Helper to plot metrics to specific axes."""
        # Simplified metric plot for report
        steps = []
        avg_metric = []

        for cp in self.data["checkpoints"]:
            metadata = cp.get("metadata", {})
            step = metadata.get("step", 0)
            metrics = metadata.get("metrics", {})

            if metrics:
                steps.append(step)
                # Average all numeric metrics
                numeric_vals = [v for v in metrics.values() if isinstance(v, (int, float))]
                avg_metric.append(sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0)

        if steps:
            ax.plot(steps, avg_metric, linewidth=2, color="#2196F3")
            ax.set_xlabel("Step")
            ax.set_ylabel("Average Metric Value")
            ax.set_title("Metrics Over Time")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                TEXT_POSITION_CENTER, TEXT_POSITION_CENTER, "No metrics", ha="center", va="center"
            )
            ax.axis("off")

    def _plot_growth_to_ax(self, ax):
        """Helper to plot growth to specific axes."""
        steps = []
        total_neurons = []

        for cp in self.data["checkpoints"]:
            metadata = cp.get("metadata", {})
            step = metadata.get("step", 0)
            regions = metadata.get("regions", {})

            if regions:
                steps.append(step)
                total = sum(r.get("n_neurons", 0) for r in regions.values())
                total_neurons.append(total)

        if steps:
            ax.plot(steps, total_neurons, linewidth=2, color="#4CAF50", marker="o")
            ax.set_xlabel("Step")
            ax.set_ylabel("Total Neurons")
            ax.set_title("Neurogenesis")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                TEXT_POSITION_CENTER,
                TEXT_POSITION_CENTER,
                "No growth data",
                ha="center",
                va="center",
            )
            ax.axis("off")


def quick_monitor(checkpoint_dir: str) -> None:
    """Quick monitoring function - shows all info."""
    monitor = TrainingMonitor(checkpoint_dir)
    monitor.show_all()
