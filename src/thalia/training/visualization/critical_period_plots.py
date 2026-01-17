"""
Critical Period Visualization

Visualize critical period windows and their progression over training.
Shows how learning rate multipliers change across developmental stages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from thalia.learning.critical_periods import CriticalPeriodGating


def plot_critical_period_windows(
    max_steps: int = 500000,
    gating: Optional[CriticalPeriodGating] = None,
    figsize: tuple = (14, 8),
) -> Figure:
    """Plot all critical period windows over training timeline.

    Shows when each domain is in early/peak/late phase and
    the corresponding learning rate multiplier.

    Args:
        max_steps: Maximum training steps to plot
        gating: CriticalPeriodGating instance (creates default if None)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure
    """
    if gating is None:
        gating = CriticalPeriodGating()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get all domains
    domains = gating.get_all_domains()

    # Colors for domains
    colors = {
        "motor": "#FF6B6B",
        "face_recognition": "#4ECDC4",
        "phonology": "#45B7D1",
        "grammar": "#96CEB4",
        "semantics": "#FFEAA7",
    }

    # Plot multiplier curves
    steps = np.arange(0, max_steps, 1000)

    for domain in domains:
        multipliers = []
        for step in steps:
            status = gating.get_window_status(domain, int(step))
            multipliers.append(status["multiplier"])

        color = colors.get(domain, "#95A5A6")
        ax.plot(
            steps / 1000,  # Convert to thousands
            multipliers,
            label=domain.replace("_", " ").title(),
            linewidth=2.5,
            color=color,
            alpha=0.9,
        )

    # Add reference lines
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Baseline (1.0x)")
    ax.axhline(y=1.2, color="green", linestyle=":", linewidth=1, alpha=0.3)
    ax.axhline(y=0.5, color="red", linestyle=":", linewidth=1, alpha=0.3)

    # Styling
    ax.set_xlabel("Training Step (thousands)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Learning Rate Multiplier", fontsize=12, fontweight="bold")
    ax.set_title("Critical Period Windows Across Development", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_xlim(0, max_steps / 1000)
    ax.set_ylim(0, 1.4)

    # Add stage markers
    stages = [
        (0, 50, "Stage -0.5\nSensorimotor"),
        (50, 100, "Stage 0\nPhonology"),
        (100, 200, "Stage 1\nToddler"),
        (200, 350, "Stage 2\nGrammar"),
        (350, 500, "Stage 3\nReading"),
    ]

    for start, end, label in stages:
        if end <= max_steps / 1000:
            ax.axvspan(start, end, alpha=0.05, color="gray")
            ax.text(
                (start + end) / 2,
                1.35,
                label,
                ha="center",
                va="center",
                fontsize=8,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
            )

    plt.tight_layout()
    return fig


def plot_critical_period_timeline(
    gating: Optional[CriticalPeriodGating] = None,
    figsize: tuple = (14, 6),
) -> Figure:
    """Plot critical period timeline showing early/peak/late phases.

    Shows a Gantt-chart style view of when each domain is in
    different phases.

    Args:
        gating: CriticalPeriodGating instance (creates default if None)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure
    """
    if gating is None:
        gating = CriticalPeriodGating()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get all domains and their windows
    domains = gating.get_all_domains()
    y_positions = {domain: i for i, domain in enumerate(domains)}

    # Colors for phases
    phase_colors = {
        "early": "#FFCCCB",  # Light red
        "peak": "#90EE90",  # Light green
        "late": "#FFE4B5",  # Light yellow/beige
    }

    max_step = 500000

    for domain in domains:
        start, end = gating.get_optimal_age(domain)
        y = y_positions[domain]

        # Early phase (before window)
        if start > 0:
            ax.barh(
                y,
                start / 1000,
                left=0,
                height=0.8,
                color=phase_colors["early"],
                edgecolor="black",
                linewidth=0.5,
            )

        # Peak phase (during window)
        ax.barh(
            y,
            (end - start) / 1000,
            left=start / 1000,
            height=0.8,
            color=phase_colors["peak"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Late phase (after window)
        if end < max_step:
            ax.barh(
                y,
                (max_step - end) / 1000,
                left=end / 1000,
                height=0.8,
                color=phase_colors["late"],
                edgecolor="black",
                linewidth=0.5,
            )

        # Add domain label
        ax.text(
            -10,
            y,
            domain.replace("_", " ").title(),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        # Add window markers
        ax.text(
            start / 1000,
            y,
            f" {start//1000}k",
            ha="left",
            va="center",
            fontsize=8,
            color="darkgreen",
        )
        ax.text(
            end / 1000,
            y,
            f"{end//1000}k ",
            ha="right",
            va="center",
            fontsize=8,
            color="darkgreen",
        )

    # Styling
    ax.set_xlabel("Training Step (thousands)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Critical Period Timeline: Early → Peak → Late Phases", fontsize=14, fontweight="bold"
    )
    ax.set_yticks([])
    ax.set_xlim(-50, max_step / 1000)
    ax.set_ylim(-0.5, len(domains) - 0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(color=phase_colors["early"], label="Early Phase (0.5x)"),
        mpatches.Patch(color=phase_colors["peak"], label="Peak Phase (1.2-1.25x)"),
        mpatches.Patch(color=phase_colors["late"], label="Late Phase (declining to 0.2-0.3x)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    return fig


def plot_domain_status(
    domain: str,
    training_history: List[Dict],
    gating: Optional[CriticalPeriodGating] = None,
    figsize: tuple = (12, 6),
) -> Figure:
    """Plot detailed status for a specific domain over training.

    Shows multiplier, progress, and phase for one domain.

    Args:
        domain: Domain name (e.g., 'phonology')
        training_history: List of metric dicts from training
        gating: CriticalPeriodGating instance (creates default if None)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure
    """
    if gating is None:
        gating = CriticalPeriodGating()

    # Extract data
    steps = [m.get("global_step", 0) for m in training_history]
    multipliers = [m.get(f"critical_period/{domain}_multiplier", 1.0) for m in training_history]
    progress = [m.get(f"critical_period/{domain}_progress", 0.0) for m in training_history]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Multiplier over time
    ax1.plot(steps, multipliers, linewidth=2, color="#45B7D1", label="Multiplier")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.fill_between(steps, 1.0, multipliers, alpha=0.2, color="#45B7D1")
    ax1.set_ylabel("Learning Rate Multiplier", fontsize=11, fontweight="bold")
    ax1.set_title(
        f'Critical Period Status: {domain.replace("_", " ").title()}',
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right")

    # Mark phases
    start, end = gating.get_optimal_age(domain)
    ax1.axvspan(0, start, alpha=0.1, color="red", label="Early")
    ax1.axvspan(start, end, alpha=0.1, color="green", label="Peak")
    if max(steps) > end:
        ax1.axvspan(end, max(steps), alpha=0.1, color="orange", label="Late")

    # Plot 2: Progress through window
    ax2.plot(steps, progress, linewidth=2, color="#96CEB4", label="Progress")
    ax2.fill_between(steps, 0, progress, alpha=0.2, color="#96CEB4")
    ax2.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Window Progress", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    return fig


def plot_training_metrics_with_critical_periods(
    training_history: List[Dict],
    metrics_to_plot: List[str],
    gating: Optional[CriticalPeriodGating] = None,
    figsize: tuple = (14, 10),
) -> Figure:
    """Plot training metrics alongside critical period status.

    Shows how performance correlates with critical period windows.

    Args:
        training_history: List of metric dicts from training
        metrics_to_plot: List of metric names to plot (e.g., ['accuracy', 'loss'])
        gating: CriticalPeriodGating instance (creates default if None)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure
    """
    if gating is None:
        gating = CriticalPeriodGating()

    # Extract steps
    steps = [m.get("global_step", 0) for m in training_history]

    # Create figure with subplots
    n_metrics = len(metrics_to_plot)
    n_domains = len(gating.get_all_domains())
    fig, axes = plt.subplots(n_metrics + 1, 1, figsize=figsize, sharex=True)

    if n_metrics == 0:
        axes = [axes]

    # Plot metrics
    for i, metric_name in enumerate(metrics_to_plot):
        ax = axes[i]
        values = [m.get(metric_name, 0.0) for m in training_history]
        ax.plot(steps, values, linewidth=2, label=metric_name)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

    # Plot critical period multipliers
    ax = axes[-1]
    for domain in gating.get_all_domains():
        multipliers = [m.get(f"critical_period/{domain}_multiplier", 1.0) for m in training_history]
        ax.plot(
            steps, multipliers, linewidth=1.5, label=domain.replace("_", " ").title(), alpha=0.8
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Training Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("CP Multiplier", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle("Training Metrics with Critical Period Windows", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def save_critical_period_plots(
    output_dir: str,
    training_history: Optional[List[Dict]] = None,
    gating: Optional[CriticalPeriodGating] = None,
):
    """Save all critical period visualization plots.

    Args:
        output_dir: Directory to save plots
        training_history: Optional training history for detailed plots
        gating: CriticalPeriodGating instance (creates default if None)
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if gating is None:
        gating = CriticalPeriodGating()

    # Plot 1: Critical period windows
    fig = plot_critical_period_windows(gating=gating)
    fig.savefig(output_path / "critical_period_windows.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path / 'critical_period_windows.png'}")

    # Plot 2: Timeline
    fig = plot_critical_period_timeline(gating=gating)
    fig.savefig(output_path / "critical_period_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path / 'critical_period_timeline.png'}")

    # Plot 3: Per-domain status (if training history available)
    if training_history:
        for domain in gating.get_all_domains():
            fig = plot_domain_status(domain, training_history, gating)
            fig.savefig(output_path / f"critical_period_{domain}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {output_path / f'critical_period_{domain}.png'}")


__all__ = [
    "plot_critical_period_windows",
    "plot_critical_period_timeline",
    "plot_domain_status",
    "plot_training_metrics_with_critical_periods",
    "save_critical_period_plots",
]
