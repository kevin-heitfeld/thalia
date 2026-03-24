"""Learning / training plots — weights, eligibility traces, BCM thresholds."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from thalia.diagnostics.diagnostics_report import DiagnosticsReport
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

from ._helpers import PlotConfig, get_cmap


def plot_weight_distributions(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Weight mean \u00b1 std and sparsity over time per synapse group."""
    if report.learning is None or not report.learning.weight_trajectories:
        return

    keys = sorted(report.learning.weight_trajectories.keys())
    n = len(keys)
    if n == 0:
        return

    hs = report.homeostasis
    t_ms = hs.gain_sample_times_ms
    if len(t_ms) < 2:
        return

    fig, axes = plt.subplots(n, 1, figsize=(cfg.timeline_width, max(3.0 * n, cfg.timeline_height)),
                             sharex=True, squeeze=False)
    cmap = get_cmap("tab10")

    for i, key in enumerate(keys):
        ax = axes[i, 0]
        traj = report.learning.weight_trajectories[key]
        n_steps = min(len(t_ms), traj.shape[0])
        t = t_ms[:n_steps]
        means = traj[:n_steps, 0]
        stds = traj[:n_steps, 1]
        valid = ~np.isnan(means)
        if valid.sum() < 2:
            ax.set_title(key, fontsize=7)
            continue

        color = cmap(i % 10)
        ax.plot(t[valid], means[valid], color=color, linewidth=1.0, label="mean")
        ax.fill_between(
            t[valid],
            (means - stds)[valid],
            (means + stds)[valid],
            alpha=0.2, color=color,
        )
        ax.set_ylabel("Weight", fontsize=7)
        ax.set_title(key, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right")

    axes[-1, 0].set_xlabel("Time (ms)", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "19_weight_distributions.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 19_weight_distributions.png")


def plot_eligibility_traces(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Mean |eligibility| and LTP/LTD ratio over time per synapse group."""
    if report.learning is None or not report.learning.eligibility_trajectories:
        return

    keys = sorted(report.learning.eligibility_trajectories.keys())
    if not keys:
        return

    hs = report.homeostasis
    t_ms = hs.gain_sample_times_ms
    if len(t_ms) < 2:
        return

    fig, (ax_elig, ax_ratio) = plt.subplots(2, 1, figsize=(cfg.timeline_width, cfg.timeline_height),
                                              sharex=True)
    cmap = get_cmap("tab10")

    for i, key in enumerate(keys):
        elig = report.learning.eligibility_trajectories[key]
        n_steps = min(len(t_ms), len(elig))
        t = t_ms[:n_steps]
        valid = ~np.isnan(elig[:n_steps])
        if valid.sum() < 2:
            continue
        color = cmap(i % 10)
        ax_elig.plot(t[valid], elig[:n_steps][valid], color=color, linewidth=0.8,
                     alpha=0.8, label=key)

    ax_elig.set_ylabel("Mean |eligibility|", fontsize=7)
    ax_elig.set_title("Eligibility Traces Over Time", fontsize=9)
    ax_elig.legend(fontsize=5, ncol=2, loc="upper right")
    ax_elig.tick_params(labelsize=6)

    # LTP/LTD ratio on log scale
    if rec._eligibility_ltp_ltd_ratio_history is not None:
        for i, key in enumerate(keys):
            if key not in report.learning.update_magnitude_trajectories:
                continue
            idx = rec._learning_keys.index(key) if key in rec._learning_keys else -1
            if idx < 0:
                continue
            ratio = rec._eligibility_ltp_ltd_ratio_history[:rec._gain_sample_step, idx]
            n_steps = min(len(t_ms), len(ratio))
            t = t_ms[:n_steps]
            valid = ~np.isnan(ratio[:n_steps]) & (ratio[:n_steps] > 0)
            if valid.sum() < 2:
                continue
            color = cmap(i % 10)
            ax_ratio.semilogy(t[valid], ratio[:n_steps][valid], color=color, linewidth=0.8,
                              alpha=0.8, label=key)

    ax_ratio.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax_ratio.set_ylabel("LTP/LTD ratio", fontsize=7)
    ax_ratio.set_xlabel("Time (ms)", fontsize=7)
    ax_ratio.legend(fontsize=5, ncol=2, loc="upper right")
    ax_ratio.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "20_eligibility_traces.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 20_eligibility_traces.png")


def plot_bcm_thresholds(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """BCM sliding threshold (\u03b8) trajectories for BCM synapses."""
    if report.learning is None or not report.learning.bcm_theta_trajectories:
        return

    keys = sorted(report.learning.bcm_theta_trajectories.keys())
    if not keys:
        return

    hs = report.homeostasis
    t_ms = hs.gain_sample_times_ms
    if len(t_ms) < 2:
        return

    fig, ax = plt.subplots(figsize=(cfg.timeline_width, cfg.timeline_height))
    cmap = get_cmap("tab10")

    for i, key in enumerate(keys):
        traj = report.learning.bcm_theta_trajectories[key]
        n_steps = min(len(t_ms), len(traj))
        t = t_ms[:n_steps]
        valid = ~np.isnan(traj[:n_steps])
        if valid.sum() < 2:
            continue
        color = cmap(i % 10)
        ax.plot(t[valid], traj[:n_steps][valid], color=color, linewidth=1.0,
                alpha=0.8, label=key)

    ax.set_xlabel("Time (ms)", fontsize=7)
    ax.set_ylabel("BCM \u03b8", fontsize=7)
    ax.set_title("BCM Sliding Threshold Trajectories", fontsize=9)
    ax.legend(fontsize=5, ncol=2, loc="upper right")
    ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "21_bcm_thresholds.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 21_bcm_thresholds.png")
