"""Homeostatic gain, STP efficacy, and apical/basal conductance plots."""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from thalia.diagnostics.diagnostics_report import DiagnosticsReport
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

from ._helpers import PlotConfig, get_cmap


def plot_homeostatic_gains(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Homeostatic gain trajectories over time."""
    hs = report.homeostasis
    if not hs.gain_trajectories or len(hs.gain_sample_times_ms) <= 1:
        return
    t_ms = hs.gain_sample_times_ms
    fig, ax = plt.subplots(figsize=(cfg.timeline_width, cfg.timeline_height))
    cmap = get_cmap("tab20")
    for i, (key, traj) in enumerate(sorted(hs.gain_trajectories.items())):
        valid = ~np.isnan(traj)
        if valid.sum() < 2:
            continue
        ax.plot(t_ms[valid], traj[valid], linewidth=0.8, alpha=0.7,
                color=cmap(i % 20), label=key)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=1.0, label="Collapse threshold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("g_L_scale")
    ax.set_title("Homeostatic Gain Trajectories")
    ax.set_ylim(0, 2.1)
    ax.legend(fontsize=5, ncol=3, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_homeostatic_gains.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 06_homeostatic_gains.png")


def plot_stp_efficacy(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """STP efficacy (x\u00b7u) trajectories per synapse over time."""
    hs_stp = report.homeostasis.stp_efficacy_history
    if not hs_stp or len(report.homeostasis.gain_sample_times_ms) <= 1:
        return
    t_ms = report.homeostasis.gain_sample_times_ms
    fig, ax = plt.subplots(figsize=(cfg.timeline_width, cfg.timeline_height))
    cmap_stp = get_cmap("tab20")
    for i, (key, traj) in enumerate(sorted(hs_stp.items())):
        valid = ~np.isnan(traj)
        if valid.sum() < 2:
            continue
        ax.plot(t_ms[valid], traj[valid], linewidth=0.8, alpha=0.7,
                color=cmap_stp(i % 20), label=key)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("STP efficacy (x\u00b7u)")
    ax.set_title("STP Efficacy Trajectories (per synapse)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5, ncol=3, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "06b_stp_efficacy.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 06b_stp_efficacy.png")


def plot_apical_basal_conductance(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Apical vs. basal AMPA conductance per population (TwoCompartmentLIF only)."""
    if (
        rec._g_apical_samples is None
        or rec._g_exc_samples is None
        or rec._cond_sample_step == 0
    ):
        return
    n_cond = rec._cond_sample_step

    labels: List[str] = []
    basal_means: List[float] = []
    apical_means: List[float] = []
    for pidx, (rn, pn) in enumerate(rec._pop_keys):
        apical_slab = rec._g_apical_samples[:n_cond, pidx, :]
        apical_mean = float(np.nanmean(apical_slab))
        if np.isnan(apical_mean):
            continue
        basal_slab = rec._g_exc_samples[:n_cond, pidx, :]
        basal_mean = float(np.nanmean(basal_slab))
        if np.isnan(basal_mean):
            basal_mean = 0.0
        labels.append(f"{pn}\n({rn})")
        basal_means.append(basal_mean)
        apical_means.append(apical_mean)

    if not labels:
        return

    n = len(labels)
    y = np.arange(n)
    bar_h = 0.35
    fig, ax = plt.subplots(figsize=(10, min(cfg.max_fig_height, max(4, n * 0.55))))
    ax.barh(y - bar_h / 2, basal_means, bar_h, label="Basal (g_exc AMPA)", color="#3498db", alpha=0.85)
    ax.barh(y + bar_h / 2, apical_means, bar_h, label="Apical (g_exc_apical)", color="#e67e22", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean conductance")
    ax.set_title("Apical vs. Basal AMPA Conductance per Population")
    ax.legend(fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "12_apical_basal_conductance.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 12_apical_basal_conductance.png")
