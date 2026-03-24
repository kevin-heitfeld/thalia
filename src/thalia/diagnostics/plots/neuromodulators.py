"""Neuromodulator concentration plots."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from thalia.diagnostics.diagnostics_report import DiagnosticsReport
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

from ._helpers import PlotConfig, get_cmap


def plot_neuromodulator_conc(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Neuromodulator receptor concentration trajectories over time."""
    if not report.neuromodulator_levels:
        return
    t_ms = report.homeostasis.gain_sample_times_ms
    nm_levels = report.neuromodulator_levels
    n_receptors = len(nm_levels)
    n_col_nm = min(4, n_receptors)
    n_row_nm = (n_receptors + n_col_nm - 1) // n_col_nm
    fig_nm, axes_nm = plt.subplots(
        n_row_nm, n_col_nm,
        figsize=(n_col_nm * cfg.col_width, n_row_nm * cfg.compact_row_height),
        squeeze=False,
        sharex=True,
    )
    for nm_i, (key, traj) in enumerate(sorted(nm_levels.items())):
        ax = axes_nm[nm_i // n_col_nm][nm_i % n_col_nm]
        valid = ~np.isnan(traj)
        if valid.sum() > 1 and len(t_ms) >= valid.sum():
            ax.plot(t_ms[:valid.sum()], traj[valid], linewidth=1.0, color="#9b59b6")
        ax.set_title(key, fontsize=6)
        ax.set_ylabel("Concentration", fontsize=6)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.tick_params(labelsize=5)
    for nm_i in range(n_receptors, n_row_nm * n_col_nm):
        axes_nm[nm_i // n_col_nm][nm_i % n_col_nm].set_visible(False)
    fig_nm.supxlabel("Time (ms)", fontsize=8)
    fig_nm.suptitle("Neuromodulator Receptor Concentrations", fontsize=10)
    plt.tight_layout()
    fig_nm.savefig(os.path.join(output_dir, "11_neuromodulator_conc.png"), dpi=150)
    plt.close(fig_nm)
    print(f"  \u2713 11_neuromodulator_conc.png")


def plot_neuromodulator_conc_by_region(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Neuromodulator receptor concentrations grouped by brain region."""
    if not report.neuromodulator_levels:
        return
    t_ms = report.homeostasis.gain_sample_times_ms
    nm_levels = report.neuromodulator_levels

    by_region: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for key in sorted(nm_levels.keys()):
        rn, mod_name = key.split("/", 1)
        if rn not in by_region:
            by_region[rn] = []
        by_region[rn].append((mod_name, nm_levels[key]))

    if not by_region:
        return

    n_regions = len(by_region)
    n_col = min(3, n_regions)
    n_row = (n_regions + n_col - 1) // n_col
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(n_col * (cfg.col_width + 1.0), n_row * cfg.row_height),
        squeeze=False,
        sharex=True,
    )
    colors = [get_cmap("tab10")(i) for i in range(10)]

    for r_i, (rn, mod_list) in enumerate(sorted(by_region.items())):
        ax = axes[r_i // n_col][r_i % n_col]
        for m_i, (mod_name, traj) in enumerate(mod_list):
            valid = ~np.isnan(traj)
            if valid.sum() > 1 and len(t_ms) >= valid.sum():
                ax.plot(
                    t_ms[: valid.sum()], traj[valid],
                    linewidth=1.2, color=colors[m_i % 10], label=mod_name,
                )
        ax.set_title(rn, fontsize=8)
        ax.set_ylabel("Concentration", fontsize=6)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=5, loc="upper right")
        ax.tick_params(labelsize=5)

    for r_i in range(n_regions, n_row * n_col):
        axes[r_i // n_col][r_i % n_col].set_visible(False)

    fig.supxlabel("Time (ms)", fontsize=8)
    fig.suptitle("Neuromodulator Concentrations by Region", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "14_neuromodulator_conc_by_region.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 14_neuromodulator_conc_by_region.png")
