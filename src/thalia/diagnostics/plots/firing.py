"""Firing-rate, ISI, raster, Fano, and correlation distribution plots."""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..diagnostics_report import DiagnosticsReport
from ..diagnostics_snapshot import RecorderSnapshot

from ._helpers import (
    PlotConfig,
    get_cmap,
    pop_color,
    rank_populations_by_health,
    rank_regions_by_health,
    unhealthiness,
)


def plot_firing_rates(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Horizontal bar chart of mean firing rates per population."""
    fig, ax = plt.subplots(figsize=(cfg.timeline_width, min(cfg.max_fig_height, max(6, rec._n_pops * cfg.height_per_pop))))
    labels, values, colors = [], [], []
    for rn, rs in report.regions.items():
        for pn, ps in rs.populations.items():
            labels.append(f"{rn}:{pn}")
            values.append(ps.mean_fr_hz)
            colors.append(pop_color(ps))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_title("Population Firing Rates  (green=bio-ok, orange=out-of-range, red=silent)")
    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_firing_rates.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 01_firing_rates.png")


def plot_population_heatmap(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Population firing-rate heatmap over time."""
    if report.pop_rate_binned is None:
        return
    pop_keys = rec._pop_keys
    pr = report.pop_rate_binned
    pr_hz = pr * (1000.0 / rec.config.rate_bin_ms)
    t_axis = np.arange(pr_hz.shape[0]) * rec.config.rate_bin_ms
    fig, ax = plt.subplots(figsize=(cfg.timeline_width + 2, min(cfg.max_fig_height, max(6, rec._n_pops * cfg.height_per_pop * 0.9))))
    vmax = float(np.percentile(pr_hz, 99)) or 1.0
    im = ax.imshow(
        pr_hz.T, aspect="auto", origin="lower", interpolation="none",
        extent=[t_axis[0], t_axis[-1], 0, rec._n_pops], cmap="hot", vmin=0, vmax=vmax,
    )
    ax.set_yticks(np.arange(rec._n_pops) + 0.5)
    ax.set_yticklabels([f"{rn}:{pn}" for rn, pn in pop_keys], fontsize=5)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Population Firing Rate Heatmap (Hz)")
    plt.colorbar(im, ax=ax, label="Firing rate (Hz)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_population_heatmap.png"), dpi=120)
    plt.close(fig)
    print(f"  \u2713 02_population_heatmap.png")


def plot_isi_distributions(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """ISI histograms for a selection of key populations."""
    ranked = rank_populations_by_health(report)
    key_pops = [(rn, pn) for rn, pn in ranked if (rn, pn) in rec._spike_times][:12]
    if not key_pops:
        return
    ncols = min(4, len(key_pops))
    nrows = (len(key_pops) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cfg.col_width, nrows * cfg.row_height))
    axes = np.array(axes).flatten()
    for ax_i, key in enumerate(key_pops):
        ax = axes[ax_i]
        all_isis: List[float] = []
        for times in rec._spike_times[key]:
            if len(times) >= 2:
                all_isis.extend(np.diff(times))
        if all_isis:
            isis_ms = np.array(all_isis, dtype=np.float32) * rec.dt_ms
            ax.hist(
                isis_ms, bins=50, range=(0, float(np.percentile(isis_ms, 98))),
                color="#3498db", edgecolor="none", density=True
            )
            cv = np.std(isis_ms) / np.mean(isis_ms) if np.mean(isis_ms) > 0 else 0
            ax.set_title(f"{key[0]}:{key[1]}\nCV={cv:.2f}", fontsize=7)
            ax.set_xlabel("ISI (ms)", fontsize=7)
            ax.tick_params(labelsize=6)
    for ax_i in range(len(key_pops), len(axes)):
        axes[ax_i].set_visible(False)
    fig.suptitle("Inter-Spike Interval Distributions", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_isi_distributions.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 05_isi_distributions.png")


def plot_spike_raster(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Spike raster plots for key regions."""
    if not rec._spike_times:
        return
    ranked_regions = rank_regions_by_health(report)
    existing = [rn for rn in ranked_regions if rn in rec._region_pop_indices][:6]
    if not existing:
        return
    fig, axes = plt.subplots(
        len(existing), 1,
        figsize=(cfg.timeline_width + 2, min(cfg.max_fig_height, len(existing) * cfg.compact_row_height)),
        sharex=True,
    )
    if len(existing) == 1:
        axes = [axes]
    t_ms_max = report.n_timesteps * rec.dt_ms
    for ax, rn in zip(axes, existing):
        y_offset = 0
        pop_names = [rec._pop_keys[i][1] for i in rec._region_pop_indices[rn]]
        pop_colors_raster = get_cmap("tab10")
        for pi, pn in enumerate(pop_names):
            key = (rn, pn)
            if key not in rec._spike_times:
                continue
            n_neurons = len(rec._spike_times[key])
            n_show = min(80, n_neurons)
            col = pop_colors_raster(pi % 10)
            for ni in range(n_show):
                times = rec._spike_times[key][ni]
                if times:
                    t_vals = np.array(times, dtype=np.float32) * rec.dt_ms
                    ax.scatter(
                        t_vals, np.full_like(t_vals, y_offset + ni),
                        s=1.0, c=[col], marker="|", linewidths=0.5
                    )
            ax.axhline(y_offset + n_show, color="gray", linewidth=0.3)
            ax.text(t_ms_max * 1.002, y_offset + n_show / 2, pn, fontsize=6, va="center")
            y_offset += n_show + 5
        ax.set_ylabel(rn, fontsize=8)
        ax.set_xlim(0, t_ms_max)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Spike Rasters (sample neurons)", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "08_raster_plots.png"), dpi=130)
    plt.close(fig)
    print(f"  \u2713 08_raster_plots.png")


def plot_fano_scaling(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Fano factor vs bin width for top populations."""
    entries: list[tuple[str, str, list[tuple[float, float]]]] = []
    for rs in report.regions.values():
        for ps in rs.populations.values():
            if ps.fano_scaling:
                entries.append((ps.region_name, ps.population_name, ps.fano_scaling))
    if not entries:
        return
    entries.sort(
        key=lambda e: (-unhealthiness(report.regions[e[0]].populations[e[1]]),
                       -report.regions[e[0]].populations[e[1]].total_spikes),
    )
    entries = entries[:12]
    n_cols = min(4, len(entries))
    n_rows = (len(entries) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cfg.col_width * n_cols, cfg.row_height * n_rows), squeeze=False)
    for i, (rn, pn, scaling) in enumerate(entries):
        ax = axes[i // n_cols][i % n_cols]
        bin_widths = [s[0] for s in scaling]
        ff_values = [s[1] for s in scaling]
        ax.plot(bin_widths, ff_values, "o-", markersize=4, linewidth=1.5)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Poisson")
        ax.set_xscale("log")
        ax.set_xlabel("Bin width (ms)", fontsize=7)
        ax.set_ylabel("Fano factor", fontsize=7)
        ax.set_title(f"{rn}:{pn}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5)
    for i in range(len(entries), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)
    fig.suptitle("Fano Factor Scaling (FF vs Bin Width)", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "15_fano_scaling.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 15_fano_scaling.png")


def plot_correlation_distribution(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Histogram of pairwise correlation coefficients."""
    entries: list[tuple[str, str, np.ndarray]] = []
    for rs in report.regions.values():
        for ps in rs.populations.values():
            if ps.pairwise_correlation_distribution is not None and len(ps.pairwise_correlation_distribution) > 0:
                entries.append((ps.region_name, ps.population_name, ps.pairwise_correlation_distribution))
    if not entries:
        return
    entries.sort(
        key=lambda e: (-unhealthiness(report.regions[e[0]].populations[e[1]]),
                       -report.regions[e[0]].populations[e[1]].total_spikes),
    )
    entries = entries[:12]
    n_cols = min(4, len(entries))
    n_rows = (len(entries) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cfg.col_width * n_cols, cfg.row_height * n_rows), squeeze=False)
    for i, (rn, pn, r_values) in enumerate(entries):
        ax = axes[i // n_cols][i % n_cols]
        ax.hist(r_values, bins=30, range=(-1.0, 1.0), color="steelblue", edgecolor="black", linewidth=0.3)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
        mean_r = float(np.mean(r_values))
        ax.axvline(mean_r, color="red", linestyle="-", linewidth=1.0, label=f"mean={mean_r:.3f}")
        ax.set_xlabel("Pearson r", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.set_title(f"{rn}:{pn}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5)
    for i in range(len(entries), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)
    fig.suptitle("Pairwise Correlation Distribution", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "16_correlation_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 16_correlation_distribution.png")
