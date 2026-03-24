"""Temporal / voltage / conductance plots."""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from thalia.diagnostics.diagnostics_report import DiagnosticsReport
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

from ._helpers import PlotConfig, get_cmap, rank_populations_by_health


def plot_voltage_traces(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Sample neuron voltage traces for key populations."""
    if report.raw_voltages is None:
        return
    ranked = rank_populations_by_health(report)
    sampled: List[Tuple[str, str, int]] = []
    for rn, pn in ranked:
        if len(sampled) >= 6:
            break
        if (rn, pn) not in rec._pop_index:
            continue
        idx = rec._pop_index[(rn, pn)]
        if not np.all(np.isnan(report.raw_voltages[:, idx, :])):
            sampled.append((rn, pn, idx))
    if not sampled:
        return
    fig, axes = plt.subplots(len(sampled), 1, figsize=(cfg.timeline_width + 2, min(cfg.max_fig_height, len(sampled) * cfg.trace_row_height)), sharex=True)
    if len(sampled) == 1:
        axes = [axes]
    t_ms = np.arange(report.n_timesteps, dtype=np.float32) * rec.dt_ms
    for ax, (rn, pn, pidx) in zip(axes, sampled):
        v = report.raw_voltages[:, pidx, :]
        n_valid = int(rec._pop_sizes[pidx])
        n_show = min(8, n_valid, v.shape[1])
        for ni in range(n_show):
            col = get_cmap("tab10")(ni)
            y = v[:, ni]
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() > 1:
                ax.plot(t_ms[valid_mask], y[valid_mask], linewidth=0.5, alpha=0.7, color=col)
        ax.set_ylabel(f"{pn}\n({rn})", fontsize=6)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Sample Neuron Voltage Traces", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_voltage_traces.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 07_voltage_traces.png")


def plot_ei_phase_portrait(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """E/I conductance phase portraits for key populations."""
    if rec._g_exc_samples is None or rec._g_inh_samples is None:
        return
    if np.all(np.isnan(rec._g_exc_samples)) or np.all(np.isnan(rec._g_inh_samples)):
        return
    pop_keys = rec._pop_keys
    key_ei_pops = ["relay", "ca1", "l23_pyr", "l4_pyr"]
    sampled_ei: List[Tuple[str, str, int]] = []
    for rn, pn in pop_keys:
        if any(t in pn for t in key_ei_pops) and len(sampled_ei) < 6:
            idx = rec._pop_index[(rn, pn)]
            sampled_ei.append((rn, pn, idx))
    if not sampled_ei:
        return
    fig, axes = plt.subplots(1, len(sampled_ei), figsize=(len(sampled_ei) * cfg.row_height, cfg.row_height))
    if len(sampled_ei) == 1:
        axes = [axes]
    n_cond = rec._cond_sample_step
    for ax, (rn, pn, pidx) in zip(axes, sampled_ei):
        g_e = rec._g_exc_samples[:n_cond, pidx, :].flatten()
        g_i = rec._g_inh_samples[:n_cond, pidx, :].flatten()
        valid = ~(np.isnan(g_e) | np.isnan(g_i))
        if valid.sum() > 5:
            ax.scatter(g_e[valid], g_i[valid], s=2, alpha=0.4, c="#3498db")
        ax.set_xlabel("g_exc", fontsize=7)
        ax.set_ylabel("g_inh", fontsize=7)
        ax.set_title(f"{pn}\n({rn})", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("E/I Conductance Phase Portraits", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "09_ei_phase_portrait.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 09_ei_phase_portrait.png")


def plot_voltage_distributions(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Per-population voltage distributions as violin plots."""
    if report.raw_voltages is None:
        return
    pops_with_data: List[Tuple[str, str, int]] = []
    for pidx, (rn, pn) in enumerate(rec._pop_keys):
        if not np.all(np.isnan(report.raw_voltages[:, pidx, :])):
            pops_with_data.append((rn, pn, pidx))
    if not pops_with_data:
        return
    n = len(pops_with_data)
    n_col = min(4, n)
    n_row = (n + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * cfg.col_width, n_row * cfg.row_height), squeeze=False)
    for i, (rn, pn, pidx) in enumerate(pops_with_data):
        ax = axes[i // n_col][i % n_col]
        v_flat = report.raw_voltages[:, pidx, :].flatten()
        v_flat = v_flat[~np.isnan(v_flat)]
        if len(v_flat) < 10:
            ax.set_visible(False)
            continue
        vp = ax.violinplot(v_flat, showmedians=True, showextrema=False)
        for body in vp.get("bodies", []):
            body.set_facecolor("#3498db")
            body.set_alpha(0.7)
        rs = report.regions.get(rn)
        pop_stats = rs.populations.get(pn) if rs else None
        bc = (
            pop_stats.voltage_bimodality
            if pop_stats is not None and not np.isnan(pop_stats.voltage_bimodality)
            else None
        )
        subtitle = f"({rn})"
        if bc is not None:
            subtitle += f"  BC={bc:.3f}"
        ax.set_title(f"{pn}\n{subtitle}", fontsize=6)
        ax.set_xticks([])
        ax.set_ylabel("Voltage (mV)", fontsize=6)
        ax.tick_params(labelsize=6)
    for i in range(len(pops_with_data), n_row * n_col):
        axes[i // n_col][i % n_col].set_visible(False)
    fig.suptitle("Population Voltage Distributions", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "13_voltage_distributions.png"), dpi=130)
    plt.close(fig)
    print(f"  \u2713 13_voltage_distributions.png")


def plot_ei_lag(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """E/I conductance lag per region as a bar chart."""
    regions: list[tuple[str, float, float]] = []
    for rn, rs in report.regions.items():
        if not np.isnan(rs.ei_lag_ms) and not np.isnan(rs.ei_xcorr_peak):
            regions.append((rn, rs.ei_lag_ms, rs.ei_xcorr_peak))
    if not regions:
        return
    regions.sort(key=lambda x: x[0])
    names = [r[0] for r in regions]
    lags = [r[1] for r in regions]
    xcorrs = [r[2] for r in regions]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cfg.timeline_width, cfg.timeline_height))
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, lags, color="steelblue", edgecolor="black", linewidth=0.3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("E\u2192I lag (ms)", fontsize=8)
    ax1.set_title("Inhibition Lag After Excitation", fontsize=9)
    ax1.tick_params(labelsize=6)
    ax2.barh(y_pos, xcorrs, color="coral", edgecolor="black", linewidth=0.3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=6)
    ax2.set_xlabel("Peak cross-correlation", fontsize=8)
    ax2.set_title("E/I Cross-Correlation Magnitude", fontsize=9)
    ax2.tick_params(labelsize=6)
    fig.suptitle("E/I Lag Cross-Correlation", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "17_ei_lag.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 17_ei_lag.png")


def plot_effective_synaptic_gain(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Effective synaptic gain per inter-region tract."""
    gains = report.effective_synaptic_gain
    if not gains:
        return
    items = sorted(gains.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig_height = max(cfg.timeline_height, len(labels) * 0.3)
    fig, ax = plt.subplots(figsize=(cfg.timeline_width, fig_height))
    y_pos = np.arange(len(labels))
    colors = ["coral" if v > 0.60 else "steelblue" if v >= 0.05 else "gray" for v in values]
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Effective gain (Pearson r at causal lag)", fontsize=8)
    ax.set_title("Effective Synaptic Gain per Tract", fontsize=9)
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.axvline(0.60, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "18_effective_synaptic_gain.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 18_effective_synaptic_gain.png")
