"""Spectral analysis plots: power spectra, coherence matrices, spectrograms."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as sp_spectrogram

from thalia.diagnostics.diagnostics_report import DiagnosticsReport
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

from ._helpers import PlotConfig


def plot_region_spectra(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Per-region normalised band-power bar charts."""
    osc = report.oscillations
    n_plots = len(report.region_keys or [])
    if n_plots == 0:
        return
    ncols = 4
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cfg.col_width, nrows * cfg.compact_row_height))
    axes = np.array(axes).flatten()
    colors_bar = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    for ax_i, rn in enumerate(report.region_keys or []):
        ax = axes[ax_i]
        bp = osc.region_band_power.get(rn, {})
        bands = list(bp.keys())
        powers = [bp[b] for b in bands]
        ax.bar(bands, powers, color=colors_bar[: len(bands)])
        ax.set_title(rn, fontsize=7, pad=2)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)
        dom_freq = osc.region_dominant_freq.get(rn, 0.0)
        if dom_freq > 0:
            ax.text(0.98, 0.95, f"{dom_freq:.1f} Hz", ha="right", va="top",
                    transform=ax.transAxes, fontsize=7)
    for ax_i in range(n_plots, len(axes)):
        axes[ax_i].set_visible(False)
    fig.suptitle("Per-Region Power Spectra", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_region_spectra.png"), dpi=150)
    plt.close(fig)
    print(f"  \u2713 03_region_spectra.png")


def plot_coherence(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Cross-regional coherence matrices (theta, beta, gamma)."""
    osc = report.oscillations
    n_r = len(osc.region_order)
    for band_label, coh_mat, fname in [
        ("Theta (4\u20138 Hz)",   osc.coherence_theta, "04_coherence_theta.png"),
        ("Beta (13\u201330 Hz)",  osc.coherence_beta,  "04b_coherence_beta.png"),
        ("Gamma (30\u2013100 Hz)", osc.coherence_gamma, "04c_coherence_gamma.png"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, n_r * 0.5), min(cfg.max_fig_height, max(6, n_r * 0.5))))
        cmap = plt.cm.YlOrRd.copy()  # type: ignore[attr-defined]
        cmap.set_bad(color="lightgrey")
        im = ax.imshow(coh_mat, vmin=0, vmax=1, cmap=cmap, aspect="equal")
        ax.set_xticks(np.arange(n_r))
        ax.set_xticklabels(osc.region_order, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(n_r))
        ax.set_yticklabels(osc.region_order, fontsize=6)
        ax.set_title(f"Cross-Regional Coherence \u2014 {band_label}")
        plt.colorbar(im, ax=ax, label="Coherence")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  \u2713 {fname}")


def plot_spectrograms(
    rec: RecorderSnapshot, report: DiagnosticsReport, output_dir: str, cfg: PlotConfig,
) -> None:
    """Time-frequency spectrograms of population rates per region."""
    n_steps = rec._n_recorded or rec.config.n_timesteps
    if n_steps <= 0 or rec._n_regions <= 0:
        return
    bin_steps = max(1, int(rec.config.rate_bin_ms / rec.dt_ms))
    n_bins = n_steps // bin_steps
    fs_bins = 1000.0 / rec.config.rate_bin_ms

    if report.region_rate_binned is not None and report.region_rate_binned.shape == (n_bins, rec._n_regions):
        region_rate_binned = report.region_rate_binned
    else:
        region_rate_binned = np.zeros((n_bins, rec._n_regions), dtype=np.float32)
        for b in range(n_bins):
            start, end = b * bin_steps, (b + 1) * bin_steps
            region_rate_binned[b] = rec._region_spike_counts[start:end].sum(axis=0)

    n_col = min(4, rec._n_regions)
    n_row = (rec._n_regions + n_col - 1) // n_col
    fig_sg, axes_sg = plt.subplots(n_row, n_col, figsize=(n_col * cfg.col_width, n_row * cfg.row_height), squeeze=False, sharex=True)
    nperseg_sg = min(n_bins, 128)
    for r_idx, rn in enumerate(rec._region_keys):
        ax = axes_sg[r_idx // n_col][r_idx % n_col]
        trace = region_rate_binned[:, r_idx].astype(float)
        if trace.std() > 0 and nperseg_sg >= 8:
            f_sg, t_sg, Sxx = sp_spectrogram(
                trace, fs=fs_bins, nperseg=nperseg_sg,
                noverlap=nperseg_sg // 2, window="hann", scaling="density"
            )
            f_mask = f_sg <= 100.0
            ax.pcolormesh(
                t_sg, f_sg[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-12),
                shading="gouraud", cmap="magma",
            )
        ax.set_title(rn, fontsize=7)
        ax.set_ylabel("Freq (Hz)", fontsize=6)
        ax.tick_params(labelsize=5)
    for r_idx in range(rec._n_regions, n_row * n_col):
        axes_sg[r_idx // n_col][r_idx % n_col].set_visible(False)
    fig_sg.supxlabel("Time (s)", fontsize=8)
    fig_sg.suptitle("Population Rate Spectrograms", fontsize=10)
    plt.tight_layout()
    fig_sg.savefig(os.path.join(output_dir, "10_spectrograms.png"), dpi=130)
    plt.close(fig_sg)
    print(f"  \u2713 10_spectrograms.png")
