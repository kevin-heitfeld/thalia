"""Oscillatory analysis orchestrator â€” delegates to analysis_spectral and analysis_neural_coupling."""

from __future__ import annotations

import numpy as np

from .analysis_spectral import (
    compute_band_powers,
    compute_coherence_matrices,
    compute_integration_tau,
    compute_pac_per_region,
)
from .analysis_neural_coupling import (
    compute_beta_burst_stats,
    compute_ca3_ca1_theta_sequence,
    compute_cerebellar_metrics,
    compute_hfo_per_region,
    compute_laminar_cascade,
    compute_plv_theta_per_region,
    compute_relay_burst_mode,
    compute_spike_avalanches,
    compute_swr_ca3_ca1_coupling,
)
from .diagnostics_types import OscillatoryStats, RecorderSnapshot


def compute_oscillatory_stats(
    rec: RecorderSnapshot,
    pop_rate_binned: np.ndarray,
    region_rate_binned: np.ndarray,
    n_bins: int,
    T: int,
) -> OscillatoryStats:
    """Orchestrate all oscillatory analyses and assemble OscillatoryStats."""
    (
        region_band_power,
        region_dominant_freq,
        region_dominant_band,
        global_band_power,
        global_dominant_freq,
    ) = compute_band_powers(rec, region_rate_binned, n_bins)

    coh_theta, coh_beta, coh_gamma, freq_resolution_hz = compute_coherence_matrices(
        rec, region_rate_binned, n_bins, T
    )
    pac_modulation_index = compute_pac_per_region(rec, T)
    hfo_band_power = compute_hfo_per_region(rec, T)
    plv_theta, plv_theta_used_fallback = compute_plv_theta_per_region(rec, T)
    avalanche_exponent, avalanche_r2, avalanche_branching_ratio = compute_spike_avalanches(rec, T)
    purkinje_dcn_corr, io_pairwise_corr = compute_cerebellar_metrics(rec, T)
    beta_burst_stats = compute_beta_burst_stats(rec, region_rate_binned, n_bins)
    relay_burst_mode = compute_relay_burst_mode(rec, T)
    swr_ca3_ca1_coupling = compute_swr_ca3_ca1_coupling(rec, T)
    ca3_ca1_theta_sequence = compute_ca3_ca1_theta_sequence(rec, T)
    region_integration_tau_ms = compute_integration_tau(rec, region_rate_binned, n_bins)
    laminar_cascade_latencies = compute_laminar_cascade(rec, T)

    return OscillatoryStats(
        region_band_power=region_band_power,
        region_dominant_freq=region_dominant_freq,
        region_dominant_band=region_dominant_band,
        coherence_theta=coh_theta,
        coherence_beta=coh_beta,
        coherence_gamma=coh_gamma,
        region_order=list(rec._region_keys),
        global_dominant_freq_hz=global_dominant_freq,
        global_band_power=global_band_power,
        freq_resolution_hz=freq_resolution_hz,
        pac_modulation_index=pac_modulation_index,
        hfo_band_power=hfo_band_power,
        plv_theta=plv_theta,
        plv_theta_used_fallback=plv_theta_used_fallback,
        avalanche_exponent=avalanche_exponent,
        avalanche_r2=avalanche_r2,
        avalanche_branching_ratio=avalanche_branching_ratio,
        purkinje_dcn_corr=purkinje_dcn_corr,
        io_pairwise_corr=io_pairwise_corr,
        beta_burst_stats=beta_burst_stats,
        relay_burst_mode=relay_burst_mode,
        swr_ca3_ca1_coupling=swr_ca3_ca1_coupling,
        ca3_ca1_theta_sequence=ca3_ca1_theta_sequence,
        region_integration_tau_ms=region_integration_tau_ms,
        laminar_cascade_latencies=laminar_cascade_latencies,
    )
