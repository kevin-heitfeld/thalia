"""Diagnostics analysis orchestrator — thin entry point that delegates to sub-modules."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

from .diagnostics_types import (
    DiagnosticsReport,
    PopulationStats,
    RegionStats,
)
from .analysis_connectivity import compute_connectivity_stats
from .analysis_health import assess_health
from .analysis_homeostasis import compute_homeostatic_stats
from .analysis_oscillations import compute_oscillatory_stats
from .analysis_population import compute_population_stats
from .analysis_region import compute_region_stats

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


def analyze(rec: "DiagnosticsRecorder") -> DiagnosticsReport:
    """Compute and return a complete :class:`DiagnosticsReport`.

    Call this via ``DiagnosticsRecorder.analyze()``.
    """
    T = rec._n_recorded or rec.config.n_timesteps

    # Population-level stats
    pop_stats: Dict[Tuple[str, str], PopulationStats] = {}
    for idx, (rn, pn) in enumerate(rec._pop_keys):
        pop_stats[(rn, pn)] = compute_population_stats(rec, idx, T)

    # Region-level stats
    region_stats: Dict[str, RegionStats] = {}
    for rn in rec._region_keys:
        region_stats[rn] = compute_region_stats(rec, rn, pop_stats)

    # Binned population rates [n_bins, n_pops]: spikes / neuron per bin.
    # Vectorized: reshape [n_bins*bin_steps, n_pops] → [n_bins, bin_steps, n_pops], sum time axis.
    bin_steps = max(1, int(rec.config.rate_bin_ms / rec.dt_ms))
    n_bins = T // bin_steps
    if n_bins > 0:
        pop_rate_binned = (
            rec._pop_spike_counts[: n_bins * bin_steps]
            .reshape(n_bins, bin_steps, rec._n_pops)
            .sum(axis=1)
            .astype(np.float32)
            / np.maximum(rec._pop_sizes, 1).astype(np.float32)
        )
    else:
        pop_rate_binned = np.zeros((0, rec._n_pops), dtype=np.float32)

    # Region binned rates derived from pop_rate_binned for strict consistency.
    # region_rate[b, r] = Σ_p (pop_rate[b,p] * pop_size[p]) / Σ_p pop_size[p]
    # Weights are sourced from pop_stats[...].n_neurons — the same field used by
    # _weighted_mean_fr() — so the time-mean of region_rate_binned[:,r] is
    # guaranteed to equal region_stats[rn].mean_fr_hz by construction.
    region_rate_binned = np.zeros((n_bins, rec._n_regions), dtype=np.float32)
    if n_bins > 0:
        for r_idx, rn in enumerate(rec._region_keys):
            p_indices = rec._region_pop_indices[rn]
            pop_keys_r = [rec._pop_keys[i] for i in p_indices]
            pop_weights_r = np.array([pop_stats[k].n_neurons for k in pop_keys_r], dtype=np.float32)
            total_neurons = float(pop_weights_r.sum())
            if total_neurons > 0:
                region_rate_binned[:, r_idx] = (
                    (pop_rate_binned[:, p_indices] * pop_weights_r).sum(axis=1)
                    / total_neurons
                )

    oscillations = compute_oscillatory_stats(rec, pop_rate_binned, region_rate_binned, n_bins, T)
    connectivity = compute_connectivity_stats(rec, T)
    homeostasis = compute_homeostatic_stats(rec)
    health = assess_health(rec, pop_stats, region_stats, connectivity, oscillations, homeostasis)

    report = DiagnosticsReport(
        timestamp=time.time(),
        simulation_time_ms=T * rec.dt_ms,
        n_timesteps=T,
        mode=rec.config.mode,
        regions=region_stats,
        oscillations=oscillations,
        connectivity=connectivity,
        homeostasis=homeostasis,
        health=health,
        pop_keys=list(rec._pop_keys),
        region_keys=list(rec._region_keys),
        pop_rate_binned=pop_rate_binned,
        region_rate_binned=region_rate_binned,
        raw_spike_counts=rec._pop_spike_counts[:T].copy(),
    )

    # Neuromodulator concentration histories (both modes)
    if rec._n_nm_receptors > 0:
        n_steps = rec._gain_sample_step
        nm_levels: Dict[str, np.ndarray] = {}
        for nm_idx, (rn, mod_name) in enumerate(rec._nm_receptor_keys):
            key = f"{rn}/{mod_name}"
            nm_levels[key] = rec._nm_concentration_history[:n_steps, nm_idx].copy()
        report.neuromodulator_levels = nm_levels

    if rec.config.mode == "full":
        assert rec._voltages is not None
        assert rec._g_exc_samples is not None
        assert rec._g_inh_samples is not None
        report.raw_voltages = rec._voltages[:T].copy()
        report.voltage_sample_times_ms = np.arange(T, dtype=np.float32) * rec.dt_ms
        if rec._cond_sample_step > 0:
            ci = rec.config.conductance_sample_interval_steps
            report.conductance_sample_times_ms = (
                np.arange(0, T, ci, dtype=np.float32) * rec.dt_ms
            )[: rec._cond_sample_step]
            report.raw_g_exc = rec._g_exc_samples[: rec._cond_sample_step].copy()
            report.raw_g_inh = rec._g_inh_samples[: rec._cond_sample_step].copy()

    return report
