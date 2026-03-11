"""Diagnostics analysis orchestrator — thin entry point that delegates to sub-modules."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Tuple

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


def detect_transient_step(rec: "DiagnosticsRecorder", T: int) -> int:
    """Detect the end of the onset transient; return the first step to analyse.

    Scans homeostatic gain trajectories for the earliest gain-sample index at
    which ALL valid (non-NaN, non-zero) trajectories have changed by less than
    2 % over a sliding backward window of 50 gain-sample points.  Converts to
    a timestep index and returns it.

    Falls back to 300 ms (≈ 5 × STP equilibration τ at 30 Hz thalamic drive)
    when homeostasis is disabled or no valid gain trajectories exist.  This
    covers the STP depletion transient even when gains are all constant.

    The result is capped at T // 2 so the analysis window is never shorter
    than half the recording.  Returns 0 when T < 200 (too short to detect).
    """
    if T < 200:
        return 0

    n_steps = rec._gain_sample_step
    gi_steps = max(1, int(rec.config.gain_sample_interval_ms / rec.dt_ms))

    # ── Primary: homeostatic gain settling ───────────────────────────────────
    if n_steps >= 20:
        window = min(50, n_steps // 4)
        valid_indices: List[int] = []
        for idx in range(rec._n_pops):
            vals = rec._g_L_scale_history[:n_steps, idx]
            clean = vals[~np.isnan(vals)]
            if len(clean) > 0 and float(np.mean(np.abs(clean))) > 1e-6:
                valid_indices.append(idx)

        if valid_indices:
            for g in range(window, n_steps):
                settled = True
                for idx in valid_indices:
                    traj = rec._g_L_scale_history[g - window : g + 1, idx]
                    valid = traj[~np.isnan(traj)]
                    if len(valid) < max(4, window // 4):
                        settled = False
                        break
                    half = len(valid) // 2
                    past_mean = float(np.mean(valid[:half]))
                    recent_mean = float(np.mean(valid[half:]))
                    if (
                        abs(past_mean) > 1e-6
                        and abs(recent_mean - past_mean) / abs(past_mean) > 0.02
                    ):
                        settled = False
                        break
                if settled:
                    t0 = min(g * gi_steps, T // 2)
                    return t0

    # ── Fallback: STP equilibration (300 ms ≈ 5 × τ_STP) ────────────────────
    return min(int(300.0 / rec.dt_ms), T // 2)


def analyze(rec: "DiagnosticsRecorder") -> DiagnosticsReport:
    """Compute and return a complete :class:`DiagnosticsReport`.

    Call this via ``DiagnosticsRecorder.analyze()``.
    """
    T = rec._n_recorded or rec.config.n_timesteps

    # Detect and exclude the onset transient from steady-state analyses.
    t0 = detect_transient_step(rec, T)
    T_eff = T - t0
    if t0 > 0:
        print(
            f"\n  [transient] Excluding first {t0} steps "
            f"({t0 * rec.dt_ms:.0f} ms) as network onset transient.  "
            f"Steady-state analyses use {T_eff} steps ({T_eff * rec.dt_ms:.0f} ms)."
        )

    # Population-level stats (computed on the post-transient window [t0:T])
    pop_stats: Dict[Tuple[str, str], PopulationStats] = {}
    for idx, (rn, pn) in enumerate(rec._pop_keys):
        pop_stats[(rn, pn)] = compute_population_stats(rec, idx, T, t0)

    # Region-level stats
    region_stats: Dict[str, RegionStats] = {}
    for rn in rec._region_keys:
        region_stats[rn] = compute_region_stats(rec, rn, pop_stats)

    # Binned population rates [n_bins, n_pops]: spikes / neuron per bin.
    # Computed over [t0 : t0 + n_bins*bin_steps] so spectral analyses operate
    # on the steady-state portion of the recording only.
    bin_steps = max(1, int(rec.config.rate_bin_ms / rec.dt_ms))
    n_bins = T_eff // bin_steps
    if n_bins > 0:
        pop_rate_binned = (
            rec._pop_spike_counts[t0 : t0 + n_bins * bin_steps]
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

    # Raw buffer accesses in oscillatory sub-functions (pac, hfo, plv, etc.) use T
    # (the full recording) so they can apply their own temporal filtering.
    oscillations = compute_oscillatory_stats(rec, pop_rate_binned, region_rate_binned, n_bins, T)
    connectivity = compute_connectivity_stats(rec, T)
    homeostasis = compute_homeostatic_stats(rec)
    health = assess_health(rec, pop_stats, region_stats, connectivity, oscillations, homeostasis)

    report = DiagnosticsReport(
        timestamp=time.time(),
        simulation_time_ms=T * rec.dt_ms,
        n_timesteps=T,
        transient_steps=t0,
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
