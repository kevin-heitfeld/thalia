"""Diagnostics analysis orchestrator — thin entry point that delegates to sub-modules."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from ._helpers import bin_counts_2d
from .diagnostics_metrics import PopulationStats, RegionStats
from .diagnostics_report import DiagnosticsReport
from .diagnostics_snapshot import RecorderSnapshot
from .analysis_connectivity import compute_connectivity_stats
from .analysis_health import assess_health
from .analysis_homeostasis import compute_homeostatic_stats
from .analysis_learning import compute_learning_stats
from .brain_state_classifier import classify_brain_state
from .coupling import compute_effective_synaptic_gain
from .analysis_spectral import compute_oscillatory_stats
from .analysis_population import compute_population_stats
from .analysis_region import compute_region_stats


def _find_settling_step(
    history: np.ndarray,
    n_steps: int,
    valid_indices: List[int],
    window: int,
    threshold_pct: float,
) -> int:
    """Find the earliest gain-sample index at which all trajectories have settled.

    Returns n_steps if no settling point is found.
    """
    for g in range(window, n_steps):
        settled = True
        for idx in valid_indices:
            traj = history[g - window : g + 1, idx]
            valid = traj[~np.isnan(traj)]
            if len(valid) < max(4, window // 4):
                settled = False
                break
            half = len(valid) // 2
            past_mean = float(np.mean(valid[:half]))
            recent_mean = float(np.mean(valid[half:]))
            if (
                abs(past_mean) > 1e-6
                and abs(recent_mean - past_mean) / abs(past_mean) > threshold_pct
            ):
                settled = False
                break
        if settled:
            return g
    return n_steps


def detect_transient_step(rec: RecorderSnapshot, T: int) -> int:
    """Detect the end of the onset transient; return the first step to analyse.

    Uses three independent signals and takes the latest settling point:

    1. **Homeostatic gain trajectories** — scans for the earliest gain-sample
       index at which ALL valid trajectories have changed by less than 2 % over
       a sliding backward window of 50 gain-sample points.

    2. **STP efficacy trajectories** — same sliding-window test on x·u efficacy
       histories (5 % threshold, since STP is noisier than gain).

    3. **Firing rate settling** — bins raw spike counts at
       ``gain_sample_interval_ms`` resolution, then applies the same
       sliding-window test (10 % threshold) across all populations with
       non-trivial activity.  This directly detects when population firing
       rates reach steady state, complementing the proxy signals from gain
       and STP.

    Falls back to 3 × max(τ_d) from the STP configs (clamped to [300, 2000] ms)
    when no valid trajectories exist.  This covers the STP depletion transient
    even when gains are all constant.

    The result is capped at T // 2 so the analysis window is never shorter
    than half the recording.  Returns 0 when T < 200 (too short to detect).
    """
    if T < 200:
        return 0

    n_steps = rec._gain_sample_step
    gi_steps = max(1, int(rec.config.gain_sample_interval_ms / rec.dt_ms))
    cap = T // 2

    # ── Compute STP-based fallback from actual τ_d values ────────────────────
    max_tau_d = 60.0  # baseline assumption
    if rec._stp_configs:
        max_tau_d = max(max_tau_d, max(cfg[1] for cfg in rec._stp_configs))
    # 3 × max(τ_d) covers STP convergence at moderate presynaptic rates.
    # Clamp to [300, 2000] ms to avoid both too-short and absurd fallbacks.
    stp_fallback_ms = max(300.0, min(2000.0, 3.0 * max_tau_d))
    fallback_step = min(int(stp_fallback_ms / rec.dt_ms), cap)

    if n_steps < 20:
        return fallback_step

    window = min(50, n_steps // 4)

    # ── Signal 1: homeostatic gain settling ──────────────────────────────────
    gain_settle = n_steps  # default: not settled
    gain_valid: List[int] = []
    for idx in range(rec._n_pops):
        vals = rec._g_L_scale_history[:n_steps, idx]
        clean = vals[~np.isnan(vals)]
        if len(clean) > 0 and float(np.mean(np.abs(clean))) > 1e-6:
            gain_valid.append(idx)
    if gain_valid:
        gain_settle = _find_settling_step(
            rec._g_L_scale_history, n_steps, gain_valid, window,
            rec.config.transient_gain_threshold_pct,
        )

    # ── Signal 2: STP efficacy settling (noisier) ────────────────────────────
    stp_settle = n_steps  # default: not settled
    n_stp = rec._stp_efficacy_history.shape[1] if rec._stp_efficacy_history.ndim == 2 else 0
    stp_valid: List[int] = []
    for idx in range(n_stp):
        vals = rec._stp_efficacy_history[:n_steps, idx]
        clean = vals[~np.isnan(vals)]
        if len(clean) > 0 and float(np.mean(np.abs(clean))) > 1e-6:
            stp_valid.append(idx)
    if stp_valid:
        stp_settle = _find_settling_step(
            rec._stp_efficacy_history, n_steps, stp_valid, window,
            rec.config.transient_stp_threshold_pct,
        )

    # ── Signal 3: firing rate settling ────────────────────────────────────────
    # Bin raw spike counts at the same cadence as gain/STP sampling so the
    # same _find_settling_step logic and window size apply directly.
    rate_settle = n_steps  # default: not settled
    rate_valid: List[int] = []
    n_rate_bins = min(n_steps, T // gi_steps)
    if n_rate_bins >= 20:
        usable_steps = n_rate_bins * gi_steps
        rate_binned = (
            bin_counts_2d(rec._pop_spike_counts[:usable_steps], n_rate_bins, gi_steps)
            / np.maximum(rec._pop_sizes, 1).astype(np.float64)
        )
        for idx in range(rec._n_pops):
            col = rate_binned[:, idx]
            col_mean: float = float(np.mean(col))
            if col_mean > 1e-6:
                rate_valid.append(idx)
        if rate_valid:
            rate_settle = _find_settling_step(
                rate_binned, n_rate_bins, rate_valid, window,
                rec.config.transient_rate_threshold_pct,
            )

    # ── Combine: take the latest of all valid settling signals ───────────────
    candidates: List[int] = []
    if gain_valid:
        candidates.append(gain_settle)
    if stp_valid:
        candidates.append(stp_settle)
    if rate_valid:
        candidates.append(rate_settle)

    if candidates:
        detected = max(candidates)
        if detected < n_steps:
            return min(detected * gi_steps, cap)

    return fallback_step


def analyze(rec: RecorderSnapshot) -> DiagnosticsReport:
    """Compute and return a complete :class:`DiagnosticsReport`."""
    T = rec._n_recorded or rec.config.n_timesteps

    if rec._running_stats is not None:
        total = rec._running_stats.total_n_recorded
        print(
            f"\n  [streaming] Analysing window of {T} steps "
            f"({T * rec.dt_ms:.0f} ms) from total run of "
            f"{total} steps ({total * rec.dt_ms:.0f} ms)."
        )

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
            bin_counts_2d(rec._pop_spike_counts[t0 : t0 + n_bins * bin_steps], n_bins, bin_steps).astype(np.float32)
            / np.maximum(rec._pop_sizes, 1).astype(np.float32)
        )
    else:
        pop_rate_binned = np.zeros((0, rec._n_pops), dtype=np.float32)

    # Region binned rates derived from pop_rate_binned for strict consistency.
    # region_rate[b, r] = Σ_p (pop_rate[b,p] * pop_size[p]) / Σ_p pop_size[p]
    # Weights are sourced from pop_stats[...].n_neurons — the same field used by
    # weighted_mean_fr() — so the time-mean of region_rate_binned[:,r] is
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
    learning = compute_learning_stats(rec)
    synaptic_gain = compute_effective_synaptic_gain(rec, region_rate_binned, n_bins)
    inferred_brain_state = classify_brain_state(oscillations)
    health = assess_health(rec, pop_stats, region_stats, connectivity, oscillations, homeostasis, learning, inferred_brain_state)

    report = DiagnosticsReport(
        timestamp=time.time(),
        simulation_time_ms=T * rec.dt_ms,
        n_timesteps=T,
        transient_steps=t0,
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
        effective_synaptic_gain=synaptic_gain if synaptic_gain else None,
        learning=learning,
    )

    # Neuromodulator concentration histories (both modes)
    if rec._n_nm_receptors > 0:
        n_steps = rec._gain_sample_step
        nm_levels: Dict[str, np.ndarray] = {}
        for nm_idx, (rn, mod_name) in enumerate(rec._nm_receptor_keys):
            key = f"{rn}/{mod_name}"
            nm_levels[key] = rec._nm_concentration_history[:n_steps, nm_idx].copy()
        report.neuromodulator_levels = nm_levels

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
