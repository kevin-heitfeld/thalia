"""Hippocampal coupling analysis — PLV, HFO, SWR, and theta-sequence metrics.

Merged from ``plv.py``, ``hfo.py``, ``swr.py``, and ``theta_sequence.py``.
All four analyses target CA3/CA1 populations within hippocampal regions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d
from scipy.signal import butter as sp_butter
from scipy.signal import filtfilt as sp_filtfilt
from scipy.signal import hilbert as sp_hilbert
from scipy.signal import welch as sp_welch

from thalia.diagnostics._helpers import cross_correlate_at_lags
from thalia.diagnostics.analysis_spectral import band_power
from thalia.diagnostics.diagnostics_metrics import (
    PlvThetaStats,
    SwrCouplingRegionStats,
    ThetaSequenceRegionStats,
)
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot


# ═══════════════════════════════════════════════════════════════════════════════
# PLV — spike–theta phase-locking value
# ═══════════════════════════════════════════════════════════════════════════════


def compute_plv_theta_per_region(rec: RecorderSnapshot, T: int) -> PlvThetaStats:
    """Compute spike–theta PLV for CA1 pyramidal cells in each hippocampal region.

    Returns empty dicts when < 500 ms simulated.

    Reference signal for theta-phase extraction:
    The theta rhythm is driven by the medial septum (MS) GABAergic pacemaker.
    Using the target region's own spike count as the reference is circular and
    inflates PLV.  We therefore try to use the MS GABA population as the
    reference first; only if that signal is absent or too sparse (<10 spikes)
    do we fall back to the hippocampal region-internal signal.  When the
    fallback is used, PLV values should be interpreted with caution.

    Returns:
        :class:`PlvThetaStats` with ``values`` and ``used_fallback`` dicts keyed by region name.
    """
    plv_theta: Dict[str, float] = {}
    plv_theta_used_fallback: Dict[str, bool] = {}
    if T * rec.dt_ms < 500.0:
        return PlvThetaStats()
    fs_plv = 1000.0 / rec.dt_ms
    nyq_plv = fs_plv / 2.0
    theta_lo, theta_hi = 4.0, 8.0
    if nyq_plv <= theta_hi:
        return PlvThetaStats()

    # ── Build medial-septum GABA reference signal once ─────────────────
    ms_gaba_idx = [
        i for i, (rn_k, pn_k) in enumerate(rec._pop_keys)
        if "medial_septum" in rn_k.lower() and "gaba" in pn_k.lower()
    ]
    ms_ref: Optional[np.ndarray] = None
    if ms_gaba_idx:
        candidate = rec._pop_spike_counts[:T, ms_gaba_idx].sum(axis=1).astype(np.float64)
        if candidate.sum() >= 10:
            ms_ref = candidate

    try:
        b_plv, a_plv = sp_butter(
            4, [theta_lo / nyq_plv, theta_hi / nyq_plv], btype="bandpass"
        )
    except ValueError:
        return PlvThetaStats()

    for rn_plv in rec._region_keys:
        if "hippocampus" not in rn_plv.lower():
            continue

        if ms_ref is not None:
            ref_signal = ms_ref
            used_fallback = False
        else:
            p_idx_plv = rec._region_pop_indices[rn_plv]
            fallback = rec._pop_spike_counts[:T, p_idx_plv].sum(axis=1).astype(np.float64)
            if fallback.sum() < 10:
                plv_theta[rn_plv] = float("nan")
                continue
            ref_signal = fallback
            used_fallback = True
        plv_theta_used_fallback[rn_plv] = used_fallback

        try:
            theta_phase_plv = np.angle(sp_hilbert(sp_filtfilt(b_plv, a_plv, ref_signal)))
        except ValueError:
            plv_theta[rn_plv] = float("nan")
            continue

        ca1_pyr_keys = [
            (rn_plv, pn_plv)
            for _, (rn_plv2, pn_plv) in enumerate(rec._pop_keys)
            if rn_plv2 == rn_plv and "ca1" in pn_plv.lower() and "pyr" in pn_plv.lower()
        ]
        spike_phases_plv: List[float] = []
        for key_plv in ca1_pyr_keys:
            if key_plv not in rec._spike_times:
                continue
            for times_plv in rec._spike_times[key_plv]:
                for t_plv in times_plv:
                    if 0 <= t_plv < len(theta_phase_plv):
                        spike_phases_plv.append(theta_phase_plv[t_plv])
        if len(spike_phases_plv) >= 10:
            plv_theta[rn_plv] = float(
                np.abs(np.mean(np.exp(1j * np.array(spike_phases_plv))))
            )
        else:
            plv_theta[rn_plv] = float("nan")
    return PlvThetaStats(values=plv_theta, used_fallback=plv_theta_used_fallback)


# ═══════════════════════════════════════════════════════════════════════════════
# HFO — high-frequency oscillation band-power
# ═══════════════════════════════════════════════════════════════════════════════


def compute_hfo_per_region(rec: RecorderSnapshot, T: int) -> Dict[str, float]:
    """Compute HFO (100–250 Hz) band-power fraction for each hippocampal CA1 population.

    Requires dt_ms ≤ 1.0 and ≥ 200 ms of simulation.  Returns an empty dict otherwise.
    """
    hfo_band_power: Dict[str, float] = {}
    if T * rec.dt_ms < 200.0 or rec.dt_ms > 1.0:
        return hfo_band_power
    fs_raw_hfo = 1000.0 / rec.dt_ms
    for rn in rec._region_keys:
        if "hippocampus" not in rn.lower():
            continue
        ca1_indices = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn and "ca1" in p.lower()
        ]
        if not ca1_indices:
            continue
        raw_ca1 = rec._pop_spike_counts[:T, ca1_indices].sum(axis=1).astype(np.float64)
        if raw_ca1.sum() < 10:
            hfo_band_power[rn] = 0.0
            continue
        nperseg_hfo = min(len(raw_ca1), 512)
        if nperseg_hfo < 8:
            continue
        f_hfo, psd_hfo = sp_welch(
            raw_ca1, fs=fs_raw_hfo, nperseg=nperseg_hfo, window="hann", scaling="density"
        )
        hfo_pow = band_power(psd_hfo, f_hfo, 100.0, 250.0)
        total_pow = band_power(psd_hfo, f_hfo, 1.0, fs_raw_hfo / 2.0)
        hfo_band_power[rn] = float(hfo_pow / total_pow) if total_pow > 0 else 0.0
    return hfo_band_power


# ═══════════════════════════════════════════════════════════════════════════════
# Shared CA3→CA1 cross-correlation helper
# ═══════════════════════════════════════════════════════════════════════════════


def _ca3_ca1_xcorr(
    rec: RecorderSnapshot,
    T: int,
    *,
    sigma_ms: float,
    lag_lo_ms: float,
    lag_hi_ms: float,
    min_duration_ms: float,
    require_fine_dt: bool,
) -> Dict[str, tuple]:
    """Compute Gaussian-smoothed CA3→CA1 cross-correlation for hippocampal regions.

    Shared implementation for SWR coupling and theta-sequence analysis.
    Returns a dict mapping region name to ``(best_corr, best_lag_ms)`` tuples,
    or ``None`` values when populations are absent or silent.
    """
    result: Dict[str, tuple] = {}
    if T * rec.dt_ms < min_duration_ms:
        return result
    if require_fine_dt and rec.dt_ms > 1.0:
        return result

    dt_ms = rec.dt_ms
    lag_lo = max(1, int(lag_lo_ms / dt_ms))
    lag_hi = max(lag_lo + 1, int(lag_hi_ms / dt_ms))
    sigma_steps = max(1.0, sigma_ms / dt_ms)

    for rn in rec._region_keys:
        if "hippocampus" not in rn.lower():
            continue
        ca3_indices = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn and "ca3" in p.lower()
        ]
        ca1_indices = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn and "ca1" in p.lower()
        ]
        if not ca3_indices or not ca1_indices:
            continue

        ca3_raw = rec._pop_spike_counts[:T, ca3_indices].sum(axis=1).astype(np.float64)
        ca1_raw = rec._pop_spike_counts[:T, ca1_indices].sum(axis=1).astype(np.float64)
        if ca3_raw.sum() < 10 or ca1_raw.sum() < 10:
            result[rn] = None
            continue

        ca3_smooth = sp_gaussian_filter1d(ca3_raw, sigma=sigma_steps)
        ca1_smooth = sp_gaussian_filter1d(ca1_raw, sigma=sigma_steps)

        best_corr, best_lag_idx = cross_correlate_at_lags(ca3_smooth, ca1_smooth, lag_lo, lag_hi)
        best_lag_ms = best_lag_idx * dt_ms if not np.isnan(best_lag_idx) else float("nan")
        result[rn] = (best_corr, best_lag_ms)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SWR — sharp-wave ripple CA3→CA1 coupling
# ═══════════════════════════════════════════════════════════════════════════════


def compute_swr_ca3_ca1_coupling(rec: RecorderSnapshot, T: int) -> Dict[str, SwrCouplingRegionStats]:
    """Compute CA3→CA1 cross-correlation to validate SWR temporal coupling.

    Applies a Gaussian smoothing kernel (σ=5 ms) to raw CA3 and CA1
    spike-count signals, then computes the Pearson cross-correlation at causal
    lags 5–50 ms — covering monosynaptic Schaffer collateral latency (5–10 ms),
    disynaptic paths via interneurons (10–20 ms), and recurrent/feedback
    coupling during SWR events (20–50 ms).

    Requires ``dt_ms ≤ 1.0`` and ≥ 200 ms of simulation.

    Returns:
        Dict keyed by region name.  Each value dict contains
        ``ca3_ca1_xcorr_peak`` and ``ca3_ca1_lag_ms``.
    """
    raw = _ca3_ca1_xcorr(
        rec, T,
        sigma_ms=5.0, lag_lo_ms=5.0, lag_hi_ms=50.0,
        min_duration_ms=200.0, require_fine_dt=True,
    )
    result: Dict[str, SwrCouplingRegionStats] = {}
    for rn, val in raw.items():
        if val is None:
            result[rn] = SwrCouplingRegionStats()
        else:
            result[rn] = SwrCouplingRegionStats(ca3_ca1_xcorr_peak=val[0], ca3_ca1_lag_ms=val[1])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Theta-sequence — CA3→CA1 cross-correlation at theta timescales
# ═══════════════════════════════════════════════════════════════════════════════


def compute_ca3_ca1_theta_sequence(rec: RecorderSnapshot, T: int) -> Dict[str, ThetaSequenceRegionStats]:
    """Compute CA3→CA1 cross-correlation at theta-sequence timescales (5–30 ms).

    Within each theta cycle, CA3 place cells activate before CA1 place cells
    via Schaffer collaterals, compressing sequences into ~125 ms theta windows
    (Foster & Wilson 2007; Dragoi & Buzsáki 2006).  This function measures
    whether CA3 population activity reliably precedes CA1 activity at the
    5–30 ms causal lags expected from this feedforward connectivity.

    A 25 ms Gaussian smoothing kernel (≈ one theta half-cycle at 8 Hz) is
    used instead of the 5 ms kernel in :func:`compute_swr_ca3_ca1_coupling`,
    preserving theta-modulated rate fluctuations rather than sharp SWR
    transients.

    Requires ≥ 500 ms of simulation.  Both ``ca3`` and ``ca1`` population name
    substrings must be present in the hippocampal region.  Unlike the SWR
    check this function has no ``dt_ms`` upper limit, though coarser timesteps
    reduce lag resolution.

    Returns:
        Dict keyed by region name.  Each value dict contains:

        * ``xcorr_peak``  — peak Pearson cross-correlation in the 5–30 ms window
        * ``peak_lag_ms`` — lag (ms) at which the peak occurs

        ``NaN`` for both keys when either population is silent or the
        activity variance is too low for a meaningful correlation.
    """
    raw = _ca3_ca1_xcorr(
        rec, T,
        sigma_ms=25.0, lag_lo_ms=5.0, lag_hi_ms=30.0,
        min_duration_ms=500.0, require_fine_dt=False,
    )
    result: Dict[str, ThetaSequenceRegionStats] = {}
    for rn, val in raw.items():
        if val is None:
            result[rn] = ThetaSequenceRegionStats()
        else:
            result[rn] = ThetaSequenceRegionStats(xcorr_peak=val[0], peak_lag_ms=val[1])
    return result
