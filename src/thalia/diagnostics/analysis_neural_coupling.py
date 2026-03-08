"""Neural coupling analysis: PLV, SWR, HFO, cerebellar metrics, beta bursts, avalanches, relay burst mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d
from scipy.signal import butter as sp_butter
from scipy.signal import filtfilt as sp_filtfilt
from scipy.signal import hilbert as sp_hilbert
from scipy.signal import welch as sp_welch

from .analysis_spectral import band_power

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


# =============================================================================
# SHARED HELPER
# =============================================================================


def _bin_spike_times_to_array(
    st_list: List[List[int]],
    n_bins: int,
    bin_steps: int,
) -> np.ndarray:
    """Accumulate spike times into a binned population-count vector (vectorized)."""
    vec = np.zeros(n_bins, dtype=np.float64)
    non_empty = [st for st in st_list if st]
    if not non_empty:
        return vec
    all_spikes = np.concatenate([np.asarray(st, dtype=np.int64) for st in non_empty])
    bins_idx = all_spikes // bin_steps
    mask = (bins_idx >= 0) & (bins_idx < n_bins)
    np.add.at(vec, bins_idx[mask], 1.0)
    return vec


# =============================================================================
# PLV THETA
# =============================================================================


def compute_plv_theta_per_region(
    rec: "DiagnosticsRecorder",
    T: int,
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Compute spike–theta PLV for CA1 pyramidal cells in each hippocampal region.

    Full mode only; returns empty dicts in stats mode or when < 500 ms simulated.

    Reference signal for theta-phase extraction:
    The theta rhythm is driven by the medial septum (MS) GABAergic pacemaker.
    Using the target region's own spike count as the reference is circular and
    inflates PLV.  We therefore try to use the MS GABA population as the
    reference first; only if that signal is absent or too sparse (<10 spikes)
    do we fall back to the hippocampal region-internal signal.  When the
    fallback is used, PLV values should be interpreted with caution.

    Returns:
        ``(plv_theta, plv_theta_used_fallback)`` — both dicts keyed by region name.
    """
    plv_theta: Dict[str, float] = {}
    plv_theta_used_fallback: Dict[str, bool] = {}
    if rec.config.mode != "full" or T * rec.dt_ms < 500.0:
        return plv_theta, plv_theta_used_fallback
    fs_plv = 1000.0 / rec.dt_ms
    nyq_plv = fs_plv / 2.0
    theta_lo, theta_hi = 4.0, 8.0
    if nyq_plv <= theta_hi:
        return plv_theta, plv_theta_used_fallback

    # ── Build medial-septum GABA reference signal once ─────────────────
    # Prefer MS GABA populations; they rhythmically burst at theta frequency
    # and are the canonical theta generator (Freund & Buzsáki 1996).
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
        return plv_theta, plv_theta_used_fallback

    for rn_plv in rec._region_keys:
        if "hippocampus" not in rn_plv.lower():
            continue

        # Determine which signal to extract theta phase from.
        if ms_ref is not None:
            ref_signal = ms_ref
            used_fallback = False
        else:
            # Fallback: region-internal signal (circular, but better than nothing).
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
    return plv_theta, plv_theta_used_fallback


# =============================================================================
# SWR CA3→CA1 COUPLING
# =============================================================================


def compute_swr_ca3_ca1_coupling(
    rec: "DiagnosticsRecorder",
    T: int,
) -> Dict[str, Dict[str, float]]:
    """Compute CA3→CA1 cross-correlation to validate SWR temporal coupling.

    Applies a Gaussian smoothing kernel (σ=5 ms) to raw CA3 and CA1
    spike-count signals, then computes the Pearson cross-correlation at causal
    lags 10–30 ms — the physiological window in which a CA3 sharp-wave
    excitatory burst precedes the CA1 ripple (Buzsáki 2015).

    Requires ``dt_ms ≤ 1.0`` and ≥ 200 ms of simulation.  Both ``ca3`` and
    ``ca1`` population name substrings must be present in the hippocampal
    region.

    Returns:
        Dict keyed by region name.  Each value dict contains:

        * ``ca3_ca1_xcorr_peak`` — peak Pearson cross-correlation at causal lags
        * ``ca3_ca1_lag_ms``     — lag (ms) at the peak

        ``NaN`` for both keys when either population is silent or has constant
        activity (zero standard deviation).
    """
    result: Dict[str, Dict[str, float]] = {}
    if T * rec.dt_ms < 200.0 or rec.dt_ms > 1.0:
        return result

    dt_ms = rec.dt_ms
    lag_lo = max(1, int(10.0 / dt_ms))
    lag_hi = max(lag_lo + 1, int(30.0 / dt_ms))
    sigma_steps = max(1.0, 5.0 / dt_ms)

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
            result[rn] = {"ca3_ca1_xcorr_peak": float("nan"), "ca3_ca1_lag_ms": float("nan")}
            continue

        ca3_smooth = sp_gaussian_filter1d(ca3_raw, sigma=sigma_steps)
        ca1_smooth = sp_gaussian_filter1d(ca1_raw, sigma=sigma_steps)
        ca3_z = ca3_smooth - ca3_smooth.mean()
        ca1_z = ca1_smooth - ca1_smooth.mean()
        std3 = float(ca3_z.std())
        std1 = float(ca1_z.std())
        if std3 < 1e-9 or std1 < 1e-9:
            result[rn] = {"ca3_ca1_xcorr_peak": float("nan"), "ca3_ca1_lag_ms": float("nan")}
            continue

        best_corr = -float("inf")
        best_lag_ms = float("nan")
        for lag in range(lag_lo, lag_hi + 1):
            if lag >= T:
                break
            corr = float(np.dot(ca3_z[:T - lag], ca1_z[lag:])) / (std3 * std1 * (T - lag))
            if corr > best_corr:
                best_corr = corr
                best_lag_ms = float(lag) * dt_ms
        result[rn] = {
            "ca3_ca1_xcorr_peak": best_corr if not np.isinf(best_corr) else float("nan"),
            "ca3_ca1_lag_ms": best_lag_ms,
        }
    return result


# =============================================================================
# HFO
# =============================================================================


def compute_hfo_per_region(
    rec: "DiagnosticsRecorder",
    T: int,
) -> Dict[str, float]:
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


# =============================================================================
# CEREBELLAR METRICS
# =============================================================================


def compute_cerebellar_metrics(
    rec: "DiagnosticsRecorder",
    T: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute cerebellar timing metrics: Purkinje–DCN anti-correlation and IO pairwise synchrony.

    Returns:
        ``(purkinje_dcn_corr, io_pairwise_corr)`` — dicts keyed by region name.
    """
    purkinje_dcn_corr: Dict[str, float] = {}
    io_pairwise_corr: Dict[str, float] = {}
    bin_steps_cb = max(1, int(10.0 / rec.dt_ms))  # 10 ms bins for Purkinje-DCN
    for rn_cb in rec._region_keys:
        purk_idx = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn_cb and "purkinje" in p.lower()
        ]
        dcn_idx = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn_cb and "dcn" in p.lower()
        ]
        if purk_idx and dcn_idx:
            n_bins_cb = T // bin_steps_cb
            if n_bins_cb >= 4:
                purk_counts = rec._pop_spike_counts[:T, purk_idx].sum(axis=1).astype(np.float64)
                dcn_counts  = rec._pop_spike_counts[:T, dcn_idx ].sum(axis=1).astype(np.float64)
                p_b = purk_counts[:n_bins_cb * bin_steps_cb].reshape(n_bins_cb, bin_steps_cb).sum(axis=1)
                d_b = dcn_counts [:n_bins_cb * bin_steps_cb].reshape(n_bins_cb, bin_steps_cb).sum(axis=1)
                if p_b.std() > 0 and d_b.std() > 0:
                    purkinje_dcn_corr[rn_cb] = float(np.corrcoef(p_b, d_b)[0, 1])

        io_idx = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn_cb and "inferior_olive" in p.lower()
        ]
        if io_idx and rec.config.mode == "full":
            bin_steps_io = max(1, int(200.0 / rec.dt_ms))
            n_bins_io = T // bin_steps_io
            if n_bins_io >= 4 and len(io_idx) >= 2:
                io_key_list = [
                    (rn_cb, p)
                    for _, (r2, p) in enumerate(rec._pop_keys)
                    if r2 == rn_cb and "inferior_olive" in p.lower()
                ]
                io_pair_rs: List[float] = []
                for ki, io_key_i in enumerate(io_key_list):
                    if io_key_i not in rec._spike_times:
                        continue
                    for _kj, io_key_j in enumerate(io_key_list[ki + 1:], start=ki + 1):
                        if io_key_j not in rec._spike_times:
                            continue
                        vi = _bin_spike_times_to_array(rec._spike_times[io_key_i], n_bins_io, bin_steps_io)
                        vj = _bin_spike_times_to_array(rec._spike_times[io_key_j], n_bins_io, bin_steps_io)
                        if vi.std() > 0 and vj.std() > 0:
                            io_pair_rs.append(float(np.corrcoef(vi, vj)[0, 1]))
                if io_pair_rs:
                    io_pairwise_corr[rn_cb] = float(np.mean(io_pair_rs))
    return purkinje_dcn_corr, io_pairwise_corr


# =============================================================================
# BETA BURST ANALYSIS
# =============================================================================

_BG_MOTOR_KEYWORDS = frozenset({
    "stn", "gpe", "gpi", "striatum", "cortex_motor",
    "putamen", "caudate", "subthalamic", "globus_pallidus",
})


def compute_beta_burst_stats(
    rec: "DiagnosticsRecorder",
    region_rate_binned: np.ndarray,
    n_bins: int,
) -> Dict[str, Dict[str, float]]:
    """Detect beta bursts in BG and motor cortex regions.

    Band-pass filters the population rate signal to 13–30 Hz, computes the
    Hilbert amplitude envelope, thresholds at the 75th percentile, and detects
    contiguous suprathreshold epochs (Tinkhauser et al. 2017; Little et al. 2012).

    Only computed for BG/motor regions and when rate_bin_ms ≤ 16 ms
    (Nyquist ≥ 30 Hz) and at least 500 ms of data are available.

    Returns:
        Dict keyed by region name.  Each value dict contains:
        * ``n_bursts``        — number of detected bursts (≥ 100 ms each)
        * ``mean_duration_ms`` — mean burst duration in ms
        * ``max_duration_ms``  — maximum burst duration in ms
        * ``mean_ibi_ms``      — mean inter-burst interval (NaN when ≤ 1 burst)
    """
    result: Dict[str, Dict[str, float]] = {}
    rate_bin_ms = rec.config.rate_bin_ms
    fs = 1000.0 / rate_bin_ms
    nyq = fs / 2.0
    beta_lo, beta_hi = 13.0, 30.0
    if nyq <= beta_hi:
        return result  # Cannot resolve beta band at this sampling rate
    if n_bins < int(500.0 / rate_bin_ms):
        return result  # Too short for meaningful burst statistics
    # Minimum burst duration: 2 cycles at 20 Hz = 100 ms
    min_burst_steps = max(1, int(100.0 / rate_bin_ms))
    try:
        b_beta, a_beta = sp_butter(4, [beta_lo / nyq, beta_hi / nyq], btype="bandpass")
    except ValueError:
        return result

    for r_idx, rn in enumerate(rec._region_keys):
        rn_lower = rn.lower()
        if not any(kw in rn_lower for kw in _BG_MOTOR_KEYWORDS):
            continue
        trace = region_rate_binned[:, r_idx].astype(np.float64)
        if trace.sum() < 1e-9:
            continue
        try:
            filtered = sp_filtfilt(b_beta, a_beta, trace)
        except ValueError:
            continue
        envelope = np.abs(sp_hilbert(filtered))
        threshold = float(np.percentile(envelope, 75))
        if threshold <= 0:
            continue
        above = envelope > threshold
        burst_durations: List[int] = []
        ibi_steps: List[int] = []
        in_burst = False
        burst_start = 0
        last_burst_end = -1
        for i, v in enumerate(above):
            if v and not in_burst:
                in_burst = True
                burst_start = i
            elif not v and in_burst:
                in_burst = False
                dur = i - burst_start
                if dur >= min_burst_steps:
                    if last_burst_end >= 0:
                        ibi_steps.append(burst_start - last_burst_end)
                    burst_durations.append(dur)
                    last_burst_end = i
        if in_burst:
            dur = n_bins - burst_start
            if dur >= min_burst_steps:
                if last_burst_end >= 0:
                    ibi_steps.append(burst_start - last_burst_end)
                burst_durations.append(dur)
        n_bursts = len(burst_durations)
        result[rn] = {
            "n_bursts": float(n_bursts),
            "mean_duration_ms": (
                float(np.mean(burst_durations)) * rate_bin_ms if n_bursts > 0 else float("nan")
            ),
            "max_duration_ms": (
                float(max(burst_durations)) * rate_bin_ms if n_bursts > 0 else float("nan")
            ),
            "mean_ibi_ms": (
                float(np.mean(ibi_steps)) * rate_bin_ms if len(ibi_steps) > 0 else float("nan")
            ),
        }
    return result


# =============================================================================
# SPIKE AVALANCHES
# =============================================================================


def compute_spike_avalanches(
    rec: "DiagnosticsRecorder",
    T: int,
) -> Tuple[float, float, float]:
    """Fit a power-law to the spike avalanche size distribution and compute the
    branching ratio σ (Beggs & Plenz 2003).

    Only runs when ``config.compute_avalanches`` is True.  Bins spikes at
    ``config.avalanche_bin_ms`` resolution (Beggs & Plenz 2003).

    Returns:
        ``(avalanche_exponent, avalanche_r2, branching_ratio)`` — all NaN when
        unavailable.

        * ``branching_ratio`` (σ): pooled ratio of total descendant spikes to
          total ancestor spikes across all consecutive active-bin pairs.
          σ ≈ 1 → critical; σ < 1 → subcritical (healthy rest); σ > 1 → supercritical.
    """
    nan = float("nan")
    if not rec.config.compute_avalanches or T <= 0:
        return nan, nan, nan
    # Use the mean axonal delay across all tracts as the bin width — this is the
    # natural timescale for avalanche propagation in simulation (no electrode averaging).
    # Fall back to config.avalanche_bin_ms (default 4 ms, Beggs & Plenz 2003) when
    # the brain has no tracts.
    if rec.brain.axonal_tracts:
        _mean_delay_ms = float(
            np.mean([t.spec.delay_ms for t in rec.brain.axonal_tracts.values()])
        )
        _bin_ms_av = max(rec.dt_ms, _mean_delay_ms)
    else:
        _bin_ms_av = rec.config.avalanche_bin_ms
    bin_steps_av = max(1, int(_bin_ms_av / rec.dt_ms))
    n_bins_av = T // bin_steps_av
    total_counts_av = (
        rec._pop_spike_counts[:T]
        .sum(axis=1)[: n_bins_av * bin_steps_av]
        .reshape(n_bins_av, bin_steps_av)
        .sum(axis=1)
    )
    # Branching ratio σ: pooled descendants/ancestors over consecutive active bins.
    # Only pairs where the ancestor bin is non-zero are included (Beggs & Plenz 2003).
    _ancestors = total_counts_av[:-1]
    _descendants = total_counts_av[1:]
    _active = _ancestors > 0
    branching_ratio = (
        float(_descendants[_active].sum() / _ancestors[_active].sum())
        if _active.sum() >= 4
        else nan
    )

    av_sizes: List[int] = []
    current_av = 0
    for cnt in total_counts_av:
        if cnt > 0:
            current_av += int(cnt)
        elif current_av > 0:
            av_sizes.append(current_av)
            current_av = 0
    if current_av > 0:
        av_sizes.append(current_av)
    if len(av_sizes) < 20:
        return nan, nan, branching_ratio
    av_arr = np.array(av_sizes, dtype=np.float64)
    s_min, s_max = av_arr.min(), av_arr.max()
    if s_max <= s_min:
        return nan, nan, branching_ratio
    bins_av = np.logspace(np.log10(s_min), np.log10(s_max), 25)
    hist_av, _ = np.histogram(av_arr, bins=bins_av, density=True)
    mid_av = 0.5 * (bins_av[:-1] + bins_av[1:])
    valid_av = hist_av > 0
    if valid_av.sum() < 5:
        return nan, nan, branching_ratio
    log_x = np.log10(mid_av[valid_av])
    log_y = np.log10(hist_av[valid_av])
    coeffs = np.polyfit(log_x, log_y, 1)
    y_pred = np.polyval(coeffs, log_x)
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coeffs[0]), r2, branching_ratio


# =============================================================================
# RELAY BURST MODE
# =============================================================================

_RELAY_KEYWORDS = frozenset({
    "relay", "thalamus_relay", "lgn", "vpl", "vpm", "mgn",
    "pulvinar", "lateral_geniculate", "ventrobasal",
})


def compute_relay_burst_mode(
    rec: "DiagnosticsRecorder",
    T: int,  # noqa: ARG001  (T reserved for consistency with sibling functions)
) -> Dict[str, float]:
    """Compute the short-ISI fraction for relay populations in thalamic regions.

    Short ISIs (< 15 ms) are the hallmark of T-channel low-threshold spike
    (LTS) burst doublets and triplets in relay cells (McCormick & Huguenard
    1992).  A significant fraction (≥ 5 %) indicates active burst mode;
    near-zero indicates tonic Poisson-like firing.

    Full mode only — requires per-neuron spike times from ``rec._spike_times``.
    Returns an empty dict when spike time data is unavailable.

    Returns:
        Dict keyed by region name.  Each value is the fraction of ISIs < 15 ms
        across all sampled relay neurons in that region.
    """
    if not rec._spike_times:
        return {}

    short_isi_ms = 15.0
    result: Dict[str, float] = {}

    # Collect relay population keys grouped by region.
    region_relay_keys: Dict[str, List[Tuple[str, str]]] = {}
    for rn, pn in rec._pop_keys:
        rn_lower = rn.lower()
        pn_lower = pn.lower()
        if "thalamus" not in rn_lower and "thalamic" not in rn_lower:
            continue
        if not any(kw in pn_lower for kw in _RELAY_KEYWORDS):
            continue
        region_relay_keys.setdefault(rn, []).append((rn, pn))

    for rn, pop_keys in region_relay_keys.items():
        n_short = 0
        n_total = 0
        for key in pop_keys:
            if key not in rec._spike_times:
                continue
            for neuron_steps in rec._spike_times[key]:
                if len(neuron_steps) < 2:
                    continue
                isis = np.diff(
                    np.array(neuron_steps, dtype=np.float64) * rec.dt_ms
                )
                n_short += int(np.sum(isis < short_isi_ms))
                n_total += len(isis)
        if n_total > 0:
            result[rn] = n_short / n_total

    return result


# =============================================================================
# CA3→CA1 THETA-SEQUENCE COUPLING
# =============================================================================


def compute_ca3_ca1_theta_sequence(
    rec: "DiagnosticsRecorder",
    T: int,
) -> Dict[str, Dict[str, float]]:
    """Compute CA3\u2192CA1 cross-correlation at theta-sequence timescales (5\u201330 ms).

    Within each theta cycle, CA3 place cells activate before CA1 place cells
    via Schaffer collaterals, compressing sequences into ~125 ms theta windows
    (Foster & Wilson 2007; Dragoi & Buzs\u00e1ki 2006).  This function measures
    whether CA3 population activity reliably precedes CA1 activity at the
    5\u201330 ms causal lags expected from this feedforward connectivity.

    A 25 ms Gaussian smoothing kernel (\u2248 one theta half-cycle at 8 Hz) is
    used instead of the 5 ms kernel in :func:`compute_swr_ca3_ca1_coupling`,
    preserving theta-modulated rate fluctuations rather than sharp SWR
    transients.

    Requires \u2265 500 ms of simulation.  Both ``ca3`` and ``ca1`` population name
    substrings must be present in the hippocampal region.  Unlike the SWR
    check this function has no ``dt_ms`` upper limit, though coarser timesteps
    reduce lag resolution.

    Returns:
        Dict keyed by region name.  Each value dict contains:

        * ``xcorr_peak``  \u2014 peak Pearson cross-correlation in the 5\u201330 ms window
        * ``peak_lag_ms`` \u2014 lag (ms) at which the peak occurs

        ``NaN`` for both keys when either population is silent or the
        activity variance is too low for a meaningful correlation.
    """
    result: Dict[str, Dict[str, float]] = {}
    if T * rec.dt_ms < 500.0:
        return result

    dt_ms = rec.dt_ms
    lag_lo = max(1, int(5.0 / dt_ms))
    lag_hi = max(lag_lo + 1, int(30.0 / dt_ms))
    sigma_steps = max(1.0, 25.0 / dt_ms)  # 25 ms \u2248 one theta half-cycle

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
            result[rn] = {"xcorr_peak": float("nan"), "peak_lag_ms": float("nan")}
            continue

        ca3_smooth = sp_gaussian_filter1d(ca3_raw, sigma=sigma_steps)
        ca1_smooth = sp_gaussian_filter1d(ca1_raw, sigma=sigma_steps)
        ca3_z = ca3_smooth - ca3_smooth.mean()
        ca1_z = ca1_smooth - ca1_smooth.mean()
        std3 = float(ca3_z.std())
        std1 = float(ca1_z.std())
        if std3 < 1e-9 or std1 < 1e-9:
            result[rn] = {"xcorr_peak": float("nan"), "peak_lag_ms": float("nan")}
            continue

        best_corr = -float("inf")
        best_lag_ms = float("nan")
        for lag in range(lag_lo, lag_hi + 1):
            if lag >= T:
                break
            corr = float(np.dot(ca3_z[:T - lag], ca1_z[lag:])) / (std3 * std1 * (T - lag))
            if corr > best_corr:
                best_corr = corr
                best_lag_ms = float(lag) * dt_ms
        result[rn] = {
            "xcorr_peak": best_corr if not np.isinf(best_corr) else float("nan"),
            "peak_lag_ms": best_lag_ms,
        }
    return result


# =============================================================================
# CORTICAL LAMINAR CASCADE
# =============================================================================

_CORTICAL_TAGS = frozenset({"cortex", "prefrontal", "entorhinal"})

# Layer tier → substrings that identify a population as belonging to that tier.
# Checked against the lowercase population name.
_LAYER_KEYWORDS: Dict[str, List[str]] = {
    "l4":  ["l4"],
    "l23": ["l23", "l2_3"],
    "l5":  ["l5"],
    "l6":  ["l6"],
}


def _detect_thalamic_volleys(rec: "DiagnosticsRecorder", T: int) -> List[int]:
    """Return timestep indices of thalamic relay volleys.

    A volley is any timestep where the combined spike count across all thalamic
    relay populations reaches ≥ 5 % of the total relay neurons.  Consecutive
    events within 5 ms are merged to the first, so a sustained burst is counted
    as a single volley.
    """
    total_relay_spikes = np.zeros(T, dtype=np.int64)
    total_relay_neurons = 0

    for idx, (rn, pn) in enumerate(rec._pop_keys):
        rn_lower = rn.lower()
        pn_lower = pn.lower()
        if "thalamus" not in rn_lower and "thalamic" not in rn_lower:
            continue
        if not any(kw in pn_lower for kw in _RELAY_KEYWORDS):
            continue
        total_relay_spikes += rec._pop_spike_counts[:T, idx].astype(np.int64)
        total_relay_neurons += int(rec._pop_sizes[idx])

    if total_relay_neurons == 0:
        return []

    threshold = max(1.0, 0.05 * total_relay_neurons)
    hot = np.where(total_relay_spikes >= threshold)[0]
    if len(hot) == 0:
        return []

    # Merge events within 5 ms
    merge_steps = max(1, int(round(5.0 / rec.dt_ms)))
    events: List[int] = [int(hot[0])]
    for ts in hot[1:]:
        if ts - events[-1] > merge_steps:
            events.append(int(ts))
    return events


def compute_laminar_cascade(
    rec: "DiagnosticsRecorder",
    T: int,
) -> Dict[str, Dict[str, float]]:
    """Compute mean first-spike latency per cortical layer after thalamic volleys.

    For each cortical region (those whose name contains "cortex", "prefrontal",
    or "entorhinal"), and for each layer tier (L4, L2/3, L5, L6), this function
    measures how quickly the first spike in that tier arrives after each thalamic
    relay volley.  Averaging over volleys gives a stable estimate of the
    thalamocortical feedforward latency hierarchy.

    Expected latency order (Thomson & Bannister 2003; Sakata & Harris 2009):
        L4 < L2/3 < L5

    Full mode only — requires per-neuron spike times in ``rec._spike_times``.
    Returns an empty dict when spike time data or thalamic volleys are absent.

    Returns
    -------
    Dict keyed by cortical region name.  Each value is a dict with keys:
        ``l4_lat_ms``, ``l23_lat_ms``, ``l5_lat_ms``, ``l6_lat_ms``.
    A tier key is omitted when no matching population is present in the region;
    its value is NaN when matching populations exist but produced no spikes in
    any post-volley window.
    """
    if not rec._spike_times:
        return {}

    volley_timesteps = _detect_thalamic_volleys(rec, T)
    if not volley_timesteps:
        return {}

    window_steps = int(round(50.0 / rec.dt_ms))  # 50 ms measurement window
    result: Dict[str, Dict[str, float]] = {}

    for rn in rec._region_keys:
        rn_lower = rn.lower()
        if not any(tag in rn_lower for tag in _CORTICAL_TAGS):
            continue

        # Build sorted spike-time arrays per layer tier for this region.
        tier_arrays: Dict[str, List[np.ndarray]] = {}
        for tier, kws in _LAYER_KEYWORDS.items():
            arrays: List[np.ndarray] = []
            for r, pn in rec._pop_keys:
                if r != rn:
                    continue
                pn_lower = pn.lower()
                if not any(kw in pn_lower for kw in kws):
                    continue
                key = (r, pn)
                if key not in rec._spike_times:
                    continue
                flat: List[int] = []
                for neuron_spikes in rec._spike_times[key]:
                    flat.extend(neuron_spikes)
                if flat:
                    arrays.append(np.sort(np.array(flat, dtype=np.int64)))
            if arrays:
                tier_arrays[tier] = arrays

        if not tier_arrays:
            continue

        # Measure first-spike latency per tier per volley event.
        tier_latencies: Dict[str, List[float]] = {tier: [] for tier in tier_arrays}
        for t_event in volley_timesteps:
            t_end = t_event + window_steps
            for tier, arrays in tier_arrays.items():
                first_spike: Optional[int] = None
                for arr in arrays:
                    lo = int(np.searchsorted(arr, t_event, side="left"))
                    if lo < len(arr) and arr[lo] < t_end:
                        cand = int(arr[lo])
                        if first_spike is None or cand < first_spike:
                            first_spike = cand
                if first_spike is not None:
                    tier_latencies[tier].append((first_spike - t_event) * rec.dt_ms)

        region_result: Dict[str, float] = {}
        for tier, lats in tier_latencies.items():
            region_result[f"{tier}_lat_ms"] = float(np.mean(lats)) if lats else float("nan")
        result[rn] = region_result

    return result
