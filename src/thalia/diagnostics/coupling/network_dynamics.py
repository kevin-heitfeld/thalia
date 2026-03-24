"""Network-level dynamics — spike avalanches, beta bursts, and synaptic gain.

Merged from ``avalanches.py``, ``motor.py``, and ``synaptic_gain.py``.
These analyses operate at the whole-network or inter-region level rather
than targeting a specific brain structure.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter as sp_butter
from scipy.signal import filtfilt as sp_filtfilt
from scipy.signal import hilbert as sp_hilbert

from thalia.diagnostics._helpers import bin_counts_1d, cross_correlate_at_lags
from thalia.diagnostics.diagnostics_metrics import AvalancheStats, BetaBurstRegionStats
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot


# ═══════════════════════════════════════════════════════════════════════════════
# Spike avalanches — power-law fit and branching ratio
# ═══════════════════════════════════════════════════════════════════════════════


def compute_spike_avalanches(rec: RecorderSnapshot, T: int) -> AvalancheStats:
    """Fit a power-law to the spike avalanche size distribution and compute the
    branching ratio σ (Beggs & Plenz 2003).

    Skipped when ``config.compute_avalanches`` is False (auto-disabled for
    short runs < 2000 timesteps).  Bins spikes at the mean axonal delay across
    all tracts (natural propagation timescale), falling back to
    ``config.avalanche_bin_ms`` when no tracts are present.

    Returns:
        ``(avalanche_exponent, avalanche_r2, branching_ratio)`` — all NaN when
        unavailable.

        * ``branching_ratio`` (σ): pooled ratio of total descendant spikes to
          total ancestor spikes across all consecutive active-bin pairs.
          σ ≈ 1 → critical; σ < 1 → subcritical (healthy rest); σ > 1 → supercritical.
    """
    nan = float("nan")
    if not rec.config.compute_avalanches or T <= 0:
        return AvalancheStats()

    if rec._tract_delay_ms:
        _mean_delay_ms = float(np.mean(rec._tract_delay_ms))
        _bin_ms_av = max(rec.dt_ms, _mean_delay_ms)
    else:
        _bin_ms_av = rec.config.avalanche_bin_ms

    bin_steps_av = max(1, int(_bin_ms_av / rec.dt_ms))
    n_bins_av = T // bin_steps_av
    total_counts_av = bin_counts_1d(
        rec._pop_spike_counts[:T].sum(axis=1), n_bins_av, bin_steps_av,
    )

    # Branching ratio σ
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
        return AvalancheStats(branching_ratio=branching_ratio)

    av_arr = np.array(av_sizes, dtype=np.float64)
    s_min, s_max = av_arr.min(), av_arr.max()
    if s_max <= s_min:
        return nan, nan, branching_ratio
    bins_av = np.logspace(np.log10(s_min), np.log10(s_max), 25)
    hist_av, _ = np.histogram(av_arr, bins=bins_av, density=True)
    mid_av = 0.5 * (bins_av[:-1] + bins_av[1:])
    valid_av = hist_av > 0
    if valid_av.sum() < 5:
        return AvalancheStats(branching_ratio=branching_ratio)
    log_x = np.log10(mid_av[valid_av])
    log_y = np.log10(hist_av[valid_av])
    coeffs = np.polyfit(log_x, log_y, 1)
    y_pred = np.polyval(coeffs, log_x)
    ss_res = float(np.sum((log_y - y_pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return AvalancheStats(exponent=float(coeffs[0]), r2=r2, branching_ratio=branching_ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# Beta bursts — basal ganglia and motor cortex
# ═══════════════════════════════════════════════════════════════════════════════


_BG_MOTOR_KEYWORDS = frozenset({
    "stn", "gpe", "gpi", "striatum", "cortex_motor",
    "putamen", "caudate", "subthalamic", "globus_pallidus",
})


def compute_beta_burst_stats(
    rec: RecorderSnapshot,
    region_rate_binned: np.ndarray,
    n_bins: int,
) -> Dict[str, BetaBurstRegionStats]:
    """Detect beta bursts in BG and motor cortex regions.

    Band-pass filters the population rate signal to 13–30 Hz, computes the
    Hilbert amplitude envelope, thresholds at the 75th percentile, and detects
    contiguous suprathreshold epochs.

    Only computed for BG/motor regions and when rate_bin_ms ≤ 16 ms
    (Nyquist ≥ 30 Hz) and at least 500 ms of data are available.

    Returns:
        Dict keyed by region name.  Each value dict contains:
        * ``n_bursts``        — number of detected bursts (≥ 100 ms each)
        * ``mean_duration_ms`` — mean burst duration in ms
        * ``max_duration_ms``  — maximum burst duration in ms
        * ``mean_ibi_ms``      — mean inter-burst interval (NaN when ≤ 1 burst)
    """
    result: Dict[str, BetaBurstRegionStats] = {}
    rate_bin_ms = rec.config.rate_bin_ms
    fs = 1000.0 / rate_bin_ms
    nyq = fs / 2.0
    beta_lo, beta_hi = 13.0, 30.0
    if nyq <= beta_hi:
        return result
    if n_bins < int(500.0 / rate_bin_ms):
        return result
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
        result[rn] = BetaBurstRegionStats(
            n_bursts=float(n_bursts),
            mean_duration_ms=(
                float(np.mean(burst_durations)) * rate_bin_ms if n_bursts > 0 else float("nan")
            ),
            max_duration_ms=(
                float(max(burst_durations)) * rate_bin_ms if n_bursts > 0 else float("nan")
            ),
            mean_ibi_ms=(
                float(np.mean(ibi_steps)) * rate_bin_ms if len(ibi_steps) > 0 else float("nan")
            ),
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Effective synaptic gain — inter-region tract efficacy
# ═══════════════════════════════════════════════════════════════════════════════


def compute_effective_synaptic_gain(
    rec: RecorderSnapshot,
    region_rate_binned: np.ndarray,
    n_bins: int,
) -> Dict[str, float]:
    """Estimate effective synaptic gain per inter-region tract.

    For each tract connecting distinct regions, computes the Pearson cross-
    correlation between the pre-region and post-region binned firing rate
    time series at the lag closest to the tract's axonal delay.  This provides
    a linear estimate of how much pre-synaptic activity in the source region
    drives post-synaptic rate changes in the target region.

    Only inter-region tracts with at least 20 rate bins and non-zero variance
    in both signals are included.  Returns a dict of ``{tract_label: gain}``.
    """
    if n_bins < 20 or region_rate_binned.shape[0] < 20:
        return {}

    rate_bin_ms = rec.config.rate_bin_ms

    seen_pairs: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for tract_idx, sid in enumerate(rec._tract_keys):
        src_r = sid.source_region
        tgt_r = sid.target_region
        if src_r == tgt_r:
            continue
        pair_key = (src_r, tgt_r)
        if pair_key in seen_pairs:
            continue
        delay_ms = rec._tract_delay_ms[tract_idx] if tract_idx < len(rec._tract_delay_ms) else 0.0
        label = f"{src_r}\u2192{tgt_r}"
        seen_pairs[pair_key] = (label, delay_ms)

    result: Dict[str, float] = {}
    for (src_r, tgt_r), (label, delay_ms) in seen_pairs.items():
        src_idx = rec._region_index.get(src_r)
        tgt_idx = rec._region_index.get(tgt_r)
        if src_idx is None or tgt_idx is None:
            continue

        src_rate = region_rate_binned[:n_bins, src_idx].astype(np.float64)
        tgt_rate = region_rate_binned[:n_bins, tgt_idx].astype(np.float64)

        lag_bins = max(0, int(round(delay_ms / rate_bin_ms)))
        lag_bins = min(lag_bins, n_bins // 3)

        xcorr, _ = cross_correlate_at_lags(src_rate, tgt_rate, lag_bins, lag_bins)
        if not np.isnan(xcorr):
            result[label] = xcorr

    return result
