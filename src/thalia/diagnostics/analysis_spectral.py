"""Spectral analysis: band powers, coherence matrices, PAC, integration tau, orchestrator."""

from __future__ import annotations

import itertools
import random
import warnings
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit as sp_curve_fit
from scipy.signal import coherence as sp_coherence
from scipy.signal import welch as sp_welch

from .bio_ranges import EEG_BANDS, freq_to_band
from .diagnostics_metrics import OscillatoryStats
from .diagnostics_snapshot import RecorderSnapshot


# =============================================================================
# SPECTRAL HELPERS
# =============================================================================


def band_power(psd: np.ndarray, freqs: np.ndarray, f_min: float, f_max: float) -> float:
    """Integrate PSD (from Welch) over [f_min, f_max] Hz using the trapezoid rule."""
    mask = (freqs >= f_min) & (freqs <= f_max)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _coherence(
    x: np.ndarray,
    y: np.ndarray,
    f_min: float,
    f_max: float,
    rate_bin_ms: float,
) -> float:
    """Mean magnitude-squared coherence in [f_min, f_max] Hz (Welch-based)."""
    fs = 1.0 / (rate_bin_ms / 1000.0)
    nperseg = min(len(x), 512)
    if nperseg < 8:
        return 0.0
    freqs, Cxy = sp_coherence(x, y, fs=fs, nperseg=nperseg, window="hann")
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return 0.0
    return float(Cxy[mask].mean())


# =============================================================================
# BAND POWERS
# =============================================================================


def compute_band_powers(
    rec: RecorderSnapshot,
    region_rate_binned: np.ndarray,
    n_bins: int,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, float],
    Dict[str, str],
    Dict[str, float],
    float,
    Dict[str, Dict[str, float]],
]:
    """Compute per-region and global Welch PSD band powers and dominant frequencies.

    Returns:
        ``(region_band_power, region_dominant_freq, region_dominant_band,
        global_band_power, global_dominant_freq_hz, region_band_power_absolute)``
    """
    fs = 1.0 / (rec.config.rate_bin_ms / 1000.0)
    nperseg = min(n_bins, 512)

    region_band_power: Dict[str, Dict[str, float]] = {}
    region_band_power_absolute: Dict[str, Dict[str, float]] = {}
    region_dominant_freq: Dict[str, float] = {}
    region_dominant_band: Dict[str, str] = {}

    for r_idx, rn in enumerate(rec._region_keys):
        trace = region_rate_binned[:, r_idx].astype(np.float64)
        if trace.sum() < 1e-9 or n_bins < 8:
            region_band_power[rn] = {b: 0.0 for b in EEG_BANDS}
            region_band_power_absolute[rn] = {b: 0.0 for b in EEG_BANDS}
            region_dominant_freq[rn] = 0.0
            region_dominant_band[rn] = "none"
            continue

        freqs, psd = sp_welch(trace, fs=fs, nperseg=nperseg, window="hann", scaling="density")

        raw_bp = {b: band_power(psd, freqs, f1, f2) for b, (f1, f2) in EEG_BANDS.items()}
        total = sum(raw_bp.values()) or 1.0
        region_band_power[rn] = {b: v / total for b, v in raw_bp.items()}
        region_band_power_absolute[rn] = dict(raw_bp)

        psd_no_dc = psd[1:]
        freqs_no_dc = freqs[1:]
        dom_freq = float(freqs_no_dc[int(np.argmax(psd_no_dc))]) if len(psd_no_dc) > 0 else 0.0
        region_dominant_freq[rn] = dom_freq

        region_dominant_band[rn] = freq_to_band(dom_freq)

    global_trace = region_rate_binned.sum(axis=1).astype(np.float64)
    global_band_power: Dict[str, float] = {}
    global_dominant_freq = 0.0
    if global_trace.sum() > 1e-9 and n_bins >= 8:
        freqs, psd = sp_welch(global_trace, fs=fs, nperseg=nperseg, window="hann", scaling="density")
        raw_bp = {b: band_power(psd, freqs, f1, f2) for b, (f1, f2) in EEG_BANDS.items()}
        total = sum(raw_bp.values()) or 1.0
        global_band_power = {b: v / total for b, v in raw_bp.items()}
        psd_no_dc = psd[1:]
        freqs_no_dc = freqs[1:]
        global_dominant_freq = float(freqs_no_dc[int(np.argmax(psd_no_dc))]) if len(psd_no_dc) > 0 else 0.0
    else:
        global_band_power = {b: 0.0 for b in EEG_BANDS}

    return region_band_power, region_dominant_freq, region_dominant_band, global_band_power, global_dominant_freq, region_band_power_absolute


# =============================================================================
# COHERENCE MATRICES
# =============================================================================


def compute_coherence_matrices(
    rec: RecorderSnapshot,
    region_rate_binned: np.ndarray,
    n_bins: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute cross-regional coherence matrices (theta, beta, gamma) and spectral resolution.

    Gamma coherence is only computed when Nyquist ≥ 100 Hz (rate_bin_ms ≤ 5 ms);
    otherwise the gamma matrix is filled with NaN.

    Returns:
        ``(coh_theta, coh_beta, coh_gamma, freq_resolution_hz)``
    """
    rate_bin_ms = rec.config.rate_bin_ms
    fs = 1.0 / (rate_bin_ms / 1000.0)
    nperseg = min(n_bins, 512)
    n_r = rec._n_regions

    coh_theta = np.full((n_r, n_r), np.nan, dtype=np.float32)
    coh_beta  = np.full((n_r, n_r), np.nan, dtype=np.float32)
    np.fill_diagonal(coh_theta, 1.0)
    np.fill_diagonal(coh_beta,  1.0)

    gamma_computable = (1000.0 / rate_bin_ms) >= 200.0  # Nyquist ≥ 100 Hz
    if gamma_computable:
        coh_gamma = np.full((n_r, n_r), np.nan, dtype=np.float32)
        np.fill_diagonal(coh_gamma, 1.0)
    else:
        coh_gamma = np.full((n_r, n_r), np.nan, dtype=np.float32)

    region_activity = region_rate_binned.sum(axis=0)
    active_r = np.where(region_activity > 0)[0].tolist()
    max_pairs = rec.config.coherence_n_pairs
    all_pairs = list(itertools.combinations(active_r, 2))
    if len(all_pairs) > max_pairs:
        rng = random.Random(rec.config.coherence_seed)
        all_pairs = rng.sample(all_pairs, max_pairs)
    for ri_val, rj_val in all_pairs:
        ri, rj = int(ri_val), int(rj_val)
        x = region_rate_binned[:, ri].astype(np.float64)
        y = region_rate_binned[:, rj].astype(np.float64)
        coh_theta[ri, rj] = coh_theta[rj, ri] = _coherence(x, y,  4.0,   8.0, rate_bin_ms)
        coh_beta [ri, rj] = coh_beta [rj, ri] = _coherence(x, y, 13.0,  30.0, rate_bin_ms)
        if gamma_computable:
            coh_gamma[ri, rj] = coh_gamma[rj, ri] = _coherence(x, y, 30.0, 100.0, rate_bin_ms)

    freq_resolution_hz = float("nan")
    if n_bins >= 8:
        freq_resolution_hz = fs / nperseg
        if n_bins < 64:
            warnings.warn(
                f"Spectral frequency resolution degraded: only {n_bins} rate bins "
                f"({T * rec.dt_ms:.0f} ms at {rec.config.rate_bin_ms:.0f} ms/bin).  "
                f"Frequency resolution = {freq_resolution_hz:.2f} Hz.  "
                f"Simulate ≥ {int(64 * rec.config.rate_bin_ms)} ms for sub-1 Hz resolution.",
                UserWarning,
                stacklevel=4,
            )

    return coh_theta, coh_beta, coh_gamma, freq_resolution_hz


# =============================================================================
# APERIODIC (1/f) EXPONENT
# =============================================================================


def _aperiodic_model(log_freq: np.ndarray, offset: float, exponent: float) -> np.ndarray:
    """Linear model in log-log space: log10(PSD) = offset - exponent * log10(f)."""
    return offset - exponent * log_freq


def compute_aperiodic_exponent(
    rec: RecorderSnapshot,
    region_rate_binned: np.ndarray,
    n_bins: int,
) -> Dict[str, float]:
    """Fit the aperiodic (1/f) spectral exponent χ per region.

    The power spectral density of neural signals follows a characteristic
    1/f^χ power law in the aperiodic (background) component.  This
    exponent reflects the balance between excitation and inhibition:

    * χ ≈ 1.0–2.0: healthy cortical activity (He 2014; Donoghue et al. 2020).
    * χ < 0.5: flattened spectrum — suggests epileptiform or
      noise-dominated activity (reduced temporal correlation).
    * χ > 3.0: very steep spectrum — suggests over-inhibition or
      disconnected low-frequency fluctuations.

    Method:
        1. Compute Welch PSD (same parameters as band_power analysis).
        2. Restrict to 2–40 Hz (avoids DC artefacts and aliasing edge).
        3. Fit a linear model in log₁₀–log₁₀ space using least squares:
           ``log₁₀(PSD) = offset − χ · log₁₀(f)``
        4. Require R² ≥ 0.70 for reporting.
    """
    result: Dict[str, float] = {}
    fs = 1.0 / (rec.config.rate_bin_ms / 1000.0)
    nperseg = min(n_bins, 512)

    if n_bins < 32:
        return result  # need a reasonable spectral estimate

    # Fitting range: 2–40 Hz avoids DC leak and stays well below Nyquist
    # for typical 10 ms bins (Nyquist = 50 Hz).
    f_fit_lo = 2.0
    f_fit_hi = min(40.0, fs / 2.0 - 1.0)
    if f_fit_hi <= f_fit_lo:
        return result

    for r_idx, rn in enumerate(rec._region_keys):
        trace = region_rate_binned[:n_bins, r_idx].astype(np.float64)
        if trace.sum() < 1e-9:
            continue

        freqs, psd = sp_welch(trace, fs=fs, nperseg=nperseg, window="hann", scaling="density")

        # Select fitting range
        mask = (freqs >= f_fit_lo) & (freqs <= f_fit_hi) & (psd > 0)
        if mask.sum() < 5:
            continue

        log_f = np.log10(freqs[mask])
        log_psd = np.log10(psd[mask])

        # Ordinary least-squares fit in log-log space
        try:
            popt, _ = sp_curve_fit(
                _aperiodic_model,
                log_f,
                log_psd,
                p0=[np.mean(log_psd), 1.5],
                maxfev=1000,
            )
            exponent = float(popt[1])
        except Exception:
            continue

        # R² quality gate
        pred = _aperiodic_model(log_f, *popt)
        ss_res = float(np.sum((log_psd - pred) ** 2))
        ss_tot = float(np.sum((log_psd - log_psd.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 >= 0.70:
            result[rn] = exponent

    return result


# =============================================================================
# INTEGRATION TAU
# =============================================================================


def _exp_decay(t: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-t / tau)


def compute_integration_tau(
    rec: RecorderSnapshot,
    region_rate_binned: np.ndarray,
    n_bins: int,
) -> Dict[str, float]:
    """Compute the population-rate autocorrelation time constant (τ_int) per region.

    Fits an exponential decay A·exp(−lag/τ) to the normalised autocorrelation of the
    region's binned firing rate.  Lags in the window 1–1000 ms are used; shorter
    lags are skipped because the auto-correlation is trivially 1 at lag 0 and the
    smooth upsweep at lag 1 reflects spectral roll-off rather than temporal memory.

    Requirements for a valid fit:
    - ≥ 1000 ms of recording (100 bins at 10 ms / bin)
    - ≥ 1 non-zero bin in the region
    - Fitted τ ∈ (1, 5000) ms (sanity bounds)
    - R² of the exponential fit ≥ 0.50
    """
    result: Dict[str, float] = {}
    if n_bins < 100:
        return result  # need ≥ 1 s at default 10 ms bins

    bin_ms = rec.config.rate_bin_ms
    max_lag_bins = min(n_bins - 1, int(1000.0 / bin_ms))  # lag window: up to 1 s
    min_lag_bins = max(1, int(1.0 / bin_ms))               # skip lag-0 artefact
    lag_bins = np.arange(min_lag_bins, max_lag_bins + 1)
    lag_ms = lag_bins * bin_ms

    for r_idx, rn in enumerate(rec._region_keys):
        trace = region_rate_binned[:n_bins, r_idx].astype(np.float64)
        if np.sum(trace > 0) < 10:
            continue  # not enough activity

        # Normalised autocorrelation via FFT (unbiased estimator)
        trace_z = trace - trace.mean()
        n = len(trace_z)
        fft = np.fft.rfft(trace_z, n=2 * n)
        acf_full = np.fft.irfft(fft * np.conj(fft))[:n]
        # Unbiased normalisation: divide by (n - lag) not n
        acf_norm = np.array([
            acf_full[lag] / ((n - lag) * np.var(trace_z)) if np.var(trace_z) > 0 else 0.0
            for lag in lag_bins
        ])

        # Only fit over lags where acf is positive (exponential cannot fit sign changes)
        pos_mask = acf_norm > 0
        if pos_mask.sum() < 5:
            continue

        try:
            popt, _ = sp_curve_fit(
                _exp_decay,
                lag_ms[pos_mask],
                acf_norm[pos_mask],
                p0=[100.0],
                bounds=(1.0, 5000.0),
                maxfev=2000,
            )
            tau_fit = float(popt[0])
        except Exception:
            continue

        # Compute R² on all positive-lag points
        pred = _exp_decay(lag_ms[pos_mask], tau_fit)
        ss_res = float(np.sum((acf_norm[pos_mask] - pred) ** 2))
        ss_tot = float(np.sum((acf_norm[pos_mask] - acf_norm[pos_mask].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 >= 0.50:
            result[rn] = tau_fit

    return result


# =============================================================================
# OSCILLATORY STATS ORCHESTRATOR
# =============================================================================


def compute_oscillatory_stats(
    rec: RecorderSnapshot,
    pop_rate_binned: np.ndarray,
    region_rate_binned: np.ndarray,
    n_bins: int,
    T: int,
) -> OscillatoryStats:
    """Orchestrate all oscillatory analyses and assemble OscillatoryStats."""
    # Local imports to avoid circular dependency (coupling modules import from this file).
    from .coupling import (
        compute_beta_burst_stats,
        compute_ca3_ca1_theta_sequence,
        compute_cerebellar_metrics,
        compute_cfc_per_region,
        compute_hfo_per_region,
        compute_laminar_cascade,
        compute_plv_theta_per_region,
        compute_relay_burst_mode,
        compute_spike_avalanches,
        compute_spike_field_coherence,
        compute_swr_ca3_ca1_coupling,
    )

    (
        region_band_power,
        region_dominant_freq,
        region_dominant_band,
        global_band_power,
        global_dominant_freq,
        region_band_power_absolute,
    ) = compute_band_powers(rec, region_rate_binned, n_bins)

    coh_theta, coh_beta, coh_gamma, freq_resolution_hz = compute_coherence_matrices(
        rec, region_rate_binned, n_bins, T
    )
    cfc_results, lfp_methods = compute_cfc_per_region(rec, T)
    hfo_band_power = compute_hfo_per_region(rec, T)
    plv_theta = compute_plv_theta_per_region(rec, T)
    spike_field = compute_spike_field_coherence(rec, T, region_band_power)
    avalanche = compute_spike_avalanches(rec, T)
    cerebellar = compute_cerebellar_metrics(rec, T)
    beta_burst_stats = compute_beta_burst_stats(rec, region_rate_binned, n_bins)
    relay_burst_mode = compute_relay_burst_mode(rec, T)
    swr_ca3_ca1_coupling = compute_swr_ca3_ca1_coupling(rec, T)
    ca3_ca1_theta_sequence = compute_ca3_ca1_theta_sequence(rec, T)
    region_integration_tau_ms = compute_integration_tau(rec, region_rate_binned, n_bins)
    region_aperiodic_exponent = compute_aperiodic_exponent(rec, region_rate_binned, n_bins)
    laminar_cascade_latencies = compute_laminar_cascade(rec, T)

    return OscillatoryStats(
        region_band_power=region_band_power,
        region_band_power_absolute=region_band_power_absolute,
        region_dominant_freq=region_dominant_freq,
        region_dominant_band=region_dominant_band,
        coherence_theta=coh_theta,
        coherence_beta=coh_beta,
        coherence_gamma=coh_gamma,
        region_order=list(rec._region_keys),
        global_dominant_freq_hz=global_dominant_freq,
        global_band_power=global_band_power,
        freq_resolution_hz=freq_resolution_hz,
        cfc_results=cfc_results,
        lfp_proxy_methods=lfp_methods,
        hfo_band_power=hfo_band_power,
        plv_theta=plv_theta,
        spike_field_coherence=spike_field,
        avalanche=avalanche,
        cerebellar=cerebellar,
        beta_burst_stats=beta_burst_stats,
        relay_burst_mode=relay_burst_mode,
        swr_ca3_ca1_coupling=swr_ca3_ca1_coupling,
        ca3_ca1_theta_sequence=ca3_ca1_theta_sequence,
        region_integration_tau_ms=region_integration_tau_ms,
        region_aperiodic_exponent=region_aperiodic_exponent,
        laminar_cascade_latencies=laminar_cascade_latencies,
    )
