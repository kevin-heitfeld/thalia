"""Spectral analysis: band powers, coherence matrices, PAC, integration tau."""

from __future__ import annotations

import itertools
import random
import warnings
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit as sp_curve_fit
from scipy.signal import butter as sp_butter
from scipy.signal import coherence as sp_coherence
from scipy.signal import filtfilt as sp_filtfilt
from scipy.signal import hilbert as sp_hilbert
from scipy.signal import welch as sp_welch

from .bio_ranges import EEG_BANDS, PAC_SPECS, freq_to_band
from .diagnostics_types import RecorderSnapshot


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
    nperseg = min(len(x), 256)
    if nperseg < 8:
        return 0.0
    freqs, Cxy = sp_coherence(x, y, fs=fs, nperseg=nperseg, window="hann")
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return 0.0
    return float(Cxy[mask].mean())


# =============================================================================
# THETA–GAMMA PAC
# =============================================================================


def _compute_pac_mi(
    signal: np.ndarray,
    fs: float,
    theta_band: Tuple[float, float] = (4.0, 8.0),
    gamma_band: Tuple[float, float] = (30.0, 100.0),
) -> float:
    """Compute Mean Vector Length (MVL) theta–gamma PAC modulation index.

    Requires at least 500 ms of signal and a sampling rate that supports gamma
    frequencies (fs > 2 × gamma_band[1]).  Returns NaN otherwise.

    Args:
        signal: 1-D population rate trace sampled at ``fs`` Hz.
        fs: Sampling rate in Hz (typically 1000 / dt_ms).
        theta_band: (f_lo, f_hi) for theta phase extraction (Hz).
        gamma_band: (f_lo, f_hi) for gamma amplitude extraction (Hz).

    Returns:
        MVL modulation index ∈ [0, 1], or NaN.
    """
    nyq = fs / 2.0
    if len(signal) < int(0.5 * fs) or nyq <= gamma_band[1]:
        return float("nan")
    sig = signal.astype(np.float64)
    # Theta phase via bandpass + Hilbert
    b_t, a_t = sp_butter(4, [theta_band[0] / nyq, theta_band[1] / nyq], btype="bandpass")
    theta_phase = np.angle(sp_hilbert(sp_filtfilt(b_t, a_t, sig)))
    # Gamma amplitude envelope via bandpass + Hilbert
    b_g, a_g = sp_butter(4, [gamma_band[0] / nyq, gamma_band[1] / nyq], btype="bandpass")
    gamma_amp = np.abs(sp_hilbert(sp_filtfilt(b_g, a_g, sig)))
    # Mean Vector Length
    return float(np.abs(np.mean(gamma_amp * np.exp(1j * theta_phase))))


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
]:
    """Compute per-region and global Welch PSD band powers and dominant frequencies.

    Returns:
        ``(region_band_power, region_dominant_freq, region_dominant_band,
        global_band_power, global_dominant_freq_hz)``
    """
    fs = 1.0 / (rec.config.rate_bin_ms / 1000.0)
    nperseg = min(n_bins, 256)

    region_band_power: Dict[str, Dict[str, float]] = {}
    region_dominant_freq: Dict[str, float] = {}
    region_dominant_band: Dict[str, str] = {}

    for r_idx, rn in enumerate(rec._region_keys):
        trace = region_rate_binned[:, r_idx].astype(np.float64)
        if trace.sum() < 1e-9 or n_bins < 8:
            region_band_power[rn] = {b: 0.0 for b in EEG_BANDS}
            region_dominant_freq[rn] = 0.0
            region_dominant_band[rn] = "none"
            continue

        freqs, psd = sp_welch(trace, fs=fs, nperseg=nperseg, window="hann", scaling="density")

        raw_bp = {b: band_power(psd, freqs, f1, f2) for b, (f1, f2) in EEG_BANDS.items()}
        total = sum(raw_bp.values()) or 1.0
        region_band_power[rn] = {b: v / total for b, v in raw_bp.items()}

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

    return region_band_power, region_dominant_freq, region_dominant_band, global_band_power, global_dominant_freq


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
    nperseg = min(n_bins, 256)
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
# PAC PER REGION
# =============================================================================


def compute_pac_per_region(
    rec: RecorderSnapshot,
    T: int,
) -> Dict[str, float]:
    """Compute PAC (MVL modulation index) for each region matching a :data:`PAC_SPECS` entry.

    Uses Gaussian-smoothed native-dt spike counts (σ ≈ 5 ms) as the LFP proxy.
    Multiple specs may match the same region; the last match wins (more specific
    region substrings should appear later in ``PAC_SPECS`` if needed).
    Returns an empty dict when the simulation is shorter than 500 ms.
    """
    from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d

    pac_modulation_index: Dict[str, float] = {}
    if T * rec.dt_ms < 500.0:
        return pac_modulation_index
    fs_raw = 1000.0 / rec.dt_ms
    for rn in rec._region_keys:
        rn_lower = rn.lower()
        spec = next((s for s in reversed(PAC_SPECS) if s.region.lower() in rn_lower), None)
        if spec is None:
            continue
        p_indices = rec._region_pop_indices[rn]
        total_neurons = int(sum(rec._pop_sizes[i] for i in p_indices))
        if total_neurons == 0:
            continue
        raw_counts = rec._pop_spike_counts[:T, p_indices].sum(axis=1).astype(np.float64)
        sigma_steps = 5.0 / rec.dt_ms
        smooth_counts = sp_gaussian_filter1d(raw_counts, sigma=sigma_steps)
        phase_band = EEG_BANDS[spec.phase_band]
        amp_band = EEG_BANDS[spec.amp_band]
        pac_modulation_index[rn] = _compute_pac_mi(smooth_counts, fs_raw, phase_band, amp_band)
    return pac_modulation_index


# =============================================================================
# INTEGRATION TAU
# =============================================================================


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

    References: Murray et al. 2014 (Nature Neuroscience); Wasmuht et al. 2018.
    """
    result: Dict[str, float] = {}
    if n_bins < 100:
        return result  # need ≥ 1 s at default 10 ms bins

    bin_ms = rec.config.rate_bin_ms
    max_lag_bins = min(n_bins - 1, int(1000.0 / bin_ms))  # lag window: up to 1 s
    min_lag_bins = max(1, int(1.0 / bin_ms))               # skip lag-0 artefact
    lag_bins = np.arange(min_lag_bins, max_lag_bins + 1)
    lag_ms = lag_bins * bin_ms

    def _exp_decay(t: np.ndarray, tau: float) -> np.ndarray:
        return np.exp(-t / tau)

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
