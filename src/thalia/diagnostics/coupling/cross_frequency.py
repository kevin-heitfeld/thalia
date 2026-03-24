"""Generic cross-frequency coupling (CFC) — PAC, AAC, PFC for arbitrary band pairs.

Supports three coupling types:

* **PAC** (Phase–Amplitude Coupling): Phase of low-frequency band modulates
  amplitude of high-frequency band.  Metric: Mean Vector Length (MVL)
  (Canolty et al. 2006).
* **AAC** (Amplitude–Amplitude Coupling): Correlation between amplitude
  envelopes of two frequency bands.  Metric: Pearson correlation of
  envelopes (Bruns et al. 2000).
* **PFC** (Phase–Frequency Coupling): Phase of low-frequency band modulates
  the instantaneous frequency of high-frequency band.  Metric: circular–linear
  correlation between low-freq phase and high-freq instantaneous frequency
  (Tort et al. 2008).

All functions accept a 1-D signal (LFP proxy) and return a scalar metric.
The orchestrator :func:`compute_cfc_per_region` applies :data:`CFC_SPECS`
to each region, constructs LFP proxies, and produces :class:`CFCResult` entries.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter as sp_butter
from scipy.signal import filtfilt as sp_filtfilt
from scipy.signal import hilbert as sp_hilbert

from thalia.diagnostics.bio_spectral_specs import CFC_SPECS, EEG_BANDS
from thalia.diagnostics.coupling.lfp_proxy import LfpMethod, build_all_region_lfp_proxies
from thalia.diagnostics.diagnostics_metrics import CFCResult
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot

# Minimum simulation duration (ms) for meaningful CFC estimation.
_MIN_DURATION_MS: float = 500.0


# =============================================================================
# INDIVIDUAL CFC METRICS
# =============================================================================


def _bandpass_analytic(
    signal: np.ndarray,
    fs: float,
    band: Tuple[float, float],
) -> np.ndarray | None:
    """Bandpass filter *signal* and return the analytic signal (Hilbert).

    Returns ``None`` if the Nyquist frequency is below the upper band edge
    or the filter fails.
    """
    nyq = fs / 2.0
    if nyq <= band[1]:
        return None
    try:
        b, a = sp_butter(4, [band[0] / nyq, band[1] / nyq], btype="bandpass")
        return sp_hilbert(sp_filtfilt(b, a, signal.astype(np.float64)))
    except ValueError:
        return None


def compute_pac_mvl(
    signal: np.ndarray,
    fs: float,
    phase_band: Tuple[float, float],
    amp_band: Tuple[float, float],
) -> float:
    """Phase–amplitude coupling via Mean Vector Length (Canolty et al. 2006).

    Returns MVL modulation index in [0, 1], or NaN if the signal is too short
    or the sampling rate cannot resolve the amplitude band.
    """
    if len(signal) < int(0.5 * fs):
        return float("nan")
    analytic_phase = _bandpass_analytic(signal, fs, phase_band)
    analytic_amp = _bandpass_analytic(signal, fs, amp_band)
    if analytic_phase is None or analytic_amp is None:
        return float("nan")
    theta_phase = np.angle(analytic_phase)
    gamma_amp = np.abs(analytic_amp)
    return float(np.abs(np.mean(gamma_amp * np.exp(1j * theta_phase))))


def compute_aac(
    signal: np.ndarray,
    fs: float,
    band_a: Tuple[float, float],
    band_b: Tuple[float, float],
) -> float:
    """Amplitude–amplitude coupling via Pearson envelope correlation (Bruns et al. 2000).

    Returns Pearson *r* between the amplitude envelopes of two frequency bands,
    or NaN if the signal is too short or filtering fails.
    """
    if len(signal) < int(0.5 * fs):
        return float("nan")
    analytic_a = _bandpass_analytic(signal, fs, band_a)
    analytic_b = _bandpass_analytic(signal, fs, band_b)
    if analytic_a is None or analytic_b is None:
        return float("nan")
    env_a = np.abs(analytic_a)
    env_b = np.abs(analytic_b)
    std_a = float(np.std(env_a))
    std_b = float(np.std(env_b))
    if std_a < 1e-12 or std_b < 1e-12:
        return float("nan")
    return float(np.corrcoef(env_a, env_b)[0, 1])


def compute_pfc(
    signal: np.ndarray,
    fs: float,
    phase_band: Tuple[float, float],
    freq_band: Tuple[float, float],
) -> float:
    """Phase–frequency coupling: low-freq phase modulates high-freq instantaneous frequency.

    Metric: circular–linear correlation between the phase of *phase_band* and
    the instantaneous frequency of *freq_band* (Tort et al. 2008).
    Returns a value in [0, 1], or NaN if the signal is too short.
    """
    if len(signal) < int(0.5 * fs):
        return float("nan")
    analytic_phase = _bandpass_analytic(signal, fs, phase_band)
    analytic_freq = _bandpass_analytic(signal, fs, freq_band)
    if analytic_phase is None or analytic_freq is None:
        return float("nan")
    low_phase = np.angle(analytic_phase)
    # Instantaneous frequency = d(phase)/dt × fs / (2π)
    high_phase = np.unwrap(np.angle(analytic_freq))
    inst_freq = np.diff(high_phase) * fs / (2.0 * np.pi)
    # Align lengths
    low_phase = low_phase[:-1]
    # Circular–linear correlation: R = sqrt(r_cs² + r_cc² − 2·r_cs·r_cc·r_sc_cc) / (1 − r_sc_cc²)
    # Simplified: correlation between inst_freq and cos/sin of phase
    cos_p = np.cos(low_phase)
    sin_p = np.sin(low_phase)
    n = len(inst_freq)
    if n < 10:
        return float("nan")
    std_f = float(np.std(inst_freq))
    if std_f < 1e-12:
        return float("nan")
    r_cf = float(np.corrcoef(cos_p, inst_freq)[0, 1])
    r_sf = float(np.corrcoef(sin_p, inst_freq)[0, 1])
    r_cs = float(np.corrcoef(cos_p, sin_p)[0, 1])
    denom = 1.0 - r_cs * r_cs
    if denom < 1e-12:
        return float("nan")
    rho_sq = (r_cf * r_cf + r_sf * r_sf - 2.0 * r_cf * r_sf * r_cs) / denom
    return float(np.sqrt(max(0.0, rho_sq)))


# =============================================================================
# DISPATCH
# =============================================================================

_CFC_DISPATCH = {
    "pac": compute_pac_mvl,
    "aac": compute_aac,
    "pfc": compute_pfc,
}


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def compute_cfc_per_region(
    rec: RecorderSnapshot,
    T: int,
) -> Tuple[List[CFCResult], Dict[str, LfpMethod]]:
    """Compute cross-frequency coupling for each region × spec in :data:`CFC_SPECS`.

    Uses the best available LFP proxy for each region: current-based
    (from recorded conductances and voltages) when available, otherwise
    Gaussian-smoothed spike rate.

    Returns
    -------
    (results, lfp_methods)
        ``results`` — list of :class:`CFCResult`, one per matching region × spec.
        ``lfp_methods`` — dict mapping region name → LFP proxy method used.
    """
    results: List[CFCResult] = []
    lfp_methods: Dict[str, LfpMethod] = {}
    if T * rec.dt_ms < _MIN_DURATION_MS:
        return results, lfp_methods

    fs_raw = 1000.0 / rec.dt_ms

    # Build LFP proxies for all regions (current-based preferred, spike-rate fallback).
    lfp_cache, method_cache = build_all_region_lfp_proxies(rec, T)
    lfp_methods = method_cache

    for rn in rec._region_keys:
        rn_lower = rn.lower()
        matching_specs = [s for s in CFC_SPECS if s.region.lower() in rn_lower]
        if not matching_specs:
            continue

        lfp_proxy = lfp_cache.get(rn)
        if lfp_proxy is None:
            continue

        for spec in matching_specs:
            fn = _CFC_DISPATCH.get(spec.coupling_type)
            if fn is None:
                continue
            phase_band = EEG_BANDS.get(spec.phase_band)
            amp_band = EEG_BANDS.get(spec.amp_band)
            if phase_band is None or amp_band is None:
                continue
            value = fn(lfp_proxy, fs_raw, phase_band, amp_band)
            results.append(CFCResult(
                region=rn,
                coupling_type=spec.coupling_type,
                phase_band=spec.phase_band,
                amp_band=spec.amp_band,
                value=value,
            ))

    return results, lfp_methods
