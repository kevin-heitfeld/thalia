"""Utilities for analyzing oscillations in spike trains using FFT and autocorrelation."""

from typing import Optional, Tuple

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import correlate

from thalia.constants import DEFAULT_DT_MS


def measure_oscillation(
    spike_history: list[float],
    dt_ms:float = DEFAULT_DT_MS,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Detect dominant oscillation frequency in spike train using FFT.

    Args:
        spike_history: List of spike counts over time (length N)
        dt_ms: Timestep in milliseconds (default: 1.0)
        freq_range: Tuple of (min_hz, max_hz) to search for peak.
                   If None, searches full spectrum.

    Returns:
        Tuple of (dominant_freq_hz, spectral_power)
        Returns (0.0, 0.0) if no oscillation detected

    Example:
        >>> spike_counts = [10, 5, 2, 8, 15, 20, 12, 5, 3, 10, ...]
        >>> freq, power = measure_oscillation(spike_counts)
        >>> print(f"Oscillating at {freq:.1f} Hz (power={power:.3f})")
    """
    N = len(spike_history)
    if N < 10:
        return 0.0, 0.0

    # FFT of spike count time series
    yf = fft(spike_history)
    xf = fftfreq(N, dt_ms / 1000.0)  # Convert ms to seconds for Hz

    # Take positive frequencies only
    xf = xf[: N // 2]
    power = 2.0 / N * np.abs(yf[: N // 2])

    # Apply frequency range filter if specified
    if freq_range is not None:
        min_hz, max_hz = freq_range
        freq_mask = (xf >= min_hz) & (xf <= max_hz)
        if not np.any(freq_mask):
            return 0.0, 0.0

        filtered_power = power[freq_mask]
        filtered_freqs = xf[freq_mask]

        if len(filtered_power) > 0:
            peak_idx = np.argmax(filtered_power)
            return filtered_freqs[peak_idx], filtered_power[peak_idx]
    else:
        # Find global peak (excluding DC component at freq=0)
        if len(xf) > 1:
            peak_idx = np.argmax(power[1:]) + 1  # Skip DC
            return xf[peak_idx], power[peak_idx]

    return 0.0, 0.0


def measure_periodicity(
    spike_history: list[float],
    dt_ms:float = DEFAULT_DT_MS,
    period_range_ms: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Detect periodic activity via autocorrelation.

    Args:
        spike_history: List of spike counts over time
        dt_ms: Timestep in milliseconds
        period_range_ms: Tuple of (min_period_ms, max_period_ms) to search.
                        If None, searches 10-200ms range.

    Returns:
        Tuple of (period_ms, autocorrelation_strength)
        period_ms: Dominant period in milliseconds
        strength: Autocorrelation coefficient at peak [0, 1]
        Returns (0.0, 0.0) if no periodicity detected

    Example:
        >>> spike_counts = [...]  # L6 activity over 200ms
        >>> period, strength = measure_periodicity(spike_counts)
        >>> print(f"Period: {period:.1f}ms (r={strength:.2f})")
        >>> # For gamma: expect period ~25ms (40 Hz)
    """
    N = len(spike_history)
    if N < 20:
        return 0.0, 0.0

    # Compute autocorrelation
    acorr = correlate(spike_history, spike_history, mode="full")
    acorr = acorr[len(acorr) // 2 :]  # Take positive lags only

    # Normalize by zero-lag value
    if acorr[0] > 0:
        acorr = acorr / acorr[0]
    else:
        return 0.0, 0.0

    # Set period range to search
    if period_range_ms is None:
        min_period_ms, max_period_ms = 10.0, 200.0
    else:
        min_period_ms, max_period_ms = period_range_ms

    # Convert period to lag indices
    min_lag = max(1, int(min_period_ms / dt_ms))
    max_lag = min(len(acorr) - 1, int(max_period_ms / dt_ms))

    if min_lag >= max_lag:
        return 0.0, 0.0

    # Find first peak after lag 0 (avoid trivial peak at lag=0)
    peak_idx = np.argmax(acorr[min_lag:max_lag]) + min_lag
    period_ms = peak_idx * dt_ms
    strength = acorr[peak_idx]

    return period_ms, strength


def power_spectrum(
    spike_history: list[float],
    dt_ms:float = DEFAULT_DT_MS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute full power spectrum of spike train.

    Args:
        spike_history: List of spike counts over time
        dt_ms: Timestep in milliseconds

    Returns:
        Tuple of (frequencies_hz, power_density)
        frequencies_hz: Array of frequencies in Hz
        power_density: Spectral power at each frequency

    Example:
        >>> freqs, power = power_spectrum(l6_spike_counts)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(freqs, power)
        >>> plt.xlabel('Frequency (Hz)')
        >>> plt.ylabel('Power')
        >>> plt.xlim(0, 100)  # Focus on gamma range
    """
    N = len(spike_history)

    # FFT
    yf = fft(spike_history)
    xf = fftfreq(N, dt_ms / 1000.0)  # Hz

    # Positive frequencies only
    xf = xf[: N // 2]
    power = 2.0 / N * np.abs(yf[: N // 2])

    return xf, power


def detect_gamma_oscillation(
    spike_history: list[float],
    dt_ms:float = DEFAULT_DT_MS,
    gamma_range: Tuple[float, float] = (30.0, 80.0),
    min_power_threshold: float = 0.5,
) -> bool:
    """Detect if spike train shows gamma oscillation (30-80 Hz).

    Args:
        spike_history: List of spike counts over time
        dt_ms: Timestep in milliseconds
        gamma_range: Frequency range for gamma (default: 30-80 Hz)
        min_power_threshold: Minimum spectral power to consider valid

    Returns:
        True if gamma oscillation detected, False otherwise

    Example:
        >>> l6_spikes = [...]  # 200ms of L6 activity
        >>> has_gamma = detect_gamma_oscillation(l6_spikes)
        >>> assert has_gamma, "L6→TRN loop should generate gamma"
    """
    freq, power = measure_oscillation(
        spike_history,
        dt_ms=dt_ms,
        freq_range=gamma_range,
    )

    if freq == 0.0:
        return False

    return power >= min_power_threshold


def detect_theta_oscillation(
    spike_history: list[float],
    dt_ms:float = DEFAULT_DT_MS,
    theta_range: Tuple[float, float] = (4.0, 12.0),
    min_power_threshold: float = 0.3,
) -> bool:
    """Detect if spike train shows theta oscillation (4-12 Hz).

    Args:
        spike_history: List of spike counts over time
        dt_ms: Timestep in milliseconds
        theta_range: Frequency range for theta (default: 4-12 Hz)
        min_power_threshold: Minimum spectral power to consider valid

    Returns:
        True if theta oscillation detected, False otherwise

    Example:
        >>> ca3_spikes = [...]  # 500ms of CA3 activity
        >>> has_theta = detect_theta_oscillation(ca3_spikes)
        >>> # May not be exactly 8 Hz without septum coordination
    """
    freq, power = measure_oscillation(
        spike_history,
        dt_ms=dt_ms,
        freq_range=theta_range,
    )

    if freq == 0.0:
        return False

    return power >= min_power_threshold
