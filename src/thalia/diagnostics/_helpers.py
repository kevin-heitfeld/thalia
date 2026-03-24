"""Shared helpers for diagnostics modules."""

from __future__ import annotations

from typing import Callable, List, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def deterministic_rng(region: str, pop: str, mode: str) -> np.random.Generator:
    """Create a deterministic, call-local RNG for reproducible neuron sampling.

    Seeds are derived from (region, pop, mode) so different analysis functions
    draw independent but reproducible subsets from the same population.
    """
    return np.random.default_rng(seed=hash((region, pop, mode)) & 0xFFFFFFFF)


def bin_counts_1d(counts: np.ndarray, n_bins: int, bin_steps: int) -> np.ndarray:
    """Bin a 1-D timestep-count vector into *n_bins* bins of width *bin_steps*.

    Only the first ``n_bins * bin_steps`` elements are used; any trailing
    remainder is discarded.

    Returns:
        1-D float64 array of length *n_bins*.
    """
    return (
        counts[: n_bins * bin_steps]
        .reshape(n_bins, bin_steps)
        .sum(axis=1)
        .astype(np.float64)
    )


def bin_counts_2d(counts: np.ndarray, n_bins: int, bin_steps: int) -> np.ndarray:
    """Bin a 2-D (timesteps x columns) count matrix, preserving the column dimension.

    Only the first ``n_bins * bin_steps`` rows are used; any trailing
    remainder is discarded.

    Returns:
        2-D float64 array of shape ``(n_bins, n_cols)``.
    """
    n_cols = counts.shape[1]
    return (
        counts[: n_bins * bin_steps]
        .reshape(n_bins, bin_steps, n_cols)
        .sum(axis=1)
        .astype(np.float64)
    )


# ─────────────────────────────────────────────────────────────────────────────
# E/I conductance grouping
# ─────────────────────────────────────────────────────────────────────────────
# Canonical definitions: AMPA + NMDA = excitatory, GABA-A + GABA-B = inhibitory.
# All helpers are NaN-aware: a NaN component is treated as absent (skipped).


def combine_excitatory(g_ampa: float, g_nmda: float) -> float:
    """AMPA + NMDA, skipping NaN components."""
    if np.isnan(g_ampa):
        return g_nmda
    if np.isnan(g_nmda):
        return g_ampa
    return g_ampa + g_nmda


def combine_inhibitory(g_gaba_a: float, g_gaba_b: float) -> float:
    """GABA-A + GABA-B, skipping NaN components."""
    if np.isnan(g_gaba_a):
        return g_gaba_b
    if np.isnan(g_gaba_b):
        return g_gaba_a
    return g_gaba_a + g_gaba_b


def compute_ei_ratio(g_ampa: float, g_nmda: float, g_inh: float) -> float:
    """(AMPA + NMDA) / inhibitory.  Returns NaN when inhibitory ≤ 0."""
    exc = combine_excitatory(g_ampa, g_nmda)
    if np.isnan(exc) or np.isnan(g_inh) or g_inh <= 0:
        return float("nan")
    return exc / g_inh


def compute_nmda_fraction(g_ampa: float, g_nmda: float) -> float:
    """NMDA / (AMPA + NMDA).  Returns NaN when total excitatory ≤ 0 or inputs are NaN."""
    if np.isnan(g_ampa) or np.isnan(g_nmda):
        return float("nan")
    exc_total = g_ampa + g_nmda
    if exc_total <= 0:
        return float("nan")
    return g_nmda / exc_total


def compute_ei_current_ratio(
    g_ampa: float,
    g_nmda: float,
    g_gaba_a: float,
    g_gaba_b: float,
    E_E: float,
    E_nmda: float,
    E_I: float,
    E_GABA_B: float,
    V: float,
) -> float:
    """Driving-force-weighted E/I current ratio.

    Computes I_exc / |I_inh| where currents account for reversal potentials:
        I_exc  = g_ampa·(E_E − V) + g_nmda·(E_nmda − V)
        I_inh  = g_gaba_a·(E_I − V) + g_gaba_b·(E_GABA_B − V)

    Returns NaN when inputs are insufficient or inhibitory current ≈ 0.
    """
    # Excitatory current (positive when V < E_E)
    i_exc = 0.0
    has_exc = False
    if not np.isnan(g_ampa) and not np.isnan(E_E):
        i_exc += g_ampa * (E_E - V)
        has_exc = True
    if not np.isnan(g_nmda) and not np.isnan(E_nmda):
        i_exc += g_nmda * (E_nmda - V)
        has_exc = True

    # Inhibitory current (negative when V > E_I)
    i_inh = 0.0
    has_inh = False
    if not np.isnan(g_gaba_a) and not np.isnan(E_I):
        i_inh += g_gaba_a * (E_I - V)
        has_inh = True
    if not np.isnan(g_gaba_b) and not np.isnan(E_GABA_B):
        i_inh += g_gaba_b * (E_GABA_B - V)
        has_inh = True

    if not has_exc or not has_inh or np.isnan(V):
        return float("nan")

    abs_inh = abs(i_inh)
    if abs_inh < 1e-12:
        return float("nan")

    return i_exc / abs_inh


# ─────────────────────────────────────────────────────────────────────────────
# Correlation helpers
# ─────────────────────────────────────────────────────────────────────────────


def safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient with NaN and zero-variance safety.

    Returns ``NaN`` when either signal has near-zero standard deviation
    (< 1e-9) — i.e. constant or silent — rather than raising or returning
    a misleading value.
    """
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cross_correlate_at_lags(
    x: np.ndarray,
    y: np.ndarray,
    lag_lo: int,
    lag_hi: int,
) -> Tuple[float, float]:
    """Z-scored Pearson cross-correlation over a range of causal lags.

    Computes Pearson *r* between ``x[:-lag]`` and ``y[lag:]`` for each
    integer lag in ``[lag_lo, lag_hi]``, returning the peak correlation
    and the lag (in timestep indices) at which it occurs.

    Args:
        x: Pre-synaptic signal (1-D, already mean-centred or raw).
        y: Post-synaptic signal (1-D, same length as *x*).
        lag_lo: Minimum lag (inclusive, ≥ 0).
        lag_hi: Maximum lag (inclusive).

    Returns:
        ``(best_r, best_lag)`` — peak Pearson *r* and corresponding lag
        index.  Both are ``NaN`` when either signal has zero variance.
    """
    nan = float("nan")
    T = len(x)
    x_z = x - x.mean()
    y_z = y - y.mean()
    std_x = float(x_z.std())
    std_y = float(y_z.std())
    if std_x < 1e-9 or std_y < 1e-9:
        return nan, nan

    best_corr = -float("inf")
    best_lag = nan
    for lag in range(lag_lo, lag_hi + 1):
        if lag >= T:
            break
        n_overlap = T - lag
        corr = float(np.dot(x_z[:n_overlap], y_z[lag:])) / (std_x * std_y * n_overlap)
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag)

    if np.isinf(best_corr):
        return nan, nan
    return best_corr, best_lag


# ─────────────────────────────────────────────────────────────────────────────
# DA burst-mode detection
# ─────────────────────────────────────────────────────────────────────────────


def is_da_burst_mode(
    rn: str,
    pn: str,
    ps: object,
    config: object,
) -> bool:
    """Return ``True`` if the population (*rn*, *pn*) is a DA neuron in burst mode.

    Checks:
    1. Population tags match DA source / DA neuron tags.
    2. ISI CV exceeds ``config.da_burst_isi_cv``.
    3. Fraction of ISIs < 80 ms exceeds ``config.da_burst_isi_fraction``.
    4. DA burst events are present (or not yet computed).

    *ps* must expose ``isi_cv``, ``fraction_isi_lt_80ms``, and
    ``da_burst_events_per_s`` attributes (a ``PopulationStats`` instance).
    *config* must expose ``da_burst_isi_cv`` and ``da_burst_isi_fraction``
    (a ``HealthThresholds`` instance).
    """
    from .region_tags import DA_NEURON_TAGS, DA_SOURCE_TAGS, matches_any

    if not (matches_any(rn, DA_SOURCE_TAGS) and matches_any(pn, DA_NEURON_TAGS)):
        return False
    isi_cv = getattr(ps, "isi_cv", float("nan"))
    frac = getattr(ps, "fraction_isi_lt_80ms", float("nan"))
    burst_rate = getattr(ps, "da_burst_events_per_s", float("nan"))
    return (
        not np.isnan(isi_cv)
        and isi_cv > getattr(config, "da_burst_isi_cv", 1.0)
        and not np.isnan(frac)
        and frac > getattr(config, "da_burst_isi_fraction", 0.3)
        and (np.isnan(burst_rate) or burst_rate > 0.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signal grouping helpers
# ─────────────────────────────────────────────────────────────────────────────


def nan_safe_compute(
    array: np.ndarray,
    min_count: int,
    fn: Callable[[np.ndarray], T],
    *,
    default: T = float("nan"),  # type: ignore[assignment]
) -> T:
    """Apply *fn* to the non-NaN elements of *array* if at least *min_count* exist.

    Returns *default* (NaN by default) when fewer than *min_count* finite values
    are present.  Avoids the repeated ``vals = a[~isnan(a)]; if len(vals) < N``
    boilerplate that appears in virtually every analysis function.
    """
    clean = array[~np.isnan(array)]
    if len(clean) < min_count:
        return default
    return fn(clean)


def bin_spike_times_to_array(
    st_list: List[List[int]],
    n_bins: int,
    bin_steps: int,
) -> np.ndarray:
    """Accumulate spike times into a binned population-count vector (vectorised).

    Args:
        st_list: Per-neuron lists of spike timestep indices.
        n_bins: Number of output bins.
        bin_steps: Width of each bin in simulation timesteps.

    Returns:
        1-D float64 array of length *n_bins* with summed spike counts per bin.
    """
    vec = np.zeros(n_bins, dtype=np.float64)
    non_empty = [st for st in st_list if st]
    if not non_empty:
        return vec
    all_spikes = np.concatenate([np.asarray(st, dtype=np.int64) for st in non_empty])
    bins_idx = all_spikes // bin_steps
    mask = (bins_idx >= 0) & (bins_idx < n_bins)
    np.add.at(vec, bins_idx[mask], 1.0)
    return vec
