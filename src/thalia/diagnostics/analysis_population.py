"""Population statistics — firing rates, ISI, Fano factor, SFA, burst detection.

These module-level functions accept a :class:`DiagnosticsRecorder` and operate on
its pre-allocated buffers without belonging to the class itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit as sp_curve_fit

from .bio_ranges import bio_range
from .diagnostics_types import PopulationStats

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


# =============================================================================
# NETWORK STATE CLASSIFIER
# =============================================================================


def _classify_network_state(ps: PopulationStats) -> str:
    """Return a joint network-state label from CV, FF, pairwise ρ and burst events.

    States
    ------
    ``"AI"``    Asynchronous-Irregular  — CV 0.7–1.3, FF ~1, ρ < 0.15
    ``"SR"``    Synchronous-Regular     — CV < 0.5,  FF < 0.5, ρ > 0.3
    ``"SI"``    Synchronous-Irregular   — CV > 1.3,  FF > 2,   ρ > 0.2
    ``"burst"`` Epileptiform            — burst_events > 0.05 or CV > 2.0
    ``"unknown"`` insufficient data (< 20 spikes, or CV/FF are NaN)
    """
    if ps.total_spikes < 20:
        return "unknown"
    cv = ps.isi_cv
    ff = ps.per_neuron_ff
    rho = ps.pairwise_correlation
    if np.isnan(cv) or np.isnan(ff):
        return "unknown"

    # Epileptiform / burst
    if (
        ps.fraction_burst_events > 0.05
        or cv > 2.0
        or ff > 4.0
    ):
        return "burst"

    rho_ok = not np.isnan(rho)

    # Synchronous-Regular
    if cv < 0.5 and ff < 0.5 and (rho_ok and rho > 0.3):
        return "SR"

    # Synchronous-Irregular
    if cv > 1.3 and ff > 2.0 and (rho_ok and rho > 0.2):
        return "SI"

    # Asynchronous-Irregular — require positive pairwise-correlation evidence;
    # NaN ρ (e.g. populations with n < 2) must not be treated as "uncorrelated".
    if 0.7 <= cv <= 1.3 and 0.5 <= ff <= 2.5 and rho_ok and rho < 0.15:
        return "AI"

    return "unknown"


# =============================================================================
# POPULATION STATISTICS  —  result containers
# =============================================================================


class _FrStatsResult(NamedTuple):
    """Per-neuron firing-rate summary returned by :func:`_compute_fr_stats`."""
    mean_fr_hz: float
    std_fr_hz: float
    fraction_silent: float
    fraction_hyperactive: float
    hyperactive_threshold_hz: float
    hist: np.ndarray
    edges: np.ndarray


class _IsiResult(NamedTuple):
    """ISI statistics returned by :func:`_compute_isi_stats`."""
    isi_mean_ms: float
    isi_cv: float
    fraction_bursting: float
    fraction_refractory_violations: float
    isi_cv2: float
    fraction_isi_lt_80ms: float


_NAN_ISI_RESULT = _IsiResult(
    isi_mean_ms=np.nan,
    isi_cv=np.nan,
    fraction_bursting=np.nan,
    fraction_refractory_violations=np.nan,
    isi_cv2=np.nan,
    fraction_isi_lt_80ms=np.nan,
)

_NAN_FR_RESULT = _FrStatsResult(
    mean_fr_hz=0.0,
    std_fr_hz=0.0,
    fraction_silent=1.0,
    fraction_hyperactive=0.0,
    hyperactive_threshold_hz=np.nan,
    hist=np.zeros(20, dtype=np.float32),
    edges=np.zeros(21, dtype=np.float32),
)

# NOTE: Fano-factor and pairwise-correlation sampling previously used module-level
# RNGs (_RNG_FF, _RNG_PC) that advanced state across analysis calls.  In sweep mode
# this made patterns 2–4 sample different neuron subsets than pattern 1, breaking
# cross-pattern comparability.  Both functions now create a call-local RNG seeded
# deterministically from the population identity.


# =============================================================================
# POPULATION STATISTICS  —  sub-functions
# =============================================================================


def _compute_fr_stats(
    rec: "DiagnosticsRecorder",
    pop_idx: int,
    T: int,
    total_spikes: int,
) -> _FrStatsResult:
    """Compute per-neuron firing rates, histogram, and silent/hyperactive fractions."""
    if total_spikes == 0:
        return _NAN_FR_RESULT
    rn, pn = rec._pop_keys[pop_idx]
    n_neurons = int(rec._pop_sizes[pop_idx])
    sim_s = T * rec.dt_ms / 1000.0
    key = (rn, pn)

    if rec.config.mode == "full" and key in rec._spike_times:
        spike_counts_per_neuron = np.array(
            [len(times) for times in rec._spike_times[key]], dtype=np.float32
        )
        # Account for neurons with no spikes
        if len(spike_counts_per_neuron) < n_neurons:
            pad = np.zeros(n_neurons - len(spike_counts_per_neuron), dtype=np.float32)
            spike_counts_per_neuron = np.concatenate([spike_counts_per_neuron, pad])
    else:
        # Stats mode: use the per-neuron cumulative spike counts recorded during
        # simulation.  This correctly captures silent and hyperactive sub-populations
        # rather than assuming all neurons fire at the population average rate.
        spike_counts_per_neuron = rec._per_neuron_spike_counts[pop_idx][:n_neurons].astype(
            np.float32
        )

    fr_per_neuron_hz = spike_counts_per_neuron / max(sim_s, 1e-9)
    mean_fr_hz = float(fr_per_neuron_hz.mean())
    std_fr_hz = float(fr_per_neuron_hz.std())
    fraction_silent = float((fr_per_neuron_hz < 0.01).mean())
    # Use 1.5× bio upper bound as the hyperactive threshold so populations
    # with legitimately high baseline rates (Purkinje, GPe, PV) are not
    # spuriously flagged.  Cap at 200 Hz to catch populations in true
    # pathological overdrive regardless of how wide their bio range is.
    # Fall back to 50 Hz for unknown populations.
    _bio_upper = bio_range(rn, pn)
    hyperactive_threshold_hz = min(_bio_upper[1] * 1.5, 200.0) if _bio_upper is not None else 50.0
    fraction_hyperactive = float((fr_per_neuron_hz > hyperactive_threshold_hz).mean())
    max_fr = max(float(fr_per_neuron_hz.max()), 1.0)
    hist, edges = np.histogram(fr_per_neuron_hz, bins=20, range=(0.0, max_fr))
    return _FrStatsResult(
        mean_fr_hz=mean_fr_hz,
        std_fr_hz=std_fr_hz,
        fraction_silent=fraction_silent,
        fraction_hyperactive=fraction_hyperactive,
        hyperactive_threshold_hz=hyperactive_threshold_hz,
        hist=hist.astype(np.float32),
        edges=edges.astype(np.float32),
    )


def _compute_isi_stats(
    spike_times_list: List[List[int]],
    dt_ms: float,
) -> _IsiResult:
    """Compute ISI mean, CV, CV₂, burst fraction, refractory violations, DA marker.

    Parameters
    ----------
    spike_times_list:
        Per-neuron spike-time lists (integer timestep indices).
        Equivalent to ``rec._spike_times[(rn, pn)]``.
    dt_ms:
        Simulation timestep in milliseconds.
    """
    # Collect integer-step diffs per neuron, then concatenate once and scale.
    # Avoids the O(N_spikes) Python list overhead of .tolist() + list.extend().
    chunks: List[np.ndarray] = []
    for times in spike_times_list:
        if len(times) >= 2:
            chunks.append(np.diff(times))
    if not chunks:
        return _NAN_ISI_RESULT

    arr = np.concatenate(chunks).astype(np.float32) * dt_ms  # ms
    isi_mean_ms = float(arr.mean())
    isi_cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else np.nan
    frac_burst = float((arr < 10.0).mean())
    # Refractory period violation — ISI < 2 ms signals a missing refractory bug.
    frac_refrac = float((arr < 2.0).mean())
    # CV₂ — local irregularity, insensitive to rate non-stationarity.
    # Computed per neuron (≥ 3 spikes required) and averaged across neurons so that
    # cross-neuron ISI boundaries cannot contaminate consecutive-pair differences.
    cv2_vals: List[float] = []
    for times in spike_times_list:
        if len(times) >= 3:
            arr_n = np.diff(np.array(times, dtype=np.float64)) * dt_ms  # ms
            diffs_n = np.abs(np.diff(arr_n))
            sums_n = arr_n[:-1] + arr_n[1:]
            valid_n = sums_n > 0
            if valid_n.any():
                cv2_vals.append(float(np.mean(2.0 * diffs_n[valid_n] / sums_n[valid_n])))
    isi_cv2 = float(np.mean(cv2_vals)) if cv2_vals else np.nan
    # Fraction of ISIs < 80 ms for DA burst-mode detection.
    frac_isi_lt_80ms = float((arr < 80.0).mean())
    return _IsiResult(
        isi_mean_ms=isi_mean_ms,
        isi_cv=isi_cv,
        fraction_bursting=frac_burst,
        fraction_refractory_violations=frac_refrac,
        isi_cv2=isi_cv2,
        fraction_isi_lt_80ms=frac_isi_lt_80ms,
    )


def _compute_fano_factor(
    spike_times_list: Optional[List[List[int]]],
    pop_counts_col: np.ndarray,
    n_neurons: int,
    T: int,
    ff_bin_steps: int,
    is_full_mode: bool,
    rn: str,
    pn: str,
) -> float:
    """Fano factor (var/mean of binned spike counts).

    Full mode: mean per-neuron FF averaged over up to 50 sampled neurons.
    This is unaffected by between-neuron correlations, unlike the population
    FF which scales as ``1 + (N−1)·ρ``.

    Stats mode: population-level FF from binned population spike sums.

    Parameters
    ----------
    spike_times_list:
        Per-neuron spike-time lists, or ``None`` to force the stats-mode path.
    pop_counts_col:
        Population spike counts per timestep, shape ``[≥T]``.
    n_neurons:
        Population size.
    T:
        Number of recorded timesteps.
    ff_bin_steps:
        Bin width in timesteps (typically ``int(50 / dt_ms)``).
    is_full_mode:
        Whether the recorder is in ``"full"`` mode.
    rn:
        Region name — used to seed the call-local RNG deterministically.
    pn:
        Population name — used to seed the call-local RNG deterministically.
    """
    n_ff_bins = T // ff_bin_steps
    if n_ff_bins < 2:
        return np.nan

    if is_full_mode and spike_times_list is not None:
        n_sample_ff = min(50, n_neurons)
        rng_ff = np.random.default_rng(seed=hash((rn, pn, "ff")) & 0xFFFFFFFF)
        ff_sample_idxs = rng_ff.choice(n_neurons, size=n_sample_ff, replace=False)
        per_neuron_ff: List[float] = []
        for nidx in ff_sample_idxs:
            bins = np.zeros(n_ff_bins, dtype=np.float64)
            for spike_t in spike_times_list[nidx]:
                b = spike_t // ff_bin_steps
                if 0 <= b < n_ff_bins:
                    bins[b] += 1.0
            m = bins.mean()
            if m > 0:
                per_neuron_ff.append(bins.var() / m)
        return float(np.mean(per_neuron_ff)) if per_neuron_ff else np.nan
    else:
        # Stats mode: vectorised population-level FF (no Python loop).
        pop_counts_flat = pop_counts_col[: n_ff_bins * ff_bin_steps]
        ff_counts = pop_counts_flat.reshape(n_ff_bins, ff_bin_steps).sum(axis=1).astype(
            np.float64
        )
        ff_mean = ff_counts.mean()
        return float(ff_counts.var() / ff_mean) if ff_mean > 0 else np.nan


def _compute_pairwise_correlation(
    spike_times_list: List[List[int]],
    n_neurons: int,
    T: int,
    corr_bin_steps: int,
    rn: str = "",
    pn: str = "",
) -> float:
    """Mean Pearson r across randomly sampled neuron pairs (full mode only).

    Bins each neuron's spike train into ``corr_bin_steps``-wide windows and
    computes Pearson r for up to C(30, 2) = 435 pairs.

    Parameters
    ----------
    spike_times_list:
        Per-neuron spike-time lists from ``rec._spike_times[(rn, pn)]``.
    n_neurons:
        Population size.
    T:
        Number of recorded timesteps.
    corr_bin_steps:
        Bin width in timesteps (typically ``int(100 / dt_ms)``).
    rn:
        Region name — used to seed the call-local RNG deterministically.
    pn:
        Population name — used to seed the call-local RNG deterministically.
    """
    if n_neurons < 2:
        return np.nan
    n_corr_bins = T // corr_bin_steps
    if n_corr_bins < 2:
        return np.nan

    n_sample = min(30, n_neurons)
    rng_pc = np.random.default_rng(seed=hash((rn, pn, "pc")) & 0xFFFFFFFF)
    sample_idxs = rng_pc.choice(n_neurons, size=n_sample, replace=False)
    spike_mat = np.zeros((n_sample, n_corr_bins), dtype=np.float64)
    for i, nidx in enumerate(sample_idxs):
        for spike_t in spike_times_list[nidx]:
            bin_idx = spike_t // corr_bin_steps
            if 0 <= bin_idx < n_corr_bins:
                spike_mat[i, bin_idx] += 1.0

    # Compute the correlation matrix over *active* neurons only, but normalise
    # the sum by the *total* C(n_sample, 2) pairs (including silent–X pairs,
    # which contribute 0 correlation).  This prevents inflating the reported
    # correlation for sparsely active populations where only a minority fire.
    n_sample = spike_mat.shape[0]
    n_total_pairs = n_sample * (n_sample - 1) // 2
    if n_total_pairs == 0:
        return np.nan
    row_stds = spike_mat.std(axis=1)
    active = spike_mat[row_stds > 1e-9]
    n_active = active.shape[0]
    if n_active == 0:
        # All neurons silent — correlation is undefined (not "zero").
        return np.nan
    if n_active == 1:
        # Only silent–X pairs exist; their correlation is zero by definition.
        return 0.0
    corr = np.corrcoef(active)  # [n_active, n_active]
    active_pairs_sum = float(corr[np.triu_indices(n_active, k=1)].sum())
    return active_pairs_sum / n_total_pairs


def _compute_burst_events(
    pop_counts_col: np.ndarray,
    n_neurons: int,
    T: int,
    burst_win_steps: int,
    burst_coactivation_fraction: float,
) -> float:
    """Fraction of burst windows exceeding the Binomial(N, p) + 2σ threshold.

    A burst event is any ``burst_win_steps``-wide window whose total spike count
    exceeds ``p·N + 2·√(p·(1−p)·N)`` where *p* = *burst_coactivation_fraction*.

    Using the Binomial+2σ threshold guards against false positives in small
    populations: for N=10, p=0.30 → mean+2σ ≈ 5.9 (60 % co-fire required) vs.
    the naïve 30 % flat threshold of only 3 neurons.

    Parameters
    ----------
    pop_counts_col:
        Population spike counts per timestep, shape ``[≥T]``.
    n_neurons:
        Population size.
    T:
        Number of recorded timesteps.
    burst_win_steps:
        Burst-window width in timesteps (typically ``int(20 / dt_ms)``).
    burst_coactivation_fraction:
        Expected spontaneous co-activation fraction (Binomial mean parameter p).
        Passed from :attr:`HealthThresholds.burst_coactivation_fraction` (via ``DiagnosticsConfig.thresholds``).
    """
    n_burst_wins = T // burst_win_steps
    if n_burst_wins < 1 or n_neurons <= 0:
        return np.nan
    p = burst_coactivation_fraction
    burst_threshold = p * n_neurons + 2.0 * np.sqrt(p * (1.0 - p) * n_neurons)
    windows = np.add.reduceat(
        pop_counts_col[: n_burst_wins * burst_win_steps],
        np.arange(0, n_burst_wins * burst_win_steps, burst_win_steps),
    )
    return float((windows >= burst_threshold).mean())


def _compute_sfa_index(
    pop_counts_col: np.ndarray,
    n_neurons: int,
    T: int,
    dt_ms: float,
) -> float:
    """Early/late firing-rate ratio (SFA index, E3).

    Compares mean FR in the first 25 % of the recording vs the last 25 %.
    ``sfa_index > 1`` → adapting; ``≈ 1`` → non-adapting (PV, FSI, TAN).
    Works in both full and stats mode (uses population spike counts).
    Returns ``nan`` if fewer than 4 total spikes.
    """
    total = int(pop_counts_col[:T].sum())
    if total < 4 or n_neurons <= 0:
        return np.nan
    quarter = max(1, T // 4)
    dur_s = quarter * dt_ms / 1000.0
    early_rate = float(pop_counts_col[:quarter].sum()) / n_neurons / dur_s
    late_rate = float(pop_counts_col[T - quarter:T].sum()) / n_neurons / dur_s
    if late_rate > 0:
        return min(early_rate / late_rate, 20.0)
    if early_rate > 0:
        return 20.0  # complete silence by end — maximally adapted
    return np.nan


def _compute_sfa_tau(
    pop_counts_col: np.ndarray,
    n_neurons: int,
    T: int,
    dt_ms: float,
    rate_bin_ms: float,
) -> float:
    """Fit FR(t) = FR_ss + (FR_0 − FR_ss)·exp(−t/τ) and return τ in ms.

    Returns ``nan`` if fewer than 20 total spikes, fewer than 10 rate bins,
    or if ``scipy.optimize.curve_fit`` does not converge.
    """
    total = int(pop_counts_col[:T].sum())
    if total < 20 or n_neurons <= 0:
        return np.nan
    bin_steps = max(1, int(rate_bin_ms / dt_ms))
    n_bins = T // bin_steps
    if n_bins < 10:
        return np.nan
    rate_trace = (
        pop_counts_col[: n_bins * bin_steps]
        .reshape(n_bins, bin_steps).sum(axis=1).astype(np.float64)
        / max(n_neurons, 1)
        / (bin_steps * dt_ms / 1000.0)
    )
    t_trace = np.arange(n_bins, dtype=np.float64) * rate_bin_ms
    fr0_est = float(rate_trace[:3].mean())
    ss_est = float(rate_trace[-3:].mean())
    # Require FR to decrease over the recording — if it stays flat or ramps up
    # the neuron is not adapting and a decay fit is meaningless.
    if fr0_est <= ss_est:
        return np.nan
    try:
        def _exp_decay(t: np.ndarray, fr_ss: float, fr0: float, tau: float) -> np.ndarray:
            return fr_ss + (fr0 - fr_ss) * np.exp(-t / tau)  # type: ignore[return-value]
        popt, _ = sp_curve_fit(
            _exp_decay, t_trace, rate_trace,
            p0=[ss_est, fr0_est, 100.0],
            bounds=([0.0, 0.0, 1.0], [np.inf, np.inf, 5000.0]),
            maxfev=1000,
        )
        return float(popt[2])
    except (ValueError, RuntimeError):
        return np.nan


def _compute_da_burst_rate(
    spike_times_list: List[List[int]],
    dt_ms: float,
    duration_ms: float,
) -> float:
    """Count DA-style burst events per second across all neurons.

    A burst event is ≥3 consecutive ISIs < 80 ms followed by a pause ISI > 200 ms.
    Events are summed across neurons and divided by the recording duration in seconds.
    Returns NaN if no neurons have sufficient spikes.
    """
    if duration_ms <= 0:
        return np.nan
    total_events = 0
    any_eligible = False
    for times in spike_times_list:
        if len(times) < 5:  # need ≥4 ISIs to form a burst + pause
            continue
        any_eligible = True
        isis_ms = np.diff(np.array(times, dtype=np.float64)) * dt_ms
        i = 0
        while i < len(isis_ms) - 1:  # need at least one ISI after position i
            # Count consecutive intra-burst ISIs (< 80 ms)
            burst_len = 0
            j = i
            while j < len(isis_ms) and isis_ms[j] < 80.0:
                burst_len += 1
                j += 1
            # Burst is valid if ≥3 ISIs < 80 ms and followed by a pause > 200 ms
            if burst_len >= 3 and j < len(isis_ms) and isis_ms[j] > 200.0:
                total_events += 1
                i = j + 1  # skip past the pause
            else:
                i += 1
    if not any_eligible:
        return np.nan
    return float(total_events) / (duration_ms / 1000.0)


def _compute_updown_durations(
    rate_trace: np.ndarray,
    dt_ms: float,
) -> Tuple[float, float]:
    """Mean up-state and down-state epoch durations (ms) via run-length analysis.

    Binarises the population spike-count trace at its grand mean, then measures
    contiguous epoch lengths.  Using population rate (all N neurons) rather than
    the mean of O(8) sampled voltage traces gives a much more reliable signal,
    consistent with how Steriade et al. (2001) measured up/down states from
    population-averaged LFP.

    Parameters
    ----------
    rate_trace:
        Population spike counts per timestep, shape ``[T]``.
    dt_ms:
        Simulation timestep in milliseconds.

    Returns ``(up_dur_ms, down_dur_ms)``; both NaN when insufficient data.
    """
    trace = rate_trace.astype(np.float64)
    if len(trace) < 20 or trace.sum() == 0:
        return np.nan, np.nan
    threshold = float(trace.mean())
    up = trace >= threshold
    # Run-length encode using diff-of-cumsum trick.
    changes = np.diff(up.astype(np.int8), prepend=np.int8(int(up[0]) ^ 1))
    run_starts = np.where(changes != 0)[0]
    run_lengths = np.diff(np.append(run_starts, len(up)))
    run_states = up[run_starts]
    up_durs = run_lengths[run_states] * dt_ms
    down_durs = run_lengths[~run_states] * dt_ms
    up_dur_ms = float(up_durs.mean()) if len(up_durs) > 0 else np.nan
    down_dur_ms = float(down_durs.mean()) if len(down_durs) > 0 else np.nan
    return up_dur_ms, down_dur_ms


def _compute_voltage_bimodality(
    voltages_pop: np.ndarray,
) -> float:
    """Sarle's bimodality coefficient on per-neuron time-averaged voltages.

    Computes the **time-mean voltage for each sampled neuron** (shape ``[V]``),
    then applies Sarle's BC to the distribution of those V means.  This ensures
    every data point is spatially independent — one per neuron — rather than
    recycling the same V neurons across T timesteps and inflating sample size
    without improving spatial coverage.

    BC = (skewness² + 1) / kurtosis.  BC > 0.555 indicates bimodality
    consistent with up/down state dynamics.

    Requires ``voltage_sample_size ≥ 20`` sampled neurons; returns NaN with
    fewer, because the BC of only 8 means is highly sensitive to individual
    neuron outliers.

    Parameters
    ----------
    voltages_pop:
        Voltage samples for one population, shape ``[T, V]`` (may contain
        ``nan`` for unsampled neurons).
    """
    # Per-neuron time-mean: shape [V].  nanmean handles partially-NaN columns.
    per_neuron_mean = np.nanmean(voltages_pop, axis=0)  # [V]
    valid_v = per_neuron_mean[~np.isnan(per_neuron_mean)]
    if len(valid_v) < 20:
        return np.nan
    mu = float(valid_v.mean())
    centred = valid_v.astype(np.float64) - mu
    m2 = float(np.mean(centred ** 2))
    m3 = float(np.mean(centred ** 3))
    m4 = float(np.mean(centred ** 4))
    if m2 > 0 and m4 > 0:
        skewness = m3 / (m2 ** 1.5)
        # Sarle (1984): BC = (skew² + 1) / κ  where κ = m4/m2² (total kurtosis).
        # excess_kurtosis = m4/m2² − 3, so  excess_kurtosis + 3 = m4/m2².
        # The formula is therefore CORRECT — denominator IS total kurtosis.
        excess_kurtosis = m4 / (m2 ** 2) - 3.0
        denominator = excess_kurtosis + 3.0  # = m4/m2² = total kurtosis
        if denominator > 0:
            return float((skewness ** 2 + 1.0) / denominator)
    return np.nan


# =============================================================================
# POPULATION STATISTICS  —  orchestrator
# =============================================================================


def compute_population_stats(
    rec: "DiagnosticsRecorder", pop_idx: int, T: int
) -> PopulationStats:
    """Compute all statistics for a single population by delegating to sub-functions."""
    rn, pn = rec._pop_keys[pop_idx]
    n_neurons = int(rec._pop_sizes[pop_idx])
    pop_counts_col = rec._pop_spike_counts[:T, pop_idx]
    total_spikes = int(pop_counts_col.sum())
    key = (rn, pn)
    is_full = rec.config.mode == "full"
    spike_times_list: List[List[int]] | None = (
        rec._spike_times.get(key) if is_full else None
    )

    # Per-neuron firing rates and histogram
    fr = _compute_fr_stats(rec, pop_idx, T, total_spikes)

    # ISI statistics (full mode only)
    isi = (
        _compute_isi_stats(spike_times_list, rec.dt_ms)
        if is_full and spike_times_list is not None
        else _NAN_ISI_RESULT
    )

    # Fano factor
    ff_bin_steps = max(1, int(50.0 / rec.dt_ms))
    _ff_value = _compute_fano_factor(
        spike_times_list, pop_counts_col, n_neurons, T, ff_bin_steps, is_full,
        rn=rn, pn=pn,
    )
    per_neuron_ff = _ff_value if is_full else np.nan
    population_ff = _ff_value if not is_full else np.nan

    # Pairwise correlation (full mode, ≥ 2 neurons)
    pairwise_correlation = (
        _compute_pairwise_correlation(
            spike_times_list, n_neurons, T,
            corr_bin_steps=max(1, int(100.0 / rec.dt_ms)),
            rn=rn, pn=pn,
        )
        if is_full and spike_times_list is not None and n_neurons >= 2
        else np.nan
    )

    # Epileptiform burst events
    fraction_burst_events = _compute_burst_events(
        pop_counts_col, n_neurons, T,
        burst_win_steps=max(1, int(20.0 / rec.dt_ms)),
        burst_coactivation_fraction=rec.config.thresholds.burst_coactivation_fraction,
    )

    # DA-style burst event rate (full mode)
    da_burst_events_per_s = (
        _compute_da_burst_rate(spike_times_list, rec.dt_ms, T * rec.dt_ms)
        if is_full and spike_times_list is not None
        else np.nan
    )

    # SFA index
    sfa_index = _compute_sfa_index(pop_counts_col, n_neurons, T, rec.dt_ms)

    # SFA time constant
    sfa_tau_ms = _compute_sfa_tau(
        pop_counts_col, n_neurons, T, rec.dt_ms, rec.config.rate_bin_ms
    )

    # Voltage bimodality (full mode only)
    voltage_bimodality = (
        _compute_voltage_bimodality(rec._voltages[:T, pop_idx, :])
        if is_full and rec._voltages is not None
        else np.nan
    )

    # Up/down state epoch durations — use population rate (always available).
    if total_spikes >= 20:
        up_state_duration_ms, down_state_duration_ms = _compute_updown_durations(
            pop_counts_col, rec.dt_ms
        )
    else:
        up_state_duration_ms = np.nan
        down_state_duration_ms = np.nan

    # Apical AMPA conductance (TwoCompartmentLIF full mode only).
    if is_full and rec._g_apical_samples is not None:
        cond_step = rec._cond_sample_step
        slab = rec._g_apical_samples[:cond_step, pop_idx, :]
        mean_g_exc_apical = float(np.nanmean(slab)) if cond_step > 0 else np.nan
    else:
        mean_g_exc_apical = np.nan

    # Network state classifier — build the final object once, then
    # fill in network_state in place (PopulationStats is not frozen).
    ps = PopulationStats(
        region_name=rn, population_name=pn, n_neurons=n_neurons,
        mean_fr_hz=fr.mean_fr_hz, std_fr_hz=fr.std_fr_hz,
        fraction_silent=fr.fraction_silent, fraction_hyperactive=fr.fraction_hyperactive,
        hyperactive_threshold_hz=fr.hyperactive_threshold_hz,
        total_spikes=total_spikes,
        isi_mean_ms=isi.isi_mean_ms, isi_cv=isi.isi_cv,
        fraction_bursting=isi.fraction_bursting,
        fraction_refractory_violations=isi.fraction_refractory_violations,
        isi_cv2=isi.isi_cv2, fraction_isi_lt_80ms=isi.fraction_isi_lt_80ms,
        sfa_index=sfa_index, per_neuron_ff=per_neuron_ff, population_ff=population_ff,
        pairwise_correlation=pairwise_correlation,
        fraction_burst_events=fraction_burst_events,
        da_burst_events_per_s=da_burst_events_per_s,
        fr_histogram=fr.hist, fr_histogram_edges=fr.edges,
        bio_range_hz=bio_range(rn, pn),
        network_state="unknown",
        voltage_bimodality=voltage_bimodality,
        up_state_duration_ms=up_state_duration_ms,
        down_state_duration_ms=down_state_duration_ms,
        sfa_tau_ms=sfa_tau_ms,
        mean_g_exc_apical=mean_g_exc_apical,
    )
    ps.network_state = _classify_network_state(ps)
    return ps
