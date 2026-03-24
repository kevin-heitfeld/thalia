"""Tests for analysis_population.py — ISI stats, Fano factor, burst detection, SFA, classify."""

from __future__ import annotations

import numpy as np

from thalia.diagnostics.analysis_population import (
    classify_network_state,
    compute_burst_events,
    compute_fano_factor,
    compute_isi_stats,
    compute_pairwise_correlation,
    compute_sfa_index,
)
from thalia.diagnostics.diagnostics_metrics import PopulationStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poisson_spike_times(
    n_neurons: int,
    rate_hz: float,
    duration_ms: float,
    dt_ms: float = 1.0,
    seed: int = 42,
) -> list[list[int]]:
    """Generate Poisson spike trains as per-neuron lists of timestep indices."""
    rng = np.random.default_rng(seed)
    T = int(duration_ms / dt_ms)
    prob = rate_hz * dt_ms / 1000.0
    result: list[list[int]] = []
    for _ in range(n_neurons):
        spikes = np.where(rng.random(T) < prob)[0]
        result.append(spikes.tolist())
    return result


def _make_regular_spike_times(
    n_neurons: int,
    isi_steps: int,
    duration_steps: int,
    jitter_steps: int = 0,
    seed: int = 42,
) -> list[list[int]]:
    """Generate regularly-spaced spike trains (low CV)."""
    rng = np.random.default_rng(seed)
    result: list[list[int]] = []
    for _ in range(n_neurons):
        times = list(range(isi_steps, duration_steps, isi_steps))
        if jitter_steps > 0:
            times = [max(0, t + int(rng.integers(-jitter_steps, jitter_steps + 1))) for t in times]
        result.append(sorted(set(times)))
    return result


def _make_burst_spike_times(
    n_neurons: int,
    duration_steps: int,
    burst_isi_steps: int = 3,
    spikes_per_burst: int = 5,
    inter_burst_steps: int = 200,
    seed: int = 42,
) -> list[list[int]]:
    """Generate bursty spike trains (high CV, high burst fraction)."""
    rng = np.random.default_rng(seed)
    result: list[list[int]] = []
    for _ in range(n_neurons):
        times: list[int] = []
        t = rng.integers(10, 50)
        while t < duration_steps:
            for s in range(spikes_per_burst):
                spike_t = t + s * burst_isi_steps
                if spike_t < duration_steps:
                    times.append(spike_t)
            t += inter_burst_steps + rng.integers(-20, 20)
        result.append(sorted(set(times)))
    return result


def _pop_counts_from_spike_times(
    spike_times_list: list[list[int]],
    T: int,
) -> np.ndarray:
    """Convert spike times to population spike counts per timestep."""
    counts = np.zeros(T, dtype=np.int32)
    for times in spike_times_list:
        for t in times:
            if 0 <= t < T:
                counts[t] += 1
    return counts


# =====================================================================
# ISI statistics
# =====================================================================


class TestComputeIsiStats:

    def test_poisson_cv_near_one(self) -> None:
        """Poisson process should have ISI CV ≈ 1.0."""
        st = _make_poisson_spike_times(50, rate_hz=20.0, duration_ms=5000.0)
        result = compute_isi_stats(st, dt_ms=1.0)
        assert 0.7 < result.isi_cv < 1.4, f"Poisson ISI CV = {result.isi_cv}"

    def test_regular_cv_low(self) -> None:
        """Regular spike trains should have ISI CV close to zero."""
        st = _make_regular_spike_times(30, isi_steps=50, duration_steps=5000, jitter_steps=1)
        result = compute_isi_stats(st, dt_ms=1.0)
        assert result.isi_cv < 0.2, f"Regular ISI CV = {result.isi_cv}"

    def test_bursty_cv_high(self) -> None:
        """Bursty spike trains should have high ISI CV."""
        st = _make_burst_spike_times(30, duration_steps=5000)
        result = compute_isi_stats(st, dt_ms=1.0)
        assert result.isi_cv > 1.0, f"Bursty ISI CV = {result.isi_cv}"

    def test_burst_fraction_bursty(self) -> None:
        """Bursty trains should have high fraction of ISI < 10ms."""
        st = _make_burst_spike_times(30, duration_steps=5000, burst_isi_steps=3)
        result = compute_isi_stats(st, dt_ms=1.0)
        assert result.fraction_bursting > 0.3, f"Burst fraction = {result.fraction_bursting}"

    def test_burst_fraction_regular(self) -> None:
        """Regular trains with wide ISI should have near-zero burst fraction."""
        st = _make_regular_spike_times(30, isi_steps=100, duration_steps=5000)
        result = compute_isi_stats(st, dt_ms=1.0)
        assert result.fraction_bursting < 0.01

    def test_no_refractory_violations_normal(self) -> None:
        """Normal spike trains shouldn't violate refractory (ISI < 2ms)."""
        st = _make_poisson_spike_times(30, rate_hz=10.0, duration_ms=5000.0)
        # Poisson at 10 Hz with dt=1ms → ISIs mostly ≫ 2ms
        result = compute_isi_stats(st, dt_ms=1.0)
        assert result.fraction_refractory_violations < 0.05

    def test_dead_time_correction(self) -> None:
        """CV_corrected should be smaller than CV when tau_ref > 0."""
        st = _make_regular_spike_times(30, isi_steps=20, duration_steps=5000, jitter_steps=3)
        result_no_ref = compute_isi_stats(st, dt_ms=1.0, tau_ref_ms=0.0)
        result_ref = compute_isi_stats(st, dt_ms=1.0, tau_ref_ms=2.0)
        # With dead-time correction, CV_corrected should increase because
        # effective_mean = mean(ISI) - tau_ref is smaller, making std / effective_mean larger
        assert result_ref.isi_cv_corrected >= result_no_ref.isi_cv_corrected

    def test_empty_spike_times(self) -> None:
        """Empty spike trains should return NaN ISI stats."""
        result = compute_isi_stats([], dt_ms=1.0)
        assert np.isnan(result.isi_cv)
        assert np.isnan(result.isi_mean_ms)

    def test_single_neuron_single_spike(self) -> None:
        """Single spike means no ISIs → NaN."""
        result = compute_isi_stats([[100]], dt_ms=1.0)
        assert np.isnan(result.isi_cv)

    def test_cv2_poisson(self) -> None:
        """CV₂ for Poisson should be ≈ 1.0 (local irregularity measure)."""
        st = _make_poisson_spike_times(50, rate_hz=20.0, duration_ms=5000.0)
        result = compute_isi_stats(st, dt_ms=1.0)
        assert 0.6 < result.isi_cv2 < 1.4, f"CV₂ = {result.isi_cv2}"

    def test_cv2_regular(self) -> None:
        """CV₂ for regular trains should be low."""
        st = _make_regular_spike_times(30, isi_steps=50, duration_steps=5000, jitter_steps=1)
        result = compute_isi_stats(st, dt_ms=1.0)
        assert result.isi_cv2 < 0.3, f"CV₂ = {result.isi_cv2}"


# =====================================================================
# Fano factor
# =====================================================================


class TestComputeFanoFactor:

    def test_poisson_ff_near_one(self) -> None:
        """Poisson process should have Fano factor ≈ 1.0."""
        st = _make_poisson_spike_times(50, rate_hz=15.0, duration_ms=5000.0)
        T = 5000
        counts = _pop_counts_from_spike_times(st, T)
        ff = compute_fano_factor(st, counts, 50, T, ff_bin_steps=50, rn="test", pn="pyr")
        assert 0.5 < ff < 2.0, f"Poisson FF = {ff}"

    def test_regular_ff_low(self) -> None:
        """Regular spike trains should have Fano factor < 1."""
        st = _make_regular_spike_times(50, isi_steps=50, duration_steps=5000, jitter_steps=1)
        T = 5000
        counts = _pop_counts_from_spike_times(st, T)
        ff = compute_fano_factor(st, counts, 50, T, ff_bin_steps=50, rn="test", pn="pyr")
        assert ff < 0.5, f"Regular FF = {ff}"

    def test_bursty_ff_high(self) -> None:
        """Bursty spike trains should have Fano factor > 1."""
        st = _make_burst_spike_times(50, duration_steps=5000)
        T = 5000
        counts = _pop_counts_from_spike_times(st, T)
        ff = compute_fano_factor(st, counts, 50, T, ff_bin_steps=50, rn="test", pn="pyr")
        assert ff > 1.5, f"Bursty FF = {ff}"

    def test_no_spikes_returns_nan(self) -> None:
        """Zero spikes → NaN."""
        st: list[list[int]] = [[] for _ in range(10)]
        T = 1000
        counts = np.zeros(T, dtype=np.int32)
        ff = compute_fano_factor(st, counts, 10, T, ff_bin_steps=50, rn="test", pn="pyr")
        assert np.isnan(ff)

    def test_fallback_to_population_level(self) -> None:
        """When spike_times_list is None, population-level FF should still work."""
        rng = np.random.default_rng(42)
        T = 5000
        # Roughly Poisson-like population counts
        counts = rng.poisson(5, T).astype(np.int32)
        ff = compute_fano_factor(None, counts, 100, T, ff_bin_steps=50, rn="test", pn="pyr")
        assert 0.3 < ff < 3.0, f"Population-level FF = {ff}"


# =====================================================================
# Pairwise correlation
# =====================================================================


class TestPairwiseCorrelation:

    def test_independent_poisson_low_correlation(self) -> None:
        """Independent Poisson spike trains → low pairwise ρ."""
        st = _make_poisson_spike_times(30, rate_hz=15.0, duration_ms=5000.0)
        rho = compute_pairwise_correlation(st, 30, 5000, corr_bin_steps=100, rn="test", pn="pyr")
        assert rho < 0.10, f"Independent ρ = {rho}"

    def test_identical_spike_trains_high_correlation(self) -> None:
        """Identical spike trains → high pairwise ρ."""
        base = _make_poisson_spike_times(1, rate_hz=20.0, duration_ms=5000.0)[0]
        st = [list(base) for _ in range(30)]
        rho = compute_pairwise_correlation(st, 30, 5000, corr_bin_steps=100, rn="test", pn="pyr")
        assert rho > 0.8, f"Identical ρ = {rho}"

    def test_single_neuron_returns_nan(self) -> None:
        """Single neuron → NaN (no pairs)."""
        rho = compute_pairwise_correlation([[10, 20, 30]], 1, 1000, corr_bin_steps=100)
        assert np.isnan(rho)

    def test_all_silent_returns_nan(self) -> None:
        """All silent neurons → NaN."""
        st: list[list[int]] = [[] for _ in range(10)]
        rho = compute_pairwise_correlation(st, 10, 1000, corr_bin_steps=100, rn="test", pn="pyr")
        assert np.isnan(rho)


# =====================================================================
# Burst events
# =====================================================================


class TestBurstEvents:

    def test_no_bursts_in_sparse_activity(self) -> None:
        """Low-rate activity should produce negligible burst events."""
        rng = np.random.default_rng(42)
        T = 5000
        counts = rng.poisson(0.5, T).astype(np.int32)
        frac = compute_burst_events(counts, 100, T, burst_win_steps=20, burst_coactivation_fraction=0.3)
        assert frac < 0.05

    def test_synchronized_bursts_detected(self) -> None:
        """Strong synchronised bursts should be detected."""
        T = 5000
        n_neurons = 50
        counts = np.zeros(T, dtype=np.int32)
        # Create periodic synchronous bursts: all neurons fire during burst windows
        for start in range(100, T, 500):
            counts[start:start + 5] = n_neurons
        frac = compute_burst_events(counts, n_neurons, T, burst_win_steps=20, burst_coactivation_fraction=0.3)
        assert frac > 0.02, f"Burst fraction = {frac}"


# =====================================================================
# SFA index
# =====================================================================


class TestSfaIndex:

    def test_constant_rate_sfa_near_one(self) -> None:
        """Constant firing → SFA index ≈ 1.0."""
        T = 2000
        n_neurons = 50
        rng = np.random.default_rng(42)
        counts = rng.poisson(2, T).astype(np.int32)
        sfa = compute_sfa_index(counts, n_neurons, T, dt_ms=1.0)
        assert 0.7 < sfa < 1.5, f"Constant-rate SFA = {sfa}"

    def test_adapting_population(self) -> None:
        """Early burst with late rate suppression → SFA index > 1."""
        T = 2000
        n_neurons = 50
        counts = np.zeros(T, dtype=np.int32)
        # First quarter: high rate
        counts[:500] = 5
        # Last quarter: low rate
        counts[1500:] = 1
        sfa = compute_sfa_index(counts, n_neurons, T, dt_ms=1.0)
        assert sfa > 2.0, f"Adapting SFA = {sfa}"

    def test_too_few_spikes(self) -> None:
        """< 4 spikes → NaN."""
        counts = np.zeros(1000, dtype=np.int32)
        counts[100] = 2
        sfa = compute_sfa_index(counts, 10, 1000, dt_ms=1.0)
        assert np.isnan(sfa)


# =====================================================================
# Network state classifier
# =====================================================================


def _make_pop_stats(**kwargs) -> PopulationStats:
    """Create a minimal PopulationStats with specified fields, defaults for the rest."""
    defaults = dict(
        region_name="test",
        population_name="pyr",
        n_neurons=100,
        mean_fr_hz=10.0,
        std_fr_hz=3.0,
        fraction_silent=0.0,
        fraction_hyperactive=0.0,
        hyperactive_threshold_hz=100.0,
        total_spikes=5000,
        isi_mean_ms=100.0,
        isi_cv=1.0,
        isi_cv_corrected=1.0,
        fraction_bursting=0.05,
        fraction_refractory_violations=0.0,
        isi_cv2=1.0,
        isi_cv2_population=1.0,
        fraction_isi_lt_80ms=0.1,
        da_burst_events_per_s=0.0,
        sfa_index=1.0,
        per_neuron_ff=1.0,
        pairwise_correlation=0.05,
        fraction_burst_events=0.01,
        fr_histogram=np.zeros(20, dtype=np.float32),
        fr_histogram_edges=np.zeros(21, dtype=np.float32),
        bio_range_hz=(5.0, 25.0),
    )
    defaults.update(kwargs)
    return PopulationStats(**defaults)


class TestClassifyNetworkState:

    def test_ai_state(self) -> None:
        """AI: CV [0.7, 1.3], FF [0.5, 2.5], ρ < 0.15."""
        ps = _make_pop_stats(isi_cv=1.0, per_neuron_ff=1.0, pairwise_correlation=0.05)
        assert classify_network_state(ps) == "AI"

    def test_sr_state(self) -> None:
        """SR: CV < 0.5, FF < 0.5, ρ > 0.3."""
        ps = _make_pop_stats(isi_cv=0.2, per_neuron_ff=0.3, pairwise_correlation=0.5)
        assert classify_network_state(ps) == "SR"

    def test_si_state(self) -> None:
        """SI: CV > 1.3, FF > 2, ρ > 0.2."""
        ps = _make_pop_stats(isi_cv=1.5, per_neuron_ff=3.0, pairwise_correlation=0.3)
        assert classify_network_state(ps) == "SI"

    def test_burst_by_events(self) -> None:
        """Epileptiform: burst_events > 0.05."""
        ps = _make_pop_stats(fraction_burst_events=0.1)
        assert classify_network_state(ps) == "burst"

    def test_burst_by_cv(self) -> None:
        """Epileptiform: CV > 2.0."""
        ps = _make_pop_stats(isi_cv=2.5, per_neuron_ff=1.0, pairwise_correlation=0.05)
        assert classify_network_state(ps) == "burst"

    def test_burst_by_ff(self) -> None:
        """Epileptiform: FF > 4.0."""
        ps = _make_pop_stats(isi_cv=1.0, per_neuron_ff=5.0, pairwise_correlation=0.05)
        assert classify_network_state(ps) == "burst"

    def test_unknown_too_few_spikes(self) -> None:
        """< 20 total spikes → unknown."""
        ps = _make_pop_stats(total_spikes=10)
        assert classify_network_state(ps) == "unknown"

    def test_unknown_nan_cv(self) -> None:
        """NaN CV → unknown."""
        ps = _make_pop_stats(isi_cv=float("nan"))
        assert classify_network_state(ps) == "unknown"

    def test_unknown_ambiguous(self) -> None:
        """Metrics don't fit any clean category → unknown."""
        ps = _make_pop_stats(isi_cv=0.6, per_neuron_ff=1.5, pairwise_correlation=0.10)
        assert classify_network_state(ps) == "unknown"
