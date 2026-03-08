"""Region-level statistics — conductance extraction and population aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

from .diagnostics_types import PopulationStats, RegionStats

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


def _weighted_mean_fr(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    rn: str,
) -> float:
    """Neuron-weighted mean firing rate for all populations in region *rn*.

    Uses :attr:`PopulationStats.n_neurons` as weights so the result is
    identical to the time-averaged value of :func:`analyze`'s
    ``region_rate_binned`` column for the same region.
    """
    pops = [ps for (r, _pn), ps in pop_stats.items() if r == rn]
    if not pops:
        return 0.0
    weights = np.array([p.n_neurons for p in pops], dtype=np.float64)
    rates   = np.array([p.mean_fr_hz for p in pops], dtype=np.float64)
    total   = float(weights.sum())
    return float((weights * rates).sum() / total) if total > 0 else 0.0


def _extract_region_conductances(
    rec: "DiagnosticsRecorder",
    region_name: str,
) -> Tuple[float, float, float, float, float]:
    """Extract mean AMPA, NMDA, and GABA conductances for a region.

    Returns ``(mean_g_exc, mean_g_nmda, mean_g_inh, mean_g_gaba_a, mean_g_gaba_b)``
    — all NaN in stats mode or when conductance samples are unavailable.

    Excitatory numerator = AMPA (g_E) + NMDA when available.
    Inhibitory denominator = GABA-A (g_I) + GABA-B when available.
    GABA-A and GABA-B are also returned separately for the §1.2 ratio check.
    """
    if (
        rec.config.mode != "full"
        or rec._g_exc_samples is None
        or rec._g_inh_samples is None
    ):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    p_indices = rec._region_pop_indices[region_name]
    cond_step = rec._cond_sample_step
    # Neuron counts per population — used as weights so that a small PV population
    # (e.g. 10 neurons) does not dominate a large pyramidal layer (e.g. 400 neurons)
    # just because conductance magnitude is higher.
    weights = np.array([float(rec._pop_sizes[pi]) for pi in p_indices], dtype=np.float64)

    def _pop_means(arr: np.ndarray) -> np.ndarray:
        """Per-population nanmean over [cond_step × C] conductance samples.

        Vectorised: advanced-indexing selects all populations at once into a
        [cond_step, n_pops_in_region, C] slab; a single nanmean call over
        axes (0, 2) replaces the O(n_pops) Python loop.
        """
        # shape: [cond_step, n_pops_in_region, C]
        slab = arr[:cond_step][:, p_indices, :]
        return np.nanmean(slab, axis=(0, 2))  # [n_pops_in_region]

    def _wavg(pop_means: np.ndarray) -> float:
        """Neuron-count-weighted mean, ignoring NaN populations."""
        valid = ~np.isnan(pop_means)
        if not valid.any():
            return float(np.nan)
        w = weights[valid]
        return float((w * pop_means[valid]).sum() / w.sum())

    exc_pm   = _pop_means(rec._g_exc_samples)
    inh_pm   = _pop_means(rec._g_inh_samples)   # GABA-A per population
    nmda_pm  = _pop_means(rec._g_nmda_samples)  if rec._g_nmda_samples   is not None \
               else np.full(len(p_indices), np.nan)
    gaba_b_pm = _pop_means(rec._g_gaba_b_samples) if rec._g_gaba_b_samples is not None \
               else np.full(len(p_indices), np.nan)

    # Combined inhibitory per population: GABA-A + GABA-B (linearity of expectation
    # makes this equivalent to averaging the per-sample element-wise sums, but without
    # the alignment hazard of the old flat-concatenation approach).
    inh_combined_pm = np.where(np.isnan(gaba_b_pm), inh_pm, inh_pm + gaba_b_pm)

    return (
        _wavg(exc_pm),
        _wavg(nmda_pm),
        _wavg(inh_combined_pm),
        _wavg(inh_pm),     # mean_g_gaba_a
        _wavg(gaba_b_pm),
    )


def compute_region_stats(
    rec: "DiagnosticsRecorder",
    region_name: str,
    pop_stats: Dict[Tuple[str, str], PopulationStats],
) -> RegionStats:
    """Aggregate population stats into a RegionStats."""
    pops = {
        pn: pop_stats[(rn, pn)]
        for rn, pn in rec._pop_keys
        if rn == region_name
    }
    # Weighted mean: large excitatory populations must not be outweighed by
    # small fast-firing interneurons.  An unweighted mean of a 400-neuron L4 pyr
    # population at 5 Hz and a 20-neuron PV population at 40 Hz would yield
    # 22.5 Hz — inflated by the population contributing only 5 % of neurons.
    mean_fr = _weighted_mean_fr(pop_stats, region_name)
    total_spikes = sum(p.total_spikes for p in pops.values())

    # E/I balance: mean conductances across sampled neurons in this region.
    # Excitatory numerator = AMPA (g_E) + NMDA (g_nmda) when available.
    # Inhibitory denominator = GABA-A (g_I) + GABA-B (g_gaba_b) when available.
    mean_g_exc, mean_g_nmda, mean_g_inh, mean_g_gaba_a, mean_g_gaba_b = _extract_region_conductances(rec, region_name)

    # --- D1/D2 competition index --------------------------------------
    # Pearson correlation between D1 and D2 MSN population rates.
    # Only computed for striatal regions that have both D1 and D2 populations.
    # Two bin widths:
    #   200 ms — action-selection epoch (Mink 1996)
    #   50 ms  — striatal mutual-inhibition timescale; catches within-window
    #             alternation that 200 ms bins blur to near-zero correlation.
    d1_d2_competition_index_200ms = np.nan
    d1_d2_competition_index_50ms = np.nan
    if "striatum" in region_name.lower():
        T_local = rec._n_recorded or rec.config.n_timesteps
        d1_idx = [i for i, (r, p) in enumerate(rec._pop_keys) if r == region_name and "d1" in p.lower()]
        d2_idx = [i for i, (r, p) in enumerate(rec._pop_keys) if r == region_name and "d2" in p.lower()]
        if d1_idx and d2_idx:
            d1_counts = rec._pop_spike_counts[:T_local, d1_idx].sum(axis=1).astype(np.float64)
            d2_counts = rec._pop_spike_counts[:T_local, d2_idx].sum(axis=1).astype(np.float64)

            def _pearson_binned(bin_ms: float) -> float:
                bs = max(1, int(bin_ms / rec.dt_ms))
                nb = T_local // bs
                if nb < 4:
                    return np.nan
                d1_b = d1_counts[:nb * bs].reshape(nb, bs).sum(axis=1)
                d2_b = d2_counts[:nb * bs].reshape(nb, bs).sum(axis=1)
                if d1_b.std() == 0 or d2_b.std() == 0:
                    return np.nan
                return float(np.corrcoef(d1_b, d2_b)[0, 1])

            d1_d2_competition_index_200ms = _pearson_binned(200.0)
            d1_d2_competition_index_50ms = _pearson_binned(50.0)

    return RegionStats(
        region_name=region_name,
        populations=pops,
        mean_fr_hz=mean_fr,
        total_spikes=total_spikes,
        mean_g_exc=mean_g_exc,
        mean_g_nmda=mean_g_nmda,
        mean_g_inh=mean_g_inh,
        mean_g_gaba_a=mean_g_gaba_a,
        mean_g_gaba_b=mean_g_gaba_b,
        d1_d2_competition_index_200ms=d1_d2_competition_index_200ms,
        d1_d2_competition_index_50ms=d1_d2_competition_index_50ms,
    )
