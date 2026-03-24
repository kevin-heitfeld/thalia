"""Learning / training diagnostics analysis.

Computes weight distribution summaries, eligibility trace statistics,
BCM threshold dynamics, homeostatic–plasticity interaction, representational
stability, and DA–eligibility temporal alignment from recorded learning
state trajectories.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .diagnostics_metrics import LearningStats, STDPTimingStats, SynapseLearningSummary, WeightDistStats
from .diagnostics_snapshot import RecorderSnapshot


def _weight_dist_from_row(row: np.ndarray) -> WeightDistStats:
    """Construct a WeightDistStats from a [5]-shaped array (mean, std, min, max, sparsity)."""
    return WeightDistStats(
        mean=float(row[0]),
        std=float(row[1]),
        min_val=float(row[2]),
        max_val=float(row[3]),
        sparsity=float(row[4]),
    )


def compute_learning_stats(rec: RecorderSnapshot) -> Optional[LearningStats]:
    """Compute learning diagnostics from a recorder snapshot.

    Returns ``None`` when no learning strategies are present.
    """
    if not rec._learning_keys:
        return None

    n_steps = rec._gain_sample_step
    if n_steps < 2:
        return None

    gi_steps = max(1, int(rec.config.gain_sample_interval_ms / rec.dt_ms))
    sample_times_ms = np.array(rec._gain_sample_times[:n_steps], dtype=np.float32) * rec.dt_ms

    # ── Per-synapse summaries ────────────────────────────────────────────
    synapse_summaries: Dict[str, SynapseLearningSummary] = {}
    weight_trajectories: Dict[str, List[Tuple[float, WeightDistStats]]] = {}
    update_mag_trajectories: Dict[str, np.ndarray] = {}
    eligibility_trajectories: Dict[str, np.ndarray] = {}
    bcm_theta_trajectories: Dict[str, np.ndarray] = {}

    for learn_idx, (rn, syn_id_str) in enumerate(rec._learning_keys):
        # Infer strategy type from the stored type names
        # (learning_keys stores (region, synapse_id_str) pairs)
        strategy_type = "unknown"

        # Weight distribution trajectory
        wdt: List[Tuple[float, WeightDistStats]] = []
        if rec._weight_dist_history is not None and n_steps > 0:
            for s in range(n_steps):
                row = rec._weight_dist_history[s, learn_idx, :]
                if not np.isnan(row[0]):
                    wdt.append((float(sample_times_ms[s]), _weight_dist_from_row(row)))

        key = f"{rn}:{syn_id_str}"
        weight_trajectories[key] = wdt

        # Weight start/end stats
        w_start = _weight_dist_from_row(rec._weight_dist_history[0, learn_idx, :]) if wdt else WeightDistStats(0, 0, 0, 0, 0)
        w_end = _weight_dist_from_row(rec._weight_dist_history[n_steps - 1, learn_idx, :]) if wdt else WeightDistStats(0, 0, 0, 0, 0)

        # Weight update magnitude
        update_mag = np.full(n_steps, np.nan, dtype=np.float32)
        if rec._weight_update_magnitude_history is not None:
            update_mag = rec._weight_update_magnitude_history[:n_steps, learn_idx].copy()
        update_mag_trajectories[key] = update_mag
        mean_update_mag = float(np.nanmean(update_mag)) if not np.all(np.isnan(update_mag)) else 0.0

        # Weight drift
        w_drift = 0.0
        if w_start.mean != 0 and abs(w_start.mean) > 1e-10:
            w_drift = (w_end.mean - w_start.mean) / abs(w_start.mean)

        # Eligibility
        elig_traj = np.full(n_steps, np.nan, dtype=np.float32)
        if rec._eligibility_mean_history is not None:
            elig_traj = rec._eligibility_mean_history[:n_steps, learn_idx].copy()
        eligibility_trajectories[key] = elig_traj
        mean_elig = float(np.nanmean(elig_traj)) if not np.all(np.isnan(elig_traj)) else float("nan")

        ltp_ltd_traj = np.full(n_steps, np.nan, dtype=np.float32)
        if rec._eligibility_ltp_ltd_ratio_history is not None:
            ltp_ltd_traj = rec._eligibility_ltp_ltd_ratio_history[:n_steps, learn_idx].copy()
        ltp_ltd_ratio = float(np.nanmean(ltp_ltd_traj)) if not np.all(np.isnan(ltp_ltd_traj)) else float("nan")

        # BCM theta
        bcm_theta_start = float("nan")
        bcm_theta_end = float("nan")
        if rec._bcm_theta_history is not None and learn_idx in rec._bcm_keys:
            bcm_idx_in_list = rec._bcm_keys.index(learn_idx)
            theta_traj = rec._bcm_theta_history[:n_steps, bcm_idx_in_list].copy()
            bcm_theta_trajectories[key] = theta_traj
            strategy_type = "BCM"
            valid_theta = theta_traj[~np.isnan(theta_traj)]
            if len(valid_theta) > 0:
                bcm_theta_start = float(valid_theta[0])
                bcm_theta_end = float(valid_theta[-1])

        synapse_summaries[key] = SynapseLearningSummary(
            synapse_id=syn_id_str,
            strategy_type=strategy_type,
            weight_start=w_start,
            weight_end=w_end,
            mean_update_magnitude=mean_update_mag,
            weight_drift=w_drift,
            mean_eligibility=mean_elig,
            ltp_ltd_ratio=ltp_ltd_ratio,
            bcm_theta_start=bcm_theta_start,
            bcm_theta_end=bcm_theta_end,
        )

    # ── Homeostatic correction rate ──────────────────────────────────────
    homeostatic_correction_rate: Dict[str, np.ndarray] = {}
    if rec._homeostatic_correction_rate is not None:
        for idx, (rn, pn) in enumerate(rec._pop_keys):
            corr = rec._homeostatic_correction_rate[:n_steps, idx].copy()
            if not np.all(np.isnan(corr)):
                homeostatic_correction_rate[f"{rn}:{pn}"] = corr

    # ── Representational stability ───────────────────────────────────────
    popvec_stability = float("nan")
    if rec._popvec_snapshots is not None and len(rec._popvec_snapshot_times) >= 2:
        n_snaps = len(rec._popvec_snapshot_times)
        correlations: List[float] = []
        for i in range(1, n_snaps):
            v0 = rec._popvec_snapshots[i - 1, :]
            v1 = rec._popvec_snapshots[i, :]
            valid = ~(np.isnan(v0) | np.isnan(v1))
            if valid.sum() > 2:
                v0c = v0[valid]
                v1c = v1[valid]
                if np.std(v0c) > 1e-10 and np.std(v1c) > 1e-10:
                    correlations.append(float(np.corrcoef(v0c, v1c)[0, 1]))
        if correlations:
            popvec_stability = float(np.mean(correlations))

    # ── DA–eligibility temporal alignment ────────────────────────────────
    da_eligibility_alignment = _compute_da_eligibility_alignment(rec, n_steps)

    # ── STDP timing distributions ────────────────────────────────────────
    stdp_timing = _compute_stdp_timing(rec)

    return LearningStats(
        synapse_summaries=synapse_summaries,
        weight_trajectories=weight_trajectories,
        update_magnitude_trajectories=update_mag_trajectories,
        eligibility_trajectories=eligibility_trajectories,
        bcm_theta_trajectories=bcm_theta_trajectories,
        homeostatic_correction_rate=homeostatic_correction_rate,
        popvec_stability=popvec_stability,
        sample_times_ms=sample_times_ms,
        da_eligibility_alignment=da_eligibility_alignment,
        stdp_timing=stdp_timing,
    )


def _compute_da_eligibility_alignment(
    rec: RecorderSnapshot,
    n_steps: int,
) -> Dict[str, float]:
    """Compute DA–eligibility temporal alignment for DA-gated synapses.

    For each synapse with eligibility traces, measures the fraction of
    timesteps where non-zero eligibility overlaps with above-baseline
    dopamine concentration.  A perfect score (1.0) means every time there's
    eligibility, DA is present to gate it.

    Requires both eligibility and neuromodulator history to be recorded.
    """
    result: Dict[str, float] = {}
    if (
        rec._eligibility_mean_history is None
        or rec._n_nm_receptors == 0
        or n_steps < 10
    ):
        return result

    # Find DA-related neuromodulator indices
    da_nm_indices: List[int] = []
    for nm_idx, (rn, mod_name) in enumerate(rec._nm_receptor_keys):
        mod_lower = mod_name.lower()
        if "da" in mod_lower or "dopamine" in mod_lower:
            da_nm_indices.append(nm_idx)

    if not da_nm_indices:
        return result

    # Aggregate DA concentration across all DA receptors: mean per timestep
    da_conc = np.nanmean(
        rec._nm_concentration_history[:n_steps, da_nm_indices], axis=1
    )
    da_baseline = float(np.nanmedian(da_conc))
    da_active = da_conc > da_baseline * 1.1  # DA above 110% of baseline

    for learn_idx, (rn, syn_id_str) in enumerate(rec._learning_keys):
        elig = rec._eligibility_mean_history[:n_steps, learn_idx]
        elig_present = elig > 1e-6  # Non-trivial eligibility
        n_elig = int(np.sum(elig_present))
        if n_elig < 5:
            continue
        # Alignment = fraction of elig-present steps where DA is also active
        n_aligned = int(np.sum(elig_present & da_active))
        key = f"{rn}:{syn_id_str}"
        result[key] = float(n_aligned) / float(n_elig)

    return result


def _parse_synapse_id_str(sid_str: str) -> Tuple[str, str, str, str]:
    """Parse the human-readable SynapseId string format.

    Format: ``"src_region:src_pop → tgt_region:tgt_pop (receptor_type)"``

    Returns (source_region, source_population, target_region, target_population).
    """
    src_part, rest = sid_str.split(" \u2192 ", 1)
    tgt_part = rest.rsplit(" (", 1)[0]
    src_region, src_pop = src_part.split(":", 1)
    tgt_region, tgt_pop = tgt_part.split(":", 1)
    return src_region, src_pop, tgt_region, tgt_pop


def _compute_stdp_timing(rec: RecorderSnapshot) -> Dict[str, STDPTimingStats]:
    """Compute pre-post spike timing distributions for each learning pathway.

    For each STDP-enabled pathway, samples neuron pairs and collects all
    spike-time differences Δt = t_post − t_pre within ±window_ms.
    Uses emission times (axonal delays not subtracted).

    Returns a dict keyed by ``"region:synapse_id_str"``.
    """
    result: Dict[str, STDPTimingStats] = {}
    if not rec._learning_keys:
        return result

    window_ms = rec.config.thresholds.learning.stdp_timing_window_ms
    max_pairs = rec.config.thresholds.learning.stdp_timing_max_pairs
    window_steps = window_ms / rec.dt_ms
    n_bins = 21  # Histogram bins for ±window

    rng = np.random.default_rng(42)

    for _learn_idx, (rn, syn_id_str) in enumerate(rec._learning_keys):
        try:
            src_r, src_p, tgt_r, tgt_p = _parse_synapse_id_str(syn_id_str)
        except (ValueError, IndexError):
            continue

        pre_spikes = rec._spike_times.get((src_r, src_p))
        post_spikes = rec._spike_times.get((tgt_r, tgt_p))
        if pre_spikes is None or post_spikes is None:
            continue

        n_pre = len(pre_spikes)
        n_post = len(post_spikes)
        if n_pre == 0 or n_post == 0:
            continue

        # Find neurons that actually spiked (avoid empty pairs)
        active_pre = [i for i in range(n_pre) if pre_spikes[i]]
        active_post = [i for i in range(n_post) if post_spikes[i]]
        if not active_pre or not active_post:
            continue

        # Sample neuron pairs
        n_possible = len(active_pre) * len(active_post)
        n_sample = min(max_pairs, n_possible)
        if n_possible <= max_pairs:
            pairs = [(pi, qi) for pi in active_pre for qi in active_post]
        else:
            pre_idx = rng.integers(0, len(active_pre), size=n_sample)
            post_idx = rng.integers(0, len(active_post), size=n_sample)
            pairs = [(active_pre[pi], active_post[qi])
                     for pi, qi in zip(pre_idx.tolist(), post_idx.tolist())]

        # Collect timing deltas
        all_deltas: List[float] = []
        for pre_i, post_j in pairs:
            t_pre = np.asarray(pre_spikes[pre_i], dtype=np.float64)
            t_post = np.asarray(post_spikes[post_j], dtype=np.float64)
            if len(t_pre) == 0 or len(t_post) == 0:
                continue
            # Broadcast: delta[i, j] = t_post[i] - t_pre[j]
            delta = t_post[:, None] - t_pre[None, :]
            within_window = delta[(delta >= -window_steps) & (delta <= window_steps)]
            if len(within_window) > 0:
                all_deltas.extend((within_window * rec.dt_ms).tolist())

        if len(all_deltas) < 5:
            continue

        deltas = np.asarray(all_deltas, dtype=np.float64)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas))
        ltp_frac = float(np.mean(deltas > 0))

        edges = np.linspace(-window_ms, window_ms, n_bins + 1)
        counts, _ = np.histogram(deltas, bins=edges)

        key = f"{rn}:{syn_id_str}"
        result[key] = STDPTimingStats(
            mean_delta_ms=mean_delta,
            std_delta_ms=std_delta,
            ltp_fraction=ltp_frac,
            n_pairs=len(deltas),
            histogram_edges_ms=edges,
            histogram_counts=counts.astype(np.int32),
        )

    return result
