"""Thalamocortical coupling analysis — relay burst mode and laminar cascade latency.

Merged from ``relay_burst.py`` and ``laminar.py``.
Both analyse thalamus→cortex signal propagation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from thalia.diagnostics.diagnostics_metrics import LaminarCascadeRegionStats
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot
from thalia.diagnostics.region_tags import (
    CORTICAL_TAGS,
    L4_TAGS,
    L23_TAGS,
    L5_TAGS,
    L6_TAGS,
)


_RELAY_KEYWORDS = frozenset({
    "relay", "thalamus_relay", "lgn", "vpl", "vpm", "mgn",
    "pulvinar", "lateral_geniculate", "ventrobasal",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Relay burst mode — T-channel LTS detection via short ISI fraction
# ═══════════════════════════════════════════════════════════════════════════════


def compute_relay_burst_mode(rec: RecorderSnapshot, T: int) -> Dict[str, float]:
    """Compute the short-ISI fraction for relay populations in thalamic regions.

    Short ISIs (< 15 ms) are the hallmark of T-channel low-threshold spike
    (LTS) burst doublets and triplets in relay cells (McCormick & Huguenard
    1992).  A significant fraction (≥ 5 %) indicates active burst mode;
    near-zero indicates tonic Poisson-like firing.

    Requires per-neuron spike times from ``rec._spike_times``.
    Returns an empty dict when spike time data is unavailable.

    Returns:
        Dict keyed by region name.  Each value is the fraction of ISIs < 15 ms
        across all sampled relay neurons in that region.
    """
    if not rec._spike_times:
        return {}

    short_isi_ms = 15.0
    result: Dict[str, float] = {}

    region_relay_keys: Dict[str, List[tuple[str, str]]] = {}
    for rn, pn in rec._pop_keys:
        rn_lower = rn.lower()
        pn_lower = pn.lower()
        if "thalamus" not in rn_lower and "thalamic" not in rn_lower:
            continue
        if not any(kw in pn_lower for kw in _RELAY_KEYWORDS):
            continue
        region_relay_keys.setdefault(rn, []).append((rn, pn))

    for rn, pop_keys in region_relay_keys.items():
        n_short = 0
        n_total = 0
        for key in pop_keys:
            if key not in rec._spike_times:
                continue
            for neuron_steps in rec._spike_times[key]:
                if len(neuron_steps) < 2:
                    continue
                isis = np.diff(
                    np.array(neuron_steps, dtype=np.float64) * rec.dt_ms
                )
                n_short += int(np.sum(isis < short_isi_ms))
                n_total += len(isis)
        if n_total > 0:
            result[rn] = n_short / n_total

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Laminar cascade — thalamic volley → cortical layer latency
# ═══════════════════════════════════════════════════════════════════════════════


_LAYER_KEYWORDS: Dict[str, frozenset[str]] = {
    "l4":  L4_TAGS,
    "l23": L23_TAGS,
    "l5":  L5_TAGS,
    "l6":  L6_TAGS,
}


def detect_thalamic_volleys(rec: RecorderSnapshot, T: int) -> List[int]:
    """Return timestep indices of thalamic relay volleys.

    A volley is any timestep where the combined spike count across all thalamic
    relay populations reaches ≥ 5 % of the total relay neurons.  Consecutive
    events within 5 ms are merged to the first, so a sustained burst is counted
    as a single volley.
    """
    total_relay_spikes = np.zeros(T, dtype=np.int64)
    total_relay_neurons = 0

    for idx, (rn, pn) in enumerate(rec._pop_keys):
        rn_lower = rn.lower()
        pn_lower = pn.lower()
        if "thalamus" not in rn_lower and "thalamic" not in rn_lower:
            continue
        if not any(kw in pn_lower for kw in _RELAY_KEYWORDS):
            continue
        total_relay_spikes += rec._pop_spike_counts[:T, idx].astype(np.int64)
        total_relay_neurons += int(rec._pop_sizes[idx])

    if total_relay_neurons == 0:
        return []

    threshold = max(1.0, 0.05 * total_relay_neurons)
    hot = np.where(total_relay_spikes >= threshold)[0]
    if len(hot) == 0:
        return []

    merge_steps = max(1, int(round(5.0 / rec.dt_ms)))
    events: List[int] = [int(hot[0])]
    for ts in hot[1:]:
        if ts - events[-1] > merge_steps:
            events.append(int(ts))
    return events


def compute_laminar_cascade(rec: RecorderSnapshot, T: int) -> Dict[str, LaminarCascadeRegionStats]:
    """Compute mean first-spike latency per cortical layer after thalamic volleys.

    For each cortical region (those whose name contains "cortex", "prefrontal",
    or "entorhinal"), and for each layer tier (L4, L2/3, L5, L6), this function
    measures how quickly the first spike in that tier arrives after each thalamic
    relay volley.  Averaging over volleys gives a stable estimate of the
    thalamocortical feedforward latency hierarchy.

    Expected latency order (Thomson & Bannister 2003; Sakata & Harris 2009):
        L4 < L2/3 < L5

    Requires per-neuron spike times in ``rec._spike_times``.
    Returns an empty dict when spike time data or thalamic volleys are absent.

    Returns:
        Dict keyed by cortical region name.  Each value is a dict with keys
        ``l4_lat_ms``, ``l23_lat_ms``, ``l5_lat_ms``, ``l6_lat_ms``.
        A tier key is omitted when no matching population is present in the
        region; its value is NaN when matching populations exist but produced
        no spikes in any post-volley window.
    """
    if not rec._spike_times:
        return {}

    volley_timesteps = detect_thalamic_volleys(rec, T)
    if not volley_timesteps:
        return {}

    window_steps = int(round(50.0 / rec.dt_ms))
    result: Dict[str, LaminarCascadeRegionStats] = {}

    for rn in rec._region_keys:
        rn_lower = rn.lower()
        if not any(tag in rn_lower for tag in CORTICAL_TAGS):
            continue

        tier_arrays: Dict[str, List[np.ndarray]] = {}
        for tier, kws in _LAYER_KEYWORDS.items():
            arrays: List[np.ndarray] = []
            for r, pn in rec._pop_keys:
                if r != rn:
                    continue
                pn_lower = pn.lower()
                if not any(kw in pn_lower for kw in kws):
                    continue
                key = (r, pn)
                if key not in rec._spike_times:
                    continue
                flat: List[int] = []
                for neuron_spikes in rec._spike_times[key]:
                    flat.extend(neuron_spikes)
                if flat:
                    arrays.append(np.sort(np.array(flat, dtype=np.int64)))
            if arrays:
                tier_arrays[tier] = arrays

        if not tier_arrays:
            continue

        tier_latencies: Dict[str, List[float]] = {tier: [] for tier in tier_arrays}
        for t_event in volley_timesteps:
            t_end = t_event + window_steps
            for tier, arrays in tier_arrays.items():
                first_spike: Optional[int] = None
                for arr in arrays:
                    lo = int(np.searchsorted(arr, t_event, side="left"))
                    if lo < len(arr) and arr[lo] < t_end:
                        cand = int(arr[lo])
                        if first_spike is None or cand < first_spike:
                            first_spike = cand
                if first_spike is not None:
                    tier_latencies[tier].append((first_spike - t_event) * rec.dt_ms)

        region_result: Dict[str, float] = {}
        for tier, lats in tier_latencies.items():
            region_result[f"{tier}_lat_ms"] = float(np.mean(lats)) if lats else float("nan")
        result[rn] = LaminarCascadeRegionStats(
            l4_lat_ms=region_result.get("l4_lat_ms", float("nan")),
            l23_lat_ms=region_result.get("l23_lat_ms", float("nan")),
            l5_lat_ms=region_result.get("l5_lat_ms", float("nan")),
            l6_lat_ms=region_result.get("l6_lat_ms", float("nan")),
        )

    return result
