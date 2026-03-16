"""Connectivity analysis — axonal tract transmission and delay verification."""

from __future__ import annotations

from typing import List

import numpy as np

from .diagnostics_types import ConnectivityStats, RecorderSnapshot


def compute_connectivity_stats(rec: RecorderSnapshot, T: int) -> ConnectivityStats:
    """Analyse axonal tract transmission and verify delays."""
    tracts: List[ConnectivityStats.TractStats] = []

    for tract_idx, synapse_id in enumerate(rec._tract_keys):
        sent = rec._tract_sent[:T, tract_idx]
        total_sent = int(sent.sum())
        transmission_ratio = float((sent > 0).mean())
        # A tract is broken only when it transmitted zero spikes over the entire
        # recording window.  The old `> 0.01` threshold false-alarmed on bursty
        # patterns (e.g. _sensory_burst) that are active on < 1% of timesteps.
        is_functional = total_sent > 0

        # Expected delay from tract spec
        expected_delay_ms = rec._tract_delay_ms[tract_idx]

        # Measured delay: cross-correlation between source and target populations.
        # Also scan the anti-causal (negative-lag) window to detect reversed connections.
        measured_delay_ms = np.nan
        anticausal_peak_ms = np.nan
        if rec.config.mode == "full" and is_functional:
            tgt_key = (synapse_id.target_region, synapse_id.target_population)
            tgt_idx = rec._pop_index.get(tgt_key)
            if tgt_idx is not None:
                src = sent.astype(np.float64)
                tgt = rec._pop_spike_counts[:T, tgt_idx].astype(np.float64)
                if src.std() > 0 and tgt.std() > 0:
                    xcorr = np.correlate(
                        tgt - tgt.mean(), src - src.mean(), mode="full"
                    )
                    lag_range = int(min(100, max(expected_delay_ms * 3, 10) / rec.dt_ms))
                    center = len(src) - 1

                    # Causal window: target follows source (positive lags)
                    c_lo = max(0, center)
                    c_hi = min(len(xcorr), center + lag_range)
                    if c_hi > c_lo:
                        causal_seg = xcorr[c_lo:c_hi]
                        peak_idx = int(np.argmax(causal_seg))
                        measured_delay_ms = float(peak_idx) * rec.dt_ms
                        causal_max = float(causal_seg[peak_idx])
                    else:
                        causal_max = 0.0

                    # Anti-causal window: target precedes source (negative lags)
                    # A dominant anti-causal peak means the connection is reversed.
                    ac_lo = max(0, center - lag_range)
                    ac_hi = min(len(xcorr), center)
                    if ac_hi > ac_lo:
                        anticausal_seg = xcorr[ac_lo:ac_hi]
                        ac_peak_idx = int(np.argmax(anticausal_seg))
                        anticausal_peak_ms = float(lag_range - ac_peak_idx) * rec.dt_ms
                        anticausal_max = float(anticausal_seg[ac_peak_idx])
                    else:
                        anticausal_max = 0.0

                    # Mark as reversed when anti-causal peak is at least 2× stronger
                    # than the causal peak, guarding against noise at lag ≈0.
                    if anticausal_max > causal_max * 2.0:
                        # Swap so measured_delay_ms represents the dominant peak.
                        measured_delay_ms = -anticausal_peak_ms

        tracts.append(
            ConnectivityStats.TractStats(
                synapse_id=synapse_id,
                spikes_sent=total_sent,
                transmission_ratio=transmission_ratio,
                is_functional=is_functional,
                measured_delay_ms=measured_delay_ms,
                expected_delay_ms=expected_delay_ms,
                anticausal_peak_ms=anticausal_peak_ms,
            )
        )

    broken = [t for t in tracts if not t.is_functional]
    return ConnectivityStats(
        tracts=tracts,
        n_functional=len(tracts) - len(broken),
        n_broken=broken,
    )
