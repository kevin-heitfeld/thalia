"""Homeostatic analysis — gain trajectories, STP efficacy history, and final state."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .diagnostics_types import HomeostaticStats, RecorderSnapshot


def compute_homeostatic_stats(rec: RecorderSnapshot) -> HomeostaticStats:
    """Summarise homeostatic gain trajectories, STP efficacy history, and STP final state."""
    n_steps = rec._gain_sample_step
    sample_times = np.array(rec._gain_sample_times, dtype=np.float32) * rec.dt_ms

    gain_trajectories: Dict[str, np.ndarray] = {}
    for idx, (rn, pn) in enumerate(rec._pop_keys):
        vals = rec._g_L_scale_history[:n_steps, idx]
        if not np.all(np.isnan(vals)) and not np.all(vals == 0):
            gain_trajectories[f"{rn}:{pn}"] = vals.copy()

    # STP efficacy trajectories
    stp_efficacy_hist: Dict[str, np.ndarray] = {}
    for stp_idx, (rn, syn_id) in enumerate(rec._stp_keys):
        vals = rec._stp_efficacy_history[:n_steps, stp_idx]
        if not np.all(np.isnan(vals)):
            stp_efficacy_hist[str(syn_id)] = vals.copy()

    # STP final state
    if hasattr(rec, "_capture_stp_final_state"):
        stp_final: Dict[str, Dict[str, float]] = rec._capture_stp_final_state()
    else:
        stp_final = dict(rec._stp_final_state)

    return HomeostaticStats(
        gain_trajectories=gain_trajectories,
        gain_sample_times_ms=sample_times,
        stp_efficacy_history=stp_efficacy_hist,
        stp_final_state=stp_final,
    )
