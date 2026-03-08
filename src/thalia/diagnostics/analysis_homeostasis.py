"""Homeostatic analysis — gain trajectories, STP efficacy history, and final state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from .diagnostics_types import HomeostaticStats

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


def compute_homeostatic_stats(rec: "DiagnosticsRecorder") -> HomeostaticStats:
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

    # STP final state snapshot
    stp_final: Dict[str, Dict[str, float]] = {}
    for region in rec.brain.regions.values():
        if not hasattr(region, "stp_modules"):
            continue
        for syn_id, stp_mod in region.stp_modules.items():
            if hasattr(stp_mod, "x") and hasattr(stp_mod, "u"):
                key = str(syn_id)
                with torch.no_grad():
                    stp_final[key] = {
                        "mean_x": float(stp_mod.x.mean().item()),
                        "mean_u": float(stp_mod.u.mean().item()),
                        "efficacy": float((stp_mod.x * stp_mod.u).mean().item()),
                    }

    return HomeostaticStats(
        gain_trajectories=gain_trajectories,
        gain_sample_times_ms=sample_times,
        stp_efficacy_history=stp_efficacy_hist,
        stp_final_state=stp_final,
    )
