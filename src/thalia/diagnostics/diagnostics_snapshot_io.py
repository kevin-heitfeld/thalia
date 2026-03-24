"""Snapshot save/load and report serialisation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from thalia.typing import SynapseId

from .diagnostics_config import (
    DiagnosticsConfig,
    HealthThresholds,
    _THRESHOLD_GROUPS,
)
from .diagnostics_report import DiagnosticsReport
from .diagnostics_snapshot import RecorderSnapshot


# =============================================================================
# SNAPSHOT SAVE/LOAD
# =============================================================================


def save_snapshot(snapshot: RecorderSnapshot, path: str) -> None:
    """Save a RecorderSnapshot to disk for later re-loading and analysis."""
    if not path.endswith(".npz"):
        path = path + ".npz"

    arrays: Dict[str, np.ndarray] = {}

    # ── Config / metadata → JSON string ────────────────────────────────
    cfg_dict = asdict(snapshot.config)

    meta = {
        "dt_ms": snapshot.dt_ms,
        "config": cfg_dict,
        "n_recorded": snapshot._n_recorded,
        "gain_sample_step": snapshot._gain_sample_step,
        "cond_sample_step": snapshot._cond_sample_step,
        "gain_sample_times": snapshot._gain_sample_times,
        "pop_keys": [[rn, pn] for rn, pn in snapshot._pop_keys],
        "region_keys": snapshot._region_keys,
        "region_pop_indices": {rn: v for rn, v in snapshot._region_pop_indices.items()},
        "tract_keys": [sid.to_key() for sid in snapshot._tract_keys],
        "stp_keys": [[rn, sid.to_key()] for rn, sid in snapshot._stp_keys],
        "nm_receptor_keys": [[rn, path_] for rn, path_ in snapshot._nm_receptor_keys],
        "nm_source_pop_keys": [[rn, pn] for rn, pn in snapshot._nm_source_pop_keys],
        "pop_polarities": [[rn, pn, pol] for (rn, pn), pol in snapshot._pop_polarities.items()],
        "tract_delay_ms": snapshot._tract_delay_ms,
        "homeostasis_target_hz": [[rn, pn, hz] for (rn, pn), hz in snapshot._homeostasis_target_hz.items()],
        "stp_configs": [[U, tau_d, tau_f] for U, tau_d, tau_f in snapshot._stp_configs],
        "stp_final_state": snapshot._stp_final_state,
        "tract_weight_stats": snapshot._tract_weight_stats,
        "pop_neuron_params": [
            [rn, pn, params]
            for (rn, pn), params in snapshot._pop_neuron_params.items()
        ],
        "learning_keys": snapshot._learning_keys,
        "bcm_keys": snapshot._bcm_keys,
        "popvec_snapshot_times": snapshot._popvec_snapshot_times,
    }
    arrays["_meta"] = np.array(json.dumps(meta), dtype=object)

    # ── Pop sizes ───────────────────────────────────────────────────────
    arrays["_pop_sizes"] = snapshot._pop_sizes

    # ── Spike count buffers ─────────────────────────────────────────────
    arrays["_pop_spike_counts"]    = snapshot._pop_spike_counts[: snapshot._n_recorded]
    arrays["_region_spike_counts"] = snapshot._region_spike_counts[: snapshot._n_recorded]
    if snapshot._n_tracts > 0:
        arrays["_tract_sent"] = snapshot._tract_sent[: snapshot._n_recorded]

    for i, arr in enumerate(snapshot._per_neuron_spike_counts):
        arrays[f"_per_neuron_{i}"] = arr

    # ── Sample indices ──────────────────────────────────────────────────
    for i, idx in enumerate(snapshot._v_sample_idx):
        arrays[f"_v_idx_{i}"] = idx
    for i, idx in enumerate(snapshot._c_sample_idx):
        arrays[f"_c_idx_{i}"] = idx

    # ── Spike times ─────────────────────────────────────────
    for pop_idx, key in enumerate(snapshot._pop_keys):
        if key in snapshot._spike_times:
            nested = snapshot._spike_times[key]
            nidx_list: List[int] = []
            ts_list: List[int] = []
            for ni, times in enumerate(nested):
                for t_val in times:
                    nidx_list.append(ni)
                    ts_list.append(t_val)
            if nidx_list:
                arrays[f"_st_nidx_{pop_idx}"] = np.array(nidx_list, dtype=np.int32)
                arrays[f"_st_ts_{pop_idx}"]   = np.array(ts_list,   dtype=np.int32)

    # ── Trajectory buffers ──────────────────────────────────────────────
    n_gs = snapshot._gain_sample_step
    arrays["_g_L_scale_history"]     = snapshot._g_L_scale_history[:n_gs]
    arrays["_stp_efficacy_history"]  = snapshot._stp_efficacy_history[:n_gs]
    arrays["_nm_concentration_history"] = snapshot._nm_concentration_history[:n_gs]

    # ── Learning trajectory buffers ──────────────────────────────────────
    if snapshot._learning_keys:
        n_learn = len(snapshot._learning_keys)
        arrays["_weight_dist_history"] = snapshot._weight_dist_history[:n_gs, :n_learn]
        arrays["_weight_update_magnitude_history"] = snapshot._weight_update_magnitude_history[:n_gs, :n_learn]
        arrays["_eligibility_mean_history"] = snapshot._eligibility_mean_history[:n_gs, :n_learn]
        arrays["_eligibility_ltp_ltd_ratio_history"] = snapshot._eligibility_ltp_ltd_ratio_history[:n_gs, :n_learn]
        if snapshot._bcm_keys:
            n_bcm = len(snapshot._bcm_keys)
            arrays["_bcm_theta_history"] = snapshot._bcm_theta_history[:n_gs, :n_bcm]
        arrays["_homeostatic_correction_rate"] = snapshot._homeostatic_correction_rate[:n_gs]
        if snapshot._popvec_snapshots is not None and len(snapshot._popvec_snapshot_times) > 0:
            n_pv = len(snapshot._popvec_snapshot_times)
            arrays["_popvec_snapshots"] = snapshot._popvec_snapshots[:n_pv]

    # ── State buffers ─────────────────────────────────────────
    if snapshot._voltages is not None:
        arrays["_voltages"] = snapshot._voltages[: snapshot._n_recorded]
    n_cs = snapshot._cond_sample_step
    if n_cs > 0:
        if snapshot._g_exc_samples is not None:
            arrays["_g_exc_samples"]    = snapshot._g_exc_samples[:n_cs]
        if snapshot._g_inh_samples is not None:
            arrays["_g_inh_samples"]    = snapshot._g_inh_samples[:n_cs]
        if snapshot._g_nmda_samples is not None:
            arrays["_g_nmda_samples"]   = snapshot._g_nmda_samples[:n_cs]
        if snapshot._g_gaba_b_samples is not None:
            arrays["_g_gaba_b_samples"] = snapshot._g_gaba_b_samples[:n_cs]
        if snapshot._g_apical_samples is not None:
            arrays["_g_apical_samples"] = snapshot._g_apical_samples[:n_cs]
        if snapshot._g_plateau_samples is not None:
            arrays["_g_plateau_samples"] = snapshot._g_plateau_samples[:n_cs]

    if snapshot._v_dend_samples is not None:
        arrays["_v_dend_samples"] = snapshot._v_dend_samples[: snapshot._n_recorded]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"  \u2713 Saved recorder snapshot \u2192 {path}")


def _reconstruct_thresholds(d: Dict[str, Any]) -> HealthThresholds:
    """Reconstruct :class:`HealthThresholds` from a nested dict.

    Snapshots produce nested dicts (``{"firing": {...}, ...}``).
    """
    kwargs: Dict[str, Any] = {}
    for group_name, cls in _THRESHOLD_GROUPS.items():
        sub = d.get(group_name, {})
        if isinstance(sub, dict):
            kwargs[group_name] = cls(**sub)
        else:
            kwargs[group_name] = sub
    overrides = d.get("regional_overrides", {})
    if overrides:
        kwargs["regional_overrides"] = overrides
    return HealthThresholds(**kwargs)


def load_snapshot(path: str) -> RecorderSnapshot:
    """Load a snapshot previously saved with :meth:`save_snapshot`."""
    if not path.endswith(".npz"):
        path = path + ".npz"

    data = np.load(path, allow_pickle=True)
    meta: dict = json.loads(str(data["_meta"].item()))

    # ── Config ──────────────────────────────────────────────────────────
    cfg_dict = meta["config"]
    thresh_dict = cfg_dict.pop("thresholds", {})
    thresholds = _reconstruct_thresholds(thresh_dict)
    config = DiagnosticsConfig(**cfg_dict, thresholds=thresholds)

    dt_ms: float = float(meta["dt_ms"])
    n_recorded: int = int(meta["n_recorded"])
    gain_sample_step: int = int(meta["gain_sample_step"])
    cond_sample_step: int = int(meta["cond_sample_step"])
    gain_sample_times: List[int] = [int(x) for x in meta["gain_sample_times"]]

    # ── Index ────────────────────────────────────────────────────────────
    pop_keys: List[Tuple[str, str]] = [
        (str(rn), str(pn)) for rn, pn in meta["pop_keys"]
    ]
    pop_index: Dict[Tuple[str, str], int] = {k: i for i, k in enumerate(pop_keys)}
    n_pops = len(pop_keys)

    pop_sizes: np.ndarray = data["_pop_sizes"]

    region_keys: List[str] = [str(r) for r in meta["region_keys"]]
    region_index: Dict[str, int] = {r: i for i, r in enumerate(region_keys)}
    n_regions = len(region_keys)
    region_pop_indices: Dict[str, List[int]] = {
        str(rn): [int(x) for x in v]
        for rn, v in meta["region_pop_indices"].items()
    }

    tract_keys: List[SynapseId] = [
        SynapseId.from_key(k) for k in meta["tract_keys"]
    ]
    tract_index: Dict[SynapseId, int] = {k: i for i, k in enumerate(tract_keys)}
    n_tracts = len(tract_keys)

    stp_keys: List[Tuple[str, SynapseId]] = [
        (str(rn), SynapseId.from_key(k)) for rn, k in meta["stp_keys"]
    ]
    nm_receptor_keys: List[Tuple[str, str]] = [
        (str(rn), str(p)) for rn, p in meta["nm_receptor_keys"]
    ]
    n_nm_receptors = len(nm_receptor_keys)
    nm_source_pop_keys: List[Tuple[str, str]] = [
        (str(rn), str(pn)) for rn, pn in meta.get("nm_source_pop_keys", [])
    ]

    # ── Static brain metadata ────────────────────────────────────────────
    pop_polarities: Dict[Tuple[str, str], str] = {
        (str(rn), str(pn)): str(pol)
        for rn, pn, pol in meta.get("pop_polarities", [])
    }
    tract_delay_ms: List[float] = [float(d) for d in meta.get("tract_delay_ms", [])]
    homeostasis_target_hz: Dict[Tuple[str, str], float] = {
        (str(rn), str(pn)): float(hz)
        for rn, pn, hz in meta.get("homeostasis_target_hz", [])
    }
    stp_configs: List[Tuple[float, float, float]] = [
        (float(U), float(tau_d), float(tau_f))
        for U, tau_d, tau_f in meta.get("stp_configs", [])
    ]
    stp_final_state: Dict[str, Dict[str, float]] = {
        str(k): {str(sk): float(sv) for sk, sv in v.items()}
        for k, v in meta.get("stp_final_state", {}).items()
    }
    tract_weight_stats: Dict[str, Dict[str, float]] = {
        str(k): {str(sk): float(sv) for sk, sv in v.items()}
        for k, v in meta.get("tract_weight_stats", {}).items()
    }
    pop_neuron_params: Dict[Tuple[str, str], Dict[str, float]] = {
        (str(rn), str(pn)): {str(pk): float(pv) for pk, pv in params.items()}
        for rn, pn, params in meta.get("pop_neuron_params", [])
    }

    # ── Sample indices ───────────────────────────────────────────────────
    v_sample_idx: List[np.ndarray] = [
        data[f"_v_idx_{i}"] for i in range(n_pops)
    ]
    c_sample_idx: List[np.ndarray] = [
        data[f"_c_idx_{i}"] for i in range(n_pops)
    ]

    # ── Spike buffers ────────────────────────────────────────────────────
    pop_spike_counts: np.ndarray = data["_pop_spike_counts"]
    region_spike_counts: np.ndarray = data["_region_spike_counts"]
    tract_sent_raw = data["_tract_sent"] if n_tracts > 0 and "_tract_sent" in data else None
    tract_sent: np.ndarray = (
        tract_sent_raw if tract_sent_raw is not None
        else np.zeros((n_recorded, 0), dtype=np.int32)
    )

    per_neuron_spike_counts: List[np.ndarray] = [
        data[f"_per_neuron_{i}"] for i in range(n_pops)
    ]

    # ── Spike times ──────────────────────────────────────────────────────
    spike_times: Dict[Tuple[str, str], List[List[int]]] = {}
    for pop_idx, key in enumerate(pop_keys):
        nidx_key = f"_st_nidx_{pop_idx}"
        ts_key   = f"_st_ts_{pop_idx}"
        if nidx_key in data:
            nidx_arr = data[nidx_key]
            ts_arr   = data[ts_key]
            n_neurons_pop = int(pop_sizes[pop_idx])
            nested: List[List[int]] = [[] for _ in range(n_neurons_pop)]
            for ni_val, ts_val in zip(nidx_arr.tolist(), ts_arr.tolist()):
                nested[int(ni_val)].append(int(ts_val))
            spike_times[key] = nested

    # ── Trajectory buffers ────────────────────────────────────────────────
    g_L_scale_history: np.ndarray = data["_g_L_scale_history"]
    stp_efficacy_history: np.ndarray = data["_stp_efficacy_history"]
    nm_concentration_history: np.ndarray = data["_nm_concentration_history"]

    # ── State buffers ─────────────────────────────────────────
    voltages: Optional[np.ndarray] = (
        data["_voltages"] if "_voltages" in data else None
    )
    g_exc_samples: Optional[np.ndarray] = (
        data["_g_exc_samples"] if "_g_exc_samples" in data else None
    )
    g_inh_samples: Optional[np.ndarray] = (
        data["_g_inh_samples"] if "_g_inh_samples" in data else None
    )
    g_nmda_samples: Optional[np.ndarray] = (
        data["_g_nmda_samples"] if "_g_nmda_samples" in data else None
    )
    g_gaba_b_samples: Optional[np.ndarray] = (
        data["_g_gaba_b_samples"] if "_g_gaba_b_samples" in data else None
    )
    g_apical_samples: Optional[np.ndarray] = (
        data["_g_apical_samples"] if "_g_apical_samples" in data else None
    )
    v_dend_samples: Optional[np.ndarray] = (
        data["_v_dend_samples"] if "_v_dend_samples" in data else None
    )
    g_plateau_samples: Optional[np.ndarray] = (
        data["_g_plateau_samples"] if "_g_plateau_samples" in data else None
    )

    # ── Learning trajectory buffers ──────────────────────────────────────
    learning_keys_raw = meta.get("learning_keys", [])
    learning_keys: List[Tuple[str, str]] = [
        (str(k[0]), str(k[1])) if isinstance(k, (list, tuple)) and len(k) == 2
        else (str(k), "") for k in learning_keys_raw
    ]
    bcm_keys: List[int] = [int(k) for k in meta.get("bcm_keys", [])]
    popvec_snapshot_times: List[int] = [int(t) for t in meta.get("popvec_snapshot_times", [])]
    n_learn = len(learning_keys)
    n_bcm = len(bcm_keys)

    weight_dist_history = (
        data["_weight_dist_history"] if "_weight_dist_history" in data
        else np.full((gain_sample_step, max(n_learn, 1), 5), np.nan, dtype=np.float32)
    )
    weight_update_magnitude_history = (
        data["_weight_update_magnitude_history"] if "_weight_update_magnitude_history" in data
        else np.full((gain_sample_step, max(n_learn, 1)), np.nan, dtype=np.float32)
    )
    eligibility_mean_history = (
        data["_eligibility_mean_history"] if "_eligibility_mean_history" in data
        else np.full((gain_sample_step, max(n_learn, 1)), np.nan, dtype=np.float32)
    )
    eligibility_ltp_ltd_ratio_history = (
        data["_eligibility_ltp_ltd_ratio_history"] if "_eligibility_ltp_ltd_ratio_history" in data
        else np.full((gain_sample_step, max(n_learn, 1)), np.nan, dtype=np.float32)
    )
    bcm_theta_history = (
        data["_bcm_theta_history"] if "_bcm_theta_history" in data
        else np.full((gain_sample_step, max(n_bcm, 1)), np.nan, dtype=np.float32)
    )
    homeostatic_correction_rate = (
        data["_homeostatic_correction_rate"] if "_homeostatic_correction_rate" in data
        else np.full((gain_sample_step, n_pops), np.nan, dtype=np.float32)
    )
    popvec_snapshots: Optional[np.ndarray] = (
        data["_popvec_snapshots"] if "_popvec_snapshots" in data else None
    )

    return RecorderSnapshot(
        config=config,
        dt_ms=dt_ms,
        _pop_keys=pop_keys,
        _pop_index=pop_index,
        _n_pops=n_pops,
        _pop_sizes=pop_sizes,
        _region_keys=region_keys,
        _region_index=region_index,
        _n_regions=n_regions,
        _region_pop_indices=region_pop_indices,
        _tract_keys=tract_keys,
        _tract_index=tract_index,
        _n_tracts=n_tracts,
        _stp_keys=stp_keys,
        _nm_receptor_keys=nm_receptor_keys,
        _n_nm_receptors=n_nm_receptors,
        _nm_source_pop_keys=nm_source_pop_keys,
        _v_sample_idx=v_sample_idx,
        _c_sample_idx=c_sample_idx,
        _n_recorded=n_recorded,
        _gain_sample_step=gain_sample_step,
        _cond_sample_step=cond_sample_step,
        _gain_sample_times=gain_sample_times,
        _pop_spike_counts=pop_spike_counts,
        _per_neuron_spike_counts=per_neuron_spike_counts,
        _region_spike_counts=region_spike_counts,
        _tract_sent=tract_sent,
        _spike_times=spike_times,
        _voltages=voltages,
        _g_exc_samples=g_exc_samples,
        _g_inh_samples=g_inh_samples,
        _g_nmda_samples=g_nmda_samples,
        _g_gaba_b_samples=g_gaba_b_samples,
        _g_apical_samples=g_apical_samples,
        _v_dend_samples=v_dend_samples,
        _g_plateau_samples=g_plateau_samples,
        _g_L_scale_history=g_L_scale_history,
        _stp_efficacy_history=stp_efficacy_history,
        _nm_concentration_history=nm_concentration_history,
        _pop_polarities=pop_polarities,
        _tract_delay_ms=tract_delay_ms,
        _homeostasis_target_hz=homeostasis_target_hz,
        _stp_configs=stp_configs,
        _stp_final_state=stp_final_state,
        _tract_weight_stats=tract_weight_stats,
        _pop_neuron_params=pop_neuron_params,
        _learning_keys=learning_keys,
        _weight_dist_history=weight_dist_history,
        _weight_update_magnitude_history=weight_update_magnitude_history,
        _eligibility_mean_history=eligibility_mean_history,
        _eligibility_ltp_ltd_ratio_history=eligibility_ltp_ltd_ratio_history,
        _bcm_theta_history=bcm_theta_history,
        _bcm_keys=bcm_keys,
        _homeostatic_correction_rate=homeostatic_correction_rate,
        _popvec_snapshots=popvec_snapshots,
        _popvec_snapshot_times=popvec_snapshot_times,
    )


# =============================================================================
# REPORT SAVE
# =============================================================================


def save_report(report: DiagnosticsReport, output_dir: str) -> None:
    """Save report summary (JSON) and raw traces (NPZ) to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    def _clean(v: Any) -> Any:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v

    summary: Dict[str, Any] = {
        "timestamp": report.timestamp,
        "simulation_time_ms": report.simulation_time_ms,
        "n_timesteps": report.n_timesteps,
        "performance": report.performance.to_dict() if report.performance else None,
        "health": report.health.to_dict(),
        "region_firing_rates_hz": {
            rn: _clean(rs.mean_fr_hz) for rn, rs in report.regions.items()
        },
        "population_firing_rates_hz": {
            f"{rn}:{pn}": _clean(ps.mean_fr_hz)
            for rn, rs in report.regions.items()
            for pn, ps in rs.populations.items()
        },
        "stp_final_state": report.homeostasis.stp_final_state,
        "global_dominant_freq_hz": _clean(report.oscillations.global_dominant_freq_hz),
        "global_band_power": {k: _clean(v) for k, v in report.oscillations.global_band_power.items()},
        "region_aperiodic_exponent": {
            k: _clean(v) for k, v in report.oscillations.region_aperiodic_exponent.items()
        },
        "neuromodulator_peak_conc": {
            key: _clean(float(np.nanmax(traj)))
            for key, traj in (report.neuromodulator_levels or {}).items()
        } if report.neuromodulator_levels else {},
        "connectivity_jitter_ms": {
            str(ts.synapse_id): _clean(ts.transmission_jitter_ms)
            for ts in report.connectivity.tracts
        },
        "inferred_brain_state": report.health.inferred_brain_state,
    }

    # ── Learning summary ─────────────────────────────────────────────────
    if report.learning is not None:
        learning_json: Dict[str, Any] = {
            "popvec_stability": _clean(report.learning.popvec_stability),
            "da_eligibility_alignment": {
                k: _clean(v) for k, v in report.learning.da_eligibility_alignment.items()
            },
            "synapse_summaries": {},
        }
        for key, s in report.learning.synapse_summaries.items():
            learning_json["synapse_summaries"][key] = {
                "strategy_type": s.strategy_type,
                "weight_drift": _clean(s.weight_drift),
                "mean_update_magnitude": _clean(s.mean_update_magnitude),
                "mean_eligibility": _clean(s.mean_eligibility),
                "ltp_ltd_ratio": _clean(s.ltp_ltd_ratio),
                "weight_start_mean": _clean(s.weight_start.mean),
                "weight_end_mean": _clean(s.weight_end.mean),
                "weight_end_std": _clean(s.weight_end.std),
                "weight_end_sparsity": _clean(s.weight_end.sparsity),
            }
        if report.learning.stdp_timing:
            learning_json["stdp_timing"] = {
                key: {
                    "mean_delta_ms": _clean(t.mean_delta_ms),
                    "std_delta_ms": _clean(t.std_delta_ms),
                    "ltp_fraction": _clean(t.ltp_fraction),
                    "n_pairs": t.n_pairs,
                }
                for key, t in report.learning.stdp_timing.items()
            }
        summary["learning"] = learning_json

    json_path = os.path.join(output_dir, "diagnostics_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  \u2713 Saved JSON summary  \u2192 {json_path}")

    # NPZ traces
    traces: Dict[str, np.ndarray] = {}
    if report.raw_spike_counts is not None:
        traces["pop_spike_counts"] = report.raw_spike_counts
    if report.pop_rate_binned is not None:
        traces["pop_rate_binned"] = report.pop_rate_binned
    if report.raw_voltages is not None:
        traces["voltages"] = report.raw_voltages
    if report.raw_g_exc is not None:
        traces["g_exc"] = report.raw_g_exc
    if report.raw_g_inh is not None:
        traces["g_inh"] = report.raw_g_inh
    for key, traj in report.homeostasis.gain_trajectories.items():
        traces[f"gain_{key.replace(':', '_')}"] = traj
    if report.neuromodulator_levels:
        for key, traj in report.neuromodulator_levels.items():
            traces[f"nm_{key.replace('/', '_').replace('.', '_')}"] = traj
    if traces:
        npz_path = os.path.join(output_dir, "diagnostics_traces.npz")
        np.savez_compressed(npz_path, **traces)
        print(f"  \u2713 Saved NPZ traces   \u2192 {npz_path}")
