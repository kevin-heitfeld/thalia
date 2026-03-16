"""
Diagnostics I/O — text reporting and file saving.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from thalia.typing import SynapseId

from .diagnostics_types import (
    DiagnosticsConfig,
    DiagnosticsReport,
    HealthThresholds,
    RecorderSnapshot,
)

if TYPE_CHECKING:
    from thalia.brain import Brain


# =============================================================================
# TEXT REPORT
# =============================================================================

_REPORT_WIDTH = 80


def _sep(char: str = "=") -> str:
    return char * _REPORT_WIDTH


def _print_health_section(report: DiagnosticsReport) -> None:
    print(f"\n{_sep('─')}")
    print("HEALTH")
    print(_sep('─'))
    h = report.health
    icon = "✓ HEALTHY" if h.is_healthy else "✗ ISSUES DETECTED"
    print(f"  {icon}")
    print(f"  Brain state         : {h.global_brain_state}")
    print(f"  Stability score     : {h.stability_score:.2f}")
    print(
        f"  Biological check    : "
        f"{h.n_populations_ok} ok  |  {h.n_populations_low} low  |  "
        f"{h.n_populations_high} high  |  {h.n_populations_unknown} unknown"
    )
    if h.critical_issues:
        print(f"\n  🔴 CRITICAL ({len(h.critical_issues)}):")
        for iss in h.critical_issues:
            print(f"    • {iss}")
    if h.warnings:
        print(f"\n  ⚠  WARNINGS ({len(h.warnings)}):")
        for w_msg in h.warnings:
            print(f"    • {w_msg}")
    if not h.critical_issues and not h.warnings:
        print("  ✓ No issues")


def _print_region_section(report: DiagnosticsReport, detailed: bool) -> None:
    print(f"\n{_sep('─')}")
    print("REGION ACTIVITY")
    print(_sep('─'))
    active = [rn for rn, rs in report.regions.items() if rs.is_active]
    silent = [rn for rn, rs in report.regions.items() if not rs.is_active]
    print(f"  Active: {len(active)}/{len(report.regions)}")
    if silent:
        print(f"  Silent: {', '.join(silent)}")
    print()

    for rn, rs in report.regions.items():
        icon = "✓" if rs.is_active else "✗"
        if not np.isnan(rs.ei_ratio):
            ei_label = "(A+N)/I" if not np.isnan(rs.mean_g_nmda) else "E/I"
            ei_str = f"  {ei_label}={rs.ei_ratio:.2f}"
        else:
            ei_str = ""
        print(f"  {icon} {rn}  {rs.mean_fr_hz:.2f} Hz avg | {rs.total_spikes:,} spikes{ei_str}")

        if detailed:
            for pn, ps in rs.populations.items():
                bio_str = ""
                if ps.bio_range_hz is not None:
                    lo, hi = ps.bio_range_hz
                    bio_str = f"  [target {lo:.0f}–{hi:.0f} Hz]"
                status_icon = {"ok": "✓", "low": "⚠", "high": "⚠", "unknown": " "}[ps.bio_plausibility]
                isi_str = ""
                if not np.isnan(ps.isi_cv):
                    cv2_part = f"/CV₂={ps.isi_cv2:.2f}" if not np.isnan(ps.isi_cv2) else ""
                    isi_str = f"  CV={ps.isi_cv:.2f}{cv2_part}"
                sfa_str = ""
                if not np.isnan(ps.sfa_index):
                    sfa_str = f"  SFA={ps.sfa_index:.2f}"
                    if not np.isnan(ps.sfa_tau_ms):
                        sfa_str += f"(\u03c4={ps.sfa_tau_ms:.0f}ms)"
                ff_str = ""
                if not np.isnan(ps.per_neuron_ff):
                    ff_str = f"  FF={ps.per_neuron_ff:.2f}"
                elif not np.isnan(ps.population_ff):
                    ff_str = f"  popFF={ps.population_ff:.2f}"
                refrac_str = ""
                if not np.isnan(ps.fraction_refractory_violations) and ps.fraction_refractory_violations > 0:
                    refrac_str = f"  \u26d4refrac={ps.fraction_refractory_violations * 100:.1f}%"
                state_str = ""
                if ps.network_state != "unknown":
                    state_str = f"  [{ps.network_state}]"
                apical_str = ""
                if not np.isnan(ps.mean_g_exc_apical):
                    apical_str = f"  g_apical={ps.mean_g_exc_apical:.4f}"
                print(
                    f"    {status_icon} {pn:30s} {ps.mean_fr_hz:6.2f} Hz  "
                    f"({ps.n_neurons} neurons, {ps.total_spikes:,} spikes)"
                    f"{isi_str}{sfa_str}{ff_str}{refrac_str}{state_str}{bio_str}{apical_str}"
                )


def _print_oscillations_section(report: DiagnosticsReport, detailed: bool) -> None:
    print(f"\n{_sep('─')}")
    print("OSCILLATORY DYNAMICS")
    print(_sep('─'))

    osc = report.oscillations

    print(f"  Global dominant: {osc.global_dominant_freq_hz:.1f} Hz")
    if not np.isnan(osc.freq_resolution_hz):
        res_flag = "  ⚠" if osc.freq_resolution_hz > 1.0 else "   "
        print(f"{res_flag} Spectral frequency resolution: {osc.freq_resolution_hz:.2f} Hz")
    else:
        print("  Spectral frequency resolution: N/A (too few recorded spikes)")

    if not np.isnan(osc.avalanche_exponent):
        exp_flag = "  ⚠" if osc.avalanche_exponent > -1.0 or osc.avalanche_exponent < -2.5 else "   "
        sigma_str = (
            f"  σ={osc.avalanche_branching_ratio:.3f}"
            if not np.isnan(osc.avalanche_branching_ratio)
            else ""
        )
        sigma_flag = (
            "  ⚠"
            if not np.isnan(osc.avalanche_branching_ratio) and osc.avalanche_branching_ratio > 1.05
            else "   "
        )
        print(
            f"{exp_flag} Avalanche exponent: {osc.avalanche_exponent:.2f}"
            f"  R²={osc.avalanche_r2:.2f}"
            f"{sigma_str}"
            + (f"  (supercritical)" if sigma_flag.strip() == "⚠" else "")
        )
    else:
        print("  Avalanche exponent: N/A (too few avalanches detected)")

    print("  Global band power:")
    for band, pwr in osc.global_band_power.items():
        bar = "█" * int(pwr * 30)
        print(f"    {band:>6s}  {bar} {pwr:.3f}")

    if detailed:
        print("\n  Per-region dominant frequency:")
        for rn in report.region_keys or []:
            freq = osc.region_dominant_freq.get(rn, 0.0)
            dom_band = osc.region_dominant_band.get(rn, "?")
            if freq > 0:
                print(f"    {rn:35s}  {freq:6.1f} Hz  ({dom_band})")
            else:
                print(f"    {rn:35s}   silent")

        if osc.pac_modulation_index:
            print("\n  Theta\u2013gamma PAC (hippocampal regions):")
            for rn, mi in sorted(osc.pac_modulation_index.items()):
                if np.isnan(mi):
                    print(f"    {rn:35s}  MI = N/A  (simulation too short or fs too low)")
                else:
                    bar = "\u2588" * max(0, int(mi * 50))
                    flag = "  \u26a0" if mi < 0.01 else "   "
                    print(f"  {flag} {rn:35s}  MI = {mi:.4f}  {bar}")

        if osc.hfo_band_power:
            print("\n  HFO band power 100\u2013250 Hz (CA1 populations):")
            for rn, pwr in sorted(osc.hfo_band_power.items()):
                flag = "  \u26a0" if pwr > 0.05 else "   "  # >5 % HFO may indicate SWR or artefact
                print(f"  {flag} {rn:35s}  {pwr:.4f}")

        if osc.swr_ca3_ca1_coupling:
            print("\n  SWR CA3→CA1 coupling (10–30 ms causal cross-correlation):")
            for rn_swr, coupling in sorted(osc.swr_ca3_ca1_coupling.items()):
                xcorr = coupling.get("ca3_ca1_xcorr_peak", float("nan"))
                lag_ms = coupling.get("ca3_ca1_lag_ms", float("nan"))
                hfo_pwr = osc.hfo_band_power.get(rn_swr, 0.0)
                if np.isnan(xcorr):
                    print(f"     {rn_swr:35s}  xcorr=N/A  (silent or constant activity)")
                else:
                    coupled = xcorr >= 0.05 and hfo_pwr > 0.01
                    flag_swr = "   " if coupled else "  ⚠"
                    lag_str = f"  lag={lag_ms:.0f}ms" if not np.isnan(lag_ms) else ""
                    print(
                        f"  {flag_swr} {rn_swr:35s}"
                        f"  xcorr={xcorr:.3f}{lag_str}  HFO={hfo_pwr:.4f}"
                    )

        if osc.beta_burst_stats:
            print("\n  Beta burst analysis — BG / motor cortex (13–30 Hz, 75th-pct threshold):")
            for rn_bb, bb in sorted(osc.beta_burst_stats.items()):
                n_b = bb.get("n_bursts", 0.0)
                max_d = bb.get("max_duration_ms", float("nan"))
                mean_d = bb.get("mean_duration_ms", float("nan"))
                mean_ibi = bb.get("mean_ibi_ms", float("nan"))
                flag = "  \u26a0" if not np.isnan(max_d) and max_d > 400.0 else "   "
                ibi_str = f"  IBI={mean_ibi:.0f}ms" if not np.isnan(mean_ibi) else ""
                if np.isnan(max_d):
                    print(f"  {flag} {rn_bb:35s}  n={n_b:.0f}  no bursts ≥100 ms")
                else:
                    print(
                        f"  {flag} {rn_bb:35s}"
                        f"  n={n_b:.0f}  mean={mean_d:.0f}ms  max={max_d:.0f}ms{ibi_str}"
                    )

        if osc.relay_burst_mode:
            print("\n  Thalamic relay burst mode (T-channel LTS):")
            for rn_rb, frac in sorted(osc.relay_burst_mode.items()):
                burst = frac >= 0.05
                flag_rb = "   " if burst else "  ⚠"
                mode_str = "burst mode" if burst else "tonic mode"
                print(
                    f"  {flag_rb} {rn_rb:35s}"
                    f"  short-ISI={frac:.3f}  [{mode_str}]"
                )


def _print_connectivity_section(report: DiagnosticsReport, detailed: bool) -> None:
    print(f"\n{_sep('─')}")
    print("CONNECTIVITY")
    print(_sep('─'))
    conn = report.connectivity
    print(f"  Functional axonal tracts: {conn.n_functional}/{len(conn.tracts)}")
    if conn.n_broken:
        print("  Non-functional tracts:")
        for bt in conn.n_broken:
            print(f"    ✗ {bt.synapse_id}  ({bt.transmission_ratio*100:.1f}%)")
    if detailed:
        print()
        for ts in conn.tracts:
            delay_str = ""
            if not np.isnan(ts.measured_delay_ms):
                diff = ts.measured_delay_ms - ts.expected_delay_ms
                delay_str = (
                    f"  delay: measured={ts.measured_delay_ms:.0f}ms "
                    f"expected={ts.expected_delay_ms:.0f}ms (Δ{diff:+.0f}ms)"
                )
            icon = "✓" if ts.is_functional else "✗"
            print(
                f"  {icon} {str(ts.synapse_id):65s}"
                f"  {ts.transmission_ratio*100:5.1f}%  "
                f"({ts.spikes_sent:,} spikes){delay_str}"
            )


def _print_homeostasis_section(report: DiagnosticsReport) -> None:
    print(f"\n{_sep('─')}")
    print("HOMEOSTATIC GAINS (g_L_scale)")
    print(_sep('─'))
    hs = report.homeostasis
    if not hs.gain_trajectories:
        print("  No gain data recorded")
    else:
        print(f"  {'Population':<35s}  {'Initial':>8s}  {'Final':>8s}  {'Change':>8s}")
        print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}")
        for key in sorted(hs.gain_trajectories):
            traj = hs.gain_trajectories[key]
            valid = traj[~np.isnan(traj)]
            if len(valid) < 2:
                continue
            init, final = valid[0], valid[-1]
            pct = (final - init) / init * 100 if init > 0 else 0
            status = "⚠" if final < 0.3 or final > 1.9 else " "
            print(f"  {status} {key:<35s}  {init:8.3f}  {final:8.3f}  {pct:+7.1f}%")

    if hs.stp_final_state:
        print(f"\n  STP state at end of recording (mean efficacy = u·x):")
        for key, state in sorted(hs.stp_final_state.items()):
            print(
                f"    {key:<60s}  x={state['mean_x']:.3f}  u={state['mean_u']:.3f}"
                f"  eff={state['efficacy']:.3f}"
            )


def _print_neuromodulators_section(report: DiagnosticsReport) -> None:
    if not report.neuromodulator_levels:
        return
    print(f"\n{_sep('─')}")
    print("NEUROMODULATOR RECEPTOR CONCENTRATIONS")
    print(_sep('─'))
    print(f"  {'Receptor':<55s}  {'Peak':>6s}  {'Mean':>6s}  {'Last':>6s}")
    print(f"  {'-'*55}  {'-'*6}  {'-'*6}  {'-'*6}")
    for key in sorted(report.neuromodulator_levels):
        traj = report.neuromodulator_levels[key]
        valid = traj[~np.isnan(traj)]
        if len(valid) == 0:
            print(f"  {'  ' + key:<55s}  {'N/A':>6s}  {'N/A':>6s}  {'N/A':>6s}")
            continue
        peak = float(np.max(valid))
        mean = float(np.mean(valid))
        last = float(valid[-1])
        flag = "⚠" if peak < 1e-6 else " "
        print(f"  {flag} {key:<55s}  {peak:6.3f}  {mean:6.3f}  {last:6.3f}")


def print_report(report: DiagnosticsReport, detailed: bool = True) -> None:
    """Print a formatted text report to stdout."""
    print(f"\n{_sep()}")
    print("THALIA BRAIN DIAGNOSTICS REPORT")
    print(_sep())
    print(f"  Simulation time : {report.simulation_time_ms:.0f} ms  ({report.n_timesteps} timesteps)")
    print(f"  Mode            : {report.mode}")
    print(f"  Regions         : {len(report.regions)}")
    n_total = sum(len(r.populations) for r in report.regions.values())
    print(f"  Populations     : {n_total}")
    _print_health_section(report)
    _print_region_section(report, detailed)
    _print_oscillations_section(report, detailed)
    _print_connectivity_section(report, detailed)
    _print_homeostasis_section(report)
    _print_neuromodulators_section(report)
    print(f"\n{_sep()}\n")


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
    # HealthThresholds is a nested dataclass — asdict handles it recursively.

    meta = {
        "dt_ms": snapshot.dt_ms,
        "config": cfg_dict,
        "n_recorded": snapshot._n_recorded,
        "gain_sample_step": snapshot._gain_sample_step,
        "cond_sample_step": snapshot._cond_sample_step,
        "gain_sample_times": snapshot._gain_sample_times,
        # Index lists
        "pop_keys": [[rn, pn] for rn, pn in snapshot._pop_keys],
        "region_keys": snapshot._region_keys,
        "region_pop_indices": {rn: v for rn, v in snapshot._region_pop_indices.items()},
        "tract_keys": [sid.to_key() for sid in snapshot._tract_keys],
        "stp_keys": [[rn, sid.to_key()] for rn, sid in snapshot._stp_keys],
        "nm_receptor_keys": [[rn, path_] for rn, path_ in snapshot._nm_receptor_keys],
        "nm_source_pop_keys": [[rn, pn] for rn, pn in snapshot._nm_source_pop_keys],
        # Static brain metadata
        "pop_polarities": [[rn, pn, pol] for (rn, pn), pol in snapshot._pop_polarities.items()],
        "tract_delay_ms": snapshot._tract_delay_ms,
        "homeostasis_target_hz": [[rn, pn, hz] for (rn, pn), hz in snapshot._homeostasis_target_hz.items()],
        "stp_configs": [[U, tau_d, tau_f] for U, tau_d, tau_f in snapshot._stp_configs],
        "stp_final_state": snapshot._stp_final_state,
        # Weight stats per tract
        "tract_weight_stats": snapshot._tract_weight_stats,
        # Neuron params per population (tuple keys → list keys for JSON)
        "pop_neuron_params": [
            [rn, pn, params]
            for (rn, pn), params in snapshot._pop_neuron_params.items()
        ],
    }
    arrays["_meta"] = np.array(json.dumps(meta), dtype=object)

    # ── Pop sizes ───────────────────────────────────────────────────────
    arrays["_pop_sizes"] = snapshot._pop_sizes

    # ── Spike count buffers ─────────────────────────────────────────────
    arrays["_pop_spike_counts"]    = snapshot._pop_spike_counts[: snapshot._n_recorded]
    arrays["_region_spike_counts"] = snapshot._region_spike_counts[: snapshot._n_recorded]
    if snapshot._n_tracts > 0:
        arrays["_tract_sent"] = snapshot._tract_sent[: snapshot._n_recorded]

    # Per-neuron spike counts
    for i, arr in enumerate(snapshot._per_neuron_spike_counts):
        arrays[f"_per_neuron_{i}"] = arr

    # ── Sample indices ──────────────────────────────────────────────────
    for i, idx in enumerate(snapshot._v_sample_idx):
        arrays[f"_v_idx_{i}"] = idx
    for i, idx in enumerate(snapshot._c_sample_idx):
        arrays[f"_c_idx_{i}"] = idx

    # ── Spike times (full mode) ─────────────────────────────────────────
    # Stored as flat (nidx, ts) pairs per population.
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

    # ── Full-mode state buffers ─────────────────────────────────────────
    if snapshot.config.mode == "full":
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

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"  ✓ Saved recorder snapshot → {path}")


def load_snapshot(path: str) -> RecorderSnapshot:
    """Load a snapshot previously saved with :meth:`save_snapshot`.

    Args:
        path: Path to the ``.npz`` file (the ``.npz`` suffix is appended
                if absent).

    Returns:
        A fully reconstituted :class:`RecorderSnapshot`.
    """
    if not path.endswith(".npz"):
        path = path + ".npz"

    data = np.load(path, allow_pickle=True)
    meta: dict = json.loads(str(data["_meta"].item()))

    # ── Config ──────────────────────────────────────────────────────────
    cfg_dict = meta["config"]
    thresh_dict = cfg_dict.pop("thresholds", {})
    thresholds = HealthThresholds(**thresh_dict)
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

    # ── Full-mode state buffers ───────────────────────────────────────────
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
    )


# =============================================================================
# REPORT SAVE/LOAD
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
        "mode": report.mode,
        "is_healthy": report.health.is_healthy,
        "stability_score": _clean(report.health.stability_score),
        "critical_issues": report.health.critical_issues,
        "warnings": report.health.warnings,
        "population_count": {
            "ok": report.health.n_populations_ok,
            "low": report.health.n_populations_low,
            "high": report.health.n_populations_high,
            "unknown": report.health.n_populations_unknown,
        },
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
        "neuromodulator_peak_conc": {
            key: _clean(float(np.nanmax(traj)))
            for key, traj in (report.neuromodulator_levels or {}).items()
        } if report.neuromodulator_levels else {},
    }

    json_path = os.path.join(output_dir, "diagnostics_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved JSON summary  → {json_path}")

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
        print(f"  ✓ Saved NPZ traces   → {npz_path}")


# =============================================================================
# BRAIN SUMMARY HELPERS
# =============================================================================


def print_brain_config(brain: Brain) -> None:
    """Print a compact overview of the brain's region list."""
    print(f"\n{'═'*80}")
    print("BRAIN CONFIGURATION")
    print(f"{'═'*80}")
    print(f"  Device        : {brain.device}")
    print(f"  dt_ms         : {brain.dt_ms} ms")
    print(f"  Axonal tracts : {len(brain.axonal_tracts)}")
    print(f"  Regions       : {len(brain.regions)}")
    for region_name in brain.regions:
        print(f"    - {region_name}")


def print_neuron_populations(brain: Brain) -> None:
    """Print per-population neuron counts for every region."""
    print(f"\n{'═'*80}")
    print("NEURON POPULATION SIZES")
    print(f"{'═'*80}")
    total: int = 0
    for region_name, region in brain.regions.items():
        region_total: int = sum(int(p.n_neurons) for p in region.neuron_populations.values())
        total += region_total
        print(f"  {region_name}  [{region_total} neurons]")
        for pop_name, pop in region.neuron_populations.items():
            print(f"    {pop_name:<42s} {int(pop.n_neurons):>6d}  ({pop.__class__.__name__})")
    print(f"\n  TOTAL : {total:,} neurons")


def print_synaptic_weights(brain: Brain, heading: str = "SYNAPTIC WEIGHTS") -> None:
    """Print weight statistics and STP parameters for every synapse."""
    print(f"\n{'═'*80}")
    print(heading)
    print(f"{'═'*80}")
    for region in brain.regions.values():
        for synapse_id, weights in region.synaptic_weights.items():
            stp = region.stp_modules.get(synapse_id, None)
            if stp is not None:
                stp_str = (
                    f"STP U={stp.config.U:.2f}  "
                    f"τd={stp.config.tau_d:.0f}ms  "
                    f"τf={stp.config.tau_f:.0f}ms"
                )
            else:
                stp_str = "no STP"
            shape_str = "×".join(str(d) for d in weights.shape)
            print(
                f"  {str(synapse_id):<90s}"
                f"  {shape_str:>11s}  "
                f"μ={weights.mean():.5f}  "
                f"σ={weights.std():.5f}  "
                f"min={weights.min():.5f}  "
                f"max={weights.max():.5f}  "
                f"[{stp_str}]"
            )
