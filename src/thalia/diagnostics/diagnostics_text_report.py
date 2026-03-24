"""Console text-report printing for diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .diagnostics_report import DiagnosticsReport

if TYPE_CHECKING:
    from thalia.brain import Brain
    from thalia.brain.configs import BrainConfig


_REPORT_WIDTH = 100


def _sep(char: str = "=") -> str:
    return char * _REPORT_WIDTH


def _print_health_section(report: DiagnosticsReport) -> None:
    print(f"\n{_sep('─')}")
    print("HEALTH")
    print(_sep('─'))
    h = report.health
    print(f"  Brain state         : {h.global_brain_state}")
    print(f"  Inferred physiology : {h.inferred_brain_state}")
    print(
        f"  Biological check    : "
        f"{h.n_populations_ok} ok  |  {h.n_populations_low} low  |  "
        f"{h.n_populations_high} high  |  {h.n_populations_unknown} unknown"
    )
    if h.critical_issues:
        print(f"\n  \U0001f534 CRITICAL ({len(h.critical_issues)}):")
        for iss in h.critical_issues:
            print(f"    \u2022 {iss}")
    if h.warnings:
        print(f"\n  \u26a0  WARNINGS ({len(h.warnings)}):")
        for w_msg in h.warnings:
            print(f"    \u2022 {w_msg}")
    if not h.critical_issues and not h.warnings:
        print("  \u2713 No issues")


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
        icon = "\u2713" if rs.is_active else "\u2717"
        if not np.isnan(rs.ei_ratio):
            ei_label = "(A+N)/I" if not np.isnan(rs.mean_g_nmda) else "E/I"
            ei_str = f"  {ei_label}={rs.ei_ratio:.2f}"
            if not np.isnan(rs.ei_current_ratio):
                ei_str += f"  I_E/I_I={rs.ei_current_ratio:.2f}"
        else:
            ei_str = ""
        print(f"  {icon} {rn}  {rs.mean_fr_hz:.2f} Hz avg | {rs.total_spikes:,} spikes{ei_str}")

        if detailed:
            for pn, ps in rs.populations.items():
                bio_str = ""
                if ps.bio_range_hz is not None:
                    lo, hi = ps.bio_range_hz
                    bio_str = f"  [target {lo:.0f}\u2013{hi:.0f} Hz]"
                status_icon = {"ok": "\u2713", "low": "\u26a0", "high": "\u26a0", "unknown": " "}[ps.bio_plausibility]
                isi_str = ""
                if not np.isnan(ps.isi_cv):
                    cv2_part = f"/CV\u2082={ps.isi_cv2:.2f}" if not np.isnan(ps.isi_cv2) else ""
                    isi_str = f"  CV={ps.isi_cv:.2f}{cv2_part}"
                sfa_str = ""
                if not np.isnan(ps.sfa_index):
                    sfa_str = f"  SFA={ps.sfa_index:.2f}"
                    if not np.isnan(ps.sfa_tau_ms):
                        sfa_str += f"(\u03c4={ps.sfa_tau_ms:.0f}ms)"
                ff_str = ""
                if not np.isnan(ps.per_neuron_ff):
                    ff_str = f"  FF={ps.per_neuron_ff:.2f}"
                refrac_str = ""
                if not np.isnan(ps.fraction_refractory_violations) and ps.fraction_refractory_violations > 0:
                    refrac_str = f"  \u26d4refrac={ps.fraction_refractory_violations * 100:.1f}%"
                state_str = ""
                if ps.network_state != "unknown":
                    state_str = f"  [{ps.network_state}]"
                apical_str = ""
                if not np.isnan(ps.mean_g_exc_apical):
                    apical_str = f"  g_apical={ps.mean_g_exc_apical:.4f}"
                dend_str = ""
                if not np.isnan(ps.bap_attenuation_ratio):
                    dend_str += f"  bAP={ps.bap_attenuation_ratio:.3f}"
                if not np.isnan(ps.nmda_plateau_fraction) and ps.nmda_plateau_fraction > 0:
                    dend_str += f"  plateau={ps.nmda_plateau_fraction:.1%}"
                if not np.isnan(ps.coincidence_gain):
                    dend_str += f"  coinc={ps.coincidence_gain:.2f}×"
                print(
                    f"    {status_icon} {pn:30s} {ps.mean_fr_hz:6.2f} Hz  "
                    f"({ps.n_neurons} neurons, {ps.total_spikes:,} spikes)"
                    f"{isi_str}{sfa_str}{ff_str}{refrac_str}{state_str}{bio_str}{apical_str}{dend_str}"
                )


def _print_oscillations_section(report: DiagnosticsReport, detailed: bool) -> None:
    print(f"\n{_sep('─')}")
    print("OSCILLATORY DYNAMICS")
    print(_sep('─'))

    osc = report.oscillations

    print(f"  Global dominant: {osc.global_dominant_freq_hz:.1f} Hz")
    if not np.isnan(osc.freq_resolution_hz):
        res_flag = "  \u26a0" if osc.freq_resolution_hz > 1.0 else "   "
        print(f"{res_flag} Spectral frequency resolution: {osc.freq_resolution_hz:.2f} Hz")
    else:
        print("  Spectral frequency resolution: N/A (too few recorded spikes)")

    if not np.isnan(osc.avalanche.exponent):
        exp_flag = "  \u26a0" if osc.avalanche.exponent > -1.0 or osc.avalanche.exponent < -2.5 else "   "
        sigma_str = (
            f"  \u03c3={osc.avalanche.branching_ratio:.3f}"
            if not np.isnan(osc.avalanche.branching_ratio)
            else ""
        )
        sigma_flag = (
            "  \u26a0"
            if not np.isnan(osc.avalanche.branching_ratio) and osc.avalanche.branching_ratio > 1.05
            else "   "
        )
        print(
            f"{exp_flag} Avalanche exponent: {osc.avalanche.exponent:.2f}"
            f"  R\u00b2={osc.avalanche.r2:.2f}"
            f"{sigma_str}"
            + (f"  (supercritical)" if sigma_flag.strip() == "\u26a0" else "")
        )
    else:
        print("  Avalanche exponent: N/A (too few avalanches detected)")

    if osc.region_aperiodic_exponent:
        chis = list(osc.region_aperiodic_exponent.values())
        mean_chi = float(np.nanmean(chis))
        print(f"  Aperiodic (1/f) exponent: mean χ={mean_chi:.2f}  (n={len(chis)} regions)")
    else:
        print("  Aperiodic (1/f) exponent: N/A (insufficient data)")

    print("  Global band power:")
    for band, pwr in osc.global_band_power.items():
        bar = "\u2588" * int(pwr * 30)
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

        if osc.region_aperiodic_exponent:
            print("\n  Per-region aperiodic (1/f) exponent:")
            low = report.oscillations.region_aperiodic_exponent
            for rn in report.region_keys or []:
                chi = low.get(rn, float("nan"))
                if np.isnan(chi):
                    print(f"    {rn:35s}  χ = N/A")
                else:
                    flag = "  ⚠" if chi < 0.5 or chi > 3.0 else "   "
                    print(f"  {flag} {rn:35s}  χ = {chi:.2f}")

        if osc.cfc_results:
            _TYPE_LABELS = {"pac": "PAC", "aac": "AAC", "pfc": "PFC"}
            print("\n  Cross-frequency coupling (CFC):")
            if osc.lfp_proxy_methods:
                methods_used = set(osc.lfp_proxy_methods.values())
                _METHOD_LABELS = {"current": "current-based", "spike_rate": "spike-rate"}
                methods_str = ", ".join(sorted(_METHOD_LABELS.get(m, m) for m in methods_used))
                print(f"    LFP proxy: {methods_str}")

            for r in sorted(osc.cfc_results, key=lambda x: (x.region, x.coupling_type)):
                label = _TYPE_LABELS.get(r.coupling_type, r.coupling_type.upper())
                band_pair = f"{r.phase_band}\u2013{r.amp_band}"
                if np.isnan(r.value):
                    print(f"    {r.region:35s}  {band_pair:12s} {label} = N/A")
                else:
                    bar = "\u2588" * max(0, int(r.value * 50))
                    flag = "  \u26a0" if r.value < 0.01 else "   "
                    print(f"  {flag} {r.region:35s}  {band_pair:12s} {label} = {r.value:.4f}  {bar}")

        if osc.hfo_band_power:
            print("\n  HFO band power 100\u2013250 Hz (CA1 populations):")
            for rn, pwr in sorted(osc.hfo_band_power.items()):
                flag = "  \u26a0" if pwr > 0.05 else "   "
                print(f"  {flag} {rn:35s}  {pwr:.4f}")

        if osc.swr_ca3_ca1_coupling:
            print("\n  SWR CA3\u2192CA1 coupling (10\u201330 ms causal cross-correlation):")
            for rn_swr, coupling in sorted(osc.swr_ca3_ca1_coupling.items()):
                xcorr = coupling.ca3_ca1_xcorr_peak
                lag_ms = coupling.ca3_ca1_lag_ms
                hfo_pwr = osc.hfo_band_power.get(rn_swr, 0.0)
                if np.isnan(xcorr):
                    print(f"     {rn_swr:35s}  xcorr=N/A  (silent or constant activity)")
                else:
                    coupled = xcorr >= 0.05 and hfo_pwr > 0.01
                    flag_swr = "   " if coupled else "  \u26a0"
                    lag_str = f"  lag={lag_ms:.0f}ms" if not np.isnan(lag_ms) else ""
                    print(
                        f"  {flag_swr} {rn_swr:35s}"
                        f"  xcorr={xcorr:.3f}{lag_str}  HFO={hfo_pwr:.4f}"
                    )

        if osc.beta_burst_stats:
            print("\n  Beta burst analysis \u2014 BG / motor cortex (13\u201330 Hz, 75th-pct threshold):")
            for rn_bb, bb in sorted(osc.beta_burst_stats.items()):
                n_b = bb.n_bursts
                max_d = bb.max_duration_ms
                mean_d = bb.mean_duration_ms
                mean_ibi = bb.mean_ibi_ms
                flag = "  \u26a0" if not np.isnan(max_d) and max_d > 400.0 else "   "
                ibi_str = f"  IBI={mean_ibi:.0f}ms" if not np.isnan(mean_ibi) else ""
                if np.isnan(max_d):
                    print(f"  {flag} {rn_bb:35s}  n={n_b:.0f}  no bursts \u2265100 ms")
                else:
                    print(
                        f"  {flag} {rn_bb:35s}"
                        f"  n={n_b:.0f}  mean={mean_d:.0f}ms  max={max_d:.0f}ms{ibi_str}"
                    )

        if osc.relay_burst_mode:
            print("\n  Thalamic relay burst mode (T-channel LTS):")
            for rn_rb, frac in sorted(osc.relay_burst_mode.items()):
                burst = frac >= 0.05
                flag_rb = "   " if burst else "  \u26a0"
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
            print(f"    \u2717 {bt.synapse_id}  ({bt.transmission_ratio*100:.1f}%)")
    if detailed:
        print()
        for ts in conn.tracts:
            delay_str = ""
            if not np.isnan(ts.measured_delay_ms):
                diff = ts.measured_delay_ms - ts.expected_delay_ms
                jitter_part = ""
                if not np.isnan(ts.transmission_jitter_ms):
                    jitter_part = f" jitter={ts.transmission_jitter_ms:.1f}ms"
                delay_str = (
                    f"  delay: measured={ts.measured_delay_ms:.0f}ms "
                    f"expected={ts.expected_delay_ms:.0f}ms (\u0394{diff:+.0f}ms)"
                    f"{jitter_part}"
                )
            icon = "\u2713" if ts.is_functional else "\u2717"
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
            status = "\u26a0" if final < 0.3 or final > 1.9 else " "
            print(f"  {status} {key:<35s}  {init:8.3f}  {final:8.3f}  {pct:+7.1f}%")

    if hs.stp_final_state:
        print(f"\n  STP state at end of recording (mean efficacy = u\u00b7x):")
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
        flag = "\u26a0" if peak < 1e-6 else " "
        print(f"  {flag} {key:<55s}  {peak:6.3f}  {mean:6.3f}  {last:6.3f}")


def _print_performance_section(report: DiagnosticsReport) -> None:
    perf = report.performance
    if perf is None:
        return
    print(f"\n{_sep('─')}")
    print("PERFORMANCE")
    print(_sep('─'))
    print(f"  Wall-clock  : {perf.wall_clock_s:.2f} s  ({perf.us_per_step:.1f} µs/step)")
    print(f"  Forward     : {perf.forward_s:.2f} s  ({perf.forward_pct:.1f}%)")
    print(f"  Recording   : {perf.record_s:.2f} s  ({perf.record_pct:.1f}%)")
    print(f"  Overhead    : {perf.overhead_pct:.1f}%  (input generation, etc.)")
    if perf.analysis_s > 0:
        print(f"  Analysis    : {perf.analysis_s:.2f} s")


def _print_learning_section(report: DiagnosticsReport, detailed: bool) -> None:
    learning = report.learning
    if learning is None:
        return
    print(f"\n{_sep('\u2500')}")
    print("LEARNING / PLASTICITY")
    print(_sep('\u2500'))
    n_synapses = len(learning.synapse_summaries)
    print(f"  Synapses with learning : {n_synapses}")
    print(f"  Pop-vector stability   : {learning.popvec_stability:.3f}")

    # STDP timing summary
    if learning.stdp_timing:
        print(f"\n  {'Pathway':<55s} {'mean \u0394t':>8s} {'LTP%':>6s} {'#pairs':>7s}")
        print(f"  {'\u2500'*55} {'\u2500'*8} {'\u2500'*6} {'\u2500'*7}")
        for key, timing in sorted(learning.stdp_timing.items()):
            # Shorten key for display
            label = key if len(key) <= 55 else key[:52] + "..."
            print(
                f"  {label:<55s} "
                f"{timing.mean_delta_ms:>+7.1f} "
                f"{timing.ltp_fraction:>5.0%} "
                f"{timing.n_pairs:>7d}"
            )

    if detailed:
        # Per-synapse weight summary
        if learning.synapse_summaries:
            print(f"\n  {'Synapse':<55s} {'drift':>7s} {'|dW|':>9s} {'elig':>8s} {'LTP/D':>6s}")
            print(f"  {'\u2500'*55} {'\u2500'*7} {'\u2500'*9} {'\u2500'*8} {'\u2500'*6}")
            for key, s in sorted(learning.synapse_summaries.items()):
                label = key if len(key) <= 55 else key[:52] + "..."
                ltp_str = f"{s.ltp_ltd_ratio:.1f}" if not np.isnan(s.ltp_ltd_ratio) else "n/a"
                elig_str = f"{s.mean_eligibility:.2e}" if not np.isnan(s.mean_eligibility) else "n/a"
                print(
                    f"  {label:<55s} "
                    f"{s.weight_drift:>+6.0%} "
                    f"{s.mean_update_magnitude:>9.2e} "
                    f"{elig_str:>8s} "
                    f"{ltp_str:>6s}"
                )

        # DA-eligibility alignment
        if learning.da_eligibility_alignment:
            print(f"\n  DA\u2013eligibility alignment:")
            for key, align in sorted(learning.da_eligibility_alignment.items()):
                label = key if len(key) <= 55 else key[:52] + "..."
                print(f"    {label:<55s} {align:.0%}")


def print_report(report: DiagnosticsReport, detailed: bool = True) -> None:
    """Print a formatted text report to stdout."""
    print(f"\n{_sep()}")
    print("THALIA BRAIN DIAGNOSTICS REPORT")
    print(_sep())
    print(f"  Simulation time : {report.simulation_time_ms:.0f} ms  ({report.n_timesteps} timesteps)")
    print(f"  Regions         : {len(report.regions)}")
    n_total = sum(len(r.populations) for r in report.regions.values())
    print(f"  Populations     : {n_total}")
    _print_performance_section(report)
    _print_health_section(report)
    _print_region_section(report, detailed)
    _print_oscillations_section(report, detailed)
    _print_connectivity_section(report, detailed)
    _print_homeostasis_section(report)
    _print_neuromodulators_section(report)
    _print_learning_section(report, detailed)
    print(f"\n{_sep()}\n")


# =============================================================================
# BRAIN SUMMARY HELPERS
# =============================================================================


def print_brain_config(brain: Brain) -> None:
    """Print a compact overview of the brain's region list."""
    print(f"\n{'\u2550'*80}")
    print("BRAIN CONFIGURATION")
    print(f"{'\u2550'*80}")
    config: BrainConfig = brain.config
    for field in config.__dataclass_fields__.values():
        value = getattr(config, field.name)
        print(f"  {field.name:25s}: {value}")
    print(f"  Device        : {brain.device}")
    print(f"  Axonal tracts : {len(brain.axonal_tracts)}")
    print(f"  Regions       : {len(brain.regions)}")
    for region_name in brain.regions:
        print(f"    - {region_name}")


def print_neuron_populations(brain: Brain) -> None:
    """Print per-population neuron counts for every region."""
    print(f"\n{'\u2550'*80}")
    print("NEURON POPULATION SIZES")
    print(f"{'\u2550'*80}")
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
    print(f"\n{'\u2550'*80}")
    print(heading)
    print(f"{'\u2550'*80}")
    for region in brain.regions.values():
        for synapse_id, weights in region.synaptic_weights.items():
            stp = region.stp_modules.get(synapse_id, None)
            if stp is not None:
                stp_str = (
                    f"STP U={stp.config.U:.2f}  "
                    f"\u03c4d={stp.config.tau_d:.0f}ms  "
                    f"\u03c4f={stp.config.tau_f:.0f}ms"
                )
            else:
                stp_str = "no STP"
            shape_str = "\u00d7".join(str(d) for d in weights.shape)
            print(
                f"  {str(synapse_id):<90s}"
                f"  {shape_str:>11s}  "
                f"\u03bc={weights.mean():.5f}  "
                f"\u03c3={weights.std():.5f}  "
                f"min={weights.min():.5f}  "
                f"max={weights.max():.5f}  "
                f"[{stp_str}]"
            )
