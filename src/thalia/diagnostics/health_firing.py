"""Firing-rate health checks — per-population biological plausibility."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .bio_ranges import adaptation_expected_for
from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    HealthThresholds,
    PopulationStats,
)
from .region_tags import DA_SOURCE_TAGS, UPDOWN_STATE_TAGS, matches_any


def _check_fr_ranges(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
    population_status: Dict[str, str],
) -> None:
    """Silent / low / high firing-rate range checks."""
    for (rn, pn), ps in pop_stats.items():
        pop_key = f"{rn}:{pn}"
        status = ps.bio_plausibility
        population_status[pop_key] = status

        # DA burst-mode detection — compute before the bio-plausibility checks so
        # burst-fired DA neurons are not false-alarmed as "High FR".
        _is_da = (
            matches_any(rn, DA_SOURCE_TAGS)
            and "da" in pn.lower()
        )
        _da_burst_mode = (
            _is_da
            and not np.isnan(ps.isi_cv)
            and ps.isi_cv > 1.0
            and not np.isnan(ps.fraction_isi_lt_80ms)
            and ps.fraction_isi_lt_80ms > 0.20
            and (np.isnan(ps.da_burst_events_per_s) or ps.da_burst_events_per_s > 0.0)
        )

        # Critical: completely silent (no spikes at all)
        # Only flag as critical if the population is *expected* to fire (bio_range_hz[0] > 0).
        # Populations with a lower bound of 0 Hz (e.g. DG bistratified target 0–1 Hz) are
        # not required to fire in any given 500 ms window — silence is within target.
        if ps.total_spikes == 0:
            if ps.bio_range_hz is not None and ps.bio_range_hz[0] > 0:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=f"SILENT: {rn}:{pn} (expected {ps.bio_range_hz[0]:.0f}–{ps.bio_range_hz[1]:.0f} Hz)"))
            elif ps.bio_range_hz is None:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=f"Silent: {rn}:{pn} — no spikes recorded"))

        # Critical: far outside biological range
        if status == "low" and ps.bio_range_hz is not None:
            lo, _ = ps.bio_range_hz
            if ps.mean_fr_hz < lo * 0.2:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"SEVERELY LOW: {rn}:{pn} = {ps.mean_fr_hz:.1f} Hz "
                        f"(target {lo:.0f}–{ps.bio_range_hz[1]:.0f} Hz)"
                    )))
            else:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Low FR: {rn}:{pn} = {ps.mean_fr_hz:.1f} Hz "
                        f"(target {lo:.0f}–{ps.bio_range_hz[1]:.0f} Hz)"
                    )))
        elif status == "high" and ps.bio_range_hz is not None:
            _, hi = ps.bio_range_hz
            if _da_burst_mode:
                # Reward-context burst firing is biologically expected — report
                # informatively rather than as a fault.
                burst_rate_str = (
                    f"  burst_rate={ps.da_burst_events_per_s:.2f}/s"
                    if not np.isnan(ps.da_burst_events_per_s)
                    else ""
                )
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"DA burst mode (expected): {rn}:{pn}  "
                        f"FR={ps.mean_fr_hz:.1f} Hz  CV={ps.isi_cv:.2f}  "
                        f"ISI<80ms={ps.fraction_isi_lt_80ms * 100:.0f}%"
                        + burst_rate_str
                    )))
            elif ps.mean_fr_hz > hi * 5.0:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"HYPERACTIVE: {rn}:{pn} = {ps.mean_fr_hz:.0f} Hz "
                        f"(target {ps.bio_range_hz[0]:.0f}–{hi:.0f} Hz)"
                    )))
            else:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"High FR: {rn}:{pn} = {ps.mean_fr_hz:.1f} Hz "
                        f"(target {ps.bio_range_hz[0]:.0f}–{hi:.0f} Hz)"
                    )))


def _check_isi_and_fano(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
    config: HealthThresholds,
) -> None:
    """ISI CV, CV₂, Fano factor, and refractory-violation checks."""
    for (rn, pn), ps in pop_stats.items():
        pop_key = f"{rn}:{pn}"

        # ISI CV irregularity check
        if not np.isnan(ps.isi_cv):
            if ps.isi_cv < config.isi_cv_regular_threshold and adaptation_expected_for(rn, pn) is True:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Regular firing (CV={ps.isi_cv:.2f}): {rn}:{pn} "
                        f"— may indicate synchrony or adaptation saturation"
                    )))
            if ps.isi_cv > 2.5:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=f"Burst-dominated irregularity (CV={ps.isi_cv:.2f}): {rn}:{pn}"))

            # Purkinje cell pacemaker regularity check.
            # Healthy simple-spike output has CV ≈ 0.08–0.25 (Häusser & Clark 1997).
            # A Purkinje cell with CV > 0.5 has lost its intrinsic pacemaker drive —
            # the cerebellar circuit is misfiring (e.g. missing climbing-fibre input
            # reset, excess inhibition from Golgi cells, or synaptic runaway).
            if "purkinje" in pn.lower():
                if ps.isi_cv > 0.5:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Purkinje pacemaker regularity lost: {rn}:{pn}  "
                            f"CV={ps.isi_cv:.2f}  (expected < 0.25, Häusser & Clark 1997) "
                            f"— intrinsic pacemaker drive may be absent or disrupted"
                        )))
                elif ps.isi_cv > 0.25:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Purkinje regularity degraded: {rn}:{pn}  "
                            f"CV={ps.isi_cv:.2f}  (expected < 0.25) "
                            f"— mild irregularity; check climbing-fibre ratio"
                        )))

        # CV₂ local irregularity check (full mode only)
        if not np.isnan(ps.isi_cv2) and ps.total_spikes >= 6:
            if ps.isi_cv2 < config.isi_cv2_low_threshold:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Low CV₂ (local regularity): {rn}:{pn}  CV₂={ps.isi_cv2:.2f} "
                        f"— pacemaker-like inter-spike intervals; possible AHP saturation "
                        f"or oscillation lock-in"
                    )))
            elif ps.isi_cv2 > 1.8:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"High CV₂ (local burst irregularity): {rn}:{pn}  "
                        f"CV₂={ps.isi_cv2:.2f} — locally bursty discharge pattern"
                    )))

        # Fano Factor variability check (mode-dependent semantics and thresholds).
        # Full mode  → per_neuron_ff: independent of between-neuron synchrony; Poisson ≈ 1.
        # Stats mode → population_ff: scales as 1 + (N−1)·ρ; healthy AI-state FF ≈ 1 + (N−1)·0.01.
        _ff_per = ps.per_neuron_ff
        _ff_pop = ps.population_ff
        if not np.isnan(_ff_per) and ps.total_spikes >= 10:
            if _ff_per > 3.0:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"High Fano Factor: {rn}:{pn}  per-neuron FF={_ff_per:.2f} "
                        f"(Poisson≈1.0) — spike-count variance exceeds epileptiform threshold; "
                        f"check for synchronous population bursting"
                    )))
            elif _ff_per < 0.3:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Low Fano Factor: {rn}:{pn}  per-neuron FF={_ff_per:.2f} "
                        f"(Poisson≈1.0) — rigid pacemaker-like regularity; "
                        f"possible strong inhibitory entrainment"
                    )))
        if not np.isnan(_ff_pop) and ps.total_spikes >= 10:
            # Threshold: warn when implied mean pairwise correlation > 10 %.
            # FF_pop = 1 + (N−1)·ρ  →  threshold at ρ = 0.10.
            n = max(1, ps.n_neurons)
            ff_pop_high = max(3.0, 1.0 + (n - 1) * 0.10)
            if _ff_pop > ff_pop_high:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"High population Fano Factor: {rn}:{pn}  population FF={_ff_pop:.2f} "
                        f"(threshold {ff_pop_high:.1f} for N={n}, implies >10%% pairwise synchrony) — "
                        f"check for synchronous population bursting"
                    )))
            elif _ff_pop < 0.3:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Low population Fano Factor: {rn}:{pn}  population FF={_ff_pop:.2f} "
                        f"— rigid pacemaker-like regularity at population level"
                    )))

        # Refractory period violation check — ISI < 2 ms is a spike-reset bug.
        if not np.isnan(ps.fraction_refractory_violations) and ps.fraction_refractory_violations > 0:
            issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                message=(
                    f"REFRACTORY VIOLATION: {rn}:{pn}  "
                    f"{ps.fraction_refractory_violations * 100:.2f}% of ISIs < 2 ms "
                    f"— spike-reset is not enforcing the absolute refractory period"
                )))


def _check_adaptation(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
    skip_sfa_check: bool = False,
) -> None:
    """Spike-frequency adaptation index and τ plausibility checks."""
    if skip_sfa_check:
        return
    for (rn, pn), ps in pop_stats.items():
        pop_key = f"{rn}:{pn}"

        # SFA index check
        if not np.isnan(ps.sfa_index) and ps.total_spikes >= 10:
            _adaptation_exp = adaptation_expected_for(rn, pn)
            if _adaptation_exp is True and ps.sfa_index < 1.30:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"No adaptation: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(expected >1.3 for adapting cell type — missing AHP?)"
                    )))
            if _adaptation_exp is True and ps.sfa_index > 3.0:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"RUNAWAY ADAPTATION: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(>3.0) — neuron will self-silence within the first 25% of any "
                        f"sustained trial; check AHP conductance magnitude or Ca²⁺ channel gain"
                    )))
            if _adaptation_exp is False and ps.sfa_index > 2.0:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Unexpected adaptation: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(PV/FSI/TAN should be non-adapting)"
                    )))
            if _adaptation_exp is False and ps.sfa_index < 0.8:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Sub-physiological SFA in non-adapting cell: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(<0.8) — depression detected; check AHP conductance or inhibitory entrainment"
                    )))

        # SFA time-constant (τ) plausibility check
        if not np.isnan(ps.sfa_tau_ms) and ps.total_spikes >= 20:
            _adaptation_exp = adaptation_expected_for(rn, pn)
            if _adaptation_exp is True:
                if ps.sfa_tau_ms < 20.0:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Suspiciously fast adaptation τ: {rn}:{pn}  τ={ps.sfa_tau_ms:.0f} ms "
                            f"(< 20 ms — possible AMPA over-activation or missing AHP)"
                        )))
                elif ps.sfa_tau_ms > 1000.0:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Very slow adaptation τ: {rn}:{pn}  τ={ps.sfa_tau_ms:.0f} ms "
                            f"(> 1000 ms — too slow for gain control within a single trial)"
                        )))
            elif _adaptation_exp is False:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Non-adapting cell type shows FR decay: {rn}:{pn}  τ={ps.sfa_tau_ms:.0f} ms "
                        f"(PV/FSI/TAN — check AHP conductance or inhibitory entrainment)"
                    )))


def _check_synchrony_and_state(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
    config: HealthThresholds,
    sensory_pattern: str = "",
) -> None:
    """Epileptiform burst detection, pairwise correlation, network-state classifier,
    FR heterogeneity, and up/down state bimodality checks."""
    _is_slow_wave = sensory_pattern == "slow_wave"

    for (rn, pn), ps in pop_stats.items():
        pop_key = f"{rn}:{pn}"

        # Epileptiform burst detection.
        # During slow-wave (NREM) up-states, high burst co-activation is a
        # normal feature of the state — suppress the CRITICAL so that it does
        # not obscure genuine pathology in other patterns.
        if not np.isnan(ps.fraction_burst_events) and ps.fraction_burst_events > config.epileptiform_burst_threshold:
            if _is_slow_wave:
                issues.append(HealthIssue(severity="info", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Burst co-activation (expected in slow_wave up-state): {rn}:{pn} "
                        f"({ps.fraction_burst_events * 100:.1f}% of 20 ms windows exceeded threshold)"
                    )))
            else:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"EPILEPTIFORM BURSTING: {rn}:{pn} "
                        f"({ps.fraction_burst_events * 100:.1f}% of 20 ms windows "
                        f"exceed Binomial(N,{config.burst_coactivation_fraction}) mean+2\u03c3 co-activation threshold)"
                    )))

        # Pairwise correlation / AI-state check
        if not np.isnan(ps.pairwise_correlation):
            if ps.pairwise_correlation > config.pairwise_rho_critical:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"NETWORK SYNCHRONISATION: {rn}:{pn} "
                        f"pairwise \u03c1={ps.pairwise_correlation:.2f}"
                    )))
            elif ps.pairwise_correlation > config.pairwise_rho_warn and adaptation_expected_for(rn, pn) is True:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Cortical synchrony elevated (\u03c1={ps.pairwise_correlation:.2f}): "
                        f"{rn}:{pn} — possible sync state"
                    )))

        # Network state classifier check
        # Uses adaptation_expected_for() as the authoritative source for which
        # populations are expected to operate in the AI state (pyramidal,
        # relay, MSN, granule — all adapting principal cells).
        _ai_expected_ns = adaptation_expected_for(rn, pn) is True
        if _ai_expected_ns:
            if ps.network_state == "SR":
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Synchronous-Regular state: {rn}:{pn}  CV={ps.isi_cv:.2f}  "
                        f"per-neuron FF={ps.per_neuron_ff:.2f}  \u03c1={ps.pairwise_correlation:.2f}  "
                        f"\u2014 recurrent inhibition may be too weak"
                    )))
            elif ps.network_state == "SI":
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Synchronous-Irregular state: {rn}:{pn}  CV={ps.isi_cv:.2f}  "
                        f"per-neuron FF={ps.per_neuron_ff:.2f}  \u03c1={ps.pairwise_correlation:.2f}  "
                        f"\u2014 possible E/I runaway or missing lateral inhibition"
                    )))

        # FR-CV across neurons — spatial firing heterogeneity
        fr_cv_ns = ps.fr_cv_across_neurons
        if not np.isnan(fr_cv_ns) and ps.total_spikes >= 20:
            if fr_cv_ns < config.fr_heterogeneity_low and adaptation_expected_for(rn, pn) is True:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Low FR heterogeneity: {rn}:{pn}  FR-CV={fr_cv_ns:.2f} "
                        f"(expected 1.4\u20132.0) \u2014 artificially homogeneous firing; "
                        f"check parameter diversity"
                    )))
            elif fr_cv_ns > config.fr_heterogeneity_high:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Winner-take-all dynamics: {rn}:{pn}  FR-CV={fr_cv_ns:.2f} "
                        f"(>{config.fr_heterogeneity_high}) \u2014 extreme spike concentration in a minority of neurons"
                    )))

        # Up/Down state detection (bimodality coefficient, full mode)
        if not np.isnan(ps.voltage_bimodality) and ps.voltage_bimodality > config.voltage_bimodality_threshold:
            _updown_region = matches_any(rn, UPDOWN_STATE_TAGS)
            if _updown_region:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Up/Down state dynamics: {rn}:{pn}  BC={ps.voltage_bimodality:.3f} "
                        f"(>{config.voltage_bimodality_threshold:.3f}) \u2014 bimodal voltage distribution consistent with "
                        f"up/down slow oscillations"
                    )))
            ud, dd = ps.up_state_duration_ms, ps.down_state_duration_ms
            if not np.isnan(ud) and ud > 700.0:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Prolonged up state: {rn}:{pn}  up_dur={ud:.0f} ms "
                        f"(> 700 ms) \u2014 may indicate insufficient after-hyperpolarisation "
                        f"or K\u207a channel blockade; healthy NREM up states: 300\u2013500 ms "
                        f"(Steriade et al. 2001)"
                    )))
            elif not np.isnan(ud) and ud < 200.0:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Up/Down timing: {rn}:{pn}  up_dur={ud:.0f} ms "
                        f"(expected 200\u2013700 ms, Steriade et al. 2001)"
                    )))
            if not np.isnan(dd) and not (200.0 <= dd <= 700.0):
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Up/Down timing: {rn}:{pn}  down_dur={dd:.0f} ms "
                        f"(expected 200\u2013700 ms, Steriade et al. 2001)"
                    )))
            if not np.isnan(ud) and not np.isnan(dd) and dd > 0.0:
                ud_ratio = ud / dd
                if ud_ratio > 4.0:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Pathological up/down ratio: {rn}:{pn}  "
                            f"up/down={ud_ratio:.1f} (up={ud:.0f} ms  down={dd:.0f} ms)  "
                            f"(> 4.0) \u2014 expected \u2264 2.0 for healthy NREM slow oscillations; "
                            f"insufficient hyperpolarising K\u207a conductance predicts epileptiform transition "
                            f"(Steriade et al. 2001)"
                        )))


def check_population_firing(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
    population_status: Dict[str, str],
    config: HealthThresholds,
    skip_sfa_check: bool = False,
    sensory_pattern: str = "",
) -> None:
    """Per-population biological plausibility checks.

    Appends :class:`HealthIssue` objects (category ``"firing"``) to *issues*
    and populates *population_status* (``pop_key → bio_plausibility``) in-place.

    Parameters
    ----------
    config:
        Diagnostics configuration carrying the health-check thresholds.
    skip_sfa_check:
        When ``True`` the SFA index health warnings are suppressed.  Set this
        when a ramping input pattern is in use, since a monotonically increasing
        stimulus causes all populations to appear adapted regardless of cellular
        spike-frequency adaptation properties.
    sensory_pattern:
        Active sensory pattern name (e.g. ``"slow_wave"``).  When set to
        ``"slow_wave"``, burst co-activation CRITICALs are downgraded to info
        messages because high co-activation is a normal feature of NREM up-states.
    """
    _check_fr_ranges(pop_stats, issues, population_status)
    _check_isi_and_fano(pop_stats, issues, config)
    _check_adaptation(pop_stats, issues, skip_sfa_check)
    _check_synchrony_and_state(pop_stats, issues, config, sensory_pattern=sensory_pattern)
