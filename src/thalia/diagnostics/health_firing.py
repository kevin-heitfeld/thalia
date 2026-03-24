"""Firing-rate health checks — per-population biological plausibility."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .bio_ranges import adaptation_expected_for, is_pacemaker_population, skip_burst_check_for, skip_sync_check_for
from .diagnostics_config import FiringThresholds
from .diagnostics_metrics import PopulationStats
from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext
from .region_tags import PURKINJE_TAGS, UPDOWN_STATE_TAGS, matches_any
from ._helpers import is_da_burst_mode


def _check_fr_ranges(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
    population_status: Dict[str, str],
    config: FiringThresholds = FiringThresholds(),
) -> None:
    """Silent / low / high firing-rate range checks."""
    for (rn, pn), ps in pop_stats.items():
        pop_key = f"{rn}:{pn}"
        status = ps.bio_plausibility
        population_status[pop_key] = status

        # DA burst-mode detection — compute before the bio-plausibility checks so
        # burst-fired DA neurons are not false-alarmed as "High FR".
        _da_burst_mode = is_da_burst_mode(rn, pn, ps, config)

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
            if ps.mean_fr_hz < lo * config.fr_severely_low_multiplier:
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
            elif ps.mean_fr_hz > hi * config.fr_hyperactive_multiplier:
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
    config: FiringThresholds,
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
            if ps.isi_cv > config.isi_cv_burst:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=f"Burst-dominated irregularity (CV={ps.isi_cv:.2f}): {rn}:{pn}"))

            # Purkinje cell pacemaker regularity check.
            # Healthy simple-spike output has CV ≈ 0.08–0.25.
            # A Purkinje cell with CV > 0.5 has lost its intrinsic pacemaker drive —
            # the cerebellar circuit is misfiring (e.g. missing climbing-fibre input
            # reset, excess inhibition from Golgi cells, or synaptic runaway).
            if matches_any(pn, PURKINJE_TAGS):
                if ps.isi_cv > config.purkinje_cv_high:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Purkinje pacemaker regularity lost: {rn}:{pn}  "
                            f"CV={ps.isi_cv:.2f}  (expected < {config.purkinje_cv_warn}) "
                            f"— intrinsic pacemaker drive may be absent or disrupted"
                        )))
                elif ps.isi_cv > config.purkinje_cv_warn:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Purkinje regularity degraded: {rn}:{pn}  "
                            f"CV={ps.isi_cv:.2f}  (expected < {config.purkinje_cv_warn}) "
                            f"— mild irregularity; check climbing-fibre ratio"
                        )))

        # CV₂ local irregularity check — skip for known autonomous pacemakers
        # where low CV₂ is biologically expected (regular tonic firing).
        if not np.isnan(ps.isi_cv2) and ps.total_spikes >= 6:
            if ps.isi_cv2 < config.isi_cv2_low_threshold and not is_pacemaker_population(rn, pn):
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Low CV₂ (local regularity): {rn}:{pn}  CV₂={ps.isi_cv2:.2f} "
                        f"— pacemaker-like inter-spike intervals; possible AHP saturation "
                        f"or oscillation lock-in"
                    )))
            elif ps.isi_cv2 > config.isi_cv2_high:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"High CV₂ (local burst irregularity): {rn}:{pn}  "
                        f"CV₂={ps.isi_cv2:.2f} — locally bursty discharge pattern"
                    )))

        # Fano Factor variability check.
        # per_neuron_ff: independent of between-neuron synchrony; Poisson ≈ 1.
        _ff_per = ps.per_neuron_ff
        if not np.isnan(_ff_per) and ps.total_spikes >= 10:
            if _ff_per > config.fano_factor_high:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"High Fano Factor: {rn}:{pn}  per-neuron FF={_ff_per:.2f} "
                        f"(Poisson≈1.0) — spike-count variance exceeds epileptiform threshold; "
                        f"check for synchronous population bursting"
                    )))
            elif _ff_per < config.fano_factor_low:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Low Fano Factor: {rn}:{pn}  per-neuron FF={_ff_per:.2f} "
                        f"(Poisson≈1.0) — rigid pacemaker-like regularity; "
                        f"possible strong inhibitory entrainment"
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
    config: FiringThresholds = FiringThresholds(),
    skip_sfa_check: bool = False,
) -> None:
    """Spike-frequency adaptation index and τ plausibility checks."""
    if skip_sfa_check:
        return
    for (rn, pn), ps in pop_stats.items():
        pop_key = f"{rn}:{pn}"

        # SFA index check.
        # Require ≥ 30 spikes AND mean_fr_hz ≥ 5 Hz.  At low firing rates the
        # ISI exceeds tau_adapt_ms, so adaptation decays between spikes and SFA ≈ 1
        # regardless of adapt_increment — the measurement is unreliable below
        # 5 Hz in a 3-second simulation (< 15 spikes per neuron = only 2-3 ISI
        # pairs to compare, too noisy).
        if not np.isnan(ps.sfa_index) and ps.total_spikes >= 30 and ps.mean_fr_hz >= 5.0:
            _adaptation_exp = adaptation_expected_for(rn, pn)
            if _adaptation_exp is True and ps.sfa_index < config.sfa_absent:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"No adaptation: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(expected >1.3 for adapting cell type — missing AHP?)"
                    )))
            if _adaptation_exp is True and ps.sfa_index > config.sfa_runaway:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"RUNAWAY ADAPTATION: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(>{config.sfa_runaway}) — neuron will self-silence within the first 25% of any "
                        f"sustained trial; check AHP conductance magnitude or Ca²⁺ channel gain"
                    )))
            if _adaptation_exp is False and ps.sfa_index > config.sfa_unexpected:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Unexpected adaptation: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(PV/FSI/TAN should be non-adapting)"
                    )))
            if _adaptation_exp is False and ps.sfa_index < config.sfa_subphysiological:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Sub-physiological SFA in non-adapting cell: {rn}:{pn}  SFA={ps.sfa_index:.2f} "
                        f"(<0.8) — depression detected; check AHP conductance or inhibitory entrainment"
                    )))

        # SFA time-constant (τ) plausibility check
        if not np.isnan(ps.sfa_tau_ms) and ps.total_spikes >= 30 and ps.mean_fr_hz >= 5.0:
            _adaptation_exp = adaptation_expected_for(rn, pn)
            if _adaptation_exp is True:
                if ps.sfa_tau_ms < config.sfa_tau_fast_ms:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Suspiciously fast adaptation τ: {rn}:{pn}  τ={ps.sfa_tau_ms:.0f} ms "
                            f"(< 20 ms — possible AMPA over-activation or missing AHP)"
                        )))
                elif ps.sfa_tau_ms > config.sfa_tau_slow_ms:
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
    config: FiringThresholds,
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
        # Also suppress for tonically-firing pacemaker populations (GPi/GPe/SNr/
        # cerebellum Purkinje+DCN) where 100 % co-activation is expected.
        if not np.isnan(ps.fraction_burst_events) and ps.fraction_burst_events > config.epileptiform_burst_threshold:
            if skip_burst_check_for(rn, pn):
                pass  # tonic pacemaker — burst statistic is not meaningful
            elif _is_slow_wave:
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
        # Skip the CRITICAL flag for pacemaker populations where high ρ arises
        # from shared tonic drive rather than pathological network synchrony.
        if not np.isnan(ps.pairwise_correlation):
            if not skip_sync_check_for(rn, pn) and ps.pairwise_correlation > config.pairwise_rho_critical:
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

        # Up/Down state detection (bimodality coefficient)
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
            # Up/Down timing checks are only meaningful for regions where slow
            # oscillations are biologically expected.  Durations < 20 ms are
            # computational artefacts (oscillation-driven voltage fluctuations
            # misclassified as state transitions) and are suppressed entirely.
            if _updown_region and not np.isnan(ud) and ud > 20.0:
                if ud > config.up_state_max_ms:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Prolonged up state: {rn}:{pn}  up_dur={ud:.0f} ms "
                            f"(> {config.up_state_max_ms:.0f} ms) \u2014 may indicate insufficient after-hyperpolarisation "
                            f"or K\u207a channel blockade; healthy NREM up states: {config.up_state_min_ms:.0f}\u2013{config.up_state_max_ms:.0f} ms "
                            f"(Steriade et al. 2001)"
                        )))
                elif ud < config.up_state_min_ms:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Up/Down timing: {rn}:{pn}  up_dur={ud:.0f} ms "
                            f"(expected {config.up_state_min_ms:.0f}\u2013{config.up_state_max_ms:.0f} ms)"
                        )))
            if _updown_region and not np.isnan(dd) and dd > 20.0 and not (config.up_state_min_ms <= dd <= config.up_state_max_ms):
                issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                    message=(
                        f"Up/Down timing: {rn}:{pn}  down_dur={dd:.0f} ms "
                        f"(expected {config.up_state_min_ms:.0f}–{config.up_state_max_ms:.0f} ms)"
                    )))
            if _updown_region and not np.isnan(ud) and not np.isnan(dd) and dd > 20.0 and ud > 20.0:
                ud_ratio = ud / dd
                if ud_ratio > config.up_down_ratio_max:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.FIRING, population=pop_key, region=rn,
                        message=(
                            f"Pathological up/down ratio: {rn}:{pn}  "
                            f"up/down={ud_ratio:.1f} (up={ud:.0f} ms  down={dd:.0f} ms)  "
                        f"(> {config.up_down_ratio_max:.1f}) — expected ≤ 2.0 for healthy NREM slow oscillations; "
                            f"insufficient hyperpolarising K\u207a conductance predicts epileptiform transition "
                            f"(Steriade et al. 2001)"
                        )))


def check_population_firing(ctx: HealthCheckContext) -> None:
    """Per-population biological plausibility checks.

    Appends :class:`HealthIssue` objects (category ``"firing"``) to *issues*
    and populates *population_status* (``pop_key → bio_plausibility``) in-place.
    """
    pop_stats = ctx.pop_stats
    issues = ctx.issues
    population_status = ctx.population_status
    config = ctx.thresholds.firing
    skip_sfa_check = ctx.skip_sfa_check
    sensory_pattern = ctx.sensory_pattern
    _check_fr_ranges(pop_stats, issues, population_status, config)
    _check_isi_and_fano(pop_stats, issues, config)
    _check_adaptation(pop_stats, issues, config, skip_sfa_check)
    _check_synchrony_and_state(pop_stats, issues, config, sensory_pattern=sensory_pattern)
