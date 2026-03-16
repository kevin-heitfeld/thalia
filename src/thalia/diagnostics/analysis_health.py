"""Health assessment — biological plausibility checks and overall health scoring."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from .bio_ranges import (
    adaptation_expected_for,
)
from .diagnostics_types import (
    ConnectivityStats,
    HealthCategory,
    HealthIssue,
    HealthReport,
    HomeostaticStats,
    OscillatoryStats,
    PopulationStats,
    RecorderSnapshot,
    RegionStats,
)
from .health_cerebellar import (
    check_cerebellar_coupling,
    check_inferior_olive,
    check_mossy_fibre_granule_drive,
)
from .health_compartment import (
    check_two_compartment_apical_basal,
)
from .health_connectivity import (
    check_connectivity,
)
from .health_ei_oscillations import (
    check_ei_balance,
    check_integration_tau,
    check_oscillatory_bands,
    check_pac,
    check_tau_hierarchy,
)
from .health_firing import (
    check_population_firing,
)
from .health_hippocampus import (
    check_hippocampal_theta_plv,
    check_medial_septum_theta_pacemaker,
    check_swr_hfo_coupling,
    check_theta_sequence,
)
from .health_homeostasis import (
    check_homeostasis,
    check_stp_directionality,
    check_stp_final_state,
)
from .health_interneurons import (
    check_interneuron_fr_balance,
    check_interneuron_ratio,
    check_interneuron_subtype_balance,
)
from .health_laminar import (
    check_laminar_cascade,
)
from .health_neuromodulators import (
    check_d1_d2_da_balance,
    check_neuromodulator_levels,
    check_neuromodulator_phasic,
    check_neuromodulators,
    check_nm_downstream_effects,
    check_nm_oscillation_gating,
)
from .health_striatum import (
    check_basal_ganglia_pathway,
)
from .health_thalamus import (
    check_relay_burst_mode,
    check_trn_relay_gating,
)


# =============================================================================
# GLOBAL BRAIN-STATE CLASSIFIER
# =============================================================================


def _compute_global_brain_state(
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    oscillations: OscillatoryStats,
) -> str:
    """Return a slash-separated global brain-state label ``"<state>/<osc>/<crit>"``.

    Components
    ----------
    dominant_state
        Majority-vote of ``network_state`` across all populations for which
        ``adaptation_expected_for(region, pop) is True`` (pyramidal, relay,
        MSN, granule cells).  These are the principal cells whose asynchronous-
        irregular regime defines healthy waking-state dynamics.  Ties are
        broken: AI > unknown > SI > SR > burst (most- to least-healthy).

    dominant_oscillation
        Global dominant EEG band from :attr:`OscillatoryStats.global_dominant_freq_hz`.
        Returns ``"none"`` when no activity (0 Hz).

    criticality
        Derived from avalanche branching ratio σ:
        ``"supercrit"`` (σ > 1.05), ``"critical"`` (0.9 < σ ≤ 1.05),
        ``"sub-crit"`` (σ ≤ 0.9), ``"unknown"`` (avalanche data absent / NaN).

    Examples
    --------
    ``"AI/theta/sub-crit"`` — healthy hippocampal waking state.
    ``"SI/beta/supercrit"`` — pathological basal-ganglia over-synchrony.
    ``"burst/gamma/unknown"`` — epileptiform activity, short run (no avalanche).
    """
    # ── Dominant state ────────────────────────────────────────────────────
    # Only count populations where adaptation_expected is True (principal cells).
    state_counts: Dict[str, int] = {}
    for (rn, pn), ps in pop_stats.items():
        if adaptation_expected_for(rn, pn) is not True:
            continue
        if ps.total_spikes < 20:
            continue  # skip populations with insufficient data
        s = ps.network_state
        state_counts[s] = state_counts.get(s, 0) + 1

    if state_counts:
        # Priority order: AI first (most-desired outcome), then unknown, SI, SR, burst.
        priority = ["AI", "unknown", "SI", "SR", "burst"]
        max_count = max(state_counts.values())
        # Among states tied for the highest count, pick the most-healthy one.
        dominant_state = next(
            (s for s in priority if state_counts.get(s, 0) == max_count),
            max(state_counts, key=state_counts.__getitem__),
        )
    else:
        dominant_state = "unknown"

    # ── Dominant oscillation ──────────────────────────────────────────────
    dom_freq = oscillations.global_dominant_freq_hz
    if np.isnan(dom_freq) or dom_freq <= 0.0:
        dominant_osc = "none"
    else:
        from .bio_ranges import EEG_BANDS
        dominant_osc = "gamma"  # default for frequencies above all defined bands
        for band, (f_lo, f_hi) in EEG_BANDS.items():
            if f_lo <= dom_freq < f_hi:
                dominant_osc = band
                break

    # ── Criticality ───────────────────────────────────────────────────────
    sigma = oscillations.avalanche_branching_ratio
    if np.isnan(sigma):
        criticality = "unknown"
    elif sigma > 1.05:
        criticality = "supercrit"
    elif sigma > 0.9:
        criticality = "critical"
    else:
        criticality = "sub-crit"

    return f"{dominant_state}/{dominant_osc}/{criticality}"


# =============================================================================
# HEALTH ASSESSMENT — orchestrator
# =============================================================================


def assess_health(
    rec: RecorderSnapshot,
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    region_stats: Dict[str, RegionStats],
    connectivity: ConnectivityStats,
    oscillations: OscillatoryStats,
    homeostasis: HomeostaticStats,
) -> HealthReport:
    """Assess overall brain health with per-population biological plausibility.

    Each checker appends structured ``HealthIssue`` objects (with ``population``
    and ``region`` context) directly to the shared ``issues`` list, making
    ``HealthIssue`` the primary source of truth for all health findings.
    """
    issues: List[HealthIssue] = []
    population_status: Dict[str, str] = {}

    check_basal_ganglia_pathway(pop_stats, issues)
    check_cerebellar_coupling(oscillations, issues)
    check_connectivity(connectivity, rec, issues)
    check_d1_d2_da_balance(rec, pop_stats, issues)
    check_ei_balance(region_stats, issues)
    check_hippocampal_theta_plv(oscillations, issues, rec.config.thresholds)
    check_homeostasis(rec, issues, pop_stats)
    check_inferior_olive(pop_stats, issues)
    check_integration_tau(oscillations, issues)
    check_interneuron_fr_balance(rec, pop_stats, issues)
    check_interneuron_ratio(rec, issues)
    check_interneuron_subtype_balance(rec, issues)
    check_laminar_cascade(oscillations, issues)
    check_medial_septum_theta_pacemaker(pop_stats, issues)
    check_mossy_fibre_granule_drive(pop_stats, issues)
    check_neuromodulator_levels(rec, issues)
    check_neuromodulator_phasic(rec, issues)
    check_neuromodulators(rec, pop_stats, issues)
    check_nm_downstream_effects(rec, pop_stats, issues)
    check_nm_oscillation_gating(rec, oscillations, issues)
    check_oscillatory_bands(region_stats, oscillations, issues)
    check_pac(oscillations, issues)
    check_population_firing(
        pop_stats, issues, population_status,
        config=rec.config.thresholds,
        skip_sfa_check=rec.config.skip_sfa_health_check,
        sensory_pattern=rec.config.sensory_pattern,
    )
    check_relay_burst_mode(oscillations, issues, rec.config.sensory_pattern)
    check_stp_directionality(rec, issues)
    check_stp_final_state(rec, homeostasis, issues)
    check_swr_hfo_coupling(oscillations, issues)
    check_tau_hierarchy(oscillations, issues)
    check_theta_sequence(oscillations, issues)
    check_trn_relay_gating(rec, issues)
    check_two_compartment_apical_basal(rec, pop_stats, issues)

    n_ok      = sum(1 for s in population_status.values() if s == "ok")
    n_low     = sum(1 for s in population_status.values() if s == "low")
    n_high    = sum(1 for s in population_status.values() if s == "high")
    n_unknown = sum(1 for s in population_status.values() if s not in ("ok", "low", "high"))

    # Cascade-silence suppression.
    # When a region has ≥1 incoming tract and ALL of them are broken, that
    # region's silence is a downstream consequence of the upstream fault, not an
    # independent issue.  Downgrade its CRITICAL SILENT issues to warnings so that
    # the report surface shows the one root-cause broken-tract CRITICAL rather than
    # a cascade of identical SILENT CRITICALs that obscure it.
    #
    # We build the reachability map from both lists:
    #   connectivity.tracts    — functional tracts (is_functional=True)
    #   connectivity.n_broken  — broken tracts (is_functional=False)
    _all_incoming: Dict[str, Set[str]] = {}   # target → set of all source regions
    _broken_incoming: Dict[str, Set[str]] = {}  # target → set of broken source regions
    for _ts in list(connectivity.tracts) + list(connectivity.n_broken):
        _tgt = _ts.synapse_id.target_region
        _src = _ts.synapse_id.source_region
        if _tgt == _src:
            continue
        _all_incoming.setdefault(_tgt, set()).add(_src)
    for _ts in connectivity.n_broken:
        _tgt = _ts.synapse_id.target_region
        _src = _ts.synapse_id.source_region
        if _tgt == _src:
            continue
        _broken_incoming.setdefault(_tgt, set()).add(_src)
    # A region is cascade-silent iff it has incoming tracts and every one is broken.
    _cascade_silent: Set[str] = {
        _rn for _rn, _srcs in _all_incoming.items()
        if _srcs and _srcs == _broken_incoming.get(_rn, set())
    }
    if _cascade_silent:
        for _idx, _issue in enumerate(issues):
            if (
                _issue.severity == "critical"
                and _issue.category == "firing"
                and "SILENT:" in _issue.message
                and _issue.region in _cascade_silent
            ):
                _broken_srcs = sorted(_broken_incoming.get(str(_issue.region), set()))
                issues[_idx] = HealthIssue(
                    severity="warning",
                    category=HealthCategory.FIRING,
                    population=_issue.population,
                    region=_issue.region,
                    message=(
                        _issue.message.replace("SILENT:", "Cascade-silent:", 1)
                        + f"  [all inputs broken: {', '.join(_broken_srcs)}]"
                    ),
                )

    # Derive plain str lists from structured issues for HealthReport fields.
    critical = [i.message for i in issues if i.severity == "critical"]
    health_warnings = [i.message for i in issues if i.severity == "warning"]
    n_crit = len(critical)
    n_warn = len(health_warnings)
    # Normalize penalty weights by population count (groups of 10) so that the
    # score is comparable regardless of brain size.  A single-region run with
    # 1–9 populations uses no normalization (divisor=1); a 40-population run
    # divides each per-issue penalty by 4, keeping the score on the same scale.
    n_pops = max(1, len(rec._pop_keys))
    norm = max(1, n_pops // 10)
    crit_w = rec.config.thresholds.critical_weight / norm
    warn_w = rec.config.thresholds.warning_weight / norm
    stability = max(
        0.0,
        1.0 - n_crit * crit_w - n_warn * warn_w,
    )
    is_healthy = n_crit == 0 and stability > 0.7

    global_brain_state = _compute_global_brain_state(
        pop_stats, oscillations
    )

    return HealthReport(
        is_healthy=is_healthy,
        stability_score=stability,
        critical_issues=critical,
        warnings=health_warnings,
        population_status=population_status,
        n_populations_ok=n_ok,
        n_populations_low=n_low,
        n_populations_high=n_high,
        n_populations_unknown=n_unknown,
        all_issues=issues,
        global_brain_state=global_brain_state,
    )
