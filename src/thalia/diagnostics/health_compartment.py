"""Two-compartment neuron health checks — apical/basal conductance segregation
and dendritic computation validation (bAP, NMDA plateaus, coincidence detection)."""

from __future__ import annotations

import numpy as np

from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext
from .region_tags import CORTICAL_TAGS, matches_any
from .sensory_patterns import SENSORY_PATTERNS


def check_two_compartment_apical_basal(ctx: HealthCheckContext) -> None:
    """Check that TwoCompartmentLIF apical compartments are receiving distinct drive.

    For populations that report a non-NaN ``mean_g_exc_apical`` (i.e.
    :class:`~thalia.brain.neurons.TwoCompartmentLIF` populations), two conditions
    are checked:

    **Apical compartment silent during active drive**
    When the basal compartment (``g_E`` / ``g_E_basal``) has measurable AMPA
    drive but ``g_E_apical ≈ 0``, the apical dendrite is not receiving any
    top-down input.  This is *expected* for some patterns (e.g. ``"none"``
    / ``"background"``).  For waking-state patterns (anything other than
    ``"none"`` / ``"background"``) it indicates that the top-down feedback
    projections are either absent or disconnected.
    Threshold: ``mean_g_exc_apical < 0.02`` when ``mean_g_exc_basal > 0.01``.

    **Identically uniform apical/basal ratio**
    If ``|ratio − 1.0| < 0.02`` the two compartments are receiving almost
    exactly equal drive, which eliminates the functional distinction between
    feedforward (basal) and feedback (apical) pathways.  This is flagged as an
    info-level note.

    Only checked for populations in cortical / prefrontal regions where the
    apical/basal distinction is anatomically meaningful.  Silently skipped
    when ``mean_g_exc_apical`` is NaN (ConductanceLIF).
    """
    rec, pop_stats, issues = ctx.rec, ctx.pop_stats, ctx.issues
    config = ctx.thresholds.regional
    _NON_WAKING = frozenset({"none", "background", "slow_wave"})
    _WAKING_PATTERNS = frozenset(SENSORY_PATTERNS.keys()) - _NON_WAKING
    pattern = rec.config.sensory_pattern or ""
    is_waking = pattern in _WAKING_PATTERNS

    for (rn, pn), ps in pop_stats.items():
        if np.isnan(ps.mean_g_exc_apical):
            continue
        if not matches_any(rn, CORTICAL_TAGS):
            continue

        apical = ps.mean_g_exc_apical

        # Retrieve basal mean from the recorder's conductance samples.
        pop_idx = rec._pop_index.get((rn, pn))
        if pop_idx is None or rec._g_exc_samples is None:
            continue
        cond_step = rec._cond_sample_step
        if cond_step == 0:
            continue
        basal_slab = rec._g_exc_samples[:cond_step, pop_idx, :]
        basal = float(np.nanmean(basal_slab))
        if np.isnan(basal):
            continue

        # Check 1: silent apical during waking-state pattern.
        if is_waking and basal > config.basal_min_drive and apical < config.apical_silent:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.CONNECTIVITY,
                population=f"{rn}:{pn}",
                region=rn,
                message=(
                    f"Two-compartment apical silent: {rn}:{pn}  "
                    f"g_E_apical={apical:.4f}, g_E_basal={basal:.4f}  "
                    f"(pattern={pattern!r}) "
                    f"— apical dendrite is receiving no excitatory drive; "
                    f"top-down feedback projection may be absent or disconnected"
                ),
            ))

        # Check 2: apical/basal ratio ≈ 1.0 — functionally indistinct compartments.
        if basal > config.basal_min_drive and not np.isnan(apical) and apical > config.basal_min_drive:
            ratio = apical / basal
            if abs(ratio - 1.0) < config.compartment_ratio_tolerance:
                issues.append(HealthIssue(
                    severity="info",
                    category=HealthCategory.CONNECTIVITY,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"Two-compartment apical≈basal conductance: {rn}:{pn}  "
                        f"apical/basal ratio={ratio:.3f}  "
                        f"(g_apical={apical:.4f}, g_basal={basal:.4f}, "
                        f"pattern={pattern!r}) "
                        f"— apical and basal compartments are receiving equal drive; "
                        f"feedforward/feedback segregation may not be functioning "
                        f"as expected"
                    ),
                ))


def check_two_compartment_dendritic_computation(ctx: HealthCheckContext) -> None:
    """Validate dendritic computation in TwoCompartmentLIF populations.

    Three aspects are checked (only for populations with non-NaN dendritic
    metrics, i.e. identified as TwoCompartmentLIF during analysis):

    **bAP attenuation ratio**
    The ratio of dendritic voltage deflection to somatic spike amplitude
    should reflect biological back-propagating action potential attenuation.
    Expected range: 0.05–0.8 (Larkum et al. 2001; Stuart & Häusser 2001).
    A ratio of 0 means bAP is absent; > 1.0 means dendritic voltage exceeds
    somatic (possible Ca spike or pathological coupling).

    **NMDA plateau absence when expected**
    If the neuron config has ``enable_nmda_plateau=True`` (captured in
    ``_pop_neuron_params``) but the measured plateau fraction is 0, the
    NMDA plateau mechanism is not engaging. This may indicate insufficient
    apical NMDA drive or dendritic depolarisation never reaching threshold.

    **Coincidence detection — basal-apical gain modulation**
    Two-compartment neurons should fire more when both compartments receive
    simultaneous input (coincidence detection). A ``coincidence_gain ≈ 1.0``
    means apical input has no modulatory effect on firing, defeating the
    purpose of the second compartment.
    Expected healthy range: 1.2–5.0 (Larkum et al. 2004; Larkum 2013).
    """
    rec, pop_stats, issues = ctx.rec, ctx.pop_stats, ctx.issues

    for (rn, pn), ps in pop_stats.items():
        # Only applies to populations identified as two-compartment.
        if np.isnan(ps.mean_g_exc_apical):
            continue

        # ── bAP attenuation ratio ────────────────────────────────────
        if not np.isnan(ps.bap_attenuation_ratio):
            ratio = ps.bap_attenuation_ratio
            if ratio < 0.01:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.CONNECTIVITY,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"bAP absent: {rn}:{pn}  "
                        f"attenuation ratio={ratio:.4f}  "
                        f"— back-propagating action potentials are not "
                        f"reaching the apical dendrite; STDP coincidence "
                        f"detection at apical synapses will not function"
                    ),
                ))
            elif ratio > 1.0:
                issues.append(HealthIssue(
                    severity="info",
                    category=HealthCategory.CONNECTIVITY,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"Dendritic amplification: {rn}:{pn}  "
                        f"bAP ratio={ratio:.3f}  "
                        f"— dendritic voltage deflection exceeds somatic spike; "
                        f"possible Ca²⁺ spike or excessive coupling"
                    ),
                ))

        # ── NMDA plateau absence when enabled ────────────────────────
        params = rec._pop_neuron_params.get((rn, pn), {})
        nmda_enabled = params.get("enable_nmda_plateau", 0.0) > 0.5
        if nmda_enabled and not np.isnan(ps.nmda_plateau_fraction):
            if ps.nmda_plateau_fraction < 1e-6:
                issues.append(HealthIssue(
                    severity="info",
                    category=HealthCategory.CONNECTIVITY,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"NMDA plateau never triggered: {rn}:{pn}  "
                        f"(enable_nmda_plateau=True but plateau fraction=0)  "
                        f"— apical NMDA drive or dendritic depolarisation may be "
                        f"insufficient to reach plateau threshold"
                    ),
                ))

        # ── Coincidence detection gain ───────────────────────────────
        if not np.isnan(ps.coincidence_gain):
            gain = ps.coincidence_gain
            if gain < 1.1:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.CONNECTIVITY,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"No coincidence detection: {rn}:{pn}  "
                        f"gain={gain:.2f}  "
                        f"— apical input does not increase firing beyond "
                        f"basal-only drive; the second compartment has no "
                        f"modulatory effect"
                    ),
                ))
            elif gain > 10.0:
                issues.append(HealthIssue(
                    severity="info",
                    category=HealthCategory.CONNECTIVITY,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"Extreme coincidence gain: {rn}:{pn}  "
                        f"gain={gain:.1f}  "
                        f"— apical input multiplicatively amplifies firing "
                        f"far beyond expected range (1.2–5.0)"
                    ),
                ))
