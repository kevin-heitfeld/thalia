"""Striatal and basal-ganglia pathway health checks.

Checks in this module go beyond D1/D2 competition (which is covered by
:func:`~.health_oscillations.check_oscillatory_bands`) and verify the
structural integrity of the **indirect pathway** (D2 MSN → GPe → STN → SNr/GPi)
and the tonic activity of basal-ganglia output nuclei.
"""

from __future__ import annotations

import numpy as np

from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext
from .region_tags import (
    ARKYPALLIDAL_TAGS,
    D2_MSN_TAGS,
    GPE_TAGS,
    GPI_TAGS,
    SNR_TAGS,
    STN_TAGS,
    matches_any,
)


def check_basal_ganglia_pathway(ctx: HealthCheckContext) -> None:
    """Validate indirect-pathway ordering and basal-ganglia output-nucleus tonicity.

    Checks performed
    ----------------
    1. **GPe tonic suppression**: When D2 MSN populations are active (mean FR >
       0.5 Hz), GPe prototypic neurons should fire at 30–80 Hz.  A GPe FR below
       20 Hz while D2 MSNs are active indicates that D2 MSN inhibition has
       silenced GPe, which would also silence their disinhibitory braking effect
       on STN — the indirect pathway is broken (Bergman et al. 1998).

    2. **STN hyperactivity**: STN neurons fire tonically at 10–40 Hz.  A mean FR
       > 60 Hz indicates runaway excitation, typically caused by absent GPe
       inhibition (DeLong 1990 parkinsonian model) or pathological STN–GPe
       pacemaker oscillations.

    3. **SNr/GPi output tonic activity**: SNr and GPi provide tonic GABAergic
       inhibition of thalamus and brainstem targets at 30–80 Hz.  Firing below
       10 Hz collapses this gateway inhibition, leading to motor dis-inhibition
       and involuntary movements (Albin et al. 1989).
    """
    pop_stats, issues = ctx.pop_stats, ctx.issues
    config = ctx.thresholds.regional
    # Collect per-region mean FRs for each BG nucleus
    d2_active_regions: set[str] = set()
    gpe_stats:  list[tuple[str, str, float]] = []  # (rn, pn, mean_fr_hz)
    stn_stats:  list[tuple[str, str, float]] = []
    snr_stats:  list[tuple[str, str, float]] = []
    gpi_stats:  list[tuple[str, str, float]] = []

    for (rn, pn), ps in pop_stats.items():
        if np.isnan(ps.mean_fr_hz):
            continue
        if matches_any(pn, D2_MSN_TAGS) and ps.mean_fr_hz > 0.5:
            d2_active_regions.add(rn)
        if matches_any(rn, GPE_TAGS):
            gpe_stats.append((rn, pn, ps.mean_fr_hz))
        if matches_any(rn, STN_TAGS):
            stn_stats.append((rn, pn, ps.mean_fr_hz))
        if matches_any(rn, SNR_TAGS):
            snr_stats.append((rn, pn, ps.mean_fr_hz))
        if matches_any(rn, GPI_TAGS):
            gpi_stats.append((rn, pn, ps.mean_fr_hz))

    # ── 1. GPe tonic suppression ─────────────────────────────────────────────
    for rn, pn, fr in gpe_stats:
        # Only flag when D2 MSNs in the same brain are active (otherwise GPe
        # silence may simply reflect a basal-ganglia-free simulation).
        d2_nearby = bool(d2_active_regions)
        # Arkypallidal neurons (~25% of GPe) fire at 5–20 Hz physiologically
        # (Abdi et al. 2015), not 30–80 Hz like prototypic neurons.  Skip them.
        is_arkypallidal = matches_any(pn, ARKYPALLIDAL_TAGS)
        if d2_nearby and not is_arkypallidal and fr < config.gpe_min_fr_hz:
            pop_key = f"{rn}:{pn}"
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.FIRING,
                population=pop_key,
                region=rn,
                message=(
                    f"GPe tonic suppression: {rn}:{pn}  FR={fr:.1f} Hz "
                    f"(expected 30\u201380 Hz while D2 MSNs are active) \u2014 "
                    f"D2 MSN inhibition may have silenced GPe; "
                    f"indirect pathway STN disinhibition is broken (Bergman et al. 1998)"
                ),
            ))

    # ── 2. STN hyperactivity ─────────────────────────────────────────────────
    for rn, pn, fr in stn_stats:
        pop_key = f"{rn}:{pn}"
        if fr > config.stn_max_fr_hz:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.FIRING,
                population=pop_key,
                region=rn,
                message=(
                    f"STN hyperactive: {rn}:{pn}  FR={fr:.1f} Hz "
                    f"(expected 10\u201340 Hz) \u2014 absent GPe inhibition or "
                    f"pathological STN\u2013GPe pacemaker oscillation; "
                    f"check GPe\u2192STN connectivity (DeLong 1990)"
                ),
            ))

    # ── 3. SNr/GPi tonic gateway activity ────────────────────────────────────
    for rn, pn, fr in snr_stats + gpi_stats:
        pop_key = f"{rn}:{pn}"
        if fr < config.bg_output_min_fr_hz:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.FIRING,
                population=pop_key,
                region=rn,
                message=(
                    f"BG output nucleus under-active: {rn}:{pn}  FR={fr:.1f} Hz "
                    f"(expected 30\u201380 Hz) \u2014 tonic GABAergic gating of thalamus "
                    f"collapsed; check D1 MSN\u2192SNr/GPi direct pathway weights "
                    f"(Albin et al. 1989)"
                ),
            ))
