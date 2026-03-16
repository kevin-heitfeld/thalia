"""Striatal and basal-ganglia pathway health checks.

Checks in this module go beyond D1/D2 competition (which is covered by
:func:`~.health_ei_oscillations.check_oscillatory_bands`) and verify the
structural integrity of the **indirect pathway** (D2 MSN → GPe → STN → SNr/GPi)
and the tonic activity of basal-ganglia output nuclei.

References
----------
- Albin, Young & Penney 1989 *Trends Neurosci* — indirect/direct pathway model.
- Mink 1996 *Prog Neurobiol* — focused inhibition via basal ganglia.
- DeLong 1990 *Trends Neurosci* — STN–GPe pacemaker oscillation.
- Bergman et al. 1998 *J Neurophysiol* — GPe tonic activity in parkinsonian models.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    PopulationStats,
)


# Matched against region names case-insensitively (substring match).
_GPE_TAGS = ("globus_pallidus", "gpe")
_STN_TAGS = ("subthalamic", "stn")
_SNR_TAGS = ("substantia_nigra_reticulata", "snr")
_GPI_TAGS = ("globus_pallidus_internal", "globus_pallidus_interna", "gpi")
_D2_POP_TAGS = ("d2",)
_D1_POP_TAGS = ("d1",)


def _matches(name: str, tags: tuple[str, ...]) -> bool:
    nl = name.lower()
    return any(t in nl for t in tags)


def check_basal_ganglia_pathway(
    pop_stats: "Dict[Tuple[str, str], PopulationStats]",
    issues: List[HealthIssue],
) -> None:
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
    # Collect per-region mean FRs for each BG nucleus
    d2_active_regions: set[str] = set()
    gpe_stats:  list[tuple[str, str, float]] = []  # (rn, pn, mean_fr_hz)
    stn_stats:  list[tuple[str, str, float]] = []
    snr_stats:  list[tuple[str, str, float]] = []
    gpi_stats:  list[tuple[str, str, float]] = []

    for (rn, pn), ps in pop_stats.items():
        if np.isnan(ps.mean_fr_hz):
            continue
        if _matches(pn, _D2_POP_TAGS) and ps.mean_fr_hz > 0.5:
            d2_active_regions.add(rn)
        if _matches(rn, _GPE_TAGS):
            gpe_stats.append((rn, pn, ps.mean_fr_hz))
        if _matches(rn, _STN_TAGS):
            stn_stats.append((rn, pn, ps.mean_fr_hz))
        if _matches(rn, _SNR_TAGS):
            snr_stats.append((rn, pn, ps.mean_fr_hz))
        if _matches(rn, _GPI_TAGS):
            gpi_stats.append((rn, pn, ps.mean_fr_hz))

    # ── 1. GPe tonic suppression ─────────────────────────────────────────────
    for rn, pn, fr in gpe_stats:
        # Only flag when D2 MSNs in the same brain are active (otherwise GPe
        # silence may simply reflect a basal-ganglia-free simulation).
        d2_nearby = bool(d2_active_regions)
        # Arkypallidal neurons (~25% of GPe) fire at 5–20 Hz physiologically
        # (Abdi et al. 2015), not 30–80 Hz like prototypic neurons.  Skip them.
        is_arkypallidal = "arkypallidal" in pn.lower()
        if d2_nearby and not is_arkypallidal and fr < 20.0:
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
        if fr > 60.0:
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
        if fr < 10.0:
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
