"""Two-compartment neuron health checks — apical/basal conductance segregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    PopulationStats,
)
from .region_tags import CORTICAL_TAGS, matches_any
from .sensory_patterns import SENSORY_PATTERNS

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


def check_two_compartment_apical_basal(
    rec: "DiagnosticsRecorder",
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
) -> None:
    """Check that TwoCompartmentLIF apical compartments are receiving distinct drive.

    For populations that report a non-NaN ``mean_g_exc_apical`` (i.e.
    :class:`~thalia.brain.neurons.TwoCompartmentLIF` populations recorded in
    full mode), two conditions are checked:

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
    when ``mean_g_exc_apical`` is NaN (stats mode or ConductanceLIF).
    """
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
        if is_waking and basal > 0.01 and apical < 0.02:
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
        if basal > 0.01 and not np.isnan(apical) and apical > 0.01:
            ratio = apical / basal
            if abs(ratio - 1.0) < 0.02:
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
                        f"as expected (Larkum 2013)"
                    ),
                ))
