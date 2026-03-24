"""Temporal integration time-constant health checks.

Per-region τ_int bounds and cross-tier cortical hierarchy validation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .bio_ranges import integration_tau_range
from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext
from .region_tags import (
    MOTOR_CORTEX_TAGS,
    PREFRONTAL_TAGS,
    SENSORY_CORTEX_TAGS,
    matches_any,
)


def check_integration_tau(ctx: HealthCheckContext) -> None:
    """Check firing-rate autocorrelation time constants against biological references.

    The temporal integration time constant τ_int reflects intrinsic neural memory
    and varies systematically across the cortical hierarchy (Murray et al. 2014):

    - Prefrontal cortex: τ ≈ 100–400 ms
    - Motor cortex:      τ ≈  40–150 ms
    - Sensory cortex:    τ ≈  20– 50 ms

    Deviations indicate:
    - τ too short for PFC: persistent activity absent — working-memory substrate broken.
    - τ too long for sensory cortex: sensory cortex stuck in reverberant activity;
      possible runaway recurrent excitation.
    """
    oscillations, issues = ctx.oscillations, ctx.issues
    for rn, tau_ms in oscillations.region_integration_tau_ms.items():
        if np.isnan(tau_ms):
            continue
        ref = integration_tau_range(rn)
        if ref is None:
            continue
        tau_min, tau_max = ref
        if tau_ms < tau_min:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Short integration τ: {rn}  τ={tau_ms:.0f} ms "
                    f"(expected {tau_min:.0f}–{tau_max:.0f} ms; "
                    f"Murray et al. 2014) — "
                    + (
                        "working-memory persistent activity absent (PFC)"
                        if matches_any(rn, PREFRONTAL_TAGS)
                        else "intrinsic temporal integration weaker than expected"
                    )
                ),
            ))
        elif tau_ms > tau_max:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Long integration τ: {rn}  τ={tau_ms:.0f} ms "
                    f"(expected {tau_min:.0f}–{tau_max:.0f} ms; "
                    f"Murray et al. 2014) — "
                    + (
                        "sensory cortex showing reverberant integration; "
                        "check recurrent excitation strength"
                        if matches_any(rn, SENSORY_CORTEX_TAGS)
                        else "integration time constant exceeds expected range"
                    )
                ),
            ))


def check_tau_hierarchy(ctx: HealthCheckContext) -> None:
    """Check that the cortical τ_int gradient is preserved across the hierarchy.

    Murray et al. 2014 (*Nature Neuroscience*) showed that the population-rate
    autocorrelation time constant increases monotonically from sensory to
    prefrontal cortex:

        cortex_sensory  τ ≈  20– 50 ms
        cortex_motor    τ ≈  40–150 ms
        prefrontal      τ ≈ 100–400 ms

    Each region is checked individually by :func:`check_integration_tau`.  This
    function performs the *cross-tier* check: even when every region's τ is
    within its own reference range, the ordering can still be violated (e.g.
    PFC τ = 105 ms < motor cortex τ = 148 ms, both within range but inverted).

    The check requires that the *mean* τ over all regions in each tier satisfies
    the ordering.  Filtering to mean (rather than min/max) is robust to brains
    with multiple cortical sub-regions in the same tier.  The check is skipped
    when fewer than two tiers have valid (non-NaN) tau values.
    """
    oscillations, issues = ctx.oscillations, ctx.issues
    tau_map = oscillations.region_integration_tau_ms

    sensory_taus = {
        rn: tau for rn, tau in tau_map.items()
        if not np.isnan(tau) and matches_any(rn, SENSORY_CORTEX_TAGS)
    }
    motor_taus = {
        rn: tau for rn, tau in tau_map.items()
        if not np.isnan(tau) and matches_any(rn, MOTOR_CORTEX_TAGS)
    }
    pfc_taus = {
        rn: tau for rn, tau in tau_map.items()
        if not np.isnan(tau) and matches_any(rn, PREFRONTAL_TAGS)
    }

    # Skip if fewer than 2 tiers present (can't assess a gradient).
    if sum(bool(t) for t in [sensory_taus, motor_taus, pfc_taus]) < 2:
        return

    def _mean(d: Dict[str, float]) -> float:
        return sum(d.values()) / len(d)

    s_mean = _mean(sensory_taus) if sensory_taus else None
    m_mean = _mean(motor_taus)   if motor_taus   else None
    p_mean = _mean(pfc_taus)     if pfc_taus     else None

    pairs: List[tuple[str, str]] = []
    if s_mean is not None and m_mean is not None and s_mean > m_mean:
        pairs.append((f"sensory τ={s_mean:.0f} ms", f"motor τ={m_mean:.0f} ms"))
    if m_mean is not None and p_mean is not None and m_mean > p_mean:
        pairs.append((f"motor τ={m_mean:.0f} ms", f"PFC τ={p_mean:.0f} ms"))
    if s_mean is not None and p_mean is not None and m_mean is None and s_mean > p_mean:
        pairs.append((f"sensory τ={s_mean:.0f} ms", f"PFC τ={p_mean:.0f} ms"))

    for higher, lower in pairs:
        issues.append(HealthIssue(
            severity="warning",
            category=HealthCategory.OSCILLATIONS,
            message=(
                f"Inverted cortical τ_int hierarchy: {higher} > {lower} — "
                f"expected sensory ≤ motor ≤ PFC "
                f"(Murray et al. 2014 Nature Neurosci)"
            ),
        ))
